"""Hand-written Metal kernels for the fp16 MLX trunk's memory-bound work.

Profiling (L=238, fp16) puts ~17% of a trunk pass outside the
matmuls, mostly in ops that MLX's compiled kernels run well below the ~180 GB/s
the M1 Pro reaches on a plain copy: the layout changes. Each kernel here keeps
the matmuls on MLX's GEMM and fuses one memory-bound step into a single pass:

  gate_cf       proj [B,N,4C] -> a, b [B,C,L,L]: signal*sigmoid(gate)(*mask), written
                channels-first through a threadgroup tile, so both the read and the
                write are coalesced (the compiled kernel reads with a 2 KB stride).
  ln_cf_to_cl   contraction output [B,C,L,L] -> LayerNorm over C, written [B,L,L,C]:
                transpose + norm_mix in one pass instead of a copy plus a LayerNorm.
  resgate_ln    z + mixed*sigmoid(g) -> (z', LayerNorm(z')): the tri-mul's residual
                fused with the next op's input LayerNorm (z' is still written once).

All arithmetic is fp32 inside the kernels (LayerNorm statistics included), with
one rounding to the storage dtype per output, so results match the MLX ops to
fp16 rounding. The 1/256 contraction pre-scale lives in the weights and norm_mix's
eps (mlx_trunk.py); nothing here changes it.

``block_forward``/``make_trunk`` mirror mlx_trunk's but call these kernels; the
transition residual uses mx.addmm (GEMM epilogue) instead of a separate add.

Usage:
    from esmfold2_mlx import mlx_trunk_kernels as mk
    fn = mk.make_trunk(mx.float16)          # same signature as mlx_trunk.make_trunk
    z_out = fn(params, z, mask)             # params from mlx_trunk.trunk_params(..., mx.float16)
"""

from functools import partial

import mlx.core as mx

from . import mlx_trunk as mt

# Channels-first rows (c, p0..p0+TP) start at flat index c*N + p0, which is only
# 4-aligned when N is; so each row segment is moved as aligned 4-element vectors
# plus scalar ends (work item j < nq: a vector; j == nq: the ends).
_HEADER = """
#define SEGMENT(s, cnt) uint head = min(uint((4 - (s) % 4) % 4), (cnt)); uint nq = ((cnt) - head) / 4;
"""

# --- gate_cf ----------------------------------------------------------------
# Threadgroup of 256 handles TP positions x TC channels of one operand: vector
# loads of 4 channels (8 B), gate in fp32, tile, then transposed vector stores.
# Tile shape chosen by a timing sweep.
_GATE_TP, _GATE_TC = 128, 32
_GATE_SRC = """
    const uint TQ = TC / 4, RPI = 256 / TQ, IPR = TP / 4 + 1;
    uint c0 = threadgroup_position_in_grid.x * TC;   // in [0, 2C): a's channels then b's
    uint p0 = threadgroup_position_in_grid.y * TP;
    uint bi = threadgroup_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;
    uint cnt = min(uint(TP), uint(N) - p0);
    threadgroup T tile[TP][TC + 1];
    const device T* P = proj + (ulong)bi * N * 4 * C;
    uint cq = (tid % TQ) * 4;
    for (uint r = tid / TQ; r < cnt; r += RPI) {
        uint pos = p0 + r;
        float4 s = float4(*(const device vec<T, 4>*)(P + (ulong)pos * 4 * C + c0 + cq));
        float4 g = float4(*(const device vec<T, 4>*)(P + (ulong)pos * 4 * C + 2 * C + c0 + cq));
        float4 v = s / (1.0f + metal::exp(-g));
        if (HAS_MASK) v *= float(mask[(ulong)bi * N + pos]);
        for (uint k = 0; k < 4; k++) tile[r][cq + k] = T(v[k]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < TC * IPR; i += 256) {
        uint c = i / IPR, j = i % IPR, cc = c0 + c;
        device T* O = cc < C ? a : b;
        ulong s = ((ulong)bi * C + (cc < C ? cc : cc - C)) * N + p0;
        SEGMENT(s, cnt)
        if (j < nq) {
            uint k = head + 4 * j;
            *(device vec<T, 4>*)(O + s + k) = vec<T, 4>(tile[k][c], tile[k + 1][c], tile[k + 2][c], tile[k + 3][c]);
        } else if (j == nq) {
            for (uint k = 0; k < head; k++) O[s + k] = tile[k][c];
            for (uint k = head + 4 * nq; k < cnt; k++) O[s + k] = tile[k][c];
        }
    }
"""
_gate_k = mx.fast.metal_kernel(name="trunk_gate_cf", input_names=["proj", "mask"], output_names=["a", "b"],
                               source=_GATE_SRC, header=_HEADER)


def gate_cf(proj, mask, c):
    """proj [B, L, L, 4c] -> (a, b) [B, c, L, L]; mask [B, L, L] or None.
    Same as mlx_trunk._gated_cf(proj, mask, 0, c), _gated_cf(proj, mask, c, c)."""
    B, L = proj.shape[0], proj.shape[1]
    N = L * L
    TP, TC = _GATE_TP, _GATE_TC
    assert proj.shape[-1] == 4 * c and c % TC == 0
    m = mask if mask is not None else mx.zeros((1,), proj.dtype)
    return tuple(_gate_k(inputs=[proj, m],
                         template=[("T", proj.dtype), ("N", N), ("C", c), ("TP", TP), ("TC", TC),
                                   ("HAS_MASK", mask is not None)],
                         grid=(256 * (2 * c // TC), (N + TP - 1) // TP, B), threadgroup=(256, 1, 1),
                         output_shapes=[(B, c, L, L)] * 2, output_dtypes=[proj.dtype] * 2))


# --- ln_cf_to_cl --------------------------------------------------------------
# Threadgroup of 256 (8 simdgroups) handles TP positions x all C channels: loads
# along positions (4-wide vectors when N % 4 == 0) into a position-major tile, then
# one simdgroup per position reduces over channels (fp32) and writes the row with
# vector stores. The tile's row stride is an odd number of 4-byte banks (C + 2
# halves): 0.459 -> 0.415 ms at L=238 vs C + 4.
_LNT_TP = 32
_LNT_SRC = """
    const uint QPR = TP / 4, NQ = C / 128;
    uint p0 = threadgroup_position_in_grid.y * TP;
    uint bi = threadgroup_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x, lane = tid % 32, sg = tid / 32;
    uint cnt = min(uint(TP), uint(N) - p0);
    threadgroup T tile[TP][C + 2];
    if (N % 4 == 0) {  // rows 8-byte aligned, and cnt % 4 == 0: 4-position vector loads
        for (uint i = tid; i < C * QPR; i += 256) {
            uint c = i / QPR, k = 4 * (i % QPR);
            if (k >= cnt) continue;
            vec<T, 4> q = *(const device vec<T, 4>*)(x + ((ulong)bi * C + c) * N + p0 + k);
            for (uint m = 0; m < 4; m++) tile[k + m][c] = q[m];
        }
    } else {  // rows misaligned: scalar loads, a simdgroup per channel row (measured faster than SEGMENT here)
        for (uint c = sg; c < C; c += 8)
            if (lane < cnt) tile[lane][c] = x[((ulong)bi * C + c) * N + p0 + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 wv[NQ], bv[NQ];
    for (uint i = 0; i < NQ; i++) {
        wv[i] = float4(*(const device vec<T, 4>*)(w + 128 * i + 4 * lane));
        bv[i] = float4(*(const device vec<T, 4>*)(bias + 128 * i + 4 * lane));
    }
    float e = eps[0];
    for (uint r = sg; r < cnt; r += 8) {
        float4 v[NQ];
        float s = 0.0f;
        for (uint i = 0; i < NQ; i++) {
            threadgroup vec<T, 2>* t2 = (threadgroup vec<T, 2>*)(&tile[r][128 * i + 4 * lane]);
            v[i] = float4(float2(t2[0]), float2(t2[1]));
            s += v[i].x + v[i].y + v[i].z + v[i].w;
        }
        float mean = simd_sum(s) / C, q = 0.0f;
        for (uint i = 0; i < NQ; i++) { v[i] -= mean; q += dot(v[i], v[i]); }
        float inv = metal::rsqrt(simd_sum(q) / C + e);
        device T* O = out + ((ulong)bi * N + p0 + r) * C;
        for (uint i = 0; i < NQ; i++)
            *(device vec<T, 4>*)(O + 128 * i + 4 * lane) = vec<T, 4>(v[i] * inv * wv[i] + bv[i]);
    }
"""
_lnt_k = mx.fast.metal_kernel(name="trunk_ln_cf_to_cl", input_names=["x", "w", "bias", "eps"], output_names=["out"],
                              source=_LNT_SRC)


def ln_cf_to_cl(x, w, b, eps):
    """x [B, C, L, L] -> LayerNorm over C of x.transpose(0, 2, 3, 1), shape [B, L, L, C]."""
    B, C, L = x.shape[0], x.shape[1], x.shape[2]
    N = L * L
    assert C % 128 == 0
    return _lnt_k(inputs=[x, w, b, mx.array([eps], mx.float32)],
                  template=[("T", x.dtype), ("N", N), ("C", C), ("TP", _LNT_TP)],
                  grid=(256, (N + _LNT_TP - 1) // _LNT_TP, B), threadgroup=(256, 1, 1),
                  output_shapes=[(B, L, L, C)], output_dtypes=[x.dtype])[0]


# --- resgate_ln ---------------------------------------------------------------
# One simdgroup per [C] row (8 rows per threadgroup), 4-channel vector loads/stores.
_RGLN_SRC = """
    const uint NQ = C / 128;
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= R) return;
    ulong o = (ulong)row * C + 4 * lane;
    float4 v[NQ];
    float s = 0.0f;
    for (uint i = 0; i < NQ; i++) {
        ulong k = o + 128 * i;
        float4 g = float4(*(const device vec<T, 4>*)(gate + k));
        vec<T, 4> zn = vec<T, 4>(float4(*(const device vec<T, 4>*)(z + k))
                                 + float4(*(const device vec<T, 4>*)(m + k)) / (1.0f + metal::exp(-g)));
        *(device vec<T, 4>*)(z_out + k) = zn;
        v[i] = float4(zn);  // normalize the stored (rounded) residual, as the unfused path does
        s += v[i].x + v[i].y + v[i].z + v[i].w;
    }
    float mean = simd_sum(s) / C, q = 0.0f;
    for (uint i = 0; i < NQ; i++) { v[i] -= mean; q += dot(v[i], v[i]); }
    float inv = metal::rsqrt(simd_sum(q) / C + eps[0]);
    for (uint i = 0; i < NQ; i++) {
        uint c = 128 * i + 4 * lane;
        float4 wv = float4(*(const device vec<T, 4>*)(w + c)), bv = float4(*(const device vec<T, 4>*)(bias + c));
        *(device vec<T, 4>*)(n_out + o + 128 * i) = vec<T, 4>(v[i] * inv * wv + bv);
    }
"""
_rgln_k = mx.fast.metal_kernel(name="trunk_resgate_ln", input_names=["z", "m", "gate", "w", "bias", "eps"],
                               output_names=["z_out", "n_out"], source=_RGLN_SRC)


def resgate_ln(z, m, g, w, b, eps):
    """(z + m * sigmoid(g), layer_norm(that, w, b, eps)); all [..., C] channels-last."""
    C = z.shape[-1]
    R = z.size // C
    assert C % 128 == 0
    return tuple(_rgln_k(inputs=[z, m, g, w, b, mx.array([eps], mx.float32)],
                         template=[("T", z.dtype), ("R", R), ("C", C)],
                         grid=(32, 8 * ((R + 7) // 8), 1), threadgroup=(32, 8, 1),
                         output_shapes=[z.shape] * 2, output_dtypes=[z.dtype] * 2))


# --- trunk ----------------------------------------------------------------------
ALL = ("gate", "ln_mix", "resgate", "addmm")  # every fusion; subsets are kept for ablation


def _trimul(p, z, n, mask, outgoing, mix_eps, next_ln, use):
    """One triangle update given n = its input LayerNorm of z. Returns (z + update,
    next_ln(z + update)), the residual and the next LayerNorm fused if "resgate"."""
    c = z.shape[-1]
    proj = n @ p["w_in"]
    if "gate" in use:
        a, b = gate_cf(proj, mask, c)
    else:
        a, b = mt._gated_cf(proj, mask, 0, c), mt._gated_cf(proj, mask, c, c)
    contracted = a @ b.swapaxes(2, 3) if outgoing else a.swapaxes(2, 3) @ b
    if "ln_mix" in use:
        mixed = ln_cf_to_cl(contracted, p["ln_mix_w"], p["ln_mix_b"], mix_eps)
    else:
        mixed = mx.fast.layer_norm(contracted.transpose(0, 2, 3, 1), p["ln_mix_w"], p["ln_mix_b"], mix_eps)
    mixed = mixed @ p["w_emit"]
    g = n @ p["w_out_gate"]
    if "resgate" in use:
        return resgate_ln(z, mixed, g, *next_ln, mt.EPS)
    z = z + mixed * mx.sigmoid(g)
    return z, mx.fast.layer_norm(z, *next_ln, mt.EPS)


def block_forward(p, z, mask, mix_eps=mt.EPS, use=ALL):
    n = mx.fast.layer_norm(z, p["out"]["ln_in_w"], p["out"]["ln_in_b"], mt.EPS)
    z, n = _trimul(p["out"], z, n, mask, True, mix_eps, (p["in"]["ln_in_w"], p["in"]["ln_in_b"]), use)
    z, h = _trimul(p["in"], z, n, mask, False, mix_eps, (p["ln_w"], p["ln_b"]), use)
    x = h @ p["w12"]
    x1, x2 = mx.split(x, 2, axis=-1)
    s = x1 * mx.sigmoid(x1) * x2  # SwiGLU; MLX's compiled kernel is already near copy bandwidth
    return mx.addmm(z, s, p["w3"]) if "addmm" in use else z + s @ p["w3"]  # addmm: residual in the GEMM


def trunk_forward(params, z, mask=None, mix_eps=mt.EPS, use=ALL):
    """z in any float dtype; computed in the params' dtype, returned as fp32."""
    dtype = params[0]["w3"].dtype
    z = z.astype(dtype)
    mask = None if mask is None else mask.astype(dtype)
    for p in params:
        z = block_forward(p, z, mask, mix_eps, use)
    return z.astype(mx.float32)


def make_trunk(dtype=mx.float16, use=ALL):
    """Compiled trunk for params made with ``mlx_trunk.trunk_params(..., dtype)``."""
    mix_eps = mt.EPS / mt.CONTRACT_SCALE**2 if dtype == mx.float16 else mt.EPS
    return mx.compile(partial(trunk_forward, mix_eps=mix_eps, use=tuple(use)))
