"""The ESMFold2 folding trunk ported to MLX, using the PyTorch checkpoint's weights.

The trunk is 24 identical PairUpdateBlocks, each: triangle update (outgoing),
triangle update (incoming), SwiGLU transition, all on a [B, L, L, 256] pair grid.
Everything stays fp32.

Layout choices (measured at L=238, compiled, one trunk pass):
the original port took 1.72 s; this one 1.59 s. The win is in the tri-mul:
projections stay channels-last (MLX's fp32 GEMM is ~20% slower channels-first),
and the gating is computed on transposed views so the fused elementwise kernel
writes the channels-first contraction operands directly, instead of a separate
transpose copy. ``mask=None`` skips the visibility multiply (all-ones mask).

Precision: ``dtype=mx.float16`` stores
weights and every intermediate, including the residual stream, in fp16; MLX's
matmuls and fast.layer_norm still accumulate in fp32. The triangle contraction
sums L products and reaches ~64k at L=238, at fp16's 65504 limit, so the left
operand's projection is divided by 256 and norm_mix's eps by 256^2 to match:
LN(c/s, eps/s^2) == LN(c, eps) exactly.

Usage:
    from esmfold2_mlx.mlx_trunk import trunk_params, make_trunk
    params = trunk_params(torch_model.folding_trunk)             # or dtype=mx.float16
    z_out = make_trunk(mx.float32)(params, z, mask)               # compiled; mask may be None
"""

from functools import partial

import mlx.core as mx

EPS = 1e-5  # esm.models.esmfold2.layers._EPS and nn.LayerNorm's default
CF = (0, 3, 1, 2)  # [B, L, L, C] -> [B, C, L, L]
CONTRACT_SCALE = 256.0  # fp16 only; a power of two, so the rescale is exact


def _mx(t, dtype=mx.float32):
    return mx.array(t.detach().float().cpu().numpy()).astype(dtype)


def _trimul_params(tri, dtype):
    eng = tri._engine
    c = eng.latent_channels
    assert eng.proj_bundle.weight.shape[0] == 4 * c
    # proj_bundle rows are [signal_a | signal_b | gate_a | gate_b]; transposed once
    # here so every call is a plain x @ W
    w_in = eng.proj_bundle.weight.detach().float().T.clone()
    if dtype == mx.float16:
        w_in[:, :c] /= CONTRACT_SCALE  # left contraction operand; see module docstring
    _d = lambda t: _mx(t, dtype)  # noqa: E731
    return {
        "ln_in_w": _d(eng.norm_start.weight), "ln_in_b": _d(eng.norm_start.bias),
        "w_in": _d(w_in),
        "ln_mix_w": _d(eng.norm_mix.weight), "ln_mix_b": _d(eng.norm_mix.bias),
        "w_emit": _d(eng.proj_emit.weight.T), "w_out_gate": _d(eng.proj_gate.weight.T),
    }


def trunk_params(folding_trunk, dtype=mx.float32):
    """Copy every block's weights out of a PyTorch FoldingTrunk, stored in ``dtype``."""
    blocks = []
    for blk in folding_trunk.blocks:
        tr = blk.pair_transition
        blocks.append({
            "out": _trimul_params(blk.tri_mul_out, dtype), "in": _trimul_params(blk.tri_mul_in, dtype),
            "ln_w": _mx(tr.norm.weight, dtype), "ln_b": _mx(tr.norm.bias, dtype),
            "w12": _mx(tr.ffn.w12.weight.T, dtype), "w3": _mx(tr.ffn.w3.weight.T, dtype),
        })
    mx.eval(blocks)
    return blocks


def _gated_cf(proj, mask, lo, c):
    # Elementwise ops on transposed views: compiled, this is one kernel that
    # writes the [B, c, L, L] operand contiguously (no separate transpose copy).
    r = proj[..., lo : lo + c].transpose(CF) * mx.sigmoid(proj[..., 2 * c + lo : 3 * c + lo].transpose(CF))
    return r if mask is None else r * mask[:, None]


def _trimul(p, z, mask, outgoing, mix_eps):
    c = z.shape[-1]
    n = mx.fast.layer_norm(z, p["ln_in_w"], p["ln_in_b"], EPS)
    proj = n @ p["w_in"]
    a, b = _gated_cf(proj, mask, 0, c), _gated_cf(proj, mask, c, c)
    # Triangle contraction as one batched matmul per channel, channels-first;
    # the swapaxes are free (matmul transpose flags):
    #   outgoing: out[d,i,j] = sum_k a[d,i,k] b[d,j,k]
    #   incoming: out[d,i,j] = sum_k a[d,k,i] b[d,k,j]
    contracted = a @ b.swapaxes(2, 3) if outgoing else a.swapaxes(2, 3) @ b
    mixed = mx.fast.layer_norm(contracted.transpose(0, 2, 3, 1), p["ln_mix_w"], p["ln_mix_b"], mix_eps) @ p["w_emit"]
    return mixed * mx.sigmoid(n @ p["w_out_gate"])


def block_forward(p, z, mask, mix_eps=EPS):
    z = z + _trimul(p["out"], z, mask, True, mix_eps)
    z = z + _trimul(p["in"], z, mask, False, mix_eps)
    h = mx.fast.layer_norm(z, p["ln_w"], p["ln_b"], EPS)
    x = h @ p["w12"]
    x1, x2 = mx.split(x, 2, axis=-1)
    return z + (x1 * mx.sigmoid(x1) * x2) @ p["w3"]  # SwiGLU: silu(x1) * x2


def trunk_forward(params, z, mask=None, mix_eps=EPS):
    """z in any float dtype; computed in the params' dtype, returned as fp32."""
    dtype = params[0]["w3"].dtype
    z = z.astype(dtype)
    mask = None if mask is None else mask.astype(dtype)
    for p in params:
        z = block_forward(p, z, mask, mix_eps)
    return z.astype(mx.float32)


def make_trunk(dtype=mx.float32):
    """Compiled trunk for params made with ``trunk_params(..., dtype)``."""
    mix_eps = EPS / CONTRACT_SCALE**2 if dtype == mx.float16 else EPS
    return mx.compile(partial(trunk_forward, mix_eps=mix_eps))
