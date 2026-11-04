"""Faster MLX diffusion sampler, built on mlx_diffusion.py (which stays unchanged).

Options (all default off = numerically the same path as mlx_diffusion):
  tok16   token transformer matmuls + attention in fp16 (fp16 weights, fp32 accumulation
          inside the GEMM, fp32 residual stream / LayerNorm / gates)
  cond16  noise conditioning (s transitions, adaLN / output-gate projections) in fp16
  pair16  per-fold pair conditioning + the 12 pair biases in fp16 (needed by tok16's
          fp16 attention anyway: the bias is the SDPA mask)
  atom16  atom encoder / decoder matmuls in fp16 (residual stream fp32)
  horn    Kabsch rotation on the GPU (Horn quaternion) instead of the 3x3 SVD
          on the CPU stream, so a step never waits on the CPU
  tokq8   8-bit quantized token-transformer weights (mx.quantized_matmul)
Coordinates, the Euler update, noise and the alignment stay fp32 in every mode.
The token attention runs in the pair-bias dtype (fp16 with pair16 or tok16).

Findings (6B head, M1 Pro): tok16 + cond16 + pair16 is the adopted set
(var_mlx_diffusion16): per-call error ~0.02 Å RMS, all finite up to 476 aa, sample
1.2-1.25x faster. cond16 alone gives most of the conditioning gain; pair16 does not
speed up the per-fold context; fp16 attention with fp32 matmuls is slower than full
tok16; atom16 and horn gave no measurable gain; tokq8 is slower (MLX's quantized
matmul at M <= 238 runs at ~1/2 the speed of fp32 here) and ~4x the error of fp16.

Usage:
    from esmfold2_mlx.mlx_diffusion16 import MlxDiffusion
    MlxDiffusion(head, tok16=True, cond16=True, pair16=True, horn=True)
"""

import math

import mlx.core as mx
import numpy as np
import torch

from . import mlx_diffusion as base
from .mlx_diffusion import (COND_BUDGET, RMS_EPS, _layer_norm, _local_attention, _rope, _silu, _to_mx, _det3)

F16, F32 = mx.float16, mx.float32
HORN_SQUARINGS = 24
Q_GROUP = 64


def _cast(tree, dtype, keys=None):
    if isinstance(tree, dict):
        return {k: (_cast(v, dtype) if keys is None or k in keys else v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_cast(v, dtype) for v in tree)
    return tree.astype(dtype)


def _mm(x, w):
    """x @ w in w's precision (fp16 GEMMs accumulate in fp32 in MLX); a dict w is an
    8-bit quantized weight (tokq8)."""
    if isinstance(w, dict):
        return mx.quantized_matmul(x.astype(w["s"].dtype), w["q"], w["s"], w["b"], transpose=True,
                                   group_size=Q_GROUP, bits=8)
    return x.astype(w.dtype) @ w


def _quant(w, dtype=F32):
    q, s, b = mx.quantize(w.T.astype(dtype), group_size=Q_GROUP, bits=8)
    return {"q": q, "s": s, "b": b}


def _transition(p, x):
    h = _layer_norm(x, p["ln"])
    return (_silu(_mm(h, p["a"])) * _mm(h, p["b"])) @ p["out"]


def noise_conditioning(p, s0, t, sigma):
    t_noise = 0.25 * mx.log(mx.maximum(t / sigma, 1e-20))
    n = mx.cos(2.0 * math.pi * (t_noise[:, None] * p["fourier_w"] + p["fourier_b"]))
    s = s0 + (_layer_norm(n, p["noise_ln"]) @ p["noise_proj"])[:, None]
    for blk in p["s_transitions"]:
        s = s + _transition(blk, s).astype(F32)
    pt = p["token"]
    return {"s_tok": _mm(_layer_norm(s, p["s_step_ln"]), p["s_to_token"]).astype(F32),
            "ada": _mm(_layer_norm(s), pt["ada_w"]) + pt["ada_b"],
            "og": mx.sigmoid(_mm(s, pt["og_w"]) + pt["og_b"])}


def pair_biases(p, z_trunk, rel_pos, n_heads, out_dtype):
    z = _mm(_layer_norm(mx.concatenate([z_trunk, rel_pos], axis=-1), p["ln"]), p["proj"]).astype(F32)
    for blk in p["transitions"]:
        z = z + _transition(blk, z).astype(F32)
    B, L = z.shape[:2]
    bias = _mm(_layer_norm(z), p["bias_w"]) + p["bias_b"]
    return bias.astype(out_dtype).reshape(B, L, L, -1, n_heads).transpose(3, 0, 4, 1, 2)


def _attn_pair_bias(p, x, out_gate, bias, n_heads):
    B, L, d = x.shape
    hd = d // n_heads
    qkvg = _mm(x, p["qkvg"]) + p["qkvg_b"]
    q, k, v, g = mx.split(qkvg, 4, axis=-1)
    q, k, v = (t.reshape(B, L, n_heads, hd).transpose(0, 2, 1, 3).astype(bias.dtype) for t in (q, k, v))
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask=bias)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, d).astype(g.dtype) * mx.sigmoid(g)
    return out_gate.astype(F32) * _mm(o, p["o"]).astype(F32)


def _cond_transition(p, x, out_gate):
    x1, x2 = mx.split(_mm(x, p["swish"]), 2, axis=-1)
    return out_gate.astype(F32) * _mm(_silu(x1) * x2, p["out"]).astype(F32)


def token_transformer(p, a, cond, pair_bias, n_heads):
    nb = len(p["attn"])
    ada = [t.astype(F32) for t in mx.split(cond["ada"], 4 * nb, axis=-1)]
    og = mx.split(cond["og"], 2 * nb, axis=-1)
    for i in range(nb):
        x = mx.sigmoid(ada[4 * i]) * _layer_norm(a) + ada[4 * i + 1]
        a = a + _attn_pair_bias(p["attn"][i], x, og[2 * i], pair_bias[i], n_heads)
        x = mx.sigmoid(ada[4 * i + 2]) * _layer_norm(a) + ada[4 * i + 3]
        a = a + _cond_transition(p["trans"][i], x, og[2 * i + 1])
    return a


def _swa_block_apply(p, mod, x, ctx, n_heads):
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
    B, N, d = x.shape
    hd = d // n_heads
    h = mx.fast.rms_norm(x, None, RMS_EPS) * (1 + scale_a) + shift_a
    qkv = _mm(h, p["Wqkv"]).astype(F32).reshape(B, N, 3, n_heads, hd)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = _rope(mx.fast.rms_norm(q, None, RMS_EPS), ctx["cos"], ctx["sin"]).astype(mx.bfloat16)
    k = _rope(mx.fast.rms_norm(k, None, RMS_EPS), ctx["cos"], ctx["sin"]).astype(mx.bfloat16)
    v = v.astype(mx.bfloat16)
    if "swa_local_mask" in ctx:
        o = _local_attention(q, k, v, ctx["swa_local_mask"], ctx["swa_local_mask"].shape[2], hd ** -0.5)
    else:
        o = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
            scale=hd ** -0.5, mask=ctx["swa_mask"]).transpose(0, 2, 1, 3).reshape(B, N, d)
    o = o.astype(F32) * ctx["valid"]
    o = _mm(o * mx.sigmoid(_mm(h, p["gate"]).astype(F32)), p["out"]).astype(F32)
    x = x + gate_a * o
    h = mx.fast.rms_norm(x, None, RMS_EPS) * (1 + scale_f) + shift_f
    x1, x2 = mx.split(_mm(h, p["up"]), 2, axis=-1)
    return x + gate_f * ((_silu(x1) * x2) @ p["down"]).astype(F32)


def denoise(p, ctx, cond, x_noisy, t, sigma, n_atom_heads, n_token_heads):
    r = x_noisy / mx.sqrt(t * t + sigma * sigma)[:, None, None]
    q = ctx["c"] + r @ p["coords"]
    for blk, mod in zip(p["enc_blocks"], ctx["enc_mods"]):
        q = _swa_block_apply(blk, mod, q, ctx, n_atom_heads)
    q_skip = q
    a = ctx["atom_to_token_mean"] @ mx.maximum(_mm(q, p["a2t"]).astype(F32), 0.0)
    a = a + cond["s_tok"]
    a = _layer_norm(token_transformer(p["token"], a, cond, ctx["pair_bias"], n_token_heads), p["token_ln"])
    q = q_skip + mx.take(_mm(a, p["t2a"]).astype(F32), ctx["tok_idx"], axis=1)
    for blk, mod in zip(p["dec_blocks"], ctx["dec_mods"]):
        q = _swa_block_apply(blk, mod, q, ctx, n_atom_heads)
    r_update = _layer_norm(q, p["dec_ln"]) @ p["dec_out"]
    s2, t2 = sigma * sigma, t * t
    out = (s2 / (s2 + t2))[:, None, None] * x_noisy
    return out + ((sigma * t) / mx.sqrt(s2 + t2))[:, None, None] * r_update, a


def horn_rotation(H, squarings=HORN_SQUARINGS):
    """Rotation R maximising tr(R^T H) (SVD-Kabsch with det fix) via Horn's quaternion
    method: top eigenvector of a 4x4 symmetric matrix by repeated squaring. H: [B, 3, 3]
    = sum w (x_gt - mu_gt)^T (x - mu) as in base.rigid_align (so R = U D Vh there)."""
    S = H.swapaxes(-1, -2)  # S[a, b] = sum x_a y_b
    sxx, sxy, sxz = S[..., 0, 0], S[..., 0, 1], S[..., 0, 2]
    syx, syy, syz = S[..., 1, 0], S[..., 1, 1], S[..., 1, 2]
    szx, szy, szz = S[..., 2, 0], S[..., 2, 1], S[..., 2, 2]
    N = mx.stack([
        sxx + syy + szz, syz - szy, szx - sxz, sxy - syx,
        syz - szy, sxx - syy - szz, sxy + syx, szx + sxz,
        szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy,
        sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz,
    ], axis=-1).reshape(*H.shape[:-2], 4, 4)
    fro = lambda A: mx.sqrt((A * A).sum(axis=(-2, -1), keepdims=True))  # noqa: E731
    A = N / mx.maximum(fro(N), 1e-30) + mx.eye(4)
    for _ in range(squarings):
        A = A @ A
        A = A / fro(A)
    col = mx.argmax(mx.sqrt((A * A).sum(axis=-2)), axis=-1)  # [B]
    q = mx.take_along_axis(A, col[..., None, None].astype(mx.int32) * mx.ones((1, 4, 1), dtype=mx.int32), axis=-1)[..., 0]
    q = q / mx.sqrt((q * q).sum(axis=-1, keepdims=True))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return mx.stack([
        w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z,
    ], axis=-1).reshape(*H.shape[:-2], 3, 3)


def rigid_align(x, x_gt, w, horn):
    if not horn:
        return base.rigid_align(x, x_gt, w)
    w = w[..., None]
    denom = mx.maximum(w.sum(axis=-2, keepdims=True), 1e-8)
    mu = (x * w).sum(axis=-2, keepdims=True) / denom
    mu_gt = (x_gt * w).sum(axis=-2, keepdims=True) / denom
    x_c, xgt_c = x - mu, x_gt - mu_gt
    H = (w * xgt_c).swapaxes(-1, -2) @ x_c
    R = horn_rotation(H)
    return x_c @ R.swapaxes(-1, -2) + mu_gt


class MlxDiffusion(base.MlxDiffusion):
    def __init__(self, head, compile=True, tok16=False, cond16=False, pair16=False, atom16=False, horn=False,
                 tokq8=False):
        super().__init__(head, compile=False)
        self.opts = dict(tok16=tok16, cond16=cond16, pair16=pair16, atom16=atom16, horn=horn, tokq8=tokq8)
        p = self.p
        if tokq8:  # 8-bit token-transformer weights (scales in fp16 if tok16, else fp32)
            dt = F16 if tok16 else F32
            p["token"]["attn"] = [{**b, "qkvg": _quant(b["qkvg"], dtype=dt), "o": _quant(b["o"], dtype=dt)}
                                  for b in p["token"]["attn"]]
            p["token"]["trans"] = [{k: _quant(v, dtype=dt) for k, v in b.items()} for b in p["token"]["trans"]]
            tok16 = False if not tok16 else "biasonly"
        if tok16 == "biasonly":
            p["token"]["attn"] = [{**b, "qkvg_b": b["qkvg_b"].astype(F16)} for b in p["token"]["attn"]]
        elif tok16:
            p["token"]["attn"] = [_cast(b, F16, ("qkvg", "qkvg_b", "o")) for b in p["token"]["attn"]]
            p["token"]["trans"] = _cast(p["token"]["trans"], F16)
        if cond16:
            p["s_transitions"] = [_cast(b, F16, ("a", "b", "out")) for b in p["s_transitions"]]
            p["token"] = {**p["token"], **_cast({k: p["token"][k] for k in ("ada_w", "ada_b", "og_w", "og_b")}, F16)}
            p["s_to_token"] = p["s_to_token"].astype(F16)
        if pair16:
            pp = p["pair"]
            pp["proj"], pp["bias_w"], pp["bias_b"] = (pp[k].astype(F16) for k in ("proj", "bias_w", "bias_b"))
            pp["transitions"] = [_cast(b, F16, ("a", "b", "out")) for b in pp["transitions"]]
        if atom16:
            for blks in ("enc_blocks", "dec_blocks"):
                p[blks] = [_cast(b, F16) for b in p[blks]]
            p["a2t"], p["t2a"] = p["a2t"].astype(F16), p["t2a"].astype(F16)
        self.bias_dtype = F16 if (tok16 or pair16) else F32
        self._step = mx.compile(self._step_fn) if compile else self._step_fn
        self._cond_bytes = 2 if cond16 else 4

    def _step_fn(self, p, ctx, cond, x, rot, shift, eps, coef):
        t_hat, eps_std, k = coef[0:1], coef[1], coef[2]
        m = ctx["atom_mask"][..., None]
        mean = (x * m).sum(axis=1, keepdims=True) / mx.maximum(m.sum(axis=1, keepdims=True), 1.0)
        x = (x - mean) @ rot + shift
        x_noisy = x + eps_std * eps
        x_den, a = denoise(p, ctx, cond, x_noisy, t_hat, self.sigma, self.n_atom_heads, self.n_token_heads)
        x_noisy = rigid_align(x_noisy, x_den, ctx["atom_mask"], self.opts["horn"])
        return x_noisy + k * (x_noisy - x_den), a

    @torch.no_grad()
    def _context(self, kw):
        """base._context, but with this module's pair_biases (precision per options)."""
        return _rebind(base.MlxDiffusion._context,
                       pair_biases=lambda p, z, r, h: pair_biases(p, z, r, h, self.bias_dtype))(self, kw)

    @torch.no_grad()
    def sample(self, **kw):
        """base.sample with this module's noise_conditioning (and its byte budget)."""
        cw = self._cond_width
        self._cond_width = cw * self._cond_bytes // 4  # COND_BUDGET counts fp32 bytes
        try:
            return _rebind(base.MlxDiffusion.sample, noise_conditioning=noise_conditioning)(self, **kw)
        finally:
            self._cond_width = cw


def _rebind(fn, **overrides):
    """``fn`` (unwrapped from torch.no_grad) re-created with some of its module globals replaced."""
    import types
    f = getattr(fn, "__wrapped__", fn)
    return types.FunctionType(f.__code__, {**f.__globals__, **overrides}, f.__name__, f.__defaults__, f.__closure__)
