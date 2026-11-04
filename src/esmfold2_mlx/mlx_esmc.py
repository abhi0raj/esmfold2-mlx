"""ESMC (the ESMFold2 language model) in MLX, with fp16 or 8-/4-bit quantised weights.

Drop-in for ``model.esmc``: ``compute_lm_hidden_states`` calls
``esmc(input_ids=..., sequence_id=..., output_hidden_states=True)`` and reads
``.hidden_states`` ([n_layers + 1, B, L, D]: every block's input plus the final
LayerNorm output), so this is a torch ``nn.Module`` with that forward that
returns the hidden states as an fp32 MPS tensor.

Each block is upstream's EsmcUnifiedTransformerBlock: pre-LN attention with
QK-LayerNorm (over all heads, weight only) and RoPE (NeoX half rotation), then
pre-LN SwiGLU, both residuals divided by sqrt(n_layers / 36) (1.49 for 80 layers).

Precision: weights are fp16 (``bits=None``) or quantised with ``mx.quantize``
(group size 64, fp16 scales and biases); matmuls take fp16 inputs and accumulate
in fp32. The residual stream stays fp32 (``residual=mx.float32``): it is the sum
of 80 blocks' outputs and also the hidden states the fold reads, so keeping it in
fp32 costs one cast per matmul input and avoids compounding fp16 rounding (and
the fp16 range limit) over 80 layers. ``residual=mx.float16`` is the all-fp16
scheme, for comparison.

Converted weights are cached in ``paths.cache_dir()`` (``esmc_q8.safetensors`` etc.),
so only the first load streams the 25 GB of fp32 shards.

Usage:
    from esmfold2_mlx.fast6b import Shards
    from esmfold2_mlx.mlx_esmc import MLXEsmc
    esmc = MLXEsmc.from_shards(Shards(), bits=8)   # None = fp16
    hs = esmc(input_ids=ids, sequence_id=seq_id).hidden_states
"""

import math
import time
from pathlib import Path

from .paths import cache_dir
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch
from torch import nn

GROUP = 64
EPS = 1e-5
CACHE = cache_dir()
LINEARS = ("wqkv", "wo", "w1", "w2")


def _mx(t, dtype=mx.float16):
    return mx.array(t.detach().float().cpu().numpy()).astype(dtype)


def layer_params(hf, i, bits):
    """One block's weights from its published (HF-named, fp32) tensors.
    Linear weights keep torch's [out, in] layout (x @ W.T; quantised along ``in``)."""
    g = lambda name: hf[f"esmc.layers.{i}.{name}"]  # noqa: E731
    lin = {
        "wqkv": torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")]),
        "wo": g("self_attn.o_proj.weight"),
        # gate rows first: upstream applies silu(x1) * x2 to fc1 = [gate | up]
        "w1": torch.cat([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")]),
        "w2": g("mlp.down_proj.weight"),
    }
    p = {"ln1_w": _mx(g("input_layernorm.weight"), mx.float32), "ln1_b": _mx(g("input_layernorm.bias"), mx.float32),
         "q_ln": _mx(g("self_attn.q_norm.weight"), mx.float32), "k_ln": _mx(g("self_attn.k_norm.weight"), mx.float32),
         "ln2_w": _mx(g("post_attention_layernorm.weight"), mx.float32),
         "ln2_b": _mx(g("post_attention_layernorm.bias"), mx.float32)}
    for name, w in lin.items():
        if bits is None:
            p[name] = _mx(w)
        else:
            # quantise from fp32, not from an fp16 copy
            wq, scales, biases = mx.quantize(_mx(w, mx.float32), group_size=GROUP, bits=bits)
            p[name], p[name + "_s"], p[name + "_b"] = wq, scales.astype(mx.float16), biases.astype(mx.float16)
    return p


def _linear(p, name, x, bits):
    if bits is None:
        return x @ p[name].T
    return mx.quantized_matmul(x, p[name], p[name + "_s"], p[name + "_b"], transpose=True, group_size=GROUP, bits=bits)


def block_forward(p, x, mask, n_heads, res_scale, bits):
    """EsmcUnifiedTransformerBlock.forward; ``x`` in the residual dtype."""
    B, L, D = x.shape
    hd = D // n_heads
    h = mx.fast.layer_norm(x.astype(mx.float32), p["ln1_w"], p["ln1_b"], EPS).astype(mx.float16)
    q, k, v = mx.split(_linear(p, "wqkv", h, bits), 3, axis=-1)
    # QK-LayerNorm over the full width (all heads), in fp32
    q = mx.fast.layer_norm(q.astype(mx.float32), p["q_ln"], None, EPS)
    k = mx.fast.layer_norm(k.astype(mx.float32), p["k_ln"], None, EPS)
    heads = lambda t: t.reshape(B, L, n_heads, hd).transpose(0, 2, 1, 3)  # noqa: E731
    q = mx.fast.rope(heads(q), hd, traditional=False, base=10000.0, scale=1.0, offset=0).astype(mx.float16)
    k = mx.fast.rope(heads(k), hd, traditional=False, base=10000.0, scale=1.0, offset=0).astype(mx.float16)
    o = mx.fast.scaled_dot_product_attention(q, k, heads(v), scale=hd ** -0.5, mask=mask)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, D)
    x = x + (_linear(p, "wo", o, bits) / res_scale).astype(x.dtype)
    h = mx.fast.layer_norm(x.astype(mx.float32), p["ln2_w"], p["ln2_b"], EPS).astype(mx.float16)
    x1, x2 = mx.split(_linear(p, "w1", h, bits), 2, axis=-1)
    y = x1 * mx.sigmoid(x1) * x2  # silu(x1) * x2, fp16
    return x + (_linear(p, "w2", y, bits) / res_scale).astype(x.dtype)


class MLXEsmc(nn.Module):
    def __init__(self, embed, layers, norm_w, bits, n_heads=40, residual=mx.float32):
        super().__init__()
        self.embed, self.layers, self.norm_w = embed, layers, norm_w
        self.bits, self.n_heads, self.residual = bits, n_heads, residual
        self.res_scale = math.sqrt(len(layers) / 36)
        self._block = mx.compile(lambda p, x, m: block_forward(p, x, m, n_heads, self.res_scale, bits))
        self._block_nomask = mx.compile(lambda p, x: block_forward(p, x, None, n_heads, self.res_scale, bits))

    # ---- construction --------------------------------------------------------

    @classmethod
    def from_shards(cls, shards, bits=8, cache=True, **kw):
        """Convert from the fp32 checkpoint (streamed layer by layer), or load the cache."""
        path = CACHE / f"esmc_{'f16' if bits is None else f'q{bits}'}.safetensors"
        if cache and path.exists():
            return cls.from_cache(path, bits, **kw)
        t0 = time.time()
        layers = []
        for i in range(80):
            layers.append(layer_params(shards.esmc_layer(i), i, bits))
            mx.eval(layers[-1])
        embed = _mx(shards.get("esmc.embed_tokens.weight"), mx.float32)
        norm_w = _mx(shards.get("esmc.norm.weight"), mx.float32)
        print(f"[mlx_esmc] converted ESMC ({'fp16' if bits is None else f'{bits}-bit'}) in {time.time() - t0:.0f}s",
              flush=True)
        if cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            flat = {"embed": embed, "norm_w": norm_w}
            for i, p in enumerate(layers):
                flat.update({f"layers.{i}.{k}": v for k, v in p.items()})
            mx.save_safetensors(str(path), flat)
        return cls(embed, layers, norm_w, bits, **kw)

    @classmethod
    def from_cache(cls, path, bits, **kw):
        flat = mx.load(str(path))
        n = 1 + max(int(k.split(".")[1]) for k in flat if k.startswith("layers."))
        layers = [{} for _ in range(n)]
        for k, v in flat.items():
            if k.startswith("layers."):
                _, i, name = k.split(".", 2)
                layers[int(i)][name] = v
        return cls(flat["embed"], layers, flat["norm_w"], bits, **kw)

    def nbytes(self):
        return sum(v.nbytes for p in self.layers for v in p.values()) + self.embed.nbytes + self.norm_w.nbytes

    # ---- forward ---------------------------------------------------------------

    # nn.Module API the fold calls on its ESMC; nothing to do here.
    def to(self, *a, **k):
        return self

    def hidden_states(self, input_ids, sequence_id=None):
        """mx [n_layers + 1, B, L, D] fp32: each block's input, then the final LayerNorm."""
        ids = mx.array(input_ids.detach().cpu().numpy().astype(np.int32))
        x = self.embed[ids].astype(self.residual)
        mask = None
        if sequence_id is not None:
            sid = sequence_id.detach().cpu()
            # upstream: token i attends to j iff same chain id (padding = -1 is its own "chain")
            allowed = (sid[:, :, None] == sid[:, None, :])
            if not bool(allowed.all()):
                mask = mx.array(allowed.numpy())[:, None]
        out = []
        for p in self.layers:
            out.append(x.astype(mx.float32))
            x = self._block_nomask(p, x) if mask is None else self._block(p, x, mask)
            mx.async_eval(x)  # bounds the lazy graph to one block
        out.append(mx.fast.layer_norm(x.astype(mx.float32), self.norm_w, None, EPS))
        hs = mx.stack(out)
        mx.eval(hs)
        return hs

    def forward(self, input_ids=None, sequence_id=None, output_hidden_states=True, **_):
        from .var_mlx_trunk import to_torch
        hs = self.hidden_states(input_ids, sequence_id)
        device = input_ids.device
        return SimpleNamespace(hidden_states=to_torch(hs, device) if device.type == "mps"
                               else torch.from_numpy(np.array(hs)))
