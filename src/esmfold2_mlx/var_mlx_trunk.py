"""Variant: run the 24-block folding trunk in MLX (fp32) inside real folds.

``model.folding_trunk`` becomes a thin torch ``nn.Module`` (so forward hooks
still work) that hands the pair grid to MLX, runs the compiled MLX trunk from
``mlx_trunk`` and hands the result back as an fp32 MPS tensor.

Handoff: MLX -> torch is zero-copy (DLPack gives an MPS tensor on the same
Metal buffer). torch -> MLX via DLPack is one copy, but MLX reads the buffer
without waiting on torch's queue, so torch must be synchronized first. Both
are a few ms at L=300 (vs ~2.7 s per trunk pass).

Usage:
    python tools/bench_e2e.py --variants fast_trunk,mlx_trunk
"""

import mlx.core as mx
import numpy as np
import torch
from torch import nn

from . import mlx_trunk


class MLXFoldingTrunk(nn.Module):
    def __init__(self, folding_trunk, dtype=mx.float32):
        super().__init__()
        self.params = mlx_trunk.trunk_params(folding_trunk, dtype)
        # Whole-trunk compile: first call at a new length costs only ~20-80 ms
        # extra, so recompiling per protein is fine.
        self._fn = mlx_trunk.make_trunk(dtype)

    # FoldingTrunk API the model calls on its trunk; MLX ignores both.
    def set_chunk_size(self, chunk_size):
        pass

    def set_kernel_backend(self, backend):
        pass

    def forward(self, pair, pair_attention_mask=None):
        # A single-chain fold has an all-ones mask; skipping it saves a multiply per tri-mul.
        mask = None
        if pair_attention_mask is not None and not bool(pair_attention_mask.all()):
            mask = to_mlx(pair_attention_mask.float())
        out = self._fn(self.params, to_mlx(pair.float()), mask)
        mx.eval(out)
        return to_torch(out, pair.device)


def to_mlx(t):
    t = t.detach().contiguous()
    if t.device.type == "mps":
        torch.mps.synchronize()  # MLX copies straight from the buffer; torch's writes must have landed
        return mx.from_dlpack(t)
    return mx.array(t.numpy())


def to_torch(a, device):
    if device.type == "mps":
        return torch.from_dlpack(a)  # zero-copy: shares the MLX Metal buffer
    return torch.from_numpy(np.array(a)).to(device)


def apply(model):
    model.folding_trunk = MLXFoldingTrunk(model.folding_trunk)
