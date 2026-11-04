"""Variant (6B release model): every folding-trunk stack in MLX fp16 with hand-written Metal kernels.

The release model has four FoldingTrunk stacks: the main trunk (24 blocks), lm_encoder (4),
the parcae coda (2) and the confidence head's trunk (4). Each runs in MLX fp16 with the 1/256
contraction pre-scale, through mlx_trunk_kernels.make_trunk (gate -> channels-first operands,
transpose + norm_mix LayerNorm, residual + next LayerNorm, transition residual via addmm), as
mlx_trunk16k does for the experimental checkpoints. Same weights.

Usage:
    python tools/bench_e2e.py --model 6B --variants fast_trunk,6b_mlx_trunks16k,mlx_diffusion16,6b_adaptive
"""

import mlx.core as mx

from . import mlx_trunk_kernels
from .var_mlx_trunk import MLXFoldingTrunk


def _wrap(trunk):
    if not isinstance(trunk, MLXFoldingTrunk):
        trunk = MLXFoldingTrunk(trunk, dtype=mx.float16)
    assert trunk.params[0]["w3"].dtype == mx.float16, "the kernels assume the fp16 trunk scheme"
    trunk._fn = mlx_trunk_kernels.make_trunk(mx.float16)  # same call signature as mlx_trunk.make_trunk's
    return trunk


def apply(model):
    model.folding_trunk = _wrap(model.folding_trunk)
    if model.lm_encoder is not None:
        model.lm_encoder = _wrap(model.lm_encoder)
    model.parcae_coda = _wrap(model.parcae_coda)
    model.confidence_head.folding_trunk = _wrap(model.confidence_head.folding_trunk)
