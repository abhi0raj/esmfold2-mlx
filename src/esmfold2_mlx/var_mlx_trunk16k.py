"""Variant: the fp16 MLX folding trunk with hand-written Metal kernels for its
memory-bound steps (gate -> channels-first operands, transpose + norm_mix LayerNorm,
residual + next LayerNorm; transition residual via addmm). Weights, fp16 scheme and
1/256 contraction pre-scale as in mlx_trunk.py; the kernels are in mlx_trunk_kernels.py.

Usage:
    python tools/bench_e2e.py --variants fast_trunk,mlx_trunk16k,mlx_diffusion16,adaptive_loops
"""

import mlx.core as mx

from . import mlx_trunk_kernels
from .var_mlx_trunk import MLXFoldingTrunk


def apply(model):
    trunk = MLXFoldingTrunk(model.folding_trunk, dtype=mx.float16)
    trunk._fn = mlx_trunk_kernels.make_trunk(mx.float16)  # same call signature as mlx_trunk.make_trunk's
    model.folding_trunk = trunk
