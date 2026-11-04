"""Variant: the MLX diffusion sampler with the token transformer, noise conditioning and
pair biases in fp16 (mlx_diffusion16 with tok16 + cond16 + pair16). Coordinates, residual
streams, LayerNorms, the alignment and the Euler update stay fp32.

Usage:
    python tools/bench_e2e.py --model 6B --variants fast_trunk,6b_mlx_trunks16k,mlx_diffusion16,6b_adaptive
"""

from .mlx_diffusion16 import MlxDiffusion

OPTS = dict(tok16=True, cond16=True, pair16=True)


def apply(model, **opts) -> None:
    head = model.structure_head
    upstream = head.sample
    mlx_head = MlxDiffusion(head, **(opts or OPTS))

    def sample(**kw):
        if (kw["s_inputs"].shape[0] == 1 and kw.get("num_diffusion_samples", 1) == 1
                and not kw.get("return_atom_repr", False)):
            return mlx_head.sample(**kw)
        return upstream(**kw)

    head.sample = sample
