"""Composable speed variants: each is a module ``var_<name>.py`` with ``apply(model)``.

``fast_trunk`` is built in. Variants are applied in the order given, so later
ones can build on earlier ones (e.g. ``fast_trunk,mlx_trunk16k``).
"""

import importlib

import torch
from esm.models.esmfold2 import EsmFold2ExperimentalModel

from . import fast_trunk
from . import mps_compat  # noqa: F401

# Same fold architecture; they differ only in the ESMC language model feeding it.
MODELS = {
    "300M": "biohub/ESMFold2-Experimental-Fast-base300M-step1500k",
    "600M": "biohub/ESMFold2-Experimental-Fast-base600M-step1500k",
}

# Defaults shared by fold.py, regression_check.py and benchmark.py.
DEFAULT_MODEL = "300M"
# Fastest stack that passed the accuracy gate (Metal-kernel
# trunk + adaptive recycling; worst held-out dTM -0.0029). mlx_* need Apple Silicon.
DEFAULT_STACK = "fast_trunk,mlx_trunk16k,mlx_diffusion16,adaptive_loops"

# The release ESMFold2-Fast (ESMC-6B, different trunk) is loaded by fast6b.py instead;
# the suffix picks its ESMC backend (see fast6b.ESMC_MODES).
MODELS_6B = {"6B": "mlx8", "6B-q4": "mlx4", "6B-f16": "mlx16", "6B-torch16": "torch16"}
# The 6B model's recycling ("parcae") differs from the experimental checkpoints', so
# it has its own validated stack; adaptive_loops was tuned on 300M only.
DEFAULT_STACK_6B = "fast_trunk,6b_mlx_trunks16k,mlx_diffusion16,6b_adaptive"


def default_stack(model):
    """The validated speed stack for ``model``."""
    return DEFAULT_STACK_6B if model in MODELS_6B else DEFAULT_STACK


# Memory options (memtrim.py): same arithmetic, ~10-15 GB less peak footprint for 6B.
DEFAULT_MEMTRIM = "empty_cache,drop_torch_diff,mlx_cache"
DEFAULT_MEMTRIM_6B = DEFAULT_MEMTRIM + ",stream_esmc"


def default_memtrim(model):
    return DEFAULT_MEMTRIM_6B if model in MODELS_6B else DEFAULT_MEMTRIM


def apply_memtrim(model, model_id, spec="default"):
    """Shared by fold.py and the checking tools so they test exactly what ships."""
    if spec == "none":
        return model
    from . import memtrim
    spec = default_memtrim(model_id) if spec == "default" else spec
    memtrim.wrap(model, checkpoint=MODELS.get(model_id, model_id), **memtrim.parse(spec))
    return model


def load_model(model=DEFAULT_MODEL, device="mps", lm_dtype=torch.float32):
    """An experimental checkpoint ("300M", "600M" or a Hub id). Upstream forces the
    ESMC language model to bf16, whose matmul trips a Metal assertion on M1, so
    it is loaded in ``lm_dtype`` (fp32 by default). "6B..." ids go to fast6b.load_6b."""
    if model in MODELS_6B:
        from .fast6b import load_6b
        return load_6b(esmc=MODELS_6B[model], device=device)

    def finalize_esmc(self):
        self.esmc = self.esmc.to(device=self.device, dtype=lm_dtype).eval()

    EsmFold2ExperimentalModel.finalize_esmc = finalize_esmc
    return EsmFold2ExperimentalModel.from_pretrained(MODELS.get(model, model), device=device).eval()


def apply_variants(model, names):
    for name in [n for n in names if n]:
        if name == "fast_trunk":
            fast_trunk.apply(model)
        else:
            try:
                module = importlib.import_module(f"{__package__}.var_{name}")
            except ModuleNotFoundError as e:
                if e.name != f"{__package__}.var_{name}":
                    raise
                raise ValueError(f"unknown speed variant {name!r}: this release ships the default stacks only") from None
            module.apply(model)
    return model
