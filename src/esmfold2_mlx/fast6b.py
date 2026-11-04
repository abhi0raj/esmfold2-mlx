"""Load biohub/ESMFold2-Fast (the release ESMFold2 with ESMC-6B bundled) on a 32 GB Mac.

The checkpoint is 26 GB of fp32 in six safetensors shards, 25.4 GB of it ESMC.
Upstream's from_pretrained reads every shard into one dict and only then casts
ESMC to bf16, which needs the full 26 GB in RAM plus the cast copy, and bf16
matmul asserts on M1. Here the model is built on the meta device and filled from
the memory-mapped shards one ESMC layer at a time: each layer's 13 tensors are
read, cast, packed into upstream's fused layout and moved to the GPU before the
next layer is touched, so peak RAM is the finished model plus ~0.3 GB.

ESMC backends (``esmc=``):
  "torch16"  upstream EsmcModel on MPS, fp16 weights and activations
  "mlx16"    mlx_esmc.MLXEsmc: fp16 weights, fp32 residual stream (see mlx_esmc.py)
  "mlx8", "mlx4"  the same with 8-/4-bit weights (mx.quantize, group size 64)
  "none"     no language model (tests of the fold stack only)

Fixes the release model needs on MPS, applied by ``load_6b``:
  - its forward has no ``seed`` argument (the experimental one does, and every tool
    here passes it), so ``seed=`` is accepted and seeds the whole forward: unlike
    the experimental model, the release trunk draws random numbers too (random
    initial pair state, per-loop LM dropout), not just the diffusion sampler;
  - ESMC hidden states are cast to fp32 before the fp32 LanguageModelShim (upstream
    relies on CUDA autocast for the mixed dtypes; there is none on MPS).

Usage:
    from esmfold2_mlx.fast6b import load_6b
    model = load_6b(esmc="mlx8")
    out = model.infer_protein(seq, num_loops=3, num_sampling_steps=20, num_diffusion_samples=1, seed=0)
    print(float(out["plddt"].mean()), float(out["ptm"]))
"""

import json
from functools import wraps
from pathlib import Path

from .paths import cache_dir

import torch
from accelerate import init_empty_weights
from esm.models.esmc.checkpoint_layout import published_to_native
from esm.models.esmfold2.config import EsmFold2Config, default_module_flags
from esm.models.esmfold2.hf_checkpoint import hf_state_dict_to_native
from esm.models.esmfold2.layers import _seed_context
from esm.models.esmfold2.model import EsmFold2Model
from esm.models.hub import resolve_model_dir
from safetensors import safe_open

from . import mps_compat  # noqa: F401

CHECKPOINT = "biohub/ESMFold2-Fast"
ESMC_LAYERS = 80
ESMC_MODES = ("torch16", "mlx16", "mlx8", "mlx4", "none")
CACHE = cache_dir()


class Shards:
    """Lazy access to the checkpoint's tensors by name (safetensors mmap; nothing
    is read until asked for)."""

    def __init__(self, name=CHECKPOINT):
        self.dir = Path(resolve_model_dir(name, local_files_only=True))
        self.weight_map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._files = {}

    def get(self, key):
        shard = self.weight_map[key]
        if shard not in self._files:
            self._files[shard] = safe_open(str(self.dir / shard), framework="pt")
        return self._files[shard].get_tensor(key)

    def keys(self, prefix=""):
        return [k for k in self.weight_map if k.startswith(prefix)]

    def esmc_layer(self, i):
        """Layer i's tensors under their published (HF) names, fp32 on CPU."""
        return {k: self.get(k) for k in self.keys(f"esmc.layers.{i}.")}


def config(shards):
    return EsmFold2Config.from_pretrained(str(shards.dir), **default_module_flags(shards.dir))


def _assign(module, state):
    """Load a partial state dict into a meta-initialised module, in place."""
    res = module.load_state_dict(state, strict=False, assign=True)
    assert not res.unexpected_keys, res.unexpected_keys[:5]


def load_torch_esmc(shards, device="mps", dtype=torch.float16, esmc_config=None):
    """Upstream EsmcModel with weights in ``dtype`` on ``device``, filled layer by layer."""
    from esm.models.esmc import EsmcModel

    with init_empty_weights():
        esmc = EsmcModel(esmc_config or config(shards).esmc_config)

    def cast(sd):
        return {k: v.to(dtype=dtype).to(device) for k, v in sd.items()}

    _assign(esmc, cast({"embed.weight": shards.get("esmc.embed_tokens.weight"),
                        "transformer.norm.weight": shards.get("esmc.norm.weight")}))
    for i in range(ESMC_LAYERS):
        # cast before packing q|k|v and gate|up, so the fp32 copies die with the layer
        layer = {k[len("esmc."):]: v for k, v in cast(shards.esmc_layer(i)).items()}
        _assign(esmc, published_to_native(layer))
    esmc.to(device)  # recomputes RoPE inv_freq (fp32) on the device
    return esmc.eval().requires_grad_(False)


def load_6b(esmc="mlx8", device="mps", trunk_dtype=torch.float32, name=CHECKPOINT):
    """The release model with its trunk, heads and diffusion in ``trunk_dtype`` on
    ``device`` and ESMC per ``esmc`` (see module docstring)."""
    assert esmc in ESMC_MODES, esmc
    shards = Shards(name)
    with init_empty_weights():
        model = EsmFold2Model(config(shards))

    # Everything but ESMC: 0.75 GB fp32, converted from the HF layout in one go.
    rest = {k: shards.get(k) for k in shards.weight_map if not k.startswith("esmc.")}
    rest = {k: v.to(device=device, dtype=trunk_dtype) for k, v in hf_state_dict_to_native(rest).items()}
    _assign(model, rest)
    del rest

    if esmc == "torch16":
        model.esmc = load_torch_esmc(shards, device, torch.float16, model.config.esmc_config)
    else:
        model.esmc = None
        if esmc.startswith("mlx"):
            from .mlx_esmc import MLXEsmc
            model.esmc = MLXEsmc.from_shards(shards, bits={"mlx16": None, "mlx8": 8, "mlx4": 4}[esmc])

    allowed = tuple(EsmFold2Model._keys_to_ignore_on_load_missing)
    import re
    meta = [n for n, p in model.named_parameters() if p.is_meta]
    unexpected = [n for n in meta if not any(re.search(pat, n) for pat in allowed)]
    assert not unexpected, f"parameters never loaded: {unexpected[:10]}"
    model._materialize_uninitialized(device="cpu")  # the 3 unused confidence-head modules
    model.to(device)  # non-persistent buffers (ESMC RoPE recomputes inv_freq on move)
    model.eval().requires_grad_(False)
    model._fast6b = {"esmc": esmc, "name": name}
    return patch_release(model)


def patch_release(model):
    """The two MPS fixes from the module docstring; idempotent."""
    if getattr(model, "_fast6b_patched", False):
        return model
    forward = model.forward

    @wraps(forward)
    def seeded_forward(*args, seed=None, **kwargs):
        with _seed_context(seed):
            return forward(*args, **kwargs)

    model.forward = seeded_forward
    lm = model._compute_lm_hidden_states
    model._compute_lm_hidden_states = lambda *a, **k: lm(*a, **k).float()
    model._fast6b_patched = True
    return model
