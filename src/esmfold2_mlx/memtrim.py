"""Memory trimming for the default fold path (mainly the 6B model). None of
these options changes the arithmetic: the same weights and the same kernels run,
only where the bytes live between uses changes.

Options (``apply(model, **opts)``; ``parse("a,b")``; ``"all"`` = every option):

  empty_cache      torch.mps.empty_cache() once after loading and after every fold:
                   the MPS caching allocator otherwise keeps the freed fp32 copies of
                   the torch modules that the MLX variants replaced (1.26 GB for 6B).
  drop_torch_diff  free the torch weights of structure_head.diffusion_module (0.49 GB
                   fp32), which mlx_diffusion has already copied. The torch sampler is
                   only the fallback for calls MLX does not cover (several diffusion
                   samples, batched inputs); such a call reloads the weights from the
                   checkpoint first (``_restore``), so behaviour is unchanged.
  mlx_cache        mx.clear_cache() after every fold, so MLX's buffer cache does not
                   hold the fold's freed activations while the process idles.
  mlx_limit        mx.set_cache_limit(1 GB): bounds MLX's cache during folds too (alternative
                   to mlx_cache; not in "all").
  stream_esmc      (6B, MLX ESMC from the converted-weights cache) do not keep the 6.7 GB
                   of 8-bit ESMC weights resident between folds: each fold re-opens the
                   cached safetensors lazily (mx.load reads a tensor only when it is
                   evaluated) and drops each block's weights as soon as that block has
                   run. Between folds those bytes live in the file cache, which macOS can
                   drop without writing to swap. Trades a re-read per fold for a
                   footprint ~6.7 GB lower.

Usage:
    from esmfold2_mlx import memtrim
    memtrim.apply(model, **memtrim.parse("empty_cache,drop_torch_diff,mlx_cache"))
    ... fold ...
    memtrim.after_fold(model, **opts)   # or use memtrim.wrap(model, **opts) to do it automatically
"""

import gc

import mlx.core as mx
import torch

STREAM_WINDOW = 4  # ESMC blocks in flight with stream_esmc (each ~84 MB at 8 bit)
OPTIONS = ("empty_cache", "drop_torch_diff", "mlx_cache", "stream_esmc", "mlx_limit")
ALL = ("empty_cache", "drop_torch_diff", "mlx_cache", "stream_esmc")  # "all"; mlx_limit is an alternative to mlx_cache
MLX_CACHE_LIMIT = 1 << 30  # mlx_limit: bytes of freed buffers MLX may keep cached, also *during* a fold


def parse(spec):
    spec = (spec or "none").strip()
    if spec == "none":
        return {k: False for k in OPTIONS}
    names = ALL if spec == "all" else [s for s in spec.split(",") if s]
    bad = [n for n in names if n not in OPTIONS]
    assert not bad, f"unknown memtrim options {bad}; known {OPTIONS}"
    return {k: (k in names) for k in OPTIONS}


# ---- drop_torch_diff -----------------------------------------------------------

def _drop_torch_diffusion(model):
    dm = model.structure_head.diffusion_module
    if getattr(model, "_memtrim_diff_dropped", False):
        return
    names = [n for n, _ in dm.named_parameters()]
    for mod in dm.modules():
        for k, p in list(mod._parameters.items()):
            if p is not None:
                mod._parameters[k] = torch.nn.Parameter(torch.empty_like(p, device="meta"), requires_grad=False)
    model._memtrim_diff_dropped = True
    model._memtrim_diff_names = names

    head = model.structure_head
    mlx_sample = head.sample  # the MLX diffusion variant's dispatcher

    def sample(**kw):
        uses_mlx = (kw["s_inputs"].shape[0] == 1 and kw.get("num_diffusion_samples", 1) == 1
                    and not kw.get("return_atom_repr", False))
        if not uses_mlx and getattr(model, "_memtrim_diff_dropped", False):
            _restore_torch_diffusion(model)
        return mlx_sample(**kw)

    head.sample = sample


def _restore_torch_diffusion(model):
    """Reload structure_head.diffusion_module weights from the checkpoint (fallback path)."""
    dm = model.structure_head.diffusion_module
    device = next(p for p in model.parameters() if not p.is_meta).device
    if hasattr(model, "_fast6b"):
        from esm.models.esmfold2.hf_checkpoint import hf_state_dict_to_native

        from .fast6b import Shards
        sh = Shards(model._fast6b["name"])
        raw = {k: sh.get(k) for k in sh.weight_map if not k.startswith("esmc.")}
        full = hf_state_dict_to_native(raw)
    else:
        from esm.models.hub import read_safetensors_dir, resolve_model_dir
        cls = type(model)
        local = resolve_model_dir(model._memtrim_ckpt)
        raw = cls._normalize_checkpoint_layout(read_safetensors_dir(local))
        full = cls._adapt_checkpoint_keys(raw, set(model.state_dict().keys()))
    prefix = "structure_head.diffusion_module."
    sd = {k[len(prefix):]: v.to(device=device, dtype=torch.float32) for k, v in full.items() if k.startswith(prefix)}
    res = dm.load_state_dict(sd, strict=False, assign=True)
    assert not res.missing_keys, res.missing_keys[:5]
    model._memtrim_diff_dropped = False


# ---- stream_esmc -----------------------------------------------------------------

def _stream_esmc(model):
    from .mlx_esmc import CACHE, MLXEsmc
    esmc = getattr(model, "esmc", None)
    if not isinstance(esmc, MLXEsmc) or getattr(esmc, "_memtrim_stream", False):
        return
    path = CACHE / f"esmc_{'f16' if esmc.bits is None else f'q{esmc.bits}'}.safetensors"
    assert path.exists(), f"stream_esmc needs the cached weights at {path}"
    n_layers = len(esmc.layers)
    esmc.layers = None  # drop the resident (possibly still lazy) copies
    esmc._memtrim_stream = True

    def layers_once():
        flat = mx.load(str(path))  # lazy: nothing is read until a tensor is evaluated
        out = [{} for _ in range(n_layers)]
        for k, v in flat.items():
            if k.startswith("layers."):
                _, i, name = k.split(".", 2)
                out[int(i)][name] = v
        return out

    def hidden_states(input_ids, sequence_id=None):
        import numpy as np
        ids = mx.array(input_ids.detach().cpu().numpy().astype(np.int32))
        x = esmc.embed[ids].astype(esmc.residual)
        mask = None
        if sequence_id is not None:
            sid = sequence_id.detach().cpu()
            allowed = (sid[:, :, None] == sid[:, None, :])
            if not bool(allowed.all()):
                mask = mx.array(allowed.numpy())[:, None]
        layers = layers_once()
        out, pending = [], []
        for i in range(n_layers):
            p = layers[i]
            layers[i] = None  # this block's weights die once its outputs are computed
            out.append(x.astype(mx.float32))
            x = esmc._block_nomask(p, x) if mask is None else esmc._block(p, x, mask)
            mx.async_eval(x)
            del p
            # async_eval returns at once, so without a bound Python would schedule (and read)
            # all 80 blocks' weights ahead of the GPU. Waiting on the block STREAM_WINDOW back
            # keeps at most ~STREAM_WINDOW blocks of weights in memory while the next reads overlap.
            pending.append(x)
            if len(pending) > STREAM_WINDOW:
                mx.eval(pending.pop(0))
        out.append(mx.fast.layer_norm(x.astype(mx.float32), esmc.norm_w, None, 1e-5))
        hs = mx.stack(out)
        mx.eval(hs)
        return hs

    esmc.hidden_states = hidden_states
    # embed / norm_w are small; force them resident now
    mx.eval(esmc.embed, esmc.norm_w)


# ---- driver ----------------------------------------------------------------------

def apply(model, empty_cache=False, drop_torch_diff=False, mlx_cache=False, stream_esmc=False, mlx_limit=False,
          checkpoint=None):
    """``checkpoint``: Hub id / dir the (non-6B) model was loaded from, so the torch sampler's
    weights can be reloaded if a call needs the fallback; without it drop_torch_diff is skipped."""
    if drop_torch_diff and not hasattr(model, "_fast6b"):
        model._memtrim_ckpt = checkpoint
        drop_torch_diff = checkpoint is not None
    if drop_torch_diff:
        _drop_torch_diffusion(model)
    if stream_esmc:
        _stream_esmc(model)
    if mlx_limit:
        mx.set_cache_limit(MLX_CACHE_LIMIT)
    gc.collect()
    if empty_cache:
        torch.mps.empty_cache()
    if mlx_cache:
        mx.clear_cache()
    return model


def after_fold(model, empty_cache=False, mlx_cache=False, **_):
    if empty_cache:
        torch.mps.empty_cache()
    if mlx_cache:
        mx.clear_cache()


def wrap(model, checkpoint=None, **opts):
    """Apply the options and run after_fold automatically after every infer_protein call."""
    apply(model, checkpoint=checkpoint, **opts)
    infer = model.infer_protein

    def infer_protein(*a, **k):
        out = infer(*a, **k)
        after_fold(model, **opts)
        return out

    model.infer_protein = infer_protein
    return model
