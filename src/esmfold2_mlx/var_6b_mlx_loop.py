"""Variant (6B release model): the parcae recycling loop in MLX, with an optional
early exit once the predicted distances stop changing.

The release model's loop (EsmFold2Model._run_one_loop) does, per step,

    lm_z_i = dropout(lm_z, p=0.25)          # forced on at inference; torch RNG
    u      = z_init + lm_encoder(lm_z_i)    # 4-block FoldingTrunk
    z      = a * z + LN(u) @ B^T            # "parcae" linear recurrence, fp32
    z      = folding_trunk(z)               # 24 blocks

and after the loop forward runs readout -> coda (2 blocks) -> distogram / diffusion.

With 6b_mlx_trunks16k every trunk already runs in MLX, but each step hands the pair grid
torch -> MLX -> torch twice (lm_encoder, trunk) with a torch sync each time, and does the
LayerNorm / recurrence in PyTorch. Here ``model._run_one_loop`` (an instance attribute;
forward calls it by name) is replaced: the dropout stays in PyTorch, so the random draws
and their order are exactly upstream's, and its result is the only per-step handoff; the
lm_encoder, LayerNorm, recurrence and trunk run as one compiled MLX step, and z stays in
MLX until the loop ends. Needs the MLX trunks (6b_mlx_trunks16k) applied
first; it calls whatever compiled trunk function they hold.

Early exit (``adaptive=True``): after step k >= MIN_STEPS (and before the last scheduled
step) compare the distogram's expected distance per pair with the previous step's; if the
95th percentile of |change| is below THRESHOLD, return z. That is the same fold as
num_loops = k - 1: the skipped steps draw nothing from the RNG, so the diffusion sees
the same random stream. ``source`` picks what the
distogram is computed from: "coda" = the real output path (readout -> coda -> head; on exit
that coda output is handed to forward, so the check costs nothing at the exit step), or
"readout" = readout -> head, skipping the 2 coda blocks. Calibrated on a separate
20-protein calibration set, never on the held-out set.

Usage:
    python tools/bench_e2e.py --model 6B \
        --variants fast_trunk,6b_mlx_trunks16k,mlx_diffusion16,6b_mlx_loop
"""

from functools import partial

import mlx.core as mx
import torch
import torch.nn.functional as F
from torch import nn

from .var_mlx_trunk import MLXFoldingTrunk, to_mlx, to_torch

# Early-exit policy (set by var_6b_adaptive; off here).
SOURCE = "coda"
THRESHOLD = None
MIN_STEPS = 2
# On exit, still draw the skipped steps' dropout masks (a few ms each, results discarded) so
# the diffusion sees the same random stream as the full-length fold: a converged protein then
# gets the default stack's structure, not a fold with a shifted diffusion seed. False makes
# an exit after k steps identical to a num_loops=k-1 fold instead.
BURN_RNG = True

# 6B distogram: 64 bins over 2-22 A (prepare_input.compute_distogram_conditioning); the
# last bin is open-ended, its centre placed half a bin past 22 A. Only differences of
# expected distance are used, so the exact centres matter little.
_EDGES = mx.linspace(2.0, 22.0, 65)
CENTERS = (_EDGES[:-1] + _EDGES[1:]) / 2


def _mx32(t):
    return mx.array(t.detach().float().cpu().numpy())


def _step(lm_fn, trunk_fn, lm_params, trunk_params, ln_w, ln_b, a, bT, lz, z, z_init, mask):
    refined = lm_fn(lm_params, lz, mask)  # fp32 out
    inj = mx.fast.layer_norm(z_init + refined, ln_w, ln_b, 1e-5)
    z = a * z + inj @ bT
    return trunk_fn(trunk_params, z, mask)


def _expected_distance(r, w, b):
    logits = (r + r.swapaxes(1, 2)) @ w + b
    return mx.softmax(logits, axis=-1) @ CENTERS


def p95(x):
    """95th percentile of a flat array (linear interpolation, as torch.quantile)."""
    x = mx.sort(x.flatten())
    pos = 0.95 * (x.size - 1)
    lo = int(pos)
    hi = min(lo + 1, x.size - 1)
    return x[lo] + (pos - lo) * (x[hi] - x[lo])


class _Cached(nn.Module):
    """Wraps parcae_readout / parcae_coda: when the adaptive loop exits it has already
    computed coda(readout(z)); forward's readout(z) then returns that and coda passes it on."""

    def __init__(self, inner, owner, role):
        super().__init__()
        self.inner = inner
        self.__dict__["_owner"] = owner
        self.role = role

    def set_chunk_size(self, chunk_size):
        if hasattr(self.inner, "set_chunk_size"):
            self.inner.set_chunk_size(chunk_size)

    def set_kernel_backend(self, backend):
        if hasattr(self.inner, "set_kernel_backend"):
            self.inner.set_kernel_backend(backend)

    def forward(self, x, *a, **k):
        o = self._owner
        if self.role == "readout" and o._exit_z is not None and x is o._exit_z:
            return o._exit_out
        if self.role == "coda" and o._exit_out is not None and x is o._exit_out:
            o._exit_z = o._exit_out = None
            return x
        return self.inner(x, *a, **k)


class MLXLoop:
    def __init__(self, model, adaptive=False, threshold=None, source=None, min_steps=None, record=False,
                 burn_rng=None):
        coda = model.parcae_coda.inner if isinstance(model.parcae_coda, _Cached) else model.parcae_coda
        assert isinstance(model.folding_trunk, MLXFoldingTrunk) and isinstance(model.lm_encoder, MLXFoldingTrunk) \
            and isinstance(coda, MLXFoldingTrunk), "apply 6b_mlx_trunks(16k) first"
        self.coda = coda
        self.model = model
        self.adaptive, self.record = adaptive, record
        self.threshold = THRESHOLD if threshold is None else threshold
        self.source = source or SOURCE
        self.min_steps = MIN_STEPS if min_steps is None else min_steps
        self.burn_rng = BURN_RNG if burn_rng is None else burn_rng
        self.ln_w, self.ln_b = _mx32(model.parcae_input_norm.weight), _mx32(model.parcae_input_norm.bias)
        assert abs(model.parcae_input_norm.eps - 1e-5) < 1e-12
        self.readout_wT = _mx32(model.parcae_readout.weight).T
        self.disto_w, self.disto_b = _mx32(model.distogram_head.weight).T, _mx32(model.distogram_head.bias)
        mx.eval(self.ln_w, self.ln_b, self.readout_wT, self.disto_w, self.disto_b)
        self._exit_z = self._exit_out = None
        self.steps_run = 0
        self.history = []
        self.compile()

    def compile(self):
        """(Re)build the compiled step from the trunks' current functions (kernels or not)."""
        m = self.model
        self._step_fn = mx.compile(partial(_step, m.lm_encoder._fn, m.folding_trunk._fn))

    # --- convergence signal -------------------------------------------------
    def _coda_out(self, z, mask):
        r = z @ self.readout_wT
        return self.coda._fn(self.coda.params, r, mask)

    def _signal(self, z, mask, source):
        """Expected distance per pair if the loop stopped at z (and, for "coda", the coda output)."""
        if source == "coda":
            out = self._coda_out(z, mask)
            return _expected_distance(out, self.disto_w, self.disto_b), out
        return _expected_distance(z @ self.readout_wT, self.disto_w, self.disto_b), None

    # --- the loop -------------------------------------------------------------
    def run(self, z, z_init, lm_z, _msa_inputs, pair_mask, a, b_mat, tok_mask, total_steps):
        m = self.model
        assert _msa_inputs is None and m.msa_encoder is None, "MSA input not supported"
        assert lm_z is not None and m.lm_encoder is not None
        cfg = m.config.lm_encoder
        assert getattr(cfg, "per_loop_lm_dropout", False)
        p = cfg.lm_dropout
        dev = z.device
        mask = None
        if pair_mask is not None and not bool(pair_mask.all()):
            mask = to_mlx(pair_mask.float())
        zm, zi = to_mlx(z.float()), to_mlx(z_init.float())
        am, bT = to_mlx(a.float()), to_mlx(b_mat.float()).T
        lm_params, tr_params = m.lm_encoder.params, m.folding_trunk.params
        self.history, self.steps_run = [], 0
        self._exit_z = self._exit_out = None
        prev = {}
        check = self.adaptive or self.record
        for k in range(1, total_steps + 1):
            lm_z_i = F.dropout(lm_z, p=p, training=True)  # upstream's draw, same order
            lz = to_mlx(lm_z_i.to(z_init.dtype))
            zm = self._step_fn(lm_params, tr_params, self.ln_w, self.ln_b, am, bT, lz, zm, zi, mask)
            self.steps_run = k
            if not check:
                continue
            if self.record:
                self._record(k, zm, mask, prev)
                continue
            # adaptive: signals are needed from step min_steps - 1 up to the second-to-last step
            if k < self.min_steps - 1 or k == total_steps:
                mx.eval(zm)
                continue
            ed, out = self._signal(zm, mask, self.source)
            if k >= self.min_steps and "ed" in prev:
                d = p95(mx.abs(ed - prev["ed"]))
                mx.eval(d, *([out] if out is not None else []))
                d = float(d.item())
                self.history.append({f"{self.source}_p95": d})
                if d < self.threshold:
                    if self.burn_rng:
                        for _ in range(total_steps - k):
                            F.dropout(lm_z, p=p, training=True)
                    z_out = to_torch(zm, dev)
                    if out is not None:  # forward's readout -> coda will return this
                        self._exit_z, self._exit_out = z_out, to_torch(out, dev)
                    return z_out
            else:
                mx.eval(ed)
            prev["ed"] = ed
        mx.eval(zm)
        return to_torch(zm, dev)

    def _record(self, k, zm, mask, prev):
        """Calibration: every candidate signal after every step (no exit)."""
        ed_c, _ = self._signal(zm, mask, "coda")
        ed_r, _ = self._signal(zm, mask, "readout")
        mx.eval(zm, ed_c, ed_r)
        if prev:
            row = {"step": k}
            for name, ed in (("coda", ed_c), ("readout", ed_r)):
                dd = mx.abs(ed - prev[name])
                close = (ed < 12) | (prev[name] < 12)
                row[f"{name}_p95"] = float(p95(dd).item())
                row[f"{name}_mean"] = float(dd.mean().item())
                row[f"{name}_max"] = float(dd.max().item())
                row[f"{name}_close"] = float((mx.sum(dd * close) / mx.maximum(mx.sum(close), 1)).item())
            dz = zm - prev["z"]
            row["z_rel"] = float((mx.sqrt(mx.sum(dz * dz)) / mx.sqrt(mx.sum(zm * zm))).item())
            self.history.append(row)
        prev.update(coda=ed_c, readout=ed_r, z=zm)


def apply(model, **kw):
    loop = MLXLoop(model, **kw)
    model._run_one_loop = loop.run  # instance attribute: forward's self._run_one_loop(...) finds it
    if loop.adaptive and loop.source == "coda" and not isinstance(model.parcae_readout, _Cached):
        model.parcae_readout = _Cached(model.parcae_readout, loop, "readout")
        model.parcae_coda = _Cached(model.parcae_coda, loop, "coda")
    model._mlx_loop = loop
    return loop
