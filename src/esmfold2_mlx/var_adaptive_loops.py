"""Variant: adaptive recycling. Stop running trunk passes once the pair
representation has converged, per protein; optionally spend extra passes on
proteins that have not converged by the last scheduled pass.

The recycling loop lives in forward (which we don't edit), so this wraps
whatever ``model.folding_trunk`` is (apply it last, e.g. on top of
``mlx_trunk16k``). Per fold it compares each trunk output with the previous one;
once the change is below THRESHOLD it returns that output for every remaining
call without running the trunk, which makes the fold identical to one run with
fewer loops. ``model.forward`` is wrapped (instance attribute) to reset state
and learn the scheduled number of passes.

Extra passes: forward's loop is ``z = trunk(z_init + pair_loop_proj(z))``. On
the first call the input is ``z_init + pair_loop_proj(0)``, so z_init is
recovered exactly as ``input - pair_loop_proj(0)`` and extra passes inside the
last call repeat the same recurrence, i.e. the fold matches one with more loops.

Result (held-out, check_accuracy): mean dTM +0.0003, worst -0.003, mean 3.54 of 4
passes, 5% less fold time; ubiquitin 0.93 -> 0.62 s, GFP 6.1 -> 5.1 s.

Usage:
    python tools/bench_e2e.py --variants fast_trunk,mlx_trunk16k,mlx_diffusion16,adaptive_loops
"""

import torch
from torch import nn

# Chosen on a separate 20-protein calibration set, never the held-out set: the largest saving
# for which no calibration protein's structure moved (TM >= 0.98 vs the loops-3 fold, same seed).
# 95th percentile over pairs of |change| of the distogram's expected distance between passes, Å.
METRIC = "disto_p95"
THRESHOLD = 0.6  # calibration picked 0.614; rounded down
MIN_PASSES = 2  # a comparison needs two passes; passes before this always run
# Extra passes after the scheduled ones when not converged. Off: on calibration the proteins
# that never converge are low-confidence folds (TM ~0.4) where loops 4-6 don't help.
MAX_EXTRA = 0

# Distogram bins: 128 bins with 127 edges evenly spaced over 2-52 Å (as in the confidence head).
_EDGES = torch.linspace(2.0, 52.0, 127)
_CENTERS = torch.cat([_EDGES[:1] - 0.2, (_EDGES[1:] + _EDGES[:-1]) / 2, _EDGES[-1:] + 0.2])


def expected_distance(z, disto_head):
    """Distogram head's expected distance per pair (Å), as forward computes the logits."""
    logits = disto_head(z + z.transpose(-2, -3)).float()
    return logits.softmax(-1) @ _CENTERS.to(z.device)


def metrics(prev, cur, disto_head, only=None):
    """Change between consecutive trunk outputs ([1, L, L, d]); all cheap next to a
    trunk pass. ``only`` names the one metric to compute (otherwise all, for calibration)."""
    prev, cur = prev.float(), cur.float()
    if only == "z_rel":
        return {"z_rel": float((cur - prev).norm() / cur.norm())}
    diff = cur - prev
    row_change = diff.norm(dim=(-2, -1)) / cur.norm(dim=(-2, -1))  # per residue i, over j and channels
    out = {
        "z_rel": float(diff.norm() / cur.norm()),
        "z_rel_max": float(row_change.max()),
    }
    if only in out:
        return out
    centers = _CENTERS.to(cur.device)
    lp, lc = (disto_head(z + z.transpose(-2, -3)).float() for z in (prev, cur))
    ep, ec = (l.softmax(-1) @ centers for l in (lp, lc))  # expected distance, Å
    dd = (ec - ep).abs()
    close = (ec < 12) | (ep < 12)  # pairs that matter for the fold
    out["disto_mean"] = float(dd.mean())
    out["disto_p95"] = float(dd.flatten().quantile(0.95))
    out["disto_close"] = float(dd[close].mean()) if bool(close.any()) else 0.0
    out["disto_argmax"] = float((lp.argmax(-1) != lc.argmax(-1)).float().mean())
    # Contact map (P(d < 8 Å)) agreement: what the structure module mostly needs.
    k = int((centers < 8).sum())
    cp, cc = lp.softmax(-1)[..., :k].sum(-1), lc.softmax(-1)[..., :k].sum(-1)
    out["contact_flip"] = float(((cp > 0.5) != (cc > 0.5)).float().sum() / ((cp > 0.5) | (cc > 0.5)).float().sum().clamp(min=1))
    return out


class AdaptiveTrunk(nn.Module):
    def __init__(self, trunk, model, metric=METRIC, threshold=THRESHOLD, min_passes=MIN_PASSES,
                 max_extra=MAX_EXTRA, extra_threshold=None, record_all=False):
        super().__init__()
        self.trunk = trunk
        # Plain attributes (not submodules) so the model's state dict is unchanged.
        self.__dict__["_proj"] = model.pair_loop_proj
        self.__dict__["_disto"] = model.distogram_head
        self.metric, self.threshold, self.min_passes = metric, threshold, min_passes
        self.max_extra = max_extra
        self.record_all = record_all
        # Extra passes stop at the same convergence test unless given their own.
        self.extra_threshold = threshold if extra_threshold is None else extra_threshold
        self.reset(None)

    def set_chunk_size(self, chunk_size):
        self.trunk.set_chunk_size(chunk_size)

    def set_kernel_backend(self, backend):
        self.trunk.set_kernel_backend(backend)

    def reset(self, scheduled_passes):
        self.scheduled = scheduled_passes
        self.calls = 0
        self.passes = 0  # trunk passes actually run
        self.prev = self.prev_ed = None
        self.z_init = None
        self.converged = False
        self.history = []  # metrics() after each pass from the second on

    def _pass(self, pair, mask):
        out = self.trunk(pair, pair_attention_mask=mask)
        self.passes += 1
        if self.record_all:
            if self.prev is not None:
                self.history.append(metrics(self.prev, out, self._disto))
        elif self.metric == "disto_p95":
            # Fast path: one head evaluation and one sync per pass (~10 ms at L=200), reusing
            # the previous pass's expected distances.
            ed = expected_distance(out, self._disto)
            if self.prev is not None:
                self.history.append({"disto_p95": float((ed - self.prev_ed).abs().flatten().quantile(0.95))})
            self.prev_ed = ed
        elif self.prev is not None:
            self.history.append(metrics(self.prev, out, self._disto, self.metric))
        self.prev = out
        return out

    def forward(self, pair, pair_attention_mask=None):
        self.calls += 1
        if self.converged:
            return self.prev  # same as a fold with fewer loops
        if self.calls == 1 and self.max_extra:
            self.z_init = pair - self._proj(torch.zeros_like(pair))
        out = self._pass(pair, pair_attention_mask)
        if self.passes >= self.min_passes and self.history[-1][self.metric] < self.threshold:
            self.converged = True
            return out
        if self.scheduled is not None and self.calls == self.scheduled and self.max_extra:
            for _ in range(self.max_extra):
                out = self._pass(self.z_init + self._proj(out), pair_attention_mask)
                if self.history[-1][self.metric] < self.extra_threshold:
                    break
        return out


def apply(model, **kw):
    trunk = AdaptiveTrunk(model.folding_trunk, model, **kw)
    model.folding_trunk = trunk
    forward = model.forward

    def adaptive_forward(*a, num_loops=None, **k):
        n = num_loops if num_loops is not None else model.config.num_loops
        trunk.reset(n + 1)
        return forward(*a, num_loops=num_loops, **k)

    model.forward = adaptive_forward  # nn.Module.__call__ dispatches to the instance attribute
    return trunk
