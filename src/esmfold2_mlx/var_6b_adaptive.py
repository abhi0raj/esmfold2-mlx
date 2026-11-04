"""Variant (6B release model): adaptive recycling for the parcae loop.

var_6b_mlx_loop with the early exit on: after step k >= 2 of the scheduled num_loops+1,
stop once the 95th percentile over pairs of |change| in the distogram's expected distance
(computed on the real output path, readout -> coda -> distogram head) since the previous
step is below 0.5 A; the coda output of the exit step is handed on to forward, and the
skipped steps' dropout draws are still made, so the diffusion sees the default stack's
random stream.

The threshold was calibrated on a separate 20-protein calibration set, never on the held-out
set. The rule (every calibration fold, 20 proteins x 2 seeds, at TM >= 0.98 vs its own
full-loop prediction, same seed; mean dTM >= -0.002, worst >= -0.02) allows coda_p95 up to
0.55 (the next recorded value fails); rounded down to 0.5 for margin: 2.65 of 4 steps on
calibration.

Usage:
    python tools/bench_e2e.py --model 6B --variants fast_trunk,6b_mlx_trunks16k,mlx_diffusion16,6b_adaptive
"""

from . import var_6b_mlx_loop

THRESHOLD = 0.5
SOURCE = "coda"
MIN_STEPS = 2


def apply(model, **kw):
    cfg = dict(adaptive=True, threshold=THRESHOLD, source=SOURCE, min_steps=MIN_STEPS, burn_rng=True)
    cfg.update(kw)
    return var_6b_mlx_loop.apply(model, **cfg)
