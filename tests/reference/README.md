# Regression references

Seed-0 structures (all atoms, `.npy`) and fold times (`meta.json`) from each model's default
fold path, one folder per `<model>__<stack>`. `tools/regression_check.py` compares against the
folder for the current default stack.

Current defaults, recorded 2026-10-03 with `--memtrim default` (what `esmfold2-mlx fold` runs):

| | 300M (`--tol 0.15`) | 6B (`--tol 0.08`) |
|---|---|---|
| identical re-runs (ubiquitin / 7QFG / GFP), Å | 0.013-0.075 / 0.036-0.102 / 0.019-0.045 | 0.018-0.025 / 0.021-0.026 / 0.023-0.029 |
| ubiquitin to 1UBQ | 0.69 Å | 0.84 Å |
| trunk at L=476 | finite, max 482 | finite, max 7,040 (confidence trunk 32,928) |
| barnase:barstar complex (1BRS), re-run | 0.009 Å | 0.017 Å (ipTM 0.96) |
| negative control (exit code 1) | 600M checkpoint: 0.40 / 4.86 / 1.99 Å, complex 0.40 Å | 4-bit ESMC: 0.03 / 0.16 / 0.29 Å |

Adaptive recycling makes re-runs spread more than the fixed-loop stacks did (0.004-0.023 Å):
a protein near the exit threshold can stop one pass earlier or later. Tolerances are about
3x the largest re-run difference; both negative controls still fail on at least two proteins.
Run a control with `--model 600M --ref-model 300M` or `--model 6B-q4 --ref-model 6B --stack <6B default>`.

Re-record only after an intended numerical change has passed tools/check_accuracy.py:
    python tools/regression_check.py --model <m> --memtrim default --update

The complex reference (`barnase_barstar.npy`) is recorded automatically the first time a
reference folder without one is checked.
