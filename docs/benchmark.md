# ESMFold2 benchmark, 2026-10-03

Apple M1 Pro (16-core GPU), macOS 26.6.2, torch 2.11.0, MLX 0.32.3.

Subset of the original run: the three superseded-default rows are omitted because this release no longer ships those stacks.
Median of 3 folds after warm-up, 3 loops, 1 sample, seed 0. Seconds per fold (speed-up vs the same model's upstream code).

| Configuration | Steps | ubiquitin_76 | 7QFG_146 | gfp_238 |
|---|---|---|---|---|
| Upstream code | 50 | 2.72 (1.0x) | 14.34 (1.0x) | 41.49 (1.0x) |
| + fast_trunk | 50 | 2.25 (1.2x) | 5.79 (2.5x) | 15.07 (2.8x) |
| + 20 steps | 20 | 1.57 (1.7x) | 4.50 (3.2x) | 12.67 (3.3x) |
| Default stack, 50 steps | 50 | 0.83 (3.3x) | 2.77 (5.2x) | 5.14 (8.1x) |
| Default stack | 20 | 0.58 (4.7x) | 2.38 (6.0x) | 4.58 (9.1x) |
| Default stack, 600M | 20 | 0.60 (4.5x) | 2.41 (5.9x) | 4.64 (8.9x) |
| 6B: upstream code (fp16 ESMC) | 20 | 2.73 (1.0x) | 16.53 (1.0x) | 48.86 (1.0x) |
| 6B: default stack (8-bit ESMC) | 20 | 1.06 (2.6x) | 2.88 (5.7x) | 6.49 (7.5x) |
