# esmfold2-mlx

[Biohub's ESMFold2](https://huggingface.co/biohub/ESMFold2) on Apple silicon: the same model, several times
faster than Biohub's own code on the same Mac, and with 2.7x less memory for the 6B model. Accuracy is unchanged
for the 300M model; the 6B's 8-bit language model costs a little (mean TM-score -0.002, details below).
Unofficial port: it is not affiliated with Biohub. Weights come from Biohub's own repositories.

Measured on an M1 Pro (16-core GPU, 32 GB):

| | 76 aa | 146 aa | 238 aa |
|---|---|---|---|
| **ESMFold2-Fast (6B)**, Biohub's code* | 2.73 s | 16.53 s | 48.86 s |
| **ESMFold2-Fast (6B)**, this package | 1.06 s | 2.88 s | **6.49 s** (7.5x) |
| 300M experimental checkpoint, Biohub's code* | 2.72 s | 14.34 s | 41.49 s |
| 300M experimental checkpoint, this package | 0.83 s | 2.77 s | **5.14 s** (8.1x) |

Seconds per fold, median of 3 after warm-up, 3 recycling loops, seed 0, with the same settings on both sides of
each pair: 20 sampling steps for the 6B rows, 50 for the 300M rows. At this package's default of 20 steps the
300M folds the 238-residue protein in 4.58 s (9.1x). Peak memory folding GFP (238 aa) with the 6B, measured in
separate runs: 18.4 GB for Biohub's code, 6.9 GB here (5.6 GB steady). Full table: [docs/benchmark.md](docs/benchmark.md).

\* Biohub's code does not run on a Mac as shipped, so both baselines carry the minimum fixes it needs to run
(see Caveats). None of them is a speed-up.

## Install

Needs an Apple-silicon Mac and Python 3.12. From a clone of this repository:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install .
```

The first run downloads Biohub's weights from Hugging Face. The 300M checkpoint is about 0.7 GB. The 6B needs
the 26 GB checkpoint once, plus a 6.7 GB 8-bit copy of its language model that this package builds and
caches in `~/.cache/esmfold2-mlx` (set `ESMFOLD2_MLX_CACHE` to move it).

## Use

```bash
esmfold2-mlx fold --seq MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG --name ubiquitin
esmfold2-mlx fold --reference 1UBQ                     # ubiquitin, scored against the crystal structure
esmfold2-mlx fold --fasta designs.fasta                # many sequences, model loaded once
esmfold2-mlx fold --model 6B --seq MKT...              # the released ESMFold2-Fast: prints pLDDT and pTM
esmfold2-mlx fold --model 6B --seq BINDER:TARGET --name cx   # complex (chains joined by ':') -> cx.cif, prints ipTM
```

Structures go to `./outputs` (`--out-dir` to change it). For many folds, keep a model loaded:

```bash
esmfold2-mlx serve start --model 6B &
esmfold2-mlx serve fold --model 6B --seq MKT... --name mine
esmfold2-mlx serve stop --model 6B
```

From Python:

```python
from esmfold2_mlx import Folder

f = Folder("6B")                  # loads and warms up once
res = f.fold("MKT...", "mine")    # dict: pdb path, seconds, plddt, ptm
```

Defaults: 3 recycling loops, and 20 sampling steps for 64-217 residues (50 outside that range, where 20
has not been tested). The 300M and 600M experimental checkpoints have no confidence head, so there is no
pLDDT or pTM for them.

## How a fold runs

What happens when you type `esmfold2-mlx fold --model 6B --seq MKT...`:

1. **Load Biohub's model.** The 6B checkpoint is 26 GB, almost all of it the ESMC language model. The package
   builds the model empty, fills in everything except ESMC (0.75 GB), then converts ESMC to 8-bit one layer at a
   time, so the full 26 GB is never in memory at once. The 6.7 GB result is saved, so later runs skip this step.
   The 300M and 600M checkpoints are small and load the normal way.
2. **Swap in faster parts.** Each model has a default list of swaps (`variants.py`). Every swap replaces one
   piece of Biohub's model with a faster version that uses the same weights:
   - the folding trunk runs in MLX (Apple's machine-learning framework) at fp16, with hand-written Metal kernels
     for the steps that are limited by memory speed;
   - the diffusion sampler, which places the atoms, runs in MLX with an fp16 network and fp32 coordinates;
   - the recycling loop stops early once the predicted distances stop changing.
3. **Trim memory** (`memtrim.py`). After each fold, cached buffers and duplicate weights are freed, and the 8-bit
   language model is read back from disk for each fold instead of being kept in memory.
4. **Fold.** Biohub's own `infer_protein` runs as normal and calls the swapped parts: the language model reads
   the sequence, the trunk refines a pairwise map of the residues over up to 3 recycling loops, the diffusion
   sampler turns that into 3D coordinates in 20 steps, and (6B only) the confidence head scores the result.
   Everything not swapped is Biohub's code, unchanged.
5. **Write the result:** a `.pdb` file (`.cif` for complexes), the confidence scores, and the RMSD to the
   crystal structure if you passed `--reference`.

## What changed, and what it costs in accuracy

Most of the time is spent in one recurrent trunk, so that is what moved first. Each change was kept only if it
passed an accuracy gate on held-out proteins.

| Change, in the order it was made | Speed-up on GFP (238 aa, 300M) over the row above |
|---|---|
| Split the trunk's projection weights instead of its activations (PyTorch on Apple GPUs is about 10x slower on split views) | 2.8x |
| 20 diffusion steps instead of 50 (validated for 64-217 aa only) | 1.2x |
| Folding trunk in MLX, fp16, fused kernels | 1.8x |
| Diffusion sampler in MLX | 1.2x |
| Hand-written Metal kernels for the memory-bound trunk steps | 1.04x (the trunk is limited by its matrix multiplications) |
| Adaptive recycling: stop once predicted distances stop changing | 1.26x (none on proteins that need every pass) |
| Mixed-precision diffusion: fp16 network, fp32 coordinates | 1.01x (on the 6B, its sampler stage alone: 1.4x) |

The first row speeds up Biohub's PyTorch trunk. The MLX trunk in row 3 then replaces that trunk entirely, so
its 1.8x is measured against the already-fixed version, and in the default stacks the first fix only affects the
few PyTorch layers of the same kind that remain. For the 6B, the 8-bit language model and memory trimming are
what bring peak memory from 18.4 GB to 6.9 GB.

Accuracy, on 24 held-out proteins released after the training cutoff with no close homologue before it:

- **300M:** the shipped default against Biohub's model run in PyTorch, same settings: mean TM-score 0.8747
  against 0.8748, worst single protein -0.003.
- **6B:** checked in two steps. First, the 8-bit MLX version against Biohub's model in PyTorch with an fp16
  language model, at matched settings (3 loops, 14 sampling steps, same seeds): mean TM-score 0.896 against
  0.898, worst single protein -0.039. Then each later speed-up (Metal kernels, adaptive recycling,
  mixed-precision diffusion) against the version before it: combined mean -0.0007 (95% CI -0.002 to +0.001),
  worst -0.009, which is within the model's own variation between random seeds. The 8-bit language model is a
  real change to the numbers, not only to the runtime, so it is the part worth checking on your own proteins.

Check it yourself with the held-out set in `eval_set/` and the tools in `tools/`:

```bash
uv pip install ".[eval]"
python tools/regression_check.py --memtrim default            # ~1 min: default path against stored structures
python tools/check_accuracy.py --variants fast_trunk,mlx_trunk16k,mlx_diffusion16,adaptive_loops   # ~4 min
python tools/benchmark.py                                     # ~10 min, writes outputs/benchmark/
```

Run these from the repository root. Close other GPU-heavy apps before timing anything.

## Caveats

- **Both baselines are patched to run.** Biohub's code does not run on a Mac as shipped: its bf16 language
  model fails on Apple GPUs, PyTorch's Apple backend lacks one integer operation it uses, and the 6B loader reads
  all 26 GB into memory before converting it. So the baselines run the language model in fp32 (300M) or fp16
  (6B), send that one operation to the CPU, and load the 6B with this package's layer-by-layer loader plus two
  small fixes (a `seed` argument and one dtype cast). The rest of Biohub's code runs unchanged.
- Timings are from one machine. Other Apple chips are untested.
- Versions are pinned to the ones the numbers came from (`torch` 2.11, `mlx` 0.32). The MPS and MLX workarounds
  are version-sensitive, and a newer PyTorch changes random numbers on MPS for the same seed. After upgrading
  anything, run `tools/regression_check.py`. `requirements-lock.txt` has the exact set.
- Seeds matter for complexes: some interfaces are bimodal. For binder screening, fold two or three seeds and
  keep the highest ipTM.

## Credit and licence

ESMFold2 is by Biohub ([paper](https://www.biorxiv.org/content/10.64898/2026.06.03.729735),
[code](https://github.com/Biohub/esm), [weights](https://huggingface.co/biohub/ESMFold2)). If you use
this, cite their paper:

```bibtex
@misc{candido2026language,
  title = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and others},
  year = {2026},
  url = {https://www.biorxiv.org/content/10.64898/2026.06.03.729735},
  note = {Preprint}
}
```

Use of ESMFold2 is subject to Biohub's Acceptable Use Policy (see the model card). This package is MIT
licensed; see `LICENSE` and `NOTICE`.
