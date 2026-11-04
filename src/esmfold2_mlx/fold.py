"""Fold proteins with ESMFold2 locally on Apple Silicon.

Usage:
    esmfold2-mlx fold --reference 1UBQ                  # ubiquitin, scored vs the crystal structure
    esmfold2-mlx fold --seq MKT... --name mine          # -> outputs/mine.pdb
    esmfold2-mlx fold --fasta designs.fasta             # many sequences, model loaded once
    esmfold2-mlx fold --model 600M                      # larger language model
    esmfold2-mlx fold --model 6B                        # ESMFold2-Fast: ESMC-6B (8-bit) + confidence (pLDDT, pTM)
    esmfold2-mlx fold --seq CHAIN_A:CHAIN_B --name cx   # complex: chains separated by ':' -> outputs/cx.cif
    esmfold2-mlx fold --stack fast_trunk                # PyTorch-only path; --stack "" = upstream code
"""

import argparse
import re
import time
import warnings
from pathlib import Path

import torch

from . import mps_compat  # noqa: F401
from .variants import DEFAULT_MODEL, apply_memtrim, apply_variants, default_stack, load_model

# Expected on a Mac: CUDA autocast is a no-op, SVD falls back to CPU, and the
# experimental configs use pre-rename field names.
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*linalg_svd.*")
warnings.filterwarnings("ignore", message=".*pre-alignment ESMFold2 field.*")
warnings.filterwarnings("ignore", message=".*No MSA provided.*")  # complexes: single-sequence mode is the design

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
LM_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# On 24 held-out monomers of 64-217 aa (eval_set/), 20 steps matched 50
# (mean dTM +0.001, worst -0.017). Outside that range keep upstream's 50 until tested.
VALIDATED_LENGTHS = range(64, 218)


def read_fasta(path):
    records, name = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            name = re.sub(r"[^\w.-]+", "_", line[1:].split()[0]) if line[1:].strip() else f"seq{len(records)}"
            records.append([name, ""])
        elif line.strip():
            records[-1][1] += line.strip().upper()
    return [(n, s) for n, s in records if s]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default=UBIQUITIN)
    p.add_argument("--name", default="ubiquitin")
    p.add_argument("--fasta", help="fold every record; overrides --seq/--name")
    p.add_argument("--model", default=DEFAULT_MODEL, help='"300M", "600M", "6B" (8-bit ESMC; also 6B-f16) or a Hub id')
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--stack", default=None,
                   help="comma-separated speed variants (variants.py); default: the model's validated stack; '' = upstream")
    p.add_argument("--num-loops", type=int, default=3, help="trunk recycles (more = more compute, better accuracy)")
    p.add_argument("--num-sampling-steps", type=int, default=None,
                   help="diffusion denoising steps (default: 20 inside the validated length range, else 50)")
    p.add_argument("--num-samples", type=int, default=1, help="diffusion samples drawn from the same trunk output")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference", help="PDB ID to score against by C-alpha RMSD, e.g. 1UBQ (single sequence only)")
    p.add_argument("--memtrim", default="default",
                   help="memory options (memtrim.py; numerically neutral): 'default' = per model, 'none', or a list")
    p.add_argument("--lm-dtype", default="fp32", choices=list(LM_DTYPES),
                   help="ESMC language-model precision; upstream forces bf16, which M1 MPS can't matmul")
    p.add_argument("--out-dir", default="outputs", help="where structures are written")
    args = p.parse_args()

    jobs = read_fasta(args.fasta) if args.fasta else [(args.name, args.seq)]
    if args.stack is None:
        args.stack = default_stack(args.model)
    t0 = time.time()
    model = apply_variants(load_model(args.model, args.device, LM_DTYPES[args.lm_dtype]), args.stack.split(","))
    apply_memtrim(model, args.model, args.memtrim)
    print(f"loaded {args.model} with stack [{args.stack}] on {args.device} in {time.time() - t0:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, seq in jobs:
        if ":" in seq:
            fold_one_complex(model, name, seq.split(":"), args, out_dir)
            continue
        steps = args.num_sampling_steps or (20 if len(seq) in VALIDATED_LENGTHS else 50)
        t0 = time.time()
        out = model.infer_protein(seq, seed=args.seed, num_loops=args.num_loops, num_sampling_steps=steps,
                                  num_diffusion_samples=args.num_samples)
        dt = time.time() - t0
        if "ptm" in out:  # only models with a confidence head (6B)
            print(f"{name}: mean pLDDT {float(out['plddt'].mean()):.2f}, pTM {float(out['ptm'].mean()):.2f}")
        if "plddt" not in out:
            # The experimental checkpoints ship without a confidence head; the PDB
            # writer still wants per-token pLDDT for the B-factor column.
            out["plddt"] = torch.zeros(out["token_attention_mask"].shape)
        out_path = out_dir / f"{name}.pdb"
        out_path.write_text(model.output_to_pdb(out))
        print(f"{name}: {len(seq)} aa, {steps} steps, {dt:.1f}s -> {out_path}")

    if args.reference and len(jobs) == 1:
        from .score import ca_rmsd_to_pdb

        print(f"C-alpha RMSD vs {args.reference.upper()}: {ca_rmsd_to_pdb(out_path, args.reference):.2f} Å")


def fold_one_complex(model, name, chains, args, out_dir):
    from .complex_fold import fold_complex

    # Fast stacks validated against the PyTorch path at 20 steps on 24 held-out heterodimers of
    # 60-383 residues in total
    steps = args.num_sampling_steps or 20
    t0 = time.time()
    res = fold_complex(model, chains, seed=args.seed, num_loops=args.num_loops, num_sampling_steps=steps)
    dt = time.time() - t0
    if res.iptm is not None:
        print(f"{name}: ipTM {res.iptm:.2f}, pTM {res.ptm:.2f}")
    out_path = out_dir / f"{name}.cif"
    out_path.write_text(res.complex.to_mmcif())
    print(f"{name}: {len(chains)} chains, {'+'.join(str(len(c)) for c in chains)} aa, {steps} steps, "
          f"{dt:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
