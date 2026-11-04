"""Sweep num_loops x num_sampling_steps over the held-out set; score by TM-score.

Appends one row per fold to a CSV, skipping rows already present, so it can be
stopped and resumed.

Usage:
    python tools/run_eval.py                                   # default grid, 300M model
    python tools/run_eval.py --loops 3 --steps 50 --model biohub/ESMFold2-Experimental-Fast-base600M-step1500k
"""

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from esm.models.esmfold2 import EsmFold2ExperimentalModel
from tmtools import tm_align

from esmfold2_mlx import fast_trunk
from esmfold2_mlx import mps_compat  # noqa: F401

warnings.filterwarnings("ignore")

THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}
FIELDS = ["model", "pdb_id", "length", "loops", "steps", "seed", "seconds", "tm_score", "rmsd"]


def ca_trace(pdb_text):
    """C-alpha coords and one-letter sequence of the first protein chain (first model, altloc A)."""
    chain = None
    coords, seq, seen = [], [], set()
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")) or line[12:16].strip() != "CA" or line[16] not in " A":
            continue
        if line[17:20] not in THREE_TO_ONE:
            continue
        chain = chain or line[21]  # RCSB label IDs can differ from the file's author chain IDs
        key = (line[21], line[22:27])
        if line[21] != chain or key in seen:
            continue
        seen.add(key)
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        seq.append(THREE_TO_ONE[line[17:20]])
    return np.array(coords), "".join(seq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="biohub/ESMFold2-Experimental-Fast-base300M-step1500k")
    p.add_argument("--loops", type=int, nargs="+", default=[0, 1, 3, 6])
    p.add_argument("--steps", type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--out", default="eval_set/eval_results.csv")
    args = p.parse_args()

    entries = json.loads(Path("eval_set/eval_set.json").read_text())
    out = Path(args.out)
    done = set()
    if out.exists():
        with out.open() as f:
            done = {(r["model"], r["pdb_id"], int(r["loops"]), int(r["steps"]), int(r["seed"])) for r in csv.DictReader(f)}
    else:
        out.parent.mkdir(exist_ok=True)
        with out.open("w") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    def finalize_esmc(self):
        self.esmc = self.esmc.to(device=self.device, dtype=torch.float32).eval()

    EsmFold2ExperimentalModel.finalize_esmc = finalize_esmc
    model = EsmFold2ExperimentalModel.from_pretrained(args.model, device="mps").eval()
    fast_trunk.apply(model)
    model.infer_protein(entries[0]["sequence"], num_loops=0, num_sampling_steps=5, num_diffusion_samples=1)  # warm-up

    short = args.model.split("/")[-1]
    total = len(entries) * len(args.loops) * len(args.steps) * len(args.seeds)
    n = 0
    for e in entries:
        ref_xyz, ref_seq = ca_trace(Path(f"eval_set/{e['pdb_id']}.pdb").read_text())
        for loops in args.loops:
            for steps in args.steps:
                for seed in args.seeds:
                    n += 1
                    if (short, e["pdb_id"], loops, steps, seed) in done:
                        continue
                    torch.mps.synchronize()
                    t0 = time.perf_counter()
                    o = model.infer_protein(e["sequence"], num_loops=loops, num_sampling_steps=steps,
                                            num_diffusion_samples=1, seed=seed)
                    torch.mps.synchronize()
                    secs = time.perf_counter() - t0
                    o["plddt"] = torch.zeros(o["token_attention_mask"].shape)  # PDB writer needs it
                    pred_xyz, pred_seq = ca_trace(model.output_to_pdb(o))
                    r = tm_align(pred_xyz, ref_xyz, pred_seq, ref_seq)
                    row = dict(model=short, pdb_id=e["pdb_id"], length=e["length"], loops=loops, steps=steps,
                               seed=seed, seconds=f"{secs:.2f}", tm_score=f"{r.tm_norm_chain2:.4f}",
                               rmsd=f"{r.rmsd:.3f}")
                    with out.open("a") as f:
                        csv.DictWriter(f, FIELDS).writerow(row)
                    print(f"[{n}/{total}] {e['pdb_id']} L={e['length']} loops={loops} steps={steps} seed={seed}"
                          f"  TM={r.tm_norm_chain2:.3f}  {secs:.1f}s", flush=True)


if __name__ == "__main__":
    main()
