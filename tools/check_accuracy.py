"""Accuracy gate for a variant stack: fold all 24 held-out proteins at the
recommended setting (3 loops, 20 steps, seeds 0 and 1) and compare TM-scores
with the fast_trunk baseline rows already in eval_set/eval_results.csv (300M model;
for another --model the deltas measure the change of model).

A variant passes if the mean change is within seed noise (|mean dTM| <= 0.005)
and no protein drops by more than 0.03. Close other GPU-heavy apps first.

Usage:
    python tools/check_accuracy.py --variants fast_trunk,mlx_trunk16k,mlx_diffusion16,adaptive_loops
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tmtools import tm_align

from run_eval import ca_trace
from esmfold2_mlx.variants import apply_memtrim, apply_variants, load_model

warnings.filterwarnings("ignore")
LOOPS, SEEDS = 3, (0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--model", default="300M", help='"300M", "600M" or a Hub id')
    p.add_argument("--memtrim", default="none", help="memtrim spec as in fold.py ('default' = what fold.py ships)")
    p.add_argument("--steps", type=int, default=20)
    # For models whose defaults differ (6B): the baseline stays 300M at 3 loops and
    # --baseline-steps (default: --steps).
    p.add_argument("--loops", type=int, default=LOOPS)
    p.add_argument("--baseline-steps", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--proteins", nargs="+", default=None, help="subset of pdb ids (default: all 24)")
    args = p.parse_args()
    tag = args.tag or args.variants
    STEPS = args.steps

    base = pd.read_csv("eval_set/eval_results.csv")
    base_steps = args.baseline_steps or STEPS
    base = base[(base.loops == LOOPS) & (base.steps == base_steps)].groupby("pdb_id").tm_score.mean()

    model = apply_variants(load_model(args.model), args.variants.split(","))
    apply_memtrim(model, args.model, args.memtrim)
    rows = []
    for e in json.loads(Path("eval_set/eval_set.json").read_text()):
        if args.proteins and e["pdb_id"] not in args.proteins:
            continue
        ref_xyz, ref_seq = ca_trace(Path(f"eval_set/{e['pdb_id']}.pdb").read_text())
        tms, extra = [], {}
        for seed in args.seeds:
            t0 = time.perf_counter()
            o = model.infer_protein(e["sequence"], num_loops=args.loops, num_sampling_steps=STEPS,
                                    num_diffusion_samples=1, seed=seed)
            extra[f"seconds_s{seed}"] = round(time.perf_counter() - t0, 2)
            # checkpoints with a confidence head (6B): keep its pLDDT (0-1) and pTM
            for k in ("plddt", "ptm"):
                if k in o:
                    extra[f"{k}_s{seed}"] = float(o[k].float().mean())
            o["plddt"] = torch.zeros(o["token_attention_mask"].shape)
            xyz, seq = ca_trace(model.output_to_pdb(o))
            tms.append(tm_align(xyz, ref_xyz, seq, ref_seq).tm_norm_chain2)
            extra[f"tm_s{seed}"] = tms[-1]
        rows.append({"pdb_id": e["pdb_id"], "tm": float(np.mean(tms)), "tm_baseline": float(base[e["pdb_id"]]),
                     **extra})
        print(f"{e['pdb_id']}  TM {rows[-1]['tm']:.3f}  baseline {rows[-1]['tm_baseline']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df["delta"] = df.tm - df.tm_baseline
    passed = abs(df.delta.mean()) <= 0.005 and df.delta.min() >= -0.03
    summary = dict(tag=tag, model=args.model, steps=STEPS, loops=args.loops, variants=args.variants, mean_tm=round(df.tm.mean(), 4),
                   mean_delta=round(df.delta.mean(), 4), worst_delta=round(df.delta.min(), 4),
                   worst_protein=df.loc[df.delta.idxmin(), "pdb_id"], passed=bool(passed))
    print(json.dumps(summary), flush=True)
    out = Path("outputs/accuracy")
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"{tag.replace(',', '+')}.csv", index=False)
    with Path("outputs/accuracy_results.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
