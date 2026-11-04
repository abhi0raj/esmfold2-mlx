"""End-to-end fold timing for a stack of variants, with a structure check.

Times whole folds (median of --repeats, after a warm-up), then one extra fold
with stage hooks for a breakdown. Compares the seed-0 structure against the
reference saved by ``--save-ref`` (the fast_trunk baseline) by aligned
all-atom RMSD. Appends one JSON line per length to outputs/e2e_results.jsonl.
Close other GPU-heavy apps first.

Usage:
    python tools/bench_e2e.py --variants fast_trunk --save-ref
    python tools/bench_e2e.py --variants fast_trunk,mlx_trunk16k,mlx_diffusion16,adaptive_loops --tag default
"""

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from esmfold2_mlx.variants import apply_memtrim, apply_variants, load_model

warnings.filterwarnings("ignore")

GFP = ("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQ"
       "ERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTP"
       "IGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK")
SEQS = {
    "ubiquitin_76": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "7QFG_146": next(e["sequence"] for e in json.loads(Path("eval_set/eval_set.json").read_text())
                     if e["pdb_id"] == "7QFG"),
    "gfp_238": GFP,
}
REF_DIR = Path("outputs/e2e_ref")


def sync():
    torch.mps.synchronize()


def aligned_rmsd(P, Q):
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean()))


def coords(out):
    x = out["sample_atom_coords"].detach().float().cpu()
    if x.dim() == 4:
        x = x[:, 0]
    mask = out["atom_attention_mask"].bool().cpu()[0] if "atom_attention_mask" in out else None
    x = x[0]
    return (x[mask] if mask is not None else x).numpy()


def stage_breakdown(model, fold):
    totals = {"esmc": 0.0, "trunk": 0.0, "diffusion": 0.0}
    handles = []
    stages = [(model.esmc, "esmc"), (model.folding_trunk, "trunk")]
    # release (6B) model only: its extra pair stacks and the confidence head
    for attr, name in [("lm_encoder", "lm_encoder"), ("parcae_coda", "coda"), ("confidence_head", "confidence")]:
        if getattr(model, attr, None) is not None and hasattr(model, "parcae_readout"):
            stages.append((getattr(model, attr), name))
            totals[name] = 0.0
    for mod, name in stages:
        start = {}
        handles.append(mod.register_forward_pre_hook(
            lambda *_, s=start: (sync(), s.__setitem__("t", time.perf_counter()))[-1]))
        handles.append(mod.register_forward_hook(
            lambda *_, s=start, n=name: (sync(), totals.__setitem__(n, totals[n] + time.perf_counter() - s["t"]))[-1]))
    head = model.structure_head
    orig = head.sample

    def sample(*a, **k):
        sync()
        t0 = time.perf_counter()
        r = orig(*a, **k)
        sync()
        totals["diffusion"] += time.perf_counter() - t0
        return r

    head.sample = sample
    sync()
    t0 = time.perf_counter()
    fold()
    sync()
    wall = time.perf_counter() - t0
    head.sample = orig
    for h in handles:
        h.remove()
    totals["other"] = wall - sum(totals.values())
    return {k: round(v, 3) for k, v in totals.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="fast_trunk")
    p.add_argument("--model", default="300M", help='"300M", "600M", "6B" (variants.MODELS_6B) or a Hub id')
    p.add_argument("--memtrim", default="none", help="memtrim spec as in fold.py ('default' = what fold.py ships)")
    p.add_argument("--tag", default=None)
    p.add_argument("--lengths", nargs="+", default=list(SEQS))
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--save-ref", action="store_true", help="store seed-0 structures as the reference")
    args = p.parse_args()
    names = args.variants.split(",")
    tag = args.tag or args.variants

    model = apply_variants(load_model(args.model), names)
    apply_memtrim(model, args.model, args.memtrim)
    for key in args.lengths:
        seq = SEQS[key]
        kw = dict(num_loops=args.loops, num_sampling_steps=args.steps, num_diffusion_samples=1, seed=0)
        fold = lambda: model.infer_protein(seq, **kw)  # noqa: E731
        out = fold()  # warm-up (Metal shader compilation, MLX compile)
        sync()
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = fold()
            sync()
            times.append(time.perf_counter() - t0)
        stages = stage_breakdown(model, fold)

        x = coords(out)
        # References are per model: a different checkpoint folds differently.
        suffix = "" if args.model == "300M" else f"_{args.model.split('/')[-1]}"
        ref_path = REF_DIR / f"{key}_l{args.loops}_s{args.steps}{suffix}.npy"
        if args.save_ref:
            REF_DIR.mkdir(parents=True, exist_ok=True)
            np.save(ref_path, x)
        rmsd = aligned_rmsd(x, np.load(ref_path)) if ref_path.exists() else None

        row = dict(tag=tag, model=args.model, variants=names, protein=key, length=len(seq), loops=args.loops, steps=args.steps,
                   median_s=round(statistics.median(times), 3), times=[round(t, 3) for t in times],
                   stages=stages, rmsd_vs_ref=None if rmsd is None else round(rmsd, 4),
                   finite=bool(np.isfinite(x).all()))
        print(json.dumps(row), flush=True)
        with Path("outputs/e2e_results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
