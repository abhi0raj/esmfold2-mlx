"""Regression check for a model's default fold path (variants.default_stack).

Folds the three benchmark proteins (seed 0, 3 loops, 20 steps) and fails if
  - any coordinate is non-finite,
  - a structure moves more than --tol Å (aligned all-atom RMSD) from the stored
    reference in tests/reference/,
  - ubiquitin's C-alpha RMSD to the 1UBQ crystal structure exceeds 1.5 Å,
  - the trunk produces non-finite values on a complex-scale input (GFP's real trunk
    input tiled 2x2 to L=476): guards the fp16 contraction's overflow headroom,
  - the two-chain path (barnase-barstar, complex_fold.py) moves more than --tol Å from its
    reference or gives non-finite coordinates.
Fold times more than 20% slower than the reference print a warning (--strict: fail).

Run after changing any model code or upgrading torch/MLX. ~1 min. Close other GPU-heavy apps first.

Usage:
    python tools/regression_check.py
    python tools/regression_check.py --update   # re-record references
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from bench_e2e import SEQS, aligned_rmsd, coords
from esmfold2_mlx.score import ca_rmsd_to_pdb
from esmfold2_mlx.variants import DEFAULT_MODEL, apply_memtrim, apply_variants, default_stack, load_model

warnings.filterwarnings("ignore")
REF = Path("tests/reference")
COMPLEX = ["AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR",
           "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS"]  # 1BRS
FOLD = dict(num_loops=3, num_sampling_steps=20, num_diffusion_samples=1, seed=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--memtrim", default="none", help="memtrim spec as in fold.py ('default' = what fold.py ships)")
    p.add_argument("--ref-model", default=None,
                   help="compare against another model's references (negative control), e.g. --model 600M --ref-model 300M")
    p.add_argument("--stack", default=None, help="default: the model's validated stack")
    p.add_argument("--tol", type=float, default=None,
                   help="max aligned all-atom RMSD (Å) to the reference; default 0.15, or 0.08 for 6B "
                        "(re-runs agree to 0.075 / 0.029 Å; see tests/reference/README.md)")
    p.add_argument("--update", action="store_true", help="overwrite the references with this run")
    p.add_argument("--strict", action="store_true", help="also fail on a >20%% slowdown")
    p.add_argument("--skip-long", action="store_true", help="skip the L=476 finiteness check (~10 s)")
    # Only with --update; a check re-runs the fold settings stored in meta.json.
    p.add_argument("--loops", type=int, default=FOLD["num_loops"])
    p.add_argument("--steps", type=int, default=FOLD["num_sampling_steps"])
    args = p.parse_args()
    if args.stack is None:
        args.stack = default_stack(args.model)
    if args.tol is None:
        args.tol = 0.08 if args.model.startswith("6B") else 0.15

    ref_dir = REF / f"{(args.ref_model or args.model).split('/')[-1]}__{args.stack.replace(',', '+') or 'upstream'}"
    meta_path = ref_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not meta and not args.update:
        sys.exit(f"no reference in {ref_dir}; run with --update first")

    model = apply_variants(load_model(args.model), args.stack.split(","))
    apply_memtrim(model, args.model, args.memtrim)
    trunk = model.folding_trunk
    while hasattr(trunk, "trunk"):  # unwrap e.g. adaptive_loops to the trunk that does the maths
        trunk = trunk.trunk
    captured = {}
    grab = lambda key: lambda m, a, k: captured.__setitem__(key, (a[0] if a else k["pair"]).detach())  # noqa: E731
    trunk.register_forward_pre_hook(grab("z"), with_kwargs=True)
    # 6B: the fused recycling loop (6b_adaptive) calls the MLX trunk directly, so the hook above never
    # fires; the pair state it hands to the readout is the same kind of input (the trunk's own output).
    if hasattr(model, "parcae_readout"):
        model.parcae_readout.register_forward_pre_hook(grab("z_loop"), with_kwargs=True)
    conf_trunk = getattr(getattr(model, "confidence_head", None), "folding_trunk", None)
    if conf_trunk is not None:  # the 6B's confidence trunk has the largest activations
        conf_trunk.register_forward_pre_hook(grab("z_conf"), with_kwargs=True)
    fold = dict(FOLD, num_loops=args.loops, num_sampling_steps=args.steps) if args.update else meta.get("fold", FOLD)
    failures, warnings_, record = [], [], {"model": args.model, "stack": args.stack, "fold": fold, "seconds": {}}
    for name, seq in SEQS.items():
        model.infer_protein(seq, **fold)  # warm-up / compile
        torch.mps.synchronize()
        t0 = time.perf_counter()
        out = model.infer_protein(seq, **fold)
        torch.mps.synchronize()
        secs = time.perf_counter() - t0
        x = coords(out)
        record["seconds"][name] = round(secs, 3)
        line = f"{name:13s} {secs:6.2f}s"

        if not np.isfinite(x).all():
            failures.append(f"{name}: non-finite coordinates")
        if args.update:
            ref_dir.mkdir(parents=True, exist_ok=True)
            np.save(ref_dir / f"{name}.npy", x)
        else:
            rmsd = aligned_rmsd(x, np.load(ref_dir / f"{name}.npy"))
            line += f"  RMSD to reference {rmsd:.3f} Å"
            if rmsd > args.tol:
                failures.append(f"{name}: {rmsd:.3f} Å from reference (tol {args.tol})")
            ref_secs = meta["seconds"][name]
            if secs > 1.2 * ref_secs:
                warnings_.append(f"{name}: {secs:.2f}s vs reference {ref_secs:.2f}s")

        if name == "ubiquitin_76":
            out["plddt"] = torch.zeros(out["token_attention_mask"].shape)
            pdb = Path("outputs/regression_ubiquitin.pdb")
            pdb.write_text(model.output_to_pdb(out))
            ca = ca_rmsd_to_pdb(pdb, "1UBQ")
            line += f"  C-alpha RMSD to 1UBQ {ca:.2f} Å"
            if ca > 1.5:
                failures.append(f"ubiquitin: {ca:.2f} Å from 1UBQ")
        print(line, flush=True)

    if not args.skip_long:  # inputs from the last fold (GFP), tiled 2x2
        tests = [("trunk", trunk, captured.get("z", captured.get("z_loop")))]
        if conf_trunk is not None:
            tests.append(("confidence trunk", conf_trunk, captured.get("z_conf")))
        for label, t, z in tests:
            if z is None:
                failures.append(f"{label}: no input captured for the L=476 check")
                continue
            big = z.repeat(1, 2, 2, 1)
            with torch.no_grad():
                out = t(big, pair_attention_mask=torch.ones(big.shape[:-1], device=big.device))
            finite = bool(torch.isfinite(out).all())
            print(f"{label} at L={big.shape[1]} (GFP input tiled 2x2): finite={finite}, "
                  f"max |z| {out.abs().max().item():.0f}")
            if not finite:
                failures.append(f"{label} non-finite at L={big.shape[1]}")

    # Two chains go through upstream's input builder (chain ids, per-chain numbering), a path the
    # monomers above never touch. Its reference is recorded on first run if missing.
    from esmfold2_mlx.complex_fold import fold_complex

    for _ in range(2):  # warm-up, then timed
        t0 = time.perf_counter()
        res = fold_complex(model, COMPLEX, seed=fold["seed"], num_loops=fold["num_loops"],
                           num_sampling_steps=fold["num_sampling_steps"])
        secs = time.perf_counter() - t0
    x = np.asarray(res.complex.atom_coordinates, dtype=np.float64).reshape(-1, 3)
    x = x[np.isfinite(x).all(1)]
    line = f"{'barnase:barstar':13s} {secs:6.2f}s"
    ref_path = ref_dir / "barnase_barstar.npy"
    if len(x) == 0:
        failures.append("complex: no finite coordinates")
    elif args.update or not ref_path.exists():
        ref_dir.mkdir(parents=True, exist_ok=True)
        np.save(ref_path, x)
        record.setdefault("seconds", {})["barnase_barstar"] = round(secs, 3)
        if not args.update:  # first run with an older reference folder: add it
            meta["seconds"]["barnase_barstar"] = round(secs, 3)
            meta_path.write_text(json.dumps(meta, indent=1))
        line += "  reference recorded"
    else:
        rmsd = aligned_rmsd(x, np.load(ref_path))
        line += f"  RMSD to reference {rmsd:.3f} Å"
        if rmsd > args.tol:
            failures.append(f"complex: {rmsd:.3f} Å from reference (tol {args.tol})")
    if res.iptm is not None:
        line += f"  ipTM {res.iptm:.2f}"
    print(line, flush=True)

    if args.update:
        meta_path.write_text(json.dumps(record, indent=1))
        print(f"references written to {ref_dir}")
        return
    for w in warnings_:
        print(f"WARN slower: {w}")
    for f in failures:
        print(f"FAIL {f}")
    if failures or (args.strict and warnings_):
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
