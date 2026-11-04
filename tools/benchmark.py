"""Official benchmark: every headline configuration, timed the same way, in one run.

Runs bench_e2e.py for each configuration below, one after another, then
writes outputs/benchmark/<date>.json and outputs/benchmark/latest.md with the
machine and library versions. ~10 min. Close other GPU-heavy apps first.

Usage:
    python tools/benchmark.py
    python tools/benchmark.py --report-only   # rewrite latest.md from the last saved run
"""

import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import torch

from esmfold2_mlx.variants import DEFAULT_STACK, DEFAULT_STACK_6B

PY = sys.executable
CONFIGS = [  # (label, model, stack, steps, memtrim)
    ("Upstream code", "300M", "", 50, "none"),
    ("+ fast_trunk", "300M", "fast_trunk", 50, "none"),
    ("+ 20 steps", "300M", "fast_trunk", 20, "none"),
    ("Default stack, 50 steps", "300M", DEFAULT_STACK, 50, "default"),
    ("Default stack", "300M", DEFAULT_STACK, 20, "default"),
    ("Default stack, 600M", "600M", DEFAULT_STACK, 20, "default"),
    ("6B: upstream code (fp16 ESMC)", "6B-torch16", "", 20, "none"),
    ("6B: default stack (8-bit ESMC)", "6B", DEFAULT_STACK_6B, 20, "default"),
]


def latest(rows):
    """Last row per protein: a same-day re-run reuses the tags, so older rows must not leak in."""
    return list({r["protein"]: r for r in rows}.values())


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
    cores = next((l.split(":")[1].strip() for l in gpu.splitlines() if "Total Number of Cores" in l), "?")
    return {"chip": chip, "gpu_cores": cores, "macos": platform.mac_ver()[0],
            "torch": torch.__version__, "mlx": mx.__version__}


def main():
    if "--report-only" in sys.argv:  # regenerate latest.md from the newest saved run
        out = Path("outputs/benchmark")
        newest = sorted(out.glob("20*.json"))[-1]
        d = json.loads(newest.read_text())
        write_report(out, newest.stem, d["machine"], d["results"])
        return
    stamp = datetime.date.today().isoformat()
    results = []
    for label, model, stack, steps, memtrim in CONFIGS:
        tag = f"official:{stamp}:{label}"
        subprocess.run([PY, str(Path(__file__).with_name("bench_e2e.py")), "--model", model, "--variants", stack,
                        "--steps", str(steps), "--memtrim", memtrim, "--tag", tag], check=True)
        rows = [json.loads(l) for l in Path("outputs/e2e_results.jsonl").read_text().splitlines()]
        results.append({"label": label, "model": model, "stack": stack, "steps": steps, "memtrim": memtrim,
                        "rows": latest(r for r in rows if r["tag"] == tag)})

    out = Path("outputs/benchmark")
    out.mkdir(parents=True, exist_ok=True)
    info = machine()
    (out / f"{stamp}.json").write_text(json.dumps({"machine": info, "results": results}, indent=1))

    write_report(out, stamp, info, results)


def write_report(out, stamp, info, results):
    for res in results:
        res["rows"] = latest(res["rows"])
    proteins = [r["protein"] for r in results[0]["rows"]]
    # Speed-ups are against the *same model family's* upstream row (stack ""), so the
    # 6B rows are compared with upstream 6B, not with upstream 300M.
    family = lambda m: "6B" if m.startswith("6B") else "exp"  # noqa: E731
    base = {family(r["model"]): {x["protein"]: x["median_s"] for x in r["rows"]} for r in results if r["stack"] == ""}
    md = [f"# ESMFold2 benchmark, {stamp}", "",
          f"{info['chip']} ({info['gpu_cores']}-core GPU), macOS {info['macos']}, torch {info['torch']}, MLX {info['mlx']}.",
          "Median of 3 folds after warm-up, 3 loops, 1 sample, seed 0. Seconds per fold "
          "(speed-up vs the same model's upstream code).", "",
          "| Configuration | Steps | " + " | ".join(proteins) + " |",
          "|---|---|" + "---|" * len(proteins)]
    for res in results:
        ref = base[family(res["model"])]
        cells = [f"{r['median_s']:.2f} ({ref[r['protein']] / r['median_s']:.1f}x)" for r in res["rows"]]
        md.append(f"| {res['label']} | {res['steps']} | " + " | ".join(cells) + " |")
    (out / "latest.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
