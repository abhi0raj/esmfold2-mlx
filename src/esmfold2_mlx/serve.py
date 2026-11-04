"""Keep-loaded ESMFold2: pay imports, model load, compile and shader warm-up once,
then fold any number of sequences from a tiny client.

    # start (foreground; Ctrl-C to stop). Prints "ready" when the warm-up fold is done.
    esmfold2-mlx serve start --model 6B                 # default stack + memtrim defaults
    # fold from another shell: the client imports only socket/json (~0.05 s start)
    esmfold2-mlx serve fold --model 6B --seq MKT... --name mine     # -> outputs/mine.pdb
    esmfold2-mlx serve fold --model 6B --fasta designs.fasta
    esmfold2-mlx serve stop --model 6B

In Python (notebooks, pipelines) use the same object the server uses:

    from esmfold2_mlx import Folder
    f = Folder("6B")                  # load + warm-up
    res = f.fold("MKT...", "mine")    # dict: pdb path, seconds, plddt, ptm

Folds are the same calls fold.py makes (same stack, steps rule, seed, PDB writer),
so outputs are identical to fold.py's for the same arguments. One fold at a time;
requests queue on the socket.
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

from .paths import cache_dir

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def sock_path(model):
    return str(cache_dir() / f"esmfold2_{model.split('/')[-1]}.sock")


class Folder:
    """A loaded, warmed-up model with fold.py's per-fold behaviour."""

    def __init__(self, model="300M", stack=None, num_loops=3, lm_dtype="fp32", memtrim="default",
                 warmup=True, out_dir="outputs", log=print):
        import warnings
        warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
        warnings.filterwarnings("ignore", message=".*linalg_svd.*")
        warnings.filterwarnings("ignore", message=".*pre-alignment ESMFold2 field.*")
        import torch

        from . import variants
        from .fold import LM_DTYPES, VALIDATED_LENGTHS
        self.torch, self.validated = torch, VALIDATED_LENGTHS
        t0 = time.time()
        self.model_name = model
        self.stack = variants.default_stack(model) if stack is None else stack
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = variants.apply_variants(variants.load_model(model, device, LM_DTYPES[lm_dtype]),
                                             self.stack.split(","))
        if memtrim == "default":
            memtrim = variants.default_memtrim(model)
        variants.apply_memtrim(self.model, model, memtrim or "none")
        self.memtrim = memtrim
        self.num_loops = num_loops
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.load_s = time.time() - t0
        log(f"loaded {model} [{self.stack}] memtrim={memtrim} in {self.load_s:.1f}s")
        if warmup:
            t0 = time.time()
            self.fold(UBIQUITIN, None)  # Metal shaders, MLX compile, MPSGraph caches, lazy ESMC weights
            log(f"warm-up fold {time.time() - t0:.1f}s")

    def fold(self, seq, name, num_sampling_steps=None, seed=0, num_samples=1, num_loops=None):
        torch = self.torch
        seq = seq.strip().upper()
        steps = num_sampling_steps or (20 if len(seq) in self.validated else 50)
        t0 = time.time()
        out = self.model.infer_protein(seq, seed=seed, num_loops=num_loops or self.num_loops,
                                       num_sampling_steps=steps, num_diffusion_samples=num_samples)
        res = {"name": name, "length": len(seq), "steps": steps}
        if "ptm" in out:
            res["plddt"] = round(float(out["plddt"].mean()), 4)
            res["ptm"] = round(float(out["ptm"].mean()), 4)
        if "plddt" not in out:
            out["plddt"] = torch.zeros(out["token_attention_mask"].shape)
        res["fold_s"] = round(time.time() - t0, 3)
        if name is not None:
            path = self.out_dir / f"{name}.pdb"
            path.write_text(self.model.output_to_pdb(out))
            res["pdb"] = str(path)
        res["seconds"] = round(time.time() - t0, 3)
        return res


def _serve(args):
    folder = Folder(args.model, stack=args.stack, num_loops=args.num_loops, memtrim=args.memtrim,
                    out_dir=args.out_dir,
                    log=lambda m: print(m, flush=True))
    path = sock_path(args.model)
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(16)
    print(f"ready on {path}", flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            with conn, conn.makefile("rw") as f:
                for line in f:
                    req = json.loads(line)
                    if req.get("cmd") == "stop":
                        f.write(json.dumps({"ok": True}) + "\n")
                        f.flush()
                        return
                    try:
                        res = folder.fold(req["seq"], req.get("name"), req.get("num_sampling_steps"),
                                          req.get("seed", 0), req.get("num_samples", 1), req.get("num_loops"))
                    except Exception as e:  # report to the client, keep serving
                        res = {"error": f"{type(e).__name__}: {e}"}
                    f.write(json.dumps(res) + "\n")
                    f.flush()
    finally:
        srv.close()
        os.unlink(path)


def _read_fasta(path):
    # same parsing as fold.read_fasta, without importing torch
    import re
    records = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            name = re.sub(r"[^\w.-]+", "_", line[1:].split()[0]) if line[1:].strip() else f"seq{len(records)}"
            records.append([name, ""])
        elif line.strip():
            records[-1][1] += line.strip().upper()
    return [(n, s) for n, s in records if s]


def _client(args):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path(args.model))
    except (FileNotFoundError, ConnectionRefusedError):
        sys.exit(f"no server for {args.model}; start one with: esmfold2-mlx serve start --model {args.model}")
    with s, s.makefile("rw") as f:
        if args.cmd == "stop":
            f.write(json.dumps({"cmd": "stop"}) + "\n")
            f.flush()
            print(f.readline().strip())
            return
        jobs = _read_fasta(args.fasta) if args.fasta else [(args.name, args.seq)]
        for name, seq in jobs:
            req = {"name": name, "seq": seq, "seed": args.seed}
            if args.num_sampling_steps:
                req["num_sampling_steps"] = args.num_sampling_steps
            f.write(json.dumps(req) + "\n")
            f.flush()
            res = json.loads(f.readline())
            if "error" in res:
                print(f"{name}: ERROR {res['error']}")
                continue
            conf = f", pLDDT {res['plddt']:.2f}, pTM {res['ptm']:.2f}" if "ptm" in res else ""
            print(f"{name}: {res['length']} aa, {res['steps']} steps, {res['seconds']:.2f}s{conf} -> {res['pdb']}",
                  flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["start", "fold", "stop"])
    p.add_argument("--model", default="300M")
    p.add_argument("--stack", default=None)
    p.add_argument("--num-loops", type=int, default=3)
    p.add_argument("--memtrim", default="default", help="memtrim options, 'none', or 'default' (per model)")
    p.add_argument("--out-dir", default="outputs", help="where the server writes structures")
    p.add_argument("--seq", default=UBIQUITIN)
    p.add_argument("--name", default="ubiquitin")
    p.add_argument("--fasta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-sampling-steps", type=int, default=None)
    args = p.parse_args()
    if args.cmd == "start":
        _serve(args)
    else:
        _client(args)


if __name__ == "__main__":
    main()
