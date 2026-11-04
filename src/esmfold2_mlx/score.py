"""Score a predicted structure against an experimental one from the PDB."""

import urllib.request
from pathlib import Path

from .paths import cache_dir

import numpy as np


def _ca_coords(pdb_text: str, chain: str | None = None) -> dict[int, np.ndarray]:
    coords = {}
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL"):
            break  # NMR entries: first model only
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in " A":
            if chain is None:
                chain = line[21]
            if line[21] == chain:
                coords[int(line[22:26])] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return coords


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD after optimal rigid superposition of P onto Q."""
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U @ Vt))  # avoid reflections
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean()))


def ca_rmsd_to_pdb(pred_path: Path, pdb_id: str) -> float:
    cache = cache_dir() / "pdb" / f"{pdb_id.upper()}.pdb"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", cache)
    pred = _ca_coords(Path(pred_path).read_text())
    ref = _ca_coords(cache.read_text())
    # Prediction numbers residues from 1 in sequence order; match by position.
    shared = sorted(set(pred) & set(ref))
    return kabsch_rmsd(np.stack([pred[i] for i in shared]), np.stack([ref[i] for i in shared]))
