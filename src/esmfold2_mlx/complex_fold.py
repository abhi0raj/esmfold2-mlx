"""Fold protein complexes (several chains) with any loaded model and speed stack.

    from esmfold2_mlx.complex_fold import fold_complex
    res = fold_complex(model, ["MKT...", "GSH..."], seed=0)   # MolecularComplexResult
    open("out.cif", "w").write(res.complex.to_mmcif())

Single chains go through ``infer_protein``; complexes need upstream's input builder, which
sets chain ids (asym_id), per-chain residue numbering and the chain-pair features.
"""

import torch

from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput

_BUILDER = None


def builder():
    global _BUILDER
    if _BUILDER is None:
        _BUILDER = ESMFold2InputBuilder()
    return _BUILDER


@torch.no_grad()
def fold_complex(model, chains, seed=0, num_loops=3, num_sampling_steps=20, chain_ids=None):
    chain_ids = chain_ids or [chr(ord("A") + i) for i in range(len(chains))]
    inp = StructurePredictionInput(sequences=[ProteinInput(id=c, sequence=s) for c, s in zip(chain_ids, chains)])
    dev = next(model.parameters()).device
    features, infos = builder().prepare_input(inp, seed=seed, device=dev)
    out = model(**features, num_loops=num_loops, num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=1, seed=seed)
    if "plddt" not in out:  # experimental checkpoints have no confidence head
        out["plddt"] = torch.zeros(out["sample_atom_coords"].shape[:2])
    return builder().decode(out, features, infos)
