"""Shims for PyTorch ops that ESMFold2 uses but Apple's MPS backend lacks.

Import before running the model. Each patch routes only the unsupported
case to CPU and leaves everything else on the GPU.
"""

import torch

_scatter_reduce_ = torch.Tensor.scatter_reduce_


def _scatter_reduce_mps_int64(self, dim, index, src, reduce, *, include_self=True):
    # MPS has no int64 scatter_reduce. ESMFold2 only hits this for small
    # token->residue index bookkeeping, so the CPU round-trip is free.
    if self.device.type == "mps" and self.dtype == torch.int64:
        res = _scatter_reduce_(self.cpu(), dim, index.cpu(), src.cpu(), reduce, include_self=include_self)
        return self.copy_(res)
    return _scatter_reduce_(self, dim, index, src, reduce, include_self=include_self)


torch.Tensor.scatter_reduce_ = _scatter_reduce_mps_int64
