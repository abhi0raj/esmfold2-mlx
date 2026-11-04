"""Faster folding trunk on MPS, numerically identical to upstream.

Upstream projects the pair to one 1024-wide tensor, then split()s it into a
signal half and a gate half. On CUDA a fused kernel hides that; on MPS,
elementwise ops on those non-contiguous halves take a slow strided path
(~150 ms vs ~15 ms per tri-mul at L=238). Splitting the *weights* instead of
the *activations* gives contiguous outputs and the same numbers.

Usage:
    from esmfold2_mlx import fast_trunk
    fast_trunk.apply(model)
"""

import torch
import torch.nn.functional as F
from esm.models.esmfold2.layers import TriangleMultiplicativeBlock, Transition


def _trimul_forward(self, pair_grid, visibility=None):
    if visibility is None:
        visibility = pair_grid.new_ones(pair_grid.shape[:-1])
    normalized_grid = self.norm_start(pair_grid)
    signal = F.linear(normalized_grid, self._w_signal)
    gate_logits = F.linear(normalized_grid, self._w_gate)
    routed = signal * torch.sigmoid(gate_logits) * visibility.unsqueeze(-1)
    left_stream, right_stream = routed.float().chunk(2, dim=-1)
    if self._chunk_size is not None:
        contracted = self._triangular_contract_chunked(left_stream, right_stream, self._chunk_size)
    else:
        contracted = self._triangular_contract(left_stream, right_stream)
    mixed = self.proj_emit(self.norm_mix(contracted))
    return mixed * torch.sigmoid(self.proj_gate(normalized_grid))


def _transition_forward(self, x):
    def ffn(x):
        h = self.norm(x)
        return x + F.linear(F.silu(F.linear(h, self._w1)) * F.linear(h, self._w2), self.ffn.w3.weight)

    if self._chunk_size is None or x.shape[1] <= self._chunk_size:
        return ffn(x)
    return torch.cat([ffn(x[:, s : s + self._chunk_size]) for s in range(0, x.shape[1], self._chunk_size)], dim=1)


def apply(model, chunk_size: int | None = None) -> None:
    """Patch every tri-mul and transition in ``model`` in place.

    ``chunk_size=None`` runs the triangle contraction in one shot, which is ~3x
    faster on MPS; pass an int (e.g. 64) to cap memory for long sequences.
    """
    for m in model.modules():
        if isinstance(m, TriangleMultiplicativeBlock):
            assert m.proj_bundle.bias is None
            w = m.proj_bundle.weight.detach()
            m._w_signal = w[: 2 * m.latent_channels].contiguous()
            m._w_gate = w[2 * m.latent_channels :].contiguous()
            m.forward = _trimul_forward.__get__(m)
        elif isinstance(m, Transition):
            assert m.ffn.w12.bias is None and m.ffn.w3.bias is None
            w = m.ffn.w12.weight.detach()
            m._w1 = w[: m.ffn.hidden_features].contiguous()
            m._w2 = w[m.ffn.hidden_features :].contiguous()
            m.forward = _transition_forward.__get__(m)
    model.set_chunk_size(chunk_size)
