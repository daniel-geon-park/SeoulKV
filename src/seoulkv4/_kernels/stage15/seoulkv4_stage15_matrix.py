"""Stage-1.5 matrix-query layout used by the production direct program."""

from __future__ import annotations

import torch

from seoulkv4._kernels.stage15.stage15_sfpu_native import coordinate_order


def prepare_matrix_query(q):
    """Tile ``[slots,4,256]`` q15 and absorb the prefix's x16 scale."""
    if q.ndim != 3 or q.shape[1:] != (4, 256):
        raise ValueError("matrix Stage 1.5 requires [slots,4,256] q15")
    out = torch.zeros(
        (1, q.shape[0], 32, 256), dtype=torch.bfloat16, device=q.device
    )
    order = coordinate_order().to(q.device)
    out[0, :, :4] = q[..., order] * 16
    return out
