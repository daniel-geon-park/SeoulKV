"""Build a query-independent Stage-I cell using Torch only."""

import math

import torch

from seoulkv4._clustering import _midrise, _refined_hierarchical_ivf
from seoulkv4._codecs.index import (
    GROUP_SCALE_SIZE,
    encode_group_scale_cell,
    fingerprint,
    validate_static_cell,
)


def centroid_count(retrieval_tokens):
    # About four keys per centroid before spill. Pad the codebook to full
    # 32-centroid scale groups even when sink/local are not tile-aligned.
    return math.ceil(retrieval_tokens / (4 * GROUP_SCALE_SIZE)) * GROUP_SCALE_SIZE


@torch.no_grad()
def build_stage1_cell(
    k, *, p, lam, seed, r0, r1, device=None, coarse_clusters=128, iters=10
):
    """Share the static index builder between the public API and fixture tool."""
    nc = centroid_count(r1 - r0)
    if not 0 <= r0 < r1 <= len(k) or r1 - r0 < nc or not 0 < p <= nc:
        raise ValueError("not enough retrieval keys for spill2 / requested nprobe")
    x = k[r0:r1].to(device=device or k.device, dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("retrieval keys must be finite")
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    centroids, vals, off, _, _, stats = _refined_hierarchical_ivf(
        x,
        nc,
        2,
        seed,
        coarse_clusters=min(coarse_clusters, max(2, nc // 2)),
        local_iters=iters,
    )
    cell, _ = encode_group_scale_cell(
        dict(
            centroids=_midrise(centroids, 4).half().cpu(),
            offsets=off.cpu().int(),
            values=vals.cpu().int() + r0,
            context=len(k),
            r0=r0,
            r1=r1,
            spill=2,
            p=p,
            lam=lam,
            seed=seed,
            key_hash=fingerprint(k.cpu().bfloat16()),
            builder=stats,
        )
    )
    validate_static_cell(cell)
    return cell
