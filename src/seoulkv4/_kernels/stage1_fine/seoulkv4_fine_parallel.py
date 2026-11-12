"""Portable contracts for the parallel Fine rewrite (no device imports).

Exact per-centroid BF16 lookup tables remain 32 bytes. Cohort selection must
retain P per head, not P divided by the number of cohorts.
"""

from dataclasses import dataclass
import itertools
import math

import torch


def allocate_workers(live_counts, total_workers=108, max_workers=32):
    """Choose four-core Fine groups from current coarse-union sizes.

    The objective is query-local workload balance only: no selected answer,
    score, or sampled-history statistic participates.  Native K1024 requires
    every four-worker cohort to cover at most 1,024 padded centroids.
    """
    live_counts = tuple(int(v) for v in live_counts)
    if not live_counts or any(v <= 0 for v in live_counts):
        raise ValueError("positive live coarse counts required")
    if total_workers % 4 or max_workers % 4:
        raise ValueError("Fine worker counts must be multiples of four")
    choices = range(4, max_workers + 1, 4)
    best = None
    for workers in itertools.product(choices, repeat=len(live_counts)):
        if sum(workers) != total_workers:
            continue
        maxima = []
        valid = True
        for live, count in zip(live_counts, workers):
            blocks = max(4, math.ceil(live / 32))
            # Mirror build()'s contiguous worker partition exactly.
            widths = [
                blocks * (i + 1) // count - blocks * i // count for i in range(count)
            ]
            cohort_widths = [sum(widths[i : i + 4]) for i in range(0, count, 4)]
            if max(cohort_widths) > 32:
                valid = False
                break
            maxima.append(max(widths))
        if not valid:
            continue
        key = (
            max(maxima),
            tuple(sorted(maxima, reverse=True)),
            sum(v * v for v in maxima),
            workers,
        )
        if best is None or key < best[0]:
            best = (key, workers)
    if best is None:
        raise ValueError("no valid four-core Fine allocation")
    return best[1]


def coalesced_scale_page_slots(selected_ids, group_size=32, page_items=16):
    """Map selected IDs to compact group-scale page slots used by Fine.

    One scale covers ``group_size`` physical centroids and one 64-byte page
    contains ``page_items`` FP32 scales. Production IDs are ascending, so every
    physical scale page is fetched once.
    The mapping deliberately remains correct for arbitrary diagnostic order:
    revisiting a non-adjacent page creates another cache slot.
    """
    pages = [(int(i) // group_size) // page_items for i in selected_ids]
    slots = []
    fetched = []
    last = None
    slot = -1
    for page in pages:
        if page != last:
            fetched.append(page)
            slot += 1
            last = page
        slots.append(slot)
    return tuple(fetched), tuple(slots)


def truth_tables(levels):
    """Transpose the 16x16 BF16 bit matrix, without changing its byte size."""
    if levels.shape[-1] != 16 or not torch.isfinite(levels).all():
        raise ValueError("Fine requires sixteen finite levels per centroid")
    bits = levels.to(torch.bfloat16).contiguous().view(torch.int16).long() & 65535
    planes = torch.arange(16, device=bits.device)
    return (((bits[..., :, None] >> planes) & 1) << planes[:, None]).sum(-2).int()


def lookup_bits(tables, codes):
    """Independent integer reference for the SFPU variable-shift decoder."""
    if tables.shape[-1] != 16 or tables.shape[:-1] != codes.shape[:-1]:
        raise ValueError("table and code batch dimensions must match")
    if ((codes < 0) | (codes > 15)).any():
        raise ValueError("Fine codes must be in [0, 15]")
    bits = torch.zeros_like(codes, dtype=torch.int64)
    for bit in range(16):
        bits |= ((tables[..., bit, None].long() >> codes.long()) & 1) << bit
    return bits.to(torch.int16)


@dataclass(frozen=True)
class Cohort:
    cell: int
    begin: int
    end: int
    quarters: tuple[tuple[int, int], ...]


def schedule(live_counts, cohort_counts=(7, 7, 7, 6)):
    """Contiguous, 32-row-aligned quarters; ends include invalid tail lanes."""
    if len(live_counts) != len(cohort_counts):
        raise ValueError("one cohort count per cell is required")
    result = []
    for cell, (live, groups) in enumerate(zip(live_counts, cohort_counts)):
        if live < 0 or groups < 1:
            raise ValueError(
                "nonnegative live count and positive cohort count required"
            )
        tiles = (live + 31) // 32
        for group in range(groups):
            first, last = tiles * group // groups, tiles * (group + 1) // groups
            quarters = tuple(
                (
                    32 * (first + (last - first) * q // 4),
                    32 * (first + (last - first) * (q + 1) // 4),
                )
                for q in range(4)
            )
            result.append(Cohort(cell, first * 32, last * 32, quarters))
    return result


def tree_top_p(scores, allowed, p, cohorts):
    """CPU specification: masked local P, balanced exact-P pairwise merges.

    Returns source positions, NOT cluster IDs. This is an independent test
    oracle, not a host selection step in the decode path.
    """
    if scores.shape != allowed.shape or scores.ndim != 2 or scores.shape[0] != 4:
        raise ValueError("expected four head score and eligibility rows")
    if p < 1 or (allowed.sum(-1) < p).any():
        raise ValueError("each head must have at least P eligible centroids")
    masked = scores.masked_fill(~allowed, -torch.inf)
    frontiers = []
    for cohort in cohorts:
        end = min(cohort.end, scores.shape[-1])
        begin = min(cohort.begin, end)
        k = min(p, end - begin)
        frontiers.append(masked[:, begin:end].topk(k, dim=-1).indices + begin)
    while len(frontiers) > 1:
        merged = []
        for i in range(0, len(frontiers), 2):
            ids = torch.cat(frontiers[i : i + 2], dim=-1)
            ranks = masked.gather(1, ids).topk(min(p, ids.shape[-1]), dim=-1).indices
            merged.append(ids.gather(1, ranks))
        frontiers = merged
    return frontiers[0]
