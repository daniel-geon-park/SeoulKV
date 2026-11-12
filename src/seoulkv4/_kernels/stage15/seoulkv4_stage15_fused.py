"""Production Blackhole Stage-1.5 scorer and shared-union L1 handoff.

The only supported implementation is the single-program producer-owned-L1
pipeline.  Historical separate scorer/TopK/union programs were slower and are
intentionally not selectable here.
"""

from __future__ import annotations

from seoulkv4._kernels.stage2.seoulkv4_stage2_fused import GQA_GROUP

__all__ = ["build_stage15"]


def _stage15_groups(ttnn, device, slots, workers, group=GQA_GROUP, core_points=None):
    """Pack ``slots * workers`` disjoint physical four-core rectangles."""
    need_lines = slots * workers
    need_cores = need_lines * group
    if core_points is not None:
        points = list(core_points)
        if len(points) < need_cores:
            raise ValueError(f"need {need_cores} worker cores, got {len(points)}")
        width = workers * group
        return [points[i * width : (i + 1) * width] for i in range(slots)]

    grid = device.compute_with_storage_grid_size()
    reverse = {}
    for ly in range(grid.y):
        for lx in range(grid.x):
            p = device.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
            reverse[(p.x, p.y)] = (lx, ly)

    unused = set(reverse)
    groups = []
    xs = [p[0] for p in reverse]
    ys = [p[1] for p in reverse]
    for y0 in range(min(ys), max(ys)):
        for x0 in range(min(xs), max(xs)):
            physical = [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
            if all(p in unused for p in physical):
                groups.append([reverse[p] for p in physical])
                unused.difference_update(physical)
                if len(groups) == need_lines:
                    break
        if len(groups) == need_lines:
            break

    def take_runs(axis):
        fixed_values = sorted({p[1 - axis] for p in unused})
        for fixed in fixed_values:
            varying = sorted(p[axis] for p in unused if p[1 - axis] == fixed)
            run = []
            previous = None
            for value in varying + [None]:
                if value is not None and (previous is None or value == previous + 1):
                    run.append(value)
                else:
                    while len(run) >= group:
                        chunk, run = run[:group], run[group:]
                        physical = (
                            [(v, fixed) for v in chunk]
                            if axis == 0
                            else [(fixed, v) for v in chunk]
                        )
                        groups.append([reverse[p] for p in physical])
                        unused.difference_update(physical)
                        if len(groups) == need_lines:
                            return True
                    run = [] if value is None else [value]
                previous = value
        return False

    if len(groups) < need_lines and not take_runs(0):
        take_runs(1)
    if len(groups) < need_lines:
        raise ValueError(
            f"need {need_lines} disjoint physical {group}-core rectangles "
            f"({need_cores} cores); packed only {len(groups)} groups"
        )
    flat = [point for members in groups[:need_lines] for point in members]
    width = workers * group
    return [flat[i * width : (i + 1) * width] for i in range(slots)]


def build_stage15(
    ttnn,
    device,
    q15,
    prefix2,
    candidates,
    cluster_ids,
    *,
    slots,
    context,
    n15=13056,
    m15=2048,
    union_m15=1600,
    workers=6,
    group=GQA_GROUP,
    core_points=None,
    debug_snapshots=False,
    posting_lists=None,
):
    """Build the sole production Stage-1.5 program.

    Merged Stage-I Fine supplies cluster IDs and the distributed ingest
    expands/deduplicates spill-2 postings directly into the narrow scorer
    handoff.  The historical synthetic pre-swizzled input is not supported.
    """
    if posting_lists is None:
        raise ValueError("production Stage1.5 requires Stage-I posting lists")
    if candidates is not None:
        raise ValueError("cluster input must not also supply a token pool")
    if len(posting_lists) != slots or any(
        cell["context"] != context for cell in posting_lists
    ):
        raise ValueError("CSR cell count/context mismatch")
    if workers < group or group != 4:
        raise ValueError("Stage 1.5 requires G=4 and at least four shards")
    if context % 4 or n15 % (workers * group * 32) or m15 > n15:
        raise ValueError(
            "require context divisible by 4, N15 divisible by W*G*32, and M15<=N15"
        )
    if m15 != 2048 or n15 // workers < m15:
        raise ValueError(
            "native TopK-XL Stage 1.5 requires M15=2048 and LOCAL>=M15"
        )
    if not 0 < union_m15 <= m15 or union_m15 % 16:
        raise ValueError("union_m15 must be an aligned value in (0, M15]")

    local = n15 // workers
    blocks = (local + 1023) // 1024
    if blocks > 4 or context > (1 << 20):
        raise ValueError("Stage 1.5 supports ceil(N15/W/1024)<=4 and context<=1M")
    groups = _stage15_groups(ttnn, device, slots, workers, group, core_points)

    from seoulkv4._kernels.stage15.seoulkv4_stage15_distributed_ingest import (
        build as build_ingest,
    )

    ingest = build_ingest(
        ttnn,
        device,
        posting_lists,
        cluster_ids,
        prefix2,
        capacity=n15,
        minimum=m15,
        groups=groups,
        diagnostics=debug_snapshots,
    )
    prefix2, candidates, cluster_ids = (
        ingest["prefix"],
        ingest["ids"],
        ingest["clusters"],
    )

    from seoulkv4._kernels.stage15.seoulkv4_stage15_direct import build_direct

    scorer = build_direct(
        ttnn,
        device,
        q15,
        prefix2,
        candidates,
        cluster_ids,
        groups=groups,
        slots=slots,
        context=context,
        n15=n15,
        workers=workers,
        pad=blocks * 1024,
        union_m15=union_m15,
        debug_snapshots=debug_snapshots,
        cluster_ingest=ingest,
    )
    from seoulkv4._kernels.stage15.seoulkv4_stage15_ingest import attach

    return attach(scorer, ingest)
