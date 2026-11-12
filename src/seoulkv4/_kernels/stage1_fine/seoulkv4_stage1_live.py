"""Live Stage-I Coarse -> Fine production handoff for SeoulKV4."""

from __future__ import annotations

import os
import torch

from seoulkv4._kernels.common.seoulkv4_cluster_blocks import encounter_union
from seoulkv4._kernels.stage1_coarse.seoulkv4_coarse_score import (
    build_all as build_coarse,
)
from seoulkv4._kernels.stage1_coarse.seoulkv4_coarse_select import (
    readback as read_coarse,
)
from seoulkv4._kernels.stage1_fine.seoulkv4_fine_packed import (
    build_all as build_fine,
)
from seoulkv4._kernels.query_prepare import host as query_prepare
from seoulkv4._kernels.stage1_coarse.abi import CoarseSemaphore


def _worker_groups(device):
    grid = device.compute_with_storage_grid_size()
    points = [(x, y) for y in range(grid.y) for x in range(grid.x)]
    if len(points) < 108:
        raise ValueError("live Stage I requires 108 Tensix workers")
    return [points[:28], points[28:56], points[56:84], points[84:108]]


def _overlap_groups(device):
    """Reserve eight Q workers; Fine keeps its original 108 workers."""
    grid = device.compute_with_storage_grid_size()
    points = [(x, y) for y in range(grid.y) for x in range(grid.x)]
    query_count = 4 * query_prepare.abi.WRITERS_PER_GROUP
    cohorts = min(27, (len(points) - query_count) // 4)
    if cohorts < 16:
        raise ValueError("Coarse/Q overlap requires at least four cohorts per cell")
    groups, at = [], 0
    for cell in range(4):
        workers = 4 * (cohorts // 4 + (cell < cohorts % 4))
        groups.append(points[at : at + workers])
        at += workers
    return groups, points[at : at + query_count]


def _selected_clusters(tt, fine_state):
    packet = tt.to_torch(fine_state["output"]).flatten().long()
    count, status = int(packet[0]), int(packet[1])
    if status:
        raise RuntimeError(f"Fine status={status}")
    if not 0 <= count <= packet.numel() - 16:
        raise RuntimeError(f"invalid Fine cluster count={count}")
    return packet[16 : 16 + count].clone()


def _read_handoff(tt, cells, coarse_states, fine_states, args):
    """Read the query-dependent Coarse/Fine handoff after synchronization."""
    coarse_tags = []
    selected = []
    pools = []
    for cell, coarse_state, fine_state in zip(cells, coarse_states, fine_states):
        tags = read_coarse(tt, coarse_state["selection"])
        coarse_tags.append(tags.clone())
        cell["coarse_ids"] = (tags & 0x0FFFFFFF).int()
        cell["head_mask"] = ((tags >> 28) & 15).byte()
        fine_state["cell"] = cell
        chosen = _selected_clusters(tt, fine_state)
        selected.append(chosen)
        pools.append(encounter_union(cell, chosen))

    counts = [len(pool) for pool in pools]
    if min(counts) < args.skv4_stage15_m or max(counts) > args.skv4_stage15_candidates:
        raise ValueError(
            f"live Stage-I key counts {counts} outside "
            f"[{args.skv4_stage15_m},{args.skv4_stage15_candidates}]; "
            "change capacity, never truncate"
        )
    padded = torch.zeros((4, args.skv4_stage15_candidates), dtype=torch.long)
    for index, pool in enumerate(pools):
        padded[index, : len(pool)] = pool
    return coarse_tags, selected, pools, counts, padded


def prepare(tt, device, k, q, args, *, index):
    """Build both Stage-I programs before observing any runtime survivor count."""
    if (args.n_conc, args.seq_len, args.n_kv_heads, args.n_heads) != (1, 1, 4, 16):
        raise ValueError("live Stage I currently supports c1/seq1/GQA4x4 only")
    attention_zones = index._zones
    live_cells = index._live_cells(q)

    groups = _worker_groups(device)
    coarse_groups, query_points = _overlap_groups(device)
    prepared_query = query_prepare.build(
        tt,
        device,
        k.float().abs().amax(dim=1).clamp_min(1e-9),
        [cell["lam"] for cell in live_cells],
        profile=os.environ.get("SKV4_QUERY_PREPARE_PROFILE", "0") == "1",
        core_points=query_points,
        consumer_groups=coarse_groups,
        consumer_ready_semaphore=int(CoarseSemaphore.QUERY_READY),
    )
    query_prepare.bind(tt, prepared_query, q)
    coarse = build_coarse(
        tt,
        device,
        live_cells,
        coarse_groups,
        select=True,
        debug_heads=getattr(args, "skv4_debug_indices", False),
        external_queries=prepared_query["coarse"],
        external_norms=prepared_query["norm"],
        wait_query=True,
    )
    qp = prepared_query["program"]
    cp = coarse["descriptor"]
    combined = tt.ProgramDescriptor(
        kernels=[*qp.kernels, *cp.kernels],
        # QP common allocations stay before owner-only allocations; Coarse
        # ranges are disjoint, so their CB IDs/addresses cannot disturb them.
        cbs=[*qp.cbs, *cp.cbs],
        semaphores=[*qp.semaphores, *cp.semaphores],
    )
    combined_io = [*prepared_query["io"], *coarse["io"]]

    def run_coarse():
        tt.generic_op(combined_io, combined)

    tag_tensors = [state["selection"]["output"] for state in coarse["states"]]
    upper_tensors = [state["upper"] for state in coarse["states"]]
    fine = build_fine(
        tt,
        device,
        live_cells,
        groups,
        score=True,
        select=True,
        external_tags=tag_tensors,
        dynamic_external=True,
        shared_upper=upper_tensors,
        external_queries=prepared_query["fine"],
    )

    # Untimed initialization supplies host references/capacity checks only.
    # Fine itself was already built without observing these runtime counts.
    run_coarse()
    fine["run"]()
    tt.synchronize_device(device)
    coarse_tags, selected, pools, counts, padded = _read_handoff(
        tt, live_cells, coarse["states"], fine["states"], args
    )

    def run():
        run_coarse()
        fine["run"]()

    return dict(
        cells=live_cells,
        attention_zones=attention_zones,
        meta={"nprobe_source": index._nprobe_source},
        coarse=coarse,
        coarse_tags=coarse_tags,
        fine=fine["states"],
        selected=selected,
        pools=pools,
        counts=counts,
        candidates=padded,
        run=run,
        program_count=2,
        programs=(dict(descriptor=combined, io=combined_io, run=run_coarse), fine),
        query_prepare=prepared_query,
        query_prepare_overlapped=True,
        worker_groups=[len(group) for group in groups],
        coarse_worker_groups=[len(group) for group in coarse_groups],
        cluster_ids=[state["output"] for state in fine["states"]],
    )


def bind_query(tt, state, q):
    """Overwrite the one stable raw GDDR query buffer."""
    query_prepare.bind(tt, state["query_prepare"], q)


def refresh_handoff(tt, state, args):
    """Refresh host references from the most recently completed query."""
    values = _read_handoff(
        tt, state["cells"], state["coarse"]["states"], state["fine"], args
    )
    coarse_tags, selected, pools, counts, candidates = values
    state.update(
        coarse_tags=coarse_tags,
        selected=selected,
        pools=pools,
        counts=counts,
        candidates=candidates,
    )
    return candidates, counts


def verify_initial(tt, state, stage15):
    """Verify the initialization query before any per-run Q replacement."""
    for coarse_state, expected_tags in zip(
        state["coarse"]["states"], state["coarse_tags"]
    ):
        got = read_coarse(tt, coarse_state["selection"])
        if not torch.equal(got, expected_tags):
            raise AssertionError("live Coarse tags changed across replay")
    for fine_state, expected in zip(state["fine"], state["selected"]):
        got = _selected_clusters(tt, fine_state)
        if not torch.equal(got, expected):
            raise AssertionError("live Fine cluster union changed across replay")
    from seoulkv4._kernels.stage15.seoulkv4_stage15_ingest import verify

    _, status = verify(tt, stage15["cluster_ingest"], state["cells"], state["selected"])
    stage15["cluster_ingest_status"] = status.tolist()
    state["live_verified"] = True


def metadata(state):
    return dict(
        summary_precomputed=True,
        summary_codec="group-scale-32-v1",
        coarse="live int2-prepacked unsigned-native-BFP4 dot, FP32 recenter/intervals and runtime compact survivor tags",
        coarse_upper_layout="native-swar-32-v1 (64B/centroid, one shared T1 tensor)",
        fine="runtime-dynamic merged 2->4-bit Fine, block-cached shared upper words",
        selection=(
            "Coarse K512 lower-bound thresholds -> upper-bound survivor masks; "
            "Fine odd-BF16/post-scale K512"
        ),
        output_contract="live cluster IDs/counts in T1; no expanded token handoff",
        centroid_codec="fp16-scale-odd-v1",
        centroid_scale_group=32,
        centroid_scale_t1_bytes=sum(
            ((len(cell["group_fine_scale"]) + 15) // 16) * 64 for cell in state["cells"]
        ),
        interval_output_t2_bytes=0,
        debug_head_positions_t2_bytes=sum(
            0 if item["heads"] is None else 4 * int(cell["p"]) * 4
            for item, cell in zip(state["coarse"]["states"], state["cells"])
        ),
        coarse_survivors=[len(tags) for tags in state["coarse_tags"]],
        reference_live_candidates=state["counts"],
        program_count=state["program_count"],
        workers=sum(state["worker_groups"]),
        worker_groups=state["worker_groups"],
        coarse_worker_groups=state["coarse_worker_groups"],
        query_prepare_workers=len(state["query_prepare"]["core_points"]),
        query_prepare_overlapped=state["query_prepare_overlapped"],
        nprobe_source=state["meta"].get("nprobe_source"),
        live_verified=state.get("live_verified", False),
    )
