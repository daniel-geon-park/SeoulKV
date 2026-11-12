"""Shared helpers for the production distributed Stage-1.5 ingest."""

import torch


PREFIX_PAGE_BYTES = 4096


def attach(scorer, ingest):
    """Attach metadata for ingestion fused into the scorer program."""
    if not scorer.get("cluster_ingest_fused"):
        raise ValueError("real Stage1.5 must include ingestion in the scorer program")
    return dict(
        scorer,
        cluster_ingest=ingest,
        stage15_cluster_size=None,
        stage15_cluster_fetch_bytes=None,
        stage15_key_prefix_bytes=64,
        stage15_scorer_page_bytes=256,
        stage15_prefix_physical_page_bytes=PREFIX_PAGE_BYTES,
        stage15_input_contract=(
            "cluster IDs/counts; Stage1.5 bulk-fetches cluster blocks then "
            "deduplicates tokens"
        ),
    )


def verify(tt, bridge, cells, selected):
    """Verify the sole fixed-worker distributed-ingest layout."""
    from seoulkv4._codecs.index import expand_lists
    from seoulkv4._kernels.common.seoulkv4_cluster_blocks import encounter_union

    status = tt.to_torch(bridge["counts"]).reshape(len(cells), 16).long()
    direct_count_only = not bridge.get("diagnostics", True)
    if direct_count_only:
        pools = []
        for slot, cell in enumerate(cells):
            if status[slot, 1]:
                raise RuntimeError(
                    f"CSR bridge slot{slot} live={int(status[slot, 0])} "
                    f"status={int(status[slot, 1])}; no truncation allowed"
                )
            expected = encounter_union(cell, selected[slot])
            assert int(status[slot, 0]) == len(expected)
            assert torch.equal(
                expected.sort().values, expand_lists(cell, selected[slot])
            )
            pools.append(expected)
        bridge["input_prefix_verified"] = False
        bridge["direct_count_verified"] = True
        return pools, status

    if bridge.get("scorer_layout") != "fixed-worker-live-prefix-v1":
        raise ValueError("obsolete Stage1.5 ingest layout")
    if bridge.get("prefix_layout") != "narrow-key-major-v1":
        raise ValueError("obsolete Stage1.5 prefix layout")

    from seoulkv4._kernels.stage15.seoulkv4_stage15_distributed_ingest import (
        balanced_worker_ranges,
    )

    ids = tt.to_torch(bridge["ids"]).reshape(len(cells), -1).long()
    prefixes = tt.to_torch(bridge["prefix"]).reshape(len(cells), -1, 64)
    pools = []
    for slot, cell in enumerate(cells):
        if status[slot, 1]:
            raise RuntimeError(
                f"CSR bridge slot{slot} live={int(status[slot, 0])} "
                f"status={int(status[slot, 1])}; no truncation allowed"
            )
        live = int(status[slot, 0])
        expected = encounter_union(cell, selected[slot])
        ranges = balanced_worker_ranges(
            live, bridge["workers_per_cell"][slot], bridge["capacity"]
        )
        pool = torch.cat(
            [ids[slot, first : first + hi - lo] for lo, hi, first in ranges]
        )
        for lo, hi, first in ranges:
            assert torch.equal(ids[slot, first : first + hi - lo], expected[lo:hi])
            assert torch.equal(
                prefixes[slot, first : first + hi - lo],
                bridge["host_prefix"][slot, expected[lo:hi]],
            )
        assert torch.equal(pool, expected)
        assert torch.equal(pool.sort().values, expand_lists(cell, selected[slot]))
        counts = cell["offsets"][1:].long() - cell["offsets"][:-1].long()
        chosen = counts[selected[slot].long()]
        assert int(status[slot, 2]) == int(chosen.sum())
        assert int(status[slot, 4]) == int(chosen.sum()) * 64
        metadata_bytes = bridge["layouts"][slot].get("posting_metadata_bytes", 4)
        expected_bytes = chosen * 64 + ((chosen * metadata_bytes + 63) // 64) * 64
        assert int(status[slot, 5]) == int(expected_bytes.sum())
        assert int(status[slot, 6]) == int((chosen > 0).sum())
        pools.append(pool)
    bridge["input_prefix_verified"] = True
    bridge["direct_count_verified"] = True
    return pools, status
