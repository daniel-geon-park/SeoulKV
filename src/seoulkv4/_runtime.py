"""Assemble the unchanged SeoulKV4 device programs."""

import os

import torch

from seoulkv4._codecs.keys import _stage15_key_codec, encode_planes
from seoulkv4._kernels.query_prepare import abi as query_prepare_abi
from seoulkv4._kernels.stage1_fine import seoulkv4_stage1_live as stage1_live
from seoulkv4._kernels.stage15 import seoulkv4_stage15_fused as fused_stage15
from seoulkv4._kernels.stage16 import seoulkv4_stage16_fused as fused_stage16

PRODUCTION_PROGRAMS = (
    "query_prepare_coarse",
    "fine",
    "stage15",
    "refinement_attention",
)


def _ttnn():
    try:
        import ttnn
    except ImportError as exc:
        raise RuntimeError(
            "SeoulKV4 requires a caller-supplied Blackhole TTNN runtime (validated with 0.77.0)"
        ) from exc
    return ttnn


def _chain_mark(message):
    if os.environ.get("SKV4_CHAIN_DEBUG") == "1":
        print(f"[skv4-chain] {message}", flush=True)


def _stage16_local_m(args, n):
    requested = args.skv4_stage16_local_m or 160
    return min(requested, n // 4)


def initialize(ttnn, device, k, v, q, index, args):
    context, d = k.shape[1], k.shape[-1]
    group = args.n_heads // args.n_kv_heads
    slots = args.n_conc * args.n_kv_heads
    n = group * args.skv4_stage15_union_m

    stage1_state = stage1_live.prepare(ttnn, device, k, q, args, index=index)
    stage15_candidates = stage1_state["candidates"]
    candidates = stage15_candidates
    stage15_lookup = None
    stage15_cluster_lookup = stage1_state["cluster_ids"]

    encoded = encode_planes(_stage15_key_codec(k), args.n_conc, args.n_kv_heads)
    stage16_inputs = {
        "residual2": ttnn.from_torch(
            encoded["residual2"].reshape(1, 1, slots * context, d // 4),
            device=device,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        "residual4": ttnn.from_torch(
            encoded["residual4"].reshape(1, 1, slots * context, d // 2),
            device=device,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    }
    # Stage1.5 prefill duplicates upper bits into cluster blocks; do not
    # upload or retain a second token-major GDDR copy.
    stage16_inputs["prefix2"] = encoded["prefix2_token"]
    if args.skv4_debug_indices:
        # Debug-only host reference. Production keeps the stage-start
        # cluster gather in distributed L1.
        stage16_inputs["reference_prefix2_token"] = encoded["prefix2_token"]
    prepared_query = stage1_state["query_prepare"]
    stage16_inputs["q15"] = prepared_query["stage15"]
    del encoded

    # sparse_sdpa stores V in the leading columns and K in the trailing
    # columns.  Q=[0,q] therefore computes ordinary attention and returns V.
    kv_one = torch.cat((v, k), dim=-1)
    kv_host = (
        kv_one.unsqueeze(0)
        .expand(args.n_conc, -1, -1, -1)
        .reshape(slots, 1, context, 2 * d)
        .float()
        .contiguous()
    )
    kv8 = ttnn.from_torch(
        kv_host,
        device=device,
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # One compact L1 Q allocation serves every slot. The attention reader
    # uses global (slot,head) rows and builds its padded CB locally.
    q_sparse = [prepared_query["compact"]] * slots

    selector15 = fused_stage15.build_stage15(
        ttnn,
        device,
        stage16_inputs["q15"],
        stage16_inputs["prefix2"],
        stage15_lookup,
        stage15_cluster_lookup,
        slots=slots,
        context=context,
        n15=args.skv4_stage15_candidates,
        m15=args.skv4_stage15_m,
        union_m15=args.skv4_stage15_union_m,
        workers=args.skv4_stage15_workers,
        debug_snapshots=args.skv4_debug_indices,
        posting_lists=stage1_state["cells"],
    )
    lookup = selector15["ids"]
    # The common per-head-control ABI still reserves this accessor. Union
    # kernels never read it, so reuse an existing L1 tensor.
    stage16_inputs["scores2"] = lookup
    stage16_inputs["union_inputs"] = {
        "prefix2": selector15["prefix2"],
        "query": prepared_query["matrix"],
        "input_counts": selector15["union_counts"],
    }
    selector = fused_stage16.build_full_stage16(
        ttnn,
        device,
        # The inherited `sq` accessor is unused by union kernels. Reuse
        # the existing compact allocation rather than allocate dead Q.
        prepared_query["compact"],
        stage16_inputs["residual2"],
        stage16_inputs["residual4"],
        lookup,
        stage16_inputs["scores2"],
        q_sparse,
        kv8,
        slots=slots,
        context=context,
        n16=n,
        m16=args.skv4_stage16_m,
        workers=4,
        local_m=_stage16_local_m(args, n),
        group=group,
        topk=args.topk,
        debug_indices=args.skv4_debug_indices,
        profile_stop=10,
        sfpu_planes=8,
        per_head_input=False,
        union_inputs=stage16_inputs["union_inputs"],
        attention_zones=stage1_state["attention_zones"],
    )
    # Compile and initialize the complete direct-launch chain.
    _chain_mark("prepare A enqueue")
    selector15["run"]()
    _chain_mark("prepare A synchronize")
    ttnn.synchronize_device(device)
    stage1_live.verify_initial(ttnn, stage1_state, selector15)
    _chain_mark("prepare A complete; B enqueue")
    selector["run"]()
    _chain_mark("prepare B synchronize")
    ttnn.synchronize_device(device)
    _chain_mark("prepare B complete")
    return {
        "ttnn": ttnn,
        "stage1_state": stage1_state,
        "device": device,
        "kv8": kv8,
        "q_sparse": q_sparse,
        "candidates": candidates,
        "group": group,
        "slots": slots,
        "n_conc": args.n_conc,
        "n_kv_heads": args.n_kv_heads,
        "head_dim": d,
        "context": context,
        "topk": args.topk,
        "selector": selector,
        "selector15": selector15,
        "stage16_inputs": stage16_inputs,
    }


def bind_query(state, q, args):
    expected = (state["n_conc"], 1, args.n_heads, state["head_dim"])
    if tuple(q.shape) != expected:
        raise ValueError(
            f"unexpected SeoulKV4 query shape {tuple(q.shape)}, expected {expected}"
        )
    stage1_live.bind_query(state["ttnn"], state["stage1_state"], q)


def enqueue(state):
    state["stage1_state"]["run"]()
    _chain_mark("run A enqueue")
    state["selector15"]["run"]()
    _chain_mark("run B enqueue")
    state["selector"]["run"]()


def metadata(state):
    implementation = (
        "Stage 1.5 -> source-owned GQA union -> matrix Stage 1.6 -> "
        "union matrix Stage II -> sparse attention"
    )
    return {
        "implementation": (
            "Live Stage-I coarse -> runtime-dynamic merged Fine (cluster IDs) -> "
            "Stage1.5 CSR ingestion -> " + implementation
        ),
        "attention_zones": state["stage1_state"]["attention_zones"].metadata(),
        "attention_t2_payload_bytes": (
            state["stage1_state"]["attention_zones"].count()
            * state["n_conc"]
            * state["n_kv_heads"]
            * state["group"]
            * 2
            * state["head_dim"]
        ),
        "device": "Blackhole p150a",
        "stage1": stage1_live.metadata(state["stage1_state"]),
        "query_preparation_included": True,
        "trace_programs": list(PRODUCTION_PROGRAMS),
        "query_binding": {
            "host_uploads_per_invocation": 1,
            "host_query_bytes": 8192,
            "raw_query_tier": "T2/GDDR",
            "derived_query_tier": "T1/L1",
            "derived_query_bytes": 336128,
            "static_scale_bytes": 4 * 32768,
            "static_zero_bytes": query_prepare_abi.SCRATCH_BYTES,
            "attention_input": "compact 512-B L1 row/head; diagonal CB built locally",
        },
        "implemented_stages": [
            "stage1_coarse",
            "stage1_fine",
            "stage1.5",
            "stage1.6",
            "stage2",
            "sparse_attention",
        ],
        "stage15_input": "cluster IDs/counts from merged fine; Stage1.5 fetches spill2 cluster blocks then deduplicates tokens",
        "stage15_cluster_ingest": {
            "owner": "stage1.5",
            "spill": 2,
            "program_count": 0,  # embedded reader phase, not an extra launch
            "reader_groups": len(state["selector15"]["cluster_ingest"]["parts"]),
            "internal_l1_tensor_count": state["selector15"]["cluster_ingest"][
                "internal_l1_tensor_count"
            ],
            "status": state["selector15"].get("cluster_ingest_status"),
            "dedup": "local spill2 mate ownership with selected-cluster bitmap; first-occurrence order",
            "key_prefix_bytes": 64,
            "internal_l1_page_bytes": state["selector15"]["cluster_ingest"][
                "prefix_page_bytes"
            ],
            "scorer_logical_block_bytes": 256,
            "ingest_kind": state["selector15"]["cluster_ingest"]["kind"],
            "ingest_workers_per_cell": state["selector15"]["cluster_ingest"].get(
                "workers_per_cell"
            ),
            "ingest_handoff": "distributed in-place compaction; bulk L1 spans to disjoint4KiB page owners",
            "gddr_request": "one contiguous block per nonempty cluster, IDs plus duplicated prefixes",
            "status_columns": [
                "unique_keys",
                "error",
                "postings",
                "clusters",
                "prefix_bytes",
                "block_read_bytes",
                "block_requests",
            ],
            "cluster_layout": state["selector15"]["cluster_ingest"]["layouts"],
            "input_prefix_verified": state["selector15"]["cluster_ingest"].get(
                "input_prefix_verified", False
            ),
            "direct_count_verified": state["selector15"]["cluster_ingest"].get(
                "direct_count_verified", False
            ),
            "scorer_query_source": "L1",
            "fused_with_scorer": state["selector15"]["cluster_ingest_fused"],
        },
        "stage15_candidates": state["selector15"]["stage15_candidates"],
        "stage15_program_count": state["selector15"]["program_count"],
        "full_chain_program_count": (
            state.get("selector", {}).get("program_count", 1)
            + state["selector15"]["program_count"]
            + state["stage1_state"]["program_count"]
        ),
        "stage15_output_tensor_count": state["selector15"].get(
            "persistent_tensor_count"
        ),
        "stage15_debug_snapshot_tensor_count": state["selector15"].get(
            "snapshot_tensor_count"
        ),
        "stage15_scratch_bytes_per_core": state["selector15"].get(
            "scratch_bytes_per_core"
        ),
        "stage15_retained_input_bytes_per_core": state["selector15"].get(
            "retained_input_bytes_per_core"
        ),
        "stage15_per_head": state["selector15"]["stage15_per_head"],
        "stage15_union_per_head": state["selector15"]["stage15_union_per_head"],
        "stage15_candidate_shards": state["selector15"]["stage15_shards"],
        "stage15_cluster_fetch_bytes": state["selector15"][
            "stage15_cluster_fetch_bytes"
        ],
        "stage15_keys_per_packed_tile": state["selector15"][
            "stage15_keys_per_packed_tile"
        ],
        "stage15_compact_query_tiles": state["selector15"][
            "stage15_compact_query_tiles"
        ],
        "stage15_aggregate_centering": state["selector15"][
            "stage15_aggregate_centering"
        ],
        "stage15_worker_cores_per_slot": state["selector15"][
            "spatial_workers_per_slot"
        ],
        "stage15_handoff": "source-owned GQA union in L1; balanced live Stage-1.6 output",
        "stage15_selection": (
            "matrix scorers assemble W shard rows per head; local TopK-XL emits "
            "native score/position payloads without a scalar remap program. "
            "W-way payload merge emits top-2048/head; only the first "
            f"{state['selector15']['stage15_union_per_head']}/head enter the "
            "variable GQA union. Each head builds one bitmap; original scorers "
            "combine their source slices "
            "and compact cached 64-byte prefixes; source counts balance four "
            "output partitions without score routing or candidate loss"
        ),
        "stage15_scoring": state["selector15"]["stage15_scoring"],
        "stage15_record_layout": state["selector15"].get(
            "stage15_record_layout", "key-major"
        ),
        "stage15_conversion_epoch": state["selector15"].get("stage15_conversion_epoch"),
        "stage15_conversion_batch": state["selector15"].get(
            "stage15_conversion_batch", 1
        ),
        "stage15_early_compact": state["selector15"].get(
            "stage15_early_compact", False
        ),
        "stage15_head_ready": state["selector15"].get("stage15_head_ready", False),
        "stage15_shared_query": state["selector15"].get("stage15_shared_query", False),
        "stage15_gddr_policy": state["selector15"]["stage15_gddr_policy"],
        "stage15_gather_batch": state["selector15"].get("stage15_gather_batch", 0),
        "stage15_frontier_ready": state["selector15"].get(
            "stage15_frontier_ready", False
        ),
        "stage15_early_bitmap": state["selector15"].get("stage15_early_bitmap", False),
        "stage15_prefix_delivery": state["selector15"]["stage15_prefix_delivery"],
        "stage15_route_l1_bytes": state["selector15"]["stage15_route_l1_bytes"],
        "stage15_packed_cache_l1_bytes": state["selector15"][
            "stage15_packed_cache_l1_bytes"
        ],
        "stage15_candidate_cache_l1_bytes": state["selector15"][
            "stage15_candidate_cache_l1_bytes"
        ],
        "stage15_scorer_shards": state["selector15"]["stage15_scorer_shards"],
        "stage2_input": "GQA union partitioned by original key shard",
        "stage16_input": "source-owned L1 union IDs + packed upper bits; no inherited scores",
        "stage16_input_count": "live balanced Stage-1.5 union count",
        "stage16_candidates": state["selector"]["stage16_candidates"],
        "stage16_dynamic_input_count": state.get("selector", {}).get(
            "stage16_dynamic_input_count", False
        ),
        "stage16_schedule": state.get("selector", {}).get("stage16_schedule", {}),
        "refinement_fusion": {
            "program_count": 1,
            "frontier_publication_ranks": [256, 400],
            "cb_bytes_per_core": state["selector"]["fusion_cb_bytes"],
            "list_packet_bytes": [1088, 576],
            "attention_separate_program": False,
            "attention_clean_compute_state": True,
        },
        "attention_overlap": state["selector"].get("attention_overlap"),
        "union_boundary": state.get("selector", {}).get(
            "union_boundary", "union-complete"
        ),
        "stage15_deferred_unswizzle": False,
        "stage15_deferred_prefetch": False,
        "stage16_per_head": state["selector"]["stage16_per_head"],
        "stage16_local_per_shard": state["selector"]["stage16_local_per_shard"],
        "stage2_candidates": "dynamic union, at most 4*M16",
        "stage2_kernel": "fused-stage15",
        "program_count": state.get("selector", {}).get("program_count"),
        "stage16_scoring": state.get("selector", {}).get("stage16_scoring"),
        "stage16_gather": state.get("selector", {}).get("stage16_gather"),
        "stage16_merge": state.get("selector", {}).get("stage16_merge"),
        "stage16_frontier_packet_bytes": state.get("selector", {}).get(
            "stage16_frontier_packet_bytes"
        ),
        "stage16_frontier_packets_per_slot": state.get("selector", {}).get(
            "stage16_frontier_packets_per_slot"
        ),
        "stage16_profile5_payload": state.get("selector", {}).get(
            "stage16_profile5_payload"
        ),
        "stage16_score_exchange_bytes_per_slot": state.get("selector", {}).get(
            "stage16_score_exchange_bytes"
        ),
        "stage16_score_exchange_remote_bytes_per_slot": state.get("selector", {}).get(
            "stage16_score_exchange_remote_bytes"
        ),
        "dispatch_mode": state.get("selector", {}).get("dispatch_mode"),
        "inter_program_handoff": (
            "interleaved L1 metadata: fused refinement -> attention top-100 IDs; "
            "separate compact L1 middle-bit pages; upper bits reuse Stage-1.5 L1"
            if state.get("selector", {}).get("program_count", 1) >= 2
            else None
        ),
        "retained_refinement_l1_bytes": (
            state.get("selector", {}).get("handoff_l1_bytes", 0)
            + state.get("selector", {}).get("middle_cache_l1_bytes", 0)
        ),
        "inter_program_handoff_l1_bytes": 0,
        "inter_program_metadata_l1_bytes": 0,
        "intra_program_topk_handoff": "allocated512B/head L1 IDs; one ready notification/head to clean attention workers",
        "stage16_middle_cache_l1_bytes": state.get("selector", {}).get(
            "middle_cache_l1_bytes", 0
        ),
        "stage2_middle_source": state.get("selector", {}).get("stage2_middle_source"),
        "union_profile_phase": "full",
        "stage2_workers_per_slot": 4,
        "stage2_k_dtype": "retained upper bits + low nibble -> two exact BF16 matrix operands",
        "direct_code_scoring": False,
        "direct_tile_assembly": True,
        "decode_maps": None,
        "stage2_stream_topk": state.get("selector", {}).get("stage2_stream_topk", True),
        "stage2_local_selection": state.get("selector", {}).get(
            "stage2_local_selection"
        ),
        "stage2_decode_partition": (
            "four candidate-quarter workers per original shard; decode each union key once, "
            "score four queries per FPU matmul, transpose score quarters to four symmetric "
            "head-local TopK-XL workers"
        ),
        "stage2_score_dtype": "BF16",
        "topk": state["topk"],
        "attention_kv_dtype": "FP8_E4M3",
        "candidate_pool_scope": "Stage-1.5 GQA union, then Stage-1.6/Stage-II GQA union",
        "stage16_handoff": state.get("selector", {}).get("stage16_handoff"),
        "stage2_prefix_handoff_l1_bytes": state["slots"]
        * state["candidates"].shape[-1]
        * 64,
        "stage16_partition": "source-ordered deduplicated union balanced by live count across four workers",
        "stage16_local_selection": (
            "matrix score quarters return to one worker per GQA head; local TopK-XL; "
            "direct native sorted frontier -> head-root merge"
        ),
        "missing_stages": [],
        "stage2_intercore_noc": (
            f"Stage1.6 frontier per slot: {state['selector']['stage16_frontier_packets_per_slot']} x "
            f"{state['selector']['stage16_frontier_packet_bytes']} B direct native packets; "
            "union tags return to original shards; Stage II sends local top100 frontiers to head roots"
        ),
        "stage2_intermediate_scores": "L1 only",
        "topk_handoff": "persistent L1 final-ID mailbox inside fused refinement/attention",
    }
