"""Fused refinement and clean-core sparse attention with persistent L1.

The two refinement phases reuse the same workers and explicitly transition
operation state after one compute startup. Stage II starts consuming each
head's 256-entry list as soon as it is ready. Upper and middle bits stay in
allocated L1 pages; only the low nibble is fetched from GDDR in Stage II.
Attention uses16 disjoint workers, preserving clean TRISC state while hiding Q
preparation and starting each head when its final100 IDs are ready.
"""

from seoulkv4._kernels.stage16.fusion import combine
from seoulkv4._kernels.stage16.attention_overlap import combine as combine_attention
from seoulkv4._kernels.stage16.fusion_abi import FRONTIER_PACKET_BYTES
from seoulkv4.config import STAGE16_PER_HEAD, STAGE16_PER_SHARD


def build_chain(builder, ttnn, device, inputs, options):
    options = dict(options)
    attention_zones = options.pop("attention_zones", None)
    if (
        options["workers"] != 4
        or options["local_m"] != STAGE16_PER_SHARD
        or options["m16"] != STAGE16_PER_HEAD
        or options["profile_stop"] != 10
    ):
        raise ValueError(
            f"production union boundary requires W4/local{STAGE16_PER_SHARD}/"
            f"M{STAGE16_PER_HEAD}/full stop10"
        )
    local = options["n16"] // options["workers"]
    # Compact middle-bit pages survive program switches, with no root-sized
    # native-tile allocation. Upper bits already live in the Stage-1.5 cache.
    page_bytes = max(4096, ((local + 1) * 4 + 63) // 64 * 64)
    handoff = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, options["slots"] * options["workers"], page_bytes // 4)),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    middle_pages = ((local + 31) // 32) * 2
    middle = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, options["slots"] * options["workers"] * middle_pages, 256)),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    a = builder(
        ttnn,
        device,
        *inputs,
        **dict(
            options,
            profile_stop=7,
            union_profile_phase="full",
            _union_split_role="stage16",
            _union_handoff=handoff,
            _union_middle=middle,
        ),
    )
    b = builder(
        ttnn,
        device,
        *inputs,
        **dict(
            options,
            profile_stop=9,
            _union_split_role="stage2",
            _union_handoff=handoff,
            _union_middle=middle,
        ),
    )
    c = builder(
        ttnn,
        device,
        *inputs,
        **dict(
            options,
            profile_stop=10,
            union_profile_phase="full",
            _union_split_role="attention",
            _union_handoff=handoff,
        ),
    )
    # The lower-level builders supply the existing tensor/accessor ABI only.
    # Their split descriptors are never dispatched by the production chain.
    fused = combine(ttnn, a, b, n16=options["n16"], context=options["context"])

    combined = combine_attention(
        ttnn,
        device,
        fused,
        c,
        n16=options["n16"],
        context=options["context"],
        debug=options.get("debug_indices", False),
        zones=attention_zones,
    )
    programs = (combined,)
    return dict(
        c,
        # Attention owns the output tensors; fused refinement owns every
        # selector diagnostic and selector-specific metadata.
        debug_indices=b.get("debug_indices"),
        debug_union_counts=b.get("debug_union_counts"),
        stage2_stream_topk=b.get("stage2_stream_topk"),
        stage2_local_selection=b.get("stage2_local_selection"),
        stage2_scoring=b.get("stage2_scoring"),
        stage16_dynamic_input_count=a.get("stage16_dynamic_input_count", False),
        stage16_schedule=a.get("stage16_schedule", {}),
        frontier_publication_ranks=[STAGE16_PER_HEAD],
        list_packet_bytes=[FRONTIER_PACKET_BYTES],
        union_boundary="frontier-stream",
        program_count=len(programs),
        programs=programs,
        handoff_tensor=handoff,
        handoff_l1_bytes=options["slots"] * options["workers"] * page_bytes,
        middle_cache_tensor=middle,
        middle_cache_l1_bytes=options["slots"]
        * options["workers"]
        * middle_pages
        * 1024,
        stage2_middle_source="L1 retained Stage-1.6 bits",
        fusion_cb_specs=fused["fusion_cb_specs"],
        fusion_cb_bytes=fused["fusion_cb_bytes"],
        attention_overlap=combined["attention_overlap"],
        attention_worker_points=combined["attention_worker_points"],
        run=combined["run"],
        dispatch_mode="64 refinement +16 clean attention workers, head-local release",
    )
