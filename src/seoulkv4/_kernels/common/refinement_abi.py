"""Shared Stage1.6, Stage II and sparse-attention argument/CB/semaphore ABI.

These ordered programs intentionally reuse the same CB/semaphore slots and
persistent L1 handoffs. Keep one specification rather than stage-local copies.
"""

from enum import IntEnum, unique

from seoulkv4._kernels.common.argument_abi import CompileABI, Field, layout


@unique
class RefinementCB(IntEnum):
    """Fixed combined ABI; scratch slots are reused across split programs.

    Slots 4/5 also carry selector inputs/partials; 21/22 are sparse-attention
    gather request/ack queues after refinement has finished using them.
    """

    ATTENTION_QUERY_ROW = 0
    DECODED_KEYS = 1
    SPARSE_KV_RECORDS = 2
    SPARSE_KV_TILES = 3
    ATTENTION_NEG_INF = 4
    ATTENTION_MASK = 5
    ATTENTION_SCALE = 6
    ATTENTION_QK = 7
    ATTENTION_MAX_A = 8
    ATTENTION_MAX_B = 9
    ATTENTION_SUM_A = 10
    ATTENTION_SUM_B = 11
    ATTENTION_OUT_A = 12
    ATTENTION_OUT_B = 13
    ATTENTION_CORRECTION = 14
    ATTENTION_OUT_INTERMEDIATE = 15
    ATTENTION_OUT_ROW = 16
    LOCAL_FRONTIERS = 17
    LIVE_CONTROL = 18
    ATTENTION_COLUMN_IDENTITY = 19
    ATTENTION_RECIPROCAL = 20
    QUERY_COEFFICIENTS = 21
    TOPK_MERGE_SCRATCH = 22
    MATRIX_QUERY = 23
    DECODE_SCRATCH = 24
    PACKED_CANDIDATES = 25
    PACKED_MIDDLE = 26
    SCORE_SCRATCH = 27
    SCORE_TILES = 28
    EXCHANGE_TILES = 29
    SELECTOR_CHANNEL = 30
    SHARED_SCRATCH = 31


@unique
class RefinementSemaphore(IntEnum):
    """Combined Stage1.6/II ABI, reused by sequential split programs.

    PEER_EXCHANGE_READY orders Stage1.6 union lists; Stage II reuses it for merge
    acknowledgements. STAGE2_FRONTIERS_READY also joins diagnostic union outputs.
    ROW_VALID_SOURCE alone starts at one. Reserved slots retain their allocation.
    """

    FRONTIERS_READY = 0
    RESERVED_1 = 1
    RESERVED_2 = 2
    ROW_DESTINATIONS_READY = 3
    ROW_DATA_READY = 4
    ROW_VALID_SOURCE = 5
    RESERVED_6 = 6
    SELECTION_DESTINATIONS_READY = 7
    SELECTION_DATA_READY = 8
    SELECTION_DONE = 9
    STAGE2_IDS_READY = 10
    PEER_EXCHANGE_READY = 11
    STAGE2_FRONTIERS_READY = 12
    SCORES_READY = 13
    RESERVED_14 = 14
    PACKED_FRONTIERS_READY = 15


REFINEMENT_READER_BASE = layout(
    "sq",
    "residual2",
    "residual4",
    "candidates",
    "scores2",
    "attention_query",
    "kv",
    "slot",
    "context",
    "shard",
    "part",
    "root_x",
    "root_y",
    "row_start_x",
    "row_start_y",
    "row_end_x",
    "row_end_y",
)
REFINEMENT_WRITER_BASE = layout(
    "output",
    "kv",
    "slot",
    "context",
    "shard",
    "debug_indices",
    "part",
    Field("head_peers", 8),
    "root_x",
    "root_y",
    "row_start_x",
    "row_start_y",
    "row_end_x",
    "row_end_y",
    "debug_scores",
    "sq",
)
# Slot 1 is consumed by the upstream sparse-SDPA compute kernel (tok_count).
# Preserve slot 2 as part of the external prefix rather than repurposing it.
REFINEMENT_COMPUTE_RT = layout("shard", "tok_count", "upstream_reserved", "part")
REFINEMENT_EXTENSION_CT = layout("local16_pad", "m16_pad", "profile", "stage2_planes")
SPARSE_SDPA_CT = layout(
    "h",
    "dht",
    "v_dht",
    "skt",
    "scale",
    "cb_q_rm",
    "cb_q_in",
    "cb_k_rm",
    "cb_k_in",
    "cb_neginf",
    "cb_mask_part",
    "cb_scale",
    "cb_qk_im",
    "cb_max_a",
    "cb_max_b",
    "cb_sum_a",
    "cb_sum_b",
    "cb_out_a",
    "cb_out_b",
    "cb_corr",
    "cb_out_im",
    "cb_out_rm",
    "cb_ctrl",
    "cb_col_identity",
    "cb_recip_scratch",
    "scaled_kv",
    "cb_k_rope_rm",
    "cb_k_scale_bcast",
    "cb_k_latent_tile",
    "cb_k_rope_tile",
    "math_approx_mode",
    "query_subblock",
    "packed_row_bytes",
)


def refinement_abi(kind, *, union=True, dynamic=False, handoff=False, middle=False):
    if kind == "reader":
        rt = REFINEMENT_READER_BASE
        tensor_names = (
            "sq",
            "residual2",
            "residual4",
            "candidates",
            "scores2",
            "attention_query",
            "kv",
        )
        if union:
            rt = rt.then(layout("prefix2", "my_x", "my_y"))
            tensor_names += ("prefix2",)
        if dynamic:
            rt = rt.then(layout("input_counts"))
            tensor_names += ("input_counts",)
        if handoff:
            rt = rt.then(layout("handoff"))
            tensor_names += ("handoff",)
        if middle:
            rt = rt.then(layout("middle"))
            tensor_names += ("middle",)
        ct = CompileABI(
            layout("n16", "workers", "group", "m16", "d", "profile", "per_head_input"),
            tensor_names,
        )
    elif kind == "writer":
        rt = REFINEMENT_WRITER_BASE
        tensor_names = ("output", "kv", "debug_indices", "debug_scores", "sq")
        if union:
            rt = rt.then(layout("union_query"))
            tensor_names += ("union_query",)
        if handoff:
            rt = rt.then(layout("handoff"))
            tensor_names += ("handoff",)
        ct = CompileABI(
            layout(
                "n16",
                "workers",
                "group",
                "m16",
                "local_m",
                "topk",
                "d",
                "debug",
                "profile",
                "per_head_input",
            ),
            tensor_names,
        )
    else:
        raise ValueError(kind)
    return rt, ct


def refinement_defines(kind, *, role, dynamic):
    result = REFINEMENT_COMPUTE_RT.cpp_defines("SKV4_ARG_RF_COMPUTE_RT")
    for field in REFINEMENT_EXTENSION_CT.fields:
        offset = REFINEMENT_EXTENSION_CT.offset(field.name)
        expression = "sparse_sdpa::compute_ct_arg::END"
        if offset:
            expression += f" + {offset}"
        result += f"#define SKV4_ARG_RF_EXT_{field.name.upper()} {expression}\n"
    if kind != "compute":
        rt, ct = refinement_abi(
            kind,
            dynamic=dynamic,
            handoff=role is not None,
            middle=role in ("stage16", "stage2"),
        )
        result += rt.cpp_defines("SKV4_ARG_RF_RT") + ct.cpp_defines("SKV4_ARG_RF_CT")
        if kind == "writer":
            result += (
                f"#define SKV4_ARG_RF_RT_HEAD_PEERS_Y {rt.offset('head_peers') + 1}\n"
            )
    return result
