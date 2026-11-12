"""Coarse argument, circular-buffer and semaphore ABI; shared by host/codegen."""

from enum import IntEnum, unique

from seoulkv4._kernels.common.argument_abi import CompileABI, Field, layout
from seoulkv4._kernels.common.selection_abi import (
    SELECT_READER_TAIL,
    SELECT_COMPUTE_RT,
    SCORE_COMPUTE_CT,
)

# Scheduling only: decoded BFP4 tiles per centroid chunk are always eight.
OPERAND_BATCH = 2


@unique
class CoarseCB(IntEnum):
    QUERY = 0
    DECODED_KEYS = 1
    PACKED_UPPER = 5
    METADATA = 6
    LOWER_SCORE_TILES = 16
    UPPER_SCORE_TILES = 17
    LOCAL_SCORE_TILES = 19
    FRONTIER_VALUES = 20
    FRONTIER_INDICES = 21
    SELECTOR_SCRATCH = 22
    PEER_FRONTIERS = 23
    MERGE_VALUES = 24
    MERGE_INDICES = 25
    WORKER_COUNTS = 28
    LOWER_INTERVALS = 29
    UPPER_INTERVALS = 30
    CONTROL = 31


@unique
class CoarseSemaphore(IntEnum):
    COHORT_SCORES_READY = 0
    FRONTIER_LEVEL_0 = 1
    FRONTIER_LEVEL_1 = 2
    FRONTIER_LEVEL_2 = 3
    FRONTIER_LEVEL_3 = 4
    FRONTIER_LEVEL_4 = 5
    FRONTIER_LEVEL_5 = 6
    FRONTIER_LEVEL_6 = 7
    FRONTIER_LEVEL_7 = 8
    THRESHOLD_HEADS_READY = 9
    THRESHOLDS_READY = 10
    WORKER_COUNTS_READY = 11
    PREFIX_OFFSETS_READY = 12
    MASK_REAR_READY = 13
    QUERY_READY = 14


COARSE_READER_BASE = layout(
    "upper", "scales", "residual", "qnorm", "query", "first_chunk"
)
COARSE_WRITER_TAIL = layout(
    "base",
    "cohort",
    "head",
    "worker",
    "workers",
    "root_x",
    "root_y",
    "parent_x",
    "parent_y",
    "parent_level",
    Field("cohort_peers", 8),
)


def coarse_abi(role, *, select=True, debug_heads=False, workers=0):
    if role == "reader":
        return (
            (
                COARSE_READER_BASE.then(SELECT_READER_TAIL).then(
                    layout("score_chunks", "live")
                )
                if select
                else COARSE_READER_BASE
            ),
            CompileABI(
                layout() if select else layout("chunks"),
                ("upper", "scales", "residual", "qnorm", "query"),
            ),
        )
    if role == "compute":
        return (
            SELECT_COMPUTE_RT.then(layout("n")) if select else layout(),
            CompileABI(layout("phase")) if select else SCORE_COMPUTE_CT,
        )
    if role != "writer":
        raise ValueError(role)
    if not select:
        return layout("output", "first_chunk"), CompileABI(
            layout("chunks"), ("output",)
        )
    return (
        layout("output", *(("heads",) if debug_heads else ()), "first_chunk")
        .then(COARSE_WRITER_TAIL)
        .then(layout("n", "total", "live", "p"))
        .then(layout(Field("worker_peers", workers * 2))),
        CompileABI(
            layout(),
            ("output", "heads") if debug_heads else ("output",),
        ),
    )


def coarse_defines(role, *, select=True, debug_heads=False):
    rt, ct = coarse_abi(role, select=select, debug_heads=debug_heads)
    # These conditional declarations preserve both selector and isolated-score
    # source contracts. No second manually numbered C++ table is maintained.
    definitions = (
        rt.cpp_defines("SKV4_ARG_RT")
        + ct.cpp_defines("SKV4_ARG_CT")
        + f"#define SKV4_COARSE_OPERAND_BATCH {OPERAND_BATCH}\n"
    )
    if role == "reader":
        definitions += "#define SKV4_ARG_ABI_COARSE_READER 1\n"
    if role == "writer" and select:
        definitions += "#define SKV4_ARG_ABI_COARSE_WRITER 1\n"
        definitions += (
            f"#define SKV4_ARG_RT_COHORT_PEERS_Y {rt.offset('cohort_peers') + 1}\n"
            f"#define SKV4_ARG_RT_WORKER_PEERS_Y {rt.offset('worker_peers') + 1}\n"
        )
    return definitions
