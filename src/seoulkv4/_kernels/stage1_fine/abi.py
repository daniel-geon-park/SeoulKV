"""Fine argument, circular-buffer and semaphore ABI; shared by host/codegen."""

from enum import IntEnum, unique

from seoulkv4._kernels.common.argument_abi import CompileABI, Field, layout
from seoulkv4._kernels.common.selection_abi import (
    SELECT_READER_TAIL,
    SELECT_COMPUTE_RT,
    SCORE_COMPUTE_CT,
)


@unique
class FineCB(IntEnum):
    QUERY = 0
    DECODED_KEYS = 1
    SELECTED_TAGS = 2
    PACKED_KEY_PLANES = 5
    SCALES = 6
    LIVE_CONTROL = 7
    UPPER_CACHE = 8
    UPPER_JOB = 10
    UPPER_DONE = 11
    SCORE_TILES = 16
    LOCAL_SCORE_TILES = 19
    FRONTIER_VALUES = 20
    FRONTIER_INDICES = 21
    SELECTOR_SCRATCH = 22
    PEER_FRONTIERS = 23
    MERGE_VALUES = 24
    MERGE_INDICES = 25
    UNION_SCRATCH = 27


@unique
class FineSemaphore(IntEnum):
    COHORT_SCORES_READY = 0
    FRONTIER_LEVEL_0 = 1
    FRONTIER_LEVEL_1 = 2
    FRONTIER_LEVEL_2 = 3
    UNION_BITMAPS_READY = 4
    UNION_COUNTS_READY = 5
    UNION_OFFSETS_READY = 6


FINE_READER_BASE = layout("upper", "lower", "tables", "ids", "query", "first")
FINE_WRITER_TAIL = layout(
    "first",
    "base",
    "cohort",
    "head",
    "root_x",
    "root_y",
    "parent_x",
    "parent_y",
    "parent_level",
    Field("cohort_peers", 8),
    "tag_header",
)
WORKER_TAIL = layout("worker", "workers")


def fine_abi(role, *, select=True, dynamic=False, dual_risc=False):
    if role == "reader":
        rt = FINE_READER_BASE
        if select:
            rt = rt.then(SELECT_READER_TAIL)
        if dynamic:
            rt = rt.then(WORKER_TAIL)
        ct = CompileABI(
            (
                layout("capacity", "score", "chunks")
                if select
                else layout("capacity", "score")
            ),
            ("upper", "lower", "tables", "ids", "query"),
        )
    elif role == "compute":
        rt, ct = SELECT_COMPUTE_RT if select else layout(), SCORE_COMPUTE_CT
    elif role == "writer":
        if not select:
            return layout("output", "first"), CompileABI(
                layout("n", "score"), ("output",)
            )
        rt = layout("scores", "tags", "output", "heads").then(FINE_WRITER_TAIL)
        if dynamic:
            rt = rt.then(WORKER_TAIL)
        if dual_risc:
            rt = rt.then(layout("upper"))
        ct = CompileABI(
            layout("n", "score", "total", "live", "p", "out", "max_local"),
            ("scores", "tags", "output", "heads") + (("upper",) if dual_risc else ()),
        )
    else:
        raise ValueError(role)
    return rt, ct


def fine_defines(role, *, select, dynamic, dual_risc):
    rt, ct = fine_abi(role, select=select, dynamic=dynamic, dual_risc=dual_risc)
    result = rt.cpp_defines("SKV4_ARG_RT") + ct.cpp_defines("SKV4_ARG_CT")
    if role == "writer" and select:
        result += (
            f"#define SKV4_ARG_RT_COHORT_PEERS_Y {rt.offset('cohort_peers') + 1}\n"
        )
    return result
