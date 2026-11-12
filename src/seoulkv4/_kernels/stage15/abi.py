"""Stage1.5 and standalone/fused ingest argument, CB and semaphore ABI."""

from enum import IntEnum, unique

from seoulkv4._kernels.common.argument_abi import CompileABI, Field, layout


@unique
class Stage15CB(IntEnum):
    QUERY = 0
    DECODED_KEYS = 1
    QUERY_CONTROL = 3
    COMPUTE_CONTROL = 4
    WRITER_CONTROL = 6
    TOKEN_IDS = 7
    HEAD_SCORES = 8
    LOCAL_SCORE_TILES = 9
    FRONTIER_VALUES = 10
    FRONTIER_INDICES = 11
    PEER_VALUES = 12
    PEER_INDICES = 13
    MERGED_VALUES = 14
    MERGED_INDICES = 15
    SCORE_TILES = 16
    UNION_BITMAP = 17
    UNION_LAYOUT = 18
    UNION_CONTROL = 19
    UNION_RECORDS = 20
    DEBUG_SCRATCH = 21
    BITMAP_EXCHANGE = 22
    INGEST_SCRATCH = 23
    QUERY_READY = 24
    PACKED_PREFIX = 25
    COMPACT_TOPK_INPUT = 26
    TILE_LAYOUT_SCRATCH = 27


@unique
class StandaloneIngestCB(IntEnum):
    INGEST_WORKSPACE = 0


@unique
class Stage15Semaphore(IntEnum):
    SHARD_SCORES_READY = 0
    EPOCH_JOIN = 1
    EPOCH_RELEASE = 2
    BITMAP_INPUT_READY = 3
    BITMAP_PARTS_READY = 4
    FRONTIERS_READY = 5
    QUERY_DESTINATIONS_READY = 6
    QUERY_DATA_READY = 7
    INGEST_JOIN = 8
    INGEST_RELEASE = 9
    TREE_FRONTIERS_READY = 10
    COMPACTION_READY = 11


@unique
class StandaloneIngestSemaphore(IntEnum):
    INGEST_JOIN = 0
    INGEST_RELEASE = 1


STAGE15_HEADER = layout("worker", "slot", "context")
STAGE15_IO = layout(
    "query", "packed", "candidates", "clusters", "ids", "prefix", "counts"
)
STAGE15_DEBUG_IO = layout(
    "shard_scores",
    "shard_indices",
    "local_scores",
    "local_indices",
    "head_scores",
    "head_indices",
)
STAGE15_SCALARS = layout("n", "workers", "pad", "m", "debug", "chunks", "bitmap_domain")
STAGE15_PHASE_RT = layout("chunks", "tail", "shard", "live")
STAGE15_COMPUTE_RT = layout("worker").then(STAGE15_PHASE_RT)
STAGE15_COMPUTE_CT = CompileABI(layout("n", "phase", "k", "workers"))
INGEST_IO = layout("fine", "descriptors", "blocks", "ids_out", "prefix", "status")
INGEST_CT = CompileABI(
    layout(
        "nc",
        "context",
        "capacity",
        "fine_width",
        "slot",
        "minimum",
        "page_bytes",
        "pages",
        "block_bytes",
        "workers",
        "local",
        "diagnostics",
        "direct_cb",
        "posting_capacity",
    ),
    tuple(field.name for field in INGEST_IO.fields),
)


def stage15_abi(workers, *, debug=True):
    io = STAGE15_IO.then(STAGE15_DEBUG_IO) if debug else STAGE15_IO
    rt = STAGE15_HEADER.then(layout(Field("peers", workers * 8))).then(io)
    return rt, CompileABI(STAGE15_SCALARS, tuple(field.name for field in io.fields))


def ingest_runtime(workers):
    return INGEST_IO.then(layout("worker", Field("peers", workers * 2)))


def stage15_defines(role, *, ct_offset=0, rt_offset=0):
    if role == "compute":
        return (
            STAGE15_COMPUTE_RT.cpp_defines("SKV4_ARG_S15_RT")
            + STAGE15_PHASE_RT.cpp_defines("SKV4_ARG_S15_PHASE_RT")
            + STAGE15_COMPUTE_CT.cpp_defines("SKV4_ARG_S15_CT")
        )
    result = (
        STAGE15_HEADER.cpp_defines("SKV4_ARG_S15_RT")
        + STAGE15_IO.then(STAGE15_DEBUG_IO).cpp_defines("SKV4_ARG_S15_IO")
        + CompileABI(STAGE15_SCALARS).cpp_defines("SKV4_ARG_S15_CT")
    )
    if role == "reader":
        result += INGEST_CT.scalars.cpp_defines("SKV4_ARG_INGEST_CT", base=ct_offset)
        result += INGEST_IO.cpp_defines("SKV4_ARG_INGEST_RT", base=rt_offset)
        result += f"#define SKV4_ARG_INGEST_CT_ACCESSORS {ct_offset + INGEST_CT.scalars.words}\n"
    return result


def ingest_defines():
    rt = ingest_runtime(0)
    return (
        rt.cpp_defines("SKV4_ARG_INGEST_RT")
        + INGEST_CT.cpp_defines("SKV4_ARG_INGEST_CT")
        + f"#define SKV4_ARG_INGEST_RT_PEERS_Y {rt.offset('peers') + 1}\n"
    )
