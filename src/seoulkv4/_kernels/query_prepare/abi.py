"""QueryPrepare owns its argument words and circular-buffer IDs."""

from enum import IntEnum, unique

from seoulkv4._kernels.common.argument_abi import CompileABI, Field, layout


@unique
class QueryPrepareCB(IntEnum):
    RAW_QUERY = 0
    QUERY_TILES = 2
    SCALE_TILES = 3
    NORM_TILE = 4
    # FP32 transport: normal products rounded to BF16, exceptions unchanged.
    # Writer-only data, not a generic unrounded FP32 arithmetic operand.
    PRODUCT_TILES = 5
    OUTPUT_SCRATCH = 6


@unique
class QueryPrepareSemaphore(IntEnum):
    OPERANDS_READY = 0
    RAW_READY = 1
    EARLY_WRITER_DONE = 2


WRITERS_PER_GROUP = 2
PACKET_BYTES = 16384 // WRITERS_PER_GROUP
PACKET_ELEMENTS = PACKET_BYTES // 2
SCRATCH_BYTES = 5 * PACKET_BYTES + 64
MAX_COARSE_CONSUMERS = 28

READER_RT = layout("raw_query", "scales", "zeros", "group")
READER_CT = CompileABI(layout(), ("raw_query", "scales", "zeros"))
COMPUTE_CT = CompileABI(layout("lambda_bits"), ())
WRITER_RT = layout(
    "coarse",
    "fine",
    "norm",
    "stage15",
    "matrix",
    "compact",
    "zeros",
    "group",
    "worker",
    Field("peers", 2 * (WRITERS_PER_GROUP - 1)),
    "owner_x",
    "owner_y",
    "consumer_count",
    Field("consumers", 2 * MAX_COARSE_CONSUMERS),
)
WRITER_CT = CompileABI(
    layout("consumer_ready_semaphore"),
    ("coarse", "fine", "norm", "stage15", "matrix", "compact", "zeros"),
)
