"""Production fused-refinement CB and semaphore resources."""

from enum import IntEnum, unique
from seoulkv4._kernels.common.argument_abi import layout


ATTENTION_DESTINATION = layout("attention_x", "attention_y")


@unique
class AttentionSemaphore(IntEnum):
    # Attention's original semaphore0 has no active consumer. It is disjoint
    # from refinement's ranges, but has the same program-wide semaphore base.
    ATTENTION_IDS_READY = 0


@unique
class FusionCB(IntEnum):
    FRONTIER_INPUT = 0
    DECODED_KEYS = 1
    STAGE16_SCORE_STAGING = 2
    TOPK_SECOND_INPUT = 4
    COPY_WITNESS = 5
    UNION_META = 14
    STAGE2_QUERY = 15
    STAGE2_FRONTIER = 16
    STAGE16_FRONTIER = 17
    LIVE_CONTROL = 18
    STAGE16_MERGE = 22
    MATRIX_QUERY = 23
    DECODE_SCRATCH = 24
    PACKED_CANDIDATES = 25
    PACKED_MIDDLE = 26
    SOURCE_IDS = 27
    SCORE_TILES = 28
    STAGE2_EXCHANGE = 29
    SELECTOR_INPUT = 30
    WRITER_SCRATCH = 31


@unique
class FusionSemaphore(IntEnum):
    MIDDLE_COUNT = 0
    MIDDLE_RELEASE = 1
    HEAD0_READY = 2
    HEAD1_READY = 3
    HEAD2_READY = 4
    HEAD3_READY = 5
    STAGE16_SCORES_READY = 6
    STAGE16_FRONTIERS_READY = 7
    STAGE2_SCORES_READY = 8
    STAGE2_FRONTIERS_READY = 9
    MERGE_ACK = 10
