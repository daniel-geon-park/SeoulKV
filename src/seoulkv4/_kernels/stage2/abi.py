"""Standalone sharded Stage-II selector semaphore ABI."""

from enum import IntEnum, unique


@unique
class Stage2ShardedSemaphore(IntEnum):
    FRONTIERS_READY = 0
