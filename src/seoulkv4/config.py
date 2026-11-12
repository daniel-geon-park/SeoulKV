"""Configuration for the validated single-cache Blackhole profile."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SeoulKV4Config:
    stage15_capacity: int = 13056
    debug: bool = False

    def __post_init__(self):
        n = self.stage15_capacity
        if type(n) is not int or not 12288 <= n <= 24576 or n % (6 * 4 * 32):
            raise ValueError(
                "stage15_capacity must be a multiple of 768 in [12288, 24576]"
            )
        if type(self.debug) is not bool:
            raise TypeError("debug must be a bool")


@dataclass(frozen=True)
class _KernelOptions:
    """Private settings passed to existing stage builders, never argparse."""

    context_len: int
    skv4_stage15_candidates: int
    skv4_debug_indices: bool
    n_conc = seq_len = 1
    n_kv_heads = 4
    n_heads = 16
    topk = 100
    skv4_stage15_m = 2048
    skv4_stage15_union_m = 1600
    skv4_stage15_workers = 6
    skv4_stage16_m = 400
    skv4_stage16_local_m = 160
