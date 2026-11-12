"""Explicit opt-in source generation for the live Coarse program."""

import os

from seoulkv4._kernels.codegen import coarse


def sources(*, select=True, debug_heads=False, profile=None, wait_query=False):
    if profile is None:
        profile = os.environ.get("SKV4_COARSE_PROFILE") == "1"
    return (
        coarse.reader(
            select=select,
            debug_heads=debug_heads,
            profile=profile,
            wait_query=wait_query,
        ),
        coarse.compute(select=select, debug_heads=debug_heads, profile=profile),
        coarse.writer(select=select, debug_heads=debug_heads, profile=profile),
    )
