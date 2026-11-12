"""Explicit fetch clocks for the isolated NCRISC Fine input control."""

import os


def reader_source(
    *, external_tags=True, dynamic=True, profile=None, fetch_profile=None
):
    from seoulkv4._kernels.stage1_fine.seoulkv4_fine_packed import sources

    if profile is None:
        profile = os.environ.get("SKV4_FINE_PACKED_PROFILE") == "1"
    if fetch_profile is None:
        fetch_profile = profile and os.environ.get("SKV4_FINE_FETCH_PROFILE") == "1"
    return sources(
        select=True,
        shared_upper=True,
        external_tags=external_tags,
        dynamic=dynamic,
        dual_risc=False,
        profile=profile,
        wait_profile="",
        fetch_profile=fetch_profile,
    )[0]
