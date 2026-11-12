"""Explicit opt-in generation for packed Fine component clocks."""


def instrument(*, profile=None):
    from seoulkv4._kernels.stage1_fine.seoulkv4_fine_packed import sources

    reader, compute, writer = sources(profile=profile, wait_profile="")
    return reader, writer, compute
