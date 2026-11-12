"""Explicit Fine issue/completion clock source generation."""


def instrument(*, mode=None, profile=None, select=False):
    from seoulkv4._kernels.stage1_fine.seoulkv4_fine_packed import sources

    return sources(select=select, profile=profile, wait_profile=mode)[1]
