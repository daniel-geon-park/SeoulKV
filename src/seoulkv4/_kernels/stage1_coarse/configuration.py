"""Multicast identical cell configuration without moving per-core CB storage."""

from seoulkv4._kernels.common.core_ranges import core_ranges


def consolidate(tt, states, groups):
    """Coarse-only: each cell has uniform, full-group CBs in the same order.

    A CB is merged only when its ENTIRE allocation prefix is identical.
    Matching just the current size can move its address after unequal earlier
    allocations, breaking peer transfers that rely on identical local pointers.
    """
    counts = {len(state["cb_specs"]) for state in states}
    if len(counts) != 1 or len(states) != len(groups):
        raise ValueError("Coarse cell configuration shape mismatch")
    prefixes = [() for _ in states]
    cbs = []
    for slot in range(counts.pop()):
        merged = {}
        for cell, (state, points) in enumerate(zip(states, groups)):
            spec = state["cb_specs"][slot]
            prefixes[cell] += (spec,)
            _, combined = merged.setdefault(prefixes[cell], (spec, []))
            combined.extend(points)
        for (index, dtype, page, count), points in merged.values():
            cbs.append(
                tt.CBDescriptor(
                    total_size=page * count,
                    core_ranges=core_ranges(tt, points),
                    format_descriptors=[
                        tt.CBFormatDescriptor(
                            buffer_index=index, data_format=dtype, page_size=page
                        )
                    ],
                )
            )
    merged_sems = {}
    for state, points in zip(states, groups):
        for sem in state["descriptor"].semaphores:
            merged_sems.setdefault((sem.id, sem.initial_value), []).extend(points)
    semaphores = [
        tt.SemaphoreDescriptor(
            id=identity,
            initial_value=value,
            core_ranges=core_ranges(tt, points),
        )
        for (identity, value), points in merged_sems.items()
    ]
    return cbs, semaphores
