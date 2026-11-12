"""Clean-core refinement/attention compositor; explicit sources, no rewriting."""

import os

from seoulkv4._kernels.common.core_ranges import core_ranges
from seoulkv4._kernels.common.refinement_abi import refinement_abi
from seoulkv4._kernels.codegen import (
    attention as attention_codegen,
    refinement_fusion,
)
from seoulkv4._kernels.common.attention_zones import AttentionZones
from seoulkv4._kernels.stage16 import attention_abi
from seoulkv4._kernels.stage16.fusion_abi import (
    ATTENTION_DESTINATION,
    AttentionSemaphore,
)


def attention_workers(device, groups):
    used = {tuple(point) for group in groups for point in group}
    if len(used) != 16 * len(groups):
        raise ValueError("refinement must retain16 distinct workers/cell")
    grid = device.compute_with_storage_grid_size()
    if any(not (0 <= x < grid.x and 0 <= y < grid.y) for x, y in used):
        raise ValueError("refinement core outside usable runtime grid")
    free = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in used]
    need = 4 * len(groups)
    if len(free) < need:
        raise ValueError(
            f"need{need} additional clean attention cores, have{len(free)}"
        )
    return free[:need]


def combine(tt, device, fused, attention, *, n16, context, debug=False, zones=None):
    zones = AttentionZones(context, 0, 0) if zones is None else zones
    if zones.context != context:
        raise ValueError("attention zone context mismatch")
    zones.count()
    groups = fused["worker_groups"]
    points = attention_workers(device, groups)
    ranges = core_ranges(tt, points)
    old_roots = [group[head * 4] for group in groups for head in range(4)]
    physical = [device.worker_core_from_logical_core(tt.CoreCoord(*p)) for p in points]
    profile = os.environ.get("SKV4_NATIVE_MARKERS") == "1" and os.environ.get(
        "SKV4_NATIVE_PROFILE_ROLE", ""
    ) in ("", "attention", "stage16")
    rt_layout, _ = refinement_abi("writer", union=True, handoff=True)
    writer_args = tt.RuntimeArgs()
    for slot, group in enumerate(groups):
        for index, (x, y) in enumerate(group):
            dest = physical[slot * 4 + index // 4]
            prefix = list(fused["descriptor"].kernels[2].runtime_args[x][y])
            if len(prefix) != rt_layout.words:
                raise ValueError("unexpected refinement writer ABI")
            writer_args[x][y] = prefix + ATTENTION_DESTINATION.pack(
                attention_x=dest.x, attention_y=dest.y
            )
    kernels = []
    for index, donor in enumerate(fused["descriptor"].kernels):
        kernels.append(
            tt.KernelDescriptor(
                kernel_source=(
                    refinement_fusion.writer(
                        local=n16 // 4,
                        id_bits=max(1, (context - 1).bit_length()),
                        attention=True,
                    )
                    if index == 2
                    else donor.kernel_source
                ),
                source_type=tt.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=donor.core_ranges,
                compile_time_args=list(donor.compile_time_args),
                runtime_args=writer_args if index == 2 else donor.runtime_args,
                config=donor.config,
            )
        )
    for index, donor in enumerate(attention["descriptor"].kernels):
        args = tt.RuntimeArgs()
        source = donor.kernel_source
        compile_args = list(donor.compile_time_args)
        bindings = attention["attention_bindings"]
        if index in (0, 2):
            source = (
                attention_codegen.reader if index == 0 else attention_codegen.writer
            )(profile=profile)
            if index == 0:
                compile_args = attention_abi.READER_CT.pack(
                    tt,
                    tensors={
                        name: bindings[name] for name in ("query", "kv", "handoff")
                    },
                    d=256,
                    handoff_bytes=bindings["handoff_bytes"],
                )
            else:
                compile_args = attention_abi.WRITER_CT.pack(
                    tt,
                    tensors=dict(output=attention["outputs"][0], kv=bindings["kv"]),
                    d=256,
                )
        for head_index, (old, new) in enumerate(zip(old_roots, points)):
            slot, head = divmod(head_index, 4)
            if index == 0:
                values = attention_abi.READER_RT.pack(
                    query=bindings["query"].buffer_address(),
                    kv=bindings["kv"].buffer_address(),
                    handoff=bindings["handoff"].buffer_address(),
                    slot=slot,
                    head=head,
                    context=context,
                    sink=zones.sink,
                    local=zones.local,
                )
            elif index == 2:
                values = attention_abi.WRITER_RT.pack(
                    output=attention["outputs"][slot].buffer_address(),
                    kv=bindings["kv"].buffer_address(),
                    slot=slot,
                    head=head,
                    context=context,
                )
            else:
                values = list(donor.runtime_args[old[0]][old[1]])
            args[new[0]][new[1]] = values
        kernels.append(
            tt.KernelDescriptor(
                kernel_source=source,
                source_type=tt.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=compile_args,
                runtime_args=args,
                config=donor.config,
            )
        )
    # Reuse opaque format descriptors: v0.77's data_format getter is broken.
    cbs = [
        *fused["descriptor"].cbs,
        *[
            tt.CBDescriptor(
                total_size=cb.total_size,
                core_ranges=ranges,
                format_descriptors=list(cb.format_descriptors),
            )
            for cb in attention["descriptor"].cbs
        ],
    ]
    sems = [
        *fused["descriptor"].semaphores,
        tt.SemaphoreDescriptor(
            id=int(AttentionSemaphore.ATTENTION_IDS_READY),
            core_ranges=ranges,
            initial_value=0,
        ),
    ]
    descriptor = tt.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)
    io, seen = [], set()
    for tensor in (*fused["io"], *attention["io"]):
        if id(tensor) not in seen:
            io.append(tensor)
            seen.add(id(tensor))

    def run():
        tt.generic_op(io, descriptor)
        return attention["outputs"]

    return dict(
        attention,
        debug_indices=fused.get("debug_indices"),
        debug_union_counts=fused.get("debug_union_counts"),
        fusion_cb_bytes=fused["fusion_cb_bytes"],
        fusion_cb_specs=fused["fusion_cb_specs"],
        descriptor=descriptor,
        io=io,
        run=run,
        attention_worker_points=points,
        refinement_worker_groups=groups,
        attention_overlap=dict(
            refinement_workers=sum(map(len, groups)),
            attention_workers=len(points),
            attention_cb_bytes_per_core=sum(
                cb.total_size for cb in attention["descriptor"].cbs
            ),
            attention_logical=points,
            # Serving API returns virtual NoC coordinates, not CSV physical IDs.
            attention_noc_virtual=[[p.x, p.y] for p in physical],
            final_ids_notifications_per_invocation=len(points),
            query_prepared_before_ids=True,
            dense_attention_before_ids=True,
            zones=zones.metadata(),
        ),
    )
