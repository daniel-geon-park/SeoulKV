"""Dense 2-bit centroid decode and four-head coarse scoring on Blackhole.

This is the first Stage-I coarse checkpoint.  It deliberately exports scores
to DRAM for validation; the production survivor path will keep them in L1.
Each 2,048-byte input page contains a losslessly prepacked 32-centroid block.
The SFPU synthesizes native BFP4 levels ``{0,1,2,3}``, the FPU computes four
BF16-input/FP32-accumulated dot products, and SFPU recenters before row scaling.
"""

from __future__ import annotations

from seoulkv4._kernels.stage1_coarse.abi import CoarseCB as CB
from seoulkv4._kernels.stage1_coarse.abi import OPERAND_BATCH
from seoulkv4._kernels.common.query_metadata import SUM_WORD_OFFSET

import torch

from seoulkv4._kernels.stage1_coarse.coarse_native_codec import (
    prepack_upper,
)
from seoulkv4._kernels.stage15.stage15_sfpu_native import coordinate_order
from seoulkv4._kernels.common.core_ranges import core_ranges as _ranges
from seoulkv4._kernels.stage1_coarse.configuration import consolidate


READER = (
    f"#define SKV4_COARSE_OPERAND_BATCH {OPERAND_BATCH}\n"
    + r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CHUNKS = get_compile_time_arg_val(0);
    constexpr auto ua = TensorAccessorArgs<1>();
    constexpr auto sa = TensorAccessorArgs<ua.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto na = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto qa = TensorAccessorArgs<na.next_compile_time_args_offset()>();
    const auto upper = TensorAccessor(ua, get_arg_val<uint32_t>(0), 4096);
    const auto scales = TensorAccessor(sa, get_arg_val<uint32_t>(1), 64);
    const auto residual = TensorAccessor(ra, get_arg_val<uint32_t>(2), 64);
    const auto qnorm = TensorAccessor(na, get_arg_val<uint32_t>(3), 64);
    const auto query = TensorAccessor(qa, get_arg_val<uint32_t>(4), 2048);
    const uint32_t first_chunk = get_arg_val<uint32_t>(5);

    // Raw UInt32 row PACK overwrites only native mantissas. All batched BFP4
    // ring headers are constant, including across trace replay.
    auto operand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
    for (uint32_t tile = 0; tile < 8 * SKV4_COARSE_OPERAND_BATCH; ++tile)
        for (uint32_t word = 0; word < 16; ++word)
            operand[tile * 144 + word] = 0x80808080u;
    __sync_synchronize();

    // Scale-sorted physical chunks share one FP32 scale.  One aligned page per
    // chunk avoids per-column scale selection in MATH.  Residual norms remain
    // per centroid so the certified conservative interval is unchanged.
    cb_reserve_back(6, 1);
    const uint32_t table = get_write_ptr(6);
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
        noc_async_read(get_noc_addr(first_chunk + chunk, scales),
                       table + chunk * 64, 64);
        noc_async_read(get_noc_addr(first_chunk + chunk, residual),
                       table + CHUNKS * 64 + chunk * 64, 64);
    }
    noc_async_read(get_noc_addr(0, qnorm), table + CHUNKS * 128, 64);

    cb_reserve_back(0, 8);
    for (uint32_t tile = 0; tile < 8; ++tile)
        noc_async_read(get_noc_addr(tile, query),
                       get_write_ptr(0) + tile * 2048, 2048);
    noc_async_read_barrier();
    cb_push_back(6, 1);
    cb_push_back(0, 8);

    // A physical 4-KiB T1 page contains 64 records.  Each scorer iteration
    // reads one 2-KiB half in one bulk NoC transaction.
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
        cb_reserve_back(5, 1);
        const uint32_t global = first_chunk + chunk;
        noc_async_read(get_noc_addr(global >> 1, upper)
                           + (global & 1u) * 2048,
                       get_write_ptr(5), 2048);
        noc_async_read_barrier();
        cb_push_back(5, 1);
    }
}
"""
)


WRITER = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CHUNKS = get_compile_time_arg_val(0);
    constexpr auto oa = TensorAccessorArgs<1>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    const uint32_t first_chunk = get_arg_val<uint32_t>(1);
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
        cb_wait_front(16, 1);
        cb_wait_front(17, 1);
        noc_async_write(get_read_ptr(16),
                        get_noc_addr((first_chunk + chunk) * 2, output), 2048);
        noc_async_write(get_read_ptr(17),
                        get_noc_addr((first_chunk + chunk) * 2 + 1, output), 2048);
        noc_async_write_barrier();
        cb_pop_front(16, 1);
        cb_pop_front(17, 1);
    }
}
"""


def compute_source():
    """Emit the native Coarse interval scorer without adapting another kernel."""
    from seoulkv4._kernels.codegen.coarse_compute import COMPUTE

    return COMPUTE


def _upload(tt, device, value, dtype, *, dram=False, tiled=False):
    return tt.from_torch(
        value,
        dtype=dtype,
        device=device,
        layout=tt.TILE_LAYOUT if tiled else tt.ROW_MAJOR_LAYOUT,
        memory_config=tt.DRAM_MEMORY_CONFIG if dram else tt.L1_MEMORY_CONFIG,
    )


def _launch_key(source, ct, config_type, config_options):
    """All compile/config behavior, excluding core-specific runtime addresses."""
    return (source, tuple(ct), config_type, tuple(sorted(config_options.items())))


def _merge_launch_groups(tt, cell_groups):
    """Share one kernel handle across cells with identical compilation inputs."""
    merged = {}
    for groups in cell_groups:
        for group in groups:
            target = merged.setdefault(
                group["key"],
                dict(
                    source=group["source"],
                    ct=group["ct"],
                    config=group["config"],
                    points=[],
                    rt=tt.RuntimeArgs(),
                ),
            )
            for x, y in group["points"]:
                if (x, y) in target["points"]:
                    raise ValueError("duplicate core in a Coarse kernel launch")
                target["points"].append((x, y))
                target["rt"][x][y] = group["rt"][x][y]
    return list(merged.values())


def _descriptors(tt, groups):
    return [
        tt.KernelDescriptor(
            kernel_source=group["source"],
            source_type=tt.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_ranges(tt, group["points"]),
            compile_time_args=group["ct"],
            runtime_args=group["rt"],
            config=group["config"],
        )
        for group in groups
    ]


def build_cell(
    tt,
    device,
    cell,
    points,
    *,
    select=False,
    debug_heads=False,
    external_query=None,
    external_norm=None,
    wait_query=False,
):
    """Build one dense cell scorer over a fixed, disjoint worker group."""
    if debug_heads and not select:
        raise ValueError("debug head export requires the Coarse selector")
    ncentroids = int(cell["upper"].shape[0])
    padded = (ncentroids + 63) // 64 * 64
    chunks = padded // 32
    if not points or len(points) != len(set(points)) or len(points) > chunks:
        raise ValueError("coarse scoring needs distinct workers, at most one per chunk")

    upper = torch.zeros((padded, 64), dtype=torch.uint8)
    upper[:ncentroids] = cell["upper"]
    upper = _upload(tt, device, prepack_upper(upper).reshape(-1, 4096), tt.uint8)

    scales_per_centroid = torch.zeros(padded, dtype=torch.float32)
    scales_per_centroid[:ncentroids] = cell["scale"].float() / 4
    if not torch.equal(
        scales_per_centroid.reshape(chunks, 32),
        scales_per_centroid.reshape(chunks, 32)[:, :1].expand(-1, 32),
    ):
        raise ValueError("Stage-I coarse requires one shared scale per 32 centroids")
    scales = torch.zeros((chunks, 16), dtype=torch.float32)
    scales[:, 0] = scales_per_centroid.reshape(chunks, 32)[:, 0]
    scales = _upload(
        tt,
        device,
        scales.contiguous().view(torch.uint8).reshape(-1, 64),
        tt.uint8,
    )

    residual = torch.zeros(padded, dtype=torch.bfloat16)
    residual[:ncentroids] = cell["coarse_residual_norm"].bfloat16()
    residual = _upload(
        tt,
        device,
        residual.contiguous().view(torch.uint8).reshape(-1, 64),
        tt.uint8,
    )
    qnorm = external_norm
    if qnorm is None:
        value = torch.zeros(16, dtype=torch.float32)
        value[:4] = cell["q"].float().norm(dim=-1) * float(cell["lam"])
        value[SUM_WORD_OFFSET : SUM_WORD_OFFSET + 4] = (
            cell["q"].bfloat16().float().sum(dim=-1)
        )
        qnorm = _upload(
            tt, device, value.contiguous().view(torch.uint8).reshape(1, 64), tt.uint8
        )
    query = external_query
    if query is None:
        value = torch.zeros((1, 1, 32, 256), dtype=torch.bfloat16)
        value[0, 0, :4] = cell["q"].bfloat16()[:, coordinate_order()]
        query = _upload(tt, device, value, tt.bfloat16, tiled=True)

    # The interval tensor belongs only to the isolated scoring checkpoint.
    # Production selection consumes CB16/17 in the selector writer and must not
    # allocate or retain an unused GDDR score tensor.
    output = None
    if not select:
        output = tt.allocate_tensor_on_device(
            tt.Shape((chunks, 2, 32, 32)),
            tt.bfloat16,
            tt.TILE_LAYOUT,
            device,
            tt.DRAM_MEMORY_CONFIG,
        )

    shards = []
    for worker, point in enumerate(points):
        first = chunks * worker // len(points)
        end = chunks * (worker + 1) // len(points)
        shards.append((point, first, end - first))
    max_chunks = max(span for _, _, span in shards)
    cores = _ranges(tt, points)
    cb_specs = [
        (CB.QUERY, tt.bfloat16, 2048, 8),
        (CB.DECODED_KEYS, tt.bfloat4_b, 576, 8 * OPERAND_BATCH),
        (CB.PACKED_UPPER, tt.uint32, 2048, 2),
        (CB.METADATA, tt.uint8, max_chunks * 128 + 64, 1),
        (CB.LOWER_SCORE_TILES, tt.bfloat16, 2048, 2),
        (CB.UPPER_SCORE_TILES, tt.bfloat16, 2048, 2),
    ]
    cbs = [
        tt.CBDescriptor(
            total_size=page * count,
            core_ranges=cores,
            format_descriptors=[
                tt.CBFormatDescriptor(
                    buffer_index=index, data_format=dtype, page_size=page
                )
            ],
        )
        for index, dtype, page, count in cb_specs
    ]

    groups = {}

    def add_kernel(source, ct, config_type, config_options, point, runtime):
        key = _launch_key(source, ct, config_type, config_options)
        group = groups.setdefault(
            key,
            dict(
                key=key,
                source=source,
                ct=ct,
                config=config_type(**config_options),
                points=[],
                rt=tt.RuntimeArgs(),
            ),
        )
        group["points"].append(point)
        group["rt"][point[0]][point[1]] = runtime

    compute = compute_source()
    selection = None
    semaphores = []
    reader_text = READER
    writer_text = WRITER
    if select:
        from seoulkv4._kernels.stage1_coarse import seoulkv4_coarse_select

        selection = seoulkv4_coarse_select.prepare(
            tt,
            device,
            cell,
            padded,
            points,
            shards,
            debug_heads=debug_heads,
        )
        cbs.extend(selection["cbs"])
        cb_specs.extend(selection["cb_specs"])
        semaphores = selection["semaphores"]
    from seoulkv4._kernels.stage1_coarse.seoulkv4_coarse_profile import (
        sources as coarse_sources,
    )

    reader_text, compute, writer_text = coarse_sources(
        select=select, debug_heads=debug_heads, wait_query=wait_query
    )
    from seoulkv4._kernels.stage1_coarse.abi import coarse_abi

    reader_rt, reader_ct = coarse_abi("reader", select=select)
    compute_rt, compute_ct = coarse_abi("compute", select=select)
    writer_rt, writer_ct = coarse_abi(
        "writer", select=select, debug_heads=debug_heads, workers=len(points)
    )
    reader_tensors = dict(
        upper=upper, scales=scales, residual=residual, qnorm=qnorm, query=query
    )
    writer_tensors = dict(output=selection["output"] if select else output)
    if debug_heads:
        writer_tensors["heads"] = selection["heads"]
    for index, (point, first, local_chunks) in enumerate(shards):
        select_args = selection["args"](index) if selection is not None else None
        writer_fields = dict(
            output=writer_tensors["output"].buffer_address(), first_chunk=first
        )
        if select:
            writer_fields.update(select_args["writer"])
            writer_fields.update(
                n=local_chunks * 32, total=padded, live=ncentroids, p=int(cell["p"])
            )
            if debug_heads:
                writer_fields["heads"] = selection["heads"].buffer_address()
        add_kernel(
            reader_text,
            reader_ct.pack(
                tt,
                tensors=reader_tensors,
                **({} if select else dict(chunks=local_chunks)),
            ),
            tt.ReaderConfigDescriptor,
            {},
            point,
            reader_rt.pack(
                **{
                    name: tensor.buffer_address()
                    for name, tensor in reader_tensors.items()
                },
                first_chunk=first,
                **(select_args["reader"] if select else {}),
                **(dict(score_chunks=local_chunks, live=ncentroids) if select else {}),
            ),
        )
        add_kernel(
            compute,
            compute_ct.pack(
                tt, tensors={}, phase=2, **({} if select else dict(n=local_chunks * 32))
            ),
            tt.ComputeConfigDescriptor,
            dict(
                math_fidelity=tt.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=True,
            ),
            point,
            compute_rt.pack(
                **(dict(select_args["compute"], n=local_chunks * 32) if select else {})
            ),
        )
        add_kernel(
            writer_text,
            writer_ct.pack(
                tt,
                tensors=writer_tensors,
                **({} if select else dict(chunks=local_chunks)),
            ),
            tt.WriterConfigDescriptor,
            {},
            point,
            writer_rt.pack(**writer_fields),
        )

    kernels = _descriptors(tt, groups.values())
    descriptor = tt.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    io = [upper, scales, residual, qnorm, query]
    if output is not None:
        io.append(output)
    if selection is not None:
        io.extend(selection["io"])

    def run():
        tt.generic_op(io, descriptor)

    return dict(
        run=run,
        descriptor=descriptor,
        cb_specs=cb_specs,
        io=io,
        output=output if selection is None else selection["output"],
        intervals=output,
        heads=None if selection is None else selection["heads"],
        selection=selection,
        ncentroids=ncentroids,
        padded=padded,
        chunks=chunks,
        workers=len(points),
        shards=shards,
        launch_groups=list(groups.values()),
        upper=upper,
        query=query,
        qnorm=qnorm,
    )


def build_all(
    tt,
    device,
    cells,
    groups,
    *,
    select=False,
    debug_heads=False,
    external_queries=None,
    external_norms=None,
    wait_query=False,
):
    """Build four disjoint cell scorers as one TT program."""
    if len(cells) != len(groups):
        raise ValueError("one worker group per cell required")
    for values in (external_queries, external_norms):
        if values is not None and len(values) != len(cells):
            raise ValueError("one external query/norm tensor per cell required")
    points = [point for group in groups for point in group]
    if len(points) != len(set(points)):
        raise ValueError("coarse cell worker groups must be disjoint")
    states = [
        build_cell(
            tt,
            device,
            cell,
            group,
            select=select,
            debug_heads=debug_heads,
            external_query=None
            if external_queries is None
            else external_queries[index],
            external_norm=None if external_norms is None else external_norms[index],
            wait_query=wait_query,
        )
        for index, (cell, group) in enumerate(zip(cells, groups))
    ]
    launches = _merge_launch_groups(tt, [state["launch_groups"] for state in states])
    cbs, semaphores = consolidate(tt, states, groups)
    descriptor = tt.ProgramDescriptor(
        kernels=_descriptors(tt, launches),
        cbs=cbs,
        semaphores=semaphores,
    )
    io = [tensor for state in states for tensor in state["io"]]

    def run():
        tt.generic_op(io, descriptor)

    return dict(
        run=run,
        descriptor=descriptor,
        io=io,
        states=states,
        workers=sum(state["workers"] for state in states),
        program_count=1,
    )
