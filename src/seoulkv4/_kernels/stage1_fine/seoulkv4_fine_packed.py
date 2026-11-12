"""Packed Fine decoder/scorer with single-program native TopP selection.

T1 upper planes and one FP16-valued scale per 32-centroid group (stored as
FP32); T2 lower64B records. The reader gathers32 centroid records per batch
directly into left/right UInt32 faces.
Metadata scratch is immutable through MATH consumption. No scalar nibble LUT
expansion, no host dequantization, no GDDR intermediate in the scoring path.
"""

from seoulkv4._kernels.stage1_fine.abi import FineCB as CB

import os

import torch

from seoulkv4._kernels.stage15.stage15_sfpu_native import coordinate_order
from seoulkv4._kernels.topk.tt_topk_large_fixed import _ranges


def sources(
    *,
    select=False,
    shared_upper=False,
    external_tags=False,
    dynamic=False,
    dual_risc=None,
    profile=None,
    wait_profile=None,
    fetch_profile=False,
    cleanup_profile=None,
):
    """Generate all roles from their explicit input and selection contract."""
    from seoulkv4._kernels.codegen import fine

    if profile is None:
        profile = os.environ.get("SKV4_FINE_PACKED_PROFILE") == "1"
    if wait_profile is None:
        wait_profile = os.environ.get("SKV4_FINE_WAIT_PROFILE", "")
    if wait_profile not in ("", "issue", "split"):
        raise ValueError("SKV4_FINE_WAIT_PROFILE must be issue or split")
    if wait_profile and not profile:
        raise ValueError("Fine wait clocks require SKV4_FINE_PACKED_PROFILE=1")
    if dual_risc is None:
        dual_risc = select and shared_upper
    mode = wait_profile or ("packed" if profile else "plain")
    if cleanup_profile is None:
        cleanup_profile = os.environ.get("SKV4_REPLAY_CLEANUP_PROFILE") == "1"
    if cleanup_profile and not profile and select:
        mode = "cleanup"
    if fetch_profile:
        if not profile:
            raise ValueError("isolated fetch diagnostics require packed profiling")
        mode = "fetch_" + wait_profile if wait_profile else "fetch"
    options = dict(
        select=select,
        shared_upper=shared_upper,
        external_tags=external_tags,
        dynamic=dynamic,
        dual_risc=dual_risc,
        profile=mode,
    )
    return tuple(emit(**options) for emit in (fine.reader, fine.compute, fine.writer))


READER = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t LIVE = get_compile_time_arg_val(0);
    constexpr uint32_t CHUNKS = (LIVE + 31) / 32;
    constexpr auto ua = TensorAccessorArgs<2>();
    constexpr auto la = TensorAccessorArgs<ua.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto qa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    const auto upper = TensorAccessor(ua, get_arg_val<uint32_t>(0), 64);
    const auto lower = TensorAccessor(la, get_arg_val<uint32_t>(1), 64);
    const auto tables = TensorAccessor(ta, get_arg_val<uint32_t>(2), 64);
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(3), 64);
    const auto query = TensorAccessor(qa, get_arg_val<uint32_t>(4), 2048);
    const uint32_t ip = get_write_ptr(2);
    auto selected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ip);
    for (uint32_t page = 0; page < CHUNKS * 2; ++page) {
        noc_async_read(get_noc_addr(get_arg_val<uint32_t>(5) / 16 + page, ids), ip + page * 64, 64);
    }
    noc_async_read_barrier();

    cb_reserve_back(6, 1);
    const uint32_t meta = get_write_ptr(6);
    constexpr uint32_t PADDED = CHUNKS * 32;
    constexpr uint32_t CACHE_OFF = PADDED * 4;
    // Read scale pages from the compact group table into a run-coalesced cache.
    // Each FP32 entry applies to 32 physical centroids, and one 64-B page therefore
    // covers 512 centroids. IDs are ascending in production, so each page is
    // fetched once; arbitrary diagnostic order remains correct and merely
    // refetches a page.
    uint32_t page_count = 0;
    uint32_t last_page = 0xffffffffu;
    for (uint32_t key = 0; key < LIVE; ++key) {
        const uint32_t page = selected[key] / 512;
        if (page != last_page) {
            noc_async_read(get_noc_addr(page, tables),
                meta + CACHE_OFF + page_count * 64, 64);
            ++page_count;
            last_page = page;
            if ((page_count & 31u) == 0) noc_async_read_barrier();
        }
    }
    noc_async_read_barrier();

    // Prefetch first residual-plane chunk. Its DMA can progress while this
    // NCRISC compacts the already-resident scale cache below.
    cb_reserve_back(5, 1);
    const uint32_t first_tile = get_write_ptr(5);
    if (32 > LIVE) {
        auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(first_tile);
        for (uint32_t i = 0; i < 1024; ++i) words[i] = 0;
    }
    for (uint32_t row = 0; row < 32 && row < LIVE; ++row) {
        const uint32_t key = selected[row];
        const uint32_t dst = first_tile + (row / 16) * 2048 + (row % 16) * 64;
        noc_async_read(get_noc_addr(key, lower), dst, 64);
        noc_async_read(get_noc_addr(key, upper), dst + 1024, 64);
    }

    auto table_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
    auto cached_pages = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        meta + CACHE_OFF);
    uint32_t page_slot = 0;
    last_page = 0xffffffffu;
    for (uint32_t key = 0; key < LIVE; ++key) {
        const uint32_t page = selected[key] / 512;
        if (page != last_page) {
            if (last_page != 0xffffffffu) ++page_slot;
            last_page = page;
        }
        table_words[key] = cached_pages[
            page_slot * 16 + ((selected[key] >> 5) & 15u)];
    }
    for (uint32_t w = LIVE; w < PADDED; ++w) table_words[w] = 0;
    __sync_synchronize();
    cb_push_back(6, 1);

    cb_reserve_back(0, 8);
    for (uint32_t t = 0; t < 8; ++t) {
        noc_async_read(get_noc_addr(t, query), get_write_ptr(0) + t * 2048, 2048);
    }
    noc_async_read_barrier();
    cb_push_back(0, 8);
    // The query fence also completes the first residual-plane chunk.
    cb_push_back(5, 1);
    for (uint32_t chunk = 1; chunk < CHUNKS; ++chunk) {
        cb_reserve_back(5, 1);
        const uint32_t tile = get_write_ptr(5);
        if (chunk * 32 + 32 > LIVE) {
            auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
            for (uint32_t i = 0; i < 1024; ++i) words[i] = 0;
        }
        for (uint32_t row = 0; row < 32 && chunk * 32 + row < LIVE; ++row) {
            const uint32_t key = selected[chunk * 32 + row];
            const uint32_t dst = tile + (row / 16) * 2048 + (row % 16) * 64;
            noc_async_read(get_noc_addr(key, lower), dst, 64);
            noc_async_read(get_noc_addr(key, upper), dst + 1024, 64);
        }
        noc_async_read_barrier();
        cb_push_back(5, 1);
    }
}
"""


def compute_source():
    """Generate Fine BF16 scoring with an explicit postscale helper."""
    from seoulkv4._kernels.codegen.fine_postscale_codegen import SOURCE

    return (
        r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void clear_score() {
    #pragma GCC unroll 32
    for(uint32_t i=0;i<32;++i)
        sfpi::dst_reg[i].mode<sfpi::DataLayout::F32>()=sfpi::vFloat(0.0f);
}
"""
        + SOURCE
        + r"""
}
#endif

inline void expand_eight() {
    cb_reserve_back(1,8);
    tile_regs_acquire();
    copy_tile(5,0,0);
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        expand_eight_planes,0,VectorMode::None));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1,1);
    pack_tile(2,1);
    pack_tile(3,1);
    pack_tile(4,1);
    pack_tile(5,1);
    pack_tile(6,1);
    pack_tile(7,1);
    pack_tile(0,1);
    tile_regs_release();
    cb_push_back(1,8);
}

void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0), PHASE=get_compile_time_arg_val(1);
    compute_kernel_hw_startup(5,0,1);

    uint32_t table_base = 0;
    // This mailbox is a DATA-readiness fence, not only an address transfer.
    // MATH's scalar L1 loads can run ahead of queued Tensix copy instructions.
    cb_wait_front(6,1);
    UNPACK(mailbox_write(ckernel::ThreadId::MathThreadId,
        get_local_cb_interface(6).fifo_rd_ptr << 4));
    MATH(table_base = mailbox_read(ckernel::ThreadId::UnpackThreadId));
    cb_wait_front(0,8);
    if constexpr(PHASE>0) {
        // Unfused TopK-XL leaves SFPU index tracking enabled across programs.
        // Custom SFPI execute calls do not initialize this state themselves.
        MATH(SFPU_UNARY_INIT(unused));
        for(uint32_t chunk=0;chunk<N/32;++chunk) {
            cb_wait_front(5,1);
            // A packed chunk is consumed immediately; the writer can drain
            // its scores while MATH starts decoding the next chunk.
            copy_tile_to_dst_init_short_with_dt(1,5);
            pack_reconfig_data_format(1);
            
            expand_eight();
            cb_pop_front(5,1);
            if constexpr(PHASE!=3) {
                if constexpr(PHASE==2) {
                    // Matrix maps Q -> SrcB, K -> SrcA, unlike tile-copy.
                    reconfig_data_format<SrcOrder::Reverse>(0,1);
                    matmul_init(0,1,1);
                    pack_reconfig_data_format(16);
                }
                cb_wait_front(1,8);
                if constexpr(PHASE==2) {
                    cb_reserve_back(16,1);
                    tile_regs_acquire();
                    if(chunk==0) {
                        // A prior unfused TopK-XL leaves the first FPU output
                        // contaminated even after ZEROACC. Materialize zeros
                        // once; later chunks are cleared by the PACK release.
                        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                            DST_ACCUM_MODE,clear_score,0,VectorMode::None));
                        // SFPU execute preserves source zero flags; restore
                        // the normal floating-point matmul interpretation.
                        reconfig_data_format<SrcOrder::Reverse>(0,1);
                    }
                    for(uint32_t kt=0;kt<8;++kt) matmul_tiles(0,1,kt,kt,0);
                    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                        DST_ACCUM_MODE,fine_post_scale,0,VectorMode::None,
                        table_base + chunk * 128));

                    tile_regs_commit(); tile_regs_wait();
                    pack_tile(0,16);
                    tile_regs_release(); cb_push_back(16,1);
                }
                cb_pop_front(1,8);
            }
        }
    }
    cb_pop_front(0,8); cb_pop_front(6,1);
}
"""
    )


WRITER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0), SCORE = get_compile_time_arg_val(1);
    constexpr auto oa = TensorAccessorArgs<2>();
    constexpr uint32_t CB = SCORE ? 16 : 1, TILES = SCORE ? 1 : 8;
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    for (uint32_t chunk = 0; chunk < N / 32; ++chunk) {
        cb_wait_front(CB, TILES);
        for (uint32_t t = 0; t < TILES; ++t) {
            noc_async_write(get_read_ptr(CB) + t * 2048,
                get_noc_addr((get_arg_val<uint32_t>(1) + chunk) * TILES + t, output), 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(CB, TILES);
    }
}
"""


def build(
    tt,
    device,
    cell,
    selected,
    *,
    score=False,
    points=None,
    select=False,
    external_tags=None,
    dynamic_external=False,
    shared_upper=None,
    external_query=None,
):
    """Contiguous32-key worker shards. Decode exports are diagnostic only."""
    selected = selected.long()
    if select and (not score or not torch.equal(selected, cell["coarse_ids"].long())):
        raise ValueError("selection requires scoring the sorted full coarse fixture")
    if external_tags is not None and not select:
        raise ValueError("external Coarse tags require native Fine selection")
    if dynamic_external and external_tags is None:
        raise ValueError("dynamic Fine requires Program C's external tag tensor")
    if (
        selected.ndim != 1
        or not len(selected)
        or (selected < 0).any()
        or (selected >= len(cell["levels"])).any()
    ):
        raise ValueError("nonempty in-range centroid IDs required")
    live_capacity = len(cell["levels"]) if dynamic_external else len(selected)
    n = max(128 if select else 32, (live_capacity + 31) // 32 * 32)

    def upload(value, dtype, *, dram=False, tiled=False):
        return tt.from_torch(
            value,
            dtype=dtype,
            device=device,
            layout=tt.TILE_LAYOUT if tiled else tt.ROW_MAJOR_LAYOUT,
            memory_config=tt.DRAM_MEMORY_CONFIG if dram else tt.L1_MEMORY_CONFIG,
        )

    upper = upload(cell["upper"], tt.uint8) if shared_upper is None else shared_upper
    query_order = coordinate_order()
    lower_value = cell["lower"]
    if shared_upper is not None:
        from seoulkv4._kernels.stage1_coarse.coarse_native_codec import (
            fine_coordinate_order,
            prepack_fine_lower,
        )

        lower_value = prepack_fine_lower(lower_value)
        query_order = fine_coordinate_order()
    lower = upload(lower_value, tt.uint8, dram=True)
    if cell.get("centroid_codec") != "fp16-scale-odd-v1":
        raise ValueError(
            "convert the reference cell with upgrade_cell before Fine construction"
        )
    if cell.get("group_scale_size") != 32:
        raise ValueError("Fine requires one resident scale per 32-centroid group")
    scales = cell["fine_scale"].float().reshape(-1, 32)
    if not torch.equal(scales, scales[:, :1].expand_as(scales)):
        raise ValueError("Fine centroid scales are not shared within each group")
    group_scales = cell.get("group_fine_scale")
    if group_scales is None:
        raise ValueError("Fine group-scale table is missing")
    group_scales = group_scales.float()
    if not torch.equal(group_scales, scales[:, 0]):
        raise ValueError("Fine group-scale table disagrees with the reference view")
    group_scales = torch.nn.functional.pad(group_scales, (0, (-len(group_scales)) % 16))
    tables = upload(
        group_scales.contiguous().view(torch.uint8).reshape(-1, 64), tt.uint8
    )
    if external_tags is None:
        ids = torch.zeros(n, dtype=torch.int32)
        ids[: len(selected)] = selected.int()
        ids = upload(ids.reshape(-1, 16), tt.uint32)
    else:
        ids = external_tags
    query = external_query
    if query is None:
        value = torch.zeros(1, 1, 32, 256, dtype=torch.bfloat16)
        value[0, 0, :4] = cell["q"][:, query_order]
        query = upload(value, tt.bfloat16, tiled=True)
    output = tt.allocate_tensor_on_device(
        tt.Shape((n // 32, 1 if score else 8, 32, 32)),
        tt.bfloat16,
        tt.TILE_LAYOUT,
        device,
        tt.L1_MEMORY_CONFIG,
    )
    points = (
        ([(i, 0) for i in range(4)] if select else [(0, 0)])
        if points is None
        else list(points)
    )
    if not points or len(points) != len(set(points)):
        raise ValueError("nonempty distinct worker cores required")
    points = points[: n // 32]
    if select:
        points = points[: len(points) // 4 * 4]
        if not points:
            raise ValueError("native Fine selection needs at least four cores")
    shards = []
    for i, point in enumerate(points):
        first = (n // 32) * i // len(points)
        end = (n // 32) * (i + 1) // len(points)
        shards.append(
            (
                point,
                first,
                (end - first) * 32,
                max(0, min(len(selected), end * 32) - first * 32),
            )
        )
    max_local = max(row[2] for row in shards)
    if max_local * 72 + 45056 > 750000:
        raise ValueError(
            "packed Fine worker scratch exceeds conservative750k budget; use more workers"
        )
    cores = _ranges(tt, points)

    kernel_groups = {}

    def kernel(source, ct, config, point, runtime):
        # Each role has one fixed config in this builder. Share a kernel handle
        # across workers with identical code/CT args; runtime args remain per
        # core. v0.77 Program(descriptor) creates a handle PER descriptor, even
        # when JIT deduplicates the compiled binary. Avoid 3*workers handles.
        key = (source, tuple(ct), type(config))
        group = kernel_groups.setdefault(
            key,
            dict(source=source, ct=ct, config=config, points=[], rt=tt.RuntimeArgs()),
        )
        group["points"].append(point)
        group["rt"][point[0]][point[1]] = runtime

    def descriptors():
        return [
            tt.KernelDescriptor(
                kernel_source=group["source"],
                source_type=tt.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=_ranges(tt, group["points"]),
                compile_time_args=group["ct"],
                runtime_args=group["rt"],
                config=group["config"],
            )
            for group in kernel_groups.values()
        ]

    cb_specs = [
        (CB.QUERY, tt.bfloat16, 2048, 8),
        (CB.DECODED_KEYS, tt.bfloat16, 2048, 8),
        (CB.SELECTED_TAGS, tt.uint32, max_local * 4, 1),
        (CB.PACKED_KEY_PLANES, tt.uint32, 4096, 2),
        (CB.SCALES, tt.uint8, max_local * 68, 1),
        (CB.SCORE_TILES, tt.bfloat16, 2048, 2),
    ]
    if dynamic_external:
        cb_specs.append((CB.LIVE_CONTROL, tt.uint32, 64, 1))
    if shared_upper is not None:
        cb_specs.append((CB.UPPER_CACHE, tt.uint32, 2048, 1))
    dual_reader = shared_upper is not None and select
    if dual_reader:
        cb_specs.extend(
            ((CB.UPPER_JOB, tt.uint32, 64, 1), (CB.UPPER_DONE, tt.uint32, 64, 1))
        )
    cbs = [
        tt.CBDescriptor(
            total_size=page * count,
            core_ranges=cores,
            format_descriptors=[
                tt.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=page)
            ],
        )
        for i, dtype, page, count in cb_specs
    ]
    selection = None
    semaphores = []
    if select:
        from seoulkv4._kernels.stage1_fine import (
            seoulkv4_fine_native_select as native_select,
        )

        selection = native_select.prepare(
            tt,
            device,
            cell,
            output,
            n,
            points,
            shards,
            external_tags=external_tags,
            extra_worker_bytes=128 if dual_reader else 0,
        )
        cbs.extend(selection["cbs"])
        semaphores = selection["semaphores"]
    if dual_reader:
        from seoulkv4._kernels.stage1_fine import seoulkv4_fine_dual_reader

        seoulkv4_fine_dual_reader.require_supported_profile()
    reader_source, compute_text, writer_source = sources(
        select=select,
        shared_upper=shared_upper is not None,
        external_tags=external_tags is not None,
        dynamic=dynamic_external,
        dual_risc=dual_reader,
    )
    from seoulkv4._kernels.stage1_fine.abi import fine_abi

    reader_rt, reader_ct = fine_abi("reader", select=select, dynamic=dynamic_external)
    compute_rt, compute_ct = fine_abi("compute", select=select)
    writer_rt, writer_ct = fine_abi(
        "writer", select=select, dynamic=dynamic_external, dual_risc=dual_reader
    )
    reader_tensors = dict(upper=upper, lower=lower, tables=tables, ids=ids, query=query)
    writer_tensors = (
        dict(zip(("scores", "tags", "output", "heads"), selection["io"]))
        if select
        else dict(output=output)
    )
    if dual_reader:
        writer_tensors["upper"] = upper
    for index, (point, first, local_n, live) in enumerate(shards):
        native_args = selection["args"](index) if select else None
        worker_fields = (
            dict(worker=index, workers=len(shards)) if dynamic_external else {}
        )
        kernel(
            reader_source,
            reader_ct.pack(
                tt,
                tensors=reader_tensors,
                capacity=local_n if dynamic_external else live,
                score=int(score),
                **(dict(chunks=local_n // 32) if select else {}),
            ),
            tt.ReaderConfigDescriptor(),
            point,
            reader_rt.pack(
                **{
                    name: tensor.buffer_address()
                    for name, tensor in reader_tensors.items()
                },
                first=first * 32,
                **(native_args["reader"] if select else {}),
                **worker_fields,
            ),
        )
        kernel(
            compute_text,
            compute_ct.pack(tt, tensors={}, n=local_n, phase=2 if score else 3),
            tt.ComputeConfigDescriptor(
                math_fidelity=tt.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=True,
            ),
            point,
            compute_rt.pack(**(native_args["compute"] if select else {})),
        )
        kernel(
            writer_source,
            writer_ct.pack(
                tt,
                tensors=writer_tensors,
                n=local_n,
                score=1 if select else int(score),
                **(
                    dict(
                        total=n,
                        live=len(selected),
                        p=cell["p"],
                        out=selection["outwidth"],
                        max_local=selection["max_local"],
                    )
                    if select
                    else {}
                ),
            ),
            tt.WriterConfigDescriptor(),
            point,
            writer_rt.pack(
                **{
                    name: tensor.buffer_address()
                    for name, tensor in writer_tensors.items()
                },
                first=first,
                **(native_args["writer"] if select else {}),
                **worker_fields,
            ),
        )
    descriptor = tt.ProgramDescriptor(
        kernels=descriptors(), cbs=cbs, semaphores=semaphores
    )
    io = [upper, lower, tables, ids, query, output]
    if selection is not None:
        io.extend(selection["io"][1:])

    def run():
        tt.generic_op(io, descriptor)

    return dict(
        run=run,
        descriptor=descriptor,
        io=io,
        output=output if selection is None else selection["output"],
        scores=output,
        heads=None if selection is None else selection["heads"],
        heads_are_positions=selection is not None,
        tt=tt,
        cell=cell,
        decoded=None,
        n=n,
        workers=len(points),
        shards=shards,
        native_selection=select,
        dynamic_external=dynamic_external,
        input_supply="dual-risc" if dual_reader else "ncrisc",
        centroid_scale_group=32,
        centroid_scale_t1_bytes=group_scales.numel() * 4,
        query=query,
        query_order=query_order,
    )


def build_all(
    tt,
    device,
    cells,
    groups,
    *,
    score=False,
    select=False,
    external_tags=None,
    dynamic_external=False,
    shared_upper=None,
    external_queries=None,
):
    """Single-program, disjoint-cell scorer/decoder correctness checkpoint."""
    if len(cells) != len(groups):
        raise ValueError("one group per cell required")
    points = [point for group in groups for point in group]
    if len(points) != len(set(points)):
        raise ValueError("cell worker groups must be disjoint")
    if external_tags is not None and len(external_tags) != len(cells):
        raise ValueError("one external Coarse tag tensor per Fine cell required")
    if shared_upper is not None and len(shared_upper) != len(cells):
        raise ValueError("one shared Coarse upper tensor per Fine cell required")
    if external_queries is not None and len(external_queries) != len(cells):
        raise ValueError("one external query tensor per Fine cell required")
    states = [
        build(
            tt,
            device,
            cell,
            cell["coarse_ids"],
            score=score,
            points=group,
            select=select,
            external_tags=None if external_tags is None else external_tags[index],
            dynamic_external=dynamic_external,
            shared_upper=None if shared_upper is None else shared_upper[index],
            external_query=None
            if external_queries is None
            else external_queries[index],
        )
        for index, (cell, group) in enumerate(zip(cells, groups))
    ]
    descriptor = tt.ProgramDescriptor(
        kernels=[kernel for state in states for kernel in state["descriptor"].kernels],
        cbs=[cb for state in states for cb in state["descriptor"].cbs],
        semaphores=[sem for state in states for sem in state["descriptor"].semaphores],
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
