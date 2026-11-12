"""Direct merge of merge-ready TopK-XL score/UInt32-ID lists on Blackhole.

Each K=2048 input list must already use the native rank-lane transform
``((rank & 0x7f) << 4) | (rank >> 7)``.  List zero is descending and later
lists are ascending.  The paired output intentionally stays in native order.
"""

from __future__ import annotations

import os


_READER = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr auto sa = TensorAccessorArgs<2>();
    constexpr auto ia = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(2);
    const auto scores = TensorAccessor(sa, scores_addr, W * K * 2);
    const auto ids = TensorAccessor(ia, ids_addr, W * K * 4);
    CircularBuffer score_cb(0);
    CircularBuffer id_cb(1);
    Noc noc;

    for (uint32_t shard = 0; shard < W; ++shard) {
        score_cb.reserve_back(2);
        id_cb.reserve_back(2);
        noc.async_read(scores, score_cb, K * 2,
                       {.page_id = row, .offset_bytes = shard * K * 2},
                       {.offset_bytes = 0});
        noc.async_read(ids, id_cb, K * 4,
                       {.page_id = row, .offset_bytes = shard * K * 4},
                       {.offset_bytes = 0});
        noc.async_read_barrier();
        score_cb.push_back(2);
        id_cb.push_back(2);
    }
}
"""


_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    static_assert(K == 2048, "direct payload merge currently requires K=2048");
    constexpr uint32_t TILES = 2;
    constexpr uint32_t SURV_VALUES = 0;
    constexpr uint32_t SURV_IDS = 2;
    constexpr uint32_t IN_VALUES = 4;
    constexpr uint32_t IN_IDS = 6;

    compute_kernel_hw_startup(0, 4);
    pack_untilize_dest_init<TILES, TILES>(4);
    CircularBuffer score_cb(0);
    CircularBuffer id_cb(1);
    CircularBuffer out_score_cb(4);
    CircularBuffer out_id_cb(5);

    tile_regs_acquire();
    for (uint32_t shard = 0; shard < W; ++shard) {
        const uint32_t value_dst = shard == 0 ? SURV_VALUES : IN_VALUES;
        const uint32_t id_dst = shard == 0 ? SURV_IDS : IN_IDS;

        score_cb.wait_front(TILES);
        topk_xl_copy_tile_init(0);
        topk_xl_copy_tile<K>(0, value_dst, 0, K);
        score_cb.pop_front(TILES);

        id_cb.wait_front(TILES);
        topk_xl_copy_tile_init(1);
        topk_xl_copy_tile<K>(1, id_dst, 0, K);
        id_cb.pop_front(TILES);

        if (shard != 0) {
            // copy_tile_init changes TopK's MOP/address modifiers, so restore
            // the complete unfused state after both operands are resident.
            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(SURV_VALUES);
            topk_xl_rebuild<K, false>(SURV_VALUES, false);
        }
    }
    tile_regs_commit();
    tile_regs_wait();

    out_score_cb.reserve_back(1);
    pack_untilize_dest<TILES, TILES>(4, 1, 0, SURV_VALUES);
    out_score_cb.push_back(1);

    pack_untilize_dest_init<TILES, TILES>(5);
    out_id_cb.reserve_back(1);
    pack_untilize_dest<TILES, TILES>(5, 1, 0, SURV_IDS);
    out_id_cb.push_back(1);
    tile_regs_release();
}
"""


_WRITER = r"""
#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr auto sa = TensorAccessorArgs<1>();
    constexpr auto ia = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(2);
    const auto scores = TensorAccessor(sa, scores_addr, K * 2);
    const auto ids = TensorAccessor(ia, ids_addr, K * 4);
    CircularBuffer score_cb(4);
    CircularBuffer id_cb(5);
    Noc noc;

    score_cb.wait_front(1);
    id_cb.wait_front(1);
    noc.async_write(score_cb, scores, K * 2,
                    {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
    noc.async_write(id_cb, ids, K * 4,
                    {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
    noc.async_write_barrier();
    score_cb.pop_front(1);
    id_cb.pop_front(1);
}
"""


def _ranges(ttnn, points):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in points]
    )


def build_payload_merge(
    ttnn, device, scores, ids, *, rows, workers=6, k=2048, core_points=None
):
    """Merge W native-layout K-lists while carrying arbitrary UInt32 IDs."""
    if k != 2048 or workers < 1:
        raise ValueError("require K=2048 and W>=1")
    if core_points is None:
        grid = device.compute_with_storage_grid_size()
        core_points = [(x, y) for y in range(grid.y) for x in range(grid.x)][:rows]
    else:
        core_points = list(core_points)[:rows]
    if len(core_points) != rows:
        raise ValueError(f"need {rows} cores, got {len(core_points)}")
    cores = _ranges(ttnn, core_points)

    out_scores = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, rows, k)),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    out_ids = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, rows, k)),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    tensors = [scores, ids, out_scores, out_ids]
    cta = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in tensors]
    rr, cr, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for row, (x, y) in enumerate(core_points):
        rr[x][y] = [scores.buffer_address(), ids.buffer_address(), row]
        cr[x][y] = []
        wr[x][y] = [out_scores.buffer_address(), out_ids.buffer_address(), row]

    from seoulkv4._kernels.codegen import component_kernels

    profile = os.environ.get("SKV4_NATIVE_MARKERS") == "1"
    reader_source = component_kernels.payload_reader(profile=profile)
    compute_source = component_kernels.payload_compute(profile=profile)
    writer_source = component_kernels.payload_writer(profile=profile)
    reader = ttnn.KernelDescriptor(
        kernel_source=reader_source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[k, workers, *cta[0], *cta[1]],
        runtime_args=rr,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=compute_source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[k, workers],
        runtime_args=cr,
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
        ),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=writer_source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[k, *cta[2], *cta[3]],
        runtime_args=wr,
        config=ttnn.WriterConfigDescriptor(),
    )

    def cb(index, dtype, page_size, pages):
        return ttnn.CBDescriptor(
            total_size=page_size * pages,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index,
                    data_format=dtype,
                    page_size=page_size,
                )
            ],
        )

    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader, compute, writer],
        cbs=[
            cb(0, ttnn.bfloat16, 2048, 4),
            cb(1, ttnn.uint32, 4096, 4),
            cb(4, ttnn.bfloat16, k * 2, 2),
            cb(5, ttnn.uint32, k * 4, 2),
        ],
    )
    io = [scores, ids, out_scores, out_ids]

    def run():
        ttnn.generic_op(io, descriptor)
        return out_scores, out_ids

    return {
        "run": run,
        "scores": out_scores,
        "ids": out_ids,
        "descriptor": descriptor,
        "io": io,
        "core_points": core_points,
    }
