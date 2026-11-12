"""Fixed-output wrapper for Blackhole's TopK-XL large-index kernel.

``ttnn.experimental.topk_large_indices`` allocates its output on every Python
call.  That is convenient for ordinary TTNN graphs, but SeoulKV needs a stable
L1 address so the following program can consume the frontier under trace
replay.  This module instantiates the same installed TT-Metal kernels through
``generic_op`` and binds them to a caller-owned output tensor.
"""

from __future__ import annotations


_READER = (
    "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/"
    "device/kernels/reader.cpp"
)
_COMPUTE = (
    "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/"
    "device/kernels/compute.cpp"
)
_WRITER = (
    "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/"
    "device/kernels/writer.cpp"
)


_PAYLOAD_WRITER = r"""
#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {

template <uint32_t source_slices_per_row, uint32_t output_slices_per_row,
          uint32_t slice_bytes>
FORCE_INLINE void copy_row_to_scratch(
    CircularBuffer& src_cb, CircularBuffer& scratch_cb, const Noc& noc) {
    constexpr uint32_t transfer_bytes = slice_bytes;
    const uint32_t src_base = src_cb.get_read_ptr();
    const uint32_t dst_base = scratch_cb.get_write_ptr();
    const uint32_t noc_id = noc.get_noc_id();
    const auto local_src = [noc_id](uint32_t addr) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = static_cast<uint32_t>(my_x[noc_id]),
            .noc_y = static_cast<uint32_t>(my_y[noc_id]), .addr = addr};
    };
    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{}, transfer_bytes, local_src(src_base));
    for (uint32_t dst_slice = 0; dst_slice < output_slices_per_row; ++dst_slice) {
        const uint32_t tile_col = dst_slice >> 2;
        const uint32_t face_col = dst_slice & 0x1;
        const uint32_t face_row_offset =
            (dst_slice & 0x2) ? source_slices_per_row / 2 : 0;
        const uint32_t src_slice =
            (2 * tile_col) + face_col + face_row_offset;
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(dst_base + dst_slice * slice_bytes),
            transfer_bytes, local_src(src_base + src_slice * slice_bytes),
            {.offset_bytes = 0});
    }
    noc.async_read_barrier();
}

}  // namespace

void kernel_main() {
    constexpr uint32_t CB_INDICES = get_compile_time_arg_val(0);
    constexpr uint32_t CB_SCRATCH = get_compile_time_arg_val(1);
    constexpr uint32_t POS_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t SOURCE_SLICES = get_compile_time_arg_val(3);
    constexpr uint32_t OUTPUT_SLICES = get_compile_time_arg_val(4);
    constexpr uint32_t SLICE_BYTES = get_compile_time_arg_val(5);
    constexpr uint32_t WIDTH = get_compile_time_arg_val(6);
    constexpr uint32_t K = get_compile_time_arg_val(7);
    constexpr uint32_t W = get_compile_time_arg_val(8);
    constexpr uint32_t G = get_compile_time_arg_val(9);
    constexpr auto pia = TensorAccessorArgs<10>();
    constexpr auto lia = TensorAccessorArgs<pia.next_compile_time_args_offset()>();
    constexpr auto lsa = TensorAccessorArgs<lia.next_compile_time_args_offset()>();
    constexpr auto oia = TensorAccessorArgs<lsa.next_compile_time_args_offset()>();
    constexpr auto osa = TensorAccessorArgs<oia.next_compile_time_args_offset()>();

    const uint32_t positions_addr = get_arg_val<uint32_t>(0);
    const uint32_t list_ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t list_scores_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_ids_addr = get_arg_val<uint32_t>(3);
    const uint32_t out_scores_addr = get_arg_val<uint32_t>(4);
    const uint32_t row = get_arg_val<uint32_t>(5);

    const auto positions = TensorAccessor(pia, positions_addr, POS_BYTES);
    const auto list_ids = TensorAccessor(lia, list_ids_addr, WIDTH * 4);
    const auto list_scores = TensorAccessor(lsa, list_scores_addr, WIDTH * 2);
    const auto out_ids = TensorAccessor(oia, out_ids_addr, K * 4);
    const auto out_scores = TensorAccessor(osa, out_scores_addr, K * 2);

    constexpr uint32_t IDS_BYTES = WIDTH * 4;
    constexpr uint32_t SCORES_BYTES = WIDTH * 2;
    constexpr uint32_t OUT_IDS_OFF = POS_BYTES + IDS_BYTES + SCORES_BYTES;
    constexpr uint32_t OUT_SCORES_OFF = OUT_IDS_OFF + K * 4;
    constexpr uint32_t SHARD_IDS_BYTES = K * 4;

    CircularBuffer indices_cb(CB_INDICES);
    CircularBuffer scratch_cb(CB_SCRATCH);
    Noc noc;
    indices_cb.wait_front(1);
    scratch_cb.reserve_back(1);
    copy_row_to_scratch<SOURCE_SLICES, OUTPUT_SLICES, SLICE_BYTES>(
        indices_cb, scratch_cb, noc);
    indices_cb.pop_front(1);
    scratch_cb.push_back(1);
    scratch_cb.wait_front(1);

    const uint32_t base = scratch_cb.get_read_ptr();
    volatile tt_l1_ptr uint32_t* p =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);

    // Start the contiguous traffic before the scalar prefix scan.  The BRISC
    // can inspect the local positions while these NoC commands are in flight.
    noc.async_write(scratch_cb, positions, POS_BYTES,
                    {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
    noc.async_read(list_scores, scratch_cb, SCORES_BYTES,
                   {.page_id = row, .offset_bytes = 0},
                   {.offset_bytes = POS_BYTES + IDS_BYTES});

    // Both the global output and each shard frontier are score ordered.  Walk
    // backward only until every contributing shard has appeared; its first
    // occurrence is the prefix end.  BF16 cutoff ties may violate stable
    // shard order.  We intentionally accept arbitrary tie selection here to
    // avoid a full K-position pass.  Eight read-ahead entries cover the
    // measured worst BF16 tie disorder (7) at negligible transfer cost.
    uint32_t prefix[W] = {};
    constexpr uint32_t TIE_SLOP = 8;
    uint32_t unseen = (1u << W) - 1;
    for (uint32_t rank = K; rank > 0 && unseen != 0; --rank) {
        const uint32_t src = p[rank - 1];
        const uint32_t shard = src / K;
        const uint32_t bit = 1u << shard;
        if (unseen & bit) {
            const uint32_t end = src - shard * K + 1 + TIE_SLOP;
            prefix[shard] = end < K ? end : K;
            unseen &= ~bit;
        }
    }

    // Gather IDs directly from the sorted shard-frontier pages.
    for (uint32_t shard = 0; shard < W; ++shard) {
        if (prefix[shard] == 0) continue;
        noc.async_read(list_ids, scratch_cb, prefix[shard] * 4,
                       {.page_id = row,
                        .offset_bytes = shard * SHARD_IDS_BYTES},
                       {.offset_bytes = POS_BYTES + shard * SHARD_IDS_BYTES});
    }
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* li =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + POS_BYTES);
    volatile tt_l1_ptr uint16_t* ls =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            base + POS_BYTES + IDS_BYTES);
    volatile tt_l1_ptr uint32_t* oi =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + OUT_IDS_OFF);
    volatile tt_l1_ptr uint16_t* os =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + OUT_SCORES_OFF);
    for (uint32_t rank = 0; rank < K; ++rank) {
        const uint32_t src = p[rank];
        const uint32_t shard = src / K;
        const uint32_t local_rank = src - shard * K;
        // A non-stable BF16 tie can place a later equal-score local rank
        // before the inferred prefix end.  Canonicalize it to the loaded
        // boundary candidate instead of touching unread L1.
        const uint32_t id_src = local_rank < prefix[shard]
            ? src : shard * K + prefix[shard] - 1;
        const uint32_t dest = (rank % G) * (K / G) + (rank / G);
        oi[dest] = li[id_src];
        os[dest] = ls[src];
    }
    noc.async_write(CoreLocalMem<uint32_t>(base + OUT_IDS_OFF), out_ids,
                    K * 4, {.offset_bytes = 0},
                    {.page_id = row, .offset_bytes = 0});
    noc.async_write(CoreLocalMem<uint32_t>(base + OUT_SCORES_OFF), out_scores,
                    K * 2, {.offset_bytes = 0},
                    {.page_id = row, .offset_bytes = 0});
    noc.async_write_barrier();
    scratch_cb.pop_front(1);
}
"""


def _ranges(ttnn, points):
    return ttnn.CoreRangeSet([
        ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
        for x, y in points
    ])


def build_topk_large_indices(
    ttnn,
    device,
    values,
    *,
    rows: int,
    width: int,
    k: int = 2048,
    valid_length: int | None = None,
    core_points=None,
    payload=None,
):
    """Build Blackhole TopK-XL with a stable row-major UInt32 L1 output.

    This intentionally supports only SeoulKV's K=2048 operating point.  Each
    active core processes one row, matching the 16 independent query heads in
    the final Stage-1.5 merge.  ``width`` is the physical row stride while
    ``valid_length`` can hide a padded suffix.
    """
    if k != 2048:
        raise ValueError("fixed TopK-XL wrapper currently requires k=2048")
    if rows <= 0 or width < k or width % 16:
        raise ValueError("require rows>0, width>=k, and width divisible by 16")
    valid = width if valid_length is None else valid_length
    if not 0 < valid <= width:
        raise ValueError("valid_length must be in [1,width]")

    if core_points is None:
        grid = device.compute_with_storage_grid_size()
        core_points = [
            (x, y) for y in range(grid.y) for x in range(grid.x)
        ][:rows]
    else:
        core_points = list(core_points)
    if len(core_points) < rows:
        raise ValueError(f"need {rows} cores, got {len(core_points)}")
    core_points = core_points[:rows]
    cores = _ranges(ttnn, core_points)

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, rows, k)), ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG,
    )
    in_cta = list(ttnn.TensorAccessorArgs(values).get_compile_time_args())
    out_cta = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    tile_elements = 1024
    tile_bytes_in = tile_elements * 2
    chunk_bytes = k * 2
    tiles_per_chunk = 2
    num_chunks = (valid + k - 1) // k
    tail_elements = valid - (num_chunks - 1) * k
    tail_bytes = tail_elements * 2
    input_row_bytes = width * 2
    output_row_bytes = k * 4

    rr = ttnn.RuntimeArgs()
    cr = ttnn.RuntimeArgs()
    wr = ttnn.RuntimeArgs()
    if payload is not None:
        list_ids = payload["list_ids"]
        list_scores = payload["list_scores"]
        out_ids = payload["out_ids"]
        out_scores = payload["out_scores"]
        group = int(payload["group"])
        workers = width // k
        if width != workers * k:
            raise ValueError("payload rows must be W contiguous K-entry lists")
        payload_tensors = [output, list_ids, list_scores, out_ids, out_scores]
        payload_cta = [
            list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
            for t in payload_tensors
        ]
    for row, (x, y) in enumerate(core_points):
        rr[x][y] = [
            values.buffer_address(), row, 1, num_chunks,
            tail_bytes, input_row_bytes,
        ]
        cr[x][y] = [1, num_chunks, tail_elements]
        if payload is None:
            wr[x][y] = [output.buffer_address(), row, 1]
        else:
            wr[x][y] = [
                output.buffer_address(), list_ids.buffer_address(),
                list_scores.buffer_address(), out_ids.buffer_address(),
                out_scores.buffer_address(), row,
            ]

    reader = ttnn.KernelDescriptor(
        kernel_source=_READER,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[0, chunk_bytes, tile_bytes_in,
                           tiles_per_chunk, *in_cta],
        runtime_args=rr,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        # ComputeBodyMode::FusedSegmented.  At six chunks this is one fused
        # segment and splits UInt32 indices only after the final survivor.
        compile_time_args=[0, 1, k, 2],
        runtime_args=cr,
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=True, dst_full_sync_en=True,
        ),
    )
    if payload is None:
        writer_source = _WRITER
        writer_source_type = ttnn.KernelDescriptor.SourceType.FILE_PATH
        writer_cta = [1, 2, output_row_bytes, 128, 128, 64, *out_cta]
        scratch_bytes = output_row_bytes
    else:
        writer_source = _PAYLOAD_WRITER
        writer_source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        writer_cta = [
            1, 2, output_row_bytes, 128, 128, 64,
            width, k, workers, group,
            *[v for args_ in payload_cta for v in args_],
        ]
        scratch_bytes = (
            output_row_bytes + width * 4 + width * 2 + k * 4 + k * 2
        )
        scratch_bytes = (scratch_bytes + 63) // 64 * 64
    writer = ttnn.KernelDescriptor(
        kernel_source=writer_source,
        source_type=writer_source_type,
        core_ranges=cores,
        # K=2048: 128 input 16-element slices, 128 output slices.
        compile_time_args=writer_cta,
        runtime_args=wr,
        config=ttnn.WriterConfigDescriptor(),
    )

    def cb(index, dtype, page_size, pages):
        return ttnn.CBDescriptor(
            total_size=page_size * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=page_size,
            )],
        )

    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader, compute, writer],
        cbs=[
            cb(0, ttnn.bfloat16, tile_bytes_in, 4),
            cb(1, ttnn.float32, output_row_bytes, 2),
            cb(2, ttnn.uint8, scratch_bytes, 1),
        ],
    )
    io = [values, output] if payload is None else [values, *payload_tensors]

    def run():
        ttnn.generic_op(io, descriptor)
        return output

    return {
        "run": run,
        "output": output,
        "descriptor": descriptor,
        "io": io,
        "core_points": core_points,
        "rows": rows,
        "width": width,
        "k": k,
        "valid_length": valid,
    }
