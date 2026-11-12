"""Explicit native Coarse and selector source generation. No completed source edits."""

from seoulkv4._kernels.stage1_coarse.abi import CoarseSemaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

from seoulkv4._kernels.stage1_coarse.abi import CoarseCB as CB
from seoulkv4._kernels.common.cb_layout import cpp_defines

from seoulkv4._kernels.codegen.preprocessor import profile_defines
from seoulkv4._kernels.stage1_coarse.abi import coarse_defines
from seoulkv4._kernels.common.query_metadata import (
    cpp_defines as query_metadata_defines,
)


def reader(*, select, debug_heads, profile, wait_query=False):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if debug_heads and (not select):
        raise ValueError("debug head export requires selection")
    if profile and (not select):
        raise ValueError("Coarse endpoint profile requires selection")
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + coarse_defines("reader", select=select, debug_heads=debug_heads)
        + profile_defines(profile=profile, select=select, debug_heads=debug_heads)
        + f"#define SKV4_CODEGEN_WAIT_QUERY {int(bool(wait_query))}\n"
        + (
            r"""#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
#include "tools/profiler/kernel_profiler.hpp"
#endif
#endif
#include "api/dataflow/dataflow_api.h"
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
// fine-ingest-meter-v1
inline __attribute__((always_inline)) uint32_t skv_tick() {
    asm volatile("" ::: "memory");
    uint32_t t=*reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory"); return t;
}
uint32_t skv_sum_CO_R_METADATA; uint32_t skv_end_CO_R_METADATA; bool skv_valid_CO_R_METADATA;
uint32_t skv_sum_CO_R_QUERY; uint32_t skv_end_CO_R_QUERY; bool skv_valid_CO_R_QUERY;
uint32_t skv_sum_CO_R_UPPER_STREAM; uint32_t skv_end_CO_R_UPPER_STREAM; bool skv_valid_CO_R_UPPER_STREAM;
uint32_t skv_sum_CO_R_FRONTIER; uint32_t skv_end_CO_R_FRONTIER; bool skv_valid_CO_R_FRONTIER;
uint32_t skv_sum_CO_R_MASK; uint32_t skv_end_CO_R_MASK; bool skv_valid_CO_R_MASK;
#endif
FORCE_INLINE uint32_t coarse_ordered_bf16_pair(uint32_t pair) {
    // Two independent halfword XOR masks, with unsigned wraparound.
    // Preserve the scalar raw-bit order, including signed zero/NaN/inf.
    const uint32_t signs = (pair >> 15) & 0x00010001u;
    return pair ^ (0x80008000u | ((signs << 16) - signs));
}
FORCE_INLINE uint16_t coarse_ordered_bf16(uint16_t bits) {
    return bits ^ (0x8000u | (0u - (bits >> 15)));
}
#endif
void kernel_main() {
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
    skv_sum_CO_R_METADATA=0;skv_sum_CO_R_QUERY=0;skv_sum_CO_R_UPPER_STREAM=0;skv_sum_CO_R_FRONTIER=0;skv_sum_CO_R_MASK=0;skv_valid_CO_R_METADATA=false;skv_valid_CO_R_QUERY=false;skv_valid_CO_R_UPPER_STREAM=false;skv_valid_CO_R_FRONTIER=false;skv_valid_CO_R_MASK=false;
#endif
#endif
#if SKV4_CODEGEN_SELECT
    const uint32_t CHUNKS = get_arg_val<uint32_t>(SKV4_ARG_RT_SCORE_CHUNKS);
#else
    constexpr uint32_t CHUNKS = get_compile_time_arg_val(SKV4_ARG_CT_CHUNKS);
#endif
    constexpr auto ua = TensorAccessorArgs<SKV4_ARG_CT_ACCESSORS>();
    constexpr auto sa = TensorAccessorArgs<ua.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto na = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto qa = TensorAccessorArgs<na.next_compile_time_args_offset()>();
    const auto upper = TensorAccessor(ua, get_arg_val<uint32_t>(SKV4_ARG_RT_UPPER), 4096);
    const auto scales = TensorAccessor(sa, get_arg_val<uint32_t>(SKV4_ARG_RT_SCALES), 64);
    const auto residual = TensorAccessor(ra, get_arg_val<uint32_t>(SKV4_ARG_RT_RESIDUAL), 64);
    const auto qnorm = TensorAccessor(na, get_arg_val<uint32_t>(SKV4_ARG_RT_QNORM), 64);
    const auto query = TensorAccessor(qa, get_arg_val<uint32_t>(SKV4_ARG_RT_QUERY), 2048);
    const uint32_t first_chunk = get_arg_val<uint32_t>(SKV4_ARG_RT_FIRST_CHUNK);
    // Raw UInt32 row PACK overwrites only native mantissas. All batched BFP4
    // ring headers are constant, including across trace replay.
    auto operand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_DECODED_KEYS));
    for (uint32_t tile = 0; tile < 8 * SKV4_COARSE_OPERAND_BATCH; ++tile)
        for (uint32_t word = 0; word < 16; ++word)
            operand[tile * 144 + word] = 0x80808080u;
    __sync_synchronize();
    // Scale-sorted physical chunks share one FP32 scale.  One aligned page per
    // chunk avoids per-column scale selection in MATH.  Residual norms remain
    // per centroid so the certified conservative interval is unchanged.
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
    uint32_t tp_CO_R_METADATA=skv_tick();
#endif
#endif
    cb_reserve_back(SKV4_CB_METADATA, 1);
    const uint32_t table = get_write_ptr(SKV4_CB_METADATA);
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
        noc_async_read(get_noc_addr(first_chunk + chunk, scales),
                       table + chunk * 64, 64);
        noc_async_read(get_noc_addr(first_chunk + chunk, residual),
                       table + CHUNKS * 64 + chunk * 64, 64);
    }
#if SKV4_CODEGEN_WAIT_QUERY
    // The producer publishes both disjoint Q halves and norm/sum before
    // this signal. Never issue a read against last replay's operands.
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(SKV4_SEM_QUERY_READY)), 1);
#endif
    noc_async_read(get_noc_addr(0, qnorm), table + CHUNKS * 128, 64);
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
    skv_end_CO_R_METADATA=skv_tick(); skv_valid_CO_R_METADATA=true;
    skv_sum_CO_R_METADATA+=skv_end_CO_R_METADATA-tp_CO_R_METADATA;
    uint32_t tp_CO_R_QUERY=skv_tick();
#endif
#endif
    cb_reserve_back(SKV4_CB_QUERY, 8);
    for (uint32_t tile = 0; tile < 8; ++tile)
        noc_async_read(get_noc_addr(tile, query),
                       get_write_ptr(SKV4_CB_QUERY) + tile * 2048, 2048);
    noc_async_read_barrier();
    cb_push_back(SKV4_CB_METADATA, 1);
    cb_push_back(SKV4_CB_QUERY, 8);
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
    skv_end_CO_R_QUERY=skv_tick(); skv_valid_CO_R_QUERY=true;
    skv_sum_CO_R_QUERY+=skv_end_CO_R_QUERY-tp_CO_R_QUERY;
    uint32_t tp_CO_R_UPPER_STREAM=skv_tick();
#endif
#endif
    // A physical 4-KiB T1 page contains 64 records.  Each scorer iteration
    // reads one 2-KiB half in one bulk NoC transaction.
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
        cb_reserve_back(SKV4_CB_PACKED_UPPER, 1);
        const uint32_t global = first_chunk + chunk;
        noc_async_read(get_noc_addr(global >> 1, upper)
                           + (global & 1u) * 2048,
                       get_write_ptr(SKV4_CB_PACKED_UPPER), 2048);
        noc_async_read_barrier();
        cb_push_back(SKV4_CB_PACKED_UPPER, 1);
    }
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
    skv_end_CO_R_UPPER_STREAM=skv_tick(); skv_valid_CO_R_UPPER_STREAM=true;
    skv_sum_CO_R_UPPER_STREAM+=skv_end_CO_R_UPPER_STREAM-tp_CO_R_UPPER_STREAM;
    uint32_t tp_CO_R_FRONTIER=skv_tick();
#endif
    /*@CO_R_FRONTIER_B*/
    const uint32_t width = get_arg_val<uint32_t>(SKV4_ARG_RT_WIDTH);
    const uint32_t cohort = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT);
    const uint32_t cohorts = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORTS);
    const uint32_t chunks = (width + 511) / 512;
    cb_reserve_back(SKV4_CB_LOCAL_SCORE_TILES, chunks);
    auto row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_LOCAL_SCORE_TILES));
    const uint32_t tail = width - (chunks - 1) * 512;
    for (uint32_t j = tail; j < 512; ++j)
        row[(chunks - 1) * 1024 + j] = 0xff80;
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_COHORT_SCORES_READY)), 4);
    cb_push_back(SKV4_CB_LOCAL_SCORE_TILES, chunks);
    for (uint32_t level = 0, stride = 1; stride < cohorts; ++level, stride *= 2) {
        if (cohort % (2 * stride)) break;
        if (cohort + stride >= cohorts) continue;
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_FRONTIER_LEVEL_0 + level)), 1);
        cb_reserve_back(SKV4_CB_MERGE_VALUES, 1);
#if SKV4_CODEGEN_DEBUG_HEADS
        cb_reserve_back(SKV4_CB_MERGE_INDICES, 1);
        const uint32_t src = get_write_ptr(SKV4_CB_PEER_FRONTIERS) + level * 3072;
        noc_async_read(get_noc_addr(src), get_write_ptr(SKV4_CB_MERGE_VALUES), 1024);
        noc_async_read(get_noc_addr(src + 1024), get_write_ptr(SKV4_CB_MERGE_INDICES), 2048);
#else
        const uint32_t src = get_write_ptr(SKV4_CB_PEER_FRONTIERS) + level * 2048;
        noc_async_read(get_noc_addr(src), get_write_ptr(SKV4_CB_MERGE_VALUES), 2048);
#endif
        noc_async_read_barrier();
        cb_push_back(SKV4_CB_MERGE_VALUES, 1);
#if SKV4_CODEGEN_DEBUG_HEADS
        cb_push_back(SKV4_CB_MERGE_INDICES, 1);
#endif
#endif
    }
#if SKV4_CODEGEN_SELECT
    /*@CO_R_FRONTIER_E*/
#if SKV4_PROFILE
    skv_end_CO_R_FRONTIER=skv_tick(); skv_valid_CO_R_FRONTIER=true;
    skv_sum_CO_R_FRONTIER+=skv_end_CO_R_FRONTIER-tp_CO_R_FRONTIER;
    uint32_t tp_CO_R_MASK=skv_tick();
#endif
    /*@CO_R_MASK_B*/
    const uint32_t LIVE = get_arg_val<uint32_t>(SKV4_ARG_RT_LIVE);
    const uint32_t N = CHUNKS * 32;
    const uint32_t threshold_control = get_write_ptr(SKV4_CB_CONTROL);
    auto threshold = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(threshold_control);
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_THRESHOLDS_READY)), 1);
    // NCRISC handles the upper half while BRISC handles the lower half.  The
    // two ordered tag lists occupy fixed, non-overlapping regions in CB22;
    // BRISC joins the local count before publishing the existing one-packet
    // worker count to root0.
    const uint32_t scratch = get_write_ptr(SKV4_CB_SELECTOR_SCRATCH);
    auto out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    auto upper_pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(SKV4_CB_UPPER_INTERVALS));
    const auto upper0 = upper_pairs;
    const auto upper1 = upper_pairs + N / 2;
    const auto upper2 = upper_pairs + N;
    const auto upper3 = upper_pairs + 3 * N / 2;
    const uint16_t t0 = coarse_ordered_bf16(threshold[0]);
    const uint16_t t1 = coarse_ordered_bf16(threshold[8]);
    const uint16_t t2 = coarse_ordered_bf16(threshold[16]);
    const uint16_t t3 = coarse_ordered_bf16(threshold[24]);
    const uint32_t first = first_chunk * 32;
    const uint32_t limit = first >= LIVE ? 0 :
        (first + N < LIVE ? N : LIVE - first);
    const uint32_t split = N / 2;
    const uint32_t rear_base = 16 + split + 3;
    uint32_t count = 0;
    uint32_t pos = split;
    for (; pos + 1 < limit; pos += 2) {
        const uint32_t pair = pos >> 1;
        const uint32_t v0 = upper0[pair];
        const uint32_t v1 = upper1[pair];
        const uint32_t v2 = upper2[pair];
        const uint32_t v3 = upper3[pair];
        const uint32_t mask0 =
            (static_cast<uint32_t>((v0 & 0xffffu) >= t0) << 0) |
            (static_cast<uint32_t>((v1 & 0xffffu) >= t1) << 1) |
            (static_cast<uint32_t>((v2 & 0xffffu) >= t2) << 2) |
            (static_cast<uint32_t>((v3 & 0xffffu) >= t3) << 3);
        const uint32_t mask1 =
            (static_cast<uint32_t>((v0 >> 16) >= t0) << 0) |
            (static_cast<uint32_t>((v1 >> 16) >= t1) << 1) |
            (static_cast<uint32_t>((v2 >> 16) >= t2) << 2) |
            (static_cast<uint32_t>((v3 >> 16) >= t3) << 3);
        if (mask0) out[rear_base + count++] = (first + pos) | (mask0 << 28);
        if (mask1) out[rear_base + count++] = (first + pos + 1) | (mask1 << 28);
    }
    if (pos < limit) {
        const uint32_t pair = pos >> 1;
        const uint32_t mask =
            (static_cast<uint32_t>((upper0[pair] & 0xffffu) >= t0) << 0) |
            (static_cast<uint32_t>((upper1[pair] & 0xffffu) >= t1) << 1) |
            (static_cast<uint32_t>((upper2[pair] & 0xffffu) >= t2) << 2) |
            (static_cast<uint32_t>((upper3[pair] & 0xffffu) >= t3) << 3);
        if (mask) out[rear_base + count++] = (first + pos) | (mask << 28);
    }
    out[2] = count;
    __sync_synchronize();
    noc_semaphore_inc(get_noc_addr(get_semaphore(SKV4_SEM_MASK_REAR_READY)), 1);
    noc_async_atomic_barrier();
    /*@CO_R_MASK_E*/
#if SKV4_PROFILE
    skv_end_CO_R_MASK=skv_tick(); skv_valid_CO_R_MASK=true;
    skv_sum_CO_R_MASK+=skv_end_CO_R_MASK-tp_CO_R_MASK;
    DeviceTimestampedData("SKVA_CO_R_METADATA",skv_sum_CO_R_METADATA);
    DeviceTimestampedData("SKVA_CO_R_QUERY",skv_sum_CO_R_QUERY);
    DeviceTimestampedData("SKVA_CO_R_UPPER_STREAM",skv_sum_CO_R_UPPER_STREAM);
    DeviceTimestampedData("SKVA_CO_R_FRONTIER",skv_sum_CO_R_FRONTIER);
    DeviceTimestampedData("SKVA_CO_R_MASK",skv_sum_CO_R_MASK);
    if(skv_valid_CO_R_METADATA) DeviceTimestampedData("SKVE_CO_R_METADATA",skv_end_CO_R_METADATA);
    if(skv_valid_CO_R_QUERY) DeviceTimestampedData("SKVE_CO_R_QUERY",skv_end_CO_R_QUERY);
    if(skv_valid_CO_R_UPPER_STREAM) DeviceTimestampedData("SKVE_CO_R_UPPER_STREAM",skv_end_CO_R_UPPER_STREAM);
    if(skv_valid_CO_R_FRONTIER) DeviceTimestampedData("SKVE_CO_R_FRONTIER",skv_end_CO_R_FRONTIER);
    if(skv_valid_CO_R_MASK) DeviceTimestampedData("SKVE_CO_R_MASK",skv_end_CO_R_MASK);
#endif
}
#endif
"""
        )
    )


def compute(*, select, debug_heads, profile):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if debug_heads and (not select):
        raise ValueError("debug head export requires selection")
    if profile and (not select):
        raise ValueError("Coarse endpoint profile requires selection")
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + coarse_defines("compute", select=select, debug_heads=debug_heads)
        + query_metadata_defines()
        + profile_defines(profile=profile, select=select, debug_heads=debug_heads)
        + (
            r"""#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
#include "tools/profiler/kernel_profiler.hpp"
#endif
#endif
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#if SKV4_CODEGEN_SELECT
#include "api/compute/experimental/topk_xl.h"
#else
#include "api/compute/reg_api.h"
#endif
#include "api/compute/pack.h"
#if !SKV4_CODEGEN_SELECT
#include "api/compute/experimental/pack_rows_to_addr.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#endif
#include "api/dataflow/circular_buffer.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/experimental/pack_rows_to_addr.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
// fine-ingest-meter-v1
inline __attribute__((always_inline)) uint32_t skv_tick() {
    asm volatile("" ::: "memory");
    uint32_t t=*reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory"); return t;
}
uint32_t skv_sum_CO_C_SCORE;
uint32_t skv_sum_CO_C_DECODE;
uint32_t skv_sum_CO_C_OPERAND_WAIT;
uint32_t skv_sum_CO_C_MATMUL;
uint32_t skv_sum_CO_C_INTERVAL;
uint32_t skv_sum_CO_C_PACK;
uint32_t skv_sum_CO_C_LOCAL_TOPK;
uint32_t skv_sum_CO_C_TREE;
#endif
#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void global_positions(uint32_t base) {
    #pragma GCC unroll 1
    for (uint32_t i = 0; i < 16; ++i) {
        sfpi::vUInt pos = sfpi::dst_reg[32 + i].mode<sfpi::DataLayout::U32>();
        sfpi::dst_reg[32 + i].mode<sfpi::DataLayout::U32>() = pos + base;
    }
}
}
#endif
inline void fine_sort_chunk(uint32_t dst, uint32_t active, bool ascending) {
    cb_wait_front(SKV4_CB_LOCAL_SCORE_TILES, 1);
    topk_xl_copy_tile_init(SKV4_CB_LOCAL_SCORE_TILES);
    topk_xl_copy_tile<512>(SKV4_CB_LOCAL_SCORE_TILES, dst, 0, active);
    cb_pop_front(SKV4_CB_LOCAL_SCORE_TILES, 1);
    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<512, 0>(dst);
    topk_xl_init<512, true>();
    topk_xl_local_sort<512>(dst, ascending);
#if SKV4_CODEGEN_DEBUG_HEADS
    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<512>(dst);
    topk_xl_separate_indices_row_major_advance_chunk_base<512>();
#endif
}
#if !SKV4_PROFILE
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/experimental/pack_rows_to_addr.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#endif
#endif
#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void clear_score() {
    #pragma GCC unroll 32
    for(uint32_t i=0;i<32;++i)
        sfpi::dst_reg[i].mode<sfpi::DataLayout::F32>()=sfpi::vFloat(0.0f);
}
inline sfpi::vUInt expand_native_eight(sfpi::vUInt x) {
    x = x & 0xffffu;
    x = (x | sfpi::shft(x, 8)) & 0x00ff00ffu;
    x = (x | sfpi::shft(x, 4)) & 0x0f0f0f0fu;
    x = (x | sfpi::shft(x, 2)) & 0x33333333u;
    return sfpi::shft(x, 1);
}
template <uint32_t PAGE, uint32_t OUTPUT>
inline void expand_native_page() {
    #pragma GCC unroll 4
    for (uint32_t row = 0; row < 4; ++row) {
        sfpi::vUInt packed = sfpi::dst_reg[(PAGE / 2) * 4 + row]
            .mode<sfpi::DataLayout::U32>();
        sfpi::vUInt word = expand_native_eight(
            sfpi::shft(packed, -int(16 * (PAGE % 2))));
        sfpi::dst_reg[OUTPUT * 32 + row]
            .mode<sfpi::DataLayout::U32>() = word;
    }
}
inline void coarse_scale_interval(
    uint32_t scale_address, uint32_t norm_address, uint32_t qnorm_address) {
    const auto scale_words = reinterpret_cast<const uint32_t*>(scale_address);
    const auto norms = reinterpret_cast<const uint16_t*>(norm_address);
    const auto qnorms = reinterpret_cast<const uint16_t*>(qnorm_address);
    // SFPI7.69 cannot predicate runtime32-bit immediates. Match the proven
    // two-halfword norm load below instead of selecting full FP32 bit words.
    const auto sums = qnorms + 2 * SKV4_QUERY_SUM_WORD_OFFSET;
    sfpi::vUInt sum_lo = uint32_t(sums[0]);
    sfpi::vUInt sum_hi = uint32_t(sums[1]);
    sfpi::vUInt qlo = uint32_t(qnorms[0]);
    sfpi::vUInt qhi = uint32_t(qnorms[1]);
    v_if (sfpi::vConstTileId >= 16) {
        sum_lo = uint32_t(sums[2]); sum_hi = uint32_t(sums[3]);
        qlo = uint32_t(qnorms[2]); qhi = uint32_t(qnorms[3]);
    } v_endif;
    v_if (sfpi::vConstTileId >= 32) {
        sum_lo = uint32_t(sums[4]); sum_hi = uint32_t(sums[5]);
        qlo = uint32_t(qnorms[4]); qhi = uint32_t(qnorms[5]);
    } v_endif;
    v_if (sfpi::vConstTileId >= 48) {
        sum_lo = uint32_t(sums[6]); sum_hi = uint32_t(sums[7]);
        qlo = uint32_t(qnorms[6]); qhi = uint32_t(qnorms[7]);
    } v_endif;
    const sfpi::vFloat qn = sfpi::as<sfpi::vFloat>(sfpi::shft(qhi, 16) | qlo);
    const sfpi::vFloat correction =
        sfpi::as<sfpi::vFloat>(sfpi::shft(sum_hi, 16) | sum_lo) * 3.0f;
    const sfpi::vFloat scale =
        sfpi::as<sfpi::vFloat>(sfpi::vUInt(scale_words[0]));
    // Only these four upper-row vectors are consumed by the selector.  The
    // remaining 28 rows are deliberately left stale instead of spending one
    // SFPU pass clearing data that never crosses the compute/writer ABI.
    #pragma GCC unroll 2
    for (uint32_t face = 0; face < 2; ++face) {
        // Even/odd columns share the same lane predicate. Select their u16
        // norms together, without predicated runtime-u32 SFPI immediates.
        sfpi::vUInt even = uint32_t(norms[face * 16]);
        sfpi::vUInt odd = uint32_t(norms[face * 16 + 1]);
        #pragma GCC unroll 1
        for (uint32_t column = 1; column < 8; ++column) {
            v_if ((sfpi::vConstTileId & 15) >= 2 * column) {
                even = uint32_t(norms[face * 16 + 2 * column]);
                odd = uint32_t(norms[face * 16 + 2 * column + 1]);
            } v_endif;
        }
        #pragma GCC unroll 2
        for (uint32_t parity = 0; parity < 2; ++parity) {
            sfpi::vUInt nw = even;
            if (parity) nw = odd;
            sfpi::vFloat norm = sfpi::as<sfpi::vFloat>(sfpi::shft(nw, 16));
            sfpi::vFloat margin = qn * norm;
            sfpi::vFloat centered =
                sfpi::dst_reg[face * 8 + parity] * 2.0f - correction;
            sfpi::vFloat value = centered * scale;
            sfpi::dst_reg[face * 8 + parity] = value - margin;
            sfpi::dst_reg[32 + face * 8 + parity] = value + margin;
        }
    }
}
}
#endif
inline void expand_eight() {
    cb_reserve_back(SKV4_CB_DECODED_KEYS, 8);
    tile_regs_acquire();
    copy_tile(SKV4_CB_PACKED_UPPER, 0, 0);
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        expand_native_page, (0, 1), 0, VectorMode::None));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        expand_native_page, (1, 2), 0, VectorMode::None));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        expand_native_page, (2, 3), 0, VectorMode::None));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        expand_native_page, (3, 4), 0, VectorMode::None));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        expand_native_page, (4, 5), 0, VectorMode::None));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        expand_native_page, (5, 6), 0, VectorMode::None));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        expand_native_page, (6, 7), 0, VectorMode::None));
    // Last page reads input vectors12..15 and overwrites only0..3. All
    // earlier input consumers have finished; eight outputs fit Dst0..7.
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        expand_native_page, (7, 0), 0, VectorMode::None));
    tile_regs_commit();
    tile_regs_wait();
    #pragma GCC unroll 1
    for (uint32_t page = 0; page < 8; ++page) {
        PACK({
            const uint32_t at = get_local_cb_interface(SKV4_CB_DECODED_KEYS).fifo_wr_ptr;
            pack_rows_to_addr((page + 1) & 7u, at + page * 36 + 4);
        });
    }
    tile_regs_release();
    cb_push_back(SKV4_CB_DECODED_KEYS, 8);
}
void kernel_main() {
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
    skv_sum_CO_C_SCORE=0;skv_sum_CO_C_DECODE=0;skv_sum_CO_C_OPERAND_WAIT=0;skv_sum_CO_C_MATMUL=0;skv_sum_CO_C_INTERVAL=0;skv_sum_CO_C_PACK=0;skv_sum_CO_C_LOCAL_TOPK=0;skv_sum_CO_C_TREE=0;
#endif
#endif
#if SKV4_CODEGEN_SELECT
    const uint32_t N = get_arg_val<uint32_t>(SKV4_ARG_RT_N);
#else
    constexpr uint32_t N = get_compile_time_arg_val(SKV4_ARG_CT_N);
#endif
    constexpr uint32_t PHASE = get_compile_time_arg_val(SKV4_ARG_CT_PHASE);
    compute_kernel_hw_startup(SKV4_CB_PACKED_UPPER,SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS);
    uint32_t table_base = 0;
    cb_wait_front(SKV4_CB_METADATA,1);
    UNPACK(mailbox_write(ckernel::ThreadId::MathThreadId,
        get_local_cb_interface(SKV4_CB_METADATA).fifo_rd_ptr << 4));
    MATH(table_base = mailbox_read(ckernel::ThreadId::UnpackThreadId));
    cb_wait_front(SKV4_CB_QUERY,8);
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
    uint32_t tp_CO_C_SCORE=skv_tick();
#endif
#endif
    if constexpr(PHASE>0) {
        // Unfused TopK-XL leaves SFPU index tracking enabled across programs.
        // Custom SFPI execute calls do not initialize this state themselves.
        MATH(SFPU_UNARY_INIT(unused));
        for(uint32_t first=0;first<N/32;first+=SKV4_COARSE_OPERAND_BATCH) {
            const uint32_t remaining=N/32-first;
            const uint32_t count=remaining<SKV4_COARSE_OPERAND_BATCH
                ?remaining:SKV4_COARSE_OPERAND_BATCH;
            cb_wait_front(SKV4_CB_PACKED_UPPER,1);
            // Decode a bounded batch before switching to matrix/score PACK.
            // Keep global chunk indices for scale/norm lookup and tail batches.
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
            uint32_t tp_CO_C_DECODE=skv_tick();
#endif
#endif
            copy_tile_to_dst_init_short_with_dt(SKV4_CB_DECODED_KEYS,SKV4_CB_PACKED_UPPER);
            pack_reconfig_data_format(SKV4_CB_PACKED_UPPER);
            pack_rows_to_addr_init(8);
            for(uint32_t local=0;local<count;++local) {
                cb_wait_front(SKV4_CB_PACKED_UPPER,1);
                expand_eight();
                cb_pop_front(SKV4_CB_PACKED_UPPER,1);
            }
            pack_rows_to_addr_uninit();
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
            skv_sum_CO_C_DECODE+=skv_tick()-tp_CO_C_DECODE;
#endif
#endif
            if constexpr(PHASE!=3) {
                if constexpr(PHASE==2) {
                    // Matrix maps Q -> SrcB, K -> SrcA, unlike tile-copy.
                    reconfig_full_operand<SrcOrder::Reverse>(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS);
                    matmul_init(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS,1);
                    pack_reconfig_data_format(SKV4_CB_LOWER_SCORE_TILES);
                    pack_init(SKV4_CB_LOWER_SCORE_TILES);
                }
                for(uint32_t local=0;local<count;++local) {
                    const uint32_t chunk=first+local;
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
                    uint32_t tp_CO_C_OPERAND_WAIT=skv_tick();
#endif
#endif
                    cb_wait_front(SKV4_CB_DECODED_KEYS,8);
                    if constexpr(PHASE==2) {
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
                        skv_sum_CO_C_OPERAND_WAIT+=skv_tick()-tp_CO_C_OPERAND_WAIT;
#endif
#endif
                        cb_reserve_back(SKV4_CB_LOWER_SCORE_TILES,1);
                        cb_reserve_back(SKV4_CB_UPPER_SCORE_TILES,1);
                        tile_regs_acquire();
                        if(chunk==0) {
                            // TopK-XL's first FPU output needs explicit clearing.
                            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                                DST_ACCUM_MODE,clear_score,0,VectorMode::None));
                            reconfig_full_operand<SrcOrder::Reverse>(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS);
                        }
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
                        uint32_t tp_CO_C_MATMUL=skv_tick();
#endif
#endif
                        for(uint32_t kt=0;kt<8;++kt) matmul_tiles(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS,kt,kt,0);
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
                        skv_sum_CO_C_MATMUL+=skv_tick()-tp_CO_C_MATMUL;
                        uint32_t tp_CO_C_INTERVAL=skv_tick();
#endif
#endif
                        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                            DST_ACCUM_MODE,coarse_scale_interval,0,VectorMode::None,
                            table_base + chunk * 64,
                            table_base + N * 2 + chunk * 64,
                            table_base + N * 4));
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
                        skv_sum_CO_C_INTERVAL+=skv_tick()-tp_CO_C_INTERVAL;
                        uint32_t tp_CO_C_PACK=skv_tick();
#endif
#endif
                        tile_regs_commit(); tile_regs_wait();
                        pack_tile(0,SKV4_CB_LOWER_SCORE_TILES);
                        pack_tile(1,SKV4_CB_UPPER_SCORE_TILES);
                        tile_regs_release(); cb_push_back(SKV4_CB_LOWER_SCORE_TILES,1); cb_push_back(SKV4_CB_UPPER_SCORE_TILES,1);
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
                        skv_sum_CO_C_PACK+=skv_tick()-tp_CO_C_PACK;
#endif
#endif
                    }
                    cb_pop_front(SKV4_CB_DECODED_KEYS,8);
                }
            }
        }
    }
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
    skv_sum_CO_C_SCORE+=skv_tick()-tp_CO_C_SCORE;
#endif
#endif
    cb_pop_front(SKV4_CB_QUERY,8); cb_pop_front(SKV4_CB_METADATA,1);
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DEBUG_HEADS
#if SKV4_PROFILE
    uint32_t tp_CO_C_LOCAL_TOPK=skv_tick();
#endif
#endif
    // The FPU scorer has released Dst and all score CBs. Same cores now sort
    // one head's complete cohort row; no additional program boundary.
    const uint32_t chunks = get_arg_val<uint32_t>(SKV4_ARG_RT_CHUNKS);
    const uint32_t tail = get_arg_val<uint32_t>(SKV4_ARG_RT_TAIL);
    const uint32_t cohort = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT);
    const uint32_t cohorts = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORTS);
    pack_reconfig_data_format(SKV4_CB_FRONTIER_VALUES);
    tile_regs_acquire();
#if SKV4_CODEGEN_DEBUG_HEADS
#if SKV4_PROFILE
    uint32_t tp_CO_C_LOCAL_TOPK=skv_tick();
#endif
    topk_xl_separate_indices_row_major_init_static<0, 0>();
#endif
    fine_sort_chunk(0, chunks == 1 ? tail : 512, false);
#if SKV4_CODEGEN_DEBUG_HEADS
    if (chunks == 1) {
        topk_xl_init<512, false>();
        topk_xl_rebuild<512, false>(0, false);
#else
    for (uint32_t chunk = 1; chunk < chunks; ++chunk) {
        fine_sort_chunk(1, chunk + 1 == chunks ? tail : 512, true);
        topk_xl_init<512, true>();
        topk_xl_merge<512, true>(0);
        topk_xl_rebuild<512, true>(0, false);
#endif
#endif
    }
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_DEBUG_HEADS
    for (uint32_t chunk = 1; chunk < chunks; ++chunk) {
        fine_sort_chunk(2, chunk + 1 == chunks ? tail : 512, true);
        topk_xl_init<512, false>();
        topk_xl_merge<512, false>(0);
        topk_xl_rebuild<512, false>(0, false);
    }
    MATH(SFPU_UNARY_INIT(unused));
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, global_positions, 0,
        VectorMode::None, get_arg_val<uint32_t>(SKV4_ARG_RT_BASE)));
#endif
#if SKV4_PROFILE
    skv_sum_CO_C_LOCAL_TOPK+=skv_tick()-tp_CO_C_LOCAL_TOPK;
    uint32_t tp_CO_C_TREE=skv_tick();
#endif
    for (uint32_t stride = 1; stride < cohorts; stride *= 2) {
        if (cohort % (2 * stride)) break;
        if (cohort + stride >= cohorts) continue;
#if SKV4_CODEGEN_DEBUG_HEADS
        // Keep our own frontier in Dst. Only the child's paired native packet
        // crosses NoC; no parent pack/reimport at intermediate levels.
#endif
        cb_wait_front(SKV4_CB_MERGE_VALUES, 1);
        topk_xl_copy_tile_init(SKV4_CB_MERGE_VALUES);
#if SKV4_CODEGEN_DEBUG_HEADS
        topk_xl_copy_tile<512>(SKV4_CB_MERGE_VALUES, 2, 0, 512);
#else
        topk_xl_copy_tile<512>(SKV4_CB_MERGE_VALUES, 1, 0, 512);
#endif
        cb_pop_front(SKV4_CB_MERGE_VALUES, 1);
#if SKV4_CODEGEN_DEBUG_HEADS
        cb_wait_front(SKV4_CB_MERGE_INDICES, 1);
        topk_xl_copy_tile_init(SKV4_CB_MERGE_INDICES);
        topk_xl_copy_tile<512>(SKV4_CB_MERGE_INDICES, 3, 0, 512);
        cb_pop_front(SKV4_CB_MERGE_INDICES, 1);
        topk_xl_init<512, false>();
        topk_xl_rebuild<512, false>(2, true);
        topk_xl_merge<512, false>(0);
        topk_xl_rebuild<512, false>(0, false);
#else
        topk_xl_init<512, true>();
        topk_xl_rebuild<512, true>(1, true);
        topk_xl_merge<512, true>(0);
        topk_xl_rebuild<512, true>(0, false);
#endif
    }
#if SKV4_PROFILE
    skv_sum_CO_C_TREE+=skv_tick()-tp_CO_C_TREE;
#endif
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(SKV4_CB_FRONTIER_VALUES, 1);
    pack_reconfig_data_format(SKV4_CB_FRONTIER_VALUES);
    pack_tile(0, SKV4_CB_FRONTIER_VALUES);
    cb_push_back(SKV4_CB_FRONTIER_VALUES, 1);
#if SKV4_CODEGEN_DEBUG_HEADS
    cb_reserve_back(SKV4_CB_FRONTIER_INDICES, 1);
    pack_reconfig_data_format(SKV4_CB_FRONTIER_INDICES);
    pack_tile(1, SKV4_CB_FRONTIER_INDICES);
    cb_push_back(SKV4_CB_FRONTIER_INDICES, 1);
#endif
    tile_regs_release();
#if SKV4_PROFILE
    DeviceTimestampedData("SKVA_CO_C_SCORE",skv_sum_CO_C_SCORE);
    DeviceTimestampedData("SKVA_CO_C_DECODE",skv_sum_CO_C_DECODE);
    DeviceTimestampedData("SKVA_CO_C_OPERAND_WAIT",skv_sum_CO_C_OPERAND_WAIT);
    DeviceTimestampedData("SKVA_CO_C_MATMUL",skv_sum_CO_C_MATMUL);
    DeviceTimestampedData("SKVA_CO_C_INTERVAL",skv_sum_CO_C_INTERVAL);
    DeviceTimestampedData("SKVA_CO_C_PACK",skv_sum_CO_C_PACK);
    DeviceTimestampedData("SKVA_CO_C_LOCAL_TOPK",skv_sum_CO_C_LOCAL_TOPK);
    DeviceTimestampedData("SKVA_CO_C_TREE",skv_sum_CO_C_TREE);
#endif
}
#endif
"""
        )
    )


def writer(*, select, debug_heads, profile):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if debug_heads and (not select):
        raise ValueError("debug head export requires selection")
    if profile and (not select):
        raise ValueError("Coarse endpoint profile requires selection")
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + coarse_defines("writer", select=select, debug_heads=debug_heads)
        + profile_defines(profile=profile, select=select, debug_heads=debug_heads)
        + (
            r"""#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
#include "tools/profiler/kernel_profiler.hpp"
#endif
#endif
#include "api/dataflow/dataflow_api.h"
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
// fine-ingest-meter-v1
inline __attribute__((always_inline)) uint32_t skv_tick() {
    asm volatile("" ::: "memory");
    uint32_t t=*reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory"); return t;
}
uint32_t skv_sum_CO_W_INTERVAL_COPY; uint32_t skv_end_CO_W_INTERVAL_COPY; bool skv_valid_CO_W_INTERVAL_COPY;
uint32_t skv_sum_CO_W_SCORE_ROUTE; uint32_t skv_end_CO_W_SCORE_ROUTE; bool skv_valid_CO_W_SCORE_ROUTE;
uint32_t skv_sum_CO_W_FRONTIER; uint32_t skv_end_CO_W_FRONTIER; bool skv_valid_CO_W_FRONTIER;
uint32_t skv_sum_CO_W_THRESHOLD; uint32_t skv_end_CO_W_THRESHOLD; bool skv_valid_CO_W_THRESHOLD;
uint32_t skv_sum_CO_W_MASK; uint32_t skv_end_CO_W_MASK; bool skv_valid_CO_W_MASK;
uint32_t skv_sum_CO_W_PREFIX; uint32_t skv_end_CO_W_PREFIX; bool skv_valid_CO_W_PREFIX;
uint32_t skv_sum_CO_W_EXPORT; uint32_t skv_end_CO_W_EXPORT; bool skv_valid_CO_W_EXPORT;
uint32_t skv_sum_CO_W_REPLAY_CLEANUP; uint32_t skv_end_CO_W_REPLAY_CLEANUP; bool skv_valid_CO_W_REPLAY_CLEANUP;
#endif
FORCE_INLINE uint32_t coarse_ordered_bf16_pair(uint32_t pair) {
    // Two independent halfword XOR masks, with unsigned wraparound.
    // Preserve the scalar raw-bit order, including signed zero/NaN/inf.
    const uint32_t signs = (pair >> 15) & 0x00010001u;
    return pair ^ (0x80008000u | ((signs << 16) - signs));
}
FORCE_INLINE uint16_t ordered_bf16(uint16_t bits) {
    return bits ^ (0x8000u | (0u - (bits >> 15)));
}
#endif
void kernel_main() {
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
    skv_sum_CO_W_INTERVAL_COPY=0;skv_sum_CO_W_SCORE_ROUTE=0;skv_sum_CO_W_FRONTIER=0;skv_sum_CO_W_THRESHOLD=0;skv_sum_CO_W_MASK=0;skv_sum_CO_W_PREFIX=0;skv_sum_CO_W_EXPORT=0;skv_sum_CO_W_REPLAY_CLEANUP=0;skv_valid_CO_W_INTERVAL_COPY=false;skv_valid_CO_W_SCORE_ROUTE=false;skv_valid_CO_W_FRONTIER=false;skv_valid_CO_W_THRESHOLD=false;skv_valid_CO_W_MASK=false;skv_valid_CO_W_PREFIX=false;skv_valid_CO_W_EXPORT=false;skv_valid_CO_W_REPLAY_CLEANUP=false;
#endif
    const uint32_t N = get_arg_val<uint32_t>(SKV4_ARG_RT_N);
    const uint32_t TOTAL = get_arg_val<uint32_t>(SKV4_ARG_RT_TOTAL);
    const uint32_t LIVE = get_arg_val<uint32_t>(SKV4_ARG_RT_LIVE);
    const uint32_t P = get_arg_val<uint32_t>(SKV4_ARG_RT_P);
    constexpr auto oa = TensorAccessorArgs<SKV4_ARG_CT_ACCESSORS>();
#if SKV4_CODEGEN_DEBUG_HEADS
    constexpr auto ha = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
#endif
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(SKV4_ARG_RT_OUTPUT), 4096);
#if SKV4_CODEGEN_DEBUG_HEADS
    const auto heads = TensorAccessor(ha, get_arg_val<uint32_t>(SKV4_ARG_RT_HEADS), P * 4);
#endif
    const uint32_t first_chunk = get_arg_val<uint32_t>(SKV4_ARG_RT_FIRST_CHUNK);
    const uint32_t base = get_arg_val<uint32_t>(SKV4_ARG_RT_BASE);
    const uint32_t cohort = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT);
    const uint32_t head = get_arg_val<uint32_t>(SKV4_ARG_RT_HEAD);
    const uint32_t worker = get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER);
    const uint32_t workers = get_arg_val<uint32_t>(SKV4_ARG_RT_WORKERS);
    const uint32_t root_x = get_arg_val<uint32_t>(SKV4_ARG_RT_ROOT_X);
    const uint32_t root_y = get_arg_val<uint32_t>(SKV4_ARG_RT_ROOT_Y);
    const uint32_t parent_x = get_arg_val<uint32_t>(SKV4_ARG_RT_PARENT_X);
    const uint32_t parent_y = get_arg_val<uint32_t>(SKV4_ARG_RT_PARENT_Y);
    const uint32_t parent_level = get_arg_val<uint32_t>(SKV4_ARG_RT_PARENT_LEVEL);
    const uint32_t lower_addr = get_write_ptr(SKV4_CB_LOWER_INTERVALS);
    const uint32_t upper_addr = get_write_ptr(SKV4_CB_UPPER_INTERVALS);
    auto upper = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(upper_addr);
    auto lower_pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lower_addr);
    auto upper_pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(upper_addr);
    // Preserve lower BF16 rows for TopK. Preorder upper pairs during the
    // overlapped streaming copy, outside the exposed threshold-mask tail.
#if SKV4_PROFILE
    uint32_t tp_CO_W_INTERVAL_COPY=skv_tick();
#endif
    for (uint32_t chunk = 0; chunk < N / 32; ++chunk) {
#else
    constexpr uint32_t CHUNKS = get_compile_time_arg_val(SKV4_ARG_CT_CHUNKS);
    constexpr auto oa = TensorAccessorArgs<SKV4_ARG_CT_ACCESSORS>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(SKV4_ARG_RT_OUTPUT), 2048);
    const uint32_t first_chunk = get_arg_val<uint32_t>(SKV4_ARG_RT_FIRST_CHUNK);
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
#endif
        cb_wait_front(SKV4_CB_LOWER_SCORE_TILES, 1);
        cb_wait_front(SKV4_CB_UPPER_SCORE_TILES, 1);
#if SKV4_CODEGEN_SELECT
        auto lo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_LOWER_SCORE_TILES));
        auto hi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_UPPER_SCORE_TILES));
        for (uint32_t h = 0; h < 4; ++h)
            for (uint32_t j = 0; j < 32; j += 2) {
                const uint32_t pos = chunk * 32 + j;
                const uint32_t lane = h * 16 + (j & 15u) + (j / 16) * 256;
                const uint32_t dst = (h * N + pos) >> 1;
                const uint32_t global = first_chunk * 32 + pos;
                uint32_t lo_pair = lo[lane >> 1];
                uint32_t hi_pair = hi[lane >> 1];
                if (global >= LIVE) {
                    lo_pair = 0xff80ff80u;
                    hi_pair = 0xff80ff80u;
                } else if (global + 1 >= LIVE) {
                    lo_pair = (lo_pair & 0x0000ffffu) | 0xff800000u;
                    hi_pair = (hi_pair & 0x0000ffffu) | 0xff800000u;
                }
                lower_pairs[dst] = lo_pair;
                upper_pairs[dst] = coarse_ordered_bf16_pair(hi_pair);
            }
#else
        noc_async_write(get_read_ptr(SKV4_CB_LOWER_SCORE_TILES),
                        get_noc_addr((first_chunk + chunk) * 2, output), 2048);
        noc_async_write(get_read_ptr(SKV4_CB_UPPER_SCORE_TILES),
                        get_noc_addr((first_chunk + chunk) * 2 + 1, output), 2048);
        noc_async_write_barrier();
#endif
        cb_pop_front(SKV4_CB_LOWER_SCORE_TILES, 1);
        cb_pop_front(SKV4_CB_UPPER_SCORE_TILES, 1);
    }
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE
    skv_end_CO_W_INTERVAL_COPY=skv_tick(); skv_valid_CO_W_INTERVAL_COPY=true;
    skv_sum_CO_W_INTERVAL_COPY+=skv_end_CO_W_INTERVAL_COPY-tp_CO_W_INTERVAL_COPY;
    uint32_t tp_CO_W_SCORE_ROUTE=skv_tick();
#endif
    // Four large spans per worker feed the four head owners in this cohort.
    for (uint32_t h = 0; h < 4; ++h) {
        const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS + h * 2);
        const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS_Y + h * 2);
        uint32_t src = lower_addr + h * N * 2;
        uint32_t offset = first_chunk * 32 - base;
        uint32_t remaining = N;
        while (remaining) {
            const uint32_t in_page = offset & 511u;
            const uint32_t take = remaining < 512 - in_page
                ? remaining : 512 - in_page;
            const uint32_t dst = get_write_ptr(SKV4_CB_LOCAL_SCORE_TILES)
                + (offset / 512) * 2048 + in_page * 2;
            noc_async_write(src, get_noc_addr(x, y, dst), take * 2);
            src += take * 2;
            offset += take;
            remaining -= take;
#endif
        }
#if SKV4_CODEGEN_SELECT
    }
    noc_async_write_barrier();
    for (uint32_t h = 0; h < 4; ++h)
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS + h * 2),
            get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS_Y + h * 2), get_semaphore(SKV4_SEM_COHORT_SCORES_READY)), 1);
    /*@COARSE_FRONTIER_B*/
#if SKV4_PROFILE
    skv_end_CO_W_SCORE_ROUTE=skv_tick(); skv_valid_CO_W_SCORE_ROUTE=true;
    skv_sum_CO_W_SCORE_ROUTE+=skv_end_CO_W_SCORE_ROUTE-tp_CO_W_SCORE_ROUTE;
    uint32_t tp_CO_W_FRONTIER=skv_tick();
#endif
    cb_wait_front(SKV4_CB_FRONTIER_VALUES, 1);
#if SKV4_CODEGEN_DEBUG_HEADS
    cb_wait_front(SKV4_CB_FRONTIER_INDICES, 1);
#endif
    const uint32_t scratch = get_write_ptr(SKV4_CB_SELECTOR_SCRATCH);
    if (cohort) {
#if SKV4_CODEGEN_DEBUG_HEADS
        noc_async_read(get_noc_addr(get_read_ptr(SKV4_CB_FRONTIER_VALUES)), scratch, 1024);
        noc_async_read(get_noc_addr(get_read_ptr(SKV4_CB_FRONTIER_INDICES)), scratch + 1024, 2048);
        noc_async_read_barrier();
        noc_async_write(scratch, get_noc_addr(parent_x, parent_y,
            get_write_ptr(SKV4_CB_PEER_FRONTIERS) + parent_level * 3072), 3072);
#else
        // One bulk packet carries the native fused K512 frontier.  No local
        // self-copy and no row-major UInt32 companion payload are needed.
        noc_async_write(get_read_ptr(SKV4_CB_FRONTIER_VALUES), get_noc_addr(parent_x, parent_y,
            get_write_ptr(SKV4_CB_PEER_FRONTIERS) + parent_level * 2048), 2048);
#endif
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(parent_x, parent_y,
            get_semaphore(SKV4_SEM_FRONTIER_LEVEL_0 + parent_level)), 1);
    } else {
#if SKV4_CODEGEN_DEBUG_HEADS
        auto native = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_read_ptr(SKV4_CB_FRONTIER_INDICES));
        auto positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t debug_rank = 0; debug_rank < P; ++debug_rank) {
            const uint32_t debug_lane =
                (debug_rank >> 5) + ((debug_rank & 31u) << 4);
            positions[debug_rank] = native[debug_lane];
        }
        noc_async_write(scratch, get_noc_addr(head, heads), P * 4);
#else
        // K512's fused result retains the native 32x16 rank matrix.  The high
        // half is the BF16 score; the low half is only an internal tie-break.
#endif
        const uint32_t rank = P - 1;
        const uint32_t lane = (rank >> 5) + ((rank & 31u) << 4);
#if SKV4_CODEGEN_DEBUG_HEADS
        auto scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
#else
        auto fused = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
#endif
            get_read_ptr(SKV4_CB_FRONTIER_VALUES));
        const uint32_t control = get_write_ptr(SKV4_CB_CONTROL);
        auto local = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(control);
#if SKV4_CODEGEN_DEBUG_HEADS
        local[head * 8] = scores[lane];
#else
        local[head * 8] = fused[lane] >> 16;
#endif
        noc_async_write(control + head * 16,
            get_noc_addr(root_x, root_y, control + head * 16), 16);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_THRESHOLD_HEADS_READY)), 1);
    }
    cb_pop_front(SKV4_CB_FRONTIER_VALUES, 1);
#if SKV4_CODEGEN_DEBUG_HEADS
    cb_pop_front(SKV4_CB_FRONTIER_INDICES, 1);
#endif
    /*@COARSE_FRONTIER_E*/
#if SKV4_PROFILE
    skv_end_CO_W_FRONTIER=skv_tick(); skv_valid_CO_W_FRONTIER=true;
    skv_sum_CO_W_FRONTIER+=skv_end_CO_W_FRONTIER-tp_CO_W_FRONTIER;
    uint32_t tp_CO_W_THRESHOLD=skv_tick();
#endif
    const uint32_t ids_control = get_write_ptr(SKV4_CB_WORKER_COUNTS);
    const uint32_t threshold_control = get_write_ptr(SKV4_CB_CONTROL);
    auto threshold = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(threshold_control);
    if (worker == 0) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_THRESHOLD_HEADS_READY)), 4);
        for (uint32_t w = 0; w < workers; ++w) {
            const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER_PEERS + w * 2);
            const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER_PEERS_Y + w * 2);
            noc_async_write(threshold_control,
                get_noc_addr(x, y, threshold_control), 64);
        }
        noc_async_write_barrier();
        for (uint32_t w = 0; w < workers; ++w)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER_PEERS + w * 2),
                get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER_PEERS_Y + w * 2), get_semaphore(SKV4_SEM_THRESHOLDS_READY)), 1);
    }
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_THRESHOLDS_READY)), 1);
#if SKV4_PROFILE
    skv_end_CO_W_THRESHOLD=skv_tick(); skv_valid_CO_W_THRESHOLD=true;
    skv_sum_CO_W_THRESHOLD+=skv_end_CO_W_THRESHOLD-tp_CO_W_THRESHOLD;
    uint32_t tp_CO_W_MASK=skv_tick();
#endif
    // Tags are ascending within each contiguous source range and carry the
    // four-head eligibility mask in their high nibble. BRISC owns the lower
    // half; NCRISC concurrently owns the upper half in READER_MASK_TAIL.
    auto out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const auto upper0 = upper_pairs;
    const auto upper1 = upper_pairs + N / 2;
    const auto upper2 = upper_pairs + N;
    const auto upper3 = upper_pairs + 3 * N / 2;
    const uint16_t t0 = ordered_bf16(threshold[0]);
    const uint16_t t1 = ordered_bf16(threshold[8]);
    const uint16_t t2 = ordered_bf16(threshold[16]);
    const uint16_t t3 = ordered_bf16(threshold[24]);
    const uint32_t first = first_chunk * 32;
    const uint32_t limit = first >= LIVE ? 0 :
        (first + N < LIVE ? N : LIVE - first);
    const uint32_t split = limit < N / 2 ? limit : N / 2;
    uint32_t count = 0;
    uint32_t pos = 0;
    for (; pos + 1 < split; pos += 2) {
        const uint32_t pair = pos >> 1;
        const uint32_t v0 = upper0[pair];
        const uint32_t v1 = upper1[pair];
        const uint32_t v2 = upper2[pair];
        const uint32_t v3 = upper3[pair];
        const uint32_t mask0 =
            (static_cast<uint32_t>((v0 & 0xffffu) >= t0) << 0) |
            (static_cast<uint32_t>((v1 & 0xffffu) >= t1) << 1) |
            (static_cast<uint32_t>((v2 & 0xffffu) >= t2) << 2) |
            (static_cast<uint32_t>((v3 & 0xffffu) >= t3) << 3);
        const uint32_t mask1 =
            (static_cast<uint32_t>((v0 >> 16) >= t0) << 0) |
            (static_cast<uint32_t>((v1 >> 16) >= t1) << 1) |
            (static_cast<uint32_t>((v2 >> 16) >= t2) << 2) |
            (static_cast<uint32_t>((v3 >> 16) >= t3) << 3);
        if (mask0) out[16 + count++] = (first + pos) | (mask0 << 28);
        if (mask1) out[16 + count++] = (first + pos + 1) | (mask1 << 28);
    }
    if (pos < split) {
        const uint32_t pair = pos >> 1;
        const uint32_t mask =
            (static_cast<uint32_t>((upper0[pair] & 0xffffu) >= t0) << 0) |
            (static_cast<uint32_t>((upper1[pair] & 0xffffu) >= t1) << 1) |
            (static_cast<uint32_t>((upper2[pair] & 0xffffu) >= t2) << 2) |
            (static_cast<uint32_t>((upper3[pair] & 0xffffu) >= t3) << 3);
        if (mask) out[16 + count++] = (first + pos) | (mask << 28);
    }
    const uint32_t front_count = count;
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_MASK_REAR_READY)), 1);
    __sync_synchronize();
    const uint32_t rear_count = out[2];
    count = front_count + rear_count;
#if SKV4_PROFILE
    skv_end_CO_W_MASK=skv_tick(); skv_valid_CO_W_MASK=true;
    skv_sum_CO_W_MASK+=skv_end_CO_W_MASK-tp_CO_W_MASK;
    uint32_t tp_CO_W_PREFIX=skv_tick();
#endif
    // One 64-byte count packet per worker. Root0 computes globally sorted
    // destination offsets because worker ranges are source ordered.
    auto count_slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        ids_control + worker * 64);
    for (uint32_t i = 0; i < 16; ++i) count_slot[i] = 0;
    count_slot[0] = count;
    noc_async_write(reinterpret_cast<uint32_t>(count_slot),
        get_noc_addr(root_x, root_y, ids_control + worker * 64), 64);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_WORKER_COUNTS_READY)), 1);
    if (worker == 0) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_WORKER_COUNTS_READY)),
            workers);
        uint32_t total = 0;
        for (uint32_t w = 0; w < workers; ++w) {
            auto slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                ids_control + w * 64);
            slot[1] = total;
            total += slot[0];
        }
        for (uint32_t w = 0; w < workers; ++w) {
            const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER_PEERS + w * 2);
            const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER_PEERS_Y + w * 2);
            noc_async_write(ids_control + w * 64,
                get_noc_addr(x, y, ids_control + w * 64), 64);
        }
        for (uint32_t i = 0; i < 16; ++i) out[i] = 0xffffffffu;
        out[0] = total;
        out[1] = 0;
        noc_async_write(scratch, get_noc_addr(0, output), 64);
        noc_async_write_barrier();
        for (uint32_t w = 0; w < workers; ++w)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER_PEERS + w * 2),
                get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER_PEERS_Y + w * 2), get_semaphore(SKV4_SEM_PREFIX_OFFSETS_READY)), 1);
    }
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_PREFIX_OFFSETS_READY)), 1);
#if SKV4_PROFILE
    skv_end_CO_W_PREFIX=skv_tick(); skv_valid_CO_W_PREFIX=true;
    skv_sum_CO_W_PREFIX+=skv_end_CO_W_PREFIX-tp_CO_W_PREFIX;
    uint32_t tp_CO_W_EXPORT=skv_tick();
#endif
    const uint32_t prefix = count_slot[1];
    const uint32_t front_shift = prefix & 3u;
    if (front_shift)
        for (uint32_t i = front_count; i > 0; --i)
            out[16 + front_shift + i - 1] = out[16 + i - 1];
    uint32_t copied = 0;
    uint32_t logical_start = (16 + prefix) * 4;
    uint32_t source = scratch + (16 + front_shift) * 4;
    while (copied < front_count * 4) {
        const uint32_t logical = logical_start + copied;
        const uint32_t page_left = 4096 - logical % 4096;
        const uint32_t bytes = front_count * 4 - copied < page_left
            ? front_count * 4 - copied : page_left;
        noc_async_write(source + copied,
            get_noc_addr(logical / 4096, output) + logical % 4096, bytes);
        copied += bytes;
    }
    const uint32_t rear_base = 16 + N / 2 + 3;
    const uint32_t rear_logical_index = 16 + prefix + front_count;
    const uint32_t rear_shift = (rear_logical_index - rear_base) & 3u;
    if (rear_shift)
        for (uint32_t i = rear_count; i > 0; --i)
            out[rear_base + rear_shift + i - 1] = out[rear_base + i - 1];
    copied = 0;
    logical_start = rear_logical_index * 4;
    source = scratch + (rear_base + rear_shift) * 4;
    while (copied < rear_count * 4) {
        const uint32_t logical = logical_start + copied;
        const uint32_t page_left = 4096 - logical % 4096;
        const uint32_t bytes = rear_count * 4 - copied < page_left
            ? rear_count * 4 - copied : page_left;
        noc_async_write(source + copied,
            get_noc_addr(logical / 4096, output) + logical % 4096, bytes);
        copied += bytes;
    }
    noc_async_write_barrier();
    // Remote semaphore increments are non-posted NoC atomics.  A program may
    // otherwise report completion while they are still in flight and let the
    // following Fine/Stage1.5 program reuse the same L1 semaphore addresses.
#if SKV4_PROFILE
    uint32_t tp_CO_W_REPLAY_CLEANUP=skv_tick();
#endif
    /*@CO_W_REPLAY_CLEANUP_B*/
    noc_async_atomic_barrier();
    // Semaphores are persistent L1 state; ProgramDescriptor initial values
    // are not replay initialization.  sem12 is the final all-worker release,
    // so every local Stage-I dependency has completed before this reset.
    for (uint32_t sem = SKV4_SEM_COHORT_SCORES_READY; sem <= SKV4_SEM_MASK_REAR_READY; ++sem)
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem)), 0);
    /*@CO_W_REPLAY_CLEANUP_E*/
#if SKV4_PROFILE
    skv_end_CO_W_REPLAY_CLEANUP=skv_tick(); skv_valid_CO_W_REPLAY_CLEANUP=true;
    skv_sum_CO_W_REPLAY_CLEANUP+=skv_end_CO_W_REPLAY_CLEANUP-tp_CO_W_REPLAY_CLEANUP;
    skv_end_CO_W_EXPORT=skv_tick(); skv_valid_CO_W_EXPORT=true;
    skv_sum_CO_W_EXPORT+=skv_end_CO_W_EXPORT-tp_CO_W_EXPORT;
    DeviceTimestampedData("SKVA_CO_W_INTERVAL_COPY",skv_sum_CO_W_INTERVAL_COPY);
    DeviceTimestampedData("SKVA_CO_W_SCORE_ROUTE",skv_sum_CO_W_SCORE_ROUTE);
    DeviceTimestampedData("SKVA_CO_W_FRONTIER",skv_sum_CO_W_FRONTIER);
    DeviceTimestampedData("SKVA_CO_W_THRESHOLD",skv_sum_CO_W_THRESHOLD);
    DeviceTimestampedData("SKVA_CO_W_MASK",skv_sum_CO_W_MASK);
    DeviceTimestampedData("SKVA_CO_W_PREFIX",skv_sum_CO_W_PREFIX);
    DeviceTimestampedData("SKVA_CO_W_EXPORT",skv_sum_CO_W_EXPORT);
    DeviceTimestampedData("SKVA_CO_W_REPLAY_CLEANUP",skv_sum_CO_W_REPLAY_CLEANUP);
    if(skv_valid_CO_W_INTERVAL_COPY) DeviceTimestampedData("SKVE_CO_W_INTERVAL_COPY",skv_end_CO_W_INTERVAL_COPY);
    if(skv_valid_CO_W_SCORE_ROUTE) DeviceTimestampedData("SKVE_CO_W_SCORE_ROUTE",skv_end_CO_W_SCORE_ROUTE);
    if(skv_valid_CO_W_FRONTIER) DeviceTimestampedData("SKVE_CO_W_FRONTIER",skv_end_CO_W_FRONTIER);
    if(skv_valid_CO_W_THRESHOLD) DeviceTimestampedData("SKVE_CO_W_THRESHOLD",skv_end_CO_W_THRESHOLD);
    if(skv_valid_CO_W_MASK) DeviceTimestampedData("SKVE_CO_W_MASK",skv_end_CO_W_MASK);
    if(skv_valid_CO_W_PREFIX) DeviceTimestampedData("SKVE_CO_W_PREFIX",skv_end_CO_W_PREFIX);
    if(skv_valid_CO_W_EXPORT) DeviceTimestampedData("SKVE_CO_W_EXPORT",skv_end_CO_W_EXPORT);
    if(skv_valid_CO_W_REPLAY_CLEANUP) DeviceTimestampedData("SKVE_CO_W_REPLAY_CLEANUP",skv_end_CO_W_REPLAY_CLEANUP);
#endif
}
#endif
"""
        )
    )
