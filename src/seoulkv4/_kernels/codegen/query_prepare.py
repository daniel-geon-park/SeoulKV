"""Explicit QueryPrepare sources; no edits of rendered kernels."""

from seoulkv4._kernels.common.cb_layout import cpp_defines
from seoulkv4._kernels.query_prepare import abi
from seoulkv4._kernels.common.query_metadata import (
    cpp_defines as query_metadata_defines,
)


PHASES = {
    "reader": ("R_SETUP", "R_FETCH", "R_ZERO", "R_LAYOUT", "R_PUBLISH"),
    "compute": ("C_SETUP", "C_WAIT", "C_NORM", "C_PRODUCT", "C_RELEASE"),
    "writer": (
        "W_SETUP",
        "W_WAIT_RAW",
        "W_EARLY_LAYOUT",
        "W_WAIT_NORM",
        "W_EARLY_PUBLISH",
        "W_EARLY_JOIN",
        "W_WAIT_PRODUCT",
        "W_ZERO",
        "W_LAYOUT",
        "W_ISSUE",
        "W_BARRIER",
        "W_RELEASE",
    ),
}

PROFILE_MACROS = r"""
#if SKV4_QP_PROFILE
#include "tools/profiler/kernel_profiler.hpp"
// Read the device clock at the actual phase boundary. Profiler packets are
// emitted only after the work, so their emission timestamps are NOT boundaries.
#define QP_SAMPLE(VAR) do { \
    asm volatile("" ::: "memory"); \
    VAR = *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); \
    asm volatile("" ::: "memory"); \
} while (0);
#define QP_BEGIN(NAME) uint32_t qp_begin_##NAME, qp_end_##NAME; QP_SAMPLE(qp_begin_##NAME)
#define QP_END(NAME) QP_SAMPLE(qp_end_##NAME)
#define QP_DUMP(NAME) \
    DeviceTimestampedData("QPB_" #NAME, qp_begin_##NAME); \
    DeviceTimestampedData("QPE_" #NAME, qp_end_##NAME);
#define QP_DECLARE_POINT(NAME) uint32_t qp_point_##NAME;
#define QP_FIRST_POINT(NAME, INDEX) do { \
    if ((INDEX) == 0) { QP_SAMPLE(qp_point_##NAME) } \
} while (0);
#define QP_DUMP_POINT(NAME) DeviceTimestampedData("QPT_" #NAME, qp_point_##NAME);
#else
#define QP_BEGIN(NAME)
#define QP_END(NAME)
#define QP_DUMP(NAME)
#define QP_DECLARE_POINT(NAME)
#define QP_FIRST_POINT(NAME, INDEX)
#define QP_DUMP_POINT(NAME)
#endif
"""


READER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    QP_BEGIN(R_SETUP)
    constexpr auto qa = TensorAccessorArgs<SKV4_ARG_QP_CT_ACCESSORS>();
    constexpr auto sa = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto za = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(SKV4_ARG_QP_RT_RAW_QUERY), 2048);
    const auto scales = TensorAccessor(sa, get_arg_val<uint32_t>(SKV4_ARG_QP_RT_SCALES), 32768);
    const auto zeros = TensorAccessor(za, get_arg_val<uint32_t>(SKV4_ARG_QP_RT_ZEROS), SKV4_QP_SCRATCH_BYTES);
    const uint32_t group = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_GROUP);
    cb_reserve_back(SKV4_CB_RAW_QUERY, 1);
    cb_reserve_back(SKV4_CB_SCALE_TILES, 8);
    QP_END(R_SETUP)
    QP_BEGIN(R_FETCH)
    noc_async_read(get_noc_addr(group, q), get_write_ptr(SKV4_CB_RAW_QUERY), 2048);
    noc_async_read(get_noc_addr(group, scales), get_write_ptr(SKV4_CB_SCALE_TILES), 32768);
    noc_async_read_barrier();
    QP_END(R_FETCH)
    QP_BEGIN(R_ZERO)
    auto raw = reinterpret_cast<const uint16_t*>(get_write_ptr(SKV4_CB_RAW_QUERY));
    cb_reserve_back(SKV4_CB_QUERY_TILES, 8);
    auto qt = reinterpret_cast<uint32_t*>(get_write_ptr(SKV4_CB_QUERY_TILES));
    noc_async_read(get_noc_addr(0, zeros), get_write_ptr(SKV4_CB_QUERY_TILES), 16384);
    noc_async_read_barrier();
    QP_END(R_ZERO)
    QP_BEGIN(R_LAYOUT)
    const auto words = reinterpret_cast<const uint32_t*>(raw);
    for (uint32_t h = 0; h < 4; ++h) {
        for (uint32_t tile = 0; tile < 8; ++tile) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t src = h * 128 + tile * 16 + face * 8;
                const uint32_t dst = tile * 512 + face * 128 + h * 8;
                // One aligned32-byte face row, two BF16 values per word.
                #pragma GCC unroll 8
                for (uint32_t word = 0; word < 8; ++word)
                    qt[dst + word] = words[src + word];
            }
        }
    }
    QP_END(R_LAYOUT)
    QP_BEGIN(R_PUBLISH)
    __sync_synchronize();
    cb_push_back(SKV4_CB_QUERY_TILES, 8);
    cb_push_back(SKV4_CB_SCALE_TILES, 8);
    cb_push_back(SKV4_CB_RAW_QUERY, 1);
    QP_END(R_PUBLISH)
    QP_DUMP(R_SETUP)
    QP_DUMP(R_FETCH)
    QP_DUMP(R_ZERO)
    QP_DUMP(R_LAYOUT)
    QP_DUMP(R_PUBLISH)
}
"""

COMPUTE = r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#ifdef TRISC_MATH
#include "llk_sfpu/ckernel_sfpu_reduce.h"
#endif
#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void qp_clear() {
    #pragma GCC unroll 1
    for (uint32_t i = 0; i < 32; ++i)
        sfpi::dst_reg[32 + i].mode<sfpi::DataLayout::F32>() = sfpi::vFloat(0.0f);
}
inline void qp_square_accumulate() {
    #pragma GCC unroll 1
    for (uint32_t i = 0; i < 32; ++i) {
        sfpi::vFloat q = sfpi::dst_reg[i].mode<sfpi::DataLayout::F32>();
        sfpi::vFloat square = q * q;
        sfpi::dst_reg[32 + i].mode<sfpi::DataLayout::F32>() =
            sfpi::dst_reg[32 + i].mode<sfpi::DataLayout::F32>() + square;
        sfpi::dst_reg[64 + i].mode<sfpi::DataLayout::F32>() =
            sfpi::dst_reg[64 + i].mode<sfpi::DataLayout::F32>() + q;
    }
}
inline void qp_norm_reduce() {
    init_reduce<PoolType::SUM, DataFormat::Float32, true>();
    calculate_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, DataFormat::Float32, true>();
}
inline void qp_round_product() {
    // Normal-range powers of two commute with BF16 rounding. Keep exceptional
    // words intact for the exact scalar underflow/overflow path. FP32 PACK
    // transports these bits unchanged; no BF16 PACK rounding mode is assumed.
    #pragma GCC unroll 1
    for (uint32_t i = 0; i < 32; ++i) {
        sfpi::vUInt bits = sfpi::dst_reg[i].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt magnitude = bits & 0x7fffffffu;
        v_if (magnitude >= 0x04800000u) {
            v_if (magnitude < 0x7f800000u) {
                sfpi::vUInt rounded = bits + 0x7fffu
                    + (sfpi::shft(bits, -16) & 1u);
                v_if ((rounded & 0x7f800000u) != 0x7f800000u) {
                    bits = rounded & 0xffff0000u;
                } v_endif;
            } v_endif;
        } v_endif;
        sfpi::dst_reg[i].mode<sfpi::DataLayout::U32>() = bits;
    }
}
inline void qp_clear_sum() {
    #pragma GCC unroll 1
    for (uint32_t i = 0; i < 32; ++i)
        sfpi::dst_reg[64 + i].mode<sfpi::DataLayout::F32>() = sfpi::vFloat(0.0f);
}
}
#endif
void kernel_main() {
    QP_DECLARE_POINT(C_FIRST_COPY_ISSUED)
    QP_DECLARE_POINT(C_FIRST_SQUARE_RETURNED)
    QP_BEGIN(C_SETUP)
    compute_kernel_hw_startup(SKV4_CB_QUERY_TILES, SKV4_CB_QUERY_TILES, SKV4_CB_NORM_TILE);
    QP_END(C_SETUP)
    QP_BEGIN(C_WAIT)
    cb_wait_front(SKV4_CB_QUERY_TILES, 8);
    cb_wait_front(SKV4_CB_SCALE_TILES, 8);
    QP_END(C_WAIT)
    QP_BEGIN(C_NORM)
    cb_reserve_back(SKV4_CB_NORM_TILE, 2);
    tile_regs_acquire();
    MATH(SFPU_UNARY_INIT(unused));
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
        qp_clear, 0, VectorMode::None));
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
        qp_clear_sum, 0, VectorMode::None));
    for (uint32_t tile = 0; tile < 8; ++tile) {
        copy_tile_to_dst_init_short(SKV4_CB_QUERY_TILES);
        copy_tile(SKV4_CB_QUERY_TILES, tile, 0);
        QP_FIRST_POINT(C_FIRST_COPY_ISSUED, tile)
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
            qp_square_accumulate, 0, VectorMode::None));
        QP_FIRST_POINT(C_FIRST_SQUARE_RETURNED, tile)
    }
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
        qp_norm_reduce, 1, VectorMode::None));
    sqrt_tile_init();
    sqrt_tile(1);
    binop_with_scalar_tile_init();
    mul_unary_tile(1, get_compile_time_arg_val(SKV4_ARG_QP_CT_LAMBDA_BITS));
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
        qp_norm_reduce, 2, VectorMode::None));
    tile_regs_commit(); tile_regs_wait();
    pack_tile(1, SKV4_CB_NORM_TILE);
    pack_tile(2, SKV4_CB_NORM_TILE);
    tile_regs_release();
    cb_push_back(SKV4_CB_NORM_TILE, 2);
    QP_END(C_NORM)
    QP_BEGIN(C_PRODUCT)

    pack_reconfig_data_format(SKV4_CB_PRODUCT_TILES);
    for (uint32_t tile = 0; tile < 8; ++tile) {
        cb_reserve_back(SKV4_CB_PRODUCT_TILES, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_QUERY_TILES, SKV4_CB_QUERY_TILES);
        copy_tile(SKV4_CB_QUERY_TILES, tile, 0);
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_QUERY_TILES, SKV4_CB_SCALE_TILES);
        copy_tile(SKV4_CB_SCALE_TILES, tile, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
            qp_round_product, 0, VectorMode::None));
        tile_regs_commit(); tile_regs_wait();
        pack_tile(0, SKV4_CB_PRODUCT_TILES);
        tile_regs_release();
        cb_push_back(SKV4_CB_PRODUCT_TILES, 1);
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_SCALE_TILES, SKV4_CB_QUERY_TILES);
    }
    QP_END(C_PRODUCT)
    QP_BEGIN(C_RELEASE)
    cb_pop_front(SKV4_CB_QUERY_TILES, 8);
    cb_pop_front(SKV4_CB_SCALE_TILES, 8);
    QP_END(C_RELEASE)
    QP_DUMP(C_SETUP)
    QP_DUMP(C_WAIT)
    QP_DUMP(C_NORM)
    QP_DUMP(C_PRODUCT)
    QP_DUMP(C_RELEASE)
    QP_DUMP_POINT(C_FIRST_COPY_ISSUED)
    QP_DUMP_POINT(C_FIRST_SQUARE_RETURNED)
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
// Divide finite FP32 by an exact power of two, including gradual underflow.
inline uint32_t downscale(uint32_t bits, uint32_t shift) {
    const uint32_t exp = (bits >> 23) & 255, sign = bits & 0x80000000u;
    if (exp == 255 || (bits & 0x7fffffffu) == 0) return bits;
    if (exp > shift) return bits - (shift << 23);
    const uint32_t m = (bits & 0x7fffffu) | (exp ? 0x800000u : 0);
    const uint32_t n = exp ? shift + 1 - exp : shift;
    const uint32_t floor = m >> n, lost = m & ((1u << n) - 1);
    const uint32_t half = 1u << (n - 1);
    return sign | (floor + (lost > half || (lost == half && (floor & 1))));
}
inline uint16_t bf16(uint32_t bits) {
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t((bits >> 16) | 64u);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1)) >> 16);
}
inline uint16_t times16(uint16_t bits) {
    const uint32_t sign = bits & 0x8000u, exp = (bits >> 7) & 255;
    if (exp == 255) return bits;
    if (exp) return uint16_t(sign | (exp + 4 >= 255 ? 0x7f80u : bits + (4u << 7)));
    uint32_t m = (bits & 127u) << 4, e = 1;
    if (m < 128) return uint16_t(sign | m);
    while (m >= 256) { m >>= 1; ++e; }
    return uint16_t(sign | (e << 7) | (m & 127u));
}
inline uint32_t tile_offset(uint32_t row, uint32_t col) {
    return (row / 16 * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16;
}
__attribute__((noinline,noclone))
void exceptional_scaled_query(uint32_t product, uint16_t& q15, uint16_t& upper,
                              uint16_t& lower) {
    q15 = times16(bf16(downscale(product, 6)));
    lower = bf16(downscale(product, 8));
    upper = times16(lower);
}
inline void scaled_query(uint32_t product, uint16_t& q15, uint16_t& upper,
                         uint16_t& lower) {
    const uint32_t magnitude = product & 0x7fffffffu;
    if (magnitude >= 0x04800000u && magnitude < 0x7f800000u
            && (product & 0xffffu) == 0) {
        const uint16_t base = product >> 16;
        q15 = base - (2u << 7);
        upper = base - (4u << 7);
        lower = base - (8u << 7);
    } else {
        // Includes subnormals, BF16 overflow boundaries, signed zero and NaN.
        // This branch is required numerical coverage, not a legacy-path gate.
        exceptional_scaled_query(product, q15, upper, lower);
    }
}
inline uint16_t lower_query(uint32_t product) {
    const uint32_t magnitude = product & 0x7fffffffu;
    if (magnitude >= 0x04800000u && magnitude < 0x7f800000u
            && (product & 0xffffu) == 0)
        return uint16_t((product >> 16) - (8u << 7));
    return bf16(downscale(product, 8));
}
void kernel_main() {
    QP_BEGIN(W_SETUP)
    const uint32_t worker = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_WORKER);
    constexpr auto ca = TensorAccessorArgs<SKV4_ARG_QP_CT_ACCESSORS>();
    constexpr auto fa = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto na = TensorAccessorArgs<fa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<na.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    constexpr auto za = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    const auto coarse = TensorAccessor(ca, get_arg_val<uint32_t>(SKV4_ARG_QP_RT_COARSE), 2048);
    const auto fine = TensorAccessor(fa, get_arg_val<uint32_t>(SKV4_ARG_QP_RT_FINE), 2048);
    const auto norm = TensorAccessor(na, get_arg_val<uint32_t>(SKV4_ARG_QP_RT_NORM), 64);
    const auto stage15 = TensorAccessor(pa, get_arg_val<uint32_t>(SKV4_ARG_QP_RT_STAGE15), 2048);
    const auto matrix = TensorAccessor(ma, get_arg_val<uint32_t>(SKV4_ARG_QP_RT_MATRIX), 2048);
    const auto compact = TensorAccessor(ra, get_arg_val<uint32_t>(SKV4_ARG_QP_RT_COMPACT), 512);
    const auto zeros = TensorAccessor(za, get_arg_val<uint32_t>(SKV4_ARG_QP_RT_ZEROS), SKV4_QP_SCRATCH_BYTES);
    const uint32_t group = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_GROUP);
    // Clear the whole scratch through one bulk API read while waiting for Q.
    // The NoC implementation may split this into physical burst packets.
    cb_reserve_back(SKV4_CB_OUTPUT_SCRATCH, 1);
    const uint32_t scratch = get_write_ptr(SKV4_CB_OUTPUT_SCRATCH);
    noc_async_read(get_noc_addr(0, zeros), scratch, SKV4_QP_SCRATCH_BYTES);
    QP_END(W_SETUP)
    QP_BEGIN(W_WAIT_RAW)
    if (worker == 0) {
        cb_wait_front(SKV4_CB_RAW_QUERY, 1);
        // Raw-only outputs must not depend on later-stage product tiles.
        for (uint32_t peer = 0; peer < SKV4_QP_WRITERS - 1; ++peer) {
            const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_PEERS + peer * 2);
            const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_PEERS + peer * 2 + 1);
            noc_async_write(get_read_ptr(SKV4_CB_RAW_QUERY),
                            get_noc_addr(x, y, get_read_ptr(SKV4_CB_RAW_QUERY)), 2048);
        }
        noc_async_write_barrier();
        for (uint32_t peer = 0; peer < SKV4_QP_WRITERS - 1; ++peer) {
            const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_PEERS + peer * 2);
            const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_PEERS + peer * 2 + 1);
            noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(SKV4_QP_SEM_RAW_READY)), 1);
        }
    } else {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(SKV4_QP_SEM_RAW_READY)), 1);
        __sync_synchronize();
    }
    QP_END(W_WAIT_RAW)
    QP_BEGIN(W_EARLY_LAYOUT)
    // Drain scratch zero-fill before touching either early or late packets.
    noc_async_read_barrier();
    auto cq = reinterpret_cast<uint16_t*>(scratch);
    auto nq = reinterpret_cast<uint32_t*>(scratch + 5 * SKV4_QP_PACKET_BYTES);
    auto raw = reinterpret_cast<const uint16_t*>(get_read_ptr(SKV4_CB_RAW_QUERY));
    for (uint32_t h = 0; h < 4; ++h) {
        for (uint32_t c = 0; c < SKV4_QP_COLUMNS; ++c) {
            const uint32_t src = worker * SKV4_QP_COLUMNS + c;
            const uint32_t off = SKV4_QP_PRODUCT_OFFSET[c] + h * 16;
            cq[off] = raw[h * 256 + SKV4_QP_COARSE_SOURCE[src]];
        }
    }
    QP_END(W_EARLY_LAYOUT)
    QP_BEGIN(W_WAIT_NORM)
    if (worker == 0) {
        cb_wait_front(SKV4_CB_NORM_TILE, 2);
        auto norms = reinterpret_cast<const uint32_t*>(get_read_ptr(SKV4_CB_NORM_TILE));
        for (uint32_t h = 0; h < 4; ++h) {
            nq[h] = norms[tile_offset(h, 0)];
            nq[SKV4_QUERY_SUM_WORD_OFFSET + h] = norms[1024 + tile_offset(h, 0)];
        }
    }
    QP_END(W_WAIT_NORM)
    QP_BEGIN(W_EARLY_PUBLISH)
    __sync_synchronize();
    for (uint32_t tile = 0; tile < SKV4_QP_PACKET_BYTES / 2048; ++tile) {
        const uint32_t dest = worker * (SKV4_QP_PACKET_BYTES / 2048) + tile;
        noc_async_write(scratch + tile * 2048, get_noc_addr(dest, coarse), 2048);
    }
    if (worker == 0) noc_async_write(scratch + 5 * SKV4_QP_PACKET_BYTES, get_noc_addr(0, norm), 64);
    noc_async_write_barrier();
    QP_END(W_EARLY_PUBLISH)
    QP_BEGIN(W_EARLY_JOIN)
    const uint32_t consumers = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_CONSUMER_COUNT);
    if (consumers != 0) {
        if (worker == 0) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(SKV4_QP_SEM_EARLY_WRITER_DONE)), SKV4_QP_WRITERS - 1);
            for (uint32_t i = 0; i < consumers; ++i) {
                const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_CONSUMERS + 2 * i);
                const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_CONSUMERS + 2 * i + 1);
                noc_semaphore_inc(get_noc_addr(x, y,
                    get_semaphore(get_compile_time_arg_val(SKV4_ARG_QP_CT_CONSUMER_READY_SEMAPHORE))), 1);
            }
        } else {
            noc_semaphore_inc(get_noc_addr(
                get_arg_val<uint32_t>(SKV4_ARG_QP_RT_OWNER_X),
                get_arg_val<uint32_t>(SKV4_ARG_QP_RT_OWNER_Y),
                get_semaphore(SKV4_QP_SEM_EARLY_WRITER_DONE)), 1);
        }
    }
    QP_END(W_EARLY_JOIN)
    QP_BEGIN(W_WAIT_PRODUCT)
    if (worker == 0) {
        cb_wait_front(SKV4_CB_PRODUCT_TILES, 8);
        // All peers own uniformly allocated common CBs. Copy bulk inputs once;
        // readiness is sent only AFTER the write barrier. No helper reads live
        // owner CBs, so owner release needs no delayed acknowledgement.
        for (uint32_t peer = 0; peer < SKV4_QP_WRITERS - 1; ++peer) {
            const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_PEERS + peer * 2);
            const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_PEERS + peer * 2 + 1);
            noc_async_write(get_read_ptr(SKV4_CB_PRODUCT_TILES),
                            get_noc_addr(x, y, get_read_ptr(SKV4_CB_PRODUCT_TILES)), 32768);
        }
        noc_async_write_barrier();
        for (uint32_t peer = 0; peer < SKV4_QP_WRITERS - 1; ++peer) {
            const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_PEERS + peer * 2);
            const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_QP_RT_PEERS + peer * 2 + 1);
            noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(SKV4_QP_SEM_OPERANDS_READY)), 1);
        }
    } else {
        auto ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(SKV4_QP_SEM_OPERANDS_READY));
        noc_semaphore_wait(ready, 1);
        // The semaphore is initialized by each program launch, not reset here
        // after the owner could have notified us (that would lose readiness).
        __sync_synchronize();
    }
    QP_END(W_WAIT_PRODUCT)
    QP_BEGIN(W_ZERO)
    noc_async_read_barrier();
    QP_END(W_ZERO)
    QP_BEGIN(W_LAYOUT)
    auto fq = reinterpret_cast<uint16_t*>(scratch + SKV4_QP_PACKET_BYTES);
    auto pq = reinterpret_cast<uint16_t*>(scratch + 2 * SKV4_QP_PACKET_BYTES);
    auto mu = reinterpret_cast<uint16_t*>(scratch + 3 * SKV4_QP_PACKET_BYTES);
    auto ml = reinterpret_cast<uint16_t*>(scratch + 4 * SKV4_QP_PACKET_BYTES);
    auto products = reinterpret_cast<const uint32_t*>(get_read_ptr(SKV4_CB_PRODUCT_TILES));
    for (uint32_t h = 0; h < 4; ++h) {
        for (uint32_t c = 0; c < SKV4_QP_COLUMNS; ++c)
            fq[SKV4_QP_PRODUCT_OFFSET[c] + h * 16] =
                raw[h * 256 + SKV4_QP_FINE_SOURCE[worker * SKV4_QP_COLUMNS + c]];
        for (uint32_t i = 0; i < SKV4_QP_BOTH_COUNT[worker]; ++i) {
            const auto entry = SKV4_QP_BOTH_WORK[worker][i];
            const uint32_t row = h * 16;
            const uint32_t product = products[entry[0] + row];
            uint16_t q15, q16, q16upper;
            scaled_query(product, q15, q16upper, q16);
            const uint32_t off = entry[2] + row;
            pq[off] = q15;
            mu[off] = q16upper;
            ml[entry[3] + row] = q16;
        }
        for (uint32_t i = 0; i < SKV4_QP_COARSE_COUNT[worker]; ++i) {
            const auto entry = SKV4_QP_COARSE_WORK[worker][i];
            const uint32_t row = h * 16;
            uint16_t q15, q16, q16upper;
            scaled_query(products[entry[0] + row], q15, q16upper, q16);
            const uint32_t off = entry[2] + row;
            pq[off] = q15;
            mu[off] = q16upper;
        }
        for (uint32_t i = 0; i < SKV4_QP_MATRIX_COUNT[worker]; ++i) {
            const auto entry = SKV4_QP_MATRIX_WORK[worker][i];
            ml[entry[1] + h * 16] = lower_query(products[entry[0] + h * 16]);
        }
    }
    for (uint32_t h = worker; h < 4; h += SKV4_QP_WRITERS)
        noc_async_write(get_read_ptr(SKV4_CB_RAW_QUERY) + h * 512,
                        get_noc_addr(group * 4 + h, compact), 512);
    QP_END(W_LAYOUT)
    QP_BEGIN(W_ISSUE)
    for (uint32_t tile = 0; tile < SKV4_QP_PACKET_BYTES / 2048; ++tile) {
        const uint32_t dest = worker * (SKV4_QP_PACKET_BYTES / 2048) + tile;
        noc_async_write(scratch + SKV4_QP_PACKET_BYTES + tile * 2048, get_noc_addr(dest, fine), 2048);
        noc_async_write(scratch + 2 * SKV4_QP_PACKET_BYTES + tile * 2048, get_noc_addr(group * 8 + dest, stage15), 2048);
        noc_async_write(scratch + 3 * SKV4_QP_PACKET_BYTES + tile * 2048, get_noc_addr(group * 16 + dest, matrix), 2048);
        noc_async_write(scratch + 4 * SKV4_QP_PACKET_BYTES + tile * 2048, get_noc_addr(group * 16 + 8 + dest, matrix), 2048);
    }
    QP_END(W_ISSUE)
    QP_BEGIN(W_BARRIER)
    noc_async_write_barrier();
    QP_END(W_BARRIER)
    QP_BEGIN(W_RELEASE)
    if (worker == 0) {
        cb_pop_front(SKV4_CB_RAW_QUERY, 1);
        cb_pop_front(SKV4_CB_NORM_TILE, 2);
        cb_pop_front(SKV4_CB_PRODUCT_TILES, 8);
    }
    QP_END(W_RELEASE)
    QP_DUMP(W_SETUP)
    QP_DUMP(W_WAIT_RAW)
    QP_DUMP(W_EARLY_LAYOUT)
    QP_DUMP(W_WAIT_NORM)
    QP_DUMP(W_EARLY_PUBLISH)
    QP_DUMP(W_EARLY_JOIN)
    QP_DUMP(W_WAIT_PRODUCT)
    QP_DUMP(W_ZERO)
    QP_DUMP(W_LAYOUT)
    QP_DUMP(W_ISSUE)
    QP_DUMP(W_BARRIER)
    QP_DUMP(W_RELEASE)
}
"""


def writer_offsets(coarse_order, fine_order, matrix_order):
    """FP32/BF16 element offsets for one canonical coordinate, head zero."""

    def physical(c):
        return (c // 32) * 1024 + ((c % 32) // 16) * 256 + c % 16

    tables = {"PRODUCT_OFFSET": tuple(physical(c) for c in range(256))}
    for name, order in (
        ("COARSE", coarse_order),
        ("FINE", fine_order),
        ("MATRIX", matrix_order),
    ):
        values = tuple(map(int, order))
        if sorted(values) != list(range(256)):
            raise ValueError(f"invalid {name} coordinate permutation")
        inverse = [0] * 256
        for dest, source in enumerate(values):
            inverse[source] = physical(dest)
        tables[name + "_DEST"] = tuple(inverse)
    return tables


def writer_worklists(tables):
    """Canonical coordinates touching each disjoint output-tile packet."""
    return [
        tuple(
            c
            for c in range(256)
            if any(
                tables[name + "_DEST"][c] // abi.PACKET_ELEMENTS == worker
                for name in ("COARSE", "MATRIX")
            )
        )
        for worker in range(abi.WRITERS_PER_GROUP)
    ]


def writer_schedules(tables):
    """Pre-resolve ownership, product offsets and packet-local destinations."""
    schedules = {
        name: [[] for _ in range(abi.WRITERS_PER_GROUP)]
        for name in ("BOTH", "COARSE", "MATRIX")
    }
    for worker, coordinates in enumerate(writer_worklists(tables)):
        for c in coordinates:
            product = tables["PRODUCT_OFFSET"][c]
            cd, md = tables["COARSE_DEST"][c], tables["MATRIX_DEST"][c]
            coarse = cd // abi.PACKET_ELEMENTS == worker
            matrix = md // abi.PACKET_ELEMENTS == worker
            if coarse and matrix:
                schedules["BOTH"][worker].append(
                    (product, c, cd % abi.PACKET_ELEMENTS, md % abi.PACKET_ELEMENTS)
                )
            elif coarse:
                schedules["COARSE"][worker].append(
                    (product, c, cd % abi.PACKET_ELEMENTS)
                )
            else:
                schedules["MATRIX"][worker].append((product, md % abi.PACKET_ELEMENTS))
    return schedules


def emit(*, role, coarse_order=(), fine_order=(), matrix_order=(), profile=False):
    """Emit a complete kernel from explicit ABI and static layout tables."""
    rt, ct = {
        "reader": (abi.READER_RT, abi.READER_CT),
        "compute": (None, abi.COMPUTE_CT),
        "writer": (abi.WRITER_RT, abi.WRITER_CT),
    }[role]
    prefix = (
        "#include <cstdint>\n"
        + f"#define SKV4_QP_PROFILE {int(bool(profile))}\n"
        + f"#define SKV4_QP_SCRATCH_BYTES {abi.SCRATCH_BYTES}\n"
        + f"#define SKV4_QP_PACKET_BYTES {abi.PACKET_BYTES}\n"
        + f"#define SKV4_QP_PACKET_ELEMENTS {abi.PACKET_ELEMENTS}\n"
        + f"#define SKV4_QP_COLUMNS {256 // abi.WRITERS_PER_GROUP}\n"
        + f"#define SKV4_QP_WRITERS {abi.WRITERS_PER_GROUP}\n"
        + query_metadata_defines()
        + PROFILE_MACROS
        + cpp_defines(abi.QueryPrepareCB)
        + ct.cpp_defines("SKV4_ARG_QP_CT")
    )
    if rt is not None:
        prefix += rt.cpp_defines("SKV4_ARG_QP_RT")
    if role == "writer":
        tables = writer_offsets(coarse_order, fine_order, matrix_order)
        for name, values in tables.items():
            if name != "PRODUCT_OFFSET":
                continue
            prefix += (
                f"constexpr uint16_t SKV4_QP_{name}[256] = {{"
                + ",".join(map(str, values))
                + "};\n"
            )
        prefix += (
            "constexpr uint16_t SKV4_QP_COARSE_SOURCE[256] = {"
            + ",".join(map(str, map(int, coarse_order)))
            + "};\n"
        )
        prefix += (
            "constexpr uint16_t SKV4_QP_FINE_SOURCE[256] = {"
            + ",".join(map(str, map(int, fine_order)))
            + "};\n"
        )
        for sem in abi.QueryPrepareSemaphore:
            prefix += f"#define SKV4_QP_SEM_{sem.name} {int(sem)}\n"
        schedules = writer_schedules(tables)
        for name, fields in (("BOTH", 4), ("COARSE", 3), ("MATRIX", 2)):
            lists = schedules[name]
            prefix += f"constexpr uint16_t SKV4_QP_{name}_COUNT[SKV4_QP_WRITERS] = {{"
            prefix += ",".join(str(len(row)) for row in lists) + "};\n"
            width = max(1, max(map(len, lists)))
            prefix += f"constexpr uint16_t SKV4_QP_{name}_WORK[SKV4_QP_WRITERS][{width}][{fields}] = {{"
            prefix += ",".join(
                "{"
                + ",".join("{" + ",".join(map(str, entry)) + "}" for entry in row)
                + "}"
                for row in lists
            )
            prefix += "};\n"
    return prefix + {"reader": READER, "compute": COMPUTE, "writer": WRITER}[role]
