"""Native BFP4 Coarse scorer with source-local lower and upper intervals.

This body is independently declared rather than patched from Stage 1.5.
BF16/FP32 arithmetic, PACK state and CB ownership are unchanged.
"""

from seoulkv4._kernels.stage1_coarse.abi import CoarseCB as CB
from seoulkv4._kernels.stage1_coarse.abi import OPERAND_BATCH
from seoulkv4._kernels.common.cb_layout import cpp_defines

from seoulkv4._kernels.common.selection_abi import SCORE_COMPUTE_CT
from seoulkv4._kernels.common.query_metadata import (
    cpp_defines as query_metadata_defines,
)

COMPUTE = (
    cpp_defines(CB)
    + f"#define SKV4_COARSE_OPERAND_BATCH {OPERAND_BATCH}\n"
    + SCORE_COMPUTE_CT.cpp_defines("SKV4_ARG_CT")
    + query_metadata_defines()
    + r"""
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
    // Last page reads input vectors12..15 and overwrites only0..3.
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
    constexpr uint32_t N=get_compile_time_arg_val(SKV4_ARG_CT_N), PHASE=get_compile_time_arg_val(SKV4_ARG_CT_PHASE);
    compute_kernel_hw_startup(SKV4_CB_PACKED_UPPER,SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS);

    uint32_t table_base = 0;
    cb_wait_front(SKV4_CB_METADATA,1);
    UNPACK(mailbox_write(ckernel::ThreadId::MathThreadId,
        get_local_cb_interface(SKV4_CB_METADATA).fifo_rd_ptr << 4));
    MATH(table_base = mailbox_read(ckernel::ThreadId::UnpackThreadId));
    cb_wait_front(SKV4_CB_QUERY,8);
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
            copy_tile_to_dst_init_short_with_dt(SKV4_CB_DECODED_KEYS,SKV4_CB_PACKED_UPPER);
            pack_reconfig_data_format(SKV4_CB_PACKED_UPPER);
            pack_rows_to_addr_init(8);
            for(uint32_t local=0;local<count;++local) {
                cb_wait_front(SKV4_CB_PACKED_UPPER,1);
                expand_eight();
                cb_pop_front(SKV4_CB_PACKED_UPPER,1);
            }
            pack_rows_to_addr_uninit();
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
                    cb_wait_front(SKV4_CB_DECODED_KEYS,8);
                    if constexpr(PHASE==2) {
                        cb_reserve_back(SKV4_CB_LOWER_SCORE_TILES,1);
                        cb_reserve_back(SKV4_CB_UPPER_SCORE_TILES,1);
                        tile_regs_acquire();
                        if(chunk==0) {
                            // TopK-XL's first FPU output needs explicit clearing.
                            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                                DST_ACCUM_MODE,clear_score,0,VectorMode::None));
                            reconfig_full_operand<SrcOrder::Reverse>(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS);
                        }
                        for(uint32_t kt=0;kt<8;++kt) matmul_tiles(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS,kt,kt,0);
                        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                            DST_ACCUM_MODE,coarse_scale_interval,0,VectorMode::None,
                            table_base + chunk * 64,
                            table_base + N * 2 + chunk * 64,
                            table_base + N * 4));
                        tile_regs_commit(); tile_regs_wait();
                        pack_tile(0,SKV4_CB_LOWER_SCORE_TILES);
                        pack_tile(1,SKV4_CB_UPPER_SCORE_TILES);
                        tile_regs_release(); cb_push_back(SKV4_CB_LOWER_SCORE_TILES,1); cb_push_back(SKV4_CB_UPPER_SCORE_TILES,1);
                    }
                    cb_pop_front(SKV4_CB_DECODED_KEYS,8);
                }
            }
        }
    }
    cb_pop_front(SKV4_CB_QUERY,8); cb_pop_front(SKV4_CB_METADATA,1);
}
"""
)
