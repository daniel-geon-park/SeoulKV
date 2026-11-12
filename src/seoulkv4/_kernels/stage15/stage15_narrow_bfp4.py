"""Compact 32x16 UInt32 Stage1.5 input decoded directly by SFPU.

This experimental path keeps each key as sixteen packed UInt32 words.  One
2,048-byte narrow tile therefore carries 32 complete keys and replaces the
scalar NCRISC conversion into a hole-padded 4,096-byte native-layout page.

The standalone witness packs byte-exact native BFP4. Production deliberately
packs the decoded Dst values as BF16 before matmul: v0.77's fused BFP4
producer/consumer bridge produced non-finite scores even though the isolated
BFP4 bytes and input records were exact. The BF16 bridge is both correct and
faster end to end because it removes the scalar native-layout pass.
"""

from seoulkv4._kernels.stage15.stage15_sfpu_native import COMPUTE as CONTROL


def compute_source():
    """Independently declared kernel fragment; no source adaptation."""
    return r"""
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
template <uint32_t FIRST>
inline void expand_four_planes() {
    #pragma GCC unroll 2
    for(uint32_t face_row=0;face_row<2;++face_row) {
        #pragma GCC unroll 8
        for(uint32_t v=0;v<8;++v) {
            sfpi::vUInt packed=sfpi::dst_reg[face_row*8+v]
                .mode<sfpi::DataLayout::U32>();
            // Keep the packed vector live across all four output tiles.
            #pragma GCC unroll 4
            for(uint32_t plane=0;plane<4;++plane) {
            #pragma GCC unroll 2
            for(uint32_t half=0;half<2;++half) {
                sfpi::vUInt code=sfpi::shft(packed,-int(4*(FIRST+plane)+2*half)) & 3u;
                sfpi::vInt centered=sfpi::vInt(code+code)-3;
                sfpi::vFloat val=sfpi::convert<sfpi::vFloat>(
                    sfpi::convert<sfpi::vSMag>(centered),
                    sfpi::RoundMode::Nearest);
                sfpi::dst_reg[(plane+1)*32+face_row*16+half*8+v]
                    .mode<sfpi::DataLayout::F32>()=val;
            }
            }
        }
    }
}
}
#endif

template <uint32_t FIRST>
inline void expand_four() {
    cb_reserve_back(1,4);
    tile_regs_acquire();
    copy_tile(5,0,0);
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,expand_four_planes,
        (FIRST),0,VectorMode::None));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1,1);
    pack_tile(2,1);
    pack_tile(3,1);
    pack_tile(4,1);
    tile_regs_release();
    cb_push_back(1,4);
}

void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0), PHASE=get_compile_time_arg_val(1);
    compute_kernel_hw_startup(5,0,1);
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
            expand_four<0>(); expand_four<4>();
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
                    tile_regs_commit(); tile_regs_wait();
                    pack_tile(0,16);
                    tile_regs_release(); cb_push_back(16,1);
                }
                cb_pop_front(1,8);
            }
        }
    }
    cb_pop_front(0,8);
}
"""


COMPUTE = compute_source()
