"""Shared-input Stage 1.6: partition keys, decode once, score all GQA heads.

Specialize Program A only. Each worker's CB26 holds its gathered key quarter.
The reader persists its useful faces to a separate compact L1 tensor for
Stage II; no consumer depends on this scratch CB surviving a program switch.
CB24 holds exact BF16 4-bit keys, CB23 the first eight union-query tiles.
Four bulk score writes per worker transpose candidate ownership to head ownership.
"""

from seoulkv4._kernels.common.refinement_abi import RefinementSemaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines


def tile_partition(chunks, part):
    """32-key-aligned balanced partition, with any extra tile at the front."""
    return part * (chunks // 4) + min(part, chunks % 4), chunks // 4 + (
        part < chunks % 4
    )


def native_helpers():
    """Independently declared kernel fragment; no source adaptation."""
    return r"""#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void s16_clear_score() {
    #pragma GCC unroll 32
    for(uint32_t i=0;i<32;++i)
        sfpi::dst_reg[i].mode<sfpi::DataLayout::F32>()=sfpi::vFloat(0.0f);
}
template <uint32_t FIRST>
inline void s16_expand_four_planes() {
    #pragma GCC unroll 2
    for(uint32_t face_row=0;face_row<2;++face_row) {
        #pragma GCC unroll 8
        for(uint32_t v=0;v<8;++v) {
            sfpi::vUInt packed=sfpi::dst_reg[face_row*16+v]
                .mode<sfpi::DataLayout::U32>();
            sfpi::vUInt packed_upper=sfpi::dst_reg[face_row*16+8+v]
                .mode<sfpi::DataLayout::U32>();
            // Keep the packed vector live across all four output tiles.
            #pragma GCC unroll 4
            for(uint32_t plane=0;plane<4;++plane) {
            #pragma GCC unroll 2
            for(uint32_t half=0;half<2;++half) {
                sfpi::vUInt code=sfpi::shft(packed,-int(4*(FIRST+plane)+2*half)) & 3u;
                sfpi::vUInt upper=sfpi::shft(packed_upper,-int(4*(FIRST+plane)+2*half)) & 3u;
                sfpi::vUInt full=sfpi::shft(upper,2)+code;
                sfpi::vInt centered=sfpi::vInt(full+full)-15;
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
inline void s16_expand_four(uint32_t tile_index) {
    cb_reserve_back(24,4);
    tile_regs_acquire();
    copy_tile(26,tile_index,0);
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,s16_expand_four_planes,
        (FIRST),0,VectorMode::None));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1,24);
    pack_tile(2,24);
    pack_tile(3,24);
    pack_tile(4,24);
    tile_regs_release();
    cb_push_back(24,4);
}

"""


SCORING = r"""
    compute_kernel_hw_startup(26,23,24);
    if constexpr(PROFILE<3) return;
    {
    constexpr uint32_t CHUNKS=LOCAL16_PAD/32;
    const uint32_t part=get_arg_val<uint32_t>(3);
    const uint32_t first=part*(CHUNKS/4)+(part<CHUNKS%4?part:CHUNKS%4);
    const uint32_t count=CHUNKS/4+(part<CHUNKS%4);
    cb_wait_front(23,8);
    cb_wait_front(26,count);
    MATH(SFPU_UNARY_INIT(unused));
    for(uint32_t local_chunk=0;local_chunk<count;++local_chunk) {
        copy_tile_to_dst_init_short_with_dt(24,26);
        pack_reconfig_data_format(24);
        s16_expand_four<0>(local_chunk); s16_expand_four<4>(local_chunk);
        // Consume this chunk before decoding the next one; writer drains in parallel.
    reconfig_data_format<SrcOrder::Reverse>(23,24);
    matmul_init(23,24,1);
    pack_reconfig_data_format(28);
        cb_wait_front(24,8); cb_reserve_back(28,1); tile_regs_acquire();
        if(local_chunk==0) {
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                DST_ACCUM_MODE,s16_clear_score,0,VectorMode::None));
            reconfig_data_format<SrcOrder::Reverse>(23,24);
        }
        for(uint32_t kt=0;kt<8;++kt) matmul_tiles(23,24,kt,kt,0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,28);
        tile_regs_release(); cb_push_back(28,1); cb_pop_front(24,8);
    }
    // Pop the whole multicast train exactly once, after all chunk-local uses.
    cb_pop_front(26,count);
    cb_pop_front(23,8);
    }

"""

QUERY = r"""
    // Reuse the existing union query: first eight tiles are the four heads,
    // plane-major coordinates, with x16 folded into exact BF16 exponents.
    if constexpr(PROFILE>=3) {
        cb_reserve_back(23,8);
        const uint32_t qp=get_write_ptr(23);
        for(uint32_t t=0;t<8;++t)
            noc_async_read(get_noc_addr(slot*16+t,uq),qp+t*2048,2048);
        noc_async_read_barrier(); cb_push_back(23,8);
    }

"""

EXCHANGE = (
    semaphore_defines(SEM)
    + r"""
    constexpr uint32_t CHUNKS=LOCAL16_PAD/32;
    constexpr uint32_t QUARTER_BYTES=((CHUNKS+3)/4)*64;
    const uint32_t first=part*(CHUNKS/4)+(part<CHUNKS%4?part:CHUNKS%4);
    const uint32_t count=CHUNKS/4+(part<CHUNKS%4);
    // CB2 is raw writer-local staging, not a FIFO. No counters are advanced.
    const uint32_t rows=get_write_ptr(2);
    for(uint32_t chunk=0;chunk<count;++chunk) {
        cb_wait_front(28,1);
        auto tile=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(28));
        for(uint32_t h=0;h<4;++h) {
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rows+h*QUARTER_BYTES);
            for(uint32_t c=0;c<32;++c)
                dst[chunk*32+c]=tile[h*16+(c<16?c:256+c-16)];
        }
        cb_pop_front(28,1);
    }
    // RISC-V stores retire before their L1 fill is visible to the NoC. Drain
    // the local fill before the NoC sources these compact score runs.
    __sync_synchronize();
    // Disjoint quarters of the existing head-local TopK score row. No key bits
    // or per-key messages move here, only four contiguous BF16 score runs.
    if(count) for(uint32_t h=0;h<4;++h)
        noc_async_write(rows+h*QUARTER_BYTES,
            get_noc_addr(row_start_x+h,row_start_y,
                         shared_scratch+TOPK_INPUT_OFFSET+first*64),count*64);
    noc_async_write_barrier();
    for(uint32_t h=0;h<4;++h)
        noc_semaphore_inc(get_noc_addr(row_start_x+h,row_start_y,get_semaphore(SKV4_SEM_SCORES_READY)),1);
    noc.async_atomic_barrier();
    auto score_ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_SCORES_READY));
    noc_semaphore_wait(score_ready,4); noc_semaphore_set(score_ready,0);
"""
)

PREFIX_ADD = r"""
#ifdef TRISC_MATH
namespace ckernel::sfpu {
template<uint32_t BASE>
inline void s16_add_prefix() {
    #pragma GCC unroll 32
    for(uint32_t i=0;i<32;++i) {
        sfpi::vFloat a=sfpi::dst_reg[BASE*32+i].mode<sfpi::DataLayout::F32>();
        sfpi::vFloat b=sfpi::dst_reg[(BASE+1)*32+i].mode<sfpi::DataLayout::F32>();
        sfpi::vUInt bits=sfpi::reinterpret<sfpi::vUInt>(a+b);
        // Match numeric/bfloat16.h fp32_to_bf16 (round-to-nearest-even),
        // before TopK inserts positions into the now-zero low 16 bits.
        sfpi::vUInt odd=sfpi::shft(bits,-16)&1u;
        bits=bits+odd+0x7fffu;
        sfpi::dst_reg[BASE*32+i].mode<sfpi::DataLayout::U32>()=bits&0xffff0000u;
    }
}
}
#endif
"""
