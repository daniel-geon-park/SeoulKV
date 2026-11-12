"""Candidate-local Stage-II union refinement for the Stage-1.6-only TT path.

Specialize Stage 1.6 via the matrix/transport/sorted modules. Exchange four
position-tagged top-M lists, attach token IDs and deduplicate at
the original key shard, and fetch low nibbles once per union key. Recompute all
four heads with two exact BF16 matrix operands (upper/lower nibble), not another
head's inherited score. Only local top-100 frontiers move after scoring.

Source splices deliberately share the existing Stage-1.6/attention protocols;
each boundary is checked rather than silently replacing an arbitrary string.
"""

import torch
from seoulkv4._kernels.common.refinement_abi import RefinementSemaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

from seoulkv4._kernels.codegen import refinement


def sources(
    *,
    profile_phase="full",
    split_role=None,
    dynamic_input_counts=False,
    profile=False,
    id_bits=None,
    prefix_page_keys=64,
    prefix_pages=1,
    retain_middle=False,
    debug=False,
):
    spec = dict(
        role=split_role,
        dynamic=dynamic_input_counts,
        phase=profile_phase,
        profile=profile,
        id_bits=id_bits,
        prefix_page_keys=prefix_page_keys,
        prefix_pages=prefix_pages,
        retain_middle=retain_middle,
        debug=debug,
    )
    return tuple(
        emit(**spec)
        for emit in (refinement.compute, refinement.reader, refinement.writer)
    )


def matrix_query(q):
    out = torch.zeros((1, q.shape[0], 64, 256), dtype=torch.bfloat16)
    order2 = torch.arange(256).reshape(16, 16).T.flatten()
    order4 = torch.arange(256).reshape(32, 8).T.flatten()
    out[0, :, :4] = q[..., order2] * 16
    out[0, :, 32:36] = q[..., order4]
    return out


EXPAND = r"""
#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void union_clear() {
    #pragma GCC unroll 32
    for(uint32_t i=0;i<32;++i)
        sfpi::dst_reg[i].mode<sfpi::DataLayout::F32>()=sfpi::vFloat(0.0f);
}
template<bool HIGH>
__attribute__((noinline,noclone)) void union_expand(uint32_t plane) {
    // Decode four planes into Dst 2..5 without releasing input Dst 0/1.
    // Only HIGH/LOW are specialized; both output addresses remain immediate.
    const int shift=-int(4*plane);
    #pragma GCC unroll 2
    for(uint32_t fr=0;fr<2;++fr) {
        #pragma GCC unroll 8
        for(uint32_t v=0;v<8;++v) {
            if constexpr(HIGH) {
                // Both output faces reuse the same two packed Dst vectors.
                sfpi::vUInt hi=sfpi::dst_reg[fr*16+v].mode<sfpi::DataLayout::U32>();
                sfpi::vUInt mid=sfpi::dst_reg[32+fr*16+v].mode<sfpi::DataLayout::U32>();
                hi=sfpi::shft(hi,shift);
                mid=sfpi::shft(mid,shift);
                #pragma GCC unroll 4
                for(uint32_t pair=0;pair<4;++pair)
                    #pragma GCC unroll 2
                    for(uint32_t fc=0;fc<2;++fc) {
                        sfpi::vUInt code=sfpi::shft(sfpi::shft(hi,-int(4*pair+2*fc))&3u,2)
                            +(sfpi::shft(mid,-int(4*pair+2*fc))&3u);
                        sfpi::vInt level=sfpi::vInt(code+code)-15;
                        sfpi::dst_reg[64+pair*32+fr*16+fc*8+v].mode<sfpi::DataLayout::F32>()=
                            sfpi::convert<sfpi::vFloat>(
                                sfpi::convert<sfpi::vSMag>(level),
                                sfpi::RoundMode::Nearest);
                    }
            } else {
                #pragma GCC unroll 2
                for(uint32_t fc=0;fc<2;++fc) {
                    sfpi::vUInt low=sfpi::dst_reg[fr*16+fc*8+v].mode<sfpi::DataLayout::U32>();
                    low=sfpi::shft(low,shift);
                    #pragma GCC unroll 4
                    for(uint32_t pair=0;pair<4;++pair) {
                        sfpi::vUInt code=sfpi::shft(low,-int(4*pair))&15u;
                        sfpi::vInt level=sfpi::vInt(code+code)-15;
                        sfpi::dst_reg[64+pair*32+fr*16+fc*8+v].mode<sfpi::DataLayout::F32>()=
                            sfpi::convert<sfpi::vFloat>(
                                sfpi::convert<sfpi::vSMag>(level),
                                sfpi::RoundMode::Nearest);
                    }
                }
            }
        }
    }
}
}
#endif
__attribute__((noinline,noclone)) void union_expand_tile(bool high, uint32_t plane) {
    if(high) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (true),0,VectorMode::None,plane));
    } else {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (false),0,VectorMode::None,plane));
    }
    tile_regs_commit(); tile_regs_wait();
    pack_tile(2,1);
    pack_tile(3,1);
    pack_tile(4,1);
    pack_tile(5,1);
}
__attribute__((noinline,noclone)) void union_expand_eight(bool high, uint32_t input) {
    #pragma GCC unroll 1
    for(uint32_t t=0;t<8;t+=4) {
        cb_reserve_back(1,4);
        tile_regs_acquire();
        copy_tile(25,input,0);
        if(high) copy_tile(25,input+1,1);
        union_expand_tile(high,t);
        tile_regs_release(); cb_push_back(1,4);
    }
}
"""

MATRIX = r"""
    // Reader-owned control mailbox broadcasts the variable local union size
    // to all three TRISCs using the stock sparse-SDPA primitive.
    cb_wait_front(18,1);
    const uint32_t count=ckernel::read_tile_value(18,0,0);
    cb_pop_front(18,1);
    cb_wait_front(23,16);
    constexpr uint32_t CAP=LOCAL16_PAD/32*3;
    cb_wait_front(25,CAP);
    MATH(SFPU_UNARY_INIT(unused));
    for(uint32_t chunk=0;chunk<(count+31)/32;++chunk) {
        copy_tile_to_dst_init_short_with_dt(5,25);
        pack_reconfig_data_format(1);
        union_expand_eight(true,chunk*3);
        union_expand_eight(false,chunk*3+2);
        cb_wait_front(1,16);
        reconfig_data_format<SrcOrder::Reverse>(23,1);
        matmul_init(23,1,1);
        pack_reconfig_data_format(28);
        cb_reserve_back(28,1);
        tile_regs_acquire();
        // The preceding TopK modifies Dst state: explicitly initialize the
        // accumulator before the first FPU score (composition regression).
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
            union_clear,0,VectorMode::None));
        reconfig_data_format<SrcOrder::Reverse>(23,1);
        for(uint32_t kt=0;kt<16;++kt) matmul_tiles(23,1,kt,kt,0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,28);
        tile_regs_release(); cb_push_back(28,1); cb_pop_front(1,16);
    }
    cb_pop_front(25,CAP); cb_pop_front(23,16);
"""

READER = (
    semaphore_defines(SEM)
    + r"""
    if(!owner) return;
    volatile tt_l1_ptr uint32_t* ready=
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_IDS_READY));
    noc_semaphore_wait(ready,1); noc_semaphore_set(ready,0);
    cb_wait_front(30,1);
    volatile tt_l1_ptr uint32_t* handoff=
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(30));
    const uint32_t count=handoff[0];
    constexpr uint32_t CAP=LOCAL16_PAD/32*3;
    if constexpr(PROFILE>=8) cb_reserve_back(25,CAP);
    const uint32_t dst=get_write_ptr(25);
    const uint32_t retained=get_write_ptr(26); // full Stage-1.6 ring wrapped
    const uint32_t high=scratch+16384;
    for(uint32_t j=0;j<count;++j) {
        const uint32_t tag=handoff[1+j], local=(tag>>20)-shard*LOCAL16;
        const uint32_t tile=j/32, row=j%32;
        const uint32_t lane=(row/16)*2048+(row%16)*64;
        const uint32_t oldlane=(local/32)*4096+((local%32)/16)*2048+(local%16)*64;
        noc_async_read(get_noc_addr(my_x[0],my_y[0],high+local*64),
                       dst+tile*12288+lane,64);
        noc_async_read(get_noc_addr(my_x[0],my_y[0],retained+oldlane),
                       dst+tile*12288+4096+lane,64);
        const uint64_t src=get_noc_addr(slot*context+(tag&((1u<<20)-1)),r4);
        noc_async_read(src,dst+tile*12288+8192+lane,64);
        noc_async_read(src+64,dst+tile*12288+8192+lane+1024,64);
    }
    noc_async_read_barrier();
    cb_pop_front(30,1);
    if constexpr(PROFILE<=7) return;
    cb_reserve_back(18,1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(18))=count;
    cb_push_back(18,1); cb_push_back(25,CAP);
    if constexpr(PROFILE<10) return;

"""
)

HANDOFF = (
    semaphore_defines(SEM)
    + r"""
        // All owners publish tagged top-M frontiers once, in bulk. Each
        // owner keeps only union entries belonging to its original key shard.
        constexpr uint32_t UNION_OFFSET=16384+LOCAL16*64;
        const uint32_t exchange=shared_scratch+UNION_OFFSET;
        volatile tt_l1_ptr uint32_t* lists=
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(exchange);
        for(uint32_t j=0;j<M16;++j) lists[shard*M16+j]=hi16[j];
        for(uint32_t dest=0;dest<G;++dest) if(dest!=shard) {
            const uint32_t x=get_arg_val<uint32_t>(7+2*dest);
            const uint32_t y=get_arg_val<uint32_t>(8+2*dest);
            noc_async_write(exchange+shard*M16*4,
                get_noc_addr(x,y,exchange+shard*M16*4),M16*4);
        }
        noc_async_write_barrier();
        for(uint32_t dest=0;dest<G;++dest) if(dest!=shard)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(7+2*dest),
                get_arg_val<uint32_t>(8+2*dest),get_semaphore(SKV4_SEM_PEER_EXCHANGE_READY)),1);
        noc_async_atomic_barrier();
        auto union_ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_PEER_EXCHANGE_READY));
        noc_semaphore_wait(union_ready,G-1); noc_semaphore_set(union_ready,0);
        uint32_t seen[(LOCAL16+31)/32]={};
        cb_reserve_back(30,1);
        auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(30));
        union_ids=reinterpret_cast<uint32_t*>(exchange+G*M16*4);
        for(uint32_t j=0;j<G*M16;++j) {
            const uint32_t tag=lists[j], pos=tag>>20;
            if(pos/LOCAL16!=shard) continue;
            const uint32_t local=pos%LOCAL16, mask=1u<<(local%32);
            if(seen[local/32]&mask) continue;
            seen[local/32]|=mask;
            union_ids[union_count]=tag&ID_MASK;
            dst[1+union_count++]=tag;
        }
        dst[0]=union_count;
        cb_push_back(30,1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_IDS_READY)),1);
        if constexpr(PROFILE>=8) {
            cb_reserve_back(23,16);
            const uint32_t qp=get_write_ptr(23);
            for(uint32_t t=0;t<16;++t)
                noc_async_read(get_noc_addr(slot*16+t,uq),qp+t*2048,2048);
            noc_async_read_barrier(); cb_push_back(23,16);
        }
    }
"""
)

SELECT = (
    semaphore_defines(SEM)
    + r"""
    if constexpr(PROFILE<=7) { cb_pop_front(31,1); return; }
    if(!owner) { cb_pop_front(31,1); return; }
    constexpr uint32_t HP=16384+LOCAL16*64+G*M16*4+LOCAL16*4;
    auto hv_all=reinterpret_cast<uint16_t*>(shared_scratch+HP);
    auto hi_all=reinterpret_cast<uint32_t*>(shared_scratch+HP+G*K*2);
    for(uint32_t j=0;j<G*K;++j) { hv_all[j]=ordered_bf16(neg_inf_bf16); hi_all[j]=0xffffffffu; }
    for(uint32_t chunk=0;chunk<(union_count+31)/32;++chunk) {
        cb_wait_front(28,1);
        const uint32_t base=get_read_ptr(28);
        if constexpr(PROFILE>=9) {
            const uint32_t begin=chunk*32, end=std::min(begin+32,union_count);
            for(uint32_t h=0;h<G;++h) for(uint32_t j=begin;j<end;++j) {
                const uint16_t value=score_at(base,j-begin,h);
                heap_insert<K>(ordered_bf16(value),union_ids[j],hv_all+h*K,hi_all+h*K);
            }
        }
        cb_pop_front(28,1);
    }
    if constexpr(PROFILE==8) { cb_pop_front(31,1); return; }
    // Exact local top-100 is sufficient to recover global top-100. Exchange
    // four small frontiers, never expanded keys or per-key per-head scores.
    for(uint32_t h=0;h<G;++h) for(uint32_t j=0;j<KLOCAL;++j) {
        packet[h*ROW_WORDS+2*j]=j<K?hv_all[h*K+j]:ordered_bf16(neg_inf_bf16);
        packet[h*ROW_WORDS+2*j+1]=j<K?hi_all[h*K+j]:0xffffffffu;
    }
    for(uint32_t h=0;h<G;++h) {
        const uint32_t src=local_packet+h*ROW_BYTES;
        if(h==shard) {
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive+shard*ROW_BYTES);
            auto p=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
            for(uint32_t j=0;j<ROW_WORDS;++j) dst[j]=p[j];
        } else noc_async_write(src,get_noc_addr(get_arg_val<uint32_t>(7+2*h),
                   get_arg_val<uint32_t>(8+2*h),receive+shard*ROW_BYTES),ROW_BYTES);
    }
    noc_async_write_barrier();
    for(uint32_t h=0;h<G;++h) if(h!=shard)
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(7+2*h),
            get_arg_val<uint32_t>(8+2*h),get_semaphore(SKV4_SEM_STAGE2_FRONTIERS_READY)),1);
    noc_async_atomic_barrier();
    auto done=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_FRONTIERS_READY));
    noc_semaphore_wait(done,G-1); noc_semaphore_set(done,0);
    uint16_t hv[K]; uint32_t hi[K];
    for(uint32_t j=0;j<K;++j) { hv[j]=ordered_bf16(neg_inf_bf16); hi[j]=0xffffffffu; }
    for(uint32_t w=0;w<W;++w) {
        auto p=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive+w*ROW_BYTES);
        for(uint32_t j=0;j<K;++j) heap_insert<K>(p[2*j],p[2*j+1],hv,hi);
    }
"""
)

# Received packets remain in L1. Sort their BF16 scores with positional IDs;
# only the surviving positions are remapped to full 32-bit frontier tags.
MERGE_TOPK = r"""
        static_assert(W * KLOCAL <= 1024);
        cb_reserve_back(30, 1);
        auto merge_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(30));
        for(uint32_t j=0;j<1024;++j) {
            uint16_t bits=neg_inf_bf16;
            if(j<W*KLOCAL) {
                auto src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    receive+(j/KLOCAL)*EXCHANGE_SEG_BYTES);
                const uint16_t ordered=static_cast<uint16_t>(src[2*(j%KLOCAL)]);
                bits=raw_bf16(ordered);
            }
            merge_scores[j]=bits;
        }
        cb_push_back(30,1);
        cb_wait_front(17,1);
        auto ranked=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(17));
        for(uint32_t j=0;j<M16;++j) {
            const uint32_t lane=(j>>6)+((j&63u)<<4);
            const uint32_t pos=decode_topk_xl_k1024(ranked[lane]);
            auto src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                receive+(pos/KLOCAL)*EXCHANGE_SEG_BYTES);
            hi16[j]=src[2*(pos%KLOCAL)+1];
        }
        cb_pop_front(17,1);
"""


PROFILE_PHASES = ("input", "copy", "decode", "pack", "matmul", "full")
