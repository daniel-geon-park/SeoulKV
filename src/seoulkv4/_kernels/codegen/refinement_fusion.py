"""Explicit fused S16/II phase bodies, never transformed rendered kernels."""

from seoulkv4._kernels.common.cb_layout import cpp_defines
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as sem_defines
from seoulkv4._kernels.common.refinement_abi import refinement_defines, refinement_abi
from seoulkv4._kernels.stage16.fusion_abi import (
    FusionCB,
    FusionSemaphore,
    ATTENTION_DESTINATION,
    AttentionSemaphore,
)
from seoulkv4._kernels.stage16.seoulkv4_stage16_matrix import native_helpers
from seoulkv4._kernels.stage16.seoulkv4_union_matrix import EXPAND


def header(kind, *, local, id_bits, attention=False):
    writer_rt, _ = refinement_abi("writer", union=True, handoff=True)
    return (
        cpp_defines(FusionCB)
        + sem_defines(FusionSemaphore)
        + refinement_defines(kind, role="stage16", dynamic=True)
        + sem_defines(AttentionSemaphore)
        + ATTENTION_DESTINATION.cpp_defines("SKV4_ARG_FATT_RT", base=writer_rt.words)
        + f"#define SKV4_CODEGEN_FUSED_ATTENTION {int(attention)}\n"
        + f"#define SKV4_FUSED_LOCAL {local}\n#define SKV4_FUSED_ID_BITS {id_bits}\n"
    )


DATAFLOW = r"""
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t LOCAL=SKV4_FUSED_LOCAL;
constexpr uint32_t PAD=(LOCAL+31)/32*32;
constexpr uint32_t QUARTER_CHUNKS=(PAD/32+3)/4;
constexpr uint32_t ID_BITS=SKV4_FUSED_ID_BITS;
constexpr uint32_t ID_MASK=(1u<<ID_BITS)-1;
constexpr uint32_t LIST_BYTES=1664;
constexpr uint32_t META_BYTES=(LOCAL+1)*4<=4096?4096:((LOCAL+1)*4+63)/64*64;
constexpr uint32_t PREFIX_KEYS=LOCAL<1024?LOCAL:1024;
constexpr uint32_t PREFIX_PAGES=(LOCAL+PREFIX_KEYS-1)/PREFIX_KEYS;
constexpr uint32_t MIDDLE_PAGES=PAD/16;
inline uint32_t sem(uint32_t id) { return get_semaphore(id); }
inline void await(uint32_t id, uint32_t count) {
    auto p=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem(id));
    noc_semaphore_wait(p,count); noc_semaphore_set(p,0);
}
"""

READER = r"""
void kernel_main() {
    constexpr auto sq=TensorAccessorArgs<SKV4_ARG_RF_CT_ACCESSORS>();
    constexpr auto r2a=TensorAccessorArgs<sq.next_compile_time_args_offset()>();
    constexpr auto r4a=TensorAccessorArgs<r2a.next_compile_time_args_offset()>();
    constexpr auto cia=TensorAccessorArgs<r4a.next_compile_time_args_offset()>();
    constexpr auto s2=TensorAccessorArgs<cia.next_compile_time_args_offset()>();
    constexpr auto aq=TensorAccessorArgs<s2.next_compile_time_args_offset()>();
    constexpr auto kv=TensorAccessorArgs<aq.next_compile_time_args_offset()>();
    constexpr auto pa=TensorAccessorArgs<kv.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto ha=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto ma=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto r2=TensorAccessor(r2a,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_RESIDUAL2),64);
    const auto r4=TensorAccessor(r4a,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_RESIDUAL4),128);
    const auto ids=TensorAccessor(cia,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_CANDIDATES),LOCAL*16);
    const auto upper=TensorAccessor(pa,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_PREFIX2),PREFIX_KEYS*64);
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_INPUT_COUNTS),16);
    const auto handoff=TensorAccessor(ha,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HANDOFF),META_BYTES);
    const auto middle=TensorAccessor(ma,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_MIDDLE),1024);
    const uint32_t slot=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SLOT);
    const uint32_t context=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_CONTEXT);
    const uint32_t shard=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SHARD);
    const uint32_t part=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_PART);
    const uint32_t root_x=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROOT_X);
    const uint32_t root_y=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROOT_Y);
    const uint32_t row_x=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_START_X);
    cb_reserve_back(SKV4_CB_SOURCE_IDS,1);
    const uint32_t meta=get_write_ptr(SKV4_CB_SOURCE_IDS);
    noc_async_read(get_noc_addr(slot,counts),meta,16); noc_async_read_barrier();
    const uint32_t live=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta)[shard];
    if(live>LOCAL) __builtin_trap();
    const uint32_t skip=(shard*LOCAL*4)&63u;
    if(live) noc_async_read(get_noc_addr(slot,ids)+shard*LOCAL*4-skip,
                            meta+64,live*4+skip);
    noc_async_read_barrier(); cb_push_back(SKV4_CB_SOURCE_IDS,1);
    cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=live;
    cb_push_back(SKV4_CB_LIVE_CONTROL,1);
    auto tokens=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta+64+skip);
    const uint32_t chunks=(live+31)/32;
    const uint32_t first=part*(chunks/4)+(part<chunks%4?part:chunks%4);
    const uint32_t count=chunks/4+(part<chunks%4);
    if(count) cb_reserve_back(SKV4_CB_PACKED_MIDDLE,count);
    const uint32_t packed=get_write_ptr(SKV4_CB_PACKED_MIDDLE);
    for(uint32_t c=0;c<count;++c) {
        const uint32_t begin=(first+c)*32;
        for(uint32_t row=0;row<32;++row) {
            const uint32_t source=begin+row<live?begin+row:0;
            noc_async_read(get_noc_addr(slot*context+tokens[source],r2),
                           packed+c*4096+(row/16)*2048+(row%16)*64,64);
        }
        for(uint32_t face=0;face<2;++face) {
            const uint32_t at=begin+face*16;
            const uint32_t valid=at<live?(live-at<16?live-at:16):0;
            const uint32_t dst=packed+c*4096+face*2048+1024;
            if(valid) noc_async_read(get_noc_addr((slot*4+shard)*PREFIX_PAGES+at/PREFIX_KEYS,upper)
                                     +(at%PREFIX_KEYS)*64,dst,valid*64);
            if(valid<16) {
                auto tail=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst+valid*64);
                for(uint32_t j=0;j<(16-valid)*16;++j) tail[j]=0;
            }
        }
        noc_async_read_barrier(); cb_push_back(SKV4_CB_PACKED_MIDDLE,1);
    }
    for(uint32_t face=0;face<count*2;++face)
        noc_async_write(packed+face*2048,
            get_noc_addr((slot*4+shard)*MIDDLE_PAGES+first*2+face,middle),1024);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(root_x,root_y,sem(SKV4_SEM_MIDDLE_COUNT)),1);
    noc_async_atomic_barrier();
    if(part==0) {
        await(SKV4_SEM_MIDDLE_COUNT,4);
        for(uint32_t p=1;p<4;++p)
            noc_semaphore_inc(get_noc_addr(row_x+p,root_y,sem(SKV4_SEM_MIDDLE_RELEASE)),1);
        noc_async_atomic_barrier();
    } else await(SKV4_SEM_MIDDLE_RELEASE,1);
    const uint32_t front=get_write_ptr(SKV4_CB_FRONTIER_INPUT);
    auto lists=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(front);
    cb_reserve_back(SKV4_CB_UNION_META,1);
    auto tags=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_UNION_META));
    uint32_t seen[(LOCAL+31)/32]={};
    uint32_t total=0, mine=0, tile=0;
    for(uint32_t h=0;h<4;++h) {
     for(uint32_t packet=0;packet<2;++packet) {
      // Do not reset after publication1: publication2 may already be present.
      auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem(SKV4_SEM_HEAD0_READY+h));
      noc_semaphore_wait_min(ready,packet+1);
      if(packet==1) noc_semaphore_set(ready,0);
      const uint32_t offset=packet?1088:0,bytes=packet?576:1088;
      noc_async_read(get_noc_addr(slot*4+h,handoff)+offset,
                     front+h*LIST_BYTES+offset,bytes);
      noc_async_read_barrier();
      for(uint32_t j=packet?256:0;j<(packet?400:256);++j) {
        const uint32_t pos=lists[h*(LIST_BYTES/4)+16+j];
        const uint32_t local=pos-shard*LOCAL;
        if(local>=live) continue;
        const uint32_t mask=1u<<(local%32);
        if(seen[local/32]&mask) continue;
        seen[local/32]|=mask;
        const uint32_t tag=(pos<<ID_BITS)|tokens[local];
        tags[1+total]=tag;
        const bool own=(total/32)%4==part;
        ++total;
        if(!own) continue;
        const uint32_t row=mine%32;
        if(row==0) { cb_reserve_back(SKV4_CB_PACKED_CANDIDATES,3); tile=get_write_ptr(SKV4_CB_PACKED_CANDIDATES); }
        const uint32_t lane=(row/16)*2048+(row%16)*64;
        noc_async_read(get_noc_addr((slot*4+shard)*PREFIX_PAGES+local/PREFIX_KEYS,upper)
                       +(local%PREFIX_KEYS)*64,tile+lane,64);
        noc_async_read(get_noc_addr((slot*4+shard)*MIDDLE_PAGES+local/16,middle)
                       +(local%16)*64,tile+4096+lane,64);
        const uint64_t low=get_noc_addr(slot*context+(tag&ID_MASK),r4);
        noc_async_read(low,tile+8192+lane,64);
        noc_async_read(low+64,tile+8192+lane+1024,64);
        ++mine;
        if(mine%32==0) {
            noc_async_read_barrier(); cb_push_back(SKV4_CB_PACKED_CANDIDATES,3);
            cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=32;
            cb_push_back(SKV4_CB_LIVE_CONTROL,1);
        }
      }
     }
    }
    if(mine%32) {
        noc_async_read_barrier(); cb_push_back(SKV4_CB_PACKED_CANDIDATES,3);
        cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=mine%32;
        cb_push_back(SKV4_CB_LIVE_CONTROL,1);
    }
    if(total>640) __builtin_trap();
    tags[0]=total; __sync_synchronize(); cb_push_back(SKV4_CB_UNION_META,1);
    for(uint32_t message=0;message<2;++message) {
        const uint32_t value=message?total:0;
        cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=value;
        cb_push_back(SKV4_CB_LIVE_CONTROL,1);
    }
}
"""

COMPUTE_HEADERS = r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
"""

TOPK = r"""
template<uint32_t K, bool SEPARATE>
void sort_half(uint32_t input, uint32_t output, uint32_t count, bool second) {
    tile_regs_acquire(); topk_xl_copy_tile_init(input);
    topk_xl_copy_tile<K>(input,0,0,count);
    topk_xl_add_lsb_indices_init();
    if constexpr(SEPARATE) topk_xl_add_lsb_indices<K,0>(0);
    else { if(second) topk_xl_add_lsb_indices<K,1>(0); else topk_xl_add_lsb_indices<K,0>(0); }
    topk_xl_init<K,true>(); topk_xl_local_sort<K>(0,false);
    if constexpr(SEPARATE) {
        topk_xl_separate_indices_row_major_init_static<0,0>();
        topk_xl_separate_indices_row_major_reinit();
        topk_xl_separate_indices_row_major<K>(0);
        topk_xl_separate_indices_row_major_advance_chunk_base<K>();
        topk_xl_init<K,false>(); topk_xl_rebuild<K,false>(0,false);
    }
    tile_regs_commit(); cb_reserve_back(output,1); tile_regs_wait();
    pack_reconfig_data_format(output); pack_tile(SEPARATE?1:0,output);
    cb_push_back(output,1); tile_regs_release(); cb_pop_front(input,1);
}
"""

COMPUTE = r"""
void kernel_main() {
    compute_kernel_hw_startup(SKV4_CB_PACKED_MIDDLE,SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH);
    const uint32_t part=get_arg_val<uint32_t>(SKV4_ARG_RF_COMPUTE_RT_PART);
    cb_wait_front(SKV4_CB_LIVE_CONTROL,1);
    const uint32_t live=ckernel::read_tile_value(SKV4_CB_LIVE_CONTROL,0,0);
    cb_pop_front(SKV4_CB_LIVE_CONTROL,1);
    const uint32_t chunks=(live+31)/32;
    const uint32_t count=chunks/4+(part<chunks%4);
    cb_wait_front(SKV4_CB_MATRIX_QUERY,8); MATH(SFPU_UNARY_INIT(unused));
    for(uint32_t c=0;c<count;++c) {
        cb_wait_front(SKV4_CB_PACKED_MIDDLE,c+1);
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_DECODE_SCRATCH,SKV4_CB_PACKED_MIDDLE);
        pack_reconfig_data_format(SKV4_CB_DECODE_SCRATCH);
        s16_expand_four<0>(c); s16_expand_four<4>(c);
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH);
        matmul_init(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH,1);
        pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
        cb_wait_front(SKV4_CB_DECODE_SCRATCH,8); cb_reserve_back(SKV4_CB_SCORE_TILES,1);
        tile_regs_acquire();
        if(c==0) {
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,s16_clear_score,0,VectorMode::None));
            reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH);
        }
        for(uint32_t k=0;k<8;++k) matmul_tiles(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH,k,k,0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,SKV4_CB_SCORE_TILES);
        tile_regs_release(); cb_push_back(SKV4_CB_SCORE_TILES,1); cb_pop_front(SKV4_CB_DECODE_SCRATCH,8);
    }
    if(count) cb_pop_front(SKV4_CB_PACKED_MIDDLE,count);
    cb_pop_front(SKV4_CB_MATRIX_QUERY,8);
    cb_wait_front(SKV4_CB_SELECTOR_INPUT,1);
    sort_half<1024,false>(SKV4_CB_SELECTOR_INPUT,SKV4_CB_STAGE16_FRONTIER,live<1024?live:1024,false);
    if(live>1024) {
        cb_wait_front(SKV4_CB_TOPK_SECOND_INPUT,1);
        sort_half<1024,false>(SKV4_CB_TOPK_SECOND_INPUT,SKV4_CB_STAGE16_FRONTIER,live-1024,true);
    }
    // Operation-specific transition; do not run hardware startup again.
    cb_wait_front(SKV4_CB_STAGE2_QUERY,16); MATH(SFPU_UNARY_INIT(unused));
    for(;;) {
        cb_wait_front(SKV4_CB_LIVE_CONTROL,1);
        const uint32_t n=ckernel::read_tile_value(SKV4_CB_LIVE_CONTROL,0,0);
        cb_pop_front(SKV4_CB_LIVE_CONTROL,1);
        if(n==0) break;
        cb_wait_front(SKV4_CB_PACKED_CANDIDATES,3);
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_COPY_WITNESS,SKV4_CB_PACKED_CANDIDATES);
        pack_reconfig_data_format(SKV4_CB_DECODED_KEYS);
        union_expand_eight(true,0); union_expand_eight(false,2);
        cb_wait_front(SKV4_CB_DECODED_KEYS,16);
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_STAGE2_QUERY,SKV4_CB_DECODED_KEYS);
        matmul_init(SKV4_CB_STAGE2_QUERY,SKV4_CB_DECODED_KEYS,1);
        pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
        cb_reserve_back(SKV4_CB_SCORE_TILES,1); tile_regs_acquire();
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,union_clear,0,VectorMode::None));
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_STAGE2_QUERY,SKV4_CB_DECODED_KEYS);
        for(uint32_t k=0;k<16;++k) matmul_tiles(SKV4_CB_STAGE2_QUERY,SKV4_CB_DECODED_KEYS,k,k,0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,SKV4_CB_SCORE_TILES);
        tile_regs_release(); cb_push_back(SKV4_CB_SCORE_TILES,1);
        cb_pop_front(SKV4_CB_DECODED_KEYS,16); cb_pop_front(SKV4_CB_PACKED_CANDIDATES,3);
    }
    cb_pop_front(SKV4_CB_STAGE2_QUERY,16);
    cb_wait_front(SKV4_CB_LIVE_CONTROL,1);
    const uint32_t total=ckernel::read_tile_value(SKV4_CB_LIVE_CONTROL,0,0);
    cb_pop_front(SKV4_CB_LIVE_CONTROL,1);
    cb_wait_front(SKV4_CB_SELECTOR_INPUT,1);
    sort_half<512,true>(SKV4_CB_SELECTOR_INPUT,SKV4_CB_STAGE2_FRONTIER,total<512?total:512,false);
    if(total>512) {
        cb_wait_front(SKV4_CB_TOPK_SECOND_INPUT,1);
        sort_half<512,true>(SKV4_CB_TOPK_SECOND_INPUT,SKV4_CB_STAGE2_FRONTIER,total-512,false);
    }
}
"""


def compute(*, local, id_bits):
    return (
        header("compute", local=local, id_bits=id_bits)
        + COMPUTE_HEADERS
        + native_helpers()
        + EXPAND
        + TOPK
        + COMPUTE
    )


def reader(*, local, id_bits):
    return header("reader", local=local, id_bits=id_bits) + DATAFLOW + READER


WRITER = r"""
inline uint16_t ordered(uint16_t v) { return v&0x8000u?uint16_t(~v):uint16_t(v^0x8000u); }
inline uint16_t score_at(uint32_t base,uint32_t col,uint32_t head) {
    return *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+head*32+
        (col<16?col*2:512+(col-16)*2));
}
inline uint32_t decode1024(uint32_t w) {
    uint32_t v=w&0xffffu;
    return ((v>>6)&15u)+((v&15u)<<4)+(((v>>10)&1u)<<8)
        +(((v>>4)&1u)<<9)+(((v>>11)&1u)<<10);
}
void kernel_main() {
    constexpr bool DEBUG=get_compile_time_arg_val(SKV4_ARG_RF_CT_DEBUG)!=0;
    constexpr auto oa=TensorAccessorArgs<SKV4_ARG_RF_CT_ACCESSORS>();
    constexpr auto kv=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto da=TensorAccessorArgs<kv.next_compile_time_args_offset()>();
    constexpr auto ds=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sq=TensorAccessorArgs<ds.next_compile_time_args_offset()>();
    constexpr auto qa=TensorAccessorArgs<sq.next_compile_time_args_offset()>();
    constexpr auto ha=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_UNION_QUERY),2048);
    const auto handoff=TensorAccessor(ha,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HANDOFF),META_BYTES);
    const auto debug=TensorAccessor(da,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_DEBUG_INDICES),512);
    const uint32_t slot=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SLOT);
    const uint32_t shard=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SHARD);
    const uint32_t part=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_PART);
    const uint32_t row_x=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_START_X);
    const uint32_t row_y=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_START_Y);
    uint32_t hx[4],hy[4];
    for(uint32_t h=0;h<4;++h) {
        hx[h]=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*h);
        hy[h]=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*h);
    }
    cb_reserve_back(SKV4_CB_MATRIX_QUERY,8);
    for(uint32_t t=0;t<8;++t)
        noc_async_read(get_noc_addr(slot*16+t,q),get_write_ptr(SKV4_CB_MATRIX_QUERY)+t*2048,2048);
    noc_async_read_barrier(); cb_push_back(SKV4_CB_MATRIX_QUERY,8);
    cb_wait_front(SKV4_CB_SOURCE_IDS,1);
    auto meta=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_SOURCE_IDS));
    const uint32_t live=meta[shard],chunks=(live+31)/32;
    const uint32_t first=part*(chunks/4)+(part<chunks%4?part:chunks%4);
    const uint32_t count=chunks/4+(part<chunks%4);
    const uint32_t scratch=get_write_ptr(SKV4_CB_WRITER_SCRATCH);
    const uint32_t staged=get_write_ptr(SKV4_CB_STAGE16_SCORE_STAGING);
    const uint32_t row=scratch+6144;
    for(uint32_t c=0;c<count;++c) {
        cb_wait_front(SKV4_CB_SCORE_TILES,1);
        const uint32_t tile=get_read_ptr(SKV4_CB_SCORE_TILES);
        for(uint32_t h=0;h<4;++h) {
            auto p=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staged+h*QUARTER_CHUNKS*64);
            for(uint32_t j=0;j<32;++j) p[c*32+j]=score_at(tile,j,h);
        }
        cb_pop_front(SKV4_CB_SCORE_TILES,1);
    }
    __sync_synchronize();
    if(count) for(uint32_t h=0;h<4;++h)
        noc_async_write(staged+h*QUARTER_CHUNKS*64,get_noc_addr(row_x+h,row_y,row+first*64),count*64);
    noc_async_write_barrier();
    for(uint32_t h=0;h<4;++h)
        noc_semaphore_inc(get_noc_addr(row_x+h,row_y,sem(SKV4_SEM_STAGE16_SCORES_READY)),1);
    noc_async_atomic_barrier(); await(SKV4_SEM_STAGE16_SCORES_READY,4);
    auto source=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(row);
    cb_reserve_back(SKV4_CB_SELECTOR_INPUT,1);
    auto input=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_SELECTOR_INPUT));
    for(uint32_t j=0;j<1024;++j) input[j]=j<live?source[j]:0xff80u;
    cb_push_back(SKV4_CB_SELECTOR_INPUT,1);
    if(live>1024) {
        cb_reserve_back(SKV4_CB_TOPK_SECOND_INPUT,1);
        auto p=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_TOPK_SECOND_INPUT));
        for(uint32_t j=0;j<1024;++j) p[j]=j+1024<live?source[j+1024]:0xff80u;
        cb_push_back(SKV4_CB_TOPK_SECOND_INPUT,1);
    }
    // Early II query copy is independent of the integer local TopK.
    cb_reserve_back(SKV4_CB_STAGE2_QUERY,16);
    for(uint32_t t=0;t<16;++t)
        noc_async_read(get_noc_addr(slot*16+t,q),get_write_ptr(SKV4_CB_STAGE2_QUERY)+t*2048,2048);
    noc_async_read_barrier(); cb_push_back(SKV4_CB_STAGE2_QUERY,16);
    const uint32_t halves=live>1024?2:1;
    cb_wait_front(SKV4_CB_STAGE16_FRONTIER,halves);
    auto ranked=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_STAGE16_FRONTIER));
    auto packet=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    uint32_t i0=0,i1=0;
    for(uint32_t rank=0;rank<512;++rank) {
        uint32_t word=0xff800000u;
        if(rank<160) {
            const uint32_t w0=i0<1024?ranked[(i0>>6)+((i0&63u)<<4)]:0xff800000u;
            const uint32_t w1=halves==2&&i1<live-1024?
                ranked[1024+(i1>>6)+((i1&63u)<<4)]:0xff800000u;
            const bool take=ordered(w1>>16)>ordered(w0>>16);
            const uint32_t w=take?w1:w0;
            if(take) ++i1; else ++i0;
            word=(w&0xffff0000u)|(shard*LOCAL+decode1024(w));
        }
        const uint32_t at=shard?511-rank:rank;
        packet[((at&31u)<<4)|(at>>5)]=word;
    }
    cb_pop_front(SKV4_CB_STAGE16_FRONTIER,halves);
    __sync_synchronize();
    const uint32_t merge=get_write_ptr(SKV4_CB_STAGE16_MERGE);
    noc_async_write(scratch,get_noc_addr(hx[part],hy[part],merge+shard*4096),2048);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(hx[part],hy[part],sem(SKV4_SEM_STAGE16_FRONTIERS_READY)),1);
    noc_async_atomic_barrier();
    if(part==0) {
        await(SKV4_SEM_STAGE16_FRONTIERS_READY,4);
        auto pages=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(merge);
        const auto fetch=[&](uint32_t s,uint32_t rank)->uint32_t {
            if(rank>=160) return 0xff800000u;
            const uint32_t at=s?511-rank:rank;
            return pages[s*1024+((at&31u)<<4)+(at>>5)];
        };
        uint32_t i0=0,i1=0,i2=0,i3=0;
        uint32_t w0=fetch(0,0),w1=fetch(1,0),w2=fetch(2,0),w3=fetch(3,0);
        uint16_t s0=ordered(w0>>16),s1=ordered(w1>>16);
        uint16_t s2=ordered(w2>>16),s3=ordered(w3>>16);
        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch+10240);
        for(uint32_t j=0;j<16;++j) out[j]=j?0:400;
        for(uint32_t rank=0;rank<400;++rank) {
            const bool a=s1>s0,b=s3>s2;
            const uint16_t sa=a?s1:s0,sb=b?s3:s2;
            const uint32_t winner=sb>sa?(b?3:2):(a?1:0);
            if(winner==0) {
                out[rank+16]=w0&0xffffu; w0=fetch(0,++i0); s0=ordered(w0>>16);
            } else if(winner==1) {
                out[rank+16]=w1&0xffffu; w1=fetch(1,++i1); s1=ordered(w1>>16);
            } else if(winner==2) {
                out[rank+16]=w2&0xffffu; w2=fetch(2,++i2); s2=ordered(w2>>16);
            } else {
                out[rank+16]=w3&0xffffu; w3=fetch(3,++i3); s3=ordered(w3>>16);
            }
            if(rank+1==256) {
                __sync_synchronize();
                noc_async_write(scratch+10240,get_noc_addr(slot*4+shard,handoff),1088);
                noc_async_write_barrier();
                for(uint32_t s=0;s<4;++s) for(uint32_t p=0;p<4;++p)
                    noc_semaphore_inc(get_noc_addr(hx[s]+p,hy[s],sem(SKV4_SEM_HEAD0_READY+shard)),1);
                noc_async_atomic_barrier();
            }
        }
        // The published prefix and the remaining144 positions own disjoint
        // cache lines. No extra scalar stack-to-mailbox copy is needed.
        __sync_synchronize();
        noc_async_write(scratch+10240+1088,get_noc_addr(slot*4+shard,handoff)+1088,576);
        noc_async_write_barrier();
        for(uint32_t s=0;s<4;++s) for(uint32_t p=0;p<4;++p)
            noc_semaphore_inc(get_noc_addr(hx[s]+p,hy[s],sem(SKV4_SEM_HEAD0_READY+shard)),1);
        noc_async_atomic_barrier();
    }
    cb_pop_front(SKV4_CB_SOURCE_IDS,1);
    cb_wait_front(SKV4_CB_UNION_META,1);
    auto tags=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_UNION_META));
    const uint32_t total=tags[0],nchunks=(total+31)/32;
    const uint32_t own_chunks=nchunks/4+(part<nchunks%4);
    const uint32_t own_count=(total/32/4+(part<(total/32)%4))*32+
        (part==(total/32)%4?total%32:0);
    auto temp=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
    for(uint32_t c=0;c<own_chunks;++c) {
        cb_wait_front(SKV4_CB_SCORE_TILES,1);
        const uint32_t tile=get_read_ptr(SKV4_CB_SCORE_TILES);
        for(uint32_t h=0;h<4;++h) for(uint32_t j=0;j<32;++j)
            temp[h*160+c*32+j]=c*32+j<own_count?score_at(tile,j,h):0xff80u;
        cb_pop_front(SKV4_CB_SCORE_TILES,1);
    }
    cb_reserve_back(SKV4_CB_SELECTOR_INPUT,1);
    const uint32_t score_page=get_write_ptr(SKV4_CB_SELECTOR_INPUT);
    auto scores=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(score_page);
    for(uint32_t j=total;j<1024;++j) scores[j]=0xff80u;
    __sync_synchronize();
    for(uint32_t h=0;h<4;++h) for(uint32_t c=0;c<own_chunks;++c)
        noc_async_write(scratch+(h*160+c*32)*2,
            get_noc_addr(row_x+h,row_y,score_page+(c*4+part)*64),64);
    noc_async_write_barrier();
    for(uint32_t h=0;h<4;++h)
        noc_semaphore_inc(get_noc_addr(row_x+h,row_y,sem(SKV4_SEM_STAGE2_SCORES_READY)),1);
    noc_async_atomic_barrier(); await(SKV4_SEM_STAGE2_SCORES_READY,4);
    if(total>512) {
        cb_reserve_back(SKV4_CB_TOPK_SECOND_INPUT,1);
        auto p=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_TOPK_SECOND_INPUT));
        for(uint32_t j=0;j<512;++j) p[j]=scores[512+j];
        cb_push_back(SKV4_CB_TOPK_SECOND_INPUT,1);
    }
    cb_push_back(SKV4_CB_SELECTOR_INPUT,1);
    cb_wait_front(SKV4_CB_STAGE2_FRONTIER,total>512?2:1);
    auto indices=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_STAGE2_FRONTIER));
    uint32_t c0=0,c1=0;
    for(uint32_t rank=0;rank<100;++rank) {
        const uint32_t p0=indices[(c0>>5)+((c0&31u)<<4)];
        const uint32_t p1=total>512?indices[1024+(c1>>5)+((c1&31u)<<4)]+512:0xffffffffu;
        const uint16_t s0=p0<total&&p0<512?ordered(scores[p0]):ordered(0xff80u);
        const uint16_t s1=p1<total?ordered(scores[p1]):ordered(0xff80u);
        const bool second=s1>s0;
        const uint32_t pos=second?p1:p0;
        if(second) ++c1; else ++c0;
        packet[2*rank]=second?s1:s0;
        packet[2*rank+1]=pos<total?tags[1+pos]&ID_MASK:0xffffffffu;
    }
    cb_pop_front(SKV4_CB_STAGE2_FRONTIER,total>512?2:1);
    const uint32_t receive=get_write_ptr(SKV4_CB_STAGE2_EXCHANGE);
    __sync_synchronize();
    noc_async_write(scratch,get_noc_addr(hx[part],hy[part],receive+shard*800),800);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(hx[part],hy[part],sem(SKV4_SEM_STAGE2_FRONTIERS_READY)),1);
    noc_async_atomic_barrier();
    if(part==0) {
        await(SKV4_SEM_STAGE2_FRONTIERS_READY,4);
        auto p=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive);
        uint32_t cursors[4]={};
        auto final=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch+4096);
        for(uint32_t j=0;j<128;++j) final[j]=0xffffffffu;
        for(uint32_t rank=0;rank<100;++rank) {
            uint32_t winner=0;
            for(uint32_t s=1;s<4;++s)
                if(p[s*200+2*cursors[s]]>p[winner*200+2*cursors[winner]]) winner=s;
            final[rank]=p[winner*200+2*cursors[winner]+1]; ++cursors[winner];
        }
        __sync_synchronize();
        noc_async_write(scratch+4096,get_noc_addr(slot*4+shard,handoff),512);
        if constexpr(DEBUG) {
            final[100]=total; __sync_synchronize();
            noc_async_write(scratch+4096,get_noc_addr(slot*4+shard,debug),512);
        }
        noc_async_write_barrier();
#if SKV4_CODEGEN_FUSED_ATTENTION
        // One complete head result, never per-ID messages. Both groups use
        // the SAME program-wide TENSIX semaphore base/offset.
        noc_semaphore_inc(get_noc_addr(
            get_arg_val<uint32_t>(SKV4_ARG_FATT_RT_ATTENTION_X),
            get_arg_val<uint32_t>(SKV4_ARG_FATT_RT_ATTENTION_Y),
            sem(SKV4_SEM_ATTENTION_IDS_READY)),1);
        noc_async_atomic_barrier();
#endif
        for(uint32_t s=0;s<4;++s)
            noc_semaphore_inc(get_noc_addr(hx[s]+shard,hy[s],sem(SKV4_SEM_MERGE_ACK)),1);
        noc_async_atomic_barrier();
    }
    await(SKV4_SEM_MERGE_ACK,1); cb_pop_front(SKV4_CB_UNION_META,1);
}
"""


def writer(*, local, id_bits, attention=False):
    return (
        header("writer", local=local, id_bits=id_bits, attention=attention)
        + DATAFLOW
        + WRITER
    )
