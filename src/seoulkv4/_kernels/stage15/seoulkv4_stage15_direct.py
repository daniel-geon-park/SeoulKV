"""Single-program Stage1.5 with producer-owned L1, no intermediate tensors.

Only final IDs/prefix/live counts survive the program. Optional debug snapshots
are outputs, never inputs to the pipeline. Uniform CB layout makes peer CB
addresses equal; physical coordinates select the owning core.
"""

from seoulkv4._kernels.stage15.abi import Stage15Semaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

from seoulkv4._kernels.stage15.abi import Stage15CB as CB

import os
from seoulkv4._kernels.codegen import stage15 as source_codegen


def native_profile_enabled():
    """Honor the same diagnostic role filter as refinement kernels."""
    selected = os.environ.get("SKV4_NATIVE_PROFILE_ROLE", "")
    return os.environ.get("SKV4_NATIVE_MARKERS") == "1" and selected in ("", "stage15")


COMMON = (
    semaphore_defines(SEM)
    + r"""
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t N=get_compile_time_arg_val(0),W=get_compile_time_arg_val(1);
constexpr uint32_t PAD=get_compile_time_arg_val(2),M=get_compile_time_arg_val(3);
constexpr bool DEBUG=get_compile_time_arg_val(4);
constexpr uint32_t CHUNKS=get_compile_time_arg_val(5),G=4,K=2048,SW=W*G;
constexpr uint32_t BITMAP_DOMAIN=get_compile_time_arg_val(6);
constexpr uint32_t N15=N*SW,BITMAP_BYTES=((BITMAP_DOMAIN+511)/512)*64;
constexpr uint32_t FAST_BITMAP_BYTES=((6*1024+511)/512)*64;
constexpr uint32_t BM_PARTS=4,BM_IDS=M/BM_PARTS,BM_ID_BYTES=((BM_IDS*4+63)/64)*64;
static_assert(M%BM_PARTS==0 && SW>=G*BM_PARTS);
constexpr uint32_t WORDS=N/32,STRIDE=((WORDS*4+123)/64)*64;
constexpr uint32_t POS_OFF=G*STRIDE,COUNT_BYTES=((SW*4+63)/64)*64;
constexpr uint32_t ID_STRIDE=N*4+64,ID_OUT_OFF=N*64,OC_OFF=ID_OUT_OFF+G*ID_STRIDE;
constexpr uint32_t PEERS=3,ADDR=PEERS+SW*2;
constexpr auto qa=TensorAccessorArgs<7>();
constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
constexpr auto ca=TensorAccessorArgs<ka.next_compile_time_args_offset()>();
constexpr auto cla=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
constexpr auto oa=TensorAccessorArgs<cla.next_compile_time_args_offset()>();
constexpr auto pa=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
constexpr auto cta=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
constexpr auto drs=TensorAccessorArgs<cta.next_compile_time_args_offset()>();
constexpr auto dri=TensorAccessorArgs<drs.next_compile_time_args_offset()>();
constexpr auto dls=TensorAccessorArgs<dri.next_compile_time_args_offset()>();
constexpr auto dli=TensorAccessorArgs<dls.next_compile_time_args_offset()>();
constexpr auto dhs=TensorAccessorArgs<dli.next_compile_time_args_offset()>();
constexpr auto dhi=TensorAccessorArgs<dhs.next_compile_time_args_offset()>();
inline uint64_t peer(uint32_t worker,uint32_t addr) {
    return get_noc_addr(get_arg_val<uint32_t>(PEERS+2*worker),
                       get_arg_val<uint32_t>(PEERS+2*worker+1),addr);
}
inline void notify(uint32_t sem,uint32_t owner) {
    noc_semaphore_inc(peer(owner,get_semaphore(sem)),1);
    noc_async_atomic_barrier();
}
inline uint32_t stage15_source_position(uint32_t p,uint32_t unique) {
    if(unique>6*1024) return p;
    if(p>=6*1024) return 0xffffffffu;
    const uint32_t shard=p/1024,local=p%1024;
    const uint32_t global_chunks=(unique+31)/32;
    uint32_t compact=0;
    for(uint32_t q=0;q<G;++q) {
        const uint32_t worker=shard*G+q;
        const uint32_t lo=(global_chunks*worker/SW)*32;
        const uint32_t hi0=(global_chunks*(worker+1)/SW)*32;
        const uint32_t hi=hi0<unique?hi0:unique;
        const uint32_t live=hi>lo?hi-lo:0;
        if(local<compact+live) return worker*N+(local-compact);
        compact+=live;
    }
    return 0xffffffffu;
}
inline void release(uint32_t first,uint32_t count) {
    for(uint32_t w=first;w<first+count;++w)
        noc_semaphore_inc(peer(w,get_semaphore(SKV4_SEM_EPOCH_RELEASE)),1);
    noc_async_atomic_barrier();
}
inline void await_epoch(uint32_t epoch) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_EPOCH_RELEASE)),epoch);
}
inline void slot_barrier(uint32_t expected,uint32_t epoch) {
    if(get_arg_val<uint32_t>(0)==0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_EPOCH_JOIN)),expected);
        release(0,SW);
    }
    await_epoch(epoch);
    if(expected==2*SW+G) {
        /*@S15D_SLOT_RESET_B*/
        if(get_arg_val<uint32_t>(0)==0)
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_EPOCH_JOIN)),0);
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_EPOCH_RELEASE)),0);
        /*@S15D_SLOT_RESET_E*/
    }
}
"""
)


READER = (
    semaphore_defines(SEM)
    + r"""
void kernel_main() {
    const uint32_t worker=get_arg_val<uint32_t>(0),slot=get_arg_val<uint32_t>(1);
    const uint32_t ctx=get_arg_val<uint32_t>(2);
    const auto query=TensorAccessor(qa,get_arg_val<uint32_t>(ADDR),2048);
    const auto packed=TensorAccessor(ka,get_arg_val<uint32_t>(ADDR+1),256);
    const auto candidates=TensorAccessor(ca,get_arg_val<uint32_t>(ADDR+2),N15*4);
    const auto clusters=TensorAccessor(cla,get_arg_val<uint32_t>(ADDR+3),N15);
    const auto ids_out=TensorAccessor(oa,get_arg_val<uint32_t>(ADDR+4),G*M*4);
    constexpr uint32_t PAGES=(M+1023)/1024;
    const auto prefix_out=TensorAccessor(pa,get_arg_val<uint32_t>(ADDR+5),65536);
    const auto counts_out=TensorAccessor(cta,get_arg_val<uint32_t>(ADDR+6),G*4);
    /*@S15D_QUERY_IDS_B*/
    cb_reserve_back(0,8); cb_reserve_back(5,N/32); cb_reserve_back(7,1);
    // Receive directly in the local TopK input, before any peer writes it.
    cb_reserve_back(9,CHUNKS*2);
    const uint32_t qp=get_write_ptr(0),kp=get_write_ptr(5),ip=get_write_ptr(7);
    const uint32_t clusterp=get_write_ptr(3),recv=get_write_ptr(9);
    for(uint32_t t=0;t<8;++t)
        noc_async_read(get_noc_addr(slot*8+t,query),qp+t*2048,2048);
    noc_async_read(get_noc_addr(slot,candidates)+worker*N*4,ip,N*4);
    const uint32_t off=worker*N,skip=off&63u;
    noc_async_read(get_noc_addr(slot,clusters)+off-skip,clusterp,((skip+N+63)/64)*64);
    noc_async_read_barrier();
    cb_push_back(0,8); cb_push_back(7,1);
    /*@S15D_QUERY_IDS_E*/
    /*@S15D_GDDR_B*/
    auto cluster_ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clusterp+skip);
    for(uint32_t c=0;c<N/4;++c)
        noc_async_read(get_noc_addr(slot*(ctx/4)+cluster_ids[c],packed),
                       kp+(c/4)*2048+(c%4)*256,256);
    noc_async_read_barrier(); cb_push_back(5,N/32);
    /*@S15D_GDDR_E*/

    /*@S15D_SCORE_READY_B*/
    if(worker%G==0) {
        auto score_ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_SHARD_SCORES_READY));
        noc_semaphore_wait(score_ready,G);
        // Program descriptors and L1 state persist across dispatches.  This
        // is a per-dispatch equality barrier, not a monotonic global epoch.
        /*@S15D_SCORE_RESET_B*/
        noc_semaphore_set(score_ready,0);
        /*@S15D_SCORE_RESET_E*/
        release(worker,G);
    }
    await_epoch(1);
    cb_push_back(9,CHUNKS*2);
    /*@S15D_SCORE_READY_E*/
    // All native producer frontiers are retained until the head merges finish.
    slot_barrier(SW,2);
    /*@S15D_FRONTIER_READ_B*/
    if(worker<G) {
        for(uint32_t shard=1;shard<W;++shard) {
            cb_reserve_back(12,2); cb_reserve_back(13,2);
            noc_async_read(peer(shard*G+worker,get_read_ptr(10)),get_write_ptr(12),K*2);
            noc_async_read(peer(shard*G+worker,get_read_ptr(11)),get_write_ptr(13),K*4);
            noc_async_read_barrier(); cb_push_back(12,2); cb_push_back(13,2);
        }
    }
    /*@S15D_FRONTIER_READ_E*/
    if(worker>=G && worker<G*BM_PARTS) {
        // One bulk ID packet, one private bitmap, one bulk result packet.
        // Original source ownership below is unchanged by this temporary job.
        auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_BITMAP_INPUT_READY));
        noc_semaphore_wait(ready,1); noc_semaphore_set(ready,0);
        /*@S15D_BM_HELP_FILL_B*/
        const uint32_t part=worker/G,head=worker%G,bp=get_write_ptr(17);
        auto bm=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bp);
        auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(22));
        for(uint32_t i=0;i<BITMAP_BYTES/4;++i) bm[i]=0;
        for(uint32_t i=0;i<BM_IDS;++i) {
            const uint32_t p=ids[i];
            if(p<BITMAP_DOMAIN) bm[p>>5] |= 1u<<(p&31u);
        }
        /*@S15D_BM_HELP_FILL_E*/
        /*@S15D_BM_HELP_SEND_B*/
        __sync_synchronize();
        noc_async_write(bp,peer(head,get_write_ptr(22)+BM_ID_BYTES+(part-1)*BITMAP_BYTES),BITMAP_BYTES);
        noc_async_write_barrier(); notify(SKV4_SEM_BITMAP_PARTS_READY,head);
        /*@S15D_BM_HELP_SEND_E*/
    }
    slot_barrier(SW+G,3);  // all four head-owner bitmaps are ready

    /*@S15D_BITMAP_READ_B*/
    const uint32_t ub=get_write_ptr(18),bitmap=get_write_ptr(17);
    const uint32_t offset=worker*WORDS*4,aligned=offset&~63u,bskip=offset-aligned;
    const uint32_t take=((bskip+WORDS*4+63)/64)*64;
    for(uint32_t h=0;h<G;++h)
        if(h!=worker) noc_async_read(peer(h,bitmap+aligned),ub+h*STRIDE,take);
    noc_async_read_barrier();
    /*@S15D_BITMAP_READ_E*/
    /*@S15D_UNION_ENUM_B*/
    auto selected=reinterpret_cast<tt_l1_ptr uint32_t*>(ub+POS_OFF);
    uint32_t count=0;
    for(uint32_t j=0;j<WORDS;++j) {
        uint32_t word=0;
        for(uint32_t h=0;h<G;++h) {
            const uint32_t at=(h==worker?bitmap+offset:ub+h*STRIDE+bskip)+j*4;
            word |= *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(at);
        }
        for(uint32_t bit=0;bit<32;++bit)
            if(word&(1u<<bit)) selected[count++]=j*32+bit;
    }
    /*@S15D_UNION_ENUM_E*/
    /*@S15D_COUNT_PUBLISH_B*/
    const uint32_t countp=get_write_ptr(19);
    auto counts=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(countp);
    counts[worker]=count;
    __sync_synchronize();
    if(worker) noc_async_write(countp+worker*4,peer(0,countp+worker*4),4);
    noc_async_write_barrier();
    notify(SKV4_SEM_EPOCH_JOIN,0);
    /*@S15D_COUNT_PUBLISH_E*/
    /*@S15D_COUNT_ROOT_WAIT_B*/
    slot_barrier(2*SW+G,4);
    /*@S15D_COUNT_ROOT_WAIT_E*/
    /*@S15D_COUNT_PEER_READ_B*/
    if(worker) { noc_async_read(peer(0,countp),countp,COUNT_BYTES); noc_async_read_barrier(); }
    /*@S15D_COUNT_PEER_READ_E*/

    /*@S15D_COMPACT_B*/
    uint32_t start=0,total=0;
    for(uint32_t w=0;w<SW;++w) { if(w<worker) start+=counts[w]; total+=counts[w]; }
    const uint32_t out=get_write_ptr(20),q=total/G,r=total%G;
    auto token_ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ip);
    // Original gather tiles and IDs, not an exported/re-read compact cache.
    for(uint32_t j=0;j<count;++j) {
        const uint32_t p=selected[j];
        auto src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kp+(p/16)*2048+(p%16)*64);
        auto dst=reinterpret_cast<tt_l1_ptr uint32_t*>(out+j*64);
        #pragma GCC unroll 16
        for(uint32_t word=0;word<16;++word) dst[word]=src[word];
    }
    /*@S15D_COMPACT_E*/
    /*@S15D_DELIVER_B*/
    for(uint32_t owner=0;owner<G;++owner) {
        const uint32_t begin=owner*q+(owner<r?owner:r),live=q+(owner<r),end=begin+live;
        const uint32_t lo=start>begin?start:begin,hi=start+count<end?start+count:end;
        if(lo<hi) {
            const uint32_t src_begin=lo-start,out_begin=lo-begin,take=hi-lo;
            const uint32_t idp=out+ID_OUT_OFF+owner*ID_STRIDE+(out_begin*4&15u);
            auto oi=reinterpret_cast<tt_l1_ptr uint32_t*>(idp);
            for(uint32_t j=0;j<take;++j) oi[j]=token_ids[selected[src_begin+j]];
            __sync_synchronize();
            noc_async_write(idp,get_noc_addr(slot,ids_out)+(owner*M+out_begin)*4,take*4);
            for(uint32_t sent=0;sent<take;) {
                const uint32_t at=out_begin+sent,room=1024-at%1024;
                const uint32_t n=take-sent<room?take-sent:room;
                noc_async_write(out+(src_begin+sent)*64,
                    get_noc_addr((slot*G+owner)*PAGES+at/1024,prefix_out)+(at%1024)*64,n*64);
                sent+=n;
            }
        }
        if(worker==0) reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out+OC_OFF)[owner]=live;
    }
    if(worker==0) {
        __sync_synchronize(); noc_async_write(out+OC_OFF,get_noc_addr(slot,counts_out),G*4);
    }
    noc_async_write_barrier();
    cb_pop_front(7,1);
    /*@S15D_DELIVER_E*/
}
"""
)


WRITER = (
    semaphore_defines(SEM)
    + r"""
void kernel_main() {
    const uint32_t worker=get_arg_val<uint32_t>(0),slot=get_arg_val<uint32_t>(1);
    const uint32_t shard=worker/G,quarter=worker%G;
    const uint32_t base=get_write_ptr(8);
    /*@S15D_SCORE_ASSEMBLE_B*/
    for(uint32_t chunk=0;chunk<N/32;++chunk) {
        /*@S15D_SCORE_WAIT_B*/
        cb_wait_front(16,1);
        /*@S15D_SCORE_WAIT_E*/
        /*@S15D_SCORE_COPY_B*/
        auto tile=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(16));
        for(uint32_t h=0;h<G;++h) {
            auto row=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                h==quarter?get_write_ptr(9)+quarter*N*2:base+h*N*2);
            for(uint32_t c=0;c<32;++c)
                row[chunk*32+c]=tile[h*16+(c<16?c:256+c-16)];
        }
        /*@S15D_SCORE_COPY_E*/
        cb_pop_front(16,1);
    }
    /*@S15D_SCORE_ASSEMBLE_E*/
    /*@S15D_SCORE_ROUTE_B*/
    for(uint32_t h=0;h<G;++h)
        if(h!=quarter) noc_async_write(base+h*N*2,peer(shard*G+h,get_write_ptr(9)+quarter*N*2),N*2);
    __sync_synchronize();
    noc_async_write_barrier();
    notify(SKV4_SEM_SHARD_SCORES_READY,shard*G);
    /*@S15D_SCORE_ROUTE_E*/
    if constexpr(DEBUG) {
        // Optional snapshots are write-only diagnostics, never the transport.
        const auto ds=TensorAccessor(drs,get_arg_val<uint32_t>(ADDR+7),PAD*2);
        const auto di=TensorAccessor(dri,get_arg_val<uint32_t>(ADDR+8),PAD*4);
        const uint32_t tmp=get_write_ptr(21),tail=PAD-G*N;
        auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tmp);
        for(uint32_t i=0;i<N;++i) ids[i]=worker*N+i;
        for(uint32_t i=N;i<N+tail;++i) ids[i]=0xffffffffu;
        auto pad=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp+(N+tail)*4);
        for(uint32_t i=0;i<tail;++i) pad[i]=0xff80u;
        for(uint32_t h=0;h<G;++h) {
            const uint32_t row=(slot*W+shard)*G+h;
            const uint32_t src=h==quarter?get_write_ptr(9)+quarter*N*2:base+h*N*2;
            noc_async_write(src,get_noc_addr(row,ds)+quarter*N*2,N*2);
            noc_async_write(tmp,get_noc_addr(row,di)+quarter*N*4,(N+(quarter==3?tail:0))*4);
            if(quarter==3 && tail) noc_async_write(tmp+(N+tail)*4,get_noc_addr(row,ds)+G*N*2,tail*2);
        }
        noc_async_write_barrier();
    }
    cb_wait_front(10,2); cb_wait_front(11,2);
    if constexpr(DEBUG) {
        const auto ds=TensorAccessor(dls,get_arg_val<uint32_t>(ADDR+9),W*K*2);
        const auto di=TensorAccessor(dli,get_arg_val<uint32_t>(ADDR+10),W*K*4);
        const uint32_t row=slot*G+quarter;
        noc_async_write(get_read_ptr(10),get_noc_addr(row,ds)+shard*K*2,K*2);
        noc_async_write(get_read_ptr(11),get_noc_addr(row,di)+shard*K*4,K*4);
        noc_async_write_barrier();
    }
    notify(SKV4_SEM_EPOCH_JOIN,0);
    if(worker<G) {
        cb_wait_front(15,1);
        if constexpr(DEBUG) {
            cb_wait_front(14,1);
            const auto ds=TensorAccessor(dhs,get_arg_val<uint32_t>(ADDR+11),K*2);
            const auto di=TensorAccessor(dhi,get_arg_val<uint32_t>(ADDR+12),K*4);
            noc_async_write(get_read_ptr(14),get_noc_addr(slot*G+worker,ds),K*2);
            noc_async_write(get_read_ptr(15),get_noc_addr(slot*G+worker,di),K*4);
            noc_async_write_barrier();
            cb_pop_front(14,1);
        }
        /*@S15D_BITMAP_BUILD_B*/
        /*@S15D_BM_ID_BULK_B*/
        const uint32_t idp=get_read_ptr(15),scratch=get_write_ptr(22);
        for(uint32_t part=1;part<BM_PARTS;++part)
            noc_async_write(idp+part*BM_IDS*4,peer(part*G+worker,scratch),BM_IDS*4);
        noc_async_write_barrier();
        for(uint32_t part=1;part<BM_PARTS;++part) notify(SKV4_SEM_BITMAP_INPUT_READY,part*G+worker);
        /*@S15D_BM_ID_BULK_E*/
        /*@S15D_BM_ROOT_BUILD_B*/
        const uint32_t bp=get_write_ptr(17);
        auto bitmap=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bp);
        auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idp);
        for(uint32_t i=0;i<BITMAP_BYTES/4;++i) bitmap[i]=0;
        for(uint32_t i=0;i<BM_IDS;++i) {
            const uint32_t p=ids[i];
            if(p<BITMAP_DOMAIN) bitmap[p>>5] |= 1u<<(p&31u);
        }
        /*@S15D_BM_ROOT_BUILD_E*/
        /*@S15D_BM_JOIN_B*/
        auto done=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_BITMAP_PARTS_READY));
        noc_semaphore_wait(done,BM_PARTS-1); noc_semaphore_set(done,0);
        /*@S15D_BM_JOIN_E*/
        /*@S15D_BM_OR_B*/
        auto partial=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch+BM_ID_BYTES);
        for(uint32_t i=0;i<BITMAP_BYTES/4;++i)
            bitmap[i] |= partial[i] | partial[BITMAP_BYTES/4+i] | partial[2*(BITMAP_BYTES/4)+i];
        /*@S15D_BM_OR_E*/
        __sync_synchronize();
        cb_pop_front(15,1);
        /*@S15D_BITMAP_BUILD_E*/
        notify(SKV4_SEM_EPOCH_JOIN,0);
    }
    await_epoch(3); // every merge has consumed the retained native frontiers
    cb_pop_front(10,2); cb_pop_front(11,2);
}
"""
)


def compute_source(n, workers, debug_snapshots=False):
    from seoulkv4._kernels.stage15.abi import STAGE15_COMPUTE_CT

    return source_codegen.compute(
        n=n, profile=native_profile_enabled()
    ), STAGE15_COMPUTE_CT.pack(None, tensors={}, n=n, phase=2, k=2048, workers=workers)


def build_direct(
    tt,
    device,
    query,
    packed,
    candidates,
    clusters,
    *,
    groups,
    slots,
    context,
    n15,
    workers,
    pad,
    union_m15,
    debug_snapshots=False,
    cluster_ingest,
):
    from seoulkv4._kernels.stage2.seoulkv4_stage2_fused import _ranges

    if cluster_ingest.get("kind") != "distributed":
        raise ValueError("Stage1.5 requires distributed cluster ingestion")
    n = n15 // (workers * 4)
    local = n * 4
    chunks = (local + 2047) // 2048
    bitmap_domain = n15
    points = [p for g in groups for p in g]
    cores = _ranges(tt, points)

    def alloc(rows, cols, dtype):
        return tt.allocate_tensor_on_device(
            tt.Shape((1, 1, rows, cols)),
            dtype,
            tt.ROW_MAJOR_LAYOUT,
            device,
            tt.L1_MEMORY_CONFIG,
        )

    ids = alloc(slots, 4 * union_m15, tt.uint32)
    prefix = alloc(slots * 4 * ((union_m15 + 1023) // 1024), 65536, tt.uint8)
    counts = alloc(slots, 4, tt.uint32)
    shapes = [
        (slots * workers * 4, pad, tt.bfloat16),
        (slots * workers * 4, pad, tt.uint32),
        (slots * 4, workers * 2048, tt.bfloat16),
        (slots * 4, workers * 2048, tt.uint32),
        (slots * 4, 2048, tt.bfloat16),
        (slots * 4, 2048, tt.uint32),
    ]
    debug = [alloc(*s) for s in shapes] if debug_snapshots else []
    io = [query, packed, candidates, clusters, ids, prefix, counts, *debug]
    from seoulkv4._kernels.stage15.abi import (
        INGEST_IO,
        STAGE15_COMPUTE_RT,
        stage15_abi,
    )

    runtime_abi, compile_abi = stage15_abi(workers, debug=debug_snapshots)
    io_names = compile_abi.tensors
    named_io = dict(zip(io_names, io))
    ct = compile_abi.pack(
        tt,
        tensors=named_io,
        n=n,
        workers=workers,
        pad=pad,
        m=union_m15,
        debug=int(debug_snapshots),
        chunks=chunks,
        bitmap_domain=bitmap_domain,
    )
    rt, cr = tt.RuntimeArgs(), tt.RuntimeArgs()
    for slot, group in enumerate(groups):
        physical = [
            device.worker_core_from_logical_core(tt.CoreCoord(*p)) for p in group
        ]
        for first in range(0, len(physical), 4):
            quad = physical[first : first + 4]
            area = (max(p.x for p in quad) - min(p.x for p in quad) + 1) * (
                max(p.y for p in quad) - min(p.y for p in quad) + 1
            )
            if len({(p.x, p.y) for p in quad}) != 4 or area != 4:
                raise ValueError("shared query requires four-core physical rectangles")
        peers = [v for c in physical for v in (c.x, c.y)]
        for worker, (x, y) in enumerate(group):
            addresses = {name: t.buffer_address() for name, t in named_io.items()}
            rt[x][y] = runtime_abi.pack(
                worker=worker, slot=slot, context=context, peers=peers, **addresses
            )
            cr[x][y] = STAGE15_COMPUTE_RT.pack(
                worker=worker,
                chunks=chunks,
                tail=local - (chunks - 1) * 2048,
                shard=worker // 4,
                live=local,
            )
    compute, cct = compute_source(n, workers, debug_snapshots)

    def kernel(src, config, cta, rta):
        return tt.KernelDescriptor(
            kernel_source=src,
            source_type=tt.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=cta,
            runtime_args=rta,
            config=config,
        )

    cbs = []
    scratch_bytes = 0

    def cb(i, dtype, size, pages=1, *, tile=None):
        nonlocal scratch_bytes
        scratch_bytes += size * pages
        fmt = dict(buffer_index=i, data_format=dtype, page_size=size)
        if tile is not None:
            fmt["tile"] = tile
        cbs.append(
            tt.CBDescriptor(
                total_size=size * pages,
                core_ranges=cores,
                format_descriptors=[tt.CBFormatDescriptor(**fmt)],
            )
        )

    cb(CB.QUERY, tt.bfloat16, 2048, 8)
    cb(CB.DECODED_KEYS, tt.bfloat16, 2048, 8)
    from seoulkv4._kernels.stage15.stage15_overlap import gather_batch

    profile = native_profile_enabled()
    writer_source = source_codegen.writer(profile=profile, debug=debug_snapshots)
    cb(CB.QUERY_CONTROL, tt.uint8, ((n + 123) // 64) * 64)
    cb(
        CB.PACKED_PREFIX,
        tt.uint32,
        2048,
        n // 32,
        tile=tt.TileDescriptor(height=32, width=16),
    )
    cb(CB.TOKEN_IDS, tt.uint32, n * 4)
    cb(CB.HEAD_SCORES, tt.uint8, 4 * n * 2)
    cb(CB.LOCAL_SCORE_TILES, tt.bfloat16, 2048, chunks * 2)
    cb(CB.FRONTIER_VALUES, tt.bfloat16, 2048, 2)
    cb(CB.FRONTIER_INDICES, tt.uint32, 4096, 2)
    cb(CB.PEER_VALUES, tt.bfloat16, 2048, workers)
    cb(CB.PEER_INDICES, tt.uint32, 4096, workers)
    cb(CB.MERGED_VALUES, tt.bfloat16, 4096)
    cb(CB.MERGED_INDICES, tt.uint32, 8192)
    cb(CB.SCORE_TILES, tt.bfloat16, 2048, 2)
    bitmap_bytes = ((bitmap_domain + 511) // 512) * 64
    stride = ((n // 32 * 4 + 123) // 64) * 64
    cb(CB.UNION_BITMAP, tt.uint32, bitmap_bytes)
    cb(CB.UNION_LAYOUT, tt.uint8, 4 * stride + n * 4)
    cb(CB.UNION_CONTROL, tt.uint32, ((workers * 16 + 63) // 64) * 64)
    cb(CB.UNION_RECORDS, tt.uint8, n * 64 + 4 * (n * 4 + 64) + 64)
    # Aligned helper ID receive area + three disjoint partial-bitmap packets.
    cb(CB.BITMAP_EXCHANGE, tt.uint32, ((union_m15 + 63) // 64) * 64 + 3 * bitmap_bytes)
    if debug_snapshots:
        tail = pad - local
        cb(CB.DEBUG_SCRATCH, tt.uint8, ((4 * (n + tail) + tail * 2 + 63) // 64) * 64)
    readers = []
    ingest_sem = SEM.INGEST_JOIN
    cb(
        CB.INGEST_SCRATCH,
        tt.uint8,
        max(part["scratch"] for part in cluster_ingest["parts"]),
    )
    cb(CB.COMPUTE_CONTROL, tt.uint32, 64)
    cb(CB.WRITER_CONTROL, tt.uint32, 64)
    cb(CB.QUERY_READY, tt.uint32, 64)
    cb(CB.COMPACT_TOPK_INPUT, tt.bfloat16, 2048)
    cb(CB.TILE_LAYOUT_SCRATCH, tt.uint8, 8192)
    # Keep existing CB addresses symmetric, including the extra scratch.
    fused_runtime = runtime_abi.then(INGEST_IO.prefixed("ingest"))
    rt_offset = fused_runtime.offset("ingest_fine")
    for slot, (group, part) in enumerate(zip(groups, cluster_ingest["parts"])):
        ict = list(ct) + part["compile"]
        irt = tt.RuntimeArgs()
        for x, y in group:
            # Both fragments have already been checked by their named packers.
            irt[x][y] = list(rt[x][y]) + INGEST_IO.pack(
                **{name: t.buffer_address() for name, t in part["tensors"].items()}
            )
            assert len(irt[x][y]) == fused_runtime.words
        source = source_codegen.reader(
            profile=profile,
            ct_offset=len(ct),
            rt_offset=rt_offset,
            ingest_sem=ingest_sem,
            debug=debug_snapshots,
        )
        readers.append(
            tt.KernelDescriptor(
                kernel_source=source,
                source_type=tt.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=_ranges(tt, group),
                compile_time_args=ict,
                runtime_args=irt,
                config=tt.ReaderConfigDescriptor(),
            )
        )
        io.extend(part["io"])
    descriptor = tt.ProgramDescriptor(
        kernels=[
            *readers,
            kernel(
                compute,
                tt.ComputeConfigDescriptor(
                    math_fidelity=tt.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=True,
                ),
                cct,
                cr,
            ),
            kernel(writer_source, tt.WriterConfigDescriptor(), ct, rt),
        ],
        cbs=cbs,
        semaphores=[
            tt.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0)
            for i in SEM
        ],
    )

    def run():
        tt.generic_op(io, descriptor)
        return ids, None

    snapshots = debug if debug else [None] * 6
    return dict(
        run=run,
        ids=ids,
        scores=None,
        prefix2=prefix,
        union_counts=counts,
        descriptor=descriptor,
        io=io,
        program_count=1,
        cluster_ingest_fused=True,
        raw_scores=snapshots[0],
        raw_ids=snapshots[1],
        list_scores=snapshots[2],
        list_ids=snapshots[3],
        head_scores=snapshots[4],
        head_positions=snapshots[5],
        local_topk=None,
        topk=None,
        union=dict(ids=ids, prefix2=prefix, counts=counts),
        stage15_head_frontier_layout="runtime-compact-pair-or-source-position",
        stage15_bitmap_domain=bitmap_domain,
        stage15_candidates=n15,
        stage15_per_head=2048,
        stage15_union_per_head=union_m15,
        stage15_cluster_size=4,
        stage15_cluster_fetch_bytes=256,
        stage15_keys_per_packed_tile=32,
        stage15_compact_query_tiles=8,
        stage15_aggregate_centering=False,
        stage15_record_layout="narrow-key-major-v1",
        stage15_conversion_epoch=None,
        stage15_conversion_batch=1,
        stage15_early_compact=True,
        stage15_head_ready=True,
        stage15_shared_query=True,
        stage15_gather_batch=gather_batch(),
        stage15_frontier_ready=False,
        stage15_early_bitmap=True,
        stage15_scoring="compact narrow-tile SFPU BF16 + four-head matmul",
        stage15_gddr_policy="stage-boundary only; original CB input retained",
        stage15_prefix_delivery="source-owned direct CB to balanced L1 output",
        stage15_route_l1_bytes=0,
        stage15_packed_cache_l1_bytes=0,
        stage15_candidate_cache_l1_bytes=0,
        stage15_scorer_shards=workers * 4,
        stage15_shards=workers,
        worker_groups=groups,
        spatial_workers_per_slot=workers * 4,
        stage15_handoff="source-owned union; balanced live output",
        stage15_dynamic_input_count=True,
        stage15_worker_capacity=n,
        stage16_candidates=4 * union_m15,
        debug_snapshots=debug_snapshots,
        persistent_tensor_count=3,
        snapshot_tensor_count=len(debug),
        scratch_bytes_per_core=scratch_bytes,
        retained_input_bytes_per_core=n * (128 + 4),
    )
