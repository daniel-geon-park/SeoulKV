"""Explicit Stage1.5 dataflow, matrix, TopK and tree phases. No rendered-source edits."""

from seoulkv4._kernels.stage15.abi import Stage15Semaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

from seoulkv4._kernels.stage15.abi import Stage15CB as CB
from seoulkv4._kernels.common.cb_layout import cpp_defines
from seoulkv4._kernels.stage15.abi import StandaloneIngestSemaphore
from seoulkv4._kernels.stage15.abi import StandaloneIngestCB

from seoulkv4._kernels.codegen.preprocessor import profile_defines
from seoulkv4._kernels.stage15.abi import stage15_defines, ingest_defines


def reader(*, profile, ct_offset, rt_offset, ingest_sem, debug=True):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + stage15_defines("reader", ct_offset=ct_offset, rt_offset=rt_offset)
        + profile_defines(profile=profile, debug=bool(debug))
        + (
            r"""#if SKV4_PROFILE
#include "tools/profiler/kernel_profiler.hpp"
#endif
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t N=get_compile_time_arg_val(SKV4_ARG_S15_CT_N),W=get_compile_time_arg_val(SKV4_ARG_S15_CT_WORKERS);
constexpr uint32_t PAD=get_compile_time_arg_val(SKV4_ARG_S15_CT_PAD),M=get_compile_time_arg_val(SKV4_ARG_S15_CT_M);
constexpr bool DEBUG=get_compile_time_arg_val(SKV4_ARG_S15_CT_DEBUG);
constexpr uint32_t CHUNKS=get_compile_time_arg_val(SKV4_ARG_S15_CT_CHUNKS),G=4,K=2048,SW=W*G;
constexpr uint32_t BITMAP_DOMAIN=get_compile_time_arg_val(SKV4_ARG_S15_CT_BITMAP_DOMAIN);
constexpr uint32_t N15=N*SW,BITMAP_BYTES=((BITMAP_DOMAIN+511)/512)*64;
constexpr uint32_t FAST_BITMAP_BYTES=((6*1024+511)/512)*64;
constexpr uint32_t BM_PARTS=4,BM_IDS=M/BM_PARTS,BM_ID_BYTES=((BM_IDS*4+63)/64)*64;
static_assert(M%BM_PARTS==0 && SW>=G*BM_PARTS);
constexpr uint32_t WORDS=N/32,STRIDE=((WORDS*4+123)/64)*64;
constexpr uint32_t POS_OFF=G*STRIDE,COUNT_BYTES=((SW*4+63)/64)*64;
constexpr uint32_t ID_STRIDE=N*4+64,ID_OUT_OFF=N*64,OC_OFF=ID_OUT_OFF+G*ID_STRIDE;
constexpr uint32_t PEERS=SKV4_ARG_S15_RT_END,ADDR=PEERS+SW*2;
constexpr auto qa=TensorAccessorArgs<SKV4_ARG_S15_CT_ACCESSORS>();
constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
constexpr auto ca=TensorAccessorArgs<ka.next_compile_time_args_offset()>();
constexpr auto cla=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
constexpr auto oa=TensorAccessorArgs<cla.next_compile_time_args_offset()>();
constexpr auto pa=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
constexpr auto cta=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
#if SKV4_CODEGEN_DEBUG
constexpr auto drs=TensorAccessorArgs<cta.next_compile_time_args_offset()>();
constexpr auto dri=TensorAccessorArgs<drs.next_compile_time_args_offset()>();
constexpr auto dls=TensorAccessorArgs<dri.next_compile_time_args_offset()>();
constexpr auto dli=TensorAccessorArgs<dls.next_compile_time_args_offset()>();
constexpr auto dhs=TensorAccessorArgs<dli.next_compile_time_args_offset()>();
constexpr auto dhi=TensorAccessorArgs<dhs.next_compile_time_args_offset()>();
#endif
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
    if(get_arg_val<uint32_t>(SKV4_ARG_S15_RT_WORKER)==0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_EPOCH_JOIN)),expected);
        release(0,SW);
    }
    await_epoch(epoch);
    if(expected==2*SW+G) {
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SLOT_RESET_B", 0);
#endif
        if(get_arg_val<uint32_t>(SKV4_ARG_S15_RT_WORKER)==0)
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_EPOCH_JOIN)),0);
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_EPOCH_RELEASE)),0);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SLOT_RESET_E", 0);
#endif
    }
}
namespace distributed_input {
constexpr uint32_t NC = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_NC), CTX = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_CONTEXT);
constexpr uint32_t CAP = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_CAPACITY), FW = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_FINE_WIDTH);
constexpr uint32_t SLOT = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_SLOT), MINIMUM = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_MINIMUM);
constexpr uint32_t PAGE = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_PAGE_BYTES), PAGES = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_PAGES);
constexpr uint32_t BLOCK = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_BLOCK_BYTES), NW = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_WORKERS);
constexpr uint32_t LOCAL = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_LOCAL), BATCH = 32;
constexpr bool DIAGNOSTICS = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_DIAGNOSTICS);
constexpr bool DIRECT_CB = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_DIRECT_CB);
constexpr uint32_t WORKER_CAP = CAP / NW;
static_assert(CAP % NW == 0 && WORKER_CAP % 32 == 0);
constexpr uint32_t IDCAP = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_POSTING_CAPACITY);
constexpr uint32_t IDBYTES = ((IDCAP * 4 + 63) / 64) * 64;
constexpr uint32_t MW = ((NC + 31) / 32 + NW * 16 - 1) / (NW * 16) * 16, COUNT_BYTES = NW * 32;
constexpr uint32_t OWNER_KEYS = ((CAP / 64 + NW - 1) / NW) * 64;
constexpr uint32_t ITMP = ((OWNER_KEYS * 4 + 16 + 63) / 64) * 64;
constexpr uint32_t JOIN = """
            + str(ingest_sem)
            + r""", RELEASE = """
            + str(ingest_sem)
            + r""" + 1;
constexpr auto fa = TensorAccessorArgs<SKV4_ARG_INGEST_CT_ACCESSORS>();
constexpr auto da = TensorAccessorArgs<fa.next_compile_time_args_offset()>();
constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
constexpr auto ia = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
constexpr auto pa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
constexpr auto sa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
inline uint64_t ip_peer(uint32_t w, uint32_t addr) {
    return peer(w, addr);
}
inline void arrive() {
    noc_semaphore_inc(ip_peer(0, get_semaphore(JOIN)), 1);
    noc_async_atomic_barrier();
}
inline void root_wait(uint32_t epoch) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(JOIN)),
                           epoch * NW);
}
inline void publish() {
    for (uint32_t w = 0; w < NW; ++w)
        noc_semaphore_inc(ip_peer(w, get_semaphore(RELEASE)), 1);
    noc_async_atomic_barrier();
}
inline void await_phase(uint32_t epoch) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(RELEASE)),
                           epoch);
}
__attribute__((noinline)) void run() {
    uint64_t start = 0;
    if constexpr (DIAGNOSTICS)
        start = c_tensix_core::read_wall_clock();
    const auto fine = TensorAccessor(fa, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_FINE), 64);
    const auto descriptors = TensorAccessor(da, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_DESCRIPTORS), 64);
    const auto blocks = TensorAccessor(ba, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_BLOCKS), PAGE);
    const auto ids_out = TensorAccessor(ia, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_IDS_OUT), 64);
    const auto prefix = TensorAccessor(pa, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_PREFIX), 4096);
    const auto status = TensorAccessor(sa, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_STATUS), 64);
    const uint32_t worker = get_arg_val<uint32_t>(SKV4_ARG_S15_RT_WORKER), base = get_write_ptr(SKV4_CB_INGEST_SCRATCH);
    const uint32_t dp = base + FW * 4, rp = dp + LOCAL * 64;
    const uint32_t compactp = rp + LOCAL * BLOCK, ip = compactp + IDCAP * 64,
                   xp = ip + IDBYTES;
    const uint32_t bp = xp + IDBYTES, cp = bp + MW * NW * 4, gp = cp + COUNT_BYTES;
    const uint32_t np = gp;
    const uint32_t op = np + (DIRECT_CB ? 0 : OWNER_KEYS * 64);
    const uint32_t tp = op + (DIRECT_CB ? 0 : OWNER_KEYS * 4);
    const uint32_t sp = tp + ITMP;
    auto f = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(base);
    auto d = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(dp);
    auto ids = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(ip);
    auto positions = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(xp);
    auto bitmap = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(bp);
    auto counts = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(cp);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_SETUP_B", 0);
#endif
    noc_async_read(get_noc_addr(0, fine), base, 64);
    noc_async_read_barrier();
    const uint32_t n = f[0];
    uint32_t error = f[1] ? 1 : 0;
    if (n > FW - 16)
        error = 2;
    if (!error)
        for (uint32_t t = 1; t < (16 + n + 15) / 16; ++t)
            noc_async_read(get_noc_addr(t, fine), base + t * 64, 64);
    noc_async_read_barrier();
    const uint32_t first = error ? 0 : n * worker / NW, last = error ? 0 : n * (worker + 1) / NW;
    if (!error)
        for (uint32_t i = first; i < last; ++i) {
            if (f[16 + i] >= NC)
                error = 3;
            if constexpr (DIAGNOSTICS)
                if (i && f[16 + i] <= f[15 + i])
                    error = 3;
        }
    const uint32_t count = error ? 0 : last - first;
    uint32_t cursor = 0, hi = error ? 0 : n;
    while (cursor < hi) {
        const uint32_t mid = (cursor + hi) / 2;
        if ((f[16 + mid] >> 5) < worker * MW)
            cursor = mid + 1;
        else
            hi = mid;
    }
    for (uint32_t word = worker * MW; word < (worker + 1) * MW; ++word) {
        uint32_t bits = 0;
        while (!error && cursor < n && (f[16 + cursor] >> 5) == word) {
            bits |= 1u << (f[16 + cursor] & 31);
            ++cursor;
        }
        bitmap[word] = bits;
    }
    __sync_synchronize();
    if (worker)
        noc_async_write(bp + worker * MW * 4, ip_peer(0, bp + worker * MW * 4), MW * 4);
    noc_async_write_barrier();
    arrive();
    if (!worker) {
        root_wait(1);
        for (uint32_t w = 1; w < NW; ++w)
            noc_async_write(bp, ip_peer(w, bp), MW * NW * 4);
        noc_async_write_barrier();
        publish();
    }
    await_phase(1);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_SETUP_E", 0);
    DeviceTimestampedData("SKV_DI_FETCH_B", 0);
    DeviceTimestampedData("SKV_DI_DESCRIPTORS_B", 0);
#endif
    for (uint32_t off = 0; off < count; off += BATCH) {
        const uint32_t end = off + BATCH < count ? off + BATCH : count;
        for (uint32_t i = off; i < end; ++i)
            noc_async_read(get_noc_addr(f[16 + first + i] / 4, descriptors), dp + i * 64, 64);
        noc_async_read_barrier();
    }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_DESCRIPTORS_E", 0);
    DeviceTimestampedData("SKV_DI_GDDR_BLOCKS_B", 0);
#endif
    uint32_t read_bytes = 0, requests = 0, postings = 0;
    for (uint32_t off = 0; off < count; off += BATCH) {
        const uint32_t end = off + BATCH < count ? off + BATCH : count;
        for (uint32_t i = off; i < end; ++i) {
            auto m = d + i * 16 + (f[16 + first + i] % 4) * 4;
            const uint32_t pg = m[0], at = m[1], nk = m[2], size = m[3];
            if constexpr (DIAGNOSTICS)
                if (pg >= PAGES || at % 64 || at > PAGE || size > PAGE - at || size > BLOCK ||
                    nk > CTX || size != ((nk * 8 + 63) / 64) * 64 + nk * 64) {
                    error = 6;
                    continue;
                }
            if (!nk)
                continue;
            noc_async_read(get_noc_addr(pg, blocks) + at, rp + i * BLOCK, size);
            if constexpr (DIAGNOSTICS) {
                read_bytes += size;
                ++requests;
                postings += nk;
            }
        }
        noc_async_read_barrier();
    }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_GDDR_BLOCKS_E", 0);
    DeviceTimestampedData("SKV_DI_FETCH_E", 0);
    DeviceTimestampedData("SKV_DI_COMPACT_B", 0);
#endif
    uint32_t live = 0, compact_requests = 0;
    if (!error)
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t cid = f[16 + first + i];
            auto m = d + i * 16 + (cid % 4) * 4;
            const uint32_t nk = m[2], meta = ((nk * 8 + 63) / 64) * 64, begin = live;
            auto tokens = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(rp + i * BLOCK);
            // Read ALL metadata of this block before in-place compaction can overwrite it.
            for (uint32_t j = 0; j < nk; ++j) {
                const uint32_t token = tokens[2 * j], mate = tokens[2 * j + 1];
                if constexpr (DIAGNOSTICS)
                    if (token >= CTX || (mate != 0xffffffffu && mate >= NC) || mate == cid) {
                        error = 3;
                        continue;
                    }
                if (mate > cid || !(bitmap[mate >> 5] & (1u << (mate & 31)))) {
                    ids[live] = token;
                    positions[live] = j;
                    ++live;
                }
            }
            for (uint32_t k = begin; k < live;) {
                const uint32_t posting = positions[k];
                uint32_t run = 1;
                while (k + run < live && positions[k + run] == posting + run)
                    ++run;
                // Source blocks remain immutable until the single final join.
                // Coalesce adjacent records rather than issuing one NoC per key.
                noc_async_read(ip_peer(worker, rp + i * BLOCK + meta + posting * 64),
                               compactp + k * 64, run * 64);
                k += run;
                if (++compact_requests % BATCH == 0)
                    noc_async_read_barrier();
            }
        }
    noc_async_read_barrier();
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_COMPACT_E", 0);
    DeviceTimestampedData("SKV_DI_COUNTS_B", 0);
#endif
    for (uint32_t i = 0; i < (DIAGNOSTICS ? 8u : 4u); ++i)
        counts[worker * 8 + i] = 0;
    counts[worker * 8] = live;
    counts[worker * 8 + 1] = error;
    if constexpr (DIAGNOSTICS) {
        counts[worker * 8 + 4] = postings;
        counts[worker * 8 + 5] = read_bytes;
        counts[worker * 8 + 6] = requests;
    }
    __sync_synchronize();
    if (worker)
        noc_async_write(cp + worker * 32, ip_peer(0, cp + worker * 32), 32);
    noc_async_write_barrier();
    arrive();
    if (!worker) {
        root_wait(2);
        uint32_t sum = 0, err = 0;
        for (uint32_t w = 0; w < NW; ++w) {
            counts[w * 8 + 2] = sum;
            sum += counts[w * 8];
            err |= counts[w * 8 + 1];
        }
        if (!err) {
            if (sum > CAP)
                err = 4;
            else if (sum < MINIMUM)
                err = 5;
        }
        for (uint32_t w = 0; w < NW; ++w) {
            counts[w * 8 + 1] = err;
            counts[w * 8 + 3] = sum;
        }
        __sync_synchronize();
        for (uint32_t w = 1; w < NW; ++w)
            noc_async_write(cp, ip_peer(w, cp), COUNT_BYTES);
        noc_async_write_barrier();
        publish();
    }
    await_phase(2);
    error = counts[worker * 8 + 1];
    const uint32_t unique = counts[worker * 8 + 3];
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_COUNTS_E", 0);
    DeviceTimestampedData("SKV_DI_GATHER_B", 0);
#endif
    const uint32_t live_chunks = error ? 0 : (unique + 31) / 32;
    const uint32_t ch0 = live_chunks * worker / NW,
                   ch1 = live_chunks * (worker + 1) / NW;
    const uint32_t lo = ch0 * 32,
                   hi_out = ch1 * 32 < unique ? ch1 * 32 : unique;
    const uint32_t active = error ? 0 : hi_out - lo;
    static_assert(DIRECT_CB, "narrow Stage1.5 requires direct handoff");
    cb_reserve_back(SKV4_CB_TOKEN_IDS, 1);
    const uint32_t own_idp = get_write_ptr(SKV4_CB_TOKEN_IDS);
    auto own_ids = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(own_idp);
    for (uint32_t k = 0; k < WORKER_CAP; ++k)
        own_ids[k] = 0xffffffffu;
    const uint32_t publish_chunks = active ? (active + 31) / 32 : 1;
    cb_reserve_back(SKV4_CB_PACKED_PREFIX, publish_chunks);
    const uint32_t direct_base = get_write_ptr(SKV4_CB_PACKED_PREFIX);
    f[2] = direct_base;
    // Publish scalar controls before the data pages. The compute kernel can
    // wait on CB25 while this NCRISC performs the bulk peer reads.
    cb_push_back(SKV4_CB_TOKEN_IDS, 1);
    cb_reserve_back(SKV4_CB_COMPUTE_CONTROL, 1);
    cb_reserve_back(SKV4_CB_WRITER_CONTROL, 1);
    auto live_compute = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
        get_write_ptr(SKV4_CB_COMPUTE_CONTROL));
    auto live_writer = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
        get_write_ptr(SKV4_CB_WRITER_CONTROL));
    live_compute[0] = active;
    live_compute[1] = publish_chunks;
    live_compute[2] = unique;
    live_writer[0] = active;
    live_writer[1] = publish_chunks;
    live_writer[2] = unique;
    __sync_synchronize();
    cb_push_back(SKV4_CB_COMPUTE_CONTROL, 1);
    cb_push_back(SKV4_CB_WRITER_CONTROL, 1);
    if (!error)
        for (uint32_t w = 0; w < NW; ++w) {
            const uint32_t src_start = counts[w * 8 + 2];
            const uint32_t src_end = src_start + counts[w * 8];
            const uint32_t a = lo > src_start ? lo : src_start;
            const uint32_t b = hi_out < src_end ? hi_out : src_end;
            if (a >= b)
                continue;
            const uint32_t from = a - src_start, take = b - a,
                           dest = a - lo, skip = (from * 4) & 15u;
            // One bulk key-record read and one aligned ID read per owning
            // worker span. Compact key-major source and narrow destination
            // have the same 64-byte row order, so no intermediate gp copy or
            // scalar bit/layout rearrangement is needed.
            noc_async_read(ip_peer(w, compactp + from * 64),
                           direct_base + dest * 64, take * 64);
            noc_async_read(ip_peer(w, ip + (from * 4 & ~15u)), tp,
                           ((skip + take * 4 + 15) / 16) * 16);
            noc_async_read_barrier();
            auto in = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tp + skip);
            for (uint32_t k = 0; k < take; ++k)
                own_ids[dest + k] = in[k];
        }
    // Only the final partial page needs scalar work. The scorer masks these
    // rows by ``active``, but zeroing keeps diagnostic snapshots deterministic.
    auto direct_words = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(direct_base);
    for (uint32_t key = active; key < publish_chunks * 32; ++key)
        for (uint32_t word = 0; word < 16; ++word)
            direct_words[key * 16 + word] = 0;
    __sync_synchronize();
    cb_push_back(SKV4_CB_PACKED_PREFIX, publish_chunks);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_GATHER_E", 0);
    DeviceTimestampedData("SKV_DI_NATIVE_B", 0);
#endif
    // Narrow pages were populated directly by DI_GATHER.
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_NATIVE_E", 0);
    DeviceTimestampedData("SKV_DI_OUTPUT_B", 0);
#endif
    // Keep a fixed, non-overlapping scorer segment per worker, but populate
    // only its balanced live prefix.  The fused scorer consumes ``active`` and
    // never reads stale tails.  Prefix transfers are page-bounded bulk writes.
    if constexpr (!DIRECT_CB) {
        const uint32_t scorer_first = SLOT * CAP + worker * WORKER_CAP;
        const uint32_t record_bytes = ((active + 3) / 4) * 256;
        for (uint32_t copied = 0; copied < record_bytes;) {
            const uint32_t logical = scorer_first * 64 + copied;
            const uint32_t bytes = record_bytes - copied < 4096 - logical % 4096
                ? record_bytes - copied : 4096 - logical % 4096;
            noc_async_write(np + copied,
                            get_noc_addr(logical / 4096, prefix) + logical % 4096,
                            bytes);
            copied += bytes;
        }
        for (uint32_t copied = 0; copied < active * 4;) {
            const uint32_t logical = scorer_first * 4 + copied;
            const uint32_t bytes = active * 4 - copied < 64 - logical % 64
                ? active * 4 - copied : 64 - logical % 64;
            noc_async_write(op + copied,
                            get_noc_addr(logical / 64, ids_out) + logical % 64,
                            bytes);
            copied += bytes;
        }
    } else if constexpr (DIAGNOSTICS) {
        // Debug-only mirror for the untimed replay verifier.  Production
        // keeps the direct CB25/CB7 handoff and compiles these T1 writes out.
        const uint32_t scorer_first = SLOT * CAP + worker * WORKER_CAP;
        for (uint32_t chunk = 0; chunk < publish_chunks; ++chunk) {
            const uint32_t logical = scorer_first * 64 + chunk * 2048;
            noc_async_write(
                direct_base + chunk * 2048,
                get_noc_addr(logical / 4096, prefix) + logical % 4096,
                2048);
        }
        for (uint32_t copied = 0; copied < active * 4;) {
            const uint32_t logical = scorer_first * 4 + copied;
            const uint32_t bytes = active * 4 - copied < 64 - logical % 64
                ? active * 4 - copied : 64 - logical % 64;
            noc_async_write(
                own_idp + copied,
                get_noc_addr(logical / 64, ids_out) + logical % 64,
                bytes);
            copied += bytes;
        }
    }
    noc_async_write_barrier();
    arrive();
    if (!worker) {
        root_wait(3);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_INGEST_JOIN_RESET_B", 0);
#endif
        // JOIN is a three-phase per-dispatch epoch.  Its L1 semaphore is
        // persistent across generic_op calls, so clear it only after every
        // worker has reached the final phase.
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(JOIN)), 0);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_INGEST_JOIN_RESET_E", 0);
#endif
        auto s = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sp);
        for (uint32_t i = 0; i < (DIAGNOSTICS ? 16u : 2u); ++i)
            s[i] = 0;
        s[0] = unique;
        s[1] = error;
        if constexpr (DIAGNOSTICS) {
            s[3] = n;
            for (uint32_t w = 0; w < NW; ++w) {
                s[2] += counts[w * 8 + 4];
                s[5] += counts[w * 8 + 5];
                s[6] += counts[w * 8 + 6];
            }
            s[4] = s[2] * 64;
            const uint64_t end = c_tensix_core::read_wall_clock();
            s[7] = uint32_t(start);
            s[8] = uint32_t(start >> 32);
            s[9] = uint32_t(end);
            s[10] = uint32_t(end >> 32);
        }
        noc_async_write(sp, get_noc_addr(SLOT, status), 64);
        noc_async_write_barrier();
        publish();
    }
    await_phase(3);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_INGEST_RELEASE_RESET_B", 0);
#endif
    // The final publish proves that this worker observed all three releases.
    // Reset its local per-dispatch epoch before returning from the kernel.
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(RELEASE)), 0);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_INGEST_RELEASE_RESET_E", 0);
#endif
    // Same NCRISC immediately enters the scorer reader.  Reuse the no-longer
    // needed Fine header as a private scalar handoff; no NoC or host work.
    f[0] = active;
    f[1] = error;
    // f[2] remains the direct-CB prefix base published above.  CP4 also
    // needs the exact global unique count to select pair versus fallback, so
    // keep it in a distinct word instead of corrupting the retained prefix
    // pointer before the source-owned union delivery.
    f[3] = unique;
    __sync_synchronize();
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_OUTPUT_E", 0);
#endif
}
}
void kernel_main() {
    // Query has no candidate dependency. Reserve every destination and
    // publish the representative before ingestion so its writer can multicast
    // while NCRISC fetches and compacts cluster records.
    const uint32_t query_worker=get_arg_val<uint32_t>(SKV4_ARG_S15_RT_WORKER);
    const uint32_t query_slot=get_arg_val<uint32_t>(SKV4_ARG_S15_RT_SLOT);
    const auto query_early=TensorAccessor(
        qa,get_arg_val<uint32_t>(ADDR),2048);
    cb_reserve_back(SKV4_CB_QUERY,8);
    const uint32_t query_ptr=get_write_ptr(SKV4_CB_QUERY);
    if(query_worker) notify(SKV4_SEM_QUERY_DESTINATIONS_READY,0);
    if(query_worker==0) {
        for(uint32_t t=0;t<8;++t)
            noc_async_read(get_noc_addr(query_slot*8+t,query_early),
                           query_ptr+t*2048,2048);
        noc_async_read_barrier();
        cb_push_back(SKV4_CB_QUERY,8);
    }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_TOTAL_B", 0);
#endif
    distributed_input::run();
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_DI_TOTAL_E", 0);
#endif
    const uint32_t LIVE_KEYS=
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_INGEST_SCRATCH))[0];
    const uint32_t UNIQUE=
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_INGEST_SCRATCH))[3];
    const uint32_t LIVE_CHUNKS=LIVE_KEYS?(LIVE_KEYS+31)/32:1;
    const uint32_t worker=get_arg_val<uint32_t>(SKV4_ARG_S15_RT_WORKER),slot=get_arg_val<uint32_t>(SKV4_ARG_S15_RT_SLOT);
    const uint32_t ctx=get_arg_val<uint32_t>(SKV4_ARG_S15_RT_CONTEXT);
    const auto query=TensorAccessor(qa,get_arg_val<uint32_t>(ADDR),2048);
    const auto packed=TensorAccessor(ka,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_PACKED),4096);
    const auto candidates=TensorAccessor(ca,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_CANDIDATES),64);
    const auto clusters=TensorAccessor(cla,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_CLUSTERS),64);
    const auto ids_out=TensorAccessor(oa,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_IDS),G*M*4);
    constexpr uint32_t PAGES=(M+1023)/1024;
    const auto prefix_out=TensorAccessor(pa,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_PREFIX),65536);
    const auto counts_out=TensorAccessor(cta,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_COUNTS),G*4);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_QUERY_IDS_B", 0);
#endif
    const uint32_t cp3_chunks=(UNIQUE+31)/32;
    uint32_t cp3_live=0;
    for(uint32_t cp3_q=0;cp3_q<G;++cp3_q) {
        const uint32_t cp3_w=(worker/G)*G+cp3_q;
        const uint32_t cp3_lo=(cp3_chunks*cp3_w/(W*G))*32;
        const uint32_t cp3_hi0=(cp3_chunks*(cp3_w+1)/(W*G))*32;
        const uint32_t cp3_hi=cp3_hi0<UNIQUE?cp3_hi0:UNIQUE;
        if(cp3_hi>cp3_lo) cp3_live+=cp3_hi-cp3_lo;
    }
    const bool cp3_pair=UNIQUE<=6*1024;
    const uint32_t cp3_pages=cp3_pair?
        (((worker/G)&1u)?0:1):((cp3_live+2047)/2048)*2;
    // Receive directly in the local TopK input, before any peer writes it.
    if(cp3_pages) cb_reserve_back(SKV4_CB_LOCAL_SCORE_TILES,cp3_pages);
    const uint32_t qp=get_write_ptr(SKV4_CB_QUERY),
        kp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_INGEST_SCRATCH))[2],
        ip=get_read_ptr(SKV4_CB_TOKEN_IDS);
    const uint32_t clusterp=get_write_ptr(SKV4_CB_QUERY_CONTROL),recv=get_write_ptr(SKV4_CB_LOCAL_SCORE_TILES);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_QUERY_IDS_E", 0);
    DeviceTimestampedData("SKV_S15D_GDDR_B", 0);
    DeviceTimestampedData("SKV_S15D_GDDR_E", 0);
#endif
    if(worker) {
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_QUERY_JOIN_B", 0);
#endif
        auto valid=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_QUERY_DATA_READY));
        noc_semaphore_wait(valid,1); noc_semaphore_set(valid,0);
        cb_push_back(SKV4_CB_QUERY,8);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_QUERY_JOIN_E", 0);
#endif
    }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SCORE_READY_B", 0);
#endif
    if(worker%G==0) {
        auto score_ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_SHARD_SCORES_READY));
        noc_semaphore_wait(score_ready,G);
        // Program descriptors and L1 state persist across dispatches.  This
        // is a per-dispatch equality barrier, not a monotonic global epoch.
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SCORE_RESET_B", 0);
#endif
        noc_semaphore_set(score_ready,0);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SCORE_RESET_E", 0);
#endif
        release(worker,G);
    }
    await_epoch(1);
    if(cp3_pages) cb_push_back(SKV4_CB_LOCAL_SCORE_TILES,cp3_pages);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SCORE_READY_E", 0);
#endif
    // All native producer frontiers are retained until the head merges finish.
    if(UNIQUE<=6*1024 && ((worker/G)&1u)==0) {
        cb_reserve_back(SKV4_CB_COMPACT_TOPK_INPUT,1);
        auto child_ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(SKV4_SEM_TREE_FRONTIERS_READY));
        noc_semaphore_wait_min(child_ready,1);
        if((worker/G)!=0) noc_semaphore_set(child_ready,0);
        cb_push_back(SKV4_CB_COMPACT_TOPK_INPUT,1);
    }
    if(worker<G) {
        auto frontiers=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_FRONTIERS_READY));
        noc_semaphore_wait(frontiers,UNIQUE<=6*1024?3:W);
        noc_semaphore_set(frontiers,0);
        // Each head has six stable CB10/11 frontiers; unrelated heads need
        // not be ready. Retain all frontiers until the bitmap epoch below.
    }
    const uint32_t tree_shard=worker/G;
    const bool tree_compact_pair=UNIQUE<=6*1024;
    if(tree_compact_pair && tree_shard==0) {
        cb_reserve_back(SKV4_CB_PEER_VALUES,4);
        cb_reserve_back(SKV4_CB_PEER_INDICES,4);
        auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(SKV4_SEM_TREE_FRONTIERS_READY));
        // Root retained the odd-shard child-row notification (count 1), then
        // receives the two completed pair frontiers (counts 2 and 3).
        noc_semaphore_wait_min(ready,3);
        noc_semaphore_set(ready,0);
        cb_push_back(SKV4_CB_PEER_VALUES,4);
        cb_push_back(SKV4_CB_PEER_INDICES,4);
    } else if(!tree_compact_pair && (tree_shard&1u)==0) {
        const uint32_t receives=tree_shard==0?3:1;
        cb_reserve_back(SKV4_CB_PEER_VALUES,receives*2);
        cb_reserve_back(SKV4_CB_PEER_INDICES,receives*2);
        auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(SKV4_SEM_TREE_FRONTIERS_READY));
        noc_semaphore_wait_min(ready,receives);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_TREE_RESET_B", 0);
#endif
        noc_semaphore_set(ready,0);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_TREE_RESET_E", 0);
#endif
        cb_push_back(SKV4_CB_PEER_VALUES,receives*2);
        cb_push_back(SKV4_CB_PEER_INDICES,receives*2);
    }
    if(worker>=G && worker<G*BM_PARTS) {
        // Private helper bitmap has no dependency on the incoming IDs.
        const uint32_t active_bitmap_bytes=
            UNIQUE<=6*1024?FAST_BITMAP_BYTES:BITMAP_BYTES;
        {
            auto zero=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_UNION_BITMAP));
            for(uint32_t i=0;i<active_bitmap_bytes/4;++i) zero[i]=0;
        }
        // One bulk ID packet, one private bitmap, one bulk result packet.
        // Original source ownership below is unchanged by this temporary job.
        auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_BITMAP_INPUT_READY));
        noc_semaphore_wait(ready,1); noc_semaphore_set(ready,0);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BM_HELP_FILL_B", 0);
#endif
        const uint32_t part=worker/G,head=worker%G,bp=get_write_ptr(SKV4_CB_UNION_BITMAP);
        auto bm=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bp);
        auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_BITMAP_EXCHANGE));
        for(uint32_t i=0;i<BM_IDS;++i) {
            const uint32_t p=ids[i];
            if(p<BITMAP_DOMAIN) bm[p>>5] |= 1u<<(p&31u);
        }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BM_HELP_FILL_E", 0);
    DeviceTimestampedData("SKV_S15D_BM_HELP_SEND_B", 0);
#endif
        __sync_synchronize();
        noc_async_write(bp,peer(head,get_write_ptr(SKV4_CB_BITMAP_EXCHANGE)+BM_ID_BYTES+(part-1)*BITMAP_BYTES),active_bitmap_bytes);
        noc_async_write_barrier(); notify(SKV4_SEM_BITMAP_PARTS_READY,head);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BM_HELP_SEND_E", 0);
#endif
    }
    slot_barrier(G,2);  // all four head-owner bitmaps are ready
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BITMAP_READ_B", 0);
#endif
    const uint32_t ub=get_write_ptr(SKV4_CB_UNION_LAYOUT),bitmap=get_write_ptr(SKV4_CB_UNION_BITMAP);
    const bool compact_bitmap=UNIQUE<=6*1024;
    uint32_t offset,live_words;
    if(compact_bitmap) {
        const uint32_t source_shard=worker/G;
        const uint32_t global_chunks=(UNIQUE+31)/32;
        const uint32_t source_lo=(global_chunks*worker/SW)*32;
        const uint32_t shard_lo=(global_chunks*(source_shard*G)/SW)*32;
        const uint32_t encoded_start=source_shard*1024+(source_lo-shard_lo);
        offset=(encoded_start/32)*4;
        live_words=(LIVE_KEYS+31)/32;
    } else {
        offset=worker*WORDS*4;
        live_words=WORDS;
    }
    const uint32_t aligned=offset&~63u,bskip=offset-aligned;
    const uint32_t take=((bskip+live_words*4+63)/64)*64;
    for(uint32_t h=0;h<G;++h)
        if(h!=worker && take)
            noc_async_read(peer(h,bitmap+aligned),ub+h*STRIDE,take);
    noc_async_read_barrier();
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BITMAP_READ_E", 0);
    DeviceTimestampedData("SKV_S15D_UNION_ENUM_B", 0);
#endif
    auto selected=reinterpret_cast<tt_l1_ptr uint32_t*>(ub+POS_OFF);
    uint32_t count=0;
    for(uint32_t j=0;j<live_words;++j) {
        uint32_t word=0;
        for(uint32_t h=0;h<G;++h) {
            const uint32_t at=(h==worker?bitmap+offset:ub+h*STRIDE+bskip)+j*4;
            word |= *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(at);
        }
        for(uint32_t bit=0;bit<32 && j*32+bit<LIVE_KEYS;++bit)
            if(word&(1u<<bit)) selected[count++]=j*32+bit;
    }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_UNION_ENUM_E", 0);
    DeviceTimestampedData("SKV_S15D_COUNT_PUBLISH_B", 0);
#endif
    const uint32_t countp=get_write_ptr(SKV4_CB_UNION_CONTROL);
    auto counts=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(countp);
    counts[worker]=count;
    __sync_synchronize();
    auto compact_sync=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(SKV4_SEM_COMPACTION_READY));
    noc_semaphore_set(compact_sync,1);
    if(worker) noc_async_write(countp+worker*4,peer(0,countp+worker*4),4);
    noc_async_write_barrier();
    notify(SKV4_SEM_EPOCH_JOIN,0);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_COUNT_PUBLISH_E", 0);
#endif
    // Root publishes the complete count array before doing its own compact.
    // Other workers do independent local work while counts are in flight.
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_COUNT_ROOT_WAIT_B", 0);
#endif
    if(worker==0) slot_barrier(SW+G,3);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_COUNT_ROOT_WAIT_E", 0);
    DeviceTimestampedData("SKV_S15D_COMPACT_B", 0);
#endif
    const uint32_t out=get_write_ptr(SKV4_CB_UNION_RECORDS);
    auto token_ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ip);
    // Original gather tiles and IDs, not an exported/re-read compact cache.
    const uint32_t compact_split=count/2;
    for(uint32_t j=compact_split;j<count;++j) {
        const uint32_t p=selected[j];
        auto src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kp+(p/32)*2048+(p%32)*64);
        auto dst=reinterpret_cast<tt_l1_ptr uint32_t*>(out+j*64);
        #pragma GCC unroll 16
        for(uint32_t word=0;word<16;++word) dst[word]=src[word];
    }
    noc_semaphore_wait_min(compact_sync,2);
    noc_semaphore_set(compact_sync,0);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_COMPACT_E", 0);
    DeviceTimestampedData("SKV_S15D_COUNT_PEER_READ_B", 0);
#endif
    if(worker) {
        slot_barrier(SW+G,3);
        noc_async_read(peer(0,countp),countp,COUNT_BYTES); noc_async_read_barrier();
    }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_COUNT_PEER_READ_E", 0);
#endif
    uint32_t start=0,total=0;
    for(uint32_t w=0;w<SW;++w) { if(w<worker) start+=counts[w]; total+=counts[w]; }
    const uint32_t q=total/G,r=total%G;
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_DELIVER_B", 0);
#endif
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
    cb_pop_front(SKV4_CB_TOKEN_IDS,1);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_DELIVER_E", 0);
#endif
}
"""
        )
    )


def compute(*, profile, n):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + stage15_defines("compute")
        + profile_defines(profile=profile)
        + (
            r"""#if SKV4_PROFILE
#include "tools/profiler/kernel_profiler.hpp"
#endif
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#include <cstdint>
#include "api/compute/pack_untilize.h"
template<typename T> inline T s15_compute_0_arg(uint32_t i) { return get_arg_val<T>(SKV4_ARG_S15_RT_CHUNKS+i); }
#define get_arg_val s15_compute_0_arg
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
    cb_reserve_back(SKV4_CB_DECODED_KEYS,4);
    tile_regs_acquire();
    copy_tile(SKV4_CB_PACKED_PREFIX,0,0);
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,expand_four_planes,
        (FIRST),0,VectorMode::None));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1,SKV4_CB_DECODED_KEYS);
    pack_tile(2,SKV4_CB_DECODED_KEYS);
    pack_tile(3,SKV4_CB_DECODED_KEYS);
    pack_tile(4,SKV4_CB_DECODED_KEYS);
    tile_regs_release();
    cb_push_back(SKV4_CB_DECODED_KEYS,4);
}
__attribute__((noinline)) void s15_compute_0() {
    constexpr uint32_t N=get_compile_time_arg_val(SKV4_ARG_S15_CT_N), PHASE=get_compile_time_arg_val(SKV4_ARG_S15_CT_PHASE);
    compute_kernel_hw_startup(SKV4_CB_PACKED_PREFIX,SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS);
    cb_wait_front(SKV4_CB_QUERY,8);
    cb_wait_front(SKV4_CB_COMPUTE_CONTROL,1);
    const uint32_t LIVE_CHUNKS=read_tile_value(SKV4_CB_COMPUTE_CONTROL,0,1);
    if constexpr(PHASE>0) {
        // Unfused TopK-XL leaves SFPU index tracking enabled across programs.
        // Custom SFPI execute calls do not initialize this state themselves.
        MATH(SFPU_UNARY_INIT(unused));
        for(uint32_t chunk=0;chunk<LIVE_CHUNKS;++chunk) {
            cb_wait_front(SKV4_CB_PACKED_PREFIX,1);
            // A packed chunk is consumed immediately; the writer can drain
            // its scores while MATH starts decoding the next chunk.
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_C_DECODE_B", 0);
#endif
            copy_tile_to_dst_init_short_with_dt(SKV4_CB_DECODED_KEYS,SKV4_CB_PACKED_PREFIX);
            pack_reconfig_data_format(SKV4_CB_DECODED_KEYS);
            expand_four<0>(); expand_four<4>();
            cb_pop_front(SKV4_CB_PACKED_PREFIX,1);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_C_DECODE_E", 0);
    DeviceTimestampedData("SKV_S15D_C_MATMUL_B", 0);
#endif
            if constexpr(PHASE!=3) {
                if constexpr(PHASE==2) {
                    // Matrix maps Q -> SrcB, K -> SrcA, unlike tile-copy.
                    reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS);
                    matmul_init(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS,1);
                    pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
                }
                cb_wait_front(SKV4_CB_DECODED_KEYS,8);
                if constexpr(PHASE==2) {
                    cb_reserve_back(SKV4_CB_SCORE_TILES,1);
                    tile_regs_acquire();
                    if(chunk==0) {
                        // A prior unfused TopK-XL leaves the first FPU output
                        // contaminated even after ZEROACC. Materialize zeros
                        // once; later chunks are cleared by the PACK release.
                        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                            DST_ACCUM_MODE,clear_score,0,VectorMode::None));
                        // SFPU execute preserves source zero flags; restore
                        // the normal floating-point matmul interpretation.
                        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS);
                    }
                    for(uint32_t kt=0;kt<8;++kt) matmul_tiles(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS,kt,kt,0);
                    tile_regs_commit(); tile_regs_wait();
                    // The writer consumes only face-0 rows 0..3 and face-1
                    // rows 0..3.  Packing the first 20 face rows preserves
                    // those byte offsets while avoiding the unused 1,408 B.
                    pack_rows_init(20);
                    pack_rows(0,SKV4_CB_SCORE_TILES);
                    pack_rows_uninit();
                    // pack_rows replaces the default packer MOP.  The next
                    // chunk's decoded BF16 key tiles use ordinary pack_tile.
                    pack_init(SKV4_CB_DECODED_KEYS);
                    tile_regs_release(); cb_push_back(SKV4_CB_SCORE_TILES,1);
                }
                cb_pop_front(SKV4_CB_DECODED_KEYS,8);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_C_MATMUL_E", 0);
#endif
            }
        }
    }
    cb_pop_front(SKV4_CB_COMPUTE_CONTROL,1);
    cb_pop_front(SKV4_CB_QUERY,8);
}
#undef get_arg_val
template<typename T> inline T s15_compute_1_arg(uint32_t i) { return get_arg_val<T>(SKV4_ARG_S15_RT_CHUNKS+i); }
#define get_arg_val s15_compute_1_arg
#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void source_remap_large(
        uint32_t base,uint32_t q1,uint32_t q2,uint32_t q3,uint32_t live) {
    #pragma GCC unroll 4
    for(uint32_t i=0;i<64;++i) {
        sfpi::vUInt v=sfpi::dst_reg[64+i].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt out=v+base;
        v_if(v>=q1) { out=v+base+"""
            + str(1 * n)
            + r"""u-q1; }
        v_endif;
        v_if(v>=q2) { out=v+base+"""
            + str(2 * n)
            + r"""u-q2; }
        v_endif;
        v_if(v>=q3) { out=v+base+"""
            + str(3 * n)
            + r"""u-q3; }
        v_endif;
        v_if(v>=live) { out=0xffffffffu; }
        v_endif;
        sfpi::dst_reg[64+i].mode<sfpi::DataLayout::U32>()=out;
    }
}
}
#endif
inline void chunk_large(CircularBuffer& in,uint32_t dst,uint32_t active,bool ascending) {
    in.wait_front(2);
    topk_xl_copy_tile_init(SKV4_CB_LOCAL_SCORE_TILES);
    topk_xl_copy_tile<2048>(SKV4_CB_LOCAL_SCORE_TILES,dst,0,active);
    in.pop_front(2);
    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<2048,0>(dst);
    topk_xl_init<2048,true>();
    topk_xl_local_sort<2048>(dst,ascending);
    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<2048>(dst);
    topk_xl_separate_indices_row_major_advance_chunk_base<2048>();
}
__attribute__((noinline)) void s15_compute_1() {
    cb_wait_front(SKV4_CB_QUERY_READY,1);
    const uint32_t live=ckernel::read_tile_value(SKV4_CB_QUERY_READY,0,9);
    const uint32_t unique=ckernel::read_tile_value(SKV4_CB_QUERY_READY,0,10);
    const bool compact_pair=unique<=6*1024;
    const uint32_t chunks=(live+2047)/2048;
    const uint32_t tail=live?(live-(chunks-1)*2048):1;
    const uint32_t shard=get_arg_val<uint32_t>(SKV4_ARG_S15_PHASE_RT_SHARD),local=get_arg_val<uint32_t>(SKV4_ARG_S15_PHASE_RT_LIVE);
    const uint32_t q1=ckernel::read_tile_value(SKV4_CB_QUERY_READY,0,6);
    const uint32_t q2=ckernel::read_tile_value(SKV4_CB_QUERY_READY,0,7);
    const uint32_t q3=ckernel::read_tile_value(SKV4_CB_QUERY_READY,0,8);
    reconfig_data_format(SKV4_CB_LOCAL_SCORE_TILES,SKV4_CB_LOCAL_SCORE_TILES);
    pack_reconfig_data_format(SKV4_CB_FRONTIER_VALUES);
    CircularBuffer in(SKV4_CB_LOCAL_SCORE_TILES),values(SKV4_CB_FRONTIER_VALUES),indices(SKV4_CB_FRONTIER_INDICES);
    if(compact_pair && (shard&1u)) return;
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_C_LOCAL_B", 0);
#endif
    tile_regs_acquire();
    MATH(SFPU_UNARY_INIT(unused));
    if(compact_pair) {
        const uint32_t global_chunks=(unique+31)/32;
        uint32_t pair_live[2];
        for(uint32_t side=0;side<2;++side) {
            uint32_t total=0;
            const uint32_t pair_shard=shard+side;
            for(uint32_t q=0;q<4;++q) {
                const uint32_t w=pair_shard*4+q;
                const uint32_t lo=(global_chunks*w/24)*32;
                const uint32_t hi0=(global_chunks*(w+1)/24)*32;
                const uint32_t hi=hi0<unique?hi0:unique;
                if(hi>lo) total+=hi-lo;
            }
            pair_live[side]=total;
        }
        if(shard==0) topk_xl_separate_indices_row_major_init_static<0,0>();
        else if(shard==2) topk_xl_separate_indices_row_major_init_static<0,2048>();
        else topk_xl_separate_indices_row_major_init_static<0,4096>();
        in.wait_front(1);
        topk_xl_copy_tile_init(SKV4_CB_LOCAL_SCORE_TILES);
        topk_xl_copy_tile<1024>(SKV4_CB_LOCAL_SCORE_TILES,0,0,pair_live[0]?pair_live[0]:1);
        in.pop_front(1);
        CircularBuffer child(SKV4_CB_COMPACT_TOPK_INPUT);
        child.wait_front(1);
        topk_xl_copy_tile_init(SKV4_CB_COMPACT_TOPK_INPUT);
        topk_xl_copy_tile<1024>(SKV4_CB_COMPACT_TOPK_INPUT,1,0,pair_live[1]?pair_live[1]:1);
        child.pop_front(1);
        topk_xl_add_lsb_indices_init();
        topk_xl_add_lsb_indices<2048,0>(0);
        topk_xl_init<2048,true>();
        topk_xl_local_sort<2048>(0,false);
        topk_xl_separate_indices_row_major_reinit();
        topk_xl_separate_indices_row_major<2048>(0);
        topk_xl_separate_indices_row_major_advance_chunk_base<2048>();
        topk_xl_init<2048,false>();
        topk_xl_rebuild<2048,false>(0,shard!=0);
    } else {
        topk_xl_separate_indices_row_major_init_static<0,0>();
        chunk_large(in,0,chunks==1?tail:2048,false);
        if(chunks==1) {
            topk_xl_init<2048,false>();
            topk_xl_rebuild<2048,false>(0,shard!=0);
        }
        for(uint32_t c=1;c<chunks;++c) {
            chunk_large(in,4,c+1==chunks?tail:2048,true);
            topk_xl_init<2048,false>();
            topk_xl_merge<2048,false>(0);
            topk_xl_rebuild<2048,false>(0,c+1==chunks && shard!=0);
        }
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
            source_remap_large,0,VectorMode::None,
            shard*"""
            + str(4 * n)
            + r"""u,q1,q2,q3,live));
    }
    tile_regs_commit(); tile_regs_wait();
    values.reserve_back(2);
    pack_tile(0,SKV4_CB_FRONTIER_VALUES); pack_tile(1,SKV4_CB_FRONTIER_VALUES);
    values.push_back(2);
    pack_reconfig_data_format(SKV4_CB_FRONTIER_INDICES);
    indices.reserve_back(2);
    pack_tile(2,SKV4_CB_FRONTIER_INDICES); pack_tile(3,SKV4_CB_FRONTIER_INDICES);
    indices.push_back(2);
    tile_regs_release();
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_C_LOCAL_E", 0);
#endif
}
#undef get_arg_val
template<typename T> inline T s15_compute_2_arg(uint32_t i) { return get_arg_val<T>(SKV4_ARG_S15_RT_CHUNKS+i); }
#define get_arg_val s15_compute_2_arg
__attribute__((noinline)) void s15_compute_2() {
    constexpr uint32_t K=get_compile_time_arg_val(SKV4_ARG_S15_CT_K);
    constexpr uint32_t W=get_compile_time_arg_val(SKV4_ARG_S15_CT_WORKERS);
    static_assert(K==2048 && W==6);
    constexpr uint32_t TILES=2,SURV_VALUES=0,SURV_IDS=2;
    constexpr uint32_t IN_VALUES=4,IN_IDS=6;
    const uint32_t shard=get_arg_val<uint32_t>(SKV4_ARG_S15_PHASE_RT_SHARD);
    CircularBuffer control(SKV4_CB_QUERY_READY);
    control.wait_front(1);
    const bool compact_pair=ckernel::read_tile_value(SKV4_CB_QUERY_READY,0,10)<=6*1024;
    const uint32_t incoming=compact_pair?2:(shard==0?3:1);
    reconfig_data_format(SKV4_CB_PEER_VALUES,SKV4_CB_PEER_VALUES);
    pack_untilize_dest_init<TILES,TILES>(SKV4_CB_MERGED_VALUES);
    CircularBuffer own_score(SKV4_CB_FRONTIER_VALUES),own_id(SKV4_CB_FRONTIER_INDICES),local_ready(SKV4_CB_COMPACT_TOPK_INPUT);
    CircularBuffer score_cb(SKV4_CB_PEER_VALUES),id_cb(SKV4_CB_PEER_INDICES),out_score(SKV4_CB_MERGED_VALUES),out_id(SKV4_CB_MERGED_INDICES);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_C_MERGE_B", 0);
#endif
    tile_regs_acquire();
    if(!compact_pair) local_ready.wait_front(1);
    own_score.wait_front(TILES);
    topk_xl_copy_tile_init(SKV4_CB_FRONTIER_VALUES);
    topk_xl_copy_tile<K>(SKV4_CB_FRONTIER_VALUES,SURV_VALUES,0,K);
    own_id.wait_front(TILES);
    topk_xl_copy_tile_init(SKV4_CB_FRONTIER_INDICES);
    topk_xl_copy_tile<K>(SKV4_CB_FRONTIER_INDICES,SURV_IDS,0,K);
    if(!compact_pair) local_ready.pop_front(1);
    if(shard!=0) {
        topk_xl_init<K,false>();
        topk_xl_rebuild<K,false>(SURV_VALUES,false);
    }
    for(uint32_t edge=0;edge<incoming;++edge) {
        score_cb.wait_front(TILES);
        topk_xl_copy_tile_init(SKV4_CB_PEER_VALUES);
        topk_xl_copy_tile<K>(SKV4_CB_PEER_VALUES,IN_VALUES,0,K);
        score_cb.pop_front(TILES);
        id_cb.wait_front(TILES);
        topk_xl_copy_tile_init(SKV4_CB_PEER_INDICES);
        topk_xl_copy_tile<K>(SKV4_CB_PEER_INDICES,IN_IDS,0,K);
        id_cb.pop_front(TILES);
        topk_xl_init<K,false>();
        topk_xl_merge<K,false>(SURV_VALUES);
        const bool ascending=shard!=0 && edge+1==incoming;
        topk_xl_rebuild<K,false>(SURV_VALUES,ascending);
    }
    tile_regs_commit(); tile_regs_wait();
    out_score.reserve_back(1);
    pack_untilize_dest<TILES,TILES>(SKV4_CB_MERGED_VALUES,1,0,SURV_VALUES);
    out_score.push_back(1);
    pack_untilize_dest_init<TILES,TILES>(SKV4_CB_MERGED_INDICES);
    out_id.reserve_back(1);
    pack_untilize_dest<TILES,TILES>(SKV4_CB_MERGED_INDICES,1,0,SURV_IDS);
    out_id.push_back(1);
    tile_regs_release();
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_C_MERGE_E", 0);
#endif
}
#undef get_arg_val
void kernel_main() {
    s15_compute_0(); s15_compute_1();
    const uint32_t s15_worker=get_arg_val<uint32_t>(SKV4_ARG_S15_RT_WORKER);
    const uint32_t s15_shard=s15_worker/4;
    const bool s15_compact_pair=
        ckernel::read_tile_value(SKV4_CB_QUERY_READY,0,10)<=6*1024;
    if((s15_compact_pair && s15_shard==0) ||
       (!s15_compact_pair && (s15_shard&1u)==0)) s15_compute_2();
    cb_pop_front(SKV4_CB_QUERY_READY,1);
}"""
        )
    )


def writer(*, profile, debug=True):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + stage15_defines("writer")
        + profile_defines(profile=profile, debug=bool(debug))
        + (
            r"""#if SKV4_PROFILE
#include "tools/profiler/kernel_profiler.hpp"
#endif
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t N=get_compile_time_arg_val(SKV4_ARG_S15_CT_N),W=get_compile_time_arg_val(SKV4_ARG_S15_CT_WORKERS);
constexpr uint32_t PAD=get_compile_time_arg_val(SKV4_ARG_S15_CT_PAD),M=get_compile_time_arg_val(SKV4_ARG_S15_CT_M);
constexpr bool DEBUG=get_compile_time_arg_val(SKV4_ARG_S15_CT_DEBUG);
constexpr uint32_t CHUNKS=get_compile_time_arg_val(SKV4_ARG_S15_CT_CHUNKS),G=4,K=2048,SW=W*G;
constexpr uint32_t BITMAP_DOMAIN=get_compile_time_arg_val(SKV4_ARG_S15_CT_BITMAP_DOMAIN);
constexpr uint32_t N15=N*SW,BITMAP_BYTES=((BITMAP_DOMAIN+511)/512)*64;
constexpr uint32_t FAST_BITMAP_BYTES=((6*1024+511)/512)*64;
constexpr uint32_t BM_PARTS=4,BM_IDS=M/BM_PARTS,BM_ID_BYTES=((BM_IDS*4+63)/64)*64;
static_assert(M%BM_PARTS==0 && SW>=G*BM_PARTS);
constexpr uint32_t WORDS=N/32,STRIDE=((WORDS*4+123)/64)*64;
constexpr uint32_t POS_OFF=G*STRIDE,COUNT_BYTES=((SW*4+63)/64)*64;
constexpr uint32_t ID_STRIDE=N*4+64,ID_OUT_OFF=N*64,OC_OFF=ID_OUT_OFF+G*ID_STRIDE;
constexpr uint32_t PEERS=SKV4_ARG_S15_RT_END,ADDR=PEERS+SW*2;
constexpr auto qa=TensorAccessorArgs<SKV4_ARG_S15_CT_ACCESSORS>();
constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
constexpr auto ca=TensorAccessorArgs<ka.next_compile_time_args_offset()>();
constexpr auto cla=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
constexpr auto oa=TensorAccessorArgs<cla.next_compile_time_args_offset()>();
constexpr auto pa=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
constexpr auto cta=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
#if SKV4_CODEGEN_DEBUG
constexpr auto drs=TensorAccessorArgs<cta.next_compile_time_args_offset()>();
constexpr auto dri=TensorAccessorArgs<drs.next_compile_time_args_offset()>();
constexpr auto dls=TensorAccessorArgs<dri.next_compile_time_args_offset()>();
constexpr auto dli=TensorAccessorArgs<dls.next_compile_time_args_offset()>();
constexpr auto dhs=TensorAccessorArgs<dli.next_compile_time_args_offset()>();
constexpr auto dhi=TensorAccessorArgs<dhs.next_compile_time_args_offset()>();
#endif
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
    if(get_arg_val<uint32_t>(SKV4_ARG_S15_RT_WORKER)==0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_EPOCH_JOIN)),expected);
        release(0,SW);
    }
    await_epoch(epoch);
    if(expected==2*SW+G) {
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SLOT_RESET_B", 0);
#endif
        if(get_arg_val<uint32_t>(SKV4_ARG_S15_RT_WORKER)==0)
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_EPOCH_JOIN)),0);
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_EPOCH_RELEASE)),0);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SLOT_RESET_E", 0);
#endif
    }
}
void kernel_main() {
    const uint32_t worker=get_arg_val<uint32_t>(SKV4_ARG_S15_RT_WORKER),slot=get_arg_val<uint32_t>(SKV4_ARG_S15_RT_SLOT);
    const uint32_t shard=worker/G,quarter=worker%G;
    const uint32_t base=get_write_ptr(SKV4_CB_HEAD_SCORES);
    cb_wait_front(SKV4_CB_WRITER_CONTROL,1);
    const auto live=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_WRITER_CONTROL));
    const uint32_t LIVE_KEYS=live[0],LIVE_CHUNKS=live[1],UNIQUE=live[2];
    if(worker==0) {
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_QUERY_FANOUT_B", 0);
#endif
        cb_wait_front(SKV4_CB_QUERY,8);
        auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_QUERY_DESTINATIONS_READY));
        noc_semaphore_wait(ready,SW-1); noc_semaphore_set(ready,0);
        const uint32_t qp=get_read_ptr(SKV4_CB_QUERY);
        // Each existing four-core group is a physically dense rectangle.
        for(uint32_t shard=0;shard<W;++shard) {
            uint32_t x0=255,y0=255,x1=0,y1=0;
            for(uint32_t h=0;h<G;++h) {
                const uint32_t x=get_arg_val<uint32_t>(PEERS+2*(shard*G+h));
                const uint32_t y=get_arg_val<uint32_t>(PEERS+2*(shard*G+h)+1);
                x0=x<x0?x:x0; x1=x>x1?x:x1; y0=y<y0?y:y0; y1=y>y1?y:y1;
            }
            // get_noc_multicast_addr maps coordinates but does NOT reorder
            // endpoints. NoC1 is mirrored, unlike the reader's NoC0.
            const uint64_t dst=noc_index==0?get_noc_multicast_addr(x0,y0,x1,y1,qp)
                                           :get_noc_multicast_addr(x1,y1,x0,y0,qp);
            noc_async_write_multicast(qp,dst,
                                      8*2048,G-(shard==0));
        }
        noc_async_write_barrier();
        // One notification per worker, never per tile/key. Destination CB
        // addresses and readiness were established before the bulk writes.
        for(uint32_t w=1;w<SW;++w) notify(SKV4_SEM_QUERY_DATA_READY,w);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_QUERY_FANOUT_E", 0);
#endif
    }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SCORE_ASSEMBLE_B", 0);
#endif
    for(uint32_t chunk=0;chunk<LIVE_CHUNKS;++chunk) {
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SCORE_WAIT_B", 0);
#endif
        cb_wait_front(SKV4_CB_SCORE_TILES,1);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SCORE_WAIT_E", 0);
    DeviceTimestampedData("SKV_S15D_SCORE_COPY_B", 0);
#endif
        auto tile=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_SCORE_TILES));
        for(uint32_t h=0;h<G;++h) {
            auto row=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                base+h*N*2);
            #pragma GCC unroll 8
            for(uint32_t pair=0;pair<8;++pair) {
                row[chunk*16+pair]=tile[h*8+pair];
                row[chunk*16+8+pair]=tile[128+h*8+pair];
            }
        }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SCORE_COPY_E", 0);
#endif
        cb_pop_front(SKV4_CB_SCORE_TILES,1);
    }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SCORE_ASSEMBLE_E", 0);
    DeviceTimestampedData("SKV_S15D_SCORE_ROUTE_B", 0);
#endif
    uint32_t quarter_live[G],quarter_off[G];
    const uint32_t global_chunks=(UNIQUE+31)/32;
    uint32_t shard_live=0;
    for(uint32_t q=0;q<G;++q) {
        const uint32_t w=shard*G+q;
        const uint32_t ch0=global_chunks*w/(W*G);
        const uint32_t ch1=global_chunks*(w+1)/(W*G);
        const uint32_t lo=ch0*32;
        const uint32_t hi=ch1*32<UNIQUE?ch1*32:UNIQUE;
        quarter_off[q]=shard_live;
        quarter_live[q]=hi>lo?hi-lo:0;
        shard_live+=quarter_live[q];
    }
    auto src=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+quarter*N*2);
    const bool compact_pair=UNIQUE<=6*1024;
    if(!compact_pair || (shard&1u)==0) {
        auto own=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            get_write_ptr(SKV4_CB_LOCAL_SCORE_TILES)+quarter_off[quarter]*2);
        for(uint32_t i=0;i<LIVE_KEYS;++i) own[i]=src[i];
        if(!shard_live)
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_LOCAL_SCORE_TILES))[0]=0xff80u;
    }
    __sync_synchronize();
    if(compact_pair && (shard&1u)) {
        for(uint32_t h=0;h<G;++h)
            if(LIVE_KEYS)
                noc_async_write(base+h*N*2,
                    peer((shard-1)*G+h,
                         get_write_ptr(SKV4_CB_COMPACT_TOPK_INPUT)+quarter_off[quarter]*2),
                    LIVE_KEYS*2);
    } else {
        for(uint32_t h=0;h<G;++h)
            if(h!=quarter && LIVE_KEYS)
                noc_async_write(base+h*N*2,
                    peer(shard*G+h,get_write_ptr(SKV4_CB_LOCAL_SCORE_TILES)+quarter_off[quarter]*2),
                    LIVE_KEYS*2);
    }
    noc_async_write_barrier();
    if(compact_pair && (shard&1u))
        notify(SKV4_SEM_TREE_FRONTIERS_READY,(shard-1)*G+quarter);
    cb_reserve_back(SKV4_CB_QUERY_READY,1);
    auto compact_control=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(SKV4_CB_QUERY_READY));
    compact_control[0]=shard_live;
    for(uint32_t q=0;q<G;++q) {
        compact_control[1+q]=quarter_live[q];
        compact_control[5+q]=quarter_off[q];
    }
    compact_control[9]=shard_live;
    compact_control[10]=UNIQUE;
    __sync_synchronize();
    cb_push_back(SKV4_CB_QUERY_READY,1);
    cb_pop_front(SKV4_CB_WRITER_CONTROL,1);
    notify(SKV4_SEM_SHARD_SCORES_READY,shard*G);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_SCORE_ROUTE_E", 0);
#endif
#if SKV4_CODEGEN_DEBUG
    if constexpr(DEBUG) {
        // Optional snapshots are write-only diagnostics, never the transport.
        const auto ds=TensorAccessor(drs,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_SHARD_SCORES),PAD*2);
        const auto di=TensorAccessor(dri,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_SHARD_INDICES),PAD*4);
        const uint32_t tmp=get_write_ptr(SKV4_CB_DEBUG_SCRATCH),tail=PAD-G*N;
        auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tmp);
        for(uint32_t i=0;i<N;++i) ids[i]=worker*N+i;
        for(uint32_t i=N;i<N+tail;++i) ids[i]=0xffffffffu;
        auto pad=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp+(N+tail)*4);
        for(uint32_t i=0;i<tail;++i) pad[i]=0xff80u;
        for(uint32_t h=0;h<G;++h) {
            const uint32_t row=(slot*W+shard)*G+h;
            const uint32_t src=base+h*N*2;
            noc_async_write(src,get_noc_addr(row,ds)+quarter*N*2,N*2);
            noc_async_write(tmp,get_noc_addr(row,di)+quarter*N*4,(N+(quarter==3?tail:0))*4);
            if(quarter==3 && tail) noc_async_write(tmp+(N+tail)*4,get_noc_addr(row,ds)+G*N*2,tail*2);
        }
        noc_async_write_barrier();
    }
#endif
    if(!compact_pair || (shard&1u)==0) {
    cb_wait_front(SKV4_CB_FRONTIER_VALUES,2); cb_wait_front(SKV4_CB_FRONTIER_INDICES,2);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_K1024_EXPAND_B", 0);
#endif
    // The compact-pair path emits native K2048 rows directly. Above the
    // 6,144-entry threshold every fallback shard uses the native K2048
    // selector as well, so no scalar K1024-to-K2048 expansion remains.
    if(!compact_pair && (shard&1u)==0) {
        cb_reserve_back(SKV4_CB_COMPACT_TOPK_INPUT,1);
        cb_push_back(SKV4_CB_COMPACT_TOPK_INPUT,1);
    }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_K1024_EXPAND_E", 0);
#endif
    if(!compact_pair && (shard&1u)) {
        const uint32_t parent=(shard-1)*G+quarter;
        noc_async_write(get_read_ptr(SKV4_CB_FRONTIER_VALUES),
            peer(parent,get_write_ptr(SKV4_CB_PEER_VALUES)),K*2);
        noc_async_write(get_read_ptr(SKV4_CB_FRONTIER_INDICES),
            peer(parent,get_write_ptr(SKV4_CB_PEER_INDICES)),K*4);
        noc_async_write_barrier();
        notify(SKV4_SEM_TREE_FRONTIERS_READY,parent);
    }
#if SKV4_CODEGEN_DEBUG
    if constexpr(DEBUG) {
        const auto ds=TensorAccessor(dls,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_LOCAL_SCORES),W*K*2);
        const auto di=TensorAccessor(dli,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_LOCAL_INDICES),W*K*4);
        const uint32_t row=slot*G+quarter;
        noc_async_write(
            get_read_ptr(SKV4_CB_FRONTIER_VALUES),get_noc_addr(row,ds)+shard*K*2,K*2);
        noc_async_write(
            get_read_ptr(SKV4_CB_FRONTIER_INDICES),get_noc_addr(row,di)+shard*K*4,K*4);
        noc_async_write_barrier();
    }
#endif
    notify(SKV4_SEM_FRONTIERS_READY,quarter);
    if(compact_pair && (shard==2 || shard==4)) {
        const uint32_t edge=shard==2?0:1;
        const uint32_t parent=quarter;
        noc_async_write(get_read_ptr(SKV4_CB_FRONTIER_VALUES),
            peer(parent,get_write_ptr(SKV4_CB_PEER_VALUES)+edge*K*2),K*2);
        noc_async_write(get_read_ptr(SKV4_CB_FRONTIER_INDICES),
            peer(parent,get_write_ptr(SKV4_CB_PEER_INDICES)+edge*K*4),K*4);
        noc_async_write_barrier();
        notify(SKV4_SEM_TREE_FRONTIERS_READY,parent);
    } else if(!compact_pair && (shard==2 || shard==4)) {
        cb_wait_front(SKV4_CB_MERGED_VALUES,1); cb_wait_front(SKV4_CB_MERGED_INDICES,1);
        const uint32_t edge=shard==2?1:2;
        const uint32_t parent=quarter;
        noc_async_write(get_read_ptr(SKV4_CB_MERGED_VALUES),
            peer(parent,get_write_ptr(SKV4_CB_PEER_VALUES)+edge*K*2),K*2);
        noc_async_write(get_read_ptr(SKV4_CB_MERGED_INDICES),
            peer(parent,get_write_ptr(SKV4_CB_PEER_INDICES)+edge*K*4),K*4);
        noc_async_write_barrier();
        notify(SKV4_SEM_TREE_FRONTIERS_READY,parent);
        cb_pop_front(SKV4_CB_MERGED_VALUES,1); cb_pop_front(SKV4_CB_MERGED_INDICES,1);
    }
    }
    if(worker<G) {
        // Global merge is still running on TRISC; clear only our own bitmap.
        const uint32_t active_bitmap_bytes=
            UNIQUE<=6*1024?FAST_BITMAP_BYTES:BITMAP_BYTES;
        {
            auto zero=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_UNION_BITMAP));
            for(uint32_t i=0;i<active_bitmap_bytes/4;++i) zero[i]=0;
        }
        cb_wait_front(SKV4_CB_MERGED_VALUES,1); cb_wait_front(SKV4_CB_MERGED_INDICES,1);
#if SKV4_CODEGEN_DEBUG
        if constexpr(DEBUG) {
            const auto ds=TensorAccessor(dhs,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_HEAD_SCORES),K*2);
            const auto di=TensorAccessor(dhi,get_arg_val<uint32_t>(ADDR+SKV4_ARG_S15_IO_HEAD_INDICES),K*4);
            noc_async_write(get_read_ptr(SKV4_CB_MERGED_VALUES),get_noc_addr(slot*G+worker,ds),K*2);
            noc_async_write(get_read_ptr(SKV4_CB_MERGED_INDICES),get_noc_addr(slot*G+worker,di),K*4);
            noc_async_write_barrier();
        }
#endif
        cb_pop_front(SKV4_CB_MERGED_VALUES,1);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BITMAP_BUILD_B", 0);
    DeviceTimestampedData("SKV_S15D_BM_ID_BULK_B", 0);
#endif
        const uint32_t idp=get_read_ptr(SKV4_CB_MERGED_INDICES),scratch=get_write_ptr(SKV4_CB_BITMAP_EXCHANGE);
        for(uint32_t part=1;part<BM_PARTS;++part)
            noc_async_write(idp+part*BM_IDS*4,peer(part*G+worker,scratch),BM_IDS*4);
        noc_async_write_barrier();
        for(uint32_t part=1;part<BM_PARTS;++part) notify(SKV4_SEM_BITMAP_INPUT_READY,part*G+worker);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BM_ID_BULK_E", 0);
    DeviceTimestampedData("SKV_S15D_BM_ROOT_BUILD_B", 0);
#endif
        const uint32_t bp=get_write_ptr(SKV4_CB_UNION_BITMAP);
        auto bitmap=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bp);
        auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idp);
        for(uint32_t i=0;i<BM_IDS;++i) {
            const uint32_t p=ids[i];
            if(p<BITMAP_DOMAIN) bitmap[p>>5] |= 1u<<(p&31u);
        }
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BM_ROOT_BUILD_E", 0);
    DeviceTimestampedData("SKV_S15D_BM_JOIN_B", 0);
#endif
        auto done=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_BITMAP_PARTS_READY));
        noc_semaphore_wait(done,BM_PARTS-1); noc_semaphore_set(done,0);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BM_JOIN_E", 0);
    DeviceTimestampedData("SKV_S15D_BM_OR_B", 0);
#endif
        auto partial=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch+BM_ID_BYTES);
        for(uint32_t i=0;i<active_bitmap_bytes/4;++i)
            bitmap[i] |= partial[i] | partial[BITMAP_BYTES/4+i] | partial[2*(BITMAP_BYTES/4)+i];
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BM_OR_E", 0);
#endif
        __sync_synchronize();
        cb_pop_front(SKV4_CB_MERGED_INDICES,1);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_BITMAP_BUILD_E", 0);
#endif
        notify(SKV4_SEM_EPOCH_JOIN,0);
    }
    await_epoch(2); // every merge has consumed the retained native frontiers
    auto compact_sync=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(SKV4_SEM_COMPACTION_READY));
    noc_semaphore_wait_min(compact_sync,1);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_COMPACT_HELP_B", 0);
#endif
    const uint32_t compact_count=*(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_UNION_CONTROL))+worker);
    const uint32_t compact_split=compact_count/2;
    auto compact_selected=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(SKV4_CB_UNION_LAYOUT)+POS_OFF);
    const uint32_t compact_keys=get_write_ptr(SKV4_CB_PACKED_PREFIX),compact_out=get_write_ptr(SKV4_CB_UNION_RECORDS);
    for(uint32_t j=0;j<compact_split;++j) {
        const uint32_t p=compact_selected[j];
        auto src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            compact_keys+(p/32)*2048+(p%32)*64);
        auto dst=reinterpret_cast<tt_l1_ptr uint32_t*>(compact_out+j*64);
        #pragma GCC unroll 16
        for(uint32_t word=0;word<16;++word) dst[word]=src[word];
    }
    __sync_synchronize();
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15D_COMPACT_HELP_E", 0);
#endif
    noc_semaphore_set(compact_sync,2);
    if(!compact_pair || (shard&1u)==0) { cb_pop_front(SKV4_CB_FRONTIER_VALUES,2); cb_pop_front(SKV4_CB_FRONTIER_INDICES,2); }
}
"""
        )
    )


def ingest(*, semaphore=StandaloneIngestSemaphore.INGEST_JOIN):
    return (
        cpp_defines(CB)
        + semaphore_defines(StandaloneIngestSemaphore)
        + ingest_defines()
        + cpp_defines(StandaloneIngestCB)
        + (
            r"""
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t NC = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_NC), CTX = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_CONTEXT);
constexpr uint32_t CAP = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_CAPACITY), FW = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_FINE_WIDTH);
constexpr uint32_t SLOT = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_SLOT), MINIMUM = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_MINIMUM);
constexpr uint32_t PAGE = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_PAGE_BYTES), PAGES = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_PAGES);
constexpr uint32_t BLOCK = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_BLOCK_BYTES), NW = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_WORKERS);
constexpr uint32_t LOCAL = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_LOCAL), BATCH = 32;
constexpr bool DIAGNOSTICS = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_DIAGNOSTICS);
constexpr bool DIRECT_CB = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_DIRECT_CB);
constexpr uint32_t WORKER_CAP = CAP / NW;
static_assert(CAP % NW == 0 && WORKER_CAP % 32 == 0);
constexpr uint32_t IDCAP = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_POSTING_CAPACITY);
constexpr uint32_t IDBYTES = ((IDCAP * 4 + 63) / 64) * 64;
constexpr uint32_t MW = ((NC + 31) / 32 + NW * 16 - 1) / (NW * 16) * 16, COUNT_BYTES = NW * 32;
constexpr uint32_t OWNER_KEYS = ((CAP / 64 + NW - 1) / NW) * 64;
constexpr uint32_t ITMP = ((OWNER_KEYS * 4 + 16 + 63) / 64) * 64;
constexpr uint32_t JOIN = """
            + str(semaphore)
            + ", RELEASE = "
            + str(semaphore)
            + r""" + 1;
constexpr auto fa = TensorAccessorArgs<SKV4_ARG_INGEST_CT_ACCESSORS>();
constexpr auto da = TensorAccessorArgs<fa.next_compile_time_args_offset()>();
constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
constexpr auto ia = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
constexpr auto pa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
constexpr auto sa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
inline uint64_t ip_peer(uint32_t w, uint32_t addr) {
    return get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_PEERS + 2 * w), get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_PEERS_Y + 2 * w), addr);
}
inline void arrive() {
    noc_semaphore_inc(ip_peer(0, get_semaphore(JOIN)), 1);
    noc_async_atomic_barrier();
}
inline void root_wait(uint32_t epoch) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(JOIN)),
                           epoch * NW);
}
inline void publish() {
    for (uint32_t w = 0; w < NW; ++w)
        noc_semaphore_inc(ip_peer(w, get_semaphore(RELEASE)), 1);
    noc_async_atomic_barrier();
}
inline void await_phase(uint32_t epoch) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(RELEASE)),
                           epoch);
}
void kernel_main() {
    uint64_t start = 0;
    if constexpr (DIAGNOSTICS)
        start = c_tensix_core::read_wall_clock();
    const auto fine = TensorAccessor(fa, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_FINE), 64);
    const auto descriptors = TensorAccessor(da, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_DESCRIPTORS), 64);
    const auto blocks = TensorAccessor(ba, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_BLOCKS), PAGE);
    const auto ids_out = TensorAccessor(ia, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_IDS_OUT), 64);
    const auto prefix = TensorAccessor(pa, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_PREFIX), 4096);
    const auto status = TensorAccessor(sa, get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_STATUS), 64);
    const uint32_t worker = get_arg_val<uint32_t>(SKV4_ARG_INGEST_RT_WORKER), base = get_write_ptr(SKV4_CB_INGEST_WORKSPACE);
    const uint32_t dp = base + FW * 4, rp = dp + LOCAL * 64;
    const uint32_t compactp = rp + LOCAL * BLOCK, ip = compactp + IDCAP * 64,
                   xp = ip + IDBYTES;
    const uint32_t bp = xp + IDBYTES, cp = bp + MW * NW * 4, gp = cp + COUNT_BYTES;
    const uint32_t np = gp + OWNER_KEYS * 64;
    const uint32_t op = np + (DIRECT_CB ? 0 : OWNER_KEYS * 64);
    const uint32_t tp = op + (DIRECT_CB ? 0 : OWNER_KEYS * 4);
    const uint32_t sp = tp + ITMP;
    auto f = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(base);
    auto d = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(dp);
    auto ids = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(ip);
    auto positions = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(xp);
    auto bitmap = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(bp);
    auto counts = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(cp);
    /*@DI_SETUP_B*/
    noc_async_read(get_noc_addr(0, fine), base, 64);
    noc_async_read_barrier();
    const uint32_t n = f[0];
    uint32_t error = f[1] ? 1 : 0;
    if (n > FW - 16)
        error = 2;
    if (!error)
        for (uint32_t t = 1; t < (16 + n + 15) / 16; ++t)
            noc_async_read(get_noc_addr(t, fine), base + t * 64, 64);
    noc_async_read_barrier();
    const uint32_t first = error ? 0 : n * worker / NW, last = error ? 0 : n * (worker + 1) / NW;
    if (!error)
        for (uint32_t i = first; i < last; ++i) {
            if (f[16 + i] >= NC)
                error = 3;
            if constexpr (DIAGNOSTICS)
                if (i && f[16 + i] <= f[15 + i])
                    error = 3;
        }
    const uint32_t count = error ? 0 : last - first;
    uint32_t cursor = 0, hi = error ? 0 : n;
    while (cursor < hi) {
        const uint32_t mid = (cursor + hi) / 2;
        if ((f[16 + mid] >> 5) < worker * MW)
            cursor = mid + 1;
        else
            hi = mid;
    }
    for (uint32_t word = worker * MW; word < (worker + 1) * MW; ++word) {
        uint32_t bits = 0;
        while (!error && cursor < n && (f[16 + cursor] >> 5) == word) {
            bits |= 1u << (f[16 + cursor] & 31);
            ++cursor;
        }
        bitmap[word] = bits;
    }
    __sync_synchronize();
    if (worker)
        noc_async_write(bp + worker * MW * 4, ip_peer(0, bp + worker * MW * 4), MW * 4);
    noc_async_write_barrier();
    arrive();
    if (!worker) {
        root_wait(1);
        for (uint32_t w = 1; w < NW; ++w)
            noc_async_write(bp, ip_peer(w, bp), MW * NW * 4);
        noc_async_write_barrier();
        publish();
    }
    await_phase(1);
    /*@DI_SETUP_E*/
    /*@DI_FETCH_B*/
    /*@DI_DESCRIPTORS_B*/
    for (uint32_t off = 0; off < count; off += BATCH) {
        const uint32_t end = off + BATCH < count ? off + BATCH : count;
        for (uint32_t i = off; i < end; ++i)
            noc_async_read(get_noc_addr(f[16 + first + i] / 4, descriptors), dp + i * 64, 64);
        noc_async_read_barrier();
    }
    /*@DI_DESCRIPTORS_E*/
    /*@DI_GDDR_BLOCKS_B*/
    uint32_t read_bytes = 0, requests = 0, postings = 0;
    for (uint32_t off = 0; off < count; off += BATCH) {
        const uint32_t end = off + BATCH < count ? off + BATCH : count;
        for (uint32_t i = off; i < end; ++i) {
            auto m = d + i * 16 + (f[16 + first + i] % 4) * 4;
            const uint32_t pg = m[0], at = m[1], nk = m[2], size = m[3];
            if constexpr (DIAGNOSTICS)
                if (pg >= PAGES || at % 64 || at > PAGE || size > PAGE - at || size > BLOCK ||
                    nk > CTX || size != ((nk * 8 + 63) / 64) * 64 + nk * 64) {
                    error = 6;
                    continue;
                }
            if (!nk)
                continue;
            noc_async_read(get_noc_addr(pg, blocks) + at, rp + i * BLOCK, size);
            if constexpr (DIAGNOSTICS) {
                read_bytes += size;
                ++requests;
                postings += nk;
            }
        }
        noc_async_read_barrier();
    }
    /*@DI_GDDR_BLOCKS_E*/
    /*@DI_FETCH_E*/
    /*@DI_COMPACT_B*/
    uint32_t live = 0, compact_requests = 0;
    if (!error)
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t cid = f[16 + first + i];
            auto m = d + i * 16 + (cid % 4) * 4;
            const uint32_t nk = m[2], meta = ((nk * 8 + 63) / 64) * 64, begin = live;
            auto tokens = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(rp + i * BLOCK);
            // Read ALL metadata of this block before in-place compaction can overwrite it.
            for (uint32_t j = 0; j < nk; ++j) {
                const uint32_t token = tokens[2 * j], mate = tokens[2 * j + 1];
                if constexpr (DIAGNOSTICS)
                    if (token >= CTX || (mate != 0xffffffffu && mate >= NC) || mate == cid) {
                        error = 3;
                        continue;
                    }
                if (mate > cid || !(bitmap[mate >> 5] & (1u << (mate & 31)))) {
                    ids[live] = token;
                    positions[live] = j;
                    ++live;
                }
            }
            for (uint32_t k = begin; k < live;) {
                const uint32_t posting = positions[k];
                uint32_t run = 1;
                while (k + run < live && positions[k + run] == posting + run)
                    ++run;
                // Source blocks remain immutable until the single final join.
                // Coalesce adjacent records rather than issuing one NoC per key.
                noc_async_read(ip_peer(worker, rp + i * BLOCK + meta + posting * 64),
                               compactp + k * 64, run * 64);
                k += run;
                if (++compact_requests % BATCH == 0)
                    noc_async_read_barrier();
            }
        }
    noc_async_read_barrier();
    /*@DI_COMPACT_E*/
    /*@DI_COUNTS_B*/
    for (uint32_t i = 0; i < (DIAGNOSTICS ? 8u : 4u); ++i)
        counts[worker * 8 + i] = 0;
    counts[worker * 8] = live;
    counts[worker * 8 + 1] = error;
    if constexpr (DIAGNOSTICS) {
        counts[worker * 8 + 4] = postings;
        counts[worker * 8 + 5] = read_bytes;
        counts[worker * 8 + 6] = requests;
    }
    __sync_synchronize();
    if (worker)
        noc_async_write(cp + worker * 32, ip_peer(0, cp + worker * 32), 32);
    noc_async_write_barrier();
    arrive();
    if (!worker) {
        root_wait(2);
        uint32_t sum = 0, err = 0;
        for (uint32_t w = 0; w < NW; ++w) {
            counts[w * 8 + 2] = sum;
            sum += counts[w * 8];
            err |= counts[w * 8 + 1];
        }
        if (!err) {
            if (sum > CAP)
                err = 4;
            else if (sum < MINIMUM)
                err = 5;
        }
        for (uint32_t w = 0; w < NW; ++w) {
            counts[w * 8 + 1] = err;
            counts[w * 8 + 3] = sum;
        }
        __sync_synchronize();
        for (uint32_t w = 1; w < NW; ++w)
            noc_async_write(cp, ip_peer(w, cp), COUNT_BYTES);
        noc_async_write_barrier();
        publish();
    }
    await_phase(2);
    error = counts[worker * 8 + 1];
    const uint32_t unique = counts[worker * 8 + 3];
    /*@DI_COUNTS_E*/
    /*@DI_GATHER_B*/
    const uint32_t live_chunks = error ? 0 : (unique + 31) / 32;
    const uint32_t ch0 = live_chunks * worker / NW,
                   ch1 = live_chunks * (worker + 1) / NW;
    const uint32_t lo = ch0 * 32, hi_out = ch1 * 32 < unique ? ch1 * 32 : unique;
    const uint32_t active = error ? 0 : hi_out - lo;
    if constexpr (DIRECT_CB)
        cb_reserve_back(SKV4_CB_TOKEN_IDS, 1);
    const uint32_t own_idp = DIRECT_CB ? get_write_ptr(SKV4_CB_TOKEN_IDS) : op;
    auto own_ids = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(own_idp);
    for (uint32_t k = 0; k < (DIRECT_CB ? WORKER_CAP : OWNER_KEYS); ++k)
        own_ids[k] = 0xffffffffu;
    if (!error)
        for (uint32_t w = 0; w < NW; ++w) {
            const uint32_t src_start = counts[w * 8 + 2], src_end = src_start + counts[w * 8];
            const uint32_t a = lo > src_start ? lo : src_start,
                           b = hi_out < src_end ? hi_out : src_end;
            if (a >= b)
                continue;
            const uint32_t from = a - src_start, take = b - a, dest = a - lo,
                           skip = (from * 4) & 15u;
            // Two bulk reads per source/destination span, not per record. IDs use
            // aligned staging because their destination's low address bits may differ.
            noc_async_read(ip_peer(w, compactp + from * 64), gp + dest * 64, take * 64);
            noc_async_read(ip_peer(w, ip + (from * 4 & ~15u)), tp,
                           ((skip + take * 4 + 15) / 16) * 16);
            noc_async_read_barrier();
            auto in = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tp + skip);
            for (uint32_t k = 0; k < take; ++k)
                own_ids[dest + k] = in[k];
        }
    /*@DI_GATHER_E*/
    /*@DI_NATIVE_B*/
    auto raw = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(gp);
    const uint32_t publish_chunks = active ? (active + 31) / 32 : 1;
    uint32_t direct_base = 0;
    if constexpr (DIRECT_CB) {
        // CB25 has one producer (this NCRISC) and one consumer (the scorer
        // TRISC) for the complete program lifetime. Reserve its bounded ring
        // once so the backing addresses remain stable for survivor delivery.
        cb_reserve_back(SKV4_CB_PACKED_PREFIX, publish_chunks);
        direct_base = get_write_ptr(SKV4_CB_PACKED_PREFIX);
        f[2] = direct_base;

        // The scorer's BFP4 output headers are invariant. Initialize them
        // before publishing the work control, exactly as the old reader did.
        for (uint32_t t = 0; t < 8; ++t) {
            auto e = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
                get_write_ptr(SKV4_CB_DECODED_KEYS) + t * 576);
            for (uint32_t j = 0; j < 16; ++j)
                e[j] = 0x80808080u;
        }
        cb_push_back(SKV4_CB_TOKEN_IDS, 1);

        // Publish controls before data pages. The consumers may now wait on
        // CB25 without creating a control/data circular wait.
        cb_reserve_back(SKV4_CB_COMPUTE_CONTROL, 1);
        cb_reserve_back(SKV4_CB_WRITER_CONTROL, 1);
        auto live_compute = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
            get_write_ptr(SKV4_CB_COMPUTE_CONTROL));
        auto live_writer = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
            get_write_ptr(SKV4_CB_WRITER_CONTROL));
        live_compute[0] = active;
        live_compute[1] = publish_chunks;
        live_compute[2] = unique;
        live_writer[0] = active;
        live_writer[1] = publish_chunks;
        live_writer[2] = unique;
        __sync_synchronize();
        cb_push_back(SKV4_CB_COMPUTE_CONTROL, 1);
        cb_push_back(SKV4_CB_WRITER_CONTROL, 1);
    }
    for (uint32_t k = 0; k < publish_chunks * 32; ++k) {
        volatile tt_l1_ptr uint32_t *dst;
        if constexpr (DIRECT_CB) {
            const uint32_t chunk = k / 32, row = k % 32;
            const uint32_t cluster = row / 4, key = row % 4;
            dst = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
                direct_base + chunk * 4096 + (cluster / 4) * 2048 +
                (cluster % 4) * 256) + 2 * key;
        } else {
            dst = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(np) +
                  (k / 4) * 64 + 2 * (k % 4);
        }
        const bool valid = k < active;
#pragma GCC unroll 8
        for (uint32_t g = 0; g < 8; ++g) {
            const uint32_t x = valid ? raw[k * 16 + 2 * g] : 0,
                           y = valid ? raw[k * 16 + 2 * g + 1] : 0, at = (g / 2) * 16 + g % 2;
            dst[at] = (x & 65535u) | (y << 16);
            dst[at + 8] = (x >> 16) | (y & 0xffff0000u);
        }
        if constexpr (DIRECT_CB)
            if ((k & 31u) == 31u) {
                __sync_synchronize();
                cb_push_back(SKV4_CB_PACKED_PREFIX, 1);
            }
    }
    /*@DI_NATIVE_E*/
    /*@DI_OUTPUT_B*/
    // Keep a fixed, non-overlapping scorer segment per worker, but populate
    // only its balanced live prefix.  The fused scorer consumes ``active`` and
    // never reads stale tails.  Prefix transfers are page-bounded bulk writes.
    if constexpr (!DIRECT_CB) {
        const uint32_t scorer_first = SLOT * CAP + worker * WORKER_CAP;
        const uint32_t record_bytes = ((active + 3) / 4) * 256;
        for (uint32_t copied = 0; copied < record_bytes;) {
            const uint32_t logical = scorer_first * 64 + copied;
            const uint32_t bytes = record_bytes - copied < 4096 - logical % 4096
                ? record_bytes - copied : 4096 - logical % 4096;
            noc_async_write(np + copied,
                            get_noc_addr(logical / 4096, prefix) + logical % 4096,
                            bytes);
            copied += bytes;
        }
        for (uint32_t copied = 0; copied < active * 4;) {
            const uint32_t logical = scorer_first * 4 + copied;
            const uint32_t bytes = active * 4 - copied < 64 - logical % 64
                ? active * 4 - copied : 64 - logical % 64;
            noc_async_write(op + copied,
                            get_noc_addr(logical / 64, ids_out) + logical % 64,
                            bytes);
            copied += bytes;
        }
    } else if constexpr (DIAGNOSTICS) {
        // Debug-only mirror for the untimed replay verifier.  Production
        // keeps the direct CB25/CB7 handoff and compiles these T1 writes out.
        const uint32_t scorer_first = SLOT * CAP + worker * WORKER_CAP;
        for (uint32_t chunk = 0; chunk < publish_chunks; ++chunk) {
            const uint32_t logical = scorer_first * 64 + chunk * 2048;
            noc_async_write(
                direct_base + chunk * 4096,
                get_noc_addr(logical / 4096, prefix) + logical % 4096,
                1024);
            noc_async_write(
                direct_base + chunk * 4096 + 2048,
                get_noc_addr((logical + 1024) / 4096, prefix) +
                    (logical + 1024) % 4096,
                1024);
        }
        for (uint32_t copied = 0; copied < active * 4;) {
            const uint32_t logical = scorer_first * 4 + copied;
            const uint32_t bytes = active * 4 - copied < 64 - logical % 64
                ? active * 4 - copied : 64 - logical % 64;
            noc_async_write(
                own_idp + copied,
                get_noc_addr(logical / 64, ids_out) + logical % 64,
                bytes);
            copied += bytes;
        }
    }
    noc_async_write_barrier();
    arrive();
    if (!worker) {
        root_wait(3);
        /*@S15D_INGEST_JOIN_RESET_B*/
        // JOIN is a three-phase per-dispatch epoch.  Its L1 semaphore is
        // persistent across generic_op calls, so clear it only after every
        // worker has reached the final phase.
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(JOIN)), 0);
        /*@S15D_INGEST_JOIN_RESET_E*/
        auto s = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sp);
        for (uint32_t i = 0; i < (DIAGNOSTICS ? 16u : 2u); ++i)
            s[i] = 0;
        s[0] = unique;
        s[1] = error;
        if constexpr (DIAGNOSTICS) {
            s[3] = n;
            for (uint32_t w = 0; w < NW; ++w) {
                s[2] += counts[w * 8 + 4];
                s[5] += counts[w * 8 + 5];
                s[6] += counts[w * 8 + 6];
            }
            s[4] = s[2] * 64;
            const uint64_t end = c_tensix_core::read_wall_clock();
            s[7] = uint32_t(start);
            s[8] = uint32_t(start >> 32);
            s[9] = uint32_t(end);
            s[10] = uint32_t(end >> 32);
        }
        noc_async_write(sp, get_noc_addr(SLOT, status), 64);
        noc_async_write_barrier();
        publish();
    }
    await_phase(3);
    /*@S15D_INGEST_RELEASE_RESET_B*/
    // The final publish proves that this worker observed all three releases.
    // Reset its local per-dispatch epoch before returning from the kernel.
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(RELEASE)), 0);
    /*@S15D_INGEST_RELEASE_RESET_E*/
    // Same NCRISC immediately enters the scorer reader.  Reuse the no-longer
    // needed Fine header as a private scalar handoff; no NoC or host work.
    f[0] = active;
    f[1] = error;
    // f[2] remains the direct-CB prefix base published above.  CP4 also
    // needs the exact global unique count to select pair versus fallback, so
    // keep it in a distinct word instead of corrupting the retained prefix
    // pointer before the source-owned union delivery.
    f[3] = unique;
    __sync_synchronize();
    /*@DI_OUTPUT_E*/
}
"""
        )
    )
