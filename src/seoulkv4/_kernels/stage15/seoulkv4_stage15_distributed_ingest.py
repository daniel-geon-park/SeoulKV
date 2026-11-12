"""Complete distributed Stage1.5 input path, fused into scorer NCRISCs.

Fetch workers compact retained records in-place; page owners gather bulk spans
from those L1 buffers and write disjoint native pages. No central payload owner,
no per-token NoC, no second GDDR fetch. Two monotonic semaphore epochs implement
bitmap/count/output joins. No mailbox or standalone fallback remains.
"""

import math

import torch
from seoulkv4._kernels.stage15.abi import StandaloneIngestSemaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

from seoulkv4._kernels.stage15.abi import StandaloneIngestCB, INGEST_CT
from seoulkv4._kernels.codegen import stage15 as source_codegen

from seoulkv4._kernels.common.seoulkv4_cluster_blocks import pack_cluster_blocks


def balanced_worker_ranges(unique, workers, capacity):
    """Kernel-exact 32-key tile balancing into fixed scorer segments."""
    if capacity % workers or capacity // workers % 32:
        raise ValueError("capacity must form 32-key-aligned worker segments")
    chunks = (unique + 31) // 32
    out = []
    for worker in range(workers):
        c0 = chunks * worker // workers
        c1 = chunks * (worker + 1) // workers
        lo = c0 * 32
        hi = min(c1 * 32, unique)
        out.append((lo, hi, worker * (capacity // workers)))
    return out


def compact_shard_layout(unique, shard, workers=6, group=4):
    """Kernel-exact live quarter lengths and compact offsets for one shard."""
    chunks = (unique + 31) // 32
    counts = []
    for quarter in range(group):
        worker = shard * group + quarter
        lo = chunks * worker // (workers * group) * 32
        hi = min(chunks * (worker + 1) // (workers * group) * 32, unique)
        counts.append(max(0, hi - lo))
    offsets = []
    total = 0
    for count in counts:
        offsets.append(total)
        total += count
    return counts, offsets, total


SOURCE = INGEST_CT.cpp_defines("SKV4_ARG_INGEST_CT") + r"""
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t NC = get_compile_time_arg_val(0), CTX = get_compile_time_arg_val(1);
constexpr uint32_t CAP = get_compile_time_arg_val(2), FW = get_compile_time_arg_val(3);
constexpr uint32_t SLOT = get_compile_time_arg_val(4), MINIMUM = get_compile_time_arg_val(5);
constexpr uint32_t PAGE = get_compile_time_arg_val(6), PAGES = get_compile_time_arg_val(7);
constexpr uint32_t BLOCK = get_compile_time_arg_val(8), NW = get_compile_time_arg_val(9);
constexpr uint32_t LOCAL = get_compile_time_arg_val(10), BATCH = 32;
constexpr bool DIAGNOSTICS = get_compile_time_arg_val(11);
constexpr bool DIRECT_CB = get_compile_time_arg_val(12);
constexpr uint32_t WORKER_CAP = CAP / NW;
static_assert(CAP % NW == 0 && WORKER_CAP % 32 == 0);
constexpr uint32_t IDCAP = get_compile_time_arg_val(SKV4_ARG_INGEST_CT_POSTING_CAPACITY);
constexpr uint32_t IDBYTES = ((IDCAP * 4 + 63) / 64) * 64;
constexpr uint32_t MW = ((NC + 31) / 32 + NW * 16 - 1) / (NW * 16) * 16, COUNT_BYTES = NW * 32;
constexpr uint32_t OWNER_KEYS = ((CAP / 64 + NW - 1) / NW) * 64;
constexpr uint32_t ITMP = ((OWNER_KEYS * 4 + 16 + 63) / 64) * 64;
constexpr uint32_t JOIN = INGEST_SEM, RELEASE = INGEST_SEM + 1;
constexpr auto fa = TensorAccessorArgs<SKV4_ARG_INGEST_CT_ACCESSORS>();
constexpr auto da = TensorAccessorArgs<fa.next_compile_time_args_offset()>();
constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
constexpr auto ia = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
constexpr auto pa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
constexpr auto sa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
inline uint64_t ip_peer(uint32_t w, uint32_t addr) {
    return get_noc_addr(get_arg_val<uint32_t>(7 + 2 * w), get_arg_val<uint32_t>(8 + 2 * w), addr);
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
    const auto fine = TensorAccessor(fa, get_arg_val<uint32_t>(0), 64);
    const auto descriptors = TensorAccessor(da, get_arg_val<uint32_t>(1), 64);
    const auto blocks = TensorAccessor(ba, get_arg_val<uint32_t>(2), PAGE);
    const auto ids_out = TensorAccessor(ia, get_arg_val<uint32_t>(3), 64);
    const auto prefix = TensorAccessor(pa, get_arg_val<uint32_t>(4), 4096);
    const auto status = TensorAccessor(sa, get_arg_val<uint32_t>(5), 64);
    const uint32_t worker = get_arg_val<uint32_t>(6), base = get_write_ptr(0);
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
        cb_reserve_back(7, 1);
    const uint32_t own_idp = DIRECT_CB ? get_write_ptr(7) : op;
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
        cb_reserve_back(25, publish_chunks);
        direct_base = get_write_ptr(25);
        f[2] = direct_base;

        // The scorer's BFP4 output headers are invariant. Initialize them
        // before publishing the work control, exactly as the old reader did.
        for (uint32_t t = 0; t < 8; ++t) {
            auto e = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
                get_write_ptr(1) + t * 576);
            for (uint32_t j = 0; j < 16; ++j)
                e[j] = 0x80808080u;
        }
        cb_push_back(7, 1);

        // Publish controls before data pages. The consumers may now wait on
        // CB25 without creating a control/data circular wait.
        cb_reserve_back(4, 1);
        cb_reserve_back(6, 1);
        auto live_compute = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
            get_write_ptr(4));
        auto live_writer = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
            get_write_ptr(6));
        live_compute[0] = active;
        live_compute[1] = publish_chunks;
        live_compute[2] = unique;
        live_writer[0] = active;
        live_writer[1] = publish_chunks;
        live_writer[2] = unique;
        __sync_synchronize();
        cb_push_back(4, 1);
        cb_push_back(6, 1);
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
                cb_push_back(25, 1);
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


def posting_bound(offsets, selected_clusters):
    """Maximum raw postings in any selected_clusters distinct CSR rows."""
    counts = offsets.diff().long()
    if selected_clusters < 0 or bool((counts < 0).any()):
        raise ValueError("invalid CSR row/count bound")
    return int(counts.topk(min(selected_clusters, len(counts))).values.sum())


def layout(nc, fw, capacity, workers, block, *, posting_capacity=None):
    local = math.ceil((fw - 16) / workers)
    # A cluster block includes 8B of token/mate metadata per 64B prefix.
    # Even with aligned metadata, floor(block/72) bounds the posting count.
    idcap = local * (block // (64 + 2 * 4)) if posting_capacity is None else posting_capacity
    idbytes = math.ceil(idcap * 4 / 64) * 64
    mw = math.ceil(math.ceil(nc / 32) / (workers * 16)) * 16
    own = math.ceil((capacity // 64) / workers) * 64
    itmp = math.ceil((own * 4 + 16) / 64) * 64
    scratch = (
        fw * 4
        + local * 64
        + local * block
        + idcap * 64
        + 2 * idbytes
        + mw * workers * 4
        + workers * 32
        + itmp
        + 64
    )
    return local, scratch


def build(
    tt,
    device,
    cells,
    cluster_ids,
    token_prefix,
    *,
    capacity,
    minimum=2048,
    groups,
    diagnostics=True,
):
    from seoulkv4._kernels.topk.tt_topk_large_fixed import _ranges

    if capacity <= 0 or capacity % 64:
        raise ValueError("capacity must be a positive multiple of64")
    if len(cells) != len(groups) or len(cells) != len(cluster_ids):
        raise ValueError("cell/group mismatch")
    if not cells or any(not g or len(g) % 4 for g in groups):
        raise ValueError(
            "each ingestion group must have a positive multiple of four workers"
        )
    if any(capacity % len(g) or capacity // len(g) % 32 for g in groups):
        raise ValueError("capacity must form 32-key-aligned scorer segments")
    if len({p for g in groups for p in g}) != sum(map(len, groups)):
        raise ValueError("overlapping groups")
    if token_prefix.device.type != "cpu":
        raise ValueError("prefill prefixes must be CPU tensors")
    host = token_prefix.reshape(len(cells), cells[0]["context"], 64)
    packs = [pack_cluster_blocks(c, p, include_mates=True) for c, p in zip(cells, host)]
    specs = []
    for c, h, g, pack in zip(cells, cluster_ids, groups, packs):
        if not g:
            raise ValueError("empty ingestion group")
        fw = math.prod(h.shape)
        block = max(64, pack["stats"]["max_block_bytes"])
        local = math.ceil((fw - 16) / len(g))
        # Any worker owns at most LOCAL distinct selected clusters. Their raw
        # postings cannot exceed the sum of the LOCAL largest lists globally.
        # Deduplication only decreases this count; no runtime truncation occurs.
        posting_capacity = posting_bound(c["offsets"], local)
        local, scratch = layout(
            len(c["offsets"]) - 1, fw, capacity, len(g), block,
            posting_capacity=posting_capacity,
        )
        if scratch > 750000:
            raise ValueError(
                f"distributed ingestion requires {scratch}B/core; "
                "the supported limit is 750000B/core"
            )
        specs.append((fw, block, local, scratch, posting_capacity))
    slots = len(cells)

    def alloc(rows, cols, dtype):
        return tt.allocate_tensor_on_device(
            tt.Shape((1, 1, rows, cols)),
            dtype,
            tt.ROW_MAJOR_LAYOUT,
            device,
            tt.L1_MEMORY_CONFIG,
        )

    ids = alloc(slots * capacity // 16, 16, tt.uint32)
    prefix = alloc(slots * capacity // 64, 4096, tt.uint8)
    status = alloc(slots, 16, tt.uint32)
    clusters = tt.from_torch(
        torch.arange(capacity // 4)
        .int()
        .expand(slots, -1)
        .contiguous()
        .reshape(-1, 16),
        device=device,
        dtype=tt.uint32,
        layout=tt.ROW_MAJOR_LAYOUT,
        memory_config=tt.L1_MEMORY_CONFIG,
    )
    storage = [ids, prefix, status, clusters]
    parts = []
    kernels = []
    cbs = []
    for slot, (c, h, g, pack, spec) in enumerate(
        zip(cells, cluster_ids, groups, packs, specs)
    ):
        fw, block, local, scratch, posting_capacity = spec
        st = pack["stats"]
        d = tt.from_torch(
            pack["descriptors"],
            device=device,
            dtype=tt.uint32,
            layout=tt.ROW_MAJOR_LAYOUT,
            memory_config=tt.L1_MEMORY_CONFIG,
        )
        b = tt.from_torch(
            pack["pages"],
            device=device,
            dtype=tt.uint8,
            layout=tt.ROW_MAJOR_LAYOUT,
            memory_config=tt.DRAM_MEMORY_CONFIG,
        )
        io = [h, d, b, ids, prefix, status]
        from seoulkv4._kernels.stage15.abi import INGEST_CT, ingest_runtime

        scalar_fields = dict(
            nc=len(c["offsets"]) - 1,
            context=c["context"],
            capacity=capacity,
            fine_width=fw,
            slot=slot,
            minimum=minimum,
            page_bytes=st["page_bytes"],
            pages=st["pages"],
            block_bytes=block,
            workers=len(g),
            local=local,
            diagnostics=int(diagnostics),
            direct_cb=1,
            posting_capacity=posting_capacity,
        )
        named_tensors = dict(zip(INGEST_CT.tensors, io))
        constants = INGEST_CT.scalars.pack(**scalar_fields)
        ct = INGEST_CT.pack(tt, tensors=named_tensors, **scalar_fields)
        parts.append(
            dict(
                io=io,
                constants=constants,
                compile=ct,
                tensors=named_tensors,
                scratch=scratch,
            )
        )
        storage.extend([d, b])
        rt = tt.RuntimeArgs()
        physical = [device.worker_core_from_logical_core(tt.CoreCoord(*p)) for p in g]
        peers = [v for p in physical for v in (p.x, p.y)]
        runtime_abi = ingest_runtime(len(g))
        for w, (x, y) in enumerate(g):
            rt[x][y] = runtime_abi.pack(
                **{name: t.buffer_address() for name, t in named_tensors.items()},
                worker=w,
                peers=peers,
            )
        cores = _ranges(tt, g)
        kernels.append(
            tt.KernelDescriptor(
                kernel_source=source_codegen.ingest(semaphore=SEM.INGEST_JOIN),
                source_type=tt.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=tt.ReaderConfigDescriptor(),
            )
        )
        cbs.append(
            tt.CBDescriptor(
                total_size=scratch,
                core_ranges=cores,
                format_descriptors=[
                    tt.CBFormatDescriptor(
                        buffer_index=StandaloneIngestCB.INGEST_WORKSPACE,
                        data_format=tt.uint8,
                        page_size=scratch,
                    )
                ],
            )
        )
    allcores = _ranges(tt, [p for g in groups for p in g])
    desc = tt.ProgramDescriptor(
        kernels=kernels,
        cbs=cbs,
        semaphores=[
            tt.SemaphoreDescriptor(id=i, core_ranges=allcores, initial_value=0)
            for i in SEM
        ],
    )
    io = [t for p in parts for t in p["io"]]
    return dict(
        kind="distributed",
        run=lambda: tt.generic_op(io, desc),
        descriptor=desc,
        ids=ids,
        prefix=prefix,
        counts=status,
        clusters=clusters,
        storage=storage,
        input_clusters=cluster_ids,
        program_count=1,
        capacity=capacity,
        internal_l1_tensor_count=4 + slots,
        layouts=[p["stats"] for p in packs],
        host_prefix=host,
        parts=parts,
        prefix_page_bytes=4096,
        workers_per_cell=[len(g) for g in groups],
        scorer_layout="fixed-worker-live-prefix-v1",
        prefix_layout="narrow-key-major-v1",
        diagnostics=diagnostics,
    )
