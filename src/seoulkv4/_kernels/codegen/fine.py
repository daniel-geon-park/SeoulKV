"""Explicit Fine source generation for packed input, live selection and diagnostics."""

from seoulkv4._kernels.stage1_fine.abi import FineSemaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

from seoulkv4._kernels.stage1_fine.abi import FineCB as CB
from seoulkv4._kernels.common.cb_layout import cpp_defines

from seoulkv4._kernels.codegen.preprocessor import profile_defines
from seoulkv4._kernels.stage1_fine.abi import fine_defines


def reader(*, select, shared_upper, external_tags, dynamic, dual_risc, profile):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if external_tags and (not select):
        raise ValueError("external tags require selection")
    if dynamic and (not external_tags):
        raise ValueError("dynamic count requires external tags")
    if dual_risc and (not (select and shared_upper)):
        raise ValueError("dual RISC requires shared upper and selection")
    if profile not in (
        "plain",
        "packed",
        "issue",
        "split",
        "fetch",
        "fetch_issue",
        "fetch_split",
        "cleanup",
    ):
        raise ValueError("unknown Fine diagnostic profile")
    if profile.startswith("fetch") and (not shared_upper or dual_risc):
        raise ValueError("fetch diagnostics require the isolated NCRISC control")
    if profile == "cleanup" and (not select):
        raise ValueError("cleanup clocks require the selector")
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + fine_defines("reader", select=select, dynamic=dynamic, dual_risc=dual_risc)
        + profile_defines(
            profile=profile,
            select=select,
            external_tags=external_tags,
            dynamic=dynamic,
            shared_upper=shared_upper,
            dual_risc=dual_risc,
        )
        + (
            r"""#if SKV4_PROFILE_PACKED_CLOCKS
#include "tools/profiler/kernel_profiler.hpp"
#endif
#include "api/dataflow/dataflow_api.h"
#if SKV4_PROFILE_PACKED_CLOCKS
// fine-ingest-meter-v1
inline __attribute__((always_inline)) uint32_t skv_tick() {
    asm volatile("" ::: "memory");
    uint32_t t=*reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory"); return t;
}
uint32_t skv_sum_FP_R_IDS;
uint32_t skv_sum_FP_R_META_RESERVE;
uint32_t skv_sum_FP_R_TABLE_FETCH;
uint32_t skv_sum_FP_R_FIRST_PLANES_ISSUE;
uint32_t skv_sum_FP_R_TABLE_COMPACT;
uint32_t skv_sum_FP_R_QUERY;
uint32_t skv_sum_FP_R_RING_WAIT;
uint32_t skv_sum_FP_R_TAIL_ZERO;
uint32_t skv_sum_FP_R_PLANES_FETCH;
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
#if SKV4_PROFILE_FETCH_GROUP
uint32_t skv_sum_FNR_LO_ISSUE, skv_end_FNR_LO_ISSUE; bool skv_valid_FNR_LO_ISSUE;
uint32_t skv_sum_FNR_HI_ISSUE, skv_end_FNR_HI_ISSUE; bool skv_valid_FNR_HI_ISSUE;
uint32_t skv_sum_FNR_MIXED_WAIT, skv_end_FNR_MIXED_WAIT; bool skv_valid_FNR_MIXED_WAIT;
uint32_t skv_sum_FNR_SHUFFLE, skv_end_FNR_SHUFFLE; bool skv_valid_FNR_SHUFFLE;
uint32_t skv_sum_FNR_LO_TAIL, skv_end_FNR_LO_TAIL; bool skv_valid_FNR_LO_TAIL;
uint32_t skv_sum_FNR_Q_FIRST_FENCE, skv_end_FNR_Q_FIRST_FENCE; bool skv_valid_FNR_Q_FIRST_FENCE;
#endif
#endif
#else
#if SKV4_PROFILE_FETCH_GROUP
uint32_t skv_sum_FNR_LO_ISSUE, skv_end_FNR_LO_ISSUE; bool skv_valid_FNR_LO_ISSUE;
uint32_t skv_sum_FNR_HI_ISSUE, skv_end_FNR_HI_ISSUE; bool skv_valid_FNR_HI_ISSUE;
uint32_t skv_sum_FNR_MIXED_WAIT, skv_end_FNR_MIXED_WAIT; bool skv_valid_FNR_MIXED_WAIT;
uint32_t skv_sum_FNR_SHUFFLE, skv_end_FNR_SHUFFLE; bool skv_valid_FNR_SHUFFLE;
uint32_t skv_sum_FNR_LO_TAIL, skv_end_FNR_LO_TAIL; bool skv_valid_FNR_LO_TAIL;
uint32_t skv_sum_FNR_Q_FIRST_FENCE, skv_end_FNR_Q_FIRST_FENCE; bool skv_valid_FNR_Q_FIRST_FENCE;
#endif
#endif
#endif
#endif
void kernel_main() {
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
#if SKV4_PROFILE_FETCH_GROUP
    skv_sum_FNR_LO_ISSUE=0; skv_valid_FNR_LO_ISSUE=false;
    skv_sum_FNR_HI_ISSUE=0; skv_valid_FNR_HI_ISSUE=false;
    skv_sum_FNR_MIXED_WAIT=0; skv_valid_FNR_MIXED_WAIT=false;
    skv_sum_FNR_SHUFFLE=0; skv_valid_FNR_SHUFFLE=false;
    skv_sum_FNR_LO_TAIL=0; skv_valid_FNR_LO_TAIL=false;
    skv_sum_FNR_Q_FIRST_FENCE=0; skv_valid_FNR_Q_FIRST_FENCE=false;
    uint32_t fnr_misses=0;
#endif
#endif
#else
#if SKV4_PROFILE_FETCH_GROUP
    skv_sum_FNR_LO_ISSUE=0; skv_valid_FNR_LO_ISSUE=false;
    skv_sum_FNR_HI_ISSUE=0; skv_valid_FNR_HI_ISSUE=false;
    skv_sum_FNR_MIXED_WAIT=0; skv_valid_FNR_MIXED_WAIT=false;
    skv_sum_FNR_SHUFFLE=0; skv_valid_FNR_SHUFFLE=false;
    skv_sum_FNR_LO_TAIL=0; skv_valid_FNR_LO_TAIL=false;
    skv_sum_FNR_Q_FIRST_FENCE=0; skv_valid_FNR_Q_FIRST_FENCE=false;
    uint32_t fnr_misses=0;
#endif
#endif
#endif
    DeviceTimestampedData("SKV_FP_READER_B", 0);
    skv_sum_FP_R_IDS=0;skv_sum_FP_R_META_RESERVE=0;skv_sum_FP_R_TABLE_FETCH=0;skv_sum_FP_R_FIRST_PLANES_ISSUE=0;skv_sum_FP_R_TABLE_COMPACT=0;skv_sum_FP_R_QUERY=0;skv_sum_FP_R_RING_WAIT=0;skv_sum_FP_R_TAIL_ZERO=0;skv_sum_FP_R_PLANES_FETCH=0;
#endif
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    constexpr uint32_t MAX_LOCAL = get_compile_time_arg_val(SKV4_ARG_CT_CAPACITY);
#else
    constexpr uint32_t LIVE = get_compile_time_arg_val(SKV4_ARG_CT_CAPACITY);
    constexpr uint32_t CHUNKS = get_compile_time_arg_val(SKV4_ARG_CT_CHUNKS);
#endif
#else
    constexpr uint32_t LIVE = get_compile_time_arg_val(SKV4_ARG_CT_CAPACITY);
    constexpr uint32_t CHUNKS = get_compile_time_arg_val(SKV4_ARG_CT_CHUNKS);
#endif
    constexpr auto ua = TensorAccessorArgs<SKV4_ARG_CT_ACCESSORS>();
#else
    constexpr uint32_t LIVE = get_compile_time_arg_val(SKV4_ARG_CT_CAPACITY);
    constexpr uint32_t CHUNKS = (LIVE + 31) / 32;
    constexpr auto ua = TensorAccessorArgs<SKV4_ARG_CT_ACCESSORS>();
#endif
    constexpr auto la = TensorAccessorArgs<ua.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto qa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
#if SKV4_CODEGEN_SHARED_UPPER
    const auto upper = TensorAccessor(ua, get_arg_val<uint32_t>(SKV4_ARG_RT_UPPER), 4096);
#else
    const auto upper = TensorAccessor(ua, get_arg_val<uint32_t>(SKV4_ARG_RT_UPPER), 64);
#endif
    const auto lower = TensorAccessor(la, get_arg_val<uint32_t>(SKV4_ARG_RT_LOWER), 64);
    const auto tables = TensorAccessor(ta, get_arg_val<uint32_t>(SKV4_ARG_RT_TABLES), 64);
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(SKV4_ARG_RT_IDS), 4096);
#else
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(SKV4_ARG_RT_IDS), 64);
#endif
#else
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(SKV4_ARG_RT_IDS), 64);
#endif
    const auto query = TensorAccessor(qa, get_arg_val<uint32_t>(SKV4_ARG_RT_QUERY), 2048);
    const uint32_t ip = get_write_ptr(SKV4_CB_SELECTED_TAGS);
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
    const uint32_t upper_cache = get_write_ptr(SKV4_CB_UPPER_CACHE);
    uint32_t cached_upper_block = 0xffffffffu;
#endif
#else
    const uint32_t upper_cache = get_write_ptr(SKV4_CB_UPPER_CACHE);
    uint32_t cached_upper_block = 0xffffffffu;
#endif
#endif
    auto selected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ip);
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    // Program C writes total live count in the first word. Every RISC derives
    // the same 32-entry partition; no host count or per-worker control DMA.
    noc_async_read(get_noc_addr(0, ids), ip, 64);
    noc_async_read_barrier();
    const uint32_t TOTAL_LIVE = selected[0];
    const uint32_t TOTAL_CHUNKS = (TOTAL_LIVE + 31) / 32;
    const uint32_t WORKER = get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER);
    const uint32_t WORKERS = get_arg_val<uint32_t>(SKV4_ARG_RT_WORKERS);
    const uint32_t FIRST_CHUNK = TOTAL_CHUNKS * WORKER / WORKERS;
    const uint32_t END_CHUNK = TOTAL_CHUNKS * (WORKER + 1) / WORKERS;
    const uint32_t FIRST = FIRST_CHUNK * 32;
    const uint32_t CHUNKS = END_CHUNK - FIRST_CHUNK;
    const uint32_t LOCAL_PADDED = CHUNKS * 32;
    const uint32_t LIVE = TOTAL_LIVE > FIRST
        ? (TOTAL_LIVE - FIRST < LOCAL_PADDED ? TOTAL_LIVE - FIRST : LOCAL_PADDED)
        : 0;
    const uint32_t COHORT = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT);
    const uint32_t COHORT_FIRST_CHUNK = TOTAL_CHUNKS * (COHORT * 4) / WORKERS;
    const uint32_t COHORT_END_CHUNK = TOTAL_CHUNKS * ((COHORT + 1) * 4) / WORKERS;
    const uint32_t COHORT_BASE = COHORT_FIRST_CHUNK * 32;
    const uint32_t COHORT_WIDTH = (COHORT_END_CHUNK - COHORT_FIRST_CHUNK) * 32;
#endif
#endif
#endif
#if SKV4_PROFILE_PACKED_CLOCKS
    uint32_t tp_FP_R_IDS=skv_tick();
#endif
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
    uint32_t copied_ids = 0;
#if SKV4_CODEGEN_DYNAMIC
    const uint32_t ID_BYTES = CHUNKS * 32 * 4;
#else
    constexpr uint32_t ID_BYTES = CHUNKS * 32 * 4;
#endif
    while (copied_ids < ID_BYTES) {
#if SKV4_CODEGEN_DYNAMIC
        const uint32_t logical = (16 + FIRST) * 4 + copied_ids;
#else
        const uint32_t logical = (16 + get_arg_val<uint32_t>(SKV4_ARG_RT_FIRST)) * 4 + copied_ids;
#endif
        const uint32_t page_left = 4096 - logical % 4096;
        const uint32_t bytes = ID_BYTES - copied_ids < page_left
            ? ID_BYTES - copied_ids : page_left;
        noc_async_read(get_noc_addr(logical / 4096, ids) + logical % 4096,
                       ip + copied_ids, bytes);
        copied_ids += bytes;
#else
    for (uint32_t page = 0; page < CHUNKS * 2; ++page) {
        noc_async_read(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RT_FIRST) / 16 + page, ids), ip + page * 64, 64);
#endif
#else
    for (uint32_t page = 0; page < CHUNKS * 2; ++page) {
        noc_async_read(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RT_FIRST) / 16 + page, ids), ip + page * 64, 64);
#endif
    }
    noc_async_read_barrier();
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_R_IDS+=skv_tick()-tp_FP_R_IDS;
    uint32_t tp_FP_R_META_RESERVE=skv_tick();
#endif
    cb_reserve_back(SKV4_CB_SCALES, 1);
    const uint32_t meta = get_write_ptr(SKV4_CB_SCALES);
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    const uint32_t PADDED = CHUNKS * 32;
    constexpr uint32_t CACHE_OFF = MAX_LOCAL * 4;
#else
    constexpr uint32_t PADDED = CHUNKS * 32;
    constexpr uint32_t CACHE_OFF = PADDED * 4;
#endif
#else
    constexpr uint32_t PADDED = CHUNKS * 32;
    constexpr uint32_t CACHE_OFF = PADDED * 4;
#endif
#else
    constexpr uint32_t PADDED = CHUNKS * 32;
    constexpr uint32_t CACHE_OFF = PADDED * 4;
#endif
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_R_META_RESERVE+=skv_tick()-tp_FP_R_META_RESERVE;
    uint32_t tp_FP_R_TABLE_FETCH=skv_tick();
#endif
    // Read scale pages from the compact group table into a run-coalesced cache.
    // Each FP32 entry applies to 32 physical centroids, and one 64-B page therefore
    // covers 512 centroids. IDs are ascending in production, so each page is
    // fetched once; arbitrary diagnostic order remains correct and merely
    // refetches a page.
    uint32_t page_count = 0;
    uint32_t last_page = 0xffffffffu;
    for (uint32_t key = 0; key < LIVE; ++key) {
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
        const uint32_t page = (selected[key] & 0x0fffffffu) / 512;
#else
        const uint32_t page = selected[key] / 512;
#endif
#else
        const uint32_t page = selected[key] / 512;
#endif
        if (page != last_page) {
            noc_async_read(get_noc_addr(page, tables),
                meta + CACHE_OFF + page_count * 64, 64);
            ++page_count;
            last_page = page;
            if ((page_count & 31u) == 0) noc_async_read_barrier();
        }
    }
    noc_async_read_barrier();
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_R_TABLE_FETCH+=skv_tick()-tp_FP_R_TABLE_FETCH;
    uint32_t tp_FP_R_FIRST_PLANES_ISSUE=skv_tick();
#endif
    // Prefetch first residual-plane chunk. Its DMA can progress while this
    // NCRISC compacts the already-resident scale cache below.
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    if (CHUNKS) {
#endif
#endif
#endif
        cb_reserve_back(SKV4_CB_PACKED_KEY_PLANES, 1);
        const uint32_t first_tile = get_write_ptr(SKV4_CB_PACKED_KEY_PLANES);
    if (32 > LIVE) {
        auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(first_tile);
        for (uint32_t i = 0; i < 1024; ++i) words[i] = 0;
    }
    for (uint32_t row = 0; row < 32 && row < LIVE; ++row) {
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
        const uint32_t key = selected[row] & 0x0fffffffu;
#else
        const uint32_t key = selected[row];
#endif
#else
        const uint32_t key = selected[row];
#endif
        const uint32_t dst = first_tile + (row / 16) * 2048 + (row % 16) * 64;
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
        { const uint32_t sample = skv_tick();
#endif
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
        { const uint32_t sample = skv_tick();
#endif
#endif
#endif
#endif
        noc_async_read(get_noc_addr(key, lower), dst, 64);
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
            skv_end_FNR_LO_ISSUE=skv_tick(); skv_valid_FNR_LO_ISSUE=true;
            skv_sum_FNR_LO_ISSUE+=skv_end_FNR_LO_ISSUE-sample;
        }
#endif
#endif
        {
            const uint32_t block = key >> 5;
            if (block != cached_upper_block) {
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                { const uint32_t sample = skv_tick();
#endif
#endif
                noc_async_read(get_noc_addr(block >> 1, upper) + (block & 1u) * 2048,
                               upper_cache, 2048);
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                    skv_end_FNR_HI_ISSUE=skv_tick(); skv_valid_FNR_HI_ISSUE=true;
                    skv_sum_FNR_HI_ISSUE+=skv_end_FNR_HI_ISSUE-sample; ++fnr_misses;
                }
                { const uint32_t sample = skv_tick();
#endif
#endif
                noc_async_read_barrier();
                cached_upper_block = block;
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                    skv_end_FNR_MIXED_WAIT=skv_tick(); skv_valid_FNR_MIXED_WAIT=true;
                    skv_sum_FNR_MIXED_WAIT+=skv_end_FNR_MIXED_WAIT-sample;
#endif
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
            skv_end_FNR_LO_ISSUE=skv_tick(); skv_valid_FNR_LO_ISSUE=true;
            skv_sum_FNR_LO_ISSUE+=skv_end_FNR_LO_ISSUE-sample;
        }
#endif
#endif
        {
            const uint32_t block = key >> 5;
            if (block != cached_upper_block) {
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                { const uint32_t sample = skv_tick();
#endif
#endif
                noc_async_read(get_noc_addr(block >> 1, upper) + (block & 1u) * 2048,
                               upper_cache, 2048);
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                    skv_end_FNR_HI_ISSUE=skv_tick(); skv_valid_FNR_HI_ISSUE=true;
                    skv_sum_FNR_HI_ISSUE+=skv_end_FNR_HI_ISSUE-sample; ++fnr_misses;
                }
                { const uint32_t sample = skv_tick();
#endif
#endif
                noc_async_read_barrier();
                cached_upper_block = block;
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                    skv_end_FNR_MIXED_WAIT=skv_tick(); skv_valid_FNR_MIXED_WAIT=true;
                    skv_sum_FNR_MIXED_WAIT+=skv_end_FNR_MIXED_WAIT-sample;
#endif
#endif
            }
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
            }
            { const uint32_t sample = skv_tick();
#endif
#endif
            const auto src = reinterpret_cast<const tt_l1_ptr uint32_t*>(upper_cache);
            auto out = reinterpret_cast<tt_l1_ptr uint32_t*>(dst + 1024);
            const uint32_t k = key & 31u;
            const uint32_t base = (k / 16) * 64 + (k % 16) * 2;
            #pragma GCC unroll 4
            for (uint32_t group = 0; group < 4; ++group) {
                out[group * 4] = src[base + group * 128];
                out[group * 4 + 1] = src[base + group * 128 + 1];
                out[group * 4 + 2] = src[base + group * 128 + 32];
                out[group * 4 + 3] = src[base + group * 128 + 33];
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
            }
                skv_end_FNR_SHUFFLE=skv_tick(); skv_valid_FNR_SHUFFLE=true;
                skv_sum_FNR_SHUFFLE+=skv_end_FNR_SHUFFLE-sample;
#endif
#endif
            }
        }
#endif
#else
        noc_async_read(get_noc_addr(key, upper), dst + 1024, 64);
#endif
    }
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if !SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
    skv_sum_FP_R_FIRST_PLANES_ISSUE+=skv_tick()-tp_FP_R_FIRST_PLANES_ISSUE;
    uint32_t tp_FP_R_TABLE_COMPACT=skv_tick();
#else
#if SKV4_PROFILE_FETCH_GROUP
            }
            { const uint32_t sample = skv_tick();
#endif
#endif
#else
    skv_sum_FP_R_FIRST_PLANES_ISSUE+=skv_tick()-tp_FP_R_FIRST_PLANES_ISSUE;
    uint32_t tp_FP_R_TABLE_COMPACT=skv_tick();
#endif
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
    skv_sum_FP_R_FIRST_PLANES_ISSUE+=skv_tick()-tp_FP_R_FIRST_PLANES_ISSUE;
    uint32_t tp_FP_R_TABLE_COMPACT=skv_tick();
#else
#if SKV4_PROFILE_FETCH_GROUP
            }
            { const uint32_t sample = skv_tick();
#endif
#endif
#else
    skv_sum_FP_R_FIRST_PLANES_ISSUE+=skv_tick()-tp_FP_R_FIRST_PLANES_ISSUE;
    uint32_t tp_FP_R_TABLE_COMPACT=skv_tick();
#endif
#endif
#endif
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
    // Zeroed tail and the descriptor must drain before BRISC sees the job.
    cb_reserve_back(SKV4_CB_UPPER_JOB, 1);
    auto job = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_UPPER_JOB));
    job[0] = first_tile;
    job[1] = 0;
    job[2] = LIVE < 32 ? LIVE : 32;
    __sync_synchronize();
    cb_push_back(SKV4_CB_UPPER_JOB, 1);
#else
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
            }
                { const uint32_t sample = skv_tick();
#endif
#endif
#endif
#endif
            const auto src = reinterpret_cast<const tt_l1_ptr uint32_t*>(upper_cache);
            auto out = reinterpret_cast<tt_l1_ptr uint32_t*>(dst + 1024);
            const uint32_t k = key & 31u;
            const uint32_t base = (k / 16) * 64 + (k % 16) * 2;
            #pragma GCC unroll 4
            for (uint32_t group = 0; group < 4; ++group) {
                out[group * 4] = src[base + group * 128];
                out[group * 4 + 1] = src[base + group * 128 + 1];
                out[group * 4 + 2] = src[base + group * 128 + 32];
                out[group * 4 + 3] = src[base + group * 128 + 33];
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
            }
                skv_end_FNR_SHUFFLE=skv_tick(); skv_valid_FNR_SHUFFLE=true;
                skv_sum_FNR_SHUFFLE+=skv_end_FNR_SHUFFLE-sample;
#endif
#endif
            }
        }
    }
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if !SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_R_FIRST_PLANES_ISSUE+=skv_tick()-tp_FP_R_FIRST_PLANES_ISSUE;
    uint32_t tp_FP_R_TABLE_COMPACT=skv_tick();
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_R_FIRST_PLANES_ISSUE+=skv_tick()-tp_FP_R_FIRST_PLANES_ISSUE;
    uint32_t tp_FP_R_TABLE_COMPACT=skv_tick();
#endif
#endif
#endif
#endif
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    }
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_R_FIRST_PLANES_ISSUE+=skv_tick()-tp_FP_R_FIRST_PLANES_ISSUE;
    uint32_t tp_FP_R_TABLE_COMPACT=skv_tick();
#endif
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_R_FIRST_PLANES_ISSUE+=skv_tick()-tp_FP_R_FIRST_PLANES_ISSUE;
    uint32_t tp_FP_R_TABLE_COMPACT=skv_tick();
#endif
#endif
    auto table_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
    auto cached_pages = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        meta + CACHE_OFF);
    uint32_t page_slot = 0;
    last_page = 0xffffffffu;
    for (uint32_t key = 0; key < LIVE; ++key) {
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
        const uint32_t page = (selected[key] & 0x0fffffffu) / 512;
#else
        const uint32_t page = selected[key] / 512;
#endif
#else
        const uint32_t page = selected[key] / 512;
#endif
        if (page != last_page) {
            if (last_page != 0xffffffffu) ++page_slot;
            last_page = page;
        }
        table_words[key] = cached_pages[
            page_slot * 16 + ((selected[key] >> 5) & 15u)];
    }
    for (uint32_t w = LIVE; w < PADDED; ++w) table_words[w] = 0;
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    cb_reserve_back(SKV4_CB_LIVE_CONTROL, 1);
    auto dynamic = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(SKV4_CB_LIVE_CONTROL));
    dynamic[0] = CHUNKS;
    dynamic[1] = (COHORT_WIDTH + 511) / 512;
    dynamic[2] = dynamic[1] ? COHORT_WIDTH - (dynamic[1] - 1) * 512 : 0;
    dynamic[3] = COHORT_BASE;
    dynamic[4] = TOTAL_CHUNKS * 32;
    dynamic[5] = TOTAL_LIVE;
    dynamic[6] = FIRST;
    dynamic[7] = LIVE;
#endif
#endif
#endif
    __sync_synchronize();
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    cb_push_back(SKV4_CB_LIVE_CONTROL, 1);
#endif
#endif
#endif
    cb_push_back(SKV4_CB_SCALES, 1);
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_R_TABLE_COMPACT+=skv_tick()-tp_FP_R_TABLE_COMPACT;
    uint32_t tp_FP_R_QUERY=skv_tick();
#endif
    cb_reserve_back(SKV4_CB_QUERY, 8);
    for (uint32_t t = 0; t < 8; ++t) {
        noc_async_read(get_noc_addr(t, query), get_write_ptr(SKV4_CB_QUERY) + t * 2048, 2048);
    }
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
    { const uint32_t sample = skv_tick();
#endif
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
    { const uint32_t sample = skv_tick();
#endif
#endif
#endif
#endif
    noc_async_read_barrier();
    cb_push_back(SKV4_CB_QUERY, 8);
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
        skv_end_FNR_Q_FIRST_FENCE=skv_tick(); skv_valid_FNR_Q_FIRST_FENCE=true;
        skv_sum_FNR_Q_FIRST_FENCE+=skv_end_FNR_Q_FIRST_FENCE-sample;
    }
#endif
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
        skv_end_FNR_Q_FIRST_FENCE=skv_tick(); skv_valid_FNR_Q_FIRST_FENCE=true;
        skv_sum_FNR_Q_FIRST_FENCE+=skv_end_FNR_Q_FIRST_FENCE-sample;
    }
#endif
#endif
#endif
#endif
    // The query fence also completes the first residual-plane chunk.
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_CODEGEN_SHARED_UPPER
    __sync_synchronize();
#if SKV4_CODEGEN_DUAL_RISC
    if (CHUNKS) { cb_wait_front(SKV4_CB_UPPER_DONE, 1); cb_pop_front(SKV4_CB_UPPER_DONE, 1); __sync_synchronize(); cb_push_back(SKV4_CB_PACKED_KEY_PLANES, 1); }
#else
    if (CHUNKS) cb_push_back(SKV4_CB_PACKED_KEY_PLANES, 1);
#endif
#else
    if (CHUNKS) cb_push_back(SKV4_CB_PACKED_KEY_PLANES, 1);
#endif
#else
#if SKV4_CODEGEN_SHARED_UPPER
    __sync_synchronize();
#if SKV4_CODEGEN_DUAL_RISC
    cb_wait_front(SKV4_CB_UPPER_DONE, 1); cb_pop_front(SKV4_CB_UPPER_DONE, 1);
#endif
#endif
    cb_push_back(SKV4_CB_PACKED_KEY_PLANES, 1);
#endif
#else
#if SKV4_CODEGEN_SHARED_UPPER
    __sync_synchronize();
#if SKV4_CODEGEN_DUAL_RISC
    cb_wait_front(SKV4_CB_UPPER_DONE, 1); cb_pop_front(SKV4_CB_UPPER_DONE, 1);
#endif
#endif
    cb_push_back(SKV4_CB_PACKED_KEY_PLANES, 1);
#endif
#else
#if SKV4_CODEGEN_SHARED_UPPER
    __sync_synchronize();
#endif
    cb_push_back(SKV4_CB_PACKED_KEY_PLANES, 1);
#endif
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_R_QUERY+=skv_tick()-tp_FP_R_QUERY;
#endif
    for (uint32_t chunk = 1; chunk < CHUNKS; ++chunk) {
#if SKV4_PROFILE_PACKED_CLOCKS
        uint32_t tp_FP_R_RING_WAIT=skv_tick();
#endif
        cb_reserve_back(SKV4_CB_PACKED_KEY_PLANES, 1);
#if SKV4_PROFILE_PACKED_CLOCKS
        skv_sum_FP_R_RING_WAIT+=skv_tick()-tp_FP_R_RING_WAIT;
        uint32_t tp_FP_R_TAIL_ZERO=skv_tick();
#endif
        const uint32_t tile = get_write_ptr(SKV4_CB_PACKED_KEY_PLANES);
        if (chunk * 32 + 32 > LIVE) {
            auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
            for (uint32_t i = 0; i < 1024; ++i) words[i] = 0;
        }
#if SKV4_PROFILE_PACKED_CLOCKS
        skv_sum_FP_R_TAIL_ZERO+=skv_tick()-tp_FP_R_TAIL_ZERO;
        uint32_t tp_FP_R_PLANES_FETCH=skv_tick();
#endif
        for (uint32_t row = 0; row < 32 && chunk * 32 + row < LIVE; ++row) {
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
            const uint32_t key = selected[chunk * 32 + row] & 0x0fffffffu;
#else
            const uint32_t key = selected[chunk * 32 + row];
#endif
#else
            const uint32_t key = selected[chunk * 32 + row];
#endif
            const uint32_t dst = tile + (row / 16) * 2048 + (row % 16) * 64;
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
            { const uint32_t sample = skv_tick();
#endif
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
            { const uint32_t sample = skv_tick();
#endif
#endif
#endif
#endif
            noc_async_read(get_noc_addr(key, lower), dst, 64);
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                skv_end_FNR_LO_ISSUE=skv_tick(); skv_valid_FNR_LO_ISSUE=true;
                skv_sum_FNR_LO_ISSUE+=skv_end_FNR_LO_ISSUE-sample;
            }
#endif
#endif
            {
            const uint32_t block = key >> 5;
            if (block != cached_upper_block) {
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                { const uint32_t sample = skv_tick();
#endif
#endif
                    noc_async_read(get_noc_addr(block >> 1, upper) + (block & 1u) * 2048,
                               upper_cache, 2048);
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                        skv_end_FNR_HI_ISSUE=skv_tick(); skv_valid_FNR_HI_ISSUE=true;
                        skv_sum_FNR_HI_ISSUE+=skv_end_FNR_HI_ISSUE-sample; ++fnr_misses;
                    }
                    { const uint32_t sample = skv_tick();
#endif
#endif
                noc_async_read_barrier();
                cached_upper_block = block;
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                        skv_end_FNR_MIXED_WAIT=skv_tick(); skv_valid_FNR_MIXED_WAIT=true;
                        skv_sum_FNR_MIXED_WAIT+=skv_end_FNR_MIXED_WAIT-sample;
#endif
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                skv_end_FNR_LO_ISSUE=skv_tick(); skv_valid_FNR_LO_ISSUE=true;
                skv_sum_FNR_LO_ISSUE+=skv_end_FNR_LO_ISSUE-sample;
            }
#endif
#endif
            {
            const uint32_t block = key >> 5;
            if (block != cached_upper_block) {
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                { const uint32_t sample = skv_tick();
#endif
#endif
                    noc_async_read(get_noc_addr(block >> 1, upper) + (block & 1u) * 2048,
                               upper_cache, 2048);
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                        skv_end_FNR_HI_ISSUE=skv_tick(); skv_valid_FNR_HI_ISSUE=true;
                        skv_sum_FNR_HI_ISSUE+=skv_end_FNR_HI_ISSUE-sample; ++fnr_misses;
                    }
                    { const uint32_t sample = skv_tick();
#endif
#endif
                noc_async_read_barrier();
                cached_upper_block = block;
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                        skv_end_FNR_MIXED_WAIT=skv_tick(); skv_valid_FNR_MIXED_WAIT=true;
                        skv_sum_FNR_MIXED_WAIT+=skv_end_FNR_MIXED_WAIT-sample;
#endif
#endif
#endif
#else
            noc_async_read(get_noc_addr(key, upper), dst + 1024, 64);
#endif
        }
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_DUAL_RISC
    // Zeroed tail and the descriptor must drain before BRISC sees the job.
    cb_reserve_back(SKV4_CB_UPPER_JOB, 1);
    auto job = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_UPPER_JOB));
    job[0] = tile;
    job[1] = chunk;
    job[2] = LIVE - chunk * 32 < 32 ? LIVE - chunk * 32 : 32;
    __sync_synchronize();
    cb_push_back(SKV4_CB_UPPER_JOB, 1);
#else
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
            }
                { const uint32_t sample = skv_tick();
#endif
#endif
            const auto src = reinterpret_cast<const tt_l1_ptr uint32_t*>(upper_cache);
            auto out = reinterpret_cast<tt_l1_ptr uint32_t*>(dst + 1024);
            const uint32_t k = key & 31u;
            const uint32_t base = (k / 16) * 64 + (k % 16) * 2;
            #pragma GCC unroll 4
            for (uint32_t group = 0; group < 4; ++group) {
                out[group * 4] = src[base + group * 128];
                out[group * 4 + 1] = src[base + group * 128 + 1];
                out[group * 4 + 2] = src[base + group * 128 + 32];
                out[group * 4 + 3] = src[base + group * 128 + 33];
            }
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                    skv_end_FNR_SHUFFLE=skv_tick(); skv_valid_FNR_SHUFFLE=true;
                    skv_sum_FNR_SHUFFLE+=skv_end_FNR_SHUFFLE-sample;
#endif
#endif
            }
        }
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
        }
        { const uint32_t sample = skv_tick();
#endif
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
            }
                { const uint32_t sample = skv_tick();
#endif
#endif
            const auto src = reinterpret_cast<const tt_l1_ptr uint32_t*>(upper_cache);
            auto out = reinterpret_cast<tt_l1_ptr uint32_t*>(dst + 1024);
            const uint32_t k = key & 31u;
            const uint32_t base = (k / 16) * 64 + (k % 16) * 2;
            #pragma GCC unroll 4
            for (uint32_t group = 0; group < 4; ++group) {
                out[group * 4] = src[base + group * 128];
                out[group * 4 + 1] = src[base + group * 128 + 1];
                out[group * 4 + 2] = src[base + group * 128 + 32];
                out[group * 4 + 3] = src[base + group * 128 + 33];
            }
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
                    skv_end_FNR_SHUFFLE=skv_tick(); skv_valid_FNR_SHUFFLE=true;
                    skv_sum_FNR_SHUFFLE+=skv_end_FNR_SHUFFLE-sample;
#endif
#endif
            }
        }
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_FETCH_GROUP
        }
        { const uint32_t sample = skv_tick();
#endif
#endif
#endif
#endif
        noc_async_read_barrier();
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
        cb_wait_front(SKV4_CB_UPPER_DONE, 1); cb_pop_front(SKV4_CB_UPPER_DONE, 1);
#endif
#endif
#endif
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
#if SKV4_PROFILE_FETCH_GROUP
            skv_end_FNR_LO_TAIL=skv_tick(); skv_valid_FNR_LO_TAIL=true;
            skv_sum_FNR_LO_TAIL+=skv_end_FNR_LO_TAIL-sample;
        }
#endif
#endif
#else
#if SKV4_PROFILE_FETCH_GROUP
            skv_end_FNR_LO_TAIL=skv_tick(); skv_valid_FNR_LO_TAIL=true;
            skv_sum_FNR_LO_TAIL+=skv_end_FNR_LO_TAIL-sample;
        }
#endif
#endif
#endif
        skv_sum_FP_R_PLANES_FETCH+=skv_tick()-tp_FP_R_PLANES_FETCH;
#endif
#if SKV4_CODEGEN_SHARED_UPPER
        __sync_synchronize();
#endif
    cb_push_back(SKV4_CB_PACKED_KEY_PLANES, 1);
    }
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKVA_FP_R_IDS",skv_sum_FP_R_IDS);
    DeviceTimestampedData("SKVA_FP_R_META_RESERVE",skv_sum_FP_R_META_RESERVE);
    DeviceTimestampedData("SKVA_FP_R_TABLE_FETCH",skv_sum_FP_R_TABLE_FETCH);
    DeviceTimestampedData("SKVA_FP_R_FIRST_PLANES_ISSUE",skv_sum_FP_R_FIRST_PLANES_ISSUE);
    DeviceTimestampedData("SKVA_FP_R_TABLE_COMPACT",skv_sum_FP_R_TABLE_COMPACT);
    DeviceTimestampedData("SKVA_FP_R_QUERY",skv_sum_FP_R_QUERY);
    DeviceTimestampedData("SKVA_FP_R_RING_WAIT",skv_sum_FP_R_RING_WAIT);
    DeviceTimestampedData("SKVA_FP_R_TAIL_ZERO",skv_sum_FP_R_TAIL_ZERO);
    DeviceTimestampedData("SKVA_FP_R_PLANES_FETCH",skv_sum_FP_R_PLANES_FETCH);
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_SELECT
#if !SKV4_CODEGEN_DUAL_RISC
#if SKV4_PROFILE_FETCH_GROUP
    DeviceTimestampedData("SKVA_FNR_LO_ISSUE", skv_sum_FNR_LO_ISSUE);
    if (skv_valid_FNR_LO_ISSUE) DeviceTimestampedData("SKVE_FNR_LO_ISSUE", skv_end_FNR_LO_ISSUE);
    DeviceTimestampedData("SKVA_FNR_HI_ISSUE", skv_sum_FNR_HI_ISSUE);
    if (skv_valid_FNR_HI_ISSUE) DeviceTimestampedData("SKVE_FNR_HI_ISSUE", skv_end_FNR_HI_ISSUE);
    DeviceTimestampedData("SKVA_FNR_MIXED_WAIT", skv_sum_FNR_MIXED_WAIT);
    if (skv_valid_FNR_MIXED_WAIT) DeviceTimestampedData("SKVE_FNR_MIXED_WAIT", skv_end_FNR_MIXED_WAIT);
    DeviceTimestampedData("SKVA_FNR_SHUFFLE", skv_sum_FNR_SHUFFLE);
    if (skv_valid_FNR_SHUFFLE) DeviceTimestampedData("SKVE_FNR_SHUFFLE", skv_end_FNR_SHUFFLE);
    DeviceTimestampedData("SKVA_FNR_LO_TAIL", skv_sum_FNR_LO_TAIL);
    if (skv_valid_FNR_LO_TAIL) DeviceTimestampedData("SKVE_FNR_LO_TAIL", skv_end_FNR_LO_TAIL);
    DeviceTimestampedData("SKVA_FNR_Q_FIRST_FENCE", skv_sum_FNR_Q_FIRST_FENCE);
    if (skv_valid_FNR_Q_FIRST_FENCE) DeviceTimestampedData("SKVE_FNR_Q_FIRST_FENCE", skv_end_FNR_Q_FIRST_FENCE);
    DeviceTimestampedData("FNR_COUNT_KEYS", LIVE);
    DeviceTimestampedData("FNR_COUNT_CHUNKS", CHUNKS);
    DeviceTimestampedData("FNR_COUNT_UPPER_BLOCKS", fnr_misses);
#endif
#endif
#else
#if SKV4_PROFILE_FETCH_GROUP
    DeviceTimestampedData("SKVA_FNR_LO_ISSUE", skv_sum_FNR_LO_ISSUE);
    if (skv_valid_FNR_LO_ISSUE) DeviceTimestampedData("SKVE_FNR_LO_ISSUE", skv_end_FNR_LO_ISSUE);
    DeviceTimestampedData("SKVA_FNR_HI_ISSUE", skv_sum_FNR_HI_ISSUE);
    if (skv_valid_FNR_HI_ISSUE) DeviceTimestampedData("SKVE_FNR_HI_ISSUE", skv_end_FNR_HI_ISSUE);
    DeviceTimestampedData("SKVA_FNR_MIXED_WAIT", skv_sum_FNR_MIXED_WAIT);
    if (skv_valid_FNR_MIXED_WAIT) DeviceTimestampedData("SKVE_FNR_MIXED_WAIT", skv_end_FNR_MIXED_WAIT);
    DeviceTimestampedData("SKVA_FNR_SHUFFLE", skv_sum_FNR_SHUFFLE);
    if (skv_valid_FNR_SHUFFLE) DeviceTimestampedData("SKVE_FNR_SHUFFLE", skv_end_FNR_SHUFFLE);
    DeviceTimestampedData("SKVA_FNR_LO_TAIL", skv_sum_FNR_LO_TAIL);
    if (skv_valid_FNR_LO_TAIL) DeviceTimestampedData("SKVE_FNR_LO_TAIL", skv_end_FNR_LO_TAIL);
    DeviceTimestampedData("SKVA_FNR_Q_FIRST_FENCE", skv_sum_FNR_Q_FIRST_FENCE);
    if (skv_valid_FNR_Q_FIRST_FENCE) DeviceTimestampedData("SKVE_FNR_Q_FIRST_FENCE", skv_end_FNR_Q_FIRST_FENCE);
    DeviceTimestampedData("FNR_COUNT_KEYS", LIVE);
    DeviceTimestampedData("FNR_COUNT_CHUNKS", CHUNKS);
    DeviceTimestampedData("FNR_COUNT_UPPER_BLOCKS", fnr_misses);
#endif
#endif
#endif
    DeviceTimestampedData("SKV_FP_READER_E", 0);
#if SKV4_CODEGEN_SELECT
    { DeviceZoneScopedN("FINE_NATIVE_ROUTE_WAIT");
#endif
#endif
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    const uint32_t width = COHORT_WIDTH;
#else
    const uint32_t width = get_arg_val<uint32_t>(SKV4_ARG_RT_WIDTH);
#endif
#else
    const uint32_t width = get_arg_val<uint32_t>(SKV4_ARG_RT_WIDTH);
#endif
    const uint32_t cohort = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT);
    const uint32_t cohorts = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORTS);
    const uint32_t chunks = (width + 511) / 512;
    cb_reserve_back(SKV4_CB_LOCAL_SCORE_TILES, chunks);
    auto row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_LOCAL_SCORE_TILES));
    // Every 512-score input occupies the first half of one physical tile.
    // Writers cover the live slots; clear only the final page's short tail.
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
        // Native packet arrives at a level-private inbox, immutable this step.
        const uint32_t src = get_write_ptr(SKV4_CB_PEER_FRONTIERS) + level * 2048;
        noc_async_read(get_noc_addr(src), get_write_ptr(SKV4_CB_MERGE_VALUES), 2048);
        noc_async_read_barrier();
        cb_push_back(SKV4_CB_MERGE_VALUES, 1);
#endif
    }
#if SKV4_CODEGEN_SELECT
    }
#if SKV4_PROFILE_PACKED_CLOCKS
}
#endif
#endif
"""
        )
    )


def compute(*, select, shared_upper, external_tags, dynamic, dual_risc, profile):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if external_tags and (not select):
        raise ValueError("external tags require selection")
    if dynamic and (not external_tags):
        raise ValueError("dynamic count requires external tags")
    if dual_risc and (not (select and shared_upper)):
        raise ValueError("dual RISC requires shared upper and selection")
    if profile not in (
        "plain",
        "packed",
        "issue",
        "split",
        "fetch",
        "fetch_issue",
        "fetch_split",
        "cleanup",
    ):
        raise ValueError("unknown Fine diagnostic profile")
    if profile.startswith("fetch") and (not shared_upper or dual_risc):
        raise ValueError("fetch diagnostics require the isolated NCRISC control")
    if profile == "cleanup" and (not select):
        raise ValueError("cleanup clocks require the selector")
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + fine_defines("compute", select=select, dynamic=dynamic, dual_risc=dual_risc)
        + profile_defines(
            profile=profile,
            select=select,
            external_tags=external_tags,
            dynamic=dynamic,
            shared_upper=shared_upper,
            dual_risc=dual_risc,
        )
        + (
            r"""#if SKV4_CODEGEN_SELECT
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
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
    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<512>(dst);
    topk_xl_separate_indices_row_major_advance_chunk_base<512>();
}
#endif
#if SKV4_PROFILE_PACKED_CLOCKS
#include "tools/profiler/kernel_profiler.hpp"
#endif
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#if SKV4_PROFILE_PACKED_CLOCKS
// fine-ingest-meter-v1
inline __attribute__((always_inline)) uint32_t skv_tick() {
    asm volatile("" ::: "memory");
    uint32_t t=*reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory"); return t;
}
uint32_t skv_sum_FP_C_STARTUP;
uint32_t skv_sum_FP_C_TABLE_WAIT;
uint32_t skv_sum_FP_C_QUERY_WAIT;
uint32_t skv_sum_FP_C_INPUT_WAIT;
uint32_t skv_sum_FP_C_DECODE_ACQUIRE;
uint32_t skv_sum_FP_C_COPY_ISSUE;
uint32_t skv_sum_FP_C_EXPAND_ISSUE;
uint32_t skv_sum_FP_C_DECODE_DRAIN;
uint32_t skv_sum_FP_C_DECODE_PACK;
uint32_t skv_sum_FP_C_SCORE_ACQUIRE_CLEAR;
uint32_t skv_sum_FP_C_MATMUL_ISSUE;
uint32_t skv_sum_FP_C_SCORE_DRAIN;
uint32_t skv_sum_FP_C_SCORE_PACK;
#endif
#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void clear_score() {
    #pragma GCC unroll 32
    for(uint32_t i=0;i<32;++i)
        sfpi::dst_reg[i].mode<sfpi::DataLayout::F32>()=sfpi::vFloat(0.0f);
}
// fine-arithmetic-codec: fp16-scale-odd-v1
__attribute__((noinline,noclone)) void expand_eight_planes() {
    // Keep each input pair live through all eight planes. The last output
    // overwrites only this already-loaded pair in Dst0, never a future input.
    #pragma GCC unroll 1
    for (uint32_t source = 0; source < 16; ++source) {
        const uint32_t face = source / 8, v = source % 8;
        const sfpi::vUInt lo = sfpi::dst_reg[face * 16 + v]
            .mode<sfpi::DataLayout::U32>();
        const sfpi::vUInt hi = sfpi::dst_reg[face * 16 + 8 + v]
            .mode<sfpi::DataLayout::U32>();
        #pragma GCC unroll 1
        for (uint32_t output = 0; output < 16; ++output) {
            const uint32_t plane = output / 2, half = output % 2;
            const int shift = -int(4 * plane + 2 * half);
            const sfpi::vUInt code = (sfpi::shft(lo, shift) & 3u)
                | sfpi::shft(sfpi::shft(hi, shift) & 3u, 2);
            const sfpi::vInt odd = sfpi::vInt(code + code) - 15;
            const sfpi::vFloat val = sfpi::convert<sfpi::vFloat>(
                sfpi::convert<sfpi::vSMag>(odd), sfpi::RoundMode::Nearest);
            sfpi::dst_reg[((plane + 1) & 7u) * 32 + face * 16 + half * 8 + v]
                .mode<sfpi::DataLayout::F32>() = val;
        }
    }
}
inline void fine_post_scale(uint32_t address) {
    const auto scales = reinterpret_cast<const uint16_t*>(address);
    // A score vector covers four query rows and the eight even/odd columns
    // of a face. The query rows share the same centroid-column scales.
    #pragma GCC unroll 1
    for (uint32_t face = 0; face < 2; ++face) {
        // Both parities use the same lane predicate. Keep independent u16
        // loads: predicated runtime-u32 immediates trigger pinned SFPI ICEs.
        sfpi::vUInt even_lo = uint32_t(scales[face * 32]);
        sfpi::vUInt even_hi = uint32_t(scales[face * 32 + 1]);
        sfpi::vUInt odd_lo = uint32_t(scales[face * 32 + 2]);
        sfpi::vUInt odd_hi = uint32_t(scales[face * 32 + 3]);
        #pragma GCC unroll 1
        for (uint32_t j = 1; j < 8; ++j) {
            v_if ((sfpi::vConstTileId & 15) >= 2 * j) {
                even_lo = uint32_t(scales[face * 32 + 4 * j]);
                even_hi = uint32_t(scales[face * 32 + 4 * j + 1]);
                odd_lo = uint32_t(scales[face * 32 + 4 * j + 2]);
                odd_hi = uint32_t(scales[face * 32 + 4 * j + 3]);
            } v_endif;
        }
        #pragma GCC unroll 2
        for (uint32_t parity = 0; parity < 2; ++parity) {
            sfpi::vUInt lo = even_lo;
            sfpi::vUInt hi = even_hi;
            if (parity) {
                lo = odd_lo;
                hi = odd_hi;
            }
            sfpi::vFloat scale = sfpi::as<sfpi::vFloat>(sfpi::shft(hi, 16) | lo);
            // One face occupies eight SFPU Dst vectors. Offsets0/1 are the
            // even/odd columns of rows0..3; the right face is therefore8/9,
            // not16/17 (which belongs to the bottom-left face).
            sfpi::vFloat value = sfpi::dst_reg[face * 8 + parity];
            sfpi::dst_reg[face * 8 + parity] = value * scale;
        }
    }
}
}
#endif
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_WAIT_CLOCKS
#ifdef TRISC_MATH
inline uint32_t fw_wait=0, fw_issue=0, fw_tail=0, fw_work=0, fw_total=0, fw_count=0;
inline uint32_t fw_first_wait=0, fw_first_issue=0, fw_first_tail=0, fw_empty=0;
inline __attribute__((always_inline)) uint32_t fw_tick() {
    asm volatile("" ::: "memory");
    uint32_t t=*reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory");
    return t;
}
inline __attribute__((always_inline)) void fw_drain() {
    TTI_STALLWAIT(p_stall::STALL_SFPU,p_stall::WAIT_SFPU);
    tensix_sync();
}
#endif
#endif
#endif
inline void expand_eight() {
#if SKV4_PROFILE_PACKED_CLOCKS
    uint32_t tp_FP_C_DECODE_ACQUIRE=skv_tick();
#endif
    cb_reserve_back(SKV4_CB_DECODED_KEYS,8);
    tile_regs_acquire();
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_C_DECODE_ACQUIRE+=skv_tick()-tp_FP_C_DECODE_ACQUIRE;
    uint32_t tp_FP_C_COPY_ISSUE=skv_tick();
#endif
    copy_tile(SKV4_CB_PACKED_KEY_PLANES,0,0);
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_C_COPY_ISSUE+=skv_tick()-tp_FP_C_COPY_ISSUE;
    uint32_t tp_FP_C_EXPAND_ISSUE=skv_tick();
#if SKV4_PROFILE_WAIT_CLOCKS
#ifdef TRISC_MATH
    const uint32_t fw_t0=fw_tick();
#if SKV4_PROFILE_SPLIT_CLOCKS
    fw_drain();
#else
    // No completion fence.
#endif
    const uint32_t fw_t1=fw_tick();
#endif
#endif
#endif
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        expand_eight_planes,0,VectorMode::None));
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_WAIT_CLOCKS
#ifdef TRISC_MATH
    const uint32_t fw_t2=fw_tick();
#if SKV4_PROFILE_SPLIT_CLOCKS
    fw_drain();
#else
    // No completion fence.
#endif
    const uint32_t fw_t3=fw_tick();
    const uint32_t fw_e0=fw_tick();
#if SKV4_PROFILE_SPLIT_CLOCKS
    fw_drain();
#else
    // No completion fence.
#endif
    const uint32_t fw_e1=fw_tick();
    fw_wait+=fw_t1-fw_t0; fw_issue+=fw_t2-fw_t1; fw_tail+=fw_t3-fw_t2;
    fw_work+=fw_t3-fw_t1; fw_total+=fw_t3-fw_t0; fw_empty+=fw_e1-fw_e0;
    if(fw_count==0) {
        fw_first_wait=fw_t1-fw_t0; fw_first_issue=fw_t2-fw_t1;
        fw_first_tail=fw_t3-fw_t2;
    }
    ++fw_count;
#endif
#endif
    skv_sum_FP_C_EXPAND_ISSUE+=skv_tick()-tp_FP_C_EXPAND_ISSUE;
    uint32_t tp_FP_C_DECODE_DRAIN=skv_tick();
#endif
    tile_regs_commit();
    tile_regs_wait();
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_C_DECODE_DRAIN+=skv_tick()-tp_FP_C_DECODE_DRAIN;
    uint32_t tp_FP_C_DECODE_PACK=skv_tick();
#endif
    pack_tile(1,SKV4_CB_DECODED_KEYS);
    pack_tile(2,SKV4_CB_DECODED_KEYS);
    pack_tile(3,SKV4_CB_DECODED_KEYS);
    pack_tile(4,SKV4_CB_DECODED_KEYS);
    pack_tile(5,SKV4_CB_DECODED_KEYS);
    pack_tile(6,SKV4_CB_DECODED_KEYS);
    pack_tile(7,SKV4_CB_DECODED_KEYS);
    pack_tile(0,SKV4_CB_DECODED_KEYS);
    tile_regs_release();
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_C_DECODE_PACK+=skv_tick()-tp_FP_C_DECODE_PACK;
#endif
    cb_push_back(SKV4_CB_DECODED_KEYS,8);
}
#if SKV4_CODEGEN_SELECT
#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <uint32_t SCORE_TILE>
inline void fine_fuse_positions_impl(uint32_t base) {
    #pragma GCC unroll 1
    for (uint32_t i = 0; i < 32; ++i) {
        sfpi::vUInt score = sfpi::dst_reg[SCORE_TILE * 32 + i]
            .mode<sfpi::DataLayout::U32>();
        sfpi::vUInt position = sfpi::dst_reg[(SCORE_TILE + 1) * 32 + i]
            .mode<sfpi::DataLayout::U32>();
        sfpi::dst_reg[SCORE_TILE * 32 + i].mode<sfpi::DataLayout::U32>() =
            score | ((position + base) & 0xffffu);
    }
}
inline void fine_fuse_positions0(uint32_t base) {
    fine_fuse_positions_impl<0>(base);
}
inline void fine_fuse_positions1(uint32_t base) {
    fine_fuse_positions_impl<1>(base);
}
}
#endif
#endif
void kernel_main() {
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_PROFILE_WAIT_CLOCKS
#ifdef TRISC_MATH
    fw_wait=0;fw_issue=0;fw_tail=0;fw_work=0;fw_total=0;fw_count=0;fw_first_wait=0;fw_first_issue=0;fw_first_tail=0;fw_empty=0;
#endif
#endif
    DeviceTimestampedData("SKV_FP_COMPUTE_B", 0);
    skv_sum_FP_C_STARTUP=0;skv_sum_FP_C_TABLE_WAIT=0;skv_sum_FP_C_QUERY_WAIT=0;skv_sum_FP_C_INPUT_WAIT=0;skv_sum_FP_C_DECODE_ACQUIRE=0;skv_sum_FP_C_COPY_ISSUE=0;skv_sum_FP_C_EXPAND_ISSUE=0;skv_sum_FP_C_DECODE_DRAIN=0;skv_sum_FP_C_DECODE_PACK=0;skv_sum_FP_C_SCORE_ACQUIRE_CLEAR=0;skv_sum_FP_C_MATMUL_ISSUE=0;skv_sum_FP_C_SCORE_DRAIN=0;skv_sum_FP_C_SCORE_PACK=0;
#endif
    constexpr uint32_t N=get_compile_time_arg_val(SKV4_ARG_CT_N), PHASE=get_compile_time_arg_val(SKV4_ARG_CT_PHASE);
#if SKV4_PROFILE_PACKED_CLOCKS
    uint32_t tp_FP_C_STARTUP=skv_tick();
#endif
    compute_kernel_hw_startup(SKV4_CB_PACKED_KEY_PLANES,SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS);
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_C_STARTUP+=skv_tick()-tp_FP_C_STARTUP;
#endif
    uint32_t table_base = 0;
    // This mailbox is a DATA-readiness fence, not only an address transfer.
    // MATH's scalar L1 loads can run ahead of queued Tensix copy instructions.
#if SKV4_PROFILE_PACKED_CLOCKS
    uint32_t tp_FP_C_TABLE_WAIT=skv_tick();
#endif
    cb_wait_front(SKV4_CB_SCALES,1);
    UNPACK(mailbox_write(ckernel::ThreadId::MathThreadId,
        get_local_cb_interface(SKV4_CB_SCALES).fifo_rd_ptr << 4));
    MATH(table_base = mailbox_read(ckernel::ThreadId::UnpackThreadId));
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    cb_wait_front(SKV4_CB_LIVE_CONTROL, 1);
    // Unlike table_base (MATH-only), loop bounds are consumed by all three
    // TRISCs. The v0.77 helper broadcasts one identical address to each.
    const uint32_t dynamic_base = get_tile_address(7, 0);
    auto dynamic = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        dynamic_base);
    const uint32_t local_chunks = dynamic[0];
    const uint32_t cohort_chunks = dynamic[1];
    const uint32_t cohort_tail = dynamic[2];
    const uint32_t cohort_base = dynamic[3];
    cb_pop_front(SKV4_CB_LIVE_CONTROL, 1);
#endif
#endif
#endif
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_C_TABLE_WAIT+=skv_tick()-tp_FP_C_TABLE_WAIT;
    uint32_t tp_FP_C_QUERY_WAIT=skv_tick();
#endif
    cb_wait_front(SKV4_CB_QUERY,8);
#if SKV4_PROFILE_PACKED_CLOCKS
    skv_sum_FP_C_QUERY_WAIT+=skv_tick()-tp_FP_C_QUERY_WAIT;
#endif
    if constexpr(PHASE>0) {
        // Unfused TopK-XL leaves SFPU index tracking enabled across programs.
        // Custom SFPI execute calls do not initialize this state themselves.
        MATH(SFPU_UNARY_INIT(unused));
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
        for(uint32_t chunk=0;chunk<local_chunks;++chunk) {
#else
        for(uint32_t chunk=0;chunk<N/32;++chunk) {
#endif
#else
        for(uint32_t chunk=0;chunk<N/32;++chunk) {
#endif
#else
        for(uint32_t chunk=0;chunk<N/32;++chunk) {
#endif
#if SKV4_PROFILE_PACKED_CLOCKS
            uint32_t tp_FP_C_INPUT_WAIT=skv_tick();
#endif
            cb_wait_front(SKV4_CB_PACKED_KEY_PLANES,1);
#if SKV4_PROFILE_PACKED_CLOCKS
            skv_sum_FP_C_INPUT_WAIT+=skv_tick()-tp_FP_C_INPUT_WAIT;
#endif
            // A packed chunk is consumed immediately; the writer can drain
            // its scores while MATH starts decoding the next chunk.
            copy_tile_to_dst_init_short_with_dt(SKV4_CB_DECODED_KEYS,SKV4_CB_PACKED_KEY_PLANES);
            pack_reconfig_data_format(SKV4_CB_DECODED_KEYS);
            expand_eight();
            cb_pop_front(SKV4_CB_PACKED_KEY_PLANES,1);
            if constexpr(PHASE!=3) {
                if constexpr(PHASE==2) {
                    // Matrix maps Q -> SrcB, K -> SrcA, unlike tile-copy.
                    reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS);
                    matmul_init(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS,1);
                    pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
                }
#if SKV4_PROFILE_PACKED_CLOCKS
                uint32_t tp_FP_C_SCORE_ACQUIRE_CLEAR=skv_tick();
#endif
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
#if SKV4_PROFILE_PACKED_CLOCKS
                    skv_sum_FP_C_SCORE_ACQUIRE_CLEAR+=skv_tick()-tp_FP_C_SCORE_ACQUIRE_CLEAR;
                    uint32_t tp_FP_C_MATMUL_ISSUE=skv_tick();
#endif
                    for(uint32_t kt=0;kt<8;++kt) matmul_tiles(SKV4_CB_QUERY,SKV4_CB_DECODED_KEYS,kt,kt,0);
                    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                        DST_ACCUM_MODE,fine_post_scale,0,VectorMode::None,
                        table_base + chunk * 128));
#if SKV4_PROFILE_PACKED_CLOCKS
                    skv_sum_FP_C_MATMUL_ISSUE+=skv_tick()-tp_FP_C_MATMUL_ISSUE;
                    uint32_t tp_FP_C_SCORE_DRAIN=skv_tick();
#endif
                    tile_regs_commit(); tile_regs_wait();
#if SKV4_PROFILE_PACKED_CLOCKS
                    skv_sum_FP_C_SCORE_DRAIN+=skv_tick()-tp_FP_C_SCORE_DRAIN;
                    uint32_t tp_FP_C_SCORE_PACK=skv_tick();
#endif
                    pack_tile(0,SKV4_CB_SCORE_TILES);
                    tile_regs_release(); cb_push_back(SKV4_CB_SCORE_TILES,1);
#if SKV4_PROFILE_PACKED_CLOCKS
                    skv_sum_FP_C_SCORE_PACK+=skv_tick()-tp_FP_C_SCORE_PACK;
#endif
                }
                cb_pop_front(SKV4_CB_DECODED_KEYS,8);
            }
        }
    }
    cb_pop_front(SKV4_CB_QUERY,8); cb_pop_front(SKV4_CB_SCALES,1);
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKVA_FP_C_STARTUP",skv_sum_FP_C_STARTUP);
    DeviceTimestampedData("SKVA_FP_C_TABLE_WAIT",skv_sum_FP_C_TABLE_WAIT);
    DeviceTimestampedData("SKVA_FP_C_QUERY_WAIT",skv_sum_FP_C_QUERY_WAIT);
    DeviceTimestampedData("SKVA_FP_C_INPUT_WAIT",skv_sum_FP_C_INPUT_WAIT);
    DeviceTimestampedData("SKVA_FP_C_DECODE_ACQUIRE",skv_sum_FP_C_DECODE_ACQUIRE);
    DeviceTimestampedData("SKVA_FP_C_COPY_ISSUE",skv_sum_FP_C_COPY_ISSUE);
    DeviceTimestampedData("SKVA_FP_C_EXPAND_ISSUE",skv_sum_FP_C_EXPAND_ISSUE);
    DeviceTimestampedData("SKVA_FP_C_DECODE_DRAIN",skv_sum_FP_C_DECODE_DRAIN);
    DeviceTimestampedData("SKVA_FP_C_DECODE_PACK",skv_sum_FP_C_DECODE_PACK);
    DeviceTimestampedData("SKVA_FP_C_SCORE_ACQUIRE_CLEAR",skv_sum_FP_C_SCORE_ACQUIRE_CLEAR);
    DeviceTimestampedData("SKVA_FP_C_MATMUL_ISSUE",skv_sum_FP_C_MATMUL_ISSUE);
    DeviceTimestampedData("SKVA_FP_C_SCORE_DRAIN",skv_sum_FP_C_SCORE_DRAIN);
    DeviceTimestampedData("SKVA_FP_C_SCORE_PACK",skv_sum_FP_C_SCORE_PACK);
    DeviceTimestampedData("SKV_FP_COMPUTE_E", 0);
#if SKV4_PROFILE_WAIT_CLOCKS
#ifdef TRISC_MATH
    DeviceTimestampedData("FNW_WAIT",fw_wait);
    DeviceTimestampedData("FNW_ISSUE",fw_issue);
    DeviceTimestampedData("FNW_TAIL",fw_tail);
    DeviceTimestampedData("FNW_EXEC",fw_work);
    DeviceTimestampedData("FNW_TOTAL",fw_total);
    DeviceTimestampedData("FNW_COUNT",fw_count);
    DeviceTimestampedData("FNW_INITIAL_WAIT",fw_first_wait);
    DeviceTimestampedData("FNW_INITIAL_ISSUE",fw_first_issue);
    DeviceTimestampedData("FNW_INITIAL_TAIL",fw_first_tail);
    DeviceTimestampedData("FNW_EMPTY",fw_empty);
#endif
#endif
#endif
#if SKV4_CODEGEN_SELECT
    // Convert each locally sorted raw TopK coordinate to a global u16 Fine
    // position once, then keep [BF16 score | u16 position] fused through every
    // running and cohort-tree merge.
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    const uint32_t chunks = cohort_chunks;
    const uint32_t tail = cohort_tail;
    const uint32_t base = cohort_base;
#else
    const uint32_t chunks = get_arg_val<uint32_t>(SKV4_ARG_RT_CHUNKS);
    const uint32_t tail = get_arg_val<uint32_t>(SKV4_ARG_RT_TAIL);
    const uint32_t base = get_arg_val<uint32_t>(SKV4_ARG_RT_BASE);
#endif
#else
    const uint32_t chunks = get_arg_val<uint32_t>(SKV4_ARG_RT_CHUNKS);
    const uint32_t tail = get_arg_val<uint32_t>(SKV4_ARG_RT_TAIL);
    const uint32_t base = get_arg_val<uint32_t>(SKV4_ARG_RT_BASE);
#endif
    const uint32_t cohort = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT);
    const uint32_t cohorts = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORTS);
#if SKV4_PROFILE_PACKED_CLOCKS
    { DeviceZoneScopedN("FINE_NATIVE_LOCAL");
#endif
    pack_reconfig_data_format(SKV4_CB_FRONTIER_VALUES);
    tile_regs_acquire();
    topk_xl_separate_indices_row_major_init_static<0, 0>();
    fine_sort_chunk(0, chunks == 1 ? tail : 512, false);
    MATH(SFPU_UNARY_INIT(unused));
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, fine_fuse_positions0, 0,
        VectorMode::None, base));
    for (uint32_t chunk = 1; chunk < chunks; ++chunk) {
        // Tile1 becomes the next fused operand; tile2 is only the transient
        // UInt32 decode target and is overwritten on the next iteration.
        fine_sort_chunk(1, chunk + 1 == chunks ? tail : 512, true);
        MATH(SFPU_UNARY_INIT(unused));
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, fine_fuse_positions1, 0,
            VectorMode::None, base));
        topk_xl_init<512, true>();
        topk_xl_merge<512, true>(0);
        topk_xl_rebuild<512, true>(0, false);
#endif
    }
#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE_PACKED_CLOCKS
    }
    { DeviceZoneScopedN("FINE_NATIVE_TREE");
#endif
    for (uint32_t stride = 1; stride < cohorts; stride *= 2) {
        if (cohort % (2 * stride)) break;
        if (cohort + stride >= cohorts) continue;
        cb_wait_front(SKV4_CB_MERGE_VALUES, 1);
        topk_xl_copy_tile_init(SKV4_CB_MERGE_VALUES);
        topk_xl_copy_tile<512>(SKV4_CB_MERGE_VALUES, 1, 0, 512);
        cb_pop_front(SKV4_CB_MERGE_VALUES, 1);
        topk_xl_init<512, true>();
        topk_xl_rebuild<512, true>(1, true);
        topk_xl_merge<512, true>(0);
        topk_xl_rebuild<512, true>(0, false);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(SKV4_CB_FRONTIER_VALUES, 1);
    pack_reconfig_data_format(SKV4_CB_FRONTIER_VALUES);
    pack_tile(0, SKV4_CB_FRONTIER_VALUES);
    cb_push_back(SKV4_CB_FRONTIER_VALUES, 1);
    tile_regs_release();
    }
#if SKV4_PROFILE_PACKED_CLOCKS
}
#endif
#endif
"""
        )
    )


def writer(*, select, shared_upper, external_tags, dynamic, dual_risc, profile):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if external_tags and (not select):
        raise ValueError("external tags require selection")
    if dynamic and (not external_tags):
        raise ValueError("dynamic count requires external tags")
    if dual_risc and (not (select and shared_upper)):
        raise ValueError("dual RISC requires shared upper and selection")
    if profile not in (
        "plain",
        "packed",
        "issue",
        "split",
        "fetch",
        "fetch_issue",
        "fetch_split",
        "cleanup",
    ):
        raise ValueError("unknown Fine diagnostic profile")
    if profile.startswith("fetch") and (not shared_upper or dual_risc):
        raise ValueError("fetch diagnostics require the isolated NCRISC control")
    if profile == "cleanup" and (not select):
        raise ValueError("cleanup clocks require the selector")
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + fine_defines("writer", select=select, dynamic=dynamic, dual_risc=dual_risc)
        + profile_defines(
            profile=profile,
            select=select,
            external_tags=external_tags,
            dynamic=dynamic,
            shared_upper=shared_upper,
            dual_risc=dual_risc,
        )
        + (
            r"""#if SKV4_CODEGEN_SELECT
#if SKV4_PROFILE_PACKED_CLOCKS
#include "tools/profiler/kernel_profiler.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#else
#if SKV4_PROFILE_CLEANUP_CLOCKS
#include "tools/profiler/kernel_profiler.hpp"
#endif
#endif
#else
#if SKV4_PROFILE_PACKED_CLOCKS
#include "tools/profiler/kernel_profiler.hpp"
#endif
#endif
#include "api/dataflow/dataflow_api.h"
#if SKV4_PROFILE_PACKED_CLOCKS
// fine-ingest-meter-v1
inline __attribute__((always_inline)) uint32_t skv_tick() {
    asm volatile("" ::: "memory");
    uint32_t t=*reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory"); return t;
}
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
uint32_t skv_sum_FN_SERVICE_WAIT;
#else
uint32_t skv_sum_FN_SCORE_WAIT;
#endif
#else
uint32_t skv_sum_FN_SCORE_WAIT;
#endif
uint32_t skv_sum_FN_SCORE_STORE;
#else
uint32_t skv_sum_FP_W_SCORE_WAIT;
uint32_t skv_sum_FP_W_SCORE_STORE;
#endif
#endif
void kernel_main() {
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
    skv_sum_FN_SERVICE_WAIT=0;skv_sum_FN_SCORE_STORE=0;
#else
    skv_sum_FN_SCORE_WAIT=0;skv_sum_FN_SCORE_STORE=0;
#endif
#else
    skv_sum_FN_SCORE_WAIT=0;skv_sum_FN_SCORE_STORE=0;
#endif
#else
    DeviceTimestampedData("SKV_FP_WRITER_B", 0);
    skv_sum_FP_W_SCORE_WAIT=0;skv_sum_FP_W_SCORE_STORE=0;
#endif
#endif
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    constexpr uint32_t MAX_LOCAL = get_compile_time_arg_val(SKV4_ARG_CT_N);
    constexpr uint32_t MAX_TOTAL = get_compile_time_arg_val(SKV4_ARG_CT_TOTAL);
#else
    constexpr uint32_t N = get_compile_time_arg_val(SKV4_ARG_CT_N);
    constexpr uint32_t TOTAL = get_compile_time_arg_val(SKV4_ARG_CT_TOTAL);
    constexpr uint32_t LIVE = get_compile_time_arg_val(SKV4_ARG_CT_LIVE);
#endif
#else
    constexpr uint32_t N = get_compile_time_arg_val(SKV4_ARG_CT_N);
    constexpr uint32_t TOTAL = get_compile_time_arg_val(SKV4_ARG_CT_TOTAL);
    constexpr uint32_t LIVE = get_compile_time_arg_val(SKV4_ARG_CT_LIVE);
#endif
    constexpr uint32_t P = get_compile_time_arg_val(SKV4_ARG_CT_P);
    constexpr uint32_t OUT = get_compile_time_arg_val(SKV4_ARG_CT_OUT);
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if !SKV4_CODEGEN_DYNAMIC
    constexpr uint32_t MAX_LOCAL = get_compile_time_arg_val(SKV4_ARG_CT_MAX_LOCAL);
#endif
#else
    constexpr uint32_t MAX_LOCAL = get_compile_time_arg_val(SKV4_ARG_CT_MAX_LOCAL);
#endif
    constexpr auto sa = TensorAccessorArgs<SKV4_ARG_CT_ACCESSORS>();
    constexpr auto ta = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
    constexpr auto ua = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    const auto upper = TensorAccessor(ua, get_arg_val<uint32_t>(SKV4_ARG_RT_UPPER), 4096);
#else
    const auto upper = TensorAccessor(ua, get_arg_val<uint32_t>(SKV4_ARG_RT_UPPER), 4096);
#endif
#else
    const auto upper = TensorAccessor(ua, get_arg_val<uint32_t>(SKV4_ARG_RT_UPPER), 4096);
#endif
#endif
#endif
    const auto scores = TensorAccessor(sa, get_arg_val<uint32_t>(SKV4_ARG_RT_SCORES), 2048);
    //4KiB pages spread the lookup across L1 banks. One giant row placed all
    //four cells on bank0 and collided with Stage1.6's static CBs at large pools.
    const auto tags = TensorAccessor(ta, get_arg_val<uint32_t>(SKV4_ARG_RT_TAGS), 4096);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(SKV4_ARG_RT_OUTPUT), 64);
    const auto heads = TensorAccessor(ha, get_arg_val<uint32_t>(SKV4_ARG_RT_HEADS), P * 4);
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if !SKV4_CODEGEN_DYNAMIC
    const uint32_t first = get_arg_val<uint32_t>(SKV4_ARG_RT_FIRST);
    const uint32_t base = get_arg_val<uint32_t>(SKV4_ARG_RT_BASE);
#endif
#else
    const uint32_t first = get_arg_val<uint32_t>(SKV4_ARG_RT_FIRST);
    const uint32_t base = get_arg_val<uint32_t>(SKV4_ARG_RT_BASE);
#endif
    const uint32_t cohort = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT);
    const uint32_t head = get_arg_val<uint32_t>(SKV4_ARG_RT_HEAD);
    const uint32_t root_x = get_arg_val<uint32_t>(SKV4_ARG_RT_ROOT_X);
    const uint32_t root_y = get_arg_val<uint32_t>(SKV4_ARG_RT_ROOT_Y);
    const uint32_t parent_x = get_arg_val<uint32_t>(SKV4_ARG_RT_PARENT_X);
    const uint32_t parent_y = get_arg_val<uint32_t>(SKV4_ARG_RT_PARENT_Y);
    const uint32_t parent_level = get_arg_val<uint32_t>(SKV4_ARG_RT_PARENT_LEVEL);
    const uint32_t scratch = get_write_ptr(SKV4_CB_SELECTOR_SCRATCH);
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    // Read only the count header first. Writer independently derives the
    // same chunk partition as NCRISC, so no RISC-to-RISC control message.
    noc_async_read(get_noc_addr(0, tags), scratch, 64);
    noc_async_read_barrier();
    auto coarse_header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t LIVE = coarse_header[0];
    const uint32_t TOTAL_CHUNKS = (LIVE + 31) / 32;
    const uint32_t WORKER = get_arg_val<uint32_t>(SKV4_ARG_RT_WORKER);
    const uint32_t WORKERS = get_arg_val<uint32_t>(SKV4_ARG_RT_WORKERS);
    const uint32_t FIRST_CHUNK = TOTAL_CHUNKS * WORKER / WORKERS;
    const uint32_t END_CHUNK = TOTAL_CHUNKS * (WORKER + 1) / WORKERS;
    const uint32_t first = FIRST_CHUNK;
    const uint32_t N = (END_CHUNK - FIRST_CHUNK) * 32;
    const uint32_t TOTAL = TOTAL_CHUNKS * 32;
    const uint32_t base = TOTAL_CHUNKS * (cohort * 4) / WORKERS * 32;
#endif
#endif
    auto masks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t rows_addr = scratch + MAX_LOCAL * 4;
    auto row_pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_addr);
    for (uint32_t copied = 0; copied < N * 4;) {
#if SKV4_CODEGEN_EXTERNAL_TAGS
        const uint32_t offset = (get_arg_val<uint32_t>(SKV4_ARG_RT_TAG_HEADER) + first * 32) * 4 + copied;
#else
        const uint32_t offset = first * 32 * 4 + copied;
#endif
        const uint32_t page_left = 4096 - offset % 4096;
        const uint32_t bytes = N * 4 - copied < page_left ? N * 4 - copied : page_left;
        noc_async_read(get_noc_addr(offset / 4096, tags) + offset % 4096,
            scratch + copied, bytes);
        copied += bytes;
    }
    noc_async_read_barrier();
#if SKV4_PROFILE_PACKED_CLOCKS
    { DeviceZoneScopedN("FINE_SCORE_ROUTE");
#endif
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
    const uint32_t upper_cache = get_write_ptr(SKV4_CB_UPPER_CACHE);
    uint32_t cached_upper_block = 0xffffffffu;
    auto service_upper = [&]() {
        const auto job = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_UPPER_JOB));
        const uint32_t target = job[0];
        const uint32_t input_chunk = job[1];
        const uint32_t count = job[2];
        for (uint32_t row = 0; row < count; ++row) {
            const uint32_t key = masks[input_chunk * 32 + row] & 0x0fffffffu;
            const uint32_t block = key >> 5;
            if (block != cached_upper_block) {
                noc_async_read(get_noc_addr(block >> 1, upper) + (block & 1u) * 2048,
                               upper_cache, 2048);
                noc_async_read_barrier();
                cached_upper_block = block;
            }
            const auto src = reinterpret_cast<const tt_l1_ptr uint32_t*>(upper_cache);
            auto out = reinterpret_cast<tt_l1_ptr uint32_t*>(
                target + (row / 16) * 2048 + (row % 16) * 64 + 1024);
            const uint32_t k = key & 31u;
            const uint32_t base = (k / 16) * 64 + (k % 16) * 2;
            #pragma GCC unroll 4
            for (uint32_t group = 0; group < 4; ++group) {
                out[group * 4] = src[base + group * 128];
                out[group * 4 + 1] = src[base + group * 128 + 1];
                out[group * 4 + 2] = src[base + group * 128 + 32];
                out[group * 4 + 3] = src[base + group * 128 + 33];
            }
        }
        __sync_synchronize();
        cb_pop_front(SKV4_CB_UPPER_JOB, 1);
        cb_reserve_back(SKV4_CB_UPPER_DONE, 1);
        cb_push_back(SKV4_CB_UPPER_DONE, 1);
    };
#endif
#endif
#else
    constexpr uint32_t N = get_compile_time_arg_val(SKV4_ARG_CT_N), SCORE = get_compile_time_arg_val(SKV4_ARG_CT_SCORE);
    constexpr auto oa = TensorAccessorArgs<SKV4_ARG_CT_ACCESSORS>();
    constexpr uint32_t CB = SCORE ? 16 : 1, TILES = SCORE ? 1 : 8;
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(SKV4_ARG_RT_OUTPUT), 2048);
#endif
    for (uint32_t chunk = 0; chunk < N / 32; ++chunk) {
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
            uint32_t tp_FN_SERVICE_WAIT=skv_tick();
#else
            uint32_t tp_FN_SCORE_WAIT=skv_tick();
#endif
#else
            uint32_t tp_FN_SCORE_WAIT=skv_tick();
#endif
#else
        uint32_t tp_FP_W_SCORE_WAIT=skv_tick();
#endif
#endif
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
        while (!cb_pages_available_at_front(SKV4_CB_SCORE_TILES, 1)) {
            if (cb_pages_available_at_front(SKV4_CB_UPPER_JOB, 1)) service_upper();
        }
#else
        cb_wait_front(SKV4_CB_SCORE_TILES, 1);
#endif
#else
        cb_wait_front(SKV4_CB_SCORE_TILES, 1);
#endif
#else
        cb_wait_front(CB, TILES);
#endif
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_CODEGEN_SELECT
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
            skv_sum_FN_SERVICE_WAIT+=skv_tick()-tp_FN_SERVICE_WAIT;
#else
            skv_sum_FN_SCORE_WAIT+=skv_tick()-tp_FN_SCORE_WAIT;
#endif
#else
            skv_sum_FN_SCORE_WAIT+=skv_tick()-tp_FN_SCORE_WAIT;
#endif
            uint32_t tp_FN_SCORE_STORE=skv_tick();
#else
        skv_sum_FP_W_SCORE_WAIT+=skv_tick()-tp_FP_W_SCORE_WAIT;
        uint32_t tp_FP_W_SCORE_STORE=skv_tick();
#endif
#endif
#if SKV4_CODEGEN_SELECT
        const uint32_t address = get_read_ptr(SKV4_CB_SCORE_TILES);
        const auto tile_pairs = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(address);
        // Keep the diagnostic device-score tensor unchanged (unmasked).
        noc_async_write(address, get_noc_addr(first + chunk, scores), 2048);
        for (uint32_t j = 0; j < 32; j += 2) {
            const uint32_t pos = chunk * 32 + j;
            // Read each tag once for the whole GQA group. LIVE clipping also
            // masks a single valid low half in an odd final pair.
            const uint32_t eligible0 = first * 32 + pos < LIVE ? masks[pos] >> 28 : 0;
            const uint32_t eligible1 = first * 32 + pos + 1 < LIVE ? masks[pos + 1] >> 28 : 0;
            for (uint32_t h = 0; h < 4; ++h) {
                const uint32_t tile_pos = h * 16 + (j & 15u) + (j / 16) * 256;
                const uint32_t keep = ((0u - ((eligible0 >> h) & 1u)) & 0xffffu)
                    | ((0u - ((eligible1 >> h) & 1u)) << 16);
                row_pairs[(h * N + pos) >> 1] =
                    (tile_pairs[tile_pos >> 1] & keep) | (0xff80ff80u & ~keep);
            }
#else
        for (uint32_t t = 0; t < TILES; ++t) {
            noc_async_write(get_read_ptr(CB) + t * 2048,
                get_noc_addr((get_arg_val<uint32_t>(SKV4_ARG_RT_FIRST) + chunk) * TILES + t, output), 2048);
#endif
        }
        noc_async_write_barrier();
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_CODEGEN_SELECT
            skv_sum_FN_SCORE_STORE+=skv_tick()-tp_FN_SCORE_STORE;
#else
        skv_sum_FP_W_SCORE_STORE+=skv_tick()-tp_FP_W_SCORE_STORE;
#endif
#endif
#if SKV4_CODEGEN_SELECT
        cb_pop_front(SKV4_CB_SCORE_TILES, 1);
    }
    for (uint32_t h = 0; h < 4; ++h) {
        const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS + h * 2);
        const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS_Y + h * 2);
        // K512 consumes one half-filled physical tile per 512 scores. Split
        // only at those page boundaries; each span remains one bulk NoC.
        uint32_t src = rows_addr + h * N * 2;
        uint32_t offset = first * 32 - base;
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
        }
    }
    noc_async_write_barrier();
    for (uint32_t h = 0; h < 4; ++h) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS + h * 2),
            get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS_Y + h * 2), get_semaphore(SKV4_SEM_COHORT_SCORES_READY)), 1);
#else
        cb_pop_front(CB, TILES);
#endif
    }
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_CODEGEN_SELECT
    }
    { DeviceZoneScopedN("FINE_FRONTIER_WAIT_SEND");
#else
    DeviceTimestampedData("SKVA_FP_W_SCORE_WAIT",skv_sum_FP_W_SCORE_WAIT);
    DeviceTimestampedData("SKVA_FP_W_SCORE_STORE",skv_sum_FP_W_SCORE_STORE);
    DeviceTimestampedData("SKV_FP_WRITER_E", 0);
#endif
#endif
#if SKV4_CODEGEN_SELECT
    cb_wait_front(SKV4_CB_FRONTIER_VALUES, 1);
    if (cohort) {
        // One bulk packet carries the native fused K512 frontier. The low
        // half already contains the global compact Fine position.
        noc_async_write(get_read_ptr(SKV4_CB_FRONTIER_VALUES), get_noc_addr(parent_x, parent_y,
            get_write_ptr(SKV4_CB_PEER_FRONTIERS) + parent_level * 2048), 2048);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(parent_x, parent_y,
            get_semaphore(SKV4_SEM_FRONTIER_LEVEL_0 + parent_level)), 1);
    } else {
        auto native = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_FRONTIER_VALUES));
        auto positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t rank = 0; rank < P; ++rank) {
            // K512's native rank matrix is 32x16 (K1024 was 64x16).
            const uint32_t lane = (rank >> 5) + ((rank & 31u) << 4);
            positions[rank] = native[lane] & 0xffffu;
#endif
        }
#if SKV4_CODEGEN_SELECT
    }
    cb_pop_front(SKV4_CB_FRONTIER_VALUES, 1);
    // Cohort helpers have already sent their parent notification.  Flush that
    // non-posted atomic before the early kernel exit can expose this core's L1
    // semaphore region to the next program.
    noc_async_atomic_barrier();
    if (cohort)
        for (uint32_t sem = SKV4_SEM_COHORT_SCORES_READY; sem <= SKV4_SEM_UNION_OFFSETS_READY; ++sem)
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem)), 0);
#if SKV4_PROFILE_PACKED_CLOCKS
    }
    if (cohort) {
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
        DeviceTimestampedData("SKVA_FN_SERVICE_WAIT", skv_sum_FN_SERVICE_WAIT);
#else
        DeviceTimestampedData("SKVA_FN_SCORE_WAIT", skv_sum_FN_SCORE_WAIT);
#endif
#else
        DeviceTimestampedData("SKVA_FN_SCORE_WAIT", skv_sum_FN_SCORE_WAIT);
#endif
        DeviceTimestampedData("SKVA_FN_SCORE_STORE", skv_sum_FN_SCORE_STORE);
        return; }
    DeviceZoneScopedN("FINE_DISTRIBUTED_UNION");
#else
    if (cohort) return;
#endif
    // The four cohort-zero roots now own one head and one contiguous source
    // range each.  Keep the diagnostic tensor in coarse-position form; the
    // untimed host checker maps it to cluster IDs.
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_BITMAP_B", 0);
#endif
    /*@FN_DU_BITMAP_B*/
    auto positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    noc_async_write(scratch, get_noc_addr(head, heads), P * 4);
    const uint32_t union_scratch = get_write_ptr(SKV4_CB_UNION_SCRATCH);
#if SKV4_CODEGEN_EXTERNAL_TAGS
#if SKV4_CODEGEN_DYNAMIC
    constexpr uint32_t MAX_BITMAP_PAGES = (MAX_TOTAL + 511) / 512;
    const uint32_t BITMAP_PAGES = (TOTAL + 511) / 512;
    constexpr uint32_t BITMAP_BYTES = MAX_BITMAP_PAGES * 64;
    constexpr uint32_t MAX_OWNER_PAGES = (MAX_BITMAP_PAGES + 3) / 4;
#else
    constexpr uint32_t BITMAP_PAGES = (TOTAL + 511) / 512;
    constexpr uint32_t BITMAP_BYTES = BITMAP_PAGES * 64;
    constexpr uint32_t MAX_OWNER_PAGES = (BITMAP_PAGES + 3) / 4;
#endif
#else
    constexpr uint32_t BITMAP_PAGES = (TOTAL + 511) / 512;
    constexpr uint32_t BITMAP_BYTES = BITMAP_PAGES * 64;
    constexpr uint32_t MAX_OWNER_PAGES = (BITMAP_PAGES + 3) / 4;
#endif
    constexpr uint32_t MAX_SLICE_BYTES = MAX_OWNER_PAGES * 64;
    constexpr uint32_t MAX_OWNER_KEYS = MAX_OWNER_PAGES * 512;
    constexpr uint32_t REMOTE_OFF = BITMAP_BYTES;
    constexpr uint32_t LOOKUP_OFF = REMOTE_OFF + 4 * MAX_SLICE_BYTES;
    constexpr uint32_t LIST_OFF = (LOOKUP_OFF + MAX_OWNER_KEYS * 4 + 63) & ~63;
    constexpr uint32_t CONTROL_OFF =
        (LIST_OFF + (MAX_OWNER_KEYS + 16) * 4 + 63) & ~63;
    constexpr uint32_t OFFSETS_OFF = CONTROL_OFF + 4 * 64;
    constexpr uint32_t HEADER_OFF = OFFSETS_OFF + 64;
    auto own_bitmap = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(union_scratch);
    for (uint32_t w = 0; w < BITMAP_BYTES / 4; ++w) own_bitmap[w] = 0;
    uint32_t local_status = 0;
    for (uint32_t rank = 0; rank < P; ++rank) {
        const uint32_t i = positions[rank];
        if (i >= LIVE) local_status = 1;
        else own_bitmap[i / 32] |= 1u << (i % 32);
    }
    noc_async_write_barrier();
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_BITMAP_E", 0);
#endif
    /*@FN_DU_BITMAP_E*/
    // One barrier after all four local bitmaps exist.  Root zero releases the
    // other roots; subsequent transfers are one bulk bitmap slice per peer.
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_EXCHANGE_B", 0);
#endif
    /*@FN_DU_EXCHANGE_B*/
    noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_UNION_BITMAPS_READY)), 1);
    if (head == 0) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_UNION_BITMAPS_READY)), 4);
        for (uint32_t h = 1; h < 4; ++h)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS + h * 2),
                get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS_Y + h * 2), get_semaphore(SKV4_SEM_UNION_BITMAPS_READY)), 1);
    } else {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_UNION_BITMAPS_READY)), 1);
    }
    const uint32_t page_first = BITMAP_PAGES * head / 4;
    const uint32_t page_end = BITMAP_PAGES * (head + 1) / 4;
    const uint32_t owner_pages = page_end - page_first;
    const uint32_t first_pos = page_first * 512;
    const uint32_t owner_keys =
        TOTAL - first_pos < owner_pages * 512 ? TOTAL - first_pos : owner_pages * 512;
    if (owner_pages) {
        for (uint32_t h = 0; h < 4; ++h) {
            if (h == head) continue;
            const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS + h * 2);
            const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS_Y + h * 2);
            noc_async_read(get_noc_addr(x, y, union_scratch + page_first * 64),
                union_scratch + REMOTE_OFF + h * MAX_SLICE_BYTES,
                owner_pages * 64);
        }
    }
    // Load only this owner's contiguous tag slice, rather than all tags on one
    // root.  Slice starts are 2KiB aligned and retain narrow-transfer alignment.
    for (uint32_t copied = 0; copied < owner_keys * 4;) {
#if SKV4_CODEGEN_EXTERNAL_TAGS
        const uint32_t logical = (get_arg_val<uint32_t>(SKV4_ARG_RT_TAG_HEADER) + first_pos) * 4 + copied;
#else
        const uint32_t logical = first_pos * 4 + copied;
#endif
        const uint32_t page_left = 4096 - logical % 4096;
        const uint32_t bytes = owner_keys * 4 - copied < page_left
            ? owner_keys * 4 - copied : page_left;
        noc_async_read(get_noc_addr(logical / 4096, tags) + logical % 4096,
            union_scratch + LOOKUP_OFF + copied, bytes);
        copied += bytes;
    }
    noc_async_read_barrier();
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_EXCHANGE_E", 0);
#endif
    /*@FN_DU_EXCHANGE_E*/
    auto lookup = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(union_scratch + LOOKUP_OFF);
    // The self slot in REMOTE_OFF is never populated by peer DMA. Cache the
    // validated accepted union words there so enumeration does not rescan all
    // four head bitmaps and masks.
    auto accepted_cache = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        union_scratch + REMOTE_OFF + head * MAX_SLICE_BYTES);
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_COUNT_B", 0);
#endif
    /*@FN_DU_COUNT_B*/
    const auto bitmap0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        head == 0 ? union_scratch + page_first * 64
                  : union_scratch + REMOTE_OFF);
    const auto bitmap1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        head == 1 ? union_scratch + page_first * 64
                  : union_scratch + REMOTE_OFF + MAX_SLICE_BYTES);
    const auto bitmap2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        head == 2 ? union_scratch + page_first * 64
                  : union_scratch + REMOTE_OFF + 2 * MAX_SLICE_BYTES);
    const auto bitmap3 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        head == 3 ? union_scratch + page_first * 64
                  : union_scratch + REMOTE_OFF + 3 * MAX_SLICE_BYTES);
    uint32_t count = 0;
    for (uint32_t w = 0; w < owner_pages * 16; ++w) {
        const uint32_t word0 = bitmap0[w];
        const uint32_t word1 = bitmap1[w];
        const uint32_t word2 = bitmap2[w];
        const uint32_t word3 = bitmap3[w];
        uint32_t remaining = word0 | word1 | word2 | word3;
        uint32_t accepted = 0;
        const uint32_t global = first_pos + w * 32;
        // A shared position loads its tag once, but retain both outcomes:
        // any eligible selecting head accepts it, while any forbidden
        // selecting head still sets the same error status as before.
        while (remaining) {
            const uint32_t b = __builtin_ctz(remaining);
            if (global + b >= LIVE) {
                local_status = 1;
            } else {
                const uint32_t local = w * 32 + b;
                const uint32_t eligible_heads = lookup[local] >> 28;
                const uint32_t selected_heads =
                    ((word0 >> b) & 1u) |
                    (((word1 >> b) & 1u) << 1) |
                    (((word2 >> b) & 1u) << 2) |
                    (((word3 >> b) & 1u) << 3);
                if (selected_heads & ~eligible_heads & 15u)
                    local_status = 1;
                if (selected_heads & eligible_heads)
                    accepted |= 1u << b;
            }
            remaining &= remaining - 1;
        }
        accepted_cache[w] = accepted;
        count += __builtin_popcount(accepted);
    }
    auto control = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        union_scratch + CONTROL_OFF + head * 64);
    control[0] = count;
    control[1] = local_status;
    if (head) {
        noc_async_write(reinterpret_cast<uint32_t>(control),
            get_noc_addr(root_x, root_y, union_scratch + CONTROL_OFF + head * 64), 64);
        noc_async_write_barrier();
    }
    noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_UNION_COUNTS_READY)), 1);
    auto offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        union_scratch + OFFSETS_OFF);
    if (head == 0) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_UNION_COUNTS_READY)), 4);
        uint32_t total = 0;
        uint32_t status = 0;
        for (uint32_t h = 0; h < 4; ++h) {
            auto c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                union_scratch + CONTROL_OFF + h * 64);
            offsets[h] = total;
            total += c[0];
            status |= c[1];
        }
        offsets[4] = total;
        offsets[5] = status;
        for (uint32_t h = 1; h < 4; ++h) {
            const uint32_t x = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS + h * 2);
            const uint32_t y = get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS_Y + h * 2);
            noc_async_write(reinterpret_cast<uint32_t>(offsets),
                get_noc_addr(x, y, union_scratch + OFFSETS_OFF), 64);
        }
        noc_async_write_barrier();
        for (uint32_t h = 1; h < 4; ++h)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS + h * 2),
                get_arg_val<uint32_t>(SKV4_ARG_RT_COHORT_PEERS_Y + h * 2), get_semaphore(SKV4_SEM_UNION_OFFSETS_READY)), 1);
        auto header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            union_scratch + HEADER_OFF);
        for (uint32_t j = 0; j < 16; ++j) header[j] = 0xffffffff;
        header[0] = total;
        header[1] = status;
        noc_async_write(reinterpret_cast<uint32_t>(header), get_noc_addr(0, output), 64);
    } else {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_UNION_OFFSETS_READY)), 1);
    }
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_COUNT_E", 0);
#endif
    /*@FN_DU_COUNT_E*/
    // Re-enumerate only this owner's range.  Prefix-padding the local list
    // keeps the source/destination low address bits equal for narrow writes.
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_ENUM_B", 0);
#endif
    /*@FN_DU_ENUM_B*/
    const uint32_t prefix = offsets[head];
    auto list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        union_scratch + LIST_OFF) + (prefix & 15u);
    uint32_t emitted = 0;
    for (uint32_t w = 0; w < owner_pages * 16; ++w) {
        uint32_t accepted = accepted_cache[w];
        while (accepted) {
            const uint32_t b = __builtin_ctz(accepted);
            list[emitted++] = lookup[w * 32 + b] & 0x0fffffff;
            accepted &= accepted - 1;
        }
    }
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_ENUM_E", 0);
#endif
    /*@FN_DU_ENUM_E*/
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_EXPORT_B", 0);
#endif
    /*@FN_DU_EXPORT_B*/
    uint32_t copied = 0;
    const uint32_t logical_start = (16 + prefix) * 4;
    const uint32_t list_addr = reinterpret_cast<uint32_t>(list);
    while (copied < emitted * 4) {
        const uint32_t logical = logical_start + copied;
        const uint32_t bytes = emitted * 4 - copied < 64 - logical % 64
            ? emitted * 4 - copied : 64 - logical % 64;
        noc_async_write(list_addr + copied,
            get_noc_addr(logical / 64, output) + logical % 64, bytes);
        copied += bytes;
    }
    noc_async_write_barrier();
    // The distributed-union path issues several remote semaphore trains.
    // Device kernel completion does not implicitly flush non-posted atomics.
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_REPLAY_CLEANUP_B", 0);
#else
#if SKV4_PROFILE_CLEANUP_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_REPLAY_CLEANUP_B", 0);
#endif
#endif
    /*@FN_DU_REPLAY_CLEANUP_B*/
    noc_async_atomic_barrier();
    for (uint32_t sem = SKV4_SEM_COHORT_SCORES_READY; sem <= SKV4_SEM_UNION_OFFSETS_READY; ++sem)
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem)), 0);
    /*@FN_DU_REPLAY_CLEANUP_E*/
#if SKV4_PROFILE_PACKED_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_REPLAY_CLEANUP_E", 0);
    DeviceTimestampedData("SKV_FN_DU_EXPORT_E", 0);
#else
#if SKV4_PROFILE_CLEANUP_CLOCKS
    DeviceTimestampedData("SKV_FN_DU_REPLAY_CLEANUP_E", 0);
#endif
#endif
    /*@FN_DU_EXPORT_E*/
#if SKV4_PROFILE_PACKED_CLOCKS
#if SKV4_CODEGEN_SHARED_UPPER
#if SKV4_CODEGEN_DUAL_RISC
    DeviceTimestampedData("SKVA_FN_SERVICE_WAIT",skv_sum_FN_SERVICE_WAIT);
#else
    DeviceTimestampedData("SKVA_FN_SCORE_WAIT",skv_sum_FN_SCORE_WAIT);
#endif
#else
    DeviceTimestampedData("SKVA_FN_SCORE_WAIT",skv_sum_FN_SCORE_WAIT);
#endif
    DeviceTimestampedData("SKVA_FN_SCORE_STORE",skv_sum_FN_SCORE_STORE);
#endif
}
#endif
"""
        )
    )
