"""K512 threshold tree and source-owned survivor masks for Stage-I coarse.

Workers count source-ordered survivors, root0 computes one prefix sum, and the
workers bulk-write disjoint spans of one compact T1 tag list for Fine.
"""

from __future__ import annotations

from seoulkv4._kernels.stage1_coarse.abi import CoarseSemaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

from seoulkv4._kernels.stage1_coarse.abi import CoarseCB as CB

import math

import torch

from seoulkv4._kernels.common.core_ranges import core_ranges as _ranges


ORDERED_PAIR_HELPER = r"""
FORCE_INLINE uint32_t coarse_ordered_bf16_pair(uint32_t pair) {
    // Two independent halfword XOR masks, with unsigned wraparound.
    // Preserve the scalar raw-bit order, including signed zero/NaN/inf.
    const uint32_t signs = (pair >> 15) & 0x00010001u;
    return pair ^ (0x80008000u | ((signs << 16) - signs));
}
"""


READER_MASK_HELPER = (
    ORDERED_PAIR_HELPER
    + r"""
FORCE_INLINE uint16_t coarse_ordered_bf16(uint16_t bits) {
    return bits ^ (0x8000u | (0u - (bits >> 15)));
}
"""
)


FUSED_READER_TAIL = (
    semaphore_defines(SEM)
    + r"""
    /*@CO_R_FRONTIER_B*/
    const uint32_t width = get_arg_val<uint32_t>(6);
    const uint32_t cohort = get_arg_val<uint32_t>(7);
    const uint32_t cohorts = get_arg_val<uint32_t>(8);
    const uint32_t chunks = (width + 511) / 512;
    cb_reserve_back(19, chunks);
    auto row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(19));
    const uint32_t tail = width - (chunks - 1) * 512;
    for (uint32_t j = tail; j < 512; ++j)
        row[(chunks - 1) * 1024 + j] = 0xff80;
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_COHORT_SCORES_READY)), 4);
    cb_push_back(19, chunks);
    for (uint32_t level = 0, stride = 1; stride < cohorts; ++level, stride *= 2) {
        if (cohort % (2 * stride)) break;
        if (cohort + stride >= cohorts) continue;
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_FRONTIER_LEVEL_0 + level)), 1);
        cb_reserve_back(24, 1);
        const uint32_t src = get_write_ptr(23) + level * 2048;
        noc_async_read(get_noc_addr(src), get_write_ptr(24), 2048);
        noc_async_read_barrier();
        cb_push_back(24, 1);
    }
    /*@CO_R_FRONTIER_E*/
"""
)


READER_MASK_TAIL = (
    semaphore_defines(SEM)
    + r"""
    /*@CO_R_MASK_B*/
    constexpr uint32_t LIVE = get_compile_time_arg_val(1);
    constexpr uint32_t N = CHUNKS * 32;
    const uint32_t threshold_control = get_write_ptr(31);
    auto threshold = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(threshold_control);
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_THRESHOLDS_READY)), 1);

    // NCRISC handles the upper half while BRISC handles the lower half.  The
    // two ordered tag lists occupy fixed, non-overlapping regions in CB22;
    // BRISC joins the local count before publishing the existing one-packet
    // worker count to root0.
    const uint32_t scratch = get_write_ptr(22);
    auto out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    auto upper_pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(30));
    const auto upper0 = upper_pairs;
    const auto upper1 = upper_pairs + N / 2;
    const auto upper2 = upper_pairs + N;
    const auto upper3 = upper_pairs + 3 * N / 2;
    const uint16_t t0 = coarse_ordered_bf16(threshold[0]);
    const uint16_t t1 = coarse_ordered_bf16(threshold[8]);
    const uint16_t t2 = coarse_ordered_bf16(threshold[16]);
    const uint16_t t3 = coarse_ordered_bf16(threshold[24]);
    const uint32_t first = first_chunk * 32;
    const uint32_t limit = first >= LIVE ? 0 :
        (first + N < LIVE ? N : LIVE - first);
    const uint32_t split = N / 2;
    const uint32_t rear_base = 16 + split + 3;
    uint32_t count = 0;
    uint32_t pos = split;
    for (; pos + 1 < limit; pos += 2) {
        const uint32_t pair = pos >> 1;
        const uint32_t v0 = coarse_ordered_bf16_pair(upper0[pair]);
        const uint32_t v1 = coarse_ordered_bf16_pair(upper1[pair]);
        const uint32_t v2 = coarse_ordered_bf16_pair(upper2[pair]);
        const uint32_t v3 = coarse_ordered_bf16_pair(upper3[pair]);
        const uint32_t mask0 =
            (static_cast<uint32_t>((v0 & 0xffffu) >= t0) << 0) |
            (static_cast<uint32_t>((v1 & 0xffffu) >= t1) << 1) |
            (static_cast<uint32_t>((v2 & 0xffffu) >= t2) << 2) |
            (static_cast<uint32_t>((v3 & 0xffffu) >= t3) << 3);
        const uint32_t mask1 =
            (static_cast<uint32_t>((v0 >> 16) >= t0) << 0) |
            (static_cast<uint32_t>((v1 >> 16) >= t1) << 1) |
            (static_cast<uint32_t>((v2 >> 16) >= t2) << 2) |
            (static_cast<uint32_t>((v3 >> 16) >= t3) << 3);
        if (mask0) out[rear_base + count++] = (first + pos) | (mask0 << 28);
        if (mask1) out[rear_base + count++] = (first + pos + 1) | (mask1 << 28);
    }
    if (pos < limit) {
        const uint32_t pair = pos >> 1;
        const uint32_t mask =
            (static_cast<uint32_t>(coarse_ordered_bf16(upper0[pair] & 0xffffu) >= t0) << 0) |
            (static_cast<uint32_t>(coarse_ordered_bf16(upper1[pair] & 0xffffu) >= t1) << 1) |
            (static_cast<uint32_t>(coarse_ordered_bf16(upper2[pair] & 0xffffu) >= t2) << 2) |
            (static_cast<uint32_t>(coarse_ordered_bf16(upper3[pair] & 0xffffu) >= t3) << 3);
        if (mask) out[rear_base + count++] = (first + pos) | (mask << 28);
    }
    out[2] = count;
    __sync_synchronize();
    noc_semaphore_inc(get_noc_addr(get_semaphore(SKV4_SEM_MASK_REAR_READY)), 1);
    noc_async_atomic_barrier();
    /*@CO_R_MASK_E*/
"""
)


FUSED_COMPUTE_TAIL = r"""
    // Coarse keeps [BF16 score | u16 tie-break] fused through the complete
    // K512 tree.  Unlike Fine, no production consumer needs global positions.
    const uint32_t chunks = get_arg_val<uint32_t>(0);
    const uint32_t tail = get_arg_val<uint32_t>(1);
    const uint32_t cohort = get_arg_val<uint32_t>(3);
    const uint32_t cohorts = get_arg_val<uint32_t>(4);
    pack_reconfig_data_format(20);
    tile_regs_acquire();
    fine_sort_chunk(0, chunks == 1 ? tail : 512, false);
    for (uint32_t chunk = 1; chunk < chunks; ++chunk) {
        fine_sort_chunk(1, chunk + 1 == chunks ? tail : 512, true);
        topk_xl_init<1024, true>();
        topk_xl_merge<1024, true>(0);
        topk_xl_rebuild<1024, true>(0, false);
    }
    for (uint32_t stride = 1; stride < cohorts; stride *= 2) {
        if (cohort % (2 * stride)) break;
        if (cohort + stride >= cohorts) continue;
        cb_wait_front(24, 1);
        topk_xl_copy_tile_init(24);
        topk_xl_copy_tile<1024>(24, 1, 0, 512);
        cb_pop_front(24, 1);
        topk_xl_init<1024, true>();
        topk_xl_rebuild<1024, true>(1, true);
        topk_xl_merge<1024, true>(0);
        topk_xl_rebuild<1024, true>(0, false);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(20, 1);
    pack_reconfig_data_format(20);
    pack_tile(0, 20);
    cb_push_back(20, 1);
    tile_regs_release();
"""


def reader_source(*, debug_heads=False):
    """Generate the selector directly from its explicit debug contract."""
    from seoulkv4._kernels.codegen.coarse import reader

    return reader(select=True, debug_heads=debug_heads, profile=False)


def compute_source(*, debug_heads=False):
    """Generate the selector directly from its explicit debug contract."""
    from seoulkv4._kernels.codegen.coarse import compute

    return compute(select=True, debug_heads=debug_heads, profile=False)


WRITER = (
    semaphore_defines(SEM)
    + r"""
#include "api/dataflow/dataflow_api.h"
"""
    + ORDERED_PAIR_HELPER
    + r"""

FORCE_INLINE uint16_t ordered_bf16(uint16_t bits) {
    return bits ^ (0x8000u | (0u - (bits >> 15)));
}

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t TOTAL = get_compile_time_arg_val(1);
    constexpr uint32_t LIVE = get_compile_time_arg_val(2);
    constexpr uint32_t P = get_compile_time_arg_val(3);
    constexpr auto oa = TensorAccessorArgs<4>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), 4096);
    const uint32_t first_chunk = get_arg_val<uint32_t>(2);
    const uint32_t base = get_arg_val<uint32_t>(3);
    const uint32_t cohort = get_arg_val<uint32_t>(4);
    const uint32_t head = get_arg_val<uint32_t>(5);
    const uint32_t worker = get_arg_val<uint32_t>(6);
    const uint32_t workers = get_arg_val<uint32_t>(7);
    const uint32_t root_x = get_arg_val<uint32_t>(8);
    const uint32_t root_y = get_arg_val<uint32_t>(9);
    const uint32_t parent_x = get_arg_val<uint32_t>(10);
    const uint32_t parent_y = get_arg_val<uint32_t>(11);
    const uint32_t parent_level = get_arg_val<uint32_t>(12);
    const uint32_t lower_addr = get_write_ptr(29);
    const uint32_t upper_addr = get_write_ptr(30);
    auto upper = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(upper_addr);
    auto lower_pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lower_addr);
    auto upper_pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(upper_addr);

    // Preserve both interval rows source-locally. Lower rows cross NoC into
    // the TopK tree; upper rows remain untouched until threshold comparison.
    for (uint32_t chunk = 0; chunk < N / 32; ++chunk) {
        cb_wait_front(16, 1);
        cb_wait_front(17, 1);
        auto lo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(16));
        auto hi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(17));
        for (uint32_t h = 0; h < 4; ++h)
            for (uint32_t j = 0; j < 32; j += 2) {
                const uint32_t pos = chunk * 32 + j;
                const uint32_t lane = h * 16 + (j & 15u) + (j / 16) * 256;
                const uint32_t dst = (h * N + pos) >> 1;
                const uint32_t global = first_chunk * 32 + pos;
                uint32_t lo_pair = lo[lane >> 1];
                uint32_t hi_pair = hi[lane >> 1];
                if (global >= LIVE) {
                    lo_pair = 0xff80ff80u;
                    hi_pair = 0xff80ff80u;
                } else if (global + 1 >= LIVE) {
                    lo_pair = (lo_pair & 0x0000ffffu) | 0xff800000u;
                    hi_pair = (hi_pair & 0x0000ffffu) | 0xff800000u;
                }
                lower_pairs[dst] = lo_pair;
                upper_pairs[dst] = hi_pair;
            }
        cb_pop_front(16, 1);
        cb_pop_front(17, 1);
    }

    // Four large spans per worker feed the four head owners in this cohort.
    for (uint32_t h = 0; h < 4; ++h) {
        const uint32_t x = get_arg_val<uint32_t>(13 + h * 2);
        const uint32_t y = get_arg_val<uint32_t>(14 + h * 2);
        uint32_t src = lower_addr + h * N * 2;
        uint32_t offset = first_chunk * 32 - base;
        uint32_t remaining = N;
        while (remaining) {
            const uint32_t in_page = offset & 511u;
            const uint32_t take = remaining < 512 - in_page
                ? remaining : 512 - in_page;
            const uint32_t dst = get_write_ptr(19)
                + (offset / 512) * 2048 + in_page * 2;
            noc_async_write(src, get_noc_addr(x, y, dst), take * 2);
            src += take * 2;
            offset += take;
            remaining -= take;
        }
    }
    noc_async_write_barrier();
    for (uint32_t h = 0; h < 4; ++h)
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(13 + h * 2),
            get_arg_val<uint32_t>(14 + h * 2), get_semaphore(SKV4_SEM_COHORT_SCORES_READY)), 1);

    /*@COARSE_FRONTIER_B*/
    cb_wait_front(20, 1);
    const uint32_t scratch = get_write_ptr(22);
    if (cohort) {
        // One bulk packet carries the native fused K512 frontier.  No local
        // self-copy and no row-major UInt32 companion payload are needed.
        noc_async_write(get_read_ptr(20), get_noc_addr(parent_x, parent_y,
            get_write_ptr(23) + parent_level * 2048), 2048);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(parent_x, parent_y,
            get_semaphore(SKV4_SEM_FRONTIER_LEVEL_0 + parent_level)), 1);
    } else {
        // K512's fused result retains the native 32x16 rank matrix.  The high
        // half is the BF16 score; the low half is only an internal tie-break.
        const uint32_t rank = P - 1;
        const uint32_t lane = (rank >> 5) + ((rank & 31u) << 4);
        auto fused = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_read_ptr(20));
        const uint32_t control = get_write_ptr(31);
        auto local = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(control);
        local[head * 8] = fused[lane] >> 16;
        noc_async_write(control + head * 16,
            get_noc_addr(root_x, root_y, control + head * 16), 16);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_THRESHOLD_HEADS_READY)), 1);
    }
    cb_pop_front(20, 1);
    /*@COARSE_FRONTIER_E*/

    const uint32_t ids_control = get_write_ptr(28);
    const uint32_t threshold_control = get_write_ptr(31);
    auto threshold = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(threshold_control);
    if (worker == 0) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_THRESHOLD_HEADS_READY)), 4);
        for (uint32_t w = 0; w < workers; ++w) {
            const uint32_t x = get_arg_val<uint32_t>(21 + w * 2);
            const uint32_t y = get_arg_val<uint32_t>(22 + w * 2);
            noc_async_write(threshold_control,
                get_noc_addr(x, y, threshold_control), 64);
        }
        noc_async_write_barrier();
        for (uint32_t w = 0; w < workers; ++w)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(21 + w * 2),
                get_arg_val<uint32_t>(22 + w * 2), get_semaphore(SKV4_SEM_THRESHOLDS_READY)), 1);
    }
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_THRESHOLDS_READY)), 1);

    // Tags are ascending within each contiguous source range and carry the
    // four-head eligibility mask in their high nibble. BRISC owns the lower
    // half; NCRISC concurrently owns the upper half in READER_MASK_TAIL.
    auto out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const auto upper0 = upper_pairs;
    const auto upper1 = upper_pairs + N / 2;
    const auto upper2 = upper_pairs + N;
    const auto upper3 = upper_pairs + 3 * N / 2;
    const uint16_t t0 = ordered_bf16(threshold[0]);
    const uint16_t t1 = ordered_bf16(threshold[8]);
    const uint16_t t2 = ordered_bf16(threshold[16]);
    const uint16_t t3 = ordered_bf16(threshold[24]);
    const uint32_t first = first_chunk * 32;
    const uint32_t limit = first >= LIVE ? 0 :
        (first + N < LIVE ? N : LIVE - first);
    const uint32_t split = limit < N / 2 ? limit : N / 2;
    uint32_t count = 0;
    uint32_t pos = 0;
    for (; pos + 1 < split; pos += 2) {
        const uint32_t pair = pos >> 1;
        const uint32_t v0 = coarse_ordered_bf16_pair(upper0[pair]);
        const uint32_t v1 = coarse_ordered_bf16_pair(upper1[pair]);
        const uint32_t v2 = coarse_ordered_bf16_pair(upper2[pair]);
        const uint32_t v3 = coarse_ordered_bf16_pair(upper3[pair]);
        const uint32_t mask0 =
            (static_cast<uint32_t>((v0 & 0xffffu) >= t0) << 0) |
            (static_cast<uint32_t>((v1 & 0xffffu) >= t1) << 1) |
            (static_cast<uint32_t>((v2 & 0xffffu) >= t2) << 2) |
            (static_cast<uint32_t>((v3 & 0xffffu) >= t3) << 3);
        const uint32_t mask1 =
            (static_cast<uint32_t>((v0 >> 16) >= t0) << 0) |
            (static_cast<uint32_t>((v1 >> 16) >= t1) << 1) |
            (static_cast<uint32_t>((v2 >> 16) >= t2) << 2) |
            (static_cast<uint32_t>((v3 >> 16) >= t3) << 3);
        if (mask0) out[16 + count++] = (first + pos) | (mask0 << 28);
        if (mask1) out[16 + count++] = (first + pos + 1) | (mask1 << 28);
    }
    if (pos < split) {
        const uint32_t pair = pos >> 1;
        const uint32_t mask =
            (static_cast<uint32_t>(ordered_bf16(upper0[pair] & 0xffffu) >= t0) << 0) |
            (static_cast<uint32_t>(ordered_bf16(upper1[pair] & 0xffffu) >= t1) << 1) |
            (static_cast<uint32_t>(ordered_bf16(upper2[pair] & 0xffffu) >= t2) << 2) |
            (static_cast<uint32_t>(ordered_bf16(upper3[pair] & 0xffffu) >= t3) << 3);
        if (mask) out[16 + count++] = (first + pos) | (mask << 28);
    }
    const uint32_t front_count = count;
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_MASK_REAR_READY)), 1);
    __sync_synchronize();
    const uint32_t rear_count = out[2];
    count = front_count + rear_count;

    // One 64-byte count packet per worker. Root0 computes globally sorted
    // destination offsets because worker ranges are source ordered.
    auto count_slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        ids_control + worker * 64);
    for (uint32_t i = 0; i < 16; ++i) count_slot[i] = 0;
    count_slot[0] = count;
    noc_async_write(reinterpret_cast<uint32_t>(count_slot),
        get_noc_addr(root_x, root_y, ids_control + worker * 64), 64);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_WORKER_COUNTS_READY)), 1);

    if (worker == 0) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_WORKER_COUNTS_READY)),
            workers);
        uint32_t total = 0;
        for (uint32_t w = 0; w < workers; ++w) {
            auto slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                ids_control + w * 64);
            slot[1] = total;
            total += slot[0];
        }
        for (uint32_t w = 0; w < workers; ++w) {
            const uint32_t x = get_arg_val<uint32_t>(21 + w * 2);
            const uint32_t y = get_arg_val<uint32_t>(22 + w * 2);
            noc_async_write(ids_control + w * 64,
                get_noc_addr(x, y, ids_control + w * 64), 64);
        }
        for (uint32_t i = 0; i < 16; ++i) out[i] = 0xffffffffu;
        out[0] = total;
        out[1] = 0;
        noc_async_write(scratch, get_noc_addr(0, output), 64);
        noc_async_write_barrier();
        for (uint32_t w = 0; w < workers; ++w)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(21 + w * 2),
                get_arg_val<uint32_t>(22 + w * 2), get_semaphore(SKV4_SEM_PREFIX_OFFSETS_READY)), 1);
    }
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_PREFIX_OFFSETS_READY)), 1);

    const uint32_t prefix = count_slot[1];
    const uint32_t front_shift = prefix & 3u;
    if (front_shift)
        for (uint32_t i = front_count; i > 0; --i)
            out[16 + front_shift + i - 1] = out[16 + i - 1];
    uint32_t copied = 0;
    uint32_t logical_start = (16 + prefix) * 4;
    uint32_t source = scratch + (16 + front_shift) * 4;
    while (copied < front_count * 4) {
        const uint32_t logical = logical_start + copied;
        const uint32_t page_left = 4096 - logical % 4096;
        const uint32_t bytes = front_count * 4 - copied < page_left
            ? front_count * 4 - copied : page_left;
        noc_async_write(source + copied,
            get_noc_addr(logical / 4096, output) + logical % 4096, bytes);
        copied += bytes;
    }
    const uint32_t rear_base = 16 + N / 2 + 3;
    const uint32_t rear_logical_index = 16 + prefix + front_count;
    const uint32_t rear_shift = (rear_logical_index - rear_base) & 3u;
    if (rear_shift)
        for (uint32_t i = rear_count; i > 0; --i)
            out[rear_base + rear_shift + i - 1] = out[rear_base + i - 1];
    copied = 0;
    logical_start = rear_logical_index * 4;
    source = scratch + (rear_base + rear_shift) * 4;
    while (copied < rear_count * 4) {
        const uint32_t logical = logical_start + copied;
        const uint32_t page_left = 4096 - logical % 4096;
        const uint32_t bytes = rear_count * 4 - copied < page_left
            ? rear_count * 4 - copied : page_left;
        noc_async_write(source + copied,
            get_noc_addr(logical / 4096, output) + logical % 4096, bytes);
        copied += bytes;
    }
    noc_async_write_barrier();
    // Remote semaphore increments are non-posted NoC atomics.  A program may
    // otherwise report completion while they are still in flight and let the
    // following Fine/Stage1.5 program reuse the same L1 semaphore addresses.
    /*@CO_W_REPLAY_CLEANUP_B*/
    noc_async_atomic_barrier();
    // Semaphores are persistent L1 state; ProgramDescriptor initial values
    // are not replay initialization.  sem12 is the final all-worker release,
    // so every local Stage-I dependency has completed before this reset.
    for (uint32_t sem = SKV4_SEM_COHORT_SCORES_READY; sem <= SKV4_SEM_MASK_REAR_READY; ++sem)
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem)), 0);
    /*@CO_W_REPLAY_CLEANUP_E*/
}
"""
)


UNFUSED_DEBUG_FRONTIER = (
    semaphore_defines(SEM)
    + r"""    cb_wait_front(20, 1);
    cb_wait_front(21, 1);
    const uint32_t scratch = get_write_ptr(22);
    if (cohort) {
        noc_async_read(get_noc_addr(get_read_ptr(20)), scratch, 1024);
        noc_async_read(get_noc_addr(get_read_ptr(21)), scratch + 1024, 2048);
        noc_async_read_barrier();
        noc_async_write(scratch, get_noc_addr(parent_x, parent_y,
            get_write_ptr(23) + parent_level * 3072), 3072);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(parent_x, parent_y,
            get_semaphore(SKV4_SEM_FRONTIER_LEVEL_0 + parent_level)), 1);
    } else {
        auto native = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_read_ptr(21));
        auto positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t debug_rank = 0; debug_rank < P; ++debug_rank) {
            const uint32_t debug_lane =
                (debug_rank >> 5) + ((debug_rank & 31u) << 4);
            positions[debug_rank] = native[debug_lane];
        }
        noc_async_write(scratch, get_noc_addr(head, heads), P * 4);

        const uint32_t rank = P - 1;
        const uint32_t lane = (rank >> 5) + ((rank & 31u) << 4);
        auto scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            get_read_ptr(20));
        const uint32_t control = get_write_ptr(31);
        auto local = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(control);
        local[head * 8] = scores[lane];
        noc_async_write(control + head * 16,
            get_noc_addr(root_x, root_y, control + head * 16), 16);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_THRESHOLD_HEADS_READY)), 1);
    }
    cb_pop_front(20, 1);
    cb_pop_front(21, 1);
"""
)


def writer_source(*, debug_heads=False):
    """Generate the selector directly from its explicit debug contract."""
    from seoulkv4._kernels.codegen.coarse import writer

    return writer(select=True, debug_heads=debug_heads, profile=False)


def prepare(tt, device, cell, padded, points, shards, *, debug_heads=False):
    p = int(cell["p"])
    if p % 16 or not 16 <= p <= 512:
        raise ValueError("coarse K512 requires P in 16..512, multiple of16")
    if len(shards) % 4 or not 4 <= len(shards) <= 32:
        raise ValueError("coarse selection requires four-core cohorts")
    cohorts = []
    for start in range(0, len(shards), 4):
        members = shards[start : start + 4]
        base = members[0][1] * 32
        end = (members[-1][1] + members[-1][2]) * 32
        cohorts.append((base, end - base))
    if cohorts[-1][0] + cohorts[-1][1] != padded:
        raise ValueError("coarse cohorts do not cover the padded centroid range")
    max_local = max(span * 32 for _, _, span in shards)
    chunks = max(math.ceil(width / 512) for _, width in cohorts)
    page_bytes = math.ceil((16 + max_local + 6) * 4 / 64) * 64
    cores = _ranges(tt, points)
    frontier_specs = (
        [
            (CB.FRONTIER_VALUES, tt.bfloat16, 2048, 1),
            (CB.FRONTIER_INDICES, tt.uint32, 4096, 1),
            (CB.PEER_FRONTIERS, tt.uint8, 3072, 3),
            (CB.MERGE_VALUES, tt.bfloat16, 2048, 1),
            (CB.MERGE_INDICES, tt.uint32, 4096, 1),
        ]
        if debug_heads
        else [
            (CB.FRONTIER_VALUES, tt.uint32, 4096, 1),
            (CB.PEER_FRONTIERS, tt.uint8, 2048, 3),
            (CB.MERGE_VALUES, tt.uint32, 4096, 1),
        ]
    )
    specs = [
        (CB.LOCAL_SCORE_TILES, tt.bfloat16, 2048, chunks),
        (
            CB.SELECTOR_SCRATCH,
            tt.uint8,
            max(3072 if debug_heads else 2048, p * 4, page_bytes),
            1,
        ),
        (CB.WORKER_COUNTS, tt.uint8, len(points) * 64, 1),
        (CB.LOWER_INTERVALS, tt.bfloat16, max_local * 8, 1),
        (CB.UPPER_INTERVALS, tt.bfloat16, max_local * 8, 1),
        (CB.CONTROL, tt.uint8, 64, 1),
        *frontier_specs,
    ]
    cbs = [
        tt.CBDescriptor(
            total_size=page * count,
            core_ranges=cores,
            format_descriptors=[
                tt.CBFormatDescriptor(
                    buffer_index=index, data_format=dtype, page_size=page
                )
            ],
        )
        for index, dtype, page, count in specs
    ]
    semaphores = [
        tt.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in SEM
    ]
    output = tt.allocate_tensor_on_device(
        tt.Shape((math.ceil((16 + padded) / 1024), 1024)),
        tt.uint32,
        tt.ROW_MAJOR_LAYOUT,
        device,
        tt.L1_MEMORY_CONFIG,
    )
    heads = None
    if debug_heads:
        heads = tt.allocate_tensor_on_device(
            tt.Shape((4, p)),
            tt.uint32,
            tt.ROW_MAJOR_LAYOUT,
            device,
            tt.DRAM_MEMORY_CONFIG,
        )
    physical = [
        device.worker_core_from_logical_core(tt.CoreCoord(*point)) for point in points
    ]

    def args(index):
        cohort, head = divmod(index, 4)
        base, width = cohorts[cohort]
        stride = cohort & -cohort
        parent = cohort - stride if cohort else 0
        level = stride.bit_length() - 1 if cohort else 0
        root = physical[0]
        parent_core = physical[parent * 4 + head]
        group_xy = [
            value
            for core in physical[cohort * 4 : cohort * 4 + 4]
            for value in (core.x, core.y)
        ]
        all_xy = [value for core in physical for value in (core.x, core.y)]
        return dict(
            reader=dict(width=width, cohort=cohort, cohorts=len(cohorts)),
            compute=dict(
                chunks=math.ceil(width / 512),
                tail=(width - 1) % 512 + 1,
                base=base,
                cohort=cohort,
                cohorts=len(cohorts),
            ),
            writer=dict(
                base=base,
                cohort=cohort,
                head=head,
                worker=index,
                workers=len(points),
                root_x=root.x,
                root_y=root.y,
                parent_x=parent_core.x,
                parent_y=parent_core.y,
                parent_level=level,
                cohort_peers=group_xy,
                worker_peers=all_xy,
            ),
        )

    return dict(
        io=[output] + ([heads] if heads is not None else []),
        output=output,
        heads=heads,
        page_bytes=page_bytes,
        cbs=cbs,
        cb_specs=specs,
        semaphores=semaphores,
        args=args,
        cohorts=cohorts,
    )


def readback(tt, state):
    """Decode the compact header and globally sorted tag vector."""
    flat = tt.to_torch(state["output"]).long().flatten()
    count = int(flat[0])
    if not 0 <= count <= flat.numel() - 16:
        raise AssertionError(("invalid compact coarse count", count))
    return flat[16 : 16 + count]
