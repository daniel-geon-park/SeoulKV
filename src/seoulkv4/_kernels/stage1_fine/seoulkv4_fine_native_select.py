"""One-program Fine score routing, fused frontiers and balanced tree selection.

CB19: complete cohort/head score row; CB20: fused native frontier;
CB22: writer scratch; CB23: immutable-per-level fused K512 packets;
CB24: reader-published child frontier; CB26: final four-head packet.
All scratch addresses are symmetric within the cell. No intermediate GDDR.
"""

from seoulkv4._kernels.stage1_fine.abi import FineSemaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

from seoulkv4._kernels.stage1_fine.abi import FineCB as CB

import math
import os

import torch

from seoulkv4._kernels.topk.tt_topk_large_fixed import _ranges


def topology(n, shards):
    if len(shards) % 4 or not 4 <= len(shards) <= 32:
        raise ValueError("native Fine requires 4..32 workers in four-core cohorts")
    cohorts = []
    for start in range(0, len(shards), 4):
        members = shards[start : start + 4]
        base = members[0][1] * 32
        end = members[-1][1] * 32 + members[-1][2]
        cohorts.append((base, end - base))
    assert cohorts[-1][0] + cohorts[-1][1] == n
    return cohorts


def distributed_union_reference(coarse_ids, head_mask, positions, owners=4):
    """Reference the source-owned Fine union used by the device writer.

    ``positions`` are coarse-row positions, not cluster IDs.  Each owner gets
    a contiguous, 512-position-aligned range so concatenating owner outputs
    preserves the ascending ``coarse_ids`` order without another sort.
    """
    coarse_ids = torch.as_tensor(coarse_ids).long().flatten()
    head_mask = torch.as_tensor(head_mask).long().flatten()
    positions = torch.as_tensor(positions).long()
    if positions.ndim != 2 or positions.shape[0] != 4:
        raise ValueError("positions must have shape [4, P]")
    if owners != 4:
        raise ValueError("the production Fine union has four head owners")
    live = coarse_ids.numel()
    pages = (live + 511) // 512
    selected_positions = []
    status = 0
    mapped = torch.full_like(positions, 0xFFFFFFFF)
    per_head = []
    for h in range(4):
        valid = (positions[h] >= 0) & (positions[h] < live)
        valid_pos = positions[h, valid]
        allowed = ((head_mask[valid_pos] >> h) & 1).bool()
        status |= int((~valid).any() or (~allowed).any())
        accepted = valid_pos[allowed]
        per_head.append(set(accepted.tolist()))
        if allowed.any():
            original = valid.nonzero().flatten()[allowed]
            mapped[h, original] = coarse_ids[accepted]
    for owner in range(owners):
        first = pages * owner // owners * 512
        end = min(live, pages * (owner + 1) // owners * 512)
        union = set().union(*(row for row in per_head))
        selected_positions.extend(sorted(i for i in union if first <= i < end))
    selected = coarse_ids[torch.tensor(selected_positions, dtype=torch.long)]
    return selected, mapped, status


FUSED_POSITION_HELPER = r"""
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
"""


def compute_source(
    *,
    external_tags=False,
    shared_upper=False,
    dynamic=False,
    profile=None,
    wait_profile=None,
):
    """Generate this native selector role from its explicit input contract."""
    from seoulkv4._kernels.stage1_fine.seoulkv4_fine_packed import sources

    return sources(
        select=True,
        external_tags=external_tags,
        shared_upper=shared_upper,
        dynamic=dynamic,
        dual_risc=False,
        profile=profile,
        wait_profile=wait_profile,
    )[1]


FUSED_COMPUTE_TAIL = r"""
    // Convert each locally sorted raw TopK coordinate to a global u16 Fine
    // position once, then keep [BF16 score | u16 position] fused through every
    // running and cohort-tree merge.
    const uint32_t chunks = get_arg_val<uint32_t>(0);
    const uint32_t tail = get_arg_val<uint32_t>(1);
    const uint32_t base = get_arg_val<uint32_t>(2);
    const uint32_t cohort = get_arg_val<uint32_t>(3);
    const uint32_t cohorts = get_arg_val<uint32_t>(4);
    pack_reconfig_data_format(20);
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


DEBUG_COMPUTE_TAIL = r"""
    // The FPU scorer has released Dst and all score CBs. Same cores now sort
    // one head's complete cohort row; no additional program boundary.
    const uint32_t chunks = get_arg_val<uint32_t>(0);
    const uint32_t tail = get_arg_val<uint32_t>(1);
    const uint32_t cohort = get_arg_val<uint32_t>(3);
    const uint32_t cohorts = get_arg_val<uint32_t>(4);
    pack_reconfig_data_format(20);
    tile_regs_acquire();
    topk_xl_separate_indices_row_major_init_static<0, 0>();
    fine_sort_chunk(0, chunks == 1 ? tail : 512, false);
    if (chunks == 1) {
        topk_xl_init<1024, false>();
        topk_xl_rebuild<1024, false>(0, false);
    }
    for (uint32_t chunk = 1; chunk < chunks; ++chunk) {
        fine_sort_chunk(2, chunk + 1 == chunks ? tail : 512, true);
        topk_xl_init<1024, false>();
        topk_xl_merge<1024, false>(0);
        topk_xl_rebuild<1024, false>(0, false);
    }
    MATH(SFPU_UNARY_INIT(unused));
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, global_positions, 0,
        VectorMode::None, get_arg_val<uint32_t>(2)));
    for (uint32_t stride = 1; stride < cohorts; stride *= 2) {
        if (cohort % (2 * stride)) break;
        if (cohort + stride >= cohorts) continue;
        // Keep our own frontier in Dst. Only the child's paired native packet
        // crosses NoC; no parent pack/reimport at intermediate levels.
        cb_wait_front(24, 1);
        topk_xl_copy_tile_init(24);
        topk_xl_copy_tile<1024>(24, 2, 0, 512);
        cb_pop_front(24, 1);
        cb_wait_front(25, 1);
        topk_xl_copy_tile_init(25);
        topk_xl_copy_tile<1024>(25, 3, 0, 512);
        cb_pop_front(25, 1);
        topk_xl_init<1024, false>();
        topk_xl_rebuild<1024, false>(2, true);
        topk_xl_merge<1024, false>(0);
        topk_xl_rebuild<1024, false>(0, false);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(20, 1);
    pack_reconfig_data_format(20);
    pack_tile(0, 20);
    cb_push_back(20, 1);
    cb_reserve_back(21, 1);
    pack_reconfig_data_format(21);
    pack_tile(1, 21);
    cb_push_back(21, 1);
    tile_regs_release();
"""


READER_TAIL = (
    semaphore_defines(SEM)
    + r"""
    const uint32_t width = get_arg_val<uint32_t>(6);
    const uint32_t cohort = get_arg_val<uint32_t>(7);
    const uint32_t cohorts = get_arg_val<uint32_t>(8);
    const uint32_t chunks = (width + 511) / 512;
    cb_reserve_back(19, chunks);
    auto row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(19));
    // Every 512-score input occupies the first half of one physical tile.
    // Writers cover the live slots; clear only the final page's short tail.
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
        // Native packet arrives at a level-private inbox, immutable this step.
        const uint32_t src = get_write_ptr(23) + level * 2048;
        noc_async_read(get_noc_addr(src), get_write_ptr(24), 2048);
        noc_async_read_barrier();
        cb_push_back(24, 1);
    }
"""
)


DEBUG_READER_TAIL = (
    semaphore_defines(SEM)
    + r"""
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
        cb_reserve_back(25, 1);
        const uint32_t src = get_write_ptr(23) + level * 3072;
        noc_async_read(get_noc_addr(src), get_write_ptr(24), 1024);
        noc_async_read(get_noc_addr(src + 1024), get_write_ptr(25), 2048);
        noc_async_read_barrier();
        cb_push_back(24, 1);
        cb_push_back(25, 1);
    }
"""
)


WRITER = (
    semaphore_defines(SEM)
    + r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t TOTAL = get_compile_time_arg_val(2);
    constexpr uint32_t LIVE = get_compile_time_arg_val(3);
    constexpr uint32_t P = get_compile_time_arg_val(4);
    constexpr uint32_t OUT = get_compile_time_arg_val(5);
    constexpr uint32_t MAX_LOCAL = get_compile_time_arg_val(6);
    constexpr auto sa = TensorAccessorArgs<7>();
    constexpr auto ta = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto scores = TensorAccessor(sa, get_arg_val<uint32_t>(0), 2048);
    //4KiB pages spread the lookup across L1 banks. One giant row placed all
    //four cells on bank0 and collided with Stage1.6's static CBs at large pools.
    const auto tags = TensorAccessor(ta, get_arg_val<uint32_t>(1), 4096);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(2), 64);
    const auto heads = TensorAccessor(ha, get_arg_val<uint32_t>(3), P * 4);
    const uint32_t first = get_arg_val<uint32_t>(4);
    const uint32_t base = get_arg_val<uint32_t>(5);
    const uint32_t cohort = get_arg_val<uint32_t>(6);
    const uint32_t head = get_arg_val<uint32_t>(7);
    const uint32_t root_x = get_arg_val<uint32_t>(8);
    const uint32_t root_y = get_arg_val<uint32_t>(9);
    const uint32_t parent_x = get_arg_val<uint32_t>(10);
    const uint32_t parent_y = get_arg_val<uint32_t>(11);
    const uint32_t parent_level = get_arg_val<uint32_t>(12);
    const uint32_t scratch = get_write_ptr(22);
    auto masks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t rows_addr = scratch + MAX_LOCAL * 4;
    auto rows = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rows_addr);
    for (uint32_t copied = 0; copied < N * 4;) {
        const uint32_t offset = first * 32 * 4 + copied;
        const uint32_t page_left = 4096 - offset % 4096;
        const uint32_t bytes = N * 4 - copied < page_left ? N * 4 - copied : page_left;
        noc_async_read(get_noc_addr(offset / 4096, tags) + offset % 4096,
            scratch + copied, bytes);
        copied += bytes;
    }
    noc_async_read_barrier();
    for (uint32_t chunk = 0; chunk < N / 32; ++chunk) {
        cb_wait_front(16, 1);
        const uint32_t address = get_read_ptr(16);
        auto tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(address);
        // Keep the diagnostic device-score tensor unchanged (unmasked).
        noc_async_write(address, get_noc_addr(first + chunk, scores), 2048);
        for (uint32_t h = 0; h < 4; ++h) {
            for (uint32_t j = 0; j < 32; ++j) {
                const uint32_t pos = chunk * 32 + j;
                const uint32_t tile_pos = h * 16 + (j & 15u) + (j / 16) * 256;
                rows[h * N + pos] = ((masks[pos] >> (28 + h)) & 1u)
                    && first * 32 + pos < LIVE ? tile[tile_pos] : 0xff80;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
    for (uint32_t h = 0; h < 4; ++h) {
        const uint32_t x = get_arg_val<uint32_t>(13 + h * 2);
        const uint32_t y = get_arg_val<uint32_t>(14 + h * 2);
        // K512 consumes one half-filled physical tile per 512 scores. Split
        // only at those page boundaries; each span remains one bulk NoC.
        uint32_t src = rows_addr + h * N * 2;
        uint32_t offset = first * 32 - base;
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
    for (uint32_t h = 0; h < 4; ++h) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(13 + h * 2),
            get_arg_val<uint32_t>(14 + h * 2), get_semaphore(SKV4_SEM_COHORT_SCORES_READY)), 1);
    }

    cb_wait_front(20, 1);
    if (cohort) {
        // One bulk packet carries the native fused K512 frontier. The low
        // half already contains the global compact Fine position.
        noc_async_write(get_read_ptr(20), get_noc_addr(parent_x, parent_y,
            get_write_ptr(23) + parent_level * 2048), 2048);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(parent_x, parent_y,
            get_semaphore(SKV4_SEM_FRONTIER_LEVEL_0 + parent_level)), 1);
    } else {
        auto native = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(20));
        auto positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t rank = 0; rank < P; ++rank) {
            // K512's native rank matrix is 32x16 (K1024 was 64x16).
            const uint32_t lane = (rank >> 5) + ((rank & 31u) << 4);
            positions[rank] = native[lane] & 0xffffu;
        }
    }
    cb_pop_front(20, 1);
    // Cohort helpers have already sent their parent notification.  Flush that
    // non-posted atomic before the early kernel exit can expose this core's L1
    // semaphore region to the next program.
    noc_async_atomic_barrier();
    if (cohort)
        for (uint32_t sem = SKV4_SEM_COHORT_SCORES_READY; sem <= SKV4_SEM_UNION_OFFSETS_READY; ++sem)
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem)), 0);
    if (cohort) return;

    // The four cohort-zero roots now own one head and one contiguous source
    // range each.  Keep the diagnostic tensor in coarse-position form; the
    // untimed host checker maps it to cluster IDs.
    /*@FN_DU_BITMAP_B*/
    auto positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    noc_async_write(scratch, get_noc_addr(head, heads), P * 4);

    const uint32_t union_scratch = get_write_ptr(27);
    constexpr uint32_t BITMAP_PAGES = (TOTAL + 511) / 512;
    constexpr uint32_t BITMAP_BYTES = BITMAP_PAGES * 64;
    constexpr uint32_t MAX_OWNER_PAGES = (BITMAP_PAGES + 3) / 4;
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
    /*@FN_DU_BITMAP_E*/

    // One barrier after all four local bitmaps exist.  Root zero releases the
    // other roots; subsequent transfers are one bulk bitmap slice per peer.
    /*@FN_DU_EXCHANGE_B*/
    noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_UNION_BITMAPS_READY)), 1);
    if (head == 0) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_UNION_BITMAPS_READY)), 4);
        for (uint32_t h = 1; h < 4; ++h)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(13 + h * 2),
                get_arg_val<uint32_t>(14 + h * 2), get_semaphore(SKV4_SEM_UNION_BITMAPS_READY)), 1);
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
            const uint32_t x = get_arg_val<uint32_t>(13 + h * 2);
            const uint32_t y = get_arg_val<uint32_t>(14 + h * 2);
            noc_async_read(get_noc_addr(x, y, union_scratch + page_first * 64),
                union_scratch + REMOTE_OFF + h * MAX_SLICE_BYTES,
                owner_pages * 64);
        }
    }
    // Load only this owner's contiguous tag slice, rather than all tags on one
    // root.  Slice starts are 2KiB aligned and retain narrow-transfer alignment.
    for (uint32_t copied = 0; copied < owner_keys * 4;) {
        const uint32_t logical = first_pos * 4 + copied;
        const uint32_t page_left = 4096 - logical % 4096;
        const uint32_t bytes = owner_keys * 4 - copied < page_left
            ? owner_keys * 4 - copied : page_left;
        noc_async_read(get_noc_addr(logical / 4096, tags) + logical % 4096,
            union_scratch + LOOKUP_OFF + copied, bytes);
        copied += bytes;
    }
    noc_async_read_barrier();
    /*@FN_DU_EXCHANGE_E*/

    auto lookup = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(union_scratch + LOOKUP_OFF);
    // The self slot in REMOTE_OFF is never populated by peer DMA. Cache the
    // validated accepted union words there so enumeration does not rescan all
    // four head bitmaps and masks.
    auto accepted_cache = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        union_scratch + REMOTE_OFF + head * MAX_SLICE_BYTES);
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
            const uint32_t x = get_arg_val<uint32_t>(13 + h * 2);
            const uint32_t y = get_arg_val<uint32_t>(14 + h * 2);
            noc_async_write(reinterpret_cast<uint32_t>(offsets),
                get_noc_addr(x, y, union_scratch + OFFSETS_OFF), 64);
        }
        noc_async_write_barrier();
        for (uint32_t h = 1; h < 4; ++h)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(13 + h * 2),
                get_arg_val<uint32_t>(14 + h * 2), get_semaphore(SKV4_SEM_UNION_OFFSETS_READY)), 1);

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
    /*@FN_DU_COUNT_E*/

    // Re-enumerate only this owner's range.  Prefix-padding the local list
    // keeps the source/destination low address bits equal for narrow writes.
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
    /*@FN_DU_ENUM_E*/
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
    /*@FN_DU_REPLAY_CLEANUP_B*/
    noc_async_atomic_barrier();
    for (uint32_t sem = SKV4_SEM_COHORT_SCORES_READY; sem <= SKV4_SEM_UNION_OFFSETS_READY; ++sem)
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem)), 0);
    /*@FN_DU_REPLAY_CLEANUP_E*/
    /*@FN_DU_EXPORT_E*/
}
"""
)


def prepare(
    tt,
    device,
    cell,
    scores,
    n,
    points,
    shards,
    *,
    external_tags=None,
    extra_worker_bytes=0,
):
    p = int(cell["p"])
    if p % 16 or not 16 <= p <= 512:
        raise ValueError("native Fine K512 requires nprobe16..512, multiple of16")
    cohorts = topology(n, shards)
    if n > 65536:
        raise ValueError("global fused Fine positions exceed16 bits")
    outwidth = math.ceil((16 + 4 * p) / 16) * 16
    live = len(cell["coarse_ids"])
    if external_tags is None:
        tags = torch.zeros(math.ceil(n / 1024) * 1024, dtype=torch.int64)
        tags[:live] = cell["coarse_ids"].long() | (cell["head_mask"].long() << 28)
        tags = tt.from_torch(
            tags.int().reshape(-1, 1024),
            dtype=tt.uint32,
            device=device,
            layout=tt.ROW_MAJOR_LAYOUT,
            memory_config=tt.L1_MEMORY_CONFIG,
        )
        tag_header = 0
    else:
        tags = external_tags
        tag_header = 16
    output = tt.allocate_tensor_on_device(
        tt.Shape((outwidth // 16, 16)),
        tt.uint32,
        tt.ROW_MAJOR_LAYOUT,
        device,
        tt.L1_MEMORY_CONFIG,
    )
    heads = tt.allocate_tensor_on_device(
        tt.Shape((4, p)),
        tt.uint32,
        tt.ROW_MAJOR_LAYOUT,
        device,
        tt.L1_MEMORY_CONFIG,
    )
    max_local = max(row[2] for row in shards)
    chunks = max(math.ceil(width / 512) for _, width in cohorts)
    bitmap_pages = math.ceil(n / 512)
    bitmap_bytes = bitmap_pages * 64
    max_owner_pages = math.ceil(bitmap_pages / 4)
    max_slice_bytes = max_owner_pages * 64
    max_owner_keys = max_owner_pages * 512
    remote_off = bitmap_bytes
    lookup_off = remote_off + 4 * max_slice_bytes
    list_off = math.ceil((lookup_off + max_owner_keys * 4) / 64) * 64
    control_off = math.ceil((list_off + (max_owner_keys + 16) * 4) / 64) * 64
    offsets_off = control_off + 4 * 64
    header_off = offsets_off + 64
    union_bytes = header_off + 64
    specs = [
        (CB.LOCAL_SCORE_TILES, tt.bfloat16, 2048, chunks),
        (CB.FRONTIER_VALUES, tt.uint32, 4096, 1),
        (
            CB.SELECTOR_SCRATCH,
            tt.uint8,
            math.ceil(max(max_local * 12, 2048, p * 4) / 64) * 64,
            1,
        ),
        (CB.PEER_FRONTIERS, tt.uint8, 2048, 3),
        (CB.MERGE_VALUES, tt.uint32, 4096, 1),
        (CB.UNION_SCRATCH, tt.uint8, math.ceil(union_bytes / 64) * 64, 1),
    ]
    total_bytes = (
        max_local * 72
        + 45056
        + extra_worker_bytes
        + sum(page * count for _, _, page, count in specs)
    )
    if total_bytes > 1000000:
        raise ValueError(f"native Fine worker CB budget exceeded: {total_bytes} bytes")
    cores = _ranges(tt, points)
    cbs = [
        tt.CBDescriptor(
            total_size=page * count,
            core_ranges=cores,
            format_descriptors=[
                tt.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=page)
            ],
        )
        for i, dtype, page, count in specs
    ]
    sems = [
        tt.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in SEM
    ]
    physical = [
        device.worker_core_from_logical_core(tt.CoreCoord(*pnt)) for pnt in points
    ]

    def args(index):
        cohort, head = divmod(index, 4)
        base, width = cohorts[cohort]
        stride = cohort & -cohort
        parent = cohort - stride if cohort else 0
        level = stride.bit_length() - 1 if cohort else 0
        root, parent_core = physical[0], physical[parent * 4 + head]
        group_xy = [
            v for c in physical[cohort * 4 : cohort * 4 + 4] for v in (c.x, c.y)
        ]
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
                root_x=root.x,
                root_y=root.y,
                parent_x=parent_core.x,
                parent_y=parent_core.y,
                parent_level=level,
                cohort_peers=group_xy,
                tag_header=tag_header,
            ),
        )

    return dict(
        io=[scores, tags, output, heads],
        output=output,
        heads=heads,
        outwidth=outwidth,
        max_local=max_local,
        cbs=cbs,
        semaphores=sems,
        args=args,
        cohorts=cohorts,
        external_tags=external_tags is not None,
    )


def reader_source(
    *,
    external_tags=False,
    shared_upper=False,
    dynamic=False,
    profile=None,
    wait_profile=None,
):
    """Generate this native selector role from its explicit input contract."""
    from seoulkv4._kernels.stage1_fine.seoulkv4_fine_packed import sources

    return sources(
        select=True,
        external_tags=external_tags,
        shared_upper=shared_upper,
        dynamic=dynamic,
        dual_risc=False,
        profile=profile,
        wait_profile=wait_profile,
    )[0]


def writer_source(
    *,
    external_tags=False,
    shared_upper=False,
    dynamic=False,
    profile=None,
    wait_profile=None,
):
    """Generate this native selector role from its explicit input contract."""
    from seoulkv4._kernels.stage1_fine.seoulkv4_fine_packed import sources

    return sources(
        select=True,
        external_tags=external_tags,
        shared_upper=shared_upper,
        dynamic=dynamic,
        dual_risc=False,
        profile=profile,
        wait_profile=wait_profile,
    )[2]
