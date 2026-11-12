"""Fused Blackhole SeoulKV4 Stage 1.6, Stage II, and sparse attention.

Each ``(request, KV-head)`` slot uses four candidate shards.  A shard is one
physical row of four Tensix cores. In the union path every core gathers its own
packed 2-bit quarter directly into UInt32 tiles; helpers bulk-write retained
quarters to the root. It decodes once into BFP4 and matrix-scores all
four GQA heads; bulk BF16 score quarters return to each head's local TopK owner.
The per-head control still scores one head per core directly with SFPU.
Each local head worker routes its native sorted frontier directly to its head root;
root ``h`` owns sparse attention for head ``h``. With ``union_inputs`` the
Stage-II matrix scorer stays at each original key shard and scores all heads;
only small local top-100 frontiers move to the final head owners. Without it,
the previous owner-local residual-SFPU Stage II remains the control.

The Stage-1.5 handoff (candidate IDs and BF16 2-bit scores) is supplied by the
decode benchmark until the earlier stages are ported.  It is consumed from L1
inside the kernels; no selector intermediate is written to GDDR. The union
path runs Stage 1.6, Stage II, and final attention as three ordered programs
with a persistent L1 handoff. Stage II uses all four cores per candidate shard;
the attention program uses the four query-head owners.
"""

from __future__ import annotations

from seoulkv4._kernels.common.refinement_abi import RefinementSemaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

from seoulkv4._kernels.common.refinement_abi import RefinementCB as CB

import os
from seoulkv4._kernels.stage16.stage16_schedule import (
    config as stage16_schedule_config,
)
import struct

from seoulkv4._kernels.stage2.seoulkv4_stage2_fused import (
    D,
    GQA_GROUP,
    PADDED_TOPK,
    ROWS,
    TOPK,
    _ranges,
)

STAGE16_COMPUTE = r"""
#define kernel_main sparse_tail_main
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sparse_sdpa_compute.cpp"
#undef kernel_main
#include "api/compute/experimental/topk_xl.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// Stage 1.6 uses 16 packed 2-bit planes over 16 UInt32 words per key.  The
// packed tile occupies only the left faces.  Its two query tiles use both face
// columns as storage: four planes per face column, eight planes per tile.
// VectorMode::C invokes this function only for packed faces 0 and 2; adding
// QUERY_FACE_OFFSET selects the corresponding query face without processing
// the packed tile's unused right half.
template <uint32_t PLANE, uint32_t SHIFT, bool HAS_PRIOR>
inline void packed2_plane_mad_body() {
    constexpr uint32_t QUERY_TILE_OFFSET = 32 + (PLANE / 8) * 32;
    constexpr uint32_t QUERY_FACE_OFFSET = ((PLANE / 4) % 2) * 8;
    constexpr uint32_t QUERY_VECTOR_OFFSET = 2 * (PLANE % 4);
    constexpr uint32_t PRIOR_TILE_OFFSET = 96;
    for (uint32_t parity = 0; parity < 2; ++parity) {
        sfpi::vFloat qv =
            sfpi::dst_reg[QUERY_TILE_OFFSET + QUERY_FACE_OFFSET +
                          QUERY_VECTOR_OFFSET + parity]
                .mode<sfpi::DataLayout::F32>();
        for (uint32_t row_group = 0; row_group < 4; ++row_group) {
            const uint32_t packed_vector = 2 * row_group + parity;
            sfpi::vUInt packed =
                sfpi::dst_reg[packed_vector].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt code = sfpi::shft(packed, -SHIFT) & 3u;
            sfpi::vInt level = sfpi::vInt(code + code) - 3;
            sfpi::vFloat value = sfpi::convert<sfpi::vFloat>(
                sfpi::convert<sfpi::vSMag>(level),
                sfpi::RoundMode::Nearest);
            sfpi::vFloat acc = value * qv;
            if constexpr (HAS_PRIOR) {
                sfpi::vFloat prior =
                    sfpi::dst_reg[PRIOR_TILE_OFFSET + packed_vector]
                        .mode<sfpi::DataLayout::F32>();
                acc = acc + prior;
            }
            sfpi::dst_reg[PRIOR_TILE_OFFSET + packed_vector]
                .mode<sfpi::DataLayout::F32>() = acc;
        }
    }
    // The packed right face is deliberately absent.  Zero the matching
    // accumulator face on the first plane so the full-width reduction below
    // sees sixteen useful partials followed by sixteen exact zeros.
    if constexpr (!HAS_PRIOR) {
        for (uint32_t vector = 0; vector < 8; ++vector) {
            sfpi::dst_reg[PRIOR_TILE_OFFSET + 8 + vector]
                .mode<sfpi::DataLayout::F32>() = 0.0f;
        }
    }
}

// Keep all sixteen plane MADs inside one SFPU traversal.  Besides avoiding
// sixteen face-address setup/restore sequences, this keeps the combined
// Stage-1.6 + Stage-II + sparse-attention compute binary below Blackhole's
// kernel-config limit.
template <bool UNUSED>
inline void packed2_all_planes_face() {
    packed2_plane_mad_body<0, 0, false>();
    packed2_plane_mad_body<1, 2, true>();
    packed2_plane_mad_body<2, 4, true>();
    packed2_plane_mad_body<3, 6, true>();
    packed2_plane_mad_body<4, 8, true>();
    packed2_plane_mad_body<5, 10, true>();
    packed2_plane_mad_body<6, 12, true>();
    packed2_plane_mad_body<7, 14, true>();
    packed2_plane_mad_body<8, 16, true>();
    packed2_plane_mad_body<9, 18, true>();
    packed2_plane_mad_body<10, 20, true>();
    packed2_plane_mad_body<11, 22, true>();
    packed2_plane_mad_body<12, 24, true>();
    packed2_plane_mad_body<13, 26, true>();
    packed2_plane_mad_body<14, 28, true>();
    packed2_plane_mad_body<15, 30, true>();
}

// Dst tile 0 is one UInt32 tile: 32 candidates x 32 packed words. Dst tiles
// 1-2 compact the eight query nibble planes, four planes per tile.  Within a
// face each plane occupies one four-row group: its even- and odd-column
// vectors are therefore adjacent Dst vectors.  Dst tile 3 retains the FP32
// accumulator across all eight planes.
template <uint32_t PLANE, uint32_t SHIFT, bool HAS_PRIOR>
inline void packed_plane_mad_face() {
    constexpr uint32_t QUERY_TILE_OFFSET = 32 + (PLANE / 4) * 32;
    constexpr uint32_t QUERY_VECTOR_OFFSET = 2 * (PLANE % 4);
    constexpr uint32_t PRIOR_TILE_OFFSET = 96;
    // SFPLOAD exposes one face as four groups of four rows, with separate
    // 8-column vectors for even and odd columns.  The compact query tile
    // repeats each 8-value half over its four physical rows.  Select the
    // matching parity vector for every packed row group.
    for (uint32_t parity = 0; parity < 2; ++parity) {
        sfpi::vFloat qv =
            sfpi::dst_reg[QUERY_TILE_OFFSET + QUERY_VECTOR_OFFSET +
                          parity]
                .mode<sfpi::DataLayout::F32>();
        // One q load serves all four candidate row groups of this parity.
        for (uint32_t row_group = 0; row_group < 4; ++row_group) {
            const uint32_t packed_vector = 2 * row_group + parity;
            sfpi::vUInt packed =
                sfpi::dst_reg[packed_vector].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt code = sfpi::shft(packed, -SHIFT) & 15u;
            sfpi::vInt level = sfpi::vInt(code + code) - 15;
            sfpi::vFloat value = sfpi::convert<sfpi::vFloat>(
                sfpi::convert<sfpi::vSMag>(level),
                sfpi::RoundMode::Nearest);
            sfpi::vFloat acc = value * qv;
            if constexpr (HAS_PRIOR) {
                sfpi::vFloat prior =
                    sfpi::dst_reg[PRIOR_TILE_OFFSET + packed_vector]
                        .mode<sfpi::DataLayout::F32>();
                acc = acc + prior;
            }
            sfpi::dst_reg[PRIOR_TILE_OFFSET + packed_vector]
                .mode<sfpi::DataLayout::F32>() = acc;
        }
    }
}

}  // namespace ckernel::sfpu
#endif

template <uint32_t PLANE, uint32_t SHIFT, bool HAS_PRIOR>
inline void packed_plane_mad_tile(uint32_t dst_index) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, packed_plane_mad_face,
        (PLANE, SHIFT, HAS_PRIOR),
        dst_index, VectorMode::RC));
}

inline void packed2_all_planes_tile(uint32_t dst_index) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, packed2_all_planes_face,
        (false),
        dst_index, VectorMode::C));
}

// Exact local top-160 uses one Blackhole TopK-XL K=1024 sort.  The writer
// presents 928 live BF16 scores followed by -inf padding in one page. Values
// and indices remain fused as (BF16-score|u16-index) and are packed as one native tile;
// the writer reads the fixed rank-to-tile permutation directly.  Avoiding
// two K=512 sorts plus merge/rebuild matters for both latency and Blackhole's
// per-core kernel-config budget.
inline void stage16_local_topk(
    uint32_t input_id, uint32_t output_id, uint32_t live_elements) {
    constexpr uint32_t TK = 1024;
    constexpr uint32_t second_input_id = 4;
    CircularBuffer input(input_id), second_input(second_input_id), output(output_id);
    pack_reconfig_data_format(output_id);
    input.wait_front(1);

    // Sort each 1024-element half independently.  Keeping the two sorted
    // frontiers avoids the fragile in-Dst merge/rebuild path at LOCAL16=2048;
    // the writer needs only KLOCAL entries and merges those two short lists.
    tile_regs_acquire();
    topk_xl_copy_tile_init(input_id);
    const uint32_t first=live_elements<1024?live_elements:1024;
    topk_xl_copy_tile<TK>(input_id,0,0,first);
    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<TK,0>(0);
    topk_xl_init<TK,true>();
    topk_xl_local_sort<TK>(0,false);
    tile_regs_commit();
    output.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, output_id);
    output.push_back(1);
    tile_regs_release();

    if(live_elements>1024) {
        second_input.wait_front(1);
        tile_regs_acquire();
        topk_xl_copy_tile_init(second_input_id);
        // TopK-XL advances by CB page size, not tile size.  CB30 has a large
        // metadata page, so Stage-1.5's second 1024-score half is published
        // through one native-sized CB4 page and read as tile zero.
        topk_xl_copy_tile<TK>(second_input_id,0,0,live_elements-1024);
        topk_xl_add_lsb_indices_init();
        topk_xl_add_lsb_indices<TK,1>(0);
        topk_xl_init<TK,true>();
        topk_xl_local_sort<TK>(0,false);
        tile_regs_commit();
        output.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, output_id);
        output.push_back(1);
        tile_regs_release();
        second_input.pop_front(1);
    }
    input.pop_front(1);
}

void refinement_main(bool owner) {
    constexpr uint32_t LOCAL16_PAD =
        get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END);
    constexpr uint32_t M16_PAD =
        get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END + 1);
    constexpr uint32_t PROFILE =
        get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END + 2);
    constexpr uint32_t STAGE2_PLANES =
        get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END + 3);
    constexpr uint32_t D = 256;
    constexpr uint32_t packed2 = 26, query16 = 24;
    constexpr uint32_t packed4 = 25, scores_id = 28;
    // Keep every CB's producer/consumer direction fixed. CB30 carries the
    // writer-produced BF16 TopK input, CB17 the compute-produced fused result,
    // and CB24 is refilled with the Stage-II query after its Stage-1.6 query
    // pages have been consumed.
    constexpr uint32_t topk_input = 30;
    constexpr uint32_t query_pattern = 24;
    constexpr uint32_t topk_output = 17;
    constexpr uint32_t reduce16 = 23, reduce_ones = 29, partial_id = 5;

    compute_kernel_hw_startup(packed2, query16);
    if constexpr (PROFILE < 3) return;

    CircularBuffer scores(scores_id), partial_cb(partial_id);
    CircularBuffer packed2_cb(packed2), query16_cb(query16);
    CircularBuffer reduce16_cb(reduce16);
    query16_cb.wait_front(2);
    reduce16_cb.wait_front(1);

    // Stage 1.6: directly score all sixteen 2-bit planes.  Every physical
    // core owns one GQA query head; no BFP4 key tile or cross-core score tile
    // is materialized.
    for (uint32_t chunk = 0; chunk < LOCAL16_PAD / 32; ++chunk) {
        packed2_cb.wait_front(1);
        tile_regs_acquire();
        const uint32_t before_packed = chunk == 0 ? query16 : partial_id;
        copy_tile_to_dst_init_short_with_dt(before_packed, packed2);
        copy_tile(packed2, 0, 0);
        copy_tile_to_dst_init_short_with_dt(packed2, query16);
        copy_tile(query16, 0, 1);
        copy_tile(query16, 1, 2);
        packed2_all_planes_tile(0);
        tile_regs_commit();
        partial_cb.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(partial_id);
        pack_tile(3, partial_id);
        tile_regs_release();
        partial_cb.push_back(1);

        partial_cb.wait_front(1);
        scores.reserve_back(1);
        reconfig_full_operand(partial_id, reduce16);
        pack_reconfig_data_format(scores_id);
        mm_no_mop_init_short(partial_id, reduce16, true, 1, 1, 1);
        configure_row_pack_width(scores_id, 1);
        blocked_matmul_and_pack<true, 1, 1>(
            partial_id, reduce16, scores_id, 0, 0, 0, 0, 1, 1,
            1, 1, true);
        scores.push_back(1);
        partial_cb.pop_front(1);
        packed2_cb.pop_front(1);
    }
    query16_cb.pop_front(2);
    reduce16_cb.pop_front(1);

    if constexpr (PROFILE >= 4) {
        // The writer publishes one or two K=1024 BF16 tiles on dedicated CB30;
        // the fused score/index result returns on CB17.
        static_assert(LOCAL16_PAD <= 2048);
        stage16_local_topk(topk_input, topk_output, LOCAL16_PAD);
    }

    if (!owner || PROFILE < 8) {
        return;
    }

    // Stage II: packed UInt32 record -> SFPU nibble decode/query MAD -> one
    // tile-native reduction. No expanded BFP8 candidate tile is materialized.
    CircularBuffer packed_cb(packed4), query_cb(query_pattern);
    CircularBuffer ones_cb(reduce_ones);
    query_cb.wait_front(2);
    ones_cb.wait_front(4);
    for (uint32_t chunk = 0; chunk < M16_PAD / 32; ++chunk) {
        packed_cb.wait_front(1);
        // Packed code and the two compact query tiles are invariant across
        // all eight nibble planes. Load all three once and keep them plus the
        // FP32 accumulator in Dst for the complete chunk.
        tile_regs_acquire();
        const uint32_t before_first_packed = partial_id;
        copy_tile_to_dst_init_short_with_dt(before_first_packed, packed4);
        copy_tile(packed4, 0, 0);
        copy_tile_to_dst_init_short_with_dt(packed4, query_pattern);
        copy_tile(query_pattern, 0, 1);
        copy_tile(query_pattern, 1, 2);
        packed_plane_mad_tile<0, 0, false>(0);
        if constexpr (STAGE2_PLANES > 1)
            packed_plane_mad_tile<1, 4, true>(0);
        if constexpr (STAGE2_PLANES > 2)
            packed_plane_mad_tile<2, 8, true>(0);
        if constexpr (STAGE2_PLANES > 3)
            packed_plane_mad_tile<3, 12, true>(0);
        if constexpr (STAGE2_PLANES > 4)
            packed_plane_mad_tile<4, 16, true>(0);
        if constexpr (STAGE2_PLANES > 5)
            packed_plane_mad_tile<5, 20, true>(0);
        if constexpr (STAGE2_PLANES > 6)
            packed_plane_mad_tile<6, 24, true>(0);
        if constexpr (STAGE2_PLANES > 7)
            packed_plane_mad_tile<7, 28, true>(0);

        tile_regs_commit();
        partial_cb.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(partial_id);
        pack_tile(3, partial_id);
        tile_regs_release();
        partial_cb.push_back(1);

        partial_cb.wait_front(1);
        scores.reserve_back(1);
        reconfig_full_operand(partial_id, reduce_ones);
        pack_reconfig_data_format(scores_id);
        mm_no_mop_init_short(partial_id, reduce_ones, true, 1, 1, 1);
        configure_row_pack_width(scores_id, 1);
        blocked_matmul_and_pack<true, 1, 1>(
            partial_id, reduce_ones, scores_id, 0, 0, 0, 0, 1, 1,
            1, 1, true);
        scores.push_back(1);
        partial_cb.pop_front(1);
        packed_cb.pop_front(1);
    }
    query_cb.pop_front(2);
    ones_cb.pop_front(4);
}

void kernel_main() {
    constexpr uint32_t PROFILE =
        get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END + 2);
    constexpr uint32_t G = 4;
    const uint32_t shard = get_arg_val<uint32_t>(0);
    const uint32_t part = get_arg_val<uint32_t>(3);
    const bool owner = part == 0;
    refinement_main(owner);
    if constexpr (PROFILE >= 10) {
        if (owner) sparse_tail_main();
    }
}
"""


STAGE16_READER = (
    semaphore_defines(SEM)
    + r"""
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"

constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t N16 = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t M16 = get_compile_time_arg_val(3);
    constexpr uint32_t D = get_compile_time_arg_val(4);
    constexpr uint32_t PROFILE = get_compile_time_arg_val(5);
    constexpr uint32_t PARTS = 4;
    constexpr uint32_t LOCAL16 = N16 / W;
    constexpr uint32_t LOCAL16_PAD = ((LOCAL16 + 31) / 32) * 32;
    constexpr uint32_t M16_PAD = ((M16 + 31) / 32) * 32;
    constexpr uint32_t CHUNKS = LOCAL16_PAD / 32;
    constexpr uint32_t INHERITED_OFFSET = 0;
    constexpr uint32_t Q2_ROW_OFFSET =
        (LOCAL16 * sizeof(uint16_t) + 63) & ~63u;
    constexpr uint32_t STAGE2_QUERY_OFFSET = 8192;
    constexpr bool PER_HEAD_INPUT = get_compile_time_arg_val(6) != 0;
    constexpr auto sqa = TensorAccessorArgs<7>();
    constexpr auto r2a = TensorAccessorArgs<sqa.next_compile_time_args_offset()>();
    constexpr auto r4a = TensorAccessorArgs<r2a.next_compile_time_args_offset()>();
    constexpr auto cia = TensorAccessorArgs<r4a.next_compile_time_args_offset()>();
    constexpr auto s2a = TensorAccessorArgs<cia.next_compile_time_args_offset()>();
    constexpr auto aqa = TensorAccessorArgs<s2a.next_compile_time_args_offset()>();
    constexpr auto kva = TensorAccessorArgs<aqa.next_compile_time_args_offset()>();
    static_assert(N16 % W == 0);
    static_assert(W >= G);

    const uint32_t sq_addr = get_arg_val<uint32_t>(0);
    const uint32_t r2_addr = get_arg_val<uint32_t>(1);
    const uint32_t r4_addr = get_arg_val<uint32_t>(2);
    const uint32_t cand_addr = get_arg_val<uint32_t>(3);
    const uint32_t s2_addr = get_arg_val<uint32_t>(4);
    const uint32_t aq_addr = get_arg_val<uint32_t>(5);
    const uint32_t kv_addr = get_arg_val<uint32_t>(6);
    const uint32_t slot = get_arg_val<uint32_t>(7);
    const uint32_t context = get_arg_val<uint32_t>(8);
    const uint32_t shard = get_arg_val<uint32_t>(9);
    const uint32_t part = get_arg_val<uint32_t>(10);
    const uint32_t root_x = get_arg_val<uint32_t>(11);
    const uint32_t root_y = get_arg_val<uint32_t>(12);
    const uint32_t row_start_x = get_arg_val<uint32_t>(13);
    const uint32_t row_start_y = get_arg_val<uint32_t>(14);
    const uint32_t row_end_x = get_arg_val<uint32_t>(15);
    const uint32_t row_end_y = get_arg_val<uint32_t>(16);
    const bool owner = part == 0;

    const auto sq = TensorAccessor(sqa, sq_addr, D * 2);
    const auto r2 = TensorAccessor(r2a, r2_addr, D / 4);
    const auto r4 = TensorAccessor(r4a, r4_addr, D / 2);
    const auto cand = TensorAccessor(cia, cand_addr, N16 * 4);
    const auto s2 = TensorAccessor(s2a, s2_addr, N16 * 2);
    const auto aq = TensorAccessor(aqa, aq_addr, 2 * D * 2);
    const auto kv = TensorAccessor(kva, kv_addr, 2 * D);

    // Every core gets its own head's inherited Stage-1.5 scores. The writer
    // RISC independently builds this core's selector-query tiles while this
    // reader gathers records. The shard root alone also reads candidate IDs.
    cb_reserve_back(31, 1);
    const uint32_t scratch = get_write_ptr(31);
    noc_async_read(
        get_noc_addr(slot * G + part, s2) + shard * LOCAL16 * 2,
        scratch + INHERITED_OFFSET, LOCAL16 * 2);
    if (owner) {
        noc_async_read(get_noc_addr(slot * 32 + shard, sq),
                       scratch + Q2_ROW_OFFSET, D * 2);
    }

    uint32_t cand_l1 = 0;
    uint32_t cand_prefix = 0;
    if (owner || PER_HEAD_INPUT) {
        cb_reserve_back(27, 1);
        cand_l1 = get_write_ptr(27);
        const uint32_t cand_offset = shard * LOCAL16 * 4;
        cand_prefix = cand_offset & 63;
        const uint32_t cand_page = PER_HEAD_INPUT ? slot * G + part : slot;
        noc_async_read(get_noc_addr(cand_page, cand) + cand_offset - cand_prefix,
                       cand_l1, LOCAL16 * 4 + cand_prefix);
    }
    noc_async_read_barrier();
    if (owner || PER_HEAD_INPUT) cb_push_back(27, 1);

    // Launch the sparse record reads before constructing the query/reduction
    // tiles.  Those are pure local-RISC stores, so their setup latency can sit
    // under the outstanding GDDR reads instead of extending the critical path.
    cb_reserve_back(26, CHUNKS);
    const uint32_t packed_tiles = get_write_ptr(26);
    if (owner || PER_HEAD_INPUT) {
        volatile tt_l1_ptr uint32_t* candidates =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_l1 + cand_prefix);
        for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t local = chunk * 32 + row;
                const uint32_t source = local < LOCAL16 ? local : 0;
                const uint32_t face = row < 16 ? 0 : 2;
                const uint32_t rr = row & 15;
                noc_async_read(
                    get_noc_addr(slot * context + candidates[source], r2),
                    packed_tiles + chunk * 4096 + face * 1024 + rr * 64,
                    D / 4);
            }
        }
    }

    if (owner) {
        // Stage II keeps its own eight-plane query pattern in scratch until
        // this core receives the global top-400 frontier.
        volatile tt_l1_ptr uint16_t* q2_src =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                scratch + Q2_ROW_OFFSET);
        volatile tt_l1_ptr uint32_t* q2_dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                scratch + STAGE2_QUERY_OFFSET);
        for (uint32_t query_tile = 0; query_tile < 2; ++query_tile) {
            for (uint32_t face_row = 0; face_row < 2; ++face_row) {
                for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                    const uint32_t face = 2 * face_row + face_col;
                    for (uint32_t local_plane = 0; local_plane < 4;
                         ++local_plane) {
                        const uint32_t plane = 4 * query_tile + local_plane;
                        for (uint32_t row = 4 * local_plane;
                             row < 4 * local_plane + 4; ++row) {
                            const uint32_t base =
                                query_tile * 1024 + face * 256 + row * 16;
                            for (uint32_t col = 0; col < 16; ++col) {
                                const uint32_t word = 16 * face_col + col;
                                q2_dst[base + col] =
                                    uint32_t(q2_src[8 * word + plane]) << 16;
                            }
                        }
                    }
                }
            }
        }
    }

    // Each 64-byte record landed in one physical row of a UInt32 tile's left
    // face while the local query patterns above were being constructed.  The
    // right faces are intentionally untouched.
    if constexpr (PER_HEAD_INPUT) {
        noc_async_read_barrier();
    } else if (owner) {
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_ROW_DESTINATIONS_READY));
        noc_semaphore_wait(ready, PARTS - 1);
        noc_semaphore_set(ready, 0);
        const uint64_t data_dst = get_noc_multicast_addr(
            row_start_x, row_start_y, row_end_x, row_end_y, packed_tiles);
        noc_async_write_multicast_loopback_src(
            packed_tiles, data_dst, CHUNKS * 4096, PARTS, true);
        noc_async_write_barrier();
        const uint64_t valid_dst = get_noc_multicast_addr(
            row_start_x, row_start_y, row_end_x, row_end_y, get_semaphore(SKV4_SEM_ROW_DATA_READY));
        noc_semaphore_set_multicast_loopback_src(
            get_semaphore(SKV4_SEM_ROW_VALID_SOURCE), valid_dst, PARTS, false);
        noc_async_atomic_barrier();
    } else {
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_ROW_DATA_READY)), 0);
        noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_ROW_DESTINATIONS_READY)), 1);
        noc_async_atomic_barrier();
    }
    if constexpr (!PER_HEAD_INPUT) {
        noc_semaphore_wait(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_ROW_DATA_READY)), 1);
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_ROW_DATA_READY)), 0);
    }
    cb_push_back(26, CHUNKS);
    cb_push_back(31, 1);
    if constexpr (PROFILE == 1) return;
    // Stop 2 is retained as an ABI-compatible profiling point.  The removed
    // BFP4 conversion has no replacement stage, so stops 1 and 2 should match.
    if constexpr (PROFILE == 2) return;
    if constexpr (PROFILE <= 3) {
        return;
    }
    if constexpr (PROFILE < 6) {
        return;
    }

    // Stage II is owner-local. Build one query-pattern tile, then gather each
    // 128-byte low-nibble record directly into a UInt32 tile as two aligned
    // 64-byte face-row reads. Helpers have no Stage-II work or traffic.
    if (!owner) {
        return;
    }
    // CB17 is exclusively compute -> writer for the temporary TopK-XL result.
    // The writer publishes the globally merged M16 IDs on CB30, preserving a
    // single producer for that ring.  A semaphore gates the second CB30
    // consumer (reader) after compute has consumed its earlier TopK input.
    volatile tt_l1_ptr uint32_t* stage2_ids_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_IDS_READY));
    noc_semaphore_wait(stage2_ids_ready, 1);
    noc_semaphore_set(stage2_ids_ready, 0);
    cb_wait_front(30, 1);
    volatile tt_l1_ptr uint32_t* ids16 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(30));

    if constexpr (PROFILE == 6) {
        for (uint32_t j = 0; j < M16; ++j) {
            const uint32_t id = ids16[j] < context ? ids16[j] : 0;
            noc_async_read(get_noc_addr(slot * context + id, r4),
                           scratch + (j & 31u) * (D / 2), D / 2);
            if ((j & 31u) == 31u) noc_async_read_barrier();
        }
        noc_async_read_barrier();
        cb_pop_front(30, 1);
        return;
    }

    constexpr uint32_t STAGE2_CHUNKS = M16_PAD / 32;
    for (uint32_t chunk = 0; chunk < STAGE2_CHUNKS; ++chunk) {
        cb_reserve_back(25, 1);
        const uint32_t tile = get_write_ptr(25);
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t local = chunk * 32 + row;
            const uint32_t source = local < M16 ? local : 0;
            const uint32_t id = ids16[source];
            const uint64_t src = get_noc_addr(
                slot * context + id, r4);
            const uint32_t face = row < 16 ? 0 : 2;
            const uint32_t rr = row & 15;
            noc_async_read(src, tile + face * 1024 + rr * 64, 64);
            noc_async_read(src + 64, tile + (face + 1) * 1024 + rr * 64, 64);
        }
        noc_async_read_barrier();
        cb_push_back(25, 1);
        if constexpr (PROFILE == 7) {
            cb_pop_front(25, 1);
        }
    }
    cb_pop_front(30, 1);
    if constexpr (PROFILE == 7) return;
    if constexpr (PROFILE < 10) return;

    // Final sparse-attention input preparation.  One worker owns one query
    // head, so only one diagonal query and one top-100 list are active.
    cb_wait_front(30, 1);
    const uint32_t all_idx = get_read_ptr(30);
    Noc noc;
    experimental::CB q_cb(0), k_cb(2), mask_cb(5), ctrl_cb(18);
    experimental::CB req_cb(21), ack_cb(22);
    constexpr uint32_t H = 32, S = G, KCHUNK = 128;
    constexpr uint32_t q_row_bytes = 2 * D * 2;
    constexpr uint32_t kv_row_bytes = 2 * D;
    const uint32_t kv_page_offset = slot * context;

    q_cb.reserve_back(H);
    for (uint32_t h = 0; h < H; ++h) {
        noc.async_read(aq, q_cb, q_row_bytes,
                       {.page_id = h * S + shard},
                       {.offset_bytes = h * q_row_bytes});
    }
    noc.async_read_barrier();
    q_cb.push_back(H);

    volatile tt_l1_ptr uint32_t* idx =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(all_idx);
    ctrl_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ctrl =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
    ctrl[sparse_sdpa::control_message::ACTIVE_CHUNKS] = 1;
    ctrl[sparse_sdpa::control_message::VALID_KEYS] = 100;
    ctrl_cb.push_back(1);

    k_cb.reserve_back(KCHUNK);
    const uint32_t kdst = k_cb.get_write_ptr();
    req_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* req =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(req_cb.get_write_ptr());
    req[sparse_sdpa::gather_request::BASE] = 0;
    req[sparse_sdpa::gather_request::SPLIT] = 50;
    req[sparse_sdpa::gather_request::IS_LAST] = 1;
    req[sparse_sdpa::gather_request::DST_L1] = kdst;
    req_cb.push_back(1);
    sparse_sdpa::trid_ring_gather<false, 1, 1, 0, 0, sparse_sdpa::K_TRID_RING>(
        noc, kv, kdst, idx, 0, 50, 100, kv_row_bytes, kv_page_offset);
    ack_cb.wait_front(1);
    ack_cb.pop_front(1);
    noc.async_write_zeros(k_cb, (KCHUNK - 100) * kv_row_bytes,
                          {.offset_bytes = 100 * kv_row_bytes});
    noc.write_zeros_l1_barrier();
    k_cb.push_back(KCHUNK);

    mask_cb.reserve_back(1);
    volatile tt_l1_ptr uint16_t* mask =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_cb.get_write_ptr());
    for (uint32_t col = 0; col < 32; ++col) {
        const uint32_t off = col < 16 ? col : 256 + col - 16;
        mask[off] = col >= 4 ? neg_inf_bf16 : uint16_t(0);
    }
    mask_cb.push_back(1);
    cb_pop_front(30, 1);
}
"""
)


STAGE16_WRITER = (
    semaphore_defines(SEM)
    + r"""
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/numeric/bfloat16.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"

constexpr uint32_t one_bf16_packed = 0x3F803F80u;
constexpr uint16_t neg_inf_bf16 = 0xFF80u;
constexpr uint32_t padded_topk = 128;

FORCE_INLINE uint16_t ordered_bf16(uint16_t bits) {
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
}
FORCE_INLINE uint16_t raw_bf16(uint16_t ordered) {
    return (ordered & 0x8000u) ? static_cast<uint16_t>(ordered ^ 0x8000u)
                               : static_cast<uint16_t>(~ordered);
}
FORCE_INLINE uint16_t score_at(uint32_t base, uint32_t candidate, uint32_t row) {
    const uint32_t tile = base + (candidate >> 5) * 2048;
    const uint32_t col = candidate & 31;
    const uint32_t face_row = (row < 16 ? 0 : 1024) + (row & 15) * 32;
    const uint32_t off = face_row + (col < 16 ? col * 2 : 512 + (col - 16) * 2);
    return *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile + off);
}
FORCE_INLINE uint16_t add_bf16(uint16_t a, uint16_t b) {
    return fp32_to_bf16(bf16_to_fp32(a) + bf16_to_fp32(b));
}
FORCE_INLINE uint16_t scale16_bf16(uint16_t bits) {
    const uint16_t exponent = (bits >> 7) & 0xffu;
    if (exponent > 0 && exponent < 251) {
        return static_cast<uint16_t>(bits + (4u << 7));
    }
    return fp32_to_bf16(bf16_to_fp32(bits) * 16.0f);
}
FORCE_INLINE uint32_t decode_topk_xl_k1024(uint32_t word) {
    const uint32_t raw = word & 0xffffu;
    return ((raw >> 6) & 0xfu) + ((raw & 0xfu) << 4) +
           (((raw >> 10) & 1u) << 8) +
           (((raw >> 4) & 1u) << 9) +
           (((raw >> 11) & 1u) << 10);
}
template <uint32_t K>
FORCE_INLINE void heapify(uint16_t* hv, uint32_t* hi) {
    for (int32_t root = K / 2 - 1; root >= 0; --root) {
        uint16_t key = hv[root];
        uint32_t idx = hi[root];
        uint32_t pos = root;
        while (true) {
            uint32_t a = pos * 2 + 1;
            if (a >= K) break;
            uint32_t z = a + 1;
            uint32_t c = (z < K && hv[z] < hv[a]) ? z : a;
            if (hv[c] >= key) break;
            hv[pos] = hv[c]; hi[pos] = hi[c]; pos = c;
        }
        hv[pos] = key; hi[pos] = idx;
    }
}

template <uint32_t K>
FORCE_INLINE void heap_insert(uint16_t key, uint32_t idx,
                              uint16_t* hv, uint32_t* hi) {
    if (key <= hv[0]) return;
    uint32_t pos = 0;
    while (true) {
        uint32_t a = pos * 2 + 1;
        if (a >= K) break;
        uint32_t z = a + 1;
        uint32_t c = (z < K && hv[z] < hv[a]) ? z : a;
        if (hv[c] >= key) break;
        hv[pos] = hv[c]; hi[pos] = hi[c]; pos = c;
    }
    hv[pos] = key; hi[pos] = idx;
}

void kernel_main() {
    constexpr uint32_t N16 = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t M16 = get_compile_time_arg_val(3);
    constexpr uint32_t KLOCAL = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t D = get_compile_time_arg_val(6);
    constexpr bool DEBUG_INDICES = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t PROFILE = get_compile_time_arg_val(8);
    constexpr uint32_t PARTS = 4;
    constexpr uint32_t LOCAL16 = N16 / W;
    constexpr uint32_t LOCAL16_PAD = ((LOCAL16 + 31) / 32) * 32;
    constexpr uint32_t M16_PAD = ((M16 + 31) / 32) * 32;
    constexpr uint32_t INHERITED_OFFSET = 0;
    constexpr uint32_t TOPK_INPUT_OFFSET = 6144;
    constexpr uint32_t STAGE2_QUERY_OFFSET = 8192;
    constexpr uint32_t ROW_WORDS = KLOCAL * 2;
    constexpr uint32_t ROW_BYTES = ROW_WORDS * 4;
    constexpr uint32_t EXCHANGE_SEG_BYTES = ROW_BYTES;
    constexpr uint32_t ID_MASK = (1u << 20) - 1;
    constexpr bool PER_HEAD_INPUT = get_compile_time_arg_val(9) != 0;
    constexpr auto oa = TensorAccessorArgs<10>();
    constexpr auto kva = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto dia = TensorAccessorArgs<kva.next_compile_time_args_offset()>();
    constexpr auto dsa = TensorAccessorArgs<dia.next_compile_time_args_offset()>();
    constexpr auto sqa = TensorAccessorArgs<dsa.next_compile_time_args_offset()>();
    static_assert(N16 % W == 0);
    static_assert(W >= G);
    static_assert(K == 100);
    static_assert(KLOCAL <= LOCAL16);
    static_assert(KLOCAL * W >= M16);

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t context = get_arg_val<uint32_t>(3);
    const uint32_t shard = get_arg_val<uint32_t>(4);
    const uint32_t part = get_arg_val<uint32_t>(6);
    const bool owner = part == 0;
    const uint32_t debug_addr = get_arg_val<uint32_t>(5);
    const uint32_t debug_score_addr = get_arg_val<uint32_t>(21);
    const uint32_t sq_addr = get_arg_val<uint32_t>(22);
    const uint32_t root_x = get_arg_val<uint32_t>(15);
    const uint32_t root_y = get_arg_val<uint32_t>(16);
    const uint32_t row_start_x = get_arg_val<uint32_t>(17);
    const uint32_t row_start_y = get_arg_val<uint32_t>(18);
    const uint32_t row_end_x = get_arg_val<uint32_t>(19);
    const uint32_t row_end_y = get_arg_val<uint32_t>(20);
    const auto out = TensorAccessor(oa, out_addr, D * 2);
    const auto kv = TensorAccessor(kva, kv_addr, 2 * D);
    const auto debug = TensorAccessor(dia, debug_addr, padded_topk * 4);
    const auto debug_scores = TensorAccessor(dsa, debug_score_addr, M16_PAD * 16);
    const auto sq = TensorAccessor(sqa, sq_addr, D * 2);
    Noc noc;

    // BRISC builds this core's compact Stage-1.6 query pattern while NCRISC
    // performs the sparse record gathers. CB2 is idle until final attention,
    // so its first 512 bytes are safe raw scratch. Do not advance its CB
    // pointers: the sparse tail later reserves the complete 128-page ring.
    const uint32_t q16_row = get_write_ptr(2);
    noc_async_read(get_noc_addr(slot * 32 + part, sq), q16_row, D * 2);
    noc_async_read_barrier();

    cb_reserve_back(24, 2);
    volatile tt_l1_ptr uint16_t* q16_src =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(q16_row);
    volatile tt_l1_ptr uint32_t* q16_dst =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(24));
    for (uint32_t query_tile = 0; query_tile < 2; ++query_tile) {
        for (uint32_t face_row = 0; face_row < 2; ++face_row) {
            for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                const uint32_t face = 2 * face_row + face_col;
                for (uint32_t local_plane = 0; local_plane < 4;
                     ++local_plane) {
                    const uint32_t plane =
                        8 * query_tile + 4 * face_col + local_plane;
                    for (uint32_t row = 4 * local_plane;
                         row < 4 * local_plane + 4; ++row) {
                        const uint32_t base =
                            query_tile * 1024 + face * 256 + row * 16;
                        for (uint32_t col = 0; col < 16; ++col) {
                            q16_dst[base + col] =
                                uint32_t(q16_src[16 * col + plane]) << 16;
                        }
                    }
                }
            }
        }
    }
    cb_push_back(24, 2);

    // The same writer RISC supplies the row-reduction operand. The SFPU first
    // plane explicitly zeros the accumulator's unused right faces.
    cb_reserve_back(23, 1);
    volatile tt_l1_ptr uint32_t* ones16 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(23));
    for (uint32_t j = 0; j < 2048 / 4; ++j) ones16[j] = 0x3f803f80u;
    cb_push_back(23, 1);

    cb_wait_front(31, 1);
    const uint32_t shared_scratch = get_read_ptr(31);
    if constexpr (PROFILE == 2 && DEBUG_INDICES) if (shard == 0) {
        // Addressing diagnostic: export exactly what this core's reader put
        // in the inherited-score window, before scoring or scratch reuse.
        noc_async_write(
            shared_scratch + INHERITED_OFFSET,
            get_noc_addr(slot * G + part, debug_scores), LOCAL16 * 2);
        noc_async_write_barrier();
    }
    if constexpr (PROFILE < 3) {
        cb_pop_front(31, 1);
        return;
    }

    if constexpr (PROFILE == 3) {
        cb_wait_front(28, LOCAL16_PAD / 32);
        cb_pop_front(28, LOCAL16_PAD / 32);
        cb_pop_front(31, 1);
        return;
    }

    // Candidate IDs are copied once.  Scores never move between cores: each
    // core owns one query head and consumes its own SFPU-produced stream.
    uint32_t candidate_page = 0;
    if constexpr (PER_HEAD_INPUT) {
        cb_wait_front(27, 1);
        candidate_page = get_read_ptr(27);
    } else if (owner) {
        cb_wait_front(27, 1);
        candidate_page = get_read_ptr(27);
        volatile tt_l1_ptr uint32_t* selection_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_SELECTION_DESTINATIONS_READY));
        noc_semaphore_wait(selection_ready, PARTS - 1);
        noc_semaphore_set(selection_ready, 0);
        for (uint32_t peer = 1; peer < PARTS; ++peer) {
            const uint32_t x = row_start_x + peer;
            noc_async_write(
                candidate_page, get_noc_addr(x, row_start_y, candidate_page),
                LOCAL16 * 4);
        }
        noc_async_write_barrier();
        for (uint32_t peer = 1; peer < PARTS; ++peer) {
            const uint32_t x = row_start_x + peer;
            noc_semaphore_inc(get_noc_addr(x, row_start_y, get_semaphore(SKV4_SEM_SELECTION_DATA_READY)), 1);
        }
        noc.async_atomic_barrier();
    } else {
        // In the group-shared path this is a fixed-address NoC receive
        // window. Do not reserve a page that no producer will ever push.
        candidate_page = get_write_ptr(27);
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_SELECTION_DATA_READY)), 0);
        noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_SELECTION_DESTINATIONS_READY)), 1);
        noc.async_atomic_barrier();
    }
    if constexpr (!PER_HEAD_INPUT) if (!owner) {
        noc_semaphore_wait(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_SELECTION_DATA_READY)), 1);
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_SELECTION_DATA_READY)), 0);
    }

    const uint32_t cand_prefix = (shard * LOCAL16 * 4) & 63;
    volatile tt_l1_ptr uint32_t* candidates =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            candidate_page + cand_prefix);

    uint32_t receive = 0;
    if (owner) {
        cb_reserve_back(29, W);
        receive = get_write_ptr(29);
    }
    // Inherited scores have been consumed by the time the frontier is ready,
    // so the same scratch page becomes the short all-to-all packet.
    const uint32_t local_packet = shared_scratch;
    volatile tt_l1_ptr uint32_t* packet =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_packet);
    volatile tt_l1_ptr uint32_t* local_row = packet + part * ROW_WORDS;
    volatile tt_l1_ptr uint16_t* inherited =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            shared_scratch + INHERITED_OFFSET);
    volatile tt_l1_ptr uint16_t* topk_scores =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            shared_scratch + TOPK_INPUT_OFFSET);
    constexpr uint32_t SCORE_CHUNKS = LOCAL16_PAD / 32;
    for (uint32_t chunk = 0; chunk < SCORE_CHUNKS; ++chunk) {
        cb_wait_front(28, 1);
        const uint32_t epoch_scores = get_read_ptr(28);
        const uint32_t begin = chunk * 32;
        const uint32_t end = begin + 32 < LOCAL16 ? begin + 32 : LOCAL16;
        for (uint32_t j = begin; j < end; ++j) {
            const uint16_t residual = scale16_bf16(
                score_at(epoch_scores, part, j - begin));
            const uint16_t total = add_bf16(residual, inherited[j]);
            topk_scores[j] = total;
        }
        cb_pop_front(28, 1);
    }

    if constexpr (PER_HEAD_INPUT) cb_pop_front(27, 1);

    // Present the 928 scores as one row-major BF16 K=1024 page.  This copy is
    // deliberately local and contiguous; the expensive O(928 log 160) scalar
    // heap has been replaced by one SFPU TopK-XL sort in the compute thread.
    // Padding is -inf and can never survive.
    cb_reserve_back(30, 1);
    volatile tt_l1_ptr uint16_t* topk_pages =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(30));
    const uint32_t first_topk = LOCAL16_PAD < 1024 ? LOCAL16_PAD : 1024;
    for (uint32_t j = 0; j < first_topk; ++j) {
        topk_pages[j] = j < LOCAL16 ? topk_scores[j] : neg_inf_bf16;
    }
    cb_push_back(30, 1);
    if constexpr (LOCAL16_PAD > 1024) {
        cb_reserve_back(4, 1);
        volatile tt_l1_ptr uint16_t* second_topk =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(4));
        for (uint32_t j = 1024; j < LOCAL16_PAD; ++j) {
            second_topk[j - 1024] =
                j < LOCAL16 ? topk_scores[j] : neg_inf_bf16;
        }
        cb_push_back(4, 1);
    }

    cb_wait_front(17, 1);
    volatile tt_l1_ptr uint32_t* local_indices =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(17));
    for (uint32_t j = 0; j < KLOCAL; ++j) {
        // K=1024 TopK rank j is stored in native-tile lane
        // (j / 64) + 16 * (j % 64).  This is the inverse of the removed
        // transpose+untilize and costs two shifts plus one add.
        const uint32_t lane = (j >> 6) + ((j & 63u) << 4);
        const uint32_t local = decode_topk_xl_k1024(local_indices[lane]);
        local_row[2 * j] = ordered_bf16(topk_scores[local]);
        local_row[2 * j + 1] =
            ((shard * LOCAL16 + local) << 20) | candidates[local];
    }
    cb_pop_front(17, 1);

    if (!owner) {
        noc_async_write(
            reinterpret_cast<uint32_t>(local_row),
            get_noc_addr(root_x, root_y, local_packet + part * ROW_BYTES),
            ROW_BYTES);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SKV4_SEM_SELECTION_DONE)), 1);
        noc.async_atomic_barrier();
    } else {
        volatile tt_l1_ptr uint32_t* selection_done =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_SELECTION_DONE));
        noc_semaphore_wait(selection_done, PARTS - 1);
        noc_semaphore_set(selection_done, 0);
        if constexpr (!PER_HEAD_INPUT) cb_pop_front(27, 1);
    }
    if constexpr (PROFILE == 4) {
        cb_pop_front(31, 1);
        return;
    }

    uint16_t hv16[M16];
    uint32_t hi16[M16];
    if (owner) {
        // First all-to-all: destination h receives head h's local frontier
        // from every candidate shard, then merges those 4xKLOCAL entries.
        {
            volatile tt_l1_ptr uint32_t* own_src = packet + shard * ROW_WORDS;
            volatile tt_l1_ptr uint32_t* own_dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    receive + shard * EXCHANGE_SEG_BYTES);
            for (uint32_t j = 0; j < ROW_WORDS; ++j) own_dst[j] = own_src[j];
        }
        for (uint32_t dest = 0; dest < G; ++dest) {
            if (dest == shard) continue;
            const uint32_t x = get_arg_val<uint32_t>(7 + 2 * dest);
            const uint32_t y = get_arg_val<uint32_t>(8 + 2 * dest);
            noc_async_write(
                reinterpret_cast<uint32_t>(packet + dest * ROW_WORDS),
                get_noc_addr(x, y, receive + shard * EXCHANGE_SEG_BYTES),
                ROW_BYTES);
        }
        noc_async_write_barrier();
        for (uint32_t dest = 0; dest < G; ++dest) {
            if (dest == shard) continue;
            const uint32_t x = get_arg_val<uint32_t>(7 + 2 * dest);
            const uint32_t y = get_arg_val<uint32_t>(8 + 2 * dest);
            noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(SKV4_SEM_FRONTIERS_READY)), 1);
        }
        noc.async_atomic_barrier();
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_FRONTIERS_READY));
        noc_semaphore_wait_min(ready, W - 1);
        noc_semaphore_set(ready, 0);

        for (uint32_t j = 0; j < M16; ++j) {
            const uint32_t worker = j / KLOCAL;
            const uint32_t item = j - worker * KLOCAL;
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    receive + worker * EXCHANGE_SEG_BYTES);
            hv16[j] = static_cast<uint16_t>(src[2 * item]);
            hi16[j] = src[2 * item + 1];
        }
        heapify<M16>(hv16, hi16);
        for (uint32_t flat = M16; flat < W * KLOCAL; ++flat) {
            const uint32_t worker = flat / KLOCAL;
            const uint32_t item = flat - worker * KLOCAL;
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    receive + worker * EXCHANGE_SEG_BYTES);
            heap_insert<M16>(static_cast<uint16_t>(src[2 * item]),
                             src[2 * item + 1], hv16, hi16);
        }
        if constexpr (PROFILE == 5) {
            volatile tt_l1_ptr uint32_t* probe =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive);
            for (uint32_t j = 0; j < M16; ++j) probe[j] = hi16[j] & ID_MASK;
            noc_async_write(
                receive, get_noc_addr(slot * G + shard, debug_scores), M16 * 4);
            noc_async_write_barrier();
            cb_pop_front(31, 1);
            return;
        }

        cb_reserve_back(30, 1);
        volatile tt_l1_ptr uint32_t* ids16 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(30));
        for (uint32_t j = 0; j < M16; ++j) ids16[j] = hi16[j] & ID_MASK;
        cb_push_back(30, 1);
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_IDS_READY)), 1);

        // Reuse the Stage-1.6 exchange buffers as Stage-II-only compute CBs.
        // This leaves sparse-SDPA's CB14-16 pointer state untouched.
        cb_reserve_back(24, 2);
        volatile tt_l1_ptr uint32_t* qdst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(24));
        volatile tt_l1_ptr uint32_t* qsrc =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                shared_scratch + STAGE2_QUERY_OFFSET);
        for (uint32_t j = 0; j < 2 * 4096 / 4; ++j) qdst[j] = qsrc[j];
        cb_push_back(24, 2);

        volatile tt_l1_ptr uint32_t* one =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive);
        for (uint32_t j = 0; j < 512; ++j) one[j] = 0x3f803f80u;
        cb_push_back(29, W);
    }

    if constexpr (PROFILE <= 6) {
        cb_pop_front(31, 1);
        return;
    }

    // Stage II is owner-local in the packed-SFPU path. Helpers stop after
    // contributing their Stage-1.6 frontier.
    if (!owner || PROFILE == 7) {
        cb_pop_front(31, 1);
        return;
    }

    uint16_t hv[K];
    uint32_t hi[K];
    uint32_t stream_seen = 0;

    constexpr uint32_t STAGE2_CHUNKS = M16_PAD / 32;
    if constexpr (PROFILE == 8) {
        cb_wait_front(28, STAGE2_CHUNKS);
        cb_pop_front(28, STAGE2_CHUNKS);
        cb_pop_front(31, 1);
        return;
    }

    // Complete score tiles stream directly from the packed SFPU scorer into
    // the exact running top-100 heap.
    for (uint32_t chunk = 0; chunk < STAGE2_CHUNKS; ++chunk) {
        cb_wait_front(28, 1);
        const uint32_t epoch_scores = get_read_ptr(28);
        const uint32_t begin = chunk * 32;
        const uint32_t end = begin + 32 < M16 ? begin + 32 : M16;
        for (uint32_t j = begin; j < end; ++j) {
            const uint16_t residual =
                score_at(epoch_scores, shard, j - begin);
            const uint16_t inherited_score = raw_bf16(hv16[j]);
            const uint16_t total = add_bf16(residual, inherited_score);
            const uint16_t key = ordered_bf16(total);
            const uint32_t id = hi16[j] & ID_MASK;
            if constexpr (DEBUG_INDICES) {
                volatile tt_l1_ptr uint32_t* trace =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(shared_scratch);
                trace[4 * j] = id;
                trace[4 * j + 1] = residual;
                trace[4 * j + 2] = inherited_score;
                trace[4 * j + 3] = total;
            }
            if (stream_seen < K) {
                hv[stream_seen] = key;
                hi[stream_seen] = id;
                ++stream_seen;
                if (stream_seen == K) heapify<K>(hv, hi);
            } else {
                heap_insert<K>(key, id, hv, hi);
            }
        }
        cb_pop_front(28, 1);
    }

    if constexpr (DEBUG_INDICES) {
        volatile tt_l1_ptr uint32_t* trace =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(shared_scratch);
        volatile tt_l1_ptr uint32_t* qtrace =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                shared_scratch + STAGE2_QUERY_OFFSET);
        for (uint32_t j = 0; j < 32; ++j) trace[4 * M16 + j] = qtrace[j];
        noc_async_write(
            shared_scratch,
            get_noc_addr(slot * G + shard, debug_scores), M16_PAD * 16);
        noc_async_write_barrier();
    }

    if constexpr (PROFILE == 9) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive)[0] = hi[0];
        cb_pop_front(31, 1);
        return;
    }

    // CB30 remains writer-owned across all three sequential payloads: local
    // TopK input, global top-400 IDs, then final top-100 IDs. CB17 stays
    // compute->writer only for the TopK-XL result.
    cb_reserve_back(30, 1);
    volatile tt_l1_ptr uint32_t* final_ids =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(30));
    for (uint32_t j = 0; j < padded_topk; ++j) {
        final_ids[j] = j < K ? hi[j] : 0xffffffffu;
    }
    if constexpr (DEBUG_INDICES) {
        noc_async_write(reinterpret_cast<uint32_t>(final_ids),
                        get_noc_addr(slot * G + shard, debug), padded_topk * 4);
        noc_async_write_barrier();
    }
    cb_push_back(30, 1);

    // CB4 was the optional second local-TopK input.  That page has already
    // been consumed, so the owner can now reuse the same ring for sparse
    // attention's -inf column scalar.
    if constexpr (PROFILE >= 10) if (owner) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            6, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW, 1>();
        generate_bcast_col_scalar(CircularBuffer(19), one_bf16_packed);
        cb_reserve_back(4, 1);
        volatile tt_l1_ptr uint16_t* ni =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(4));
        for (uint32_t c = 0; c < 16; ++c) {
            ni[c] = neg_inf_bf16;
            ni[256 + c] = neg_inf_bf16;
        }
        cb_push_back(4, 1);
    }

    // Writer half of one sparse-attention tail.
    experimental::CB req_cb(21), ack_cb(22), idx_cb(30), out_cb(16);
    const uint32_t all_idx = idx_cb.get_read_ptr();
    constexpr uint32_t H = 32, S = G, block_tiles = D / 32;
    const uint32_t kv_page_offset = slot * context;
    req_cb.wait_front(1);
    volatile tt_l1_ptr uint32_t* req =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(req_cb.get_read_ptr());
    const uint32_t split = req[sparse_sdpa::gather_request::SPLIT];
    const uint32_t dst_l1 = req[sparse_sdpa::gather_request::DST_L1];
    req_cb.pop_front(1);
    volatile tt_l1_ptr uint32_t* idx =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(all_idx);
    sparse_sdpa::trid_ring_gather<false, 1, 1, 0, 0, sparse_sdpa::K_TRID_RING>(
        noc, kv, dst_l1, idx, 0, 0, split, 2 * D, kv_page_offset);
    ack_cb.reserve_back(1);
    ack_cb.push_back(1);

    out_cb.wait_front(block_tiles);
    for (uint32_t h = 0; h < H; ++h) {
        noc.async_write(out_cb, out, D * 2,
                        {.offset_bytes = h * D * 2},
                        {.page_id = h * S + shard});
    }
    noc.async_write_barrier();
    out_cb.pop_front(block_tiles);
    cb_pop_front(31, 1);
}
"""
)


def _stage16_groups(ttnn, device, slots, core_points=None):
    """Find non-overlapping physical 4x4 blocks, one per KV-head slot."""
    if core_points is not None:
        points = list(core_points)
        need = slots * 16
        if len(points) < need:
            raise ValueError(f"need {need} worker cores, got {len(points)}")
        return [points[i * 16 : (i + 1) * 16] for i in range(slots)]

    grid = device.compute_with_storage_grid_size()
    reverse = {}
    for ly in range(grid.y):
        for lx in range(grid.x):
            p = device.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
            reverse[(p.x, p.y)] = (lx, ly)
    xs = [p[0] for p in reverse]
    ys = [p[1] for p in reverse]
    used = set()
    groups = []
    for y0 in range(min(ys), max(ys) - 2):
        for x0 in range(min(xs), max(xs) - 2):
            physical = [(x0 + x, y0 + y) for y in range(4) for x in range(4)]
            if any(p in used or p not in reverse for p in physical):
                continue
            groups.append([reverse[p] for p in physical])
            used.update(physical)
            if len(groups) == slots:
                return groups
    raise ValueError(
        f"need {slots} non-overlapping physical 4x4 worker blocks; "
        f"this device supplies only {len(groups)}"
    )


def build_full_stage16(
    ttnn,
    device,
    q,
    residual2,
    residual4,
    indices15,
    scores2,
    q_attention,
    kv,
    *,
    slots,
    context,
    n16,
    m16=400,
    local_m=160,
    workers=4,
    group=GQA_GROUP,
    topk=TOPK,
    core_points=None,
    debug_indices=False,
    profile_stop=10,
    sfpu_planes=8,
    per_head_input=False,
    union_inputs=None,
    union_profile_phase="full",
    _union_split_role=None,
    _union_handoff=None,
    _union_middle=None,
    attention_zones=None,
):
    """Build sharded Stage 1.6 / Stage II / attention program chain."""
    if group != GQA_GROUP or workers != 4:
        raise ValueError("spatial Stage 1.6 requires G=4 and four candidate shards")
    if profile_stop not in range(1, 11):
        raise ValueError("profile_stop must be in [1,10]")
    if sfpu_planes not in range(1, 9):
        raise ValueError("sfpu_planes must be in [1,8]")
    if n16 % workers or n16 // workers < local_m:
        raise ValueError("require N16%W=0 and each shard to contain at least local_m")
    if m16 < topk or topk != TOPK:
        raise ValueError("require M16>=100 and topk=100")
    if local_m * workers < m16:
        raise ValueError("local Stage-1.6 frontiers do not cover global M16")
    union = union_inputs is not None
    if _union_split_role in ("stage16", "stage2") and _union_middle is None:
        raise ValueError(
            "split union stages require an allocated persistent middle-bit L1 cache"
        )
    if union_profile_phase != "full" and (not union or profile_stop != 8):
        raise ValueError(
            "partial union profiling requires union input and profile_stop=8"
        )
    if union and (per_head_input or sfpu_planes != 8 or local_m < topk):
        raise ValueError(
            "matrix union requires shared input, all planes, local_m>=topk"
        )
    if union and m16 % 16:
        raise ValueError("union frontier packets require M16 divisible by 16")
    if union and local_m > 256:
        raise ValueError("union exchange scratch requires local_m<=256")
    id_bits = max(1, (context - 1).bit_length())
    position_capacity = 1 << (32 - id_bits)
    if n16 > position_capacity or context > (1 << 20):
        raise ValueError(
            f"Stage-1.6 32-bit tags require N16<={position_capacity} at "
            f"context={context}, and context<=1M"
        )
    local16 = n16 // workers
    local16_pad = ((local16 + 31) // 32) * 32
    prefix_page_keys = min(local16, 1024)
    prefix_pages = (local16 + prefix_page_keys - 1) // prefix_page_keys
    m16_pad = ((m16 + 31) // 32) * 32
    if union and profile_stop >= 6 and _union_split_role is None:
        from seoulkv4._kernels.stage16.seoulkv4_union_chain import build_chain

        return build_chain(
            build_full_stage16,
            ttnn,
            device,
            (q, residual2, residual4, indices15, scores2, q_attention, kv),
            dict(
                slots=slots,
                context=context,
                n16=n16,
                m16=m16,
                local_m=local_m,
                workers=workers,
                group=group,
                topk=topk,
                core_points=core_points,
                debug_indices=debug_indices,
                profile_stop=profile_stop,
                sfpu_planes=sfpu_planes,
                per_head_input=per_head_input,
                union_inputs=union_inputs,
                union_profile_phase=union_profile_phase,
                attention_zones=attention_zones,
            ),
        )
    groups = _stage16_groups(ttnn, device, slots, core_points)
    points = [point for members in groups for point in members]
    cores = _ranges(ttnn, points)

    outputs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, ROWS, group, D)),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(slots)
    ]
    # One page per (slot, query-head).  Keeping this as one tensor makes the
    # page ID unambiguous and avoids concurrent heads racing on one slot page.
    debug_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, slots, group, PADDED_TOPK)),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    debug_scores = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, slots, group, m16_pad * 4)),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    from seoulkv4._kernels.common.refinement_abi import (
        REFINEMENT_COMPUTE_RT,
        REFINEMENT_EXTENSION_CT,
        SPARSE_SDPA_CT,
        refinement_abi,
    )

    reader_tensors = dict(
        sq=q,
        residual2=residual2,
        residual4=residual4,
        candidates=indices15,
        scores2=scores2,
        attention_query=q_attention[0],
        kv=kv,
    )
    writer_tensors = dict(
        output=outputs[0],
        kv=kv,
        debug_indices=debug_output,
        debug_scores=debug_scores,
        sq=q,
    )
    dynamic_inputs = (
        union
        and union_inputs.get("input_counts") is not None
        and _union_split_role not in ("stage2", "attention")
    )
    if union:
        reader_tensors["prefix2"] = union_inputs["prefix2"]
        writer_tensors["union_query"] = union_inputs["query"]
    if dynamic_inputs:
        reader_tensors["input_counts"] = union_inputs["input_counts"]
    if _union_handoff is not None:
        reader_tensors["handoff"] = writer_tensors["handoff"] = _union_handoff
    if _union_middle is not None:
        reader_tensors["middle"] = _union_middle
    reader_rt_abi, reader_ct_abi = refinement_abi(
        "reader",
        union=union,
        dynamic=dynamic_inputs,
        handoff=_union_handoff is not None,
        middle=_union_middle is not None,
    )
    writer_rt_abi, writer_ct_abi = refinement_abi(
        "writer", union=union, handoff=_union_handoff is not None
    )
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for slot, members in enumerate(groups):
        physical = [
            device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            for x, y in members
        ]
        roots = [physical[shard * 4] for shard in range(4)]
        noc_coords = [c for point in roots for c in (point.x, point.y)]
        for flat, (x, y) in enumerate(members):
            shard, part = divmod(flat, 4)
            root = roots[shard]
            row = physical[shard * 4 : (shard + 1) * 4]
            reader_fields = {
                name: t.buffer_address() for name, t in reader_tensors.items()
            }
            reader_fields["attention_query"] = q_attention[slot].buffer_address()
            writer_fields = {
                name: t.buffer_address() for name, t in writer_tensors.items()
            }
            writer_fields.update(
                output=outputs[slot].buffer_address(), head_peers=noc_coords
            )
            topology = dict(
                slot=slot,
                context=context,
                shard=shard,
                part=part,
                root_x=root.x,
                root_y=root.y,
                row_start_x=row[0].x,
                row_start_y=row[0].y,
                row_end_x=row[-1].x,
                row_end_y=row[-1].y,
            )
            if union:
                reader_fields.update(my_x=physical[flat].x, my_y=physical[flat].y)
            rr[x][y] = reader_rt_abi.pack(**reader_fields, **topology)
            wr[x][y] = writer_rt_abi.pack(**writer_fields, **topology)
            cr[x][y] = REFINEMENT_COMPUTE_RT.pack(
                shard=shard, tok_count=1, upstream_reserved=1, part=part
            )

    compute_source, reader_source, writer_source = (
        STAGE16_COMPUTE,
        STAGE16_READER,
        STAGE16_WRITER,
    )
    if union:
        from seoulkv4._kernels.stage16.seoulkv4_union_matrix import sources

        native_profile = os.environ.get("SKV4_NATIVE_MARKERS") == "1"
        if native_profile and _union_split_role not in (
            "stage16",
            "stage2",
            "attention",
        ):
            raise ValueError("native markers require a split union program")
        only_role = os.environ.get("SKV4_NATIVE_PROFILE_ROLE")
        compute_source, reader_source, writer_source = sources(
            profile_phase=union_profile_phase,
            split_role=_union_split_role,
            dynamic_input_counts=dynamic_inputs,
            profile=native_profile
            and (not only_role or only_role == _union_split_role),
            id_bits=id_bits,
            prefix_page_keys=prefix_page_keys,
            prefix_pages=prefix_pages,
            retain_middle=_union_middle is not None,
            debug=debug_indices,
        )
    elif os.environ.get("SKV4_NATIVE_MARKERS") == "1":
        raise ValueError("native markers require a split union program")

    reader = ttnn.KernelDescriptor(
        kernel_source=reader_source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=reader_ct_abi.pack(
            ttnn,
            tensors=reader_tensors,
            n16=n16,
            workers=workers,
            group=group,
            m16=m16,
            d=D,
            profile=profile_stop,
            per_head_input=int(per_head_input),
        ),
        runtime_args=rr,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=writer_source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=writer_ct_abi.pack(
            ttnn,
            tensors=writer_tensors,
            n16=n16,
            workers=workers,
            group=group,
            m16=m16,
            local_m=local_m,
            topk=topk,
            d=D,
            debug=int(debug_indices),
            profile=profile_stop,
            per_head_input=int(per_head_input),
        ),
        runtime_args=wr,
        config=ttnn.WriterConfigDescriptor(),
    )

    scale_bits = struct.unpack("<I", struct.pack("<f", D**-0.5))[0]
    compute_ct = SPARSE_SDPA_CT.pack(
        h=ROWS,
        dht=2 * D // 32,
        v_dht=D // 32,
        skt=PADDED_TOPK // 32,
        scale=scale_bits,
        cb_q_rm=CB.ATTENTION_QUERY_ROW,
        cb_q_in=CB.DECODED_KEYS,
        cb_k_rm=CB.SPARSE_KV_RECORDS,
        cb_k_in=CB.SPARSE_KV_TILES,
        cb_neginf=CB.ATTENTION_NEG_INF,
        cb_mask_part=CB.ATTENTION_MASK,
        cb_scale=CB.ATTENTION_SCALE,
        cb_qk_im=CB.ATTENTION_QK,
        cb_max_a=CB.ATTENTION_MAX_A,
        cb_max_b=CB.ATTENTION_MAX_B,
        cb_sum_a=CB.ATTENTION_SUM_A,
        cb_sum_b=CB.ATTENTION_SUM_B,
        cb_out_a=CB.ATTENTION_OUT_A,
        cb_out_b=CB.ATTENTION_OUT_B,
        cb_corr=CB.ATTENTION_CORRECTION,
        cb_out_im=CB.ATTENTION_OUT_INTERMEDIATE,
        cb_out_rm=CB.ATTENTION_OUT_ROW,
        cb_ctrl=CB.LIVE_CONTROL,
        cb_col_identity=CB.ATTENTION_COLUMN_IDENTITY,
        cb_recip_scratch=CB.ATTENTION_RECIPROCAL,
        scaled_kv=0,
        cb_k_rope_rm=CB.MATRIX_QUERY,
        cb_k_scale_bcast=CB.DECODE_SCRATCH,
        cb_k_latent_tile=CB.PACKED_CANDIDATES,
        cb_k_rope_tile=CB.PACKED_MIDDLE,
        math_approx_mode=0,
        query_subblock=1,
        packed_row_bytes=0,
    ) + REFINEMENT_EXTENSION_CT.pack(
        local16_pad=local16_pad,
        m16_pad=m16_pad,
        profile=profile_stop,
        stage2_planes=sfpu_planes,
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=compute_source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=cr,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
        ),
    )

    tile16 = 2048
    tile8 = ttnn.tile_size(ttnn.bfloat8_b)
    stage16_program = union and _union_split_role == "stage16"
    stage16_quarter_chunks = (local16_pad // 32 + 3) // 4
    # Split Stage II does NOT retain a full Stage1.6 packed-tile ring in CB26.
    # The frontier-stream reader uses it only for four aligned frontiers and
    # one source-ID span. Upper/middle bits reside in allocated L1 tensors.
    # The monolithic configuration is unchanged. Attention's unused decoder
    # CB24 is reduced below too; its clean split source never accesses it.
    stage2_meta_pages = local16_pad // 32
    if union and _union_split_role == "stage2":
        stage2_meta_bytes = (
            group * (((m16 + 1) * 4 + 63) // 64 * 64) + ((local16 * 4 + 126) // 64) * 64
        )
        stage2_meta_pages = (stage2_meta_bytes + 4095) // 4096

    def cb(index, dtype, page_size, pages):
        return ttnn.CBDescriptor(
            total_size=page_size * pages,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index, data_format=dtype, page_size=page_size
                )
            ],
        )

    scratch_bytes = 4 * 4096
    if stage16_program:
        # Program A needs only the four top-M16 lists and their deduplicated
        # candidate positions. Middle bits live in distributed persistent L1
        # pages, not this root's scratch; upper bits reuse Stage-1.5 storage.
        scratch_bytes += group * m16 * 4 + local16 * 4
    elif union:
        scratch_bytes += local16 * 64 + group * m16 * 4 + local16 * 4 + group * topk * 6
        scratch_bytes = ((scratch_bytes + 63) // 64) * 64
    row_bytes = local_m * 2 * 4
    cbs = [
        # Sparse-attention tail.
        cb(
            CB.ATTENTION_QUERY_ROW,
            ttnn.bfloat16,
            2 * D * 2,
            1 if stage16_program else ROWS,
        ),
        cb(CB.DECODED_KEYS, ttnn.bfloat16, tile16, 1 if stage16_program else 16),
        cb(
            CB.SPARSE_KV_RECORDS,
            ttnn.fp8_e4m3,
            2 * D,
            (
                max(1, (4 * stage16_quarter_chunks * 64 + 2 * D - 1) // (2 * D))
                if stage16_program
                else PADDED_TOPK
            ),
        ),
        cb(
            CB.SPARSE_KV_TILES,
            ttnn.bfloat8_b,
            tile8,
            1 if stage16_program else (PADDED_TOPK // 32) * (2 * D // 32),
        ),
        cb(CB.ATTENTION_NEG_INF, ttnn.bfloat16, tile16, 2),
        cb(CB.ATTENTION_MASK, ttnn.bfloat16, tile16, 1 if stage16_program else 2),
        cb(CB.ATTENTION_SCALE, ttnn.bfloat16, tile16, 1),
        cb(
            CB.ATTENTION_QK,
            ttnn.bfloat16,
            tile16,
            1 if stage16_program else PADDED_TOPK // 32,
        ),
        cb(CB.ATTENTION_MAX_A, ttnn.bfloat16, tile16, 1),
        cb(CB.ATTENTION_MAX_B, ttnn.bfloat16, tile16, 1),
        cb(CB.ATTENTION_SUM_A, ttnn.bfloat16, tile16, 1),
        cb(CB.ATTENTION_SUM_B, ttnn.bfloat16, tile16, 1),
        cb(
            CB.ATTENTION_OUT_A, ttnn.bfloat16, tile16, 1 if stage16_program else D // 32
        ),
        cb(
            CB.ATTENTION_OUT_B, ttnn.bfloat16, tile16, 1 if stage16_program else D // 32
        ),
        cb(CB.ATTENTION_CORRECTION, ttnn.bfloat16, tile16, 1),
        cb(
            CB.ATTENTION_OUT_INTERMEDIATE,
            ttnn.bfloat16,
            tile16,
            1 if stage16_program else D // 32,
        ),
        cb(
            CB.ATTENTION_OUT_ROW,
            ttnn.bfloat16,
            tile16,
            1 if stage16_program else D // 32,
        ),
        # Compute -> writer only. Stage II emits one or two native K512
        # frontiers; the shared Stage1.6 helper emits K1024 frontiers.
        cb(
            CB.LOCAL_FRONTIERS,
            ttnn.uint32 if _union_split_role == "stage2" else ttnn.float32,
            4096,
            2 if (_union_split_role == "stage2" or local16_pad > 1024) else 1,
        ),
        cb(CB.LIVE_CONTROL, ttnn.bfloat16, 16, 2),
        cb(CB.ATTENTION_COLUMN_IDENTITY, ttnn.bfloat16, tile16, 1),
        cb(CB.ATTENTION_RECIPROCAL, ttnn.bfloat16, tile16, 1),
        cb(CB.QUERY_COEFFICIENTS, ttnn.bfloat16, 32, 2),
        (
            cb(CB.TOPK_MERGE_SCRATCH, ttnn.uint32, 4096, 4)
            if union and _union_split_role != "stage2"
            else cb(CB.TOPK_MERGE_SCRATCH, ttnn.bfloat16, 16, 2)
        ),
        # Union A: native four-head query + expanded BFP4 quarter. Union B
        # and the per-head control retain their original CB descriptors.
        cb(
            CB.MATRIX_QUERY,
            ttnn.bfloat16,
            tile16,
            8 if stage16_program else (16 if union else 1),
        ),
        (
            cb(
                CB.DECODE_SCRATCH,
                ttnn.bfloat16,
                tile16,
                1 if _union_split_role == "attention" else stage16_quarter_chunks * 8,
            )
            if union and _union_split_role != "stage2"
            else cb(CB.DECODE_SCRATCH, ttnn.float32, 4096, 2)
        ),
        # Stage II: two packed UInt32 candidate tiles are enough to overlap
        # sparse GDDR gathers with SFPU scoring; no expanded BFP8 cache.
        cb(
            CB.PACKED_CANDIDATES,
            ttnn.uint32,
            4096,
            (
                (
                    ((local16_pad // 32 + 3) // 4) * 3
                    if union and _union_split_role == "stage2"
                    else (1 if stage16_program else 2)
                )
                if union
                else 2
            ),
        ),
        # Stage 1.6: tile-ready 2-bit records, left faces only.
        cb(
            CB.PACKED_MIDDLE,
            ttnn.uint32,
            4096,
            stage16_quarter_chunks if stage16_program else stage2_meta_pages,
        ),
        cb(
            CB.SCORE_SCRATCH,
            ttnn.float32,
            (
                max(4096, ((local16 + 1) * 4 + 63) // 64 * 64)
                if _union_split_role
                else 4096
            ),
            1,
        ),
        cb(
            CB.SCORE_TILES,
            ttnn.bfloat16,
            tile16,
            (
                stage16_quarter_chunks
                if stage16_program
                else max(local16_pad, m16_pad) // 32
            ),
        ),
        # Stage-1.6 exchange scratch, then persistent Stage-II reduction tile.
        cb(CB.EXCHANGE_TILES, ttnn.bfloat16, tile16, workers),
        # Writer-owned sequential channel: BF16 TopK input, M16 IDs, final IDs.
        cb(
            CB.SELECTOR_CHANNEL,
            ttnn.bfloat16,
            max(4096, ((local16 + 1) * 4 + 63) // 64 * 64) if union else 2048,
            1,
        ),
        cb(CB.SHARED_SCRATCH, ttnn.uint8, scratch_bytes, 1),
    ]
    ready = ttnn.SemaphoreDescriptor(
        id=SEM.FRONTIERS_READY, core_ranges=cores, initial_value=0
    )
    row_ready = ttnn.SemaphoreDescriptor(
        id=SEM.ROW_DESTINATIONS_READY, core_ranges=cores, initial_value=0
    )
    row_received = ttnn.SemaphoreDescriptor(
        id=SEM.ROW_DATA_READY, core_ranges=cores, initial_value=0
    )
    row_valid = ttnn.SemaphoreDescriptor(
        id=SEM.ROW_VALID_SOURCE, core_ranges=cores, initial_value=1
    )
    selection_ready = ttnn.SemaphoreDescriptor(
        id=SEM.SELECTION_DESTINATIONS_READY, core_ranges=cores, initial_value=0
    )
    selection_received = ttnn.SemaphoreDescriptor(
        id=SEM.SELECTION_DATA_READY, core_ranges=cores, initial_value=0
    )
    selection_done = ttnn.SemaphoreDescriptor(
        id=SEM.SELECTION_DONE, core_ranges=cores, initial_value=0
    )
    stage2_ids_ready = ttnn.SemaphoreDescriptor(
        id=SEM.STAGE2_IDS_READY, core_ranges=cores, initial_value=0
    )
    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader, compute, writer],
        semaphores=[
            ready,
            row_ready,
            row_received,
            row_valid,
            selection_ready,
            selection_received,
            selection_done,
            stage2_ids_ready,
            *(
                [
                    ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0)
                    for i in (
                        (
                            SEM.PEER_EXCHANGE_READY,
                            SEM.STAGE2_FRONTIERS_READY,
                            SEM.SCORES_READY,
                        )
                        if _union_split_role == "stage2"
                        else (
                            SEM.PEER_EXCHANGE_READY,
                            SEM.STAGE2_FRONTIERS_READY,
                            SEM.SCORES_READY,
                            SEM.RESERVED_14,
                            SEM.PACKED_FRONTIERS_READY,
                        )
                    )
                ]
                if union
                else []
            ),
        ],
        cbs=cbs,
    )
    io = [
        q,
        residual2,
        residual4,
        indices15,
        scores2,
        *q_attention,
        kv,
        *outputs,
        debug_output,
        debug_scores,
    ]
    if union:
        io += [union_inputs["prefix2"], union_inputs["query"]]
    if _union_handoff is not None:
        io.append(_union_handoff)
    if _union_middle is not None:
        io.append(_union_middle)

    def run():
        ttnn.generic_op(io, descriptor)
        return outputs

    return {
        "run": run,
        "outputs": outputs,
        "attention_bindings": dict(
            query=q_attention[0], kv=kv, handoff=_union_handoff,
            handoff_bytes=max(4096, ((local16 + 1) * 4 + 63) // 64 * 64),
        ),
        "debug_indices": debug_output if debug_indices else None,
        "debug_scores": debug_scores if debug_indices and not union else None,
        "debug_stage16_positions": (
            debug_scores if debug_indices and union and profile_stop == 5 else None
        ),
        "debug_stage16_scores": (
            debug_scores if debug_indices and union and profile_stop == 3 else None
        ),
        "debug_stage16_inherited": None,  # union recomputes, no inherited scores
        "debug_stage16_local": (
            debug_scores if debug_indices and union and profile_stop == 4 else None
        ),
        "debug_stage2_scores": (
            debug_scores
            if debug_indices
            and union
            and profile_stop == 8
            and _union_split_role == "stage2"
            else None
        ),
        "debug_union_counts": (
            debug_output
            if debug_indices
            and union
            and (
                profile_stop == 10
                or (_union_split_role == "stage2" and profile_stop == 9)
            )
            else None
        ),
        "debug_union_count_column": topk,
        "descriptor": descriptor,
        "io": io,
        "worker_groups": groups,
        "workers_per_slot": workers,
        "spatial_workers_per_slot": 16,
        "stage16_candidates": n16,
        "stage16_dynamic_input_count": dynamic_inputs,
        "stage16_schedule": (stage16_schedule_config() if stage16_program else {}),
        "stage16_per_head": m16,
        "stage16_local_per_shard": local_m,
        "noc_packet_bytes": row_bytes,
        "noc_packets_per_slot": group * (workers - 1) * (3 if union else 1),
        "noc_frontier_phases": (
            ["Stage-1.6 head merge", "union tags", "Stage-II head merge"]
            if union
            else ["Stage-1.6 head merge"]
        ),
        "profile_stop": profile_stop,
        "stage2_stream_topk": not (union and _union_split_role == "stage2"),
        "stage2_local_selection": (
            "one native K=512 sort per compact source-union row; exact two-K512 "
            "top-100 merge only above 512; compact local top-100 packets and exact "
            "sorted four-way global merge"
            if union and _union_split_role == "stage2"
            else "streaming scalar top-100 heap"
        ),
        "stage2_frontier_packet_bytes": topk * 8 if union else None,
        "per_head_input": per_head_input,
        "stage16_handoff": "union" if union else "per-head",
        "stage16_scoring": (
            "candidate-quarter exact BF16 4-bit matrix, full recomputation"
            if union
            else "per-head SFPU"
        ),
        "stage16_gather": (
            "all scorers, retained quarter writes to root"
            if union
            else "root gather/multicast"
        ),
        "stage16_merge": (
            "native sorted TopK-XL merge" if union else "scalar frontier merge"
        ),
        "stage16_frontier_packet_bytes": (
            (512 if m16 <= 512 else 1024) * 4 if union else row_bytes
        ),
        "stage16_frontier_packets_per_slot": workers * group if union else None,
        "stage16_profile5_payload": "candidate positions" if union else "token IDs",
        "stage16_score_exchange_bytes": n16 * group * 2 if union else 0,
        "stage16_score_exchange_remote_bytes": n16 * (group - 1) * 2 if union else 0,
        "stage16_score_exchange_remote_writes": (
            workers * group * (group - 1) if union else 0
        ),
        "stage2_scoring": (
            "candidate-local matrix, recompute 8-bit" if union else "residual SFPU"
        ),
    }
