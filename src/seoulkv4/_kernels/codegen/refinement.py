"""Explicit Stage1.6, union refinement and sparse attention kernel construction."""

from seoulkv4._kernels.common.refinement_abi import RefinementSemaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

from seoulkv4._kernels.common.refinement_abi import RefinementCB as CB
from seoulkv4._kernels.common.cb_layout import cpp_defines

from seoulkv4._kernels.codegen.preprocessor import profile_defines
from seoulkv4._kernels.common.refinement_abi import refinement_defines
from seoulkv4._kernels.stage16.fusion_abi import AttentionSemaphore


def compute(
    *,
    role,
    dynamic,
    phase,
    profile,
    id_bits=None,
    prefix_page_keys=64,
    prefix_pages=1,
    retain_middle=False,
    debug=False,
):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if role not in (None, "stage16", "stage2", "attention"):
        raise ValueError("unknown refinement role")
    if phase not in ("full", "input", "copy", "decode", "pack", "matmul"):
        raise ValueError("unknown refinement phase")
    if profile and role is None:
        raise ValueError("native markers require a split role")
    if profile and phase != "full":
        raise ValueError("native markers require the full phase")
    if profile and role == "stage16" and (not dynamic):
        raise ValueError("Stage1.6 markers require dynamic inputs")
    if (role, phase) in [
        ("attention", "copy"),
        ("attention", "decode"),
        ("attention", "input"),
        ("attention", "matmul"),
        ("attention", "pack"),
        ("stage2", "input"),
    ]:
        raise ValueError("unsupported diagnostic role/phase combination")
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + refinement_defines("compute", role=role, dynamic=dynamic)
        + profile_defines(
            profile=profile,
            role=role,
            phase=phase,
            dynamic=dynamic,
            has_id_bits=id_bits is not None,
            retain_middle=bool(retain_middle),
            debug=bool(debug),
        )
        + (
            r"""#if SKV4_PROFILE
// skv4-fine-control-v3
#include "tools/profiler/kernel_profiler.hpp"
#endif
#define kernel_main sparse_tail_main
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sparse_sdpa_compute.cpp"
#undef kernel_main
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
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
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
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
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
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
#else
void kernel_main() {
    const uint32_t part=get_arg_val<uint32_t>(SKV4_ARG_RF_COMPUTE_RT_PART);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_C_TAIL_B", 0);
#endif
    if(part==0) sparse_tail_main();
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_C_TAIL_E", 0);
#endif
#endif
#endif
#endif
}
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
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
__attribute__((noinline,noclone)) void stage16_sorted_merge() {
    // The four shard frontiers are already sorted. Their writer performs the
    // exact 4-way merge directly; a second TopK-XL pass only re-sorts padding.
}
__attribute__((noinline,noclone)) void stage16_local_topk(
    uint32_t input_id, uint32_t output_id, uint32_t live_elements) {
    constexpr uint32_t TK = 1024;
    constexpr uint32_t second_input_id = SKV4_CB_ATTENTION_NEG_INF;
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
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
    if(high) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (true),0,VectorMode::None,plane));
    } else {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (false),0,VectorMode::None,plane));
    }
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_INPUT
    if(high) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (true),0,VectorMode::None,plane));
    } else {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (false),0,VectorMode::None,plane));
    }
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
    if(high) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (true),0,VectorMode::None,plane));
    } else {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (false),0,VectorMode::None,plane));
    }
#endif
#endif
#endif
    tile_regs_commit(); tile_regs_wait();
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
    pack_tile(2,SKV4_CB_DECODED_KEYS);
    pack_tile(3,SKV4_CB_DECODED_KEYS);
    pack_tile(4,SKV4_CB_DECODED_KEYS);
    pack_tile(5,SKV4_CB_DECODED_KEYS);
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_INPUT
    pack_tile(2,SKV4_CB_DECODED_KEYS);
    pack_tile(3,SKV4_CB_DECODED_KEYS);
    pack_tile(4,SKV4_CB_DECODED_KEYS);
    pack_tile(5,SKV4_CB_DECODED_KEYS);
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE)
    pack_tile(2,SKV4_CB_DECODED_KEYS);
    pack_tile(3,SKV4_CB_DECODED_KEYS);
    pack_tile(4,SKV4_CB_DECODED_KEYS);
    pack_tile(5,SKV4_CB_DECODED_KEYS);
#endif
#endif
#endif
#endif
}
#if SKV4_CODEGEN_DEBUG
__attribute__((noinline,noclone,optimize("Os"))) void union_expand_eight(bool high, uint32_t input) {
#else
__attribute__((noinline,noclone)) void union_expand_eight(bool high, uint32_t input) {
#endif
    #pragma GCC unroll 1
    for(uint32_t t=0;t<8;t+=4) {
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
        cb_reserve_back(SKV4_CB_DECODED_KEYS,4);
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_INPUT
        cb_reserve_back(SKV4_CB_DECODED_KEYS,4);
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE)
        cb_reserve_back(SKV4_CB_DECODED_KEYS,4);
#endif
#endif
#endif
#endif
        tile_regs_acquire();
        copy_tile(SKV4_CB_PACKED_CANDIDATES,input,0);
        if(high) copy_tile(SKV4_CB_PACKED_CANDIDATES,input+1,1);
        union_expand_tile(high,t);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
        tile_regs_release(); cb_push_back(SKV4_CB_DECODED_KEYS,4);
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_INPUT
        tile_regs_release(); cb_push_back(SKV4_CB_DECODED_KEYS,4);
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY
        tile_regs_release();
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE
        tile_regs_release();
#else
        tile_regs_release(); cb_push_back(SKV4_CB_DECODED_KEYS,4);
#endif
#endif
#endif
#endif
    }
}
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#ifdef TRISC_MATH
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
    cb_reserve_back(SKV4_CB_DECODE_SCRATCH,4);
    tile_regs_acquire();
    copy_tile(SKV4_CB_PACKED_MIDDLE,tile_index,0);
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,s16_expand_four_planes,
        (FIRST),0,VectorMode::None));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1,SKV4_CB_DECODE_SCRATCH);
    pack_tile(2,SKV4_CB_DECODE_SCRATCH);
    pack_tile(3,SKV4_CB_DECODE_SCRATCH);
    pack_tile(4,SKV4_CB_DECODE_SCRATCH);
    tile_regs_release();
    cb_push_back(SKV4_CB_DECODE_SCRATCH,4);
}
void refinement_main(bool owner) {
    constexpr uint32_t LOCAL16_PAD =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_LOCAL16_PAD);
    constexpr uint32_t M16_PAD =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_M16_PAD);
    constexpr uint32_t PROFILE =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_PROFILE);
    constexpr uint32_t STAGE2_PLANES =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_STAGE2_PLANES);
    constexpr uint32_t D = 256;
    constexpr uint32_t packed2 = SKV4_CB_PACKED_MIDDLE, query16 = SKV4_CB_DECODE_SCRATCH;
    constexpr uint32_t packed4 = SKV4_CB_PACKED_CANDIDATES, scores_id = SKV4_CB_SCORE_TILES;
    // Keep every CB's producer/consumer direction fixed. CB30 carries the
    // writer-produced BF16 TopK input, CB17 the compute-produced fused result,
    // and CB24 is refilled with the Stage-II query after its Stage-1.6 query
    // pages have been consumed.
    constexpr uint32_t topk_input = SKV4_CB_SELECTOR_CHANNEL;
    constexpr uint32_t query_pattern = SKV4_CB_DECODE_SCRATCH;
    constexpr uint32_t topk_output = SKV4_CB_LOCAL_FRONTIERS;
    constexpr uint32_t reduce16 = SKV4_CB_MATRIX_QUERY, reduce_ones = SKV4_CB_EXCHANGE_TILES, partial_id = SKV4_CB_ATTENTION_MASK;
    compute_kernel_hw_startup(SKV4_CB_PACKED_MIDDLE,SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH);
#if SKV4_CODEGEN_DYNAMIC
    cb_wait_front(SKV4_CB_LIVE_CONTROL,1);
    const uint32_t stage16_live=ckernel::read_tile_value(SKV4_CB_LIVE_CONTROL,0,0);
    cb_pop_front(SKV4_CB_LIVE_CONTROL,1);
    const uint32_t stage16_padded=(stage16_live+31)/32*32;
#endif
    if constexpr(PROFILE<3) return;
    {
#if SKV4_CODEGEN_DYNAMIC
    const uint32_t stage16_chunks=(stage16_live+31)/32;
#else
    constexpr uint32_t CHUNKS=LOCAL16_PAD/32;
#endif
    const uint32_t part=get_arg_val<uint32_t>(SKV4_ARG_RF_COMPUTE_RT_PART);
#if SKV4_CODEGEN_DYNAMIC
    const uint32_t first=part*(stage16_chunks/4)+
        (part<stage16_chunks%4?part:stage16_chunks%4);
    const uint32_t count=stage16_chunks/4+(part<stage16_chunks%4);
#else
    const uint32_t first=part*(CHUNKS/4)+(part<CHUNKS%4?part:CHUNKS%4);
    const uint32_t count=CHUNKS/4+(part<CHUNKS%4);
#endif
    cb_wait_front(SKV4_CB_MATRIX_QUERY,8);
    cb_wait_front(SKV4_CB_PACKED_MIDDLE,count);
    MATH(SFPU_UNARY_INIT(unused));
    for(uint32_t local_chunk=0;local_chunk<count;++local_chunk) {
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_DECODE_SCRATCH,SKV4_CB_PACKED_MIDDLE);
        pack_reconfig_data_format(SKV4_CB_DECODE_SCRATCH);
        s16_expand_four<0>(local_chunk); s16_expand_four<4>(local_chunk);
        // Consume this chunk before decoding the next one; writer drains in parallel.
    reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH);
    matmul_init(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH,1);
    pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
        cb_wait_front(SKV4_CB_DECODE_SCRATCH,8); cb_reserve_back(SKV4_CB_SCORE_TILES,1); tile_regs_acquire();
        if(local_chunk==0) {
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                DST_ACCUM_MODE,s16_clear_score,0,VectorMode::None));
            reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH);
        }
        for(uint32_t kt=0;kt<8;++kt) matmul_tiles(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH,kt,kt,0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,SKV4_CB_SCORE_TILES);
        tile_regs_release(); cb_push_back(SKV4_CB_SCORE_TILES,1); cb_pop_front(SKV4_CB_DECODE_SCRATCH,8);
    }
    // Pop the whole multicast train exactly once, after all chunk-local uses.
    cb_pop_front(SKV4_CB_PACKED_MIDDLE,count);
    cb_pop_front(SKV4_CB_MATRIX_QUERY,8);
    }
    if constexpr (PROFILE >= 4) {
        // The writer publishes one or two K=1024 BF16 tiles on dedicated CB30;
        // the fused score/index result returns on CB17.
        static_assert(LOCAL16_PAD <= 2048);
#if SKV4_CODEGEN_DYNAMIC
        stage16_local_topk(topk_input, topk_output, stage16_padded);
#else
        stage16_local_topk(topk_input, topk_output, LOCAL16_PAD);
#endif
    }
    if constexpr(PROFILE >= 5) {
        // This is the second TopK pass: only the candidate-shard root owns
        // the globally merged frontier and publishes its CB30 input.  The
        // first (per-head local) TopK above is still executed by all cores.
        if(owner) stage16_sorted_merge();
    }
    if (!owner || PROFILE < 8) {
        return;
    }
    // Reader-owned control mailbox broadcasts the variable local union size
    // to all three TRISCs using the stock sparse-SDPA primitive.
    cb_wait_front(SKV4_CB_LIVE_CONTROL,1);
    const uint32_t count=ckernel::read_tile_value(SKV4_CB_LIVE_CONTROL,0,0);
    cb_pop_front(SKV4_CB_LIVE_CONTROL,1);
    cb_wait_front(SKV4_CB_MATRIX_QUERY,16);
    constexpr uint32_t CAP=LOCAL16_PAD/32*3;
    cb_wait_front(SKV4_CB_PACKED_CANDIDATES,CAP);
    MATH(SFPU_UNARY_INIT(unused));
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
    for(uint32_t chunk=0;chunk<(count+31)/32;++chunk) {
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_ATTENTION_MASK,SKV4_CB_PACKED_CANDIDATES);
        pack_reconfig_data_format(SKV4_CB_DECODED_KEYS);
        union_expand_eight(true,chunk*3);
        union_expand_eight(false,chunk*3+2);
        cb_wait_front(SKV4_CB_DECODED_KEYS,16);
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        matmul_init(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,1);
        pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
        cb_reserve_back(SKV4_CB_SCORE_TILES,1);
        tile_regs_acquire();
        // The preceding TopK modifies Dst state: explicitly initialize the
        // accumulator before the first FPU score (composition regression).
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
            union_clear,0,VectorMode::None));
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        for(uint32_t kt=0;kt<16;++kt) matmul_tiles(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,kt,kt,0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,SKV4_CB_SCORE_TILES);
        tile_regs_release(); cb_push_back(SKV4_CB_SCORE_TILES,1); cb_pop_front(SKV4_CB_DECODED_KEYS,16);
    }
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_INPUT)
    for(uint32_t chunk=0;chunk<(count+31)/32;++chunk) {
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_ATTENTION_MASK,SKV4_CB_PACKED_CANDIDATES);
        pack_reconfig_data_format(SKV4_CB_DECODED_KEYS);
        union_expand_eight(true,chunk*3);
        union_expand_eight(false,chunk*3+2);
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE)
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_PACK
        cb_wait_front(SKV4_CB_DECODED_KEYS,16); cb_pop_front(SKV4_CB_DECODED_KEYS,16);
#else
        cb_wait_front(SKV4_CB_DECODED_KEYS,16);
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        matmul_init(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,1);
        pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
        tile_regs_acquire();
        // The preceding TopK modifies Dst state: explicitly initialize the
        // accumulator before the first FPU score (composition regression).
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
            union_clear,0,VectorMode::None));
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        for(uint32_t kt=0;kt<16;++kt) matmul_tiles(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,kt,kt,0);
        tile_regs_commit(); tile_regs_wait();
        tile_regs_release(); cb_pop_front(SKV4_CB_DECODED_KEYS,16);
#endif
#endif
#endif
    }
#endif
#endif
    cb_pop_front(SKV4_CB_PACKED_CANDIDATES,CAP); cb_pop_front(SKV4_CB_MATRIX_QUERY,16);
}
void kernel_main() {
    constexpr uint32_t PROFILE =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_PROFILE);
    constexpr uint32_t G = 4;
    const uint32_t shard = get_arg_val<uint32_t>(SKV4_ARG_RF_COMPUTE_RT_SHARD);
    const uint32_t part = get_arg_val<uint32_t>(SKV4_ARG_RF_COMPUTE_RT_PART);
    const bool owner = part == 0;
    refinement_main(owner);
    if constexpr (PROFILE >= 10) {
        if (owner) sparse_tail_main();
    }
}
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
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
__attribute__((noinline,noclone)) void stage16_sorted_merge() {
    // The four shard frontiers are already sorted. Their writer performs the
    // exact 4-way merge directly; a second TopK-XL pass only re-sorts padding.
}
__attribute__((noinline,noclone)) void stage16_local_topk(
    uint32_t input_id, uint32_t output_id, uint32_t live_elements) {
    constexpr uint32_t TK = 1024;
    constexpr uint32_t second_input_id = SKV4_CB_ATTENTION_NEG_INF;
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
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
    if(high) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (true),0,VectorMode::None,plane));
    } else {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (false),0,VectorMode::None,plane));
    }
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_INPUT
    if(high) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (true),0,VectorMode::None,plane));
    } else {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (false),0,VectorMode::None,plane));
    }
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
    if(high) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (true),0,VectorMode::None,plane));
    } else {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (false),0,VectorMode::None,plane));
    }
#endif
#endif
#endif
    tile_regs_commit(); tile_regs_wait();
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
    pack_tile(2,SKV4_CB_DECODED_KEYS);
    pack_tile(3,SKV4_CB_DECODED_KEYS);
    pack_tile(4,SKV4_CB_DECODED_KEYS);
    pack_tile(5,SKV4_CB_DECODED_KEYS);
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_INPUT
    pack_tile(2,SKV4_CB_DECODED_KEYS);
    pack_tile(3,SKV4_CB_DECODED_KEYS);
    pack_tile(4,SKV4_CB_DECODED_KEYS);
    pack_tile(5,SKV4_CB_DECODED_KEYS);
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE)
    pack_tile(2,SKV4_CB_DECODED_KEYS);
    pack_tile(3,SKV4_CB_DECODED_KEYS);
    pack_tile(4,SKV4_CB_DECODED_KEYS);
    pack_tile(5,SKV4_CB_DECODED_KEYS);
#endif
#endif
#endif
#endif
}
__attribute__((noinline,noclone)) void union_expand_eight(bool high, uint32_t input) {
    #pragma GCC unroll 1
    for(uint32_t t=0;t<8;t+=4) {
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
        cb_reserve_back(SKV4_CB_DECODED_KEYS,4);
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_INPUT
        cb_reserve_back(SKV4_CB_DECODED_KEYS,4);
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE)
        cb_reserve_back(SKV4_CB_DECODED_KEYS,4);
#endif
#endif
#endif
#endif
        tile_regs_acquire();
        copy_tile(SKV4_CB_PACKED_CANDIDATES,input,0);
        if(high) copy_tile(SKV4_CB_PACKED_CANDIDATES,input+1,1);
        union_expand_tile(high,t);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
        tile_regs_release(); cb_push_back(SKV4_CB_DECODED_KEYS,4);
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_INPUT
        tile_regs_release(); cb_push_back(SKV4_CB_DECODED_KEYS,4);
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY
        tile_regs_release();
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE
        tile_regs_release();
#else
        tile_regs_release(); cb_push_back(SKV4_CB_DECODED_KEYS,4);
#endif
#endif
#endif
#endif
    }
}
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#ifdef TRISC_MATH
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
    cb_reserve_back(SKV4_CB_DECODE_SCRATCH,4);
    tile_regs_acquire();
    copy_tile(SKV4_CB_PACKED_MIDDLE,tile_index,0);
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,s16_expand_four_planes,
        (FIRST),0,VectorMode::None));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1,SKV4_CB_DECODE_SCRATCH);
    pack_tile(2,SKV4_CB_DECODE_SCRATCH);
    pack_tile(3,SKV4_CB_DECODE_SCRATCH);
    pack_tile(4,SKV4_CB_DECODE_SCRATCH);
    tile_regs_release();
    cb_push_back(SKV4_CB_DECODE_SCRATCH,4);
}
void refinement_main(bool owner) {
    constexpr uint32_t LOCAL16_PAD =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_LOCAL16_PAD);
    constexpr uint32_t M16_PAD =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_M16_PAD);
    constexpr uint32_t PROFILE =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_PROFILE);
    constexpr uint32_t STAGE2_PLANES =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_STAGE2_PLANES);
    constexpr uint32_t D = 256;
    constexpr uint32_t packed2 = SKV4_CB_PACKED_MIDDLE, query16 = SKV4_CB_DECODE_SCRATCH;
    constexpr uint32_t packed4 = SKV4_CB_PACKED_CANDIDATES, scores_id = SKV4_CB_SCORE_TILES;
    // Keep every CB's producer/consumer direction fixed. CB30 carries the
    // writer-produced BF16 TopK input, CB17 the compute-produced fused result,
    // and CB24 is refilled with the Stage-II query after its Stage-1.6 query
    // pages have been consumed.
    constexpr uint32_t topk_input = SKV4_CB_SELECTOR_CHANNEL;
    constexpr uint32_t query_pattern = SKV4_CB_DECODE_SCRATCH;
    constexpr uint32_t topk_output = SKV4_CB_LOCAL_FRONTIERS;
    constexpr uint32_t reduce16 = SKV4_CB_MATRIX_QUERY, reduce_ones = SKV4_CB_EXCHANGE_TILES, partial_id = SKV4_CB_ATTENTION_MASK;
    compute_kernel_hw_startup(SKV4_CB_PACKED_MIDDLE,SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH);
#if SKV4_CODEGEN_DYNAMIC
    cb_wait_front(SKV4_CB_LIVE_CONTROL,1);
    const uint32_t stage16_live=ckernel::read_tile_value(SKV4_CB_LIVE_CONTROL,0,0);
    cb_pop_front(SKV4_CB_LIVE_CONTROL,1);
    const uint32_t stage16_padded=(stage16_live+31)/32*32;
#endif
    if constexpr(PROFILE<3) return;
    {
#if SKV4_CODEGEN_DYNAMIC
    const uint32_t stage16_chunks=(stage16_live+31)/32;
#else
    constexpr uint32_t CHUNKS=LOCAL16_PAD/32;
#endif
    const uint32_t part=get_arg_val<uint32_t>(SKV4_ARG_RF_COMPUTE_RT_PART);
#if SKV4_CODEGEN_DYNAMIC
    const uint32_t first=part*(stage16_chunks/4)+
        (part<stage16_chunks%4?part:stage16_chunks%4);
    const uint32_t count=stage16_chunks/4+(part<stage16_chunks%4);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_C_WAIT_INPUT_B", 0);
#endif
#endif
#else
    const uint32_t first=part*(CHUNKS/4)+(part<CHUNKS%4?part:CHUNKS%4);
    const uint32_t count=CHUNKS/4+(part<CHUNKS%4);
#endif
    cb_wait_front(SKV4_CB_MATRIX_QUERY,8);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_C_WAIT_INPUT_E", 0);
#endif
#endif
#endif
    MATH(SFPU_UNARY_INIT(unused));
    for(uint32_t local_chunk=0;local_chunk<count;++local_chunk) {
        // Cumulative readiness: original tile indices remain stable until
        // the final pop. No producer overwrite, no per-key remote signals.
        cb_wait_front(SKV4_CB_PACKED_MIDDLE,local_chunk+1);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_C_DECODE_BF16_B", 0);
#endif
#endif
#endif
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_DECODE_SCRATCH,SKV4_CB_PACKED_MIDDLE);
        pack_reconfig_data_format(SKV4_CB_DECODE_SCRATCH);
        s16_expand_four<0>(local_chunk); s16_expand_four<4>(local_chunk);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_C_DECODE_BF16_E", 0);
#endif
#endif
#endif
        // Consume this chunk before decoding the next one; writer drains in parallel.
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_C_MATMUL_B", 0);
#endif
#endif
#endif
    reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH);
    matmul_init(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH,1);
    pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
        cb_wait_front(SKV4_CB_DECODE_SCRATCH,8); cb_reserve_back(SKV4_CB_SCORE_TILES,1); tile_regs_acquire();
        if(local_chunk==0) {
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,
                DST_ACCUM_MODE,s16_clear_score,0,VectorMode::None));
            reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH);
        }
        for(uint32_t kt=0;kt<8;++kt) matmul_tiles(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODE_SCRATCH,kt,kt,0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,SKV4_CB_SCORE_TILES);
        tile_regs_release(); cb_push_back(SKV4_CB_SCORE_TILES,1); cb_pop_front(SKV4_CB_DECODE_SCRATCH,8);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_C_MATMUL_E", 0);
#endif
#endif
#endif
    }
    // Pop the whole multicast train exactly once, after all chunk-local uses.
    cb_pop_front(SKV4_CB_PACKED_MIDDLE,count);
    cb_pop_front(SKV4_CB_MATRIX_QUERY,8);
    }
    if constexpr (PROFILE >= 4) {
        // The writer publishes one or two K=1024 BF16 tiles on dedicated CB30;
        // the fused score/index result returns on CB17.
        static_assert(LOCAL16_PAD <= 2048);
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_C_LOCAL_TOPK_B", 0);
#endif
#endif
        stage16_local_topk(topk_input, topk_output, stage16_padded);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_C_LOCAL_TOPK_E", 0);
#endif
#endif
#else
        stage16_local_topk(topk_input, topk_output, LOCAL16_PAD);
#endif
    }
    if constexpr(PROFILE >= 5) {
        // This is the second TopK pass: only the candidate-shard root owns
        // the globally merged frontier and publishes its CB30 input.  The
        // first (per-head local) TopK above is still executed by all cores.
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_C_GLOBAL_TOPK_B", 0);
#endif
#endif
#endif
        if(owner) stage16_sorted_merge();
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_C_GLOBAL_TOPK_E", 0);
#endif
#endif
#endif
    }
    if (!owner || PROFILE < 8) {
        return;
    }
    // Reader-owned control mailbox broadcasts the variable local union size
    // to all three TRISCs using the stock sparse-SDPA primitive.
    cb_wait_front(SKV4_CB_LIVE_CONTROL,1);
    const uint32_t count=ckernel::read_tile_value(SKV4_CB_LIVE_CONTROL,0,0);
    cb_pop_front(SKV4_CB_LIVE_CONTROL,1);
    cb_wait_front(SKV4_CB_MATRIX_QUERY,16);
    constexpr uint32_t CAP=LOCAL16_PAD/32*3;
    cb_wait_front(SKV4_CB_PACKED_CANDIDATES,CAP);
    MATH(SFPU_UNARY_INIT(unused));
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
    for(uint32_t chunk=0;chunk<(count+31)/32;++chunk) {
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_ATTENTION_MASK,SKV4_CB_PACKED_CANDIDATES);
        pack_reconfig_data_format(SKV4_CB_DECODED_KEYS);
        union_expand_eight(true,chunk*3);
        union_expand_eight(false,chunk*3+2);
        cb_wait_front(SKV4_CB_DECODED_KEYS,16);
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        matmul_init(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,1);
        pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
        cb_reserve_back(SKV4_CB_SCORE_TILES,1);
        tile_regs_acquire();
        // The preceding TopK modifies Dst state: explicitly initialize the
        // accumulator before the first FPU score (composition regression).
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
            union_clear,0,VectorMode::None));
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        for(uint32_t kt=0;kt<16;++kt) matmul_tiles(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,kt,kt,0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,SKV4_CB_SCORE_TILES);
        tile_regs_release(); cb_push_back(SKV4_CB_SCORE_TILES,1); cb_pop_front(SKV4_CB_DECODED_KEYS,16);
    }
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_INPUT)
    for(uint32_t chunk=0;chunk<(count+31)/32;++chunk) {
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_ATTENTION_MASK,SKV4_CB_PACKED_CANDIDATES);
        pack_reconfig_data_format(SKV4_CB_DECODED_KEYS);
        union_expand_eight(true,chunk*3);
        union_expand_eight(false,chunk*3+2);
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE)
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_PACK
        cb_wait_front(SKV4_CB_DECODED_KEYS,16); cb_pop_front(SKV4_CB_DECODED_KEYS,16);
#else
        cb_wait_front(SKV4_CB_DECODED_KEYS,16);
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        matmul_init(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,1);
        pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
        tile_regs_acquire();
        // The preceding TopK modifies Dst state: explicitly initialize the
        // accumulator before the first FPU score (composition regression).
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
            union_clear,0,VectorMode::None));
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        for(uint32_t kt=0;kt<16;++kt) matmul_tiles(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,kt,kt,0);
        tile_regs_commit(); tile_regs_wait();
        tile_regs_release(); cb_pop_front(SKV4_CB_DECODED_KEYS,16);
#endif
#endif
#endif
    }
#endif
#endif
    cb_pop_front(SKV4_CB_PACKED_CANDIDATES,CAP); cb_pop_front(SKV4_CB_MATRIX_QUERY,16);
}
void kernel_main() {
    constexpr uint32_t PROFILE =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_PROFILE);
    constexpr uint32_t G = 4;
    const uint32_t shard = get_arg_val<uint32_t>(SKV4_ARG_RF_COMPUTE_RT_SHARD);
    const uint32_t part = get_arg_val<uint32_t>(SKV4_ARG_RF_COMPUTE_RT_PART);
    const bool owner = part == 0;
    refinement_main(owner);
    if constexpr (PROFILE >= 10) {
        if (owner) sparse_tail_main();
    }
}
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
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
// Stage II never has more than 640 source-owned union entries. Sort one
// K512 half on the common path and a second half only on overflow. The writer
// merges the two exact top-100 frontiers, so no candidate is truncated.
inline void stage2_exact_k512_topk(
    uint32_t input_id, uint32_t second_input_id, uint32_t control_id,
    uint32_t output_id) {
    constexpr uint32_t TK = 512;
    CircularBuffer input(input_id), second_input(second_input_id);
    CircularBuffer control(control_id), output(output_id);
    control.wait_front(1);
    const uint32_t live = ckernel::read_tile_value(control_id, 0, 0);
    input.wait_front(1);
    tile_regs_acquire();
    topk_xl_copy_tile_init(input_id);
    topk_xl_copy_tile<TK>(input_id, 0, 0, live < TK ? live : TK);
    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<TK, 0>(0);
    topk_xl_init<TK, true>();
    topk_xl_local_sort<TK>(0, false);
    topk_xl_separate_indices_row_major_init_static<0, 0>();
    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<TK>(0);
    topk_xl_separate_indices_row_major_advance_chunk_base<TK>();
    topk_xl_init<TK, false>();
    topk_xl_rebuild<TK, false>(0, false);
    tile_regs_commit();
    output.reserve_back(1);
    tile_regs_wait();
    pack_reconfig_data_format(output_id);
    pack_tile(1, output_id);
    output.push_back(1);
    tile_regs_release();
    input.pop_front(1);
    if (live > TK) {
        second_input.wait_front(1);
        tile_regs_acquire();
        topk_xl_copy_tile_init(second_input_id);
        topk_xl_copy_tile<TK>(second_input_id, 0, 0, live - TK);
        // Keep positions local to the second half. The writer adds 512.
        topk_xl_add_lsb_indices_init();
        topk_xl_add_lsb_indices<TK, 0>(0);
        topk_xl_init<TK, true>();
        topk_xl_local_sort<TK>(0, false);
        topk_xl_separate_indices_row_major_init_static<0, 0>();
        topk_xl_separate_indices_row_major_reinit();
        topk_xl_separate_indices_row_major<TK>(0);
        topk_xl_separate_indices_row_major_advance_chunk_base<TK>();
        topk_xl_init<TK, false>();
        topk_xl_rebuild<TK, false>(0, false);
        tile_regs_commit();
        output.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(output_id);
        pack_tile(1, output_id);
        output.push_back(1);
        tile_regs_release();
        second_input.pop_front(1);
    }
    control.pop_front(1);
}
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
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
    if(high) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (true),0,VectorMode::None,plane));
    } else {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (false),0,VectorMode::None,plane));
    }
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
    if(high) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (true),0,VectorMode::None,plane));
    } else {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,union_expand,
            (false),0,VectorMode::None,plane));
    }
#endif
#endif
    tile_regs_commit(); tile_regs_wait();
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
    pack_tile(2,SKV4_CB_DECODED_KEYS);
    pack_tile(3,SKV4_CB_DECODED_KEYS);
    pack_tile(4,SKV4_CB_DECODED_KEYS);
    pack_tile(5,SKV4_CB_DECODED_KEYS);
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE)
    pack_tile(2,SKV4_CB_DECODED_KEYS);
    pack_tile(3,SKV4_CB_DECODED_KEYS);
    pack_tile(4,SKV4_CB_DECODED_KEYS);
    pack_tile(5,SKV4_CB_DECODED_KEYS);
#endif
#endif
#endif
}
__attribute__((noinline,noclone)) void union_expand_eight(bool high, uint32_t input) {
    #pragma GCC unroll 1
    for(uint32_t t=0;t<8;t+=4) {
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
        cb_reserve_back(SKV4_CB_DECODED_KEYS,4);
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE)
        cb_reserve_back(SKV4_CB_DECODED_KEYS,4);
#endif
#endif
#endif
        tile_regs_acquire();
        copy_tile(SKV4_CB_PACKED_CANDIDATES,input,0);
        if(high) copy_tile(SKV4_CB_PACKED_CANDIDATES,input+1,1);
        union_expand_tile(high,t);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
        tile_regs_release(); cb_push_back(SKV4_CB_DECODED_KEYS,4);
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY
        tile_regs_release();
#else
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE
        tile_regs_release();
#else
        tile_regs_release(); cb_push_back(SKV4_CB_DECODED_KEYS,4);
#endif
#endif
#endif
    }
}
void refinement_main(bool owner) {
    constexpr uint32_t LOCAL16_PAD =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_LOCAL16_PAD);
    constexpr uint32_t M16_PAD =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_M16_PAD);
    constexpr uint32_t PROFILE =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_PROFILE);
    constexpr uint32_t STAGE2_PLANES =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_STAGE2_PLANES);
    if(PROFILE<8) return;
    compute_kernel_hw_startup(SKV4_CB_PACKED_CANDIDATES,SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
    cb_wait_front(SKV4_CB_MATRIX_QUERY,16);
    MATH(SFPU_UNARY_INIT(unused));
    for(;;) {
        cb_wait_front(SKV4_CB_LIVE_CONTROL,1);
        const uint32_t count=ckernel::read_tile_value(SKV4_CB_LIVE_CONTROL,0,0);
        cb_pop_front(SKV4_CB_LIVE_CONTROL,1);
        if(count==0) break;
        cb_wait_front(SKV4_CB_PACKED_CANDIDATES,3);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_C_DECODE_TO_BF16_B", 0);
#endif
#endif
        copy_tile_to_dst_init_short_with_dt(SKV4_CB_ATTENTION_MASK,SKV4_CB_PACKED_CANDIDATES);
        pack_reconfig_data_format(SKV4_CB_DECODED_KEYS);
        union_expand_eight(true,0);
        union_expand_eight(false,2);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_C_DECODE_TO_BF16_E", 0);
    DeviceTimestampedData("SKV_S2_C_MATMUL_B", 0);
#endif
        cb_wait_front(SKV4_CB_DECODED_KEYS,16);
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        matmul_init(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,1);
        pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
        cb_reserve_back(SKV4_CB_SCORE_TILES,1);
        tile_regs_acquire();
        // The preceding TopK modifies Dst state: explicitly initialize the
        // accumulator before the first FPU score (composition regression).
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
            union_clear,0,VectorMode::None));
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        for(uint32_t kt=0;kt<16;++kt) matmul_tiles(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,kt,kt,0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,SKV4_CB_SCORE_TILES);
        tile_regs_release(); cb_push_back(SKV4_CB_SCORE_TILES,1); cb_pop_front(SKV4_CB_DECODED_KEYS,16);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_C_MATMUL_E", 0);
#endif
#else
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_COPY)
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_DECODE)
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_PACK
        cb_wait_front(SKV4_CB_DECODED_KEYS,16); cb_pop_front(SKV4_CB_DECODED_KEYS,16);
#else
        cb_wait_front(SKV4_CB_DECODED_KEYS,16);
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        matmul_init(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,1);
        pack_reconfig_data_format(SKV4_CB_SCORE_TILES);
        tile_regs_acquire();
        // The preceding TopK modifies Dst state: explicitly initialize the
        // accumulator before the first FPU score (composition regression).
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
            union_clear,0,VectorMode::None));
        reconfig_data_format<SrcOrder::Reverse>(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS);
        for(uint32_t kt=0;kt<16;++kt) matmul_tiles(SKV4_CB_MATRIX_QUERY,SKV4_CB_DECODED_KEYS,kt,kt,0);
        tile_regs_commit(); tile_regs_wait();
        tile_regs_release(); cb_pop_front(SKV4_CB_DECODED_KEYS,16);
#endif
#endif
#endif
#endif
        cb_pop_front(SKV4_CB_PACKED_CANDIDATES,3);
    }
    cb_pop_front(SKV4_CB_MATRIX_QUERY,16);
    if constexpr(PROFILE>=9) {
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_C_LOCAL_TOPK_B", 0);
#endif
#endif
        stage2_exact_k512_topk(SKV4_CB_SELECTOR_CHANNEL,SKV4_CB_ATTENTION_NEG_INF,SKV4_CB_LIVE_CONTROL,SKV4_CB_LOCAL_FRONTIERS);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_C_LOCAL_TOPK_E", 0);
#endif
#endif
    }
    if(!owner) return;
}
void kernel_main() {
    constexpr uint32_t PROFILE =
        get_compile_time_arg_val(SKV4_ARG_RF_EXT_PROFILE);
    constexpr uint32_t G = 4;
    const uint32_t shard = get_arg_val<uint32_t>(SKV4_ARG_RF_COMPUTE_RT_SHARD);
    const uint32_t part = get_arg_val<uint32_t>(SKV4_ARG_RF_COMPUTE_RT_PART);
    const bool owner = part == 0;
    refinement_main(owner);
    if constexpr (PROFILE >= 10) {
        if (owner) sparse_tail_main();
    }
}
#endif
#endif
#endif
"""
        )
    )


def reader(
    *,
    role,
    dynamic,
    phase,
    profile,
    id_bits=None,
    prefix_page_keys=64,
    prefix_pages=1,
    retain_middle=False,
    debug=False,
    wait_frontier=False,
):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if wait_frontier and role != "attention":
        raise ValueError("frontier wait requires the clean attention reader")
    if role not in (None, "stage16", "stage2", "attention"):
        raise ValueError("unknown refinement role")
    if phase not in ("full", "input", "copy", "decode", "pack", "matmul"):
        raise ValueError("unknown refinement phase")
    if profile and role is None:
        raise ValueError("native markers require a split role")
    if profile and phase != "full":
        raise ValueError("native markers require the full phase")
    if profile and role == "stage16" and (not dynamic):
        raise ValueError("Stage1.6 markers require dynamic inputs")
    if (role, phase) in [
        ("attention", "copy"),
        ("attention", "decode"),
        ("attention", "input"),
        ("attention", "matmul"),
        ("attention", "pack"),
        ("stage2", "input"),
    ]:
        raise ValueError("unsupported diagnostic role/phase combination")
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + refinement_defines("reader", role=role, dynamic=dynamic)
        + semaphore_defines(AttentionSemaphore)
        + profile_defines(
            profile=profile,
            role=role,
            phase=phase,
            dynamic=dynamic,
            has_id_bits=id_bits is not None,
            retain_middle=bool(retain_middle),
            debug=bool(debug),
            wait_frontier=bool(wait_frontier),
        )
        + (
            r"""#if SKV4_PROFILE
// skv4-fine-control-v3
#include "tools/profiler/kernel_profiler.hpp"
#endif
#if SKV4_CODEGEN_HAS_ID_BITS
#define SKV4_ID_BITS """
            + str(0 if id_bits is None else id_bits)
            + r"""
#define SKV4_ID_MASK ((1u << SKV4_ID_BITS) - 1u)
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE)
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
#if SKV4_CODEGEN_RETAIN_MIDDLE
#define SKV4_RETAIN_MIDDLE 1
#endif
#endif
#endif
#define SKV4_PREFIX_PAGE_KEYS """
            + str(prefix_page_keys)
            + r"""
#define SKV4_PREFIX_PAGES """
            + str(prefix_pages)
            + r"""
#endif
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"
constexpr uint16_t neg_inf_bf16 = 0xFF80u;
void kernel_main() {
    constexpr uint32_t N16 = get_compile_time_arg_val(SKV4_ARG_RF_CT_N16);
    constexpr uint32_t W = get_compile_time_arg_val(SKV4_ARG_RF_CT_WORKERS);
    constexpr uint32_t G = get_compile_time_arg_val(SKV4_ARG_RF_CT_GROUP);
    constexpr uint32_t M16 = get_compile_time_arg_val(SKV4_ARG_RF_CT_M16);
    constexpr uint32_t D = get_compile_time_arg_val(SKV4_ARG_RF_CT_D);
    constexpr uint32_t PROFILE = get_compile_time_arg_val(SKV4_ARG_RF_CT_PROFILE);
    constexpr uint32_t PARTS = 4;
    constexpr uint32_t LOCAL16 = N16 / W;
    constexpr uint32_t LOCAL16_PAD = ((LOCAL16 + 31) / 32) * 32;
    constexpr uint32_t M16_PAD = ((M16 + 31) / 32) * 32;
    constexpr uint32_t CHUNKS = LOCAL16_PAD / 32;
    constexpr uint32_t INHERITED_OFFSET = 0;
    constexpr uint32_t Q2_ROW_OFFSET =
        (LOCAL16 * sizeof(uint16_t) + 63) & ~63u;
    constexpr uint32_t STAGE2_QUERY_OFFSET = 8192;
    constexpr bool PER_HEAD_INPUT = get_compile_time_arg_val(SKV4_ARG_RF_CT_PER_HEAD_INPUT) != 0;
    constexpr auto sqa = TensorAccessorArgs<SKV4_ARG_RF_CT_ACCESSORS>();
    constexpr auto r2a = TensorAccessorArgs<sqa.next_compile_time_args_offset()>();
    constexpr auto r4a = TensorAccessorArgs<r2a.next_compile_time_args_offset()>();
    constexpr auto cia = TensorAccessorArgs<r4a.next_compile_time_args_offset()>();
    constexpr auto s2a = TensorAccessorArgs<cia.next_compile_time_args_offset()>();
    constexpr auto aqa = TensorAccessorArgs<s2a.next_compile_time_args_offset()>();
    constexpr auto kva = TensorAccessorArgs<aqa.next_compile_time_args_offset()>();
    static_assert(N16 % W == 0);
    static_assert(W >= G);
    const uint32_t sq_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SQ);
    const uint32_t r2_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_RESIDUAL2);
    const uint32_t r4_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_RESIDUAL4);
    const uint32_t cand_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_CANDIDATES);
    const uint32_t s2_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SCORES2);
    const uint32_t aq_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ATTENTION_QUERY);
    const uint32_t kv_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_KV);
    const uint32_t slot = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SLOT);
    const uint32_t context = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_CONTEXT);
    const uint32_t shard = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SHARD);
    const uint32_t part = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_PART);
    const uint32_t root_x = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROOT_X);
    const uint32_t root_y = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROOT_Y);
    const uint32_t row_start_x = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_START_X);
    const uint32_t row_start_y = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_START_Y);
    const uint32_t row_end_x = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_END_X);
    const uint32_t row_end_y = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_END_Y);
    const bool owner = part == 0;
    const auto sq = TensorAccessor(sqa, sq_addr, D * 2);
    const auto r2 = TensorAccessor(r2a, r2_addr, D / 4);
    const auto r4 = TensorAccessor(r4a, r4_addr, D / 2);
    const auto cand = TensorAccessor(cia, cand_addr, N16 * 4);
    const auto s2 = TensorAccessor(s2a, s2_addr, N16 * 2);
    const auto aq = TensorAccessor(aqa, aq_addr, D * 2);
    const auto kv = TensorAccessor(kva, kv_addr, 2 * D);
    constexpr auto p2a=TensorAccessorArgs<kva.next_compile_time_args_offset()>();
    const auto p2=TensorAccessor(p2a,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_PREFIX2),SKV4_PREFIX_PAGE_KEYS*64);
    const uint32_t my_x[1]={get_arg_val<uint32_t>(SKV4_ARG_RF_RT_MY_X)};
    const uint32_t my_y[1]={get_arg_val<uint32_t>(SKV4_ARG_RF_RT_MY_Y)};
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
#if SKV4_CODEGEN_DYNAMIC
    constexpr auto ica=TensorAccessorArgs<p2a.next_compile_time_args_offset()>();
    const auto input_counts=TensorAccessor(
        ica,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_INPUT_COUNTS),G*sizeof(uint32_t));
#endif
#else
#if SKV4_CODEGEN_DYNAMIC
    constexpr auto ica=TensorAccessorArgs<p2a.next_compile_time_args_offset()>();
    const auto input_counts=TensorAccessor(
        ica,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_INPUT_COUNTS),G*sizeof(uint32_t));
    constexpr auto hfa=TensorAccessorArgs<ica.next_compile_time_args_offset()>();
#else
    constexpr auto hfa=TensorAccessorArgs<p2a.next_compile_time_args_offset()>();
#endif
    constexpr uint32_t META_BYTES=(LOCAL16+1)*4<=4096?4096:((LOCAL16+1)*4+63)/64*64;
    constexpr uint32_t HANDOFF_BYTES=META_BYTES;
#if SKV4_CODEGEN_DYNAMIC
    const auto handoff=TensorAccessor(hfa,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HANDOFF),HANDOFF_BYTES);
#else
    const auto handoff=TensorAccessor(hfa,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HANDOFF),HANDOFF_BYTES);
#endif
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
    constexpr auto mca=TensorAccessorArgs<hfa.next_compile_time_args_offset()>();
    constexpr uint32_t MIDDLE_PAGES=LOCAL16_PAD/16;
#if SKV4_CODEGEN_DYNAMIC
    const auto middle_cache=TensorAccessor(mca,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_MIDDLE),1024);
#else
    const auto middle_cache=TensorAccessor(mca,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_MIDDLE),1024);
#endif
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
    constexpr auto mca=TensorAccessorArgs<hfa.next_compile_time_args_offset()>();
    constexpr uint32_t MIDDLE_PAGES=LOCAL16_PAD/16;
#if SKV4_CODEGEN_DYNAMIC
    const auto middle_cache=TensorAccessor(mca,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_MIDDLE),1024);
#else
    const auto middle_cache=TensorAccessor(mca,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_MIDDLE),1024);
#endif
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_R_FRONTIER_IMPORT_B", 0);
#endif
#endif
    cb_reserve_back(SKV4_CB_SHARED_SCRATCH,1);
    const uint32_t scratch=get_write_ptr(SKV4_CB_SHARED_SCRATCH), meta=get_write_ptr(SKV4_CB_SCORE_SCRATCH);
    auto tags=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
    const uint32_t raw=get_write_ptr(SKV4_CB_PACKED_MIDDLE);
    auto lists=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw);
    constexpr uint32_t LIST_BYTES=((M16+1)*4+63)/64*64;
    // Four bulk reads, not a per-candidate remote notification protocol.
    for(uint32_t h=0;h<G;++h)
        noc_async_read(get_noc_addr(slot*W+h,handoff),raw+h*LIST_BYTES,LIST_BYTES);
    // Frontiers contain candidate POSITIONS. The persistent Stage1.5 ID
    // map supplies token IDs; it is L1, never a new GDDR fetch.
    const uint32_t skip=(shard*LOCAL16*4)&63u;
    const uint32_t idp=raw+G*LIST_BYTES;
    noc_async_read(get_noc_addr(slot,cand)+(shard*LOCAL16*4-skip),
                   idp,((skip+LOCAL16*4+63)/64)*64);
#else
    if(!owner) return;
#if SKV4_CODEGEN_WAIT_FRONTIER
    // No selector dependency: prepare Q while the other64 cores refine keys.
    {
#if SKV4_PROFILE
        DeviceTimestampedData("SKV_AOV_Q_B",0);
#endif
        Noc early_noc;
        experimental::CB early_q(SKV4_CB_ATTENTION_QUERY_ROW);
        early_q.reserve_back(32);
        auto zero=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(early_q.get_write_ptr());
        for(uint32_t j=0;j<32*2*D*2/4;++j) zero[j]=0;
        early_noc.async_read(aq,early_q,D*2,
            {.page_id=slot*G+shard}, {.offset_bytes=shard*(2*D*2)+D*2});
        early_noc.async_read_barrier(); early_q.push_back(32);
#if SKV4_PROFILE
        DeviceTimestampedData("SKV_AOV_Q_E",0);
        DeviceTimestampedData("SKV_AOV_READY_B",0);
#endif
        auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_ATTENTION_IDS_READY));
        noc_semaphore_wait(ready,1); noc_semaphore_set(ready,0);
#if SKV4_PROFILE
        DeviceTimestampedData("SKV_AOV_READY_E",0);
#endif
    }
#endif
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_R_IMPORT_TOP100_B", 0);
#endif
    cb_reserve_back(SKV4_CB_SELECTOR_CHANNEL,1);
    const uint32_t final_page=get_write_ptr(SKV4_CB_SELECTOR_CHANNEL);
    noc_async_read(get_noc_addr(slot*W+shard,handoff),
                   final_page,128*4);
#endif
    noc_async_read_barrier();
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
    auto candidate_ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idp+skip);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_R_FRONTIER_IMPORT_E", 0);
    DeviceTimestampedData("SKV_S2_R_UNION_GATHER_STREAM_B", 0);
#endif
#endif
    /*@S2B_UNION_GATHER_B*/
    uint32_t seen[(LOCAL16+31)/32]={};
    uint32_t total=0,local_count=0,emitted=0;
    constexpr uint32_t LIMIT=LOCAL16<640?LOCAL16:640;
    constexpr uint32_t CAP=((LOCAL16_PAD/32+G-1)/G)*3;
    static_assert(((LIMIT+31)/32+G-1)/G*3<=CAP);
    cb_reserve_back(SKV4_CB_PACKED_CANDIDATES,CAP);
    const uint32_t dst=get_write_ptr(SKV4_CB_PACKED_CANDIDATES);
    for(uint32_t h=0;h<G;++h) {
      auto frontier=lists+h*(LIST_BYTES/4)+1;
      for(uint32_t j=0;j<M16;++j) {
        const uint32_t pos=frontier[j];
        const uint32_t local=pos-shard*LOCAL16;
        if(local>=LOCAL16) continue;
        const uint32_t mask=1u<<(local%32);
        if(seen[local/32]&mask) continue;
        seen[local/32]|=mask;
#if SKV4_CODEGEN_HAS_ID_BITS
        const uint32_t tag=(pos<< SKV4_ID_BITS)|candidate_ids[local];
#else
        const uint32_t tag=(pos<<20)|candidate_ids[local];
#endif
        tags[1+total]=tag;
        const bool mine=(total/32)%G==part;
        ++total;
        if(!mine) continue;
        const uint32_t row=local_count%32;
        const uint32_t tile=dst+(local_count/32)*12288;
        const uint32_t lane=(row/16)*2048+(row%16)*64;
        const uint32_t p2_page=(slot*W+shard)*SKV4_PREFIX_PAGES+local/SKV4_PREFIX_PAGE_KEYS;
        noc_async_read(get_noc_addr(p2_page,p2)+(local%SKV4_PREFIX_PAGE_KEYS)*64,tile+lane,64);
        const uint32_t mp=(slot*W+shard)*MIDDLE_PAGES+local/16;
        noc_async_read(get_noc_addr(mp,middle_cache)+(local%16)*64,tile+4096+lane,64);
#if SKV4_CODEGEN_HAS_ID_BITS
        const uint64_t low=get_noc_addr(slot*context+(tag&SKV4_ID_MASK),r4);
#else
        const uint64_t low=get_noc_addr(slot*context+(tag&((1u<<20)-1)),r4);
#endif
#if SKV4_PROFILE
        if (local_count == 0) {
            const uint32_t tick = *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_L);
            DeviceTimestampedData("SKV_FU_FIRST_LOW", tick);
        }
#endif
        noc_async_read(low,tile+8192+lane,64);
        noc_async_read(low+64,tile+8192+lane+1024,64);
        ++local_count;
        if(local_count%32==0) {
            noc_async_read_barrier();
            cb_push_back(SKV4_CB_PACKED_CANDIDATES,3);
            cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=32;
            cb_push_back(SKV4_CB_LIVE_CONTROL,1);
            emitted=local_count;
        }
      }
    }
    if(local_count>emitted) {
        noc_async_read_barrier();
        cb_push_back(SKV4_CB_PACKED_CANDIDATES,3);
        cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=local_count-emitted;
        cb_push_back(SKV4_CB_LIVE_CONTROL,1);
    }
    tags[0]=total;
    __sync_synchronize();
    cb_push_back(SKV4_CB_SHARED_SCRATCH,1); // metadata immutable; writer can now assemble scores
    cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=0;
    cb_push_back(SKV4_CB_LIVE_CONTROL,1); // explicit EOF, including empty local streams
    // Compute consumes this only after the streaming scorer observes EOF.
    cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=total;
    cb_push_back(SKV4_CB_LIVE_CONTROL,1);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_R_UNION_GATHER_STREAM_E", 0);
#endif
#endif
    /*@S2B_UNION_GATHER_E*/
    if(PROFILE<10 || !owner) return;
#else
    cb_push_back(SKV4_CB_SELECTOR_CHANNEL,1);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_R_IMPORT_TOP100_E", 0);
#endif
#endif
#endif
#endif
    // Every core gets its own head's inherited Stage-1.5 scores. The writer
    // RISC independently builds this core's selector-query tiles while this
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
    // reader gathers records. The shard root alone also reads candidate IDs.
    cb_reserve_back(SKV4_CB_SHARED_SCRATCH, 1);
    const uint32_t scratch = get_write_ptr(SKV4_CB_SHARED_SCRATCH);
#if SKV4_CODEGEN_DYNAMIC
    constexpr uint32_t INPUT_COUNT_OFFSET=4096;
    // Read the whole aligned 16-B count packet. A narrow read at shard*4
    // into an aligned destination violates NoC's matching-low-address rule
    // and made every shard consume shard 0's count on Blackhole.
    noc_async_read(get_noc_addr(slot,input_counts),
                   scratch+INPUT_COUNT_OFFSET,G*sizeof(uint32_t));
    noc_async_read_barrier();
    const uint32_t stage16_count_raw=
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            scratch+INPUT_COUNT_OFFSET+shard*sizeof(uint32_t));
    // Stage 1.5's union producer enforces this bound; retain the clamp as an
    // OOB safety net and validate overflow through its exported count tensor.
    const uint32_t stage16_live=
        stage16_count_raw<LOCAL16?stage16_count_raw:LOCAL16;
    cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=
        stage16_live;
    cb_push_back(SKV4_CB_LIVE_CONTROL,1);
#endif
    // Full 4-bit recomputation: no inherited-score read.
    // Matrix queries are supplied once as native four-head tiles.
    {
    // Every scorer gathers its own 32-key-aligned quarter. Helpers use CB27
    // as a raw ID window; only the root publishes that reader->writer channel.
    uint32_t cand_l1=get_write_ptr(SKV4_CB_SCORE_SCRATCH);
    const uint32_t cand_offset=shard*LOCAL16*4;
    const uint32_t cand_prefix=cand_offset&63;
    if(owner) cb_reserve_back(SKV4_CB_SCORE_SCRATCH,1);
#if SKV4_CODEGEN_DYNAMIC
    if(stage16_live) noc_async_read(
        get_noc_addr(slot,cand)+cand_offset-cand_prefix,
        cand_l1,stage16_live*4+cand_prefix);
#else
    noc_async_read(get_noc_addr(slot,cand)+cand_offset-cand_prefix,
                   cand_l1,LOCAL16*4+cand_prefix);
#endif
    noc_async_read_barrier();
    if(owner) cb_push_back(SKV4_CB_SCORE_SCRATCH,1);
#if SKV4_CODEGEN_DYNAMIC
    const uint32_t stage16_chunks=(stage16_live+31)/32;
    const uint32_t first=part*(stage16_chunks/4)+
        (part<stage16_chunks%4?part:stage16_chunks%4);
    const uint32_t count=stage16_chunks/4+(part<stage16_chunks%4);
#else
    const uint32_t first=part*(CHUNKS/4)+(part<CHUNKS%4?part:CHUNKS%4);
    const uint32_t count=CHUNKS/4+(part<CHUNKS%4);
#endif
    cb_reserve_back(SKV4_CB_PACKED_MIDDLE,count);
    const uint32_t packed_tiles=get_write_ptr(SKV4_CB_PACKED_MIDDLE);
    auto candidates=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_l1+cand_prefix);
    for(uint32_t chunk=first;chunk<first+count;++chunk)
        for(uint32_t row=0;row<32;++row) {
            const uint32_t local=chunk*32+row;
#if SKV4_CODEGEN_DYNAMIC
            const uint32_t source=local<stage16_live?local:0;
#else
            const uint32_t source=local<LOCAL16?local:0;
#endif
            noc_async_read(get_noc_addr(slot*context+candidates[source],r2),
                packed_tiles+(chunk-first)*4096+(row/16)*2048+(row%16)*64,64);
            // Upper bits were fetched by Stage 1.5: L1 only, into unused
            // right faces. Middle persistence below still exports left only.
            const uint32_t prefix_page=(slot*W+shard)*SKV4_PREFIX_PAGES+
                source/SKV4_PREFIX_PAGE_KEYS;
            noc_async_read(get_noc_addr(prefix_page,p2)+
                (source%SKV4_PREFIX_PAGE_KEYS)*64,
                packed_tiles+(chunk-first)*4096+(row/16)*2048+1024+(row%16)*64,64);
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
    // reader gathers records. The shard root alone also reads candidate IDs.
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_R_METADATA_IDS_B", 0);
#endif
#endif
#endif
    cb_reserve_back(SKV4_CB_SHARED_SCRATCH, 1);
    const uint32_t scratch = get_write_ptr(SKV4_CB_SHARED_SCRATCH);
#if SKV4_CODEGEN_DYNAMIC
    constexpr uint32_t INPUT_COUNT_OFFSET=4096;
    // Read the whole aligned 16-B count packet. A narrow read at shard*4
    // into an aligned destination violates NoC's matching-low-address rule
    // and made every shard consume shard 0's count on Blackhole.
    noc_async_read(get_noc_addr(slot,input_counts),
                   scratch+INPUT_COUNT_OFFSET,G*sizeof(uint32_t));
    noc_async_read_barrier();
    const uint32_t stage16_count_raw=
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            scratch+INPUT_COUNT_OFFSET+shard*sizeof(uint32_t));
    // Stage 1.5's union producer enforces this bound; retain the clamp as an
    // OOB safety net and validate overflow through its exported count tensor.
    const uint32_t stage16_live=
        stage16_count_raw<LOCAL16?stage16_count_raw:LOCAL16;
    cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=
        stage16_live;
    cb_push_back(SKV4_CB_LIVE_CONTROL,1);
#endif
    // Full 4-bit recomputation: no inherited-score read.
    // Matrix queries are supplied once as native four-head tiles.
    cb_push_back(SKV4_CB_SHARED_SCRATCH, 1);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_R_METADATA_IDS_E", 0);
    DeviceTimestampedData("SKV_S16_R_SPARSE_GATHER_B", 0);
#endif
#endif
#endif
    {
    // Every scorer gathers its own 32-key-aligned quarter. Helpers use CB27
    // as a raw ID window; only the root publishes that reader->writer channel.
    uint32_t cand_l1=get_write_ptr(SKV4_CB_SCORE_SCRATCH);
    const uint32_t cand_offset=shard*LOCAL16*4;
    const uint32_t cand_prefix=cand_offset&63;
    if(owner) cb_reserve_back(SKV4_CB_SCORE_SCRATCH,1);
#if SKV4_CODEGEN_DYNAMIC
    if(stage16_live) noc_async_read(
        get_noc_addr(slot,cand)+cand_offset-cand_prefix,
        cand_l1,stage16_live*4+cand_prefix);
#else
    noc_async_read(get_noc_addr(slot,cand)+cand_offset-cand_prefix,
                   cand_l1,LOCAL16*4+cand_prefix);
#endif
    noc_async_read_barrier();
    if(owner) cb_push_back(SKV4_CB_SCORE_SCRATCH,1);
#if SKV4_CODEGEN_DYNAMIC
    const uint32_t stage16_chunks=(stage16_live+31)/32;
    const uint32_t first=part*(stage16_chunks/4)+
        (part<stage16_chunks%4?part:stage16_chunks%4);
    const uint32_t count=stage16_chunks/4+(part<stage16_chunks%4);
#else
    const uint32_t first=part*(CHUNKS/4)+(part<CHUNKS%4?part:CHUNKS%4);
    const uint32_t count=CHUNKS/4+(part<CHUNKS%4);
#endif
    cb_reserve_back(SKV4_CB_PACKED_MIDDLE,count);
    const uint32_t packed_tiles=get_write_ptr(SKV4_CB_PACKED_MIDDLE);
    auto candidates=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_l1+cand_prefix);
    for(uint32_t base=0;base<count;base+=1) {
        const uint32_t take=count-base<1?count-base:1;
    for(uint32_t chunk=first+base;chunk<first+base+take;++chunk) {
        for(uint32_t row=0;row<32;++row) {
            const uint32_t local=chunk*32+row;
#if SKV4_CODEGEN_DYNAMIC
            const uint32_t source=local<stage16_live?local:0;
#else
            const uint32_t source=local<LOCAL16?local:0;
#endif
            noc_async_read(get_noc_addr(slot*context+candidates[source],r2),
                packed_tiles+(chunk-first)*4096+(row/16)*2048+(row%16)*64,64);
#else
    cb_wait_front(SKV4_CB_SELECTOR_CHANNEL, 1);
    const uint32_t all_idx = get_read_ptr(SKV4_CB_SELECTOR_CHANNEL);
    Noc noc;
    experimental::CB q_cb(SKV4_CB_ATTENTION_QUERY_ROW), k_cb(SKV4_CB_SPARSE_KV_RECORDS), mask_cb(SKV4_CB_ATTENTION_MASK), ctrl_cb(SKV4_CB_LIVE_CONTROL);
    experimental::CB req_cb(SKV4_CB_QUERY_COEFFICIENTS), ack_cb(SKV4_CB_TOPK_MERGE_SCRATCH);
    constexpr uint32_t H = 32, S = G, KCHUNK = 128;
    constexpr uint32_t q_row_bytes = 2 * D * 2;
    constexpr uint32_t kv_row_bytes = 2 * D;
    const uint32_t kv_page_offset = slot * context;
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2)
#if !SKV4_CODEGEN_WAIT_FRONTIER
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_R_QUERY_B", 0);
#endif
#endif
#endif
#if !SKV4_CODEGEN_WAIT_FRONTIER
    q_cb.reserve_back(H);
    // The current invocation's compact Q is persistent L1. Construct the
    // sparse-SDPA diagonal/zero layout locally, not via padded GDDR reads.
    {
        auto zero = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(q_cb.get_write_ptr());
        for (uint32_t i = 0; i < H * q_row_bytes / 4; ++i) zero[i] = 0;
        noc.async_read(aq, q_cb, D * 2,
                       {.page_id = slot * G + shard},
                       {.offset_bytes = shard * q_row_bytes + D * 2});
#endif
#endif
#endif
#if !SKV4_CODEGEN_WAIT_FRONTIER
        }
#endif
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
    noc_async_read_barrier();
    cb_push_back(SKV4_CB_PACKED_MIDDLE,count);
    // Persist useful faces only: each contains sixteen 64-B key records.
    // Distributed 1-KiB pages avoid both root aggregation and native padding.
    // Program A ends only after these writes complete; CB26 isn't recycled.
#ifdef SKV4_RETAIN_MIDDLE
    for(uint32_t face=0;face<count*2;++face) {
        const uint32_t page=(slot*W+shard)*MIDDLE_PAGES+first*2+face;
        noc_async_write(packed_tiles+face*2048,get_noc_addr(page,middle_cache),1024);
    }
    noc_async_write_barrier();
#endif
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
        // Consecutive L1 prefixes: one read per sixteen-key face, not per key.
        static_assert(SKV4_PREFIX_PAGE_KEYS%16==0);
        for(uint32_t face=0;face<2;++face) {
            const uint32_t local=chunk*32+face*16;
#if SKV4_CODEGEN_DYNAMIC
            const uint32_t valid=local<stage16_live?((stage16_live-local<16)?stage16_live-local:16):0;
#else
            const uint32_t valid=local<LOCAL16?((LOCAL16-local<16)?LOCAL16-local:16):0;
#endif
            const uint32_t dst=packed_tiles+(chunk-first)*4096+face*2048+1024;
            if(valid) {
                const uint32_t pg=(slot*W+shard)*SKV4_PREFIX_PAGES+local/SKV4_PREFIX_PAGE_KEYS;
                noc_async_read(get_noc_addr(pg,p2)+(local%SKV4_PREFIX_PAGE_KEYS)*64,dst,valid*64);
#else
#if !SKV4_CODEGEN_WAIT_FRONTIER
    noc.async_read_barrier();
    q_cb.push_back(H);
#endif
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2)
#if !SKV4_CODEGEN_WAIT_FRONTIER
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_R_QUERY_E", 0);
#endif
#endif
#endif
    volatile tt_l1_ptr uint32_t* idx =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(all_idx);
    ctrl_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ctrl =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
    ctrl[sparse_sdpa::control_message::ACTIVE_CHUNKS] = 1;
    ctrl[sparse_sdpa::control_message::VALID_KEYS] = 100;
    ctrl_cb.push_back(1);
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2)
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_R_K_GATHER_JOIN_B", 0);
#endif
#endif
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
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2)
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_R_K_GATHER_JOIN_E", 0);
#endif
#endif
    mask_cb.reserve_back(1);
    volatile tt_l1_ptr uint16_t* mask =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_cb.get_write_ptr());
    for (uint32_t col = 0; col < 32; ++col) {
        const uint32_t off = col < 16 ? col : 256 + col - 16;
        mask[off] = col >= 4 ? neg_inf_bf16 : uint16_t(0);
#endif
#endif
    }
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
    cb_push_back(SKV4_CB_SHARED_SCRATCH, 1);
    if constexpr (PROFILE == 1) return;
    // Stop 2 is retained as an ABI-compatible profiling point.  The removed
    // BFP4 conversion has no replacement stage, so stops 1 and 2 should match.
    if constexpr (PROFILE == 2) return;
    if constexpr (PROFILE <= 3) {
        return;
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
            // Match the original source0 padding; never consume stale tails.
            for(uint32_t j=valid;j<16;++j)
                noc_async_read(get_noc_addr((slot*W+shard)*SKV4_PREFIX_PAGES,p2),dst+j*64,64);
#else
    mask_cb.push_back(1);
    cb_pop_front(SKV4_CB_SELECTOR_CHANNEL, 1);
#endif
#endif
    }
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
    if constexpr (PROFILE < 6) {
        return;
    }
    if(!owner) return;
    volatile tt_l1_ptr uint32_t* ready=
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_IDS_READY));
    noc_semaphore_wait(ready,1); noc_semaphore_set(ready,0);
    cb_wait_front(SKV4_CB_SELECTOR_CHANNEL,1);
    volatile tt_l1_ptr uint32_t* handoff=
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_SELECTOR_CHANNEL));
    const uint32_t count=handoff[0];
    constexpr uint32_t CAP=LOCAL16_PAD/32*3;
    if constexpr(PROFILE>=8) cb_reserve_back(SKV4_CB_PACKED_CANDIDATES,CAP);
    const uint32_t dst=get_write_ptr(SKV4_CB_PACKED_CANDIDATES);
    const uint32_t retained=get_write_ptr(SKV4_CB_PACKED_MIDDLE); // full Stage-1.6 ring wrapped
    const uint32_t high=scratch+16384;
    for(uint32_t j=0;j<count;++j) {
#if SKV4_CODEGEN_HAS_ID_BITS
        const uint32_t tag=handoff[1+j], local=(tag>> SKV4_ID_BITS)-shard*LOCAL16;
#else
        const uint32_t tag=handoff[1+j], local=(tag>>20)-shard*LOCAL16;
#endif
        const uint32_t tile=j/32, row=j%32;
        const uint32_t lane=(row/16)*2048+(row%16)*64;
        const uint32_t oldlane=(local/32)*4096+((local%32)/16)*2048+(local%16)*64;
        noc_async_read(get_noc_addr(my_x[0],my_y[0],high+local*64),
                       dst+tile*12288+lane,64);
        noc_async_read(get_noc_addr(my_x[0],my_y[0],retained+oldlane),
                       dst+tile*12288+4096+lane,64);
#if SKV4_CODEGEN_HAS_ID_BITS
        const uint64_t src=get_noc_addr(slot*context+(tag&SKV4_ID_MASK),r4);
#else
        const uint64_t src=get_noc_addr(slot*context+(tag&((1u<<20)-1)),r4);
#endif
        noc_async_read(src,dst+tile*12288+8192+lane,64);
        noc_async_read(src+64,dst+tile*12288+8192+lane+1024,64);
    }
    noc_async_read_barrier();
    cb_pop_front(SKV4_CB_SELECTOR_CHANNEL,1);
    if constexpr(PROFILE<=7) return;
    cb_reserve_back(SKV4_CB_LIVE_CONTROL,1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL))=count;
    cb_push_back(SKV4_CB_LIVE_CONTROL,1); cb_push_back(SKV4_CB_PACKED_CANDIDATES,CAP);
    if constexpr(PROFILE<10) return;
    // Final sparse-attention input preparation.  One worker owns one query
    // head, so only one diagonal query and one top-100 list are active.
    cb_wait_front(SKV4_CB_SELECTOR_CHANNEL, 1);
    const uint32_t all_idx = get_read_ptr(SKV4_CB_SELECTOR_CHANNEL);
    Noc noc;
    experimental::CB q_cb(SKV4_CB_ATTENTION_QUERY_ROW), k_cb(SKV4_CB_SPARSE_KV_RECORDS), mask_cb(SKV4_CB_ATTENTION_MASK), ctrl_cb(SKV4_CB_LIVE_CONTROL);
    experimental::CB req_cb(SKV4_CB_QUERY_COEFFICIENTS), ack_cb(SKV4_CB_TOPK_MERGE_SCRATCH);
    constexpr uint32_t H = 32, S = G, KCHUNK = 128;
    constexpr uint32_t q_row_bytes = 2 * D * 2;
    constexpr uint32_t kv_row_bytes = 2 * D;
    const uint32_t kv_page_offset = slot * context;
    q_cb.reserve_back(H);
    // Match the compute ABI while materializing only one live head row.
    auto zero = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(q_cb.get_write_ptr());
    for (uint32_t i = 0; i < H * q_row_bytes / 4; ++i) zero[i] = 0;
    noc.async_read(aq, q_cb, D * 2,
                   {.page_id = slot * G + shard},
                   {.offset_bytes = shard * q_row_bytes + D * 2});
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
    cb_pop_front(SKV4_CB_SELECTOR_CHANNEL, 1);
}
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
    }
        noc_async_read_barrier();
        cb_push_back(SKV4_CB_PACKED_MIDDLE,take);
    }
    // Persist useful faces only: each contains sixteen 64-B key records.
    // Distributed 1-KiB pages avoid both root aggregation and native padding.
    // Program A ends only after these writes complete; CB26 isn't recycled.
#ifdef SKV4_RETAIN_MIDDLE
    for(uint32_t face=0;face<count*2;++face) {
        const uint32_t page=(slot*W+shard)*MIDDLE_PAGES+first*2+face;
        noc_async_write(packed_tiles+face*2048,get_noc_addr(page,middle_cache),1024);
    }
    noc_async_write_barrier();
#endif
    }
#if SKV4_PROFILE
    {
        const uint32_t tick = *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
            RISCV_DEBUG_REG_WALL_CLOCK_L);
        DeviceTimestampedData("SKV_FU_MIDDLE", tick);
    }
#endif
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_R_SPARSE_GATHER_E", 0);
#endif
#endif
#endif
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
    if(!owner) return;
    auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_IDS_READY));
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_R_EXPORT_L1_B", 0);
#endif
#endif
#endif
    noc_semaphore_wait(ready,1); noc_semaphore_set(ready,0);
    cb_wait_front(SKV4_CB_SELECTOR_CHANNEL,1);
    const uint32_t meta=get_read_ptr(SKV4_CB_SELECTOR_CHANNEL);
    const uint64_t target=get_noc_addr(slot*W+shard,handoff);
    noc_async_write(meta,target,META_BYTES);
    noc_async_write_barrier();
    // Middle bits are already persisted by the distributed scorers. Only
    // selection metadata needs an additional export here.
    cb_pop_front(SKV4_CB_SELECTOR_CHANNEL,1);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_R_EXPORT_L1_E", 0);
#endif
#endif
#endif
}
#endif
#endif
"""
        )
    )


def writer(
    *,
    role,
    dynamic,
    phase,
    profile,
    id_bits=None,
    prefix_page_keys=64,
    prefix_pages=1,
    retain_middle=False,
    debug=False,
):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if role not in (None, "stage16", "stage2", "attention"):
        raise ValueError("unknown refinement role")
    if phase not in ("full", "input", "copy", "decode", "pack", "matmul"):
        raise ValueError("unknown refinement phase")
    if profile and role is None:
        raise ValueError("native markers require a split role")
    if profile and phase != "full":
        raise ValueError("native markers require the full phase")
    if profile and role == "stage16" and (not dynamic):
        raise ValueError("Stage1.6 markers require dynamic inputs")
    if (role, phase) in [
        ("attention", "copy"),
        ("attention", "decode"),
        ("attention", "input"),
        ("attention", "matmul"),
        ("attention", "pack"),
        ("stage2", "input"),
    ]:
        raise ValueError("unsupported diagnostic role/phase combination")
    return (
        cpp_defines(CB)
        + semaphore_defines(SEM)
        + refinement_defines("writer", role=role, dynamic=dynamic)
        + profile_defines(
            profile=profile,
            role=role,
            phase=phase,
            dynamic=dynamic,
            has_id_bits=id_bits is not None,
            retain_middle=bool(retain_middle),
            debug=bool(debug),
        )
        + (
            r"""#if SKV4_PROFILE
// skv4-fine-control-v3
#include "tools/profiler/kernel_profiler.hpp"
#endif
#if SKV4_CODEGEN_HAS_ID_BITS
#define SKV4_ID_BITS """
            + str(0 if id_bits is None else id_bits)
            + r"""
#define SKV4_ID_MASK ((1u << SKV4_ID_BITS) - 1u)
#endif
#include <cstdint>
#include <algorithm>
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
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
__attribute__((noinline,noclone)) void heapify(uint16_t* hv, uint32_t* hi) {
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
__attribute__((noinline,noclone)) void heapify(uint16_t* hv, uint32_t* hi) {
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
FORCE_INLINE void heapify(uint16_t* hv, uint32_t* hi) {
#else
__attribute__((noinline,noclone)) void heapify(uint16_t* hv, uint32_t* hi) {
#endif
#endif
#endif
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
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
__attribute__((noinline,noclone)) void heap_insert(uint16_t key, uint32_t idx,
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
__attribute__((noinline,noclone)) void heap_insert(uint16_t key, uint32_t idx,
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
FORCE_INLINE void heap_insert(uint16_t key, uint32_t idx,
#else
__attribute__((noinline,noclone)) void heap_insert(uint16_t key, uint32_t idx,
#endif
#endif
#endif
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
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
__attribute__((optimize("Os"))) void kernel_main() {
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
__attribute__((optimize("Os"))) void kernel_main() {
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
void kernel_main() {
#else
__attribute__((optimize("Os"))) void kernel_main() {
#endif
#endif
#endif
    constexpr uint32_t N16 = get_compile_time_arg_val(SKV4_ARG_RF_CT_N16);
    constexpr uint32_t W = get_compile_time_arg_val(SKV4_ARG_RF_CT_WORKERS);
    constexpr uint32_t G = get_compile_time_arg_val(SKV4_ARG_RF_CT_GROUP);
    constexpr uint32_t M16 = get_compile_time_arg_val(SKV4_ARG_RF_CT_M16);
    constexpr uint32_t KLOCAL = get_compile_time_arg_val(SKV4_ARG_RF_CT_LOCAL_M);
    constexpr uint32_t K = get_compile_time_arg_val(SKV4_ARG_RF_CT_TOPK);
    constexpr uint32_t D = get_compile_time_arg_val(SKV4_ARG_RF_CT_D);
    constexpr bool DEBUG_INDICES = get_compile_time_arg_val(SKV4_ARG_RF_CT_DEBUG) != 0;
    constexpr uint32_t PROFILE = get_compile_time_arg_val(SKV4_ARG_RF_CT_PROFILE);
    constexpr uint32_t PARTS = 4;
    constexpr uint32_t LOCAL16 = N16 / W;
    constexpr uint32_t LOCAL16_PAD = ((LOCAL16 + 31) / 32) * 32;
    constexpr uint32_t M16_PAD = ((M16 + 31) / 32) * 32;
    constexpr uint32_t INHERITED_OFFSET = 0;
    constexpr uint32_t TOPK_INPUT_OFFSET = 6144;
    constexpr uint32_t STAGE2_QUERY_OFFSET = 8192;
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
    constexpr uint32_t MERGE_K=M16<=512?512:1024;
    constexpr uint32_t ROW_WORDS = MERGE_K;
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
    constexpr uint32_t MERGE_K=M16<=512?512:1024;
    constexpr uint32_t ROW_WORDS = MERGE_K;
#else
    constexpr uint32_t ROW_WORDS = KLOCAL * 2;
#endif
#endif
    constexpr uint32_t ROW_BYTES = ROW_WORDS * 4;
    constexpr uint32_t EXCHANGE_SEG_BYTES = ROW_BYTES;
#if SKV4_CODEGEN_HAS_ID_BITS
    constexpr uint32_t ID_MASK = SKV4_ID_MASK;
#else
    constexpr uint32_t ID_MASK = (1u << 20) - 1;
#endif
    constexpr bool PER_HEAD_INPUT = get_compile_time_arg_val(SKV4_ARG_RF_CT_PER_HEAD_INPUT) != 0;
    constexpr auto oa = TensorAccessorArgs<SKV4_ARG_RF_CT_ACCESSORS>();
    constexpr auto kva = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto dia = TensorAccessorArgs<kva.next_compile_time_args_offset()>();
    constexpr auto dsa = TensorAccessorArgs<dia.next_compile_time_args_offset()>();
    constexpr auto sqa = TensorAccessorArgs<dsa.next_compile_time_args_offset()>();
    static_assert(N16 % W == 0);
    static_assert(W >= G);
    static_assert(K == 100);
    static_assert(KLOCAL <= LOCAL16);
    static_assert(KLOCAL * W >= M16);
    const uint32_t out_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_OUTPUT);
    const uint32_t kv_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_KV);
    const uint32_t slot = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SLOT);
    const uint32_t context = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_CONTEXT);
    const uint32_t shard = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SHARD);
    const uint32_t part = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_PART);
    const bool owner = part == 0;
    const uint32_t debug_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_DEBUG_INDICES);
    const uint32_t debug_score_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_DEBUG_SCORES);
    const uint32_t sq_addr = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_SQ);
    const uint32_t root_x = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROOT_X);
    const uint32_t root_y = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROOT_Y);
    const uint32_t row_start_x = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_START_X);
    const uint32_t row_start_y = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_START_Y);
    const uint32_t row_end_x = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_END_X);
    const uint32_t row_end_y = get_arg_val<uint32_t>(SKV4_ARG_RF_RT_ROW_END_Y);
    const auto out = TensorAccessor(oa, out_addr, D * 2);
    const auto kv = TensorAccessor(kva, kv_addr, 2 * D);
    const auto debug = TensorAccessor(dia, debug_addr, padded_topk * 4);
    const auto debug_scores = TensorAccessor(dsa, debug_score_addr, M16_PAD * 16);
    const auto sq = TensorAccessor(sqa, sq_addr, D * 2);
    constexpr auto uqa=TensorAccessorArgs<sqa.next_compile_time_args_offset()>();
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE)
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16)
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
    constexpr auto hfa=TensorAccessorArgs<uqa.next_compile_time_args_offset()>();
    constexpr uint32_t META_BYTES=(LOCAL16+1)*4<=4096?
        4096:((LOCAL16+1)*4+63)/64*64;
    constexpr uint32_t HANDOFF_BYTES=META_BYTES;
#endif
#endif
#endif
    const auto uq=TensorAccessor(uqa,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_UNION_QUERY),2048);
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE)
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16)
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
    const auto handoff=TensorAccessor(
        hfa,get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HANDOFF),HANDOFF_BYTES);
#endif
#endif
#endif
    Noc noc;
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
    // Reuse the existing union query: first eight tiles are the four heads,
    // plane-major coordinates, with x16 folded into exact BF16 exponents.
    if constexpr(PROFILE>=3) {
        cb_reserve_back(SKV4_CB_MATRIX_QUERY,8);
        const uint32_t qp=get_write_ptr(SKV4_CB_MATRIX_QUERY);
        for(uint32_t t=0;t<8;++t)
            noc_async_read(get_noc_addr(slot*16+t,uq),qp+t*2048,2048);
        noc_async_read_barrier(); cb_push_back(SKV4_CB_MATRIX_QUERY,8);
    }
    cb_wait_front(SKV4_CB_SHARED_SCRATCH, 1);
    const uint32_t shared_scratch = get_read_ptr(SKV4_CB_SHARED_SCRATCH);
#if SKV4_CODEGEN_DYNAMIC
    constexpr uint32_t INPUT_COUNT_OFFSET=4096;
    const uint32_t stage16_live=
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            shared_scratch+INPUT_COUNT_OFFSET+shard*sizeof(uint32_t));
#endif
    if constexpr (PROFILE == 2 && DEBUG_INDICES) if (shard == 0) {
        // Addressing diagnostic: export exactly what this core's reader put
        // in the inherited-score window, before scoring or scratch reuse.
        noc_async_write(
            shared_scratch + INHERITED_OFFSET,
            get_noc_addr(slot * G + part, debug_scores), LOCAL16 * 2);
        noc_async_write_barrier();
    }
    if constexpr (PROFILE < 3) {
        cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
        return;
    }
#if SKV4_CODEGEN_DYNAMIC
    constexpr uint32_t MAX_CHUNKS=LOCAL16_PAD/32;
    constexpr uint32_t QUARTER_BYTES=((MAX_CHUNKS+3)/4)*64;
    const uint32_t stage16_chunks=(stage16_live+31)/32;
    const uint32_t first=part*(stage16_chunks/4)+
        (part<stage16_chunks%4?part:stage16_chunks%4);
    const uint32_t count=stage16_chunks/4+(part<stage16_chunks%4);
#else
    constexpr uint32_t CHUNKS=LOCAL16_PAD/32;
    constexpr uint32_t QUARTER_BYTES=((CHUNKS+3)/4)*64;
    const uint32_t first=part*(CHUNKS/4)+(part<CHUNKS%4?part:CHUNKS%4);
    const uint32_t count=CHUNKS/4+(part<CHUNKS%4);
#endif
    // CB2 is raw writer-local staging, not a FIFO. No counters are advanced.
    const uint32_t rows=get_write_ptr(SKV4_CB_SPARSE_KV_RECORDS);
    for(uint32_t chunk=0;chunk<count;++chunk) {
        cb_wait_front(SKV4_CB_SCORE_TILES,1);
        auto tile=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(SKV4_CB_SCORE_TILES));
        for(uint32_t h=0;h<4;++h) {
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rows+h*QUARTER_BYTES);
            for(uint32_t c=0;c<32;++c)
                dst[chunk*32+c]=tile[h*16+(c<16?c:256+c-16)];
        }
        cb_pop_front(SKV4_CB_SCORE_TILES,1);
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
    if(owner) cb_wait_front(SKV4_CB_SCORE_SCRATCH,1);
    const uint32_t cand_prefix=(shard*LOCAL16*4)&63;
    auto candidates=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_SCORE_SCRATCH)+cand_prefix);
    uint32_t receive = 0;
    if (owner) {
        // Raw NoC scratch only: no FIFO reservation. Stage II does not use ones.
        receive = get_write_ptr(SKV4_CB_EXCHANGE_TILES);
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
    if constexpr (PER_HEAD_INPUT) cb_pop_front(SKV4_CB_SCORE_SCRATCH, 1);
    // Present the 928 scores as one row-major BF16 K=1024 page.  This copy is
    // deliberately local and contiguous; the expensive O(928 log 160) scalar
    // heap has been replaced by one SFPU TopK-XL sort in the compute thread.
    // Padding is -inf and can never survive.
    cb_reserve_back(SKV4_CB_SELECTOR_CHANNEL, 1);
    volatile tt_l1_ptr uint16_t* topk_pages =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_SELECTOR_CHANNEL));
    const uint32_t first_topk = LOCAL16_PAD < 1024 ? LOCAL16_PAD : 1024;
    for (uint32_t j = 0; j < first_topk; ++j) {
#if SKV4_CODEGEN_DYNAMIC
        topk_pages[j] = j < stage16_live ? topk_scores[j] : neg_inf_bf16;
#else
        topk_pages[j] = j < LOCAL16 ? topk_scores[j] : neg_inf_bf16;
#endif
    }
    cb_push_back(SKV4_CB_SELECTOR_CHANNEL, 1);
#if SKV4_CODEGEN_DYNAMIC
    if (stage16_live > 1024) {
#else
    if constexpr (LOCAL16_PAD > 1024) {
#endif
        cb_reserve_back(SKV4_CB_ATTENTION_NEG_INF, 1);
        volatile tt_l1_ptr uint16_t* second_topk =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_ATTENTION_NEG_INF));
        for (uint32_t j = 1024; j < LOCAL16_PAD; ++j) {
            second_topk[j - 1024] =
#if SKV4_CODEGEN_DYNAMIC
                j < stage16_live ? topk_scores[j] : neg_inf_bf16;
#else
                j < LOCAL16 ? topk_scores[j] : neg_inf_bf16;
#endif
        }
        cb_push_back(SKV4_CB_ATTENTION_NEG_INF, 1);
    }
    if constexpr(PROFILE==3) {
        // Export the complete recomputed 4-bit score consumed by TopK.
        // This catches padding and alignment
        // errors that a residual-only diagnostic cannot see.
        if(shard==0) {
            noc_async_write(reinterpret_cast<uint32_t>(topk_pages),
#if SKV4_CODEGEN_DYNAMIC
                get_noc_addr(slot*G+part,debug_scores),stage16_live*2);
#else
                get_noc_addr(slot*G+part,debug_scores),LOCAL16*2);
#endif
            noc_async_write_barrier();
        }
        cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return;
    }
#if SKV4_CODEGEN_DYNAMIC
    cb_wait_front(SKV4_CB_LOCAL_FRONTIERS,stage16_live>1024?2:1);
#else
    cb_wait_front(SKV4_CB_LOCAL_FRONTIERS,LOCAL16_PAD>1024?2:1);
#endif
    volatile tt_l1_ptr uint32_t* local_indices =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_LOCAL_FRONTIERS));
    // Merge the two independently sorted K=1024 halves.  Only the first
    // KLOCAL ranks from either half can enter this shard's exported frontier.
    // First shard descends and subsequent shards ascend, including -inf
    // padding, to form each bitonic merge pair.
    constexpr uint32_t BITS=MERGE_K==512?5:6;
    auto half0=local_indices;
    auto half1=local_indices+1024;
    uint32_t i0=0,i1=0;
    for(uint32_t rank=0;rank<MERGE_K;++rank) {
        uint32_t word=0xff800000u;
        if(rank<KLOCAL) {
            const uint32_t l0=(i0>>6)+((i0&63u)<<4);
            const uint32_t l1=(i1>>6)+((i1&63u)<<4);
            const uint32_t w0=i0<1024?half0[l0]:0xff800000u;
#if SKV4_CODEGEN_DYNAMIC
            const uint32_t w1=(stage16_live>1024 && i1<stage16_live-1024)?
#else
            const uint32_t w1=(LOCAL16_PAD>1024 && i1<LOCAL16_PAD-1024)?
#endif
                half1[l1]:0xff800000u;
            const uint16_t s0=ordered_bf16(static_cast<uint16_t>(w0>>16));
            const uint16_t s1=ordered_bf16(static_cast<uint16_t>(w1>>16));
            const uint32_t packed=s1>s0?w1:w0;
            if(s1>s0) ++i1; else ++i0;
            const uint32_t local=decode_topk_xl_k1024(packed);
            word=(packed&0xffff0000u)|(
                PROFILE==4 ? (packed&0xffffu) : (shard*LOCAL16+local));
        }
        const uint32_t out_rank=shard==0?rank:MERGE_K-1-rank;
        local_row[((out_rank&(MERGE_K/16-1))<<4)|(out_rank>>BITS)]=word;
    }
#if SKV4_CODEGEN_DYNAMIC
    cb_pop_front(SKV4_CB_LOCAL_FRONTIERS,stage16_live>1024?2:1);
#else
    cb_pop_front(SKV4_CB_LOCAL_FRONTIERS,LOCAL16_PAD>1024?2:1);
#endif
    // Send straight to the root responsible for this query head. There is no
    // gather-at-local-root then retransmit phase. Receive slots are disjoint.
    const uint32_t head_x=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*part);
    const uint32_t head_y=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*part);
    noc_async_write(reinterpret_cast<uint32_t>(local_row),
        get_noc_addr(head_x,head_y,get_write_ptr(SKV4_CB_TOPK_MERGE_SCRATCH)+shard*4096),ROW_BYTES);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(head_x,head_y,get_semaphore(SKV4_SEM_PACKED_FRONTIERS_READY)),1);
    noc.async_atomic_barrier();
    if(owner) {
        cb_pop_front(SKV4_CB_SCORE_SCRATCH,1);
        auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_PACKED_FRONTIERS_READY));
        noc_semaphore_wait(ready,W); noc_semaphore_set(ready,0);
    }
    if constexpr (PROFILE == 4) {
        if(owner) {
            auto pages=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_write_ptr(SKV4_CB_TOPK_MERGE_SCRATCH));
            auto compact=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                shared_scratch);
            constexpr uint32_t DBITS=MERGE_K==512?5:6;
            for(uint32_t src=0;src<4;++src) for(uint32_t rank=0;rank<KLOCAL;++rank) {
                const uint32_t stored=src==0?rank:MERGE_K-1-rank;
                const uint32_t lane=((stored&(MERGE_K/16-1))<<4)|(stored>>DBITS);
                compact[src*KLOCAL+rank]=pages[src*1024+lane];
            }
            noc_async_write(shared_scratch,
                get_noc_addr(slot*G+shard,debug_scores),4*KLOCAL*4);
            noc_async_write_barrier();
        }
        cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
        return;
    }
    uint32_t union_count=0; uint32_t* union_ids=nullptr;
    uint16_t hv16[M16];
    uint32_t hi16[M16];
    if (owner) {
        // Remote writers filled four native-layout, descending logical
        // frontiers. Shard zero is stored descending; the other three are
        // physically reversed for the removed TopK-XL bitonic merge. Read
        // each in logical rank order and perform only 3*M16 comparisons.
        constexpr uint32_t RBITS=MERGE_K==512?5:6;
        auto p0=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_TOPK_MERGE_SCRATCH));
        const auto fetch=[&](uint32_t src,uint32_t rank) -> uint32_t {
            if(rank>=KLOCAL) return 0xff800000u;
            const uint32_t stored=src==0?rank:MERGE_K-1-rank;
            const uint32_t lane=((stored&(MERGE_K/16-1))<<4)|(stored>>RBITS);
            return p0[src*1024+lane];
        };
        uint32_t i0=0,i1=0,i2=0,i3=0;
        uint32_t w0=fetch(0,0),w1=fetch(1,0),w2=fetch(2,0),w3=fetch(3,0);
        uint16_t s0=ordered_bf16(w0>>16),s1=ordered_bf16(w1>>16);
        uint16_t s2=ordered_bf16(w2>>16),s3=ordered_bf16(w3>>16);
        for(uint32_t out=0;out<M16;++out) {
            // Keep all four frontier heads in registers. Only the winner's
            // cursor/load/ordered conversion changes; ties prefer lower src.
            const bool a=s1>s0,b=s3>s2;
            const uint16_t sa=a?s1:s0,sb=b?s3:s2;
            const uint32_t winner=sb>sa?(b?3:2):(a?1:0);
            if(winner==0) {
                hi16[out]=w0&0xffffu; w0=fetch(0,++i0); s0=ordered_bf16(w0>>16);
            } else if(winner==1) {
                hi16[out]=w1&0xffffu; w1=fetch(1,++i1); s1=ordered_bf16(w1>>16);
            } else if(winner==2) {
                hi16[out]=w2&0xffffu; w2=fetch(2,++i2); s2=ordered_bf16(w2>>16);
            } else {
                hi16[out]=w3&0xffffu; w3=fetch(3,++i3); s3=ordered_bf16(w3>>16);
            }
        }
        if constexpr (PROFILE == 5) {
            volatile tt_l1_ptr uint32_t* probe =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive);
            for (uint32_t j = 0; j < M16; ++j) probe[j] = hi16[j]; // stop5 diagnostic: candidate position, not token ID
            noc_async_write(
                receive, get_noc_addr(slot * G + shard, debug_scores), M16 * 4);
            noc_async_write_barrier();
            cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
            return;
        }
        // All owners publish tagged top-M frontiers once, in bulk. Each
        // owner keeps only union entries belonging to its original key shard.
        constexpr uint32_t UNION_OFFSET=16384+LOCAL16*64;
        const uint32_t exchange=shared_scratch+UNION_OFFSET;
        volatile tt_l1_ptr uint32_t* lists=
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(exchange);
        for(uint32_t j=0;j<M16;++j) lists[shard*M16+j]=hi16[j];
        for(uint32_t dest=0;dest<G;++dest) if(dest!=shard) {
            const uint32_t x=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*dest);
            const uint32_t y=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*dest);
            noc_async_write(exchange+shard*M16*4,
                get_noc_addr(x,y,exchange+shard*M16*4),M16*4);
        }
        noc_async_write_barrier();
        for(uint32_t dest=0;dest<G;++dest) if(dest!=shard)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*dest),
                get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*dest),get_semaphore(SKV4_SEM_PEER_EXCHANGE_READY)),1);
        noc_async_atomic_barrier();
        auto union_ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_PEER_EXCHANGE_READY));
        noc_semaphore_wait(union_ready,G-1); noc_semaphore_set(union_ready,0);
        uint32_t seen[(LOCAL16+31)/32]={};
        cb_reserve_back(SKV4_CB_SELECTOR_CHANNEL,1);
        auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_SELECTOR_CHANNEL));
        union_ids=reinterpret_cast<uint32_t*>(exchange+G*M16*4);
        for(uint32_t j=0;j<G*M16;++j) {
            const uint32_t pos=lists[j];
            // Unsigned subtraction also rejects earlier shards by wraparound.
            // No division/modulo in the 4*M16-entry ownership scan.
            const uint32_t local=pos-shard*LOCAL16;
            if(local>=LOCAL16) continue;
            const uint32_t word=local>>5, mask=1u<<(local&31u);
            if(seen[word]&mask) continue;
            seen[word]|=mask;
            // Original key shard owns the token-ID map. Restore the unchanged
            // Stage-II (candidate position << SKV4_ID_BITS | token ID) handoff here.
            const uint32_t id=candidates[local];
            union_ids[union_count]=id;
#if SKV4_CODEGEN_HAS_ID_BITS
            dst[1+union_count++]=(pos<< SKV4_ID_BITS)|id;
#else
            dst[1+union_count++]=(pos<<20)|id;
#endif
        }
        dst[0]=union_count;
        cb_push_back(SKV4_CB_SELECTOR_CHANNEL,1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_IDS_READY)),1);
        if constexpr(PROFILE>=8) {
            cb_reserve_back(SKV4_CB_MATRIX_QUERY,16);
            const uint32_t qp=get_write_ptr(SKV4_CB_MATRIX_QUERY);
            for(uint32_t t=0;t<16;++t)
                noc_async_read(get_noc_addr(slot*16+t,uq),qp+t*2048,2048);
            noc_async_read_barrier(); cb_push_back(SKV4_CB_MATRIX_QUERY,16);
        }
    }
    if constexpr(PROFILE<=7) { cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return; }
    if(!owner) { cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return; }
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL)
    if(owner) cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return; // diagnostic: no score producer
#endif
    constexpr uint32_t HP=16384+LOCAL16*64+G*M16*4+LOCAL16*4;
    auto hv_all=reinterpret_cast<uint16_t*>(shared_scratch+HP);
    auto hi_all=reinterpret_cast<uint32_t*>(shared_scratch+HP+G*K*2);
    for(uint32_t j=0;j<G*K;++j) { hv_all[j]=ordered_bf16(neg_inf_bf16); hi_all[j]=0xffffffffu; }
    for(uint32_t chunk=0;chunk<(union_count+31)/32;++chunk) {
        cb_wait_front(SKV4_CB_SCORE_TILES,1);
        const uint32_t base=get_read_ptr(SKV4_CB_SCORE_TILES);
        if constexpr(PROFILE>=9) {
            const uint32_t begin=chunk*32, end=std::min(begin+32,union_count);
            for(uint32_t h=0;h<G;++h) for(uint32_t j=begin;j<end;++j) {
                const uint16_t value=score_at(base,j-begin,h);
                heap_insert<K>(ordered_bf16(value),union_ids[j],hv_all+h*K,hi_all+h*K);
            }
        }
        cb_pop_front(SKV4_CB_SCORE_TILES,1);
    }
    if constexpr(PROFILE==8) { cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return; }
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
        } else noc_async_write(src,get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*h),
                   get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*h),receive+shard*ROW_BYTES),ROW_BYTES);
    }
    noc_async_write_barrier();
    for(uint32_t h=0;h<G;++h) if(h!=shard)
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*h),
            get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*h),get_semaphore(SKV4_SEM_STAGE2_FRONTIERS_READY)),1);
    noc_async_atomic_barrier();
    auto done=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_FRONTIERS_READY));
    noc_semaphore_wait(done,G-1); noc_semaphore_set(done,0);
    uint16_t hv[K]; uint32_t hi[K];
    for(uint32_t j=0;j<K;++j) { hv[j]=ordered_bf16(neg_inf_bf16); hi[j]=0xffffffffu; }
    for(uint32_t w=0;w<W;++w) {
        auto p=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive+w*ROW_BYTES);
        for(uint32_t j=0;j<K;++j) heap_insert<K>(p[2*j],p[2*j+1],hv,hi);
    }
    if constexpr (PROFILE == 9) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive)[0] = hi[0];
        cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
        return;
    }
    // CB30 remains writer-owned across all three sequential payloads: local
    // TopK input, global top-400 IDs, then final top-100 IDs. CB17 stays
    // compute->writer only for the TopK-XL result.
    cb_reserve_back(SKV4_CB_SELECTOR_CHANNEL, 1);
    volatile tt_l1_ptr uint32_t* final_ids =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_SELECTOR_CHANNEL));
    for (uint32_t j = 0; j < padded_topk; ++j) {
        final_ids[j] = j < K ? hi[j] : 0xffffffffu;
    }
    if constexpr (DEBUG_INDICES) {
        final_ids[K]=union_count;
        noc_async_write(reinterpret_cast<uint32_t>(final_ids),
                        get_noc_addr(slot * G + shard, debug), padded_topk * 4);
        noc_async_write_barrier();
    }
    if constexpr (DEBUG_INDICES) final_ids[K]=0xffffffffu;
    cb_push_back(SKV4_CB_SELECTOR_CHANNEL, 1);
    // CB4 was the optional second local-TopK input.  That page has already
    // been consumed, so the owner can now reuse the same ring for sparse
    // attention's -inf column scalar.
    if constexpr (PROFILE >= 10) if (owner) {
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_QUERY_B", 0);
#endif
#endif
#endif
    // Reuse the existing union query: first eight tiles are the four heads,
    // plane-major coordinates, with x16 folded into exact BF16 exponents.
    if constexpr(PROFILE>=3) {
        cb_reserve_back(SKV4_CB_MATRIX_QUERY,8);
        const uint32_t qp=get_write_ptr(SKV4_CB_MATRIX_QUERY);
        for(uint32_t t=0;t<8;++t)
            noc_async_read(get_noc_addr(slot*16+t,uq),qp+t*2048,2048);
        noc_async_read_barrier(); cb_push_back(SKV4_CB_MATRIX_QUERY,8);
    }
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_QUERY_E", 0);
#endif
#endif
#endif
    cb_wait_front(SKV4_CB_SHARED_SCRATCH, 1);
    const uint32_t shared_scratch = get_read_ptr(SKV4_CB_SHARED_SCRATCH);
#if SKV4_CODEGEN_DYNAMIC
    constexpr uint32_t INPUT_COUNT_OFFSET=4096;
    const uint32_t stage16_live=
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            shared_scratch+INPUT_COUNT_OFFSET+shard*sizeof(uint32_t));
#endif
    if constexpr (PROFILE == 2 && DEBUG_INDICES) if (shard == 0) {
        // Addressing diagnostic: export exactly what this core's reader put
        // in the inherited-score window, before scoring or scratch reuse.
        noc_async_write(
            shared_scratch + INHERITED_OFFSET,
            get_noc_addr(slot * G + part, debug_scores), LOCAL16 * 2);
        noc_async_write_barrier();
    }
    if constexpr (PROFILE < 3) {
        cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
        return;
    }
#if SKV4_CODEGEN_DYNAMIC
    constexpr uint32_t MAX_CHUNKS=LOCAL16_PAD/32;
    constexpr uint32_t QUARTER_BYTES=((MAX_CHUNKS+3)/4)*64;
    const uint32_t stage16_chunks=(stage16_live+31)/32;
    const uint32_t first=part*(stage16_chunks/4)+
        (part<stage16_chunks%4?part:stage16_chunks%4);
    const uint32_t count=stage16_chunks/4+(part<stage16_chunks%4);
#else
    constexpr uint32_t CHUNKS=LOCAL16_PAD/32;
    constexpr uint32_t QUARTER_BYTES=((CHUNKS+3)/4)*64;
    const uint32_t first=part*(CHUNKS/4)+(part<CHUNKS%4?part:CHUNKS%4);
    const uint32_t count=CHUNKS/4+(part<CHUNKS%4);
#endif
    // CB2 is raw writer-local staging, not a FIFO. No counters are advanced.
    const uint32_t rows=get_write_ptr(SKV4_CB_SPARSE_KV_RECORDS);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_SCORE_MATERIALIZE_B", 0);
#endif
#endif
#endif
    for(uint32_t chunk=0;chunk<count;++chunk) {
        cb_wait_front(SKV4_CB_SCORE_TILES,1);
        auto tile=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(SKV4_CB_SCORE_TILES));
        for(uint32_t h=0;h<4;++h) {
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rows+h*QUARTER_BYTES);
            for(uint32_t c=0;c<32;++c)
                dst[chunk*32+c]=tile[h*16+(c<16?c:256+c-16)];
        }
        cb_pop_front(SKV4_CB_SCORE_TILES,1);
    }
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_SCORE_MATERIALIZE_E", 0);
#endif
#endif
#endif
    // RISC-V stores retire before their L1 fill is visible to the NoC. Drain
    // the local fill before the NoC sources these compact score runs.
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_SCORE_EXCHANGE_B", 0);
#endif
#endif
#endif
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
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_SCORE_EXCHANGE_E", 0);
#endif
#endif
#endif
    if(owner) cb_wait_front(SKV4_CB_SCORE_SCRATCH,1);
    const uint32_t cand_prefix=(shard*LOCAL16*4)&63;
    auto candidates=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_SCORE_SCRATCH)+cand_prefix);
    uint32_t receive = 0;
    if (owner) {
        // Raw NoC scratch only: no FIFO reservation. Stage II does not use ones.
        receive = get_write_ptr(SKV4_CB_EXCHANGE_TILES);
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
    if constexpr (PER_HEAD_INPUT) cb_pop_front(SKV4_CB_SCORE_SCRATCH, 1);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_TOPK_INPUT_B", 0);
#endif
#endif
#endif
    // Present the 928 scores as one row-major BF16 K=1024 page.  This copy is
    // deliberately local and contiguous; the expensive O(928 log 160) scalar
    // heap has been replaced by one SFPU TopK-XL sort in the compute thread.
    // Padding is -inf and can never survive.
    cb_reserve_back(SKV4_CB_SELECTOR_CHANNEL, 1);
    volatile tt_l1_ptr uint16_t* topk_pages =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_SELECTOR_CHANNEL));
    const uint32_t first_topk = LOCAL16_PAD < 1024 ? LOCAL16_PAD : 1024;
    for (uint32_t j = 0; j < first_topk; ++j) {
#if SKV4_CODEGEN_DYNAMIC
        topk_pages[j] = j < stage16_live ? topk_scores[j] : neg_inf_bf16;
#else
        topk_pages[j] = j < LOCAL16 ? topk_scores[j] : neg_inf_bf16;
#endif
    }
    cb_push_back(SKV4_CB_SELECTOR_CHANNEL, 1);
#if SKV4_CODEGEN_DYNAMIC
    if (stage16_live > 1024) {
#else
    if constexpr (LOCAL16_PAD > 1024) {
#endif
        cb_reserve_back(SKV4_CB_ATTENTION_NEG_INF, 1);
        volatile tt_l1_ptr uint16_t* second_topk =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_ATTENTION_NEG_INF));
        for (uint32_t j = 1024; j < LOCAL16_PAD; ++j) {
            second_topk[j - 1024] =
#if SKV4_CODEGEN_DYNAMIC
                j < stage16_live ? topk_scores[j] : neg_inf_bf16;
#else
                j < LOCAL16 ? topk_scores[j] : neg_inf_bf16;
#endif
        }
        cb_push_back(SKV4_CB_ATTENTION_NEG_INF, 1);
    }
    if constexpr(PROFILE==3) {
        // Export the complete recomputed 4-bit score consumed by TopK.
        // This catches padding and alignment
        // errors that a residual-only diagnostic cannot see.
        if(shard==0) {
            noc_async_write(reinterpret_cast<uint32_t>(topk_pages),
#if SKV4_CODEGEN_DYNAMIC
                get_noc_addr(slot*G+part,debug_scores),stage16_live*2);
#else
                get_noc_addr(slot*G+part,debug_scores),LOCAL16*2);
#endif
            noc_async_write_barrier();
        }
        cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return;
    }
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_TOPK_INPUT_E", 0);
    DeviceTimestampedData("SKV_S16_W_LOCAL_MERGE_ROUTE_B", 0);
#endif
#endif
    cb_wait_front(SKV4_CB_LOCAL_FRONTIERS,stage16_live>1024?2:1);
#else
    cb_wait_front(SKV4_CB_LOCAL_FRONTIERS,LOCAL16_PAD>1024?2:1);
#endif
    volatile tt_l1_ptr uint32_t* local_indices =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_LOCAL_FRONTIERS));
    // Merge the two independently sorted K=1024 halves.  Only the first
    // KLOCAL ranks from either half can enter this shard's exported frontier.
    // First shard descends and subsequent shards ascend, including -inf
    // padding, to form each bitonic merge pair.
    constexpr uint32_t BITS=MERGE_K==512?5:6;
    auto half0=local_indices;
    auto half1=local_indices+1024;
    uint32_t i0=0,i1=0;
    for(uint32_t rank=0;rank<MERGE_K;++rank) {
        uint32_t word=0xff800000u;
        if(rank<KLOCAL) {
            const uint32_t l0=(i0>>6)+((i0&63u)<<4);
            const uint32_t l1=(i1>>6)+((i1&63u)<<4);
            const uint32_t w0=i0<1024?half0[l0]:0xff800000u;
#if SKV4_CODEGEN_DYNAMIC
            const uint32_t w1=(stage16_live>1024 && i1<stage16_live-1024)?
#else
            const uint32_t w1=(LOCAL16_PAD>1024 && i1<LOCAL16_PAD-1024)?
#endif
                half1[l1]:0xff800000u;
            const uint16_t s0=ordered_bf16(static_cast<uint16_t>(w0>>16));
            const uint16_t s1=ordered_bf16(static_cast<uint16_t>(w1>>16));
            const uint32_t packed=s1>s0?w1:w0;
            if(s1>s0) ++i1; else ++i0;
            const uint32_t local=decode_topk_xl_k1024(packed);
            word=(packed&0xffff0000u)|(
                PROFILE==4 ? (packed&0xffffu) : (shard*LOCAL16+local));
        }
        const uint32_t out_rank=shard==0?rank:MERGE_K-1-rank;
        local_row[((out_rank&(MERGE_K/16-1))<<4)|(out_rank>>BITS)]=word;
    }
#if SKV4_CODEGEN_DYNAMIC
    cb_pop_front(SKV4_CB_LOCAL_FRONTIERS,stage16_live>1024?2:1);
#else
    cb_pop_front(SKV4_CB_LOCAL_FRONTIERS,LOCAL16_PAD>1024?2:1);
#endif
    // Send straight to the root responsible for this query head. There is no
    // gather-at-local-root then retransmit phase. Receive slots are disjoint.
    const uint32_t head_x=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*part);
    const uint32_t head_y=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*part);
    noc_async_write(reinterpret_cast<uint32_t>(local_row),
        get_noc_addr(head_x,head_y,get_write_ptr(SKV4_CB_TOPK_MERGE_SCRATCH)+shard*4096),ROW_BYTES);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(head_x,head_y,get_semaphore(SKV4_SEM_PACKED_FRONTIERS_READY)),1);
    noc.async_atomic_barrier();
    if(owner) {
        cb_pop_front(SKV4_CB_SCORE_SCRATCH,1);
        auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_PACKED_FRONTIERS_READY));
        noc_semaphore_wait(ready,W); noc_semaphore_set(ready,0);
    }
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_LOCAL_MERGE_ROUTE_E", 0);
#endif
#endif
#endif
    if constexpr (PROFILE == 4) {
        if(owner) {
            auto pages=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_write_ptr(SKV4_CB_TOPK_MERGE_SCRATCH));
            auto compact=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                shared_scratch);
            constexpr uint32_t DBITS=MERGE_K==512?5:6;
            for(uint32_t src=0;src<4;++src) for(uint32_t rank=0;rank<KLOCAL;++rank) {
                const uint32_t stored=src==0?rank:MERGE_K-1-rank;
                const uint32_t lane=((stored&(MERGE_K/16-1))<<4)|(stored>>DBITS);
                compact[src*KLOCAL+rank]=pages[src*1024+lane];
            }
            noc_async_write(shared_scratch,
                get_noc_addr(slot*G+shard,debug_scores),4*KLOCAL*4);
            noc_async_write_barrier();
        }
        cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
        return;
    }
    uint32_t union_count=0; uint32_t* union_ids=nullptr;
    uint16_t hv16[M16];
    uint32_t hi16[M16];
    if (owner) {
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_GLOBAL_MERGE_B", 0);
#endif
#endif
#endif
        // Remote writers filled four native-layout, descending logical
        // frontiers. Shard zero is stored descending; the other three are
        // physically reversed for the removed TopK-XL bitonic merge. Read
        // each in logical rank order and perform only 3*M16 comparisons.
        constexpr uint32_t RBITS=MERGE_K==512?5:6;
        auto p0=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_TOPK_MERGE_SCRATCH));
        const auto fetch=[&](uint32_t src,uint32_t rank) -> uint32_t {
            if(rank>=KLOCAL) return 0xff800000u;
            const uint32_t stored=src==0?rank:MERGE_K-1-rank;
            const uint32_t lane=((stored&(MERGE_K/16-1))<<4)|(stored>>RBITS);
            return p0[src*1024+lane];
        };
        uint32_t i0=0,i1=0,i2=0,i3=0;
        uint32_t w0=fetch(0,0),w1=fetch(1,0),w2=fetch(2,0),w3=fetch(3,0);
        uint16_t s0=ordered_bf16(w0>>16),s1=ordered_bf16(w1>>16);
        uint16_t s2=ordered_bf16(w2>>16),s3=ordered_bf16(w3>>16);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_MERGE_WORK_B", 0);
#endif
#endif
#endif
#if SKV4_PROFILE
        uint32_t fusion_rank256 = 0;
        uint32_t fusion_rank400 = 0;
#endif
        for(uint32_t out=0;out<M16;++out) {
            // Keep all four frontier heads in registers. Only the winner's
            // cursor/load/ordered conversion changes; ties prefer lower src.
            const bool a=s1>s0,b=s3>s2;
            const uint16_t sa=a?s1:s0,sb=b?s3:s2;
            const uint32_t winner=sb>sa?(b?3:2):(a?1:0);
            if(winner==0) {
                hi16[out]=w0&0xffffu; w0=fetch(0,++i0); s0=ordered_bf16(w0>>16);
            } else if(winner==1) {
                hi16[out]=w1&0xffffu; w1=fetch(1,++i1); s1=ordered_bf16(w1>>16);
            } else if(winner==2) {
                hi16[out]=w2&0xffffu; w2=fetch(2,++i2); s2=ordered_bf16(w2>>16);
            } else {
                hi16[out]=w3&0xffffu; w3=fetch(3,++i3); s3=ordered_bf16(w3>>16);
            }
#if SKV4_PROFILE
            if (out + 1 == 256) {
                asm volatile("" ::: "memory");
                fusion_rank256 = *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                    RISCV_DEBUG_REG_WALL_CLOCK_L);
                asm volatile("" ::: "memory");
            }
#endif
        }
#if SKV4_PROFILE
        asm volatile("" ::: "memory");
        fusion_rank400 = *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
            RISCV_DEBUG_REG_WALL_CLOCK_L);
        asm volatile("" ::: "memory");
        // DATA carries the earlier sampled clock, not the emission time.
        DeviceTimestampedData("SKV_FU_RANK256", fusion_rank256);
        DeviceTimestampedData("SKV_FU_RANK400", fusion_rank400);
#endif
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_MERGE_WORK_E", 0);
    DeviceTimestampedData("SKV_S16_W_GLOBAL_MERGE_E", 0);
#endif
#endif
#endif
        if constexpr (PROFILE == 5) {
            volatile tt_l1_ptr uint32_t* probe =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive);
            for (uint32_t j = 0; j < M16; ++j) probe[j] = hi16[j]; // stop5 diagnostic: candidate position, not token ID
            noc_async_write(
                receive, get_noc_addr(slot * G + shard, debug_scores), M16 * 4);
            noc_async_write_barrier();
            cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
            return;
        }
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_FRONTIER_EXPORT_B", 0);
#endif
#endif
#endif
        // Final per-head list; Program B owns deduplication and residual fetch.
        cb_reserve_back(SKV4_CB_SELECTOR_CHANNEL,1);
        auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_SELECTOR_CHANNEL));
        dst[0]=M16;
        for(uint32_t j=0;j<M16;++j) dst[1+j]=hi16[j];
        cb_push_back(SKV4_CB_SELECTOR_CHANNEL,1);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_CODEGEN_DYNAMIC
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S16_W_FRONTIER_EXPORT_E", 0);
#endif
#endif
#endif
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_IDS_READY)),1);
    }
    if constexpr(PROFILE<=7) { cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return; }
    if(!owner) { cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return; }
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL)
    if(owner) cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return; // diagnostic: no score producer
#endif
    constexpr uint32_t HP=16384+LOCAL16*64+G*M16*4+LOCAL16*4;
    auto hv_all=reinterpret_cast<uint16_t*>(shared_scratch+HP);
    auto hi_all=reinterpret_cast<uint32_t*>(shared_scratch+HP+G*K*2);
    for(uint32_t j=0;j<G*K;++j) { hv_all[j]=ordered_bf16(neg_inf_bf16); hi_all[j]=0xffffffffu; }
    for(uint32_t chunk=0;chunk<(union_count+31)/32;++chunk) {
        cb_wait_front(SKV4_CB_SCORE_TILES,1);
        const uint32_t base=get_read_ptr(SKV4_CB_SCORE_TILES);
        if constexpr(PROFILE>=9) {
            const uint32_t begin=chunk*32, end=std::min(begin+32,union_count);
            for(uint32_t h=0;h<G;++h) for(uint32_t j=begin;j<end;++j) {
                const uint16_t value=score_at(base,j-begin,h);
                heap_insert<K>(ordered_bf16(value),union_ids[j],hv_all+h*K,hi_all+h*K);
            }
        }
        cb_pop_front(SKV4_CB_SCORE_TILES,1);
    }
    if constexpr(PROFILE==8) { cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return; }
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
        } else noc_async_write(src,get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*h),
                   get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*h),receive+shard*ROW_BYTES),ROW_BYTES);
    }
    noc_async_write_barrier();
    for(uint32_t h=0;h<G;++h) if(h!=shard)
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*h),
            get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*h),get_semaphore(SKV4_SEM_STAGE2_FRONTIERS_READY)),1);
    noc_async_atomic_barrier();
    auto done=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_FRONTIERS_READY));
    noc_semaphore_wait(done,G-1); noc_semaphore_set(done,0);
    uint16_t hv[K]; uint32_t hi[K];
    for(uint32_t j=0;j<K;++j) { hv[j]=ordered_bf16(neg_inf_bf16); hi[j]=0xffffffffu; }
    for(uint32_t w=0;w<W;++w) {
        auto p=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive+w*ROW_BYTES);
        for(uint32_t j=0;j<K;++j) heap_insert<K>(p[2*j],p[2*j+1],hv,hi);
    }
    if constexpr (PROFILE == 9) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive)[0] = hi[0];
        cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
        return;
    }
    // CB30 remains writer-owned across all three sequential payloads: local
    // TopK input, global top-400 IDs, then final top-100 IDs. CB17 stays
    // compute->writer only for the TopK-XL result.
    cb_reserve_back(SKV4_CB_SELECTOR_CHANNEL, 1);
    volatile tt_l1_ptr uint32_t* final_ids =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_SELECTOR_CHANNEL));
    for (uint32_t j = 0; j < padded_topk; ++j) {
        final_ids[j] = j < K ? hi[j] : 0xffffffffu;
    }
    if constexpr (DEBUG_INDICES) {
        final_ids[K]=union_count;
        noc_async_write(reinterpret_cast<uint32_t>(final_ids),
                        get_noc_addr(slot * G + shard, debug), padded_topk * 4);
        noc_async_write_barrier();
    }
    if constexpr (DEBUG_INDICES) final_ids[K]=0xffffffffu;
    cb_push_back(SKV4_CB_SELECTOR_CHANNEL, 1);
    // CB4 was the optional second local-TopK input.  That page has already
    // been consumed, so the owner can now reuse the same ring for sparse
    // attention's -inf column scalar.
    if constexpr (PROFILE >= 10) if (owner) {
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_QUERY_B", 0);
#endif
#endif
    if constexpr(PROFILE>=8) {
        cb_reserve_back(SKV4_CB_MATRIX_QUERY,16);
        const uint32_t qp=get_write_ptr(SKV4_CB_MATRIX_QUERY);
        for(uint32_t t=0;t<16;++t)
            noc_async_read(get_noc_addr(slot*16+t,uq),qp+t*2048,2048);
        noc_async_read_barrier(); cb_push_back(SKV4_CB_MATRIX_QUERY,16);
#if SKV4_PROFILE
        const uint32_t tick = *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
            RISCV_DEBUG_REG_WALL_CLOCK_L);
        DeviceTimestampedData("SKV_FU_QUERY2", tick);
#endif
    }
    cb_wait_front(SKV4_CB_SHARED_SCRATCH,1);
    uint32_t shared_scratch=get_read_ptr(SKV4_CB_SHARED_SCRATCH);
    auto meta=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_SCORE_SCRATCH));
    const uint32_t union_count=meta[0];
    const uint32_t union_chunks=(union_count+31)/32;
    constexpr uint32_t LIMIT=LOCAL16<G*KLOCAL?LOCAL16:G*KLOCAL;
    constexpr uint32_t STRIPE_CHUNKS=((LIMIT+31)/32+G-1)/G;
    const uint32_t first=part*STRIPE_CHUNKS;
    const uint32_t local_chunks=union_chunks/G+(part<union_chunks%G);
    const uint32_t full=union_count/32;
    const uint32_t local_count=(full/G+(part<full%G))*32+
        (part==full%G?union_count%32:0);
    if constexpr(PROFILE>=10) if(owner) {
#else
    if(!owner) return;
#endif
#endif
#endif
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            6, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW, 1>();
        generate_bcast_col_scalar(CircularBuffer(SKV4_CB_ATTENTION_COLUMN_IDENTITY), one_bf16_packed);
        cb_reserve_back(SKV4_CB_ATTENTION_NEG_INF, 1);
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
        volatile tt_l1_ptr uint16_t* ni =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_ATTENTION_NEG_INF));
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
        volatile tt_l1_ptr uint16_t* ni =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_ATTENTION_NEG_INF));
#else
        auto ni=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_ATTENTION_NEG_INF));
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
        for(uint32_t c=0;c<16;++c) { ni[c]=neg_inf_bf16; ni[256+c]=neg_inf_bf16; }
        cb_push_back(SKV4_CB_ATTENTION_NEG_INF,1);
    }
    constexpr uint32_t UNION_IDS_OFFSET=16384+LOCAL16*64+G*M16*4;
    auto union_ids=reinterpret_cast<uint32_t*>(shared_scratch+UNION_IDS_OFFSET);
    for(uint32_t j=0;j<union_count;++j) union_ids[j]=meta[j+1]&ID_MASK;
    const uint32_t receive=get_write_ptr(SKV4_CB_EXCHANGE_TILES), local_packet=shared_scratch;
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_QUERY_E", 0);
#endif
#endif
    if constexpr(PROFILE<=7) { cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return; }
#if !(SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL)
    cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return; // diagnostic: no score producer
#endif
    constexpr uint32_t TOPK_N=1024;
    constexpr uint32_t QUARTER_CHUNKS=STRIPE_CHUNKS;
    constexpr uint32_t QUARTER_BYTES=QUARTER_CHUNKS*64;
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_SCORE_MATERIALIZE_B", 0);
#endif
#endif
    // Materialize this candidate quarter's four score rows contiguously.
    auto staged=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(shared_scratch);
    for(uint32_t chunk=0;chunk<local_chunks;++chunk) {
        cb_wait_front(SKV4_CB_SCORE_TILES,1);
        const uint32_t base=get_read_ptr(SKV4_CB_SCORE_TILES);
        for(uint32_t h=0;h<G;++h) for(uint32_t j=0;j<32;++j) {
            const uint32_t local=chunk*32+j;
            staged[h*QUARTER_CHUNKS*32+local]=local<local_count?
                score_at(base,j,h):neg_inf_bf16;
        }
        cb_pop_front(SKV4_CB_SCORE_TILES,1);
    }
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_SCORE_MATERIALIZE_E", 0);
#endif
#endif
    // Every destination reserves its own complete score row. Source quarters
    // write disjoint 32-key-aligned runs, so no initialization/write barrier is
    // needed. Only the untouched suffix is initialized locally to -inf.
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_SCORE_EXCHANGE_B", 0);
#endif
#endif
    cb_reserve_back(SKV4_CB_SELECTOR_CHANNEL,1);
    const uint32_t score_page=get_write_ptr(SKV4_CB_SELECTOR_CHANNEL);
    auto score16=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(score_page);
    static_assert(G*STRIPE_CHUNKS*32<=TOPK_N);
    // Source workers write their round-robin chunks directly into compact
    // logical order. Only the suffix is untouched by peers.
    for(uint32_t j=union_count;j<TOPK_N;++j) score16[j]=neg_inf_bf16;
    if(local_chunks) for(uint32_t h=0;h<G;++h)
        for(uint32_t chunk=0;chunk<local_chunks;++chunk) {
            const uint32_t logical_chunk=chunk*G+part;
            noc_async_write(reinterpret_cast<uint32_t>(
                                staged+h*QUARTER_CHUNKS*32+chunk*32),
                get_noc_addr(row_start_x+h,row_start_y,
                             score_page+logical_chunk*64),64);
        }
    noc_async_write_barrier();
    for(uint32_t h=0;h<G;++h)
        noc_semaphore_inc(get_noc_addr(row_start_x+h,row_start_y,
            get_semaphore(SKV4_SEM_SCORES_READY)),1);
    noc_async_atomic_barrier();
    auto score_ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_SCORES_READY));
    noc_semaphore_wait(score_ready,G); noc_semaphore_set(score_ready,0);
    if(union_count>512) {
        cb_reserve_back(SKV4_CB_ATTENTION_NEG_INF,1);
        auto second=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_ATTENTION_NEG_INF));
        for(uint32_t j=0;j<512;++j)
            second[j]=j+512<TOPK_N?score16[j+512]:neg_inf_bf16;
        cb_push_back(SKV4_CB_ATTENTION_NEG_INF,1);
    }
    cb_push_back(SKV4_CB_SELECTOR_CHANNEL,1);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_SCORE_EXCHANGE_E", 0);
#endif
#endif
    if constexpr(PROFILE==8) {
        if(shard==0) {
            noc_async_write(score_page,
                get_noc_addr(slot*G+part,debug_scores),TOPK_N*2);
            noc_async_write_barrier();
        }
        cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return;
    }
    // K512's native row-major index matrix is 32x16. If a source union spills
    // above 512, merge the exact top-100 of each disjoint half. An element
    // below rank 100 in either half cannot enter the combined top-100.
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_LOCAL_TOPK_REMAP_B", 0);
#endif
#endif
    const uint32_t frontier_count=union_count>512?2:1;
    cb_wait_front(SKV4_CB_LOCAL_FRONTIERS,frontier_count);
    auto ranked0=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SKV4_CB_LOCAL_FRONTIERS));
    auto ranked1=ranked0+1024; // one 4096-byte CB page
    constexpr uint32_t S2_ROW_WORDS=K*2;
    constexpr uint32_t S2_ROW_BYTES=S2_ROW_WORDS*4;
    auto local_row=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_packet);
    uint32_t half_cursor0=0,half_cursor1=0;
    for(uint32_t out=0;out<K;++out) {
        const uint32_t lane0=(half_cursor0>>5)+((half_cursor0&31u)<<4);
        const uint32_t half0=ranked0[lane0];
        const bool valid0=half0<union_count && half0<512;
        const uint16_t score0=valid0?ordered_bf16(score16[half0]):
            ordered_bf16(neg_inf_bf16);
        uint32_t half1=0;
        uint16_t score1=ordered_bf16(neg_inf_bf16);
        bool valid1=false;
        if(frontier_count==2) {
            const uint32_t lane1=(half_cursor1>>5)+((half_cursor1&31u)<<4);
            half1=ranked1[lane1]+512;
            valid1=half1<union_count;
            score1=valid1?ordered_bf16(score16[half1]):
                ordered_bf16(neg_inf_bf16);
        }
        const bool take1=score1>score0;
        const uint32_t pos=take1?half1:half0;
        const uint16_t score=take1?score1:score0;
        if(take1) ++half_cursor1; else ++half_cursor0;
        const bool valid=(take1?valid1:valid0) && pos<union_count;
        local_row[2*out]=valid?score:ordered_bf16(neg_inf_bf16);
        local_row[2*out+1]=valid?union_ids[pos]:0xffffffffu;
    }
    cb_pop_front(SKV4_CB_LOCAL_FRONTIERS,frontier_count);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_LOCAL_TOPK_REMAP_E", 0);
    DeviceTimestampedData("SKV_S2_W_FRONTIER_ROUTE_B", 0);
#endif
#endif
    // Route this (source shard, query head) frontier directly to its head root.
    const uint32_t dest_x=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*part);
    const uint32_t dest_y=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*part);
    noc_async_write(local_packet,get_noc_addr(dest_x,dest_y,
        receive+shard*S2_ROW_BYTES),S2_ROW_BYTES);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(dest_x,dest_y,get_semaphore(SKV4_SEM_STAGE2_FRONTIERS_READY)),1);
    noc_async_atomic_barrier();
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_FRONTIER_ROUTE_E", 0);
#endif
#endif
    auto merged_ack=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_PEER_EXCHANGE_READY));
    if(!owner) {
        // Do not let a sender enter the next decode step until its destination
        // root has consumed this frontier and reset semaphore 12.  Without
        // this acknowledgement, a next-step increment can be erased by the
        // previous step's reset and eventually deadlock a long run.
        noc_semaphore_wait(merged_ack,1); noc_semaphore_set(merged_ack,0);
        cb_pop_front(SKV4_CB_SHARED_SCRATCH,1);
        return;
    }
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_FRONTIER_JOIN_B", 0);
#endif
#endif
    // Root `shard` owns query head `shard`: merge four exact local frontiers.
    auto done=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_STAGE2_FRONTIERS_READY));
    noc_semaphore_wait(done,G); noc_semaphore_set(done,0);
    // TopK-XL already emitted every local frontier in descending score order.
    // Preserve that information: selecting global top-K from four sorted
    // top-K lists needs only a four-way merge (3*K comparisons), not another
    // 4*K insertions into a branch-dependent size-K heap.
    auto p0=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive+0*S2_ROW_BYTES);
    auto p1=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive+1*S2_ROW_BYTES);
    auto p2=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive+2*S2_ROW_BYTES);
    auto p3=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive+3*S2_ROW_BYTES);
    uint32_t i0=0, i1=0, i2=0, i3=0;
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_FRONTIER_JOIN_E", 0);
    DeviceTimestampedData("SKV_S2_W_GLOBAL_TOP100_B", 0);
#endif
#endif
    uint32_t hi[K];
    for(uint32_t out=0;out<K;++out) {
        uint32_t winner=0;
        uint16_t best=static_cast<uint16_t>(p0[2*i0]);
        const uint16_t k1=static_cast<uint16_t>(p1[2*i1]);
        const uint16_t k2=static_cast<uint16_t>(p2[2*i2]);
        const uint16_t k3=static_cast<uint16_t>(p3[2*i3]);
        if(k1>best) { best=k1; winner=1; }
        if(k2>best) { best=k2; winner=2; }
        if(k3>best) { best=k3; winner=3; }
        if(winner==0) { hi[out]=p0[2*i0+1]; ++i0; }
        else if(winner==1) { hi[out]=p1[2*i1+1]; ++i1; }
        else if(winner==2) { hi[out]=p2[2*i2+1]; ++i2; }
        else { hi[out]=p3[2*i3+1]; ++i3; }
    }
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_GLOBAL_TOP100_E", 0);
    DeviceTimestampedData("SKV_S2_W_MERGE_ACK_B", 0);
#endif
#endif
    // Acknowledge every source-shard sender for the query head owned here.
    // The 4-core rows are physically contiguous in +x.
    for(uint32_t w=0;w<W;++w) {
        const uint32_t x=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS+2*w)+shard;
        const uint32_t y=get_arg_val<uint32_t>(SKV4_ARG_RF_RT_HEAD_PEERS_Y+2*w);
        noc_semaphore_inc(get_noc_addr(x,y,get_semaphore(SKV4_SEM_PEER_EXCHANGE_READY)),1);
    }
    noc_async_atomic_barrier();
    // This owner also sent head 0 to root 0; wait for that independent merge.
    noc_semaphore_wait(merged_ack,1); noc_semaphore_set(merged_ack,0);
#if SKV4_CODEGEN_PHASE == SKV4_PHASE_FULL
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S2_W_MERGE_ACK_E", 0);
#endif
#endif
    if constexpr (PROFILE == 9) {
        auto final_ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            local_packet);
        for(uint32_t j=0;j<padded_topk;++j)
            final_ids[j]=j<K?hi[j]:0xffffffffu;
        noc_async_write(local_packet,
                        get_noc_addr(slot*W+shard,handoff),
                        padded_topk*4);
        if constexpr(DEBUG_INDICES) {
            final_ids[K]=union_count;
            noc_async_write(local_packet,get_noc_addr(slot*G+shard,debug),
                            padded_topk*4);
        }
        noc_async_write_barrier();
        if constexpr(DEBUG_INDICES) final_ids[K]=0xffffffffu;
        cb_pop_front(SKV4_CB_SHARED_SCRATCH,1); return;
    }
    // CB30 remains writer-owned across all three sequential payloads: local
    // TopK input, global top-400 IDs, then final top-100 IDs. CB17 stays
    // compute->writer only for the TopK-XL result.
    cb_reserve_back(SKV4_CB_SELECTOR_CHANNEL, 1);
    volatile tt_l1_ptr uint32_t* final_ids =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_SELECTOR_CHANNEL));
    for (uint32_t j = 0; j < padded_topk; ++j) {
        final_ids[j] = j < K ? hi[j] : 0xffffffffu;
    }
    if constexpr (DEBUG_INDICES) {
        final_ids[K]=union_count;
        noc_async_write(reinterpret_cast<uint32_t>(final_ids),
                        get_noc_addr(slot * G + shard, debug), padded_topk * 4);
        noc_async_write_barrier();
    }
    if constexpr (DEBUG_INDICES) final_ids[K]=0xffffffffu;
    cb_push_back(SKV4_CB_SELECTOR_CHANNEL, 1);
    // CB4 was the optional second local-TopK input.  That page has already
    // been consumed, so the owner can now reuse the same ring for sparse
    // attention's -inf column scalar.
    if constexpr (PROFILE >= 10) if (owner) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            6, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW, 1>();
        generate_bcast_col_scalar(CircularBuffer(SKV4_CB_ATTENTION_COLUMN_IDENTITY), one_bf16_packed);
        cb_reserve_back(SKV4_CB_ATTENTION_NEG_INF, 1);
        volatile tt_l1_ptr uint16_t* ni =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_ATTENTION_NEG_INF));
#endif
#endif
#endif
        for (uint32_t c = 0; c < 16; ++c) {
            ni[c] = neg_inf_bf16;
            ni[256 + c] = neg_inf_bf16;
        }
        cb_push_back(SKV4_CB_ATTENTION_NEG_INF, 1);
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
    }
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
    }
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
    }
#else
    cb_wait_front(SKV4_CB_SELECTOR_CHANNEL,1);
#endif
#endif
#endif
    // Writer half of one sparse-attention tail.
    experimental::CB req_cb(SKV4_CB_QUERY_COEFFICIENTS), ack_cb(SKV4_CB_TOPK_MERGE_SCRATCH), idx_cb(SKV4_CB_SELECTOR_CHANNEL), out_cb(SKV4_CB_ATTENTION_OUT_ROW);
    const uint32_t all_idx = idx_cb.get_read_ptr();
    constexpr uint32_t H = 32, S = G, block_tiles = D / 32;
    const uint32_t kv_page_offset = slot * context;
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE)
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16)
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2)
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_W_K_HELPER_B", 0);
#endif
#endif
#endif
#endif
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
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE)
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16)
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2)
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_W_K_HELPER_E", 0);
    DeviceTimestampedData("SKV_ATTN_W_WAIT_OUTPUT_B", 0);
#endif
#endif
#endif
#endif
    out_cb.wait_front(block_tiles);
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE)
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16)
#if !(SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2)
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_W_WAIT_OUTPUT_E", 0);
    DeviceTimestampedData("SKV_ATTN_W_WRITE_OUTPUT_B", 0);
#endif
#endif
#endif
#endif
    for (uint32_t h = 0; h < H; ++h) {
        noc.async_write(out_cb, out, D * 2,
                        {.offset_bytes = h * D * 2},
                        {.page_id = h * S + shard});
    }
    noc.async_write_barrier();
    out_cb.pop_front(block_tiles);
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_NONE
    cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE16
    cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
#else
#if SKV4_CODEGEN_ROLE == SKV4_ROLE_STAGE2
    cb_pop_front(SKV4_CB_SHARED_SCRATCH, 1);
#else
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_ATTN_W_WRITE_OUTPUT_E", 0);
#endif
#endif
#endif
#endif
}
"""
        )
    )
