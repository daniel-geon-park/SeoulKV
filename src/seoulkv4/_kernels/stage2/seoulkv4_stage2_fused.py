"""Blackhole Stage-II and sparse-attention kernels.

One Tensix core owns one ``(request, KV-head)`` slot.  It gathers the slot's
random candidate records from row-major E4M3 storage, scores all four GQA query
heads, and performs four exact BF16 top-k reductions without communicating with
another Tensix.  Q, gathered keys, score tiles, and heaps remain in that core's
L1; only sentinel-padded natural token indices are written to GDDR.

``build`` exposes the selector alone so it can be certified against the stock
Blackhole ``sparse_sdpa`` tail. ``build_full`` source-merges that tail into the
same dispatch: the top-100 indices pass through an on-core circular buffer and
never touch GDDR.

``build_full_sharded`` parallelizes the expensive Stage-II scan across nearby
Tensix workers.  Its flat topology emits one packet containing exact local
top-100 rows for all four GQA heads and fans three packets into one owner.  Its
head-parallel topology exchanges one 800-byte row per destination in one NoC
round; worker h exactly merges and attends query head h.  There is no
per-candidate or tree-shaped communication in either topology.
"""

from __future__ import annotations

from seoulkv4._kernels.stage2.abi import Stage2ShardedSemaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines

import struct


ROWS = 32
GQA_GROUP = 4
D = 256
TOPK = 100
PADDED_TOPK = 128


READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);
    constexpr uint32_t D = get_compile_time_arg_val(2);
    constexpr auto qa = TensorAccessorArgs<3>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ka.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot = get_arg_val<uint32_t>(3);
    const uint32_t context = get_arg_val<uint32_t>(4);

    constexpr uint32_t q_row_bytes = D * sizeof(uint16_t);
    constexpr uint32_t k_row_bytes = D;
    constexpr uint32_t chunks = N / 32;
    static_assert(N % 32 == 0);
    static_assert(G <= 32);

    const auto q = TensorAccessor(qa, q_addr, q_row_bytes);
    const auto k = TensorAccessor(ka, k_addr, k_row_bytes);
    const auto idx = TensorAccessor(ia, idx_addr, N * sizeof(uint32_t));

    // Candidate ids are a persistent single-page CB.  Reader and writer share
    // the same on-core L1 page; only the writer pops it after all scoring.
    cb_reserve_back(24, 1);
    const uint32_t idx_l1 = get_write_ptr(24);
    noc_async_read(get_noc_addr(slot, idx), idx_l1, N * sizeof(uint32_t));

    // Q is padded to 32 rows because tilization/matmul are tile-native.  Only
    // rows [0,G) are selected, but the padding never leaves this core.
    cb_reserve_back(0, 32);
    const uint32_t q_l1 = get_write_ptr(0);
    for (uint32_t row = 0; row < 32; ++row) {
        noc_async_read(get_noc_addr(slot * 32 + row, q), q_l1 + row * q_row_bytes, q_row_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(24, 1);
    cb_push_back(0, 32);

    volatile tt_l1_ptr uint32_t* ids =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        cb_reserve_back(2, 32);
        const uint32_t dst = get_write_ptr(2);
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t natural = ids[chunk * 32 + row];
            const uint32_t page = slot * context + natural;
            noc_async_read(get_noc_addr(page, k), dst + row * k_row_bytes, k_row_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(2, 32);
    }
}
"""


COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
constexpr bool EXP_APPROX_MODE = false;  // required by the shared SDPA helpers
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t D = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = D / 32;
    constexpr uint32_t chunks = N / 32;
    static_assert(N % 32 == 0);
    static_assert(D % 32 == 0);

    CircularBuffer q_in(1), k_in(3), scores(16);
    compute_kernel_hw_startup(0, 1);
    compute_kernel_lib::tilize<DHt, 0, 1>(/*num_blocks=*/1, /*total_input_pages=*/32);
    q_in.wait_front(DHt);

    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        compute_kernel_lib::tilize<DHt, 2, 3>(/*num_blocks=*/1, /*total_input_pages=*/32);
        k_in.wait_front(DHt);
        scores.reserve_back(1);

        // scores[32 queries,32 candidates] = Q @ K^T.  K is gathered as
        // row-major E4M3 and tilized into BFP8; scores are accumulated/packed
        // as BF16 exactly as in Blackhole sparse_sdpa's QK phase.
        reconfig_full_operand(3, 1);
        pack_reconfig_data_format(16);
        mm_no_mop_init_short(1, 3, /*transpose=*/true, 1, 1, DHt);
        configure_row_pack_width(16, 1);
        blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/1>(
            1, 3, 16,
            /*in0_index_start=*/0,
            /*in1_index_start=*/0,
            /*row_subblock_idx=*/0,
            /*out_col_offset=*/0,
            /*subblock_w=*/1,
            /*subblock_h=*/1,
            /*inner_dim=*/DHt,
            /*matmul_stride=*/DHt,
            /*skip_pack_configure=*/true);
        scores.push_back(1);
        k_in.pop_front(DHt);
    }
    q_in.pop_front(DHt);
}
"""


WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint16_t ordered_bf16(uint16_t bits) {
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
}

FORCE_INLINE uint16_t score_at(uint32_t scores, uint32_t candidate, uint32_t row) {
    const uint32_t tile = scores + (candidate >> 5) * 2048;
    const uint32_t col = candidate & 31;
    const uint32_t face_row = (row < 16 ? 0 : 1024) + (row & 15) * 32;
    const uint32_t elem = face_row + (col < 16 ? col * 2 : 512 + (col - 16) * 2);
    return *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile + elem);
}

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t PAD = get_compile_time_arg_val(3);
    constexpr auto oa = TensorAccessorArgs<4>();
    static_assert(K == 100);
    static_assert(PAD == 128);

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const auto out = TensorAccessor(oa, out_addr, PAD * sizeof(uint32_t));

    cb_wait_front(24, 1);
    cb_wait_front(16, N / 32);
    volatile tt_l1_ptr uint32_t* ids =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(24));
    const uint32_t score_base = get_read_ptr(16);

    cb_reserve_back(25, 1);
    volatile tt_l1_ptr uint32_t* final =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(25));

    // Process G rows sequentially so only one 100-element heap is live.  All
    // N score tiles remain local in cb16; no row redistribution is required.
    for (uint32_t row = 0; row < G; ++row) {
        uint16_t heap_value[K];
        uint32_t heap_index[K];
        for (uint32_t j = 0; j < K; ++j) {
            heap_value[j] = ordered_bf16(score_at(score_base, j, row));
            heap_index[j] = ids[j];
        }
        for (int32_t root = K / 2 - 1; root >= 0; --root) {
            const uint16_t key = heap_value[root];
            const uint32_t idx = heap_index[root];
            uint32_t pos = root;
            while (true) {
                const uint32_t left = pos * 2 + 1;
                if (left >= K) break;
                const uint32_t right = left + 1;
                uint32_t child = left;
                if (right < K && heap_value[right] < heap_value[left]) child = right;
                if (heap_value[child] >= key) break;
                heap_value[pos] = heap_value[child];
                heap_index[pos] = heap_index[child];
                pos = child;
            }
            heap_value[pos] = key;
            heap_index[pos] = idx;
        }
        for (uint32_t j = K; j < N; ++j) {
            const uint16_t key = ordered_bf16(score_at(score_base, j, row));
            if (key <= heap_value[0]) continue;
            uint32_t pos = 0;
            while (true) {
                const uint32_t left = pos * 2 + 1;
                if (left >= K) break;
                const uint32_t right = left + 1;
                uint32_t child = left;
                if (right < K && heap_value[right] < heap_value[left]) child = right;
                if (heap_value[child] >= key) break;
                heap_value[pos] = heap_value[child];
                heap_index[pos] = heap_index[child];
                pos = child;
            }
            heap_value[pos] = key;
            heap_index[pos] = ids[j];
        }

        for (uint32_t j = 0; j < PAD; ++j) {
            final[j] = j < K ? heap_index[j] : 0xffffffffu;
        }
        noc_async_write(reinterpret_cast<uint32_t>(final),
                        get_noc_addr(row, out), PAD * sizeof(uint32_t));
        noc_async_write_barrier();
    }

    cb_pop_front(16, N / 32);
    cb_pop_front(24, 1);
}
"""


# The complete kernel reuses the shipped sparse-SDPA compute implementation,
# but replaces its reader/writer with ownership-preserving versions.  Stage II
# uses CBs 23..28; sparse attention uses its standard CBs 0..22.  The two
# phases communicate four padded index rows through cb17 in the same core.
FULL_COMPUTE = r"""
#define kernel_main sparse_tail_main
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sparse_sdpa_compute.cpp"
#undef kernel_main

void stage2_main() {
    constexpr uint32_t N = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END);
    constexpr uint32_t D = 256;
    constexpr uint32_t DHt_stage2 = D / 32;
    constexpr uint32_t chunks = N / 32;
    constexpr uint32_t q_rm = 23, q_in = 24, k_rm = 25, k_in = 26, scores_id = 28;

    CircularBuffer qcb(q_in), kcb(k_in), scores(scores_id);
    compute_kernel_hw_startup(q_rm, q_in);
    compute_kernel_lib::tilize<DHt_stage2, q_rm, q_in>(1, 32);
    qcb.wait_front(DHt_stage2);
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        compute_kernel_lib::tilize<DHt_stage2, k_rm, k_in>(1, 32);
        kcb.wait_front(DHt_stage2);
        scores.reserve_back(1);
        reconfig_full_operand(k_in, q_in);
        pack_reconfig_data_format(scores_id);
        mm_no_mop_init_short(q_in, k_in, true, 1, 1, DHt_stage2);
        configure_row_pack_width(scores_id, 1);
        blocked_matmul_and_pack<true, 1, 1>(
            q_in, k_in, scores_id, 0, 0, 0, 0, 1, 1,
            DHt_stage2, DHt_stage2, true);
        scores.push_back(1);
        kcb.pop_front(DHt_stage2);
    }
    qcb.pop_front(DHt_stage2);
}

void kernel_main() {
    stage2_main();
    sparse_tail_main();
}
"""


FULL_READER = r"""
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"

constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);
    constexpr uint32_t D = get_compile_time_arg_val(2);
    constexpr auto sqa = TensorAccessorArgs<3>();
    constexpr auto ska = TensorAccessorArgs<sqa.next_compile_time_args_offset()>();
    constexpr auto cia = TensorAccessorArgs<ska.next_compile_time_args_offset()>();
    constexpr auto aqa = TensorAccessorArgs<cia.next_compile_time_args_offset()>();
    constexpr auto kva = TensorAccessorArgs<aqa.next_compile_time_args_offset()>();

    const uint32_t sq_addr = get_arg_val<uint32_t>(0);
    const uint32_t sk_addr = get_arg_val<uint32_t>(1);
    const uint32_t cand_addr = get_arg_val<uint32_t>(2);
    const uint32_t aq_addr = get_arg_val<uint32_t>(3);
    const uint32_t kv_addr = get_arg_val<uint32_t>(4);
    const uint32_t slot = get_arg_val<uint32_t>(5);
    const uint32_t context = get_arg_val<uint32_t>(6);

    const auto sq = TensorAccessor(sqa, sq_addr, D * 2);
    const auto sk = TensorAccessor(ska, sk_addr, D);
    const auto cand = TensorAccessor(cia, cand_addr, N * 4);
    const auto aq = TensorAccessor(aqa, aq_addr, 2 * D * 2);
    const auto kv = TensorAccessor(kva, kv_addr, 2 * D);

    // ---- Stage II: one owner core, no remote-L1 addresses. ----
    cb_reserve_back(27, 1);
    const uint32_t cand_l1 = get_write_ptr(27);
    noc_async_read(get_noc_addr(slot, cand), cand_l1, N * 4);
    cb_reserve_back(23, 32);
    const uint32_t sq_l1 = get_write_ptr(23);
    for (uint32_t row = 0; row < 32; ++row) {
        noc_async_read(get_noc_addr(slot * 32 + row, sq), sq_l1 + row * D * 2, D * 2);
    }
    noc_async_read_barrier();
    cb_push_back(27, 1);
    cb_push_back(23, 32);

    volatile tt_l1_ptr uint32_t* candidates =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_l1);
    for (uint32_t chunk = 0; chunk < N / 32; ++chunk) {
        cb_reserve_back(25, 32);
        const uint32_t dst = get_write_ptr(25);
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t page = slot * context + candidates[chunk * 32 + row];
            noc_async_read(get_noc_addr(page, sk), dst + row * D, D);
        }
        noc_async_read_barrier();
        cb_push_back(25, 32);
    }

    // Writer publishes G local top-k rows into cb17.  No GDDR index tensor and
    // no cross-core semaphore are involved in the handoff.
    cb_wait_front(17, G);
    const uint32_t all_idx = get_read_ptr(17);
    Noc noc;
    experimental::CB q_cb(0), k_cb(2), mask_cb(5), ctrl_cb(18);
    experimental::CB req_cb(21), ack_cb(22);

    constexpr uint32_t H = 32, S = G, KCHUNK = 128;
    constexpr uint32_t q_row_bytes = 2 * D * 2;
    constexpr uint32_t kv_row_bytes = 2 * D;
    const uint32_t kv_page_offset = slot * context;
    for (uint32_t tok = 0; tok < G; ++tok) {
        q_cb.reserve_back(H);
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_read(aq, q_cb, q_row_bytes,
                           {.page_id = h * S + tok}, {.offset_bytes = h * q_row_bytes});
        }
        noc.async_read_barrier();
        q_cb.push_back(H);

        volatile tt_l1_ptr uint32_t* idx =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(all_idx + tok * 128 * 4);
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

        // 100 valid keys: the fourth key tile has four valid columns.
        mask_cb.reserve_back(1);
        volatile tt_l1_ptr uint16_t* mask =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_cb.get_write_ptr());
        for (uint32_t col = 0; col < 32; ++col) {
            const uint32_t off = col < 16 ? col : 256 + col - 16;
            mask[off] = col >= 4 ? neg_inf_bf16 : uint16_t(0);
        }
        mask_cb.push_back(1);
    }
    cb_pop_front(17, G);
}
"""


FULL_WRITER = r"""
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"

constexpr uint32_t one_bf16_packed = 0x3F803F80u;
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

FORCE_INLINE uint16_t ordered_bf16_full(uint16_t bits) {
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
}
FORCE_INLINE uint16_t stage_score(uint32_t base, uint32_t candidate, uint32_t row) {
    const uint32_t tile = base + (candidate >> 5) * 2048;
    const uint32_t col = candidate & 31;
    const uint32_t face_row = (row < 16 ? 0 : 1024) + (row & 15) * 32;
    const uint32_t off = face_row + (col < 16 ? col * 2 : 512 + (col - 16) * 2);
    return *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile + off);
}

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t D = get_compile_time_arg_val(3);
    constexpr auto oa = TensorAccessorArgs<4>();
    constexpr auto kva = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t context = get_arg_val<uint32_t>(3);
    const auto out = TensorAccessor(oa, out_addr, D * 2);
    const auto kv = TensorAccessor(kva, kv_addr, 2 * D);

    // Build sparse compute's persistent helper tiles while Stage II's reader
    // and matrix engine are busy.
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

    cb_wait_front(27, 1);
    cb_wait_front(28, N / 32);
    volatile tt_l1_ptr uint32_t* candidates =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(27));
    const uint32_t scores = get_read_ptr(28);

    // Exact local BF16 heaps.  cb17 is the SRAM handoff to the reader.
    for (uint32_t row = 0; row < G; ++row) {
        uint16_t hv[K]; uint32_t hi[K];
        for (uint32_t j = 0; j < K; ++j) {
            hv[j] = ordered_bf16_full(stage_score(scores, j, row)); hi[j] = candidates[j];
        }
        for (int32_t root = K / 2 - 1; root >= 0; --root) {
            uint16_t key = hv[root]; uint32_t idx = hi[root], pos = root;
            while (true) {
                uint32_t a = pos * 2 + 1;
                if (a >= K) {
                    break;
                }
                uint32_t z = a + 1;
                uint32_t c = a;
                if (z < K && hv[z] < hv[a]) {
                    c = z;
                }
                if (hv[c] >= key) {
                    break;
                }
                hv[pos] = hv[c];
                hi[pos] = hi[c];
                pos = c;
            }
            hv[pos] = key;
            hi[pos] = idx;
        }
        for (uint32_t j = K; j < N; ++j) {
            uint16_t key = ordered_bf16_full(stage_score(scores, j, row));
            if (key <= hv[0]) {
                continue;
            }
            uint32_t pos = 0;
            while (true) {
                uint32_t a = pos * 2 + 1;
                if (a >= K) {
                    break;
                }
                uint32_t z = a + 1;
                uint32_t c = a;
                if (z < K && hv[z] < hv[a]) {
                    c = z;
                }
                if (hv[c] >= key) {
                    break;
                }
                hv[pos] = hv[c];
                hi[pos] = hi[c];
                pos = c;
            }
            hv[pos] = key;
            hi[pos] = candidates[j];
        }
        cb_reserve_back(17, 1);
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(17));
        for (uint32_t j = 0; j < 128; ++j) dst[j] = j < K ? hi[j] : 0xffffffffu;
        cb_push_back(17, 1);
    }
    cb_pop_front(28, N / 32);
    cb_pop_front(27, 1);

    Noc noc;
    experimental::CB req_cb(21), ack_cb(22), idx_cb(17), out_cb(16);
    const uint32_t all_idx = idx_cb.get_read_ptr();
    constexpr uint32_t H = 32, S = G, block_tiles = D / 32;
    const uint32_t kv_page_offset = slot * context;
    for (uint32_t tok = 0; tok < G; ++tok) {
        req_cb.wait_front(1);
        volatile tt_l1_ptr uint32_t* req =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(req_cb.get_read_ptr());
        const uint32_t split = req[sparse_sdpa::gather_request::SPLIT];
        const uint32_t dst_l1 = req[sparse_sdpa::gather_request::DST_L1];
        req_cb.pop_front(1);
        volatile tt_l1_ptr uint32_t* idx =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(all_idx + tok * 128 * 4);
        sparse_sdpa::trid_ring_gather<false, 1, 1, 0, 0, sparse_sdpa::K_TRID_RING>(
            noc, kv, dst_l1, idx, 0, 0, split, 2 * D, kv_page_offset);
        ack_cb.reserve_back(1); ack_cb.push_back(1);

        out_cb.wait_front(block_tiles);
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_write(out_cb, out, D * 2,
                            {.offset_bytes = h * D * 2}, {.page_id = h * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);
    }
}
"""


# Workers split one KV head's candidate pool.  Every worker computes all four
# GQA rows.  ``HEAD_PARALLEL=false`` sends three full packets to one owner;
# ``HEAD_PARALLEL=true`` exchanges one row with each peer so worker h merges and
# attends query head h.  Both topologies have one NoC dependency round.
SHARDED_COMPUTE = r"""
#define kernel_main sparse_tail_main
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sparse_sdpa_compute.cpp"
#undef kernel_main

void sharded_stage2_main() {
    constexpr uint32_t LOCAL_PAD =
        get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END);
    constexpr uint32_t D = 256;
    constexpr uint32_t DHt_stage2 = D / 32;
    constexpr uint32_t chunks = LOCAL_PAD / 32;
    constexpr uint32_t q_rm = 23, q_in = 24, k_rm = 25, k_in = 26, scores_id = 28;

    CircularBuffer qcb(q_in), kcb(k_in), scores(scores_id);
    compute_kernel_hw_startup(q_rm, q_in);
    compute_kernel_lib::tilize<DHt_stage2, q_rm, q_in>(1, 32);
    qcb.wait_front(DHt_stage2);
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        compute_kernel_lib::tilize<DHt_stage2, k_rm, k_in>(1, 32);
        kcb.wait_front(DHt_stage2);
        scores.reserve_back(1);
        reconfig_full_operand(k_in, q_in);
        pack_reconfig_data_format(scores_id);
        mm_no_mop_init_short(q_in, k_in, true, 1, 1, DHt_stage2);
        configure_row_pack_width(scores_id, 1);
        blocked_matmul_and_pack<true, 1, 1>(
            q_in, k_in, scores_id, 0, 0, 0, 0, 1, 1,
            DHt_stage2, DHt_stage2, true);
        scores.push_back(1);
        kcb.pop_front(DHt_stage2);
    }
    qcb.pop_front(DHt_stage2);
}

void kernel_main() {
    sharded_stage2_main();
    if (get_arg_val<uint32_t>(2) != 0) {
        sparse_tail_main();
    }
}
"""


SHARDED_READER = r"""
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"

constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t D = get_compile_time_arg_val(3);
    constexpr bool HEAD_PARALLEL = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t LOCAL_N = N / W;
    constexpr uint32_t LOCAL_PAD = ((LOCAL_N + 31) / 32) * 32;
    constexpr auto sqa = TensorAccessorArgs<5>();
    constexpr auto ska = TensorAccessorArgs<sqa.next_compile_time_args_offset()>();
    constexpr auto cia = TensorAccessorArgs<ska.next_compile_time_args_offset()>();
    constexpr auto aqa = TensorAccessorArgs<cia.next_compile_time_args_offset()>();
    constexpr auto kva = TensorAccessorArgs<aqa.next_compile_time_args_offset()>();
    static_assert(N % W == 0);

    const uint32_t sq_addr = get_arg_val<uint32_t>(0);
    const uint32_t sk_addr = get_arg_val<uint32_t>(1);
    const uint32_t cand_addr = get_arg_val<uint32_t>(2);
    const uint32_t aq_addr = get_arg_val<uint32_t>(3);
    const uint32_t kv_addr = get_arg_val<uint32_t>(4);
    const uint32_t slot = get_arg_val<uint32_t>(5);
    const uint32_t context = get_arg_val<uint32_t>(6);
    const uint32_t rank = get_arg_val<uint32_t>(7);

    const auto sq = TensorAccessor(sqa, sq_addr, D * 2);
    const auto sk = TensorAccessor(ska, sk_addr, D);
    const auto cand = TensorAccessor(cia, cand_addr, N * 4);
    const auto aq = TensorAccessor(aqa, aq_addr, 2 * D * 2);
    const auto kv = TensorAccessor(kva, kv_addr, 2 * D);

    cb_reserve_back(27, 1);
    const uint32_t cand_l1 = get_write_ptr(27);
    // Blackhole DRAM starts are physically 64-byte granular.  LOCAL_N=200
    // makes odd shard offsets land at +32.  Read from the preceding aligned
    // address and skip that prefix in L1 instead of silently receiving the
    // preceding eight IDs.
    const uint32_t cand_offset = rank * LOCAL_N * 4;
    const uint32_t cand_prefix = cand_offset & 63;
    noc_async_read(get_noc_addr(slot, cand) + cand_offset - cand_prefix,
                   cand_l1, LOCAL_N * 4 + cand_prefix);
    cb_reserve_back(23, 32);
    const uint32_t sq_l1 = get_write_ptr(23);
    for (uint32_t row = 0; row < 32; ++row) {
        noc_async_read(get_noc_addr(slot * 32 + row, sq),
                       sq_l1 + row * D * 2, D * 2);
    }
    noc_async_read_barrier();
    cb_push_back(27, 1);
    cb_push_back(23, 32);

    volatile tt_l1_ptr uint32_t* candidates =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_l1 + cand_prefix);
    for (uint32_t chunk = 0; chunk < LOCAL_PAD / 32; ++chunk) {
        cb_reserve_back(25, 32);
        const uint32_t dst = get_write_ptr(25);
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t local = chunk * 32 + row;
            // The score writer ignores [LOCAL_N,LOCAL_PAD); a repeated valid
            // record avoids a scalar 6 KiB zero-fill on the dataflow RISC.
            const uint32_t natural = candidates[local < LOCAL_N ? local : 0];
            const uint32_t page = slot * context + natural;
            noc_async_read(get_noc_addr(page, sk), dst + row * D, D);
        }
        noc_async_read_barrier();
        cb_push_back(25, 32);
    }

    if constexpr (!HEAD_PARALLEL) {
        if (rank != 0) {
            return;
        }
    }

    // Flat mode receives all G rows on rank zero.  Head-parallel mode receives
    // one row on every worker, with worker rank equal to the active query head.
    constexpr uint32_t ACTIVE = HEAD_PARALLEL ? 1 : G;
    const uint32_t token_start = HEAD_PARALLEL ? rank : 0;
    cb_wait_front(17, ACTIVE);
    const uint32_t all_idx = get_read_ptr(17);
    Noc noc;
    experimental::CB q_cb(0), k_cb(2), mask_cb(5), ctrl_cb(18);
    experimental::CB req_cb(21), ack_cb(22);

    constexpr uint32_t H = 32, S = G, KCHUNK = 128;
    constexpr uint32_t q_row_bytes = 2 * D * 2;
    constexpr uint32_t kv_row_bytes = 2 * D;
    const uint32_t kv_page_offset = slot * context;
    for (uint32_t active = 0; active < ACTIVE; ++active) {
        const uint32_t tok = token_start + active;
        q_cb.reserve_back(H);
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_read(aq, q_cb, q_row_bytes,
                           {.page_id = h * S + tok}, {.offset_bytes = h * q_row_bytes});
        }
        noc.async_read_barrier();
        q_cb.push_back(H);

        volatile tt_l1_ptr uint32_t* idx =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                all_idx + active * 128 * 4);
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
    }
    cb_pop_front(17, ACTIVE);
}
"""


SHARDED_WRITER = semaphore_defines(SEM) + r"""
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"

constexpr uint32_t one_bf16_packed = 0x3F803F80u;
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

FORCE_INLINE uint16_t ordered_bf16_shard(uint16_t bits) {
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
}
FORCE_INLINE uint16_t shard_score(uint32_t base, uint32_t candidate, uint32_t row) {
    const uint32_t tile = base + (candidate >> 5) * 2048;
    const uint32_t col = candidate & 31;
    const uint32_t face_row = (row < 16 ? 0 : 1024) + (row & 15) * 32;
    const uint32_t off = face_row + (col < 16 ? col * 2 : 512 + (col - 16) * 2);
    return *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile + off);
}

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t D = get_compile_time_arg_val(4);
    constexpr bool DEBUG_INDICES = get_compile_time_arg_val(5) != 0;
    constexpr bool HEAD_PARALLEL = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t LOCAL_N = N / W;
    constexpr uint32_t LOCAL_PAD = ((LOCAL_N + 31) / 32) * 32;
    constexpr uint32_t PACKET_WORDS = G * K * 2;
    constexpr uint32_t PACKET_BYTES = PACKET_WORDS * 4;
    constexpr uint32_t ROW_WORDS = K * 2;
    constexpr uint32_t ROW_BYTES = ROW_WORDS * 4;
    constexpr auto oa = TensorAccessorArgs<7>();
    constexpr auto kva = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto dia = TensorAccessorArgs<kva.next_compile_time_args_offset()>();
    static_assert(N % W == 0);
    static_assert(K == 100);
    static_assert(!HEAD_PARALLEL || W == G);

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t context = get_arg_val<uint32_t>(3);
    const uint32_t rank = get_arg_val<uint32_t>(4);
    const uint32_t owner_x = get_arg_val<uint32_t>(5);
    const uint32_t owner_y = get_arg_val<uint32_t>(6);
    const uint32_t debug_addr = get_arg_val<uint32_t>(7);
    const auto out = TensorAccessor(oa, out_addr, D * 2);
    const auto kv = TensorAccessor(kva, kv_addr, 2 * D);
    const auto debug = TensorAccessor(dia, debug_addr, PACKET_BYTES);

    // Flat mode has one sparse-tail owner.  In head-parallel mode every worker
    // owns one query head and therefore prepares its own helper tiles.
    if (HEAD_PARALLEL || rank == 0) {
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

    cb_wait_front(27, 1);
    cb_wait_front(28, LOCAL_PAD / 32);
    const uint32_t cand_prefix = (rank * LOCAL_N * 4) & 63;
    volatile tt_l1_ptr uint32_t* candidates =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_read_ptr(27) + cand_prefix);
    const uint32_t scores = get_read_ptr(28);

    cb_reserve_back(29, W);
    const uint32_t receive = get_write_ptr(29);
    cb_reserve_back(30, 1);
    const uint32_t local_packet = get_write_ptr(30);
    volatile tt_l1_ptr uint32_t* packet =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            (!HEAD_PARALLEL && rank == 0) ? receive : local_packet);

    for (uint32_t row = 0; row < G; ++row) {
        uint16_t hv[K];
        uint32_t hi[K];
        for (uint32_t j = 0; j < K; ++j) {
            hv[j] = ordered_bf16_shard(shard_score(scores, j, row));
            hi[j] = candidates[j];
        }
        for (int32_t root = K / 2 - 1; root >= 0; --root) {
            uint16_t key = hv[root];
            uint32_t idx = hi[root];
            uint32_t pos = root;
            while (true) {
                uint32_t a = pos * 2 + 1;
                if (a >= K) break;
                uint32_t z = a + 1;
                uint32_t c = a;
                if (z < K && hv[z] < hv[a]) c = z;
                if (hv[c] >= key) break;
                hv[pos] = hv[c];
                hi[pos] = hi[c];
                pos = c;
            }
            hv[pos] = key;
            hi[pos] = idx;
        }
        for (uint32_t j = K; j < LOCAL_N; ++j) {
            uint16_t key = ordered_bf16_shard(shard_score(scores, j, row));
            if (key <= hv[0]) continue;
            uint32_t pos = 0;
            while (true) {
                uint32_t a = pos * 2 + 1;
                if (a >= K) break;
                uint32_t z = a + 1;
                uint32_t c = a;
                if (z < K && hv[z] < hv[a]) c = z;
                if (hv[c] >= key) break;
                hv[pos] = hv[c];
                hi[pos] = hi[c];
                pos = c;
            }
            hv[pos] = key;
            hi[pos] = candidates[j];
        }
        volatile tt_l1_ptr uint32_t* dst = packet + row * K * 2;
        for (uint32_t j = 0; j < K; ++j) {
            dst[2 * j] = hv[j];
            dst[2 * j + 1] = hi[j];
        }
    }
    if constexpr (DEBUG_INDICES) {
        noc_async_write(reinterpret_cast<uint32_t>(packet),
                        get_noc_addr(rank, debug), PACKET_BYTES);
        noc_async_write_barrier();
    }
    cb_pop_front(28, LOCAL_PAD / 32);
    cb_pop_front(27, 1);

    volatile tt_l1_ptr uint32_t* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_FRONTIERS_READY));
    if constexpr (HEAD_PARALLEL) {
        // One all-to-all row exchange. Worker h receives head h from every
        // candidate shard, so no second redistribution is needed before the
        // sparse-attention tail. The three remote writes are issued together
        // and followed by one arrival semaphore per destination.
        volatile tt_l1_ptr uint32_t* own_src = packet + rank * ROW_WORDS;
        volatile tt_l1_ptr uint32_t* own_dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                receive + rank * ROW_BYTES);
        for (uint32_t j = 0; j < ROW_WORDS; ++j) {
            own_dst[j] = own_src[j];
        }
        for (uint32_t dest = 0; dest < W; ++dest) {
            if (dest == rank) {
                continue;
            }
            const uint32_t dest_x = get_arg_val<uint32_t>(8 + 2 * dest);
            const uint32_t dest_y = get_arg_val<uint32_t>(9 + 2 * dest);
            noc_async_write(
                reinterpret_cast<uint32_t>(packet + dest * ROW_WORDS),
                get_noc_addr(dest_x, dest_y, receive + rank * ROW_BYTES),
                ROW_BYTES);
        }
        noc_async_write_barrier();
        for (uint32_t dest = 0; dest < W; ++dest) {
            if (dest == rank) {
                continue;
            }
            const uint32_t dest_x = get_arg_val<uint32_t>(8 + 2 * dest);
            const uint32_t dest_y = get_arg_val<uint32_t>(9 + 2 * dest);
            noc_semaphore_inc(
                get_noc_addr(dest_x, dest_y, get_semaphore(SKV4_SEM_FRONTIERS_READY)), 1);
        }
        noc_semaphore_wait_min(ready, W - 1);
        noc_semaphore_set(ready, 0);
    } else {
        if (rank != 0) {
            noc_async_write(
                local_packet,
                get_noc_addr(owner_x, owner_y, receive + rank * PACKET_BYTES),
                PACKET_BYTES);
            noc_async_write_barrier();
            noc_semaphore_inc(
                get_noc_addr(owner_x, owner_y, get_semaphore(SKV4_SEM_FRONTIERS_READY)), 1);
            return;
        }
        noc_semaphore_wait_min(ready, W - 1);
        noc_semaphore_set(ready, 0);
    }

    // Merge W exact local top-K sets.  This is globally exact: an item outside
    // its partition's top K cannot be in the global top K.
    constexpr uint32_t ACTIVE = HEAD_PARALLEL ? 1 : G;
    const uint32_t row_start = HEAD_PARALLEL ? rank : 0;
    for (uint32_t active = 0; active < ACTIVE; ++active) {
        const uint32_t row = row_start + active;
        uint16_t hv[K];
        uint32_t hi[K];
        const uint32_t row_offset = HEAD_PARALLEL ? 0 : row * ROW_BYTES;
        const uint32_t worker_stride = HEAD_PARALLEL ? ROW_BYTES : PACKET_BYTES;
        volatile tt_l1_ptr uint32_t* first =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                receive + row_offset);
        for (uint32_t j = 0; j < K; ++j) {
            hv[j] = static_cast<uint16_t>(first[2 * j]);
            hi[j] = first[2 * j + 1];
        }
        for (int32_t root = K / 2 - 1; root >= 0; --root) {
            uint16_t key = hv[root];
            uint32_t idx = hi[root];
            uint32_t pos = root;
            while (true) {
                uint32_t a = pos * 2 + 1;
                if (a >= K) break;
                uint32_t z = a + 1;
                uint32_t c = a;
                if (z < K && hv[z] < hv[a]) c = z;
                if (hv[c] >= key) break;
                hv[pos] = hv[c]; hi[pos] = hi[c]; pos = c;
            }
            hv[pos] = key; hi[pos] = idx;
        }
        for (uint32_t worker = 1; worker < W; ++worker) {
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    receive + worker * worker_stride + row_offset);
            for (uint32_t j = 0; j < K; ++j) {
                uint16_t key = static_cast<uint16_t>(src[2 * j]);
                if (key <= hv[0]) continue;
                uint32_t pos = 0;
                while (true) {
                    uint32_t a = pos * 2 + 1;
                    if (a >= K) break;
                    uint32_t z = a + 1;
                    uint32_t c = a;
                    if (z < K && hv[z] < hv[a]) c = z;
                    if (hv[c] >= key) break;
                    hv[pos] = hv[c]; hi[pos] = hi[c]; pos = c;
                }
                hv[pos] = key;
                hi[pos] = src[2 * j + 1];
            }
        }
        cb_reserve_back(17, 1);
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(17));
        for (uint32_t j = 0; j < 128; ++j) {
            dst[j] = j < K ? hi[j] : 0xffffffffu;
        }
        if constexpr (DEBUG_INDICES) {
            noc_async_write(reinterpret_cast<uint32_t>(dst),
                            get_noc_addr(W, debug) + row * 128 * 4, 128 * 4);
        }
        cb_push_back(17, 1);
    }
    if constexpr (DEBUG_INDICES) {
        noc_async_write_barrier();
    }

    Noc noc;
    experimental::CB req_cb(21), ack_cb(22), idx_cb(17), out_cb(16);
    const uint32_t all_idx = idx_cb.get_read_ptr();
    constexpr uint32_t H = 32, S = G, block_tiles = D / 32;
    const uint32_t kv_page_offset = slot * context;
    for (uint32_t active = 0; active < ACTIVE; ++active) {
        const uint32_t tok = row_start + active;
        req_cb.wait_front(1);
        volatile tt_l1_ptr uint32_t* req =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(req_cb.get_read_ptr());
        const uint32_t split = req[sparse_sdpa::gather_request::SPLIT];
        const uint32_t dst_l1 = req[sparse_sdpa::gather_request::DST_L1];
        req_cb.pop_front(1);
        volatile tt_l1_ptr uint32_t* idx =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                all_idx + active * 128 * 4);
        sparse_sdpa::trid_ring_gather<false, 1, 1, 0, 0, sparse_sdpa::K_TRID_RING>(
            noc, kv, dst_l1, idx, 0, 0, split, 2 * D, kv_page_offset);
        ack_cb.reserve_back(1);
        ack_cb.push_back(1);

        out_cb.wait_front(block_tiles);
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_write(out_cb, out, D * 2,
                            {.offset_bytes = h * D * 2}, {.page_id = h * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);
    }
}
"""


def _ranges(ttnn, points):
    return ttnn.CoreRangeSet([
        ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
        for x, y in points
    ])


def _points(device):
    grid = device.compute_with_storage_grid_size()
    return [(x, y) for y in range(grid.y) for x in range(grid.x)]


def _compact_groups(device, slots, workers, core_points=None):
    """Return ``slots`` compact, non-overlapping logical-core groups."""
    if core_points is not None:
        points = list(core_points)
        need = slots * workers
        if len(points) < need:
            raise ValueError(f"need {need} worker cores, got {len(points)}")
        return [points[i * workers:(i + 1) * workers] for i in range(slots)]

    grid = device.compute_with_storage_grid_size()
    groups = []
    # Keep each fan-in group within a physical row.  Logical-to-physical
    # remapping is supplied explicitly to the kernels, so this primarily avoids
    # accidental long groups at a row-major wraparound.
    for y in range(grid.y):
        for x0 in range(0, grid.x - workers + 1, workers):
            groups.append([(x0 + rank, y) for rank in range(workers)])
            if len(groups) == slots:
                return groups
    raise ValueError(
        f"need {slots * workers} cores in compact {workers}-worker groups; "
        f"grid {grid.x}x{grid.y} supplies only {len(groups) * workers} this way"
    )


def build(ttnn, device, q, keys, indices, *, slots, context, n, group=GQA_GROUP,
          topk=TOPK, core_points=None):
    """Build one no-cross-core selector dispatch and return its outputs."""
    if n % 32:
        raise ValueError("stage-II candidate count must be a multiple of 32")
    if topk != TOPK:
        raise ValueError(f"fused selector is currently specialized for topk={TOPK}")
    if group != GQA_GROUP:
        raise ValueError(f"fused selector is currently specialized for G={GQA_GROUP}")
    points = list(core_points) if core_points is not None else _points(device)[:slots]
    if len(points) < slots:
        raise ValueError(f"need {slots} owner cores, got {len(points)}")
    points = points[:slots]
    cores = _ranges(ttnn, points)

    outputs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, group, PADDED_TOPK)), ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(slots)
    ]

    qct = list(ttnn.TensorAccessorArgs(q).get_compile_time_args())
    kct = list(ttnn.TensorAccessorArgs(keys).get_compile_time_args())
    ict = list(ttnn.TensorAccessorArgs(indices).get_compile_time_args())
    oct = list(ttnn.TensorAccessorArgs(outputs[0]).get_compile_time_args())

    rr = ttnn.RuntimeArgs()
    wr = ttnn.RuntimeArgs()
    for slot, (x, y) in enumerate(points):
        rr[x][y] = [q.buffer_address(), keys.buffer_address(), indices.buffer_address(),
                    slot, context]
        wr[x][y] = [outputs[slot].buffer_address()]

    reader = ttnn.KernelDescriptor(
        kernel_source=READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[n, group, D, *qct, *kct, *ict],
        runtime_args=rr,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[n, D],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            # Blackhole requires 32-bit DEST whenever any CB on the core uses
            # native E4M3, even though the final score tile is packed BF16.
            fp32_dest_acc_en=True,
        ),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[n, group, topk, PADDED_TOPK, *oct],
        runtime_args=wr,
        config=ttnn.WriterConfigDescriptor(),
    )

    score_tiles = n // 32
    cbs = [
        ttnn.CBDescriptor(
            total_size=32 * D * 2, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=0, data_format=ttnn.bfloat16, page_size=D * 2)],
        ),
        ttnn.CBDescriptor(
            total_size=(D // 32) * 2048, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=1, data_format=ttnn.bfloat16, page_size=2048)],
        ),
        ttnn.CBDescriptor(
            total_size=64 * D, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=2, data_format=ttnn.fp8_e4m3, page_size=D)],
        ),
        ttnn.CBDescriptor(
            total_size=2 * (D // 32) * ttnn.tile_size(ttnn.bfloat8_b),
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=3, data_format=ttnn.bfloat8_b,
                page_size=ttnn.tile_size(ttnn.bfloat8_b))],
        ),
        ttnn.CBDescriptor(
            total_size=score_tiles * 2048, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=16, data_format=ttnn.bfloat16, page_size=2048)],
        ),
        ttnn.CBDescriptor(
            total_size=n * 4, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=24, data_format=ttnn.uint32, page_size=n * 4)],
        ),
        ttnn.CBDescriptor(
            total_size=PADDED_TOPK * 4, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=25, data_format=ttnn.uint32,
                page_size=PADDED_TOPK * 4)],
        ),
    ]
    descriptor = ttnn.ProgramDescriptor(kernels=[reader, compute, writer], cbs=cbs)
    io = [q, keys, indices, *outputs]

    def run():
        ttnn.generic_op(io, descriptor)
        return outputs

    return {
        "run": run,
        "outputs": outputs,
        "descriptor": descriptor,
        "io": io,
        "owner_cores": points,
    }


def build_full(ttnn, device, q, keys, indices, q_attention, kv, *, slots,
               context, n, group=GQA_GROUP, topk=TOPK, core_points=None):
    """Build Stage II and final sparse attention as one owner-core dispatch."""
    if n % 32 or topk != TOPK or group != GQA_GROUP:
        raise ValueError("full fused kernel requires N%32=0, G=4, topk=100")
    points = list(core_points) if core_points is not None else _points(device)[:slots]
    if len(points) < slots:
        raise ValueError(f"need {slots} owner cores, got {len(points)}")
    points = points[:slots]
    cores = _ranges(ttnn, points)

    outputs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, ROWS, group, D)), ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(slots)
    ]

    sqct = list(ttnn.TensorAccessorArgs(q).get_compile_time_args())
    skct = list(ttnn.TensorAccessorArgs(keys).get_compile_time_args())
    cict = list(ttnn.TensorAccessorArgs(indices).get_compile_time_args())
    aqct = list(ttnn.TensorAccessorArgs(q_attention[0]).get_compile_time_args())
    kvct = list(ttnn.TensorAccessorArgs(kv).get_compile_time_args())
    oct = list(ttnn.TensorAccessorArgs(outputs[0]).get_compile_time_args())

    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for slot, (x, y) in enumerate(points):
        rr[x][y] = [q.buffer_address(), keys.buffer_address(), indices.buffer_address(),
                    q_attention[slot].buffer_address(), kv.buffer_address(), slot, context]
        wr[x][y] = [outputs[slot].buffer_address(), kv.buffer_address(), slot, context]
        cr[x][y] = [0, group]  # sparse compute token start/count

    reader = ttnn.KernelDescriptor(
        kernel_source=FULL_READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[n, group, D, *sqct, *skct, *cict, *aqct, *kvct],
        runtime_args=rr,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=FULL_WRITER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[n, group, topk, D, *oct, *kvct],
        runtime_args=wr,
        config=ttnn.WriterConfigDescriptor(),
    )

    scale_bits = struct.unpack("<I", struct.pack("<f", D ** -0.5))[0]
    # Exact fixed-shape specialization of sparse_sdpa_program_factory's
    # compute-time argument vector, followed by the Stage-II candidate count.
    compute_ct = [
        ROWS, 2 * D // 32, D // 32, PADDED_TOPK // 32, scale_bits,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        15, 16, 18, 19, 20,
        0, 23, 24, 25, 26, 0, 1, 0,
        n,
    ]
    compute = ttnn.KernelDescriptor(
        kernel_source=FULL_COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=cr,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        ),
    )

    tile16 = 2048
    tile8 = ttnn.tile_size(ttnn.bfloat8_b)

    def cb(index, dtype, page_size, pages):
        return ttnn.CBDescriptor(
            total_size=page_size * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    # Sparse-SDPA CBs 0..22 (plain row-major E4M3 cache, fixed H=32,
    # S=G=4, K-chunk=128, K-dim=512, V-dim=256).
    cbs = [
        cb(0, ttnn.bfloat16, 2 * D * 2, ROWS),
        cb(1, ttnn.bfloat16, tile16, 2 * D // 32),
        cb(2, ttnn.fp8_e4m3, 2 * D, PADDED_TOPK),
        cb(3, ttnn.bfloat8_b, tile8, (PADDED_TOPK // 32) * (2 * D // 32)),
        cb(4, ttnn.bfloat16, tile16, 1),
        cb(5, ttnn.bfloat16, tile16, 1),
        cb(6, ttnn.bfloat16, tile16, 1),
        cb(7, ttnn.bfloat16, tile16, PADDED_TOPK // 32),
        cb(8, ttnn.bfloat16, tile16, 1),
        cb(9, ttnn.bfloat16, tile16, 1),
        cb(10, ttnn.bfloat16, tile16, 1),
        cb(11, ttnn.bfloat16, tile16, 1),
        cb(12, ttnn.bfloat16, tile16, D // 32),
        cb(13, ttnn.bfloat16, tile16, D // 32),
        cb(14, ttnn.bfloat16, tile16, 1),
        cb(15, ttnn.bfloat16, tile16, D // 32),
        cb(16, ttnn.bfloat16, tile16, D // 32),
        cb(17, ttnn.uint32, PADDED_TOPK * 4, group),
        cb(18, ttnn.bfloat16, 16, 2),
        cb(19, ttnn.bfloat16, tile16, 1),
        cb(20, ttnn.bfloat16, tile16, 1),
        cb(21, ttnn.bfloat16, 32, 2),
        cb(22, ttnn.bfloat16, 16, 2),
        # Stage-II CBs.  IDs 23..26 are unused by plain sparse_sdpa.
        cb(23, ttnn.bfloat16, D * 2, ROWS),
        cb(24, ttnn.bfloat16, tile16, D // 32),
        cb(25, ttnn.fp8_e4m3, D, 64),
        cb(26, ttnn.bfloat8_b, tile8, 2 * (D // 32)),
        cb(27, ttnn.uint32, n * 4, 1),
        cb(28, ttnn.bfloat16, tile16, n // 32),
    ]
    descriptor = ttnn.ProgramDescriptor(kernels=[reader, compute, writer], cbs=cbs)
    io = [q, keys, indices, *q_attention, kv, *outputs]

    def run():
        ttnn.generic_op(io, descriptor)
        return outputs

    return {
        "run": run,
        "outputs": outputs,
        "descriptor": descriptor,
        "io": io,
        "owner_cores": points,
    }


def build_full_sharded(ttnn, device, q, keys, indices, q_attention, kv, *,
                       slots, context, n, workers=4, group=GQA_GROUP,
                       topk=TOPK, core_points=None, debug_indices=False,
                       head_parallel=False):
    """Build exact worker-sharded Stage II plus sparse attention.

    Candidate rows are split evenly across ``workers``.  Each worker computes
    all G query-head score rows.  In flat mode, every non-owner sends one
    ``G * topk`` packet to the owner, which merges all rows and runs all sparse
    tails.  In head-parallel mode, worker h receives row h from every shard,
    merges that row, and runs one sparse tail.  Both mappings use one NoC
    dependency round; head-parallel mode requires ``workers == group``.
    """
    if workers < 2:
        raise ValueError("sharded full kernel requires at least two workers")
    if n % workers or topk != TOPK or group != GQA_GROUP:
        raise ValueError("sharded full kernel requires N%workers=0, G=4, topk=100")
    if head_parallel and workers != group:
        raise ValueError("head-parallel sharding requires workers == GQA group")
    local_n = n // workers
    if local_n < topk:
        raise ValueError(
            f"each shard must contain at least topk={topk} candidates; got {local_n}"
        )
    local_pad = ((local_n + 31) // 32) * 32
    groups = _compact_groups(device, slots, workers, core_points)
    points = [point for members in groups for point in members]
    cores = _ranges(ttnn, points)

    outputs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, ROWS, group, D)), ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(slots)
    ]
    packet_bytes = group * topk * 2 * 4
    packet_words = packet_bytes // 4
    debug_outputs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, workers + 1, packet_words)), ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(slots)
    ]

    sqct = list(ttnn.TensorAccessorArgs(q).get_compile_time_args())
    skct = list(ttnn.TensorAccessorArgs(keys).get_compile_time_args())
    cict = list(ttnn.TensorAccessorArgs(indices).get_compile_time_args())
    aqct = list(ttnn.TensorAccessorArgs(q_attention[0]).get_compile_time_args())
    kvct = list(ttnn.TensorAccessorArgs(kv).get_compile_time_args())
    oct = list(ttnn.TensorAccessorArgs(outputs[0]).get_compile_time_args())
    dict_ = list(ttnn.TensorAccessorArgs(debug_outputs[0]).get_compile_time_args())

    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    owner_points = []
    for slot, members in enumerate(groups):
        owner = members[0]
        owner_points.append(owner)
        physical = [
            device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            for x, y in members
        ]
        owner_phys = physical[0]
        group_noc_coords = [
            coord for point in physical for coord in (point.x, point.y)
        ]
        for rank, (x, y) in enumerate(members):
            rr[x][y] = [
                q.buffer_address(), keys.buffer_address(), indices.buffer_address(),
                q_attention[slot].buffer_address(), kv.buffer_address(), slot,
                context, rank,
            ]
            wr[x][y] = [
                outputs[slot].buffer_address(), kv.buffer_address(), slot,
                context, rank, owner_phys.x, owner_phys.y,
                debug_outputs[slot].buffer_address(),
                *group_noc_coords,
            ]
            # The stock sparse compute runtime starts with token offset/count.
            # Head-parallel workers each own one token/head; flat non-owners
            # execute only Stage II and return before the sparse tail.
            cr[x][y] = ([rank, 1, 1] if head_parallel
                        else [0, group, int(rank == 0)])

    reader = ttnn.KernelDescriptor(
        kernel_source=SHARDED_READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[
            n, workers, group, D, int(head_parallel),
            *sqct, *skct, *cict, *aqct, *kvct,
        ],
        runtime_args=rr,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=SHARDED_WRITER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[
            n, workers, group, topk, D, int(debug_indices),
            int(head_parallel),
            *oct, *kvct, *dict_,
        ],
        runtime_args=wr,
        config=ttnn.WriterConfigDescriptor(),
    )

    scale_bits = struct.unpack("<I", struct.pack("<f", D ** -0.5))[0]
    compute_ct = [
        ROWS, 2 * D // 32, D // 32, PADDED_TOPK // 32, scale_bits,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        15, 16, 18, 19, 20,
        0, 23, 24, 25, 26, 0, 1, 0,
        local_pad,
    ]
    compute = ttnn.KernelDescriptor(
        kernel_source=SHARDED_COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=cr,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        ),
    )

    tile16 = 2048
    tile8 = ttnn.tile_size(ttnn.bfloat8_b)

    def cb(index, dtype, page_size, pages):
        return ttnn.CBDescriptor(
            total_size=page_size * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    # Space for the 64-byte-aligned over-read used by odd 800-byte shards.
    candidate_page_bytes = ((local_n * 4 + 63) // 64) * 64
    cbs = [
        # Sparse-SDPA tail. Flat mode consumes it only on owners; head-parallel
        # mode consumes one page/tail on every worker.
        cb(0, ttnn.bfloat16, 2 * D * 2, ROWS),
        cb(1, ttnn.bfloat16, tile16, 2 * D // 32),
        cb(2, ttnn.fp8_e4m3, 2 * D, PADDED_TOPK),
        cb(3, ttnn.bfloat8_b, tile8, (PADDED_TOPK // 32) * (2 * D // 32)),
        cb(4, ttnn.bfloat16, tile16, 1),
        cb(5, ttnn.bfloat16, tile16, 1),
        cb(6, ttnn.bfloat16, tile16, 1),
        cb(7, ttnn.bfloat16, tile16, PADDED_TOPK // 32),
        cb(8, ttnn.bfloat16, tile16, 1),
        cb(9, ttnn.bfloat16, tile16, 1),
        cb(10, ttnn.bfloat16, tile16, 1),
        cb(11, ttnn.bfloat16, tile16, 1),
        cb(12, ttnn.bfloat16, tile16, D // 32),
        cb(13, ttnn.bfloat16, tile16, D // 32),
        cb(14, ttnn.bfloat16, tile16, 1),
        cb(15, ttnn.bfloat16, tile16, D // 32),
        cb(16, ttnn.bfloat16, tile16, D // 32),
        cb(17, ttnn.uint32, PADDED_TOPK * 4,
           1 if head_parallel else group),
        cb(18, ttnn.bfloat16, 16, 2),
        cb(19, ttnn.bfloat16, tile16, 1),
        cb(20, ttnn.bfloat16, tile16, 1),
        cb(21, ttnn.bfloat16, 32, 2),
        cb(22, ttnn.bfloat16, 16, 2),
        # Local Stage-II scan and one-round fan-in/all-to-all exchange.
        cb(23, ttnn.bfloat16, D * 2, ROWS),
        cb(24, ttnn.bfloat16, tile16, D // 32),
        cb(25, ttnn.fp8_e4m3, D, 64),
        cb(26, ttnn.bfloat8_b, tile8, 2 * (D // 32)),
        cb(27, ttnn.uint32, candidate_page_bytes, 1),
        cb(28, ttnn.bfloat16, tile16, local_pad // 32),
        cb(29, ttnn.uint32,
           topk * 2 * 4 if head_parallel else packet_bytes, workers),
        cb(30, ttnn.uint32, packet_bytes, 1),
    ]
    ready = ttnn.SemaphoreDescriptor(
        id=SEM.FRONTIERS_READY, core_ranges=cores, initial_value=0
    )
    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader, compute, writer], semaphores=[ready], cbs=cbs
    )
    io = [q, keys, indices, *q_attention, kv, *outputs, *debug_outputs]

    def run():
        ttnn.generic_op(io, descriptor)
        return outputs

    return {
        "run": run,
        "outputs": outputs,
        "debug_indices": debug_outputs if debug_indices else None,
        "descriptor": descriptor,
        "io": io,
        "owner_cores": owner_points,
        "worker_groups": groups,
        "workers_per_slot": workers,
        "local_candidates": local_n,
        "head_parallel": head_parallel,
        "noc_packet_bytes": topk * 2 * 4 if head_parallel else packet_bytes,
        "noc_packets_per_slot": (workers * (workers - 1)
                                 if head_parallel else workers - 1),
    }
