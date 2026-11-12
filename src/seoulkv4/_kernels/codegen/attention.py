"""Streaming sink/local/retrieval attention; no source-string rewriting."""

from seoulkv4._kernels.common.cb_layout import cpp_defines as cb_defines
from seoulkv4._kernels.common.refinement_abi import RefinementCB
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as sem_defines
from seoulkv4._kernels.codegen.preprocessor import profile_defines
from seoulkv4._kernels.stage16.attention_abi import cpp_defines as arg_defines
from seoulkv4._kernels.stage16.fusion_abi import AttentionSemaphore


def _header(profile):
    return (
        cb_defines(RefinementCB)
        + sem_defines(AttentionSemaphore)
        + arg_defines()
        + profile_defines(profile=profile)
        + r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"
#if SKV4_PROFILE
#include "tools/profiler/kernel_profiler.hpp"
#endif
constexpr uint32_t CHUNK_KEYS = 128;
constexpr uint32_t RETRIEVAL_KEYS = 100;
constexpr uint32_t RETRIEVAL_PACKET_BYTES = 512;
constexpr uint16_t NEG_INF = 0xff80u;
"""
    )


def reader(*, profile=False):
    return (
        _header(profile)
        + r"""
void kernel_main() {
    constexpr uint32_t D = get_compile_time_arg_val(SKV4_ATTN_R_CT_D);
    constexpr uint32_t HANDOFF_BYTES = get_compile_time_arg_val(SKV4_ATTN_R_CT_HANDOFF_BYTES);
    constexpr auto qa = TensorAccessorArgs<SKV4_ATTN_R_CT_ACCESSORS>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(SKV4_ATTN_R_RT_QUERY), D * 2);
    const auto kv = TensorAccessor(ka, get_arg_val<uint32_t>(SKV4_ATTN_R_RT_KV), D * 2);
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(SKV4_ATTN_R_RT_HANDOFF), HANDOFF_BYTES);
    const uint32_t slot = get_arg_val<uint32_t>(SKV4_ATTN_R_RT_SLOT);
    const uint32_t head = get_arg_val<uint32_t>(SKV4_ATTN_R_RT_HEAD);
    const uint32_t context = get_arg_val<uint32_t>(SKV4_ATTN_R_RT_CONTEXT);
    const uint32_t sink = get_arg_val<uint32_t>(SKV4_ATTN_R_RT_SINK);
    const uint32_t local = get_arg_val<uint32_t>(SKV4_ATTN_R_RT_LOCAL);
    const uint32_t dense = sink + local;
    const uint32_t total = dense + RETRIEVAL_KEYS;
    const uint32_t chunks = (total + CHUNK_KEYS - 1) / CHUNK_KEYS;
    Noc noc;
    experimental::CB query(SKV4_CB_ATTENTION_QUERY_ROW), keys(SKV4_CB_SPARSE_KV_RECORDS);
    experimental::CB requests(SKV4_CB_QUERY_COEFFICIENTS), acks(SKV4_CB_TOPK_MERGE_SCRATCH);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_AOV_Q_B", 0);
#endif
    query.reserve_back(32);
    const uint32_t qdst = query.get_write_ptr();
    auto zeros = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(qdst);
    for (uint32_t j = 0; j < 32 * 2 * D * 2 / 4; ++j) zeros[j] = 0;
    noc.async_read(q, query, D * 2, {.page_id = slot * 4 + head},
                   {.offset_bytes = head * (2 * D * 2) + D * 2});
    noc.async_read_barrier();
    query.push_back(32);
#if SKV4_PROFILE
    DeviceTimestampedData("SKV_AOV_Q_E", 0);
#endif
    cb_reserve_back(SKV4_CB_SHARED_SCRATCH, 1);
    const uint32_t scratch = get_write_ptr(SKV4_CB_SHARED_SCRATCH);
    auto selected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    auto indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + RETRIEVAL_PACKET_BYTES);
    cb_reserve_back(SKV4_CB_LIVE_CONTROL, 1);
    auto control = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SKV4_CB_LIVE_CONTROL));
    control[sparse_sdpa::control_message::ACTIVE_CHUNKS] = chunks;
    control[sparse_sdpa::control_message::VALID_KEYS] = total;
    cb_push_back(SKV4_CB_LIVE_CONTROL, 1);
    bool imported = false;
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const uint32_t begin = chunk * CHUNK_KEYS;
        const uint32_t valid = total - begin < CHUNK_KEYS ? total - begin : CHUNK_KEYS;
        if (!imported && begin + valid > dense) {
#if SKV4_PROFILE
            DeviceTimestampedData("SKV_AOV_READY_B", 0);
#endif
            auto ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SKV4_SEM_ATTENTION_IDS_READY));
            noc_semaphore_wait(ready, 1);
            noc_semaphore_set(ready, 0);
#if SKV4_PROFILE
            DeviceTimestampedData("SKV_AOV_READY_E", 0);
            DeviceTimestampedData("SKV_ATTN_R_IMPORT_TOP100_B", 0);
#endif
            noc_async_read(get_noc_addr(slot * 4 + head, ids), scratch, RETRIEVAL_PACKET_BYTES);
            noc_async_read_barrier();
            imported = true;
#if SKV4_PROFILE
            DeviceTimestampedData("SKV_ATTN_R_IMPORT_TOP100_E", 0);
#endif
        }
        for (uint32_t j = 0; j < valid; ++j) {
            const uint32_t pos = begin + j;
            indices[j] = pos < sink ? pos : pos < dense ? context - local + pos - sink : selected[pos - dense];
        }
        // Shared on-core index scratch stays live until both gather halves ACK.
        __sync_synchronize();
        keys.reserve_back(CHUNK_KEYS);
        const uint32_t dst = keys.get_write_ptr();
        const uint32_t split = (valid + 1) / 2;
#if SKV4_PROFILE
        DeviceTimestampedData("SKV_ATTN_R_K_GATHER_JOIN_B", chunk);
#endif
        requests.reserve_back(1);
        auto req = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(requests.get_write_ptr());
        req[SKV4_ATTN_REQ_INDEX_L1] = reinterpret_cast<uint32_t>(indices);
        req[SKV4_ATTN_REQ_SPLIT] = split;
        req[SKV4_ATTN_REQ_IS_LAST] = chunk + 1 == chunks;
        req[SKV4_ATTN_REQ_DST_L1] = dst;
        requests.push_back(1);
        sparse_sdpa::trid_ring_gather<false, 1, 1, 0, 0, sparse_sdpa::K_TRID_RING>(
            noc, kv, dst, indices, 0, split, valid, 2 * D, slot * context);
        acks.wait_front(1);
        acks.pop_front(1);
        if (valid < CHUNK_KEYS) {
            noc.async_write_zeros(keys, (CHUNK_KEYS - valid) * 2 * D,
                                   {.offset_bytes = valid * 2 * D});
            noc.write_zeros_l1_barrier();
        }
        // Only the final partial tile is masked; all previous chunks are full.
        if (chunk + 1 == chunks && valid % 32) {
            cb_reserve_back(SKV4_CB_ATTENTION_MASK, 1);
            auto mask = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_ATTENTION_MASK));
            for (uint32_t col = 0; col < 32; ++col) {
                const uint32_t offset = col < 16 ? col : 256 + col - 16;
                mask[offset] = col >= valid % 32 ? NEG_INF : uint16_t(0);
            }
            cb_push_back(SKV4_CB_ATTENTION_MASK, 1);
        }
        keys.push_back(CHUNK_KEYS);
#if SKV4_PROFILE
        DeviceTimestampedData("SKV_ATTN_R_K_GATHER_JOIN_E", chunk);
#endif
    }
}
"""
    )


def writer(*, profile=False):
    return (
        _header(profile)
        + r"""
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
void kernel_main() {
    constexpr uint32_t D = get_compile_time_arg_val(SKV4_ATTN_W_CT_D);
    constexpr auto oa = TensorAccessorArgs<SKV4_ATTN_W_CT_ACCESSORS>();
    constexpr auto ka = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(oa, get_arg_val<uint32_t>(SKV4_ATTN_W_RT_OUTPUT), D * 2);
    const auto kv = TensorAccessor(ka, get_arg_val<uint32_t>(SKV4_ATTN_W_RT_KV), D * 2);
    const uint32_t slot = get_arg_val<uint32_t>(SKV4_ATTN_W_RT_SLOT);
    const uint32_t head = get_arg_val<uint32_t>(SKV4_ATTN_W_RT_HEAD);
    const uint32_t context = get_arg_val<uint32_t>(SKV4_ATTN_W_RT_CONTEXT);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        SKV4_CB_ATTENTION_SCALE, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW, 1>();
    generate_bcast_col_scalar(CircularBuffer(SKV4_CB_ATTENTION_COLUMN_IDENTITY), 0x3f803f80u);
    cb_reserve_back(SKV4_CB_ATTENTION_NEG_INF, 1);
    auto neginf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(SKV4_CB_ATTENTION_NEG_INF));
    for (uint32_t c = 0; c < 16; ++c) {
        neginf[c] = NEG_INF;
        neginf[256 + c] = NEG_INF;
    }
    cb_push_back(SKV4_CB_ATTENTION_NEG_INF, 1);
    Noc noc;
    experimental::CB requests(SKV4_CB_QUERY_COEFFICIENTS), acks(SKV4_CB_TOPK_MERGE_SCRATCH);
    bool last = false;
    while (!last) {
        requests.wait_front(1);
        auto req = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(requests.get_read_ptr());
        const uint32_t split = req[SKV4_ATTN_REQ_SPLIT];
        const uint32_t dst = req[SKV4_ATTN_REQ_DST_L1];
        auto indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(req[SKV4_ATTN_REQ_INDEX_L1]);
        last = req[SKV4_ATTN_REQ_IS_LAST] != 0;
        requests.pop_front(1);
        sparse_sdpa::trid_ring_gather<false, 1, 1, 0, 0, sparse_sdpa::K_TRID_RING>(
            noc, kv, dst, indices, 0, 0, split, 2 * D, slot * context);
        acks.reserve_back(1);
        acks.push_back(1);
    }
    experimental::CB output(SKV4_CB_ATTENTION_OUT_ROW);
    output.wait_front(D / 32);
    for (uint32_t row = 0; row < 32; ++row) {
        noc.async_write(output, out, D * 2,
                        {.offset_bytes = row * D * 2}, {.page_id = row * 4 + head});
    }
    noc.async_write_barrier();
    output.pop_front(D / 32);
}
"""
    )
