"""Explicit source construction for standalone component witnesses and benchmarks."""

from seoulkv4._kernels.codegen.preprocessor import profile_defines


def upper_reader(*, variant):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if variant not in ("butterfly", "broadcast", "direct"):
        raise ValueError(variant)
    parts = []
    parts.append(
        r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t CASES = get_compile_time_arg_val(0);
    constexpr auto ua = TensorAccessorArgs<1>();
    constexpr auto la = TensorAccessorArgs<ua.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    const auto upper = TensorAccessor(ua, get_arg_val<uint32_t>(0), 2048);
    const auto lower = TensorAccessor(la, get_arg_val<uint32_t>(1), 64);
    const auto desc = TensorAccessor(da, get_arg_val<uint32_t>(2), 256);
    // MATH reads descriptors after UNPACK has progressed. Keep the entire
    // small control batch immutable, rather than recycling a one-page ring.
    cb_reserve_back(2, CASES);
    const uint32_t descriptors = get_write_ptr(2);
    for (uint32_t test = 0; test < CASES; ++test)
        noc_async_read(get_noc_addr(test, desc), descriptors + test * 256, 256);
    noc_async_read_barrier();
    cb_push_back(2, CASES);
    cb_reserve_back(18, CASES);
    auto clocks = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(18));
    for (uint32_t w = 0; w < CASES * 16; ++w) clocks[w] = 0;
    __sync_synchronize(); cb_push_back(18, CASES);
    for (uint32_t test = 0; test < CASES; ++test) {
        const uint32_t dp = descriptors + test * 256;
        const auto ids = reinterpret_cast<const tt_l1_ptr uint32_t*>(dp);
        cb_reserve_back(0, 2);
        const uint32_t up = get_write_ptr(0);
        // Two native blocks from persistent L1/T1, not GDDR/T2.
        for (uint32_t block = 0; block < 2; ++block)
            noc_async_read(get_noc_addr(block, upper), up + block * 2048, 2048);
        cb_reserve_back(3, 1);
        const uint32_t lp = get_write_ptr(3);
        auto pad = reinterpret_cast<tt_l1_ptr uint32_t*>(lp);
        for (uint32_t word = ids[0] * 16; word < 512; ++word) pad[word] = 0;
        // Only selected centroids fetch lower64 from T2. No payload repack.
        for (uint32_t row = 0; row < ids[0]; ++row)
            noc_async_read(get_noc_addr(ids[row + 1] & 0x0fffffffu, lower), lp + row * 64, 64);
        noc_async_read_barrier();
        __sync_synchronize();
        cb_push_back(0, 2); cb_push_back(3, 1);
    }
}
"""
    )
    return "".join(parts)


def upper_compute(*, variant):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if variant not in ("butterfly", "broadcast", "direct"):
        raise ValueError(variant)
    parts = []
    parts.append(
        r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/pack_rows_to_addr.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

inline __attribute__((always_inline)) uint32_t upper_tick() {
    asm volatile("" ::: "memory");
    uint32_t t = *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory");
    return t;
}

#ifdef TRISC_MATH
namespace ckernel::sfpu {

"""
    )
    if variant == "butterfly":
        parts.append(
            r"""inline void transpose_stage(uint32_t src, uint32_t dst, uint32_t high, uint32_t low) {
    const uint32_t h = 1u << high, l = 1u << low;
    #pragma GCC unroll 1
    for (uint32_t base = 0; base < 16; ++base) {
        if (base & (h | l)) continue;
        sfpi::vUInt a = sfpi::dst_reg[src + base].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt b = sfpi::dst_reg[src + (base | l)].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt c = sfpi::dst_reg[src + (base | h)].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt d = sfpi::dst_reg[src + (base | h | l)].mode<sfpi::DataLayout::U32>();
        sfpi::subvec_transp(a, b, c, d);
        sfpi::dst_reg[dst + base].mode<sfpi::DataLayout::U32>() = a;
        sfpi::dst_reg[dst + (base | l)].mode<sfpi::DataLayout::U32>() = b;
        sfpi::dst_reg[dst + (base | h)].mode<sfpi::DataLayout::U32>() = c;
        sfpi::dst_reg[dst + (base | h | l)].mode<sfpi::DataLayout::U32>() = d;
    }
"""
        )
    elif variant == "broadcast":
        parts.append(
            r"""inline sfpi::vUInt spread_column(sfpi::vUInt source, uint32_t column) {
    sfpi::vUInt value = 0u;
    v_if ((sfpi::vConstTileId & 15) == int32_t(2 * column)) {
        value = source;
    } v_endif;
    sfpi::vUInt shift = sfpi::subvec_shflror1(value);
    value = value | shift;
    shift = sfpi::subvec_shflror1(value);
    shift = sfpi::subvec_shflror1(shift);
    value = value | shift;
    shift = sfpi::subvec_shflror1(value);
    shift = sfpi::subvec_shflror1(shift);
    shift = sfpi::subvec_shflror1(shift);
    shift = sfpi::subvec_shflror1(shift);
    return value | shift;
}
inline void permute_block(uint32_t output_base) {
    #pragma GCC unroll 1
    for (uint32_t out = 0; out < 16; ++out) {
        const uint32_t base = (out / 8) * 2 + out % 2;
        sfpi::vUInt a = sfpi::dst_reg[base].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt b = sfpi::dst_reg[base + 4].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt c = sfpi::dst_reg[base + 8].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt d = sfpi::dst_reg[base + 12].mode<sfpi::DataLayout::U32>();
        sfpi::subvec_transp(a, b, c, d);
        sfpi::dst_reg[96].mode<sfpi::DataLayout::U32>() = a;
        sfpi::dst_reg[97].mode<sfpi::DataLayout::U32>() = b;
        sfpi::dst_reg[98].mode<sfpi::DataLayout::U32>() = c;
        sfpi::dst_reg[99].mode<sfpi::DataLayout::U32>() = d;
        const uint32_t band = (out % 8) / 2;
        #pragma GCC unroll 1
        for (uint32_t key = 0; key < 4; ++key) {
            sfpi::vUInt p0 = sfpi::dst_reg[96 + band / 2].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt p1 = sfpi::dst_reg[98 + band / 2].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt x = spread_column(p0, (band % 2) * 4 + key);
            sfpi::vUInt y = spread_column(p1, (band % 2) * 4 + key);
            v_if ((sfpi::vConstTileId & 2) != 0) { x = y; } v_endif;
            sfpi::dst_reg[128 + key].mode<sfpi::DataLayout::U32>() = x;
        }
        a = sfpi::dst_reg[128].mode<sfpi::DataLayout::U32>();
        b = sfpi::dst_reg[129].mode<sfpi::DataLayout::U32>();
        c = sfpi::dst_reg[130].mode<sfpi::DataLayout::U32>();
        d = sfpi::dst_reg[131].mode<sfpi::DataLayout::U32>();
        sfpi::subvec_transp(a, b, c, d);
        sfpi::vUInt result = 0u;
        v_if ((sfpi::vConstTileId & 12) == 0) { result = a; } v_endif;
        v_if ((sfpi::vConstTileId & 12) == 4) { result = b; } v_endif;
        v_if ((sfpi::vConstTileId & 12) == 8) { result = c; } v_endif;
        v_if ((sfpi::vConstTileId & 12) == 12) { result = d; } v_endif;
        sfpi::dst_reg[output_base + out].mode<sfpi::DataLayout::U32>() = result;
    }
"""
        )
    else:
        parts.append(
            r"""inline sfpi::vUInt direct_spread(sfpi::vUInt source, uint32_t column) {
    sfpi::vUInt value = 0u;
    v_if ((sfpi::vConstTileId & 15) == int32_t(2 * column)) {
        value = source;
    } v_endif;
    sfpi::vUInt shifted = sfpi::subvec_shflror1(value);
    value = value | shifted;
    shifted = sfpi::subvec_shflror1(value);
    shifted = sfpi::subvec_shflror1(shifted);
    value = value | shifted;
    shifted = sfpi::subvec_shflror1(value);
    shifted = sfpi::subvec_shflror1(shifted);
    shifted = sfpi::subvec_shflror1(shifted);
    shifted = sfpi::subvec_shflror1(shifted);
    return value | shifted;
"""
        )
    parts.append(
        r"""}
"""
    )
    if variant == "butterfly":
        parts.append(
            r"""
template<uint32_t DISTANCE>
inline void exchange_stage(uint32_t src, uint32_t dst, uint32_t bit) {
    const uint32_t h = 1u << bit;
    #pragma GCC unroll 1
    for (uint32_t base = 0; base < 16; ++base) {
        if (base & h) continue;
        sfpi::vUInt a = sfpi::dst_reg[src + base].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt b = sfpi::dst_reg[src + (base | h)].mode<sfpi::DataLayout::U32>();
        // Exchange the register-index bit with a lane-index bit. Route one
        // combined vector to its XOR partner, rather than broadcast each lane.
        sfpi::vUInt mixed = a;
        v_if ((sfpi::vConstTileId & int32_t(2 * DISTANCE)) == 0) {
            mixed = b;
        } v_endif;
        sfpi::vUInt shifted = mixed;
        #pragma GCC unroll 8
        for (uint32_t r = 0; r < DISTANCE; ++r)
            shifted = sfpi::subvec_shflror1(shifted);
        sfpi::vUInt crossed = shifted;
        #pragma GCC unroll 8
        for (uint32_t r = DISTANCE; r < 8 - DISTANCE; ++r)
            shifted = sfpi::subvec_shflror1(shifted);
        v_if ((sfpi::vConstTileId & int32_t(2 * DISTANCE)) == 0) {
            crossed = shifted;
        } v_endif;
        v_if ((sfpi::vConstTileId & int32_t(2 * DISTANCE)) != 0) {
            a = crossed;
        } v_endif;
        v_if ((sfpi::vConstTileId & int32_t(2 * DISTANCE)) == 0) {
            b = crossed;
        } v_endif;
        sfpi::dst_reg[dst + base].mode<sfpi::DataLayout::U32>() = a;
        sfpi::dst_reg[dst + (base | h)].mode<sfpi::DataLayout::U32>() = b;
    }
}

inline void permute_block(uint32_t output_base) {
    transpose_stage(0, 96, 3, 1);
    transpose_stage(96, 128, 3, 2);
    transpose_stage(128, 96, 1, 2);
    exchange_stage<4>(96, 128, 3);
    exchange_stage<2>(128, 96, 2);
    exchange_stage<1>(96, 128, 1);
    transpose_stage(128, 96, 2, 1);
    // Register-index bit swaps are address changes, not lane shuffles.
    #pragma GCC unroll 1
    for (uint32_t v = 0; v < 16; ++v) {
        const uint32_t flip = ((v >> 3) ^ (v >> 1)) & 1u;
        const uint32_t src = v ^ (flip << 3) ^ (flip << 1);
        sfpi::vUInt value = sfpi::dst_reg[96 + src].mode<sfpi::DataLayout::U32>();
        sfpi::dst_reg[output_base + v].mode<sfpi::DataLayout::U32>() = value;
    }
}

inline void select_rows(uint32_t descriptor_address) {
"""
        )
    elif variant == "broadcast":
        parts.append(
            r"""inline void select_rows(uint32_t descriptor_address) {
"""
        )
    else:
        parts.append(
            r"""
inline sfpi::vUInt direct_parity(uint32_t block_base, uint32_t key, uint32_t parity) {
    const uint32_t base = block_base + (key / 16) * 2 + parity;
    sfpi::vUInt a = sfpi::dst_reg[base].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt b = sfpi::dst_reg[base + 4].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt c = sfpi::dst_reg[base + 8].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt d = sfpi::dst_reg[base + 12].mode<sfpi::DataLayout::U32>();
    sfpi::subvec_transp(a, b, c, d);
    sfpi::dst_reg[96].mode<sfpi::DataLayout::U32>() = a;
    sfpi::dst_reg[97].mode<sfpi::DataLayout::U32>() = b;
    sfpi::dst_reg[98].mode<sfpi::DataLayout::U32>() = c;
    sfpi::dst_reg[99].mode<sfpi::DataLayout::U32>() = d;
    const uint32_t band = (key % 16) / 8;
    a = sfpi::dst_reg[96 + band].mode<sfpi::DataLayout::U32>();
    b = sfpi::dst_reg[98 + band].mode<sfpi::DataLayout::U32>();
    a = direct_spread(a, key % 8);
    b = direct_spread(b, key % 8);
    v_if ((sfpi::vConstTileId & 2) != 0) { a = b; } v_endif;
    b = a; c = a; d = a;
    sfpi::subvec_transp(a, b, c, d);
    sfpi::vUInt result = 0u;
    v_if ((sfpi::vConstTileId & 12) == 0) { result = a; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 4) { result = b; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 8) { result = c; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 12) { result = d; } v_endif;
    return result;
}

inline void select_native(uint32_t descriptor_address) {
"""
        )
    parts.append(
        r"""    const auto descriptor = reinterpret_cast<const uint32_t*>(descriptor_address);
    #pragma GCC unroll 1
    for (uint32_t v = 0; v < 32; ++v)
        sfpi::dst_reg[160 + v].mode<sfpi::DataLayout::U32>() = sfpi::vUInt(0u);
    #pragma GCC unroll 1
    for (uint32_t position = 0; position < descriptor[0]; ++position) {
"""
    )
    if variant == "direct":
        parts.append(
            r"""        const uint32_t id = descriptor[position + 1] & 0x0fffffffu;
"""
        )
    else:
        parts.append(
            r"""        const uint32_t id = descriptor[position + 1] & 0x0fffffffu, key = id % 32;
        const uint32_t src = (id / 32 + 1) * 32 + (key / 4) * 2;
        sfpi::vUInt a = sfpi::dst_reg[src].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt b = sfpi::dst_reg[src + 1].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt c = 0u, d = 0u;
        sfpi::subvec_transp(a, b, c, d);
        sfpi::dst_reg[96].mode<sfpi::DataLayout::U32>() = a;
        sfpi::dst_reg[97].mode<sfpi::DataLayout::U32>() = b;
        sfpi::dst_reg[98].mode<sfpi::DataLayout::U32>() = c;
        sfpi::dst_reg[99].mode<sfpi::DataLayout::U32>() = d;
        a = sfpi::dst_reg[96 + key % 4].mode<sfpi::DataLayout::U32>();
        b = a; c = a; d = a;
        sfpi::subvec_transp(a, b, c, d);
"""
        )
    parts.append(
        r"""        const uint32_t dest = 160 + (position / 16) * 8 + ((position % 16) / 4) * 2;
"""
    )
    if variant == "direct":
        parts.append(
            r"""        // Both source tiles remain native and immutable throughout selection.
        // No output or scratch range aliases either source tile.
        #pragma GCC unroll 1
        for (uint32_t parity = 0; parity < 2; ++parity) {
            sfpi::vUInt value = direct_parity((id / 32 + 1) * 32, id % 32, parity);
            sfpi::vUInt prior = sfpi::dst_reg[dest + parity].mode<sfpi::DataLayout::U32>();
            v_if ((sfpi::vConstTileId & 48) == int32_t((position % 4) * 16)) {
                prior = value;
            } v_endif;
            sfpi::dst_reg[dest + parity].mode<sfpi::DataLayout::U32>() = prior;
        }
"""
        )
    else:
        parts.append(
            r"""        sfpi::vUInt lo = sfpi::dst_reg[dest].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt hi = sfpi::dst_reg[dest + 1].mode<sfpi::DataLayout::U32>();
        v_if ((sfpi::vConstTileId & 48) == int32_t((position % 4) * 16)) {
            lo = a; hi = b;
        } v_endif;
        sfpi::dst_reg[dest].mode<sfpi::DataLayout::U32>() = lo;
        sfpi::dst_reg[dest + 1].mode<sfpi::DataLayout::U32>() = hi;
"""
        )
    parts.append(
        r"""    }
}

inline void upper_drain() {
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::WAIT_SFPU);
    tensix_sync();
}
"""
    )
    if variant == "direct":
        parts.append("")
    else:
        parts.append(
            r"""inline void permute_timed(uint32_t output_base, uint32_t clocks, uint32_t block) {
    upper_drain();  // Preceding UNPACK/SFPU completion is excluded.
    const uint32_t begin = upper_tick();
    permute_block(output_base);
    upper_drain();
    const uint32_t end = upper_tick();
    auto p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks);
    p[block] = end - begin;
}
"""
        )
    parts.append(
        r"""inline void select_timed(uint32_t descriptor, uint32_t clocks) {
    upper_drain();
    const uint32_t begin = upper_tick();
"""
    )
    if variant == "direct":
        parts.append(
            r"""    select_native(descriptor);
"""
        )
    else:
        parts.append(
            r"""    select_rows(descriptor);
"""
        )
    parts.append(
        r"""    upper_drain();
    const uint32_t end = upper_tick();
    const uint32_t empty_begin = upper_tick();
    upper_drain();
    const uint32_t empty_end = upper_tick();
    auto p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks);
    p[2] = end - begin; p[3] = empty_end - empty_begin;
    p[6] = *reinterpret_cast<const uint32_t*>(descriptor);
    p[7] = 0x13579bdfu;
    __sync_synchronize();
}

}
#endif
void kernel_main() {
    constexpr uint32_t CASES = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0, 0, 16);
    MATH(SFPU_UNARY_INIT(unused));
    copy_tile_to_dst_init_short(0);
    pack_reconfig_data_format(16);
    cb_wait_front(2, CASES); cb_wait_front(18, CASES);
    for (uint32_t test = 0; test < CASES; ++test) {
        cb_wait_front(0, 2);
        cb_reserve_back(17, 2); cb_reserve_back(16, 1);
        const uint32_t descriptor_address = get_tile_address(2, test);
        const uint32_t clocks = get_tile_address(18, test);
        tile_regs_acquire();
        MATH(const uint32_t math_begin = upper_tick());
        for (uint32_t block = 0; block < 2; ++block) {
"""
    )
    if variant == "direct":
        parts.append(
            r"""            copy_tile(0, block, block + 1);
"""
        )
    else:
        parts.append(
            r"""            copy_tile(0, block, 0);
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
                permute_timed, 0, VectorMode::None, (block + 1) * 32, clocks, block));
"""
        )
    parts.append(
        r"""        }
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
            select_timed, 0, VectorMode::None, descriptor_address, clocks));
        MATH({
            auto p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks);
            p[4] = upper_tick() - math_begin; __sync_synchronize();
        });
        tile_regs_commit(); tile_regs_wait();
        PACK({
            const uint32_t wait_begin = upper_tick();
            // tile_regs_wait queues a Tensix wait; it need not block this
            // RISC's clock read. This thread has not consumed the token,
            // so MATH_PACK stays positive until the release below.
            while (semaphore_read(semaphore::MATH_PACK) == 0) {}
            tensix_sync();
            auto p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks);
            p[8] = upper_tick() - wait_begin; __sync_synchronize();
        });
        PACK(const uint32_t pack_begin = upper_tick());
        pack_rows_to_addr_init(32);
        PACK({
            const uint32_t np = get_local_cb_interface(17).fifo_wr_ptr;
            pack_rows_to_addr(1, np);
            pack_rows_to_addr(2, np + 128);
            pack_rows_to_addr(5, get_local_cb_interface(16).fifo_wr_ptr);
        });
        pack_rows_to_addr_uninit();
        tile_regs_release();
        PACK({
            tensix_sync();
            auto p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks);
            p[5] = upper_tick() - pack_begin; __sync_synchronize();
        });
        cb_push_back(17, 2); cb_push_back(16, 1);
        cb_pop_front(0, 2);
    }
    cb_pop_front(2, CASES); cb_pop_front(18, CASES);
}
"""
    )
    return "".join(parts)


def upper_writer(*, variant):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if variant not in ("butterfly", "broadcast", "direct"):
        raise ValueError(variant)
    parts = []
    parts.append(
        r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t CASES = get_compile_time_arg_val(0);
    constexpr auto oa = TensorAccessorArgs<1>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    cb_wait_front(18, CASES);
    const uint32_t clocks = get_read_ptr(18);
    for (uint32_t test = 0; test < CASES; ++test) {
        cb_wait_front(17, 2); cb_wait_front(16, 1); cb_wait_front(3, 1);
        const uint32_t normal = get_read_ptr(17);
        noc_async_write(normal, get_noc_addr(test * 5, output), 2048);
        noc_async_write(normal + 2048, get_noc_addr(test * 5 + 1, output), 2048);
        noc_async_write(get_read_ptr(16), get_noc_addr(test * 5 + 2, output), 2048);
        noc_async_write(get_read_ptr(3), get_noc_addr(test * 5 + 3, output), 2048);
        noc_async_write(clocks + test * 64, get_noc_addr(test * 5 + 4, output), 64);
        noc_async_write_barrier();
        cb_pop_front(17, 2); cb_pop_front(16, 1); cb_pop_front(3, 1);
    }
}
"""
    )
    return "".join(parts)


def composed_reader(*, variant):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if variant not in (
        "cache",
        "repeat",
        "narrow",
        "full",
        "narrow_fused",
        "full_fused",
        "scalar",
    ):
        raise ValueError(variant)
    parts = []
    parts.append(
        r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t CASES = get_compile_time_arg_val(0);
    constexpr auto ua = TensorAccessorArgs<1>();
    constexpr auto la = TensorAccessorArgs<ua.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    constexpr auto qa = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    const auto upper = TensorAccessor(ua, get_arg_val<uint32_t>(0), 2048);
    const auto lower = TensorAccessor(la, get_arg_val<uint32_t>(1), 64);
    const auto desc = TensorAccessor(da, get_arg_val<uint32_t>(2), 1024);
    const auto query = TensorAccessor(qa, get_arg_val<uint32_t>(3), 2048);
    cb_reserve_back(11, CASES); cb_reserve_back(13, CASES);
    const uint32_t dp = get_write_ptr(11);
    auto clocks = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(13));
    for (uint32_t w = 0; w < CASES * 32; ++w) clocks[w] = 0;
    for (uint32_t test = 0; test < CASES; ++test)
        noc_async_read(get_noc_addr(test, desc), dp + test * 1024, 1024);
    cb_reserve_back(0, 8);
    for (uint32_t tile = 0; tile < 8; ++tile)
        noc_async_read(get_noc_addr(tile, query), get_write_ptr(0) + tile * 2048, 2048);
    noc_async_read_barrier(); __sync_synchronize();
    cb_push_back(11, CASES); cb_push_back(13, CASES); cb_push_back(0, 8);
    for (uint32_t test = 0; test < CASES; ++test) {
        const auto d = reinterpret_cast<const tt_l1_ptr uint32_t*>(dp + test * 1024);
        // Fixed one-chunk capacity, not a full-context alternate summary.
        cb_reserve_back(10, 32); cb_reserve_back(5, 1);
        const uint32_t np = get_write_ptr(10), lp = get_write_ptr(5);
        auto words = reinterpret_cast<tt_l1_ptr uint32_t*>(lp);
        for (uint32_t w = 0; w < 1024; ++w) words[w] = 0;
        __sync_synchronize();
        for (uint32_t slot = 0; slot < d[1]; ++slot)
            noc_async_read(get_noc_addr(d[66 + slot], upper), np + slot * 2048, 2048);
        for (uint32_t row = 0; row < d[0]; ++row)
            noc_async_read(get_noc_addr(d[2 + row] & 0x0fffffffu, lower),
                lp + (row / 16) * 2048 + (row % 16) * 64, 64);
        noc_async_read_barrier();
"""
    )
    if variant == "scalar":
        parts.append(
            r"""
        for (uint32_t row = 0; row < d[0]; ++row) {
            const uint32_t k = d[2 + row] & 31u;
            const auto src = reinterpret_cast<const tt_l1_ptr uint32_t*>(np + d[34 + row] * 2048);
            auto dst = reinterpret_cast<tt_l1_ptr uint32_t*>(lp + (row / 16) * 2048 + (row % 16) * 64 + 1024);
            const uint32_t base = (k / 16) * 64 + (k % 16) * 2;
            for (uint32_t g = 0; g < 4; ++g) {
                dst[g * 4] = src[base + g * 128];
                dst[g * 4 + 1] = src[base + g * 128 + 1];
                dst[g * 4 + 2] = src[base + g * 128 + 32];
                dst[g * 4 + 3] = src[base + g * 128 + 33];
            }
        }
        __sync_synchronize();
"""
        )
    else:
        parts.append("")
    parts.append(
        r"""        cb_push_back(10, 32); cb_push_back(5, 1);
    }
}
"""
    )
    return "".join(parts)


def composed_compute(*, variant):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    if variant not in (
        "cache",
        "repeat",
        "narrow",
        "full",
        "narrow_fused",
        "full_fused",
        "scalar",
    ):
        raise ValueError(variant)
    parts = []
    parts.append(
        r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/pack_rows_to_addr.h"
#include "api/dataflow/circular_buffer.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
inline uint32_t tick() {
    asm volatile("" ::: "memory");
    const uint32_t t = *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory"); return t;
}
inline void drain() {
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::WAIT_SFPU); tensix_sync();
}
#ifdef TRISC_MATH
namespace ckernel::sfpu {

inline sfpi::vUInt direct_spread(sfpi::vUInt source, uint32_t column) {
    sfpi::vUInt value = 0u;
    v_if ((sfpi::vConstTileId & 15) == int32_t(2 * column)) {
        value = source;
    } v_endif;
    sfpi::vUInt shifted = sfpi::subvec_shflror1(value);
    value = value | shifted;
    shifted = sfpi::subvec_shflror1(value);
    shifted = sfpi::subvec_shflror1(shifted);
    value = value | shifted;
    shifted = sfpi::subvec_shflror1(value);
    shifted = sfpi::subvec_shflror1(shifted);
    shifted = sfpi::subvec_shflror1(shifted);
    shifted = sfpi::subvec_shflror1(shifted);
    return value | shifted;
}

inline sfpi::vUInt direct_parity(uint32_t block_base, uint32_t key, uint32_t parity) {
    const uint32_t base = block_base + (key / 16) * 2 + parity;
    sfpi::vUInt a = sfpi::dst_reg[base].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt b = sfpi::dst_reg[base + 4].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt c = sfpi::dst_reg[base + 8].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt d = sfpi::dst_reg[base + 12].mode<sfpi::DataLayout::U32>();
    sfpi::subvec_transp(a, b, c, d);
"""
    )
    if variant == "narrow_fused":
        parts.append(
            r"""    sfpi::dst_reg[224].mode<sfpi::DataLayout::U32>() = a;
    sfpi::dst_reg[225].mode<sfpi::DataLayout::U32>() = b;
    sfpi::dst_reg[226].mode<sfpi::DataLayout::U32>() = c;
    sfpi::dst_reg[227].mode<sfpi::DataLayout::U32>() = d;
"""
        )
    else:
        parts.append(
            r"""    sfpi::dst_reg[192].mode<sfpi::DataLayout::U32>() = a;
    sfpi::dst_reg[193].mode<sfpi::DataLayout::U32>() = b;
    sfpi::dst_reg[194].mode<sfpi::DataLayout::U32>() = c;
    sfpi::dst_reg[195].mode<sfpi::DataLayout::U32>() = d;
"""
        )
    parts.append(
        r"""    const uint32_t band = (key % 16) / 8;
"""
    )
    if variant == "narrow_fused":
        parts.append(
            r"""    a = sfpi::dst_reg[224 + band].mode<sfpi::DataLayout::U32>();
    b = sfpi::dst_reg[226 + band].mode<sfpi::DataLayout::U32>();
"""
        )
    else:
        parts.append(
            r"""    a = sfpi::dst_reg[192 + band].mode<sfpi::DataLayout::U32>();
    b = sfpi::dst_reg[194 + band].mode<sfpi::DataLayout::U32>();
"""
        )
    parts.append(
        r"""    a = direct_spread(a, key % 8);
    b = direct_spread(b, key % 8);
    v_if ((sfpi::vConstTileId & 2) != 0) { a = b; } v_endif;
    b = a; c = a; d = a;
    sfpi::subvec_transp(a, b, c, d);
    sfpi::vUInt result = 0u;
    v_if ((sfpi::vConstTileId & 12) == 0) { result = a; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 4) { result = b; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 8) { result = c; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 12) { result = d; } v_endif;
    return result;
}


inline void clear_upper(uint32_t right) {
    #pragma GCC unroll 1
    for (uint32_t v = 0; v < 16; ++v) {
"""
    )
    if variant == "narrow_fused":
        parts.append(
            r"""        const uint32_t at = right ? (v / 8) * 16 + 8 + v % 8 : 160 + v;
"""
        )
    else:
        parts.append(
            r"""        const uint32_t at = right ? (v / 8) * 16 + 8 + v % 8 : v;
"""
        )
    parts.append(
        r"""        sfpi::dst_reg[at].mode<sfpi::DataLayout::U32>() = sfpi::vUInt(0u);
    }
}
__attribute__((noinline, noclone))
void select_block(uint32_t address, uint32_t slot, uint32_t right) {
    const auto d = reinterpret_cast<const uint32_t*>(address);
    #pragma GCC unroll 1
    for (uint32_t row = 0; row < d[0]; ++row) {
        if (d[34 + row] != slot) continue;
        const uint32_t key = d[2 + row] & 31u;
        const uint32_t dest = right
            ? (row / 16) * 16 + 8 + (row % 16) / 4 * 2
"""
    )
    if variant == "narrow_fused":
        parts.append(
            r"""            : 160 + (row / 16) * 8 + (row % 16) / 4 * 2;
"""
        )
    else:
        parts.append(
            r"""            : (row / 16) * 8 + (row % 16) / 4 * 2;
"""
        )
    parts.append(
        r"""        #pragma GCC unroll 1
        for (uint32_t parity = 0; parity < 2; ++parity) {
"""
    )
    if variant == "narrow_fused":
        parts.append(
            r"""            sfpi::vUInt value = direct_parity(192, key, parity);
"""
        )
    else:
        parts.append(
            r"""            sfpi::vUInt value = direct_parity(160, key, parity);
"""
        )
    parts.append(
        r"""            sfpi::vUInt prior = sfpi::dst_reg[dest + parity].mode<sfpi::DataLayout::U32>();
            v_if ((sfpi::vConstTileId & 48) == int32_t((row % 4) * 16)) {
                prior = value;
            } v_endif;
            sfpi::dst_reg[dest + parity].mode<sfpi::DataLayout::U32>() = prior;
        }
    }
}
inline void merge_cached_upper() {
    #pragma GCC unroll 1
    for (uint32_t v = 0; v < 16; ++v) {
        sfpi::vUInt upper = sfpi::dst_reg[160 + v].mode<sfpi::DataLayout::U32>();
        sfpi::dst_reg[(v / 8) * 16 + 8 + v % 8].mode<sfpi::DataLayout::U32>() = upper;
    }
}
inline void clear_score() {
    #pragma GCC unroll 1
    for (uint32_t v = 0; v < 32; ++v)
        sfpi::dst_reg[v].mode<sfpi::DataLayout::F32>() = sfpi::vFloat(0.0f);
}

// fine-arithmetic-codec: fp16-scale-odd-v1
template <uint32_t FIRST>
__attribute__((noinline,noclone)) void expand_four_planes(uint32_t table_address) {
    const auto scales = reinterpret_cast<const uint16_t*>(table_address);

    // Each packed pair supplies all four planes and both coordinate halves.
    // Keep only that pair live; avoid reloading Dst for every output vector.
    #pragma GCC unroll 1
    for (uint32_t source = 0; source < 16; ++source) {
        const uint32_t face = source / 8, v = source % 8;
        const sfpi::vUInt lo = sfpi::dst_reg[face * 16 + v]
            .mode<sfpi::DataLayout::U32>();
"""
    )
    if variant == "narrow" or variant == "narrow_fused":
        parts.append(
            r"""        const sfpi::vUInt hi = sfpi::dst_reg[160 + face * 8 + v]
"""
        )
    else:
        parts.append(
            r"""        const sfpi::vUInt hi = sfpi::dst_reg[face * 16 + 8 + v]
"""
        )
    parts.append(
        r"""            .mode<sfpi::DataLayout::U32>();
        #pragma GCC unroll 1
        for (uint32_t output = 0; output < 8; ++output) {
            const uint32_t plane = output / 2, half = output % 2;
            const int shift = -int(4 * (FIRST + plane) + 2 * half);
            const sfpi::vUInt code = (sfpi::shft(lo, shift) & 3u)
                | sfpi::shft(sfpi::shft(hi, shift) & 3u, 2);
            const sfpi::vInt odd = sfpi::vInt(code + code) - 15;
            const sfpi::vFloat val = sfpi::convert<sfpi::vFloat>(
                sfpi::convert<sfpi::vSMag>(odd), sfpi::RoundMode::Nearest);
            sfpi::dst_reg[(plane + 1) * 32 + face * 16 + half * 8 + v]
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
        #pragma GCC unroll 1
        for (uint32_t parity = 0; parity < 2; ++parity) {
            #pragma GCC unroll 1
            for (uint32_t part = 0; part < 2; ++part) {
                sfpi::vUInt word = uint32_t(scales[(face * 16 + parity) * 2 + part]);
                #pragma GCC unroll 1
                for (uint32_t j = 1; j < 8; ++j) {
                    v_if ((sfpi::vConstTileId & 15) >= 2 * j) {
                        word = uint32_t(scales[(face * 16 + parity + 2 * j) * 2 + part]);
                    } v_endif;
                }
                sfpi::dst_reg[5 * 32 + part].mode<sfpi::DataLayout::U32>() = word;
            }
            sfpi::vUInt lo = sfpi::dst_reg[5 * 32].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt hi = sfpi::dst_reg[5 * 32 + 1].mode<sfpi::DataLayout::U32>();
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
inline void extract(uint32_t address, uint32_t blocks, uint32_t right) {
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
        clear_upper, 0, VectorMode::None, right));
    copy_tile_to_dst_init_short_with_dt(5, 10);
    for (uint32_t slot = 0; slot < blocks; ++slot) {
"""
    )
    if variant == "narrow_fused":
        parts.append(
            r"""        copy_tile(10, slot, 6);
"""
        )
    else:
        parts.append(
            r"""        copy_tile(10, slot, 5);
"""
        )
    parts.append(
        r"""        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
            select_block, 0, VectorMode::None, address, slot, right));
    }
}
inline void pack_start(uint32_t clocks, uint32_t field) {
    PACK({
        while (semaphore_read(semaphore::MATH_PACK) == 0) {}
        tensix_sync();
        auto c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks);
        c[field] = tick();
    });
}
inline void pack_end(uint32_t clocks, uint32_t field) {
    PACK({
        tensix_sync();
        auto c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks);
        c[field] = tick(); __sync_synchronize();
    });
}
template<uint32_t FIRST, bool CACHE>
inline void decode(uint32_t descriptor, uint32_t clocks, uint32_t blocks) {
    constexpr uint32_t FIELD = FIRST == 0 ? 3 : 8;
    cb_reserve_back(1, 4); tile_regs_acquire();
    MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[FIELD] = tick(); });
"""
    )
    if variant == "narrow":
        parts.append(
            r"""        copy_tile_to_dst_init_short_with_dt(0, 5); copy_tile(5, 0, 0);
        copy_tile_to_dst_init_short_with_dt(5, 12); copy_tile(12, 0, 5);
"""
        )
    elif variant == "full":
        parts.append(
            r"""        copy_tile_to_dst_init_short_with_dt(0, 12); copy_tile(12, 0, 0);
"""
        )
    elif variant == "narrow_fused":
        parts.append(
            r"""    if constexpr(FIRST == 0) {
        copy_tile_to_dst_init_short_with_dt(0, 5); copy_tile(5, 0, 0);
        extract(descriptor, blocks, 0);
    } else {
        copy_tile_to_dst_init_short_with_dt(0, 5); copy_tile(5, 0, 0);
        copy_tile_to_dst_init_short_with_dt(5, 12); copy_tile(12, 0, 5);
    }
"""
        )
    elif variant == "full_fused":
        parts.append(
            r"""    if constexpr(FIRST == 0) {
        copy_tile_to_dst_init_short_with_dt(0, 5); copy_tile(5, 0, 0);
        extract(descriptor, blocks, 1);
    } else {
        copy_tile_to_dst_init_short_with_dt(0, 12); copy_tile(12, 0, 0);
    }
"""
        )
    elif variant == "scalar":
        parts.append(
            r"""    copy_tile_to_dst_init_short_with_dt(0, 5); copy_tile(5, 0, 0);
    if constexpr(CACHE) {
        copy_tile_to_dst_init_short_with_dt(5, 12); copy_tile(12, 0, 5);
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
            merge_cached_upper, 0, VectorMode::None));
    } else {
        // Reader assembled both faces.
    }
"""
        )
    else:
        parts.append(
            r"""    copy_tile_to_dst_init_short_with_dt(0, 5); copy_tile(5, 0, 0);
    if constexpr(CACHE) {
        copy_tile_to_dst_init_short_with_dt(5, 12); copy_tile(12, 0, 5);
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
            merge_cached_upper, 0, VectorMode::None));
    } else {
        extract(descriptor, blocks, 1);
    }
"""
        )
    parts.append(
        r"""    MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[FIELD + 1] = tick(); });
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        expand_four_planes, (FIRST), 0, VectorMode::None, descriptor + 512));
    MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[FIELD + 2] = tick(); });
    tile_regs_commit(); tile_regs_wait();
    pack_start(clocks, FIELD + 3);
"""
    )
    if variant == "narrow_fused":
        parts.append(
            r"""    if constexpr(FIRST == 0) {
    pack_reconfig_data_format(12); pack_init(12);
    pack_rows_to_addr_init(32);
    PACK(pack_rows_to_addr(5, get_local_cb_interface(12).fifo_wr_ptr));
    pack_rows_to_addr_uninit();
    }
"""
        )
    elif variant == "full_fused":
        parts.append(
            r"""    if constexpr(FIRST == 0) {
    pack_reconfig_data_format(12); pack_init(12);
    pack_rows_to_addr_init(64);
    PACK(pack_rows_to_addr(0, get_local_cb_interface(12).fifo_wr_ptr));
    pack_rows_to_addr_uninit();
    }
"""
        )
    else:
        parts.append("")
    parts.append(
        r"""    pack_reconfig_data_format(1); pack_init(1);
    pack_tile(1, 1); pack_tile(2, 1); pack_tile(3, 1); pack_tile(4, 1);
    tile_regs_release(); pack_end(clocks, FIELD + 4); cb_push_back(1, 4);
"""
    )
    if variant == "narrow_fused" or variant == "full_fused":
        parts.append(
            r"""    if constexpr(FIRST == 0) { cb_push_back(12, 1); cb_wait_front(12, 1); }
"""
        )
    else:
        parts.append("")
    parts.append(
        r"""}
void kernel_main() {
    constexpr uint32_t CASES = get_compile_time_arg_val(0);
    constexpr bool CACHE = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(5, 0, 1);
    MATH(SFPU_UNARY_INIT(unused));
    cb_wait_front(0, 8); cb_wait_front(11, CASES); cb_wait_front(13, CASES);
    for (uint32_t test = 0; test < CASES; ++test) {
        cb_wait_front(5, 1); cb_wait_front(10, 32);
        const uint32_t descriptor = get_tile_address(11, test);
        const uint32_t clocks = get_tile_address(13, test);
        const uint32_t blocks = reinterpret_cast<const uint32_t*>(descriptor)[1];
        MATH({ drain(); auto c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks);
            c[16] = tick(); c[18] = 0x13579bdfu; });
"""
    )
    if variant == "narrow":
        parts.append(
            r"""        cb_reserve_back(12, 1);
        tile_regs_acquire();
        MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[0] = tick(); });

        extract(descriptor, blocks, 0);
        MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[1] = tick(); });
        tile_regs_commit(); tile_regs_wait(); pack_start(clocks, 19);
    pack_reconfig_data_format(12); pack_init(12);
    pack_rows_to_addr_init(32);
    PACK(pack_rows_to_addr(0, get_local_cb_interface(12).fifo_wr_ptr));
    pack_rows_to_addr_uninit();

        tile_regs_release(); pack_end(clocks, 2);
        cb_push_back(12, 1); cb_wait_front(12, 1);
"""
        )
    elif variant == "full":
        parts.append(
            r"""        cb_reserve_back(12, 1);
        tile_regs_acquire();
        MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[0] = tick(); });
        copy_tile_to_dst_init_short_with_dt(0, 5); copy_tile(5, 0, 0);
        extract(descriptor, blocks, 1);
        MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[1] = tick(); });
        tile_regs_commit(); tile_regs_wait(); pack_start(clocks, 19);
    pack_reconfig_data_format(12); pack_init(12);
    pack_rows_to_addr_init(64);
    PACK(pack_rows_to_addr(0, get_local_cb_interface(12).fifo_wr_ptr));
    pack_rows_to_addr_uninit();

        tile_regs_release(); pack_end(clocks, 2);
        cb_push_back(12, 1); cb_wait_front(12, 1);
"""
        )
    elif variant == "narrow_fused" or variant == "full_fused":
        parts.append(
            r"""        cb_reserve_back(12, 1);
"""
        )
    else:
        parts.append(
            r"""        if constexpr(CACHE) {
            cb_reserve_back(12, 1); tile_regs_acquire();
            MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[0] = tick(); });
            extract(descriptor, blocks, 0);
            MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[1] = tick(); });
            tile_regs_commit(); tile_regs_wait(); pack_start(clocks, 19);
            pack_reconfig_data_format(12); pack_init(12);
            pack_rows_to_addr_init(32);
            PACK(pack_rows_to_addr(0, get_local_cb_interface(12).fifo_wr_ptr));
            pack_rows_to_addr_uninit(); tile_regs_release(); pack_end(clocks, 2);
            cb_push_back(12, 1); cb_wait_front(12, 1);
        }
"""
        )
    parts.append(
        r"""        decode<0, CACHE>(descriptor, clocks, blocks);
        decode<4, CACHE>(descriptor, clocks, blocks);
        if constexpr(CACHE) cb_pop_front(12, 1);
        cb_pop_front(5, 1); cb_pop_front(10, 32);
        cb_wait_front(1, 8); cb_reserve_back(16, 1);
        reconfig_data_format<SrcOrder::Reverse>(0, 1); matmul_init(0, 1, 1);
        tile_regs_acquire();
        MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[13] = tick(); });
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
            clear_score, 0, VectorMode::None));
        reconfig_data_format<SrcOrder::Reverse>(0, 1);
        for (uint32_t kt = 0; kt < 8; ++kt) matmul_tiles(0, 1, kt, kt, 0);
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
            fine_post_scale, 0, VectorMode::None, descriptor + 512));
        MATH({ drain(); reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clocks)[14] = tick(); });
        tile_regs_commit(); tile_regs_wait(); pack_start(clocks, 20);
        pack_reconfig_data_format(16); pack_init(16); pack_tile(0, 16);
        tile_regs_release(); pack_end(clocks, 15); cb_push_back(16, 1);
        // Writer owns the decoded CB until its diagnostic DMA completes.
        // MATH must not recycle it after merely issuing the matmul UNPACKs.
    }
    cb_pop_front(0, 8); cb_pop_front(11, CASES); cb_pop_front(13, CASES);
}
"""
    )
    return "".join(parts)


def cached_helper(*, narrow, fused):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    parts = []
    parts.append(
        r"""
inline sfpi::vUInt direct_spread(sfpi::vUInt source, uint32_t column) {
    sfpi::vUInt value = 0u;
    v_if ((sfpi::vConstTileId & 15) == int32_t(2 * column)) {
        value = source;
    } v_endif;
    sfpi::vUInt shifted = sfpi::subvec_shflror1(value);
    value = value | shifted;
    shifted = sfpi::subvec_shflror1(value);
    shifted = sfpi::subvec_shflror1(shifted);
    value = value | shifted;
    shifted = sfpi::subvec_shflror1(value);
    shifted = sfpi::subvec_shflror1(shifted);
    shifted = sfpi::subvec_shflror1(shifted);
    shifted = sfpi::subvec_shflror1(shifted);
    return value | shifted;
}

inline sfpi::vUInt direct_parity(uint32_t block_base, uint32_t key, uint32_t parity) {
    const uint32_t base = block_base + (key / 16) * 2 + parity;
    sfpi::vUInt a = sfpi::dst_reg[base].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt b = sfpi::dst_reg[base + 4].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt c = sfpi::dst_reg[base + 8].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt d = sfpi::dst_reg[base + 12].mode<sfpi::DataLayout::U32>();
    sfpi::subvec_transp(a, b, c, d);
"""
    )
    if narrow and fused:
        parts.append(
            r"""    sfpi::dst_reg[224].mode<sfpi::DataLayout::U32>() = a;
    sfpi::dst_reg[225].mode<sfpi::DataLayout::U32>() = b;
    sfpi::dst_reg[226].mode<sfpi::DataLayout::U32>() = c;
    sfpi::dst_reg[227].mode<sfpi::DataLayout::U32>() = d;
"""
        )
    else:
        parts.append(
            r"""    sfpi::dst_reg[192].mode<sfpi::DataLayout::U32>() = a;
    sfpi::dst_reg[193].mode<sfpi::DataLayout::U32>() = b;
    sfpi::dst_reg[194].mode<sfpi::DataLayout::U32>() = c;
    sfpi::dst_reg[195].mode<sfpi::DataLayout::U32>() = d;
"""
        )
    parts.append(
        r"""    const uint32_t band = (key % 16) / 8;
"""
    )
    if narrow and fused:
        parts.append(
            r"""    a = sfpi::dst_reg[224 + band].mode<sfpi::DataLayout::U32>();
    b = sfpi::dst_reg[226 + band].mode<sfpi::DataLayout::U32>();
"""
        )
    else:
        parts.append(
            r"""    a = sfpi::dst_reg[192 + band].mode<sfpi::DataLayout::U32>();
    b = sfpi::dst_reg[194 + band].mode<sfpi::DataLayout::U32>();
"""
        )
    parts.append(
        r"""    a = direct_spread(a, key % 8);
    b = direct_spread(b, key % 8);
    v_if ((sfpi::vConstTileId & 2) != 0) { a = b; } v_endif;
    b = a; c = a; d = a;
    sfpi::subvec_transp(a, b, c, d);
    sfpi::vUInt result = 0u;
    v_if ((sfpi::vConstTileId & 12) == 0) { result = a; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 4) { result = b; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 8) { result = c; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 12) { result = d; } v_endif;
    return result;
}


inline void clear_upper(uint32_t right) {
    #pragma GCC unroll 1
    for (uint32_t v = 0; v < 16; ++v) {
"""
    )
    if narrow and fused:
        parts.append(
            r"""        const uint32_t at = right ? (v / 8) * 16 + 8 + v % 8 : 160 + v;
"""
        )
    else:
        parts.append(
            r"""        const uint32_t at = right ? (v / 8) * 16 + 8 + v % 8 : v;
"""
        )
    parts.append(
        r"""        sfpi::dst_reg[at].mode<sfpi::DataLayout::U32>() = sfpi::vUInt(0u);
    }
}
__attribute__((noinline, noclone))
void select_block(uint32_t address, uint32_t slot, uint32_t right) {
    const auto d = reinterpret_cast<const uint32_t*>(address);
    #pragma GCC unroll 1
    for (uint32_t row = 0; row < d[0]; ++row) {
        if (d[34 + row] != slot) continue;
        const uint32_t key = d[2 + row] & 31u;
        const uint32_t dest = right
            ? (row / 16) * 16 + 8 + (row % 16) / 4 * 2
"""
    )
    if narrow and fused:
        parts.append(
            r"""            : 160 + (row / 16) * 8 + (row % 16) / 4 * 2;
"""
        )
    else:
        parts.append(
            r"""            : (row / 16) * 8 + (row % 16) / 4 * 2;
"""
        )
    parts.append(
        r"""        #pragma GCC unroll 1
        for (uint32_t parity = 0; parity < 2; ++parity) {
"""
    )
    if narrow and fused:
        parts.append(
            r"""            sfpi::vUInt value = direct_parity(192, key, parity);
"""
        )
    else:
        parts.append(
            r"""            sfpi::vUInt value = direct_parity(160, key, parity);
"""
        )
    parts.append(
        r"""            sfpi::vUInt prior = sfpi::dst_reg[dest + parity].mode<sfpi::DataLayout::U32>();
            v_if ((sfpi::vConstTileId & 48) == int32_t((row % 4) * 16)) {
                prior = value;
            } v_endif;
            sfpi::dst_reg[dest + parity].mode<sfpi::DataLayout::U32>() = prior;
        }
    }
}
inline void merge_cached_upper() {
    #pragma GCC unroll 1
    for (uint32_t v = 0; v < 16; ++v) {
        sfpi::vUInt upper = sfpi::dst_reg[160 + v].mode<sfpi::DataLayout::U32>();
        sfpi::dst_reg[(v / 8) * 16 + 8 + v % 8].mode<sfpi::DataLayout::U32>() = upper;
    }
}
inline void clear_score() {
    #pragma GCC unroll 1
    for (uint32_t v = 0; v < 32; ++v)
        sfpi::dst_reg[v].mode<sfpi::DataLayout::F32>() = sfpi::vFloat(0.0f);
}

// fine-arithmetic-codec: fp16-scale-odd-v1
template <uint32_t FIRST>
__attribute__((noinline,noclone)) void expand_four_planes(uint32_t table_address) {
    const auto scales = reinterpret_cast<const uint16_t*>(table_address);

    // Each packed pair supplies all four planes and both coordinate halves.
    // Keep only that pair live; avoid reloading Dst for every output vector.
    #pragma GCC unroll 1
    for (uint32_t source = 0; source < 16; ++source) {
        const uint32_t face = source / 8, v = source % 8;
        const sfpi::vUInt lo = sfpi::dst_reg[face * 16 + v]
            .mode<sfpi::DataLayout::U32>();
"""
    )
    if not narrow:
        parts.append(
            r"""        const sfpi::vUInt hi = sfpi::dst_reg[face * 16 + 8 + v]
"""
        )
    else:
        parts.append(
            r"""        const sfpi::vUInt hi = sfpi::dst_reg[160 + face * 8 + v]
"""
        )
    parts.append(
        r"""            .mode<sfpi::DataLayout::U32>();
        #pragma GCC unroll 1
        for (uint32_t output = 0; output < 8; ++output) {
            const uint32_t plane = output / 2, half = output % 2;
            const int shift = -int(4 * (FIRST + plane) + 2 * half);
            const sfpi::vUInt code = (sfpi::shft(lo, shift) & 3u)
                | sfpi::shft(sfpi::shft(hi, shift) & 3u, 2);
            const sfpi::vInt odd = sfpi::vInt(code + code) - 15;
            const sfpi::vFloat val = sfpi::convert<sfpi::vFloat>(
                sfpi::convert<sfpi::vSMag>(odd), sfpi::RoundMode::Nearest);
            sfpi::dst_reg[(plane + 1) * 32 + face * 16 + half * 8 + v]
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
        #pragma GCC unroll 1
        for (uint32_t parity = 0; parity < 2; ++parity) {
            #pragma GCC unroll 1
            for (uint32_t part = 0; part < 2; ++part) {
                sfpi::vUInt word = uint32_t(scales[(face * 16 + parity) * 2 + part]);
                #pragma GCC unroll 1
                for (uint32_t j = 1; j < 8; ++j) {
                    v_if ((sfpi::vConstTileId & 15) >= 2 * j) {
                        word = uint32_t(scales[(face * 16 + parity + 2 * j) * 2 + part]);
                    } v_endif;
                }
                sfpi::dst_reg[5 * 32 + part].mode<sfpi::DataLayout::U32>() = word;
            }
            sfpi::vUInt lo = sfpi::dst_reg[5 * 32].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt hi = sfpi::dst_reg[5 * 32 + 1].mode<sfpi::DataLayout::U32>();
            sfpi::vFloat scale = sfpi::as<sfpi::vFloat>(sfpi::shft(hi, 16) | lo);
            // One face occupies eight SFPU Dst vectors. Offsets0/1 are the
            // even/odd columns of rows0..3; the right face is therefore8/9,
            // not16/17 (which belongs to the bottom-left face).
            sfpi::vFloat value = sfpi::dst_reg[face * 8 + parity];
            sfpi::dst_reg[face * 8 + parity] = value * scale;
        }
    }
}
"""
    )
    return "".join(parts)


def upper_input_compute(*, direct):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    parts = []
    parts.append(
        r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/pack_rows_to_addr.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#ifdef TRISC_MATH
namespace ckernel::sfpu {

"""
    )
    if not direct:
        parts.append(
            r"""inline void transpose_stage(uint32_t src, uint32_t dst, uint32_t high, uint32_t low) {
    const uint32_t h = 1u << high, l = 1u << low;
    #pragma GCC unroll 1
    for (uint32_t base = 0; base < 16; ++base) {
        if (base & (h | l)) continue;
        sfpi::vUInt a = sfpi::dst_reg[src + base].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt b = sfpi::dst_reg[src + (base | l)].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt c = sfpi::dst_reg[src + (base | h)].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt d = sfpi::dst_reg[src + (base | h | l)].mode<sfpi::DataLayout::U32>();
        sfpi::subvec_transp(a, b, c, d);
        sfpi::dst_reg[dst + base].mode<sfpi::DataLayout::U32>() = a;
        sfpi::dst_reg[dst + (base | l)].mode<sfpi::DataLayout::U32>() = b;
        sfpi::dst_reg[dst + (base | h)].mode<sfpi::DataLayout::U32>() = c;
        sfpi::dst_reg[dst + (base | h | l)].mode<sfpi::DataLayout::U32>() = d;
    }
"""
        )
    else:
        parts.append(
            r"""inline sfpi::vUInt direct_spread(sfpi::vUInt source, uint32_t column) {
    sfpi::vUInt value = 0u;
    v_if ((sfpi::vConstTileId & 15) == int32_t(2 * column)) {
        value = source;
    } v_endif;
    sfpi::vUInt shifted = sfpi::subvec_shflror1(value);
    value = value | shifted;
    shifted = sfpi::subvec_shflror1(value);
    shifted = sfpi::subvec_shflror1(shifted);
    value = value | shifted;
    shifted = sfpi::subvec_shflror1(value);
    shifted = sfpi::subvec_shflror1(shifted);
    shifted = sfpi::subvec_shflror1(shifted);
    shifted = sfpi::subvec_shflror1(shifted);
    return value | shifted;
"""
        )
    parts.append(
        r"""}

"""
    )
    if not direct:
        parts.append(
            r"""template<uint32_t DISTANCE>
inline void exchange_stage(uint32_t src, uint32_t dst, uint32_t bit) {
    const uint32_t h = 1u << bit;
    #pragma GCC unroll 1
    for (uint32_t base = 0; base < 16; ++base) {
        if (base & h) continue;
        sfpi::vUInt a = sfpi::dst_reg[src + base].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt b = sfpi::dst_reg[src + (base | h)].mode<sfpi::DataLayout::U32>();
        // Exchange the register-index bit with a lane-index bit. Route one
        // combined vector to its XOR partner, rather than broadcast each lane.
        sfpi::vUInt mixed = a;
        v_if ((sfpi::vConstTileId & int32_t(2 * DISTANCE)) == 0) {
            mixed = b;
        } v_endif;
        sfpi::vUInt shifted = mixed;
        #pragma GCC unroll 8
        for (uint32_t r = 0; r < DISTANCE; ++r)
            shifted = sfpi::subvec_shflror1(shifted);
        sfpi::vUInt crossed = shifted;
        #pragma GCC unroll 8
        for (uint32_t r = DISTANCE; r < 8 - DISTANCE; ++r)
            shifted = sfpi::subvec_shflror1(shifted);
        v_if ((sfpi::vConstTileId & int32_t(2 * DISTANCE)) == 0) {
            crossed = shifted;
        } v_endif;
        v_if ((sfpi::vConstTileId & int32_t(2 * DISTANCE)) != 0) {
            a = crossed;
        } v_endif;
        v_if ((sfpi::vConstTileId & int32_t(2 * DISTANCE)) == 0) {
            b = crossed;
        } v_endif;
        sfpi::dst_reg[dst + base].mode<sfpi::DataLayout::U32>() = a;
        sfpi::dst_reg[dst + (base | h)].mode<sfpi::DataLayout::U32>() = b;
    }
"""
        )
    else:
        parts.append(
            r"""inline sfpi::vUInt direct_parity(uint32_t block_base, uint32_t key, uint32_t parity) {
    const uint32_t base = block_base + (key / 16) * 2 + parity;
    sfpi::vUInt a = sfpi::dst_reg[base].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt b = sfpi::dst_reg[base + 4].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt c = sfpi::dst_reg[base + 8].mode<sfpi::DataLayout::U32>();
    sfpi::vUInt d = sfpi::dst_reg[base + 12].mode<sfpi::DataLayout::U32>();
    sfpi::subvec_transp(a, b, c, d);
    sfpi::dst_reg[96].mode<sfpi::DataLayout::U32>() = a;
    sfpi::dst_reg[97].mode<sfpi::DataLayout::U32>() = b;
    sfpi::dst_reg[98].mode<sfpi::DataLayout::U32>() = c;
    sfpi::dst_reg[99].mode<sfpi::DataLayout::U32>() = d;
    const uint32_t band = (key % 16) / 8;
    a = sfpi::dst_reg[96 + band].mode<sfpi::DataLayout::U32>();
    b = sfpi::dst_reg[98 + band].mode<sfpi::DataLayout::U32>();
    a = direct_spread(a, key % 8);
    b = direct_spread(b, key % 8);
    v_if ((sfpi::vConstTileId & 2) != 0) { a = b; } v_endif;
    b = a; c = a; d = a;
    sfpi::subvec_transp(a, b, c, d);
    sfpi::vUInt result = 0u;
    v_if ((sfpi::vConstTileId & 12) == 0) { result = a; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 4) { result = b; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 8) { result = c; } v_endif;
    v_if ((sfpi::vConstTileId & 12) == 12) { result = d; } v_endif;
    return result;
"""
        )
    parts.append(
        r"""}

"""
    )
    if not direct:
        parts.append(
            r"""inline void permute_block(uint32_t output_base) {
    transpose_stage(0, 96, 3, 1);
    transpose_stage(96, 128, 3, 2);
    transpose_stage(128, 96, 1, 2);
    exchange_stage<4>(96, 128, 3);
    exchange_stage<2>(128, 96, 2);
    exchange_stage<1>(96, 128, 1);
    transpose_stage(128, 96, 2, 1);
    // Register-index bit swaps are address changes, not lane shuffles.
    #pragma GCC unroll 1
    for (uint32_t v = 0; v < 16; ++v) {
        const uint32_t flip = ((v >> 3) ^ (v >> 1)) & 1u;
        const uint32_t src = v ^ (flip << 3) ^ (flip << 1);
        sfpi::vUInt value = sfpi::dst_reg[96 + src].mode<sfpi::DataLayout::U32>();
        sfpi::dst_reg[output_base + v].mode<sfpi::DataLayout::U32>() = value;
    }
}

inline void select_rows(uint32_t descriptor_address) {
"""
        )
    else:
        parts.append(
            r"""inline void select_native(uint32_t descriptor_address) {
"""
        )
    parts.append(
        r"""    const auto descriptor = reinterpret_cast<const uint32_t*>(descriptor_address);
    #pragma GCC unroll 1
    for (uint32_t v = 0; v < 32; ++v)
        sfpi::dst_reg[160 + v].mode<sfpi::DataLayout::U32>() = sfpi::vUInt(0u);
    #pragma GCC unroll 1
    for (uint32_t position = 0; position < descriptor[0]; ++position) {
"""
    )
    if not direct:
        parts.append(
            r"""        const uint32_t id = descriptor[position + 1] & 0x0fffffffu, key = id % 32;
        const uint32_t src = (id / 32 + 1) * 32 + (key / 4) * 2;
        sfpi::vUInt a = sfpi::dst_reg[src].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt b = sfpi::dst_reg[src + 1].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt c = 0u, d = 0u;
        sfpi::subvec_transp(a, b, c, d);
        sfpi::dst_reg[96].mode<sfpi::DataLayout::U32>() = a;
        sfpi::dst_reg[97].mode<sfpi::DataLayout::U32>() = b;
        sfpi::dst_reg[98].mode<sfpi::DataLayout::U32>() = c;
        sfpi::dst_reg[99].mode<sfpi::DataLayout::U32>() = d;
        a = sfpi::dst_reg[96 + key % 4].mode<sfpi::DataLayout::U32>();
        b = a; c = a; d = a;
        sfpi::subvec_transp(a, b, c, d);
"""
        )
    else:
        parts.append(
            r"""        const uint32_t id = descriptor[position + 1] & 0x0fffffffu;
"""
        )
    parts.append(
        r"""        const uint32_t dest = 160 + (position / 16) * 8 + ((position % 16) / 4) * 2;
"""
    )
    if not direct:
        parts.append(
            r"""        sfpi::vUInt lo = sfpi::dst_reg[dest].mode<sfpi::DataLayout::U32>();
        sfpi::vUInt hi = sfpi::dst_reg[dest + 1].mode<sfpi::DataLayout::U32>();
        v_if ((sfpi::vConstTileId & 48) == int32_t((position % 4) * 16)) {
            lo = a; hi = b;
        } v_endif;
        sfpi::dst_reg[dest].mode<sfpi::DataLayout::U32>() = lo;
        sfpi::dst_reg[dest + 1].mode<sfpi::DataLayout::U32>() = hi;
"""
        )
    else:
        parts.append(
            r"""        // Both source tiles remain native and immutable throughout selection.
        // No output or scratch range aliases either source tile.
        #pragma GCC unroll 1
        for (uint32_t parity = 0; parity < 2; ++parity) {
            sfpi::vUInt value = direct_parity((id / 32 + 1) * 32, id % 32, parity);
            sfpi::vUInt prior = sfpi::dst_reg[dest + parity].mode<sfpi::DataLayout::U32>();
            v_if ((sfpi::vConstTileId & 48) == int32_t((position % 4) * 16)) {
                prior = value;
            } v_endif;
            sfpi::dst_reg[dest + parity].mode<sfpi::DataLayout::U32>() = prior;
        }
"""
        )
    parts.append(
        r"""    }
}

}
#endif
void kernel_main() {
    constexpr uint32_t CASES = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0, 0, 16);
    MATH(SFPU_UNARY_INIT(unused));
    copy_tile_to_dst_init_short(0);
    pack_reconfig_data_format(16);
    cb_wait_front(2, CASES);
    for (uint32_t test = 0; test < CASES; ++test) {
        cb_wait_front(0, 2);
        cb_reserve_back(17, 2); cb_reserve_back(16, 1);
        const uint32_t descriptor_address = get_tile_address(2, test);
        tile_regs_acquire();
        for (uint32_t block = 0; block < 2; ++block) {
"""
    )
    if not direct:
        parts.append(
            r"""            copy_tile(0, block, 0);
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
                permute_block, 0, VectorMode::None, (block + 1) * 32));
"""
        )
    else:
        parts.append(
            r"""            copy_tile(0, block, block + 1);
"""
        )
    parts.append(
        r"""        }
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
"""
    )
    if not direct:
        parts.append(
            r"""            select_rows, 0, VectorMode::None, descriptor_address));
"""
        )
    else:
        parts.append(
            r"""            select_native, 0, VectorMode::None, descriptor_address));
"""
        )
    parts.append(
        r"""        tile_regs_commit(); tile_regs_wait();
        pack_rows_to_addr_init(32);
        PACK({
            const uint32_t np = get_local_cb_interface(17).fifo_wr_ptr;
            pack_rows_to_addr(1, np);
            pack_rows_to_addr(2, np + 128);
            pack_rows_to_addr(5, get_local_cb_interface(16).fifo_wr_ptr);
        });
        pack_rows_to_addr_uninit();
        tile_regs_release();
        cb_push_back(17, 2); cb_push_back(16, 1);
        cb_pop_front(0, 2);
    }
    cb_pop_front(2, CASES);
}
"""
    )
    return "".join(parts)


def narrow_tile_compute(*, transpose, within_face):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    parts = []
    parts.append(
        r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void clear_all() {
    #pragma GCC unroll 32
    for(uint32_t i=0;i<32;++i)
        sfpi::dst_reg[i].mode<sfpi::DataLayout::U32>()=sfpi::vUInt(0u);
}
inline void stamp_rows() {
    #pragma GCC unroll 32
    for(uint32_t i=0;i<32;++i)
        sfpi::dst_reg[i].mode<sfpi::DataLayout::U32>()=sfpi::vUInt(i+1);
}
}
#endif

void kernel_main() {
    compute_kernel_hw_startup(0,0,16);
    MATH(SFPU_UNARY_INIT(unused));
    cb_wait_front(0,1);
"""
    )
    if not transpose and (not within_face):
        parts.append(
            r"""    copy_tile_to_dst_init_short(0,0,false);
"""
        )
    elif not transpose and within_face:
        parts.append(
            r"""    copy_tile_to_dst_init_short(0,0,true);
"""
        )
    elif transpose and (not within_face):
        parts.append(
            r"""    copy_tile_to_dst_init_short(0,1,false);
"""
        )
    else:
        parts.append(
            r"""    copy_tile_to_dst_init_short(0,1,true);
"""
        )
    parts.append(
        r"""    pack_reconfig_data_format(16);

    // First output: prove where the 32x16 narrow tile lands in a full Dst.
    cb_reserve_back(16,1);
    tile_regs_acquire();
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        clear_all,0,VectorMode::None));
    copy_tile(0,0,0);
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,16);
    tile_regs_release(); cb_push_back(16,1);

    // Second output: expose the sfpi::dst_reg[i] -> packed tile block map.
    cb_reserve_back(17,1);
    tile_regs_acquire();
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        stamp_rows,0,VectorMode::None));
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,17);
    tile_regs_release(); cb_push_back(17,1);
    cb_pop_front(0,1);
}
"""
    )
    return "".join(parts)


def packed_reader(*, full):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    parts = []
    parts.append(
        r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TILES = get_compile_time_arg_val(0);
    constexpr auto ia = TensorAccessorArgs<1>();
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t ordinal = get_arg_val<uint32_t>(1);
    const auto in = TensorAccessor(ia, in_addr, 4096);
"""
    )
    if not full:
        parts.append("")
    else:
        parts.append(
            r"""    cb_reserve_back(1, 1);
    volatile tt_l1_ptr uint32_t* one =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
    for (uint32_t i = 0; i < 512; ++i) one[i] = 0x3f803f80u;
    cb_push_back(1, 1);
"""
        )
    parts.append(
        r"""    for (uint32_t t = 0; t < TILES; ++t) {
        cb_reserve_back(0, 1);
        const uint32_t dst = get_write_ptr(0);
        noc_async_read(get_noc_addr(ordinal * TILES + t, in), dst, 4096);
        noc_async_read_barrier();
        cb_push_back(0, 1);
    }
}
"""
    )
    return "".join(parts)


def packed_writer(*, full):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    parts = []
    parts.append(
        r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TILES = get_compile_time_arg_val(0);
    constexpr auto oa = TensorAccessorArgs<1>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t ordinal = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(oa, out_addr, 2048);
    for (uint32_t t = 0; t < TILES; ++t) {
"""
    )
    if not full:
        parts.append(
            r"""        cb_wait_front(16, 1);
"""
        )
    else:
        parts.append(
            r"""        cb_wait_front(17, 1);
"""
        )
    parts.append(
        r"""        if (t == 0) {
"""
    )
    if not full:
        parts.append(
            r"""            noc_async_write(get_read_ptr(16), get_noc_addr(ordinal, out), 2048);
"""
        )
    else:
        parts.append(
            r"""            noc_async_write(get_read_ptr(17), get_noc_addr(ordinal, out), 2048);
"""
        )
    parts.append(
        r"""            noc_async_write_barrier();
        }
"""
    )
    if not full:
        parts.append(
            r"""        cb_pop_front(16, 1);
"""
        )
    else:
        parts.append(
            r"""        cb_pop_front(17, 1);
"""
        )
    parts.append(
        r"""    }
}
"""
    )
    return "".join(parts)


def payload_compute(*, profile):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    parts = []
    parts.append(
        r"""
#if SKV4_PROFILE
#include "tools/profiler/kernel_profiler.hpp"

#endif

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    static_assert(K == 2048, "direct payload merge currently requires K=2048");
    constexpr uint32_t TILES = 2;
    constexpr uint32_t SURV_VALUES = 0;
    constexpr uint32_t SURV_IDS = 2;
    constexpr uint32_t IN_VALUES = 4;
    constexpr uint32_t IN_IDS = 6;

    compute_kernel_hw_startup(0, 4);
    pack_untilize_dest_init<TILES, TILES>(4);
    CircularBuffer score_cb(0);
    CircularBuffer id_cb(1);
    CircularBuffer out_score_cb(4);
    CircularBuffer out_id_cb(5);


#if SKV4_PROFILE

    DeviceTimestampedData("SKV_S15_MERGE_C_TOPK_B", 0);

#endif
    tile_regs_acquire();
    for (uint32_t shard = 0; shard < W; ++shard) {
        const uint32_t value_dst = shard == 0 ? SURV_VALUES : IN_VALUES;
        const uint32_t id_dst = shard == 0 ? SURV_IDS : IN_IDS;

        score_cb.wait_front(TILES);
        topk_xl_copy_tile_init(0);
        topk_xl_copy_tile<K>(0, value_dst, 0, K);
        score_cb.pop_front(TILES);

        id_cb.wait_front(TILES);
        topk_xl_copy_tile_init(1);
        topk_xl_copy_tile<K>(1, id_dst, 0, K);
        id_cb.pop_front(TILES);

        if (shard != 0) {
            // copy_tile_init changes TopK's MOP/address modifiers, so restore
            // the complete unfused state after both operands are resident.
            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(SURV_VALUES);
            topk_xl_rebuild<K, false>(SURV_VALUES, false);
        }
    }
    tile_regs_commit();
    tile_regs_wait();

    out_score_cb.reserve_back(1);
    pack_untilize_dest<TILES, TILES>(4, 1, 0, SURV_VALUES);
    out_score_cb.push_back(1);

    pack_untilize_dest_init<TILES, TILES>(5);
    out_id_cb.reserve_back(1);
    pack_untilize_dest<TILES, TILES>(5, 1, 0, SURV_IDS);
    out_id_cb.push_back(1);
    tile_regs_release();

#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15_MERGE_C_TOPK_E", 0);


#endif
}
"""
    )
    return profile_defines(profile=profile) + "".join(parts)


def payload_reader(*, profile):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    parts = []
    parts.append(
        r"""
#if SKV4_PROFILE
#include "tools/profiler/kernel_profiler.hpp"

#endif

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr auto sa = TensorAccessorArgs<2>();
    constexpr auto ia = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(2);
    const auto scores = TensorAccessor(sa, scores_addr, W * K * 2);
    const auto ids = TensorAccessor(ia, ids_addr, W * K * 4);
    CircularBuffer score_cb(0);
    CircularBuffer id_cb(1);
    Noc noc;

    for (uint32_t shard = 0; shard < W; ++shard) {

#if SKV4_PROFILE

    DeviceTimestampedData("SKV_S15_MERGE_R_LISTS_B", 0);

#endif
        score_cb.reserve_back(2);
        id_cb.reserve_back(2);
        noc.async_read(scores, score_cb, K * 2,
                       {.page_id = row, .offset_bytes = shard * K * 2},
                       {.offset_bytes = 0});
        noc.async_read(ids, id_cb, K * 4,
                       {.page_id = row, .offset_bytes = shard * K * 4},
                       {.offset_bytes = 0});
        noc.async_read_barrier();
        score_cb.push_back(2);
        id_cb.push_back(2);

#if SKV4_PROFILE

    DeviceTimestampedData("SKV_S15_MERGE_R_LISTS_E", 0);

#endif
    }
}
"""
    )
    return profile_defines(profile=profile) + "".join(parts)


def payload_writer(*, profile):
    """Construct this kernel from explicit semantic layout/schedule arguments."""
    parts = []
    parts.append(
        r"""
#if SKV4_PROFILE
#include "tools/profiler/kernel_profiler.hpp"

#endif

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr auto sa = TensorAccessorArgs<1>();
    constexpr auto ia = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(2);
    const auto scores = TensorAccessor(sa, scores_addr, K * 2);
    const auto ids = TensorAccessor(ia, ids_addr, K * 4);
    CircularBuffer score_cb(4);
    CircularBuffer id_cb(5);
    Noc noc;


#if SKV4_PROFILE

    DeviceTimestampedData("SKV_S15_MERGE_W_OUTPUT_B", 0);

#endif
    score_cb.wait_front(1);
    id_cb.wait_front(1);
    noc.async_write(score_cb, scores, K * 2,
                    {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
    noc.async_write(id_cb, ids, K * 4,
                    {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
    noc.async_write_barrier();
    score_cb.pop_front(1);
    id_cb.pop_front(1);

#if SKV4_PROFILE
    DeviceTimestampedData("SKV_S15_MERGE_W_OUTPUT_E", 0);


#endif
}
"""
    )
    return profile_defines(profile=profile) + "".join(parts)
