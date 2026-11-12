"""Exact odd BF16 decoder and four-head post-dot FP32 scale application.

Declared directly: do not derive this helper by editing arithmetic decoder source.
"""

SOURCE = r"""
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
"""


def source():
    return SOURCE
