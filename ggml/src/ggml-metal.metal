#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#include "ggml-common.h"

#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal.metal
//
#if __METAL_VERSION__ < 310
#define GGML_METAL_NO_BFLOAT
#endif

#if !defined(GGML_METAL_NO_BFLOAT)
typedef matrix<bfloat, 4, 4> bfloat4x4;
#endif

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

// NOTE: this is not dequantizing - we are simply fitting the template
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

#if !defined(GGML_METAL_NO_BFLOAT)
template <typename type4x4>
void dequantize_bf16(device const bfloat4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}
#endif

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const half d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il<2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * ql = (device const uint8_t *)xb->ql;
    device const uint8_t * qh = (device const uint8_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 64*(il/8) + 32*((il/2)&1) + 16*(il&1);
    qh = qh + 32*(il/8) + 16*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint16_t  kmask1 = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);
    const uint16_t  kmask2 = il>1 ? 0xF0              : 0x0F;
    const float       coef = il>1 ? 1.f/16.f          : 1.f;
    const float ml = d_all * sc * 32.f;
    const float dl = d_all * sc * coef;
    for (int i = 0; i < 16; ++i) {
        const half q = il&1 ? ((ql[i] & kmask2) | ((qh[i] & kmask1) << 2))
                            : ((ql[i] & kmask2) | ((qh[i] & kmask1) << 4));
        reg[i/4][i%4] = dl * q - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 8*ib32;
    device const uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
        reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
        reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
    }
}

template <typename type4x4>
void dequantize_iq2_s(device const block_iq2_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * signs = qs + QK_K/8;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    for (int i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]);
        reg[i/4+2][i%4] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]);
    }
}

template <typename type4x4>
void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    const float d = xb->d;
    device const uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device const uint16_t * qh = xb->qh;
    const float dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    const float ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
    const uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml;
        reg[1][i] = dl * (grid1[i] >>  4) + ml;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml;
        reg[3][i] = dl * (grid2[i] >>  4) + ml;
    }
}

template <typename type4x4>
void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * sc = (device const uint16_t *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = scale.f16;

    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * qh = xb->qh + 2*ib32 + il;

    const float dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    const float ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    const float ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
        reg[1][i] = dl * (grid1[i] >>  4) + ml1;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
        reg[3][i] = dl * (grid2[i] >>  4) + ml2;
    }
}

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

// general-purpose kernel for addition, subtraction, multiplication and division of two tensors
// pros: works for non-contiguous tensors, supports broadcast across all dims
// cons: not very efficient
kernel void kernel_add(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        constant  int64_t & offs,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01 + offs;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1  + offs;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) + *((device float *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_sub(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        constant  int64_t & offs,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01 + offs;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1  + offs;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) - *((device float *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_mul(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) * *((device float *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_div(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) / *((device float *)(src1_ptr + i10*nb10));
    }
}

template<typename T>
kernel void kernel_repeat(
        device const char * src0,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3 % ne03;
    const int64_t i02 = i2 % ne02;
    const int64_t i01 = i1 % ne01;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device       char * dst_ptr  = dst  +  i3*nb3  +  i2*nb2  +  i1*nb1 ;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i00 = i0 % ne00;
        *((device T *)(dst_ptr + i0*nb0)) = *((device T *)(src0_ptr + i00*nb00));
    }
}

typedef decltype(kernel_repeat<float>) kernel_repeat_t;

template [[host_name("kernel_repeat_f32")]] kernel kernel_repeat_t kernel_repeat<float>;
template [[host_name("kernel_repeat_f16")]] kernel kernel_repeat_t kernel_repeat<half>;
template [[host_name("kernel_repeat_i32")]] kernel kernel_repeat_t kernel_repeat<int>;
template [[host_name("kernel_repeat_i16")]] kernel kernel_repeat_t kernel_repeat<short>;

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig % nb];
}

kernel void kernel_sub_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] - src1[tpig % nb];
}

kernel void kernel_mul_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb  [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig % nb];
}

kernel void kernel_div_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb  [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] / src1[tpig % nb];
}

kernel void kernel_scale(
        device const float * src0,
        device       float * dst,
        constant     float & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_scale_4(
        device const float4 * src0,
        device       float4 * dst,
        constant     float  & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_clamp(
        device const float * src0,
        device       float * dst,
        constant     float & min,
        constant     float & max,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] < min ? min : (src0[tpig] > max ? max : src0[tpig]);
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

kernel void kernel_sigmoid(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = 1.0f / (1.0f + exp(-src0[tpig]));
}

kernel void kernel_tanh(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = precise::tanh(x);
}

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

kernel void kernel_gelu(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    // BEWARE !!!
    // Simply using "tanh" instead of "precise::tanh" will sometimes results in NaNs!
    // This was observed with Falcon 7B and 40B models
    //
    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_quick(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_gelu_quick_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_silu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_silu_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_sqr(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src0[tpig];
}

kernel void kernel_sqrt(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sqrt(src0[tpig]);
}

kernel void kernel_sin(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sin(src0[tpig]);
}

kernel void kernel_cos(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = cos(src0[tpig]);
}

kernel void kernel_sum_rows(
        device const float * src0,
        device       float * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tpig[[thread_position_in_grid]]) {
    int64_t i3 = tpig.z;
    int64_t i2 = tpig.y;
    int64_t i1 = tpig.x;

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    device const float * src_row = (device const float *) ((device const char *) src0 + i1*nb01 + i2*nb02 + i3*nb03);
    device       float * dst_row = (device       float *) ((device       char *) dst  + i1*nb1  + i2*nb2  + i3*nb3);

    float row_sum = 0;

    for (int64_t i0 = 0; i0 < ne00; i0++) {
        row_sum += src_row[i0];
    }

    dst_row[0] = row_sum;
}

template<typename T>
kernel void kernel_soft_max(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float * psrc0 = (device const float *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    device const     T * pmask = src1 != src0 ? (device const    T *) src1         + i01*ne00 : nullptr;
    device       float * pdst  = (device       float *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = -INFINITY;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        lmax = MAX(lmax, psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        const float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float4 * psrc4 = (device const float4 *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;
    device const      T * pmask = src1 != src0 ? (device const     T *) src1         + i01*ne00/4 : nullptr;
    device       float4 * pdst4 = (device       float4 *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;

    float slope = 1.0f;

    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = -INFINITY;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;

kernel void kernel_diag_mask_inf(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant       int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i02 = tpig[2];
    const int64_t i01 = tpig[1];
    const int64_t i00 = tpig[0];

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

kernel void kernel_diag_mask_inf_8(
        device const float4 * src0,
        device       float4 * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne01,
        constant        int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {

    const int64_t i = 2*tpig[0];

    dst[i+0] = src0[i+0];
    dst[i+1] = src0[i+1];
    int64_t i4 = 4*i;
    const int64_t i02 = i4/(ne00*ne01); i4 -= i02*ne00*ne01;
    const int64_t i01 = i4/(ne00);      i4 -= i01*ne00;
    const int64_t i00 = i4;
    for (int k = 3; k >= 0; --k) {
        if (i00 + 4 + k <= n_past + i01) {
            break;
        }
        dst[i+1][k] = -INFINITY;
        if (i00 + k > n_past + i01) {
            dst[i][k] = -INFINITY;
        }
    }
}

// ref: ggml.c:ggml_compute_forward_ssm_conv_f32
// TODO: optimize
kernel void kernel_ssm_conv_f32(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    const int64_t nc  = ne10;
  //const int64_t ncs = ne00;
  //const int64_t nr  = ne01;
  //const int64_t n_t = ne1;
  //const int64_t n_s = ne2;

    device const float * s = (device const float *) ((device const char *) src0 + ir*nb01 + i2*nb00 + i3*nb02);
    device const float * c = (device const float *) ((device const char *) src1 + ir*nb11);
    device       float * x = (device       float *) ((device       char *) dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = sumf;
}

// ref: ggml.c:ggml_compute_forward_ssm_scan_f32
// TODO: optimize
kernel void kernel_ssm_scan_f32(
        device const void * src0,
        device const void * src1,
        device const void * src2,
        device const void * src3,
        device const void * src4,
        device const void * src5,
        device      float * dst,
        constant  int64_t & d_state,
        constant  int64_t & d_inner,
        constant  int64_t & n_seq_tokens,
        constant  int64_t & n_seqs,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant uint64_t & nb20,
        constant uint64_t & nb21,
        constant uint64_t & nb22,
        constant uint64_t & nb30,
        constant uint64_t & nb31,
        constant uint64_t & nb40,
        constant uint64_t & nb41,
        constant uint64_t & nb42,
        constant uint64_t & nb50,
        constant uint64_t & nb51,
        constant uint64_t & nb52,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i3 = tgpig.y;

    const int64_t nc  = d_state;
  //const int64_t nr  = d_inner;
    const int64_t n_t = n_seq_tokens;
  //const int64_t n_s = n_seqs;

    for (int64_t i2 = 0; i2 < n_t; ++i2) {
        device const float * s0 = (device const float *) ((device const char *) src0 + ir*nb01 + i3*nb02);
        device const float * x  = (device const float *) ((device const char *) src1 + ir*nb10 + i2*nb11 + i3*nb12);
        device const float * dt = (device const float *) ((device const char *) src2 + ir*nb20 + i2*nb21 + i3*nb22);
        device const float * A  = (device const float *) ((device const char *) src3 + ir*nb31);
        device const float * B  = (device const float *) ((device const char *) src4 + i2*nb41 + i3*nb42);
        device const float * C  = (device const float *) ((device const char *) src5 + i2*nb51 + i3*nb52);
        device       float * y  = (device       float *) ((device       char *) dst  + ir*nb10 + i2*nb11 + i3*nb12); // TODO: do not use src1 strides
        device       float * s  = (device       float *) ((device       char *) dst  + ir*nb01 + i3*nb02 +    nb13);

        if (i2 > 0) {
            s0 = s;
        }

        // i1 == 0
        float dt_soft_plus = dt[0] <= 20.0f ? log(1.0f + exp(dt[0])) : dt[0];
        float x_dt = x[0] * dt_soft_plus;
        float sumf = 0.0f;

        for (int64_t i0 = 0; i0 < nc; ++i0) {
            int64_t i = i0;
            float state = (s0[i] * exp(dt_soft_plus * A[i])) + (B[i0] * x_dt);
            sumf += state * C[i0];
            s[i] = state;
        }

        y[0] = sumf;
    }
}

kernel void kernel_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * sum [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * x = (device const float *) ((device const char *) src0 + tgpig*nb01);
    // MEAN
    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += x[i00];
    }
    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean  = sum[0] / ne00;

    // recenter and VARIANCE
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * y = dst + tgpig*ne00;
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = x[i00] - mean;
        sum[tpitg] += y[i00] * y[i00];
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float variance = sum[0] / ne00;

    const float scale = 1.0f/sqrt(variance + eps);
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = y[i00] * scale;
    }
}

kernel void kernel_rms_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float4 * x = (device const float4 *) ((device const char *) src0 + tgpig*nb01);

    float4 sumf = 0;
    float all_sum = 0;

    // parallel sum
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = all_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        all_sum = buf[tiisg];
        all_sum = simd_sum(all_sum);
    }

    const float mean  = all_sum/ne00;
    const float scale = 1.0f/sqrt(mean + eps);

    device float4 * y = (device float4 *) (dst + tgpig*ne00);
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
}

kernel void kernel_group_norm(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int32_t & n_groups,
        constant     float & eps,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    const int64_t ne = ne00*ne01*ne02;
    const int64_t gs = ne00*ne01*((ne02 + n_groups - 1) / n_groups);

    int start = tgpig * gs;
    int end   = start + gs;

    start += tpitg;

    if (end >= ne) {
        end = ne;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += ntg) {
        tmp += src0[j];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float mean = tmp / gs;
    tmp = 0.0f;

    for (int j = start; j < end; j += ntg) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float variance = tmp / gs;
    const float scale = 1.0f/sqrt(variance + eps);
    for (int j = start; j < end; j += ntg) {
        dst[j] *= scale;
    }
}

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// putting them in the kernel cause a significant performance penalty
#define N_DST 4        // each SIMD group works on 4 rows
#define N_SIMDGROUP 2  // number of SIMD groups in a thread group
//Note: This is a template, but strictly speaking it only applies to
//      quantizations where the block size is 32. It also does not
//      guard against the number of rows not being divisible by
//      N_DST, so this is another explicit assumption of the implementation.
template<typename block_q_type, int nr, int nsg, int nw>
void mul_vec_q_n_f32_impl(
        device const void  * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                     uint3   tgpig,
                     uint    tiisg,
                     uint    sgitg) {
    const int nb = ne00/QK4_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

  //const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

  //device const block_q_type * x = (device const block_q_type *) ((device char *) src0 + offset0);
    device const float        * y = (device const float        *) ((device char *) src1 + offset1);

    // pointers to src0 rows
    device const block_q_type * ax[nr];
    for (int row = 0; row < nr; ++row) {
        const uint offset0 = (first_row + row)*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    float yl[16]; // src1 vector cache
    float sumf[nr] = {0.f};

    const int ix = (tiisg/2);
    const int il = (tiisg%2)*8;

    device const float * yb = y + ix * QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy[2] = { 0.f, 0.f };

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

#pragma unroll
        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[im*ne0*ne1 + r1*ne0 + first_row + row] = tot;
        }
    }
}

kernel void kernel_mul_mv_q4_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,nb01,nb02,nb03,ne10,ne12,nb11,nb12,nb13,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q4_1_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32_impl<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,nb01,nb02,nb03,ne10,ne12,nb11,nb12,nb13,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q5_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,nb01,nb02,nb03,ne10,ne12,nb11,nb12,nb13,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q5_1_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,nb01,nb02,nb03,ne10,ne12,nb11,nb12,nb13,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}


#define NB_Q8_0 8

void kernel_mul_mv_q8_0_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = ne00/QK8_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

  //const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

  //device const block_q8_0 * x = (device const block_q8_0 *) ((device char *) src0 + offset0);
    device const float      * y = (device const float      *) ((device char *) src1 + offset1);

    // pointers to src0 rows
    device const block_q8_0 * ax[nr];
    for (int row = 0; row < nr; ++row) {
        const uint offset0 = (first_row + row)*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float yl[NB_Q8_0];
    float sumf[nr]={0.f};

    const int ix = tiisg/4;
    const int il = tiisg%4;

    device const float * yb = y + ix * QK8_0 + NB_Q8_0*il;

    // each thread in a SIMD group deals with NB_Q8_0 quants at a time
    for (int ib = ix; ib < nb; ib += nw/4) {
        for (int i = 0; i < NB_Q8_0; ++i) {
            yl[i] = yb[i];
        }

        for (int row = 0; row < nr; row++) {
            device const int8_t * qs = ax[row][ib].qs + NB_Q8_0*il;
            float sumq = 0.f;
            for (int iq = 0; iq < NB_Q8_0; ++iq) {
                sumq += qs[iq] * yl[iq];
            }
            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NB_Q8_0 * nw;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl(src0,src1,dst,ne00,ne01,ne02,nb01,nb02,nb03,ne10,ne12,nb11,nb12,nb13,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

#define N_MV_T_T 4

template<typename T0, typename T04, typename T1, typename T14>
void kernel_mul_mv_impl(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb00,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne11,
                   int64_t   ne12,
                  uint64_t   nb10,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3     tgpig,
                   uint      tiisg) {
    const int64_t r0 = tgpig.x;
    const int64_t rb = tgpig.y*N_MV_T_T;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    if (ne00 < 128) {
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            const uint offset1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            device const T1 * y = (device const T1 *) (src1 + offset1);

            float sumf = 0;
            for (int i = tiisg; i < ne00; i += 32) {
                sumf += (T0) x[i] * (T1) y[i];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        device const T04 * x4 = (device const T04 *) x;
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            const uint offset1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            device const T1  * y  = (device const T1  *) (src1 + offset1);
            device const T14 * y4 = (device const T14 *) y;

            float sumf = 0;
            for (int i = tiisg; i < ne00/4; i += 32) {
                for (int k = 0; k < 4; ++k) sumf += (float) (x4[i][k] * y4[i][k]);
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) (x[i] * y[i]);
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_impl<T0, T04, T1, T14>(
        src0,
        src1,
        dst,
        ne00,
        ne01,
        ne02,
        nb00,
        nb01,
        nb02,
        nb03,
        ne10,
        ne11,
        ne12,
        nb10,
        nb11,
        nb12,
        nb13,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv<half, half4, half, half4>) mul_mv_t;

template [[host_name("kernel_mul_mv_f32_f32")]]   kernel mul_mv_t kernel_mul_mv<float,  float4,  float,  float4>;
template [[host_name("kernel_mul_mv_f16_f32")]]   kernel mul_mv_t kernel_mul_mv<half,   half4,   float,  float4>;
template [[host_name("kernel_mul_mv_f16_f16")]]   kernel mul_mv_t kernel_mul_mv<half,   half4,   half,   half4>;
#if !defined(GGML_METAL_NO_BFLOAT)
template [[host_name("kernel_mul_mv_bf16_f32")]]  kernel mul_mv_t kernel_mul_mv<bfloat, bfloat4, float,  float4>;
template [[host_name("kernel_mul_mv_bf16_bf16")]] kernel mul_mv_t kernel_mul_mv<bfloat, bfloat4, bfloat, bfloat4>;
#endif

template<typename T, typename T4>
kernel void kernel_mul_mv_1row(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const T     * x = (device const T     *) (src0 + offset0);
    device const float * y = (device const float *) (src1 + offset1);

    float sumf = 0;
    if (ne00 < 128) {
        for (int i = tiisg; i < ne00; i += 32) {
            sumf += (float) x[i] * (float) y[i];
        }
        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    } else {
        device const T4     * x4 = (device const T4     *) x;
        device const float4 * y4 = (device const float4 *) y;

        for (int i = tiisg; i < ne00/4; i += 32) {
            for (int k = 0; k < 4; ++k) sumf += (float) (x4[i][k] * y4[i][k]);
        }

        float all_sum = simd_sum(sumf);

        if (tiisg == 0) {
            for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) (x[i] * y[i]);
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_1row<half, half4>) mul_mv_1row_t;

template [[host_name("kernel_mul_mv_f16_f32_1row")]]  kernel mul_mv_1row_t kernel_mul_mv_1row<half,   half4>;
#if !defined(GGML_METAL_NO_BFLOAT)
template [[host_name("kernel_mul_mv_bf16_f32_1row")]] kernel mul_mv_1row_t kernel_mul_mv_1row<bfloat, bfloat4>;
#endif

// Assumes row size (ne00) is a multiple of 4
template<typename T, typename T4>
kernel void kernel_mul_mv_l4(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]]) {

    const int nrows = ne11;
    const int64_t r0 = tgpig.x;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    device const T4 * x4 = (device const T4 *) (src0 + offset0);

    for (int r1 = 0; r1 < nrows; ++r1) {
        const uint offset1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

        device const float4 * y4 = (device const float4 *) (src1 + offset1);

        float sumf = 0;
        for (int i = tiisg; i < ne00/4; i += 32) {
            for (int k = 0; k < 4; ++k) sumf += (float) (x4[i][k] * y4[i][k]);
        }

        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_l4<half, half4>) mul_mv_l4_t;

template [[host_name("kernel_mul_mv_f16_f32_l4")]]  kernel mul_mv_l4_t kernel_mul_mv_l4<half, half4>;
#if !defined(GGML_METAL_NO_BFLOAT)
template [[host_name("kernel_mul_mv_bf16_f32_l4")]] kernel mul_mv_l4_t kernel_mul_mv_l4<bfloat, bfloat4>;
#endif

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

template<typename T>
kernel void kernel_rope_norm(
        device const    void * src0,
        device const int32_t * src1,
        device const   float * src2,
        device         float * dst,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant     int64_t & ne03,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant    uint64_t & nb03,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant     int64_t & ne2,
        constant     int64_t & ne3,
        constant    uint64_t & nb0,
        constant    uint64_t & nb1,
        constant    uint64_t & nb2,
        constant    uint64_t & nb3,
        constant         int & n_past,
        constant         int & n_dims,
        constant         int & n_ctx_orig,
        constant       float & freq_base,
        constant       float & freq_scale,
        constant       float & ext_factor,
        constant       float & attn_factor,
        constant       float & beta_fast,
        constant       float & beta_slow,
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        uint3 tgpig[[threadgroup_position_in_grid]]) {
    const int64_t i3 = tgpig[2];
    const int64_t i2 = tgpig[1];
    const int64_t i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    device const int32_t * pos = src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/n_dims;

    float cos_theta;
    float sin_theta;

    for (int64_t i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x) {
        if (i0 < n_dims) {
            const int64_t ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_neox(
        device const    void * src0,
        device const int32_t * src1,
        device const   float * src2,
        device         float * dst,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant     int64_t & ne03,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant    uint64_t & nb03,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant     int64_t & ne2,
        constant     int64_t & ne3,
        constant    uint64_t & nb0,
        constant    uint64_t & nb1,
        constant    uint64_t & nb2,
        constant    uint64_t & nb3,
        constant         int & n_past,
        constant         int & n_dims,
        constant         int & n_ctx_orig,
        constant       float & freq_base,
        constant       float & freq_scale,
        constant       float & ext_factor,
        constant       float & attn_factor,
        constant       float & beta_fast,
        constant       float & beta_slow,
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        uint3 tgpig[[threadgroup_position_in_grid]]) {
    const int64_t i3 = tgpig[2];
    const int64_t i2 = tgpig[1];
    const int64_t i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    device const int32_t * pos = src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/n_dims;

    float cos_theta;
    float sin_theta;

    for (int64_t i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x) {
        if (i0 < n_dims) {
            const int64_t ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_theta - x1*sin_theta;
            dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

typedef decltype(kernel_rope_norm<float>) kernel_rope_norm_t;
typedef decltype(kernel_rope_neox<float>) kernel_rope_neox_t;

template [[host_name("kernel_rope_norm_f32")]] kernel kernel_rope_norm_t kernel_rope_norm<float>;
template [[host_name("kernel_rope_norm_f16")]] kernel kernel_rope_norm_t kernel_rope_norm<half>;

template [[host_name("kernel_rope_neox_f32")]] kernel kernel_rope_neox_t kernel_rope_neox<float>;
template [[host_name("kernel_rope_neox_f16")]] kernel kernel_rope_neox_t kernel_rope_neox<half>;

typedef void (im2col_t)(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t iiw = tgpig[2] * s0 + tpitg[2] * d0 - p0;
    const int32_t iih = tgpig[1] * s1 + tpitg[1] * d1 - p1;

    const int32_t offset_dst =
        (tpitg[0] * tgpg[1] * tgpg[2] + tgpig[1] * tgpg[2] + tgpig[2]) * CHW +
        (tgpig[0] * (ntg[1] * ntg[2]) + tpitg[1] * ntg[2] + tpitg[2]);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int32_t offset_src = tpitg[0] * ofs0 + tgpig[0] * ofs1;
        pdst[offset_dst] = x[offset_src + iih * IW + iiw];
    }
}

template [[host_name("kernel_im2col_f32")]] kernel im2col_t kernel_im2col<float>;
template [[host_name("kernel_im2col_f16")]] kernel im2col_t kernel_im2col<half>;

typedef void (im2col_ext_t)(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        constant   int32_t & N,
        constant   int32_t & KH,
        constant   int32_t & KW,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col_ext(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        constant   int32_t & N,
        constant   int32_t & KH,
        constant   int32_t & KW,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],      // tgpg[0] = D x IC x KH x KW, CHW = IC x KH x KW
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {  // [M, 1, 1]
    const int32_t KHW = KH * KW;             // KHW == ntg[1] * ntg[2], KW == ntg[2]

    const int32_t d = tgpig[0] / CHW;
    const int32_t chw = tgpig[0] % CHW;
    const int32_t tgpig_0 = chw / KHW;  // 0 ~ (IC - 1)
    const int32_t HW = tgpig[0] % KHW;

    const int32_t tpitg_0 = (d * ntg[0]) + tpitg[0];
    if (tpitg_0 >= N) {
        return;
    }

    const int32_t tpitg_1 = HW / KW;
    const int32_t tpitg_2 = HW % KW;

    const int32_t iiw = tgpig[2] * s0 + tpitg_2 * d0 - p0;
    const int32_t iih = tgpig[1] * s1 + tpitg_1 * d1 - p1;

    const int32_t offset_dst =
        (tpitg_0 * tgpg[1] * tgpg[2] + tgpig[1] * tgpg[2] + tgpig[2]) * CHW +
        (tgpig_0 * KHW + tpitg_1 * KW + tpitg_2);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int32_t offset_src = tpitg_0 * ofs0 + tgpig_0 * ofs1;
        pdst[offset_dst] = x[offset_src + iih * IW + iiw];
    }
}

template [[host_name("kernel_im2col_ext_f32")]] kernel im2col_ext_t kernel_im2col_ext<float>;
template [[host_name("kernel_im2col_ext_f16")]] kernel im2col_ext_t kernel_im2col_ext<half>;

kernel void kernel_upscale_f32(
    device  const char * src0,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    constant     float & sf0,
    constant     float & sf1,
    constant     float & sf2,
    constant     float & sf3,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3/sf3;
    const int64_t i02 = i2/sf2;
    const int64_t i01 = i1/sf1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int64_t i00 = i0/sf0;

        device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1  +  i0*nb0);

        dst_ptr[0] = src0_ptr[0];
    }
}

kernel void kernel_pad_f32(
    device  const char * src0,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1);

    if (i1 < ne01 && i2 < ne02 && i3 < ne03) {
        for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
            if (i0 < ne00) {
                dst_ptr[i0] = src0_ptr[i0];
            } else {
                dst_ptr[i0] = 0.0f;
            }
        }

        return;
    }

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dst_ptr[i0] = 0.0f;
    }
}

kernel void kernel_arange_f32(
    device        char * dst,
    constant   int64_t & ne0,
    constant   float   & start,
    constant   float   & step,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dst_ptr[i0] = start + step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    device  const char * src0,
    device        char * dst,
    constant  uint64_t & nb1,
    constant  int      & dim,
    constant  int      & max_period,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst +  i*nb1);

    int half_ = dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (dim % 2 != 0 && tpitg.x == 0) {
        embed_data[dim] = 0.f;
    }
}

// bitonic sort implementation following the CUDA kernels as reference
typedef void (argsort_t)(
        device const float  * x,
        device     int32_t  * dst,
        constant   int64_t  & ncols,
        constant   int64_t  & ncols_pad,
        threadgroup int32_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
        device const float   * x,
        device       int32_t * dst,
        constant     int64_t & ncols,
        constant     int64_t & ncols_pad,
        threadgroup int32_t  * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    // bitonic sort
    int col = tpitg[0];
    int row = tgpig[1];

    if (col >= ncols_pad) return;

    device const float   * x_row   = x + row * ncols;
    threadgroup int32_t  * dst_row = shared_values;

    // initialize indices
    dst_row[col] = col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

template [[host_name("kernel_argsort_f32_i32_asc")]]  kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_DESC>;

kernel void kernel_leaky_relu_f32(
        device const float * src0,
        device       float * dst,
        constant     float & slope,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] > 0.0f ? src0[tpig] : src0[tpig] * slope;
}

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // key type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short D,         // head size
    short Q  = 8,    // queries per threadgroup
    short KV = 8,    // key/value processed per each simdgroup
    short C  = 32>   // cache items per threadgroup
kernel void kernel_flash_attn_ext(
        device const  char * q,
        device const  char * k,
        device const  char * v,
        device const  char * mask,
        device       float * dst,
        constant   int32_t & ne01,
        constant   int32_t & ne02,
        constant   int32_t & ne03,
        constant  uint32_t & nb01,
        constant  uint32_t & nb02,
        constant  uint32_t & nb03,
        constant   int32_t & ne11,
        constant   int32_t & ne_12_2, // assume K and V are same shape
        constant   int32_t & ne_12_3,
        constant  uint32_t & nb_12_1,
        constant  uint32_t & nb_12_2,
        constant  uint32_t & nb_12_3,
        constant  uint32_t & nb31,
        constant   int32_t & ne1,
        constant   int32_t & ne2,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint16_t & n_head_log2,
        constant     float & logit_softcap,
        threadgroup   half * shared [[threadgroup(0)]],
        ushort3  tgpig[[threadgroup_position_in_grid]],
        ushort3    ntg[[threads_per_threadgroup]],
        ushort   tiisg[[thread_index_in_simdgroup]],
        ushort   sgitg[[simdgroup_index_in_threadgroup]]) {
    const short nsg = ntg.y; // number of simdgroups

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    const short D4  = D/4;
    const short D8  = D/8;
    const short D16 = D/16;
    const short NW  = N_SIMDWIDTH;
    const short SH  = (2*C + Q); // shared memory per simdgroup in (half)

    const short TS = nsg*SH;   // shared memory size per query in (s_t)
    const short T  = D + 2*TS; // shared memory size per query in (half)

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shared +              0*D); // holds the query data
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shared +              0*D); // same as above but in q4_t
    threadgroup o_t  * so  = (threadgroup o_t  *) (shared +              0*D); // reuse query data for accumulation
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shared +              0*D); // reuse query data for accumulation
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shared + 2*sgitg*SH + Q*D); // scratch buffer for attention and diagonal matrix

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shared + sgitg*(4*16*KV) + Q*T); // scratch buffer to load K in shared memory
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shared + sgitg*(4*16*KV) + Q*T); // same as above but in k4x4_t

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shared + sgitg*(4*16*KV) + Q*T); // scratch buffer to load V in shared memory
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shared + sgitg*(4*16*KV) + Q*T); // same as above but in v4x4_t

    // store the result for all queries in local memory in 8x8 matrices (the O matrix from the paper)
    o8x8_t lo[D8];

    // load heads from Q to shared memory
    for (short j = sgitg; j < Q; j += nsg) {
        device const float4 * q4 = (device const float4 *) ((device const char *) q + ((iq1 + j)*nb01 + iq2*nb02 + iq3*nb03));

        for (short i = tiisg; i < D4; i += NW) {
            if (iq1 + j < ne01) {
                sq4[j*D4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*D4 + i] = (q4_t) 0.0f;
            }
        }
    }

    // zero out lo
    for (short i = 0; i < D8; ++i) {
        lo[i] = make_filled_simdgroup_matrix<o_t, 8>((o_t) 0.0f);
    }

    // zero out shared memory SH
    for (short j = 0; j < Q; ++j) {
        for (short i = tiisg; i < SH; i += NW) {
            ss[j*TS + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        half S[Q] = { [0 ... Q-1] = 0.0f };
        half M[Q] = { [0 ... Q-1] = -__FLT16_MAX__/2 };

        // thread indices inside the simdgroup
        // TODO: see if we can utilize quad-group functions for better performance
        //       https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (6.9.3)
        const short tx = tiisg%4;
        const short ty = tiisg/4;

        // broadcast kv
        //const short rk2 = ne02/ne12;
        //const short rk3 = ne03/ne13;

        const short ikv2 = iq2/(ne02/ne_12_2);
        const short ikv3 = iq3/(ne03/ne_12_3);

        // load the queries from shared memory into local memory
        q8x8_t mq[D8];

        for (short i = 0; i < D8; ++i) {
            simdgroup_load(mq[i], sq + i*8, D);
        }

        const bool has_mask = mask != q;

        half slope = 1.0f;

        // ALiBi
        if (max_bias > 0.0f) {
            const short h = iq2;

            const half  base = h < n_head_log2 ? m0 : m1;
            const short exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ic0 < ne11; ic0 += C*nsg) {
            const int ic = ic0 + C*sgitg;
            if (ic >= ne11) {
                break;
            }

            if (has_mask) {
                // used to detect blocks full of -INF
                half smax = -INFINITY;

                for (short j = 0; j < Q; ++j) {
                    device const half * pm = (device const half *) ((device const char *) mask + (iq1 + j)*nb31);

                    const half m = pm[ic + tiisg];

                    ss[j*TS + C + tiisg] = m;
                    smax = max(smax, m);
                }

                smax = simd_max(smax);

                if (smax == -INFINITY) {
                    continue;
                }
            }

            // Q*K^T
            {
                for (short cc = 0; cc < C/8; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    // this is compile-time check, so it does not have runtime overhead
                    if (is_same<kd4x4_t, k4x4_t>::value) {
                        // we can read directly from global memory
                        device const k_t * pk = (device const k_t *) ((device const char *) k + ((ic + 8*cc)*nb_12_1 + ikv2*nb_12_2 + ikv3*nb_12_3));

#pragma unroll
                        for (short i = 0; i < D8; ++i) {
                            k8x8_t mk;
                            simdgroup_load(mk, pk + i*8, nb_12_1/sizeof(k_t), 0, true); // transpose // TODO: use ne10

                            simdgroup_multiply_accumulate(mqk, mq[i], mk, mqk);
                        }
                    } else {
                        for (short ii = 0; ii < D16; ii += 4) {
                            device const kd4x4_t * pk4x4 = (device const kd4x4_t *) ((device const char *) k + ((ic + 8*cc + ty)*nb_12_1 + ikv2*nb_12_2 + ikv3*nb_12_3));

                            if (D16%4 == 0) {
                                // the head is evenly divisible by 4*16 = 64, so no need for bound checks
                                {
                                    k4x4_t tmp;
                                    deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                    sk4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
                                for (short k = 0; k < 4; ++k) {
                                    k8x8_t mk;

                                    simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                    simdgroup_multiply_accumulate(mqk, mq[2*(ii + k) + 0], mk, mqk);

                                    simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                    simdgroup_multiply_accumulate(mqk, mq[2*(ii + k) + 1], mk, mqk);
                                }
                            } else {
                                if (ii + tx < D16) {
                                    k4x4_t tmp;
                                    deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                    sk4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < D16; ++k) {
                                    k8x8_t mk;

                                    simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                    simdgroup_multiply_accumulate(mqk, mq[2*(ii + k) + 0], mk, mqk);

                                    simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                    simdgroup_multiply_accumulate(mqk, mq[2*(ii + k) + 1], mk, mqk);
                                }
                            }
                        }
                    }

                    // cast qk_t -> s_t
                    s8x8_t mqks(1.0f);
                    simdgroup_multiply(mqks, mqk, mqks);
                    simdgroup_store(mqks, ss + 8*cc, TS, 0, false);
                }
            }

            // online softmax
            {
                for (ushort j = 0; j < Q; ++j) {
                    const half m = M[j];

                    // scale and apply the logitcap / mask
                    half s = ss[j*TS + tiisg]*scale;

                    if (logit_softcap != 0.0f) {
                        s = logit_softcap*precise::tanh(s);
                    }

                    // mqk = mqk + mask*slope
                    s += slope*ss[j*TS + C + tiisg];

                    M[j] = simd_max(max(M[j], s));

                    const half ms = exp(m - M[j]);
                    const half vs = exp(s - M[j]);

                    S[j] = S[j]*ms + simd_sum(vs);

                    // the P matrix from the paper (Q rows, C columns)
                    ss[j*TS + tiisg] = vs;

                    // create a QxQ diagonal matrix for rescaling the output
                    if (tiisg == j) {
                        ss[j*TS + 2*C + j] = ms;
                    }
                }
            }

            // O = diag(ms)*O
            {
                s8x8_t mm;
                simdgroup_load(mm, ss + 2*C, TS, 0, false);

#pragma unroll
                for (short i = 0; i < D8; ++i) {
                    simdgroup_multiply(lo[i], mm, lo[i]);
                }
            }

            // O = O + (Q*K^T)*V
            {
                for (short cc = 0; cc < C/8; ++cc) {
                    s8x8_t ms;
                    simdgroup_load(ms, ss + 8*cc, TS, 0, false);

                    if (is_same<vd4x4_t, v4x4_t>::value) {
                        // we can read directly from global memory
                        device const v_t * pv = (device const v_t *) ((device const char *) v + ((ic + 8*cc)*nb_12_1 + ikv2*nb_12_2 + ikv3*nb_12_3));
#pragma unroll
                        for (short i = 0; i < D8; ++i) {
                            v8x8_t mv;
                            simdgroup_load(mv, pv + i*8, nb_12_1/sizeof(v_t), 0, false); // TODO: use ne20

                            simdgroup_multiply_accumulate(lo[i], ms, mv, lo[i]);
                        }
                    } else {
                        for (short ii = 0; ii < D16; ii += 4) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) ((device const char *) v + ((ic + 8*cc + ty)*nb_12_1 + ikv2*nb_12_2 + ikv3*nb_12_3));

                            if (D16%4 == 0) {
                                // no need for bound checks
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
                                for (short k = 0; k < 4; ++k) {
                                    v8x8_t mv;

                                    simdgroup_load(mv, sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 0], ms, mv, lo[2*(ii + k) + 0]);

                                    simdgroup_load(mv, sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 1], ms, mv, lo[2*(ii + k) + 1]);
                                }
                            } else {
                                if (ii + tx < D16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < D16; ++k) {
                                    v8x8_t mv;

                                    simdgroup_load(mv, sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 0], ms, mv, lo[2*(ii + k) + 0]);

                                    simdgroup_load(mv, sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 1], ms, mv, lo[2*(ii + k) + 1]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        for (short j = 0; j < Q; ++j) {
            if (tiisg == 0) {
                ss[j*TS + 0] = S[j];
                ss[j*TS + 1] = M[j];
            }
        }
    }

    // reduce the warps sequentially
    for (ushort sg = 1; sg < nsg; ++sg) {
        half S = { 0.0f };
        half M = { -__FLT16_MAX__/2 };

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each simdgroup stores its output to shared memory, reusing sq
        if (sgitg == sg) {
            for (short i = 0; i < D8; ++i) {
                simdgroup_store(lo[i], so + i*8, D, 0, false);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // the first simdgroup accumulates the results from the other simdgroups
        if (sgitg == 0) {
            for (short j = 0; j < Q; ++j) {
                const half S0 = ss[j*TS +         0];
                const half S1 = ss[j*TS + sg*SH + 0];

                const half M0 = ss[j*TS +         1];
                const half M1 = ss[j*TS + sg*SH + 1];

                M = max(M0, M1);

                const half ms0 = exp(M0 - M);
                const half ms1 = exp(M1 - M);

                S = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss[j*TS + 0] = S;
                    ss[j*TS + 1] = M;

                    ss[j*TS + 2*C + j        ] = ms0;
                    ss[j*TS + 2*C + j + sg*SH] = ms1;
                }
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            {
                s8x8_t ms0;
                s8x8_t ms1;

                simdgroup_load(ms0, ss + 2*C,         TS, 0, false);
                simdgroup_load(ms1, ss + 2*C + sg*SH, TS, 0, false);

                for (short i = 0; i < D8; ++i) {
                    o8x8_t t;

                    simdgroup_load    (t, so + i*8, D, 0, false);
                    simdgroup_multiply(t, ms1, t);

                    simdgroup_multiply_accumulate(lo[i], ms0, lo[i], t);
                }
            }
        }
    }

    // store result to shared memory (reuse sq)
    if (sgitg == 0) {
        for (short i = 0; i < D8; ++i) {
            simdgroup_store(lo[i], so + i*8, D, 0, false);
        }
    }

    device float4 * dst4 = (device float4 *) dst;

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        for (short j = 0; j < Q && iq1 + j < ne01; ++j) {
            const half S = ss[j*TS + 0];

            for (short i = tiisg; i < D4; i += NW) {
                dst4[((int64_t)iq3*ne2*ne1 + iq2 + (iq1 + j)*ne1)*D4 + i] = (float4) so4[j*D4 + i]/S;
            }
        }
    }
}

#define FA_TYPES \
    half,  half4,   simdgroup_half8x8,  \
    half,  half4x4, simdgroup_half8x8,  \
    half,  half4x4, simdgroup_half8x8,  \
    float,          simdgroup_float8x8, \
    float,          simdgroup_float8x8, \
    half,  half4,   simdgroup_half8x8

// TOOD: static_assert

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64>) flash_attn_ext_t;

template [[host_name("kernel_flash_attn_ext_f16_h64" )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  64>;
template [[host_name("kernel_flash_attn_ext_f16_h80" )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  80>;
template [[host_name("kernel_flash_attn_ext_f16_h96" )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96>;
template [[host_name("kernel_flash_attn_ext_f16_h112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  112>;
template [[host_name("kernel_flash_attn_ext_f16_h128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128>;
template [[host_name("kernel_flash_attn_ext_f16_h256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  256>;

template [[host_name("kernel_flash_attn_ext_bf16_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 64>;
template [[host_name("kernel_flash_attn_ext_bf16_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 80>;
template [[host_name("kernel_flash_attn_ext_bf16_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96>;
template [[host_name("kernel_flash_attn_ext_bf16_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 112>;
template [[host_name("kernel_flash_attn_ext_bf16_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 256>;

template [[host_name("kernel_flash_attn_ext_q4_0_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 64>;
template [[host_name("kernel_flash_attn_ext_q4_0_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 80>;
template [[host_name("kernel_flash_attn_ext_q4_0_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 96>;
template [[host_name("kernel_flash_attn_ext_q4_0_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 112>;
template [[host_name("kernel_flash_attn_ext_q4_0_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 128>;
template [[host_name("kernel_flash_attn_ext_q4_0_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 256>;

template [[host_name("kernel_flash_attn_ext_q4_1_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 64>;
template [[host_name("kernel_flash_attn_ext_q4_1_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 80>;
template [[host_name("kernel_flash_attn_ext_q4_1_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 96>;
template [[host_name("kernel_flash_attn_ext_q4_1_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 112>;
template [[host_name("kernel_flash_attn_ext_q4_1_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 128>;
template [[host_name("kernel_flash_attn_ext_q4_1_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 256>;

template [[host_name("kernel_flash_attn_ext_q5_0_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 64>;
template [[host_name("kernel_flash_attn_ext_q5_0_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 80>;
template [[host_name("kernel_flash_attn_ext_q5_0_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 96>;
template [[host_name("kernel_flash_attn_ext_q5_0_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 112>;
template [[host_name("kernel_flash_attn_ext_q5_0_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 128>;
template [[host_name("kernel_flash_attn_ext_q5_0_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 256>;

template [[host_name("kernel_flash_attn_ext_q5_1_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 64>;
template [[host_name("kernel_flash_attn_ext_q5_1_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 80>;
template [[host_name("kernel_flash_attn_ext_q5_1_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96>;
template [[host_name("kernel_flash_attn_ext_q5_1_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 112>;
template [[host_name("kernel_flash_attn_ext_q5_1_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 256>;

template [[host_name("kernel_flash_attn_ext_q8_0_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 64>;
template [[host_name("kernel_flash_attn_ext_q8_0_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 80>;
template [[host_name("kernel_flash_attn_ext_q8_0_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 96>;
template [[host_name("kernel_flash_attn_ext_q8_0_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 112>;
template [[host_name("kernel_flash_attn_ext_q8_0_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 256>;

#undef FA_TYPES

template<
    typename q4_t,
    typename q4x4_t,
    typename k4x4_t,
    typename v4x4_t,
    typename s_t,    // attention accumulation types
    typename s4_t,
    typename s4x4_t,
    typename o4x4_t,
    typename block_q,
    short nl_k,
    void (*deq_k)(device const block_q *, short, thread k4x4_t &),
    short nl_v,
    void (*deq_v)(device const block_q *, short, thread v4x4_t &),
    short D,         // head size
    short Q  = 1,    // queries per threadgroup
    short C  = 32>   // cache items per threadgroup
kernel void kernel_flash_attn_ext_vec(
        device const  char * q,
        device const  char * k,
        device const  char * v,
        device const  char * mask,
        device       float * dst,
        constant   int32_t & ne01,
        constant   int32_t & ne02,
        constant   int32_t & ne03,
        constant  uint32_t & nb01,
        constant  uint32_t & nb02,
        constant  uint32_t & nb03,
        constant   int32_t & ne11,
        constant   int32_t & ne_12_2, // assume K and V are same shape
        constant   int32_t & ne_12_3,
        constant  uint32_t & nb_12_1,
        constant  uint32_t & nb_12_2,
        constant  uint32_t & nb_12_3,
        constant  uint32_t & nb31,
        constant   int32_t & ne1,
        constant   int32_t & ne2,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        constant     float & logit_softcap,
        threadgroup   half * shared [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint3  tpitg[[thread_position_in_threadgroup]],
        uint3    ntg[[threads_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short nsg = ntg.y; // number of simdgroups

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0];

    const short D4  = D/4;
    const short D16 = D/16;
    const short NW  = N_SIMDWIDTH;
    const short NW4 = NW/4;
    const short SH  = C; // shared memory per simdgroup in (half)

    const short SF = sizeof(s_t)/sizeof(half);

    const short T = D + SF*nsg*SH; // shared memory size per query in (half)

  //threadgroup q_t    * sq    = (threadgroup q_t    *) (shared +               0*D); // holds the query data
    threadgroup q4_t   * sq4   = (threadgroup q4_t   *) (shared +               0*D); // same as above but in half4
    threadgroup q4x4_t * sq4x4 = (threadgroup q4x4_t *) (shared +               0*D); // same as above but in half4x4
    threadgroup s_t    * ss    = (threadgroup s_t    *) (shared + SF*sgitg*SH + 1*D); // scratch buffer for attention
    threadgroup s4_t   * ss4   = (threadgroup s4_t   *) (shared + SF*sgitg*SH + 1*D); // same as above but in half4
    threadgroup s4x4_t * sr4x4 = (threadgroup s4x4_t *) (shared + SF*sgitg*D  + Q*T); // scratch buffer for the results

    // store the result for all queries in local memory in 8x8 matrices (the O matrix from the paper)
    o4x4_t lo[D16/NW4];

    // load heads from Q to shared memory
    device const float4 * q4 = (device const float4 *) ((device const char *) q + (iq1*nb01 + iq2*nb02 + iq3*nb03));

    for (short i = tiisg; i < D4; i += NW) {
        if (iq1 < ne01) {
            sq4[i] = (q4_t) q4[i];
        } else {
            sq4[i] = (q4_t) (float4) 0.0f;
        }
    }

    // zero out lo
    for (short i = 0; i < D16/NW4; i += NW4) {
        lo[i] = (o4x4_t) 0.0f;
    }

    // zero out shared memory SH
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) (float4) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        // thread indices inside the simdgroup
        const short tx = tiisg%8;
        const short ty = tiisg/8;

        // broadcast kv
        //const short rk2 = ne02/ne12;
        //const short rk3 = ne03/ne13;

        const short ikv2 = iq2/(ne02/ne_12_2);
        const short ikv3 = iq3/(ne03/ne_12_3);

        // load the queries from shared memory into local memory
        k4x4_t mq[D16/NW4];

        for (short ii = 0; ii < D16; ii += NW4) {
            mq[ii/NW4] = (k4x4_t) sq4x4[ii + tx];
        }

        // pointer to the mask
        device const half * mp = (device const half *) (mask + iq1*nb31);

        float slope = 1.0f;

        // ALiBi
        if (max_bias > 0.0f) {
            const uint32_t h = iq2;

            const float base = h < n_head_log2 ? m0 : m1;
            const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

            slope = pow(base, exp);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ic0 < ne11; ic0 += C*nsg) {
            const int ic = ic0 + C*sgitg;
            if (ic >= ne11) {
                break;
            }

            // Q*K^T
            {
                // each simdgroup processes 1 query and 4 keys
                for (short cc = 0; cc < C/4; ++cc) {
                    s_t mqk = 0.0;

                    device const block_q * pk = (device const block_q *) ((device const char *) k + ((ic + 4*cc + ty)*nb_12_1 + ikv2*nb_12_2 + ikv3*nb_12_3));

#pragma unroll
                    for (short ii = 0; ii < D16; ii += NW4) {
                        const short i = ii + tx;

                        k4x4_t mk;
                        deq_k(pk + i/nl_k, i%nl_k, mk);

                        mqk +=
                            dot(mq[ii/NW4][0], mk[0]) +
                            dot(mq[ii/NW4][1], mk[1]) +
                            dot(mq[ii/NW4][2], mk[2]) +
                            dot(mq[ii/NW4][3], mk[3]);
                    }

                    // simdgroup reduce
                    // [ 0 ..  7] -> [ 0]
                    // [ 8 .. 15] -> [ 8]
                    // [16 .. 23] -> [16]
                    // [24 .. 31] -> [24]
                  //mqk += simd_shuffle_down(mqk, 16);
                  //mqk += simd_shuffle_down(mqk,  8);
                    mqk += simd_shuffle_down(mqk,  4);
                    mqk += simd_shuffle_down(mqk,  2);
                    mqk += simd_shuffle_down(mqk,  1);

                    // mqk = mqk*scale + mask*slope
                    if (tx == 0) {
                        mqk *= scale;

                        if (logit_softcap != 0.0f) {
                            mqk = logit_softcap*precise::tanh(mqk);
                        }

                        mqk += (s_t) ((mask != q) ? ((float) mp[ic + 4*cc + ty])*slope : (float) 0.0f);

                        ss[4*cc + ty] = mqk;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                const short p = tiisg;

                const float m = M;
                const float s = ss[p];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                // the P matrix from the paper (Q rows, C columns)
                ss[p] = vs;

                // O = diag(ms)*O
#pragma unroll
                for (short ii = 0; ii < D16; ii += NW4) {
                    lo[ii/NW4] *= ms;
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
#pragma unroll
                for (short cc = 0; cc < C/4; ++cc) {
                    device const block_q * pv4 = (device const block_q *) ((device const char *) v + ((ic + 4*cc + ty)*nb_12_1 + ikv2*nb_12_2 + ikv3*nb_12_3));

                    const s4x4_t ms(ss[4*cc + ty]);

#pragma unroll
                    for (short ii = 0; ii < D16; ii += NW4) {
                        const short i = ii + tx;

                        v4x4_t mv;
                        deq_v(pv4 + i/nl_v, i%nl_v, mv);

                        lo[ii/NW4] += mv*ms;
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    // simdgroup reduce
    // [ 0,  8, 16, 24] -> [ 0]
    // [ 1,  9, 17, 25] -> [ 1]
    // [ 2, 10, 18, 26] -> [ 2]
    // [ 3, 11, 19, 27] -> [ 3]
    // [ 4, 12, 20, 28] -> [ 4]
    // [ 5, 13, 21, 29] -> [ 5]
    // [ 6, 14, 22, 30] -> [ 6]
    // [ 7, 15, 23, 31] -> [ 7]
    for (short ii = 0; ii < D16; ii += NW4) {
        lo[ii/NW4][0] += simd_shuffle_down(lo[ii/NW4][0], 16);
        lo[ii/NW4][0] += simd_shuffle_down(lo[ii/NW4][0],  8);

        lo[ii/NW4][1] += simd_shuffle_down(lo[ii/NW4][1], 16);
        lo[ii/NW4][1] += simd_shuffle_down(lo[ii/NW4][1],  8);

        lo[ii/NW4][2] += simd_shuffle_down(lo[ii/NW4][2], 16);
        lo[ii/NW4][2] += simd_shuffle_down(lo[ii/NW4][2],  8);

        lo[ii/NW4][3] += simd_shuffle_down(lo[ii/NW4][3], 16);
        lo[ii/NW4][3] += simd_shuffle_down(lo[ii/NW4][3],  8);
    }

    // store results to shared memory
    for (short i = tiisg; i < D16; i += NW4) {
        sr4x4[i] = lo[i/NW4];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce
    for (short r = nsg/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[       0];
            const float S1 = ss[r*SH + 0];

            const float M0 = ss[       1];
            const float M1 = ss[r*SH + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = (s_t) S;
                ss[1] = (s_t) M;
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            for (short i = tiisg; i < D16; i += NW) {
                sr4x4[i] = sr4x4[i]*ms0 + sr4x4[i + r*D16]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float4x4 * dst44 = (device float4x4 *) dst;

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const float S = ss[0];

        for (short i = tiisg; i < D16; i += NW) {
            dst44[(iq3*ne2*ne1 + iq2 + (iq1)*ne1)*D16 + i] = (float4x4) sr4x4[i]/S;
        }
    }
}

// NOTE: can use half instead of float precision for some extra perf
//       however, by default use F32 since the op should be mostly memory bandwidth bound

#define FA_TYPES \
           half4,  half4x4,  \
                   float4x4, \
                   float4x4, \
    float, float4, float4x4, \
                   float4x4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4x4, 1, dequantize_f16, 1, dequantize_f16, 64>) flash_attn_ext_vec_t;

template [[host_name("kernel_flash_attn_ext_vec_f16_h128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4x4,    1, dequantize_f16,  1, dequantize_f16,  128>;
template [[host_name("kernel_flash_attn_ext_vec_bf16_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4x4,  1, dequantize_bf16, 1, dequantize_bf16, 128>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 2, dequantize_q4_0, 2, dequantize_q4_0, 128>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 2, dequantize_q4_1, 2, dequantize_q4_1, 128>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 2, dequantize_q5_0, 2, dequantize_q5_0, 128>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 2, dequantize_q5_1, 2, dequantize_q5_1, 128>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 2, dequantize_q8_0, 2, dequantize_q8_0, 128>;

template [[host_name("kernel_flash_attn_ext_vec_f16_h256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4x4,    1, dequantize_f16,  1, dequantize_f16,  256>;
template [[host_name("kernel_flash_attn_ext_vec_bf16_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4x4,  1, dequantize_bf16, 1, dequantize_bf16, 256>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 2, dequantize_q4_0, 2, dequantize_q4_0, 256>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 2, dequantize_q4_1, 2, dequantize_q4_1, 256>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 2, dequantize_q5_0, 2, dequantize_q5_0, 256>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 2, dequantize_q5_1, 2, dequantize_q5_1, 256>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 2, dequantize_q8_0, 2, dequantize_q8_0, 256>;

#undef FA_TYPES

template<typename T0, typename T1>
kernel void kernel_cpy(
        device  const void * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device T1 * dst_data = (device T1 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const T0 * src = (device T0 *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = (T1) src[0];
    }
}

typedef decltype(kernel_cpy<float, float>) kernel_cpy_t;

template [[host_name("kernel_cpy_f32_f32")]]   kernel kernel_cpy_t kernel_cpy<float,  float>;
template [[host_name("kernel_cpy_f32_f16")]]   kernel kernel_cpy_t kernel_cpy<float,  half>;
#if !defined(GGML_METAL_NO_BFLOAT)
template [[host_name("kernel_cpy_f32_bf16")]]  kernel kernel_cpy_t kernel_cpy<float,  bfloat>;
#endif
template [[host_name("kernel_cpy_f16_f32")]]   kernel kernel_cpy_t kernel_cpy<half,   float>;
template [[host_name("kernel_cpy_f16_f16")]]   kernel kernel_cpy_t kernel_cpy<half,   half>;
#if !defined(GGML_METAL_NO_BFLOAT)
template [[host_name("kernel_cpy_bf16_f32")]]  kernel kernel_cpy_t kernel_cpy<bfloat, float>;
template [[host_name("kernel_cpy_bf16_bf16")]] kernel kernel_cpy_t kernel_cpy<bfloat, bfloat>;
#endif

kernel void kernel_cpy_f32_q8_0(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK8_0;

    device block_q8_0 * dst_data = (device block_q8_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK8_0; i00 < ne00; i00 += ntg.x*QK8_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = src[j];
            amax = MAX(amax, fabs(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK8_0].d = d;

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = src[j]*id;

            dst_data[i00/QK8_0].qs[j] = round(x0);
        }
    }
}

kernel void kernel_cpy_f32_q4_0(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_0;

    device block_q4_0 * dst_data = (device block_q4_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_0; i00 < ne00; i00 += ntg.x*QK4_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK4_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK4_0].d = d;

        for (int j = 0; j < QK4_0/2; ++j) {
            const float x0 = src[0       + j]*id;
            const float x1 = src[QK4_0/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            dst_data[i00/QK4_0].qs[j]  = xi0;
            dst_data[i00/QK4_0].qs[j] |= xi1 << 4;
        }
    }
}

kernel void kernel_cpy_f32_q4_1(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_1;

    device block_q4_1 * dst_data = (device block_q4_1 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_1; i00 < ne00; i00 += ntg.x*QK4_1) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < QK4_1; j++) {
            const float v = src[j];
            if (min > v) min = v;
            if (max < v) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK4_1].d = d;
        dst_data[i00/QK4_1].m = min;

        for (int j = 0; j < QK4_1/2; ++j) {
            const float x0 = (src[0       + j] - min)*id;
            const float x1 = (src[QK4_1/2 + j] - min)*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            dst_data[i00/QK4_1].qs[j]  = xi0;
            dst_data[i00/QK4_1].qs[j] |= xi1 << 4;
        }
    }
}

kernel void kernel_cpy_f32_q5_0(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK5_0;

    device block_q5_0 * dst_data = (device block_q5_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK5_0; i00 < ne00; i00 += ntg.x*QK5_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK5_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK5_0].d = d;

        uint32_t qh = 0;
        for (int j = 0; j < QK5_0/2; ++j) {
            const float x0 = src[0       + j]*id;
            const float x1 = src[QK5_0/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            dst_data[i00/QK5_0].qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
        }
        thread const uint8_t * qh8 = (thread const uint8_t *)&qh;
        for (int j = 0; j < 4; ++j) {
            dst_data[i00/QK5_0].qh[j] = qh8[j];
        }
    }
}

kernel void kernel_cpy_f32_q5_1(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK5_1;

    device block_q5_1 * dst_data = (device block_q5_1 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK5_1; i00 < ne00; i00 += ntg.x*QK5_1) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float max = src[0];
        float min = src[0];

        for (int j = 1; j < QK5_1; j++) {
            const float v = src[j];
            min = v < min ? v : min;
            max = v > max ? v : max;
        }

        const float d = (max - min) / 31;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK5_1].d = d;
        dst_data[i00/QK5_1].m = min;

        uint32_t qh = 0;
        for (int j = 0; j < QK5_1/2; ++j) {
            const float x0 = (src[0       + j] - min)*id;
            const float x1 = (src[QK5_1/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            dst_data[i00/QK5_1].qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
        }
        thread const uint8_t * qh8 = (thread const uint8_t *)&qh;
        for (int j = 0; j < 4; ++j) {
            dst_data[i00/QK5_1].qh[j] = qh8[j];
        }
    }
}

static inline int best_index_int8(int n, constant float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

kernel void kernel_cpy_f32_iq4_nl(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_NL;

    device block_iq4_nl * dst_data = (device block_iq4_nl *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_NL; i00 < ne00; i00 += ntg.x*QK4_NL) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK4_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / kvalues_iq4nl_f[0];
        const float id = d ? 1.0f/d : 0.0f;

        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            const float x0 = src[0        + j]*id;
            const float x1 = src[QK4_NL/2 + j]*id;

            const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
            const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

            dst_data[i00/QK4_NL].qs[j] = xi0 | (xi1 << 4);

            const float v0 = kvalues_iq4nl_f[xi0];
            const float v1 = kvalues_iq4nl_f[xi1];
            const float w0 = src[0        + j]*src[0        + j];
            const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
            sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
            sumq2 += w0*v0*v0 + w1*v1*v1;

        }

        dst_data[i00/QK4_NL].d = sumq2 > 0 ? sumqx/sumq2 : d;

    }
}

kernel void kernel_concat(
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne10,
    constant   int64_t & ne11,
    constant   int64_t & ne12,
    constant   int64_t & ne13,
    constant  uint64_t & nb10,
    constant  uint64_t & nb11,
    constant  uint64_t & nb12,
    constant  uint64_t & nb13,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    constant   int32_t & dim,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

    device const float * x;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (device const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            x = (device const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
        }

        device float * y = (device float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

void kernel_mul_mv_q2_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_q2_K * x = (device const block_q2_K *) ((device char *) src0 + offset0);
    device const float      * y = (device const float      *) ((device char *) src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3
    const int is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += nb01/2;
            sc += nb01;
            dh += nb01/2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q3_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_q3_K * x = (device const block_q3_K *) ((device char *) src0 + offset0);
    device const float     * yy = (device const float      *) ((device char *) src1 + offset1);

    float yl[32];

    //const uint16_t kmask1 = 0x3030;
    //const uint16_t kmask2 = 0x0f0f;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int ip  = tid/4;          // 0 or 1
    const int il  = 2*((tid%4)/2);  // 0 or 2
    const int ir  = tid%2;
    const int n   = 8;
    const int l0  = n*ir;

    // One would think that the Metal compiler would figure out that ip and il can only have
    // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it
    // with these two tales.
    //
    // Possible masks for the high bit
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

    // Possible masks for the low 2 bits
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const int shift = 2*il;
    const float    v1 = il == 0 ? 4.f : 64.f;
    const float    v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const int q_offset = 32*ip + l0;
    const int y_offset = 128*ip + 32*il + l0;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[2] = {0.f};
    float sumf2[2] = {0.f};
    for (int i = ix; i < nb; i += 4) {
        for (int l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (int row = 0; row < 2; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (int l = 0; l < n; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (int l = 0; l < n; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += nb01/2;
            h  += nb01/2;
            a  += nb01/2;
            dh += nb01/2;
        }

        y1 += 4 * QK_K;
    }

    for (int row = 0; row < 2; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }
    if (tiisg == 0) {
        for (int row = 0; row < 2; ++row) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = sumf1[row];
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q4_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    //const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int first_row = r0 * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_q4_K * x = (device const block_q4_K *) ((device char *) src0 + offset0);
    device const float      * y = (device const float      *) ((device char *) src1 + offset1);

    float yl[16];
    float yh[16];
    float sumf[N_DST]={0.f}, all_sum;

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);
                acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);
                acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);
                acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += nb01/2;
            sc += nb01/2;
            dh += nb01/2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q5_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_q5_K * x = (device const block_q5_K *) ((device char *) src0 + offset0);
    device const float     * yy = (device const float      *) ((device char *) src1 + offset1);

    float sumf[2]={0.f};

    float yl[16], yh[16];

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int iq  = tid/4;
    const int ir  = tid%4;
    const int n   = 8;

    const int l0 = n*ir;
    const int q_offset = 32*iq + l0;
    const int y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (int row = 0; row < 2; ++row) {
            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            for (int l = 0; l < n; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }
            const float dall = dh[0];
            const float dmin = dh[1];
            sumf[row] += dall * (sc8[0] * (acc1[0] +  16.f*acc2[0]) +
                                 sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                 sc8[4] * (acc1[2] +  16.f*acc2[2]) +
                                 sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += nb01;
            qh += nb01;
            dh += nb01/2;
            a  += nb01/2;
        }

        y1 += 4 * QK_K;
    }

    for (int row = 0; row < 2; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q6_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int     im = tgpig.z;

    const int row = 2 * r0 + sgitg;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =  r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_q6_K * x = (device const block_q6_K *) ((device char *) src0 + offset0);
    device const float     * yy = (device const float      *) ((device char *) src1 + offset1);

    float sumf = 0;

    const int tid  = tiisg/2;
    const int ix   = tiisg%2;
    const int ip   = tid/8;         // 0 or 1
    const int il   = tid%8;
    const int n    = 4;
    const int l0   = n*il;
    const int is   = 8*ip + l0/16;

    const int y_offset = 128*ip + l0;
    const int q_offset_l = 64*ip + l0;
    const int q_offset_h = 32*ip + l0;

    for (int i = ix; i < nb; i += 2) {

        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;

        device const float * y = yy + i * QK_K + y_offset;

        const float dall = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            sums[0] += y[l+ 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += y[l+32] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += y[l+64] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            sums[3] += y[l+96] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }

        sumf += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0) {
        dst[r1*ne0 + im*ne0*ne1 + row] = tot;
    }
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

// ======================= "True" 2-bit

void kernel_mul_mv_iq2_xxs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) ((device char *) src0 + offset0);
    device const float         * y = (device const float         *) ((device char *) src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + aux8[l]);
                const uint8_t signs = shared_signs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += nb01/2;
            q2 += nb01/2;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xxs_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq2_xs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_iq2_xs * x = (device const block_iq2_xs *) ((device char *) src0 + offset0);
    device const float        * y = (device const float        *) ((device char *) src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 512);
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq2xs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (int l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + (q2[l] & 511));
                const uint8_t signs = shared_signs[(q2[l] >> 9)];
                for (int j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (int l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + (q2[l] & 511));
                const uint8_t signs = shared_signs[(q2[l] >> 9)];
                for (int j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += nb01/2;
            q2 += nb01/2;
            sc += nb01;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq3_xxs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) ((device char *) src0 + offset0);
    device const float         * y = (device const float         *) ((device char *) src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * values = (threadgroup uint32_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(values + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(values + q3[2*l+1]);
                const uint8_t signs = shared_signs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += nb01/2;
            q3  += nb01;
            gas += nb01/2;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.5f;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq3_s_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_iq3_s * x = (device const block_iq3_s *) ((device char *) src0 + offset0);
    device const float       * y = (device const float       *) ((device char *) src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * values = (threadgroup uint32_t *)shared_values;
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? values + 256 : values;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? values + 256 : values;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh    += nb01/2;
            qs    += nb01;
            qh    += nb01;
            sc    += nb01;
            signs += nb01;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq2_s_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_iq2_s * x = (device const block_iq2_s *) ((device char *) src0 + offset0);
    device const float       * y = (device const float       *) ((device char *) src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    //threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    //{
    //    int nval = 32;
    //    int pos  = (32*sgitg + tiisg)*nval;
    //    for (int i = 0; i < nval; ++i) values[pos + i] = iq2s_grid[pos + i];
    //    threadgroup_barrier(mem_flags::mem_threadgroup);
    //}

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >>  4));

            float2 sum = {0};
            for (int l = 0; l < 2; ++l) {
                //const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(values + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                //const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(values + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sum[0] += yl[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh    += nb01/2;
            qs    += nb01;
            qh    += nb01;
            sc    += nb01;
            signs += nb01;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq1_s_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_value,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_iq1_s * x = (device const block_iq1_s *) ((device char *) src0 + offset0);
    device const float       * y = (device const float       *) ((device char *) src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        float sumy = 0;
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint16_t * qh = xr->qh + ib;
        device const half     * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));

            float sum = 0;
            for (int j = 0; j < 4; ++j) {
                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += nb01/2;
            qs += nb01;
            qh += nb01/2;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq1_m_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_value,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_iq1_m * x = (device const block_iq1_m *) ((device char *) src0 + offset0);
    device const float       * y = (device const float       *) ((device char *) src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        float4 sumy = {0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint8_t  * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (int row = 0; row < N_DST; row++) {
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f};
            for (int j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

            sc += nb01/2;
            qs += nb01;
            qh += nb01;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq4_nl_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK4_NL;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_iq4_nl * x = (device const block_iq4_nl *) ((device char *) src0 + offset0);
    device const float        * y = (device const float        *) ((device char *) src1 + offset1);

    const int ix = tiisg/2;  // 0...15
    const int it = tiisg%2;  // 0 or 1

    shared_values[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK4_NL + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < nb; ib += 16) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        for (int row = 0; row < 2 && first_row + row < ne01; ++row) {

            device const block_iq4_nl & xb = x[row*nb + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

        }

        yb += 16 * QK4_NL;
    }

    for (int row = 0; row < 2 && first_row + row < ne01; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq4_xs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    const uint offset1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    device const block_iq4_xs * x = (device const block_iq4_xs *) ((device char *) src0 + offset0);
    device const float        * y = (device const float        *) ((device char *) src1 + offset1);

    const int ix = tiisg/16;  // 0 or 1
    const int it = tiisg%16;  // 0...15
    const int ib = it/2;
    const int il = it%2;

    shared_values[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        for (int row = 0; row < 2; ++row) {

            device const block_iq4_xs & xb = x[row*nb + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[1] & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

        }

        yb += 2 * QK_K;
    }

    for (int row = 0; row < 2; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb01, nb02, nb03, ne10, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_get_rows_q(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int64_t ind = tiitg; ind < ne00/16; ind += tptg.x) {
        float4x4 temp;
        dequantize_func(((device const block_q *) ((const device char *) src0 + r*nb01 + i02*nb02)) + ind/nl, ind%nl, temp);
        *(((device float4x4 *) ((device char *) dst + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

template<typename T>
kernel void kernel_get_rows_f(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((      device float *) ((      device char *)  dst + i11*nb2  + i10*nb1))[ind] =
        ((const device T     *) ((const device char *) src0 + i02*nb02 +  r*nb01))[ind];
    }
}

kernel void kernel_get_rows_i32(
        device const  void * src0,
        device const  void * src1,
        device     int32_t * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((      device int32_t *) ((      device char *) dst  + i11*nb2 + i10*nb1))[ind] =
        ((const device int32_t *) ((const device char *) src0 + i02*nb02 + r*nb01))[ind];
    }
}


#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// each block_q contains 16*nl weights
template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_mul_mm(device const  uchar * src0,
                          device const  uchar * src1,
                          device        float * dst,
                          constant    int64_t & ne00,
                          constant    int64_t & ne02,
                          constant   uint64_t & nb01,
                          constant   uint64_t & nb02,
                          constant   uint64_t & nb03,
                          constant    int64_t & ne12,
                          constant   uint64_t & nb10,
                          constant   uint64_t & nb11,
                          constant   uint64_t & nb12,
                          constant   uint64_t & nb13,
                          constant    int64_t & ne0,
                          constant    int64_t & ne1,
                          constant       uint & r2,
                          constant       uint & r3,
                          threadgroup   uchar * shared_memory [[threadgroup(0)]],
                          uint3                 tgpig[[threadgroup_position_in_grid]],
                          uint                  tiitg[[thread_index_in_threadgroup]],
                          uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup T     * sa = (threadgroup T     *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;
    const uint im = tgpig.z;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_T8x8     ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    uint   offset0 = (i12/r2)*nb02 + (i13/r3)*nb03;
    ushort offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb13 * i13
        + nb12 * i12
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        T4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup T     * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        #pragma unroll(4)
        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i],lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i],lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            #pragma unroll(8)
            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= ne0 && (r1 + 1) * BLOCK_SIZE_N <= ne1) {
        device float * C = dst + (BLOCK_SIZE_M * r0 + 32 * (sgitg &  1)) \
                               + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * ne0 + im*ne1*ne0;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = dst + (BLOCK_SIZE_M * r0) + (BLOCK_SIZE_N * r1) * ne0 + im*ne1*ne0;
        if (sgitg == 0) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                    *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}

// same as kernel_mul_mm_impl, but src1 and dst are accessed via indices stored in rowids
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
void kernel_mul_mm_id_impl(
        device const  uchar * src0,
        device const  uchar * src1,
        threadgroup ushort2 * rowids,
        device        float * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne02,
        constant   uint64_t & nb01,
        constant   uint64_t & nb02,
        constant    int64_t & ne11,
        constant    int64_t & ne12,
        constant   uint64_t & nb10,
        constant   uint64_t & nb11,
        constant   uint64_t & nb12,
        constant    int64_t & ne0,
                    int64_t   ne1,
                    int64_t   ne0ne1,
        threadgroup   uchar * shared_memory,
        uint3                 tgpig[[threadgroup_position_in_grid]],
        uint                  tiitg[[thread_index_in_threadgroup]],
        uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;

    if (r1 * BLOCK_SIZE_N >= ne1) return;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    short il = (tiitg % THREAD_PER_ROW);

    ushort offset1 = il/nl;

    threadgroup const auto & id = rowids[r1 * BLOCK_SIZE_N + thread_col];

    device const block_q * x = (device const block_q *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb12 * id[1]
        + nb11 * (id[0] % ne11)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        half4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = dst + (BLOCK_SIZE_M * r0);
        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                threadgroup const auto & jid = rowids[r1 * BLOCK_SIZE_N + j];
                int joff =  jid[0] * ne0 + jid[1] * ne0ne1;
                for (int i = 0; i < n_rows; i++) {
                    *(C + i + joff) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm_id(
        device const   uchar * src0s,
        device const   uchar * src1,
        device         float * dst,
        device const   uchar * ids,
        constant     int64_t & nei0,
        constant     int64_t & nei1,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne02,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        threadgroup    uchar * shared_memory [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {

    const int32_t i02 = tgpig.z;
    tgpig.z = 0;

    device const uchar * src0 = src0s + i02*nb02;

    // row indices
    threadgroup ushort2 * rowids = (threadgroup ushort2 *)(shared_memory + 8192);

    // TODO: parallelize this loop
    int64_t _ne1 = 0;
    for (ushort ii1 = 0; ii1 < nei1; ii1++) {
        for (ushort ii0 = 0; ii0 < nei0; ii0++) {
            int32_t id = ((device int32_t *) (ids + ii1*nbi1))[ii0];
            if (id == i02) {
                //if (tiitg == 0) {
                    rowids[_ne1] = ushort2(ii0, ii1);
                //}
                _ne1++;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    kernel_mul_mm_id_impl<block_q, nl, dequantize_func>(
        src0,
        src1,
        rowids,
        dst,
        ne00,
        ne02,
        nb01,
        nb02,
        ne11,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        _ne1,
        ne0*ne1,
        shared_memory,
        tgpig,
        tiitg,
        sgitg);
}

#define QK_NL 16

//
// get rows
//

typedef decltype(kernel_get_rows_f<float>) get_rows_f_t;

template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_f_t kernel_get_rows_f<float>;
template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_f_t kernel_get_rows_f<half>;
#if !defined(GGML_METAL_NO_BFLOAT)
template [[host_name("kernel_get_rows_bf16")]] kernel get_rows_f_t kernel_get_rows_f<bfloat>;
#endif

typedef decltype(kernel_get_rows_q<block_q4_0, 2, dequantize_q4_0>) get_rows_q_t;

template [[host_name("kernel_get_rows_q4_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_0,    2, dequantize_q4_0>;
template [[host_name("kernel_get_rows_q4_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_1,    2, dequantize_q4_1>;
template [[host_name("kernel_get_rows_q5_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_0,    2, dequantize_q5_0>;
template [[host_name("kernel_get_rows_q5_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_1,    2, dequantize_q5_1>;
template [[host_name("kernel_get_rows_q8_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q8_0,    2, dequantize_q8_0>;
template [[host_name("kernel_get_rows_q2_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_get_rows_q3_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_get_rows_q4_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_get_rows_q5_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_get_rows_q6_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_get_rows_iq2_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_get_rows_iq2_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_get_rows_iq3_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_get_rows_iq3_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_get_rows_iq2_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_get_rows_iq1_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_get_rows_iq1_m")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_get_rows_iq4_nl")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_get_rows_iq4_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, float4x4, 1, dequantize_f32>) mat_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16>;
#if !defined(GGML_METAL_NO_BFLOAT)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mat_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16>;
#endif
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<float4x4, 1, dequantize_f32>) mat_mm_id_t;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mat_mm_id_t kernel_mul_mm_id<float4x4,      1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mat_mm_id_t kernel_mul_mm_id<half4x4,       1,     dequantize_f16>;
#if !defined(GGML_METAL_NO_BFLOAT)
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<bfloat4x4,     1,     dequantize_bf16>;
#endif
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q4_0,    2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q4_1,    2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q5_0,    2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q5_1,    2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q8_0,    2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// matrix-vector multiplication
//

typedef void (kernel_mul_mv_impl_t)(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb00,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne11,
                   int64_t   ne12,
                  uint64_t   nb10,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3     tgpig,
                   uint      tiisg);

typedef void (kernel_mul_mv2_impl_t)(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb01,
                  uint64_t   nb02,
                  uint64_t   nb03,
                   int64_t   ne10,
                   int64_t   ne12,
                  uint64_t   nb11,
                  uint64_t   nb12,
                  uint64_t   nb13,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg);

template<kernel_mul_mv_impl_t impl_fn>
void mmv_fn(
        device const    char * src0,
        device const    char * src1,
        device         float * dst,
                     int64_t   ne00,
                     int64_t   ne01,
                     int64_t   ne02,
                    uint64_t   nb00,
                    uint64_t   nb01,
                    uint64_t   nb02,
                    uint64_t   nb03,
                     int64_t   ne10,
                     int64_t   ne11,
                     int64_t   ne12,
                     int64_t   ne13,
                    uint64_t   nb10,
                    uint64_t   nb11,
                    uint64_t   nb12,
                    uint64_t   nb13,
                     int64_t   ne0,
                     int64_t   ne1,
                    uint64_t   nb1,
                        uint   r2,
                        uint   r3,
        threadgroup int8_t   * shared_values,
        uint3                  tgpig,
        uint                   tiitg,
        uint                   tiisg,
        uint                   sgitg) {
    impl_fn(src0,src1,dst,ne00,ne01,ne02,nb00,nb01,nb02,nb03,ne10,ne11,ne12,nb10,nb11,nb12,nb13,ne0,ne1,r2,r3,tgpig,tiisg);
}

template<kernel_mul_mv2_impl_t impl_fn>
void mmv_fn(
        device const    char * src0,
        device const    char * src1,
        device         float * dst,
                     int64_t   ne00,
                     int64_t   ne01,
                     int64_t   ne02,
                    uint64_t   nb00,
                    uint64_t   nb01,
                    uint64_t   nb02,
                    uint64_t   nb03,
                     int64_t   ne10,
                     int64_t   ne11,
                     int64_t   ne12,
                     int64_t   ne13,
                    uint64_t   nb10,
                    uint64_t   nb11,
                    uint64_t   nb12,
                    uint64_t   nb13,
                     int64_t   ne0,
                     int64_t   ne1,
                    uint64_t   nb1,
                        uint   r2,
                        uint   r3,
        threadgroup int8_t   * shared_values,
        uint3                  tgpig,
        uint                   tiitg,
        uint                   tiisg,
        uint                   sgitg) {
    impl_fn(src0,(const device float *)src1,dst,ne00,ne01,ne02,nb01,nb02,nb03,ne10,ne12,nb11,nb12,nb13,ne0,ne1,r2,r3,shared_values,tgpig,tiisg,sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_impl<half, half4, half, half4>>) mul_mv_impl_fn_t;

template<mul_mv_impl_fn_t impl_fn>
kernel void kernel_mul_mv_id(
        device const    char * src0s,
        device const    char * src1,
        device         float * dst,
        device const    char * ids,
        constant     int64_t & nei0,
        constant     int64_t & nei1,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        threadgroup int8_t   * shared_values [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/nei0;
    const int idx = tgpig.z%nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*nbi1))[idx];

    const int64_t i11 = idx % ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*nb02;
    device const char * src1_cur = src1  + i11*nb11 + i12*nb12;
    device      float *  dst_cur = dst   + i1*ne0   + i2*ne1*ne0;

    impl_fn(
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        /* ne00 */ ne00,
        /* ne01 */ ne01,
        /* ne02 */ 1, // ne02,
        /* nb00 */ nb00,
        /* nb01 */ nb01,
        /* nb02 */ nb02,
        /* nb03 */ nb02, // ne02 == 1
        /* ne10 */ ne10,
        /* ne11 */ 1, // ne11,
        /* ne12 */ 1, // ne12,
        /* ne13 */ 1, // ne13,
        /* nb10 */ nb10,
        /* nb11 */ nb11,
        /* nb12 */ nb12,
        /* ne13 */ nb12, // ne12 == 1
        /* ne0  */ ne0,
        /* ne1  */ 1, // ne1,
        /* nb1  */ nb1,
        /* r2   */ 1,
        /* r3   */ 1,
        shared_values,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<float, float4, float, float4>>>) kernel_mul_mv_id_t;

template [[host_name("kernel_mul_mv_id_f32_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<float, float4, float, float4>>>;
template [[host_name("kernel_mul_mv_id_f16_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<half, half4, float, float4>>>;
#if !defined(GGML_METAL_NO_BFLOAT)
template [[host_name("kernel_mul_mv_id_bf16_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<bfloat, bfloat4, float, float4>>>;
#endif
template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q4_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q4_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q5_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q3_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq1_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq1_m_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_nl_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_nl_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl>>;

kernel void kernel_pool_2d_max_f32(
        device  const float * src0,
        device        float * dst,
        constant    int32_t & k0,
        constant    int32_t & k1,
        constant    int32_t & s0,
        constant    int32_t & s1,
        constant    int32_t & p0,
        constant    int32_t & p1,
        constant    int64_t & IH,
        constant    int64_t & IW,
        constant    int64_t & OH,
        constant    int64_t & OW,
        constant    int64_t & parallel_elements,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= parallel_elements) {
        return;
    }

    const int idx = gid;
    const int I_HW = IH * IW;
    const int O_HW = OH * OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / OW;
    const int cur_ow = idx % O_HW % OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * s1 - p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(IH, start_h + k1);
    const int start_w = cur_ow * s0 - p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(IW, start_w + k0);

    float res = -INFINITY;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            res = MAX(res, i_ptr[i * IW + j]);
        }
    }

    o_ptr[cur_oh * OW + cur_ow] = res;
}

kernel void kernel_pool_2d_avg_f32(
        device  const float * src0,
        device        float * dst,
        constant    int32_t & k0,
        constant    int32_t & k1,
        constant    int32_t & s0,
        constant    int32_t & s1,
        constant    int32_t & p0,
        constant    int32_t & p1,
        constant    int64_t & IH,
        constant    int64_t & IW,
        constant    int64_t & OH,
        constant    int64_t & OW,
        constant    int64_t & parallel_elements,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= parallel_elements) {
        return;
    }

    const int idx = gid;
    const int I_HW = IH * IW;
    const int O_HW = OH * OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / OW;
    const int cur_ow = idx % O_HW % OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * s1 - p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(IH, start_h + k1);
    const int start_w = cur_ow * s0 - p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(IW, start_w + k0);
    // const float scale = 1. / ((eh - bh) * (ew - bw));
    const float scale = 1. / (k0 * k1);

    float res = 0;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            float cur = i_ptr[i * IW + j];
            res += cur * scale;
        }
    }

    o_ptr[cur_oh * OW + cur_ow] = res;
}
