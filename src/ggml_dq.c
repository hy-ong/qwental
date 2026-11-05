// GGUF 區塊的 CPU 解碼，照 ggml（MIT）的 dequantize_row_*：同樣的運算順序，結果與 gguf 套件逐值相同
#include "ggml_dq.h"
#include <stdint.h>
#include <string.h>

static float h2f(uint16_t h) {
    _Float16 v;
    memcpy(&v, &h, 2);
    return (float)v;
}

static float bf2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

static const int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// Q4_K / Q5_K 的 6-bit scale 與 min
static void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void dq_q8_0(const uint8_t *b, float *y) { // 34 B：d、qs[32]
    const float d = h2f(rd16(b));
    for (int j = 0; j < 32; j++) y[j] = (float)(int8_t)b[2 + j] * d;
}

static void dq_q4_k(const uint8_t *b, float *y) { // 144 B：d、dmin、scales[12]、qs[128]
    const float d = h2f(rd16(b)), mn = h2f(rd16(b + 2));
    const uint8_t *sc = b + 4, *q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is, sc, &s, &m);
        const float d1 = d * s, m1 = mn * m;
        scale_min_k4(is + 1, sc, &s, &m);
        const float d2 = d * s, m2 = mn * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

static void dq_q5_k(const uint8_t *b, float *y) { // 176 B：d、dmin、scales[12]、qh[32]、qs[128]
    const float d = h2f(rd16(b)), mn = h2f(rd16(b + 2));
    const uint8_t *sc = b + 4, *qh = b + 16, *ql = b + 48;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is, sc, &s, &m);
        const float d1 = d * s, m1 = mn * m;
        scale_min_k4(is + 1, sc, &s, &m);
        const float d2 = d * s, m2 = mn * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

static void dq_q6_k(const uint8_t *b, float *y) { // 210 B：ql[128]、qh[64]、scales[16]（int8）、d
    const float d = h2f(rd16(b + 208));
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            const int q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static void dq_iq4_nl(const uint8_t *b, float *y) { // 18 B：d、qs[16]
    const float d = h2f(rd16(b));
    const uint8_t *qs = b + 2;
    for (int j = 0; j < 16; j++) {
        y[j] = d * kv_iq4nl[qs[j] & 0xF];
        y[j + 16] = d * kv_iq4nl[qs[j] >> 4];
    }
}

static void dq_iq4_xs(const uint8_t *b, float *y) { // 136 B：d、scales_h、scales_l[4]、qs[128]
    const float d = h2f(rd16(b));
    const uint16_t sh = rd16(b + 2);
    const uint8_t *sl = b + 4, *qs = b + 8;
    for (int ib = 0; ib < 8; ib++) {
        const int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; j++) {
            y[j] = dl * kv_iq4nl[qs[j] & 0xF];
            y[j + 16] = dl * kv_iq4nl[qs[j] >> 4];
        }
        y += 32;
        qs += 16;
    }
}

bool qw_ggml_dq_ok(int type) {
    switch (type) {
    case 0: case 1: case 8: case 12: case 13: case 14: case 20: case 23: case 30: return true;
    default: return false;
    }
}

bool qw_ggml_dq_row(int type, const void *src, float *dst, int K) {
    const uint8_t *s = src;
    switch (type) {
    case 0: memcpy(dst, src, (size_t)K * 4); return true;
    case 1: for (int i = 0; i < K; i++) dst[i] = h2f(rd16(s + 2 * i)); return true;
    case 30: for (int i = 0; i < K; i++) dst[i] = bf2f(rd16(s + 2 * i)); return true;
    case 8: for (int i = 0; i < K / 32; i++) dq_q8_0(s + 34 * i, dst + 32 * i); return true;
    case 20: for (int i = 0; i < K / 32; i++) dq_iq4_nl(s + 18 * i, dst + 32 * i); return true;
    case 12: for (int i = 0; i < K / 256; i++) dq_q4_k(s + 144 * i, dst + 256 * i); return true;
    case 13: for (int i = 0; i < K / 256; i++) dq_q5_k(s + 176 * i, dst + 256 * i); return true;
    case 14: for (int i = 0; i < K / 256; i++) dq_q6_k(s + 210 * i, dst + 256 * i); return true;
    case 23: for (int i = 0; i < K / 256; i++) dq_iq4_xs(s + 136 * i, dst + 256 * i); return true;
    default: return false;
    }
}
