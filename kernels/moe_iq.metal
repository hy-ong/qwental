// MoE（decode 與驗證）的 routed expert 用 GGUF 的 IQ / Q2_0 區塊（ISTA-DASLab GSQ-RCO IQ3_S 的 expert，位元組原樣）。
// 解碼對照 llama.cpp ggml-quants.c 的 dequantize_row_*（MIT）。共用 expert 與 router 仍是 MLX 8-bit。
// 主程式注入：E、TOPK、D、I、FGU（gate / up 的 ggml type）、FD（down 的 type）、RPS；驗證版另有 NB；kernel 名稱由主程式改名
// 前面會接上 iq_tables.metal
#include <metal_stdlib>
using namespace metal;

#define GS 64
#define CHUNK 16

// ---------------------------------------------------------------- 區塊格式
// type：(每塊值數, 每塊位元組)
#define QB_42 64
#define BB_42 18
#define QB_20 32
#define BB_20 18
#define QB_22 256
#define BB_22 82
#define QB_18 256
#define BB_18 98
#define QB_21 256
#define BB_21 110
#define QB_23 256
#define BB_23 136
#define QB_30 32 // BF16（dq32 以 32 個值為一段）
#define BB_30 64
#define QB_7 32 // Q5_1
#define BB_7 24
#define QB_8 32 // Q8_0
#define BB_8 34
#define QB_12 256 // Q4_K
#define BB_12 144
#define QB_13 256 // Q5_K
#define BB_13 176
#define QB_14 256 // Q6_K
#define BB_14 210
#define QB_(F) QB_##F
#define BB_(F) BB_##F
#define QB(F) QB_(F)
#define BB(F) BB_(F)

inline float ldh(device const uchar *p) { return float(as_type<half>(ushort(p[0] | (p[1] << 8)))); }

// 一列的第 sub 段（32 個值）→ y[32]
template <int F> inline void dq32(device const uchar *row, uint sub, thread float *y);

template <> inline void dq32<42>(device const uchar *row, uint sub, thread float *y) { // Q2_0：00 = −1、01 = 0、10 = 1、11 = 2
    device const uchar *b = row + (sub / 2) * 18;
    const float d = ldh(b);
    device const uchar *qs = b + 2 + (sub % 2) * 8;
    for (uint i = 0; i < 8; i++) {
        const uint q = qs[i];
        y[4 * i] = d * float(int(q & 3) - 1);
        y[4 * i + 1] = d * float(int((q >> 2) & 3) - 1);
        y[4 * i + 2] = d * float(int((q >> 4) & 3) - 1);
        y[4 * i + 3] = d * float(int(q >> 6) - 1);
    }
}

template <> inline void dq32<20>(device const uchar *row, uint sub, thread float *y) { // IQ4_NL
    device const uchar *b = row + sub * 18;
    const float d = ldh(b);
    for (uint j = 0; j < 16; j++) {
        const uint q = b[2 + j];
        y[j] = d * float(kvalues_iq4nl[q & 15]);
        y[j + 16] = d * float(kvalues_iq4nl[q >> 4]);
    }
}

template <> inline void dq32<22>(device const uchar *row, uint sub, thread float *y) { // IQ2_S
    device const uchar *b = row + (sub / 8) * 82;
    const uint ib = sub % 8;
    const float d = ldh(b);
    device const uchar *qs = b + 2 + 4 * ib, *signs = b + 2 + 32 + 4 * ib;
    const uint qh = b[2 + 64 + ib], sc = b[2 + 64 + 8 + ib];
    const float db0 = d * (0.5f + float(sc & 15)) * 0.25f, db1 = d * (0.5f + float(sc >> 4)) * 0.25f;
    for (uint l = 0; l < 4; l++) {
        const float dl = l < 2 ? db0 : db1;
        const ulong g = iq2s_grid[qs[l] | ((qh << (8 - 2 * l)) & 0x300)];
        const uint s = signs[l];
        for (uint j = 0; j < 8; j++) y[8 * l + j] = dl * float((g >> (8 * j)) & 255) * ((s & kmask_iq2xs[j]) ? -1.0f : 1.0f);
    }
}

template <> inline void dq32<18>(device const uchar *row, uint sub, thread float *y) { // IQ3_XXS
    device const uchar *b = row + (sub / 8) * 98;
    const uint ib = sub % 8;
    const float d = ldh(b);
    device const uchar *qs = b + 2 + 8 * ib, *ss = b + 2 + 64 + 4 * ib;
    const uint aux = uint(ss[0]) | (uint(ss[1]) << 8) | (uint(ss[2]) << 16) | (uint(ss[3]) << 24);
    const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
    for (uint l = 0; l < 4; l++) {
        const uint s = ksigns_iq2xs[(aux >> (7 * l)) & 127];
        const uint g1 = iq3xxs_grid[qs[2 * l]], g2 = iq3xxs_grid[qs[2 * l + 1]];
        for (uint j = 0; j < 4; j++) {
            y[8 * l + j] = db * float((g1 >> (8 * j)) & 255) * ((s & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            y[8 * l + 4 + j] = db * float((g2 >> (8 * j)) & 255) * ((s & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
        }
    }
}

template <> inline void dq32<21>(device const uchar *row, uint sub, thread float *y) { // IQ3_S
    device const uchar *b = row + (sub / 8) * 110;
    const uint ib = sub % 8;
    const float d = ldh(b);
    device const uchar *qs = b + 2 + 8 * ib, *signs = b + 2 + 64 + 8 + 4 * ib;
    const uint qh = b[2 + 64 + ib];
    const float db = d * float(1 + 2 * ((b[2 + 64 + 8 + 32 + ib / 2] >> (4 * (ib % 2))) & 15));
    for (uint l = 0; l < 4; l++) {
        const uint g1 = iq3s_grid[qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
        const uint g2 = iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
        const uint s = signs[l];
        for (uint j = 0; j < 4; j++) {
            y[8 * l + j] = db * float((g1 >> (8 * j)) & 255) * ((s & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            y[8 * l + 4 + j] = db * float((g2 >> (8 * j)) & 255) * ((s & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
        }
    }
}

template <> inline void dq32<23>(device const uchar *row, uint sub, thread float *y) { // IQ4_XS
    device const uchar *b = row + (sub / 8) * 136;
    const uint ib = sub % 8;
    const float d = ldh(b);
    const uint sh = uint(b[2]) | (uint(b[3]) << 8);
    const int ls = int(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4));
    const float dl = d * float(ls - 32);
    device const uchar *qs = b + 8 + 16 * ib;
    for (uint j = 0; j < 16; j++) {
        const uint q = qs[j];
        y[j] = dl * float(kvalues_iq4nl[q & 15]);
        y[j + 16] = dl * float(kvalues_iq4nl[q >> 4]);
    }
}

// ---- K-quant 與 Q8_0（dense、shared expert、lm_head 用；對照 ggml-quants.c 的 dequantize_row_q{4,5,6}_K / q8_0）
// Q4_K / Q5_K 的第 j 段（j = 0…7）scale 與 min：get_scale_min_k4
inline void kq_scale_min(device const uchar *sc, uint j, thread float &s, thread float &m) {
    if (j < 4) {
        s = float(sc[j] & 63);
        m = float(sc[j + 4] & 63);
    } else {
        s = float((sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4));
        m = float((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    }
}

template <> inline void dq32<12>(device const uchar *row, uint sub, thread float *y) { // Q4_K：d, dmin, scales[12], qs[128]
    device const uchar *b = row + (sub / 8) * 144;
    const uint j = sub % 8;
    const float d = ldh(b), dmin = ldh(b + 2);
    float sc, mn;
    kq_scale_min(b + 4, j, sc, mn);
    const float dl = d * sc, ml = dmin * mn;
    device const uchar *qs = b + 16 + 32 * (j / 2);
    const uint sh = 4 * (j % 2);
    for (uint l = 0; l < 32; l++) y[l] = dl * float((qs[l] >> sh) & 15) - ml;
}

template <> inline void dq32<13>(device const uchar *row, uint sub, thread float *y) { // Q5_K：d, dmin, scales[12], qh[32], qs[128]
    device const uchar *b = row + (sub / 8) * 176;
    const uint j = sub % 8;
    const float d = ldh(b), dmin = ldh(b + 2);
    float sc, mn;
    kq_scale_min(b + 4, j, sc, mn);
    const float dl = d * sc, ml = dmin * mn;
    device const uchar *qh = b + 16, *qs = b + 48 + 32 * (j / 2);
    const uint sh = 4 * (j % 2);
    for (uint l = 0; l < 32; l++) y[l] = dl * float(((qs[l] >> sh) & 15) | (((qh[l] >> j) & 1) << 4)) - ml;
}

template <> inline void dq32<14>(device const uchar *row, uint sub, thread float *y) { // Q6_K：ql[128], qh[64], scales[16]（int8）, d
    device const uchar *b = row + (sub / 8) * 210;
    const uint s8 = sub % 8, n = s8 / 4, r = s8 % 4; // 第 n 個 128、第 r 個 32
    const float d = ldh(b + 208);
    device const uchar *ql = b + 64 * n + 32 * (r % 2), *qh = b + 128 + 32 * n;
    device const char *sc = (device const char *)(b + 192 + 8 * n);
    const uint lsh = 4 * (r / 2), hsh = 2 * r;
    for (uint l = 0; l < 32; l++) {
        const int q = int(((ql[l] >> lsh) & 15) | (((qh[l] >> hsh) & 3) << 4)) - 32;
        y[l] = d * float(sc[l / 16 + 2 * r]) * float(q);
    }
}

template <> inline void dq32<30>(device const uchar *row, uint sub, thread float *y) { // BF16（GGUF 的未量化矩陣）：每段 64 位元組
    device const bfloat *b = (device const bfloat *)(row + sub * 64);
    for (uint l = 0; l < 32; l++) y[l] = float(b[l]);
}

template <> inline void dq32<7>(device const uchar *row, uint sub, thread float *y) { // Q5_1：d, m, qh[4]（32 個第 5 位元）, qs[16]
    device const uchar *b = row + sub * 24;
    const float d = ldh(b), m = ldh(b + 2);
    const uint qh = uint(b[4]) | (uint(b[5]) << 8) | (uint(b[6]) << 16) | (uint(b[7]) << 24);
    device const uchar *qs = b + 8;
    for (uint j = 0; j < 16; j++) {
        y[j] = d * float((qs[j] & 15) | (((qh >> j) & 1) << 4)) + m;
        y[j + 16] = d * float((qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) + m;
    }
}

template <> inline void dq32<8>(device const uchar *row, uint sub, thread float *y) { // Q8_0：d, qs[32]（int8）
    device const uchar *b = row + sub * 34;
    const float d = ldh(b);
    device const char *qs = (device const char *)(b + 2);
    for (uint l = 0; l < 32; l++) y[l] = d * float(qs[l]);
}

// 驗證用：W 的 rows 列（每列 K 個值）全部解碼成 float。grid：rows × 32（每個 lane 負責 lane、lane + 32… 段）
#ifdef DQ_TEST
kernel void dq_test(device const uchar *W [[buffer(0)]], device float *y [[buffer(1)]], constant uint2 &a [[buffer(2)]], // (K, 每列位元組)
                    uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    float v[32];
    for (uint s = lane; s < a.x / 32; s += 32) {
        dq32<DQ_TEST>(W + (ulong)r * a.y, s, v);
        for (uint i = 0; i < 32; i++) y[(ulong)r * a.x + 32 * s + i] = v[i];
    }
}
#endif

#ifndef IQ_DQ_ONLY // prefill 的 qmm 只需要上面的解碼函式
#define RBGU (D / QB(FGU) * BB(FGU)) // gate / up 每列位元組
#define RBD (I / QB(FD) * BB(FD))    // down 每列位元組

// 一列（K 個值、位元組 row）與 x 的部分內積（lane 負責第 lane、lane + 32… 段）
template <int F, int K> inline float iq_dot(device const uchar *row, device const half *x, uint lane) {
    float acc = 0;
    for (uint s = lane; s < K / 32; s += 32) {
        float w[32];
        dq32<F>(row, s, w);
        device const half4 *xp = (device const half4 *)(x + s * 32);
        for (uint q = 0; q < 8; q++) {
            const float4 h = float4(xp[q]);
            acc += w[4 * q] * h.x + w[4 * q + 1] * h.y + w[4 * q + 2] * h.z + w[4 * q + 3] * h.w;
        }
    }
    return acc;
}

template <int F, int K> inline void iq_dot2(device const uchar *ra, device const uchar *rb, device const half *x, uint lane,
                                            thread float &oa, thread float &ob) {
    float a = 0, c = 0;
    for (uint s = lane; s < K / 32; s += 32) {
        float wa[32], wb[32];
        dq32<F>(ra, s, wa);
        dq32<F>(rb, s, wb);
        device const half4 *xp = (device const half4 *)(x + s * 32);
        for (uint q = 0; q < 8; q++) {
            const float4 h = float4(xp[q]);
            a += wa[4 * q] * h.x + wa[4 * q + 1] * h.y + wa[4 * q + 2] * h.z + wa[4 * q + 3] * h.w;
            c += wb[4 * q] * h.x + wb[4 * q + 1] * h.y + wb[4 * q + 2] * h.z + wb[4 * q + 3] * h.w;
        }
    }
    oa = a;
    ob = c;
}

// MLX 8-bit 列的內積（共用 expert；同 moe_decode.metal 的 qdot<8>）
template <int K> inline float q8_dot(device const uint *W, device const bfloat *S, device const bfloat *Z, ulong row, device const half *x,
                                     uint lane) {
    constexpr uint NCH = K / CHUNK, ROWW = K * 8 / 32, NG = K / GS;
    float acc = 0;
    for (uint c = lane; c < NCH; c += 32) {
        device const half4 *xp = (device const half4 *)(x + c * CHUNK);
        float xv[CHUNK], xs = 0;
        for (int j = 0; j < 4; j++) {
            float4 h = float4(xp[j]);
            xv[4 * j] = h.x; xv[4 * j + 1] = h.y; xv[4 * j + 2] = h.z; xv[4 * j + 3] = h.w;
            xs += h.x + h.y + h.z + h.w;
        }
        const uint2 wl = ((device const uint2 *)(W + row * ROWW))[2 * c], wh = ((device const uint2 *)(W + row * ROWW))[2 * c + 1];
        const uint4 w = uint4(wl, wh);
        float d = 0;
        for (int q = 0; q < 4; q++)
            for (int n = 0; n < 4; n++) d += float((w[q] >> (8 * n)) & 255u) * xv[4 * q + n];
        const uint g = c / (GS / CHUNK);
        acc += float(S[row * NG + g]) * d + float(Z[row * NG + g]) * xs;
    }
    return acc;
}

// 由 SIMD group 0 選前 TOPK 並算權重（同 moe_decode.metal 的 select_topk）
inline void iq_select(device const float *logits, threadgroup uint *ids_out, threadgroup float *w_out, uint sg, uint lane) {
    if (sg == 0) {
        constexpr uint PER = E / 32;
        float v[PER];
        for (uint j = 0; j < PER; j++) v[j] = logits[lane * PER + j];
        float picked[TOPK];
        uint ids[TOPK];
        for (uint k = 0; k < TOPK; k++) {
            float best = -INFINITY;
            uint bi = 0;
            for (uint j = 0; j < PER; j++)
                if (v[j] > best) { best = v[j]; bi = lane * PER + j; }
            const float m = simd_max(best);
            const uint id = simd_min(best == m ? bi : 0xffffffffu);
            picked[k] = m;
            ids[k] = id;
            if (id / PER == lane) v[id % PER] = -INFINITY;
        }
        if (lane == 0) {
            float mx = picked[0], s = 0;
            for (uint k = 0; k < TOPK; k++) s += exp(picked[k] - mx);
            for (uint k = 0; k < TOPK; k++) {
                ids_out[k] = ids[k];
                w_out[k] = exp(picked[k] - mx) / s;
            }
            w_out[TOPK] = 1.0f / (1.0f + exp(-logits[E]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#ifndef NB
// ---------------------------------------------------------------- 單 token：gate/up（grid (I / 8, TOPK + 1) × 256）
kernel void moe_gateup_iq(device const uchar *Wg [[buffer(0)]], device const uchar *Wu [[buffer(3)]],
                          device const uint *Wsg [[buffer(6)]], device const bfloat *Ssg [[buffer(7)]], device const bfloat *Zsg [[buffer(8)]],
                          device const uint *Wsu [[buffer(9)]], device const bfloat *Ssu [[buffer(10)]], device const bfloat *Zsu [[buffer(11)]],
                          device const half *x [[buffer(12)]], device const float *logits [[buffer(13)]], device half *H [[buffer(14)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sel[TOPK];
    threadgroup float wsel[TOPK + 1];
    iq_select(logits, sel, wsel, sg, lane);
    const uint slot = tg.y, j = tg.x * 8 + sg;
    float gt, up;
    if (slot < TOPK) {
        const ulong r = (ulong)sel[slot] * I + j;
        iq_dot2<FGU, D>(Wg + r * RBGU, Wu + r * RBGU, x, lane, gt, up);
    } else {
        gt = q8_dot<D>(Wsg, Ssg, Zsg, j, x, lane);
        up = q8_dot<D>(Wsu, Ssu, Zsu, j, x, lane);
    }
    gt = simd_sum(gt);
    up = simd_sum(up);
    if (lane == 0) H[slot * I + j] = half(gt / (1.0f + exp(-gt)) * up);
}

// 單 token：down + 加權合併（grid D / 8 × 256）
kernel void moe_down_iq(device const uchar *Wd [[buffer(0)]], device const uint *Wsd [[buffer(3)]], device const bfloat *Ssd [[buffer(4)]],
                        device const bfloat *Zsd [[buffer(5)]], device const half *H [[buffer(6)]], device const float *logits [[buffer(7)]],
                        device half *out [[buffer(9)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sel[TOPK];
    threadgroup float wts[TOPK + 1];
    iq_select(logits, sel, wts, sg, lane);
    const uint row = tg * 8 + sg;
    float acc = 0;
    for (uint s = 0; s < TOPK; s++) acc += wts[s] * iq_dot<FD, I>(Wd + ((ulong)sel[s] * D + row) * RBD, H + s * I, lane);
    acc += wts[TOPK] * q8_dot<I>(Wsd, Ssd, Zsd, row, H + TOPK * I, lane);
    acc = simd_sum(acc);
    if (lane == 0) out[row] = half(acc);
}

// 測試：expert e 的第 r 列解開成 float（gate 或 down 的格式）
kernel void iq_dequant_row(device const uchar *W [[buffer(0)]], device float *y [[buffer(1)]], constant uint3 &a [[buffer(2)]], // row 位元組位移、K、是否 down
                           uint s [[thread_position_in_grid]]) {
    float w[32];
    if (a.z) { if (s >= I / 32) return; dq32<FD>(W + a.x, s, w); }
    else { if (s >= D / 32) return; dq32<FGU>(W + a.x, s, w); }
    for (uint i = 0; i < 32; i++) y[s * 32 + i] = w[i];
}
#else
// ---------------------------------------------------------------- 驗證（NB 個 token）：gate/up（grid (I / 8, NB · (TOPK + 1)) × 256）
kernel void moe_gateup_iqb(device const uchar *Wg [[buffer(0)]], device const uchar *Wu [[buffer(3)]],
                           device const uint *Wsg [[buffer(6)]], device const bfloat *Ssg [[buffer(7)]], device const bfloat *Zsg [[buffer(8)]],
                           device const uint *Wsu [[buffer(9)]], device const bfloat *Ssu [[buffer(10)]], device const bfloat *Zsu [[buffer(11)]],
                           device const half *x [[buffer(12)]], device const float *logits [[buffer(13)]], device half *H [[buffer(14)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sel[TOPK];
    threadgroup float wsel[TOPK + 1];
    const uint t = tg.y / (TOPK + 1), slot = tg.y % (TOPK + 1);
    iq_select(logits + (ulong)t * (E + 1), sel, wsel, sg, lane);
    device const half *xt = x + (ulong)t * D;
    device half *Ht = H + (ulong)t * (TOPK + 1) * I;
    const uint j = tg.x * 8 + sg;
    float gt, up;
    if (slot < TOPK) {
        const ulong r = (ulong)sel[slot] * I + j;
        iq_dot2<FGU, D>(Wg + r * RBGU, Wu + r * RBGU, xt, lane, gt, up);
    } else {
        gt = q8_dot<D>(Wsg, Ssg, Zsg, j, xt, lane);
        up = q8_dot<D>(Wsu, Ssu, Zsu, j, xt, lane);
    }
    gt = simd_sum(gt);
    up = simd_sum(up);
    if (lane == 0) Ht[slot * I + j] = half(gt / (1.0f + exp(-gt)) * up);
}

// 驗證：down（grid (D / 8, NB) × 256）
kernel void moe_down_iqb(device const uchar *Wd [[buffer(0)]], device const uint *Wsd [[buffer(3)]], device const bfloat *Ssd [[buffer(4)]],
                         device const bfloat *Zsd [[buffer(5)]], device const half *H [[buffer(6)]], device const float *logits [[buffer(7)]],
                         device half *out [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sel[TOPK];
    threadgroup float wts[TOPK + 1];
    const uint t = tg.y;
    iq_select(logits + (ulong)t * (E + 1), sel, wts, sg, lane);
    device const half *Ht = H + (ulong)t * (TOPK + 1) * I;
    const uint row = tg.x * 8 + sg;
    float acc = 0;
    for (uint s = 0; s < TOPK; s++) acc += wts[s] * iq_dot<FD, I>(Wd + ((ulong)sel[s] * D + row) * RBD, Ht + s * I, lane);
    acc += wts[TOPK] * q8_dot<I>(Wsd, Ssd, Zsd, row, Ht + TOPK * I, lane);
    acc = simd_sum(acc);
    if (lane == 0) out[(ulong)t * D + row] = half(acc);
}
#endif
#endif // IQ_DQ_ONLY
