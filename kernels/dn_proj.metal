// GGUF profile 的 DeltaNet 投影（decode 一個 token、驗證 NB 個 token）：K-quant / IQ4_XS / BF16 區塊直接解碼
//   dn_in_{Fqkv}_{Fz}_n{NB}：in_proj_qkv、in_proj_z、in_proj_a、in_proj_b（a / b 是 BF16）一次 dispatch（K = 2560）
//   dn_out_{F}_n{NB}：out_proj（K = 6144）
// 權重只讀一次、NB 個 token 共用解碼結果；每個 token 的運算順序與 NB 無關（驗證與 decode 逐位元相同）。
// 每個 SIMD group 算 R 列；32 個 lane 分成兩半，各自處理偶數 / 奇數個 superblock（256 個值），
// 半邊內 16 個 lane 各負責 superblock 裡 16 個值（4 組連續 4 個，x 用 half4 讀）。對照 llama.cpp 的 kernel_mul_mv_q6_K 等的分工。
// 解碼對照 ggml-quants.c 的 dequantize_row_q{4,5,6}_K / iq4_xs（MIT）；跟 moe_iq.metal 的 dq32 數值相同（乘法順序不同）
// 主程式注入：NB、NBS（NB 的字串）
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef R
#define R 2 // 每個 SIMD group 幾列
#endif
#define NSG 8

constant char dn_kv4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}; // IQ4_NL / IQ4_XS 的值表

// 每個 superblock（256 個值）的位元組數
constexpr uint dn_bb(int f) { return f == 12 ? 144 : f == 13 ? 176 : f == 14 ? 210 : f == 23 ? 136 : 512; } // BF16：512

inline float dn_h(device const uchar *p) { return float(as_type<half>(*(device const ushort *)p)); } // 2 位元組對齊

// Q4_K / Q5_K 第 j 段的 scale 與 min（get_scale_min_k4）
inline float2 dn_kq_sm(device const uchar *sc, uint j) {
    if (j < 4) return float2(float(sc[j] & 63), float(sc[j + 4] & 63));
    return float2(float((sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4)), float((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)));
}

// superblock b 裡第 j 個 lane（0…15）的 16 個值 → w[4]（各 4 個連續值），o[4] 是它們在 superblock 內的起點
template <int F> inline void dn_dq16(device const uchar *b, uint j, thread float4 *w, thread uint *o);

template <> inline void dn_dq16<14>(device const uchar *b, uint j, thread float4 *w, thread uint *o) { // Q6_K
    const uint ip = j / 8, l0 = 4 * (j % 8);
    device const ushort *q1 = (device const ushort *)(b + 64 * ip + l0), *q2 = (device const ushort *)(b + 64 * ip + 32 + l0);
    device const ushort *qh = (device const ushort *)(b + 128 + 32 * ip + l0);
    device const char *sc = (device const char *)(b + 192 + 8 * ip + l0 / 16);
    const float d = dn_h(b + 208);
    const uint a = uint(q1[0]) | (uint(q1[1]) << 16), c = uint(q2[0]) | (uint(q2[1]) << 16), h = uint(qh[0]) | (uint(qh[1]) << 16);
    const float s0 = d * float(sc[0]), s1 = d * float(sc[2]), s2 = d * float(sc[4]), s3 = d * float(sc[6]);
    for (uint i = 0; i < 4; i++) {
        const uint ai = (a >> (8 * i)) & 255, ci = (c >> (8 * i)) & 255, hi = (h >> (8 * i)) & 255;
        w[0][i] = s0 * float(int((ai & 15) | ((hi & 3) << 4)) - 32);
        w[1][i] = s1 * float(int((ci & 15) | (((hi >> 2) & 3) << 4)) - 32);
        w[2][i] = s2 * float(int((ai >> 4) | (((hi >> 4) & 3) << 4)) - 32);
        w[3][i] = s3 * float(int((ci >> 4) | (((hi >> 6) & 3) << 4)) - 32);
    }
    const uint base = 128 * ip + l0;
    o[0] = base; o[1] = base + 32; o[2] = base + 64; o[3] = base + 96;
}

// Q4_K / Q5_K：lane j → 第 p = j / 4 對 32 值段（2p、2p+1 共用 qs[32p…]），段內位置 l0 = 8·(j % 4)
template <int F> inline void dn_dq16_k45(device const uchar *b, uint j, thread float4 *w, thread uint *o) {
    const uint p = j / 4, l0 = 8 * (j % 4);
    const float d = dn_h(b), dmin = dn_h(b + 2);
    const float2 sm0 = dn_kq_sm(b + 4, 2 * p), sm1 = dn_kq_sm(b + 4, 2 * p + 1);
    const float d0 = d * sm0.x, m0 = dmin * sm0.y, d1 = d * sm1.x, m1 = dmin * sm1.y;
    const uint2 q = *(device const uint2 *)(b + (F == 13 ? 48 : 16) + 32 * p + l0);
    uint2 hb = uint2(0);
    if (F == 13) hb = *(device const uint2 *)(b + 16 + l0);
    for (uint i = 0; i < 8; i++) {
        const uint qi = ((i < 4 ? q.x : q.y) >> (8 * (i % 4))) & 255;
        const uint hi = ((i < 4 ? hb.x : hb.y) >> (8 * (i % 4))) & 255;
        const uint lo = (qi & 15) | (F == 13 ? ((hi >> (2 * p)) & 1) << 4 : 0);
        const uint up = (qi >> 4) | (F == 13 ? ((hi >> (2 * p + 1)) & 1) << 4 : 0);
        w[i / 4][i % 4] = d0 * float(lo) - m0;
        w[2 + i / 4][i % 4] = d1 * float(up) - m1;
    }
    const uint base = 64 * p + l0;
    o[0] = base; o[1] = base + 4; o[2] = base + 32; o[3] = base + 36;
}
template <> inline void dn_dq16<12>(device const uchar *b, uint j, thread float4 *w, thread uint *o) { dn_dq16_k45<12>(b, j, w, o); }
template <> inline void dn_dq16<13>(device const uchar *b, uint j, thread float4 *w, thread uint *o) { dn_dq16_k45<13>(b, j, w, o); }

template <> inline void dn_dq16<23>(device const uchar *b, uint j, thread float4 *w, thread uint *o) { // IQ4_XS
    const uint ib = j / 2, hh = j % 2;
    const float d = dn_h(b);
    const uint sh = uint(*(device const ushort *)(b + 2));
    const int ls = int(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4));
    const float dl = d * float(ls - 32);
    const uint2 q = *(device const uint2 *)(b + 8 + 16 * ib + 8 * hh);
    for (uint i = 0; i < 8; i++) {
        const uint qi = ((i < 4 ? q.x : q.y) >> (8 * (i % 4))) & 255;
        w[i / 4][i % 4] = dl * float(dn_kv4[qi & 15]);
        w[2 + i / 4][i % 4] = dl * float(dn_kv4[qi >> 4]);
    }
    const uint base = 32 * ib + 8 * hh;
    o[0] = base; o[1] = base + 4; o[2] = base + 16; o[3] = base + 20;
}

template <> inline void dn_dq16<30>(device const uchar *b, uint j, thread float4 *w, thread uint *o) { // BF16
    device const bfloat4 *p = (device const bfloat4 *)(b + 32 * j);
    for (uint i = 0; i < 4; i++) { w[i] = float4(p[i]); o[i] = 16 * j + 4 * i; }
}

// 一個 SIMD group 的 R 列 × NB 個 token：acc[r][t]（每個 lane 的部分和）
template <int F, uint KD>
inline void dn_rows(device const uchar *W, uint row0, device const half *x, uint xs, uint lane, thread float (*acc)[NB]) {
    constexpr uint BB = dn_bb(F), NSB = KD / 256, RB = NSB * BB;
    const uint hf = lane / 16, j = lane % 16;
#pragma clang loop unroll(full)
    for (uint r = 0; r < R; r++)
#pragma clang loop unroll(full)
        for (uint t = 0; t < NB; t++) acc[r][t] = 0;
    for (uint sb = hf; sb < NSB; sb += 2)
    #pragma clang loop unroll(full)
        for (uint r = 0; r < R; r++) {
            float4 w[4];
            uint o[4];
            dn_dq16<F>(W + (ulong)(row0 + r) * RB + sb * BB, j, w, o);
#pragma clang loop unroll(full)
            for (uint t = 0; t < NB; t++) {
                device const half *xt = x + (ulong)t * xs + sb * 256;
                float s = 0;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4; i++) s += dot(w[i], float4(*(device const half4 *)(xt + o[i])));
                acc[r][t] += s;
            }
        }
}

template <int F, uint KD>
inline void dn_mv(device const uchar *W, uint row0, device const half *x, uint xs, device half *y, uint ys, uint lane) {
    float acc[R][NB];
    dn_rows<F, KD>(W, row0, x, xs, lane, acc);
#pragma clang loop unroll(full)
    for (uint r = 0; r < R; r++)
#pragma clang loop unroll(full)
        for (uint t = 0; t < NB; t++) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0) y[(ulong)t * ys + row0 + r] = half(v);
        }
}

// in_proj：列 [0, 10240) qkv、[10240, 16384) z、[16384, 16432) a、[16432, 16480) b。grid：16480 / (NSG·R) × 256
struct dn_in_args {
    uint xs;          // x 的列距（元素）
    uint ys[4];       // 各輸出的列距
    uint ks;          // tensor 單元版：K 分成幾段（grid.y；> 1 時部分和寫到 P，再由 dn_ksum 依固定順序加總）
    uint pr;          //                P 每段每個 token 的列數（= 這次 dispatch 的總列數）
};
static_assert(10240 % (NSG * R) == 0 && 6144 % (NSG * R) == 0 && 48 % R == 0, "dn_in：列的分段要對齊");

template <int FQ, int FZ>
kernel void dn_in(device const uchar *Wq [[buffer(0)]], device const uchar *Wz [[buffer(1)]], device const uchar *Wa [[buffer(2)]],
                  device const uchar *Wb [[buffer(3)]], device const half *x [[buffer(4)]], device half *yq [[buffer(5)]],
                  device half *yz [[buffer(6)]], device half *ya [[buffer(7)]], device half *yb [[buffer(8)]],
                  constant dn_in_args &a [[buffer(9)]], uint tg [[threadgroup_position_in_grid]],
                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = (tg * NSG + sg) * R;
    if (row < 10240) dn_mv<FQ, 2560>(Wq, row, x, a.xs, yq, a.ys[0], lane);
    else if (row < 16384) dn_mv<FZ, 2560>(Wz, row - 10240, x, a.xs, yz, a.ys[1], lane);
    else if (row < 16432) dn_mv<30, 2560>(Wa, row - 16384, x, a.xs, ya, a.ys[2], lane);
    else if (row < 16480) dn_mv<30, 2560>(Wb, row - 16432, x, a.xs, yb, a.ys[3], lane);
}

// ---- 合併版 in_proj（dn_inf）：投影之後直接做每列各自的收尾，省掉 k_gdn_in / kb_gdn_in 那個 dispatch
//   qkv 的列 c：depthwise conv（kernel 4，跟 conv 狀態）+ SiLU → yq（FP16，q / k 還沒正規化：正規化在 gdn_fstep），並更新 conv 狀態
//              （第 0 個 token 之後的狀態就地寫回 st，第 t 個 token 之後的寫到 so[t − 1]）
//   z：照舊；a → log 衰減 A_neg·softplus(a + dt_bias)、b → β = sigmoid(b)（FP32，列距 64）
// 每個值的運算與 gdn_in_body / gdnb_in_body 相同（投影結果先捨入成 FP16，跟原本寫出再讀回一樣）
struct dn_epi {
    device const bfloat *cw;  // conv1d 權重 [10240][4]
    device float *st;         // conv 狀態 [3][10240]（這一層）
    device float *so[7];      // 驗證：第 1…7 個 token 之後的 conv 狀態
    device const float *A;    // A_neg [48]
    device const bfloat *dtb; // dt_bias [48]
};
inline float dn_silu(float x) { return x / (1.0f + exp(-x)); }
inline float dn_sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
// 第 g 列（[0, 16480)）的 NT 個結果 v → 各自的去處（一個執行緒做完一列）。平常 NT = NB；並發（DN_MS）逐個 token 用 NT = 1
template <uint NT>
inline void dn_epi_rowT(uint g, thread const float *v, device float *st, device float *const *so, device const bfloat *cw,
                        device const float *An, device const bfloat *dtb, device half *yq, device half *yz, device float *la,
                        device float *be, constant dn_in_args &a) {
    constexpr uint C = 10240;
    if (g < C) {
        float seq[3 + NT];
        seq[0] = st[g]; seq[1] = st[C + g]; seq[2] = st[2 * C + g];
        for (uint t = 0; t < NT; t++) seq[3 + t] = float(half(v[t]));
        const float w0 = float(cw[g * 4]), w1 = float(cw[g * 4 + 1]), w2 = float(cw[g * 4 + 2]), w3 = float(cw[g * 4 + 3]);
        for (uint t = 0; t < NT; t++) yq[(ulong)t * a.ys[0] + g] = half(dn_silu(w0 * seq[t] + w1 * seq[t + 1] + w2 * seq[t + 2] + w3 * seq[t + 3]));
        st[g] = seq[1]; st[C + g] = seq[2]; st[2 * C + g] = seq[3];
        for (uint t = 1; t < NT; t++) {
            device float *o = so[t - 1];
            o[g] = seq[t + 1]; o[C + g] = seq[t + 2]; o[2 * C + g] = seq[t + 3];
        }
    } else if (g < 16384) {
        for (uint t = 0; t < NT; t++) yz[(ulong)t * a.ys[1] + g - C] = half(v[t]);
    } else if (g < 16432) {
        const uint j = g - 16384;
        for (uint t = 0; t < NT; t++) {
            const float xa = float(half(v[t])) + float(dtb[j]);
            const float sp = xa > 20.0f ? xa : log(1.0f + exp(xa));
            la[t * a.ys[2] + j] = An[j] * sp;
        }
    } else {
        const uint j = g - 16432;
        for (uint t = 0; t < NT; t++) be[t * a.ys[3] + j] = dn_sigm(float(half(v[t])));
    }
}
inline void dn_epi_row(uint g, thread const float *v, thread const dn_epi &ep, device half *yq, device half *yz, device float *la,
                       device float *be, constant dn_in_args &a) {
#ifdef DN_MS
    // 並發（qwental-serve --parallel）：第 t 個 token 是另一個對話，conv 狀態各自一份（t = 0 是 st、t ≥ 1 是 so[t − 1]）；
    // 每個 token 用 NT = 1 的同一份程式（與 decode 逐位元相同）
    for (uint t = 0; t < NB; t++)
        dn_epi_rowT<1>(g, v + t, t == 0 ? ep.st : ep.so[t - 1], ep.so, ep.cw, ep.A, ep.dtb, yq + (ulong)t * a.ys[0], yz + (ulong)t * a.ys[1],
                       la + t * a.ys[2], be + t * a.ys[3], a);
#else
    dn_epi_rowT<NB>(g, v, ep.st, ep.so, ep.cw, ep.A, ep.dtb, yq, yz, la, be, a);
#endif
}
#define DN_EPI_ARGS                                                                                                              \
    device const bfloat *cw [[buffer(11)]], device float *st [[buffer(12)]], device const float *An [[buffer(13)]],             \
        device const bfloat *dtb [[buffer(14)]], device float *so1 [[buffer(15)]], device float *so2 [[buffer(16)]],            \
        device float *so3 [[buffer(17)]], device float *so4 [[buffer(18)]], device float *so5 [[buffer(19)]],                   \
        device float *so6 [[buffer(20)]], device float *so7 [[buffer(21)]]
#define DN_EPI_MAKE const dn_epi ep = {cw, st, {so1, so2, so3, so4, so5, so6, so7}, An, dtb}

template <int FQ, int FZ>
kernel void dn_inf(device const uchar *Wq [[buffer(0)]], device const uchar *Wz [[buffer(1)]], device const uchar *Wa [[buffer(2)]],
                   device const uchar *Wb [[buffer(3)]], device const half *x [[buffer(4)]], device half *yq [[buffer(5)]],
                   device half *yz [[buffer(6)]], device float *la [[buffer(7)]], device float *be [[buffer(8)]],
                   constant dn_in_args &a [[buffer(9)]], DN_EPI_ARGS, uint tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    DN_EPI_MAKE;
    const uint row = (tg * NSG + sg) * R;
    float acc[R][NB];
    if (row < 10240) dn_rows<FQ, 2560>(Wq, row, x, a.xs, lane, acc);
    else if (row < 16384) dn_rows<FZ, 2560>(Wz, row - 10240, x, a.xs, lane, acc);
    else if (row < 16432) dn_rows<30, 2560>(Wa, row - 16384, x, a.xs, lane, acc);
    else dn_rows<30, 2560>(Wb, row - 16432, x, a.xs, lane, acc);
#pragma clang loop unroll(full)
    for (uint r = 0; r < R; r++) {
        float v[NB];
#pragma clang loop unroll(full)
        for (uint t = 0; t < NB; t++) v[t] = simd_sum(acc[r][t]);
        if (lane == 0) dn_epi_row(row + r, v, ep, yq, yz, la, be, a);
    }
}

// out_proj：2560 列、K = 6144。grid：2560 / (NSG·R) × 256
template <int F>
kernel void dn_out(device const uchar *W [[buffer(0)]], device const half *x [[buffer(4)]], device half *y [[buffer(5)]],
                   constant dn_in_args &a [[buffer(9)]], uint tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    dn_mv<F, 6144>(W, (tg * NSG + sg) * R, x, a.xs, y, a.ys[0], lane);
}

// ---------------------------------------------------------------- tensor 單元版（M5 的矩陣單元，驗證 NB ≥ 2）
// 權重直接解碼進 SIMD group 的 cooperative tensor（同 qmvb_tc.metal），跟 8 列輸入（不足 8 列的部分是緩衝區裡的
// 其他資料，結果不寫出；每個 token 的結果與 NB 無關）相乘。每個 threadgroup 32 個輸出列，DN_TSG 個 SIMD group 輪流分 K，
// 最後在 threadgroup memory 依固定順序加總。
// 權重拆成 hi = half(w)、lo = half(w − hi) 兩個 FP16 運算元（兩次乘加）：精度跟 FP32 權重相當（約 22 位元），不因為矩陣單元只吃 FP16 而變差。
// x 緩衝區至少要有 8 列（VBMAX）。

// superblock b 裡位置 p（4 的倍數）起的 4 個值
template <int F> inline float4 dn_dq4(device const uchar *b, uint p);
template <> inline float4 dn_dq4<14>(device const uchar *b, uint p) { // Q6_K
    const uint ip = p / 128, r = (p % 128) / 32, l = p % 32;
    device const ushort *ql = (device const ushort *)(b + 64 * ip + 32 * (r % 2) + l), *qh = (device const ushort *)(b + 128 + 32 * ip + l);
    const uint a = (uint(ql[0]) | (uint(ql[1]) << 16)) >> (4 * (r / 2)), h = (uint(qh[0]) | (uint(qh[1]) << 16)) >> (2 * r);
    const float s = dn_h(b + 208) * float(((device const char *)b)[192 + 8 * ip + 2 * r + l / 16]);
    float4 w;
    for (uint i = 0; i < 4; i++) w[i] = s * float(int(((a >> (8 * i)) & 15) | (((h >> (8 * i)) & 3) << 4)) - 32);
    return w;
}
template <int F> inline float4 dn_dq4_k45(device const uchar *b, uint p) {
    const uint j = p / 32, l = p % 32;
    const float2 sm = dn_kq_sm(b + 4, j);
    const float d = dn_h(b) * sm.x, m = dn_h(b + 2) * sm.y;
    const uint q = *(device const uint *)(b + (F == 13 ? 48 : 16) + 32 * (j / 2) + l) >> (4 * (j % 2));
    const uint h = F == 13 ? *(device const uint *)(b + 16 + l) >> j : 0;
    float4 w;
    for (uint i = 0; i < 4; i++) w[i] = d * float(((q >> (8 * i)) & 15) | (F == 13 ? ((h >> (8 * i)) & 1) << 4 : 0)) - m;
    return w;
}
template <> inline float4 dn_dq4<12>(device const uchar *b, uint p) { return dn_dq4_k45<12>(b, p); }
template <> inline float4 dn_dq4<13>(device const uchar *b, uint p) { return dn_dq4_k45<13>(b, p); }
template <> inline float4 dn_dq4<23>(device const uchar *b, uint p) { // IQ4_XS
    const uint ib = p / 32, l = p % 32;
    const uint sh = uint(*(device const ushort *)(b + 2));
    const int ls = int(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4));
    const float dl = dn_h(b) * float(ls - 32);
    const uint q = *(device const uint *)(b + 8 + 16 * ib + (l % 16)) >> (l >= 16 ? 4 : 0);
    float4 w;
    for (uint i = 0; i < 4; i++) w[i] = dl * float(dn_kv4[(q >> (8 * i)) & 15]);
    return w;
}
template <> inline float4 dn_dq4<30>(device const uchar *b, uint p) { return float4(*(device const bfloat4 *)(b + 2 * p)); }

// 32 值段 c（32 的倍數）裡位置 l 與 l + 16（l = 0、4、8、12）各 4 個值：tensor 單元的 right input 每個 lane 正好拿這兩組
template <int F> inline void dn_dq8(device const uchar *b, uint c, uint l, thread float4 &w0, thread float4 &w1);
template <> inline void dn_dq8<14>(device const uchar *b, uint c, uint l, thread float4 &w0, thread float4 &w1) { // Q6_K
    const uint ip = c / 128, r = (c % 128) / 32;
    device const ushort *ql = (device const ushort *)(b + 64 * ip + 32 * (r % 2) + l), *qh = (device const ushort *)(b + 128 + 32 * ip + l);
    const uint ls = 4 * (r / 2), hs = 2 * r;
    const uint a0 = (uint(ql[0]) | (uint(ql[1]) << 16)) >> ls, a1 = (uint(ql[8]) | (uint(ql[9]) << 16)) >> ls;
    const uint h0 = (uint(qh[0]) | (uint(qh[1]) << 16)) >> hs, h1 = (uint(qh[8]) | (uint(qh[9]) << 16)) >> hs;
    device const char *sc = (device const char *)b + 192 + 8 * ip + 2 * r;
    const float d = dn_h(b + 208), s0 = d * float(sc[0]), s1 = d * float(sc[1]);
    for (uint i = 0; i < 4; i++) {
        w0[i] = s0 * float(int(((a0 >> (8 * i)) & 15) | (((h0 >> (8 * i)) & 3) << 4)) - 32);
        w1[i] = s1 * float(int(((a1 >> (8 * i)) & 15) | (((h1 >> (8 * i)) & 3) << 4)) - 32);
    }
}
template <int F> inline void dn_dq8_k45(device const uchar *b, uint c, uint l, thread float4 &w0, thread float4 &w1) {
    const uint j = c / 32;
    const float2 sm = dn_kq_sm(b + 4, j);
    const float d = dn_h(b) * sm.x, m = dn_h(b + 2) * sm.y;
    device const uint *qp = (device const uint *)(b + (F == 13 ? 48 : 16) + 32 * (j / 2) + l);
    const uint sh = 4 * (j % 2);
    const uint q0 = qp[0] >> sh, q1 = qp[4] >> sh;
    uint h0 = 0, h1 = 0;
    if (F == 13) { device const uint *hp = (device const uint *)(b + 16 + l); h0 = hp[0] >> j; h1 = hp[4] >> j; }
    for (uint i = 0; i < 4; i++) {
        w0[i] = d * float(((q0 >> (8 * i)) & 15) | (F == 13 ? ((h0 >> (8 * i)) & 1) << 4 : 0)) - m;
        w1[i] = d * float(((q1 >> (8 * i)) & 15) | (F == 13 ? ((h1 >> (8 * i)) & 1) << 4 : 0)) - m;
    }
}
template <> inline void dn_dq8<12>(device const uchar *b, uint c, uint l, thread float4 &w0, thread float4 &w1) { dn_dq8_k45<12>(b, c, l, w0, w1); }
template <> inline void dn_dq8<13>(device const uchar *b, uint c, uint l, thread float4 &w0, thread float4 &w1) { dn_dq8_k45<13>(b, c, l, w0, w1); }
template <> inline void dn_dq8<23>(device const uchar *b, uint c, uint l, thread float4 &w0, thread float4 &w1) { // IQ4_XS：l 與 l + 16 是同一組位元組的低 / 高 4 位元
    const uint ib = c / 32;
    const uint sh = uint(*(device const ushort *)(b + 2));
    const int ls = int(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4));
    const float dl = dn_h(b) * float(ls - 32);
    const uint q = *(device const uint *)(b + 8 + 16 * ib + l);
    for (uint i = 0; i < 4; i++) {
        w0[i] = dl * float(dn_kv4[(q >> (8 * i)) & 15]);
        w1[i] = dl * float(dn_kv4[(q >> (8 * i + 4)) & 15]);
    }
}
template <> inline void dn_dq8<30>(device const uchar *b, uint c, uint l, thread float4 &w0, thread float4 &w1) {
    w0 = float4(*(device const bfloat4 *)(b + 2 * (c + l)));
    w1 = float4(*(device const bfloat4 *)(b + 2 * (c + l + 16)));
}

#define DN_TSG 8
#define TMT 8  // 輸入列（token）
#define TNT 32 // 每個 threadgroup 的輸出列
#define TKT 32 // 每次 K

// 32 列（列 n < split 在 W0，其餘在 W1 的第 n − split 列）× 8 個 token，K 的第 ks 段（共 a.ks 段）
// a.ks == 1：直接寫 y（只寫前 NB 個 token）；否則部分和寫到 P[ks][token][g0 + 列]
template <int F, uint KD, bool EPI = false>
inline void dn_tc(device const uchar *W0, device const uchar *W1, uint split, uint n0, device const half *x, device half *y0,
                  device half *y1, uint ys, ushort sg, ushort tid, threadgroup float (*part)[TMT][TNT], constant dn_in_args &a, uint ks,
                  device float *P, uint g0) {
    constexpr uint BB = dn_bb(F), RB = KD / 256 * BB;
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX((device half *)x, T2(KD, TMT));
    constexpr auto desc = matmul2d_descriptor(TMT, TNT, TKT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto rh = op.get_right_input_cooperative_tensor<half, half, float>();
    auto rl = op.get_right_input_cooperative_tensor<half, half, float>();
    auto x0 = tX.slice<TKT, TMT>(0, 0);
    auto acc = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(x0)>,
                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(rh)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    const uint kl = KD / a.ks, kb = ks * kl; // 主程式保證 KD / ks 是 32 的倍數
    for (uint k0 = kb + sg * TKT; k0 < kb + kl; k0 += TKT * DN_TSG) {
        // 每個 lane 的 32 個元素：k ∈ {l, l + 16}（l = 0、4、8、12）× 4 列（間隔 8），每組連續 4 個 k（M5 的排列；排列不同時 QW_DN_VTEST 會抓到）
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rh.get_capacity() / 2; i += 4) {
            const auto ix = rh.get_multidimensional_index(i);
            const uint n = n0 + ix[1], l = ix[0];
            device const uchar *row = n < split ? W0 + (ulong)n * RB : W1 + (ulong)(n - split) * RB;
            float4 w[2];
            dn_dq8<F>(row + (k0 / 256) * BB, k0 % 256, l, w[0], w[1]);
            for (uint g = 0; g < 2; g++)
                for (uint c = 0; c < 4; c++) {
                    const uint e = i + g * rh.get_capacity() / 2 + c;
                    const half h = half(w[g][c]);
                    rh[e] = h;
                    rl[e] = half(w[g][c] - float(h));
                }
        }
        auto xs = tX.slice<TKT, TMT>(k0, 0);
        op.run(xs, rh, acc);
        op.run(xs, rl, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            const auto ix = acc.get_multidimensional_index(i);
            part[sg][ix[1]][ix[0]] = acc[i];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (EPI) return; // 合併版：呼叫端自己收尾
    for (uint t = tid; t < NB * TNT; t += 32 * DN_TSG) {
        const uint m = t / TNT, nn = t % TNT, n = n0 + nn;
        float v = 0;
        for (uint s = 0; s < DN_TSG; s++) v += part[s][m][nn];
        if (a.ks > 1) P[((ulong)ks * TMT + m) * a.pr + g0 + nn] = v;
        else if (n < split) y0[(ulong)m * ys + n] = half(v);
        else y1[(ulong)m * ys + n - split] = half(v);
    }
}

// in_proj：threadgroup 依序是 qkv（320 個）、z（192 個）、a + b（3 個，96 列當成一個矩陣）。grid：515 × 32·DN_TSG
template <int FQ, int FZ>
kernel void dn_in_tc(device const uchar *Wq [[buffer(0)]], device const uchar *Wz [[buffer(1)]], device const uchar *Wa [[buffer(2)]],
                     device const uchar *Wb [[buffer(3)]], device const half *x [[buffer(4)]], device half *yq [[buffer(5)]],
                     device half *yz [[buffer(6)]], device half *ya [[buffer(7)]], device half *yb [[buffer(8)]],
                     constant dn_in_args &a [[buffer(9)]], device float *P [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[DN_TSG][TMT][TNT];
    const uint r0 = tg.x * TNT;
    if (r0 < 10240) dn_tc<FQ, 2560>(Wq, Wq, 0xffffffffu, r0, x, yq, yq, a.ys[0], sg, tid, part, a, tg.y, P, r0);
    else if (r0 < 16384) dn_tc<FZ, 2560>(Wz, Wz, 0xffffffffu, r0 - 10240, x, yz, yz, a.ys[1], sg, tid, part, a, tg.y, P, r0);
    else dn_tc<30, 2560>(Wa, Wb, 48, r0 - 16384, x, ya, yb, a.ys[2], sg, tid, part, a, tg.y, P, r0); // a、b 的列距相同（48）
}
// 合併版（同 dn_inf 的收尾；只支援 K 不分段）：執行緒 nn < 32 負責第 r0 + nn 列的 NB 個 token
template <int FQ, int FZ>
kernel void dn_inf_tc(device const uchar *Wq [[buffer(0)]], device const uchar *Wz [[buffer(1)]], device const uchar *Wa [[buffer(2)]],
                      device const uchar *Wb [[buffer(3)]], device const half *x [[buffer(4)]], device half *yq [[buffer(5)]],
                      device half *yz [[buffer(6)]], device float *la [[buffer(7)]], device float *be [[buffer(8)]],
                      constant dn_in_args &a [[buffer(9)]], device float *P [[buffer(10)]], DN_EPI_ARGS, uint2 tg [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    DN_EPI_MAKE;
    threadgroup float part[DN_TSG][TMT][TNT];
    const uint r0 = tg.x * TNT;
    device half *nh = nullptr;
    if (r0 < 10240) dn_tc<FQ, 2560, true>(Wq, Wq, 0xffffffffu, r0, x, nh, nh, 0, sg, tid, part, a, 0, P, r0);
    else if (r0 < 16384) dn_tc<FZ, 2560, true>(Wz, Wz, 0xffffffffu, r0 - 10240, x, nh, nh, 0, sg, tid, part, a, 0, P, r0);
    else dn_tc<30, 2560, true>(Wa, Wb, 48, r0 - 16384, x, nh, nh, 0, sg, tid, part, a, 0, P, r0);
    if (tid >= TNT) return;
    const uint g = r0 + tid;
    if (g >= 16480) return;
    float v[NB];
    for (uint m = 0; m < NB; m++) {
        float s = 0;
        for (uint q = 0; q < DN_TSG; q++) s += part[q][m][tid];
        v[m] = s;
    }
    dn_epi_row(g, v, ep, yq, yz, la, be, a);
}

template <int F>
kernel void dn_out_tc(device const uchar *W [[buffer(0)]], device const half *x [[buffer(4)]], device half *y [[buffer(5)]],
                      constant dn_in_args &a [[buffer(9)]], device float *P [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[DN_TSG][TMT][TNT];
    dn_tc<F, 6144>(W, W, 0xffffffffu, tg.x * TNT, x, y, y, a.ys[0], sg, tid, part, a, tg.y, P, tg.x * TNT);
}

// K 分段的部分和依段的順序加總 → 輸出。列 g 的去處：[0, b0) → y0、[b0, b1) → y1、[b1, b2) → y2、其餘 → y3（各自的列距 ys[i]）
// grid：ceil(pr·NB / 256) × 256
struct dn_ksum_args {
    uint ks, pr, b0, b1, b2, ys[4];
};
[[host_name("dn_ksum_n" NBS)]] kernel void dn_ksum(device const float *P [[buffer(0)]], device half *y0 [[buffer(1)]], device half *y1 [[buffer(2)]],
                    device half *y2 [[buffer(3)]], device half *y3 [[buffer(4)]], constant dn_ksum_args &a [[buffer(5)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= a.pr * NB) return;
    const uint m = i / a.pr, g = i % a.pr;
    float v = 0;
    for (uint k = 0; k < a.ks; k++) v += P[((ulong)k * TMT + m) * a.pr + g];
    if (g < a.b0) y0[(ulong)m * a.ys[0] + g] = half(v);
    else if (g < a.b1) y1[(ulong)m * a.ys[1] + g - a.b0] = half(v);
    else if (g < a.b2) y2[(ulong)m * a.ys[2] + g - a.b1] = half(v);
    else y3[(ulong)m * a.ys[3] + g - a.b2] = half(v);
}

// 量測用（QW_DN_KB1）：同一組權重的純讀取上限。4 個緩衝區各自切成連續段，每個 threadgroup 一段，組內 uint4 合併讀取、
// 每輪 4 個獨立載入。tb[i]：第 i 個緩衝區從第幾個 threadgroup 開始；n[i]：uint4 個數。grid：tb[4] × 256
struct dn_rd_args { uint n[4], tb[5]; };
[[host_name("dn_rd_n" NBS)]] kernel void dn_rd(device const uint4 *W0 [[buffer(0)]], device const uint4 *W1 [[buffer(1)]],
                                               device const uint4 *W2 [[buffer(2)]], device const uint4 *W3 [[buffer(3)]],
                                               device uint *out [[buffer(4)]], constant dn_rd_args &a [[buffer(5)]],
                                               uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint w = tg >= a.tb[3] ? 3 : tg >= a.tb[2] ? 2 : tg >= a.tb[1] ? 1 : 0;
    device const uint4 *src = w == 0 ? W0 : w == 1 ? W1 : w == 2 ? W2 : W3;
    const uint ntg = a.tb[w + 1] - a.tb[w], n = a.n[w], per = (n + ntg - 1) / ntg, b0 = (tg - a.tb[w]) * per, e0 = min(b0 + per, n);
    uint4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint i = b0 + tid;
    for (; i + 3 * 256 < e0; i += 4 * 256) { a0 ^= src[i]; a1 ^= src[i + 256]; a2 ^= src[i + 512]; a3 ^= src[i + 768]; }
    for (; i < e0; i += 256) a0 ^= src[i];
    const uint4 acc = a0 ^ a1 ^ a2 ^ a3;
    const uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0x9e3779b9u) out[tg & 1023] = r;
}

typedef decltype(dn_in<12, 12>) dn_in_t;
typedef decltype(dn_out<12>) dn_out_t;
#define DN_IN_SIG(FQ, FZ) template [[host_name("dn_in_" #FQ "_" #FZ "_n" NBS)]] kernel dn_in_t dn_in<FQ, FZ>;
#define DN_IN_ROW(FQ) DN_IN_SIG(FQ, 12) DN_IN_SIG(FQ, 13) DN_IN_SIG(FQ, 14) DN_IN_SIG(FQ, 23)
DN_IN_ROW(12)
DN_IN_ROW(13)
DN_IN_ROW(14)
DN_IN_ROW(23)
#define DN_OUT_SIG(F) template [[host_name("dn_out_" #F "_n" NBS)]] kernel dn_out_t dn_out<F>;
DN_OUT_SIG(12)
DN_OUT_SIG(13)
DN_OUT_SIG(14)
DN_OUT_SIG(23)
typedef decltype(dn_inf<12, 12>) dn_inf_t;
#define DN_INF_SIG(FQ, FZ) template [[host_name("dn_inf_" #FQ "_" #FZ "_n" NBS)]] kernel dn_inf_t dn_inf<FQ, FZ>;
#define DN_INF_ROW(FQ) DN_INF_SIG(FQ, 12) DN_INF_SIG(FQ, 13) DN_INF_SIG(FQ, 14) DN_INF_SIG(FQ, 23)
DN_INF_ROW(12)
DN_INF_ROW(13)
DN_INF_ROW(14)
DN_INF_ROW(23)
typedef decltype(dn_inf_tc<12, 12>) dn_inf_tc_t;
#define DN_INFT_SIG(FQ, FZ) template [[host_name("dn_inf_tc_" #FQ "_" #FZ "_n" NBS)]] kernel dn_inf_tc_t dn_inf_tc<FQ, FZ>;
#define DN_INFT_ROW(FQ) DN_INFT_SIG(FQ, 12) DN_INFT_SIG(FQ, 13) DN_INFT_SIG(FQ, 14) DN_INFT_SIG(FQ, 23)
DN_INFT_ROW(12)
DN_INFT_ROW(13)
DN_INFT_ROW(14)
DN_INFT_ROW(23)
typedef decltype(dn_in_tc<12, 12>) dn_in_tc_t;
typedef decltype(dn_out_tc<12>) dn_out_tc_t;
#define DN_INT_SIG(FQ, FZ) template [[host_name("dn_in_tc_" #FQ "_" #FZ "_n" NBS)]] kernel dn_in_tc_t dn_in_tc<FQ, FZ>;
#define DN_INT_ROW(FQ) DN_INT_SIG(FQ, 12) DN_INT_SIG(FQ, 13) DN_INT_SIG(FQ, 14) DN_INT_SIG(FQ, 23)
DN_INT_ROW(12)
DN_INT_ROW(13)
DN_INT_ROW(14)
DN_INT_ROW(23)
#define DN_OUTT_SIG(F) template [[host_name("dn_out_tc_" #F "_n" NBS)]] kernel dn_out_tc_t dn_out_tc<F>;
DN_OUTT_SIG(12)
DN_OUTT_SIG(13)
DN_OUTT_SIG(14)
DN_OUTT_SIG(23)
