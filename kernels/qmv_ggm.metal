// GGUF profile 的多矩陣 matvec（attention 的 q / k / v / indexer 一次 dispatch；o_proj 單獨），decode 1 個 token、驗證 NB 個 token。
// 前面接 iq_tables.metal 與 moe_iq.metal（IQ_DQ_ONLY：ldh、kq_scale_min、kvalues_iq4nl）。
//
// 做法（同 llama.cpp 的 mul_mv_q*_K 排列）：每個 SIMD group 一列；半個 SIMD group（16 個 lane）一個 256 值的超區塊，
// 每個 lane 解 16 個權重（4 段 × 4 個連續值，位元組用 32 / 64-bit 讀），兩半輪流走 K / 256 個超區塊。
// 權重解一次、跟 NB 個 token 各乘一次（解碼成本攤到 NB 個 token）。
//
// 主程式注入：KD（K，256 的倍數）、NB（1–8；0 = 一次編出 NB 1…8 共 8 個 kernel，名稱 <qmv_ggm>_b1…_b8）、NMAT（1–4）、
//             GF0…GF3（每個矩陣的 ggml type；BF16 = 30）、YT；名稱改名。
// （token 數改成執行時參數、編一份共用的版本量過：NB = 2 / 3 / 8 慢 12–16%，所以每個 NB 各一個 kernel）
// 支援的格式：Q4_K 12、Q5_K 13、Q6_K 14、IQ4_XS 23、BF16 30（其他格式不能編譯：host 端先檢查）
#ifndef YT
#define YT half
#endif
static_assert(KD % 256 == 0, "qmv_ggm：K 必須是 256 的倍數");
#ifndef GGM_PF
static_assert(NB >= 0 && NB <= 8, "qmv_ggm：NB 0–8");
#endif
#ifndef GF1
#define GF1 GF0
#endif
#ifndef GF2
#define GF2 GF0
#endif
#ifndef GF3
#define GF3 GF0
#endif

// 每個 256 值超區塊的位元組數（BF16：256 × 2）；0 = 不支援
template <int F> constexpr uint ggm_bb() { return F == 12 ? 144 : F == 13 ? 176 : F == 14 ? 210 : F == 23 ? 136 : F == 30 ? 512 : 0; }

inline float4 nib_lo(uint w) { return float4(as_type<uchar4>(w & 0x0f0f0f0fu)); }
inline float4 nib_hi(uint w) { return float4(as_type<uchar4>((w >> 4) & 0x0f0f0f0fu)); }

// 超區塊 b（位元組起點）中 lane il（0…15）的 16 個權重：w[g] 對應超區塊內位置 p[g]…p[g] + 3
template <int F> inline void deq16(device const uchar *b, uint il, thread float4 *w, thread uint *p);

template <> inline void deq16<12>(device const uchar *b, uint il, thread float4 *w, thread uint *p) { // Q4_K
    const uint sp = il / 4, q = il % 4; // 子區塊 2sp（低 nibble）與 2sp + 1（高 nibble），各 8 個值
    const float d = ldh(b), dmin = ldh(b + 2);
    float s0, m0, s1, m1;
    kq_scale_min(b + 4, 2 * sp, s0, m0);
    kq_scale_min(b + 4, 2 * sp + 1, s1, m1);
    const float a0 = d * s0, c0 = dmin * m0, a1 = d * s1, c1 = dmin * m1;
    const uint2 qs = *(device const uint2 *)(b + 16 + 32 * sp + 8 * q);
    w[0] = a0 * nib_lo(qs.x) - c0;
    w[1] = a0 * nib_lo(qs.y) - c0;
    w[2] = a1 * nib_hi(qs.x) - c1;
    w[3] = a1 * nib_hi(qs.y) - c1;
    p[0] = 64 * sp + 8 * q;
    p[1] = p[0] + 4;
    p[2] = p[0] + 32;
    p[3] = p[0] + 36;
}

template <> inline void deq16<13>(device const uchar *b, uint il, thread float4 *w, thread uint *p) { // Q5_K
    const uint sp = il / 4, q = il % 4, j0 = 2 * sp, j1 = j0 + 1;
    const float d = ldh(b), dmin = ldh(b + 2);
    float s0, m0, s1, m1;
    kq_scale_min(b + 4, j0, s0, m0);
    kq_scale_min(b + 4, j1, s1, m1);
    const float a0 = d * s0, c0 = dmin * m0, a1 = d * s1, c1 = dmin * m1;
    const uint2 qs = *(device const uint2 *)(b + 48 + 32 * sp + 8 * q);
    const uint2 qh = *(device const uint2 *)(b + 16 + 8 * q);
    // 5 位元整數先在位元組裡拼好（低 4 位 | 第 5 位 << 4），再一次轉成 float
    w[0] = a0 * float4(as_type<uchar4>((qs.x & 0x0f0f0f0fu) | (((qh.x >> j0) & 0x01010101u) << 4))) - c0;
    w[1] = a0 * float4(as_type<uchar4>((qs.y & 0x0f0f0f0fu) | (((qh.y >> j0) & 0x01010101u) << 4))) - c0;
    w[2] = a1 * float4(as_type<uchar4>(((qs.x >> 4) & 0x0f0f0f0fu) | (((qh.x >> j1) & 0x01010101u) << 4))) - c1;
    w[3] = a1 * float4(as_type<uchar4>(((qs.y >> 4) & 0x0f0f0f0fu) | (((qh.y >> j1) & 0x01010101u) << 4))) - c1;
    p[0] = 64 * sp + 8 * q;
    p[1] = p[0] + 4;
    p[2] = p[0] + 32;
    p[3] = p[0] + 36;
}

inline uint ld4_a2(device const uchar *p) { // 4 位元組，只保證 2 位元組對齊（Q6_K 區塊 210 B）
    device const ushort *u = (device const ushort *)p;
    return uint(u[0]) | (uint(u[1]) << 16);
}

template <> inline void deq16<14>(device const uchar *b, uint il, thread float4 *w, thread uint *p) { // Q6_K
    const uint n = il / 8, l0 = 4 * (il % 8); // 第 n 個 128、位置 l0…l0 + 3，r = 0…3 四段
    const uint ql0 = ld4_a2(b + 64 * n + l0), ql1 = ld4_a2(b + 64 * n + 32 + l0), qh = ld4_a2(b + 128 + 32 * n + l0);
    const float d = ldh(b + 208);
    device const char *sc = (device const char *)(b + 192 + 8 * n + l0 / 16);
    const uint lo[4] = {ql0 & 0x0f0f0f0fu, ql1 & 0x0f0f0f0fu, (ql0 >> 4) & 0x0f0f0f0fu, (ql1 >> 4) & 0x0f0f0f0fu};
    for (uint r = 0; r < 4; r++) {
        const uint q = lo[r] | (((qh >> (2 * r)) & 0x03030303u) << 4);
        w[r] = d * float(sc[2 * r]) * (float4(as_type<uchar4>(q)) - 32.0f);
        p[r] = 128 * n + 32 * r + l0;
    }
}

template <> inline void deq16<23>(device const uchar *b, uint il, thread float4 *w, thread uint *p) { // IQ4_XS
    const uint ib = il / 2, h = il % 2;
    const float d = ldh(b);
    const uint sh = uint(b[2]) | (uint(b[3]) << 8);
    const int ls = int(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4));
    const float dl = d * float(ls - 32);
    const uint2 qs = *(device const uint2 *)(b + 8 + 16 * ib + 8 * h);
    const uchar4 a = as_type<uchar4>(qs.x & 0x0f0f0f0fu), c = as_type<uchar4>(qs.y & 0x0f0f0f0fu);
    const uchar4 e = as_type<uchar4>((qs.x >> 4) & 0x0f0f0f0fu), f = as_type<uchar4>((qs.y >> 4) & 0x0f0f0f0fu);
#define KV4(v) float4(kvalues_iq4nl[v.x], kvalues_iq4nl[v.y], kvalues_iq4nl[v.z], kvalues_iq4nl[v.w])
    w[0] = dl * KV4(a);
    w[1] = dl * KV4(c);
    w[2] = dl * KV4(e);
    w[3] = dl * KV4(f);
#undef KV4
    p[0] = 32 * ib + 8 * h;
    p[1] = p[0] + 4;
    p[2] = p[0] + 16;
    p[3] = p[0] + 20;
}

template <> inline void deq16<30>(device const uchar *b, uint il, thread float4 *w, thread uint *p) { // BF16：連續 16 個
    device const bfloat4 *v = (device const bfloat4 *)b + 4 * il;
    for (uint g = 0; g < 4; g++) {
        w[g] = float4(v[g]);
        p[g] = 16 * il + 4 * g;
    }
}

#ifdef GGM_PF
// ---------------------------------------------------------------- prefill：Y[M, N] = X[M, K] · Wᵀ（tensor ops，FP16 運算元、FP32 累加）
// 每個 threadgroup TM × TN 的輸出；K 方向每次一個 256 值的超區塊：全體執行緒用 deq16 把 TN 列解碼成 FP16 放進
// threadgroup memory（每 16 個執行緒一列，向量讀寫），再 multiply_accumulate。權重轉 FP16 與 llama.cpp 的 mul_mm 相同。
// 主程式注入：KD、GF0（格式）、TM、TN、SG、YT；名稱改名。主程式保證 N 是 TN 的倍數
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#define PF_KC 256
#define PF_NT (32 * SG)
static_assert(TN * PF_KC * 2 <= 32768, "qmm_ggm：threadgroup memory 不夠");
static_assert(PF_NT % 16 == 0 && TN % (PF_NT / 16) == 0, "qmm_ggm：TN 必須是（執行緒數 / 16）的倍數");

struct qmm_ggm_args {
    uint M, N;
};

kernel void qmm_ggm(device half *X [[buffer(0)]], device const uchar *W [[buffer(1)]], device YT *Y [[buffer(4)]],
                    constant qmm_ggm_args &a [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    constexpr ulong BB = ggm_bb<GF0>(), RB = KD / 256 * BB;
    static_assert(BB > 0, "qmm_ggm：不支援的格式");
    const uint m0 = tg.y * TM, n0 = tg.x * TN;
    threadgroup half Wt[TN * PF_KC];
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX(X, T2(KD, a.M));
    tensor<threadgroup half, T2, tensor_inline> tW(Wt, T2(PF_KC, TN));
    constexpr auto desc = matmul2d_descriptor(TM, TN, PF_KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto x0 = tX.slice(0, m0);
    auto acc = op.get_destination_cooperative_tensor<decltype(x0), decltype(tW), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    const uint il = tid % 16;
    for (uint i = 0; i < KD / 256; i++) {
        for (uint r = tid / 16; r < TN; r += PF_NT / 16) {
            float4 w[4];
            uint p[4];
            deq16<GF0>(W + (ulong)(n0 + r) * RB + (ulong)i * BB, il, w, p);
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + r * PF_KC);
            for (uint g = 0; g < 4; g++) dst[p[g] / 4] = half4(w[g]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto xs = tX.slice(i * 256, m0);
        op.run(xs, tW, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const uint n = n0 + ix[0], m = m0 + ix[1];
            if (m < a.M) Y[(ulong)m * a.N + n] = YT(acc[i]);
        }
}
#else
// 一列對 NBT 個 token 的部分內積（lane 的部分；呼叫端再 simd_sum）。量過：一個 SIMD group 兩列共用 x、
// x 先轉 float、x 先搬進 threadgroup memory 都比較慢（NB = 8 時瓶頸在每個 token 讀 x）
template <int F, uint NBT> inline void row_nb(device const uchar *row, device const half *x, uint xs, uint lane, thread float *acc) {
    constexpr uint BB = ggm_bb<F>();
    static_assert(BB > 0, "qmv_ggm：不支援的格式");
    for (uint i = lane / 16; i < KD / 256; i += 2) {
        float4 w[4];
        uint p[4];
        deq16<F>(row + (ulong)i * BB, lane % 16, w, p);
        for (uint t = 0; t < NBT; t++) {
            device const half *xb = x + (ulong)t * xs + i * 256;
            acc[t] += dot(w[0], float4(*(device const half4 *)(xb + p[0]))) + dot(w[1], float4(*(device const half4 *)(xb + p[1]))) +
                      dot(w[2], float4(*(device const half4 *)(xb + p[2]))) + dot(w[3], float4(*(device const half4 *)(xb + p[3])));
        }
    }
}

struct qmv_ggm_args {
    uint N[4];   // 每個矩陣的列數（沒用的是 0）
    uint off[4]; // 每個矩陣的起始全域列號
    uint total;
    uint xstride; // 每個 token 的輸入間距（元素）；輸出間距 = 各矩陣的 N
};

template <uint NBT>
inline void ggm_body(device const uchar *W0, device const uchar *W1, device const uchar *W2, device const uchar *W3, device const half *x,
                     device YT *y0, device YT *y1, device YT *y2, device YT *y3, constant qmv_ggm_args &a, uint tg, uint sg, uint lane) {
    const uint grow = tg * 8 + sg;
    if (grow >= a.total) return;
    uint j = 0;
    while (j + 1 < NMAT && grow >= a.off[j + 1] && a.N[j + 1] > 0) j++;
    const uint row = grow - a.off[j];
    float acc[NBT];
    for (uint t = 0; t < NBT; t++) acc[t] = 0;
    device YT *y;
    if (j == 0) { row_nb<GF0, NBT>(W0 + (ulong)row * (KD / 256 * ggm_bb<GF0>()), x, a.xstride, lane, acc); y = y0; }
#if NMAT > 1
    else if (j == 1) { row_nb<GF1, NBT>(W1 + (ulong)row * (KD / 256 * ggm_bb<GF1>()), x, a.xstride, lane, acc); y = y1; }
#endif
#if NMAT > 2
    else if (j == 2) { row_nb<GF2, NBT>(W2 + (ulong)row * (KD / 256 * ggm_bb<GF2>()), x, a.xstride, lane, acc); y = y2; }
#endif
#if NMAT > 3
    else { row_nb<GF3, NBT>(W3 + (ulong)row * (KD / 256 * ggm_bb<GF3>()), x, a.xstride, lane, acc); y = y3; }
#else
    else y = y0;
#endif
    for (uint t = 0; t < NBT; t++) {
        const float v = simd_sum(acc[t]);
        if (lane == 0) y[(ulong)t * a.N[j] + row] = YT(v);
    }
}

// grid：ceil(total / 8) × 256（每個 SIMD group 一列）
#define GGM_K(NAME, NBT)                                                                                                              \
    kernel void NAME(device const uchar *W0 [[buffer(0)]], device const uchar *W1 [[buffer(1)]], device const uchar *W2 [[buffer(2)]], \
                     device const uchar *W3 [[buffer(3)]], device const half *x [[buffer(4)]], device YT *y0 [[buffer(5)]],          \
                     device YT *y1 [[buffer(6)]], device YT *y2 [[buffer(7)]], device YT *y3 [[buffer(8)]],                         \
                     constant qmv_ggm_args &a [[buffer(9)]], uint tg [[threadgroup_position_in_grid]],                              \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                         \
        ggm_body<NBT>(W0, W1, W2, W3, x, y0, y1, y2, y3, a, tg, sg, lane);                                                            \
    }
#if NB > 0
GGM_K(qmv_ggm, NB)
#else
#define GGM_CAT2(a, b) a##_b##b
#define GGM_CAT(a, b) GGM_CAT2(a, b)
GGM_K(GGM_CAT(qmv_ggm, 1), 1)
GGM_K(GGM_CAT(qmv_ggm, 2), 2)
GGM_K(GGM_CAT(qmv_ggm, 3), 3)
GGM_K(GGM_CAT(qmv_ggm, 4), 4)
GGM_K(GGM_CAT(qmv_ggm, 5), 5)
GGM_K(GGM_CAT(qmv_ggm, 6), 6)
GGM_K(GGM_CAT(qmv_ggm, 7), 7)
GGM_K(GGM_CAT(qmv_ggm, 8), 8)
#endif

#ifdef GGM_TC
// ---------------------------------------------------------------- 矩陣單元版（驗證 NB 較大時）
// 純量版每個權重要對每個 token 各乘一次（還要各讀一次 x），NB = 8 時被算力 / L1 卡住。這裡權重解一次、直接放進
// SIMD group 的 cooperative tensor，跟 8 列輸入（不足 8 列的部分是緩衝區裡的舊資料，結果不寫出）用 tensor ops 相乘。
// 精度：權重拆成 hi = half(w)、lo = half(w − hi) 兩個 FP16 運算元各乘一次（FP32 累加），合計約 22 位元有效位數，
// 與純量版（FP32）相當；x 本來就是 FP16。
// 每個 threadgroup：32 個輸出列；NSG 個 SIMD group 輪流分 K（每次 32 個），最後在 threadgroup memory 加總。
// 主程式保證：每個矩陣的 N 是 32 的倍數、x 的列距 = K 且緩衝區至少 8 列。
// 實驗（engine 未使用，tools/ggmtest.c 的 TC=4 / 8 可測）：NB = 8、完整形狀時只比純量版快 0–22%（Q4_K / Q5_K 打平：
// 每 4 個值都要重解一次 scale，hi / lo 兩次運算也不便宜）。
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#define TC_MT 8
#define TC_NT 32
#define TC_KT 32
static_assert(KD % (TC_KT * NSG) == 0, "qmv_ggm_tc：K 必須是 32·NSG 的倍數");
static_assert(NB > 0, "qmv_ggm_tc：NB 要在編譯期決定");

// 一列第 k…k + 3 個值（k 是 4 的倍數）
template <int F> inline float4 deq4(device const uchar *row, uint k);

template <> inline float4 deq4<12>(device const uchar *row, uint k) { // Q4_K
    device const uchar *b = row + (k / 256) * 144;
    const uint j = (k % 256) / 32, l = k % 32;
    float s, m;
    kq_scale_min(b + 4, j, s, m);
    const uint q = *(device const uint *)(b + 16 + 32 * (j / 2) + l);
    return ldh(b) * s * ((j & 1) ? nib_hi(q) : nib_lo(q)) - ldh(b + 2) * m;
}

template <> inline float4 deq4<13>(device const uchar *row, uint k) { // Q5_K
    device const uchar *b = row + (k / 256) * 176;
    const uint j = (k % 256) / 32, l = k % 32;
    float s, m;
    kq_scale_min(b + 4, j, s, m);
    const uint q = *(device const uint *)(b + 48 + 32 * (j / 2) + l), h = *(device const uint *)(b + 16 + l);
    const uint q5 = (((j & 1) ? q >> 4 : q) & 0x0f0f0f0fu) | (((h >> j) & 0x01010101u) << 4);
    return ldh(b) * s * float4(as_type<uchar4>(q5)) - ldh(b + 2) * m;
}

template <> inline float4 deq4<14>(device const uchar *row, uint k) { // Q6_K
    device const uchar *b = row + (k / 256) * 210;
    const uint n = (k % 256) / 128, r = (k % 128) / 32, l = k % 32;
    const uint ql = ld4_a2(b + 64 * n + 32 * (r % 2) + l), qh = ld4_a2(b + 128 + 32 * n + l);
    const uint q = ((ql >> (4 * (r / 2))) & 0x0f0f0f0fu) | (((qh >> (2 * r)) & 0x03030303u) << 4);
    const float sc = float(((device const char *)b)[192 + 8 * n + 2 * r + l / 16]);
    return ldh(b + 208) * sc * (float4(as_type<uchar4>(q)) - 32.0f);
}

template <> inline float4 deq4<23>(device const uchar *row, uint k) { // IQ4_XS
    device const uchar *b = row + (k / 256) * 136;
    const uint ib = (k % 256) / 32, l = k % 32;
    const uint sh = uint(b[2]) | (uint(b[3]) << 8);
    const int ls = int(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4));
    const uint q = *(device const uint *)(b + 8 + 16 * ib + l % 16);
    const uchar4 v = as_type<uchar4>(l < 16 ? q & 0x0f0f0f0fu : (q >> 4) & 0x0f0f0f0fu);
    return ldh(b) * float(ls - 32) * float4(kvalues_iq4nl[v.x], kvalues_iq4nl[v.y], kvalues_iq4nl[v.z], kvalues_iq4nl[v.w]);
}

template <> inline float4 deq4<30>(device const uchar *row, uint k) { // BF16
    return float4(*(device const bfloat4 *)(row + 2 * k));
}

// 32 列（n0 起）× 8 個 token；結果寫進 part，回傳前已同步
template <int F>
inline void tc_rows(device const uchar *W, device const half *x, uint n0, uint sg, threadgroup float (*part)[TC_MT][TC_NT]) {
    constexpr ulong RB = KD / 256 * ggm_bb<F>();
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX((device half *)x, T2(KD, TC_MT));
    constexpr auto desc = matmul2d_descriptor(TC_MT, TC_NT, TC_KT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto rh = op.get_right_input_cooperative_tensor<half, half, float>();
    auto rl = op.get_right_input_cooperative_tensor<half, half, float>();
    auto x0 = tX.slice<TC_KT, TC_MT>(0, 0);
    auto acc = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(x0)>,
                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(rh)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint k0 = sg * TC_KT; k0 < KD; k0 += TC_KT * NSG) {
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rh.get_capacity(); i += 4) {
            const auto ix = rh.get_multidimensional_index(i); // (k, n)：連續 4 個 k（M5 的排列）
            const float4 w = deq4<F>(W + (ulong)(n0 + ix[1]) * RB, k0 + ix[0]);
            const half4 h = half4(w), l = half4(w - float4(h));
            rh[i] = h.x; rh[i + 1] = h.y; rh[i + 2] = h.z; rh[i + 3] = h.w;
            rl[i] = l.x; rl[i + 1] = l.y; rl[i + 2] = l.z; rl[i + 3] = l.w;
        }
        auto xs = tX.slice<TC_KT, TC_MT>(k0, 0);
        op.run(xs, rh, acc);
        op.run(xs, rl, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            const auto ix = acc.get_multidimensional_index(i);
            part[sg][ix[1]][ix[0]] = acc[i];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// grid：總列數 / 32 × 32·NSG
kernel void qmv_ggm_tc(device const uchar *W0 [[buffer(0)]], device const uchar *W1 [[buffer(1)]], device const uchar *W2 [[buffer(2)]],
                       device const uchar *W3 [[buffer(3)]], device const half *x [[buffer(4)]], device YT *y0 [[buffer(5)]],
                       device YT *y1 [[buffer(6)]], device YT *y2 [[buffer(7)]], device YT *y3 [[buffer(8)]],
                       constant qmv_ggm_args &a [[buffer(9)]], uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG][TC_MT][TC_NT];
    const uint grow = tg * TC_NT;
    uint j = 0;
    while (j + 1 < NMAT && grow >= a.off[j + 1] && a.N[j + 1] > 0) j++;
    const uint n0 = grow - a.off[j];
    device YT *y;
    if (j == 0) { tc_rows<GF0>(W0, x, n0, sg, part); y = y0; }
#if NMAT > 1
    else if (j == 1) { tc_rows<GF1>(W1, x, n0, sg, part); y = y1; }
#endif
#if NMAT > 2
    else if (j == 2) { tc_rows<GF2>(W2, x, n0, sg, part); y = y2; }
#endif
#if NMAT > 3
    else { tc_rows<GF3>(W3, x, n0, sg, part); y = y3; }
#else
    else y = y0;
#endif
    for (uint t = tid; t < NB * TC_NT; t += 32 * NSG) {
        const uint m = t / TC_NT, n = t % TC_NT;
        float v = 0;
        for (uint s = 0; s < NSG; s++) v += part[s][m][n];
        y[(ulong)m * a.N[j] + n0 + n] = YT(v);
    }
}
#endif
#endif // GGM_PF
