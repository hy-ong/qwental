// DFlash2 drafter 的 Q4_0 matvec（NB ≤ 8 列輸入；權重保留打包檔原本的 Q4_0 數值：w = d·(q − 8)，每 32 個值一個 FP16 d）。
// 載入時（src/dflash.c）把 Q4_0 區塊拆成兩個陣列，數值與位元組數都不變，只是對齊：
//   QS [N][K/2]：每 32 個值 16 位元組，byte i 的低 nibble = 區塊第 i 個值、高 nibble = 第 i + 16 個（同 Q4_0）
//   D  [N][K/32] half
// 主程式注入：K、NB（輸入列數）、NSG（TC 版每個 threadgroup 的 SIMD group 數）、YT（輸出型別）；名稱改名。
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef YT
#define YT half
#endif
#define KB (K / 32) // 每列區塊數
#define MT 8        // TC：輸入列
#define NT 32       // TC：每個 threadgroup 的輸出列
static_assert(K % 512 == 0, "df_q4：K 必須是 512 的倍數");
static_assert(NB >= 1 && NB <= MT, "df_q4：NB 必須是 1–8");

struct df_q4_args {
    uint N[4];   // 每個矩陣的列數（沒用的是 0；TC 版必須是 32 的倍數）
    uint off[4]; // 每個矩陣的起始全域列號
    uint total;
    uint xstride; // 每列輸入的間距（元素）；輸出間距 = 各矩陣的 N
};

// ---------------------------------------------------------------- 矩陣單元版
// 每個 threadgroup 32 個輸出列 × 8 列輸入（不足 8 列的部分讀緩衝區裡的舊資料，結果不寫出：各列互不影響）。
// SIMD group 輪流取 K 方向的 KT 個值。權重 w = d·(q − 8) 在 FP32 是精確的（11 + 4 位元），右運算元用 hi = half(w)
// （捨入到 FP16，相對誤差 ≤ 2⁻¹²，與輸入 x 本身是 FP16 同級；遠小於原本展開成 MLX 8-bit 的重新量化誤差）。
// Q4_LO = 1 再加 lo = half(w − hi) 乘一次，合起來就是原本的權重，但慢約一半（7680 × 2560：31 → 48 µs），預設不用。
// 量過：每個 SIMD group 先乘出一個區塊再乘 d（右運算元 = q − 8，精確）因為每塊都要等矩陣單元的結果，慢 2–3 倍。
#ifndef Q4_LO
#define Q4_LO 0
#endif
#ifndef KT
#define KT 64
#endif
static_assert(KT % 32 == 0 && K % (KT * NSG) == 0, "df_q4：K 必須是 KT·NSG 的倍數");
// part：各 SIMD group 的部分和，回傳前已同步
inline void q4_block(device const uchar *QS, device const half *D, device const half *x, uint n0, uint sg, uint lane,
                     threadgroup float (*part)[MT][NT]) {
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX((device half *)x, T2(K, MT));
    constexpr auto desc = matmul2d_descriptor(MT, NT, KT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto rh = op.get_right_input_cooperative_tensor<half, half, float>();
#if Q4_LO
    auto rl = op.get_right_input_cooperative_tensor<half, half, float>();
#endif
    auto x0 = tX.slice<KT, MT>(0, 0);
    auto acc = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(x0)>,
                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(rh)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    // 右運算元的排列（M5；df_init 先用 df_q4_layout 核對）：lane l 的第 g 組（4 個連續 k）在
    //   k = 4·(l % 2) + 8·((l / 8) % 2) + 16·(g / 4)、n = (l % 8) / 2 + 4·(l / 16) + 8·(g % 4)
    // 所以區塊內第 k 與 k + 16 個（同一個 byte 的低、高 nibble）在同一個 lane：每個 word 讀一次填兩組
    const uint kofs = 4 * (lane % 2) + 8 * ((lane / 8) % 2), nb = (lane % 8) / 2 + 4 * (lane / 16);
    for (uint k0 = sg * KT; k0 < K; k0 += KT * NSG) {
#pragma clang loop unroll(full)
        for (uint b2 = 0; b2 < KT / 32; b2++) {
            const uint kb = k0 / 32 + b2;
#pragma clang loop unroll(full)
            for (uint nq = 0; nq < 4; nq++) {
                const ulong row = n0 + nb + 8 * nq;
                const uint w = *(device const uint *)(QS + row * (K / 2) + kb * 16 + kofs);
                const float d = float(D[row * KB + kb]);
                const float4 vl = d * (float4(as_type<uchar4>(w & 0x0f0f0f0fu)) - 8.0f);
                const float4 vh = d * (float4(as_type<uchar4>((w >> 4) & 0x0f0f0f0fu)) - 8.0f);
                const uint il = 4 * (8 * b2 + nq), ih = il + 16; // 低 nibble：g = 8·b2 + nq；高 nibble：g + 4
                const half4 hl = half4(vl), hh = half4(vh);
                rh[il] = hl.x; rh[il + 1] = hl.y; rh[il + 2] = hl.z; rh[il + 3] = hl.w;
                rh[ih] = hh.x; rh[ih + 1] = hh.y; rh[ih + 2] = hh.z; rh[ih + 3] = hh.w;
#if Q4_LO
                const half4 ll = half4(vl - float4(hl)), lh = half4(vh - float4(hh));
                rl[il] = ll.x; rl[il + 1] = ll.y; rl[il + 2] = ll.z; rl[il + 3] = ll.w;
                rl[ih] = lh.x; rl[ih + 1] = lh.y; rl[ih + 2] = lh.z; rl[ih + 3] = lh.w;
#endif
            }
        }
        auto xs = tX.slice<KT, MT>(k0, 0);
        op.run(xs, rh, acc);
#if Q4_LO
        op.run(xs, rl, acc);
#endif
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            const auto ix = acc.get_multidimensional_index(i);
            part[sg][ix[1]][ix[0]] = acc[i];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// 核對上面假設的右運算元排列：out[0] = 不符的元素數（0 = 符合）。grid：1 × 32
kernel void df_q4_layout(device atomic_uint *out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr auto desc = matmul2d_descriptor(MT, NT, KT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto rh = op.get_right_input_cooperative_tensor<half, half, float>();
    uint bad = rh.get_capacity() == KT ? 0 : 1;
    for (uint16_t i = 0; i < rh.get_capacity(); i++) {
        const auto ix = rh.get_multidimensional_index(i);
        const uint g = i / 4;
        bad += uint(ix[0]) != 4 * (lane % 2) + 8 * ((lane / 8) % 2) + 16 * (g / 4) + i % 4 || uint(ix[1]) != (lane % 8) / 2 + 4 * (lane / 16) + 8 * (g % 4);
    }
    atomic_fetch_add_explicit(out, bad, memory_order_relaxed);
}

// 多矩陣（同一組輸入、1–4 個）。grid：total / 32 × 32·NSG
kernel void df_q4tc(device const uchar *W0 [[buffer(0)]], device const half *D0 [[buffer(1)]], device const uchar *W1 [[buffer(2)]],
                    device const half *D1 [[buffer(3)]], device const uchar *W2 [[buffer(4)]], device const half *D2 [[buffer(5)]],
                    device const uchar *W3 [[buffer(6)]], device const half *D3 [[buffer(7)]], device const half *x [[buffer(8)]],
                    device YT *y0 [[buffer(9)]], device YT *y1 [[buffer(10)]], device YT *y2 [[buffer(11)]], device YT *y3 [[buffer(12)]],
                    constant df_q4_args &a [[buffer(13)]], uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG][MT][NT];
    const uint grow = tg * NT;
    uint j = 0;
    while (j < 3 && a.N[j + 1] > 0 && grow >= a.off[j + 1]) j++;
    const uint n0 = grow - a.off[j];
    device const uchar *W = j == 0 ? W0 : j == 1 ? W1 : j == 2 ? W2 : W3;
    device const half *D = j == 0 ? D0 : j == 1 ? D1 : j == 2 ? D2 : D3;
    device YT *y = j == 0 ? y0 : j == 1 ? y1 : j == 2 ? y2 : y3;
    q4_block(W, D, x, n0, sg, tid % 32, part);
    for (uint t = tid; t < NB * NT; t += 32 * NSG) {
        const uint m = t / NT, n = t % NT;
        float v = 0;
        for (uint s = 0; s < NSG; s++) v += part[s][m][n];
        y[(ulong)m * a.N[j] + n0 + n] = YT(v);
    }
}

// gate / up 一起：同一個 threadgroup 算兩個矩陣的同 32 列，直接寫 silu(g)·u（half，同 df_swiglu 的算式）。
// grid：N / 32 × 32·NSG
kernel void df_q4tc_glu(device const uchar *Wg [[buffer(0)]], device const half *Dg [[buffer(1)]], device const uchar *Wu [[buffer(2)]],
                        device const half *Du [[buffer(3)]], device const half *x [[buffer(8)]], device half *y [[buffer(9)]],
                        constant df_q4_args &a [[buffer(13)]], uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float pg[NSG][MT][NT], pu[NSG][MT][NT];
    const uint n0 = tg * NT;
    q4_block(Wg, Dg, x, n0, sg, tid % 32, pg);
    q4_block(Wu, Du, x, n0, sg, tid % 32, pu);
    for (uint t = tid; t < NB * NT; t += 32 * NSG) {
        const uint m = t / NT, n = t % NT;
        float g = 0, u = 0;
        for (uint s = 0; s < NSG; s++) g += pg[s][m][n];
        for (uint s = 0; s < NSG; s++) u += pu[s][m][n];
        y[(ulong)m * a.N[0] + n0 + n] = half(g / (1.0f + exp(-g)) * u);
    }
}

// ---------------------------------------------------------------- ALU 版（精確：FP32 的 Σ d·(q − 8)·x）
// TC 版不能用時的退路（輸入不到 8 列可讀、N 不是 32 的倍數、或矩陣單元排列不符）。7680 × 2560、8 列：RPS 1 / 2 / 4 = 45 / 40 / 77 µs（TC 31 µs）。
// NB = 8 時每個權重要對每列輸入各讀一次 x，所以每個 SIMD group 算 RPS 列、x 讀進暫存器後共用。
// lane 每次負責 8 個值（一個 uint：區塊內 byte i…i + 3 的低 nibble = 第 i…i + 3 個、高 nibble = 第 i + 16…i + 19 個），
// 一次走 8 個區塊（256 個值）。grid：ceil(total / (8·RPS)) × 256（一個 SIMD group 的 RPS 列都屬於同一個矩陣：主程式保證各矩陣的 N 是 RPS 的倍數）
#ifndef RPS
#define RPS 2
#endif
kernel void df_q4mv(device const uchar *W0 [[buffer(0)]], device const half *D0 [[buffer(1)]], device const uchar *W1 [[buffer(2)]],
                    device const half *D1 [[buffer(3)]], device const uchar *W2 [[buffer(4)]], device const half *D2 [[buffer(5)]],
                    device const uchar *W3 [[buffer(6)]], device const half *D3 [[buffer(7)]], device const half *x [[buffer(8)]],
                    device YT *y0 [[buffer(9)]], device YT *y1 [[buffer(10)]], device YT *y2 [[buffer(11)]], device YT *y3 [[buffer(12)]],
                    constant df_q4_args &a [[buffer(13)]], uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint grow = (tg * 8 + sg) * RPS;
    if (grow >= a.total) return;
    uint j = 0;
    while (j < 3 && a.N[j + 1] > 0 && grow >= a.off[j + 1]) j++;
    const ulong row = grow - a.off[j];
    device const uchar *W = (j == 0 ? W0 : j == 1 ? W1 : j == 2 ? W2 : W3) + row * (K / 2);
    device const half *D = (j == 0 ? D0 : j == 1 ? D1 : j == 2 ? D2 : D3) + row * KB;
    device YT *y = j == 0 ? y0 : j == 1 ? y1 : j == 2 ? y2 : y3;
    const uint bi = lane / 4, q = lane % 4; // 這個 lane 在 8 個區塊中的第 bi 個、byte 4q…4q + 3
    float acc[RPS][NB];
    for (uint r = 0; r < RPS; r++)
        for (uint b = 0; b < NB; b++) acc[r][b] = 0;
    for (uint kb = bi; kb < KB; kb += 8) {
        half4 xl[NB], xh[NB];
        for (uint b = 0; b < NB; b++) {
            device const half4 *xp = (device const half4 *)(x + (ulong)b * a.xstride + kb * 32 + 4 * q);
            xl[b] = xp[0];
            xh[b] = xp[4];
        }
        for (uint r = 0; r < RPS; r++) {
            const uint w = *(device const uint *)(W + (ulong)r * (K / 2) + kb * 16 + 4 * q);
            const float4 l = float4(as_type<uchar4>(w & 0x0f0f0f0fu)) - 8, h = float4(as_type<uchar4>((w >> 4) & 0x0f0f0f0fu)) - 8;
            const float d = float(D[(ulong)r * KB + kb]);
            for (uint b = 0; b < NB; b++) acc[r][b] = fma(d, dot(l, float4(xl[b])) + dot(h, float4(xh[b])), acc[r][b]);
        }
    }
    for (uint r = 0; r < RPS; r++)
        for (uint b = 0; b < NB; b++) {
            const float v = simd_sum(acc[r][b]);
            if (lane == 0) y[(ulong)b * a.N[j] + row + r] = YT(v);
        }
}

// ---------------------------------------------------------------- draft vocab 的 lm_head（Q6_K 區塊，GGUF 原樣）：矩陣單元版
// 同上：每個 threadgroup 32 個輸出列 × 8 列輸入；權重 d·sc·(q − 32)（算法同 qmv_ggm 的 deq16<14>）在 FP32 算好，
// 捨入成 FP16 當右運算元（Q6_LO = 1 再加 lo 一次）。N 不必是 32 的倍數：超出的列讀最後一列、不寫出。
// 量過（106299 × 2560、7 列）：qmv_ggm（ALU）約 0.67 ms、這裡 0.49 ms（約 460 GB/s）；加 lo 0.60 ms。
// grid：ceil(N / 32) × 32·NSG
#ifndef Q6_LO
#define Q6_LO 0
#endif
#define Q6RB (K / 256 * 210)
inline float4 q6_deq4(device const uchar *row, uint k) { // 一列第 k…k + 3 個值（k 是 4 的倍數）
    device const uchar *b = row + (k / 256) * 210;
    const uint n = (k % 256) / 128, r = (k % 128) / 32, l = k % 32;
    device const ushort *pl = (device const ushort *)(b + 64 * n + 32 * (r % 2) + l), *ph = (device const ushort *)(b + 128 + 32 * n + l);
    const uint ql = uint(pl[0]) | (uint(pl[1]) << 16), qh = uint(ph[0]) | (uint(ph[1]) << 16); // 區塊 210 位元組：只保證 2 位元組對齊
    const uint q = ((ql >> (4 * (r / 2))) & 0x0f0f0f0fu) | (((qh >> (2 * r)) & 0x03030303u) << 4);
    const float sc = float(((device const char *)b)[192 + 8 * n + 2 * r + l / 16]);
    return float(*(device const half *)(b + 208)) * sc * (float4(as_type<uchar4>(q)) - 32.0f);
}
kernel void df_q6tc(device const uchar *W [[buffer(0)]], device const half *x [[buffer(8)]], device YT *y [[buffer(9)]],
                    constant df_q4_args &a [[buffer(13)]], uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG][MT][NT];
    const uint n0 = tg * NT, N = a.N[0];
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX((device half *)x, T2(K, MT));
    constexpr auto desc = matmul2d_descriptor(MT, NT, KT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto rh = op.get_right_input_cooperative_tensor<half, half, float>();
#if Q6_LO
    auto rl = op.get_right_input_cooperative_tensor<half, half, float>();
#endif
    auto x0 = tX.slice<KT, MT>(0, 0);
    auto acc = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(x0)>,
                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(rh)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint k0 = sg * KT; k0 < K; k0 += KT * NSG) {
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rh.get_capacity(); i += 4) {
            const auto ix = rh.get_multidimensional_index(i);
            const ulong row = min(n0 + uint(ix[1]), N - 1);
            const float4 v = q6_deq4(W + row * Q6RB, k0 + ix[0]);
            const half4 h = half4(v);
            rh[i] = h.x;
            rh[i + 1] = h.y;
            rh[i + 2] = h.z;
            rh[i + 3] = h.w;
#if Q6_LO
            const half4 l = half4(v - float4(h));
            rl[i] = l.x;
            rl[i + 1] = l.y;
            rl[i + 2] = l.z;
            rl[i + 3] = l.w;
#endif
        }
        auto xs = tX.slice<KT, MT>(k0, 0);
        op.run(xs, rh, acc);
#if Q6_LO
        op.run(xs, rl, acc);
#endif
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            const auto ix = acc.get_multidimensional_index(i);
            part[sg][ix[1]][ix[0]] = acc[i];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < NB * NT; t += 32 * NSG) {
        const uint m = t / NT, n = t % NT;
        if (n0 + n >= N) continue;
        float v = 0;
        for (uint s = 0; s < NSG; s++) v += part[s][m][n];
        y[(ulong)m * N + n0 + n] = YT(v);
    }
}

// ---------------------------------------------------------------- 主模型 lm_head（Q6_K、248320 列）：矩陣單元版，K 方向一次 128 個值
// 同 df_q6tc（每個 threadgroup 32 列 × 8 列輸入、權重 d·sc·(q − 32) 在 FP32 算好再捨入成 FP16），但 KT = 128 剛好是 Q6_K 的半個超區塊：
// 照右運算元的排列（M5；主程式先用 df_q6b_layout 核對），lane 的 32 組（每組 4 個連續 k）是 4 列 × 位置 l、l + 16 × r = 0…3，
// 同一列、同一個 l 的 4 個 r 共用 ql 的兩個 word 與 qh 的一個 word（df_q6tc 每 4 個值各讀一次：讀取指令多 4 倍）。
// 各列輸入互不影響：decode（NB = 1）與驗證（NB 個 token）每個 token 的算式相同。grid：ceil(N / 32) × 32·Q6B_NSG
#ifndef Q6B_NSG
#define Q6B_NSG 4
#endif
#define Q6B_KT 128
static_assert(K % (Q6B_KT * Q6B_NSG) == 0, "df_q6tcb：K 必須是 128·Q6B_NSG 的倍數");
inline uint q6b_ld4(device const uchar *p) { // 4 位元組，只保證 2 位元組對齊（Q6_K 區塊 210 B）
    device const ushort *u = (device const ushort *)p;
    return uint(u[0]) | (uint(u[1]) << 16);
}
// 核對右運算元的排列：lane l 的第 g 組在 k = 4·(l % 2) + 8·((l / 8) % 2) + 16·(g / 4)、n = (l % 8) / 2 + 4·(l / 16) + 8·(g % 4)。
// out[0] = 不符的元素數（0 = 符合）。grid：1 × 32
kernel void df_q6b_layout(device atomic_uint *out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr auto desc = matmul2d_descriptor(MT, NT, Q6B_KT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto rh = op.get_right_input_cooperative_tensor<half, half, float>();
    uint bad = rh.get_capacity() == Q6B_KT ? 0 : 1;
    for (uint16_t i = 0; i < rh.get_capacity(); i++) {
        const auto ix = rh.get_multidimensional_index(i);
        const uint g = i / 4;
        bad += uint(ix[0]) != 4 * (lane % 2) + 8 * ((lane / 8) % 2) + 16 * (g / 4) + i % 4 || uint(ix[1]) != (lane % 8) / 2 + 4 * (lane / 16) + 8 * (g % 4);
    }
    atomic_fetch_add_explicit(out, bad, memory_order_relaxed);
}
kernel void df_q6tcb(device const uchar *W [[buffer(0)]], device const half *x [[buffer(8)]], device YT *y [[buffer(9)]],
                     constant df_q4_args &a [[buffer(13)]], uint tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[Q6B_NSG][MT][NT];
    const uint n0 = tg * NT, N = a.N[0], lane = tid % 32;
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX((device half *)x, T2(K, MT));
    constexpr auto desc = matmul2d_descriptor(MT, NT, Q6B_KT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto rh = op.get_right_input_cooperative_tensor<half, half, float>();
    auto x0 = tX.slice<Q6B_KT, MT>(0, 0);
    auto acc = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(x0)>,
                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(rh)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    const uint c = 4 * (lane % 2) + 8 * ((lane / 8) % 2), nb = (lane % 8) / 2 + 4 * (lane / 16);
    device const uchar *wr[4];
#pragma clang loop unroll(full)
    for (uint q = 0; q < 4; q++) wr[q] = W + (ulong)min(n0 + nb + 8 * q, N - 1) * Q6RB; // 超出的列讀最後一列、不寫出
    for (uint k0 = sg * Q6B_KT; k0 < K; k0 += Q6B_KT * Q6B_NSG) {
        const uint hn = (k0 % 256) / 128;
#pragma clang loop unroll(full)
        for (uint q = 0; q < 4; q++) {
            device const uchar *b = wr[q] + (k0 / 256) * 210;
            const float d = float(*(device const half *)(b + 208));
#pragma clang loop unroll(full)
            for (uint hl = 0; hl < 2; hl++) {
                const uint l = c + 16 * hl;
                const uint ql0 = q6b_ld4(b + 64 * hn + l), ql1 = q6b_ld4(b + 64 * hn + 32 + l), qh = q6b_ld4(b + 128 + 32 * hn + l);
#pragma clang loop unroll(full)
                for (uint r = 0; r < 4; r++) {
                    const uint qv = (((r % 2 ? ql1 : ql0) >> (4 * (r / 2))) & 0x0f0f0f0fu) | (((qh >> (2 * r)) & 0x03030303u) << 4);
                    const float sc = float(((device const char *)b)[192 + 8 * hn + 2 * r + hl]);
                    const half4 h = half4(d * sc * (float4(as_type<uchar4>(qv)) - 32.0f)); // 同 q6_deq4 的算式
                    const uint i = 4 * (4 * (2 * r + hl) + q); // 第 g = 4·j + q 組，j = 2r + hl（k = l + 32r）
                    rh[i] = h.x;
                    rh[i + 1] = h.y;
                    rh[i + 2] = h.z;
                    rh[i + 3] = h.w;
                }
            }
        }
        auto xs = tX.slice<Q6B_KT, MT>(k0, 0);
        op.run(xs, rh, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            const auto ix = acc.get_multidimensional_index(i);
            part[sg][ix[1]][ix[0]] = acc[i];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < NB * NT; t += 32 * Q6B_NSG) {
        const uint m = t / NT, n = t % NT;
        if (n0 + n >= N) continue;
        float v = 0;
        for (uint s = 0; s < Q6B_NSG; s++) v += part[s][m][n];
        y[(ulong)m * N + n0 + n] = YT(v);
    }
}
