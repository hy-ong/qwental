// Prefill 用的量化矩陣乘法（qmm）：Y = X · Wᵀ
//   X：[M, K] half      Y：[M, N] half
//   W：[N, K] MLX 格式（uint32 從低位放無號 q，BITS = 4 或 8），S、Z：[N, K/64] bfloat，w = q·S + Z
//
// 做法（依第 0 階段量測：每段之後逐元素處理 cooperative tensor 非常慢，
//      但 multiply_accumulate 分段累加幾乎不掉速）：
//   每個 threadgroup 負責 TM×TN 的輸出；K 方向每次 KC 個：
//     1. 全體執行緒把 W 的 TN×KC 解量化成 FP16（已乘 scale、加 bias）放進 threadgroup memory
//     2. tensor 單元：acc += X[TM×KC] · Wtᵀ（multiply_accumulate）
//   全部跑完才逐元素寫出一次。
//
// 主程式注入：qmm（名稱）、BITS、KD（K）、TM、TN、SG、KC（64 的倍數）、MOE（0 / 1）
//
// MOE = 1（grouped，給 MoE prefill 用）：X 是依 expert 排序過的列（共 a.M 列），
//   tiles[tg.y] = {expert, 起始列, 結束列}；W/S/Z 是全部 expert 的權重 [E, N, …]，依 expert 位移。
//   區塊可能讀到下一個 expert 的列，但不會寫出（m < 結束列）。
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef YT
#define YT half // 輸出型別（router logits 用 float）
#endif
#define GS 64
#define NG (KD / GS)
#define ROWW (KD * BITS / 32)   // 每列幾個 uint32
#define VPW (32 / BITS)         // 每個 uint32 幾個值
#define NT (32 * SG)

struct qmm_args {
    uint M, N;
};

#ifdef IQF // MoE 的 routed expert 是 GGUF 區塊（type IQF），每列 IQRB 位元組；權重以 uchar 讀
#define WT uchar
#else
#define WT uint
#endif
kernel void qmm(device half *X [[buffer(0)]],
                device const WT *W [[buffer(1)]],
                device const bfloat *S [[buffer(2)]],
                device const bfloat *Z [[buffer(3)]],
                device YT *Y [[buffer(4)]],
                constant qmm_args &a [[buffer(5)]],
#if MOE
                device const uint4 *tiles [[buffer(6)]],
#endif
#ifdef GU // gate 與 up 一起算，直接輸出 silu(gate) · up（Y 是 H）：up 的權重在 7–9
                device const WT *Wu [[buffer(7)]],
                device const bfloat *Su [[buffer(8)]],
                device const bfloat *Zu [[buffer(9)]],
#endif
                uint2 tg [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]]) {
    const uint M = a.M;
#if MOE
    const uint4 tile = tiles[tg.y];
    const uint m0 = tile.y, m_end = tile.z;
    if (m_end <= m0) return; // 沒用到的區塊（區塊數是在 GPU 上算的，dispatch 時用上限）
    {
        const ulong e = tile.x;
#ifdef IQF
        W += e * a.N * IQRB;
#else
        W += e * a.N * ROWW;
        S += e * a.N * NG;
        Z += e * a.N * NG;
#ifdef GU
        Wu += e * a.N * ROWW;
        Su += e * a.N * NG;
        Zu += e * a.N * NG;
#endif
#endif
    }
#else
    const uint m0 = tg.y * TM, m_end = M;
#endif
    const uint n0 = tg.x * TN;
    threadgroup half Wt[TN * KC];
#ifdef GU
    threadgroup half Wt2[TN * KC];
#endif

    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX(X, T2(KD, M));
    tensor<threadgroup half, T2, tensor_inline> tW(Wt, T2(KC, TN));

    constexpr auto desc = matmul2d_descriptor(TM, TN, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto x0 = tX.slice(0, m0);
    auto acc = op.get_destination_cooperative_tensor<decltype(x0), decltype(tW), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
#ifdef GU
    tensor<threadgroup half, T2, tensor_inline> tW2(Wt2, T2(KC, TN));
    auto acc2 = op.get_destination_cooperative_tensor<decltype(x0), decltype(tW2), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc2.get_capacity(); ++i)
        if (acc2.is_valid_element(i)) acc2[i] = 0;
#endif

    constexpr uint WORDS = TN * KC / VPW; // 這一段權重有幾個 uint32
    for (uint k0 = 0; k0 < KD; k0 += KC) {
        // 1. 解量化到 threadgroup memory
#ifdef IQF
        for (uint q = tid; q < TN * (KC / 32); q += NT) { // 每個執行緒解一段 32 個值
            const uint n = q / (KC / 32), sub = q % (KC / 32);
            float v[32];
            dq32<IQF>(W + (ulong)(n0 + n) * IQRB, k0 / 32 + sub, v);
            threadgroup half *dst = Wt + n * KC + sub * 32;
            for (uint i = 0; i < 32; i++) dst[i] = half(v[i]);
        }
        if (false)
#endif
        for (uint w = tid; w < WORDS; w += NT) {
            const uint n = w / (KC / VPW), j = w % (KC / VPW);
            const uint k = k0 + j * VPW;
            const ulong row = n0 + n;
            const uint word = W[row * ROWW + k / VPW];
            const float s = float(S[row * NG + k / GS]);
            const float z = float(Z[row * NG + k / GS]);
            threadgroup half *dst = Wt + n * KC + j * VPW;
#pragma unroll
            for (uint i = 0; i < VPW; i++) dst[i] = half(float((word >> (BITS * i)) & ((1u << BITS) - 1)) * s + z);
#ifdef GU
            const uint word2 = Wu[row * ROWW + k / VPW];
            const float s2 = float(Su[row * NG + k / GS]);
            const float z2 = float(Zu[row * NG + k / GS]);
            threadgroup half *dst2 = Wt2 + n * KC + j * VPW;
#pragma unroll
            for (uint i = 0; i < VPW; i++) dst2[i] = half(float((word2 >> (BITS * i)) & ((1u << BITS) - 1)) * s2 + z2);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 2. 累加
        auto xs = tX.slice(k0, m0);
        op.run(xs, tW, acc);
#ifdef GU
        op.run(xs, tW2, acc2);
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const uint n = n0 + ix[0], m = m0 + ix[1];
#ifdef GU
            // 同 p_act2：gate、up 先轉成 half（和分開算時寫進 buffer 一樣），再 silu(g) · u
            if (m < m_end) Y[(ulong)m * a.N + n] = half(float(half(acc[i])) / (1.0f + exp(-float(half(acc[i])))) * float(half(acc2[i])));
#else
            if (m < m_end) Y[(ulong)m * a.N + n] = YT(acc[i]);
#endif
        }
}

#if MOE && defined(REGDQ) && !defined(IQF)
#define RN 32 // 只驗證過 32 × 32；改尺寸前先用短輸入驗證（錯的尺寸曾讓 GPU 卡死）
#define RK 32
static_assert(BITS == 4 && KD % 64 == 0, "qmm_reg：只支援 4-bit、K 為 64 的倍數");
// MoE grouped qmm，權重直接解碼進每個 SIMD group 的 tensor 運算元（不經 threadgroup memory、不需同步）。
// 每個 SIMD group：TM 列 × 32 欄；threadgroup 4 個 SIMD group = TM × 128 欄。每次 K 32。grid：(N / 128, tiles) × 128
kernel void qmm_reg(device half *X [[buffer(0)]], device const uint *W [[buffer(1)]], device const bfloat *S [[buffer(2)]],
                    device const bfloat *Z [[buffer(3)]], device YT *Y [[buffer(4)]], constant qmm_args &a [[buffer(5)]],
                    device const uint4 *tiles [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint4 tile = tiles[tg.y];
    const uint m0 = tile.y, m_end = tile.z;
    if (m_end <= m0) return;
    const ulong e = tile.x;
    W += e * a.N * ROWW;
    S += e * a.N * NG;
    Z += e * a.N * NG;
    const uint n0 = tg.x * (RN * 4) + sg * RN;
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX(X, T2(KD, a.M + TM));
    constexpr auto desc = matmul2d_descriptor(TM, RN, RK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto rb = op.get_right_input_cooperative_tensor<half, half, float>();
    auto x0 = tX.slice<RK, TM>(0, m0);
    auto acc = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(x0)>,
                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(rb)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint k0 = 0; k0 < KD; k0 += RK) {
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rb.get_capacity(); i += 4) {
            const auto ix = rb.get_multidimensional_index(i); // (k, n)：連續 4 個 k（M5 的排列）
            const uint k = k0 + ix[0];
            const ulong row = n0 + ix[1];
            const uint word = W[row * ROWW + k / VPW] >> (BITS * (k % VPW));
            const float s = float(S[row * NG + k / GS]), z = float(Z[row * NG + k / GS]);
            rb[i] = half(float(word & 15) * s + z);
            rb[i + 1] = half(float((word >> 4) & 15) * s + z);
            rb[i + 2] = half(float((word >> 8) & 15) * s + z);
            rb[i + 3] = half(float((word >> 12) & 15) * s + z);
        }
        auto xs = tX.slice<RK, TM>(k0, m0);
        op.run(xs, rb, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            const auto ix = acc.get_multidimensional_index(i);
            const uint n = n0 + ix[0], m = m0 + ix[1];
            if (m < m_end) Y[(ulong)m * a.N + n] = YT(acc[i]);
        }
}
#endif
