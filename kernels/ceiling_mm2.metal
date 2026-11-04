// 第 0 階段補充：tensor 單元在「K 每 64 個一段、每段乘上 scale」時的速度。
// 這是 MLX 量化格式（group size 64）的矩陣乘法一定要做的事，用來量這個額外成本。
//
// 主程式注入：TA / TB / PA / PB（型別與指標型別）、PT（部分積型別：float 或 int32_t）、
//            TM / TN / SG（區塊與 SIMD group 數）、GK（每段的 K，通常 64）、mm2（kernel 名稱）
// 資料排列：A [M, K] row-major；B [N, K] row-major（K 連續，跟 MLX 權重一樣，所以右邊要轉置）；
//          S [N, K/GK] float（每段的 scale）；C [M, N] float
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void mm2(device PA *A [[buffer(0)]],
                device PB *B [[buffer(1)]],
                device const float *S [[buffer(2)]],
                device float *Cm [[buffer(3)]],
                constant uint3 &mnk [[buffer(4)]],
                uint2 tg [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]]) {
    const int M = mnk.x, N = mnk.y, K = mnk.z;
    const int NGK = K / GK;
    tensor<device TA, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(K, M));
    tensor<device TB, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(K, N));
    const int m0 = tg.y * TM, n0 = tg.x * TN;

#if GROUPED == 4
    // 分段、但用 multiply_accumulate 直接累加到同一個 cooperative tensor（沒有逐元素處理）
    constexpr auto desc = matmul2d_descriptor(TM, TN, GK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto a0 = tA.slice(0, m0);
    auto b0 = tB.slice(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), PT>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    for (int g = 0; g < NGK; g++) {
        auto ag = tA.slice(g * GK, m0);
        auto bg = tB.slice(g * GK, n0);
        op.run(ag, bg, acc);
    }
#elif GROUPED
    // 每段 GK 個 K：部分積 → 乘上該段 scale → 累加
    // GROUPED = 1：scale 從 device 讀；2：不乘 scale（只量分段本身的成本）；3：scale 先載入 threadgroup memory
#if GROUPED == 3
    threadgroup float Ssh[TN * 160]; // 最多 K = 10240
    for (uint e = tid; e < (uint)(TN * NGK); e += 32 * SG) Ssh[e] = S[(n0 + e / NGK) * NGK + e % NGK];
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    constexpr auto desc = matmul2d_descriptor(TM, TN, GK, false, true);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto a0 = tA.slice(0, m0);
    auto b0 = tB.slice(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    for (int g = 0; g < NGK; g++) {
        auto p = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), PT>();
#pragma unroll
        for (uint16_t i = 0; i < p.get_capacity(); ++i)
            if (p.is_valid_element(i)) p[i] = 0;
        auto ag = tA.slice(g * GK, m0);
        auto bg = tB.slice(g * GK, n0);
        op.run(ag, bg, p);
#pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) {
                auto ix = p.get_multidimensional_index(i);
#if GROUPED == 1
                acc[i] += float(p[i]) * S[(n0 + ix[0]) * NGK + g];
#elif GROUPED == 2
                (void)ix;
                acc[i] += float(p[i]);
#else
                acc[i] += float(p[i]) * Ssh[ix[0] * NGK + g];
#endif
            }
    }
#else
    // 對照組：整個 K 一次算完，不做分段縮放
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto a0 = tA.slice(0, m0);
    auto b0 = tB.slice(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), PT>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    op.run(a0, b0, acc);
#endif
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const int n = n0 + ix[0], m = m0 + ix[1];
            if (m < M && n < N) Cm[(ulong)m * N + n] = float(acc[i]);
        }
}
