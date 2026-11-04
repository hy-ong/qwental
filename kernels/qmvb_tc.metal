// 驗證（一次 NB ≤ 8 個 token）的量化 matvec，用 tensor ops 的矩陣單元算：權重直接解碼進 SIMD group 的
// cooperative tensor（同 qmm.metal 的 REGDQ），跟 8 列輸入（不足 8 列的部分是緩衝區裡的舊資料，結果不寫出）相乘。
// 一般 ALU 版（qmvb.metal）每個權重要對每個 token 各算一次，NB = 8 時被算力卡住；這裡每個權重只解一次。
// 每個 threadgroup：32 個輸出列；NSG 個 SIMD group 輪流分 K，最後在 threadgroup memory 加總。
// 主程式注入：BITS（4 / 8）、K、NB、NSG、YT；名稱改名。主程式保證：N 是 32 的倍數、K 是 32·NSG 的倍數、
// x 的列距 = K、x 緩衝區至少 8 列（VBMAX）
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef YT
#define YT half
#endif
#define GS 64
#define VPW (32 / BITS)
#define ROWW (K * BITS / 32)
#define NG (K / GS)
#define MT 8  // 輸入列（token）
#define NT 32 // 每個 threadgroup 的輸出列
#define KT 32 // 每次 K
static_assert(BITS == 4 || BITS == 8, "qmvb_tc：只支援 4 / 8-bit");
static_assert(K % (KT * NSG) == 0, "qmvb_tc：K 必須是 32·NSG 的倍數");
static_assert(NB >= 1 && NB <= MT, "qmvb_tc：NB 必須是 1–8");

struct qmvb_args {
    uint N;
    uint xstride;
    uint ystride;
    uint pad;
};

struct qmvb_multi_args {
    uint N[4];
    uint off[4];
    uint total;
    uint xstride;
};

// 一個 threadgroup 的 32 列 × 8 個 token；結果寫進 part，回傳前已同步
inline void tc_block(device const uint *W, device const bfloat *S, device const bfloat *Z, device const half *x, uint n0, uint sg,
                     threadgroup float (*part)[MT][NT]) {
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX((device half *)x, T2(K, MT));
    constexpr auto desc = matmul2d_descriptor(MT, NT, KT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto rb = op.get_right_input_cooperative_tensor<half, half, float>();
    auto x0 = tX.slice<KT, MT>(0, 0);
    auto acc = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(x0)>,
                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(rb)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    // SIMD group 輪流取相鄰的 32 個 k：同一時間 NSG 個 SIMD group 一起讀每列連續的 32·NSG 個權重
    for (uint k0 = sg * KT; k0 < K; k0 += KT * NSG) {
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rb.get_capacity(); i += 4) {
            const auto ix = rb.get_multidimensional_index(i); // (k, n)：連續 4 個 k（M5 的排列）
            const uint k = k0 + ix[0];
            const ulong row = n0 + ix[1];
            const uint word = W[row * ROWW + k / VPW] >> (BITS * (k % VPW));
            const float s = float(S[row * NG + k / GS]), z = float(Z[row * NG + k / GS]);
#if BITS == 4
            rb[i] = half(float(word & 15) * s + z);
            rb[i + 1] = half(float((word >> 4) & 15) * s + z);
            rb[i + 2] = half(float((word >> 8) & 15) * s + z);
            rb[i + 3] = half(float((word >> 12) & 15) * s + z);
#else
            rb[i] = half(float(word & 255) * s + z);
            rb[i + 1] = half(float((word >> 8) & 255) * s + z);
            rb[i + 2] = half(float((word >> 16) & 255) * s + z);
            rb[i + 3] = half(float(word >> 24) * s + z);
#endif
        }
        auto xs = tX.slice<KT, MT>(k0, 0);
        op.run(xs, rb, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            const auto ix = acc.get_multidimensional_index(i);
            part[sg][ix[1]][ix[0]] = acc[i];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// grid：N / 32 × 32·NSG
kernel void qmvb_tc(device const uint *W [[buffer(0)]], device const bfloat *S [[buffer(1)]], device const bfloat *Z [[buffer(2)]],
                    device const half *x [[buffer(3)]], device YT *y [[buffer(4)]], constant qmvb_args &a [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG][MT][NT];
    const uint n0 = tg * NT;
    tc_block(W, S, Z, x, n0, sg, part);
    for (uint t = tid; t < NB * NT; t += 32 * NSG) {
        const uint m = t / NT, n = t % NT;
        float v = 0;
        for (uint s = 0; s < NSG; s++) v += part[s][m][n];
        y[(ulong)m * a.ystride + n0 + n] = YT(v);
    }
}

// 多矩陣（同一組輸入、最多 4 個、每個列數都是 32 的倍數）。grid：總列數 / 32 × 32·NSG
kernel void qmvb_tc_multi(device const uint *W0 [[buffer(0)]], device const bfloat *S0 [[buffer(1)]], device const bfloat *Z0 [[buffer(2)]],
                          device const uint *W1 [[buffer(3)]], device const bfloat *S1 [[buffer(4)]], device const bfloat *Z1 [[buffer(5)]],
                          device const uint *W2 [[buffer(6)]], device const bfloat *S2 [[buffer(7)]], device const bfloat *Z2 [[buffer(8)]],
                          device const uint *W3 [[buffer(9)]], device const bfloat *S3 [[buffer(10)]], device const bfloat *Z3 [[buffer(11)]],
                          device const half *x [[buffer(12)]], device YT *y0 [[buffer(13)]], device YT *y1 [[buffer(14)]],
                          device YT *y2 [[buffer(15)]], device YT *y3 [[buffer(16)]], constant qmvb_multi_args &a [[buffer(17)]],
                          uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG][MT][NT];
    const uint grow = tg * NT;
    uint j = 0;
    while (j < 3 && grow >= a.off[j + 1] && a.N[j + 1] > 0) j++;
    const uint n0 = grow - a.off[j];
    device const uint *W = j == 0 ? W0 : j == 1 ? W1 : j == 2 ? W2 : W3;
    device const bfloat *S = j == 0 ? S0 : j == 1 ? S1 : j == 2 ? S2 : S3;
    device const bfloat *Z = j == 0 ? Z0 : j == 1 ? Z1 : j == 2 ? Z2 : Z3;
    device YT *y = j == 0 ? y0 : j == 1 ? y1 : j == 2 ? y2 : y3;
    tc_block(W, S, Z, x, n0, sg, part);
    for (uint t = tid; t < NB * NT; t += 32 * NSG) {
        const uint m = t / NT, n = t % NT;
        float v = 0;
        for (uint s = 0; s < NSG; s++) v += part[s][m][n];
        y[(ulong)m * a.N[j] + n0 + n] = YT(v);
    }
}
