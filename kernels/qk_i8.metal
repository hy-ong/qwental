// QSA indexer 打分（prefill）的 INT8 版：QW_INT8 類別清單含 "qk" 才用（會改變精度，預設路徑仍是 qsa.metal 的 p_qsa_score）。
//   iq [T][IH][ID]（half）每個 (token, head) 一列、kblk [nb][ID]（half）每個區塊一列，各自對稱量化：
//     s = max|x| / 127（全 0 時 1），q = rint(x / s) ∈ [−127, 127]
//   acc_h[t][n] = Σ q_iq · q_k（INT8 × INT8 → INT32，tensor 單元；ID = 128 時 |acc| ≤ 128·127² ≈ 206 萬，不會溢位）
//   score[t][n] = Σ_h max(sq[t,h] · acc_h, 0) · sk[n] / √ID（sk ≥ 0，與 p_qsa_score 的 Σ_h max(iq·k, 0) / √ID 同義）
//   只寫 n < nvis(t) = (pos0 + t + 1) / CR（同 p_qsa_score）。
// 主程式注入：IH、ID、CR、TM、TN、SG
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
static_assert(ID == 128, "qk_quant：一個 SIMD group 一列（32 lane × half4）");

// 每列一個 SIMD group（threadgroup 256 = 8 列）。輸入第 r 列 → 輸出第 (r % G)·gs + r / G 列
// （iq：G = IH、gs = 每個 head 的列距，排成 [IH][gs][ID]，每個 head 一塊連續的左運算元；kblk：G = 1）
// grid：ceil(n / 8) × 256
kernel void qk_quant(device const half *x [[buffer(0)]], device int8_t *xq [[buffer(1)]], device float *xs [[buffer(2)]],
                     constant uint4 &a [[buffer(3)]], // n（列數）、G、gs（w 不用）
                     uint tgi [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tgi * 8 + sg;
    if (r >= a.x) return;
    const float4 v = float4(((device const half4 *)(x + (ulong)r * ID))[lane]);
    const float4 av = fabs(v);
    const float m = simd_max(max(max(av.x, av.y), max(av.z, av.w)));
    const float sc = m > 0 ? m / 127.0f : 1.0f, inv = 1.0f / sc;
    const ulong o = (ulong)(r % a.y) * a.z + r / a.y;
    ((device char4 *)(xq + o * ID))[lane] = char4(clamp(rint(v * inv), -127.0f, 127.0f));
    if (lane == 0) xs[o] = sc;
}

struct qk_args { uint pos0, T, sstride, qp, nbp; };

// 每個 threadgroup：TM 個 query × TN 個區塊，IH 個 head 各一次 matmul（K = ID 一次 run），4 份 INT32 累加器排列相同，
// 逐元素合併。整塊都超出 causal 範圍（n0 ≥ 這組 query 的最大 nvis）就直接返回。
// grid：(ceil(nbmax / TN), ceil(T / TM)) × 32·SG
static_assert(IH == 4, "qk_score_i8：4 個 head");
kernel void qk_score_i8(device const int8_t *q8 [[buffer(0)]], device const float *qs [[buffer(1)]],
                        device const int8_t *k8 [[buffer(2)]], device const float *ks [[buffer(3)]],
                        device float *score [[buffer(4)]], constant qk_args &a [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]]) {
    const uint m0 = tg.y * TM, n0 = tg.x * TN;
    if (m0 >= a.T) return;
    const uint tl = min((uint)TM, a.T - m0);
    if (n0 >= (a.pos0 + m0 + tl) / CR) return;
    using T2 = dextents<int32_t, 2>;
    // 左運算元每個 head [qp][ID]（qp ≥ T + TM：最後一塊超出的列讀得到、不寫出）；右運算元 [nbp][ID]（nbp ≥ nbmax + TN）
    tensor<device int8_t, T2, tensor_inline> tK((device int8_t *)k8, T2(ID, a.nbp));
    tensor<device int8_t, T2, tensor_inline> tQ0((device int8_t *)q8, T2(ID, a.qp));
    tensor<device int8_t, T2, tensor_inline> tQ1((device int8_t *)q8 + (ulong)a.qp * ID, T2(ID, a.qp));
    tensor<device int8_t, T2, tensor_inline> tQ2((device int8_t *)q8 + (ulong)a.qp * ID * 2, T2(ID, a.qp));
    tensor<device int8_t, T2, tensor_inline> tQ3((device int8_t *)q8 + (ulong)a.qp * ID * 3, T2(ID, a.qp));
    constexpr auto desc = matmul2d_descriptor(TM, TN, ID, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto kk = tK.slice(0, n0);
    auto x0 = tQ0.slice(0, m0), x1 = tQ1.slice(0, m0), x2 = tQ2.slice(0, m0), x3 = tQ3.slice(0, m0);
    auto c0 = op.get_destination_cooperative_tensor<decltype(x0), decltype(kk), int32_t>();
    auto c1 = op.get_destination_cooperative_tensor<decltype(x0), decltype(kk), int32_t>();
    auto c2 = op.get_destination_cooperative_tensor<decltype(x0), decltype(kk), int32_t>();
    auto c3 = op.get_destination_cooperative_tensor<decltype(x0), decltype(kk), int32_t>();
#pragma unroll
    for (uint16_t i = 0; i < c0.get_capacity(); ++i) { c0[i] = 0; c1[i] = 0; c2[i] = 0; c3[i] = 0; }
    op.run(x0, kk, c0);
    op.run(x1, kk, c1);
    op.run(x2, kk, c2);
    op.run(x3, kk, c3);
    const float rs = rsqrt(float(ID));
#pragma unroll
    for (uint16_t i = 0; i < c0.get_capacity(); ++i)
        if (c0.is_valid_element(i)) {
            auto ix = c0.get_multidimensional_index(i);
            const uint n = n0 + ix[0], m = m0 + ix[1];
            if (m >= a.T || n >= (a.pos0 + m + 1) / CR) continue;
            const float sc = max(float(c0[i]) * qs[m], 0.0f) + max(float(c1[i]) * qs[a.qp + m], 0.0f) +
                             max(float(c2[i]) * qs[2 * a.qp + m], 0.0f) + max(float(c3[i]) * qs[3 * a.qp + m], 0.0f);
            score[(ulong)m * a.sstride + n] = sc * ks[n] * rs;
        }
}
