// dense prefill 的 INT8 矩陣乘法：權重是「每列對稱 8-bit」（MLX 8-bit 格式，每組 scale 相同、bias = −128·scale，
// requant.c 的 c8 模式），輸入每列量化成 8-bit。兩邊都直接當 uint8 的 tensor 運算元（不解碼、不經 threadgroup memory）：
//   xu = Qa + 128（Qa = rint(x / sa)，sa = max|x| / 127），q = w' + 128（w' = round(w / sw)）
//   acc = Σ xu · q = Σ Qa·w' + 128·ΣQa + 128·Σw' + K·128²   →   y = sa · sw · (acc − 128·ΣQa − 128·Σw' − K·16384)
// Σw' 每列在載入時算好（rsum）。INT32 不會溢位：K ≤ 10240 時 acc ≤ 10240·255² ≈ 6.7 億。
// 主程式注入：KD（K）、TM、TN、SG、KC；名稱改名
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// 每列一個 threadgroup（256）：x [T][KD] half → xu [T][KD] uint8、sa [T]、asum [T]（ΣQa）
kernel void i8_quant(device const half *x [[buffer(0)]], device uint8_t *xu [[buffer(1)]], device float *sa [[buffer(2)]],
                     device int *asum [[buffer(3)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    threadgroup int ipart[8];
    device const half *xr = x + (ulong)r * KD;
    float m = 0;
    for (uint i = tid; i < KD; i += 256) m = max(m, fabs(float(xr[i])));
    m = simd_max(m);
    if (lane == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = 0;
    for (uint k = 0; k < 8; k++) amax = max(amax, part[k]);
    const float sc = amax > 0 ? amax / 127.0f : 1.0f, inv = 1.0f / sc;
    int s = 0;
    for (uint i = tid; i < KD; i += 256) {
        const int q = (int)clamp(rint(float(xr[i]) * inv), -127.0f, 127.0f);
        xu[(ulong)r * KD + i] = (uint8_t)(q + 128);
        s += q;
    }
    s = simd_sum(s);
    if (lane == 0) ipart[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int t = 0;
        for (uint k = 0; k < 8; k++) t += ipart[k];
        sa[r] = sc;
        asum[r] = t;
    }
}

struct i8_args { uint M, N; };

// grid：(N / TN, ceil(M / TM)) × 32·SG
kernel void qmm_i8(device const uint8_t *xu [[buffer(0)]], device const uint8_t *W [[buffer(1)]], device const bfloat *S [[buffer(2)]],
                   device YT *Y [[buffer(4)]], constant i8_args &a [[buffer(5)]], device const float *sa [[buffer(6)]],
                   device const int *asum [[buffer(7)]], device const int *rsum [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]]) {
    const uint m0 = tg.y * TM, n0 = tg.x * TN;
    using T2 = dextents<int32_t, 2>;
    tensor<device uint8_t, T2, tensor_inline> tX((device uint8_t *)xu, T2(KD, a.M + TM)); // 多 TM 列：最後一塊超出的部分不寫出
    tensor<device uint8_t, T2, tensor_inline> tW((device uint8_t *)W, T2(KD, a.N));
    constexpr auto desc = matmul2d_descriptor(TM, TN, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto x0 = tX.slice(0, m0);
    auto w0 = tW.slice(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(x0), decltype(w0), int32_t>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    for (uint k0 = 0; k0 < KD; k0 += KC) {
        auto xs = tX.slice(k0, m0);
        auto ws = tW.slice(k0, n0);
        op.run(xs, ws, acc);
    }
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const uint n = n0 + ix[0], m = m0 + ix[1];
            if (m < a.M) {
                const int v = acc[i] - 128 * asum[m] - 128 * rsum[n] - (int)KD * 16384;
                Y[(ulong)m * a.N + n] = YT(float(v) * sa[m] * float(S[(ulong)n * (KD / 64)]));
            }
        }
}
