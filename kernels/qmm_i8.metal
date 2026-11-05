// dense prefill 的 INT8 矩陣乘法：權重是「每列對稱 8-bit」（MLX 8-bit 格式，每組 scale 相同、bias = −128·scale，
// requant.c 的 c8 模式），輸入每列量化成 8-bit。兩邊都直接當 uint8 的 tensor 運算元（不解碼、不經 threadgroup memory）：
//   xu = Qa + 128（Qa = rint(x / sa)，sa = max|x| / 127），q = w' + 128（w' = round(w / sw)）
//   acc = Σ xu · q = Σ Qa·w' + 128·ΣQa + 128·Σw' + K·128²   →   y = sa · sw · (acc − 128·ΣQa − 128·Σw' − K·16384)
// Σw' 每列在載入時算好（rsum）。INT32 不會溢位：K ≤ 10240 時 acc ≤ 10240·255² ≈ 6.7 億。
// 主程式注入：KD（K）、TM、TN、SG、KC（0 = 整個 K 一次 run）、WT；名稱改名
// WT = 1：W 排成 [K][N]（GGUF 的 INT8 影子，qw_qlin.kn），右運算元不轉置，tensor 運算比 [N][K] 快約 12%；
//         WT = 0：[N][K]（MLX 的每列對稱 8-bit，decode 也讀同一份，不能改排列）
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#ifndef WT
#define WT 0
#endif
#define KCE (KC ? KC : KD) // KC = 0：整個 K 一次 run

// 每列一個 threadgroup（I8Q_NT 個執行緒）：x [T][KD] half → xu [T][KD] uint8、sa [T]、asum [T]（ΣQa）
// half4 一次讀進暫存器（每個執行緒 I8Q_V 組），max 與量化共用，不重讀 x。max、整數和與順序無關：結果與逐值版逐位元相同
#ifndef I8Q_NT
#define I8Q_NT 256
#endif
static_assert(KD % 4 == 0 && I8Q_NT % 32 == 0 && I8Q_NT <= 256, "i8_quant：KD / I8Q_NT");
#define I8Q_V ((KD / 4 + I8Q_NT - 1) / I8Q_NT)
kernel void i8_quant(device const half *x [[buffer(0)]], device uint8_t *xu [[buffer(1)]], device float *sa [[buffer(2)]],
                     device int *asum [[buffer(3)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[I8Q_NT / 32];
    threadgroup int ipart[I8Q_NT / 32];
    device const half4 *xr = (device const half4 *)(x + (ulong)r * KD);
    float4 v[I8Q_V];
    float m = 0;
#pragma unroll
    for (uint j = 0; j < I8Q_V; j++) {
        const uint i = tid + j * I8Q_NT;
        v[j] = i < KD / 4 ? float4(xr[i]) : float4(0);
        const float4 a = fabs(v[j]);
        m = max(m, max(max(a.x, a.y), max(a.z, a.w)));
    }
    m = simd_max(m);
    if (lane == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = 0;
    for (uint k = 0; k < I8Q_NT / 32; k++) amax = max(amax, part[k]);
    const float sc = amax > 0 ? amax / 127.0f : 1.0f, inv = 1.0f / sc;
    int s = 0;
    device uchar4 *dst = (device uchar4 *)(xu + (ulong)r * KD);
#pragma unroll
    for (uint j = 0; j < I8Q_V; j++) {
        const uint i = tid + j * I8Q_NT;
        if (i < KD / 4) {
            const int4 q = int4(clamp(rint(v[j] * inv), -127.0f, 127.0f));
            dst[i] = uchar4(q + 128);
            s += q.x + q.y + q.z + q.w;
        }
    }
    s = simd_sum(s);
    if (lane == 0) ipart[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int t = 0;
        for (uint k = 0; k < I8Q_NT / 32; k++) t += ipart[k];
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
#if WT // 權重 [K][N]（不轉置）
    tensor<device uint8_t, T2, tensor_inline> tW((device uint8_t *)W, T2(a.N, KD));
    constexpr auto desc = matmul2d_descriptor(TM, TN, KCE, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
#else
    tensor<device uint8_t, T2, tensor_inline> tW((device uint8_t *)W, T2(KD, a.N));
    constexpr auto desc = matmul2d_descriptor(TM, TN, KCE, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
#endif
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto x0 = tX.slice(0, m0);
#if WT
    auto w0 = tW.slice(n0, 0);
#else
    auto w0 = tW.slice(0, n0);
#endif
    auto acc = op.get_destination_cooperative_tensor<decltype(x0), decltype(w0), int32_t>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    for (uint k0 = 0; k0 < KD; k0 += KCE) {
        auto xs = tX.slice(k0, m0);
#if WT
        auto ws = tW.slice(n0, k0);
#else
        auto ws = tW.slice(k0, n0);
#endif
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
