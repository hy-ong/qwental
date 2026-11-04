// MoE prefill 的 grouped qmm，INT8 輸入（A8）：思路參考 oMLX 的 m5_gather_qmm_a8（Apache-2.0），程式碼是我們自己的。
//   輸入量化（a8_quant）：每列 scale = max|x| / 127，Qa = rint(x / scale)（INT8），Ra = 每 64 個的 Qa 總和
//   GEMM：4-bit 權重碼（0..15）直接當 INT8 右運算元，INT8 × INT8 → INT32，每組 64 個 K 之後校正：
//         Cf += s[n, g] · acc + z[n, g] · Ra[m, g]；最後 y[m, n] = Sa[m] · Cf
//   （x · w = Σ x (s q + z) = s Σ x q + z Σ x，x ≈ Sa · Qa）
// 主程式注入：KD（K）、名稱改名；權重 MLX affine 4-bit、group 64、[E, N, K / 8]
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define GS 64
#ifndef YT
#define YT half
#endif
#define NG (KD / GS)
#define ROWW (KD / 8)

// 每列一個 threadgroup（256）：x [P][KD] half → qa [P][KD] int8、sa [P] float、ra [P][NG] short
kernel void a8_quant(device const half *x [[buffer(0)]], device int8_t *qa [[buffer(1)]], device float *sa [[buffer(2)]],
                     device short *ra [[buffer(3)]], constant uint &P [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    if (r >= P) return;
    device const half *xr = x + (ulong)r * KD;
    float m = 0;
    for (uint i = tid; i < KD; i += 256) m = max(m, fabs(float(xr[i])));
    m = simd_max(m);
    if (lane == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = 0;
    for (uint k = 0; k < 8; k++) amax = max(amax, part[k]);
    const float sc = amax > 0 ? amax / 127.0f : 1.0f, inv = 1.0f / sc;
    if (tid == 0) sa[r] = sc;
    // 每個 SIMD group 一次處理一組 64 個（每個 lane 2 個）
    for (uint g = sg; g < NG; g += 8) {
        const uint i = g * GS + lane * 2;
        const int q0 = (int)clamp(rint(float(xr[i]) * inv), -127.0f, 127.0f);
        const int q1 = (int)clamp(rint(float(xr[i + 1]) * inv), -127.0f, 127.0f);
        qa[(ulong)r * KD + i] = (int8_t)q0;
        qa[(ulong)r * KD + i + 1] = (int8_t)q1;
        const int s = simd_sum(q0 + q1);
        if (lane == 0) ra[(ulong)r * NG + g] = (short)s;
    }
}

// grid：(N / 64, tiles) × 128（4 個 SIMD group：2 × 2，每個 32 列 × 32 欄）。tiles[t] = {expert, 起始列, 結束列}
kernel void qmm_a8(device const int8_t *qa [[buffer(0)]], device const uint *W [[buffer(1)]], device const bfloat *S [[buffer(2)]],
                   device const bfloat *Z [[buffer(3)]], device half *Y [[buffer(4)]], constant uint2 &MN [[buffer(5)]],
                   device const uint4 *tiles [[buffer(6)]], device const float *sa [[buffer(7)]], device const short *ra [[buffer(8)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint tid [[thread_index_in_threadgroup]]) {
    // 這個 threadgroup 的 64 欄 scale / bias、64 列的 Ra，先搬進 threadgroup memory（校正時每組都要讀）
    threadgroup float sS[NG * 64], sZ[NG * 64], sR[NG * 64];
    const uint4 tile = tiles[tg.y];
    if (tile.z <= tile.y) return; // 整個 threadgroup 一起離開
    const uint N = MN.y;
    const ulong e = tile.x;
    W += e * N * ROWW;
    S += e * N * NG;
    Z += e * N * NG;
    const uint nb = tg.x * 64;
    for (uint q = tid; q < NG * 64; q += 128) {
        const uint c = q % 64, g = q / 64; // 第 c 欄 / 列、第 g 組
        sS[g * 64 + c] = float(S[(ulong)(nb + c) * NG + g]);
        sZ[g * 64 + c] = float(Z[(ulong)(nb + c) * NG + g]);
        const uint m = tile.y + c;
        sR[g * 64 + c] = m < tile.z ? float(ra[(ulong)m * NG + g]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint mo = (sg / 2) * 32, m0 = tile.y + mo, m_end = tile.z;
    if (m0 >= m_end) return;
    const uint no = (sg % 2) * 32, n0 = nb + no;

    constexpr auto dM = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto dA = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dM, execution_simdgroup> opM;
    matmul2d<dA, execution_simdgroup> opA;
    using T2 = dextents<int32_t, 2>;
    tensor<device int8_t, T2, tensor_inline> tA((device int8_t *)qa, T2(KD, MN.x + 32)); // 多 32 列：最後一塊可能超出（不寫出）
    auto ct_b = opA.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto a0 = tA.slice<16, 16>(0, m0);
    auto acc0 = opA.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(a0)>, __tensor_ops_detail::__remove_addrspace_t<decltype(ct_b)>, int32_t>();
    auto acc1 = opA.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(a0)>, __tensor_ops_detail::__remove_addrspace_t<decltype(ct_b)>, int32_t>();
    constexpr int CAP = 16 * 32 / 32; // 每個 lane 的元素數（16 × 32 / 32 個 lane）
    float c0[CAP], c1[CAP];
    ushort nl[CAP], ml[CAP]; // 每個元素在 tile 裡的欄、列（迴圈外先算好）
    for (uint16_t i = 0; i < CAP; i++) {
        c0[i] = 0; c1[i] = 0;
        const auto ix = acc0.get_multidimensional_index(i);
        nl[i] = (ushort)(no + ix[0]);
        ml[i] = (ushort)(mo + ix[1]);
    }

    for (uint g = 0; g < NG; g++) {
        for (uint t = 0; t < 4; t++) { // 一組 64 = 4 段 16
            const uint k0 = g * GS + t * 16;
            // 右運算元：每個 lane 連續 4 個元素 = 同一欄 n 的連續 4 個 k（M5 的排列）
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < ct_b.get_capacity(); i += 4) {
                const auto ix = ct_b.get_multidimensional_index(i); // (k, n)
                const uint k = k0 + ix[0], n = n0 + ix[1];
                const uint word = W[(ulong)n * ROWW + k / 8] >> (4 * (k % 8));
                ct_b[i] = (int8_t)(word & 15);
                ct_b[i + 1] = (int8_t)((word >> 4) & 15);
                ct_b[i + 2] = (int8_t)((word >> 8) & 15);
                ct_b[i + 3] = (int8_t)((word >> 12) & 15);
            }
            auto x0 = tA.slice<16, 16>(k0, m0);
            auto x1 = tA.slice<16, 16>(k0, m0 + 16);
            if (t == 0) { opM.run(x0, ct_b, acc0); opM.run(x1, ct_b, acc1); }
            else { opA.run(x0, ct_b, acc0); opA.run(x1, ct_b, acc1); }
        }
        // 校正（threadgroup memory）
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < CAP; ++i) {
            const uint q = g * 64 + nl[i];
            const float sv = sS[q], zv = sZ[q];
            c0[i] += sv * float(acc0[i]) + zv * sR[g * 64 + ml[i]];
            c1[i] += sv * float(acc1[i]) + zv * sR[g * 64 + ml[i] + 16];
        }
    }
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        if (!acc0.is_valid_element(i)) continue;
        const auto ix = acc0.get_multidimensional_index(i);
        const uint n = n0 + ix[0], ma = m0 + ix[1], mb = ma + 16;
        if (ma < m_end) Y[(ulong)ma * N + n] = half(c0[i] * sa[ma]);
        if (mb < m_end) Y[(ulong)mb * N + n] = half(c1[i] * sa[mb]);
    }
}


// ---------------------------------------------------------------- uint8 × uint4b（權重不解碼）
// 輸入：xu = Qa + 128（uint8），sa、ra 同 a8_quant。權重 4-bit 碼直接當 uint4b_format 的 device tensor。
// 每組 64 個 K：acc = Σ xu·q = Σ Qa·q + 128·Σq（Σq 由每個 lane 對自己那欄即時加總）
//   Cf += s·(acc − 128·Σq) + z·Ra；y = sa · Cf
kernel void u8_quant(device const half *x [[buffer(0)]], device uint8_t *xu [[buffer(1)]], device float *sa [[buffer(2)]],
                     device short *ra [[buffer(3)]], constant uint &P [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    if (r >= P) return;
    device const half *xr = x + (ulong)r * KD;
    float m = 0;
    for (uint i = tid; i < KD; i += 256) m = max(m, fabs(float(xr[i])));
    m = simd_max(m);
    if (lane == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = 0;
    for (uint k = 0; k < 8; k++) amax = max(amax, part[k]);
    const float sc = amax > 0 ? amax / 127.0f : 1.0f, inv = 1.0f / sc;
    if (tid == 0) sa[r] = sc;
    for (uint g = sg; g < NG; g += 8) {
        const uint i = g * GS + lane * 2;
        const int q0 = (int)clamp(rint(float(xr[i]) * inv), -127.0f, 127.0f);
        const int q1 = (int)clamp(rint(float(xr[i + 1]) * inv), -127.0f, 127.0f);
        xu[(ulong)r * KD + i] = (uint8_t)(q0 + 128);
        xu[(ulong)r * KD + i + 1] = (uint8_t)(q1 + 128);
        const int s = simd_sum(q0 + q1);
        if (lane == 0) ra[(ulong)r * NG + g] = (short)s;
    }
}

// grid：(N / 64, tiles) × 128（2 × 2 個 SIMD group，各 32 列 × 32 欄）
kernel void qmm_u4(device const uint8_t *xu [[buffer(0)]], device const uint *W [[buffer(1)]], device const bfloat *S [[buffer(2)]],
                   device const bfloat *Z [[buffer(3)]], device YT *Y [[buffer(4)]], constant uint2 &MN [[buffer(5)]],
                   device const uint4 *tiles [[buffer(6)]], device const float *sa [[buffer(7)]], device const short *ra [[buffer(8)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sS[NG * 64], sZ[NG * 64], sR[NG * 64];
    threadgroup float sQ[4][32];
    const uint4 tile = tiles[tg.y];
    if (tile.z <= tile.y) return;
    const uint N = MN.y;
    const ulong e = tile.x;
    W += e * N * ROWW;
    S += e * N * NG;
    Z += e * N * NG;
    const uint nb = tg.x * 64;
    for (uint q = tid; q < NG * 64; q += 128) {
        const uint c = q % 64, g = q / 64;
        sS[g * 64 + c] = float(S[(ulong)(nb + c) * NG + g]);
        sZ[g * 64 + c] = float(Z[(ulong)(nb + c) * NG + g]);
        const uint m = tile.y + c;
        sR[g * 64 + c] = m < tile.z ? float(ra[(ulong)m * NG + g]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint mo = (sg / 2) * 32, m0 = tile.y + mo, m_end = tile.z;
    if (m0 >= m_end) return;
    const uint no = (sg % 2) * 32, n0 = nb + no;
    using T2 = dextents<int32_t, 2>;
    tensor<device uint8_t, T2, tensor_inline> tA((device uint8_t *)xu, T2(KD, MN.x + 64));
    tensor<device uint4b_format, T2, tensor_inline> tB((device uchar *)W, T2(KD, N));
    constexpr auto dM = matmul2d_descriptor(32, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<dM, execution_simdgroup> op;
    auto a0 = tA.slice<64, 32>(0, m0);
    auto b0 = tB.slice<64, 32>(0, n0);
    auto acc = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(a0)>,
                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(b0)>, int32_t>();
    constexpr int CAP = 32 * 32 / 32;
    float c[CAP];
    ushort nl[CAP], ml[CAP];
    for (uint16_t i = 0; i < CAP; i++) {
        c[i] = 0;
        const auto ix = acc.get_multidimensional_index(i);
        nl[i] = (ushort)ix[0];
        ml[i] = (ushort)(mo + ix[1]);
    }
    device const uint *wl = W + (ulong)(n0 + lane) * ROWW; // 這個 lane 負責第 n0 + lane 欄的 Σq
    for (uint g = 0; g < NG; g++) {
        auto xa = tA.slice<64, 32>(g * GS, m0);
        auto wb = tB.slice<64, 32>(g * GS, n0);
        op.run(xa, wb, acc);
        uint qs = 0;
        for (uint j = 0; j < 8; j++) {
            const uint w = wl[g * 8 + j];
            const uint lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu, t = lo + hi;
            qs += (t & 0xff) + ((t >> 8) & 0xff) + ((t >> 16) & 0xff) + (t >> 24);
        }
        sQ[sg][lane] = float(qs);
        simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < CAP; ++i) {
            const uint qn = g * 64 + no + nl[i];
            c[i] += sS[qn] * (float(acc[i]) - 128.0f * sQ[sg][nl[i]]) + sZ[qn] * sR[g * 64 + ml[i]];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < CAP; ++i) {
        const uint n = n0 + nl[i], m = tile.y + ml[i];
        if (m < m_end) Y[(ulong)m * N + n] = YT(c[i] * sa[m]);
    }
}
