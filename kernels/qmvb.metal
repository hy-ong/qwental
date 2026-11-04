// 多個 token 一起的量化 matvec（給 MTP 驗證用）：權重讀一次、跟 NB 個輸入向量各算一次
// 排列：x [NB][K]（每個 token 連續）、y [NB][N]。權重格式同 qmv.metal（MLX affine、group 64）
// 主程式注入：BITS、K、NB、ST（scale 型別）、YT（輸出型別，預設 half）；kernel 名稱也由主程式改名
#include <metal_stdlib>
using namespace metal;

#ifndef YT
#define YT half
#endif
#define GS 64
#define CHUNK 16
#define NCHUNK (K / CHUNK)
#define ROWW (K * BITS / 32)
#define NG (K / GS)

#ifndef RPSB
#define RPSB 2 // dense 版每個 SIMD group 算幾列（共用 x）
#endif

// 一列、一個 chunk 對 NB 個 token 的部分內積：acc[b] += s·Σ q·x + z·Σ x
// 權重解開一次（16 個 float），x 逐 token 讀（不全部留在暫存器，NB 大時才不會溢出）
inline void row_chunk(device const uint *W, device const ST *S, device const ST *Z, ulong row, uint c, device const half *x,
                      uint xstride, thread float *acc) {
    const uint g = c / (GS / CHUNK);
    const float s = float(S[row * NG + g]), z = float(Z[row * NG + g]);
#if BITS == 4
    // 每個 uint32 有 8 個 nibble：偶數位置（低 nibble）與奇數位置分開轉成 float4
    const uint2 w = ((device const uint2 *)(W + row * ROWW))[c];
    const float4 e0 = float4(as_type<uchar4>(w.x & 0x0f0f0f0fu)), o0 = float4(as_type<uchar4>((w.x >> 4) & 0x0f0f0f0fu));
    const float4 e1 = float4(as_type<uchar4>(w.y & 0x0f0f0f0fu)), o1 = float4(as_type<uchar4>((w.y >> 4) & 0x0f0f0f0fu));
#else
    const uint2 wl = ((device const uint2 *)(W + row * ROWW))[2 * c], wh = ((device const uint2 *)(W + row * ROWW))[2 * c + 1];
    const float4 w0 = float4(as_type<uchar4>(wl.x)), w1 = float4(as_type<uchar4>(wl.y));
    const float4 w2 = float4(as_type<uchar4>(wh.x)), w3 = float4(as_type<uchar4>(wh.y));
#endif
    for (uint b = 0; b < NB; b++) {
        device const half4 *xp = (device const half4 *)(x + (ulong)b * xstride + c * CHUNK);
        const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
        const float xs = dot(x0 + x1 + x2 + x3, float4(1));
#if BITS == 4
        // 權重 n 對應 x[n]：byte i 的低 nibble = x[2i]、高 nibble = x[2i+1]
        const float d = dot(e0, float4(x0.xz, x1.xz)) + dot(o0, float4(x0.yw, x1.yw)) + dot(e1, float4(x2.xz, x3.xz)) +
                        dot(o1, float4(x2.yw, x3.yw));
#else
        const float d = dot(w0, x0) + dot(w1, x1) + dot(w2, x2) + dot(w3, x3);
#endif
        acc[b] += s * d + z * xs;
    }
}

inline void chunk_nb(device const uint *W, device const ST *S, device const ST *Z, ulong row, uint c, device const half *x,
                     uint xstride, thread float *acc) {
    row_chunk(W, S, Z, row, c, x, xstride, acc);
}

struct qmvb_args {
    uint N;       // 輸出列數
    uint xstride; // 每個 token 的輸入間距（元素）
    uint ystride; // 每個 token 的輸出間距（元素）
    uint pad;
};

// dense：每個 SIMD group RPSB 列（共用 x）。grid：ceil(N / (8·RPSB)) × 256
kernel void qmvb(device const uint *W [[buffer(0)]], device const ST *S [[buffer(1)]], device const ST *Z [[buffer(2)]],
                 device const half *x [[buffer(3)]], device YT *y [[buffer(4)]], constant qmvb_args &a [[buffer(5)]],
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * 8 + sg) * RPSB;
    if (row0 >= a.N) return;
    float acc[RPSB][NB];
    for (uint r = 0; r < RPSB; r++)
        for (uint b = 0; b < NB; b++) acc[r][b] = 0;
    for (uint c = lane; c < NCHUNK; c += 32)
        for (uint r = 0; r < RPSB; r++)
            if (row0 + r < a.N) row_chunk(W, S, Z, row0 + r, c, x, a.xstride, acc[r]);
    for (uint r = 0; r < RPSB; r++)
        for (uint b = 0; b < NB; b++) {
            const float v = simd_sum(acc[r][b]);
            if (lane == 0 && row0 + r < a.N) y[(ulong)b * a.ystride + row0 + r] = YT(v);
        }
}

// split-K：每個 threadgroup 一列，8 個 SIMD group 各算 K 的一段再加總。grid：N × 256
kernel void qmvb_splitk(device const uint *W [[buffer(0)]], device const ST *S [[buffer(1)]], device const ST *Z [[buffer(2)]],
                        device const half *x [[buffer(3)]], device YT *y [[buffer(4)]], constant qmvb_args &a [[buffer(5)]],
                        uint row [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[8][NB];
    float acc[NB];
    for (uint b = 0; b < NB; b++) acc[b] = 0;
    for (uint c = sg * 32 + lane; c < NCHUNK; c += 256) chunk_nb(W, S, Z, row, c, x, a.xstride, acc);
    for (uint b = 0; b < NB; b++) {
        const float v = simd_sum(acc[b]);
        if (lane == 0) part[sg][b] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NB) {
        float t = 0;
        for (uint i = 0; i < 8; i++) t += part[i][tid];
        y[(ulong)tid * a.ystride + row] = YT(t);
    }
}

// 多矩陣：同一組輸入、最多 4 個矩陣（相同 K、位元數）。grid：ceil(總列數 / 8) × 256
struct qmvb_multi_args {
    uint N[4];
    uint off[4];
    uint total;
    uint xstride;
};

kernel void qmvb_multi(device const uint *W0 [[buffer(0)]], device const ST *S0 [[buffer(1)]], device const ST *Z0 [[buffer(2)]],
                       device const uint *W1 [[buffer(3)]], device const ST *S1 [[buffer(4)]], device const ST *Z1 [[buffer(5)]],
                       device const uint *W2 [[buffer(6)]], device const ST *S2 [[buffer(7)]], device const ST *Z2 [[buffer(8)]],
                       device const uint *W3 [[buffer(9)]], device const ST *S3 [[buffer(10)]], device const ST *Z3 [[buffer(11)]],
                       device const half *x [[buffer(12)]], device YT *y0 [[buffer(13)]], device YT *y1 [[buffer(14)]],
                       device YT *y2 [[buffer(15)]], device YT *y3 [[buffer(16)]], constant qmvb_multi_args &a [[buffer(17)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint grow = tg * 8 + sg;
    if (grow >= a.total) return;
    uint j = 0;
    while (j < 3 && grow >= a.off[j + 1] && a.N[j + 1] > 0) j++;
    const ulong row = grow - a.off[j];
    device const uint *W = j == 0 ? W0 : j == 1 ? W1 : j == 2 ? W2 : W3;
    device const ST *S = j == 0 ? S0 : j == 1 ? S1 : j == 2 ? S2 : S3;
    device const ST *Z = j == 0 ? Z0 : j == 1 ? Z1 : j == 2 ? Z2 : Z3;
    device YT *y = j == 0 ? y0 : j == 1 ? y1 : j == 2 ? y2 : y3;
    float acc[NB];
    for (uint b = 0; b < NB; b++) acc[b] = 0;
    for (uint c = lane; c < NCHUNK; c += 32) chunk_nb(W, S, Z, row, c, x, a.xstride, acc);
    for (uint b = 0; b < NB; b++) {
        const float v = simd_sum(acc[b]);
        if (lane == 0) y[(ulong)b * a.N[j] + row] = YT(v);
    }
}
