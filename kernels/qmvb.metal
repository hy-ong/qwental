// 多個 token 一起的量化 matvec（給 MTP 驗證用）：權重讀一次、跟 NB 個輸入向量各算一次
// 排列：x [NB][K]（每個 token 連續）、y [NB][N]。權重格式同 qmv.metal（MLX affine、group 64）
// 主程式注入：BITS、K、NB、ST（scale 型別）、YT（輸出型別，預設 half）；kernel 名稱也由主程式改名
#include <metal_stdlib>
#ifdef QW_STRICT // MLX profile（主程式注入）：safe 數學模式，不重排加總。驗證（NB 個 token）與逐 token decode 的同一段算式才會逐位元相同，
#pragma METAL fp math_mode(safe) // 推測解碼的輸出 = greedy。fast 模式下編譯器依情境（token 迴圈、展開）重排，兩邊捨入不同
#endif
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

// QW_STRICT（--exact-spec，MLX）用與 qmv.metal 相同的 chunk 寫法；預設（#else）保留 v1.0.0 的寫法，輸出逐位元不變
#ifdef QW_STRICT
// 一個 chunk（16 個權重）的運算。qmv.metal（decode）有逐字相同的一份：兩邊必須一起改，驗證才會與 decode 逐位元相同。
// 權重解成 4 個 float4（4-bit：前 8 個的偶數 / 奇數 nibble、後 8 個的偶數 / 奇數 nibble；8-bit：依序 4 個一組），
// 內積是 4 段 dot 兩兩相加（固定的加總樹：safe 模式下不重排，又比逐個累加的長依賴鏈快）
inline void unpack16(device const uint *W, ulong row, uint c, thread float4 *w4) {
#if BITS == 4
    const uint2 w = ((device const uint2 *)(W + row * ROWW))[c];
    w4[0] = float4(as_type<uchar4>(w.x & 0x0f0f0f0fu));
    w4[1] = float4(as_type<uchar4>((w.x >> 4) & 0x0f0f0f0fu));
    w4[2] = float4(as_type<uchar4>(w.y & 0x0f0f0f0fu));
    w4[3] = float4(as_type<uchar4>((w.y >> 4) & 0x0f0f0f0fu));
#else
    // 兩次 8 bytes 讀取：checkpoint 的 tensor 只保證 8 bytes 對齊
    const uint2 wl = ((device const uint2 *)(W + row * ROWW))[2 * c], wh = ((device const uint2 *)(W + row * ROWW))[2 * c + 1];
    w4[0] = float4(as_type<uchar4>(wl.x));
    w4[1] = float4(as_type<uchar4>(wl.y));
    w4[2] = float4(as_type<uchar4>(wh.x));
    w4[3] = float4(as_type<uchar4>(wh.y));
#endif
}
inline float dot16(thread const float4 *w4, float4 x0, float4 x1, float4 x2, float4 x3) {
#if BITS == 4
    return (dot(w4[0], float4(x0.xz, x1.xz)) + dot(w4[1], float4(x0.yw, x1.yw))) +
           (dot(w4[2], float4(x2.xz, x3.xz)) + dot(w4[3], float4(x2.yw, x3.yw)));
#else
    return (dot(w4[0], x0) + dot(w4[1], x1)) + (dot(w4[2], x2) + dot(w4[3], x3));
#endif
}
inline float sum16(float4 x0, float4 x1, float4 x2, float4 x3) { return dot((x0 + x1) + (x2 + x3), float4(1)); }
// 一列一個 chunk：acc += s·Σ q·x + z·Σ x
inline float chunk_acc(float acc, float s, float z, float d, float xs) { return acc + (s * d + z * xs); }

// 一列、一個 chunk 對 NB 個 token 的部分內積：權重解開一次，每個 token 的算式與 decode（qmv.metal 的 chunk_dot）相同
inline void row_chunk(device const uint *W, device const ST *S, device const ST *Z, ulong row, uint c, device const half *x,
                      uint xstride, thread float *acc) {
    const uint g = c / (GS / CHUNK);
    const float s = float(S[row * NG + g]), z = float(Z[row * NG + g]);
    float4 w4[4];
    unpack16(W, row, c, w4);
    for (uint b = 0; b < NB; b++) {
        device const half4 *xp = (device const half4 *)(x + (ulong)b * xstride + c * CHUNK);
        const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
        acc[b] = chunk_acc(acc[b], s, z, dot16(w4, x0, x1, x2, x3), sum16(x0, x1, x2, x3));
    }
}
#else
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
#endif

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

#ifdef QW_STRICT // 只有 --exact-spec 用到
#ifdef XF
// ---- 驗證用的輸入轉換（decode 把它併在 qmv.metal 的 qmv_xf 裡）：逐元素算好寫成 FP16，再由 qmvb 乘
// XF = 1：DeltaNet 的 gated RMSNorm（o、z [NB][6144]、ssp [NB][48·4]）；XF = 2：attention 的 output gate（o [NB][6144]、qg [NB][12288]）
// xf_elem 與 qmv.metal 的逐字相同（兩邊都必須一起改）
inline float xf_elem(uint d, device const half *o, device const half *aux, device const float *ssp, device const bfloat *nw) {
    float v = float(o[d]);
#if XF == 1
    const uint hh = d / 128;
    const float ss = ssp[hh * 4] + ssp[hh * 4 + 1] + ssp[hh * 4 + 2] + ssp[hh * 4 + 3];
    v = fast::divide(v * fast::rsqrt(ss / 128.0f + 1e-6f) * float(nw[d % 128]), 1.0f + fast::exp(-float(aux[d])));
#else
    v = fast::divide(v, 1.0f + fast::exp(-float(aux[(d / 256) * 512 + 256 + d % 256])));
#endif
    return float(half(v)); // 與未合併版一樣先轉成 FP16
}
// grid：(6144 / 256, token 數) × 256
kernel void qmvb_xform(device const half *o [[buffer(0)]], device const half *aux [[buffer(1)]], device const float *ssp [[buffer(2)]],
                       device const bfloat *nw [[buffer(3)]], device half *y [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    constexpr uint XD = 6144, AS = XF == 1 ? 6144 : 12288, SS = 48 * 4;
    static_assert(XD % 256 == 0, "grid 是 XD / 256 個 threadgroup");
    const uint d = gid.x, t = gid.y;
    if (d >= XD) return;
    y[(ulong)t * XD + d] = half(xf_elem(d, o + (ulong)t * XD, aux + (ulong)t * AS, ssp + (ulong)t * SS, nw));
}
#endif
#endif
