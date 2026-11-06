// Decode 用的量化矩陣 × 向量（qmv）：y = W·x
// 權重格式與 MLX 相同（affine，group size 64）：
//   W：[N, K] 以 uint32 打包，每個 uint32 從低位開始放 32/BITS 個無號整數 q
//   S、Z：[N, K/64] 的 scale 和 bias，實際權重 w = q·S + Z
// 支援一次處理多個 expert：grid 的 y 維度是第幾個被選中的 expert（slot），
// eids[slot] 是 expert 編號。dense 層就把 eids 設成 {0}、E = 1。
//
// 主程式注入的 #define：
//   BITS：4 或 8；K：輸入維度；RPS：每個 SIMD group 算幾列；SPT：每個 threadgroup 幾個 SIMD group；
//   ST：scale / bias 的型別（MLX 通常是 bfloat）；qmv：kernel 名稱
#include <metal_stdlib>
#ifdef QW_STRICT // MLX profile（主程式注入）：safe 數學模式，不重排加總。驗證（NB 個 token）與逐 token decode 的同一段算式才會逐位元相同，
#pragma METAL fp math_mode(safe) // 推測解碼的輸出 = greedy。fast 模式下編譯器依情境（token 迴圈、展開）重排，兩邊捨入不同
#endif
using namespace metal;

#ifndef YT
#define YT half // 輸出型別（lm_head 用 float）
#endif

#define GS 64
#define CHUNK 16                      // 每個 lane 每次處理 16 個權重
#define NCHUNK (K / CHUNK)            // 每列幾個 chunk
#define ROWW (K * BITS / 32)          // 每列幾個 uint32
#define NG (K / GS)                   // 每列幾個 group

static_assert(K % GS == 0, "K 必須是 64 的倍數");

struct qmv_args {
    uint N;          // 輸出維度
    uint x_stride;   // 每個 slot 的輸入間距（元素數）；所有 expert 共用同一個 x 時為 0
    uint y_stride;   // 每個 slot 的輸出間距（元素數）
    uint pad;
};

// QW_STRICT（--exact-spec，MLX）用下面的 chunk 寫法；預設（#else）保留 v1.0.0 的逐個累加，輸出逐位元不變
#ifdef QW_STRICT
// 一個 chunk（16 個權重）的運算。qmvb.metal（驗證）有逐字相同的一份：兩邊必須一起改，驗證才會與 decode 逐位元相同。
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
#endif

kernel void qmv(device const uint *W [[buffer(0)]],
                device const ST *S [[buffer(1)]],
                device const ST *Z [[buffer(2)]],
                device const half *x [[buffer(3)]],
                device YT *y [[buffer(4)]],
                device const uint *eids [[buffer(5)]],
                constant qmv_args &a [[buffer(6)]],
                uint2 tg [[threadgroup_position_in_grid]],
                uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    const uint slot = tg.y;
    const uint e = eids[slot];
    const uint N = a.N;
    W += (ulong)e * N * ROWW;
    S += (ulong)e * N * NG;
    Z += (ulong)e * N * NG;
    x += slot * a.x_stride;
    y += slot * a.y_stride;

    const uint row0 = (tg.x * SPT + sg) * RPS;
    if (row0 >= N) return;

    float acc[RPS];
    for (int r = 0; r < RPS; r++) acc[r] = 0;

    for (uint c = lane; c < NCHUNK; c += 32) {
        // 讀 16 個輸入值，同時算它們的和（給 bias 用）
        device const half4 *xp = (device const half4 *)(x + c * CHUNK);
#ifdef QW_STRICT
        const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
        const float xs = sum16(x0, x1, x2, x3);
#else
        float xv[CHUNK];
        float xs = 0;
        for (int j = 0; j < 4; j++) {
            float4 h = float4(xp[j]);
            xv[4 * j + 0] = h.x; xv[4 * j + 1] = h.y; xv[4 * j + 2] = h.z; xv[4 * j + 3] = h.w;
            xs += h.x + h.y + h.z + h.w;
        }
#endif
        const uint g = c / (GS / CHUNK);
        for (int r = 0; r < RPS; r++) {
            const uint row = row0 + r;
            if (row >= N) break;
#ifdef QW_STRICT
            float4 w4[4];
            unpack16(W, row, c, w4);
            acc[r] = chunk_acc(acc[r], float(S[(ulong)row * NG + g]), float(Z[(ulong)row * NG + g]), dot16(w4, x0, x1, x2, x3), xs);
#else
            float d = 0;
#if BITS == 4
            uint2 w = ((device const uint2 *)(W + (ulong)row * ROWW))[c];
            for (int n = 0; n < 8; n++) d += float((w.x >> (4 * n)) & 15u) * xv[n];
            for (int n = 0; n < 8; n++) d += float((w.y >> (4 * n)) & 15u) * xv[8 + n];
#else
            // 兩次 8 bytes 讀取：checkpoint 的 tensor 只保證 8 bytes 對齊
            const uint2 wl = ((device const uint2 *)(W + (ulong)row * ROWW))[2 * c], wh = ((device const uint2 *)(W + (ulong)row * ROWW))[2 * c + 1];
            const uint4 w = uint4(wl, wh);
            for (int n = 0; n < 4; n++) d += float((w.x >> (8 * n)) & 255u) * xv[n];
            for (int n = 0; n < 4; n++) d += float((w.y >> (8 * n)) & 255u) * xv[4 + n];
            for (int n = 0; n < 4; n++) d += float((w.z >> (8 * n)) & 255u) * xv[8 + n];
            for (int n = 0; n < 4; n++) d += float((w.w >> (8 * n)) & 255u) * xv[12 + n];
#endif
            const float s = float(S[(ulong)row * NG + g]);
            const float z = float(Z[(ulong)row * NG + g]);
            acc[r] += s * d + z * xs;
#endif
        }
    }
    for (int r = 0; r < RPS; r++) {
        float v = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < N) y[row0 + r] = YT(v);
    }
}

// ---------------------------------------------------------------- 特殊形狀的變體
// 共用：一個 chunk（16 個權重）的內積（未乘 scale）與 x 的總和
inline float chunk_dot(device const uint *W, ulong row, uint c, device const half *x, thread float &xs) {
    device const half4 *xp = (device const half4 *)(x + c * CHUNK);
#ifdef QW_STRICT
    const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
    xs = sum16(x0, x1, x2, x3);
    float4 w4[4];
    unpack16(W, row, c, w4);
    return dot16(w4, x0, x1, x2, x3);
#else
    float xv[CHUNK];
    xs = 0;
    for (int j = 0; j < 4; j++) {
        float4 h = float4(xp[j]);
        xv[4 * j] = h.x; xv[4 * j + 1] = h.y; xv[4 * j + 2] = h.z; xv[4 * j + 3] = h.w;
        xs += h.x + h.y + h.z + h.w;
    }
    float d = 0;
#if BITS == 4
    uint2 w = ((device const uint2 *)(W + row * ROWW))[c];
    for (int n = 0; n < 8; n++) d += float((w.x >> (4 * n)) & 15u) * xv[n];
    for (int n = 0; n < 8; n++) d += float((w.y >> (4 * n)) & 15u) * xv[8 + n];
#else
    const uint2 wl = ((device const uint2 *)(W + row * ROWW))[2 * c], wh = ((device const uint2 *)(W + row * ROWW))[2 * c + 1];
    for (int n = 0; n < 4; n++) d += float((wl.x >> (8 * n)) & 255u) * xv[n];
    for (int n = 0; n < 4; n++) d += float((wl.y >> (8 * n)) & 255u) * xv[4 + n];
    for (int n = 0; n < 4; n++) d += float((wh.x >> (8 * n)) & 255u) * xv[8 + n];
    for (int n = 0; n < 4; n++) d += float((wh.y >> (8 * n)) & 255u) * xv[12 + n];
#endif
    return d;
#endif
}

// split-K：列少、K 很長（例如 10240 → 320）。每個 threadgroup 一列，8 個 SIMD group 各算 K 的一段再加總
// grid：N 個 threadgroup × 256
kernel void qmv_splitk(device const uint *W [[buffer(0)]], device const ST *S [[buffer(1)]], device const ST *Z [[buffer(2)]],
                       device const half *x [[buffer(3)]], device YT *y [[buffer(4)]], device const uint *eids [[buffer(5)]],
                       constant qmv_args &a [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    if (row >= a.N) return;
    float acc = 0;
    for (uint c = sg * 32 + lane; c < NCHUNK; c += 256) {
        float xs;
        const float d = chunk_dot(W, row, c, x, xs);
        const uint g = c / (GS / CHUNK);
#ifdef QW_STRICT
        acc = chunk_acc(acc, float(S[(ulong)row * NG + g]), float(Z[(ulong)row * NG + g]), d, xs);
#else
        acc += float(S[(ulong)row * NG + g]) * d + float(Z[(ulong)row * NG + g]) * xs;
#endif
    }
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float t = 0;
        for (uint i = 0; i < 8; i++) t += part[i];
        y[row] = YT(t);
    }
}

// 短 K（例如 320 → 10240）：每個 SIMD group 算 2 列，每列 16 個 lane
// grid：ceil(N / 16) 個 threadgroup × 256（8 個 SIMD group × 2 列）
kernel void qmv_halfsg(device const uint *W [[buffer(0)]], device const ST *S [[buffer(1)]], device const ST *Z [[buffer(2)]],
                       device const half *x [[buffer(3)]], device YT *y [[buffer(4)]], device const uint *eids [[buffer(5)]],
                       constant qmv_args &a [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = (tg * 8 + sg) * 2 + lane / 16, sub = lane % 16;
    float acc = 0;
    if (row < a.N)
        for (uint c = sub; c < NCHUNK; c += 16) {
            float xs;
            const float d = chunk_dot(W, row, c, x, xs);
            const uint g = c / (GS / CHUNK);
#ifdef QW_STRICT
            acc = chunk_acc(acc, float(S[(ulong)row * NG + g]), float(Z[(ulong)row * NG + g]), d, xs);
#else
            acc += float(S[(ulong)row * NG + g]) * d + float(Z[(ulong)row * NG + g]) * xs;
#endif
        }
    // 16 個 lane 內加總
    for (uint o = 8; o > 0; o >>= 1) acc += simd_shuffle_xor(acc, o);
    if (sub == 0 && row < a.N) y[row] = YT(acc);
}

// 多矩陣：同一個輸入 x、最多 4 個權重矩陣（相同 K 與位元數），一次 dispatch
// 每個 SIMD group 一列；依全域列號判斷屬於哪個矩陣。grid：ceil(總列數 / 8) 個 threadgroup × 256
struct qmv_multi_args {
    uint N[4];
    uint off[4]; // 每個矩陣的起始全域列號
    uint total;
};

kernel void qmv_multi(device const uint *W0 [[buffer(0)]], device const ST *S0 [[buffer(1)]], device const ST *Z0 [[buffer(2)]],
                      device const uint *W1 [[buffer(3)]], device const ST *S1 [[buffer(4)]], device const ST *Z1 [[buffer(5)]],
                      device const uint *W2 [[buffer(6)]], device const ST *S2 [[buffer(7)]], device const ST *Z2 [[buffer(8)]],
                      device const uint *W3 [[buffer(9)]], device const ST *S3 [[buffer(10)]], device const ST *Z3 [[buffer(11)]],
                      device const half *x [[buffer(12)]], device YT *y0 [[buffer(13)]], device YT *y1 [[buffer(14)]],
                      device YT *y2 [[buffer(15)]], device YT *y3 [[buffer(16)]], constant qmv_multi_args &a [[buffer(17)]],
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
    float acc = 0;
    for (uint c = lane; c < NCHUNK; c += 32) {
        float xs;
        const float d = chunk_dot(W, row, c, x, xs);
        const uint g = c / (GS / CHUNK);
#ifdef QW_STRICT
        acc = chunk_acc(acc, float(S[row * NG + g]), float(Z[row * NG + g]), d, xs);
#else
        acc += float(S[row * NG + g]) * d + float(Z[row * NG + g]) * xs;
#endif
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = YT(acc);
}

// split-K 的 mix_down + inject：前 HCN 個 threadgroup 算 inject（最重、最先開始），其餘各算 mix_down 的一列
// inject[s] = 2 · sigmoid(Σ_j injW[s, j] · nf[j] / HCN)（BF16 權重、FP32 輸入）
#ifndef HCN
#define HCN 4
#endif
kernel void qmv_splitk_inj(device const uint *W [[buffer(0)]], device const ST *S [[buffer(1)]], device const ST *Z [[buffer(2)]],
                           device const half *x [[buffer(3)]], device YT *y [[buffer(4)]], device const uint *eids [[buffer(5)]],
                           constant qmv_args &a [[buffer(6)]], device const bfloat *injW [[buffer(7)]],
                           device const float *nf [[buffer(8)]], device float *inj [[buffer(9)]],
                           uint tgi [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[8];
    float acc = 0;
    const bool is_inj = tgi < HCN;
    const uint row = is_inj ? a.N + tgi : tgi - HCN;
    if (!is_inj) {
        for (uint c = sg * 32 + lane; c < NCHUNK; c += 256) {
            float xs;
            const float d = chunk_dot(W, row, c, x, xs);
            const uint g = c / (GS / CHUNK);
#ifdef QW_STRICT
            acc = chunk_acc(acc, float(S[(ulong)row * NG + g]), float(Z[(ulong)row * NG + g]), d, xs);
#else
            acc += float(S[(ulong)row * NG + g]) * d + float(Z[(ulong)row * NG + g]) * xs;
#endif
        }
    } else {
        const uint s = row - a.N;
        for (uint j = tid; j < K; j += 256) acc += float(injW[(ulong)s * K + j]) * nf[j];
    }
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0;
        for (uint i = 0; i < 8; i++) t += part[i];
        if (row < a.N) y[row] = YT(t);
        else inj[row - a.N] = 2.0f / (1.0f + exp(-t / HCN));
    }
}


// ---------------------------------------------------------------- 輸入轉換版（把前一步的逐元素處理併進來）
// XF = 1（DeltaNet 的 gated RMSNorm）：x[d] = o[d] · rsqrt(ss[head]/128 + eps) · w[d % 128] · sigmoid(z[d])
//         ss[head] = Σ_{j<4} ssp[head·4 + j]（gdn_step 輸出的部分平方和）
// XF = 2（attention 的 output gate）：x[d] = o[d] · sigmoid(qg[(d/256)·512 + 256 + d%256])
#ifdef XF
#ifdef QW_STRICT // --exact-spec（MLX）：與 qmvb.metal 的 qmvb_xform 共用 xf_elem
// 一個元素的轉換。qmvb.metal 的 qmvb_xform（驗證用）有一份逐字相同的副本：兩邊必須一致，驗證才會與 decode 逐位元相同。
// 本檔是 safe 數學模式（不重排），超越函數明確用 fast:: 版本（與原本 fast 模式的速度相同、結果與編譯情境無關）
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
// Σ x 的加總同 chunk_dot（sum16）：驗證走 qmvb_xform（寫成 FP16）→ qmvb，結果要與這裡逐位元相同
inline void xform_chunk(uint c, device const half *o, device const half *aux, device const float *ssp, device const bfloat *nw,
                        thread float4 *xq, thread float &xs) {
    for (uint j = 0; j < 4; j++) {
        const uint d = c * CHUNK + 4 * j;
        xq[j] = float4(xf_elem(d, o, aux, ssp, nw), xf_elem(d + 1, o, aux, ssp, nw), xf_elem(d + 2, o, aux, ssp, nw),
                       xf_elem(d + 3, o, aux, ssp, nw));
    }
    xs = sum16(xq[0], xq[1], xq[2], xq[3]);
}

kernel void qmv_xf(device const uint *W [[buffer(0)]], device const ST *S [[buffer(1)]], device const ST *Z [[buffer(2)]],
                   device const half *o [[buffer(3)]], device YT *y [[buffer(4)]], device const half *aux [[buffer(5)]],
                   constant qmv_args &a [[buffer(6)]], device const float *ssp [[buffer(7)]], device const bfloat *nw [[buffer(8)]],
                   uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8 + sg;
    if (row >= a.N) return;
    float acc = 0;
    for (uint c = lane; c < NCHUNK; c += 32) {
        float4 xq[4], w4[4];
        float xs;
        xform_chunk(c, o, aux, ssp, nw, xq, xs);
        unpack16(W, row, c, w4);
        const uint g = c / (GS / CHUNK);
        acc = chunk_acc(acc, float(S[(ulong)row * NG + g]), float(Z[(ulong)row * NG + g]), dot16(w4, xq[0], xq[1], xq[2], xq[3]), xs);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = YT(acc);
}
#else // 預設：v1.0.0 的寫法（輸出逐位元不變）
inline void xform_chunk(uint c, device const half *o, device const half *aux, device const float *ssp, device const bfloat *nw,
                        thread float *xv, thread float &xs) {
    xs = 0;
    for (uint j = 0; j < CHUNK; j++) {
        const uint d = c * CHUNK + j;
        float v = float(o[d]);
#if XF == 1
        const uint hh = d / 128;
        const float ss = ssp[hh * 4] + ssp[hh * 4 + 1] + ssp[hh * 4 + 2] + ssp[hh * 4 + 3];
        v = v * rsqrt(ss / 128.0f + 1e-6f) * float(nw[d % 128]) / (1.0f + exp(-float(aux[d])));
        v = float(half(v)); // 與未合併版一樣先轉成 FP16
#else
        v = float(half(v / (1.0f + exp(-float(aux[(d / 256) * 512 + 256 + d % 256])))));
#endif
        xv[j] = v;
        xs += v;
    }
}

kernel void qmv_xf(device const uint *W [[buffer(0)]], device const ST *S [[buffer(1)]], device const ST *Z [[buffer(2)]],
                   device const half *o [[buffer(3)]], device YT *y [[buffer(4)]], device const half *aux [[buffer(5)]],
                   constant qmv_args &a [[buffer(6)]], device const float *ssp [[buffer(7)]], device const bfloat *nw [[buffer(8)]],
                   uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8 + sg;
    if (row >= a.N) return;
    float acc = 0;
    for (uint c = lane; c < NCHUNK; c += 32) {
        float xv[CHUNK], xs;
        xform_chunk(c, o, aux, ssp, nw, xv, xs);
        float d = 0;
#if BITS == 4
        const uint2 w = ((device const uint2 *)(W + (ulong)row * ROWW))[c];
        for (int n = 0; n < 8; n++) d += float((w.x >> (4 * n)) & 15u) * xv[n];
        for (int n = 0; n < 8; n++) d += float((w.y >> (4 * n)) & 15u) * xv[8 + n];
#else
        const uint2 wl = ((device const uint2 *)(W + (ulong)row * ROWW))[2 * c], wh = ((device const uint2 *)(W + (ulong)row * ROWW))[2 * c + 1];
        for (int n = 0; n < 4; n++) d += float((wl.x >> (8 * n)) & 255u) * xv[n];
        for (int n = 0; n < 4; n++) d += float((wl.y >> (8 * n)) & 255u) * xv[4 + n];
        for (int n = 0; n < 4; n++) d += float((wh.x >> (8 * n)) & 255u) * xv[8 + n];
        for (int n = 0; n < 4; n++) d += float((wh.y >> (8 * n)) & 255u) * xv[12 + n];
#endif
        const uint g = c / (GS / CHUNK);
        acc += float(S[(ulong)row * NG + g]) * d + float(Z[(ulong)row * NG + g]) * xs;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = YT(acc);
}
#endif
#endif
