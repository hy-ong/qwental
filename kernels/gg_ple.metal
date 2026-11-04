// GGUF profile 的 embed 與 PLE（第 1 層）：token_embd（IQ4_XS）逐列精確解碼、PLE 的 F32 norm 權重與 FP32 value。
// 前面接 iq_tables.metal 與 moe_iq.metal（IQ_DQ_ONLY，提供 dq32）。
// 主程式注入：D（2560）、HC（4）、GF（token_embd 的 ggml type）、ERB（token_embd 每列位元組）
//
// PLE 的 n-gram 列（IQ4_NL：fp16 d × int8 碼本）在 CPU 精確解成 float v，再拆成兩個 FP16：hi = half(v)、lo = half(v − hi)。
// v = d·k 是 d 的 ulp（≥ 2⁻²⁴）的整數倍，hi 也是；所以 lo 也是 2⁻²⁴ 的倍數且不超過 11 位元 → hi + lo 恰好等於 v（CPU 會檢查）。
// key / value 投影把 hi、lo 當成兩個 token 一起乘（權重只讀一次），FP32 相加 = W·v（輸入不經過 FP16 捨入）。

#define EPS 1e-6f
#define HCD (HC * D)

inline float gp_sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

inline float gp_tg_sum(float v, threadgroup float *tmp, uint tid, uint ntg) {
    v = simd_sum(v);
    const uint sg = tid / 32, lane = tid % 32, nsg = (ntg + 31) / 32;
    if (lane == 0) tmp[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? tmp[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static_assert(D % 32 == 0, "gg_embed：D 必須是 32 的倍數");

struct gg_embed_args {
    uint p;       // ids 從第 p 個開始
    uint ncopy;   // 每列複製幾份（殘差流 HC 份；drafter 1 份）
    uint ystride; // 輸出每列的間距（元素）
    uint pad;
};

// embed：y[r][c·D + i] = token_embd[ids[p + r]][i]（FP32，與 GGUF 的 dequantize_row 相同，不捨入）。
// grid：列數 × (D / 32)：每個執行緒解一段 32 個值
kernel void gg_embed(device const uchar *W [[buffer(0)]], device const int *ids [[buffer(1)]], constant gg_embed_args &a [[buffer(2)]],
                     device float *y [[buffer(3)]], uint seg [[thread_index_in_threadgroup]], uint r [[threadgroup_position_in_grid]]) {
    if (seg >= D / 32) return;
    const ulong row = (ulong)ids[a.p + r];
    float v[32];
    dq32<GF>(W + row * ERB, seg, v);
    for (uint c = 0; c < a.ncopy; c++) {
        device float4 *o = (device float4 *)(y + (ulong)r * a.ystride + c * D + seg * 32);
        for (uint j = 0; j < 8; j++) o[j] = float4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
    }
}

// hi / lo 兩份的投影結果相加：y[r][n] = Y[2r][n] + Y[2r + 1][n]。grid 涵蓋 rows × N
kernel void gp_pair_sum(device const float *Y [[buffer(0)]], device float *y [[buffer(1)]], constant uint &N [[buffer(2)]],
                        constant uint &rows [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= rows * N) return;
    const uint r = i / N, n = i % N;
    y[i] = Y[(ulong)(2 * r) * N + n] + Y[(ulong)(2 * r + 1) * N + n];
}

// 分流 RMSNorm，權重 FP32（GGUF 的 ple_norm_*）；FP32 進出。grid：(HC, T) × 1024（decode：T = 1）
kernel void gp_hcnorm_w32(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint2 ntg2 [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    const uint ntg = ntg2.x, s = tg.x;
    const ulong base = (ulong)tg.y * HCD + s * D;
    float ss = 0;
    for (uint i = tid; i < D; i += ntg) { const float v = x[base + i]; ss += v * v; }
    const float r = rsqrt(gp_tg_sum(ss, tmp, tid, ntg) / D + EPS);
    for (uint i = tid; i < D; i += ntg) y[base + i] = x[base + i] * r * w[s * D + i];
}

// PLE gate（value 是 FP32）：gated[t, s, i] = sigmoid(g(t, s)) · value[t, i]。grid：(HC, T) × 1024
kernel void gp_ple_gate(device const float *key [[buffer(0)]], device const float *qry [[buffer(1)]], device const float *val [[buffer(2)]],
                        device float *gated [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint2 ntg2 [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    const uint ntg = ntg2.x, s = tg.x, t = tg.y;
    const ulong base = (ulong)t * HCD + s * D;
    float acc = 0;
    for (uint i = tid; i < D; i += ntg) acc += key[base + i] * qry[base + i];
    float g = gp_tg_sum(acc, tmp, tid, ntg) / sqrt(float(D));
    g = sqrt(max(fabs(g), 1e-6f)) * sign(g);
    const float sgv = gp_sigm(g);
    for (uint i = tid; i < D; i += ntg) gated[base + i] = sgv * val[(ulong)t * D + i];
}
