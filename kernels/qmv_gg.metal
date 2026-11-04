// GGUF profile 的 dense matvec（decode 一個 token、驗證 NB 個 token）：權重是 ggml 區塊（dq32<F> 解碼，含 BF16）。
// 前面接 iq_tables.metal 與 moe_iq.metal（IQ_DQ_ONLY，提供 dq32）。這是正確性優先的通用版本，各部分再各自特化。
// 主程式注入：GF（ggml type；不用 F：會跟 dq32 的 template 參數撞名）、KD（K）、RB（每列位元組）、NB（token 數）、YT（輸出型別）；名稱改名
#ifndef YT
#define YT half
#endif
static_assert(KD % 32 == 0, "qmv_gg：K 必須是 32 的倍數");

struct qmv_gg_args {
    uint N;       // 輸出列數
    uint xstride; // 每個 token 的輸入間距（元素）
    uint ystride; // 每個 token 的輸出間距（元素）
    uint pad;
};

// 每個 SIMD group 一列；lane 負責第 lane、lane + 32… 段（每段 32 個值）。grid：ceil(N / 8) × 256
kernel void qmv_gg(device const uchar *W [[buffer(0)]], device const half *x [[buffer(3)]], device YT *y [[buffer(4)]],
                   constant qmv_gg_args &a [[buffer(5)]], uint tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8 + sg;
    if (row >= a.N) return;
    device const uchar *wr = W + (ulong)row * RB;
    float acc[NB];
    for (uint b = 0; b < NB; b++) acc[b] = 0;
    for (uint s = lane; s < KD / 32; s += 32) {
        float v[32];
        dq32<GF>(wr, s, v);
        for (uint b = 0; b < NB; b++) {
            device const half4 *xp = (device const half4 *)(x + (ulong)b * a.xstride + 32 * s);
            float d = 0;
            for (uint i = 0; i < 8; i++) d += dot(float4(xp[i]), float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]));
            acc[b] += d;
        }
    }
    for (uint b = 0; b < NB; b++) {
        const float t = simd_sum(acc[b]);
        if (lane == 0) y[(ulong)b * a.ystride + row] = YT(t);
    }
}
