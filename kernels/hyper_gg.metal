// GGUF profile 的 hyper-connection：hc_norm F32、mix_down / mix_up BF16、inject BF16（llama.cpp 的精度：
// 全程 FP32，不把正規化後的向量捨入成 FP16）。decode（NB = 1）與驗證（NB 個 token）共用同一份原始碼。
// 主程式注入：D（2560）、HC（4）、NB；可選 GH_R / GH_G（ghc_down 每個 threadgroup 幾列、每組幾列先加總）、GU_SG（ghc_up 每個 threadgroup 幾個 SIMD group）。
//
// 一次 hyper-connection 只要兩個 kernel（原本 MLX 版是三個）：
//   ghc_down：殘差更新（h + out·inj）、分流平方和、mix_down 與 inject 的「未正規化」部分和，全部一次讀完。
//     RMSNorm 的倒數 r[s] 是每條流一個純量，所以 Σ_j W[r, j]·(h'[j]·r[s]·w[j]) = r[s] · Σ_j W[r, j]·h'[j]·w[j]：
//     先算每條流的部分和，r[s] 留給下一個 kernel 乘，不必先等整條流的平方和。
//     每個 threadgroup：GH_R 列 × 一條流（2560 欄）；每個執行緒固定 8 欄（h'·w 留在暫存器，給 GH_R 列重複使用）。
//     這個 kernel 只讀不寫狀態（h 由 ghc_up 寫回），不同 threadgroup 之間沒有競爭。
//   ghc_up：r[s] = rsqrt(平方和 / D + ε)、d = Σ_s r[s]·部分和、SiLU(d / HC) → mix_up → x = mean_s σ(u)·norm(h')，
//     同時把 h' 寫回 h、把 inject 部分和（乘上 r[s]）寫成舊格式 injp[s][q]（inj_of 照用）。
#include <metal_stdlib>
using namespace metal;

#define EPS 1e-6f
#define HCD (HC * D)
#define LR 320             // 低秩維度
#define LRI (LR + HC)      // mix_down 的列 + inject 的 HC 列
#define PDS (LRI + 4)      // 部分和每份的間距：[0, LR) mix_down、[LR, LRI) inject、[LRI] 平方和
#ifndef GH_R
#define GH_R 16
#endif
#ifndef GH_G
#define GH_G GH_R
#endif
#ifndef GU_SG
#define GU_SG 8
#endif
#define GH_CPT 8           // ghc_down 每個執行緒的欄數（一次讀 16 位元組的 BF16）
#define GH_NT (D / GH_CPT) // ghc_down 每個 threadgroup 的執行緒數（320）
#define GH_NSG (GH_NT / 32)
#ifdef NB
static_assert(D % (GH_CPT * 32) == 0, "ghc_down：D 必須是 256 的倍數");
static_assert(LR % GH_R == 0 && GH_R >= HC, "ghc_down：GH_R 必須整除 320 且至少 HC");
static_assert(LR == 32 * 10, "ghc_up：每個 lane 10 欄");
static_assert(D % GU_SG == 0 && GU_SG <= 32, "ghc_up：GU_SG 必須整除 D、最多 32");
#endif

inline float gh_silu(float x) { return x / (1.0f + exp(-x)); }
inline float gh_sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
// 釘住運算順序：fast-math 下 a·b·c 的結合順序由編譯器依 kernel 的上下文決定（decode 與驗證的 kernel 不同就可能不同），
// 中間值經過一次 simd_shuffle（自己的 lane）編譯器就不能再重排
inline float gh_pin(float v, ushort lane) { return simd_shuffle(v, lane); }
// 同 engine.metal 的 inj_of：inj[r] = 2 · sigmoid(Σ_s injp[s][r] / HC)
inline float gh_inj(device const float *injp, uint r) {
    float t = 0;
    for (uint s = 0; s < HC; s++) t += injp[s * HC + r];
    return 2.0f / (1.0f + exp(-t / HC));
}
// BF16 → float（精確）：一個 uint 兩個值，低位在前
inline float bf_lo(uint u) { return as_type<float>(u << 16); }
inline float bf_hi(uint u) { return as_type<float>(u & 0xffff0000u); }

#ifdef NB
// 一個 threadgroup 的 R 列（從 W 起、列距 HCD），x 是這個執行緒的 8 欄 × NB 個 token；結果寫到 part（每個 SIMD group 一份）
#define GH_POW2(n) ((n) <= 1 ? 1u : (n) <= 2 ? 2u : (n) <= 4 ? 4u : (n) <= 8 ? 8u : (n) <= 16 ? 16u : 32u)

// 每組 GH_G 列：讀完、算完就先 SIMD 內加總寫出，累加器只要 GH_G × NB 個（token 多時 GH_R 仍可以大，h / out 少重讀）
template <uint R>
inline void ghd_rows(device const bfloat *W, uint c0, thread const float (&x)[NB][GH_CPT], threadgroup float *part, uint sg, uint lane) {
    constexpr uint G = R < GH_G ? R : GH_G;
    static_assert(R % G == 0, "ghd_rows：R 必須是 G 的倍數");
#pragma clang loop unroll(full)
    for (uint r0 = 0; r0 < R; r0 += G) {
        float acc[G][NB];
        uint4 u[G];
#pragma clang loop unroll(full)
        for (uint r = 0; r < G; r++) u[r] = *(device const uint4 *)(W + (ulong)(r0 + r) * HCD + c0); // 先發出這組的讀取
#pragma clang loop unroll(full)
        for (uint r = 0; r < G; r++) {
            const float v[8] = {bf_lo(u[r].x), bf_hi(u[r].x), bf_lo(u[r].y), bf_hi(u[r].y),
                                bf_lo(u[r].z), bf_hi(u[r].z), bf_lo(u[r].w), bf_hi(u[r].w)};
#pragma clang loop unroll(full)
            for (uint t = 0; t < NB; t++) {
                float d = 0;
#pragma clang loop unroll(full)
                for (uint k = 0; k < 8; k++) d = fma(v[k], x[t][k], d);
                acc[r][t] = d;
            }
        }
#pragma clang loop unroll(full)
        for (uint r = 0; r < G; r++)
#pragma clang loop unroll(full)
            for (uint t = 0; t < NB; t++) {
                const float v = simd_sum(acc[r][t]);
                if (lane == 0) part[sg * (GH_R * NB + NB) + (r0 + r) * NB + t] = v;
            }
    }
}

// grid：(LR / GH_R + 1, HC) × GH_NT。最後一排 threadgroup 算 inject 的 HC 列（want_inj = 0 時直接結束）
// pd：[NB][HC][PDS]（第 t 個 token、第 s 條流的部分和）
kernel void ghc_down(device const float *h [[buffer(0)]], device const half *out [[buffer(1)]], device const float *inj [[buffer(2)]],
                     device const float *w [[buffer(3)]], device const bfloat *Wd [[buffer(4)]], device const bfloat *Wi [[buffer(5)]],
                     device float *pd [[buffer(6)]], constant uint2 &fl [[buffer(7)]], // fl.x：先套用殘差；fl.y：要 inject
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[GH_NSG * (GH_R * NB + NB)];
    const uint rb = tg.x, s = tg.y;
    const bool is_inj = rb == LR / GH_R;
    if (is_inj && !fl.y) return;
    const uint i0 = tid * GH_CPT, c0 = s * D + i0;
    float x[NB][GH_CPT], ss[NB];
    const float4 w0 = *(device const float4 *)(w + c0), w1 = *(device const float4 *)(w + c0 + 4);
#pragma clang loop unroll(full)
    for (uint t = 0; t < NB; t++) {
        const float js = fl.x ? gh_inj(inj + t * HC * HC, s) : 0.0f;
        device const float4 *hp = (device const float4 *)(h + (ulong)t * HCD + c0);
        device const half4 *op = (device const half4 *)(out + (ulong)t * D + i0);
        // 與 ghc_up 同一個式子；不套用時不讀 out（可能還沒寫過）
        const float4 a = fl.x ? fma(float4(op[0]), js, hp[0]) : hp[0], b = fl.x ? fma(float4(op[1]), js, hp[1]) : hp[1];
        ss[t] = dot(a, a) + dot(b, b);
        const float4 xa = a * w0, xb = b * w1;
        for (uint k = 0; k < 4; k++) { x[t][k] = xa[k]; x[t][4 + k] = xb[k]; }
    }
    if (is_inj) ghd_rows<HC>(Wi, c0, x, part, sg, lane);
    else ghd_rows<GH_R>(Wd + (ulong)rb * GH_R * HCD, c0, x, part, sg, lane);
#pragma clang loop unroll(full)
    for (uint t = 0; t < NB; t++) {
        const float v = simd_sum(ss[t]);
        if (lane == 0) part[sg * (GH_R * NB + NB) + GH_R * NB + t] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nrow = is_inj ? HC : GH_R;
    for (uint j = tid; j < nrow * NB + NB; j += GH_NT) {
        const uint jj = j < nrow * NB ? j : GH_R * NB + (j - nrow * NB);
        float v = 0;
        for (uint q = 0; q < GH_NSG; q++) v += part[q * (GH_R * NB + NB) + jj];
        if (j < nrow * NB) {
            const uint r = j / NB, t = j % NB;
            pd[(t * HC + s) * PDS + (is_inj ? LR : rb * GH_R) + r] = v;
        } else if (rb == 0) {
            pd[((j - nrow * NB) * HC + s) * PDS + LRI] = v;
        }
    }
}

// grid：D / GU_SG × 32·GU_SG（每個 SIMD group 負責一個 i，也就是 mix_up 的 HC 列）。
// 權重與 h / out / w 的讀取都在前置計算（部分和 → SiLU）之前發出，兩者的延遲重疊。
kernel void ghc_up(device float *h [[buffer(0)]], device const half *out [[buffer(1)]], device const float *inj [[buffer(2)]],
                   device const float *w [[buffer(3)]], device const bfloat *Wu [[buffer(4)]], device const float *pd [[buffer(5)]],
                   device half *x [[buffer(6)]], device float *injo [[buffer(7)]], constant uint2 &fl [[buffer(8)]],
                   uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sx[NB][LR];
    threadgroup float rs[NB][HC], js[NB][HC];
    const uint i = tg * GU_SG + sg;
    constexpr uint V = GH_POW2(NB * HC);
    const uint idx = lane % V, t = idx / HC, s = idx % HC; // 加總之後這個 lane 拿到的是 (t, s)
    // 1. 先發出讀取：mix_up 的 HC 列（每個 lane 10 欄）、這個 lane 的 (t, s) 的 h / out / w
    uint u[HC][5];
#pragma clang loop unroll(full)
    for (uint q = 0; q < HC; q++) {
        device const uint *wr = (device const uint *)(Wu + (ulong)(q * D + i) * LR) + lane * 5;
#pragma clang loop unroll(full)
        for (uint k = 0; k < 5; k++) u[q][k] = wr[k];
    }
    const uint tc = t < NB ? t : 0;
    const ulong hi = (ulong)tc * HCD + s * D + i;
    const float h0 = h[hi], o0 = fl.x ? float(out[(ulong)tc * D + i]) : 0.0f, w0 = w[s * D + i];
    // 2. 前置：r[t][s]、舊 inject 係數、SiLU(d / HC)
    if (tid < NB * HC) {
        const uint t2 = tid / HC, s2 = tid % HC;
        rs[t2][s2] = rsqrt(pd[(t2 * HC + s2) * PDS + LRI] / D + EPS);
        js[t2][s2] = fl.x ? gh_inj(inj + t2 * HC * HC, s2) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < NB * LR; j += 32 * GU_SG) {
        const uint t2 = j / LR, r = j % LR;
        float d = 0;
        for (uint q = 0; q < HC; q++) d += rs[t2][q] * pd[(t2 * HC + q) * PDS + r];
        sx[t2][r] = gh_silu(d / HC);
    }
    if (tg == 0 && fl.y && tid < NB * HC * HC) { // inject 部分和（乘上 r[s]）→ 舊格式 injp[t][s][q]
        const uint t2 = tid / (HC * HC), s2 = tid / HC % HC, q = tid % HC;
        injo[tid] = rs[t2][s2] * pd[(t2 * HC + s2) * PDS + LR + q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3. mix_up：NB × HC 個內積，加總之後 lane (t, s) 拿到 u[t, s·D + i]
    float acc[NB * HC];
    float v[HC][10];
#pragma clang loop unroll(full)
    for (uint q = 0; q < HC; q++)
#pragma clang loop unroll(full)
        for (uint k = 0; k < 5; k++) { v[q][2 * k] = bf_lo(u[q][k]); v[q][2 * k + 1] = bf_hi(u[q][k]); }
#pragma clang loop unroll(full)
    for (uint tt = 0; tt < NB; tt++) { // 每個 token 的 10 個輸入只從 threadgroup memory 讀一次
        float xv[10];
#pragma clang loop unroll(full)
        for (uint k = 0; k < 10; k++) xv[k] = sx[tt][lane * 10 + k];
#pragma clang loop unroll(full)
        for (uint q = 0; q < HC; q++) {
            float d = 0;
#pragma clang loop unroll(full)
            for (uint k = 0; k < 10; k++) d = fma(v[q][k], xv[k], d);
            acc[tt * HC + q] = d;
        }
    }
    float uv = 0; // simd_sum 是硬體指令：逐一加總比 shuffle 的 reduce-scatter 快很多（量過：NB = 8 時 23 對 49 µs）
#pragma clang loop unroll(full)
    for (uint j = 0; j < NB * HC; j++) { const float sj = simd_sum(acc[j]); uv = idx == j ? sj : uv; }
    // 4. lane (t, s)：殘差寫回、這條流的那一項；4 條流的 lane 相鄰，再用 2 次 shuffle 加起來
    float m = 0;
    if (t < NB) {
        const float hv = fl.x ? fma(o0, js[t][s], h0) : h0;
        if (fl.x && lane < V) h[hi] = hv;
        m = gh_sigm(uv) * gh_pin(gh_pin(hv * rs[t][s], lane) * w0, lane);
    }
    m += simd_shuffle_xor(m, 1);
    m += simd_shuffle_xor(m, 2);
    if (t < NB && s == 0 && lane < V) x[(ulong)t * D + i] = half(m / HC);
}

// ---------------------------------------------------------------- 驗證（NB ≥ 2）的三段版
// 兩段版在 NB = 8 時 ghc_down 的每個 threadgroup（GH_R 列）都要重讀 8 個 token 的 h / out、重算 h'·w，
// ghc_up 的每個 threadgroup 都要重算 NB × 320 個 SiLU，兩者都不是頻寬限制。改成：
//   ghc_dsk：每個 threadgroup = 一條流 × 256 欄（split-K，GK 段），h'·w 只算一次（放 threadgroup memory），所有 324 列都做；
//            每段的部分和寫到 pv[t][s][段][PDS]（每段就是兩段版 ghc_down 的一個 SIMD group：同樣 8 欄一個 lane、同樣 simd_sum）。
//   ghc_mid：依兩段版的順序把各段加起來（Σ_段 → r[s] → Σ_s r[s]·部分和 → SiLU），寫 sx / r / inject，只算一次。
//   ghc_upv：同 ghc_up 的 mix_up，sx / r 從 device 讀（沒有前置計算、沒有 barrier）。
// 每個運算的順序都與兩段版相同，所以 decode（兩段版）與驗證逐位元一致。
#ifndef GK
#define GK 10           // K 方向的段數（主程式的 GH_GK）
#endif
#define GKC (D / GK)    // 每段 256 欄（= 兩段版 ghc_down 一個 SIMD group 的欄數）
#ifndef GV_SG
#define GV_SG 8         // ghc_dsk 每個 threadgroup 幾個 SIMD group（主程式的 GH_VSG）
#endif
#ifndef GV_RP
#define GV_RP 10        // ghc_dsk 的列分幾份（每份一個 threadgroup；h'·w 每份重算一次，很便宜）（主程式的 GH_VRP）
#endif
#ifndef GV_G
#define GV_G 4          // ghc_dsk 每組幾列（累加器 GV_G × NB 個，加總後每個 lane 剛好拿一個）
#endif
static_assert(GK * GKC == D && GKC == 32 * GH_CPT, "ghc_dsk：每段 256 欄（每個 lane 8 欄）");
static_assert(NB <= GV_SG && GV_G * NB <= 32 && LR % (GV_RP * GV_SG * GV_G) == 0 && GV_G == HC, "ghc_dsk：尺寸");
#define PVO(t, s, c) ((((t) * HC + (s)) * GK + (c)) * PDS) // pv 的位移

// grid：(GK, HC, GV_RP) × 32·GV_SG。權重先全部發出讀取，再算 h'·w（兩者的延遲重疊）
kernel void ghc_dsk(device const float *h [[buffer(0)]], device const half *out [[buffer(1)]], device const float *inj [[buffer(2)]],
                    device const float *w [[buffer(3)]], device const bfloat *Wd [[buffer(4)]], device const bfloat *Wi [[buffer(5)]],
                    device float *pv [[buffer(6)]], constant uint2 &fl [[buffer(7)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[NB][GKC];
    const uint c = tg.x, s = tg.y, rp = tg.z;
    const uint i0 = c * GKC + lane * GH_CPT, c0 = s * D + i0;
    // 1. 這個 SIMD group 的列（每組 GV_G 列；最後一份的最後一個 SIMD group 多做 inject 的 HC 列）全部先讀
    constexpr uint NGR = LR / (GV_RP * GV_SG * GV_G);
    constexpr uint NGT = NGR + 1;
    const bool inj_sg = fl.y && rp == GV_RP - 1 && sg == GV_SG - 1;
    uint4 u[NGT][GV_G];
#pragma clang loop unroll(full)
    for (uint gi = 0; gi < NGT; gi++) {
        const bool ok = gi < NGR || inj_sg;
        device const bfloat *W = gi < NGR ? Wd + (ulong)(((rp * NGR + gi) * GV_SG + sg) * GV_G) * HCD + c0 : Wi + c0;
#pragma clang loop unroll(full)
        for (uint r = 0; r < GV_G; r++) u[gi][r] = ok ? *(device const uint4 *)(W + (ulong)r * HCD) : uint4(0);
    }
    // 2. 第 sg 個 token 的 h'·w 與平方和（式子同 ghc_down）
    if (sg < NB) {
        const uint t = sg;
        const float4 w0 = *(device const float4 *)(w + c0), w1 = *(device const float4 *)(w + c0 + 4);
        const float js = fl.x ? gh_inj(inj + t * HC * HC, s) : 0.0f;
        device const float4 *hp = (device const float4 *)(h + (ulong)t * HCD + c0);
        device const half4 *op = (device const half4 *)(out + (ulong)t * D + i0);
        const float4 a = fl.x ? fma(float4(op[0]), js, hp[0]) : hp[0], b = fl.x ? fma(float4(op[1]), js, hp[1]) : hp[1];
        const float ss = simd_sum(dot(a, a) + dot(b, b));
        if (lane == 0 && rp == 0) pv[PVO(t, s, c) + LRI] = ss;
        *(threadgroup float4 *)&xs[t][lane * GH_CPT] = a * w0;
        *(threadgroup float4 *)&xs[t][lane * GH_CPT + 4] = b * w1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float x[NB][GH_CPT];
#pragma clang loop unroll(full)
    for (uint t = 0; t < NB; t++) {
        const float4 xa = *(threadgroup const float4 *)&xs[t][lane * GH_CPT], xb = *(threadgroup const float4 *)&xs[t][lane * GH_CPT + 4];
        for (uint k = 0; k < 4; k++) { x[t][k] = xa[k]; x[t][4 + k] = xb[k]; }
    }
    // 3. 每組：GV_G × NB 個內積（同 ghc_down：8 欄一條 fma 鏈、simd_sum），加總後 lane r·NB + t 拿 (r, t)
#pragma clang loop unroll(full)
    for (uint gi = 0; gi < NGT; gi++) {
        if (gi == NGR && !inj_sg) break;
        const uint r0 = gi < NGR ? ((rp * NGR + gi) * GV_SG + sg) * GV_G : LR;
        float mine = 0;
#pragma clang loop unroll(full)
        for (uint r = 0; r < GV_G; r++) {
            const uint4 q = u[gi][r];
            const float v[8] = {bf_lo(q.x), bf_hi(q.x), bf_lo(q.y), bf_hi(q.y), bf_lo(q.z), bf_hi(q.z), bf_lo(q.w), bf_hi(q.w)};
#pragma clang loop unroll(full)
            for (uint t = 0; t < NB; t++) {
                float d = 0;
#pragma clang loop unroll(full)
                for (uint k = 0; k < 8; k++) d = fma(v[k], x[t][k], d);
                const float sm = simd_sum(d);
                mine = lane == r * NB + t ? sm : mine;
            }
        }
        if (lane < GV_G * NB) {
            const uint r = lane / NB, t = lane % NB;
            pv[PVO(t, s, c) + r0 + r] = mine;
        }
    }
}

// grid：NB × LR（每個 threadgroup 一個 token，每個執行緒一列）。sxo [NB][LR]、rso [NB][HC]、injo（同 ghc_up）
kernel void ghc_mid(device const float *pv [[buffer(0)]], device float *sxo [[buffer(1)]], device float *rso [[buffer(2)]],
                    device float *injo [[buffer(3)]], constant uint2 &fl [[buffer(4)]],
                    uint t [[threadgroup_position_in_grid]], uint r [[thread_index_in_threadgroup]]) {
    threadgroup float rs[HC];
    // 各段依序加總（同兩段版 ghc_down 的 SIMD group 加總順序）
    float v[HC];
#pragma clang loop unroll(full)
    for (uint s = 0; s < HC; s++) {
        float a = 0;
#pragma clang loop unroll(full)
        for (uint c = 0; c < GK; c++) a += pv[PVO(t, s, c) + r];
        v[s] = a;
    }
    if (r < HC) {
        float ss = 0;
#pragma clang loop unroll(full)
        for (uint c = 0; c < GK; c++) ss += pv[PVO(t, r, c) + LRI];
        const float rr = rsqrt(ss / D + EPS);
        rs[r] = rr;
        rso[t * HC + r] = rr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float d = 0;
#pragma clang loop unroll(full)
    for (uint q = 0; q < HC; q++) d = fma(rs[q], v[q], d);
    sxo[t * LR + r] = gh_silu(d / HC);
    if (fl.y && r < HC * HC) { // inject 部分和（乘上 r[s]）→ injp[t][s][q]
        const uint s2 = r / HC, q = r % HC;
        float a = 0;
#pragma clang loop unroll(full)
        for (uint c = 0; c < GK; c++) a += pv[PVO(t, s2, c) + LR + q];
        injo[t * HC * HC + r] = rs[s2] * a;
    }
}

// grid：D / GU_SG × 32·GU_SG。同 ghc_up 的步驟 3、4（數值相同），sx / r 由 ghc_mid 算好
kernel void ghc_upv(device float *h [[buffer(0)]], device const half *out [[buffer(1)]], device const float *inj [[buffer(2)]],
                    device const float *w [[buffer(3)]], device const bfloat *Wu [[buffer(4)]], device const float *sxi [[buffer(5)]],
                    device const float *rsi [[buffer(6)]], device half *x [[buffer(7)]], constant uint2 &fl [[buffer(8)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint i = tg * GU_SG + sg;
    constexpr uint V = GH_POW2(NB * HC);
    const uint idx = lane % V, t = idx / HC, s = idx % HC;
    uint u[HC][5];
#pragma clang loop unroll(full)
    for (uint q = 0; q < HC; q++) {
        device const uint *wr = (device const uint *)(Wu + (ulong)(q * D + i) * LR) + lane * 5;
#pragma clang loop unroll(full)
        for (uint k = 0; k < 5; k++) u[q][k] = wr[k];
    }
    const uint tc = t < NB ? t : 0;
    const ulong hi = (ulong)tc * HCD + s * D + i;
    const float h0 = h[hi], o0 = fl.x ? float(out[(ulong)tc * D + i]) : 0.0f, w0 = w[s * D + i];
    const float rv = rsi[tc * HC + s], jv = fl.x ? gh_inj(inj + tc * HC * HC, s) : 0.0f;
    float acc[NB * HC];
    float v[HC][10];
#pragma clang loop unroll(full)
    for (uint q = 0; q < HC; q++)
#pragma clang loop unroll(full)
        for (uint k = 0; k < 5; k++) { v[q][2 * k] = bf_lo(u[q][k]); v[q][2 * k + 1] = bf_hi(u[q][k]); }
#pragma clang loop unroll(full)
    for (uint tt = 0; tt < NB; tt++) {
        float xv[10];
        device const float2 *sp = (device const float2 *)(sxi + tt * LR + lane * 10);
#pragma clang loop unroll(full)
        for (uint k = 0; k < 5; k++) { const float2 p = sp[k]; xv[2 * k] = p.x; xv[2 * k + 1] = p.y; }
#pragma clang loop unroll(full)
        for (uint q = 0; q < HC; q++) {
            float d = 0;
#pragma clang loop unroll(full)
            for (uint k = 0; k < 10; k++) d = fma(v[q][k], xv[k], d);
            acc[tt * HC + q] = d;
        }
    }
    float uv = 0;
#pragma clang loop unroll(full)
    for (uint j = 0; j < NB * HC; j++) { const float sj = simd_sum(acc[j]); uv = idx == j ? sj : uv; }
    float m = 0;
    if (t < NB) {
        const float hv = fl.x ? fma(o0, jv, h0) : h0;
        if (fl.x && lane < V) h[hi] = hv;
        m = gh_sigm(uv) * gh_pin(gh_pin(hv * rv, lane) * w0, lane);
    }
    m += simd_shuffle_xor(m, 1);
    m += simd_shuffle_xor(m, 2);
    if (t < NB && s == 0 && lane < V) x[(ulong)t * D + i] = half(m / HC);
}
#endif // NB

#ifdef GH_PF
// ---------------------------------------------------------------- prefill（T 個 token）
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#ifndef GH_BFT
#define GH_BFT 0
#endif
#ifndef GTM
#define GTM 64
#endif
#ifndef GTN
#define GTN 16
#endif
#ifndef GSG
#define GSG 2
#endif
inline float gh_tg_sum(float v, threadgroup float *tmp, uint tid, uint ntg) {
    v = simd_sum(v);
    const uint sg = tid / 32, lane = tid % 32, nsg = (ntg + 31) / 32;
    if (lane == 0) tmp[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? tmp[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// 殘差（可選）+ 分流 RMSNorm（輸出 FP16 給 mix_down）+ inject 部分和；FP32 的正規化結果不寫出，只寫每條流的 r[t][s]
// （ghp_up 從 h 重算）。grid：(HC, T) × D / 4（每個執行緒 4 個連續元素，留在暫存器）
kernel void ghp_hc_in(device float *h [[buffer(0)]], device const half *out [[buffer(1)]], device const float *injp_in [[buffer(2)]],
                      device const float *w [[buffer(3)]], device half *yh [[buffer(4)]], device float *rs [[buffer(5)]],
                      constant uint &apply [[buffer(6)]], device const bfloat *injW [[buffer(7)]], device float *injp_out [[buffer(8)]],
                      constant uint &want_inj [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tmp[32];
    constexpr uint NT = D / 4;
    const uint s = tg.x, t = tg.y, i = tid * 4;
    device float4 *hp = (device float4 *)(h + (ulong)t * HCD + s * D + i);
    float4 x = *hp;
    if (apply) {
        const float js = gh_inj(injp_in + t * HC * HC, s);
        x = fma(float4(*(device const half4 *)(out + (ulong)t * D + i)), js, x);
        *hp = x;
    }
    const float r = rsqrt(gh_tg_sum(dot(x, x), tmp, tid, NT) / D + EPS);
    const float4 y = x * r * *(device const float4 *)(w + s * D + i);
    *(device half4 *)(yh + (ulong)t * HCD + s * D + i) = half4(y);
    if (tid == 0) rs[t * HC + s] = r;
    if (want_inj)
        for (uint q = 0; q < HC; q++) {
            const uint2 u = *(device const uint2 *)(injW + (ulong)q * HCD + s * D + i);
            const float4 wq = float4(bf_lo(u.x), bf_hi(u.x), bf_lo(u.y), bf_hi(u.y));
            const float v = gh_tg_sum(dot(wq, y), tmp, tid, NT);
            if (tid == 0) injp_out[(t * HC + s) * HC + q] = v;
        }
}

// mix_up（tensor 單元）+ 合流：x[m, i] = mean_s σ(u[m, s·D + i]) · h[m, s, i]·r[m][s]·w[s, i]，u = X · Wuᵀ（X 是 SiLU(d / HC)，FP16）。
// 每個 threadgroup：GTM 個 token × GTN 個 i，4 條流各一個累加器（同一個 i 的 4 條流在同一個 threadgroup，u 不必寫出）。
// 權重 BF16 → FP16 放進 threadgroup memory。grid：(D / GTN, ceil(M / GTM)) × 32·GSG
#define KC 64
static_assert(LR % KC == 0 && D % GTN == 0, "ghp_up：尺寸");
#if GH_BFT
// GH_BFT：BF16 權重直接當 tensor 運算元（device memory，不轉 FP16、不經 threadgroup memory）
kernel void ghp_up(device half *X [[buffer(0)]], device bfloat *Wu [[buffer(1)]], device const float *h [[buffer(2)]],
                   device const float *rs [[buffer(3)]], device const float *w [[buffer(4)]], device half *xo [[buffer(5)]],
                   constant uint &M [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) {
    const uint m0 = tg.y * GTM, i0 = tg.x * GTN;
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX(X, T2(LR, M));
    tensor<device bfloat, T2, tensor_inline> tW(Wu, T2(LR, HCD));
    constexpr auto desc = matmul2d_descriptor(GTM, GTN, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<GSG>> op;
    auto x0 = tX.slice(0, m0);
    auto w0 = tW.slice(0, i0);
    auto a0 = op.get_destination_cooperative_tensor<decltype(x0), decltype(w0), float>();
    auto a1 = op.get_destination_cooperative_tensor<decltype(x0), decltype(w0), float>();
    auto a2 = op.get_destination_cooperative_tensor<decltype(x0), decltype(w0), float>();
    auto a3 = op.get_destination_cooperative_tensor<decltype(x0), decltype(w0), float>();
#pragma unroll
    for (uint16_t i = 0; i < a0.get_capacity(); ++i)
        if (a0.is_valid_element(i)) { a0[i] = 0; a1[i] = 0; a2[i] = 0; a3[i] = 0; }
    for (uint k0 = 0; k0 < LR; k0 += KC) {
        auto xs = tX.slice(k0, m0);
        auto ws0 = tW.slice(k0, i0), ws1 = tW.slice(k0, D + i0), ws2 = tW.slice(k0, 2 * D + i0), ws3 = tW.slice(k0, 3 * D + i0);
        op.run(xs, ws0, a0);
        op.run(xs, ws1, a1);
        op.run(xs, ws2, a2);
        op.run(xs, ws3, a3);
    }
#pragma unroll
    for (uint16_t i = 0; i < a0.get_capacity(); ++i)
        if (a0.is_valid_element(i)) {
            auto ix = a0.get_multidimensional_index(i);
            const uint n = i0 + ix[0], m = m0 + ix[1];
            if (m >= M) continue;
            device const float *hm = h + (ulong)m * HCD + n;
            device const float *rm = rs + m * HC;
            const float v = gh_sigm(a0[i]) * (hm[0] * rm[0] * w[n]) + gh_sigm(a1[i]) * (hm[D] * rm[1] * w[D + n]) +
                            gh_sigm(a2[i]) * (hm[2 * D] * rm[2] * w[2 * D + n]) + gh_sigm(a3[i]) * (hm[3 * D] * rm[3] * w[3 * D + n]);
            xo[(ulong)m * D + n] = half(v / HC);
        }
}
#else
kernel void ghp_up(device half *X [[buffer(0)]], device const bfloat *Wu [[buffer(1)]], device const float *h [[buffer(2)]],
                   device const float *rs [[buffer(3)]], device const float *w [[buffer(4)]], device half *xo [[buffer(5)]],
                   constant uint &M [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NT = 32 * GSG;
    const uint m0 = tg.y * GTM, i0 = tg.x * GTN;
    threadgroup half Wt[HC * GTN * KC];
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX(X, T2(LR, M));
    tensor<threadgroup half, T2, tensor_inline> tW0(Wt, T2(KC, GTN)), tW1(Wt + GTN * KC, T2(KC, GTN)),
        tW2(Wt + 2 * GTN * KC, T2(KC, GTN)), tW3(Wt + 3 * GTN * KC, T2(KC, GTN));
    constexpr auto desc = matmul2d_descriptor(GTM, GTN, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<GSG>> op;
    auto x0 = tX.slice(0, m0);
    auto a0 = op.get_destination_cooperative_tensor<decltype(x0), decltype(tW0), float>();
    auto a1 = op.get_destination_cooperative_tensor<decltype(x0), decltype(tW0), float>();
    auto a2 = op.get_destination_cooperative_tensor<decltype(x0), decltype(tW0), float>();
    auto a3 = op.get_destination_cooperative_tensor<decltype(x0), decltype(tW0), float>();
#pragma unroll
    for (uint16_t i = 0; i < a0.get_capacity(); ++i)
        if (a0.is_valid_element(i)) { a0[i] = 0; a1[i] = 0; a2[i] = 0; a3[i] = 0; }
    for (uint k0 = 0; k0 < LR; k0 += KC) {
        for (uint q = tid; q < HC * GTN * KC / 8; q += NT) { // 每個執行緒一次 8 個 BF16
            const uint s = q / (GTN * KC / 8), n = q / (KC / 8) % GTN, j = q % (KC / 8);
            const uint4 u = *(device const uint4 *)(Wu + (ulong)(s * D + i0 + n) * LR + k0 + j * 8);
            threadgroup half *dst = Wt + (s * GTN + n) * KC + j * 8;
            dst[0] = half(bf_lo(u.x)); dst[1] = half(bf_hi(u.x)); dst[2] = half(bf_lo(u.y)); dst[3] = half(bf_hi(u.y));
            dst[4] = half(bf_lo(u.z)); dst[5] = half(bf_hi(u.z)); dst[6] = half(bf_lo(u.w)); dst[7] = half(bf_hi(u.w));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto xs = tX.slice(k0, m0);
        op.run(xs, tW0, a0);
        op.run(xs, tW1, a1);
        op.run(xs, tW2, a2);
        op.run(xs, tW3, a3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (uint16_t i = 0; i < a0.get_capacity(); ++i)
        if (a0.is_valid_element(i)) {
            auto ix = a0.get_multidimensional_index(i);
            const uint n = i0 + ix[0], m = m0 + ix[1];
            if (m >= M) continue;
            device const float *hm = h + (ulong)m * HCD + n;
            device const float *rm = rs + m * HC;
            const float v = gh_sigm(a0[i]) * (hm[0] * rm[0] * w[n]) + gh_sigm(a1[i]) * (hm[D] * rm[1] * w[D + n]) +
                            gh_sigm(a2[i]) * (hm[2 * D] * rm[2] * w[2 * D + n]) + gh_sigm(a3[i]) * (hm[3 * D] * rm[3] * w[3 * D + n]);
            xo[(ulong)m * D + n] = half(v / HC);
        }
}
#endif
// mix_down（tensor 單元）+ SiLU：Y[m, r] = SiLU((X · Wdᵀ)[m, r] / HC)，X 是正規化後的 FP16 [M][HCD]、Wd 是 BF16 [LR][HCD]。
// 每個 threadgroup：GDM 個 token × GDN 列；權重 BF16 → FP16 放進 threadgroup memory。grid：(LR / GDN, ceil(M / GDM)) × 32·GDS
#ifndef GDM
#define GDM 128
#endif
#ifndef GDN
#define GDN 64
#endif
#ifndef GDS
#define GDS 4
#endif
static_assert(LR % GDN == 0 && HCD % KC == 0, "ghp_down：尺寸");
#if GH_BFT
kernel void ghp_down(device half *X [[buffer(0)]], device bfloat *Wd [[buffer(1)]], device half *Y [[buffer(2)]],
                     constant uint &M [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
    const uint m0 = tg.y * GDM, n0 = tg.x * GDN;
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX(X, T2(HCD, M));
    tensor<device bfloat, T2, tensor_inline> tW(Wd, T2(HCD, LR));
    constexpr auto desc = matmul2d_descriptor(GDM, GDN, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<GDS>> op;
    auto x0 = tX.slice(0, m0);
    auto w0 = tW.slice(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(x0), decltype(w0), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    for (uint k0 = 0; k0 < HCD; k0 += KC) {
        auto xs = tX.slice(k0, m0);
        auto ws = tW.slice(k0, n0);
        op.run(xs, ws, acc);
    }
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const uint n = n0 + ix[0], m = m0 + ix[1];
            if (m < M) Y[(ulong)m * LR + n] = half(gh_silu(acc[i] / HC));
        }
}
#else
kernel void ghp_down(device half *X [[buffer(0)]], device const bfloat *Wd [[buffer(1)]], device half *Y [[buffer(2)]],
                     constant uint &M [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NT = 32 * GDS;
    const uint m0 = tg.y * GDM, n0 = tg.x * GDN;
    threadgroup half Wt[GDN * KC];
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX(X, T2(HCD, M));
    tensor<threadgroup half, T2, tensor_inline> tW(Wt, T2(KC, GDN));
    constexpr auto desc = matmul2d_descriptor(GDM, GDN, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<GDS>> op;
    auto x0 = tX.slice(0, m0);
    auto acc = op.get_destination_cooperative_tensor<decltype(x0), decltype(tW), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    for (uint k0 = 0; k0 < HCD; k0 += KC) {
        for (uint q = tid; q < GDN * KC / 8; q += NT) {
            const uint n = q / (KC / 8), j = q % (KC / 8);
            const uint4 u = *(device const uint4 *)(Wd + (ulong)(n0 + n) * HCD + k0 + j * 8);
            threadgroup half *dst = Wt + n * KC + j * 8;
            dst[0] = half(bf_lo(u.x)); dst[1] = half(bf_hi(u.x)); dst[2] = half(bf_lo(u.y)); dst[3] = half(bf_hi(u.y));
            dst[4] = half(bf_lo(u.z)); dst[5] = half(bf_hi(u.z)); dst[6] = half(bf_lo(u.w)); dst[7] = half(bf_hi(u.w));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto xs = tX.slice(k0, m0);
        op.run(xs, tW, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const uint n = n0 + ix[0], m = m0 + ix[1];
            if (m < M) Y[(ulong)m * LR + n] = half(gh_silu(acc[i] / HC));
        }
}
#endif
#endif // GH_PF
