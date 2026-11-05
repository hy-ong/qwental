// Prefill（一次 T 個 token）的逐元素與小型運算。矩陣乘法用 qmm.metal / gdn.metal / moe_prefill.metal。
// 排列：每個 token 一列。殘差流 h：[T][HC·D] FP32；送進 qmm 的輸入：[T][K] FP16。
// 主程式注入：D（2560）、HC（4）

#include <metal_stdlib>
using namespace metal;

#define EPS 1e-6f
#define HCD (HC * D)

inline float silu(float x) { return x / (1.0f + exp(-x)); }
inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

inline float tg_sum(float v, threadgroup float *tmp, uint tid, uint ntg) {
    v = simd_sum(v);
    const uint sg = tid / 32, lane = tid % 32, nsg = (ntg + 31) / 32;
    if (lane == 0) tmp[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? tmp[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// embed：tok[p0 + t] → h[t]（4-bit、group 64），複製成 HC 條流。grid：(D / 256, T) × 256
kernel void p_embed(device const uint *W [[buffer(0)]], device const bfloat *S [[buffer(1)]], device const bfloat *Z [[buffer(2)]],
                    device const int *tok [[buffer(3)]], constant uint &p0 [[buffer(4)]], device float *h [[buffer(5)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, t = gid.y;
    if (i >= D) return;
    const ulong row = (ulong)tok[p0 + t];
    const uint word = W[row * (D / 8) + i / 8];
    const float q = float((word >> (4 * (i % 8))) & 15u);
    const float v = float(bfloat(q * float(S[row * (D / 64) + i / 64]) + float(Z[row * (D / 64) + i / 64])));
    for (uint s = 0; s < HC; s++) h[(ulong)t * HCD + s * D + i] = v;
}

// inject 係數（由部分和）
inline float inj_of(device const float *injp, uint r) {
    float t = 0;
    for (uint s = 0; s < HC; s++) t += injp[s * HC + r];
    return 2.0f / (1.0f + exp(-t / HC));
}

// 殘差更新（可選）+ 分流 RMSNorm + inject 部分和。grid：(HC, T) × 1024
//   injp_in / injp_out：[T][HC][HC]
kernel void p_hc_in(device float *h [[buffer(0)]], device const half *out [[buffer(1)]], device const float *injp_in [[buffer(2)]],
                    device const bfloat *w [[buffer(3)]], device half *yh [[buffer(4)]], device float *yf [[buffer(5)]],
                    constant uint &apply [[buffer(6)]], device const bfloat *injW [[buffer(7)]], device float *injp_out [[buffer(8)]],
                    constant uint &want_inj [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint2 ntg2 [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    const uint ntg = ntg2.x;
    const uint s = tg.x, t = tg.y;
    device float *ht = h + (ulong)t * HCD;
    const float js = apply ? inj_of(injp_in + t * HC * HC, s) : 0.0f;
    float ss = 0;
    for (uint i = tid; i < D; i += ntg) {
        float x = ht[s * D + i];
        if (apply) {
            x += float(out[(ulong)t * D + i]) * js;
            ht[s * D + i] = x;
        }
        ss += x * x;
    }
    const float r = rsqrt(tg_sum(ss, tmp, tid, ntg) / D + EPS);
    float pj[HC];
    for (uint q = 0; q < HC; q++) pj[q] = 0;
    for (uint i = tid; i < D; i += ntg) {
        const float y = ht[s * D + i] * r * float(w[s * D + i]);
        yh[(ulong)t * HCD + s * D + i] = half(y);
        yf[(ulong)t * HCD + s * D + i] = y;
        if (want_inj)
            for (uint q = 0; q < HC; q++) pj[q] += float(injW[(ulong)q * HCD + s * D + i]) * y;
    }
    if (want_inj)
        for (uint q = 0; q < HC; q++) {
            const float v = tg_sum(pj[q], tmp, tid, ntg);
            if (tid == 0) injp_out[(t * HC + s) * HC + q] = v;
        }
}

// 只做殘差更新：h[t, s, i] += out[t, i] · inj[t][s]。grid：T·HCD
kernel void p_resid(device float *h [[buffer(0)]], device const half *out [[buffer(1)]], device const float *injp [[buffer(2)]],
                    constant uint &T [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= T * HCD) return;
    const uint t = gid / HCD, j = gid % HCD;
    h[gid] += float(out[(ulong)t * D + j % D]) * inj_of(injp + t * HC * HC, j / D);
}

// y = silu(x / HC)，n 個元素
kernel void p_silu_div(device const half *x [[buffer(0)]], device half *y [[buffer(1)]], constant uint &n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = half(silu(float(x[i]) / HC));
}

// x[t, i] = mean_s sigmoid(u[t, s·D + i]) · nf[t, s·D + i]。grid：T·D
kernel void p_hcmix(device const half *u [[buffer(0)]], device const float *nf [[buffer(1)]], device half *x [[buffer(2)]],
                    constant uint &T [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= T * D) return;
    const uint t = gid / D, i = gid % D;
    float acc = 0;
    for (uint s = 0; s < HC; s++) acc += sigm(float(u[(ulong)t * HCD + s * D + i])) * nf[(ulong)t * HCD + s * D + i];
    x[gid] = half(acc / HC);
}

// ---------------------------------------------------------------- DeltaNet

// conv4 + SiLU（整段 T 個 token，因果）＋ q/k 正規化、v、log 衰減、β。
// grid：(KH + VH, T) × 128。xin：[T][10240]（qkv 投影）；st：[3][10240]（上一段最後 3 個 token 的輸入）
// 輸出：q、k [T][16][128]、v [T][48][128]（FP16）、la、be [T][48]
// log 衰減的係數：MLX 存 A_log（BF16）、GGUF 存 A_neg = −exp(A_log)（F32；p_gdn_in_t）
inline float a_neg(device const bfloat *A_log, uint i) { return -exp(float(A_log[i])); }
inline float a_neg(device const float *A_neg, uint i) { return A_neg[i]; }
template <typename AT>
inline void p_gdn_in_body(device const half *xin, device const bfloat *w, device const float *st, device const half *ain,
                          device const half *bin, device const AT *A, device const bfloat *dt_bias, device half *q, device half *k,
                          device half *v, device float *la, device float *be, uint2 tg, uint tid, threadgroup float *tmp) {
    constexpr uint DK = 128, KH = 16, VH = 48, C = 10240;
    const uint hh = tg.x, t = tg.y;
    auto inp = [&](int tt, uint c) -> float { return tt >= 0 ? float(xin[(ulong)tt * C + c]) : st[(3 + tt) * C + c]; };
    auto conv = [&](uint c) -> float {
        const int tt = (int)t;
        const float o = float(w[c * 4]) * inp(tt - 3, c) + float(w[c * 4 + 1]) * inp(tt - 2, c) + float(w[c * 4 + 2]) * inp(tt - 1, c) +
                        float(w[c * 4 + 3]) * inp(tt, c);
        return float(half(silu(o)));
    };
    if (hh < KH) {
        const float qv = conv(hh * DK + tid), kv = conv(KH * DK + hh * DK + tid);
        const float rq = rsqrt(tg_sum(qv * qv, tmp, tid, DK) / DK + EPS);
        const float rk = rsqrt(tg_sum(kv * kv, tmp, tid, DK) / DK + EPS);
        q[((ulong)t * KH + hh) * DK + tid] = half(qv * rq / DK);
        k[((ulong)t * KH + hh) * DK + tid] = half(kv * rk * rsqrt(float(DK)));
        if (hh == 0 && tid < VH) {
            const float xa = float(ain[(ulong)t * VH + tid]) + float(dt_bias[tid]);
            const float sp = xa > 20.0f ? xa : log(1.0f + exp(xa));
            la[(ulong)t * VH + tid] = a_neg(A, tid) * sp;
            be[(ulong)t * VH + tid] = sigm(float(bin[(ulong)t * VH + tid]));
        }
    } else {
        const uint j = hh - KH;
        v[((ulong)t * VH + j) * DK + tid] = half(conv(2 * KH * DK + j * DK + tid));
    }
}

#define P_GDN_IN_ARGS(AT)                                                                                                      \
    device const half *xin [[buffer(0)]], device const bfloat *w [[buffer(1)]], device const float *st [[buffer(2)]],          \
        device const half *ain [[buffer(3)]], device const half *bin [[buffer(4)]], device const AT *A [[buffer(5)]],         \
        device const bfloat *dt_bias [[buffer(6)]], device half *q [[buffer(7)]], device half *k [[buffer(8)]],                \
        device half *v [[buffer(9)]], device float *la [[buffer(10)]], device float *be [[buffer(11)]],                       \
        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
kernel void p_gdn_in(P_GDN_IN_ARGS(bfloat)) {
    threadgroup float tmp[32];
    p_gdn_in_body(xin, w, st, ain, bin, A, dt_bias, q, k, v, la, be, tg, tid, tmp);
}
kernel void p_gdn_in_t(P_GDN_IN_ARGS(float)) {
    threadgroup float tmp[32];
    p_gdn_in_body(xin, w, st, ain, bin, A, dt_bias, q, k, v, la, be, tg, tid, tmp);
}

// 更新 conv 狀態：最後 3 個 token 的輸入（T 可能 < 3，要跟舊狀態接起來）。grid：10240
kernel void p_conv_state(device const half *xin [[buffer(0)]], device float *st [[buffer(1)]], constant uint &T [[buffer(2)]],
                         constant uint &C [[buffer(3)]], uint c [[thread_position_in_grid]]) {
    if (c >= C) return;
    float hist[3 + 3];
    for (uint j = 0; j < 3; j++) hist[j] = st[j * C + c];
    // 新序列 = 舊狀態 + T 個輸入；取最後 3 個
    float last[3];
    for (int j = 0; j < 3; j++) {
        const int idx = (int)T - 3 + j; // 相對於新輸入
        last[j] = idx >= 0 ? float(xin[(ulong)idx * C + c]) : hist[3 + idx];
    }
    for (uint j = 0; j < 3; j++) st[j * C + c] = last[j];
}

// gated RMSNorm（每個 token、每個 value head）。grid：(48, T) × 128
kernel void p_normgate(device const half *o [[buffer(0)]], device const half *z [[buffer(1)]], device const bfloat *w [[buffer(2)]],
                       device half *y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint DV = 128;
    threadgroup float tmp[32];
    const ulong base = ((ulong)tg.y * 48 + tg.x) * DV;
    const float x = float(o[base + tid]);
    const float r = rsqrt(tg_sum(x * x, tmp, tid, DV) / DV + EPS);
    y[base + tid] = half(x * r * float(w[tid]) * sigm(float(z[base + tid])));
}

// ---------------------------------------------------------------- attention（dense causal，context ≤ 預算）

inline void rope_pair(thread float &a, thread float &b, uint d, uint pos, float theta, uint rd) {
    const float f = pow(theta, -float(2 * d) / float(rd)) * float(pos);
    const float c = cos(f), s = sin(f);
    const float x1 = a, x2 = b;
    a = x1 * c - x2 * s;
    b = x2 * c + x1 * s;
}

// RoPE 的 cos / sin 表：rcs[t][d] = (cos, sin)(θ^(−2d/RD)·(pos0 + t))，d < 32（RD = 64、θ = 1e7，attention 與 indexer 相同）。
// 與 rope_pair 同一個算式（結果逐位元相同）。位置大時 sin / cos 的引數化簡很慢（pos ≥ 32768 時 p_attn_prep 慢約 2 倍），
// 原本每個 token 的 24 + 2 個 head（加上 indexer 4 個）各算一次，現在每段每個位置只算一次。grid：(32, T)
kernel void p_rope_tab(device float2 *rcs [[buffer(0)]], constant uint &pos0 [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    constexpr uint RD = 64;
    constexpr float THETA = 10000000.0f;
    const uint d = gid.x, t = gid.y;
    const float f = pow(THETA, -float(2 * d) / float(RD)) * float(pos0 + t);
    rcs[t * (RD / 2) + d] = float2(cos(f), sin(f));
}

// rope_pair 的後半：用查表的 (cos, sin) 旋轉。fma 的位置照編譯器對 rope_pair 的收縮方式寫死
// （a = fma(x1, c, −x2·s)、b = fma(x1, s, x2·c)），結果才與 rope_pair 逐位元相同（寫成 x1 * c − x2 * s 會差 1 ulp）
inline void rope_cs(thread float &a, thread float &b, float2 cs) {
    const float c = cs.x, s = cs.y;
    const float x1 = a, x2 = b;
    a = fma(x1, c, -(x2 * s));
    b = fma(x1, s, x2 * c);
}

// q/k 的 RMSNorm + RoPE，k、v 寫進 cache。grid：(24 + 2, T) × 256。qg：[T][12288]，kin/vin：[T][512]
// norm 權重型別 NT：MLX 是 bfloat，GGUF profile 是 float（_f 版）
template <typename NT>
inline void p_attn_prep_body(device const half *qg, device const half *kin, device const half *vin, device const NT *qn,
                             device const NT *kn, device half *q, device half *Kc, device half *Vc, device const float2 *rcs,
                             uint pos0, uint2 tg, uint tid, threadgroup float *tmp, threadgroup float *v) {
    constexpr uint HD = 256, QH = 24, KVH = 2, RD = 64;
    const uint hh = tg.x, t = tg.y, pos = pos0 + t;
    const bool isq = hh < QH;
    const uint kh = hh - QH;
    const float x = isq ? float(qg[(ulong)t * 12288 + hh * 2 * HD + tid]) : float(kin[(ulong)t * 512 + kh * HD + tid]);
    const float r = rsqrt(tg_sum(x * x, tmp, tid, HD) / HD + EPS);
    v[tid] = x * r * float(isq ? qn[tid] : kn[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < RD / 2) {
        float a = v[tid], b = v[tid + RD / 2];
        rope_cs(a, b, rcs[t * (RD / 2) + tid]);
        v[tid] = a;
        v[tid + RD / 2] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (isq) q[((ulong)t * QH + hh) * HD + tid] = half(v[tid]);
    else {
        Kc[((ulong)pos * KVH + kh) * HD + tid] = half(v[tid]);
        Vc[((ulong)pos * KVH + kh) * HD + tid] = vin[(ulong)t * 512 + kh * HD + tid];
    }
}

#define P_ATTN_PREP(NAME, NT)                                                                                                       \
    kernel void NAME(device const half *qg [[buffer(0)]], device const half *kin [[buffer(1)]], device const half *vin [[buffer(2)]], \
                     device const NT *qn [[buffer(3)]], device const NT *kn [[buffer(4)]], device half *q [[buffer(5)]],             \
                     device half *Kc [[buffer(6)]], device half *Vc [[buffer(7)]], constant uint &pos0 [[buffer(8)]],                \
                     device const float2 *rcs [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],                              \
                     uint tid [[thread_index_in_threadgroup]]) {                                                                     \
        threadgroup float tmp[32], v[256];                                                                                            \
        p_attn_prep_body<NT>(qg, kin, vin, qn, kn, q, Kc, Vc, rcs, pos0, tg, tid, tmp, v);                                            \
    }
P_ATTN_PREP(p_attn_prep, bfloat)
P_ATTN_PREP(p_attn_prep_f, float)

// indexer：raw_k 存進 cache（BF16 精度）。grid：(T) × 128
kernel void p_idx_rawk(device const half *iqk [[buffer(0)]], device float *rawk [[buffer(1)]], constant uint &pos0 [[buffer(2)]],
                       uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    rawk[((ulong)pos0 + t) * 128 + tid] = float(bfloat(float(iqk[(ulong)t * 640 + 4 * 128 + tid])));
}

// indexer：新完成的區塊 key（區塊 nb 的 4 個 token 都已寫入 raw_k）。grid：(nblocks) × 128，nb = nb0 + tg
template <typename NT>
inline void p_idx_blocks_body(device const float *rawk, device const NT *kn, device half *kblk, uint nb0, uint tg, uint tid,
                              threadgroup float *tmp, threadgroup float *v) {
    constexpr uint ID = 128, RD = 64, CR = 4;
    constexpr float THETA = 10000000.0f;
    const uint nb = nb0 + tg;
    float m = 0;
    for (uint j = 0; j < CR; j++) m += rawk[((ulong)nb * CR + j) * ID + tid];
    m = float(bfloat(m / CR));
    const float r = rsqrt(tg_sum(m * m, tmp, tid, ID) / ID + EPS);
    v[tid] = m * r * float(kn[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < RD / 2) {
        float a = v[tid], b = v[tid + RD / 2];
        rope_pair(a, b, tid, nb * CR, THETA, RD);
        v[tid] = a;
        v[tid + RD / 2] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kblk[(ulong)nb * ID + tid] = half(v[tid]);
}

kernel void p_idx_blocks(device const float *rawk [[buffer(0)]], device const bfloat *kn [[buffer(1)]], device half *kblk [[buffer(2)]],
                         constant uint &nb0 [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tmp[32], v[128];
    p_idx_blocks_body<bfloat>(rawk, kn, kblk, nb0, tg, tid, tmp, v);
}
kernel void p_idx_blocks_f(device const float *rawk [[buffer(0)]], device const float *kn [[buffer(1)]], device half *kblk [[buffer(2)]],
                           constant uint &nb0 [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tmp[32], v[128];
    p_idx_blocks_body<float>(rawk, kn, kblk, nb0, tg, tid, tmp, v);
}

// dense causal attention（context ≤ 2048，此時 QSA 不稀疏）。每個 threadgroup 一個 (query, kv head)：
// 12 個 SIMD group = 12 個 q head，各自 online softmax 掃過 0..pos 的 key。grid：(T, 2) × 384
kernel void p_attn_dense(device const half *q [[buffer(0)]], device const half *Kc [[buffer(1)]], device const half *Vc [[buffer(2)]],
                         device const half *qg [[buffer(3)]], device half *y [[buffer(4)]], constant uint &pos0 [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint HD = 256, QH = 24, KVH = 2, GQ = QH / KVH, PL = HD / 32;
    const uint t = tg.x, kvh = tg.y, qh = kvh * GQ + sg, pos = pos0 + t;
    const float scale = rsqrt(float(HD));
    float qv[PL], acc[PL];
    for (uint d = 0; d < PL; d++) {
        qv[d] = float(q[((ulong)t * QH + qh) * HD + lane * PL + d]) * scale;
        acc[d] = 0;
    }
    float m = -INFINITY, l = 0;
    for (uint j = 0; j <= pos; j++) {
        device const half *kr = Kc + ((ulong)j * KVH + kvh) * HD + lane * PL;
        device const half *vr = Vc + ((ulong)j * KVH + kvh) * HD + lane * PL;
        float s = 0;
        for (uint d = 0; d < PL; d++) s += qv[d] * float(kr[d]);
        s = simd_sum(s);
        const float mn = max(m, s), al = exp(m - mn), p = exp(s - mn);
        l = l * al + p;
        for (uint d = 0; d < PL; d++) acc[d] = acc[d] * al + p * float(vr[d]);
        m = mn;
    }
    // 乘上 output gate：sigmoid(qg[t, qh·512 + 256 + d])
    for (uint d = 0; d < PL; d++) {
        const uint dd = lane * PL + d;
        const float gte = float(qg[(ulong)t * 12288 + qh * 2 * HD + HD + dd]);
        y[((ulong)t * QH + qh) * HD + dd] = half(float(half(acc[d] / l)) * sigm(gte));
    }
}

// ---------------------------------------------------------------- MoE

// router logits（FP32，[T][E]）→ 每個 token 選前 TOPK、softmax；另算共用 expert 的 sigmoid gate
// grid：T 個 threadgroup × 32。sgate：[T]（shared_expert_gate · x 的結果，FP32）
kernel void p_moe_select(device const float *logits [[buffer(0)]], device const float *sgl [[buffer(1)]],
                         device uint *ids [[buffer(2)]], device float *wts [[buffer(3)]], device float *sgate [[buffer(4)]],
                         uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint E = 512, TOPK = 10, PER = E / 32;
    float v[PER];
    for (uint j = 0; j < PER; j++) v[j] = logits[(ulong)t * E + lane * PER + j];
    float picked[TOPK];
    uint idv[TOPK];
    for (uint k = 0; k < TOPK; k++) {
        float best = -INFINITY;
        uint bi = 0;
        for (uint j = 0; j < PER; j++)
            if (v[j] > best) { best = v[j]; bi = lane * PER + j; }
        const float mm = simd_max(best);
        const uint id = simd_min(best == mm ? bi : 0xffffffffu);
        picked[k] = mm;
        idv[k] = id;
        if (id / PER == lane) v[id % PER] = -INFINITY;
    }
    if (lane == 0) {
        float mx = picked[0], s = 0;
        for (uint k = 0; k < TOPK; k++) s += exp(picked[k] - mx);
        for (uint k = 0; k < TOPK; k++) {
            ids[t * TOPK + k] = idv[k];
            wts[t * TOPK + k] = exp(picked[k] - mx) / s;
        }
        sgate[t] = sigm(sgl[t]);
    }
}

// 共用 expert gate 的 logit：sgl[t] = Σ_i w[i] · x[t, i]（BF16 權重）。grid：T × 256
kernel void p_dot_bf16(device const bfloat *w [[buffer(0)]], device const half *x [[buffer(1)]], device float *y [[buffer(2)]],
                       uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tmp[32];
    float acc = 0;
    for (uint i = tid; i < D; i += 256) acc += float(w[i]) * float(x[(ulong)t * D + i]);
    acc = tg_sum(acc, tmp, tid, 256);
    if (tid == 0) y[t] = acc;
}

// router 兩階段的第二步：approx（tensor 版的近似 logits）每個 token 取前 RC 名候選，候選用 p_router 同一套公式與加總順序重算
// （值與 p_router 逐位元相同），其餘設 −∞。只要真正的前 TOPK 名落在近似的前 RC 名內，選出的 expert 與權重就和精確版一樣。
// grid：T × 32
kernel void p_router_refine(device const uint *W [[buffer(0)]], device const bfloat *S [[buffer(1)]], device const bfloat *Z [[buffer(2)]],
                            device const half *x [[buffer(3)]], device float *y [[buffer(4)]], device const float *approx [[buffer(5)]],
                            uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint E = 512, RC = 16, PER = E / 32, NCH = D / 16, ROWW = D / 4, NG = D / 64;
    float v[PER];
    for (uint j = 0; j < PER; j++) v[j] = approx[(ulong)t * E + lane * PER + j];
    uint cand[RC];
    for (uint k = 0; k < RC; k++) {
        float best = -INFINITY;
        uint bi = 0;
        for (uint j = 0; j < PER; j++)
            if (v[j] > best) { best = v[j]; bi = lane * PER + j; }
        const float mm = simd_max(best);
        const uint id = simd_min(best == mm ? bi : 0xffffffffu);
        cand[k] = id;
        if (id / PER == lane) v[id % PER] = -INFINITY;
    }
    for (uint j = 0; j < PER; j++) y[(ulong)t * E + lane * PER + j] = -INFINITY;
    simdgroup_barrier(mem_flags::mem_device);
    device const half *xt = x + (ulong)t * D;
    for (uint k = 0; k < RC; k++) {
        const uint r = cand[k];
        float acc = 0;
        for (uint c = lane; c < NCH; c += 32) {
            const uint2 wl = ((device const uint2 *)(W + (ulong)r * ROWW))[2 * c], wh = ((device const uint2 *)(W + (ulong)r * ROWW))[2 * c + 1];
            float wq[16];
            for (int n = 0; n < 4; n++) {
                wq[n] = float((wl.x >> (8 * n)) & 255u);
                wq[4 + n] = float((wl.y >> (8 * n)) & 255u);
                wq[8 + n] = float((wh.x >> (8 * n)) & 255u);
                wq[12 + n] = float((wh.y >> (8 * n)) & 255u);
            }
            const uint g = c / 4;
            const float sc = float(S[(ulong)r * NG + g]), zz = float(Z[(ulong)r * NG + g]);
            device const half4 *xp = (device const half4 *)(xt + c * 16);
            float d = 0, xs = 0;
            for (int q = 0; q < 4; q++) {
                const float4 h = float4(xp[q]);
                d += wq[4 * q] * h.x + wq[4 * q + 1] * h.y + wq[4 * q + 2] * h.z + wq[4 * q + 3] * h.w;
                xs += h.x + h.y + h.z + h.w;
            }
            acc += sc * d + zz * xs;
        }
        const float val = simd_sum(acc);
        if (lane == 0) y[(ulong)t * E + r] = val;
    }
}

// H = silu(G) · U（gate 與 up 分開的兩個 buffer），n 個元素
kernel void p_act2(device const half *G [[buffer(0)]], device const half *U [[buffer(1)]], device half *H [[buffer(2)]],
                   constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) H[i] = half(silu(float(G[i])) * float(U[i]));
}

// out[t, c] = Σ_s w[t,s] · Dm[pos_of[t,s], c] + sgate[t] · Ds[t, c]。grid：(D / 4, T)
kernel void p_moe_combine(device const half *Dm [[buffer(0)]], device const uint *pos_of [[buffer(1)]], device const float *w [[buffer(2)]],
                          device const half *Ds [[buffer(3)]], device const float *sgate [[buffer(4)]], device half *out [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    constexpr uint TOPK = 10;
    const uint c4 = gid.x, t = gid.y;
    float4 acc = 0;
    for (uint s = 0; s < TOPK; s++) {
        const uint p = t * TOPK + s;
        acc += w[p] * float4(((device const half4 *)(Dm + (ulong)pos_of[p] * D))[c4]);
    }
    acc += sgate[t] * float4(((device const half4 *)(Ds + (ulong)t * D))[c4]);
    ((device half4 *)(out + (ulong)t * D))[c4] = half4(acc);
}

// ---------------------------------------------------------------- PLE

// n-gram（多個 token）：每個 token、每個 head 一個 threadgroup。grid：(16, T) × 160
kernel void p_ngram(device const int *hist [[buffer(0)]], constant uint &p0 [[buffer(1)]], device const long *mult [[buffer(2)]],
                    device const long *off [[buffer(3)]], device const long *size [[buffer(4)]], device const uint *TW [[buffer(5)]],
                    device const bfloat *TS [[buffer(6)]], device const bfloat *TB [[buffer(7)]], device half *emb [[buffer(8)]],
                    constant uint &eos [[buffer(9)]], constant int &gbase [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint i [[thread_index_in_threadgroup]]) {
    constexpr uint DIM = 160, WORDS = 20, NGS = 5, GSZ = 32;
    const uint head = tg.x, t = tg.y, p = p0 + t;
    if (gbase >= 0) { // CPU 已抄好：第 t 個位置的第 head 列在 gbase + t·16 + head
        const ulong row = (ulong)(gbase + (int)(t * 16 + head));
        const uint w = TW[row * WORDS + i / 8];
        const float q = float((w >> (4 * (i % 8))) & 15u);
        const float v = q * float(TS[row * NGS + i / GSZ]) + float(TB[row * NGS + i / GSZ]);
        emb[(ulong)t * D + head * DIM + i] = half(float(bfloat(v)));
        return;
    }
    const long t0 = hist[p], t1 = hist[p - 1], t2 = hist[p - 2];
    const long s1 = (t1 != (long)eos) ? t1 : (long)eos;
    const long s2 = (t1 != (long)eos && t2 != (long)eos) ? t2 : (long)eos;
    ulong mixed = (ulong)t0 * (ulong)mult[0];
    mixed ^= (ulong)s1 * (ulong)mult[1];
    if (head >= 8) mixed ^= (ulong)s2 * (ulong)mult[2];
    const long sz = size[head];
    long r = (long)mixed % sz;
    if (r < 0) r += sz;
    const ulong row = (ulong)(r + off[head]);
    const uint w = TW[row * WORDS + i / 8];
    const float q = float((w >> (4 * (i % 8))) & 15u);
    const float v = q * float(TS[row * NGS + i / GSZ]) + float(TB[row * NGS + i / GSZ]);
    emb[(ulong)t * D + head * DIM + i] = half(float(bfloat(v)));
}

// FP16 → FP32（n 個）
kernel void p_h2f(device const half *x [[buffer(0)]], device float *y [[buffer(1)]], constant uint &n [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = float(x[i]);
}

// 分流 RMSNorm（FP32 進、FP32 出）。grid：(HC, T) × 1024
kernel void p_hcnorm_f(device const float *x [[buffer(0)]], device const bfloat *w [[buffer(1)]], device float *y [[buffer(2)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint2 ntg2 [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    const uint ntg = ntg2.x;
    const uint s = tg.x;
    const ulong base = (ulong)tg.y * HCD + s * D;
    float ss = 0;
    for (uint i = tid; i < D; i += ntg) { const float v = x[base + i]; ss += v * v; }
    const float r = rsqrt(tg_sum(ss, tmp, tid, ntg) / D + EPS);
    for (uint i = tid; i < D; i += ntg) y[base + i] = x[base + i] * r * float(w[s * D + i]);
}

// PLE gate：gated[t, s, i] = sigmoid(g(t, s)) · value[t, i]。grid：(HC, T) × 1024
kernel void p_ple_gate(device const float *key [[buffer(0)]], device const float *qry [[buffer(1)]], device const half *val [[buffer(2)]],
                       device float *gated [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint2 ntg2 [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    const uint ntg = ntg2.x;
    const uint s = tg.x, t = tg.y;
    const ulong base = (ulong)t * HCD + s * D;
    float acc = 0;
    for (uint i = tid; i < D; i += ntg) acc += key[base + i] * qry[base + i];
    float g = tg_sum(acc, tmp, tid, ntg) / sqrt(float(D));
    g = sqrt(max(fabs(g), 1e-6f)) * sign(g);
    const float sgv = sigm(g);
    for (uint i = tid; i < D; i += ntg) gated[base + i] = sgv * float(val[(ulong)t * D + i]);
}

// PLE conv（kernel 4、dilation 3）＋ 殘差：h[t] += gated[t] + silu(Σ_k w[c,k] · cn[t − (3 − k)·3])
// 早於本段的輸入從狀態 st（[9][HCD]，舊的在前）取。grid：(HCD / 256, T)
kernel void p_ple_conv(device float *h [[buffer(0)]], device const float *gated [[buffer(1)]], device const float *cn [[buffer(2)]],
                       device const bfloat *w [[buffer(3)]], device const float *st [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, t = gid.y;
    if (c >= HCD) return;
    auto inp = [&](int tt) -> float { return tt >= 0 ? cn[(ulong)tt * HCD + c] : st[(9 + tt) * HCD + c]; };
    const int tt = (int)t;
    const float o = float(w[c * 4]) * inp(tt - 9) + float(w[c * 4 + 1]) * inp(tt - 6) + float(w[c * 4 + 2]) * inp(tt - 3) +
                    float(w[c * 4 + 3]) * inp(tt);
    h[(ulong)t * HCD + c] += gated[(ulong)t * HCD + c] + silu(o);
}

// PLE conv 狀態：保留最後 9 個 token 的 cn。grid：HCD
kernel void p_ple_state(device const float *cn [[buffer(0)]], device float *st [[buffer(1)]], constant uint &T [[buffer(2)]],
                        uint c [[thread_position_in_grid]]) {
    if (c >= HCD) return;
    float old[9];
    for (uint j = 0; j < 9; j++) old[j] = st[j * HCD + c];
    for (int j = 0; j < 9; j++) {
        const int idx = (int)T - 9 + j;
        st[j * HCD + c] = idx >= 0 ? cn[(ulong)idx * HCD + c] : old[9 + idx];
    }
}

// 清零（GPU 上做：多層的指令是一次編碼完才執行，CPU 在編碼時清零無效）
kernel void p_zero(device uint *x [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = 0;
}

// router logits（精確的 FP32 計算，跟 decode 的 qmv 一致；tensor 單元版的 FP16 權重會讓接近平手的 expert 排名對調）
// W：8-bit [E, D]（MLX affine）。每個 SIMD group 算一個 expert、8 個 token。grid：(E / 8, ceil(T / 8)) × 256
kernel void p_router(device const uint *W [[buffer(0)]], device const bfloat *S [[buffer(1)]], device const bfloat *Z [[buffer(2)]],
                     device const half *x [[buffer(3)]], device float *y [[buffer(4)]], constant uint &T [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint E = 512, NCH = D / 16, ROWW = D / 4, NG = D / 64, TT = 8;
    const uint r = tg.x * 8 + sg, t0 = tg.y * TT;
    float acc[TT];
    for (uint j = 0; j < TT; j++) acc[j] = 0;
    for (uint c = lane; c < NCH; c += 32) {
        const uint2 wl = ((device const uint2 *)(W + (ulong)r * ROWW))[2 * c], wh = ((device const uint2 *)(W + (ulong)r * ROWW))[2 * c + 1];
        float wq[16];
        for (int n = 0; n < 4; n++) {
            wq[n] = float((wl.x >> (8 * n)) & 255u);
            wq[4 + n] = float((wl.y >> (8 * n)) & 255u);
            wq[8 + n] = float((wh.x >> (8 * n)) & 255u);
            wq[12 + n] = float((wh.y >> (8 * n)) & 255u);
        }
        const uint g = c / 4;
        const float sc = float(S[(ulong)r * NG + g]), zz = float(Z[(ulong)r * NG + g]);
        for (uint j = 0; j < TT; j++) {
            const uint t = t0 + j;
            if (t >= T) break;
            device const half4 *xp = (device const half4 *)(x + (ulong)t * D + c * 16);
            float d = 0, xs = 0;
            for (int q = 0; q < 4; q++) {
                const float4 h = float4(xp[q]);
                d += wq[4 * q] * h.x + wq[4 * q + 1] * h.y + wq[4 * q + 2] * h.z + wq[4 * q + 3] * h.w;
                xs += h.x + h.y + h.z + h.w;
            }
            acc[j] += sc * d + zz * xs;
        }
    }
    for (uint j = 0; j < TT; j++) {
        const float v = simd_sum(acc[j]);
        if (lane == 0 && t0 + j < T) y[(ulong)(t0 + j) * E + r] = v;
    }
}
