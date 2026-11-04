// Engine 的各種小運算（decode：一次一個 token）。
// 殘差流 h 用 FP32（[HC × D]）；送進 matvec 的向量用 FP16。
// 主程式注入：D（hidden 2560）、HC（4）

#include <metal_stdlib>
using namespace metal;

#define EPS 1e-6f
#define HCD (HC * D)

inline float silu(float x) { return x / (1.0f + exp(-x)); }
inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// threadgroup 內加總（最多 1024 個執行緒）
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

// ---------------------------------------------------------------- embed（4-bit、group 64）→ h（複製成 HC 條流）
kernel void k_embed(device const uint *W [[buffer(0)]], device const bfloat *S [[buffer(1)]],
                    device const bfloat *Z [[buffer(2)]], constant uint &id [[buffer(3)]],
                    device float *h [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= D) return;
    const ulong row = id;
    const uint word = W[row * (D / 8) + i / 8];
    const float q = float((word >> (4 * (i % 8))) & 15u);
    const float v = float(bfloat(q * float(S[row * (D / 64) + i / 64]) + float(Z[row * (D / 64) + i / 64])));
    for (uint s = 0; s < HC; s++) h[s * D + i] = v;
}

// ---------------------------------------------------------------- 分流 RMSNorm：HC 條流各自正規化
// grid：HC 個 threadgroup × 1024 執行緒。輸出 FP16（給 matvec）與 FP32（給後續逐元素運算）
kernel void k_hcnorm(device const float *h [[buffer(0)]], device const bfloat *w [[buffer(1)]],
                     device half *yh [[buffer(2)]], device float *yf [[buffer(3)]],
                     uint s [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    float ss = 0;
    for (uint i = tid; i < D; i += ntg) { const float x = h[s * D + i]; ss += x * x; }
    const float r = rsqrt(tg_sum(ss, tmp, tid, ntg) / D + EPS);
    for (uint i = tid; i < D; i += ntg) {
        const float y = h[s * D + i] * r * float(w[s * D + i]);
        yh[s * D + i] = half(y);
        yf[s * D + i] = y;
    }
}

// y = silu(x / HC)（hyper-connection 低秩那段）
kernel void k_silu_div(device const half *x [[buffer(0)]], device half *y [[buffer(1)]], constant uint &n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = half(silu(float(x[i]) / HC));
}

// x[i] = mean_s sigmoid(u[s,i]) · normed[s,i]
kernel void k_hcmix(device const half *u [[buffer(0)]], device const float *nf [[buffer(1)]], device half *x [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= D) return;
    float acc = 0;
    for (uint s = 0; s < HC; s++) acc += sigm(float(u[s * D + i])) * nf[s * D + i];
    x[i] = half(acc / HC);
}

// inject[s] = 2 · sigmoid(Σ_j W[s, j] · normed[j] / HC)；grid：HC 個 threadgroup × 1024
kernel void k_inject(device const bfloat *W [[buffer(0)]], device const float *nf [[buffer(1)]], device float *inj [[buffer(2)]],
                     uint s [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    float acc = 0;
    for (uint j = tid; j < HCD; j += ntg) acc += float(W[s * HCD + j]) * nf[j];
    acc = tg_sum(acc, tmp, tid, ntg);
    if (tid == 0) inj[s] = 2.0f * sigm(acc / HC);
}

// h[s, i] += out[i] · inject[s]
kernel void k_resid(device float *h [[buffer(0)]], device const half *out [[buffer(1)]], device const float *injp [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= HCD) return;
    float t = 0;
    for (uint s = 0; s < HC; s++) t += injp[s * HC + i / D];
    h[i] += float(out[i % D]) * (2.0f / (1.0f + exp(-t / HC)));
}

// ---------------------------------------------------------------- Gated DeltaNet

// depthwise causal conv（kernel 4）+ SiLU，並更新 conv 狀態（最近 3 個 token，舊的在前）
// w：[C, 4]（BF16）；st：[3, C]（FP32）
kernel void k_conv4(device const half *x [[buffer(0)]], device const bfloat *w [[buffer(1)]], device float *st [[buffer(2)]],
                    device half *y [[buffer(3)]], constant uint &C [[buffer(4)]], uint c [[thread_position_in_grid]]) {
    if (c >= C) return;
    const float x0 = st[c], x1 = st[C + c], x2 = st[2 * C + c], x3 = float(x[c]);
    const float o = float(w[c * 4]) * x0 + float(w[c * 4 + 1]) * x1 + float(w[c * 4 + 2]) * x2 + float(w[c * 4 + 3]) * x3;
    y[c] = half(silu(o));
    st[c] = x1;
    st[C + c] = x2;
    st[2 * C + c] = x3;
}

// q、k 每個 key head 做無權重 RMSNorm：q ·= 1/DK、k ·= 1/√DK；另算每個 value head 的 log 衰減與 β
// grid：KH 個 threadgroup × 128（每個執行緒一維）；第 0 個 threadgroup 另外算 a、β
kernel void k_gdn_qk(device const half *conv [[buffer(0)]], device const half *ain [[buffer(1)]],
                     device const half *bin [[buffer(2)]], device const bfloat *A_log [[buffer(3)]],
                     device const bfloat *dt_bias [[buffer(4)]], device half *q [[buffer(5)]], device half *k [[buffer(6)]],
                     device float *la [[buffer(7)]], device float *be [[buffer(8)]],
                     uint hh [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint DK = 128, KH = 16, VH = 48;
    threadgroup float tmp[32];
    const float qv = float(conv[hh * DK + tid]), kv = float(conv[KH * DK + hh * DK + tid]);
    const float rq = rsqrt(tg_sum(qv * qv, tmp, tid, DK) / DK + EPS);
    const float rk = rsqrt(tg_sum(kv * kv, tmp, tid, DK) / DK + EPS);
    q[hh * DK + tid] = half(qv * rq / DK);
    k[hh * DK + tid] = half(kv * rk * rsqrt(float(DK)));
    if (hh == 0 && tid < VH) {
        const float x = float(ain[tid]) + float(dt_bias[tid]);
        const float sp = x > 20.0f ? x : log(1.0f + exp(x)); // softplus
        la[tid] = -exp(float(A_log[tid])) * sp;
        be[tid] = sigm(float(bin[tid]));
    }
}

// RMSNormGated：每個 value head 正規化 × weight × sigmoid(z)；grid：VH × 128
kernel void k_normgate(device const half *o [[buffer(0)]], device const half *z [[buffer(1)]], device const bfloat *w [[buffer(2)]],
                       device half *y [[buffer(3)]], uint hh [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint DV = 128;
    threadgroup float tmp[32];
    const float x = float(o[hh * DV + tid]);
    const float r = rsqrt(tg_sum(x * x, tmp, tid, DV) / DV + EPS);
    y[hh * DV + tid] = half(x * r * float(w[tid]) * sigm(float(z[hh * DV + tid])));
}

// ---------------------------------------------------------------- QSA attention

// RoPE（rotate_half、前 RD 維、theta）：對單一向量就地處理，x 是該向量的 FP32 暫存
inline void rope_pair(thread float &a, thread float &b, uint d, uint pos, float theta, uint rd) {
    const float f = pow(theta, -float(2 * d) / float(rd)) * float(pos);
    const float c = cos(f), s = sin(f);
    const float x1 = a, x2 = b;
    a = x1 * c - x2 * s;
    b = x2 * c + x1 * s;
}

// attention 的 q/k/v 準備：q 與 k 做 per-head RMSNorm（有權重）＋ RoPE（前 64 維），k、v 寫進 cache[pos]
// q_proj 輸出每個 head 是 [q(256) | gate(256)]。norm 權重型別 NT：MLX 是 bfloat，GGUF profile 是 float（_f 版）
template <typename NT>
inline void attn_prep_body(device const half *qg, device const half *kin, device const half *vin, device const NT *qn,
                           device const NT *kn, device half *q, device half *Kc, device half *Vc, uint hh, uint pos, uint tid,
                           threadgroup float *tmp, threadgroup float *v) {
    constexpr uint HD = 256, QH = 24, KVH = 2, RD = 64;
    constexpr float THETA = 10000000.0f;
    const bool isq = hh < QH;
    const uint kh = hh - QH;
    const float x = isq ? float(qg[hh * 2 * HD + tid]) : float(kin[kh * HD + tid]);
    const float r = rsqrt(tg_sum(x * x, tmp, tid, HD) / HD + EPS);
    v[tid] = x * r * float(isq ? qn[tid] : kn[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < RD / 2) {
        float a = v[tid], b = v[tid + RD / 2];
        rope_pair(a, b, tid, pos, THETA, RD);
        v[tid] = a;
        v[tid + RD / 2] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (isq) q[hh * HD + tid] = half(v[tid]);
    else {
        Kc[((ulong)pos * KVH + kh) * HD + tid] = half(v[tid]);
        Vc[((ulong)pos * KVH + kh) * HD + tid] = vin[kh * HD + tid];
    }
}

// 單 token。grid：(QH + KVH) 個 threadgroup × 256
#define ATTN_PREP_K(NAME, NT)                                                                                                         \
    kernel void NAME(device const half *qg [[buffer(0)]], device const half *kin [[buffer(1)]], device const half *vin [[buffer(2)]], \
                     device const NT *qn [[buffer(3)]], device const NT *kn [[buffer(4)]], device half *q [[buffer(5)]],             \
                     device half *Kc [[buffer(6)]], device half *Vc [[buffer(7)]], constant uint &pos [[buffer(8)]],                 \
                     uint hh [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {                          \
        threadgroup float tmp[32], v[256];                                                                                            \
        attn_prep_body<NT>(qg, kin, vin, qn, kn, q, Kc, Vc, hh, pos, tid, tmp, v);                                                    \
    }
ATTN_PREP_K(k_attn_prep, bfloat)
ATTN_PREP_K(k_attn_prep_f, float)

// 多個 token 的 attention 準備（推測解碼驗證）：grid (QH + KVH, 幾個 token) × 256；token t 在位置 p0 + t
#define ATTN_PREP_KB(NAME, NT)                                                                                                        \
    kernel void NAME(device const half *qg [[buffer(0)]], device const half *kin [[buffer(1)]], device const half *vin [[buffer(2)]], \
                     device const NT *qn [[buffer(3)]], device const NT *kn [[buffer(4)]], device half *q [[buffer(5)]],             \
                     device half *Kc [[buffer(6)]], device half *Vc [[buffer(7)]], constant uint &p0 [[buffer(8)]],                  \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {                         \
        threadgroup float tmp[32], v[256];                                                                                            \
        const uint t = tg.y;                                                                                                          \
        attn_prep_body<NT>(qg + (ulong)t * 2 * 24 * 256, kin + (ulong)t * 2 * 256, vin + (ulong)t * 2 * 256, qn, kn,                  \
                           q + (ulong)t * 24 * 256, Kc, Vc, tg.x, p0 + t, tid, tmp, v);                                              \
    }
ATTN_PREP_KB(kb_attn_prep, bfloat)
ATTN_PREP_KB(kb_attn_prep_f, float)

// attention 輸出乘上 gate，多個 token：grid 涵蓋 nt × QH·HD
kernel void kb_attn_gate(device const half *o [[buffer(0)]], device const half *qg [[buffer(1)]], device half *y [[buffer(2)]],
                         constant uint &nt [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    constexpr uint HD = 256, QH = 24;
    if (i >= nt * QH * HD) return;
    const uint t = i / (QH * HD), j = i % (QH * HD), h = j / HD, d = j % HD;
    y[i] = half(float(o[i]) * sigm(float(qg[(ulong)t * 2 * QH * HD + h * 2 * HD + HD + d])));
}

// indexer 準備：4 個 q head（RMSNorm 有權重 + RoPE）；raw_k 存進 cache；滿 4 個 token 時產生區塊 key
// iqk：[ (4 + 1) × 128 ]；grid：1 個 threadgroup × 128
template <typename NT>
inline void idx_body(device const half *iqk, device const NT *qn, device const NT *kn, device half *iq, device float *rawk,
                     device half *kblk, uint pos, uint tid, threadgroup float *tmp, threadgroup float *v) {
    constexpr uint ID = 128, IH = 4, RD = 64, CR = 4;
    constexpr float THETA = 10000000.0f;
    for (uint h = 0; h < IH; h++) {
        const float x = float(iqk[h * ID + tid]);
        const float r = rsqrt(tg_sum(x * x, tmp, tid, ID) / ID + EPS);
        v[tid] = x * r * float(qn[tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < RD / 2) {
            float a = v[tid], b = v[tid + RD / 2];
            rope_pair(a, b, tid, pos, THETA, RD);
            v[tid] = a;
            v[tid + RD / 2] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        iq[h * ID + tid] = half(v[tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // raw_k（參考實作存成 BF16）
    rawk[(ulong)pos * ID + tid] = float(bfloat(float(iqk[IH * ID + tid])));
    if ((pos + 1) % CR == 0) {
        threadgroup_barrier(mem_flags::mem_device);
        const uint nb = pos / CR;
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
}

#define IDX_PREP_K(NAME, NT)                                                                                                         \
    kernel void NAME(device const half *iqk [[buffer(0)]], device const NT *qn [[buffer(1)]], device const NT *kn [[buffer(2)]],      \
                     device half *iq [[buffer(3)]], device float *rawk [[buffer(4)]], device half *kblk [[buffer(5)]],               \
                     constant uint &pos [[buffer(6)]], uint tid [[thread_index_in_threadgroup]]) {                                   \
        threadgroup float tmp[32], v[128];                                                                                            \
        idx_body<NT>(iqk, qn, kn, iq, rawk, kblk, pos, tid, tmp, v);                                                                  \
    }
IDX_PREP_K(k_idx_prep, bfloat)
IDX_PREP_K(k_idx_prep_f, float)

// prefill 的 indexer q：每個 token 4 個 head，RMSNorm（有權重）+ RoPE。iqk [T][640] → iq [T][512]。grid：T × 128
template <typename NT>
inline void idx_q_body(device const half *iqk, device const NT *qn, device half *iq, uint pos0, uint t, uint tid,
                       threadgroup float *tmp, threadgroup float *v) {
    constexpr uint ID = 128, IH = 4, RD = 64;
    constexpr float THETA = 10000000.0f;
    for (uint h = 0; h < IH; h++) {
        const float x = float(iqk[(ulong)t * 640 + h * ID + tid]);
        const float r = rsqrt(tg_sum(x * x, tmp, tid, ID) / ID + EPS);
        v[tid] = x * r * float(qn[tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < RD / 2) {
            float a = v[tid], b = v[tid + RD / 2];
            rope_pair(a, b, tid, pos0 + t, THETA, RD);
            v[tid] = a;
            v[tid + RD / 2] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        iq[(ulong)t * 512 + h * ID + tid] = half(v[tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void p_idx_q(device const half *iqk [[buffer(0)]], device const bfloat *qn [[buffer(1)]], device half *iq [[buffer(2)]],
                    constant uint &pos0 [[buffer(3)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tmp[32], v[128];
    idx_q_body<bfloat>(iqk, qn, iq, pos0, t, tid, tmp, v);
}
kernel void p_idx_q_f(device const half *iqk [[buffer(0)]], device const float *qn [[buffer(1)]], device half *iq [[buffer(2)]],
                      constant uint &pos0 [[buffer(3)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tmp[32], v[128];
    idx_q_body<float>(iqk, qn, iq, pos0, t, tid, tmp, v);
}

// 多個 token：一個 threadgroup 依序處理（區塊 key 會用到同一批前面 token 的 raw_k）。iqk [nt][640]、iq [nt][512]
#define IDX_PREP_KB(NAME, NT)                                                                                                        \
    kernel void NAME(device const half *iqk [[buffer(0)]], device const NT *qn [[buffer(1)]], device const NT *kn [[buffer(2)]],      \
                     device half *iq [[buffer(3)]], device float *rawk [[buffer(4)]], device half *kblk [[buffer(5)]],               \
                     constant uint &p0 [[buffer(6)]], constant uint &nt [[buffer(7)]], uint tid [[thread_index_in_threadgroup]]) {   \
        threadgroup float tmp[32], v[128];                                                                                            \
        for (uint t = 0; t < nt; t++) {                                                                                               \
            idx_body<NT>(iqk + t * 640, qn, kn, iq + t * 512, rawk, kblk, p0 + t, tid, tmp, v);                                       \
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);                                                  \
        }                                                                                                                             \
    }
IDX_PREP_KB(kb_idx_prep, bfloat)
IDX_PREP_KB(kb_idx_prep_f, float)

// out[h, d] ·= sigmoid(gate[h, d])，gate 在 q_proj 輸出每個 head 的後半
kernel void k_attn_gate(device const half *o [[buffer(0)]], device const half *qg [[buffer(1)]], device half *y [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    constexpr uint HD = 256, QH = 24;
    if (i >= QH * HD) return;
    const uint h = i / HD, d = i % HD;
    y[i] = half(float(o[i]) * sigm(float(qg[h * 2 * HD + HD + d])));
}

// ---------------------------------------------------------------- PLE（只有第 2 層）

// gate[s] = Σ_i key[s,i]·query[s,i] / √D → sqrt(|g|)·sign(g) → sigmoid；gated[s,i] = sig[s] · value[i]
// grid：HC 個 threadgroup × 1024
kernel void k_ple_gate(device const float *key [[buffer(0)]], device const float *qry [[buffer(1)]],
                       device const half *val [[buffer(2)]], device float *gated [[buffer(3)]],
                       uint s [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    float acc = 0;
    for (uint i = tid; i < D; i += ntg) acc += key[s * D + i] * qry[s * D + i];
    float gte = tg_sum(acc, tmp, tid, ntg) / sqrt(float(D));
    gte = sqrt(max(fabs(gte), 1e-6f)) * sign(gte);
    const float sg = sigm(gte);
    for (uint i = tid; i < D; i += ntg) gated[s * D + i] = sg * float(val[i]);
}

// h += gated + silu(conv(norm_conv(gated)))：depthwise conv kernel 4、dilation 3（狀態保存最近 9 個 token）
// cn：norm_conv(gated)（FP32，[HCD]）；st：[9, HCD]（舊的在前）
kernel void k_ple_conv(device float *h [[buffer(0)]], device const float *gated [[buffer(1)]], device const float *cn [[buffer(2)]],
                       device const bfloat *w [[buffer(3)]], device float *st [[buffer(4)]], uint c [[thread_position_in_grid]]) {
    if (c >= HCD) return;
    const float x = cn[c];
    // full = [st[0..8], x]，取 full[0]、full[3]、full[6]、full[9]
    const float o = float(w[c * 4]) * st[c] + float(w[c * 4 + 1]) * st[3 * HCD + c] + float(w[c * 4 + 2]) * st[6 * HCD + c] +
                    float(w[c * 4 + 3]) * x;
    h[c] += gated[c] + silu(o);
    for (uint j = 0; j < 8; j++) st[j * HCD + c] = st[(j + 1) * HCD + c];
    st[8 * HCD + c] = x;
}

// FP32 分流 RMSNorm（PLE 的 key / query / conv 用），輸出 FP32；grid：HC × 1024
kernel void k_hcnorm_f(device const float *x [[buffer(0)]], device const bfloat *w [[buffer(1)]], device float *y [[buffer(2)]],
                       uint s [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    float ss = 0;
    for (uint i = tid; i < D; i += ntg) { const float v = x[s * D + i]; ss += v * v; }
    const float r = rsqrt(tg_sum(ss, tmp, tid, ntg) / D + EPS);
    for (uint i = tid; i < D; i += ntg) y[s * D + i] = x[s * D + i] * r * float(w[s * D + i]);
}

// half → float
kernel void k_h2f(device const half *x [[buffer(0)]], device float *y [[buffer(1)]], constant uint &n [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = float(x[i]);
}

// 複製（除錯用：把殘差流存下來）
kernel void k_copyf(device const float *x [[buffer(0)]], device float *y [[buffer(1)]], constant uint &n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = x[i];
}

// ---------------------------------------------------------------- 合併版 hyper-connection

// （可選的）殘差更新 + 分流 RMSNorm：若 apply，先 h[s,i] += out[i]·inj[s]，再正規化
// grid：HC 個 threadgroup × 1024
// inject 係數以「部分和」保存：injp[s][r] = Σ_{j ∈ 第 s 條流} W[r, j] · normed[j]；
// 用的時候 inj[r] = 2 · sigmoid(Σ_s injp[s][r] / HC)（見 inj_of）
inline float inj_of(device const float *injp, uint r) {
    float t = 0;
    for (uint s = 0; s < HC; s++) t += injp[s * HC + r];
    return 2.0f / (1.0f + exp(-t / HC));
}

kernel void k_hc_in(device float *h [[buffer(0)]], device const half *out [[buffer(1)]], device const float *inj [[buffer(2)]],
                    device const bfloat *w [[buffer(3)]], device half *yh [[buffer(4)]], device float *yf [[buffer(5)]],
                    constant uint &apply [[buffer(6)]], device const bfloat *injW [[buffer(7)]], device float *injp_out [[buffer(8)]],
                    constant uint &want_inj [[buffer(9)]], uint s [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    const float js = apply ? inj_of(inj, s) : 0.0f;
    float ss = 0;
    for (uint i = tid; i < D; i += ntg) {
        float x = h[s * D + i];
        if (apply) {
            x += float(out[i]) * js;
            h[s * D + i] = x;
        }
        ss += x * x;
    }
    const float r = rsqrt(tg_sum(ss, tmp, tid, ntg) / D + EPS);
    float pj[HC];
    for (uint q = 0; q < HC; q++) pj[q] = 0;
    for (uint i = tid; i < D; i += ntg) {
        const float y = h[s * D + i] * r * float(w[s * D + i]);
        yh[s * D + i] = half(y);
        yf[s * D + i] = y;
        if (want_inj)
            for (uint q = 0; q < HC; q++) pj[q] += float(injW[(ulong)q * HCD + s * D + i]) * y;
    }
    if (want_inj)
        for (uint q = 0; q < HC; q++) {
            const float t = tg_sum(pj[q], tmp, tid, ntg);
            if (tid == 0) injp_out[s * HC + q] = t;
        }
}

// SiLU(d / HC) → mix_up（8-bit，K = LR）→ x[i] = mean_s sigmoid(u[s·D + i]) · nf[s·D + i]
// 每個 SIMD group 負責一個 i（4 條流各一列，共 4 列）。grid：D / 8 個 threadgroup × 256
#define LR 320
kernel void k_mixup(device const uint *W [[buffer(0)]], device const bfloat *S [[buffer(1)]], device const bfloat *Z [[buffer(2)]],
                    device const half *dn [[buffer(3)]], device const float *nf [[buffer(4)]], device half *x [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NCH = LR / 16, ROWW = LR / 4, NG = LR / 64; // 8-bit：每列 LR/4 個 uint32
    threadgroup float sx[LR], ssum[NCH];
    for (uint j = tid; j < LR; j += 256) sx[j] = silu(float(dn[j]) / HC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NCH) {
        float t = 0;
        for (uint j = 0; j < 16; j++) t += sx[tid * 16 + j];
        ssum[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = tg * 8 + sg;
    if (i >= D) return;
    float acc[HC];
    for (uint s = 0; s < HC; s++) acc[s] = 0;
    for (uint t = lane; t < HC * NCH; t += 32) {
        const uint s = t / NCH, c = t % NCH;
        const ulong row = (ulong)s * D + i;
        threadgroup const float *xv = sx + c * 16;
        const float xs = ssum[c];
        const uint2 wl = ((device const uint2 *)(W + row * ROWW))[2 * c], wh = ((device const uint2 *)(W + row * ROWW))[2 * c + 1];
        float d = 0;
        for (int n = 0; n < 4; n++) d += float((wl.x >> (8 * n)) & 255u) * xv[n];
        for (int n = 0; n < 4; n++) d += float((wl.y >> (8 * n)) & 255u) * xv[4 + n];
        for (int n = 0; n < 4; n++) d += float((wh.x >> (8 * n)) & 255u) * xv[8 + n];
        for (int n = 0; n < 4; n++) d += float((wh.y >> (8 * n)) & 255u) * xv[12 + n];
        const uint g = c / 4;
        acc[s] += float(S[row * NG + g]) * d + float(Z[row * NG + g]) * xs;
    }
    float m = 0;
    for (uint s = 0; s < HC; s++) m += sigm(simd_sum(acc[s])) * nf[s * D + i];
    if (lane == 0) x[i] = half(m / HC);
}

// log 衰減的係數 −exp(A_log)：MLX 存 A_log（BF16）；GGUF 直接存 A_neg = −exp(A_log)（F32）
inline float a_neg(device const bfloat *A_log, uint i) { return -exp(float(A_log[i])); }
inline float a_neg(device const float *A_neg, uint i) { return A_neg[i]; }

// DeltaNet：conv4（+ SiLU）與 q/k 正規化合併。grid：(KH + VH) 個 threadgroup × 128
//   tg < KH：key head tg 的 q 與 k 通道（各 128）：conv → RMSNorm → q/DK、k/√DK
//   tg ≥ KH：value head 的 128 個通道：conv → 寫 v
//   tg == 0 另外算 log 衰減與 β
//   k_gdn_in_t：A 是 GGUF 的 A_neg（F32）；value head 的順序不影響這個 kernel（每個 head 各自獨立）
template <typename AT>
inline void gdn_in_body(device const half *x, device const bfloat *w, device const float *st, device const half *ain,
                        device const half *bin, device const AT *A, device const bfloat *dt_bias, device half *q, device half *k,
                        device half *v, device float *la, device float *be, device float *sto, uint hh, uint tid, threadgroup float *tmp) {
    constexpr uint DK = 128, KH = 16, VH = 48, C = 10240;
    auto conv = [&](uint c) -> float {
        const float x0 = st[c], x1 = st[C + c], x2 = st[2 * C + c], x3 = float(x[c]);
        const float o = float(w[c * 4]) * x0 + float(w[c * 4 + 1]) * x1 + float(w[c * 4 + 2]) * x2 + float(w[c * 4 + 3]) * x3;
        sto[c] = x1;
        sto[C + c] = x2;
        sto[2 * C + c] = x3;
        return float(half(silu(o))); // 參考實作在這裡是低精度
    };
    if (hh < KH) {
        const float qv = conv(hh * DK + tid), kv = conv(KH * DK + hh * DK + tid);
        const float rq = rsqrt(tg_sum(qv * qv, tmp, tid, DK) / DK + EPS);
        const float rk = rsqrt(tg_sum(kv * kv, tmp, tid, DK) / DK + EPS);
        q[hh * DK + tid] = half(qv * rq / DK);
        k[hh * DK + tid] = half(kv * rk * rsqrt(float(DK)));
        if (hh == 0 && tid < VH) {
            const float xa = float(ain[tid]) + float(dt_bias[tid]);
            const float sp = xa > 20.0f ? xa : log(1.0f + exp(xa));
            la[tid] = a_neg(A, tid) * sp;
            be[tid] = sigm(float(bin[tid]));
        }
    } else {
        const uint j = hh - KH;
        v[j * DK + tid] = half(conv(2 * KH * DK + j * DK + tid));
    }
}


#define GDN_IN_ARGS(AT)                                                                                                           \
    device const half *x [[buffer(0)]], device const bfloat *w [[buffer(1)]], device const float *st [[buffer(2)]],            \
        device const half *ain [[buffer(3)]], device const half *bin [[buffer(4)]], device const AT *A [[buffer(5)]],         \
        device const bfloat *dt_bias [[buffer(6)]], device half *q [[buffer(7)]], device half *k [[buffer(8)]],                \
        device half *v [[buffer(9)]], device float *la [[buffer(10)]], device float *be [[buffer(11)]],                       \
        device float *sto [[buffer(12)]], /* 新的 conv 狀態寫到這裡（可以跟 st 相同；MTP 驗證時寫到另一份以便回退）*/ \
        uint hh [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
kernel void k_gdn_in(GDN_IN_ARGS(bfloat)) {
    threadgroup float tmp[32];
    gdn_in_body(x, w, st, ain, bin, A, dt_bias, q, k, v, la, be, sto, hh, tid, tmp);
}
kernel void k_gdn_in_t(GDN_IN_ARGS(float)) {
    threadgroup float tmp[32];
    gdn_in_body(x, w, st, ain, bin, A, dt_bias, q, k, v, la, be, sto, hh, tid, tmp);
}

// ---------------------------------------------------------------- 流水線 decode：token 歷史放在 GPU

// embed：從 GPU 上的 token 歷史讀目前的 token（hist[pos + 2]）
kernel void k_embed_g(device const uint *W [[buffer(0)]], device const bfloat *S [[buffer(1)]],
                      device const bfloat *Z [[buffer(2)]], device const int *hist [[buffer(3)]], constant uint &p [[buffer(4)]],
                      device float *h [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= D) return;
    const ulong row = (ulong)hist[p];
    const uint word = W[row * (D / 8) + i / 8];
    const float q = float((word >> (4 * (i % 8))) & 15u);
    const float v = float(bfloat(q * float(S[row * (D / 64) + i / 64]) + float(Z[row * (D / 64) + i / 64])));
    for (uint s = 0; s < HC; s++) h[s * D + i] = v;
}

// n-gram 查表（參考實作 NGramEmbedding）。grid：16 個 head × 160 執行緒
//   shifted[s] = hist[p − s]，若 hist[p − s .. p − 1] 中有 EOS 則改用 EOS
//   2-gram：head 0–7、3-gram：head 8–15；mixed = XOR shifted[s]·mult[s]（64 位元溢位）
//   row = floor_mod(mixed, size[head]) + offset[head]；表格 4-bit、group 32，結果轉 BF16
kernel void k_ngram(device const int *hist [[buffer(0)]], constant uint &p [[buffer(1)]], device const long *mult [[buffer(2)]],
                    device const long *off [[buffer(3)]], device const long *size [[buffer(4)]], device const uint *TW [[buffer(5)]],
                    device const bfloat *TS [[buffer(6)]], device const bfloat *TB [[buffer(7)]], device half *emb [[buffer(8)]],
                    constant uint &eos [[buffer(9)]], constant int &gbase [[buffer(10)]],
                    uint head [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    constexpr uint DIM = 160, WORDS = 20, NGS = 5, GSZ = 32;
    if (gbase >= 0) { // CPU 已把這個位置的 16 列抄到 TW / TS / TB 的第 gbase.. 列（大表不交給 GPU）
        const ulong row = (ulong)(gbase + (int)head);
        const uint w = TW[row * WORDS + i / 8];
        const float q = float((w >> (4 * (i % 8))) & 15u);
        const float v = q * float(TS[row * NGS + i / GSZ]) + float(TB[row * NGS + i / GSZ]);
        emb[head * DIM + i] = half(float(bfloat(v)));
        return;
    }
    const long t0 = hist[p], t1 = hist[p - 1], t2 = hist[p - 2];
    const long s0 = t0;
    const long s1 = (t1 != (long)eos) ? t1 : (long)eos;
    const long s2 = (t1 != (long)eos && t2 != (long)eos) ? t2 : (long)eos;
    ulong mixed = (ulong)s0 * (ulong)mult[0];
    mixed ^= (ulong)s1 * (ulong)mult[1];
    if (head >= 8) mixed ^= (ulong)s2 * (ulong)mult[2];
    const long sz = size[head];
    long r = (long)mixed % sz;
    if (r < 0) r += sz;
    const ulong row = (ulong)(r + off[head]);
    const uint w = TW[row * WORDS + i / 8];
    const float q = float((w >> (4 * (i % 8))) & 15u);
    const float v = q * float(TS[row * NGS + i / GSZ]) + float(TB[row * NGS + i / GSZ]);
    emb[head * DIM + i] = half(float(bfloat(v)));
}

// argmax 第一階段：每個 threadgroup 處理 1024 個 logits，輸出最大值與編號（同分取最小編號）
kernel void k_argmax1(device const float *lg [[buffer(0)]], device float *pv [[buffer(1)]], device uint *pi [[buffer(2)]],
                      constant uint &n [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup uint si[32];
    const uint i = tg * 1024 + tid;
    float v = i < n ? lg[i] : -INFINITY;
    uint id = i < n ? i : 0xffffffffu;
    const float m = simd_max(v);
    id = simd_min(v == m ? id : 0xffffffffu);
    v = m;
    if (tid % 32 == 0) { sv[tid / 32] = v; si[tid / 32] = id; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v2 = sv[tid];
        uint i2 = si[tid];
        const float m2 = simd_max(v2);
        i2 = simd_min(v2 == m2 ? i2 : 0xffffffffu);
        if (tid == 0) { pv[tg] = m2; pi[tg] = i2; }
    }
}

// argmax 第二階段：一個 threadgroup，把結果寫進 token 歷史 hist[p]
kernel void k_argmax2(device const float *pv [[buffer(0)]], device const uint *pi [[buffer(1)]], constant uint &m [[buffer(2)]],
                      device int *hist [[buffer(3)]], constant uint &p [[buffer(4)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup uint si[32];
    float v = -INFINITY;
    uint id = 0xffffffffu;
    for (uint j = tid; j < m; j += 1024) {
        const float x = pv[j];
        if (x > v || (x == v && pi[j] < id)) { v = x; id = pi[j]; }
    }
    float mm = simd_max(v);
    id = simd_min(v == mm ? id : 0xffffffffu);
    if (tid % 32 == 0) { sv[tid / 32] = mm; si[tid / 32] = id; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v2 = sv[tid];
        uint i2 = si[tid];
        const float m2 = simd_max(v2);
        i2 = simd_min(v2 == m2 ? i2 : 0xffffffffu);
        if (tid == 0) hist[p] = (int)i2;
    }
}

// ---------------------------------------------------------------- MTP

// 單一向量的 RMSNorm（n 個元素一起算，FP32 進、FP16 出）。grid：1 × 1024
kernel void k_norm1(device const float *x [[buffer(0)]], device const bfloat *w [[buffer(1)]], device half *y [[buffer(2)]],
                    constant uint &n [[buffer(3)]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    float ss = 0;
    for (uint i = tid; i < n; i += ntg) { const float v = x[i]; ss += v * v; }
    const float r = rsqrt(tg_sum(ss, tmp, tid, ntg) / n + EPS);
    for (uint i = tid; i < n; i += ntg) y[i] = half(x[i] * r * float(w[i]));
}

// 逐列 RMSNorm（有權重），每列 n 個；輸入 FP32（列間距 xs），輸出 FP16（列間距 ys）。grid：列數 × 1024
kernel void k_norm_rows(device const float *x [[buffer(0)]], device const bfloat *w [[buffer(1)]], device half *y [[buffer(2)]],
                        constant uint &n [[buffer(3)]], constant uint &xs [[buffer(4)]], constant uint &ys [[buffer(5)]],
                        uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    device const float *xr = x + (ulong)r * xs;
    float ss = 0;
    for (uint i = tid; i < n; i += ntg) { const float v = xr[i]; ss += v * v; }
    const float rr = rsqrt(tg_sum(ss, tmp, tid, ntg) / n + EPS);
    for (uint i = tid; i < n; i += ntg) y[(ulong)r * ys + i] = half(xr[i] * rr * float(w[i]));
}

// MTP 殘差流（多個 token）：hm[t, s, i] = fh[t, s, i] + fe[t, i]。grid 涵蓋 nt × HCD
kernel void p_mtp_combine(device const half *fh [[buffer(0)]], device const half *fe [[buffer(1)]], device float *hm [[buffer(2)]],
                          constant uint &nt [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= nt * HCD) return;
    const uint t = i / HCD, j = i % HCD;
    hm[i] = float(fh[i]) + float(fe[(ulong)t * D + j % D]);
}

// MTP 的殘差流：hm[s, i] = fc_hidden(norm(h))[s, i] + fc_embedding(norm(emb))[i]
kernel void k_mtp_combine(device const half *fh [[buffer(0)]], device const half *fe [[buffer(1)]], device float *hm [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= HCD) return;
    hm[i] = float(fh[i]) + float(fe[i % D]);
}

// ---------------------------------------------------------------- 多 token（MTP 驗證）版本：NB 由主程式注入
#ifdef NB
// 殘差（可選）+ 分流 norm + inject 部分和，每個 token 一份。grid：(HC, NB) × 1024
kernel void kb_hc_in(device float *h [[buffer(0)]], device const half *out [[buffer(1)]], device const float *inj [[buffer(2)]],
                     device const bfloat *w [[buffer(3)]], device half *yh [[buffer(4)]], device float *yf [[buffer(5)]],
                     constant uint &apply [[buffer(6)]], device const bfloat *injW [[buffer(7)]], device float *injp_out [[buffer(8)]],
                     constant uint &want_inj [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint2 ntg2 [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    const uint ntg = ntg2.x, s = tg.x, t = tg.y;
    device float *ht = h + (ulong)t * HCD;
    device const half *ot = out + (ulong)t * D;
    const float js = apply ? inj_of(inj + t * HC * HC, s) : 0.0f;
    float ss = 0;
    for (uint i = tid; i < D; i += ntg) {
        float x = ht[s * D + i];
        if (apply) {
            x += float(ot[i]) * js;
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

// SiLU + mix_up + mix，NB 個 token 共用權重讀取。grid：D / 8 × 256
kernel void kb_mixup(device const uint *W [[buffer(0)]], device const bfloat *S [[buffer(1)]], device const bfloat *Z [[buffer(2)]],
                     device const half *dn [[buffer(3)]], device const float *nf [[buffer(4)]], device half *x [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NCH = LR / 16, ROWW = LR / 4, NG = LR / 64;
    threadgroup float sx[NB][LR], ssum[NB][NCH];
    for (uint j = tid; j < NB * LR; j += 256) sx[j / LR][j % LR] = silu(float(dn[j]) / HC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NB * NCH) {
        const uint b = tid / NCH, c = tid % NCH;
        float t = 0;
        for (uint j = 0; j < 16; j++) t += sx[b][c * 16 + j];
        ssum[b][c] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = tg * 8 + sg;
    if (i >= D) return;
    float acc[NB][HC];
    for (uint b = 0; b < NB; b++)
        for (uint s = 0; s < HC; s++) acc[b][s] = 0;
    for (uint t = lane; t < HC * NCH; t += 32) {
        const uint s = t / NCH, c = t % NCH;
        const ulong row = (ulong)s * D + i;
        const uint2 wl = ((device const uint2 *)(W + row * ROWW))[2 * c], wh = ((device const uint2 *)(W + row * ROWW))[2 * c + 1];
        float wq[16];
        for (int n = 0; n < 4; n++) {
            wq[n] = float((wl.x >> (8 * n)) & 255u);
            wq[4 + n] = float((wl.y >> (8 * n)) & 255u);
            wq[8 + n] = float((wh.x >> (8 * n)) & 255u);
            wq[12 + n] = float((wh.y >> (8 * n)) & 255u);
        }
        const float sc = float(S[row * NG + c / 4]), zz = float(Z[row * NG + c / 4]);
        for (uint b = 0; b < NB; b++) {
            float d = 0;
            for (uint j = 0; j < 16; j++) d += wq[j] * sx[b][c * 16 + j];
            acc[b][s] += sc * d + zz * ssum[b][c];
        }
    }
    for (uint b = 0; b < NB; b++) {
        float m = 0;
        for (uint s = 0; s < HC; s++) m += sigm(simd_sum(acc[b][s])) * nf[(ulong)b * HCD + s * D + i];
        if (lane == 0) x[(ulong)b * D + i] = half(m / HC);
    }
}

// embed：tok[p + t] → h[t]。grid：(D / 256, NB) × 256
kernel void kb_embed(device const uint *W [[buffer(0)]], device const bfloat *S [[buffer(1)]], device const bfloat *Z [[buffer(2)]],
                     device const int *tok [[buffer(3)]], constant uint &p [[buffer(4)]], device float *h [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, t = gid.y;
    if (i >= D) return;
    const ulong row = (ulong)tok[p + t];
    const uint word = W[row * (D / 8) + i / 8];
    const float q = float((word >> (4 * (i % 8))) & 15u);
    const float v = float(bfloat(q * float(S[row * (D / 64) + i / 64]) + float(Z[row * (D / 64) + i / 64])));
    for (uint s = 0; s < HC; s++) h[(ulong)t * HCD + s * D + i] = v;
}

// DeltaNet 輸入（conv + q/k 正規化 + 衰減），NB 個 token 依序；conv 視窗直接由 [舊狀態 3 格, x_0, x_1, …] 取。
// 第 1 個 token 後的 conv 狀態寫回 st（就地），第 t 個（t ≥ 1）之後的寫到 so[t − 1]。
// x [NB][10240]、ain/bin [NB][48]（間距 48）、q/k [NB][2048]、v [NB][6144]、la/be [NB][64]。grid：(16 + 48) × 128
template <typename AT>
inline void gdnb_in_body(device const half *x, device const bfloat *w, device float *st, device const half *ain, device const half *bin,
                         device const AT *A, device const bfloat *dt_bias, device half *q, device half *k, device half *v,
                         device float *la, device float *be, device float *so1, device float *so2, device float *so3,
                         device float *so4, device float *so5, device float *so6, device float *so7, uint hh, uint tid, threadgroup float *tmp) {
    constexpr uint DK = 128, KH = 16, VH = 48, C = 10240;
    // 每個 channel 的 conv 輸出（NB 個 token），並更新兩份狀態
    auto conv = [&](uint c, thread float *outv) {
        float seq[3 + NB];
        seq[0] = st[c]; seq[1] = st[C + c]; seq[2] = st[2 * C + c];
        for (uint t = 0; t < NB; t++) seq[3 + t] = float(x[(ulong)t * C + c]);
        const float w0 = float(w[c * 4]), w1 = float(w[c * 4 + 1]), w2 = float(w[c * 4 + 2]), w3 = float(w[c * 4 + 3]);
        for (uint t = 0; t < NB; t++) outv[t] = float(half(silu(w0 * seq[t] + w1 * seq[t + 1] + w2 * seq[t + 2] + w3 * seq[t + 3])));
        st[c] = seq[1]; st[C + c] = seq[2]; st[2 * C + c] = seq[3];
        for (uint t = 1; t < NB; t++) {
            device float *o = t == 1 ? so1 : t == 2 ? so2 : t == 3 ? so3 : t == 4 ? so4 : t == 5 ? so5 : t == 6 ? so6 : so7;
            o[c] = seq[t + 1]; o[C + c] = seq[t + 2]; o[2 * C + c] = seq[t + 3];
        }
    };
    if (hh < KH) {
        float qv[NB], kv[NB];
        conv(hh * DK + tid, qv);
        conv(KH * DK + hh * DK + tid, kv);
        for (uint t = 0; t < NB; t++) {
            const float rq = rsqrt(tg_sum(qv[t] * qv[t], tmp, tid, DK) / DK + EPS);
            const float rk = rsqrt(tg_sum(kv[t] * kv[t], tmp, tid, DK) / DK + EPS);
            q[t * 2048 + hh * DK + tid] = half(qv[t] * rq / DK);
            k[t * 2048 + hh * DK + tid] = half(kv[t] * rk * rsqrt(float(DK)));
        }
        if (hh == 0 && tid < VH)
            for (uint t = 0; t < NB; t++) {
                const float xa = float(ain[t * VH + tid]) + float(dt_bias[tid]);
                const float sp = xa > 20.0f ? xa : log(1.0f + exp(xa));
                la[t * 64 + tid] = a_neg(A, tid) * sp;
                be[t * 64 + tid] = sigm(float(bin[t * VH + tid]));
            }
    } else {
        const uint j = hh - KH;
        float vv[NB];
        conv(2 * KH * DK + j * DK + tid, vv);
        for (uint t = 0; t < NB; t++) v[t * 6144 + j * DK + tid] = half(vv[t]);
    }
}

#define GDNB_IN_ARGS(AT)                                                                                                       \
    device const half *x [[buffer(0)]], device const bfloat *w [[buffer(1)]], device float *st [[buffer(2)]],                  \
        device const half *ain [[buffer(3)]], device const half *bin [[buffer(4)]], device const AT *A [[buffer(5)]],         \
        device const bfloat *dt_bias [[buffer(6)]], device half *q [[buffer(7)]], device half *k [[buffer(8)]],                \
        device half *v [[buffer(9)]], device float *la [[buffer(10)]], device float *be [[buffer(11)]],                       \
        device float *so1 [[buffer(12)]], device float *so2 [[buffer(13)]], device float *so3 [[buffer(14)]],                 \
        device float *so4 [[buffer(15)]], device float *so5 [[buffer(16)]], device float *so6 [[buffer(17)]],                 \
        device float *so7 [[buffer(18)]], uint hh [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
kernel void kb_gdn_in(GDNB_IN_ARGS(bfloat)) {
    threadgroup float tmp[32];
    gdnb_in_body(x, w, st, ain, bin, A, dt_bias, q, k, v, la, be, so1, so2, so3, so4, so5, so6, so7, hh, tid, tmp);
}
// GGUF：A 是 A_neg（F32）
kernel void kb_gdn_in_t(GDNB_IN_ARGS(float)) {
    threadgroup float tmp[32];
    gdnb_in_body(x, w, st, ain, bin, A, dt_bias, q, k, v, la, be, so1, so2, so3, so4, so5, so6, so7, hh, tid, tmp);
}

// gated RMSNorm，NB 個 token。grid：(48, NB) × 128
kernel void kb_normgate(device const half *o [[buffer(0)]], device const half *z [[buffer(1)]], device const bfloat *w [[buffer(2)]],
                        device half *y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint DV = 128;
    threadgroup float tmp[32];
    const ulong i = (ulong)tg.y * 6144 + tg.x * DV + tid;
    const float xv = float(o[i]);
    const float r = rsqrt(tg_sum(xv * xv, tmp, tid, DV) / DV + EPS);
    y[i] = half(xv * r * float(w[tid]) * sigm(float(z[i])));
}
#endif
