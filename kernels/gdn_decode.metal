// Gated DeltaNet 的 decode 單步（一個 token）
// 遞迴式同 gdn.metal：S = g·S；mem = Sᵀk；δ = β(v − mem)；S += k δᵀ；o = Sᵀq
// 狀態排列 [HV, DK, DV]（與 prefill 的分塊版相同，prefill 完可以直接接 decode）。
//
// 每個 threadgroup 負責一個 head 的 DVB 個 dv 欄：
//   1. 把 S[:, dv0 : dv0+DVB] 合併讀取進 threadgroup memory（每列補一格避免 bank 衝突）
//   2. 每個 SIMD group 負責 DVB / 8 欄，lane 分攤 dk
//   3. 整塊寫回
// 主程式注入：DK、DV、HK、HV、DVB

#include <metal_stdlib>
using namespace metal;

#define NSG 8
#define PADW (DVB + 1)

// value head → key head：MLX / HF 是分組順序（h = k·(HV/HK) + r → h / (HV/HK)）；
// GGUF 是交錯順序（h = r·HK + k → h % HK），主程式另外定義 GDN_TILED 編一份改名的版本
#ifdef GDN_TILED
#define HK_OF(h) ((h) % HK)
#else
#define HK_OF(h) ((h) / (HV / HK))
#endif

kernel void gdn_step(device const half *q [[buffer(0)]],    // [HK, DK]
                     device const half *k [[buffer(1)]],    // [HK, DK]
                     device const half *v [[buffer(2)]],    // [HV, DV]
                     device const float *a [[buffer(3)]],   // [HV] log 衰減
                     device const float *beta [[buffer(4)]],// [HV]
                     device float *S [[buffer(5)]],         // [HV, DK, DV]
                     device half *o [[buffer(6)]],          // [HV, DV]
                     device float *ssp [[buffer(7)]],       // [HV, DV / DVB] 輸出的部分平方和（給合併版 gated RMSNorm）
                     device float *So [[buffer(8)]],        // 新狀態寫到這裡（可以跟 S 相同；MTP 驗證時寫到另一份以便回退）
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NT = NSG * 32, PER = DK / 32, CPS = DVB / NSG; // 每個 SIMD group 幾欄
    const uint hv = tg.x, dv0 = tg.y * DVB, hk = HK_OF(hv);
    threadgroup float St[DK * PADW];
    device float *Sh = S + (ulong)hv * DK * DV;
    device float *Sho = So + (ulong)hv * DK * DV;

    for (uint e = tid; e < DK * DVB; e += NT) {
        const uint r = e / DVB, c = e % DVB;
        St[r * PADW + c] = Sh[r * DV + dv0 + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float gdec = exp(a[hv]), b = beta[hv];
    float kk[PER], qq[PER];
    for (uint i = 0; i < PER; i++) {
        kk[i] = float(k[hk * DK + lane * PER + i]);
        qq[i] = float(q[hk * DK + lane * PER + i]);
    }
    threadgroup float sq[NSG];
    float mysq = 0;
    for (uint cc = 0; cc < CPS; cc++) {
        const uint c = sg * CPS + cc;
        float s[PER], mem = 0;
        for (uint i = 0; i < PER; i++) {
            s[i] = St[(lane * PER + i) * PADW + c] * gdec;
            mem += s[i] * kk[i];
        }
        mem = simd_sum(mem);
        const float d = b * (float(v[hv * DV + dv0 + c]) - mem);
        float out = 0;
        for (uint i = 0; i < PER; i++) {
            s[i] += kk[i] * d;
            out += s[i] * qq[i];
            St[(lane * PER + i) * PADW + c] = s[i];
        }
        out = simd_sum(out);
        if (lane == 0) o[hv * DV + dv0 + c] = half(out);
        const float oh = float(half(out)); // 與讀回 FP16 輸出一致
        mysq += oh * oh;
    }
    if (lane == 0) sq[sg] = mysq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0;
        for (uint i = 0; i < NSG; i++) t += sq[i];
        ssp[hv * (DV / DVB) + tg.y] = t;
    }
    for (uint e = tid; e < DK * DVB; e += NT) {
        const uint r = e / DVB, c = e % DVB;
        Sho[r * DV + dv0 + c] = St[r * PADW + c];
    }
}

#ifdef NB
// NB 個 token 連續 NB 步（推測解碼驗證）：狀態只讀一次。第 1 步後的狀態寫回 S（就地），第 t 步（t ≥ 1）後寫到 So[t − 1]
// q、k [NB][HK, DK]；v、o [NB][HV, DV]；a、beta [NB][64]；ssp [NB][HV, DV / DVB]
kernel void gdn_stepn(device const half *q [[buffer(0)]], device const half *k [[buffer(1)]], device const half *v [[buffer(2)]],
                      device const float *a [[buffer(3)]], device const float *beta [[buffer(4)]], device float *S [[buffer(5)]],
                      device half *o [[buffer(6)]], device float *ssp [[buffer(7)]],
                      device float *So1 [[buffer(8)]], device float *So2 [[buffer(9)]], device float *So3 [[buffer(10)]],
                      device float *So4 [[buffer(11)]], device float *So5 [[buffer(12)]], device float *So6 [[buffer(13)]],
                      device float *So7 [[buffer(14)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NT = NSG * 32, PER = DK / 32, CPS = DVB / NSG;
    const uint hv = tg.x, dv0 = tg.y * DVB, hk = HK_OF(hv);
    threadgroup float St[DK * PADW];
    threadgroup float sq[NSG];
    device float *Sh = S + (ulong)hv * DK * DV;
    for (uint e = tid; e < DK * DVB; e += NT) {
        const uint r = e / DVB, c = e % DVB;
        St[r * PADW + c] = Sh[r * DV + dv0 + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < NB; t++) {
        const float gdec = exp(a[t * 64 + hv]), b = beta[t * 64 + hv];
        float kk[PER], qq[PER];
        for (uint i = 0; i < PER; i++) {
            kk[i] = float(k[t * HK * DK + hk * DK + lane * PER + i]);
            qq[i] = float(q[t * HK * DK + hk * DK + lane * PER + i]);
        }
        float mysq = 0;
        for (uint cc = 0; cc < CPS; cc++) {
            const uint c = sg * CPS + cc;
            float s[PER], mem = 0;
            for (uint i = 0; i < PER; i++) {
                s[i] = St[(lane * PER + i) * PADW + c] * gdec;
                mem += s[i] * kk[i];
            }
            mem = simd_sum(mem);
            const float d = b * (float(v[t * HV * DV + hv * DV + dv0 + c]) - mem);
            float out = 0;
            for (uint i = 0; i < PER; i++) {
                s[i] += kk[i] * d;
                out += s[i] * qq[i];
                St[(lane * PER + i) * PADW + c] = s[i];
            }
            out = simd_sum(out);
            if (lane == 0) o[t * HV * DV + hv * DV + dv0 + c] = half(out);
            const float oh = float(half(out));
            mysq += oh * oh;
        }
        if (lane == 0) sq[sg] = mysq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float tt = 0;
            for (uint i = 0; i < NSG; i++) tt += sq[i];
            ssp[t * HV * (DV / DVB) + hv * (DV / DVB) + tg.y] = tt;
        }
        device float *Ob = t == 0 ? S : t == 1 ? So1 : t == 2 ? So2 : t == 3 ? So3 : t == 4 ? So4 : t == 5 ? So5 : t == 6 ? So6 : So7;
        device float *Out = Ob + (ulong)hv * DK * DV;
        for (uint e = tid; e < DK * DVB; e += NT) {
            const uint r = e / DVB, c = e % DVB;
            Out[r * DV + dv0 + c] = St[r * PADW + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#ifdef GDN_DEFER
// 驗證（延後寫回狀態）：跟 gdn_stepn 相同的遞迴，但不寫任何狀態（S 只讀），改把每步的 δ（[NB][HV][DV]）寫到 dlt；
// 接受了 J 個 token 之後再由 gdn_commit 從原狀態重放 J 步（只讀寫一次狀態，不用每個 token 各存一份）
kernel void gdn_stepn_d(device const half *q [[buffer(0)]], device const half *k [[buffer(1)]], device const half *v [[buffer(2)]],
                      device const float *a [[buffer(3)]], device const float *beta [[buffer(4)]], device const float *S [[buffer(5)]],
                      device half *o [[buffer(6)]], device float *ssp [[buffer(7)]],
                      device float *dlt [[buffer(8)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NT = NSG * 32, PER = DK / 32, CPS = DVB / NSG;
    const uint hv = tg.x, dv0 = tg.y * DVB, hk = HK_OF(hv);
    threadgroup float St[DK * PADW];
    threadgroup float sq[NSG];
    device const float *Sh = S + (ulong)hv * DK * DV;
    for (uint e = tid; e < DK * DVB; e += NT) {
        const uint r = e / DVB, c = e % DVB;
        St[r * PADW + c] = Sh[r * DV + dv0 + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < NB; t++) {
        const float gdec = exp(a[t * 64 + hv]), b = beta[t * 64 + hv];
        float kk[PER], qq[PER];
        for (uint i = 0; i < PER; i++) {
            kk[i] = float(k[t * HK * DK + hk * DK + lane * PER + i]);
            qq[i] = float(q[t * HK * DK + hk * DK + lane * PER + i]);
        }
        float mysq = 0;
        for (uint cc = 0; cc < CPS; cc++) {
            const uint c = sg * CPS + cc;
            float s[PER], mem = 0;
            for (uint i = 0; i < PER; i++) {
                s[i] = St[(lane * PER + i) * PADW + c] * gdec;
                mem += s[i] * kk[i];
            }
            mem = simd_sum(mem);
            const float d = b * (float(v[t * HV * DV + hv * DV + dv0 + c]) - mem);
            if (lane == 0) dlt[((ulong)t * HV + hv) * DV + dv0 + c] = d;
            float out = 0;
            for (uint i = 0; i < PER; i++) {
                s[i] += kk[i] * d;
                out += s[i] * qq[i];
                St[(lane * PER + i) * PADW + c] = s[i];
            }
            out = simd_sum(out);
            if (lane == 0) o[t * HV * DV + hv * DV + dv0 + c] = half(out);
            const float oh = float(half(out));
            mysq += oh * oh;
        }
        if (lane == 0) sq[sg] = mysq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float tt = 0;
            for (uint i = 0; i < NSG; i++) tt += sq[i];
            ssp[t * HV * (DV / DVB) + hv * (DV / DVB) + tg.y] = tt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#endif
#endif

#ifdef GDN_COMMIT
// 延後寫回的狀態更新：S ← 重放 J 步（S = g_t·S + k_t δ_tᵀ，運算順序同 gdn_step；只差在 FMA 合併的捨入，約 1e-8）
// k [J][HK, DK]、a [J][64]、dlt [J][HV][DV]。grid：(HV, DV / DVB) × 256
kernel void gdn_commit(device const half *k [[buffer(0)]], device const float *a [[buffer(1)]], device const float *dlt [[buffer(2)]],
                       device float *S [[buffer(3)]], constant uint &J [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NT = NSG * 32, PER = DK / 32, CPS = DVB / NSG;
    const uint hv = tg.x, dv0 = tg.y * DVB, hk = HK_OF(hv);
    threadgroup float St[DK * PADW];
    device float *Sh = S + (ulong)hv * DK * DV;
    for (uint e = tid; e < DK * DVB; e += NT) {
        const uint r = e / DVB, c = e % DVB;
        St[r * PADW + c] = Sh[r * DV + dv0 + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < J; t++) {
        const float gdec = exp(a[t * 64 + hv]);
        float kk[PER];
        for (uint i = 0; i < PER; i++) kk[i] = float(k[t * HK * DK + hk * DK + lane * PER + i]);
        for (uint cc = 0; cc < CPS; cc++) {
            const uint c = sg * CPS + cc;
            const float d = dlt[((ulong)t * HV + hv) * DV + dv0 + c];
            for (uint i = 0; i < PER; i++) // 同 gdn_step 的捨入（衰減先捨入、再一次 FMA），重放後的狀態與逐 token 相同
                St[(lane * PER + i) * PADW + c] = fma(kk[i], d, St[(lane * PER + i) * PADW + c] * gdec);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < DK * DVB; e += NT) {
        const uint r = e / DVB, c = e % DVB;
        Sh[r * DV + dv0 + c] = St[r * PADW + c];
    }
}
#endif

#ifdef GDN_FUSED
// ---------------------------------------------------------------- GGUF 合併版（decode 與延後寫回的驗證共用）：一個 dispatch 做完
//   1. 待重放的 J 步（上一輪驗證接受的 token；原本是另一個 gdn_commit dispatch，狀態要多讀寫一次）
//   2. q / k 的無權重 RMSNorm（原本在 k_gdn_in；conv 已併進 dn_inf 的收尾）
//   3. NB 步遞迴：decode（defer = 0）最後把狀態寫回；驗證（defer = 1）狀態只讀，δ 與正規化後的 k 寫出供下一輪重放；
//      decode 延後寫回（defer = 2）：同驗證但重放的結果也不寫回（待重放的步數加一，累積幾步才寫一次狀態）
//   4. gated RMSNorm：每個 head 最後完成的 threadgroup（計數器）接著算整個 head 的 normgate（原本另一個 dispatch）
// 狀態放在暫存器（不經 threadgroup memory）：lane 負責 dk 的 4 列（4·lane … 4·lane + 3）、SIMD group 負責 4 欄，
// 每列 4 欄用一次 float4 讀寫。運算式與 gdn_step / gdn_commit / k_gdn_in / k_normgate 相同；
// fast-math 下編譯器的捨入選擇不一定相同（偶爾差 1 ulp，對 FP64 參考的誤差相同），decode 與驗證之間逐位元相同。
// 主程式注入：NB（每次幾個 token，≤ 8）、GF_DVB（每個 threadgroup 幾欄）
#define GF_EPS 1e-6f
#ifndef GF_DVB
#define GF_DVB 32
#endif
#define GF_NSG (GF_DVB / 4) // 每個 SIMD group 4 欄
static_assert(GF_NSG >= 4 && GF_NSG <= 32, "gdn_fstep：normgate 要至少 4 個 SIMD group（128 個執行緒）");
struct gf_args {
    uint J;     // 先重放幾步（0：沒有待重放的）
    uint defer; // 0：decode（就地寫回狀態）；1：驗證（重放後寫回，遞迴只讀，寫 δ 與 k）；2：decode 延後（完全不寫狀態，寫 δ 與 k）
};
inline float gf_sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// 第 t 個 token、key head hk 的 q / k 正規化（同 k_gdn_in 的 tg_sum：元素 32·j + lane 先 SIMD 內加總，4 份再加總）；
// 回傳這個 lane 負責的 4 列（4·lane + i）
inline void gf_qk(device const half *cv, uint t, uint hk, uint lane, thread float *qq, thread half *kh) {
    device const half *qc = cv + (ulong)t * 10240 + hk * DK, *kc = qc + HK * DK;
    float pq[4], pk[4];
    for (uint j = 0; j < 4; j++) {
        const float qv = float(qc[32 * j + lane]), kv = float(kc[32 * j + lane]);
        pq[j] = simd_sum(qv * qv);
        pk[j] = simd_sum(kv * kv);
    }
    const float rq = rsqrt(simd_sum(lane < 4 ? pq[lane] : 0.0f) / DK + GF_EPS);
    const float rk = rsqrt(simd_sum(lane < 4 ? pk[lane] : 0.0f) / DK + GF_EPS);
    for (uint i = 0; i < 4; i++) {
        const uint e = lane * 4 + i;
        kh[i] = half(float(kc[e]) * rk * rsqrt(float(DK)));
        qq[i] = float(half(float(qc[e]) * rq / DK));
    }
}

// grid：(HV, DV / GF_DVB) × 32·GF_NSG
kernel void gdn_fstep(device const half *cv [[buffer(0)]],    // [NB][10240] conv 輸出：q [0, 2048)、k [2048, 4096)、v [4096, 10240)
                      device const float *la [[buffer(1)]],   // [NB][64] log 衰減
                      device const float *beta [[buffer(2)]], // [NB][64]
                      device float *S [[buffer(3)]],          // [HV, DK, DV]（這一層）
                      device coherent(device) half *o [[buffer(4)]], // [NB][HV·DV]
                      device const half *z [[buffer(5)]],     // [NB][HV·DV]
                      device const bfloat *nw [[buffer(6)]],  // [DV] gated RMSNorm 的權重
                      device half *gy [[buffer(7)]],          // [NB][HV·DV] gated RMSNorm 的輸出
                      device atomic_uint *cnt [[buffer(8)]],  // [HV] 每個 head 完成幾個 threadgroup（最後一個歸零）
                      device half *ko [[buffer(9)]],          // 驗證：[NB][HK·DK] 正規化後的 k（下一輪重放用）
                      device float *dlt [[buffer(10)]],       // 驗證：[NB][HV][DV] δ
                      device const half *rk [[buffer(11)]],   // 重放：[J][HK·DK]
                      device const float *rg [[buffer(12)]],  // 重放：[J][64]
                      device const float *rd [[buffer(13)]],  // 重放：[J][HV][DV]
                      constant gf_args &ga [[buffer(14)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PER = DK / 32, CPS = 4;
    static_assert(PER == 4, "gdn_fstep：每個 lane 4 列 × 4 欄");
    const uint hv = tg.x, dv0 = tg.y * GF_DVB, hk = HK_OF(hv), c0 = dv0 + sg * CPS;
    device float *Sh = S + (ulong)hv * DK * DV;
    float4 s[PER]; // s[i][cc]：列 4·lane + i、欄 c0 + cc
    for (uint i = 0; i < PER; i++) s[i] = *(device const float4 *)(Sh + (lane * PER + i) * DV + c0);
#if NB > 1
    // 多個 token：q / k 正規化每個 token 只算一次（第 t % GF_NSG 個 SIMD group），經 threadgroup memory 給其他 SIMD group
    threadgroup float qs[NB][DK];
    threadgroup half ks[NB][DK];
    for (uint t = sg; t < NB; t += GF_NSG) {
        float qq[4];
        half kh[4];
        gf_qk(cv, t, hk, lane, qq, kh);
        for (uint i = 0; i < 4; i++) {
            qs[t][lane * 4 + i] = qq[i];
            ks[t][lane * 4 + i] = kh[i];
            if (ga.defer && hv == hk && tg.y == 0) ko[t * HK * DK + hk * DK + lane * 4 + i] = kh[i];
        }
    }
#endif
    // 1. 重放（同 gdn_commit）
    for (uint t = 0; t < ga.J; t++) {
        const float gdec = exp(rg[t * 64 + hv]);
        float kk[PER];
        for (uint i = 0; i < PER; i++) kk[i] = float(rk[t * HK * DK + hk * DK + lane * PER + i]);
        const float4 d4 = *(device const float4 *)(rd + ((ulong)t * HV + hv) * DV + c0);
        for (uint cc = 0; cc < CPS; cc++) {
            const float d = d4[cc];
            for (uint i = 0; i < PER; i++) s[i][cc] = fma(kk[i], d, s[i][cc] * gdec); // 同遞迴那一步的捨入（衰減先捨入、再一次 FMA）
        }
    }
    if (ga.J && ga.defer == 1)
        for (uint i = 0; i < PER; i++) *(device float4 *)(Sh + (lane * PER + i) * DV + c0) = s[i];
#if NB > 1
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    // 2–3. 每個 token 的遞迴
    for (uint t = 0; t < NB; t++) {
        float kk[PER], qq[PER];
#if NB > 1
        for (uint i = 0; i < PER; i++) { kk[i] = float(ks[t][lane * PER + i]); qq[i] = qs[t][lane * PER + i]; }
#else
        {
            half kh[4];
            gf_qk(cv, t, hk, lane, qq, kh);
            for (uint i = 0; i < PER; i++) {
                kk[i] = float(kh[i]);
                if (ga.defer && hv == hk && tg.y == 0 && sg == 0) ko[t * HK * DK + hk * DK + lane * PER + i] = kh[i];
            }
        }
#endif
        const float gdec = exp(la[t * 64 + hv]), b = beta[t * 64 + hv];
        const float4 v4 = float4(*(device const half4 *)(cv + (ulong)t * 10240 + 2 * HK * DK + hv * DV + c0));
        float4 d4;
        for (uint cc = 0; cc < CPS; cc++) {
            float mem = 0;
            for (uint i = 0; i < PER; i++) {
                s[i][cc] = s[i][cc] * gdec;
                mem = fma(s[i][cc], kk[i], mem);
            }
            mem = simd_sum(mem);
            const float d = b * (v4[cc] - mem);
            d4[cc] = d;
            float out = 0;
            for (uint i = 0; i < PER; i++) {
                s[i][cc] = fma(kk[i], d, s[i][cc]);
                out = fma(s[i][cc], qq[i], out);
            }
            out = simd_sum(out);
            if (lane == 0) o[(ulong)t * HV * DV + hv * DV + c0 + cc] = half(out);
        }
        if (ga.defer && lane == 0) *(device float4 *)(dlt + ((ulong)t * HV + hv) * DV + c0) = d4; // 4 欄一起寫（c0 是 4 的倍數）
    }
    if (ga.defer == 0)
        for (uint i = 0; i < PER; i++) *(device float4 *)(Sh + (lane * PER + i) * DV + c0) = s[i];
    // 4. 這個 head 最後完成的 threadgroup 算 gated RMSNorm（同 k_normgate / kb_normgate 的 tg_sum：元素 32·j + lane 先 SIMD 內加總，
    //    4 份再加總）；每個 SIMD group 一個 token，不用 threadgroup barrier
    threadgroup bool last;
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        last = atomic_fetch_add_explicit(cnt + hv, 1u, memory_order_relaxed) == DV / GF_DVB - 1;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last) return;
    static_assert(DV == 128, "gdn_fstep：normgate 每個 lane 4 個元素");
    for (uint t = sg; t < NB; t += GF_NSG) {
        const ulong b0 = (ulong)t * HV * DV + hv * DV;
        float x[4], p[4];
        for (uint j = 0; j < 4; j++) {
            x[j] = float(o[b0 + 32 * j + lane]);
            p[j] = simd_sum(x[j] * x[j]);
        }
        const float r = rsqrt(simd_sum(lane < 4 ? p[lane] : 0.0f) / DV + GF_EPS);
        for (uint j = 0; j < 4; j++) {
            const uint e = 32 * j + lane;
            gy[b0 + e] = half(x[j] * r * float(nw[e]) * gf_sigm(float(z[b0 + e])));
        }
    }
    if (tid == 0) atomic_store_explicit(cnt + hv, 0u, memory_order_relaxed);
}
#endif
