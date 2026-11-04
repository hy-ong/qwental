// Qwen Sparse Attention（QSA）— decode 路徑（單一 query）
// 演算法來源：mlx-lm PR #1788 的模型實作。
//
// 輸入（都已經過 RMSNorm 與 RoPE）：
//   iq   [IH, ID] half        indexer q（IH = 4、ID = 128）
//   kblk [NB, ID] half        區塊 key（每 4 個 token 的 raw_k 平均 → RMSNorm → RoPE）
//   q    [QH, HD] half        attention q（QH = 24、HD = 256）
//   K、V [L, KVH, HD] half    KV cache（KVH = 2）
// 流程：
//   1. qsa_score：score[n] = Σ_h max(iq_h · kblk_n, 0) / √ID，n < nvis = (pos + 1) / 4
//   2. qsa_topk ：一個 threadgroup 用 radix select 選出分數最高的 SEL = 512 個區塊
//   3. qsa_attn ：選中區塊的 token（+ query 自己那段未滿 4 個的尾巴）切成 NCH 段，
//                 每段一個 threadgroup：12 個 SIMD group = 同一個 kv head 的 12 個 q head
//   4. qsa_merge：合併各段的 (max, sum, acc)
// 主程式注入：IH、ID、QH、KVH、HD、SEL、CR（compress ratio = 4）、CHUNK（每段 token 數）

#include <metal_stdlib>
using namespace metal;

#define GQA (QH / KVH)

struct qsa_args {
    uint L;      // context 長度（含目前的 token）
    uint pos;    // query 位置 = L − 1
    uint nvis;   // 可選區塊數 = (pos + 1) / CR
    uint ntok;   // 要看的 token 數（稀疏時 = 選中區塊 × CR + 尾巴；不稀疏時 = L）
    uint sparse; // 0：L ≤ 預算，直接看全部
};

// grid：涵蓋 nvis 個區塊，每個執行緒一個區塊
kernel void qsa_score(device const half *iq [[buffer(0)]],
                      device const half *kblk [[buffer(1)]],
                      device float *score [[buffer(2)]],
                      constant qsa_args &a [[buffer(3)]],
                      uint n [[thread_position_in_grid]]) {
    if (n >= a.nvis) return;
    device const half4 *k = (device const half4 *)(kblk + (ulong)n * ID);
    float s = 0;
    for (uint h = 0; h < IH; h++) {
        device const half4 *qh = (device const half4 *)(iq + h * ID);
        float d = 0;
        for (uint i = 0; i < ID / 4; i++) d += dot(float4(qh[i]), float4(k[i]));
        s += max(d, 0.0f);
    }
    score[n] = s * rsqrt(float(ID));
}

// 一個 threadgroup、1024 個執行緒。分數 ≥ 0，所以 float 的位元當 uint 比大小即可。
// 4 輪、每輪 8 bit 的 radix select 找出第 SEL 大的值，再輸出所有大於它的、加上足夠數量的同值。
kernel void qsa_topk(device const float *score [[buffer(0)]],
                     device uint *sel [[buffer(1)]],      // [SEL] 選中的區塊編號
                     device uint *nsel [[buffer(2)]],     // 實際選了幾個（nvis < SEL 時較少）
                     constant qsa_args &a [[buffer(3)]],
                     uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NT = 1024;
    threadgroup atomic_uint hist[256];
    threadgroup uint s_prefix, s_mask, s_need;
    threadgroup atomic_uint s_out, s_eq;
    const uint nv = a.nvis;
    if (nv <= SEL) { // 全部選
        for (uint i = tid; i < nv; i += NT) sel[i] = i;
        if (tid == 0) *nsel = nv;
        return;
    }
    if (tid == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_need = SEL; // 還要從「前綴相符」的元素裡找第幾大
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint b = tid; b < 256; b += NT) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint prefix = s_prefix, mask = s_mask;
        for (uint i = tid; i < nv; i += NT) {
            const uint u = as_type<uint>(score[i]);
            if ((u & mask) == prefix) atomic_fetch_add_explicit(&hist[(u >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint need = s_need, acc = 0, b = 255;
            for (;; b--) {
                const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (acc + c >= need || b == 0) break;
                acc += c;
            }
            s_need = need - acc;
            s_prefix = prefix | (b << shift);
            s_mask = mask | (255u << shift);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // 門檻 = s_prefix（第 SEL 大的值）。先輸出所有大於門檻的，再補 s_need 個等於門檻的。
    const uint thr = s_prefix;
    if (tid == 0) {
        atomic_store_explicit(&s_out, 0u, memory_order_relaxed);
        atomic_store_explicit(&s_eq, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup atomic_uint *out_cnt = &s_out;
    threadgroup atomic_uint *eq_cnt = &s_eq;
    const uint need_eq = s_need;
    for (uint i = tid; i < nv; i += NT) {
        const uint u = as_type<uint>(score[i]);
        if (u > thr) sel[atomic_fetch_add_explicit(out_cnt, 1u, memory_order_relaxed)] = i;
        else if (u == thr) {
            const uint e = atomic_fetch_add_explicit(eq_cnt, 1u, memory_order_relaxed);
            if (e < need_eq) sel[SEL - need_eq + e] = i;
        }
    }
    if (tid == 0) *nsel = SEL;
}

// 第 i 個要看的 token
static inline uint tok_of(uint i, device const uint *sel, uint nsel, constant qsa_args &a) {
    if (!a.sparse) return i;
    const uint nb = nsel * CR;
    if (i < nb) return sel[i / CR] * CR + i % CR;
    return a.nvis * CR + (i - nb); // query 自己那段（從最後一個完整區塊之後到 pos）
}

// grid：(NCH, KVH)，每個 threadgroup GQA 個 SIMD group（一個 SIMD group 一個 q head）
// 每個 lane 負責 HD / 32 = 8 維
kernel void qsa_attn(device const half *q [[buffer(0)]],
                     device const half *K [[buffer(1)]],
                     device const half *V [[buffer(2)]],
                     device const uint *sel [[buffer(3)]],
                     device const uint *nsel_p [[buffer(4)]],
                     device float *part [[buffer(5)]], // [NCH, QH, HD + 2]：acc、max、sum
                     constant qsa_args &a [[buffer(6)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint2 ntg [[threadgroups_per_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PL = HD / 32;
    const uint ch = tg.x, kvh = tg.y, qh = kvh * GQA + sg;
    const uint nsel = a.sparse ? *nsel_p : 0;
    const uint ntok = a.ntok, per = (ntok + ntg.x - 1) / ntg.x;
    const uint i0 = ch * per, i1 = min(i0 + per, ntok);
    const float scale = rsqrt(float(HD));
    float qv[PL], acc[PL];
    for (uint d = 0; d < PL; d++) {
        qv[d] = float(q[qh * HD + lane * PL + d]) * scale;
        acc[d] = 0;
    }
    float m = -INFINITY, l = 0;
    for (uint i = i0; i < i1; i++) {
        const ulong t = tok_of(i, sel, nsel, a);
        device const half *kr = K + (t * KVH + kvh) * HD + lane * PL;
        device const half *vr = V + (t * KVH + kvh) * HD + lane * PL;
        float s = 0;
        for (uint d = 0; d < PL; d++) s += qv[d] * float(kr[d]);
        s = simd_sum(s);
        const float mn = max(m, s), alpha = exp(m - mn), p = exp(s - mn);
        l = l * alpha + p;
        for (uint d = 0; d < PL; d++) acc[d] = acc[d] * alpha + p * float(vr[d]);
        m = mn;
    }
    device float *o = part + ((ulong)ch * QH + qh) * (HD + 2);
    for (uint d = 0; d < PL; d++) o[lane * PL + d] = acc[d];
    if (lane == 0) {
        o[HD] = m;
        o[HD + 1] = l;
    }
}

// grid：QH 個 threadgroup × 32 個執行緒
kernel void qsa_merge(device const float *part [[buffer(0)]],
                      device half *out [[buffer(1)]], // [QH, HD]
                      constant uint &nch [[buffer(2)]],
                      uint qh [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint PL = HD / 32;
    float M = -INFINITY;
    for (uint c = 0; c < nch; c++) M = max(M, part[((ulong)c * QH + qh) * (HD + 2) + HD]);
    float Ls = 0, acc[PL];
    for (uint d = 0; d < PL; d++) acc[d] = 0;
    for (uint c = 0; c < nch; c++) {
        device const float *p = part + ((ulong)c * QH + qh) * (HD + 2);
        if (p[HD + 1] == 0) continue;
        const float w = exp(p[HD] - M);
        Ls += p[HD + 1] * w;
        for (uint d = 0; d < PL; d++) acc[d] += p[lane * PL + d] * w;
    }
    for (uint d = 0; d < PL; d++) out[qh * HD + lane * PL + d] = half(acc[d] / Ls);
}


// ================================================================ v2：較快的版本

// 打分數 v2：一個 SIMD group 一個區塊，每個 lane 讀 4 維（合併讀取）
// grid：nvis 個 SIMD group（每個 threadgroup 8 個）
kernel void qsa_score2(device const half *iq [[buffer(0)]],
                       device const half *kblk [[buffer(1)]],
                       device float *score [[buffer(2)]],
                       constant qsa_args &a [[buffer(3)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint n = tg * 8 + sg;
    if (n >= a.nvis) return;
    static_assert(ID == 128, "每個 lane 4 維");
    const float4 k = float4(((device const half4 *)(kblk + (ulong)n * ID))[lane]);
    float s = 0;
    for (uint h = 0; h < IH; h++) s += max(simd_sum(dot(float4(((device const half4 *)(iq + h * ID))[lane]), k)), 0.0f);
    if (lane == 0) score[n] = s * rsqrt(float(ID));
}

// 選取 v2：分數放暫存器，二分搜尋門檻（不用 atomic）
// 一個 threadgroup、1024 個執行緒；每個執行緒最多 MAXPER 個分數
#ifndef MAXPER
#define MAXPER 32 // 每個執行緒最多幾個分數：1024 × 32 = 32K 個區塊 = 128K context
#endif
kernel void qsa_topk2(device const float *score [[buffer(0)]],
                      device uint *sel [[buffer(1)]],
                      device uint *nsel [[buffer(2)]],
                      constant qsa_args &a [[buffer(3)]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NT = 1024, NSG = NT / 32;
    const uint nv = a.nvis;
    if (nv <= SEL) {
        for (uint i = tid; i < nv; i += NT) sel[i] = i;
        if (tid == 0) *nsel = nv;
        return;
    }
    threadgroup uint red[NSG];
    // 每個執行緒負責連續的一段（b0 起 MAXPER 個），超出範圍的填 0（分數 ≥ 0，所以 0 不會被誤選：
    // 門檻 thr 一定 > 0 或者被選的 0 剛好是補位的同值，下面寫出時另外檢查 i < nv）
    const uint b0 = tid * MAXPER;
    uint v[MAXPER];
    uint hi = 0;
#pragma unroll
    for (uint j = 0; j < MAXPER; j++) {
        const uint i = b0 + j;
        v[j] = i < nv ? as_type<uint>(score[i]) : 0u;
        hi = max(hi, v[j]);
    }
    // 計數：有幾個 ≥ t
    auto count_ge = [&](uint t) -> uint {
        uint c = 0;
#pragma unroll
        for (uint j = 0; j < MAXPER; j++) c += (b0 + j < nv && v[j] >= t) ? 1u : 0u;
        c = simd_sum(c);
        if (lane == 0) red[sg] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tot = lane < NSG ? red[lane] : 0u;
        tot = simd_sum(tot);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return tot;
    };
    hi = simd_max(hi);
    if (lane == 0) red[sg] = hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hi = simd_max(lane < NSG ? red[lane] : 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 找最大的 t 使得 count(≥ t) ≥ SEL
    uint lo = 0, top = hi; // count(≥ lo) = nv ≥ SEL
    while (lo < top) {
        const uint mid = lo + (top - lo + 1) / 2;
        if (count_ge(mid) >= SEL) lo = mid;
        else top = mid - 1;
    }
    const uint thr = lo;
    // 寫出：先所有 > thr 的，再補等於 thr 的（依位置順序，用前綴和決定寫入位置）
    uint ngt = 0, neq = 0;
#pragma unroll
    for (uint j = 0; j < MAXPER; j++)
        if (b0 + j < nv) {
            ngt += v[j] > thr;
            neq += v[j] == thr;
        }
    threadgroup uint pre_gt[NSG], pre_eq[NSG];
    uint xg = simd_prefix_exclusive_sum(ngt), xe = simd_prefix_exclusive_sum(neq);
    if (lane == 31) {
        pre_gt[sg] = xg + ngt;
        pre_eq[sg] = xe + neq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint a1 = lane < NSG ? pre_gt[lane] : 0, a2 = lane < NSG ? pre_eq[lane] : 0;
        uint s1 = simd_prefix_exclusive_sum(a1), s2 = simd_prefix_exclusive_sum(a2);
        if (lane < NSG) {
            pre_gt[lane] = s1;
            pre_eq[lane] = s2;
        }
        if (lane == NSG - 1) red[0] = s1 + a1; // 大於門檻的總數
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint total_gt = red[0];
    uint og = pre_gt[sg] + xg, oe = pre_eq[sg] + xe;
#pragma unroll
    for (uint j = 0; j < MAXPER; j++)
        if (b0 + j < nv) {
            if (v[j] > thr) sel[og++] = b0 + j;
            else if (v[j] == thr) {
                if (total_gt + oe < SEL) sel[total_gt + oe] = b0 + j;
                oe++;
            }
        }
    if (tid == 0) *nsel = SEL;
}

// 多個 query（prefill）：threadgroup z 處理第 z 個 query；分數列間距 sstride、參數 aa[z]
kernel void qsa_topk2b(device const float *score0 [[buffer(0)]],
                       device uint *sel0 [[buffer(1)]],
                       device uint *nsel0 [[buffer(2)]],
                       device const qsa_args *aa [[buffer(3)]],
                       constant uint &sstride [[buffer(4)]],
                       uint z [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    device const float *score = score0 + (ulong)z * sstride;
    device uint *sel = sel0 + (ulong)z * SEL;
    device uint *nsel = nsel0 + z;
    const qsa_args a = aa[z];
    constexpr uint NT = 1024, NSG = NT / 32;
    const uint nv = a.nvis;
    if (nv <= SEL) {
        for (uint i = tid; i < nv; i += NT) sel[i] = i;
        if (tid == 0) *nsel = nv;
        return;
    }
    threadgroup uint red[NSG];
    // 每個執行緒負責連續的一段（b0 起 MAXPER 個），超出範圍的填 0（分數 ≥ 0，所以 0 不會被誤選：
    // 門檻 thr 一定 > 0 或者被選的 0 剛好是補位的同值，下面寫出時另外檢查 i < nv）
    const uint b0 = tid * MAXPER;
    uint v[MAXPER];
    uint hi = 0;
#pragma unroll
    for (uint j = 0; j < MAXPER; j++) {
        const uint i = b0 + j;
        v[j] = i < nv ? as_type<uint>(score[i]) : 0u;
        hi = max(hi, v[j]);
    }
    // 計數：有幾個 ≥ t
    auto count_ge = [&](uint t) -> uint {
        uint c = 0;
#pragma unroll
        for (uint j = 0; j < MAXPER; j++) c += (b0 + j < nv && v[j] >= t) ? 1u : 0u;
        c = simd_sum(c);
        if (lane == 0) red[sg] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tot = lane < NSG ? red[lane] : 0u;
        tot = simd_sum(tot);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return tot;
    };
    hi = simd_max(hi);
    if (lane == 0) red[sg] = hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hi = simd_max(lane < NSG ? red[lane] : 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 找最大的 t 使得 count(≥ t) ≥ SEL
    uint lo = 0, top = hi; // count(≥ lo) = nv ≥ SEL
    while (lo < top) {
        const uint mid = lo + (top - lo + 1) / 2;
        if (count_ge(mid) >= SEL) lo = mid;
        else top = mid - 1;
    }
    const uint thr = lo;
    // 寫出：先所有 > thr 的，再補等於 thr 的（依位置順序，用前綴和決定寫入位置）
    uint ngt = 0, neq = 0;
#pragma unroll
    for (uint j = 0; j < MAXPER; j++)
        if (b0 + j < nv) {
            ngt += v[j] > thr;
            neq += v[j] == thr;
        }
    threadgroup uint pre_gt[NSG], pre_eq[NSG];
    uint xg = simd_prefix_exclusive_sum(ngt), xe = simd_prefix_exclusive_sum(neq);
    if (lane == 31) {
        pre_gt[sg] = xg + ngt;
        pre_eq[sg] = xe + neq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint a1 = lane < NSG ? pre_gt[lane] : 0, a2 = lane < NSG ? pre_eq[lane] : 0;
        uint s1 = simd_prefix_exclusive_sum(a1), s2 = simd_prefix_exclusive_sum(a2);
        if (lane < NSG) {
            pre_gt[lane] = s1;
            pre_eq[lane] = s2;
        }
        if (lane == NSG - 1) red[0] = s1 + a1; // 大於門檻的總數
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint total_gt = red[0];
    uint og = pre_gt[sg] + xg, oe = pre_eq[sg] + xe;
#pragma unroll
    for (uint j = 0; j < MAXPER; j++)
        if (b0 + j < nv) {
            if (v[j] > thr) sel[og++] = b0 + j;
            else if (v[j] == thr) {
                if (total_gt + oe < SEL) sel[total_gt + oe] = b0 + j;
                oe++;
            }
        }
    if (tid == 0) *nsel = SEL;
}

// attention v2：每次 4 個 token，讀取與加總可以重疊
inline void attn2_body(device const half *q, device const half *K, device const half *V, device const uint *sel,
                       device const uint *nsel_p, device float *part, constant qsa_args &a, uint2 tg, uint2 ntg, uint sg, uint lane) {
    constexpr uint PL = HD / 32, U = 4;
    const uint ch = tg.x, kvh = tg.y, qh = kvh * GQA + sg;
    const uint nsel = a.sparse ? *nsel_p : 0;
    const uint ntok = a.ntok, per = (ntok + ntg.x - 1) / ntg.x;
    const uint i0 = ch * per, i1 = min(i0 + per, ntok);
    const float scale = rsqrt(float(HD));
    float qv[PL], acc[PL];
    {
        const float4 q0 = float4(((device const half4 *)(q + qh * HD + lane * PL))[0]);
        const float4 q1 = float4(((device const half4 *)(q + qh * HD + lane * PL))[1]);
        qv[0] = q0.x * scale; qv[1] = q0.y * scale; qv[2] = q0.z * scale; qv[3] = q0.w * scale;
        qv[4] = q1.x * scale; qv[5] = q1.y * scale; qv[6] = q1.z * scale; qv[7] = q1.w * scale;
    }
    for (uint d = 0; d < PL; d++) acc[d] = 0;
    float m = -INFINITY, l = 0;
    for (uint i = i0; i < i1; i += U) {
        float s[U];
        float4 va[U], vb[U];
        for (uint u = 0; u < U; u++) {
            const bool ok = i + u < i1;
            const ulong t = tok_of(ok ? i + u : i0, sel, nsel, a);
            device const half4 *kr = (device const half4 *)(K + (t * KVH + kvh) * HD + lane * PL);
            device const half4 *vr = (device const half4 *)(V + (t * KVH + kvh) * HD + lane * PL);
            const float4 k0 = float4(kr[0]), k1 = float4(kr[1]);
            va[u] = float4(vr[0]);
            vb[u] = float4(vr[1]);
            float d = k0.x * qv[0] + k0.y * qv[1] + k0.z * qv[2] + k0.w * qv[3] + k1.x * qv[4] + k1.y * qv[5] +
                      k1.z * qv[6] + k1.w * qv[7];
            s[u] = ok ? simd_sum(d) : -INFINITY;
        }
        float mn = m;
        for (uint u = 0; u < U; u++) mn = max(mn, s[u]);
        const float alpha = exp(m - mn);
        l *= alpha;
        for (uint d = 0; d < PL; d++) acc[d] *= alpha;
        for (uint u = 0; u < U; u++) {
            const float p = exp(s[u] - mn);
            l += p;
            acc[0] += p * va[u].x; acc[1] += p * va[u].y; acc[2] += p * va[u].z; acc[3] += p * va[u].w;
            acc[4] += p * vb[u].x; acc[5] += p * vb[u].y; acc[6] += p * vb[u].z; acc[7] += p * vb[u].w;
        }
        m = mn;
    }
    device float *o = part + ((ulong)ch * QH + qh) * (HD + 2);
    for (uint d = 0; d < PL; d++) o[lane * PL + d] = acc[d];
    if (lane == 0) {
        o[HD] = m;
        o[HD + 1] = l;
    }
}

kernel void qsa_attn2(device const half *q [[buffer(0)]],
                      device const half *K [[buffer(1)]],
                      device const half *V [[buffer(2)]],
                      device const uint *sel [[buffer(3)]],
                      device const uint *nsel_p [[buffer(4)]],
                      device float *part [[buffer(5)]],
                      constant qsa_args &a [[buffer(6)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint2 ntg [[threadgroups_per_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    attn2_body(q, K, V, sel, nsel_p, part, a, tg, ntg, sg, lane);
}

// 多個 query（推測解碼驗證）：grid z = 第幾個 token，各自的參數 aa[z]、q / sel / part 依 z 錯開
kernel void qsa_attn2b(device const half *q [[buffer(0)]],
                       device const half *K [[buffer(1)]],
                       device const half *V [[buffer(2)]],
                       device const uint *sel [[buffer(3)]],
                       device const uint *nsel_p [[buffer(4)]],
                       device float *part [[buffer(5)]],
                       constant qsa_args *aa [[buffer(6)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       uint3 ntg [[threadgroups_per_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint z = tg.z;
    attn2_body(q + (ulong)z * QH * HD, K, V, sel + (ulong)z * SEL, nsel_p + z, part + (ulong)z * ntg.x * QH * (HD + 2), aa[z],
               tg.xy, ntg.xy, sg, lane);
}

// 合併 v2：每個 q head 一個 threadgroup、8 個 SIMD group 分攤各段
inline void merge2_body(device const float *part, device half *out, uint nch, uint qh, uint sg, uint lane, threadgroup float *sm,
                        threadgroup float *sl, threadgroup float (*sacc)[HD]) {
    constexpr uint PL = HD / 32, NS = 8;
    float m = -INFINITY, l = 0, acc[PL];
    for (uint d = 0; d < PL; d++) acc[d] = 0;
    for (uint c = sg; c < nch; c += NS) {
        device const float *p = part + ((ulong)c * QH + qh) * (HD + 2);
        const float pm = p[HD], pl = p[HD + 1];
        if (pl == 0) continue;
        const float mn = max(m, pm), a0 = exp(m - mn), a1 = exp(pm - mn);
        l = l * a0 + pl * a1;
        for (uint d = 0; d < PL; d++) acc[d] = acc[d] * a0 + p[lane * PL + d] * a1;
        m = mn;
    }
    for (uint d = 0; d < PL; d++) sacc[sg][lane * PL + d] = acc[d];
    if (lane == 0) {
        sm[sg] = m;
        sl[sg] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float M = -INFINITY;
        for (uint s2 = 0; s2 < NS; s2++) M = max(M, sm[s2]);
        float Ls = 0, o[PL];
        for (uint d = 0; d < PL; d++) o[d] = 0;
        for (uint s2 = 0; s2 < NS; s2++) {
            if (sl[s2] == 0) continue;
            const float w = exp(sm[s2] - M);
            Ls += sl[s2] * w;
            for (uint d = 0; d < PL; d++) o[d] += sacc[s2][lane * PL + d] * w;
        }
        for (uint d = 0; d < PL; d++) out[qh * HD + lane * PL + d] = half(o[d] / Ls);
    }
}

kernel void qsa_merge2(device const float *part [[buffer(0)]],
                       device half *out [[buffer(1)]],
                       constant uint &nch [[buffer(2)]],
                       uint qh [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sm[8], sl[8], sacc[8][HD];
    merge2_body(part, out, nch, qh, sg, lane, sm, sl, sacc);
}

// grid：(QH, 幾個 token)
kernel void qsa_merge2b(device const float *part [[buffer(0)]],
                        device half *out [[buffer(1)]],
                        constant uint &nch [[buffer(2)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sm[8], sl[8], sacc[8][HD];
    merge2_body(part + (ulong)tg.y * nch * QH * (HD + 2), out + (ulong)tg.y * QH * HD, nch, tg.x, sg, lane, sm, sl, sacc);
}

// ---------------------------------------------------------------- 稀疏 prefill：indexer 分數矩陣
// score[t][n] = Σ_h max(iq[t,h] · kblk[n], 0) / √ID，只寫 n < nvis(t) = (pos0 + t + 1) / CR。
// 每個執行緒一個區塊、一次 QT 個 query（q 放 threadgroup memory 共用）。grid：(ceil(nb / 256), ceil(T / QT)) × 256
#define QT 8
kernel void p_qsa_score(device const half *iq [[buffer(0)]],     // [T][IH][ID]
                        device const half *kblk [[buffer(1)]],   // [nb][ID]
                        device float *score [[buffer(2)]],       // [T][sstride]
                        constant uint &pos0 [[buffer(3)]],
                        constant uint &T [[buffer(4)]],
                        constant uint &sstride [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half4 qs[QT][IH * ID / 4];
    const uint t0 = tg.y * QT, n = tg.x * 256 + tid;
    for (uint i = tid; i < QT * IH * ID / 4; i += 256) {
        const uint q = i / (IH * ID / 4), j = i % (IH * ID / 4);
        qs[q][j] = t0 + q < T ? ((device const half4 *)(iq + (ulong)(t0 + q) * IH * ID))[j] : half4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tl = min((uint)QT, T - t0);
    const uint nvmax = (pos0 + t0 + tl) / CR; // 這組 query 中最大的可選區塊數
    if (n >= nvmax) return;
    float acc[QT][IH];
    for (uint q = 0; q < QT; q++)
        for (uint h = 0; h < IH; h++) acc[q][h] = 0;
    device const half4 *kr = (device const half4 *)(kblk + (ulong)n * ID);
    for (uint d = 0; d < ID / 4; d++) {
        const float4 k = float4(kr[d]);
        for (uint q = 0; q < QT; q++)
            for (uint h = 0; h < IH; h++) acc[q][h] += dot(float4(qs[q][h * ID / 4 + d]), k);
    }
    for (uint q = 0; q < tl; q++) {
        const uint t = t0 + q;
        if (n >= (pos0 + t + 1) / CR) continue;
        float sc = 0;
        for (uint h = 0; h < IH; h++) sc += max(acc[q][h], 0.0f);
        score[(ulong)t * sstride + n] = sc * rsqrt(float(ID));
    }
}
