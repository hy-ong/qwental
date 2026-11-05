// QSA attention（prefill）用 tensor 單元：每個 threadgroup 處理一個 (kv head, query)，不分段、不用合併。
// 結構參考 Strata 的 qsa_flash（MIT，metal/kernels/qsa_prompt.metal）：
//   一次 QCH = 64 個要看的 token：S[16 × 64] = Q[12 → 16 × 256] · Kᵀ（4 個 SIMD group 各算 64 維的部分和）
//   → 每個 head 一列做 online softmax（FP32）→ P 轉 FP16 → acc[16 × 256] += P · V（每個 SIMD group 負責 64 維輸出）
// 要看的 token 與 decode 的 qsa_attn 相同（tok_of）：稀疏時是選中的區塊 + query 自己那段尾巴，否則 0..pos。
// 輸出 out[t][QH][HD] = acc / l（FP16），之後一樣由 kb_attn_gate 乘上 gate。
// 主程式注入：QH、KVH、HD、SEL、CR。grid：(KVH, T) × 128
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define GQA (QH / KVH)
#define QCH 64
#ifndef K4
#define K4 0
#endif

struct qsa_pf_args {
    uint L, pos, nvis, ntok, sparse;
};

static inline int pf_tok(uint i, device const uint *sel, uint nsel, constant qsa_pf_args &a) {
    if (!a.sparse) return (int)i;
    const uint nb = nsel * CR;
    if (i < nb) return (int)(sel[i / CR] * CR + i % CR);
    return (int)(a.nvis * CR + (i - nb));
}

kernel void qsa_flash_pf(device const half *q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]],
                         device const uint *sel_all [[buffer(3)]], device const uint *nsel_all [[buffer(4)]],
                         device half *out [[buffer(5)]], constant qsa_pf_args *aa [[buffer(6)]],
                         uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sp[4][16 * QCH];
    threadgroup half p[16 * QCH];
    threadgroup int cid[QCH];
    threadgroup float fac[16], lsum[16], mrun[16];
    const uint g = tg.x, t = tg.y;
    constant qsa_pf_args &a = aa[t];
    device const uint *sel = sel_all + (ulong)t * SEL;
    const uint nsel = a.sparse ? nsel_all[t] : 0, ntok = a.ntok;
    const int nch = (int)((ntok + QCH - 1) / QCH);
    if (tid < 16) { mrun[tid] = -INFINITY; lsum[tid] = 0.0f; }

    constexpr auto dS = matmul2d_descriptor(16, QCH, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dO = matmul2d_descriptor(16, 64, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> opS;
    matmul2d<dO, execution_simdgroup> opO;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ((device half *)q + ((ulong)t * QH + g * GQA) * HD,
                                                               dextents<int32_t, 2>(HD, GQA));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(p, dextents<int32_t, 2>(QCH, 16));
    auto rv = opO.template get_right_input_cooperative_tensor<half, half, float>();
    auto tPh = tP.template slice<32, 16>(0, 0);
    auto acc = opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>,
                                                               __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
    auto tot = opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>,
                                                               __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
    for (uint16_t i = 0; i < tot.get_capacity(); ++i)
        if (tot.is_valid_element(i)) tot[i] = 0.0f;
    const int d0 = sg * 64;
    const ulong rs = KVH * HD; // 相鄰 token 的列距
    for (int ch = 0; ch < nch; ++ch) {
        const int c0 = ch * QCH, nc = min(QCH, (int)ntok - c0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < QCH) cid[tid] = tid < nc ? pf_tok((uint)(c0 + tid), sel, nsel, a) : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S 的部分和（這個 SIMD group 的 64 維）
        {
            auto rk = opS.template get_right_input_cooperative_tensor<half, half, float>();
            auto tA = tQ.template slice<32, 16>(d0, 0);
            auto s = opS.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>,
                                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
            for (uint16_t i = 0; i < s.get_capacity(); ++i)
                if (s.is_valid_element(i)) s[i] = 0.0f;
            for (int k0 = d0; k0 < d0 + 64; k0 += 32) {
#if K4
                // M5 的排列：每個 lane 連續 4 個元素是同一個 cell 的連續 4 維 → 一次讀 half4
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < rk.get_capacity(); i += 4) {
                    const auto ix = rk.get_multidimensional_index(i); // (k, cell)
                    const int id = cid[ix[1]];
                    const half4 w = id >= 0 ? *(device const half4 *)(K + (ulong)id * rs + g * HD + k0 + ix[0]) : half4(0.0h);
                    rk[i] = w.x; rk[i + 1] = w.y; rk[i + 2] = w.z; rk[i + 3] = w.w;
                }
#else
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < rk.get_capacity(); ++i) {
                    const auto ix = rk.get_multidimensional_index(i); // (k, cell)
                    const int id = cid[ix[1]];
                    rk[i] = id >= 0 ? K[(ulong)id * rs + g * HD + k0 + ix[0]] : half(0.0h);
                }
#endif
                tA = tQ.template slice<32, 16>(k0, 0);
                opS.run(tA, rk, s);
            }
            tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(sp[sg], dextents<int32_t, 2>(QCH, 16));
            s.store(tS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax：每列 8 個執行緒（16 列 × 8 = 128），各 8 個 token，用 shuffle 合併（scale = 1/16，精確）
        {
            const uint row = tid / 8, part = tid % 8;
            float m = row < GQA ? mrun[row] : -INFINITY;
            for (int c = part * 8; c < (int)part * 8 + 8; ++c)
                if (row < GQA && c < nc) {
                    const int qi = row * QCH + c;
                    m = max(m, ((sp[0][qi] + sp[1][qi]) + (sp[2][qi] + sp[3][qi])) * 0.0625f);
                }
            m = max(m, simd_shuffle_xor(m, 1));
            m = max(m, simd_shuffle_xor(m, 2));
            m = max(m, simd_shuffle_xor(m, 4));
            float l = 0;
            for (int c = part * 8; c < (int)part * 8 + 8; ++c) {
                const int qi = row * QCH + c;
                const float ev = (row < GQA && c < nc) ? exp(((sp[0][qi] + sp[1][qi]) + (sp[2][qi] + sp[3][qi])) * 0.0625f - m) : 0.0f;
                l += ev;
                p[qi] = half(ev);
            }
            l += simd_shuffle_xor(l, 1);
            l += simd_shuffle_xor(l, 2);
            l += simd_shuffle_xor(l, 4);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (part == 0) {
                if (row < GQA) {
                    const float f = exp(mrun[row] - m);
                    lsum[row] = lsum[row] * f + l;
                    mrun[row] = m;
                    fac[row] = f;
                } else fac[row] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // acc = P · V（這個 SIMD group 的 64 維），tot = tot · fac + acc
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) acc[i] = 0.0f;
        for (int cc = 0; cc < QCH; cc += 32) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rv.get_capacity(); ++i) {
                const auto ix = rv.get_multidimensional_index(i); // (cell, dim)
                const int id = cid[cc + ix[0]];
                rv[i] = id >= 0 ? V[(ulong)id * rs + g * HD + d0 + ix[1]] : half(0.0h);
            }
            tPh = tP.template slice<32, 16>(cc, 0);
            opO.run(tPh, rv, acc);
        }
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) tot[i] = fma(tot[i], fac[acc.get_multidimensional_index(i)[1]], acc[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < tot.get_capacity(); ++i)
        if (tot.is_valid_element(i)) {
            const auto ix = tot.get_multidimensional_index(i); // (dim, row)
            const int h = ix[1];
            if (h < GQA) out[((ulong)t * QH + g * GQA + h) * HD + d0 + ix[0]] = half(tot[i] / lsum[h]);
        }
}

// ================================================================ 4 個相鄰 query 共用 K/V 讀取
// qsa_union4：每組 4 個 query，選中區塊（稀疏）或 0..pos/4（不稀疏）的聯集 → ulist [組][MAXU] 區塊編號（遞增）、
// umask [組][MAXU] 4 位元（哪幾個 query 看得到）、ucount [組]。query 自己那段尾巴（不滿 4 個）也算一個區塊，token 是否可見
// 另以 token 位置 ≤ query 位置判斷。grid：組數 × 1024
#define MAXW 1024   // 位元圖字數：最多 32768 個區塊（context 131072）
#define MAXU 2064
kernel void qsa_union4(device const uint *sel_all [[buffer(0)]], device const uint *nsel_all [[buffer(1)]],
                       constant qsa_pf_args *aa [[buffer(2)]], device uint *ulist [[buffer(3)]], device uchar *umask [[buffer(4)]],
                       device uint *ucount [[buffer(5)]], constant uint &T [[buffer(6)]], uint grp [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint bm[4][MAXW];
    threadgroup uint wsum[32];
    for (uint q = 0; q < 4; q++) atomic_store_explicit(&bm[q][tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nbm = 0; // 最大的區塊編號 + 1
    for (uint q = 0; q < 4; q++) {
        const uint t = grp * 4 + q;
        if (t >= T) break;
        constant qsa_pf_args &a = aa[t];
        const uint tail = (a.pos + 1) % CR ? a.nvis : 0xffffffffu; // 尾巴所在的區塊
        nbm = max(nbm, a.nvis + 1);
        if (a.sparse) {
            const uint ns = nsel_all[t];
            for (uint i = tid; i < ns; i += 1024) {
                const uint b = sel_all[(ulong)t * SEL + i];
                atomic_fetch_or_explicit(&bm[q][b / 32], 1u << (b % 32), memory_order_relaxed);
            }
            if (tid == 0 && tail != 0xffffffffu) atomic_fetch_or_explicit(&bm[q][tail / 32], 1u << (tail % 32), memory_order_relaxed);
        } else {
            for (uint b = tid; b <= a.pos / CR; b += 1024) atomic_fetch_or_explicit(&bm[q][b / 32], 1u << (b % 32), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nw = (nbm + 31) / 32;
    uint w0 = 0, w1 = 0, w2 = 0, w3 = 0, cnt = 0;
    if (tid < nw) {
        w0 = atomic_load_explicit(&bm[0][tid], memory_order_relaxed);
        w1 = atomic_load_explicit(&bm[1][tid], memory_order_relaxed);
        w2 = atomic_load_explicit(&bm[2][tid], memory_order_relaxed);
        w3 = atomic_load_explicit(&bm[3][tid], memory_order_relaxed);
        cnt = popcount(w0 | w1 | w2 | w3);
    }
    const uint pre = simd_prefix_exclusive_sum(cnt);
    if (lane == 31) wsum[sg] = pre + cnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint v = wsum[lane];
        wsum[lane] = simd_prefix_exclusive_sum(v);
        if (lane == 31) ucount[grp] = wsum[lane] + v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint off = wsum[sg] + pre;
    uint comb = w0 | w1 | w2 | w3;
    while (comb) {
        const uint bit = ctz(comb);
        comb &= comb - 1;
        const uint b = tid * 32 + bit;
        if (off < MAXU) {
            ulist[(ulong)grp * MAXU + off] = b;
            umask[(ulong)grp * MAXU + off] = (uchar)(((w0 >> bit) & 1) | (((w1 >> bit) & 1) << 1) | (((w2 >> bit) & 1) << 2) | (((w3 >> bit) & 1) << 3));
        }
        off++;
    }
}

// qsa_flash4：每個 (kv head, 組) 一個 threadgroup（256 = 8 個 SIMD group）。4 個 query × 16 列（12 個 head + 4 列空白）= 64 列。
// 每次 16 個區塊（64 個 token）：S = Q·Kᵀ（SIMD group s 負責第 8s.. 個 token、4 個 query）→ 64 列 online softmax（不可見 = −∞）
// → acc += P·V（SIMD group s 負責第 32s.. 維）。輸出同 qsa_flash_pf。grid：(KVH, 組數) × 256
kernel void qsa_flash4(device const half *q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]],
                       device const uint *ulist [[buffer(3)]], device const uchar *umask [[buffer(4)]], device const uint *ucount [[buffer(5)]],
                       device half *out [[buffer(6)]], constant qsa_pf_args *aa [[buffer(7)]], constant uint &T [[buffer(8)]],
                       uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float Ssh[64 * 64];   // [列][token]
    threadgroup half P[64 * 64];      // [列][token]
    threadgroup int cid[64];
    threadgroup uchar cm[64];
    threadgroup float fac[64], lsum[64], mrun[64];
    const uint g = tg.x, grp = tg.y, t0 = grp * 4;
    const uint nu = min(ucount[grp], (uint)MAXU), nchunk = (nu + 15) / 16;
    if (tid < 64) { mrun[tid] = -INFINITY; lsum[tid] = 0.0f; }
    const ulong rs = KVH * HD;
    using T2 = dextents<int32_t, 2>;
    // S：M 16（一個 query 的 head）× N 8（token）× K 32
    constexpr auto dS = matmul2d_descriptor(16, 8, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> opS;
    // PV：M 16 × N 32（維度）× K 32（token）
    constexpr auto dO = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dO, execution_simdgroup> opO;
    tensor<threadgroup half, T2, tensor_inline> tP(P, T2(64, 64));
    auto rv = opO.get_right_input_cooperative_tensor<half, half, float>();
    auto p0s = tP.slice<32, 16>(0, 0);
    auto acc0 = opO.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(p0s)>,
                                                       __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
    auto acc1 = opO.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(p0s)>,
                                                       __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
    auto acc2 = opO.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(p0s)>,
                                                       __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
    auto acc3 = opO.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(p0s)>,
                                                       __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) { acc0[i] = 0; acc1[i] = 0; acc2[i] = 0; acc3[i] = 0; }
    const uint d0 = sg * 32;
    for (uint ch = 0; ch < nchunk; ++ch) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64) {
            const uint u = ch * 16 + tid / 4;
            if (u < nu) { cid[tid] = (int)(ulist[(ulong)grp * MAXU + u] * CR + tid % 4); cm[tid] = umask[(ulong)grp * MAXU + u]; }
            else { cid[tid] = -1; cm[tid] = 0; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- S：這個 SIMD group 的 8 個 token，4 個 query（K 每段只讀一次，4 個 query 共用）
        {
            auto rk = opS.get_right_input_cooperative_tensor<half, half, float>();
            const uint tq0 = min(t0, T - 1), tq1 = min(t0 + 1, T - 1), tq2 = min(t0 + 2, T - 1), tq3 = min(t0 + 3, T - 1);
            tensor<device half, T2, tensor_inline> tQ0((device half *)q + ((ulong)tq0 * QH + g * GQA) * HD, T2(HD, GQA));
            tensor<device half, T2, tensor_inline> tQ1((device half *)q + ((ulong)tq1 * QH + g * GQA) * HD, T2(HD, GQA));
            tensor<device half, T2, tensor_inline> tQ2((device half *)q + ((ulong)tq2 * QH + g * GQA) * HD, T2(HD, GQA));
            tensor<device half, T2, tensor_inline> tQ3((device half *)q + ((ulong)tq3 * QH + g * GQA) * HD, T2(HD, GQA));
            auto qa0 = tQ0.slice<32, 16>(0, 0);
            auto s0 = opS.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(qa0)>,
                                                             __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
            auto s1 = opS.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(qa0)>,
                                                             __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
            auto s2 = opS.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(qa0)>,
                                                             __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
            auto s3 = opS.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(qa0)>,
                                                             __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
            for (uint16_t i = 0; i < s0.get_capacity(); ++i) { s0[i] = 0; s1[i] = 0; s2[i] = 0; s3[i] = 0; }
            for (uint k0 = 0; k0 < HD; k0 += 32) {
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < rk.get_capacity(); i += 4) {
                    const auto ix = rk.get_multidimensional_index(i); // (k, token)
                    const int id = cid[sg * 8 + ix[1]];
                    const half4 w = id >= 0 ? *(device const half4 *)(K + (ulong)id * rs + g * HD + k0 + ix[0]) : half4(0.0h);
                    rk[i] = w.x; rk[i + 1] = w.y; rk[i + 2] = w.z; rk[i + 3] = w.w;
                }
                auto a0 = tQ0.slice<32, 16>(k0, 0);
                auto a1 = tQ1.slice<32, 16>(k0, 0);
                auto a2 = tQ2.slice<32, 16>(k0, 0);
                auto a3 = tQ3.slice<32, 16>(k0, 0);
                opS.run(a0, rk, s0);
                opS.run(a1, rk, s1);
                opS.run(a2, rk, s2);
                opS.run(a3, rk, s3);
            }
            for (uint16_t i = 0; i < s0.get_capacity(); ++i)
                if (s0.is_valid_element(i)) {
                    const auto ix = s0.get_multidimensional_index(i); // (token, 列)
                    const uint c = sg * 8 + ix[0], r = ix[1];
                    Ssh[r * 64 + c] = s0[i];
                    Ssh[(16 + r) * 64 + c] = s1[i];
                    Ssh[(32 + r) * 64 + c] = s2[i];
                    Ssh[(48 + r) * 64 + c] = s3[i];
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- softmax：每列 4 個執行緒（各 16 個 token），用 shuffle 合併
        {
            const uint row = tid / 4, part = tid % 4, qq = row / 16, h = row % 16, t = t0 + qq;
            const bool live = h < GQA && t < T;
            const int pos = live ? (int)aa[t].pos : -1;
            const uint bit = 1u << qq;
            float m = mrun[row];
            for (uint c = part * 16; c < part * 16 + 16; ++c)
                if (live && (cm[c] & bit) && cid[c] >= 0 && cid[c] <= pos) m = max(m, Ssh[row * 64 + c] * 0.0625f);
            m = max(m, simd_shuffle_xor(m, 1));
            m = max(m, simd_shuffle_xor(m, 2));
            float l = 0;
            for (uint c = part * 16; c < part * 16 + 16; ++c) {
                const bool vis = live && (cm[c] & bit) && cid[c] >= 0 && cid[c] <= pos;
                const float ev = vis ? exp(Ssh[row * 64 + c] * 0.0625f - m) : 0.0f;
                l += ev;
                P[row * 64 + c] = half(ev);
            }
            l += simd_shuffle_xor(l, 1);
            l += simd_shuffle_xor(l, 2);
            threadgroup_barrier(mem_flags::mem_threadgroup); // 大家都讀完 mrun 才寫
            if (part == 0) {
                const float f = (m == -INFINITY) ? 1.0f : exp(mrun[row] - m);
                lsum[row] = lsum[row] * f + l;
                mrun[row] = m;
                fac[row] = f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- 縮放舊的 acc，再加 P·V（這個 SIMD group 的 32 維）
        for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
            const uint r = acc0.get_multidimensional_index(i)[1];
            acc0[i] *= fac[r]; acc1[i] *= fac[16 + r]; acc2[i] *= fac[32 + r]; acc3[i] *= fac[48 + r];
        }
        for (uint cc = 0; cc < 64; cc += 32) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rv.get_capacity(); ++i) {
                const auto ix = rv.get_multidimensional_index(i); // (token, 維)
                const int id = cid[cc + ix[0]];
                rv[i] = id >= 0 ? V[(ulong)id * rs + g * HD + d0 + ix[1]] : half(0.0h);
            }
            auto pa = tP.slice<32, 16>(cc, 0);
            auto pb = tP.slice<32, 16>(cc, 16);
            auto pc = tP.slice<32, 16>(cc, 32);
            auto pd = tP.slice<32, 16>(cc, 48);
            opO.run(pa, rv, acc0);
            opO.run(pb, rv, acc1);
            opO.run(pc, rv, acc2);
            opO.run(pd, rv, acc3);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        const auto ix = acc0.get_multidimensional_index(i); // (維, 列)
        const uint h = ix[1], d = d0 + ix[0];
        if (h >= GQA) continue;
        float v[4] = {acc0[i], acc1[i], acc2[i], acc3[i]};
        for (uint qq = 0; qq < 4; ++qq) {
            const uint t = t0 + qq;
            if (t < T) out[((ulong)t * QH + g * GQA + h) * HD + d] = half(v[qq] / lsum[qq * 16 + h]);
        }
    }
}

// ================================================================ qsa_flash_tr：S 依 token 分工、留在暫存器（prefill 預設）
// 要看的 token、輸出都和 qsa_flash_pf 一樣；每個 (kv head, query) 一個 threadgroup（TR_NSG 個 SIMD group），一次 TR_QCH 個 token：
//   SIMD group s 算第 s·NS.. 個 token 的完整 S[16 × NS]（256 維一次加完，不用 qsa_flash_pf 的 4 份部分和，也不寫進
//   threadgroup memory）→ 各 SIMD group 的列最大值交換一次 → 各自在暫存器算 exp、寫 P（half）、列和交換一次
//   → SIMD group s 負責 HD / TR_NSG 維的 tot = tot · f + P·V（V 用 half4 讀：不轉置的右運算元每個 lane 連續 4 個元素 = 連續 4 維）。
// S 的排列（M5，量過）：lane l 持有第 r、r + 8 列，r = (l >> 1 & 3) + 4·(l >> 4)；同一列在 lane l、l^1、l^8、l^9。
// tot（16 × DS）的列也一樣，所以 online softmax 的 m、l、縮放係數都放在暫存器。
// 每段兩個 barrier；threadgroup memory 只有 P、token 編號與列的交換區（約 5 KB，qsa_flash_pf 18.6 KB），一個 GPU core 可以放
// 更多 threadgroup。12K context：qsa_flash_pf 16.0 → 9.5 ms／層。gated = 1 時輸出直接乘上 gate（qg = q 投影的輸出）。
// grid：(KVH, T) × 32·TR_NSG
#ifndef TR_QCH
#define TR_QCH 128 // 量過：64 慢 4%、192 / 256 慢 30% 以上
#endif
#ifndef TR_NSG
#define TR_NSG 8   // 量過：4 個 SIMD group 慢 15%
#endif
static_assert((TR_NSG == 4 || TR_NSG == 8) && TR_QCH % (8 * TR_NSG) == 0 && TR_QCH % 32 == 0 && TR_QCH <= 256, "");
static_assert(GQA <= 16 && HD % (32 * TR_NSG) == 0, "一個 query 的 head 要放得進 16 列");
inline float tr_sigm(float x) { return 1.0f / (1.0f + exp(-x)); } // 同 engine.metal 的 sigm
kernel void qsa_flash_tr(device const half *q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]],
                         device const uint *sel_all [[buffer(3)]], device const uint *nsel_all [[buffer(4)]],
                         device half *out [[buffer(5)]], constant qsa_pf_args *aa [[buffer(6)]],
                         device const half *qg [[buffer(7)]], constant uint &gated [[buffer(8)]],
                         uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NTH = TR_NSG * 32, NS = TR_QCH / TR_NSG, DS = HD / TR_NSG; // 執行緒數、每個 SIMD group 的 token 數（S）與維度數（P·V）
    threadgroup half p[16 * TR_QCH];
    threadgroup int cid[2][TR_QCH]; // 這段、下一段（下一段在 softmax 時先算好）
    threadgroup float pm[TR_NSG][16], ps[TR_NSG][16];
    const uint g = tg.x, t = tg.y;
    constant qsa_pf_args &a = aa[t];
    device const uint *sel = sel_all + (ulong)t * SEL;
    const uint nsel = a.sparse ? nsel_all[t] : 0, ntok = a.ntok;
    const int nch = (int)((ntok + TR_QCH - 1) / TR_QCH);
    using T2 = dextents<int32_t, 2>;
    auto tok_of = [&](int c) -> int { return c < (int)ntok ? pf_tok((uint)c, sel, nsel, a) : -1; };

    constexpr auto dS = matmul2d_descriptor(16, NS, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dO = matmul2d_descriptor(16, DS, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> opS;
    matmul2d<dO, execution_simdgroup> opO;
    tensor<device half, T2, tensor_inline> tQ((device half *)q + ((ulong)t * QH + g * GQA) * HD, T2(HD, GQA));
    tensor<threadgroup half, T2, tensor_inline> tP(p, T2(TR_QCH, 16));
    auto rv = opO.template get_right_input_cooperative_tensor<half, half, float>();
    auto rk = opS.template get_right_input_cooperative_tensor<half, half, float>();
    auto tPh = tP.template slice<32, 16>(0, 0);
    auto tA0 = tQ.template slice<32, 16>(0, 0);
    auto tot = opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>,
                                                               __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
    for (uint16_t i = 0; i < tot.get_capacity(); ++i) tot[i] = 0.0f;
    const int d0 = sg * DS;
    const ulong rs = KVH * HD; // 相鄰 token 的列距
    device const half *Kg = K + g * HD, *Vg = V + g * HD + d0;
    const uint r0 = ((lane >> 1) & 3) + 4 * (lane >> 4); // 這個 lane 的列：r0、r0 + 8
    const bool wr = (lane & 9) == 0;                     // 同一列 4 個 lane 中負責寫交換區的
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    for (int j = tid; j < TR_QCH; j += NTH) cid[0][j] = tok_of(j);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ch = 0; ch < nch; ++ch) {
        const int c0 = ch * TR_QCH, nc = min(TR_QCH, (int)ntok - c0);
        threadgroup const int *cc_ = cid[ch & 1];
        // S（這個 SIMD group 的 NS 個 token，256 維）
        auto s = opS.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>,
                                                                 __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
        for (uint16_t i = 0; i < s.get_capacity(); ++i) s[i] = 0.0f;
        for (int k0 = 0; k0 < HD; k0 += 32) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rk.get_capacity(); i += 4) {
                const auto ix = rk.get_multidimensional_index(i); // (維, token)
                const int id = cc_[sg * NS + ix[1]];
                const half4 w = id >= 0 ? *(device const half4 *)(Kg + (ulong)id * rs + k0 + ix[0]) : half4(0.0h);
                rk[i] = w.x; rk[i + 1] = w.y; rk[i + 2] = w.z; rk[i + 3] = w.w;
            }
            auto tA = tQ.template slice<32, 16>(k0, 0);
            opS.run(tA, rk, s);
        }
        // 這個 SIMD group 的列最大值（scale = 1/16，精確；超出範圍或 head ≥ GQA 的列 = −∞）
        float x0 = -INFINITY, x1 = -INFINITY;
        for (uint16_t i = 0; i < s.get_capacity(); ++i) {
            const auto ix = s.get_multidimensional_index(i); // (token, 列)
            const bool vis = s.is_valid_element(i) && ix[1] < GQA && sg * NS + ix[0] < nc;
            const float v = vis ? s[i] * 0.0625f : -INFINITY;
            s[i] = v;
            if (ix[1] < 8) x0 = max(x0, v);
            else x1 = max(x1, v);
        }
        x0 = max(x0, simd_shuffle_xor(x0, 1));
        x0 = max(x0, simd_shuffle_xor(x0, 8));
        x1 = max(x1, simd_shuffle_xor(x1, 1));
        x1 = max(x1, simd_shuffle_xor(x1, 8));
        if (wr) { pm[sg][r0] = x0; pm[sg][r0 + 8] = x1; }
        threadgroup_barrier(mem_flags::mem_threadgroup); // (A) 列最大值寫完；上一段的 P·V 也都讀完 p、列和
        if (ch + 1 < nch)
            for (int j = tid; j < TR_QCH; j += NTH) cid[(ch + 1) & 1][j] = tok_of(c0 + TR_QCH + j);
        float n0 = m0, n1 = m1;
        for (int k = 0; k < TR_NSG; ++k) { n0 = max(n0, pm[k][r0]); n1 = max(n1, pm[k][r0 + 8]); }
        const float f0 = n0 == -INFINITY ? 1.0f : exp(m0 - n0), f1 = n1 == -INFINITY ? 1.0f : exp(m1 - n1);
        float y0 = 0.0f, y1 = 0.0f;
        for (uint16_t i = 0; i < s.get_capacity(); ++i) {
            const auto ix = s.get_multidimensional_index(i);
            const bool lo = ix[1] < 8;
            const float ev = s[i] == -INFINITY ? 0.0f : exp(s[i] - (lo ? n0 : n1));
            if (lo) y0 += ev;
            else y1 += ev;
            if (s.is_valid_element(i)) p[ix[1] * TR_QCH + sg * NS + ix[0]] = half(ev);
        }
        y0 += simd_shuffle_xor(y0, 1);
        y0 += simd_shuffle_xor(y0, 8);
        y1 += simd_shuffle_xor(y1, 1);
        y1 += simd_shuffle_xor(y1, 8);
        if (wr) { ps[sg][r0] = y0; ps[sg][r0 + 8] = y1; }
        m0 = n0;
        m1 = n1;
        threadgroup_barrier(mem_flags::mem_threadgroup); // (B) p、列和、下一段的 cid 寫完；列最大值讀完
        float z0 = 0.0f, z1 = 0.0f;
        for (int k = 0; k < TR_NSG; ++k) { z0 += ps[k][r0]; z1 += ps[k][r0 + 8]; }
        l0 = l0 * f0 + z0;
        l1 = l1 * f1 + z1;
        // tot = tot · f + P · V（這個 SIMD group 的 DS 維；tot 的列和 S 的一樣是 r0、r0 + 8）
        for (uint16_t i = 0; i < tot.get_capacity(); ++i) tot[i] *= tot.get_multidimensional_index(i)[1] < 8 ? f0 : f1;
        for (int cc = 0; cc < TR_QCH; cc += 32) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rv.get_capacity(); i += 4) {
                const auto ix = rv.get_multidimensional_index(i); // (維, token)
                const int id = cc_[cc + ix[1]];
                const half4 w = id >= 0 ? *(device const half4 *)(Vg + (ulong)id * rs + ix[0]) : half4(0.0h);
                rv[i] = w.x; rv[i + 1] = w.y; rv[i + 2] = w.z; rv[i + 3] = w.w;
            }
            tPh = tP.template slice<32, 16>(cc, 0);
            opO.run(tPh, rv, tot);
        }
    }
    for (uint16_t i = 0; i < tot.get_capacity(); ++i)
        if (tot.is_valid_element(i)) {
            const auto ix = tot.get_multidimensional_index(i); // (維, 列)
            const int h = ix[1];
            if (h >= GQA) continue;
            const half o = half(tot[i] / (h < 8 ? l0 : l1));
            const uint hd = (g * GQA + h) * HD + d0 + ix[0];
            // gated：順便乘上 gate（同 kb_attn_gate：q 投影每個 head 後半的 HD 個值），省掉一個 kernel
            out[(ulong)t * QH * HD + hd] = gated ? half(float(o) * tr_sigm(float(qg[(ulong)t * 2 * QH * HD + hd + (g * GQA + h) * HD + HD]))) : o;
        }
}

// ================================================================ INT8 Q·Kᵀ（QW_INT8 類別清單含 "qk" 才用，會改變精度）
// qk8_quant：每列 HD 個 half 對稱量化成 INT8（s = max|x| / 127，全 0 時 1）。q（[T][QH][HD]）、K cache（[位置][KVH][HD]）共用。
// 每列一個 SIMD group（threadgroup 256 = 8 列）。grid：ceil(n / 8) × 256
static_assert(HD == 256, "qk8_quant：32 lane × 8 個");
kernel void qk8_quant(device const half *x [[buffer(0)]], device int8_t *xq [[buffer(1)]], device float *xs [[buffer(2)]],
                      constant uint &n [[buffer(3)]], uint tgi [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tgi * 8 + sg;
    if (r >= n) return;
    device const half4 *xr = (device const half4 *)(x + (ulong)r * HD);
    const float4 v0 = float4(xr[lane * 2]), v1 = float4(xr[lane * 2 + 1]);
    const float4 a = max(fabs(v0), fabs(v1));
    const float m = simd_max(max(max(a.x, a.y), max(a.z, a.w)));
    const float sc = m > 0 ? m / 127.0f : 1.0f, inv = 1.0f / sc;
    device char4 *o = (device char4 *)(xq + (ulong)r * HD);
    o[lane * 2] = char4(clamp(rint(v0 * inv), -127.0f, 127.0f));
    o[lane * 2 + 1] = char4(clamp(rint(v1 * inv), -127.0f, 127.0f));
    if (lane == 0) xs[r] = sc;
}

// qsa_flash_tr8：同 qsa_flash_tr，只有 S = Q·Kᵀ 改成 INT8 × INT8 → INT32（tensor 單元）再乘 q 的 (query, head) scale 與
// K 的 (token, kv head) scale。q8 [T][QH][HD]、qs [T][QH]；K8 [位置][KVH][HD]、ks [位置][KVH]。P·V 仍是 FP16。
// S 的 INT32 累加器在 scale 之後直接存 float 的位元（as_type），排列不變，後面的 online softmax 與 qsa_flash_tr 相同。
// grid：(KVH, T) × 32·TR_NSG
#ifndef QK8_KC
#define QK8_KC 32 // S 的 K 方向一次幾維
#endif
kernel void qsa_flash_tr8(device const int8_t *q8 [[buffer(0)]], device const int8_t *K8 [[buffer(1)]], device const half *V [[buffer(2)]],
                          device const uint *sel_all [[buffer(3)]], device const uint *nsel_all [[buffer(4)]],
                          device half *out [[buffer(5)]], constant qsa_pf_args *aa [[buffer(6)]],
                          device const half *qg [[buffer(7)]], constant uint &gated [[buffer(8)]],
                          device const float *qs [[buffer(9)]], device const float *ks [[buffer(10)]],
                          uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NTH = TR_NSG * 32, NS = TR_QCH / TR_NSG, DS = HD / TR_NSG;
    threadgroup half p[16 * TR_QCH];
    threadgroup int cid[2][TR_QCH];
    threadgroup float csk[2][TR_QCH]; // 這段、下一段 token 的 K scale（不可見 = 0）
    threadgroup float pm[TR_NSG][16], ps[TR_NSG][16];
    const uint g = tg.x, t = tg.y;
    constant qsa_pf_args &a = aa[t];
    device const uint *sel = sel_all + (ulong)t * SEL;
    const uint nsel = a.sparse ? nsel_all[t] : 0, ntok = a.ntok;
    const int nch = (int)((ntok + TR_QCH - 1) / TR_QCH);
    using T2 = dextents<int32_t, 2>;
    auto tok_of = [&](int c) -> int { return c < (int)ntok ? pf_tok((uint)c, sel, nsel, a) : -1; };

    constexpr auto dS = matmul2d_descriptor(16, NS, QK8_KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dO = matmul2d_descriptor(16, DS, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> opS;
    matmul2d<dO, execution_simdgroup> opO;
    tensor<device int8_t, T2, tensor_inline> tQ((device int8_t *)q8 + ((ulong)t * QH + g * GQA) * HD, T2(HD, GQA));
    tensor<threadgroup half, T2, tensor_inline> tP(p, T2(TR_QCH, 16));
    auto rv = opO.template get_right_input_cooperative_tensor<half, half, float>();
    auto rk = opS.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto tPh = tP.template slice<32, 16>(0, 0);
    auto tA0 = tQ.template slice<QK8_KC, 16>(0, 0);
    auto tot = opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>,
                                                               __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
    for (uint16_t i = 0; i < tot.get_capacity(); ++i) tot[i] = 0.0f;
    const int d0 = sg * DS;
    const ulong rs = KVH * HD;
    device const int8_t *Kg = K8 + g * HD;
    device const half *Vg = V + g * HD + d0;
    const uint r0 = ((lane >> 1) & 3) + 4 * (lane >> 4);
    const bool wr = (lane & 9) == 0;
    // 這個 lane 兩列的 q scale（列 ≥ GQA 不可見，scale 不用）
    device const float *qsr = qs + (ulong)t * QH + g * GQA;
    const float q0s = r0 < GQA ? qsr[r0] * 0.0625f : 0.0f, q1s = r0 + 8 < GQA ? qsr[r0 + 8] * 0.0625f : 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    for (int j = tid; j < TR_QCH; j += NTH) {
        const int id = tok_of(j);
        cid[0][j] = id;
        csk[0][j] = id >= 0 ? ks[(ulong)id * KVH + g] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ch = 0; ch < nch; ++ch) {
        const int c0 = ch * TR_QCH, nc = min(TR_QCH, (int)ntok - c0);
        threadgroup const int *cc_ = cid[ch & 1];
        threadgroup const float *ck_ = csk[ch & 1];
        auto s = opS.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>,
                                                                 __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, int32_t>();
        for (uint16_t i = 0; i < s.get_capacity(); ++i) s[i] = 0;
        for (int k0 = 0; k0 < HD; k0 += QK8_KC) {
#if QK8_K4 // 每個 lane 連續 4 個元素 = 同一個 token 的連續 4 維（與 half 的 K4 同樣的假設）
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rk.get_capacity(); i += 4) {
                const auto ix = rk.get_multidimensional_index(i);
                const int id = cc_[sg * NS + ix[1]];
                const char4 w = id >= 0 ? *(device const char4 *)(Kg + (ulong)id * rs + k0 + ix[0]) : char4(0);
                rk[i] = w.x; rk[i + 1] = w.y; rk[i + 2] = w.z; rk[i + 3] = w.w;
            }
#else
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rk.get_capacity(); ++i) {
                const auto ix = rk.get_multidimensional_index(i);
                const int id = cc_[sg * NS + ix[1]];
                rk[i] = id >= 0 ? Kg[(ulong)id * rs + k0 + ix[0]] : (int8_t)0;
            }
#endif
            auto tA = tQ.template slice<QK8_KC, 16>(k0, 0);
            opS.run(tA, rk, s);
        }
        float x0 = -INFINITY, x1 = -INFINITY;
        for (uint16_t i = 0; i < s.get_capacity(); ++i) {
            const auto ix = s.get_multidimensional_index(i); // (token, 列)
            const bool vis = s.is_valid_element(i) && ix[1] < GQA && sg * NS + ix[0] < nc;
            const float v = vis ? float(s[i]) * (ix[1] < 8 ? q0s : q1s) * ck_[sg * NS + ix[0]] : -INFINITY;
            s[i] = as_type<int32_t>(v);
            if (ix[1] < 8) x0 = max(x0, v);
            else x1 = max(x1, v);
        }
        x0 = max(x0, simd_shuffle_xor(x0, 1));
        x0 = max(x0, simd_shuffle_xor(x0, 8));
        x1 = max(x1, simd_shuffle_xor(x1, 1));
        x1 = max(x1, simd_shuffle_xor(x1, 8));
        if (wr) { pm[sg][r0] = x0; pm[sg][r0 + 8] = x1; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ch + 1 < nch)
            for (int j = tid; j < TR_QCH; j += NTH) {
                const int id = tok_of(c0 + TR_QCH + j);
                cid[(ch + 1) & 1][j] = id;
                csk[(ch + 1) & 1][j] = id >= 0 ? ks[(ulong)id * KVH + g] : 0.0f;
            }
        float n0 = m0, n1 = m1;
        for (int k = 0; k < TR_NSG; ++k) { n0 = max(n0, pm[k][r0]); n1 = max(n1, pm[k][r0 + 8]); }
        const float f0 = n0 == -INFINITY ? 1.0f : exp(m0 - n0), f1 = n1 == -INFINITY ? 1.0f : exp(m1 - n1);
        float y0 = 0.0f, y1 = 0.0f;
        for (uint16_t i = 0; i < s.get_capacity(); ++i) {
            const auto ix = s.get_multidimensional_index(i);
            const bool lo = ix[1] < 8;
            const float sv = as_type<float>(s[i]);
            const float ev = sv == -INFINITY ? 0.0f : exp(sv - (lo ? n0 : n1));
            if (lo) y0 += ev;
            else y1 += ev;
            if (s.is_valid_element(i)) p[ix[1] * TR_QCH + sg * NS + ix[0]] = half(ev);
        }
        y0 += simd_shuffle_xor(y0, 1);
        y0 += simd_shuffle_xor(y0, 8);
        y1 += simd_shuffle_xor(y1, 1);
        y1 += simd_shuffle_xor(y1, 8);
        if (wr) { ps[sg][r0] = y0; ps[sg][r0 + 8] = y1; }
        m0 = n0;
        m1 = n1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float z0 = 0.0f, z1 = 0.0f;
        for (int k = 0; k < TR_NSG; ++k) { z0 += ps[k][r0]; z1 += ps[k][r0 + 8]; }
        l0 = l0 * f0 + z0;
        l1 = l1 * f1 + z1;
        for (uint16_t i = 0; i < tot.get_capacity(); ++i) tot[i] *= tot.get_multidimensional_index(i)[1] < 8 ? f0 : f1;
        for (int cc = 0; cc < TR_QCH; cc += 32) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rv.get_capacity(); i += 4) {
                const auto ix = rv.get_multidimensional_index(i);
                const int id = cc_[cc + ix[1]];
                const half4 w = id >= 0 ? *(device const half4 *)(Vg + (ulong)id * rs + ix[0]) : half4(0.0h);
                rv[i] = w.x; rv[i + 1] = w.y; rv[i + 2] = w.z; rv[i + 3] = w.w;
            }
            tPh = tP.template slice<32, 16>(cc, 0);
            opO.run(tPh, rv, tot);
        }
    }
    for (uint16_t i = 0; i < tot.get_capacity(); ++i)
        if (tot.is_valid_element(i)) {
            const auto ix = tot.get_multidimensional_index(i);
            const int h = ix[1];
            if (h >= GQA) continue;
            const half o = half(tot[i] / (h < 8 ? l0 : l1));
            const uint hd = (g * GQA + h) * HD + d0 + ix[0];
            out[(ulong)t * QH * HD + hd] = gated ? half(float(o) * tr_sigm(float(qg[(ulong)t * 2 * QH * HD + hd + (g * GQA + h) * HD + HD]))) : o;
        }
}
