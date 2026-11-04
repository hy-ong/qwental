// GGUF profile 的 MoE（decode 與驗證）：routed expert、shared expert 是 ggml 區塊，router 是 BF16。
// 前面接 iq_tables.metal 與 moe_iq.metal（IQ_DQ_ONLY：ldh）。
// 主程式注入：E、TOPK、D、I、NB（最多幾個 token：decode 1、驗證 8；實際個數 nbr 在執行時給），以及下面其中一組：
//   GG_ROUTER     ：BF16 router（moe_router_bf、p_router_bf、p_router_bf_refine；這組的 NB 就是 token 數）
//   GG_PART == 0  ：查表初始化與選 expert（gg_tab_init、moe_sel_gg）
//   GG_PART == 1  ：前段 moe_pre_gg（FSG、FSU：shared gate、up 的格式；RBF：router 是 BF16）
//   GG_PART == 2  ：moe_gu_gg（FGU：routed gate/up 的格式；FSD：shared down 的格式）
//   GG_PART == 3  ：moe_dn_gg、moe_cmb_gg（FD：routed down 的格式；DSGN）
// 格式是 ggml type，0 = MLX 8-bit affine（只用在 shared）。
//
// 每個 lane 一次解 16 個值（一個「單位」）：w（half4 × 4，未乘 scale 的整數值，half 精確）、d、m，值 = d·w − m。
// 驗證時每個不重複的 expert 只讀、只解一次，再跟選到它的每個 token 相乘；每個 token 的算式與加總順序都和 decode 相同，
// 所以推測解碼與一般解碼的結果逐位元一致。

#ifdef GG_ROUTER
// ---------------------------------------------------------------- BF16 router（FP32 累加）。decode、驗證、prefill 第二階段用同一個算式：
// lane 負責第 lane、lane + 32… 個 bfloat4，最後 simd_sum（三者的 logit 逐位元相同）
inline float bf_rowdot(device const bfloat4 *w, device const half4 *x, uint lane) {
    float acc = 0;
    for (uint i = lane; i < D / 4; i += 32) acc += dot(float4(w[i]), float4(x[i]));
    return acc;
}

// decode / 驗證：logits [NB][E + 1]；第 E 列是 shared_expert_gate（BF16 [D]）。grid：(E + 1 + 7) / 8 × 256
kernel void moe_router_bf(device const bfloat *Wr [[buffer(0)]], device const bfloat *Wsg [[buffer(1)]], device const half *x [[buffer(2)]],
                          device float *logits [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg * 8 + sg;
    if (r > E) return;
    device const bfloat4 *w = (device const bfloat4 *)(r < E ? Wr + (ulong)r * D : Wsg);
    float acc[NB];
    for (uint t = 0; t < NB; t++) acc[t] = 0;
    for (uint i = lane; i < D / 4; i += 32) { // 權重只讀一次，各 token 的加總順序同 bf_rowdot
        const float4 wv = float4(w[i]);
        for (uint t = 0; t < NB; t++) acc[t] += dot(wv, float4(((device const half4 *)(x + (ulong)t * D))[i]));
    }
    for (uint t = 0; t < NB; t++) {
        const float v = simd_sum(acc[t]);
        if (lane == 0) logits[(ulong)t * (E + 1) + r] = v;
    }
}

// prefill 精確版：y [T][E]。每個 SIMD group 一個 expert、8 個 token。grid：(E / 8, ceil(T / 8)) × 256
kernel void p_router_bf(device const bfloat *W [[buffer(0)]], device const half *x [[buffer(1)]], device float *y [[buffer(2)]],
                        constant uint &T [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TT = 8;
    const uint r = tg.x * 8 + sg, t0 = tg.y * TT;
    device const bfloat4 *w = (device const bfloat4 *)(W + (ulong)r * D);
    float acc[TT];
    for (uint j = 0; j < TT; j++) acc[j] = 0;
    for (uint i = lane; i < D / 4; i += 32) {
        const float4 wv = float4(w[i]);
        for (uint j = 0; j < TT; j++)
            if (t0 + j < T) acc[j] += dot(wv, float4(((device const half4 *)(x + (ulong)(t0 + j) * D))[i]));
    }
    for (uint j = 0; j < TT; j++) {
        const float v = simd_sum(acc[j]);
        if (lane == 0 && t0 + j < T) y[(ulong)(t0 + j) * E + r] = v;
    }
}

// prefill 兩階段的第二步（同 prefill.metal 的 p_router_refine）：近似 logits 的前 RC 名用精確算式重算，其餘設 −∞。grid：T × 32
kernel void p_router_bf_refine(device const bfloat *W [[buffer(0)]], device const half *x [[buffer(1)]], device float *y [[buffer(2)]],
                               device const float *approx [[buffer(3)]], uint t [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint RC = 16, PER = E / 32;
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
    device const half4 *xt = (device const half4 *)(x + (ulong)t * D);
    for (uint k = 0; k < RC; k++) {
        const uint r = cand[k];
        const float val = simd_sum(bf_rowdot((device const bfloat4 *)(W + (ulong)r * D), xt, lane));
        if (lane == 0) y[(ulong)t * E + r] = val;
    }
}

#else
// 查表（gg_tab_init 在 GPU 上由 iq_tables.metal 產生，綁成 constant buffer，各 kernel 共用）：值都是小整數，half 精確
#define TAB_IQ2S 0       // [1024] half8：IQ2_S grid
#define TAB_IQ3XXSS 16384 // [4096] half4：IQ3_XXS grid（低 8 位元）連同 4 個符號（第 8–11 位元）→ 已帶符號的 4 個值
#define TAB_IQ3SS 49152   // [8192] half4：IQ3_S grid（低 9 位元）連同 4 個符號（第 9–12 位元）→ 已帶符號的 4 個值
#define TAB_IQ4 114688    // [256] half2：一個 byte 的低、高 nibble → kvalues_iq4nl
#define TAB_Q20 115712    // [256] half4：一個 byte 的 4 個 2-bit → q − 1
#define TAB_BYTES 117760
// 查表次數是解碼的主要成本：IQ3 的 grid 和符號合成一張表（一次查表取代 grid、sign 兩次，表 32 / 64 KB，量過比算符號快）；
// IQ2_S 的 grid 一次 8 個值，符號用乘法算（合成表要 256 KB，量過慢很多）

#define GNB 8 // gg_sel 的版面固定以 8 個 token 配置（與 NB 無關）
#define GUMAX (GNB * TOPK)

// 選 expert 的結果（moe_sel_gg 寫、gate/up 與 down 讀）
#ifndef TABQ
#define TABQ constant // 查表走 constant 位址空間（常數快取；prefill 比 device 快約 4%）
#endif
#define TABP TABQ const uchar *
struct gg_sel {
    uint U;                 // 不重複的 expert 數
    uint ue[GUMAX];         // 第 u 個不重複 expert 的編號（依第一次出現的順序：token 0 的第 0…9 名、token 1 新出現的…）
    uint mask[GUMAX];       // 哪些 token 選了它（bit t）
    uchar slot[GUMAX][GNB]; // token t 的第幾名（mask 對應位元為 1 時才有效）
    float w[GNB][TOPK + 1]; // 各 token 的 routed 權重（softmax）與 shared gate（sigmoid）
};

// 選 expert（NB 個 token）：SIMD group sg < NB 選 token sg 的前 TOPK（同 moe_decode.metal 的 select_topk：同值取編號最小的），
// 再去重（nbr ≤ NB 是這次的 token 數）。threadgroup 至少 32·nbr 個執行緒；ids、first、lead 是呼叫端的 threadgroup 陣列（[NB·TOPK]、[E]、[NB·TOPK]）
// 結果（ue、mask、slot、U、w）與逐一掃描的寫法完全相同，只是：每輪 lane 內的最大值分兩半找（深度減半）、跨 lane 同值取最小 lane
// 用 ballot（不必再做一次 simd_min）、softmax 由 10 個 lane 同時算 exp（加總仍依名次順序）、去重用前綴和與 threadgroup atomic（不必每個
// expert 逐一掃描）
template <typename LP>
inline void gg_select(LP logits, device gg_sel *ms, uint nbr, uint sg, uint lane, uint tid, uint nth, threadgroup uint *ids,
                      threadgroup atomic_uint *first, threadgroup uint *lead) {
    const uint UM = nbr * TOPK;
    for (uint i = tid; i < E; i += nth) atomic_store_explicit(&first[i], 0xffffffffu, memory_order_relaxed);
    for (uint i = tid; i < GUMAX * GNB / 4; i += nth) ((device uint *)ms->slot)[i] = 0;
    if (sg < nbr) {
        LP lg = logits + (ulong)sg * (E + 1);
        constexpr uint PER = E / 32, HP = PER / 2;
        static_assert(PER % 2 == 0, "gg_select：每個 lane 的值數要是偶數");
        float v[PER];
        for (uint j = 0; j < PER; j++) v[j] = lg[lane * PER + j];
        float pk = 0; // lane k：第 k 名的 logit
        for (uint k = 0; k < TOPK; k++) {
            float b0 = v[0], b1 = v[HP];
            uint j0 = 0, j1 = HP;
            for (uint j = 1; j < HP; j++) {
                if (v[j] > b0) { b0 = v[j]; j0 = j; }
                if (v[HP + j] > b1) { b1 = v[HP + j]; j1 = HP + j; }
            }
            if (b1 > b0) { b0 = b1; j0 = j1; } // 同值取前半（編號較小）
            const float m = simd_max(b0);
            const uint L = ctz(uint(static_cast<simd_vote::vote_t>(simd_ballot(b0 == m)))); // 同值取最小的 lane（編號較小）
            const uint bj = simd_shuffle(j0, L);
            if (lane == 0) ids[sg * TOPK + k] = L * PER + bj;
            if (lane == k) pk = m;
            for (uint j = 0; j < PER; j++) v[j] = lane == L && j == bj ? -INFINITY : v[j];
        }
        const float mx = simd_shuffle(pk, 0), ex = exp(pk - mx);
        float sm = 0;
        for (uint k = 0; k < TOPK; k++) sm += simd_shuffle(ex, k); // 依名次順序加總
        if (lane < TOPK) ms->w[sg][lane] = ex / sm;
        if (lane == TOPK) ms->w[sg][TOPK] = 1.0f / (1.0f + exp(-lg[E]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < UM) atomic_fetch_min_explicit(&first[ids[tid]], tid, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint f = 0;
    if (tid < UM) {
        f = atomic_load_explicit(&first[ids[tid]], memory_order_relaxed); // 同一個 expert 第一次出現的位置
        lead[tid] = f == tid ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { // 第一次出現的位置 → 不重複 expert 的編號 u（前綴和；lead 改存 u + 1）
        uint carry = 0;
        for (uint c = 0; c < UM; c += 32) {
            const uint i = c + lane, fl = i < UM ? lead[i] : 0u;
            const uint pre = simd_prefix_exclusive_sum(fl) + carry;
            carry += simd_sum(fl);
            if (fl) lead[i] = pre + 1;
        }
        if (lane == 0) ms->U = carry;
    }
    for (uint i = tid; i < GUMAX; i += nth) atomic_store_explicit(&first[i], 0u, memory_order_relaxed); // 改當各 u 的 mask
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (tid < UM) {
        const uint u = lead[f] - 1, t = tid / TOPK;
        atomic_fetch_or_explicit(&first[u], 1u << t, memory_order_relaxed);
        ms->slot[u][t] = uchar(tid % TOPK);
        if (f == tid) ms->ue[u] = ids[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < UM && f == tid) {
        const uint u = lead[tid] - 1;
        ms->mask[u] = atomic_load_explicit(&first[u], memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- 16 個值一個單位的解碼
// 格式參數：每塊值數、位元組；SPL：一個單位的 16 個值是 [p, p + 8) ∪ [p + 16, p + 24)（IQ4 的高低 nibble），否則連續；MIN：有 min
constexpr uint gf_qb(int F) { return F == 22 || F == 18 || F == 21 || F == 23 || F == 12 || F == 13 || F == 14 ? 256 : F == 0 || F == 42 ? 64 : 32; }
constexpr uint gf_bb(int F) {
    return F == 0 ? 64 : F == 42 || F == 20 ? 18 : F == 22 ? 82 : F == 18 ? 98 : F == 21 ? 110 : F == 23 ? 136 : F == 12 ? 144 : F == 13 ? 176
         : F == 14 ? 210 : F == 8 ? 34 : 64; // 0 = MLX 8-bit affine（group 64）：值 = s·q + z（d = s、m = −z）；30 = BF16
}
constexpr bool gf_spl(int F) { return F == 20 || F == 23; }
constexpr bool gf_min(int F) { return F == 0 || F == 12 || F == 13; }
#define RBY(F, K) ((K) / gf_qb(F) * gf_bb(F)) // 每列位元組

// 一列權重：w 在 device 或 threadgroup memory（P 是指標型別）；MLX 格式另有每 64 個值一組的 scale / bias；tab 是查表
template <typename P> struct gref {
    P w;
    device const bfloat *s, *z;
    TABP tab;
};
typedef device const uchar *DP;
typedef threadgroup const uchar *TP;

// 2 位元組對齊的讀取（各格式的區塊都是偶數長度）
inline uint ldu16(DP p) { return uint(*(device const ushort *)p); }
inline uint ldu16(TP p) { return uint(*(threadgroup const ushort *)p); }
inline float ldf16(DP p) { return float(as_type<half>(ushort(ldu16(p)))); }
inline float ldf16(TP p) { return float(as_type<half>(ushort(ldu16(p)))); }

template <typename P> inline uint2 tg4(gref<P> r, uint off, uint i) { return ((TABQ const uint2 *)(r.tab + off))[i]; } // half4 的位元
template <typename P> inline uint4 tg8(gref<P> r, uint off, uint i) { return ((TABQ const uint4 *)(r.tab + off))[i]; } // half8 的位元
inline half4 hx(uint2 v) { return as_type<half4>(v); }
// 8 個 sign 位元 → 8 個 half 的符號位元遮罩（用乘法把每兩個位元搬到第 15、31 位元，不查表）
inline uint sgn2(uint s) { return ((s & 3) * 0x40008000u) & 0x80008000u; }
inline uint4 sgn8(uint s) { return uint4(sgn2(s), sgn2(s >> 2), sgn2(s >> 4), sgn2(s >> 6)); }

// Q4_K / Q5_K 的第 j 段 scale 與 min（同 moe_iq.metal 的 kq_scale_min）
template <typename P> inline void kq_sm(P sc, uint j, thread float &s, thread float &m) {
    if (j < 4) {
        s = float(sc[j] & 63);
        m = float(sc[j + 4] & 63);
    } else {
        s = float((sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4));
        m = float((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    }
}

// w[4]：16 個值（half4 × 4，對應 x 的 [p0, p0 + 8) 與 [p1, p1 + 8)），值 = d·w − m
template <int F> struct dqf;

template <> struct dqf<0> { // MLX 8-bit：tensor 只保證 8 位元組對齊（只在 device）
    static inline void run(gref<DP> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        device const uint2 *p = (device const uint2 *)(r.w + 16 * u);
        const uint2 a = p[0], b = p[1];
        w[0] = half4(as_type<uchar4>(a.x));
        w[1] = half4(as_type<uchar4>(a.y));
        w[2] = half4(as_type<uchar4>(b.x));
        w[3] = half4(as_type<uchar4>(b.y));
        d = float(r.s[u / 4]);
        m = -float(r.z[u / 4]);
    }
};

template <> struct dqf<42> { // Q2_0：值 = d·(q − 1)
    template <typename P> static inline void run(gref<P> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        P b = r.w + (u >> 2) * 18;
        d = ldf16(b);
        m = 0;
        const uint v0 = ldu16(b + 2 + 4 * (u & 3)), v1 = ldu16(b + 4 + 4 * (u & 3));
        w[0] = hx(tg4(r, TAB_Q20, v0 & 255));
        w[1] = hx(tg4(r, TAB_Q20, v0 >> 8));
        w[2] = hx(tg4(r, TAB_Q20, v1 & 255));
        w[3] = hx(tg4(r, TAB_Q20, v1 >> 8));
    }
};

// IQ4（NL / XS）的 8 個 byte：低 nibble → w[0..1]（x 的 [p0, p0 + 8)）、高 nibble → w[2..3]（[p0 + 16, p0 + 24)）
template <typename P> inline void iq4_8(gref<P> r, P q8, thread half4 *w) {
    half2 h[8];
    for (uint i = 0; i < 4; i++) {
        const uint v = ldu16(q8 + 2 * i);
        h[2 * i] = as_type<half2>(((TABQ const uint *)(r.tab + TAB_IQ4))[v & 255]);
        h[2 * i + 1] = as_type<half2>(((TABQ const uint *)(r.tab + TAB_IQ4))[v >> 8]);
    }
    w[0] = half4(h[0].x, h[1].x, h[2].x, h[3].x);
    w[1] = half4(h[4].x, h[5].x, h[6].x, h[7].x);
    w[2] = half4(h[0].y, h[1].y, h[2].y, h[3].y);
    w[3] = half4(h[4].y, h[5].y, h[6].y, h[7].y);
}

template <> struct dqf<20> { // IQ4_NL
    template <typename P> static inline void run(gref<P> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        P b = r.w + (u >> 1) * 18;
        d = ldf16(b);
        m = 0;
        iq4_8(r, b + 2 + 8 * (u & 1), w);
    }
};

template <> struct dqf<22> { // IQ2_S：scale 每 16 個值一個
    template <typename P> static inline void run(gref<P> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        P b = r.w + (u >> 4) * 82;
        const uint ib = (u >> 1) & 7, hh = u & 1;
        const uint qs = ldu16(b + 2 + 4 * ib + 2 * hh), sg = ldu16(b + 34 + 4 * ib + 2 * hh);
        const uint qh = b[66 + ib], sc = (b[74 + ib] >> (4 * hh)) & 15;
        d = ldf16(b) * (0.5f + float(sc)) * 0.25f;
        m = 0;
        const uint4 g0 = tg8(r, TAB_IQ2S, (qs & 255) | ((qh << (8 - 4 * hh)) & 0x300)) ^ sgn8(sg & 255);
        const uint4 g1 = tg8(r, TAB_IQ2S, (qs >> 8) | ((qh << (6 - 4 * hh)) & 0x300)) ^ sgn8(sg >> 8);
        w[0] = hx(g0.xy);
        w[1] = hx(g0.zw);
        w[2] = hx(g1.xy);
        w[3] = hx(g1.zw);
    }
};

template <> struct dqf<18> { // IQ3_XXS
    template <typename P> static inline void run(gref<P> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        P b = r.w + (u >> 4) * 98;
        const uint ib = (u >> 1) & 7, hh = u & 1;
        const uint q0 = ldu16(b + 2 + 8 * ib + 4 * hh), q1 = ldu16(b + 4 + 8 * ib + 4 * hh);
        const uint aux = ldu16(b + 66 + 4 * ib) | (ldu16(b + 68 + 4 * ib) << 16);
        d = ldf16(b) * (0.5f + float(aux >> 28)) * 0.5f;
        m = 0;
        const uint a0 = (aux >> (14 * hh)) & 127, a1 = (aux >> (14 * hh + 7)) & 127;
        const uint s0 = a0 | ((popcount(a0) & 1) << 7), s1 = a1 | ((popcount(a1) & 1) << 7); // = ksigns_iq2xs[a]
        w[0] = hx(tg4(r, TAB_IQ3XXSS, (q0 & 255) | ((s0 & 15) << 8)));
        w[1] = hx(tg4(r, TAB_IQ3XXSS, (q0 >> 8) | ((s0 >> 4) << 8)));
        w[2] = hx(tg4(r, TAB_IQ3XXSS, (q1 & 255) | ((s1 & 15) << 8)));
        w[3] = hx(tg4(r, TAB_IQ3XXSS, (q1 >> 8) | ((s1 >> 4) << 8)));
    }
};

template <> struct dqf<21> { // IQ3_S
    template <typename P> static inline void run(gref<P> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        P b = r.w + (u >> 4) * 110;
        const uint ib = (u >> 1) & 7, hh = u & 1;
        const uint q0 = ldu16(b + 2 + 8 * ib + 4 * hh), q1 = ldu16(b + 4 + 8 * ib + 4 * hh);
        const uint qh = uint(b[66 + ib]) >> (4 * hh), sg = ldu16(b + 74 + 4 * ib + 2 * hh); // 這個單位的 4 個 grid 用 qh 的第 4hh…4hh+3 位元
        d = ldf16(b) * float(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib & 1))) & 15));
        m = 0;
        w[0] = hx(tg4(r, TAB_IQ3SS, (q0 & 255) | ((qh << 8) & 256) | ((sg & 15) << 9)));
        w[1] = hx(tg4(r, TAB_IQ3SS, (q0 >> 8) | ((qh << 7) & 256) | (((sg >> 4) & 15) << 9)));
        w[2] = hx(tg4(r, TAB_IQ3SS, (q1 & 255) | ((qh << 6) & 256) | (((sg >> 8) & 15) << 9)));
        w[3] = hx(tg4(r, TAB_IQ3SS, (q1 >> 8) | ((qh << 5) & 256) | ((sg >> 12) << 9)));
    }
};

template <> struct dqf<23> { // IQ4_XS
    template <typename P> static inline void run(gref<P> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        P b = r.w + (u >> 4) * 136;
        const uint ib = (u >> 1) & 7, hh = u & 1;
        const uint sh = ldu16(b + 2);
        const int ls = int(((b[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4));
        d = ldf16(b) * float(ls - 32);
        m = 0;
        iq4_8(r, b + 8 + 16 * ib + 8 * hh, w);
    }
};

template <> struct dqf<12> { // Q4_K
    template <typename P> static inline void run(gref<P> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        P b = r.w + (u >> 4) * 144;
        const uint j = (u >> 1) & 7, hh = u & 1, sh = 4 * (j & 1);
        float sc, mn;
        kq_sm(b + 4, j, sc, mn);
        d = ldf16(b) * sc;
        m = ldf16(b + 2) * mn;
        P q = b + 16 + 32 * (j / 2) + 16 * hh;
        for (uint i = 0; i < 4; i++) {
            const uint v0 = ldu16(q + 4 * i), v1 = ldu16(q + 4 * i + 2);
            w[i] = half4(half((v0 >> sh) & 15), half((v0 >> (8 + sh)) & 15), half((v1 >> sh) & 15), half((v1 >> (8 + sh)) & 15));
        }
    }
};

template <> struct dqf<13> { // Q5_K
    template <typename P> static inline void run(gref<P> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        P b = r.w + (u >> 4) * 176;
        const uint j = (u >> 1) & 7, hh = u & 1, sh = 4 * (j & 1);
        float sc, mn;
        kq_sm(b + 4, j, sc, mn);
        d = ldf16(b) * sc;
        m = ldf16(b + 2) * mn;
        P q = b + 48 + 32 * (j / 2) + 16 * hh, h = b + 16 + 16 * hh;
        for (uint i = 0; i < 4; i++) {
            const uint v0 = ldu16(q + 4 * i), v1 = ldu16(q + 4 * i + 2), h0 = ldu16(h + 4 * i), h1 = ldu16(h + 4 * i + 2);
            w[i] = half4(half(((v0 >> sh) & 15) | (((h0 >> j) & 1) << 4)), half(((v0 >> (8 + sh)) & 15) | (((h0 >> (8 + j)) & 1) << 4)),
                         half(((v1 >> sh) & 15) | (((h1 >> j) & 1) << 4)), half(((v1 >> (8 + sh)) & 15) | (((h1 >> (8 + j)) & 1) << 4)));
        }
    }
};

template <> struct dqf<14> { // Q6_K：scale 每 16 個值一個
    template <typename P> static inline void run(gref<P> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        P b = r.w + (u >> 4) * 210;
        const uint s8 = (u >> 1) & 7, hh = u & 1, n = s8 / 4, rr = s8 % 4, lsh = 4 * (rr / 2), hsh = 2 * rr;
        P ql = b + 64 * n + 32 * (rr % 2) + 16 * hh, qh = b + 128 + 32 * n + 16 * hh;
        d = ldf16(b + 208) * float(int(uint(b[192 + 8 * n + hh + 2 * rr]) << 24) >> 24);
        m = 0;
        for (uint i = 0; i < 4; i++) {
            const uint a0 = ldu16(ql + 4 * i), c0 = ldu16(qh + 4 * i), a1 = ldu16(ql + 4 * i + 2), c1 = ldu16(qh + 4 * i + 2);
            w[i] = half4(half(int(((a0 >> lsh) & 15) | (((c0 >> hsh) & 3) << 4)) - 32), half(int(((a0 >> (8 + lsh)) & 15) | (((c0 >> (8 + hsh)) & 3) << 4)) - 32),
                         half(int(((a1 >> lsh) & 15) | (((c1 >> hsh) & 3) << 4)) - 32), half(int(((a1 >> (8 + lsh)) & 15) | (((c1 >> (8 + hsh)) & 3) << 4)) - 32));
        }
    }
};

template <> struct dqf<8> { // Q8_0
    template <typename P> static inline void run(gref<P> r, uint u, thread half4 *w, thread float &d, thread float &m) {
        P b = r.w + (u >> 1) * 34;
        d = ldf16(b);
        m = 0;
        P q = b + 2 + 16 * (u & 1);
        for (uint i = 0; i < 4; i++) {
            const uint v0 = ldu16(q + 4 * i), v1 = ldu16(q + 4 * i + 2);
            w[i] = half4(half(int(v0 << 24) >> 24), half(int(v0 << 16) >> 24), half(int(v1 << 24) >> 24), half(int(v1 << 16) >> 24));
        }
    }
};

// 一個單位與 x 的內積：d·Σ w·x − m·Σ x（decode 與驗證共用，加總順序固定）
template <int F, typename XT> inline float udot(thread const half4 *w, float d, float m, device const XT *x, uint u) {
    const uint p0 = gf_spl(F) ? 32 * (u >> 1) + 8 * (u & 1) : 16 * u, p1 = p0 + (gf_spl(F) ? 16 : 8);
    device const vec<XT, 4> *a = (device const vec<XT, 4> *)(x + p0), *c = (device const vec<XT, 4> *)(x + p1);
    const float4 a0 = float4(a[0]), a1 = float4(a[1]), c0 = float4(c[0]), c1 = float4(c[1]);
    float r = d * (dot(float4(w[0]), a0) + dot(float4(w[1]), a1) + dot(float4(w[2]), c0) + dot(float4(w[3]), c1));
    if (gf_min(F)) r -= m * dot(a0 + a1 + c0 + c1, float4(1));
    return r;
}

static_assert(D % 512 == 0 && I % 128 == 0, "moe_gg：gate/up 每個 lane 5 個單位、down 每列 8 個 lane");
#define NUG (D / 16 / 32) // gate/up：一列 32 個 lane，每個 lane 幾個單位
#define NUD (I / 16 / 8)  // down：一列 8 個 lane
#ifndef DSGN
#define DSGN 4            // down：每個 threadgroup 幾個 SIMD group（各 4 列、同一個 expert）
#endif

// gate、up 各 R 列（同一個 expert 相鄰的列，列距 rby 位元組；MLX 格式的 scale / bias 列距 D / 64），mask 裡的每個 token 各一個部分和（尚未 simd_sum）。
// 每個單位先解 R 列，再對每個 token 讀一次 x、乘這 R 列（x 的讀取少 R 倍）；每列、每個 token 的算式與加總順序都和一列一列算相同
template <int FA, int FB, uint R, typename P>
inline void gu_rows(gref<P> ra, gref<P> rb, ulong rby, device const half *x, uint mask, uint lane, thread float (*ag)[NB], thread float (*au)[NB]) {
    for (uint r = 0; r < R; r++)
        for (uint t = 0; t < NB; t++) ag[r][t] = au[r][t] = 0;
    for (uint k = 0; k < NUG; k++) {
        const uint u = lane + 32 * k;
        half4 wa[R][4], wb[R][4];
        float da[R], ma[R], db[R], mb[R];
        for (uint r = 0; r < R; r++) {
            dqf<FA>::run(gref<P>{ra.w + r * rby, ra.s + r * (D / 64), ra.z + r * (D / 64), ra.tab}, u, wa[r], da[r], ma[r]);
            dqf<FB>::run(gref<P>{rb.w + r * rby, rb.s + r * (D / 64), rb.z + r * (D / 64), rb.tab}, u, wb[r], db[r], mb[r]);
        }
        for (uint t = 0; t < NB; t++)
            if ((mask >> t) & 1)
                for (uint r = 0; r < R; r++) {
                    ag[r][t] += udot<FA>(wa[r], da[r], ma[r], x + (ulong)t * D, u);
                    au[r][t] += udot<FB>(wb[r], db[r], mb[r], x + (ulong)t * D, u);
                }
    }
}

// dn_rows 先解完這個 lane 的全部單位再乘（讀取一起送出，記憶體並行度高）：routed down（每個 SIMD group 只做一個工作）一律較快；
// shared down（在 moe_gu_gg 裡）NB = 8 時暫存器不夠反而慢。兩種寫法每個 token 的算式與加總順序相同
#ifndef DNALL
#define DNALL (GG_PART == 3 || NB <= 4)
#endif
// down R 列（相鄰，列距 rby；8 個 lane 分一列，lane 負責第 q、q + 8… 個單位），mask 裡的每個 token 各一個部分和。
// token t 的 H 是 H[t][sh ? TOPK : slot[u][t]]（起點先算好）；每個單位先解 R 列，再對每個 token 讀一次 H、乘這 R 列
template <int F, uint R, typename P>
inline void dn_rows(gref<P> g, ulong rby, device const float *H, device const gg_sel *ms, uint u, uint mask, bool sh, uint q, thread float (*acc)[NB]) {
    uint Ht[NB]; // token t 的 H 列在 H 裡的位置（32 位元，比指標省暫存器）
    for (uint t = 0; t < NB; t++) {
        for (uint r = 0; r < R; r++) acc[r][t] = 0;
        Ht[t] = (t * (TOPK + 1) + (sh ? TOPK : ((mask >> t) & 1 ? ms->slot[u][t] : 0))) * I;
    }
#if DNALL // 先解完這個 lane 的全部單位（讀取一起送出），再逐 token 乘：每個 token 的加總順序不變（單位 0、1…）
    half4 w[NUD][R][4];
    float d[NUD][R], m[NUD][R];
    for (uint k = 0; k < NUD; k++)
        for (uint r = 0; r < R; r++) dqf<F>::run(gref<P>{g.w + r * rby, g.s + r * (I / 64), g.z + r * (I / 64), g.tab}, q + 8 * k, w[k][r], d[k][r], m[k][r]);
    for (uint t = 0; t < NB; t++)
        if ((mask >> t) & 1)
            for (uint k = 0; k < NUD; k++)
                for (uint r = 0; r < R; r++) acc[r][t] += udot<F>(w[k][r], d[k][r], m[k][r], H + Ht[t], q + 8 * k);
#else
    for (uint k = 0; k < NUD; k++) {
        const uint un = q + 8 * k;
        half4 w[R][4];
        float d[R], m[R];
        for (uint r = 0; r < R; r++) dqf<F>::run(gref<P>{g.w + r * rby, g.s + r * (I / 64), g.z + r * (I / 64), g.tab}, un, w[r], d[r], m[r]);
        for (uint t = 0; t < NB; t++)
            if ((mask >> t) & 1)
                for (uint r = 0; r < R; r++) acc[r][t] += udot<F>(w[r], d[r], m[r], H + Ht[t], un);
    }
#endif
}

// 8 個 lane 的部分和加總（固定順序）
inline float sum8(float v) {
    v += simd_shuffle_xor(v, 1);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 4);
    return v;
}

#if GG_PART == 0
// 查表初始化（一次）：grid 8192 個執行緒
kernel void gg_tab_init(device uchar *tab [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < 8192) // 帶符號的值 = 把 grid 值的符號位元設起來（同舊版 grid ^ 符號遮罩，逐位元相同）
        for (uint j = 0; j < 4; j++) {
            const half g = half((iq3s_grid[i & 511] >> (8 * j)) & 255);
            ((device half *)(tab + TAB_IQ3SS))[4 * i + j] = (i >> (9 + j)) & 1 ? -g : g;
        }
    if (i < 4096)
        for (uint j = 0; j < 4; j++) {
            const half g = half((iq3xxs_grid[i & 255] >> (8 * j)) & 255);
            ((device half *)(tab + TAB_IQ3XXSS))[4 * i + j] = (i >> (8 + j)) & 1 ? -g : g;
        }
    if (i < 1024) {
        const ulong g = iq2s_grid[i];
        for (uint j = 0; j < 8; j++) ((device half *)(tab + TAB_IQ2S))[8 * i + j] = half((g >> (8 * j)) & 255);
    }
    if (i < 256) {
        ((device half *)(tab + TAB_IQ4))[2 * i] = half(kvalues_iq4nl[i & 15]);
        ((device half *)(tab + TAB_IQ4))[2 * i + 1] = half(kvalues_iq4nl[i >> 4]);
        for (uint j = 0; j < 4; j++) ((device half *)(tab + TAB_Q20))[4 * i + j] = half(int((i >> (2 * j)) & 3) - 1);
    }
}

// ---------------------------------------------------------------- 選 expert（router 不是 BF16 時另外送）：一個 threadgroup。grid：1 × 32·nbr
kernel void moe_sel_gg(device const float *logits [[buffer(0)]], device gg_sel *ms [[buffer(1)]], constant uint &nbr [[buffer(2)]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint ids[NB * TOPK], lead[NB * TOPK];
    threadgroup atomic_uint first[E];
    gg_select(logits, ms, nbr, sg, lane, tid, 32 * nbr, ids, first, lead);
}

#elif GG_PART == 1
// ---------------------------------------------------------------- 1. 前段：與選取無關的部分一次送
// threadgroup 0 … NRT − 1：BF16 router（每個 SIMD group 一列，第 E 列是 shared gate；RBF 為 0 時沒有這段，router 另外送）；
// 其餘：shared expert 的 gate/up（每個 SIMD group 一個 j）→ H[t][TOPK]。
// RBF 時最後完成的 router threadgroup 接著選 expert（router 的結果在同一個 dispatch 裡交接：coherent buffer + fence + 計數器），
// 和 shared gate/up 重疊。
// grid：(NRT + I / 8) × 256
#define NRT (RBF ? (E + 1 + 7) / 8 : 0)
kernel void moe_pre_gg(device const bfloat *Wr [[buffer(0)]], device const bfloat *Wsgl [[buffer(1)]], device const half *x [[buffer(2)]],
                       device coherent(device) float *logits [[buffer(3)]], device atomic_uint *cnt [[buffer(4)]], device gg_sel *ms [[buffer(5)]],
                       device const uchar *Wsg [[buffer(6)]], device const uchar *Wsu [[buffer(7)]], device float *H [[buffer(8)]],
                       device const bfloat *Ssg [[buffer(9)]], device const bfloat *Zsg [[buffer(10)]], device const bfloat *Ssu [[buffer(11)]],
                       device const bfloat *Zsu [[buffer(12)]], TABQ const uchar *tab [[buffer(13)]], constant uint &nbr [[buffer(14)]],
                       uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    if (tg < NRT) {
        const uint r = tg * 8 + sg;
        if (r <= E) {
            device const bfloat4 *w = (device const bfloat4 *)(r < E ? Wr + (ulong)r * D : Wsgl);
            float acc[NB];
            for (uint t = 0; t < NB; t++) acc[t] = 0;
            // 這一列的權重、每個 token 的 x 都先整批讀進暫存器（讀取一起送出；逐項讀會讓每次讀取的延遲串起來，router 只有 65 個
            // threadgroup，延遲藏不住），再依序乘：每個 token 的加總順序同 moe_router_bf（逐位元相同）
            bfloat4 wr[D / 128];
            for (uint k = 0; k < D / 128; k++) wr[k] = w[lane + 32 * k];
            for (uint t = 0; t < NB; t++)
                if (t < nbr) {
                    device const half4 *xt = (device const half4 *)(x + (ulong)t * D);
                    half4 xr[D / 128];
                    for (uint k = 0; k < D / 128; k++) xr[k] = xt[lane + 32 * k];
                    for (uint k = 0; k < D / 128; k++) acc[t] += dot(float4(wr[k]), float4(xr[k]));
                }
            for (uint t = 0; t < NB; t++) {
                const float v = simd_sum(acc[t]);
                if (lane == 0 && t < nbr) logits[(ulong)t * (E + 1) + r] = v;
            }
        }
    } else {
        const uint j = (tg - NRT) * 8 + sg;
        float ag[1][NB], au[1][NB];
        gu_rows<FSG, FSU, 1>(gref<DP>{Wsg + (ulong)j * RBY(FSG, D), Ssg + (ulong)j * (D / 64), Zsg + (ulong)j * (D / 64), tab},
                             gref<DP>{Wsu + (ulong)j * RBY(FSU, D), Ssu + (ulong)j * (D / 64), Zsu + (ulong)j * (D / 64), tab}, 0, x, (1u << nbr) - 1,
                             lane, ag, au);
        for (uint t = 0; t < NB; t++) {
            const float g = simd_sum(ag[0][t]), up = simd_sum(au[0][t]);
            if (lane == 0 && t < nbr) H[((ulong)t * (TOPK + 1) + TOPK) * I + j] = g / (1.0f + exp(-g)) * up;
        }
    }
#if RBF // 最後完成的 router threadgroup 選 expert（shared gate/up 的 threadgroup 同時還在跑）
    if (tg >= NRT) return;
    threadgroup bool last;
    threadgroup uint ids[NB * TOPK], lead[NB * TOPK];
    threadgroup atomic_uint first[E];
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        last = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed) == NRT - 1;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last) return;
    gg_select(logits, ms, nbr, sg, lane, tid, 256, ids, first, lead);
    if (tid == 0) atomic_store_explicit(cnt, 0u, memory_order_relaxed);
#endif
}

#elif GG_PART == 2
// ---------------------------------------------------------------- 2. routed gate/up + SiLU → H [NB][TOPK + 1][I]（float）；同時算 shared 的 down → Psh [NB][D]
// grid：(I / GSG, nbr·TOPK + 1) × 32·GSG。y < nbr·TOPK：第 y 個不重複的 expert，每個 SIMD group 一個 j；
// y = nbr·TOPK：shared down，每個 threadgroup 4·GSG 列（每個 SIMD group 4 列，一列 8 個 lane）
#ifndef GSG
#define GSG 8
#endif
static_assert(I % (GSG * RG) == 0 && D % (4 * GSG) == 0, "moe_gu_gg：grid 的列數");
kernel void moe_gu_gg(device const uchar *Wg [[buffer(0)]], device const uchar *Wu [[buffer(1)]], device const uchar *Wsd [[buffer(2)]],
                      device float *Psh [[buffer(3)]], device const half *x [[buffer(4)]], device const gg_sel *ms [[buffer(5)]],
                      device float *H [[buffer(6)]], device const bfloat *Ssd [[buffer(7)]], device const bfloat *Zsd [[buffer(8)]],
                      TABQ const uchar *tab [[buffer(11)]], constant uint &nbr [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint u = tg.y;
    if (u == nbr * TOPK) {
        if (tg.x >= D / (4 * GSG)) return;
        const uint row = tg.x * 4 * GSG + sg * 4 + lane / 8, q = lane % 8;
        float acc[1][NB];
        dn_rows<FSD, 1>(gref<DP>{Wsd + (ulong)row * RBY(FSD, I), Ssd + (ulong)row * (I / 64), Zsd + (ulong)row * (I / 64), tab}, 0, H, ms, 0,
                        (1u << nbr) - 1, true, q, acc);
        for (uint t = 0; t < NB; t++) {
            const float v = sum8(acc[0][t]);
            if (q == 0 && t < nbr) Psh[(ulong)t * D + row] = v;
        }
        return;
    }
    if (u >= ms->U || tg.x >= I / (GSG * RG)) return;
    const uint mask = ms->mask[u], j = (tg.x * GSG + sg) * RG;
    const ulong r = (ulong)ms->ue[u] * I + j;
    float ag[RG][NB], au[RG][NB];
    gu_rows<FGU, FGU, RG>(gref<DP>{Wg + r * RBY(FGU, D), Ssd, Zsd, tab}, gref<DP>{Wu + r * RBY(FGU, D), Ssd, Zsd, tab}, RBY(FGU, D), x, mask, lane, ag, au);
    for (uint t = 0; t < NB; t++)
        if ((mask >> t) & 1) {
            device float *h = H + ((ulong)t * (TOPK + 1) + ms->slot[u][t]) * I + j;
            for (uint rr = 0; rr < RG; rr++) {
                const float g = simd_sum(ag[rr][t]), up = simd_sum(au[rr][t]);
                if (lane == 0) h[rr] = g / (1.0f + exp(-g)) * up;
            }
        }
}

#elif GG_PART == 3
// ---------------------------------------------------------------- 3. routed down → Pg [NB][TOPK][D]（float），再由 moe_cmb_gg 加權合併
// 和 gate/up 一樣每個 threadgroup 只做一個 expert 的一段列：工作分得細、同時在跑的多（原本每個 threadgroup 負責幾列、
// 在裡面輪流做所有 expert，每個 SIMD group 依序做好幾個 expert，讀取等待無法重疊；驗證 8 個 token 時 down 慢約 10%）。
// grid：(D / (4·DSGN), nbr·TOPK) × 32·DSGN；SIMD group 一次 4 列（每列 8 個 lane）
static_assert(D % (4 * DSGN) == 0, "moe_dn_gg：列數");
kernel void moe_dn_gg(device const uchar *Wd [[buffer(0)]], device const float *H [[buffer(2)]], device const gg_sel *ms [[buffer(3)]],
                      TABQ const uchar *tab [[buffer(7)]], device float *Pg [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint u = tg.y;
    if (u >= ms->U) return;
    const uint q = lane % 8, row = (tg.x * DSGN + sg) * 4 + lane / 8, mask = ms->mask[u];
    float acc[1][NB];
    dn_rows<FD, 1>(gref<DP>{Wd + ((ulong)ms->ue[u] * D + row) * RBY(FD, I), nullptr, nullptr, tab}, RBY(FD, I), H, ms, u, mask, false, q, acc);
    for (uint t = 0; t < NB; t++)
        if ((mask >> t) & 1) {
            const float v = sum8(acc[0][t]);
            if (q == 0) Pg[((ulong)t * TOPK + ms->slot[u][t]) * D + row] = v;
        }
}

// 加權合併（依名次順序，再加 shared；同 moe_dn_gg 原本的收尾）。grid：D / 256 × 256
kernel void moe_cmb_gg(device const float *Psh [[buffer(1)]], device const gg_sel *ms [[buffer(3)]], device half *out [[buffer(4)]],
                       constant uint &nbr [[buffer(8)]], device const float *Pg [[buffer(9)]], uint row [[thread_position_in_grid]]) {
    for (uint t = 0; t < nbr; t++) {
        float o = 0;
        for (uint s = 0; s < TOPK; s++) o += ms->w[t][s] * Pg[((ulong)t * TOPK + s) * D + row];
        o += ms->w[t][TOPK] * Psh[(ulong)t * D + row];
        out[(ulong)t * D + row] = half(o);
    }
}
#elif GG_PART == 5
// ---------------------------------------------------------------- prefill：routed expert 的 grouped GEMM（同 qmm.metal 的 MOE = 1，解碼換成上面的查表版）
// Y [P][N] = X [P][KD] · W_eᵀ。X 依 expert 排好，tiles[tg.y] = {expert, 起始列, 結束列}；權重 [E][N][每列位元組]（格式 PFF）。
// 每個 K 段（KC 個值）先把 TN 列解碼成 half（已乘 scale）放進 threadgroup memory，再用 tensor 單元累加。grid：(N / TN, 區塊數) × 32·SG
// （metal_tensor 與 MetalPerformancePrimitives 由主程式放在最前面 include：E、D、I 等巨集會撞到標頭裡的名稱）
//
// PFS = 1（條帶版）：區塊的 TM 列切成 SG 條、每條 TS = TM / SG 列，第 s 條由 SIMD group s 自己算（TS × TN 的 tensor 運算）；
// 超出這個 expert 的條帶整條不算。每個 expert 平均約 78 列、分布很偏，TM 64 一塊時算到的列數是實際的 1.5 倍，
// 16 列一條只剩 1.1 倍。tensor 運算的累加順序與 tile 形狀無關：結果和 PFS = 0 逐位元相同。
// 量過（M5 Max）比較慢的做法：一個 SIMD group 算多條（累加器變多）、SIMD group 改成分欄、X 先放 threadgroup memory、
// 權重兩份輪流解碼（含專門解碼的 SIMD group）、原始位元組先搬進 threadgroup memory、預取下一段、TM 32 / 48 / 96 / 128、
// KC 32 / 128、TN 32 / 128、剩下的列另用較矮的 kernel 算。tensor 運算本身（不解碼）約 40 TFLOPS，解碼約佔三成且不和
// tensor 運算重疊：主要是權重位元組的讀取延遲與查表
// PFT：IQ2_S、IQ3_XXS、IQ3_S 的 grid 查表改從 threadgroup memory 讀（表在每個 threadgroup 開頭填好，見 pfq）
using namespace mpp::tensor_ops;
static_assert(KD % KC == 0 && KC % 32 == 0, "moe_pf_gg：K 段要是 32 的倍數");
struct gg_qmm_args {
    uint M, N;
};
#ifndef PFS
#define PFS 0
#endif
#ifndef PFT
#define PFT 0
#endif
#ifndef PFDBG
#define PFDBG 0 // 量測用：1 = 不解碼（權重寫常數，看 tensor 運算本身的時間）
#endif

// threadgroup memory 的 grid 表：無號 grid（每個值一個 byte），符號另外用 xor 設定。值與 dqf 的查表完全相同
// （帶符號表的負值就是把 half 的符號位元設起來）。表小（1 KB、2 KB、IQ2_S 8 KB），隨機查表比 constant 快
#define PFT_ON (PFT && (PFF == 22 || PFF == 18 || PFF == 21))
constexpr uint pft_words(int F) { return F == 22 ? 2048 : F == 18 ? 256 : F == 21 ? 512 : 1; } // 表的 uint 數
template <uint NT> inline void pft_fill(threadgroup uint *g, uint tid) {
    if (PFF == 22)
        for (uint i = tid; i < 1024; i += NT) {
            const ulong v = iq2s_grid[i];
            g[2 * i] = uint(v);
            g[2 * i + 1] = uint(v >> 32);
        }
    else if (PFF == 18)
        for (uint i = tid; i < 256; i += NT) g[i] = iq3xxs_grid[i];
    else if (PFF == 21)
        for (uint i = tid; i < 512; i += NT) g[i] = iq3s_grid[i];
}
inline uint2 pft_h4(uint g) { return as_type<uint2>(half4(as_type<uchar4>(g))); } // 4 個 byte → half4 的位元
inline uint2 sgn4(uint s) { return uint2(sgn2(s), sgn2(s >> 2)); }               // 4 個 sign 位元 → half4 的符號遮罩

template <int F> struct pfq { // 沒有 threadgroup 表的格式不會用到
    static inline void run(DP, uint, threadgroup const uint *, thread half4 *, thread float &) {}
};
template <> struct pfq<22> { // IQ2_S（同 dqf<22>）
    static inline void run(DP W, uint u, threadgroup const uint *g, thread half4 *w, thread float &d) {
        DP b = W + (u >> 4) * 82;
        const uint ib = (u >> 1) & 7, hh = u & 1;
        const uint qs = ldu16(b + 2 + 4 * ib + 2 * hh), sg = ldu16(b + 34 + 4 * ib + 2 * hh);
        const uint qh = b[66 + ib], sc = (b[74 + ib] >> (4 * hh)) & 15;
        d = ldf16(b) * (0.5f + float(sc)) * 0.25f;
        threadgroup const uint2 *g2 = (threadgroup const uint2 *)g;
        const uint2 e0 = g2[(qs & 255) | ((qh << (8 - 4 * hh)) & 0x300)], e1 = g2[(qs >> 8) | ((qh << (6 - 4 * hh)) & 0x300)];
        w[0] = hx(pft_h4(e0.x) ^ sgn4(sg));
        w[1] = hx(pft_h4(e0.y) ^ sgn4(sg >> 4));
        w[2] = hx(pft_h4(e1.x) ^ sgn4(sg >> 8));
        w[3] = hx(pft_h4(e1.y) ^ sgn4(sg >> 12));
    }
};
template <> struct pfq<18> { // IQ3_XXS（同 dqf<18>）
    static inline void run(DP W, uint u, threadgroup const uint *g, thread half4 *w, thread float &d) {
        DP b = W + (u >> 4) * 98;
        const uint ib = (u >> 1) & 7, hh = u & 1;
        const uint q0 = ldu16(b + 2 + 8 * ib + 4 * hh), q1 = ldu16(b + 4 + 8 * ib + 4 * hh);
        const uint aux = ldu16(b + 66 + 4 * ib) | (ldu16(b + 68 + 4 * ib) << 16);
        d = ldf16(b) * (0.5f + float(aux >> 28)) * 0.5f;
        const uint a0 = (aux >> (14 * hh)) & 127, a1 = (aux >> (14 * hh + 7)) & 127;
        const uint s0 = a0 | ((popcount(a0) & 1) << 7), s1 = a1 | ((popcount(a1) & 1) << 7);
        w[0] = hx(pft_h4(g[q0 & 255]) ^ sgn4(s0));
        w[1] = hx(pft_h4(g[q0 >> 8]) ^ sgn4(s0 >> 4));
        w[2] = hx(pft_h4(g[q1 & 255]) ^ sgn4(s1));
        w[3] = hx(pft_h4(g[q1 >> 8]) ^ sgn4(s1 >> 4));
    }
};
template <> struct pfq<21> { // IQ3_S（同 dqf<21>）
    static inline void run(DP W, uint u, threadgroup const uint *g, thread half4 *w, thread float &d) {
        DP b = W + (u >> 4) * 110;
        const uint ib = (u >> 1) & 7, hh = u & 1;
        const uint q0 = ldu16(b + 2 + 8 * ib + 4 * hh), q1 = ldu16(b + 4 + 8 * ib + 4 * hh);
        const uint qh = uint(b[66 + ib]) >> (4 * hh), sg = ldu16(b + 74 + 4 * ib + 2 * hh);
        d = ldf16(b) * float(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib & 1))) & 15));
        w[0] = hx(pft_h4(g[(q0 & 255) | ((qh << 8) & 256)]) ^ sgn4(sg));
        w[1] = hx(pft_h4(g[(q0 >> 8) | ((qh << 7) & 256)]) ^ sgn4(sg >> 4));
        w[2] = hx(pft_h4(g[(q1 & 255) | ((qh << 6) & 256)]) ^ sgn4(sg >> 8));
        w[3] = hx(pft_h4(g[(q1 >> 8) | ((qh << 5) & 256)]) ^ sgn4(sg >> 12));
    }
};

// 權重 tile 的 TN 列 × KC 個值（從第 k0 個起）解碼成 half（已乘 scale）放進 Wt [TN][KC]：每個執行緒一次一個單位（16 個值）
template <uint NT> inline void pf_dq(DP W, uint RB, uint n0, uint k0, uint tid, threadgroup half *Wt, TABP tab, threadgroup const uint *gt) {
    for (uint q = tid; q < TN * (KC / 16); q += NT) {
        const uint n = q / (KC / 16), u = k0 / 16 + q % (KC / 16);
        half4 w[4];
        float d, m = 0;
        if (PFDBG == 1) {
            w[0] = w[1] = w[2] = w[3] = half4(half(n & 3));
            d = 1;
        } else if (PFT_ON)
            pfq<PFF>::run(W + (ulong)(n0 + n) * RB, u, gt, w, d);
        else
            dqf<PFF>::run(gref<DP>{W + (ulong)(n0 + n) * RB, nullptr, nullptr, tab}, u, w, d, m);
        const uint p0 = (gf_spl(PFF) ? 32 * (u >> 1) + 8 * (u & 1) : 16 * u) - k0, p1 = p0 + (gf_spl(PFF) ? 16 : 8);
        threadgroup half4 *d0 = (threadgroup half4 *)(Wt + n * KC + p0), *d1 = (threadgroup half4 *)(Wt + n * KC + p1);
        d0[0] = half4(float4(w[0]) * d - m);
        d0[1] = half4(float4(w[1]) * d - m);
        d1[0] = half4(float4(w[2]) * d - m);
        d1[1] = half4(float4(w[3]) * d - m);
    }
}

kernel void moe_pf_gg(device half *X [[buffer(0)]], device const uchar *W [[buffer(1)]], device half *Y [[buffer(4)]],
                      constant gg_qmm_args &a [[buffer(5)]], device const uint4 *tiles [[buffer(6)]], TABQ const uchar *tab [[buffer(7)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NT = 32 * SG, RB = RBY(PFF, KD);
    const uint4 tile = tiles[tg.y];
    const uint m0 = tile.y, m_end = tile.z;
    if (m_end <= m0) return; // 沒用到的區塊
    W += (ulong)tile.x * a.N * RB;
    const uint n0 = tg.x * TN;
    threadgroup half Wt[TN * KC];
#if PFT_ON
    threadgroup uint gt[pft_words(PFF)];
    static_assert(TN * KC * 2 + pft_words(PFF) * 4 <= 32768, "moe_pf_gg：threadgroup memory");
    pft_fill<NT>(gt, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#else
    threadgroup uint *gt = nullptr;
#endif
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX(X, T2(KD, a.M));
    tensor<threadgroup half, T2, tensor_inline> tW(Wt, T2(KC, TN));
#if PFS
    constexpr uint TS = TM / SG;
    static_assert(TM % SG == 0 && TS % 8 == 0, "moe_pf_gg：條帶列數");
    const uint r0 = m0 + tid / 32 * TS; // 這個 SIMD group 的條帶
    const bool live = r0 < m_end;
    constexpr auto desc = matmul2d_descriptor(TS, TN, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<1>> op;
#else
    const uint r0 = m0;
    constexpr bool live = true;
    constexpr auto desc = matmul2d_descriptor(TM, TN, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
#endif
    auto x0 = tX.slice(0, r0);
    auto acc = op.get_destination_cooperative_tensor<decltype(x0), decltype(tW), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    for (uint k0 = 0; k0 < KD; k0 += KC) {
        pf_dq<NT>(W, RB, n0, k0, tid, Wt, tab, gt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            auto xs = tX.slice(k0, r0);
            op.run(xs, tW, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live) return;
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const uint n = n0 + ix[0], mm = r0 + ix[1];
            if (mm < m_end) Y[(ulong)mm * a.N + n] = half(acc[i]);
        }
}
#endif
#endif
