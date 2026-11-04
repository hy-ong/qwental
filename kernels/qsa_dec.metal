// QSA decode / 驗證 v3：單 token decode 與 NB 個 token 的驗證共用同一組 kernel（grid 的最後一維是第幾個 token），
// 所以驗證的每個 token 與逐 token decode 逐位元相同；每層的 dispatch 從 7–9 個（驗證另加 2·NB 個）降到 5–7 個。
//   1. qsa_prep  ：q / k 的 RMSNorm + RoPE、k / v 寫進 cache、indexer q 的 RMSNorm + RoPE、raw_k 與區塊 key，各 head 平行（一次 dispatch）
//   2. qsa_score3：indexer 分數，NB 個 token 共用一次區塊 key 讀取（算式同 qsa_score2）
//   3. qsa_topk3 ：每個 token 一個 threadgroup；用打分數時順便累計的直方圖找到門檻所在的格子，只精算那一格（原本二分搜尋約 31 輪），
//                  選中的集合與 qsa_topk2 相同，輸出依位置排序
//   4. qsa_attn5 ：attention 核心，每段固定是 token 自己的連續幾個區塊（清單先放進 threadgroup memory）
//                  （qsa_attn4：舊的核心，QW_ATT4=1 對照用）
//   5. qsa_merge4：合併各段並乘上 output gate（k_attn_gate 的 dispatch 拿掉），輸出就是 o_proj 的輸入
// 主程式注入：QH、KVH、HD、IH、ID、SEL、CR、MAXPER；名稱改名
#include <metal_stdlib>
using namespace metal;

#define EPS 1e-6f
#define GQA (QH / KVH)
static_assert(HD == 256 && ID == 128 && CR == 4, "qsa_dec：形狀寫死");

struct qsa_args {
    uint L, pos, nvis, ntok, sparse; // 同 qsa.metal
};

inline float d_sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// 同 engine.metal 的 tg_sum（不參與的執行緒傳 0）
inline float d_tg_sum(float v, threadgroup float *tmp, uint tid, uint ntg) {
    v = simd_sum(v);
    const uint sg = tid / 32, lane = tid % 32, nsg = (ntg + 31) / 32;
    if (lane == 0) tmp[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? tmp[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// 同 engine.metal 的 rope_pair
inline void d_rope(thread float &a, thread float &b, uint d, uint pos, float theta, uint rd) {
    const float f = pow(theta, -float(2 * d) / float(rd)) * float(pos);
    const float c = cos(f), s = sin(f);
    const float x1 = a, x2 = b;
    a = x1 * c - x2 * s;
    b = x2 * c + x1 * s;
}

// 前 RD 維做 RoPE（v 是 threadgroup 暫存）
inline void d_rope_tg(threadgroup float *v, uint tid, uint pos) {
    constexpr uint RD = 64;
    constexpr float THETA = 10000000.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < RD / 2) {
        float a = v[tid], b = v[tid + RD / 2];
        d_rope(a, b, tid, pos, THETA, RD);
        v[tid] = a;
        v[tid + RD / 2] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float d_rawk(half x) { return float(bfloat(float(x))); } // raw_k 依參考實作存成 BF16

// ---------------------------------------------------------------- 1. 準備
// grid：(QH + KVH + IH + 1, NB) × 256。threadgroup x：0…QH−1 q head、QH…QH+KVH−1 k head（連同 v 寫進 cache）、
// 接著 IH 個 indexer q head、最後一個是 indexer 的 raw_k（滿 CR 個 token 時產生區塊 key）。token t 在位置 p0 + t。
// 算式與 attn_prep_body / idx_body 相同。區塊 key 用到的同一批 raw_k 直接從 iqk 算（不等其他 threadgroup 寫回）。
template <typename NT>
inline void prep_body(device const half *qg, device const half *kin, device const half *vin, device const NT *qn, device const NT *kn,
                      device half *q, device half *Kc, device half *Vc, uint p0, device const half *iqk, device const NT *iqn,
                      device const NT *ikn, device half *iq, device float *rawk, device half *kblk, uint2 tg, uint tid,
                      threadgroup float *tmp, threadgroup float *v) {
    const uint hh = tg.x, t = tg.y, pos = p0 + tg.y;
    if (hh < QH + KVH) {
        const bool isq = hh < QH;
        const uint kh = hh - QH;
        const float x = isq ? float(qg[(ulong)t * 2 * QH * HD + hh * 2 * HD + tid]) : float(kin[(ulong)t * KVH * HD + kh * HD + tid]);
        const float r = rsqrt(d_tg_sum(x * x, tmp, tid, HD) / HD + EPS);
        v[tid] = x * r * float(isq ? qn[tid] : kn[tid]);
        d_rope_tg(v, tid, pos);
        if (isq) q[((ulong)t * QH + hh) * HD + tid] = half(v[tid]);
        else {
            Kc[((ulong)pos * KVH + kh) * HD + tid] = half(v[tid]);
            Vc[((ulong)pos * KVH + kh) * HD + tid] = vin[(ulong)t * KVH * HD + kh * HD + tid];
        }
        return;
    }
    device const half *ik = iqk + (ulong)t * (IH + 1) * ID;
    if (hh < QH + KVH + IH) { // indexer q head
        const uint h = hh - QH - KVH;
        const float x = tid < ID ? float(ik[h * ID + tid]) : 0.0f;
        const float r = rsqrt(d_tg_sum(x * x, tmp, tid, ID) / ID + EPS);
        if (tid < ID) v[tid] = x * r * float(iqn[tid]);
        d_rope_tg(v, tid, pos);
        if (tid < ID) iq[(ulong)t * IH * ID + h * ID + tid] = half(v[tid]);
        return;
    }
    // raw_k；滿 CR 個 token 時產生區塊 key（同一批的 raw_k 從 iqk 直接算，之前的從 cache 讀）
    if (tid < ID) rawk[(ulong)pos * ID + tid] = d_rawk(ik[IH * ID + tid]);
    if ((pos + 1) % CR) return;
    const uint nb = pos / CR;
    float m = 0;
    if (tid < ID)
        for (uint j = 0; j < CR; j++) {
            const uint pj = nb * CR + j;
            m += pj >= p0 ? d_rawk(iqk[(ulong)(pj - p0) * (IH + 1) * ID + IH * ID + tid]) : rawk[(ulong)pj * ID + tid];
        }
    m = float(bfloat(m / CR));
    const float r = rsqrt(d_tg_sum(tid < ID ? m * m : 0.0f, tmp, tid, ID) / ID + EPS);
    if (tid < ID) v[tid] = m * r * float(ikn[tid]);
    d_rope_tg(v, tid, nb * CR);
    if (tid < ID) kblk[(ulong)nb * ID + tid] = half(v[tid]);
}

// norm 權重型別：MLX 是 bfloat（_bf），GGUF profile 是 float（_f）
#define QSA_PREP(NAME, NT)                                                                                                            \
    kernel void NAME(device const half *qg [[buffer(0)]], device const half *kin [[buffer(1)]], device const half *vin [[buffer(2)]], \
                     device const NT *qn [[buffer(3)]], device const NT *kn [[buffer(4)]], device half *q [[buffer(5)]],             \
                     device half *Kc [[buffer(6)]], device half *Vc [[buffer(7)]], constant uint &p0 [[buffer(8)]],                  \
                     device const half *iqk [[buffer(9)]], device const NT *iqn [[buffer(10)]], device const NT *ikn [[buffer(11)]],  \
                     device half *iq [[buffer(12)]], device float *rawk [[buffer(13)]], device half *kblk [[buffer(14)]],             \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {                         \
        threadgroup float tmp[32], v[256];                                                                                            \
        prep_body<NT>(qg, kin, vin, qn, kn, q, Kc, Vc, p0, iqk, iqn, ikn, iq, rawk, kblk, tg, tid, tmp, v);                           \
    }
QSA_PREP(qsa_prep_bf, bfloat)
QSA_PREP(qsa_prep_f, float)

// ---------------------------------------------------------------- 2. indexer 分數
// 每個 SIMD group 一個區塊（每個 lane 4 維），區塊 key 讀一次給 nt 個 token；算式同 qsa_score2。
// score[t][n]（列距 sstride），只寫 n < nvis(t)、而且 token t 是稀疏的。
// 同時累計分數高 HB 位元的直方圖 hist[t][2^HB]（device atomic；qsa_topk3 用完歸零），讓選取只要精算門檻所在的那一格。
// grid：ceil(max nvis / 8) × 256
#define HB 12
#define HSH (31 - HB) // 分數 ≥ 0：符號位元是 0，取第 30…19 位元
#define NBIN (1u << HB)
kernel void qsa_score3(device const half *iq [[buffer(0)]], device const half *kblk [[buffer(1)]], device float *score [[buffer(2)]],
                       constant qsa_args *aa [[buffer(3)]], constant uint &nt [[buffer(4)]], constant uint &sstride [[buffer(5)]],
                       device atomic_uint *hist [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint n = tg * 8 + sg;
    const float4 k = float4(((device const half4 *)(kblk + (ulong)n * ID))[lane]);
    for (uint t = 0; t < nt; t++) {
        if (!aa[t].sparse || n >= aa[t].nvis) continue;
        device const half *qt = iq + (ulong)t * IH * ID;
        float s = 0;
        for (uint h = 0; h < IH; h++) s += max(simd_sum(dot(float4(((device const half4 *)(qt + h * ID))[lane]), k)), 0.0f);
        if (lane == 0) {
            const float v = s * rsqrt(float(ID));
            score[(ulong)t * sstride + n] = v;
            atomic_fetch_add_explicit(&hist[t * NBIN + (as_type<uint>(v) >> HSH)], 1u, memory_order_relaxed);
        }
    }
}

// ---------------------------------------------------------------- 3. 選前 SEL 個區塊
// 一個 token 一個 threadgroup（1024 個執行緒，每個執行緒連續 MAXPER 個分數放暫存器）。分數 ≥ 0，float 的位元當 uint 比大小。
// 門檻 thr = 第 SEL 大的值：
//   1. 直方圖由高往低累加，找到第 SEL 個落在哪一格 b*（之前的格子全選）
//   2. 落在 b* 的分數再依接下來 8 位元做 threadgroup 直方圖，找到門檻所在的子格 b2
//   3. 落在 b2 的分數收進 threadgroup memory，各自算「比我大的有幾個」，排名對上的就是 thr
//   （b2 裡超過 CMAX 個時改用二分搜尋，結果相同。原本沒有第 2 步：b* 常有上千個，排名是 O(n²)，64K context 要 14 µs）
// 選中的集合與 qsa_topk2 相同（所有 > thr 的，再依位置補等於 thr 的），但輸出依位置排序。用完把直方圖歸零。grid：NB × 1024
#define CMAX 2048
#define NB2 256        // 第二層直方圖格數（threadgroup memory）
#define HSH2 (HSH - 8) // 第二層用第 18…11 位元
kernel void qsa_topk3(device const float *score0 [[buffer(0)]], device uint *sel0 [[buffer(1)]], device uint *nsel0 [[buffer(2)]],
                      constant qsa_args *aa [[buffer(3)]], constant uint &sstride [[buffer(4)]], device uint *hist0 [[buffer(5)]],
                      uint z [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NTH = 1024, NSG = NTH / 32, BPT = NBIN / NTH; // 每個執行緒幾格
    static_assert(NBIN % NTH == 0, "qsa_topk3：直方圖格數");
    const qsa_args a = aa[z];
    if (!a.sparse) return;
    device const float *score = score0 + (ulong)z * sstride;
    device uint *sel = sel0 + (ulong)z * SEL;
    device uint *hist = hist0 + (ulong)z * NBIN;
    const uint nv = a.nvis;
    // 一次發出：這個執行緒的分數、直方圖格子（第 tid·BPT 格起，格子由高往低編：g = NBIN − 1 − …）
    const uint b0 = tid * MAXPER;
    uint v[MAXPER];
#pragma unroll
    for (uint j = 0; j < MAXPER; j++) v[j] = b0 + j < nv ? as_type<uint>(score[b0 + j]) : 0u;
    uint hc[BPT], hsum = 0;
    for (uint j = 0; j < BPT; j++) {
        hc[j] = hist[NBIN - 1 - (tid * BPT + j)];
        hsum += hc[j];
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint j = 0; j < BPT; j++) hist[NBIN - 1 - (tid * BPT + j)] = 0; // 歸零（給下一層）
    if (nv <= SEL) {
        for (uint i = tid; i < nv; i += NTH) sel[i] = i;
        if (tid == 0) nsel0[z] = nv;
        return;
    }
    threadgroup uint red[NSG], s_bin, s_above, s_cnt, s_thr, s_ok, s_tot, s_b2, s_above2;
    threadgroup uint cand[CMAX];
    threadgroup atomic_uint h2[NB2];
    if (tid < NB2) atomic_store_explicit(&h2[tid], 0u, memory_order_relaxed);
    // 1. 由高往低的前綴和 → b*（直方圖總數不等於 nv 時不用它，例如量測時只跑了打分數）
    const uint xs = simd_prefix_exclusive_sum(hsum);
    if (lane == 31) red[sg] = xs + hsum;
    if (tid == 0) { s_cnt = 0; s_ok = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint r = lane < NSG ? red[lane] : 0u;
        const uint rx = simd_prefix_exclusive_sum(r);
        if (lane < NSG) red[lane] = rx;
        if (lane == NSG - 1) s_tot = rx + r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#ifndef QSA_NOHIST // 定義時一律走二分搜尋（測試用）
    if (s_tot == nv) { // 直方圖要剛好是這個 token 的分數（否則改用二分搜尋）
#else
    if (false) {
#endif
        uint above = red[sg] + xs; // 這個執行緒的第一格之前（較高的格子）有幾個
        for (uint j = 0; j < BPT; j++) {
            if (above < SEL && above + hc[j] >= SEL) {
                s_bin = NBIN - 1 - (tid * BPT + j);
                s_above = above;
                s_ok = 1;
            }
            above += hc[j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint thr;
    if (s_ok) {
        // 2. b* 裡再依接下來 8 位元分 256 格（threadgroup 直方圖），找到門檻所在的子格 b2
        const uint bs = s_bin, need = SEL - s_above; // b* 裡要選幾個（≥ 1）
#pragma unroll
        for (uint j = 0; j < MAXPER; j++)
            if (b0 + j < nv && (v[j] >> HSH) == bs) atomic_fetch_add_explicit(&h2[(v[j] >> HSH2) & (NB2 - 1)], 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) { // lane L 管第 NB2−1−8L … NB2−8−8L 格（由高往低）
            uint c8[8], sum = 0;
            for (uint j = 0; j < 8; j++) {
                c8[j] = atomic_load_explicit(&h2[NB2 - 1 - lane * 8 - j], memory_order_relaxed);
                sum += c8[j];
            }
            uint above = simd_prefix_exclusive_sum(sum);
            for (uint j = 0; j < 8; j++) {
                if (above < need && above + c8[j] >= need) {
                    s_b2 = NB2 - 1 - lane * 8 - j;
                    s_above2 = above;
                }
                above += c8[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 3. 子格 b2 的分數收進 cand（通常只有幾個），排名決定門檻
        const uint key = (bs << 8) | s_b2, need2 = need - s_above2;
#pragma unroll
        for (uint j = 0; j < MAXPER; j++)
            if (b0 + j < nv && (v[j] >> HSH2) == key) {
                const uint c = atomic_fetch_add_explicit((threadgroup atomic_uint *)&s_cnt, 1u, memory_order_relaxed);
                if (c < CMAX) cand[c] = v[j];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint nc = s_cnt;
        if (nc <= CMAX) {
            for (uint c = tid; c < nc; c += NTH) {
                const uint x = cand[c];
                uint gt = 0, ge = 0;
                for (uint k = 0; k < nc; k++) {
                    const uint w = cand[k];
                    gt += w > x;
                    ge += w >= x;
                }
                if (gt < need2 && ge >= need2) s_thr = x; // 同值的都寫同一個值
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            thr = s_thr;
        } else thr = 0xffffffffu; // 太多同格：改用二分搜尋
    } else thr = 0xffffffffu; // 直方圖不完整（例如沒先跑 qsa_score3）：改用二分搜尋
    if (thr == 0xffffffffu) { // 同 qsa_topk2 的二分搜尋
        auto count_ge = [&](uint t) -> uint {
            uint c = 0;
#pragma unroll
            for (uint j = 0; j < MAXPER; j++) c += (b0 + j < nv && v[j] >= t) ? 1u : 0u;
            c = simd_sum(c);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) red[sg] = c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return simd_sum(lane < NSG ? red[lane] : 0u);
        };
        uint lo = 0, top = 0x7f800000u;
        while (lo < top) {
            const uint mid = lo + (top - lo + 1) / 2;
            if (count_ge(mid) >= SEL) lo = mid;
            else top = mid - 1;
        }
        thr = lo;
    }
    // 寫出：依位置排序（qsa_attn5 依排名分段、驗證的 token 合併讀取都要靠這個順序）。
    // 選中 = 所有 > thr 的，加上依位置前 SEL − (> thr 的個數) 個等於 thr 的；兩次前綴和：先數等於 thr 的、再數選中的
    uint ngt = 0, neq = 0;
#pragma unroll
    for (uint j = 0; j < MAXPER; j++)
        if (b0 + j < nv) {
            ngt += v[j] > thr;
            neq += v[j] == thr;
        }
    threadgroup uint pre_gt[NSG], pre_eq[NSG], tgt;
    const uint xg = simd_prefix_exclusive_sum(ngt), xe = simd_prefix_exclusive_sum(neq);
    if (lane == 31) {
        pre_gt[sg] = xg + ngt;
        pre_eq[sg] = xe + neq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint a1 = lane < NSG ? pre_gt[lane] : 0, a2 = lane < NSG ? pre_eq[lane] : 0;
        uint s1 = simd_prefix_exclusive_sum(a1), s2 = simd_prefix_exclusive_sum(a2);
        if (lane < NSG) pre_eq[lane] = s2;
        if (lane == NSG - 1) tgt = s1 + a1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint neq_take = SEL - tgt;             // 等於 thr 的要選幾個（依位置）
    const uint oe0 = pre_eq[sg] + xe;            // 這個執行緒之前有幾個等於 thr 的
    const uint take = oe0 >= neq_take ? 0u : min(neq, neq_take - oe0);
    const uint nmine = ngt + take;               // 這個執行緒選中幾個
    const uint xs2 = simd_prefix_exclusive_sum(nmine);
    if (lane == 31) pre_gt[sg] = xs2 + nmine;    // pre_gt 改放選中的個數
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint a1 = lane < NSG ? pre_gt[lane] : 0;
        const uint s1 = simd_prefix_exclusive_sum(a1);
        if (lane < NSG) pre_gt[lane] = s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint o = pre_gt[sg] + xs2, ne = 0;
#pragma unroll
    for (uint j = 0; j < MAXPER; j++)
        if (b0 + j < nv) {
            bool pick = v[j] > thr;
            if (v[j] == thr) pick = ne++ < take;
            if (pick) sel[o++] = b0 + j;
        }
    if (tid == 0) nsel0[z] = SEL;
}

// 第 i 個要看的 token（同 qsa.metal 的 tok_of）
static inline uint d_tok(uint i, device const uint *sel, uint nsel, constant qsa_args &a) {
    if (!a.sparse) return i;
    const uint nb = nsel * CR;
    if (i < nb) return sel[i / CR] * CR + i % CR;
    return a.nvis * CR + (i - nb);
}

// ---------------------------------------------------------------- 4. attention 核心 + 合併 + gate
// qsa_attn2 的排列：(NCH4 段, KVH, NB) 個 threadgroup × 12 個 SIMD group（一個 q head），每個 lane 8 維、每次 U4 個 token，跨 lane 用 simd_sum。
// part [z][NCH4][QH][HD + 2]（acc、max、sum），由 qsa_merge4 合併並乘上 output gate。
// （試過的其他排列都沒有比較快，見 docs：每個 lane 一個 token 的 q·k + 每個執行緒一維的 p·v、各段先合併的 token 流、
//   最後完成的 threadgroup 合併、預取下一組 K / V。串行量測時要嘛短 context 較慢、要嘛多 token 驗證較慢）
#ifndef NCH4
#define NCH4 32 // 主程式注入（QW_NCH4）；48 時 10K context 單 token 快 5%，但短 context 與 8 個 token 的驗證都變慢
#endif
#define U4 4
kernel void qsa_attn4(device const half *q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]],
                      device const uint *sel0 [[buffer(3)]], device const uint *nsel0 [[buffer(4)]], device float *part0 [[buffer(5)]],
                      constant qsa_args *aa [[buffer(6)]], constant uint &dense_only [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PL = HD / 32;
    const uint ch = tg.x, kvh = tg.y, z = tg.z, qh = kvh * GQA + sg;
    constant qsa_args &a = aa[z];
    if (dense_only && a.sparse) return; // 稀疏的 token 交給 qsa_attn5
    device const uint *sel = sel0 + (ulong)z * SEL;
    const uint nsel = a.sparse ? nsel0[z] : 0;
    const uint ntok = a.ntok, per = (ntok + NCH4 - 1) / NCH4;
    const uint i0 = ch * per, i1 = min(i0 + per, ntok);
    const float scale = rsqrt(float(HD));
    float qv[PL], acc[PL];
    {
        device const half4 *qp = (device const half4 *)(q + ((ulong)z * QH + qh) * HD + lane * PL);
        const float4 q0 = float4(qp[0]), q1 = float4(qp[1]);
        qv[0] = q0.x * scale; qv[1] = q0.y * scale; qv[2] = q0.z * scale; qv[3] = q0.w * scale;
        qv[4] = q1.x * scale; qv[5] = q1.y * scale; qv[6] = q1.z * scale; qv[7] = q1.w * scale;
    }
    for (uint d = 0; d < PL; d++) acc[d] = 0;
    float m = -INFINITY, l = 0;
    for (uint i = i0; i < i1; i += U4) {
        float s[U4];
        float4 va[U4], vb[U4];
        for (uint u = 0; u < U4; u++) {
            const bool ok = i + u < i1;
            const ulong t = d_tok(ok ? i + u : i0, sel, nsel, a);
            device const half4 *kr = (device const half4 *)(K + (t * KVH + kvh) * HD + lane * PL);
            device const half4 *vr = (device const half4 *)(V + (t * KVH + kvh) * HD + lane * PL);
            const float4 k0 = float4(kr[0]), k1 = float4(kr[1]);
            va[u] = float4(vr[0]);
            vb[u] = float4(vr[1]);
            const float d = k0.x * qv[0] + k0.y * qv[1] + k0.z * qv[2] + k0.w * qv[3] + k1.x * qv[4] + k1.y * qv[5] + k1.z * qv[6] +
                            k1.w * qv[7];
            s[u] = ok ? simd_sum(d) : -INFINITY;
        }
        float mn = m;
        for (uint u = 0; u < U4; u++) mn = max(mn, s[u]);
        const float alpha = exp(m - mn);
        l *= alpha;
        for (uint d = 0; d < PL; d++) acc[d] *= alpha;
        for (uint u = 0; u < U4; u++) {
            const float p = exp(s[u] - mn);
            l += p;
            acc[0] += p * va[u].x; acc[1] += p * va[u].y; acc[2] += p * va[u].z; acc[3] += p * va[u].w;
            acc[4] += p * vb[u].x; acc[5] += p * vb[u].y; acc[6] += p * vb[u].z; acc[7] += p * vb[u].w;
        }
        m = mn;
    }
    device float *o = part0 + (((ulong)z * NCH4 + ch) * QH + qh) * (HD + 2);
    for (uint d = 0; d < PL; d++) o[lane * PL + d] = acc[d];
    if (lane == 0) {
        o[HD] = m;
        o[HD + 1] = l;
    }
}

// ---------------------------------------------------------------- 4b. attention 核心 v5
// token 要看的區塊清單：稀疏時是依位置排序的 SEL 個（qsa_topk3），不稀疏時是 0…nvis−1；第 c 段 = 清單的 [c·bpc, (c+1)·bpc)
// （稀疏 bpc = SEL / NCH4，不稀疏 bpc = ⌈nvis / NCH4⌉），最後一段再接上尾端 0–3 個 token（位置 nvis·CR 起）。
// 一個 threadgroup：(段 c, kv head, 一組最多 TT 個 token)，12 個 SIMD group 各一個 q head、每個 lane 8 維，一次一個區塊（U4 = CR 個 token）。
// 段的清單先放進 threadgroup memory，K / V 的位址不用等 device 讀 sel（qsa_attn4 每組 token 都先讀 sel 再讀 K），而且每次讀的
// 4 個 token 是同一個區塊（連續位址）：10K 單 token 的核心 33 → 15 µs、8 個 token 的驗證 127 → 64 µs。
// TT > 1 時組內各 token 的段依位置合併走訪：每個區塊的 K / V 只讀一次，選了它的 token 各自更新。量過反而慢（10K、真實程度的重疊：
// 8 個 token 的核心 TT = 2 慢 26%、TT = 4 慢 1.6 倍）：同一段的區塊變成依序走完，平行度不夠；各 token 分開時重疊的區塊本來就多半在快取裡。
// 每個 token 的運算只看自己的區塊序列，與同組有沒有其他 token 無關，所以 decode（一組 1 個）與驗證逐位元相同。
// part 的排列同 qsa_attn4，由 qsa_merge4 合併。
#ifndef TT
#define TT 1 // 一組幾個 token（主程式注入 QW_ATT_TT）
#endif
#define BPC (SEL / NCH4)
static_assert(SEL % NCH4 == 0 && U4 == CR, "qsa_attn5：段要剛好切開 SEL 個區塊");
static_assert(TT >= 1 && TT <= 8, "qsa_attn5：TT");
kernel void qsa_attn5(device const half *q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]],
                      device const uint *sel0 [[buffer(3)]], device float *part0 [[buffer(4)]], constant qsa_args *aa [[buffer(5)]],
                      constant uint &nt [[buffer(6)]], device atomic_uint *stat [[buffer(7)]], constant uint &dostat [[buffer(8)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PL = HD / 32, NTH = 32 * GQA, END = 0xffffffffu;
    threadgroup uint sl[TT][BPC + 1];
    const uint ch = tg.x, kvh = tg.y, t0 = tg.z * TT, qh = kvh * GQA + sg;
    const uint n = min((uint)TT, nt - t0);
    for (uint i = tid; i < TT * (BPC + 1); i += NTH) {
        const uint t = i / (BPC + 1), j = i % (BPC + 1);
        uint v = END;
        if (t < n) {
            const qsa_args a = aa[t0 + t];
            const uint nblk = a.sparse ? SEL : a.nvis, bpc = a.sparse ? BPC : (a.nvis + NCH4 - 1) / NCH4, k = ch * bpc + j;
            if (j < bpc && k < nblk) v = a.sparse ? sel0[(ulong)(t0 + t) * SEL + k] : k;
        }
        sl[t][j] = v;
    }
    const float scale = rsqrt(float(HD));
    float qv[TT][PL], acc[TT][PL], m[TT], l[TT];
    uint ix[TT];
#pragma unroll
    for (uint t = 0; t < TT; t++) {
        ix[t] = 0;
        m[t] = -INFINITY;
        l[t] = 0;
        for (uint d = 0; d < PL; d++) acc[t][d] = 0;
        if (t < n) {
            device const half4 *qp = (device const half4 *)(q + ((ulong)(t0 + t) * QH + qh) * HD + lane * PL);
            const float4 q0 = float4(qp[0]), q1 = float4(qp[1]);
            qv[t][0] = q0.x * scale; qv[t][1] = q0.y * scale; qv[t][2] = q0.z * scale; qv[t][3] = q0.w * scale;
            qv[t][4] = q1.x * scale; qv[t][5] = q1.y * scale; qv[t][6] = q1.z * scale; qv[t][7] = q1.w * scale;
        } else
            for (uint d = 0; d < PL; d++) qv[t][d] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half *Kh = K + kvh * HD + lane * PL, *Vh = V + kvh * HD + lane * PL;
    // 一次 U4 個 token 的更新（同 qsa_attn4 迴圈內的算式）；nok：前幾個有效
    auto upd = [&](uint t, thread const float4 *ka, thread const float4 *kb, thread const float4 *va, thread const float4 *vb, uint nok) {
        float s[U4];
        for (uint u = 0; u < U4; u++) {
            const float d = ka[u].x * qv[t][0] + ka[u].y * qv[t][1] + ka[u].z * qv[t][2] + ka[u].w * qv[t][3] + kb[u].x * qv[t][4] +
                            kb[u].y * qv[t][5] + kb[u].z * qv[t][6] + kb[u].w * qv[t][7];
            s[u] = u < nok ? simd_sum(d) : -INFINITY;
        }
        float mn = m[t];
        for (uint u = 0; u < U4; u++) mn = max(mn, s[u]);
        const float alpha = exp(m[t] - mn);
        l[t] *= alpha;
        for (uint d = 0; d < PL; d++) acc[t][d] *= alpha;
        for (uint u = 0; u < U4; u++) {
            const float p = exp(s[u] - mn);
            l[t] += p;
            acc[t][0] += p * va[u].x; acc[t][1] += p * va[u].y; acc[t][2] += p * va[u].z; acc[t][3] += p * va[u].w;
            acc[t][4] += p * vb[u].x; acc[t][5] += p * vb[u].y; acc[t][6] += p * vb[u].z; acc[t][7] += p * vb[u].w;
        }
        m[t] = mn;
    };
    uint steps = 0;
    for (;;) {
        uint cur[TT], b = END;
#pragma unroll
        for (uint t = 0; t < TT; t++) {
            cur[t] = sl[t][ix[t]]; // ix 最多到 BPC（那格是 END）
            b = min(b, cur[t]);
        }
        if (b == END) break;
        steps++;
        float4 ka[U4], kb[U4], va[U4], vb[U4];
        for (uint u = 0; u < U4; u++) {
            const ulong o = ((ulong)b * CR + u) * KVH * HD;
            device const half4 *kr = (device const half4 *)(Kh + o), *vr = (device const half4 *)(Vh + o);
            ka[u] = float4(kr[0]);
            kb[u] = float4(kr[1]);
            va[u] = float4(vr[0]);
            vb[u] = float4(vr[1]);
        }
#pragma unroll
        for (uint t = 0; t < TT; t++)
            if (cur[t] == b) {
                upd(t, ka, kb, va, vb, U4);
                ix[t]++;
            }
    }
    if (ch == NCH4 - 1) // 尾端：位置 nvis·CR … pos（0–3 個）
#pragma unroll
        for (uint t = 0; t < TT; t++) {
            if (t >= n) continue;
            const qsa_args a = aa[t0 + t];
            const uint r = a.ntok - (a.sparse ? SEL : a.nvis) * CR;
            if (r == 0) continue;
            float4 ka[U4], kb[U4], va[U4], vb[U4];
            for (uint u = 0; u < U4; u++) {
                const ulong o = ((ulong)a.nvis * CR + (u < r ? u : 0)) * KVH * HD;
                device const half4 *kr = (device const half4 *)(Kh + o), *vr = (device const half4 *)(Vh + o);
                ka[u] = float4(kr[0]);
                kb[u] = float4(kr[1]);
                va[u] = float4(vr[0]);
                vb[u] = float4(vr[1]);
            }
            upd(t, ka, kb, va, vb, r);
        }
#pragma unroll
    for (uint t = 0; t < TT; t++) {
        if (t >= n) continue;
        device float *o = part0 + (((ulong)(t0 + t) * NCH4 + ch) * QH + qh) * (HD + 2);
        for (uint d = 0; d < PL; d++) o[lane * PL + d] = acc[t][d];
        if (lane == 0) {
            o[HD] = m[t];
            o[HD + 1] = l[t];
        }
    }
    if (dostat && tid == 0) { // 量測用（QW_ATT5_STATS）：合併後讀了幾個區塊 vs 各 token 各讀
        uint tot = 0;
        for (uint t = 0; t < n; t++) tot += aa[t0 + t].sparse ? BPC : (aa[t0 + t].nvis + NCH4 - 1) / NCH4;
        atomic_fetch_add_explicit(&stat[0], steps, memory_order_relaxed);
        atomic_fetch_add_explicit(&stat[1], tot, memory_order_relaxed);
    }
}

// 除錯（QW_ATT5_VCHECK）：兩份 part 逐位元比較，不同的個數加到 stat[2]
kernel void qsa_cmp5(device const uint *a [[buffer(0)]], device const uint *b [[buffer(1)]], constant uint &n [[buffer(2)]],
                     device atomic_uint *stat [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n && a[i] != b[i]) atomic_fetch_add_explicit(&stat[2], 1u, memory_order_relaxed);
}

// qsa_attn4 的合併 + output gate（同 qsa_merge2 的算法），結果直接寫出 y。只處理不稀疏的 token。grid：(QH, NB) × 256
kernel void qsa_merge4(device const float *part0 [[buffer(0)]], device const half *qg [[buffer(1)]], device half *y [[buffer(2)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PL = HD / 32, NS = 8;
    threadgroup float sm[NS], sl[NS], sacc[NS][HD];
    const uint qh = tg.x, z = tg.y;
    device const float *part = part0 + (ulong)z * NCH4 * QH * (HD + 2);
    float m = -INFINITY, l = 0, acc[PL];
    for (uint d = 0; d < PL; d++) acc[d] = 0;
    for (uint c = sg; c < NCH4; c += NS) {
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
    const uint d = sg * 32 + lane; // 8 個 SIMD group 各寫 32 維
    float M = -INFINITY;
    for (uint s2 = 0; s2 < NS; s2++) M = max(M, sm[s2]);
    float Ls = 0, o = 0;
    for (uint s2 = 0; s2 < NS; s2++) {
        if (sl[s2] == 0) continue;
        const float w = exp(sm[s2] - M);
        Ls += sl[s2] * w;
        o += sacc[s2][d] * w;
    }
    const float g = float(qg[(ulong)z * 2 * QH * HD + qh * 2 * HD + HD + d]);
    y[((ulong)z * QH + qh) * HD + d] = half(float(half(o / Ls)) * d_sigm(g));
}
