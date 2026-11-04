// Decode（單一 token）的合併 MoE
// 語意（依 ref/ 參考實作的 SparseMoeBlock）：
//   logits = Wr · x（FP32 計算）；選 logits 最大的 TOPK 個 expert；
//   w = softmax(選中的 TOPK 個 logit)（只對這 10 個做 softmax）
//   out = Σ_s w_s · Wd_e (SiLU(Wg_e x) ⊙ Wu_e x) + sigmoid(wsg · x) · Wsd (SiLU(Wsg x) ⊙ Wsu x)
//
// 4 個 kernel：
//   moe_router ：router（8-bit，[E, D]）算 E 個 logit；最後一個 SIMD group 算 shared_expert_gate（BF16，[D]）
//   moe_select ：一個 SIMD group 選前 TOPK、算 softmax 與 sigmoid
//   moe_gateup ：10 個 routed（4-bit）+ 1 個共用（8-bit）expert 的 gate/up，直接輸出 SiLU(gate)·up
//   moe_down   ：每個輸出列在 kernel 內把 11 個 expert 的 down 加權加總（不需要 atomic）
//
// 權重格式同 qmv.metal（MLX affine、group 64）。gate 與 up 是分開的兩個 tensor（checkpoint 原樣 mmap，不合併）。
// 主程式注入：E、TOPK、D（hidden）、I（expert 中間維度）、RPS（每個 SIMD group 算幾列）

#include <metal_stdlib>
using namespace metal;

#define GS 64
#define CHUNK 16

// 量化列與 x 的部分內積（每個 lane 處理 chunk c = lane, lane + 32, …），回傳尚未 simd_sum 的值
template <int BITS, int K>
inline float qdot(device const uint *W, device const bfloat *S, device const bfloat *Z, ulong row,
                  device const half *x, uint lane) {
    constexpr uint NCH = K / CHUNK, ROWW = K * BITS / 32, NG = K / GS;
    float acc = 0;
    for (uint c = lane; c < NCH; c += 32) {
        device const half4 *xp = (device const half4 *)(x + c * CHUNK);
        float xv[CHUNK], xs = 0;
        for (int j = 0; j < 4; j++) {
            float4 h = float4(xp[j]);
            xv[4 * j] = h.x; xv[4 * j + 1] = h.y; xv[4 * j + 2] = h.z; xv[4 * j + 3] = h.w;
            xs += h.x + h.y + h.z + h.w;
        }
        float d = 0;
        if (BITS == 4) {
            uint2 w = ((device const uint2 *)(W + row * ROWW))[c];
            for (int n = 0; n < 8; n++) d += float((w.x >> (4 * n)) & 15u) * xv[n];
            for (int n = 0; n < 8; n++) d += float((w.y >> (4 * n)) & 15u) * xv[8 + n];
        } else {
            // 兩次 8 bytes 讀取：checkpoint 的 tensor 只保證 8 bytes 對齊
            const uint2 wl = ((device const uint2 *)(W + row * ROWW))[2 * c], wh = ((device const uint2 *)(W + row * ROWW))[2 * c + 1];
            const uint4 w = uint4(wl, wh);
            for (int n = 0; n < 4; n++) d += float((w.x >> (8 * n)) & 255u) * xv[n];
            for (int n = 0; n < 4; n++) d += float((w.y >> (8 * n)) & 255u) * xv[4 + n];
            for (int n = 0; n < 4; n++) d += float((w.z >> (8 * n)) & 255u) * xv[8 + n];
            for (int n = 0; n < 4; n++) d += float((w.w >> (8 * n)) & 255u) * xv[12 + n];
        }
        const uint g = c / (GS / CHUNK);
        acc += float(S[row * NG + g]) * d + float(Z[row * NG + g]) * xs;
    }
    return acc;
}

// 兩列（例如 gate 與 up）共用同一段 x 的內積：x 只讀一次
template <int BITS, int K>
inline void qdot2(device const uint *Wa, device const bfloat *Sa, device const bfloat *Za, ulong ra,
                  device const uint *Wb, device const bfloat *Sb, device const bfloat *Zb, ulong rb,
                  device const half *x, uint lane, thread float &oa, thread float &ob) {
    constexpr uint NCH = K / CHUNK, ROWW = K * BITS / 32, NG = K / GS;
    float acc_a = 0, acc_b = 0;
    for (uint c = lane; c < NCH; c += 32) {
        device const half4 *xp = (device const half4 *)(x + c * CHUNK);
        float xv[CHUNK], xs = 0;
        for (int j = 0; j < 4; j++) {
            float4 h = float4(xp[j]);
            xv[4 * j] = h.x; xv[4 * j + 1] = h.y; xv[4 * j + 2] = h.z; xv[4 * j + 3] = h.w;
            xs += h.x + h.y + h.z + h.w;
        }
        float da = 0, db = 0;
        if (BITS == 4) {
            const uint2 wa = ((device const uint2 *)(Wa + ra * ROWW))[c], wb = ((device const uint2 *)(Wb + rb * ROWW))[c];
            for (int n = 0; n < 8; n++) {
                da += float((wa.x >> (4 * n)) & 15u) * xv[n];
                db += float((wb.x >> (4 * n)) & 15u) * xv[n];
            }
            for (int n = 0; n < 8; n++) {
                da += float((wa.y >> (4 * n)) & 15u) * xv[8 + n];
                db += float((wb.y >> (4 * n)) & 15u) * xv[8 + n];
            }
        } else {
            const uint2 a0 = ((device const uint2 *)(Wa + ra * ROWW))[2 * c], a1 = ((device const uint2 *)(Wa + ra * ROWW))[2 * c + 1];
            const uint2 b0 = ((device const uint2 *)(Wb + rb * ROWW))[2 * c], b1 = ((device const uint2 *)(Wb + rb * ROWW))[2 * c + 1];
            const uint4 wa = uint4(a0, a1), wb = uint4(b0, b1);
            for (int q = 0; q < 4; q++)
                for (int n = 0; n < 4; n++) {
                    da += float((wa[q] >> (8 * n)) & 255u) * xv[4 * q + n];
                    db += float((wb[q] >> (8 * n)) & 255u) * xv[4 * q + n];
                }
        }
        const uint g = c / (GS / CHUNK);
        acc_a += float(Sa[ra * NG + g]) * da + float(Za[ra * NG + g]) * xs;
        acc_b += float(Sb[rb * NG + g]) * db + float(Zb[rb * NG + g]) * xs;
    }
    oa = acc_a;
    ob = acc_b;
}

// ---------------------------------------------------------------- 1. router
// grid：(E + 1) / 8 個 threadgroup × 8 個 SIMD group，一個 SIMD group 一列；第 E 列是 shared_expert_gate
kernel void moe_router(device const uint *Wr [[buffer(0)]], device const bfloat *Sr [[buffer(1)]],
                       device const bfloat *Zr [[buffer(2)]],   // router：8-bit [E, D]
                       device const bfloat *Wsg [[buffer(3)]],  // shared_expert_gate：BF16 [D]
                       device const half *x [[buffer(4)]],
                       device float *logits [[buffer(5)]],      // [E + 1]
                       uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg * 8 + sg;
    if (r > E) return;
    float acc;
    if (r < E) {
        acc = qdot<8, D>(Wr, Sr, Zr, r, x, lane);
    } else {
        device const bfloat4 *w = (device const bfloat4 *)Wsg;
        device const half4 *xv = (device const half4 *)x;
        acc = 0;
        for (uint i = lane; i < D / 4; i += 32) acc += dot(float4(w[i]), float4(xv[i]));
    }
    acc = simd_sum(acc);
    if (lane == 0) logits[r] = acc;
}

// ---------------------------------------------------------------- 2. 選取
// 一個 SIMD group。輸出 sel[TOPK] 與 wts[TOPK + 1]（最後一個是共用 expert 的 sigmoid gate）
kernel void moe_select(device const float *logits [[buffer(0)]],
                       device uint *sel [[buffer(1)]],
                       device float *wts [[buffer(2)]],
                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PER = E / 32;
    float v[PER];
    for (uint j = 0; j < PER; j++) v[j] = logits[lane * PER + j];
    float picked[TOPK];
    uint ids[TOPK];
    for (uint k = 0; k < TOPK; k++) {
        float best = -INFINITY;
        uint bi = 0;
        for (uint j = 0; j < PER; j++)
            if (v[j] > best) { best = v[j]; bi = lane * PER + j; }
        const float m = simd_max(best);
        // 同值時取編號最小的
        const uint id = simd_min(best == m ? bi : 0xffffffffu);
        picked[k] = m;
        ids[k] = id;
        if (id / PER == lane) v[id % PER] = -INFINITY;
    }
    if (lane == 0) {
        float mx = picked[0], s = 0;
        for (uint k = 0; k < TOPK; k++) s += exp(picked[k] - mx);
        for (uint k = 0; k < TOPK; k++) {
            sel[k] = ids[k];
            wts[k] = exp(picked[k] - mx) / s;
        }
        wts[TOPK] = 1.0f / (1.0f + exp(-logits[E]));
    }
}


// 由 SIMD group 0 從 logits 選出前 TOPK 並算權重，寫進 threadgroup memory（取代單獨的 moe_select dispatch）
inline void select_topk(device const float *logits, threadgroup uint *ids_out, threadgroup float *w_out, uint sg, uint lane) {
    if (sg == 0) {
        constexpr uint PER = E / 32;
        float v[PER];
        for (uint j = 0; j < PER; j++) v[j] = logits[lane * PER + j];
        float picked[TOPK];
        uint ids[TOPK];
        for (uint k = 0; k < TOPK; k++) {
            float best = -INFINITY;
            uint bi = 0;
            for (uint j = 0; j < PER; j++)
                if (v[j] > best) { best = v[j]; bi = lane * PER + j; }
            const float m = simd_max(best);
            const uint id = simd_min(best == m ? bi : 0xffffffffu);
            picked[k] = m;
            ids[k] = id;
            if (id / PER == lane) v[id % PER] = -INFINITY;
        }
        if (lane == 0) {
            float mx = picked[0], s = 0;
            for (uint k = 0; k < TOPK; k++) s += exp(picked[k] - mx);
            for (uint k = 0; k < TOPK; k++) {
                ids_out[k] = ids[k];
                w_out[k] = exp(picked[k] - mx) / s;
            }
            w_out[TOPK] = 1.0f / (1.0f + exp(-logits[E]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------- 3. gate/up + SiLU
// grid：(I / (8·RPS), TOPK + 1)；slot < TOPK 是 routed（4-bit），slot == TOPK 是共用 expert（8-bit）
// 每個 SIMD group 算 RPS 個 j：gate 第 j 列與 up 第 j 列，輸出 H[slot, j] = SiLU(gate)·up
kernel void moe_gateup(device const uint *Wg [[buffer(0)]], device const bfloat *Sg [[buffer(1)]],
                       device const bfloat *Zg [[buffer(2)]],     // routed gate：[E, I, D] 4-bit
                       device const uint *Wu [[buffer(3)]], device const bfloat *Su [[buffer(4)]],
                       device const bfloat *Zu [[buffer(5)]],     // routed up
                       device const uint *Wsg [[buffer(6)]], device const bfloat *Ssg [[buffer(7)]],
                       device const bfloat *Zsg [[buffer(8)]],    // 共用 gate：[I, D] 8-bit
                       device const uint *Wsu [[buffer(9)]], device const bfloat *Ssu [[buffer(10)]],
                       device const bfloat *Zsu [[buffer(11)]],   // 共用 up
                       device const half *x [[buffer(12)]],
                       device const uint *sel [[buffer(13)]],
                       device half *H [[buffer(14)]], // [TOPK + 1, I]
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint slot = tg.y;
    const uint j0 = (tg.x * 8 + sg) * RPS;
    if (slot < TOPK) {
        const ulong e = sel[slot];
        for (uint r = 0; r < RPS; r++) {
            const uint j = j0 + r;
            const float gt = simd_sum(qdot<4, D>(Wg, Sg, Zg, e * I + j, x, lane));
            const float up = simd_sum(qdot<4, D>(Wu, Su, Zu, e * I + j, x, lane));
            if (lane == 0) H[slot * I + j] = half(gt / (1.0f + exp(-gt)) * up);
        }
    } else {
        for (uint r = 0; r < RPS; r++) {
            const uint j = j0 + r;
            const float gt = simd_sum(qdot<8, D>(Wsg, Ssg, Zsg, j, x, lane));
            const float up = simd_sum(qdot<8, D>(Wsu, Ssu, Zsu, j, x, lane));
            if (lane == 0) H[slot * I + j] = half(gt / (1.0f + exp(-gt)) * up);
        }
    }
}

// ---------------------------------------------------------------- 4. down + 加權合併
// grid：D / (8·RPS) 個 threadgroup；每個 SIMD group 算 RPS 個輸出列，在 kernel 內加總 11 個 expert
kernel void moe_down(device const uint *Wd [[buffer(0)]], device const bfloat *Sd [[buffer(1)]],
                     device const bfloat *Zd [[buffer(2)]],
                     device const uint *Wsd [[buffer(3)]], device const bfloat *Ssd [[buffer(4)]],
                     device const bfloat *Zsd [[buffer(5)]],
                     device const half *H [[buffer(6)]],
                     device const uint *sel [[buffer(7)]],
                     device const float *wts [[buffer(8)]],
                     device half *out [[buffer(9)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tg * 8 + sg) * RPS;
    for (uint r = 0; r < RPS; r++) {
        const uint row = r0 + r;
        float acc = 0;
        for (uint s = 0; s < TOPK; s++)
            acc += wts[s] * qdot<4, I>(Wd, Sd, Zd, (ulong)sel[s] * D + row, H + s * I, lane);
        acc += wts[TOPK] * qdot<8, I>(Wsd, Ssd, Zsd, row, H + TOPK * I, lane);
        acc = simd_sum(acc);
        if (lane == 0) out[row] = half(acc);
    }
}

// ---------------------------------------------------------------- 選取併入的版本（engine 用）
kernel void moe_gateup_f(device const uint *Wg [[buffer(0)]], device const bfloat *Sg [[buffer(1)]],
                       device const bfloat *Zg [[buffer(2)]],     // routed gate：[E, I, D] 4-bit
                       device const uint *Wu [[buffer(3)]], device const bfloat *Su [[buffer(4)]],
                       device const bfloat *Zu [[buffer(5)]],     // routed up
                       device const uint *Wsg [[buffer(6)]], device const bfloat *Ssg [[buffer(7)]],
                       device const bfloat *Zsg [[buffer(8)]],    // 共用 gate：[I, D] 8-bit
                       device const uint *Wsu [[buffer(9)]], device const bfloat *Ssu [[buffer(10)]],
                       device const bfloat *Zsu [[buffer(11)]],   // 共用 up
                       device const half *x [[buffer(12)]],
                       device const float *logits [[buffer(13)]],
                       device half *H [[buffer(14)]], // [TOPK + 1, I]
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sel[TOPK];
    threadgroup float wsel[TOPK + 1];
    select_topk(logits, sel, wsel, sg, lane);
    const uint slot = tg.y;
    const uint j0 = (tg.x * 8 + sg) * RPS;
    if (slot < TOPK) {
        const ulong e = sel[slot];
        for (uint r = 0; r < RPS; r++) {
            const uint j = j0 + r;
            float gt, up;
            qdot2<4, D>(Wg, Sg, Zg, e * I + j, Wu, Su, Zu, e * I + j, x, lane, gt, up);
            gt = simd_sum(gt);
            up = simd_sum(up);
            if (lane == 0) H[slot * I + j] = half(gt / (1.0f + exp(-gt)) * up);
        }
    } else {
        for (uint r = 0; r < RPS; r++) {
            const uint j = j0 + r;
            float gt, up;
            qdot2<8, D>(Wsg, Ssg, Zsg, j, Wsu, Ssu, Zsu, j, x, lane, gt, up);
            gt = simd_sum(gt);
            up = simd_sum(up);
            if (lane == 0) H[slot * I + j] = half(gt / (1.0f + exp(-gt)) * up);
        }
    }
}


kernel void moe_down_f(device const uint *Wd [[buffer(0)]], device const bfloat *Sd [[buffer(1)]],
                     device const bfloat *Zd [[buffer(2)]],
                     device const uint *Wsd [[buffer(3)]], device const bfloat *Ssd [[buffer(4)]],
                     device const bfloat *Zsd [[buffer(5)]],
                     device const half *H [[buffer(6)]],
                     device const float *logits [[buffer(7)]],
                     device half *out [[buffer(9)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sel[TOPK];
    threadgroup float wts[TOPK + 1];
    select_topk(logits, sel, wts, sg, lane);
    const uint r0 = (tg * 8 + sg) * RPS;
    for (uint r = 0; r < RPS; r++) {
        const uint row = r0 + r;
        float acc = 0;
        for (uint s = 0; s < TOPK; s++)
            acc += wts[s] * qdot<4, I>(Wd, Sd, Zd, (ulong)sel[s] * D + row, H + s * I, lane);
        acc += wts[TOPK] * qdot<8, I>(Wsd, Ssd, Zsd, row, H + TOPK * I, lane);
        acc = simd_sum(acc);
        if (lane == 0) out[row] = half(acc);
    }
}

// ---------------------------------------------------------------- 多 token（MTP 驗證）版本：NB 由主程式注入
#ifdef NB
// router：grid ((E + 1 + 7) / 8, NB) × 256；logits [NB][E + 1]
kernel void moe_router_b(device const uint *Wr [[buffer(0)]], device const bfloat *Sr [[buffer(1)]], device const bfloat *Zr [[buffer(2)]],
                         device const bfloat *Wsg [[buffer(3)]], device const half *x [[buffer(4)]], device float *logits [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x * 8 + sg, t = tg.y;
    if (r > E) return;
    device const half *xt = x + (ulong)t * D;
    float acc;
    if (r < E) acc = qdot<8, D>(Wr, Sr, Zr, r, xt, lane);
    else {
        device const bfloat4 *w = (device const bfloat4 *)Wsg;
        device const half4 *xv = (device const half4 *)xt;
        acc = 0;
        for (uint i = lane; i < D / 4; i += 32) acc += dot(float4(w[i]), float4(xv[i]));
    }
    acc = simd_sum(acc);
    if (lane == 0) logits[(ulong)t * (E + 1) + r] = acc;
}

// gate/up：grid (I / 8, NB · (TOPK + 1)) × 256；H [NB][TOPK + 1][I]
kernel void moe_gateup_fb(device const uint *Wg [[buffer(0)]], device const bfloat *Sg [[buffer(1)]], device const bfloat *Zg [[buffer(2)]],
                          device const uint *Wu [[buffer(3)]], device const bfloat *Su [[buffer(4)]], device const bfloat *Zu [[buffer(5)]],
                          device const uint *Wsg [[buffer(6)]], device const bfloat *Ssg [[buffer(7)]], device const bfloat *Zsg [[buffer(8)]],
                          device const uint *Wsu [[buffer(9)]], device const bfloat *Ssu [[buffer(10)]], device const bfloat *Zsu [[buffer(11)]],
                          device const half *x [[buffer(12)]], device const float *logits [[buffer(13)]], device half *H [[buffer(14)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sel[TOPK];
    threadgroup float wsel[TOPK + 1];
    const uint t = tg.y / (TOPK + 1), slot = tg.y % (TOPK + 1);
    select_topk(logits + (ulong)t * (E + 1), sel, wsel, sg, lane);
    device const half *xt = x + (ulong)t * D;
    device half *Ht = H + (ulong)t * (TOPK + 1) * I;
    const uint j = tg.x * 8 + sg;
    float gt, up;
    if (slot < TOPK) {
        const ulong e = sel[slot];
        qdot2<4, D>(Wg, Sg, Zg, e * I + j, Wu, Su, Zu, e * I + j, xt, lane, gt, up);
    } else
        qdot2<8, D>(Wsg, Ssg, Zsg, j, Wsu, Ssu, Zsu, j, xt, lane, gt, up);
    gt = simd_sum(gt);
    up = simd_sum(up);
    if (lane == 0) Ht[slot * I + j] = half(gt / (1.0f + exp(-gt)) * up);
}

// down + 加權合併：grid (D / 8, NB) × 256；out [NB][D]
kernel void moe_down_fb(device const uint *Wd [[buffer(0)]], device const bfloat *Sd [[buffer(1)]], device const bfloat *Zd [[buffer(2)]],
                        device const uint *Wsd [[buffer(3)]], device const bfloat *Ssd [[buffer(4)]], device const bfloat *Zsd [[buffer(5)]],
                        device const half *H [[buffer(6)]], device const float *logits [[buffer(7)]], device half *out [[buffer(9)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sel[TOPK];
    threadgroup float wts[TOPK + 1];
    const uint t = tg.y;
    select_topk(logits + (ulong)t * (E + 1), sel, wts, sg, lane);
    device const half *Ht = H + (ulong)t * (TOPK + 1) * I;
    const uint row = tg.x * 8 + sg;
    float acc = 0;
    for (uint s = 0; s < TOPK; s++) acc += wts[s] * qdot<4, I>(Wd, Sd, Zd, (ulong)sel[s] * D + row, Ht + s * I, lane);
    acc += wts[TOPK] * qdot<8, I>(Wsd, Ssd, Zsd, row, Ht + TOPK * I, lane);
    acc = simd_sum(acc);
    if (lane == 0) out[(ulong)t * D + row] = half(acc);
}

// ---- 驗證時 expert 去重：相鄰 token 常選到同樣的 expert（8 個 token 約 54/80 不重複），每個 expert 的權重只讀一次
#define UMAX (NB * TOPK)
struct moe_msel {
    uint U;                // 不重複的 expert 數
    uint ue[UMAX];         // 第 u 個不重複 expert 的編號（依第一次出現的順序）
    uint mask[UMAX];       // 哪些 token 選了它（bit t）
    uchar slot[UMAX][NB];  // token t 的第幾名（只在 mask 對應位元為 1 時有效）
    float w[NB][TOPK + 1]; // 各 token 的 routed 權重（softmax）與 shared gate（sigmoid）
};

// 一個 threadgroup，NB 個 SIMD group（各選一個 token 的前 TOPK）。grid：1 × 32·NB
kernel void moe_sel_b(device const float *logits [[buffer(0)]], device moe_msel *ms [[buffer(1)]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint ids[UMAX];
    threadgroup atomic_uint first[E];
    threadgroup uint lead[UMAX];
    for (uint i = tid; i < E; i += 32 * NB) atomic_store_explicit(&first[i], 0xffffffffu, memory_order_relaxed);
    {
        device const float *lg = logits + (ulong)sg * (E + 1);
        constexpr uint PER = E / 32;
        float v[PER];
        for (uint j = 0; j < PER; j++) v[j] = lg[lane * PER + j];
        float picked[TOPK];
        for (uint k = 0; k < TOPK; k++) {
            float best = -INFINITY;
            uint bi = 0;
            for (uint j = 0; j < PER; j++)
                if (v[j] > best) { best = v[j]; bi = lane * PER + j; }
            const float m = simd_max(best);
            const uint id = simd_min(best == m ? bi : 0xffffffffu);
            picked[k] = m;
            if (lane == 0) ids[sg * TOPK + k] = id;
            if (id / PER == lane) v[id % PER] = -INFINITY;
        }
        if (lane == 0) { // 同 select_topk
            float mx = picked[0], s = 0;
            for (uint k = 0; k < TOPK; k++) s += exp(picked[k] - mx);
            for (uint k = 0; k < TOPK; k++) ms->w[sg][k] = exp(picked[k] - mx) / s;
            ms->w[sg][TOPK] = 1.0f / (1.0f + exp(-lg[E]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < UMAX) atomic_fetch_min_explicit(&first[ids[tid]], tid, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < UMAX) lead[tid] = atomic_load_explicit(&first[ids[tid]], memory_order_relaxed) == tid ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < UMAX && lead[tid]) {
        uint u = 0;
        for (uint i = 0; i < tid; i++) u += lead[i];
        const uint e = ids[tid];
        uint m = 0;
        for (uint t = 0; t < NB; t++) ms->slot[u][t] = 0;
        for (uint i = tid; i < UMAX; i++)
            if (ids[i] == e) { m |= 1u << (i / TOPK); ms->slot[u][i / TOPK] = uchar(i % TOPK); }
        ms->ue[u] = e;
        ms->mask[u] = m;
    }
    if (tid == 0) {
        uint n = 0;
        for (uint i = 0; i < UMAX; i++) n += lead[i];
        ms->U = n;
    }
}

// gate/up：grid (I / 8, UMAX + 1) × 256；第 UMAX 列是 shared expert。H [NB][TOPK + 1][I]
kernel void moe_gateup_db(device const uint *Wg [[buffer(0)]], device const bfloat *Sg [[buffer(1)]], device const bfloat *Zg [[buffer(2)]],
                          device const uint *Wu [[buffer(3)]], device const bfloat *Su [[buffer(4)]], device const bfloat *Zu [[buffer(5)]],
                          device const uint *Wsg [[buffer(6)]], device const bfloat *Ssg [[buffer(7)]], device const bfloat *Zsg [[buffer(8)]],
                          device const uint *Wsu [[buffer(9)]], device const bfloat *Ssu [[buffer(10)]], device const bfloat *Zsu [[buffer(11)]],
                          device const half *x [[buffer(12)]], device const moe_msel *ms [[buffer(13)]], device half *H [[buffer(14)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    const uint u = tg.y, j = tg.x * 8 + sg;
    if (u == UMAX) {
        for (uint t = 0; t < NB; t++) {
            float gt, up;
            qdot2<8, D>(Wsg, Ssg, Zsg, j, Wsu, Ssu, Zsu, j, x + (ulong)t * D, lane, gt, up);
            gt = simd_sum(gt);
            up = simd_sum(up);
            if (lane == 0) H[((ulong)t * (TOPK + 1) + TOPK) * I + j] = half(gt / (1.0f + exp(-gt)) * up);
        }
        return;
    }
    if (u >= ms->U) return;
    const ulong e = ms->ue[u];
    const uint m = ms->mask[u];
    for (uint t = 0; t < NB; t++) {
        if (!((m >> t) & 1u)) continue;
        float gt, up;
        qdot2<4, D>(Wg, Sg, Zg, e * I + j, Wu, Su, Zu, e * I + j, x + (ulong)t * D, lane, gt, up);
        gt = simd_sum(gt);
        up = simd_sum(up);
        if (lane == 0) H[((ulong)t * (TOPK + 1) + ms->slot[u][t]) * I + j] = half(gt / (1.0f + exp(-gt)) * up);
    }
}

// down + 加權合併：grid D / 8 × 256；每個 SIMD group 一列，依序走過不重複的 expert。out [NB][D]
kernel void moe_down_db(device const uint *Wd [[buffer(0)]], device const bfloat *Sd [[buffer(1)]], device const bfloat *Zd [[buffer(2)]],
                        device const uint *Wsd [[buffer(3)]], device const bfloat *Ssd [[buffer(4)]], device const bfloat *Zsd [[buffer(5)]],
                        device const half *H [[buffer(6)]], device const moe_msel *ms [[buffer(7)]], device half *out [[buffer(9)]],
                        uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8 + sg;
    float acc[NB];
    for (uint t = 0; t < NB; t++) acc[t] = 0;
    const uint U = min(ms->U, (uint)UMAX);
    for (uint u = 0; u < U; u++) {
        const ulong e = ms->ue[u];
        const uint m = ms->mask[u];
#pragma unroll
        for (uint t = 0; t < NB; t++)
            if ((m >> t) & 1u) {
                const uint s = ms->slot[u][t];
                acc[t] += ms->w[t][s] * qdot<4, I>(Wd, Sd, Zd, e * D + row, H + ((ulong)t * (TOPK + 1) + s) * I, lane);
            }
    }
#pragma unroll
    for (uint t = 0; t < NB; t++) {
        float a = acc[t] + ms->w[t][TOPK] * qdot<8, I>(Wsd, Ssd, Zsd, row, H + ((ulong)t * (TOPK + 1) + TOPK) * I, lane);
        a = simd_sum(a);
        if (lane == 0) out[(ulong)t * D + row] = half(a);
    }
}
#endif
