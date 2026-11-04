// 驗證（一次 NB ≤ 8 個 token）的 MoE，用 tensor ops 的矩陣單元算（純量版每個 token、每個 expert 各解一次權重，
// NB 大時被算力卡住）。先由 moe_sel_b（moe_decode.metal）選好每個 token 的前 TOPK，並整理出不重複的 expert。
//   moe_gu_tc  ：每個不重複的 expert（和 shared expert）讀一次 gate / up，跟 8 列 x 相乘；
//                Hu[u][t] = 權重 · SiLU(gate)·up（token t 沒選這個 expert → 0）
//   moe_dn_tc  ：out[t] = Σ_u Wd[e_u] · Hu[u][t] + Wsd · Hu[UMAX][t]，整個累加在矩陣單元裡、順序固定
// 權重解碼直接進 cooperative tensor（同 qmvb_tc.metal）。x 與 Hu 的列 ≥ NB 是舊資料，輸出一律不寫、權重一律 0。
// 主程式注入：NB、MTOPK、MD（hidden）、MI（expert 中間維度）、NSG（不用 E / D / I：會跟標頭檔的 template 參數撞名）；名稱改名
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define GS 64
#define MT 8
#define NT 32
#define KT 32
#define UMAX (NB * MTOPK)
static_assert(NB >= 1 && NB <= MT, "moe_tc：NB 必須是 1–8");
static_assert(MD % (KT * NSG) == 0 && MI % KT == 0 && MI % NT == 0 && MD % NT == 0, "moe_tc：尺寸");

struct moe_msel { // 與 moe_decode.metal 相同
    uint U;
    uint ue[UMAX];
    uint mask[UMAX];
    uchar slot[UMAX][NB];
    float w[NB][MTOPK + 1];
};

using T2 = dextents<int32_t, 2>;
constant constexpr auto DESC = matmul2d_descriptor(MT, NT, KT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);

// 右運算元：W 的列 row0 + n（n < 32）、k0 起 32 個 k，MLX affine（group 64）解碼成 half
template <int BITS, int K, typename RB>
inline void fill_w(thread RB &rb, device const uint *W, device const bfloat *S, device const bfloat *Z, ulong row0, uint k0) {
    constexpr uint VPW = 32 / BITS, ROWW = K * BITS / 32, NG = K / GS;
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < rb.get_capacity(); i += 4) {
        const auto ix = rb.get_multidimensional_index(i); // (k, n)：連續 4 個 k（M5 的排列）
        const uint k = k0 + ix[0];
        const ulong row = row0 + ix[1];
        const uint word = W[row * ROWW + k / VPW] >> (BITS * (k % VPW));
        const float s = float(S[row * NG + k / GS]), z = float(Z[row * NG + k / GS]);
        if (BITS == 4) {
            rb[i] = half(float(word & 15) * s + z);
            rb[i + 1] = half(float((word >> 4) & 15) * s + z);
            rb[i + 2] = half(float((word >> 8) & 15) * s + z);
            rb[i + 3] = half(float((word >> 12) & 15) * s + z);
        } else {
            rb[i] = half(float(word & 255) * s + z);
            rb[i + 1] = half(float((word >> 8) & 255) * s + z);
            rb[i + 2] = half(float((word >> 16) & 255) * s + z);
            rb[i + 3] = half(float(word >> 24) * s + z);
        }
    }
}

template <typename ACC>
inline void acc_to_tg(thread ACC &acc, threadgroup float (*part)[NT]) {
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            const auto ix = acc.get_multidimensional_index(i);
            part[ix[1]][ix[0]] = acc[i];
        }
}

// gate / up 的 32 列 × 8 個 token：NSG 個 SIMD group 輪流分 K（MD），結果加總到 pg / pu
template <int BITS>
inline void gu_block(device const uint *Wg, device const bfloat *Sg, device const bfloat *Zg, device const uint *Wu,
                     device const bfloat *Su, device const bfloat *Zu, ulong row0, device const half *x, uint sg,
                     threadgroup float (*pg)[MT][NT], threadgroup float (*pu)[MT][NT]) {
    tensor<device half, T2, tensor_inline> tX((device half *)x, T2(MD, MT));
    matmul2d<DESC, execution_simdgroup> op;
    auto rb = op.get_right_input_cooperative_tensor<half, half, float>();
    auto x0 = tX.slice<KT, MT>(0, 0);
    auto ag = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(x0)>,
                                                    __tensor_ops_detail::__remove_addrspace_t<decltype(rb)>, float>();
    auto au = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(x0)>,
                                                    __tensor_ops_detail::__remove_addrspace_t<decltype(rb)>, float>();
    for (uint16_t i = 0; i < ag.get_capacity(); ++i) { ag[i] = 0; au[i] = 0; }
    for (uint k0 = sg * KT; k0 < MD; k0 += KT * NSG) {
        auto xs = tX.slice<KT, MT>(k0, 0);
        fill_w<BITS, MD>(rb, Wg, Sg, Zg, row0, k0);
        op.run(xs, rb, ag);
        fill_w<BITS, MD>(rb, Wu, Su, Zu, row0, k0);
        op.run(xs, rb, au);
    }
    acc_to_tg(ag, pg[sg]);
    acc_to_tg(au, pu[sg]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// grid：(MI / 32, UMAX + 1) × 32·NSG；第 UMAX 列是 shared expert（8-bit）。Hu [UMAX + 1][8][MI]
kernel void moe_gu_tc(device const uint *Wg [[buffer(0)]], device const bfloat *Sg [[buffer(1)]], device const bfloat *Zg [[buffer(2)]],
                      device const uint *Wu [[buffer(3)]], device const bfloat *Su [[buffer(4)]], device const bfloat *Zu [[buffer(5)]],
                      device const uint *Wsg [[buffer(6)]], device const bfloat *Ssg [[buffer(7)]], device const bfloat *Zsg [[buffer(8)]],
                      device const uint *Wsu [[buffer(9)]], device const bfloat *Ssu [[buffer(10)]], device const bfloat *Zsu [[buffer(11)]],
                      device const half *x [[buffer(12)]], device const moe_msel *ms [[buffer(13)]], device half *Hu [[buffer(14)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float pg[NSG][MT][NT], pu[NSG][MT][NT];
    const uint u = tg.y, j0 = tg.x * NT;
    const bool shared = u == UMAX;
    if (!shared && u >= ms->U) return;
    if (shared) gu_block<8>(Wsg, Ssg, Zsg, Wsu, Ssu, Zsu, j0, x, sg, pg, pu);
    else {
        const ulong e = ms->ue[u];
        gu_block<4>(Wg, Sg, Zg, Wu, Su, Zu, e * MI + j0, x, sg, pg, pu);
    }
    const uint m = shared ? 0xffu : ms->mask[u];
    for (uint q = tid; q < MT * NT; q += 32 * NSG) {
        const uint t = q / NT, n = q % NT;
        half h = 0;
        if (t < NB && ((m >> t) & 1u)) {
            float g = 0, up = 0;
            for (uint s = 0; s < NSG; s++) { g += pg[s][t][n]; up += pu[s][t][n]; }
            const float w = shared ? ms->w[t][MTOPK] : ms->w[t][ms->slot[u][t]];
            h = half(w * (g / (1.0f + exp(-g)) * up));
        }
        Hu[((ulong)u * MT + t) * MI + j0 + n] = h;
    }
}

// grid：MD / 32 × 32·NSG；SIMD group s 處理 u ≡ s（mod NSG），shared expert 給 SIMD group 0；最後加總
kernel void moe_dn_tc(device const uint *Wd [[buffer(0)]], device const bfloat *Sd [[buffer(1)]], device const bfloat *Zd [[buffer(2)]],
                      device const uint *Wsd [[buffer(3)]], device const bfloat *Ssd [[buffer(4)]], device const bfloat *Zsd [[buffer(5)]],
                      device const half *Hu [[buffer(6)]], device const moe_msel *ms [[buffer(7)]], device half *out [[buffer(9)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG][MT][NT];
    const uint n0 = tg * NT;
    matmul2d<DESC, execution_simdgroup> op;
    auto rb = op.get_right_input_cooperative_tensor<half, half, float>();
    tensor<device half, T2, tensor_inline> tH0((device half *)Hu, T2(MI, MT));
    auto h0 = tH0.slice<KT, MT>(0, 0);
    auto acc = op.get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(h0)>,
                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(rb)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    const uint U = min(ms->U, (uint)UMAX);
    for (uint u = sg; u < U; u += NSG) {
        const ulong e = ms->ue[u];
        tensor<device half, T2, tensor_inline> tH((device half *)Hu + (ulong)u * MT * MI, T2(MI, MT));
        for (uint k0 = 0; k0 < MI; k0 += KT) {
            auto hs = tH.slice<KT, MT>(k0, 0);
            fill_w<4, MI>(rb, Wd, Sd, Zd, e * MD + n0, k0);
            op.run(hs, rb, acc);
        }
    }
    if (sg == 0) {
        tensor<device half, T2, tensor_inline> tH((device half *)Hu + (ulong)UMAX * MT * MI, T2(MI, MT));
        for (uint k0 = 0; k0 < MI; k0 += KT) {
            auto hs = tH.slice<KT, MT>(k0, 0);
            fill_w<8, MI>(rb, Wsd, Ssd, Zsd, n0, k0);
            op.run(hs, rb, acc);
        }
    }
    acc_to_tg(acc, part[sg]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = tid; q < NB * NT; q += 32 * NSG) {
        const uint t = q / NT, n = q % NT;
        float v = 0;
        for (uint s = 0; s < NSG; s++) v += part[s][t][n];
        out[(ulong)t * MD + n0 + n] = half(v);
    }
}
