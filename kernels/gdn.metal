// Gated DeltaNet（Qwen3.8-Flash-Next 的線性注意力）
//
// 每個 value head 有一個狀態 S ∈ R^{dk×dv}（FP32），每個 token：
//   S   = g_t · S                       （衰減，g_t = exp(a_t)，a_t ≤ 0）
//   mem = Sᵀ k_t                        （用 key 讀出目前記住的 value）
//   δ   = β_t · (v_t − mem)
//   S   = S + k_t δᵀ                    （delta rule 更新）
//   o_t = Sᵀ q_t
// q、k 已經過 L2 normalize（q 也已乘上 1/√dk）。GQA：每個 key head 對應 HV/HK 個 value head。
//
// 資料排列（batch = 1）：
//   q, k：[T, HK, DK] half      v, o：[T, HV, DV] half
//   a（log 衰減）、beta：[T, HV] float      S：[HV, DK, DV] float（就地更新）
//
// 主程式注入：DK、DV、HK、HV、NV（每個 threadgroup 處理幾個 dv 欄）

#include <metal_stdlib>
using namespace metal;

#define PER (DK / 32) // 每個 lane 負責幾個 dk

// value head → key head：MLX / HF 是分組順序（h / (HV/HK)）；GGUF 是交錯順序（h % HK，主程式定義 GDN_TILED 另編一份）
#ifdef GDN_TILED
#define HK_OF(h) ((h) % HK)
#else
#define HK_OF(h) ((h) / (HV / HK))
#endif

// ---------------------------------------------------------------- 循序版
// 跟 mlx-lm 的做法相同：一個 SIMD group 負責一個 (head, dv 欄)，32 個 lane 分攤 dk，
// 每個 token 依序更新。這是目前的基準線。
kernel void gdn_seq(device const half *q [[buffer(0)]],
                    device const half *k [[buffer(1)]],
                    device const half *v [[buffer(2)]],
                    device const float *a [[buffer(3)]],
                    device const float *beta [[buffer(4)]],
                    device float *S [[buffer(5)]],
                    device half *o [[buffer(6)]],
                    constant uint &T [[buffer(7)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint hv = tg.x;
    const uint dv = tg.y * NV + sg;
    const uint hk = HK_OF(hv);
    device float *Sh = S + (ulong)hv * DK * DV;

    float s[PER];
    for (int i = 0; i < PER; i++) s[i] = Sh[(lane * PER + i) * DV + dv];

    for (uint t = 0; t < T; t++) {
        const ulong qk = ((ulong)t * HK + hk) * DK + lane * PER;
        const ulong tv = (ulong)t * HV + hv;
        const float g = exp(a[tv]);
        const float b = beta[tv];
        const float vv = float(v[tv * DV + dv]);
        float kk[PER], qq[PER];
        float mem = 0;
        for (int i = 0; i < PER; i++) {
            kk[i] = float(k[qk + i]);
            qq[i] = float(q[qk + i]);
            s[i] *= g;
            mem += s[i] * kk[i];
        }
        mem = simd_sum(mem);
        const float d = b * (vv - mem);
        float out = 0;
        for (int i = 0; i < PER; i++) {
            s[i] += kk[i] * d;
            out += s[i] * qq[i];
        }
        out = simd_sum(out);
        if (lane == 0) o[tv * DV + dv] = half(out);
    }
    for (int i = 0; i < PER; i++) Sh[(lane * PER + i) * DV + dv] = s[i];
}

// ---------------------------------------------------------------- 分塊版（chunked，WY 表示法）
//
// 每 C 個 token 一塊。塊內令 γ_i = Π_{j≤i} g_j = exp(G_i)（G 是塊內 a 的前綴和），推導可得：
//   (I + L) Δ = diag(β)(V − diag(γ) K S₀)，其中 L_ij = β_i (k_i·k_j) e^{G_i−G_j}（j < i）
//   令 X = (I + L)⁻¹，則 Δ = U − W S₀，U = X diag(β) V，W = X diag(β γ) K
//   輸出：O = diag(γ) Q S₀ + A Δ，A_ij = (q_i·k_j) e^{G_i−G_j}（j ≤ i）
//   新狀態：S = γ_C S₀ + (diag(γ_C/γ) K)ᵀ Δ
// 第一步（prep）跟狀態無關，所有塊、所有 head 平行算；第二步（scan）每個 head 依序推進。
//
// 中間結果（每塊、每個 value head）：W [C, DK]、U [C, DV]、A [C, C]、G [C]，都是 float。
// 主程式另外注入：C（塊大小）、DVB（scan 時每個 threadgroup 負責幾個 dv 欄）

struct gdn_snap_args {
    uint n, stride; // 幾個檢查點、相鄰檢查點的間距（float 個數）
    uint chunk[4];  // 第幾塊結束時（1 起算）
};

struct gdn_chunk_args {
    uint T;       // token 數（必須是 C 的倍數）
    uint nchunk;  // T / C
};

// grid：(nchunk, HV)，每個 threadgroup 256 個執行緒
kernel void gdn_prep(device const half *q [[buffer(0)]],
                     device const half *k [[buffer(1)]],
                     device const half *v [[buffer(2)]],
                     device const float *a [[buffer(3)]],
                     device const float *beta [[buffer(4)]],
                     device float *Wo [[buffer(5)]],
                     device float *Uo [[buffer(6)]],
                     device float *Ao [[buffer(7)]],
                     device float *Go [[buffer(8)]],
                     constant gdn_chunk_args &args [[buffer(9)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NT = 256;
    const uint c = tg.x, hv = tg.y, hk = HK_OF(hv);
    const uint t0 = c * C;
    threadgroup float G[C], B[C];
    threadgroup float X[C][C]; // 先放 L，再就地換成 X = (I + L)⁻¹

    // 塊內前綴和（C ≤ 256，由執行緒 0 依序加，C 很小）
    if (tid < C) {
        B[tid] = beta[(ulong)(t0 + tid) * HV + hv];
        G[tid] = a[(ulong)(t0 + tid) * HV + hv];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0)
        for (uint i = 1; i < C; i++) G[i] += G[i - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong base = ((ulong)c * HV + hv);
    // L（嚴格下三角）與 A（含對角線的下三角）
    for (uint e = tid; e < C * C; e += NT) {
        const uint i = e / C, j = e % C;
        float kk = 0, qk = 0;
        if (j <= i) {
            device const half *ki = k + ((ulong)(t0 + i) * HK + hk) * DK;
            device const half *kj = k + ((ulong)(t0 + j) * HK + hk) * DK;
            device const half *qi = q + ((ulong)(t0 + i) * HK + hk) * DK;
            for (uint d = 0; d < DK; d++) {
                const float kjd = float(kj[d]);
                kk += float(ki[d]) * kjd;
                qk += float(qi[d]) * kjd;
            }
        }
        const float decay = j <= i ? exp(G[i] - G[j]) : 0.0f;
        X[i][j] = j < i ? B[i] * kk * decay : 0.0f;
        Ao[base * C * C + e] = qk * decay;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 前向代換：X_i = e_i − Σ_{j<i} L_ij X_j（第 i 列算完才覆蓋，L 的第 i 列在那之前都還在）
    for (uint i = 0; i < C; i++) {
        float val = 0;
        const uint col = tid;
        if (col < C) {
            val = (col == i) ? 1.0f : 0.0f;
            for (uint j = col; j < i; j++) val -= X[i][j] * X[j][col]; // X_j 的第 col 欄在 j < col 時為 0
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (col < C) X[i][col] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // W = X diag(β γ) K，U = X diag(β) V
    for (uint e = tid; e < C * DK; e += NT) {
        const uint i = e / DK, d = e % DK;
        float w = 0, u = 0;
        for (uint j = 0; j <= i; j++) {
            const float xb = X[i][j] * B[j];
            w += xb * exp(G[j]) * float(k[((ulong)(t0 + j) * HK + hk) * DK + d]);
            u += xb * float(v[((ulong)(t0 + j) * HV + hv) * DV + d]);
        }
        Wo[base * C * DK + e] = w;
        Uo[base * C * DV + e] = u; // DK == DV
    }
    if (tid < C) Go[base * C + tid] = G[tid];
}

// grid：(HV, DV / DVB)，每個 threadgroup 256 個執行緒，依序處理每一塊
kernel void gdn_scan(device const half *q [[buffer(0)]],
                     device const half *k [[buffer(1)]],
                     device const float *Wi [[buffer(2)]],
                     device const float *Ui [[buffer(3)]],
                     device const float *Ai [[buffer(4)]],
                     device const float *Gi [[buffer(5)]],
                     device float *S [[buffer(6)]],
                     device half *o [[buffer(7)]],
                     constant gdn_chunk_args &args [[buffer(8)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NT = 256;
    const uint hv = tg.x, hk = HK_OF(hv), dv0 = tg.y * DVB;
    threadgroup float Sb[DK][DVB];
    threadgroup float D[C][DVB];
    threadgroup float G[C];
    device float *Sh = S + (ulong)hv * DK * DV;

    for (uint e = tid; e < DK * DVB; e += NT) Sb[e / DVB][e % DVB] = Sh[(e / DVB) * DV + dv0 + e % DVB];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < args.nchunk; c++) {
        const uint t0 = c * C;
        const ulong base = ((ulong)c * HV + hv);
        if (tid < C) G[tid] = Gi[base * C + tid];
        // Δ = U − W S
        for (uint e = tid; e < C * DVB; e += NT) {
            const uint i = e / DVB, j = e % DVB;
            device const float *w = Wi + (base * C + i) * DK;
            float acc = Ui[(base * C + i) * DV + dv0 + j];
            for (uint d = 0; d < DK; d++) acc -= w[d] * Sb[d][j];
            D[i][j] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(γ) Q S + A Δ
        for (uint e = tid; e < C * DVB; e += NT) {
            const uint i = e / DVB, j = e % DVB;
            device const half *qi = q + ((ulong)(t0 + i) * HK + hk) * DK;
            float inter = 0;
            for (uint d = 0; d < DK; d++) inter += float(qi[d]) * Sb[d][j];
            float intra = 0;
            device const float *ar = Ai + (base * C + i) * C;
            for (uint m = 0; m <= i; m++) intra += ar[m] * D[m][j];
            o[((ulong)(t0 + i) * HV + hv) * DV + dv0 + j] = half(exp(G[i]) * inter + intra);
        }
        // S = γ_C S + (diag(γ_C/γ) K)ᵀ Δ（先算進暫存，等所有人讀完舊 S 再寫）
        constexpr uint PER_T = DK * DVB / NT;
        float ns[PER_T];
        const float gC = G[C - 1];
        for (uint r = 0; r < PER_T; r++) {
            const uint e = tid + r * NT, d = e / DVB, j = e % DVB;
            float acc = exp(gC) * Sb[d][j];
            for (uint m = 0; m < C; m++)
                acc += exp(gC - G[m]) * float(k[((ulong)(t0 + m) * HK + hk) * DK + d]) * D[m][j];
            ns[r] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < PER_T; r++) {
            const uint e = tid + r * NT;
            Sb[e / DVB][e % DVB] = ns[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint e = tid; e < DK * DVB; e += NT) Sh[(e / DVB) * DV + dv0 + e % DVB] = Sb[e / DVB][e % DVB];
}

// ---------------------------------------------------------------- 分塊版 v2：tensor 單元
//
// prep2（各塊平行）輸出，每塊、每個 value head：
//   Wh  [C, DK]      half ：W = X diag(βγ) K
//   Uf  [C, DV]      float：U = X diag(β) V
//   QA  [C, DK + C]  half ：[diag(γ) Q | A]，讓 O = [γQ | A] × [S ; Δ] 變成一個 K = DK + C 的矩陣乘法
//   Kt  [C, DK]      half ：diag(γ_C / γ) K（狀態更新用）
//   gC  [1]          float：γ_C（這一塊的總衰減）
// scan2（每個 head、每 DVB 個 dv 欄一個 threadgroup，依序推進）：
//   狀態 S 以 FP32 留在 cooperative tensor（暫存器）裡，每塊把 FP16 副本寫進 threadgroup memory 的 SD[0:DK]，
//   Δ 寫進 SD[DK:DK+C]，所以 SD 就是 [S ; Δ]。

#if defined(__HAVE_TENSOR__)
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

kernel void gdn_prep2(device const half *q [[buffer(0)]],
                      device const half *k [[buffer(1)]],
                      device const half *v [[buffer(2)]],
                      device const float *a [[buffer(3)]],
                      device const float *beta [[buffer(4)]],
                      device half *Wh [[buffer(5)]],
                      device float *Uf [[buffer(6)]],
                      device half *QA [[buffer(7)]],
                      device half *Kt [[buffer(8)]],
                      device float *gCo [[buffer(9)]],
                      constant gdn_chunk_args &args [[buffer(10)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NT = 256;
    const uint c = tg.x, hv = tg.y, hk = HK_OF(hv);
    const uint t0 = c * C;
    threadgroup float G[C], B[C];
    threadgroup float X[C][C];

    if (tid < C) {
        B[tid] = beta[(ulong)(t0 + tid) * HV + hv];
        G[tid] = a[(ulong)(t0 + tid) * HV + hv];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0)
        for (uint i = 1; i < C; i++) G[i] += G[i - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong base = ((ulong)c * HV + hv);
    device half *qa = QA + base * C * (DK + C);
    for (uint e = tid; e < C * C; e += NT) {
        const uint i = e / C, j = e % C;
        float kk = 0, qk = 0;
        if (j <= i) {
            device const half *ki = k + ((ulong)(t0 + i) * HK + hk) * DK;
            device const half *kj = k + ((ulong)(t0 + j) * HK + hk) * DK;
            device const half *qi = q + ((ulong)(t0 + i) * HK + hk) * DK;
            for (uint d = 0; d < DK; d++) {
                const float kjd = float(kj[d]);
                kk += float(ki[d]) * kjd;
                qk += float(qi[d]) * kjd;
            }
        }
        const float decay = j <= i ? exp(G[i] - G[j]) : 0.0f;
        X[i][j] = j < i ? B[i] * kk * decay : 0.0f;
        qa[i * (DK + C) + DK + j] = half(qk * decay);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < C; i++) {
        float val = 0;
        const uint col = tid;
        if (col < C) {
            val = (col == i) ? 1.0f : 0.0f;
            for (uint j = col; j < i; j++) val -= X[i][j] * X[j][col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (col < C) X[i][col] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float gC = G[C - 1];
    for (uint e = tid; e < C * DK; e += NT) {
        const uint i = e / DK, d = e % DK;
        float w = 0, u = 0;
        for (uint j = 0; j <= i; j++) {
            const float xb = X[i][j] * B[j];
            w += xb * exp(G[j]) * float(k[((ulong)(t0 + j) * HK + hk) * DK + d]);
            u += xb * float(v[((ulong)(t0 + j) * HV + hv) * DV + d]);
        }
        Wh[base * C * DK + e] = half(w);
        Uf[base * C * DV + e] = u;
        const ulong qk_idx = ((ulong)(t0 + i) * HK + hk) * DK + d;
        qa[i * (DK + C) + d] = half(exp(G[i]) * float(q[qk_idx]));
        Kt[base * C * DK + e] = half(exp(gC - G[i]) * float(k[qk_idx]));
    }
    if (tid == 0) gCo[base] = exp(gC);
}


// prep3：跟 prep2 相同的輸出，但 KKᵀ、QKᵀ、W、U 改用 tensor 單元。
// q、k、v 直接用步長讀原始排列（不先複製）。X 求完逆後轉成兩份 FP16（X·diag(β)、X·diag(βγ)）。
kernel void gdn_prep3(device half *q [[buffer(0)]],
                      device half *k [[buffer(1)]],
                      device half *v [[buffer(2)]],
                      device const float *a [[buffer(3)]],
                      device const float *beta [[buffer(4)]],
                      device half *Wh [[buffer(5)]],
                      device float *Uf [[buffer(6)]],
                      device half *QA [[buffer(7)]],
                      device half *Kt [[buffer(8)]],
                      device float *gCo [[buffer(9)]],
                      constant gdn_chunk_args &args [[buffer(10)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NT = 256;
    const uint c = tg.x, hv = tg.y, hk = HK_OF(hv);
    const uint t0 = c * C;
    threadgroup float G[C], B[C];
    threadgroup float Xf[C * C];                       // L → X（FP32）
    threadgroup half *Xb = (threadgroup half *)Xf;     // 之後重用：X·diag(β)
    threadgroup half *Xg = Xb + C * C;                 //           X·diag(βγ)

    // 塊內前綴和：兩個 SIMD group 各做 32 個，再補上前半段的總和
    if (tid < C) {
        B[tid] = beta[(ulong)(t0 + tid) * HV + hv];
        float x = a[(ulong)(t0 + tid) * HV + hv];
        G[tid] = simd_prefix_inclusive_sum(x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= 32 && tid < C) G[tid] += G[31];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong base = ((ulong)c * HV + hv);
    device half *qa = QA + base * C * (DK + C);
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tK(k + ((ulong)t0 * HK + hk) * DK, T2(DK, C), array<int, 2>{1, HK * DK});
    tensor<device half, T2, tensor_inline> tQ(q + ((ulong)t0 * HK + hk) * DK, T2(DK, C), array<int, 2>{1, HK * DK});
    tensor<device half, T2, tensor_inline> tV(v + ((ulong)t0 * HV + hv) * DV, T2(DV, C), array<int, 2>{1, HV * DV});

    // KKᵀ 與 QKᵀ（[C × DK] × [DK × C]，右邊轉置）
    constexpr auto dKK = matmul2d_descriptor(C, C, DK, false, true);
    matmul2d<dKK, execution_simdgroups<8>> opKK;
    {
        auto kk = opKK.get_destination_cooperative_tensor<decltype(tK), decltype(tK), float>();
#pragma unroll
        for (uint16_t i = 0; i < kk.get_capacity(); ++i)
            if (kk.is_valid_element(i)) kk[i] = 0;
        opKK.run(tK, tK, kk);
#pragma unroll
        for (uint16_t i = 0; i < kk.get_capacity(); ++i)
            if (kk.is_valid_element(i)) {
                auto ix = kk.get_multidimensional_index(i);
                const uint r = ix[1], j = ix[0];
                Xf[r * C + j] = j < r ? B[r] * kk[i] * exp(G[r] - G[j]) : 0.0f;
            }
        auto qk = opKK.get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
#pragma unroll
        for (uint16_t i = 0; i < qk.get_capacity(); ++i)
            if (qk.is_valid_element(i)) qk[i] = 0;
        opKK.run(tQ, tK, qk);
#pragma unroll
        for (uint16_t i = 0; i < qk.get_capacity(); ++i)
            if (qk.is_valid_element(i)) {
                auto ix = qk.get_multidimensional_index(i);
                const uint r = ix[1], j = ix[0];
                qa[r * (DK + C) + DK + j] = half(j <= r ? qk[i] * exp(G[r] - G[j]) : 0.0f);
            }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 區塊求逆 X = (I + L)⁻¹，區塊大小 16（C = 64 → 4×4 個區塊）
    // 1) 對角區塊：SIMD group b 負責第 b 個，16 步前向代換，只需 SIMD group 內同步
    {
        constexpr uint NB = 16;
        if (sg < C / NB && lane < NB) {
            const uint r0 = sg * NB, col = lane;
            for (uint i2 = 0; i2 < NB; i2++) {
                float val = (col == i2) ? 1.0f : 0.0f;
                for (uint j = col; j < i2; j++) val -= Xf[(r0 + i2) * C + r0 + j] * Xf[(r0 + j) * C + r0 + col];
                simdgroup_barrier(mem_flags::mem_threadgroup);
                Xf[(r0 + i2) * C + r0 + col] = val;
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 2) 非對角區塊：X_IJ = −X_II · Σ_{K=J}^{I−1} L_IK X_KJ
        //    依列區塊 I 由上到下、同一列內 J 由左到右；X_IJ 寫回 (I,J) 之後，
        //    同一列後面的區塊只會用到 K ≥ J' > J 的 L，所以就地覆寫是安全的。
        const uint r = tid / NB, cc = tid % NB; // 每個執行緒負責區塊內的一個元素
        for (uint I = 1; I < C / NB; I++)
            for (uint J = 0; J < I; J++) {
                float t = 0;
                for (uint K = J; K < I; K++) {
                    const uint kk0 = K * NB;
                    for (uint m = 0; m < NB; m++)
                        t += Xf[(I * NB + r) * C + kk0 + m] * Xf[(kk0 + m) * C + J * NB + cc];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                Xf[(I * NB + r) * C + J * NB + cc] = t; // 暫存 T_IJ
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float x = 0;
                for (uint m = 0; m < NB; m++) x -= Xf[(I * NB + r) * C + I * NB + m] * Xf[(I * NB + m) * C + J * NB + cc];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                Xf[(I * NB + r) * C + J * NB + cc] = x;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
    }
    // X 轉成兩份 FP16（先全部讀進暫存器，再覆蓋同一塊記憶體）
    constexpr uint XPER = C * C / NT;
    float xr[XPER];
    for (uint r = 0; r < XPER; r++) xr[r] = Xf[tid + r * NT];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < XPER; r++) {
        const uint e = tid + r * NT, j = e % C;
        Xb[e] = half(xr[r] * B[j]);
        Xg[e] = half(xr[r] * B[j] * exp(G[j]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // W = Xg K、U = Xb V（[C × C] × [C × DK]）
    tensor<threadgroup half, T2, tensor_inline> tXb(Xb, T2(C, C));
    tensor<threadgroup half, T2, tensor_inline> tXg(Xg, T2(C, C));
    constexpr auto dW = matmul2d_descriptor(C, DK, C);
    matmul2d<dW, execution_simdgroups<8>> opW;
    {
        auto w = opW.get_destination_cooperative_tensor<decltype(tXg), decltype(tK), float>();
#pragma unroll
        for (uint16_t i = 0; i < w.get_capacity(); ++i)
            if (w.is_valid_element(i)) w[i] = 0;
        opW.run(tXg, tK, w);
#pragma unroll
        for (uint16_t i = 0; i < w.get_capacity(); ++i)
            if (w.is_valid_element(i)) {
                auto ix = w.get_multidimensional_index(i);
                Wh[base * C * DK + ix[1] * DK + ix[0]] = half(w[i]);
            }
        auto u = opW.get_destination_cooperative_tensor<decltype(tXb), decltype(tV), float>();
#pragma unroll
        for (uint16_t i = 0; i < u.get_capacity(); ++i)
            if (u.is_valid_element(i)) u[i] = 0;
        opW.run(tXb, tV, u);
        tensor<device float, T2, tensor_inline> tU(Uf + base * C * DV, T2(DV, C));
        u.store(tU);
    }

    // [γQ | …] 的前半與 diag(γ_C/γ) K（逐元素）
    const float gC = G[C - 1];
    for (uint e = tid; e < C * DK; e += NT) {
        const uint i = e / DK, d = e % DK;
        const ulong idx = ((ulong)(t0 + i) * HK + hk) * DK + d;
        qa[i * (DK + C) + d] = half(exp(G[i]) * float(q[idx]));
        Kt[base * C * DK + e] = half(exp(gC - G[i]) * float(k[idx]));
    }
    if (tid == 0) gCo[base] = exp(gC);
}

// grid：(HV, DV / DVB)，每個 threadgroup 8 個 SIMD group（256 個執行緒）
kernel void gdn_scan2(device half *Wh [[buffer(0)]],      // tensor 需要非 const 指標
                      device float *Uf [[buffer(1)]],
                      device half *QA [[buffer(2)]],
                      device half *Kt [[buffer(3)]],
                      device const float *gCi [[buffer(4)]],
                      device float *S [[buffer(5)]],
                      device half *o [[buffer(6)]],
                      constant gdn_chunk_args &args [[buffer(7)]],
                      constant uint &t_base [[buffer(8)]],
                      device float *Ssnap [[buffer(9)]],               // 段中檢查點：第 k 個存在 Ssnap + k * snap.stride
                      constant gdn_snap_args &snap [[buffer(10)]],
                      uint2 tg [[threadgroup_position_in_grid]]) {
    const uint hv = tg.x, dv0 = tg.y * DVB;
    threadgroup half SD[(DK + C) * DVB]; // 列 0..DK-1 是 S（FP16 副本），列 DK.. 是 Δ
    threadgroup half *Db = SD + DK * DVB;

    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tS(SD, dextents<int32_t, 2>(DVB, DK));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tSD(SD, dextents<int32_t, 2>(DVB, DK + C));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tD(Db, dextents<int32_t, 2>(DVB, C));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tSdev(S + (ulong)hv * DK * DV, dextents<int32_t, 2>(DV, DK));
    auto tSblk = tSdev.slice(dv0, 0);

    // Δ = U − W S：[C × DK] × [DK × DVB]
    constexpr auto dD = matmul2d_descriptor(C, DVB, DK);
    matmul2d<dD, execution_simdgroups<8>> opD;
    // O = [γQ | A] × [S ; Δ]：[C × (DK + C)] × [(DK + C) × DVB]
    constexpr auto dO = matmul2d_descriptor(C, DVB, DK + C);
    matmul2d<dO, execution_simdgroups<8>> opO;
    // S += Ktᵀ Δ：[DK × C] × [C × DVB]，左邊轉置
    constexpr auto dS = matmul2d_descriptor(DK, DVB, C, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroups<8>> opS;

    using tKt_t = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    auto sT = opS.get_destination_cooperative_tensor<tKt_t, decltype(tD), float>();
    sT.load(tSblk);
#pragma unroll
    for (uint16_t i = 0; i < sT.get_capacity(); ++i)
        if (sT.is_valid_element(i)) {
            auto ix = sT.get_multidimensional_index(i);
            SD[ix[1] * DVB + ix[0]] = half(sT[i]);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < args.nchunk; c++) {
        const ulong base = ((ulong)c * HV + hv);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tW(Wh + base * C * DK, dextents<int32_t, 2>(DK, C));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> tU(Uf + base * C * DV, dextents<int32_t, 2>(DV, C));
        auto tUblk = tU.slice(dv0, 0);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tQA(QA + base * C * (DK + C), dextents<int32_t, 2>(DK + C, C));
        tKt_t tK(Kt + base * C * DK, dextents<int32_t, 2>(DK, C));

        // 1. Δ = U − W S，寫進 SD 的下半部
        auto dT = opD.get_destination_cooperative_tensor<decltype(tW), decltype(tS), float>();
#pragma unroll
        for (uint16_t i = 0; i < dT.get_capacity(); ++i)
            if (dT.is_valid_element(i)) dT[i] = 0;
        opD.run(tW, tS, dT);
        auto uT = opD.get_destination_cooperative_tensor<decltype(tW), decltype(tS), float>();
        uT.load(tUblk);
#pragma unroll
        for (uint16_t i = 0; i < dT.get_capacity(); ++i)
            if (dT.is_valid_element(i)) {
                auto ix = dT.get_multidimensional_index(i);
                Db[ix[1] * DVB + ix[0]] = half(uT[i] - dT[i]);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. O = [γQ | A] × [S ; Δ]
        auto oT = opO.get_destination_cooperative_tensor<decltype(tQA), decltype(tSD), float>();
#pragma unroll
        for (uint16_t i = 0; i < oT.get_capacity(); ++i)
            if (oT.is_valid_element(i)) oT[i] = 0;
        opO.run(tQA, tSD, oT);
#pragma unroll
        for (uint16_t i = 0; i < oT.get_capacity(); ++i)
            if (oT.is_valid_element(i)) {
                auto ix = oT.get_multidimensional_index(i);
                const ulong t = t_base + c * C + ix[1];
                o[(t * HV + hv) * DV + dv0 + ix[0]] = half(oT[i]);
            }

        // 3. S = γ_C S + Ktᵀ Δ
        const float gC = gCi[base];
#pragma unroll
        for (uint16_t i = 0; i < sT.get_capacity(); ++i)
            if (sT.is_valid_element(i)) sT[i] *= gC;
        opS.run(tK, tD, sT);
        threadgroup_barrier(mem_flags::mem_threadgroup); // 所有人都讀完舊的 S、Δ
#pragma unroll
        for (uint16_t i = 0; i < sT.get_capacity(); ++i)
            if (sT.is_valid_element(i)) {
                auto ix = sT.get_multidimensional_index(i);
                SD[ix[1] * DVB + ix[0]] = half(sT[i]);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < snap.n; k++) // 這一塊結束時的狀態 = 某個檢查點
            if (c + 1 == snap.chunk[k]) {
                tensor<device float, dextents<int32_t, 2>, tensor_inline> tK2(Ssnap + (ulong)k * snap.stride + (ulong)hv * DK * DV, dextents<int32_t, 2>(DV, DK));
                auto blk = tK2.slice(dv0, 0);
                sT.store(blk);
            }
    }
    sT.store(tSblk);
}
#endif
