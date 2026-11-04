// DFlash2 drafter 的小 kernel（矩陣乘法用 qmvb / qmm）。數學對照 Strata 的 msl/dflash2.metal 與 tools/dflash2/dflash2_ref.py（MIT）。
// 主程式注入：H（2560）、HC（主模型殘差流數 4）、NH、NKV、HD（128）、GS（卷積分組 16）、RING（環狀 KV 槽數）、EB（主模型 embedding 位元數）
#include <metal_stdlib>
using namespace metal;

#define G (H / GS) // 卷積的組數

// 主模型特徵：第 slot 份 = 4 條殘差流的平均（h：[rows][HC·H] float）→ out [rows][ostride] half，位移 slot·H
kernel void df_tap(device const float *h [[buffer(0)]], device half *out [[buffer(1)]], constant uint &ostride [[buffer(2)]],
                   constant uint &slot [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, r = gid.y;
    if (i >= H) return;
    device const float *hr = h + (ulong)r * HC * H;
    float s = 0;
    for (uint c = 0; c < HC; c++) s += hr[c * H + i];
    out[(ulong)r * ostride + slot * H + i] = half(s / HC);
}

// embedding（主模型的量化表，MLX affine group 64）：rows 個 token → x [rows][H] float
kernel void df_embed(device const uint *W [[buffer(0)]], device const bfloat *S [[buffer(1)]], device const bfloat *Z [[buffer(2)]],
                     device const int *ids [[buffer(3)]], device float *x [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, r = gid.y;
    if (i >= H) return;
    const ulong t = (ulong)ids[r];
    constexpr uint VPW = 32 / EB, ROWW = H * EB / 32;
    const uint word = W[t * ROWW + i / VPW];
    const float q = float((word >> (EB * (i % VPW))) & ((1u << EB) - 1));
    x[(ulong)r * H + i] = q * float(S[t * (H / 64) + i / 64]) + float(Z[t * (H / 64) + i / 64]);
}

// RMSNorm：一列一個 threadgroup（256）。y float、yh half 都寫（不需要的傳同一個 buffer 即可）
kernel void df_rms(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],
                   device half *yh [[buffer(3)]], constant float &eps [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device const float *xr = x + (ulong)r * H;
    float ss = 0;
    for (uint i = tid; i < H; i += 256) ss = fma(xr[i], xr[i], ss);
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0;
    for (uint k = 0; k < 8; k++) t += part[k];
    const float rs = rsqrt(t / H + eps);
    for (uint i = tid; i < H; i += 256) {
        const float v = xr[i] * rs * w[i];
        y[(ulong)r * H + i] = v;
        yh[(ulong)r * H + i] = half(v);
    }
}

// 分組動態 2-tap 卷積（block 內沿列方向，第 0 列沒有前一列）：
//   out[b][c] = Σ_{o<2, b≥o} (base[o][c] + dyn[b][part·2G + o·G + c / GS]) · x[b − o][c]（+ resid）
// dyn：kernel_projection 的輸出 [rows][4G]（[2 部分][2 tap][G]）；part 0 = prep、1 = finish
kernel void df_conv(device const float *x [[buffer(0)]], device const float *dyn [[buffer(1)]], device const float *base [[buffer(2)]],
                    device const float *resid [[buffer(3)]], device float *out [[buffer(4)]], device half *outh [[buffer(5)]],
                    constant uint &part [[buffer(6)]], constant uint &has_resid [[buffer(7)]], uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, b = gid.y;
    if (c >= H) return;
    float acc = 0;
    for (uint o = 0; o < 2; o++) {
        if (b < o) continue;
        const float k = base[(part * 2 + o) * H + c] + dyn[(ulong)b * 4 * G + part * 2 * G + o * G + c / GS];
        acc = fma(k, x[(ulong)(b - o) * H + c], acc);
    }
    if (has_resid) acc += resid[(ulong)b * H + c];
    out[(ulong)b * H + c] = acc;
    outh[(ulong)b * H + c] = half(acc);
}

// 每個 head：RMSNorm（gamma [HD]）+ RoPE（位置 pos0 + 列號；rotate_half、theta）→ half，q、k、v 三份一次 dispatch：
// head 編號 [0, nq) = q（norm + RoPE，不進環）、
// [nq, nq + nk) = k（norm + RoPE）、[nq + nk, nq + 2nk) = v（只轉 half）；k、v 依 ring 寫環狀 KV 或 [rows][nk][HD]。
// nq = 0：上下文（只有 k、v）。grid：(nq + 2nk, rows) × HD
struct df_nr3_args { uint nq, nk, ring; float eps, theta; };
kernel void df_nr3(device const half *qs [[buffer(0)]], device const half *ks [[buffer(1)]], device const half *vs [[buffer(2)]],
                   device const float *qg [[buffer(3)]], device const float *kg [[buffer(4)]], device half *qd [[buffer(5)]],
                   device half *kd [[buffer(6)]], device half *vd [[buffer(7)]], constant uint &pos0 [[buffer(8)]],
                   constant df_nr3_args &a [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4], xs[HD];
    const uint r = tg.y;
    uint hh = tg.x, heads, ring, norm;
    device const half *src;
    device const float *gamma;
    device half *dst;
    if (hh < a.nq) { src = qs; gamma = qg; dst = qd; heads = a.nq; ring = 0; norm = 1; }
    else if (hh < a.nq + a.nk) { hh -= a.nq; src = ks; gamma = kg; dst = kd; heads = a.nk; ring = a.ring; norm = 1; }
    else { hh -= a.nq + a.nk; src = vs; gamma = kg; dst = vd; heads = a.nk; ring = a.ring; norm = 0; }
    const float x = float(src[((ulong)r * heads + hh) * HD + d]);
    float v = x;
    if (norm) {
        const float ss = simd_sum(x * x);
        if (lane == 0) part[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float t = part[0] + part[1] + part[2] + part[3];
        v = x * rsqrt(t / HD + a.eps) * gamma[d];
        xs[d] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint hd2 = HD / 2, j = d % hd2;
        const float f = float(pos0 + r) * pow(a.theta, -float(2 * j) / float(HD));
        const float cs = cos(f), sn = sin(f);
        const float rot = d < hd2 ? -xs[d + hd2] : xs[d - hd2];
        v = v * cs + rot * sn;
    }
    const ulong row = ring ? (ulong)((pos0 + r) % RING) : (ulong)r;
    dst[(row * heads + hh) * HD + d] = half(v);
}

// df_conv（has_resid = 1）接著 df_rms：一列一個 threadgroup（256）。out = 卷積 + 殘差（float），再 RMSNorm → y、yh。
// 算式與兩個 kernel 分開時相同（同樣的執行緒分工與加總順序）；out 與 resid 可以是同一個 buffer。grid：rows × 256
kernel void df_conv_rms(device const float *x [[buffer(0)]], device const float *dyn [[buffer(1)]], device const float *base [[buffer(2)]],
                        device const float *resid [[buffer(3)]], device float *out [[buffer(4)]], constant uint &part [[buffer(5)]],
                        device const float *w [[buffer(6)]], device float *y [[buffer(7)]], device half *yh [[buffer(8)]],
                        constant float &eps [[buffer(9)]], uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float pr[8];
    constexpr uint PT = (H + 255) / 256;
    float o[PT];
    float ss = 0;
    for (uint t = 0; t < PT; t++) {
        const uint c = tid + 256 * t;
        if (c >= H) break;
        float acc = 0;
        for (uint q = 0; q < 2; q++) {
            if (b < q) continue;
            const float k = base[(part * 2 + q) * H + c] + dyn[(ulong)b * 4 * G + part * 2 * G + q * G + c / GS];
            acc = fma(k, x[(ulong)(b - q) * H + c], acc);
        }
        acc += resid[(ulong)b * H + c];
        out[(ulong)b * H + c] = acc;
        o[t] = acc;
        ss = fma(acc, acc, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) pr[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tt = 0;
    for (uint k = 0; k < 8; k++) tt += pr[k];
    const float rs = rsqrt(tt / H + eps);
    for (uint t = 0; t < PT; t++) {
        const uint c = tid + 256 * t;
        if (c >= H) break;
        const float v = o[t] * rs * w[c];
        y[(ulong)b * H + c] = v;
        yh[(ulong)b * H + c] = half(v);
    }
}

// attention：每個 (block 列 b, q head h) 一個 threadgroup（256 = 8 個 SIMD group）。
// key：環狀 KV 中位置 [lo_b, s) 的（lo_b = max(lo, s + b − win + 1)）+ block 自己的 BS 列（全部可見）
// q：[BS][NH][HD] half；Kb、Vb：[BS][NKV][HD] half；o：[BS][NH][HD] half
kernel void df_attn(device const half *q [[buffer(0)]], device const half *Kr [[buffer(1)]], device const half *Vr [[buffer(2)]],
                    device const half *Kb [[buffer(3)]], device const half *Vb [[buffer(4)]], device half *o [[buffer(5)]],
                    constant uint4 &a [[buffer(6)]], // s、lo、win、BS
                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sm[8], sl[8], sacc[8][HD];
    const uint b = tg.x, hh = tg.y, kv = hh / (NH / NKV);
    const uint s = a.x, BS = a.w;
    const uint lo = max(a.y, s + b + 1 > a.z ? s + b + 1 - a.z : 0u);
    const uint nctx = s > lo ? s - lo : 0, ntot = nctx + BS;
    constexpr uint PL = HD / 32;
    float qv[PL], acc[PL];
    const float scale = rsqrt(float(HD));
    for (uint d = 0; d < PL; d++) { qv[d] = float(q[((ulong)b * NH + hh) * HD + lane * PL + d]) * scale; acc[d] = 0; }
    float m = -INFINITY, l = 0;
    for (uint i = sg; i < ntot; i += 8) {
        device const half *kr, *vr;
        if (i < nctx) {
            const ulong slot = (lo + i) % RING;
            kr = Kr + (slot * NKV + kv) * HD;
            vr = Vr + (slot * NKV + kv) * HD;
        } else {
            const ulong j = i - nctx;
            kr = Kb + (j * NKV + kv) * HD;
            vr = Vb + (j * NKV + kv) * HD;
        }
        float dd = 0;
        for (uint d = 0; d < PL; d++) dd += qv[d] * float(kr[lane * PL + d]);
        const float sc = simd_sum(dd);
        const float mn = max(m, sc), al = exp(m - mn), p = exp(sc - mn);
        l = l * al + p;
        for (uint d = 0; d < PL; d++) acc[d] = acc[d] * al + p * float(vr[lane * PL + d]);
        m = mn;
    }
    for (uint d = 0; d < PL; d++) sacc[sg][lane * PL + d] = acc[d];
    if (lane == 0) { sm[sg] = m; sl[sg] = l; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float M = -INFINITY;
        for (uint k = 0; k < 8; k++) M = max(M, sm[k]);
        float L = 0, r[PL];
        for (uint d = 0; d < PL; d++) r[d] = 0;
        for (uint k = 0; k < 8; k++) {
            if (sl[k] == 0) continue;
            const float w = exp(sm[k] - M);
            L += sl[k] * w;
            for (uint d = 0; d < PL; d++) r[d] += sacc[k][lane * PL + d] * w;
        }
        for (uint d = 0; d < PL; d++) o[((ulong)b * NH + hh) * HD + lane * PL + d] = half(r[d] / L);
    }
}

// attention（分段版）：每個 (kv head, 32 個 key 的一段) 一個 threadgroup（256），同一個 kv head 的 NH / NKV 個 q head ×
// BS 列 = 32 個 query 一起算：K 段整段搬進 threadgroup memory 一次（原本每個 (列, q head) 各讀一次、每個 key 一次 simd_sum），
// 各 query 在這段的 (max, Σexp, Σexp·V) 寫到 part，df_attn_merge 再合併。key 編號 i：[0, nctx0) 是環狀 KV 的位置 lo0 + i
// （lo0 = 第 0 列的下限，各列只看 ≥ lo_b 的），[nctx0, nctx0 + BS) 是 block 自己（全部可見）。可見性、算式同 df_attn。
// grid：(段數上限, NKV) × 256；超出 ntot 的段直接結束（段數由 GPU 端的 a 決定，CPU 只給上限）
#define AG (NH / NKV)
#define ACH 32 // 每段 key 數
#define APS (HD + 2) // part 每個 query 的 float 數：m、l、o[HD]
static_assert(HD == 128 && AG * 8 == 32, "df_attn_split：HD = 128、AG × BS = 32 個 query");
inline uint attn_lo(constant uint4 &a, uint b) { const uint s = a.x; return max(a.y, s + b + 1 > a.z ? s + b + 1 - a.z : 0u); }
kernel void df_attn_split(device const half *q [[buffer(0)]], device const half *Kr [[buffer(1)]], device const half *Vr [[buffer(2)]],
                          device const half *Kb [[buffer(3)]], device const half *Vb [[buffer(4)]], device float *part [[buffer(5)]],
                          constant uint4 &a [[buffer(6)]], // s、lo、win、BS（BS 必須是 8）
                          uint2 tg [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup half4 qs[32][HD / 4], ks[ACH][HD / 4];
    threadgroup float ps[32][ACH];
    const uint c = tg.x, kv = tg.y, s = a.x, lo0 = attn_lo(a, 0);
    const uint nctx = s > lo0 ? s - lo0 : 0, ntot = nctx + 8, i0 = c * ACH;
    if (i0 >= ntot) return;
    // q：query qi = b·AG + hl ↔ (列 b, head kv·AG + hl)；scale 先乘
    const float scale = rsqrt(float(HD));
    for (uint e = t; e < 32 * HD / 4; e += 256) {
        const uint qi = e / (HD / 4), d4 = e % (HD / 4), b = qi / AG, hl = qi % AG;
        qs[qi][d4] = half4(float4(((device const half4 *)(q + ((ulong)b * NH + kv * AG + hl) * HD))[d4]) * scale);
    }
    for (uint e = t; e < ACH * HD / 4; e += 256) {
        const uint j = e / (HD / 4), d4 = e % (HD / 4), i = i0 + j;
        half4 v = half4(0);
        if (i < nctx) v = ((device const half4 *)(Kr + (((ulong)(lo0 + i) % RING) * NKV + kv) * HD))[d4];
        else if (i < ntot) v = ((device const half4 *)(Kb + ((ulong)(i - nctx) * NKV + kv) * HD))[d4];
        ks[j][d4] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 分數：執行緒 t → query t / 8、key (t % 8)·4…+3
    const uint qi = t / 8, b = qi / AG, kq = (t % 8) * 4;
    const uint lob = attn_lo(a, b);
    float sc[4] = {0, 0, 0, 0};
    for (uint d4 = 0; d4 < HD / 4; d4++) {
        const float4 q4 = float4(qs[qi][d4]);
        for (uint u = 0; u < 4; u++) sc[u] += dot(q4, float4(ks[kq + u][d4]));
    }
    for (uint u = 0; u < 4; u++) {
        const uint i = i0 + kq + u;
        const bool vis = i < ntot && (i >= nctx || lo0 + i >= lob);
        sc[u] = vis ? sc[u] : -INFINITY;
    }
    float m = max(max(sc[0], sc[1]), max(sc[2], sc[3]));
    for (uint off = 1; off < 8; off <<= 1) m = max(m, simd_shuffle_xor(m, off)); // 同一個 query 的 8 個執行緒在同一個 SIMD group
    float l = 0;
    for (uint u = 0; u < 4; u++) {
        const float p = m == -INFINITY ? 0.0f : exp(sc[u] - m);
        ps[qi][kq + u] = p;
        l += p;
    }
    for (uint off = 1; off < 8; off <<= 1) l += simd_shuffle_xor(l, off);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // P·V：執行緒 t → query t / 8、維度 (t % 8)·16…+15
    const uint d0 = (t % 8) * 16;
    float4 o[4] = {0, 0, 0, 0};
    const uint nj = min((uint)ACH, ntot - i0);
    for (uint j = 0; j < nj; j++) {
        const uint i = i0 + j;
        device const half4 *vr = (device const half4 *)(i < nctx ? Vr + (((ulong)(lo0 + i) % RING) * NKV + kv) * HD
                                                                 : Vb + ((ulong)(i - nctx) * NKV + kv) * HD) + d0 / 4;
        const float p = ps[qi][j];
        for (uint u = 0; u < 4; u++) o[u] += p * float4(vr[u]);
    }
    device float *pp = part + (((ulong)kv * ng.x + c) * 32 + qi) * APS;
    if (t % 8 == 0) { pp[0] = m; pp[1] = l; }
    for (uint u = 0; u < 4; u++) ((device float4 *)(pp + 2 + d0))[u] = o[u];
}

// 合併各段：每個 (列 b, q head) 一個 threadgroup（HD 個執行緒）。nch = 段數上限（同 df_attn_split 的 grid）。
// 各段的權重 exp(m − M) 先算好放 threadgroup memory，之後每個執行緒（一維）逐段累加。grid：(BS, NH) × HD
kernel void df_attn_merge(device const float *part [[buffer(0)]], device half *o [[buffer(1)]], constant uint4 &a [[buffer(2)]],
                          constant uint &nch [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float w[HD], wl[HD];
    const uint b = tg.x, h = tg.y, kv = h / AG, qi = b * AG + h % AG, s = a.x, lo0 = attn_lo(a, 0);
    const uint ntot = (s > lo0 ? s - lo0 : 0) + 8, nc = (ntot + ACH - 1) / ACH; // ≤ HD（主程式保證 nch ≤ HD）
    device const float *p0 = part + ((ulong)kv * nch * 32 + qi) * APS; // 第 c 段在 p0 + c·32·APS
    float M = -INFINITY;
    for (uint c = lane; c < nc; c += 32) {
        device const float *pp = p0 + (ulong)c * 32 * APS;
        if (pp[1] > 0) M = max(M, pp[0]);
    }
    M = simd_max(M);
    if (d < nc) {
        device const float *pp = p0 + (ulong)d * 32 * APS;
        const float wt = pp[1] > 0 ? exp(pp[0] - M) : 0.0f;
        w[d] = wt;
        wl[d] = wt * pp[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float L = 0, r = 0;
    for (uint c = 0; c < nc; c++) {
        if (w[c] == 0) continue;
        L += wl[c];
        r += p0[(ulong)c * 32 * APS + 2 + d] * w[c];
    }
    o[((ulong)b * NH + h) * HD + d] = half(r / L);
}

// silu(g) · u（float 輸入）→ half
kernel void df_swiglu(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]], device half *y [[buffer(2)]],
                      constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) { const float x = g[i]; y[i] = half(x / (1.0f + exp(-x)) * u[i]); }
}

// 每列 logits [rows][nv] 的前 K 名（值大的先；同值取較小的 index）。grid：rows × 1024
kernel void df_topk(device const float *lg [[buffer(0)]], device float *tv [[buffer(1)]], device int *ti [[buffer(2)]],
                    constant uint &nv [[buffer(3)]], constant uint &K [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup int bi[32];
    threadgroup int picked[32];
    device const float *x = lg + (ulong)r * nv;
    for (uint k = 0; k < K; k++) {
        float best = -INFINITY;
        int bidx = 0x7fffffff;
        for (uint i = tid; i < nv; i += 1024) {
            const float v = x[i];
            bool skip = false;
            for (uint j = 0; j < k; j++) skip |= picked[j] == (int)i;
            if (skip) continue;
            if (v > best || (v == best && (int)i < bidx)) { best = v; bidx = (int)i; }
        }
        // SIMD 內取最大
        for (uint off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(best, off);
            const int oi = simd_shuffle_down(bidx, off);
            if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
        }
        if (lane == 0) { bv[sg] = best; bi[sg] = bidx; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float B = bv[0];
            int I = bi[0];
            for (uint j = 1; j < 32; j++)
                if (bv[j] > B || (bv[j] == B && bi[j] < I)) { B = bv[j]; I = bi[j]; }
            picked[k] = I;
            tv[r * K + k] = B;
            ti[r * K + k] = I;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// 兩階段 top-K：第一階段 grid (rows, NCHUNK) × 256，每段取前 K 名寫到 cv / ci [rows][NCHUNK·K]；
// 第二階段 df_topk2（n = NCHUNK·K，用 ci 換回 index）。同值取較小的 index，與單階段相同。
kernel void df_topk1(device const float *lg [[buffer(0)]], device float *cv [[buffer(1)]], device int *ci [[buffer(2)]],
                     constant uint &nv [[buffer(3)]], constant uint &K [[buffer(4)]], constant uint &nc [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // 每個 SIMD group 一段（nc 段），只用 SIMD 內的 shuffle：不需要 threadgroup 同步
    const uint r = tg.x, c = tg.y * 8 + sg;
    if (c >= nc) return;
    const uint per = (nv + nc - 1) / nc, i0 = c * per, i1 = min(i0 + per, nv);
    device const float *x = lg + (ulong)r * nv;
    float v[24];
    uint cntv = 0;
    for (uint i = i0 + lane; i < i1 && cntv < 24; i += 32) v[cntv++] = x[i];
    for (uint k = 0; k < K; k++) {
        float best = -INFINITY;
        int bidx = 0x7fffffff;
        for (uint j = 0; j < cntv; j++) {
            const int i = (int)(i0 + lane + j * 32);
            if (v[j] > best || (v[j] == best && i < bidx)) { best = v[j]; bidx = i; }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_xor(best, off);
            const int oi = simd_shuffle_xor(bidx, off);
            if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
        }
        if (lane == 0) { cv[((ulong)r * nc + c) * K + k] = best; ci[((ulong)r * nc + c) * K + k] = bidx; }
        for (uint j = 0; j < cntv; j++) if ((int)(i0 + lane + j * 32) == bidx) v[j] = -INFINITY;
    }
}

// 第二階段：n 個候選（≤ 32 × TK2_PL，用 ci 換回 index）的前 K 名（K ≤ 16）。grid：rows × 1024。
// 先每個 SIMD group 各取自己那份（≤ 32 × TK2_PL / 32 個）的前 K 名（只用 SIMD 內的 shuffle），放進 threadgroup memory，
// 再由第 0 個 SIMD group 從 32·K 個裡取前 K 名：全程只同步一次（原本 K 輪、每輪兩次 threadgroup 同步，約 90 µs）。
// 排序同單階段：值大的先、同值取較小的 index（各階段都是同一個全序，結果與一次取完相同）。
#define TK2_PL 8
inline void tk_best(thread const float *v, thread const int *idx, uint m, thread float &best, thread int &bidx) {
    best = -INFINITY;
    bidx = 0x7fffffff;
    for (uint j = 0; j < m; j++)
        if (v[j] > best || (v[j] == best && idx[j] < bidx)) { best = v[j]; bidx = idx[j]; }
    for (uint off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_xor(best, off);
        const int oi = simd_shuffle_xor(bidx, off);
        if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
    }
}
kernel void df_topk2(device const float *cv [[buffer(0)]], device const int *ci [[buffer(1)]], device float *tv [[buffer(2)]],
                     device int *ti [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &K [[buffer(5)]],
                     uint r [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[32 * 16];
    threadgroup int si[32 * 16];
    const uint per = (n + 31) / 32; // 每個 SIMD group 的候選數（≤ 32·TK2_PL：主程式保證 n ≤ 32·32·TK2_PL）
    float v[16];
    int idx[16];
    for (uint j = 0; j < TK2_PL; j++) {
        const uint o = lane + 32 * j, q = sg * per + o;
        const bool ok = o < per && q < n;
        v[j] = ok ? cv[(ulong)r * n + q] : -INFINITY;
        idx[j] = ok ? ci[(ulong)r * n + q] : 0x7fffffff;
    }
    for (uint k = 0; k < K; k++) {
        float best;
        int bidx;
        tk_best(v, idx, TK2_PL, best, bidx);
        if (lane == 0) { sv[sg * K + k] = best; si[sg * K + k] = bidx; }
        for (uint j = 0; j < TK2_PL; j++) if (idx[j] == bidx) v[j] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    for (uint j = 0; j < K; j++) { v[j] = sv[lane + 32 * j]; idx[j] = si[lane + 32 * j]; } // 32·K 個，每個 lane K 個
    for (uint k = 0; k < K; k++) {
        float best;
        int bidx;
        tk_best(v, idx, K, best, bidx);
        if (lane == 0) { tv[r * K + k] = best; ti[r * K + k] = bidx; }
        for (uint j = 0; j < K; j++) if (idx[j] == bidx) v[j] = -INFINITY;
    }
}

// 候選挑選：由左到右，sc_j = unary[i][j] + Σ_r B[cand_j][r] · A[pred][r] · hp[i][r]；選最大的，記下它的 softmax 機率
// cand 是 draft vocab 的 index，vocab 換回 token id。out：path [rows] int、prob [rows] float
// 一個 threadgroup，K 個 SIMD group（每個候選一個，同時算）；候選的 token id 先一次查好放 threadgroup memory。
// 每一列只剩一次依賴 pred 的讀取（A[pred]）。原本一個 SIMD group 逐一算 K 個候選，每個候選三次相依的讀取，一輪約 0.3 ms
// selector 的 codebook 一個值：BF16，或 selq4 打包檔的 Q4_0（每 32 個值 18 位元組：fp16 d、16 位元組 nibble；第 j < 16 個是低 4 位元，其餘高 4 位元；值 = d·(q − 8)）
#ifndef CBQ4
#define CBQ4 0
#endif
inline float cbv(device const uchar *cb, ulong row, uint r, uint R) {
#if CBQ4
    device const uchar *b = cb + row * (R / 32 * 18) + (r / 32) * 18;
    const float d = float(as_type<half>(ushort(b[0] | (b[1] << 8))));
    const uint j = r % 32, q = j < 16 ? (b[2 + j] & 15u) : (b[2 + j - 16] >> 4);
    return d * (float(q) - 8.0f);
#else
    return float(((device const bfloat *)cb)[row * R + r]);
#endif
}

kernel void df_select(device const float *unary [[buffer(0)]], device const int *cand [[buffer(1)]], device const uint *vocab [[buffer(2)]],
                      device const float *hp [[buffer(3)]], device const uchar *A [[buffer(4)]], device const uchar *Bc [[buffer(5)]],
                      device int *path [[buffer(6)]], device float *prob [[buffer(7)]], constant uint4 &a [[buffer(8)]], // anchor、rows、K、rank
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint tok[8 * 32];
    threadgroup float scs[32];
    const uint rows = a.y, K = a.z, R = a.w;
    for (uint q = tid; q < rows * K; q += 32 * K) tok[q] = vocab[cand[q]];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int pred = (int)a.x;
    for (uint i = 0; i < rows; i++) {
        const ulong tj = tok[i * K + sg];
        float d = 0;
        for (uint r = lane; r < R; r += 32) d += cbv(Bc, tj, r, R) * cbv(A, (ulong)pred, r, R) * hp[i * R + r];
        const float sv = unary[i * K + sg] + simd_sum(d);
        if (lane == 0) scs[sg] = sv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint bj = 0;
        for (uint j = 1; j < K; j++) if (scs[j] > scs[bj]) bj = j;
        pred = (int)tok[i * K + bj];
        if (tid == 0) {
            float z = 0;
            for (uint j = 0; j < K; j++) z += exp(scs[j] - scs[bj]);
            path[i] = pred;
            prob[i] = 1.0f / z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// 推測解碼一輪的接受判定（greedy；接在驗證之後、同一個 command buffer，猜下一塊不必等 CPU）：
// j = 從頭連續猜中幾個（res[i] == tok[p0 + 3 + i]）、nx = res[j]。寫入 drafter 的 anchor（ids[0] = nx）與參數：
// prm[0..3] = df_attn 的 (s, lo, win, BS)、prm[4] = df_nr3 的位置 s、prm[8..11] = df_select 的 (anchor, rows, K, rank)、
// prm[12] = j（給 CPU 核對）。s = p0 + j + 1，lo = max(s − win + 1, df_lo, 0)（同 CPU 的算法）。
// a：nd、p0、df_lo、win、BS、rows、K、rank。一個執行緒。
struct df_acc_args { uint nd, p0, df_lo, win, bs, rows, k, rank; };
kernel void df_accept(device const int *res [[buffer(0)]], device const int *tok [[buffer(1)]], device int *ids [[buffer(2)]],
                      device uint *prm [[buffer(3)]], constant df_acc_args &a [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint j = 0;
    while (j < a.nd && res[j] == tok[a.p0 + 3 + j]) j++;
    const int nx = res[j];
    const int s = int(a.p0 + j + 1);
    int lo = s - int(a.win) + 1 > int(a.df_lo) ? s - int(a.win) + 1 : int(a.df_lo);
    lo = lo > 0 ? lo : 0;
    ids[0] = nx;
    prm[0] = uint(s); prm[1] = uint(lo); prm[2] = a.win; prm[3] = a.bs;
    prm[4] = uint(s);
    prm[8] = uint(nx); prm[9] = a.rows; prm[10] = a.k; prm[11] = a.rank;
    prm[12] = j;
}
