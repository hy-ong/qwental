// 非有限值防護的 GPU 單元測試（不用模型）：用 engine 同樣的編譯選項（fast math）跑
//   1. hsat（prefill.metal）：範圍內的有限值與 half() 逐位元相同；超出範圍的夾到 ±65504（不是 inf）；印出 NaN 進去出來是什麼
//   2. k_nfscan（engine.metal）：乾淨的列不設旗標；任一位置有 inf / −inf / NaN 的列設旗標
//   3. i8_quant（qmm_i8.metal，QW_INT8 的 dense 輸入量化）：乾淨的輸入不設旗標；有 NaN / inf 的列設旗標。印出 NaN 列量化後的
//      scale 與值（fast math 下是有限的：這就是 QW_INT8 時 NaN 在 k_nfscan 看得到之前就消失的地方）
// 用法：build/nfcheck（全部通過回傳 0）。只用幾 KB 的 GPU 記憶體
#include "gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char src_prefill[] = {
#embed "../kernels/prefill.metal"
};
static const char src_engine[] = {
#embed "../kernels/engine.metal"
};
static const char src_qmm_i8[] = {
#embed "../kernels/qmm_i8.metal"
};
static const char t_kern[] = "kernel void t_hsat(device const float *x [[buffer(0)]], device half *y [[buffer(1)]], device half *z [[buffer(2)]],\n"
                             "                   uint i [[thread_position_in_grid]]) { y[i] = hsat(x[i]); z[i] = half(x[i]); }\n"
                             "kernel void t_hsat4(device const float4 *x [[buffer(0)]], device half4 *y [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
                             "    y[i] = hsat(x[i]);\n}\n";

static bool build(qw_gpu *g, const char *a, size_t na, const char *b, size_t nb) {
    const char *defs = "#define D 2560\n#define HC 4\n";
    const size_t nd = strlen(defs);
    char *s = malloc(nd + na + nb + 1);
    memcpy(s, defs, nd);
    memcpy(s + nd, a, na);
    memcpy(s + nd + na, b, nb);
    s[nd + na + nb] = 0;
    char err[8192];
    const bool ok = qw_gpu_compile(g, s, nd + na + nb, err, sizeof err);
    if (!ok) fprintf(stderr, "編譯失敗：%s\n", err);
    free(s);
    return ok;
}

static uint16_t h_bits(const qw_buf *b, int i) { return ((const uint16_t *)b->ptr)[i]; }

int main(void) {
    qw_gpu *g = qw_gpu_open();
    if (!g) { fprintf(stderr, "沒有 GPU\n"); return 1; }
    if (!build(g, src_prefill, sizeof src_prefill, t_kern, strlen(t_kern)) || !build(g, src_engine, sizeof src_engine, "", 0)) return 1;
    qw_kernel *kh = qw_kernel_get(g, "t_hsat"), *kh4 = qw_kernel_get(g, "t_hsat4"), *ks = qw_kernel_get(g, "k_nfscan");
    if (!kh || !kh4 || !ks) { fprintf(stderr, "找不到 kernel\n"); return 1; }
    int fail = 0;

    // ---- 1. hsat
    const float xs[] = {0.0f, -0.0f, 1.5f, -2.25f, 3.14159f, 6e-8f, 1e-10f, 1000.123f, 65504.0f, -65504.0f, 65519.99f, -65519.99f,
                        65520.0f, -65520.0f, 70000.0f, -70000.0f, 1e30f, -1e30f, INFINITY, -INFINITY, NAN};
    enum { NX = sizeof xs / sizeof *xs };
    qw_buf x, y, z, x4, y4;
    if (!qw_buf_new(g, 4 * 64, &x) || !qw_buf_new(g, 2 * 64, &y) || !qw_buf_new(g, 2 * 64, &z) || !qw_buf_new(g, 16 * 8, &x4) ||
        !qw_buf_new(g, 8 * 8, &y4))
        return 1;
    memcpy(x.ptr, xs, sizeof xs);
    memcpy(x4.ptr, xs, 16 * 5); // 前 20 個（5 組 float4）
    qw_batch *b = qw_batch_begin(g, false);
    qw_batch_kernel(b, kh);
    qw_batch_buf(b, 0, &x, 0);
    qw_batch_buf(b, 1, &y, 0);
    qw_batch_buf(b, 2, &z, 0);
    qw_batch_dispatch(b, 1, 1, 1, NX, 1, 1);
    qw_batch_kernel(b, kh4);
    qw_batch_buf(b, 0, &x4, 0);
    qw_batch_buf(b, 1, &y4, 0);
    qw_batch_dispatch(b, 1, 1, 1, 5, 1, 1);
    if (qw_batch_end(b) < 0) { fprintf(stderr, "GPU 失敗\n"); return 1; }
    for (int i = 0; i < NX; i++) {
        const uint16_t hs = h_bits(&y, i), hh = h_bits(&z, i), h4 = i < 20 ? h_bits(&y4, i) : hs;
        const float a = fabsf(xs[i]);
        bool ok;
        const char *want;
        if (isnan(xs[i])) { ok = true; want = "（NaN：只記錄）"; }
        else if (a < 65520.0f) { ok = hs == hh; want = "= half(x)"; }
        else { ok = hs == (xs[i] > 0 ? 0x7bff : 0xfbff); want = "= ±65504"; }
        ok = ok && h4 == hs;
        printf("  hsat(%-12g) = 0x%04x  half() = 0x%04x  hsat4 = 0x%04x  %s %s\n", xs[i], hs, hh, h4, want, ok ? "OK" : "❌");
        fail += !ok;
    }

    // ---- 2. k_nfscan：5 列 × 248320
    enum { V = 248320, R = 5 };
    qw_buf lg, fl;
    if (!qw_buf_new(g, (size_t)R * V * 4, &lg) || !qw_buf_new(g, 64 * 4, &fl)) return 1;
    float *L = lg.ptr;
    for (size_t i = 0; i < (size_t)R * V; i++) L[i] = (float)((int)(i % 2001) - 1000) * 0.037f;
    L[(size_t)1 * V + V - 1] = INFINITY; // 最後一個
    L[(size_t)2 * V + 0] = -INFINITY;    // 第一個
    L[(size_t)3 * V + 123457] = NAN;     // 中間
    L[(size_t)4 * V + 7] = 3.4e38f;      // 很大但有限：不算
    memset(fl.ptr, 0, fl.size);
    const uint32_t n = V;
    b = qw_batch_begin(g, false);
    qw_batch_kernel(b, ks);
    qw_batch_buf(b, 0, &lg, 0);
    qw_batch_buf(b, 1, &fl, 0);
    qw_batch_bytes(b, 2, &n, 4);
    qw_batch_dispatch(b, (V / 4 + 1023) / 1024, R, 1, 1024, 1, 1);
    if (qw_batch_end(b) < 0) { fprintf(stderr, "GPU 失敗\n"); return 1; }
    const uint32_t want[R] = {0, 1, 1, 1, 0};
    for (int r = 0; r < R; r++) {
        const uint32_t f = ((const uint32_t *)fl.ptr)[r];
        printf("  k_nfscan 第 %d 列：旗標 %u（應為 %u）%s\n", r, f, want[r], f == want[r] ? "OK" : "❌");
        fail += f != want[r];
    }
    // ---- 3. i8_quant：K = 2560，4 列（0 乾淨、1 有 NaN、2 有 +inf、3 有 −inf），先只量化第 0 列，再量化全部
    {
        enum { K = 2560, RQ = 4 };
        const char *qd = "#define KD 2560\n#define TM 64\n#define TN 64\n#define SG 4\n#define KC 0\n#define WT 1\n#define YT half\n"
                         "#define I8Q_NT 256\n#define i8_quant t_i8_quant\n#define qmm_i8 t_qmm_i8\n";
        if (!build(g, qd, strlen(qd), src_qmm_i8, sizeof src_qmm_i8)) return 1;
        qw_kernel *kq = qw_kernel_get(g, "t_i8_quant");
        qw_buf qx, qu, qs, qa, qf;
        if (!kq || !qw_buf_new(g, (size_t)RQ * K * 2, &qx) || !qw_buf_new(g, (size_t)RQ * K, &qu) || !qw_buf_new(g, RQ * 4, &qs) ||
            !qw_buf_new(g, RQ * 4, &qa) || !qw_buf_new(g, 16, &qf))
            return 1;
        _Float16 *X = qx.ptr;
        for (int i = 0; i < RQ * K; i++) X[i] = (_Float16)((float)((i * 37) % 201 - 100) * 0.05f);
        X[1 * K + 0] = (_Float16)NAN;       // QW_FAULT_NAN 的位置：一列的第一個值
        X[2 * K + K - 1] = (_Float16)INFINITY;
        X[3 * K + 1000] = (_Float16)-INFINITY;
        for (int pass = 0; pass < 2; pass++) {
            const int rows = pass ? RQ : 1;
            memset(qf.ptr, 0, qf.size);
            b = qw_batch_begin(g, false);
            qw_batch_kernel(b, kq);
            qw_batch_buf(b, 0, &qx, 0);
            qw_batch_buf(b, 1, &qu, 0);
            qw_batch_buf(b, 2, &qs, 0);
            qw_batch_buf(b, 3, &qa, 0);
            qw_batch_buf(b, 4, &qf, 0);
            qw_batch_dispatch(b, rows, 1, 1, 256, 1, 1);
            if (qw_batch_end(b) < 0) { fprintf(stderr, "GPU 失敗\n"); return 1; }
            const uint32_t f = *(const uint32_t *)qf.ptr, w = pass ? 1 : 0;
            printf("  i8_quant %s：旗標 %u（應為 %u）%s\n", pass ? "含 NaN / ±inf 的列" : "只有乾淨的列", f, w, f == w ? "OK" : "❌");
            fail += f != w;
        }
        const float *S = qs.ptr;
        const uint8_t *U = qu.ptr;
        for (int r = 0; r < RQ; r++)
            printf("  i8_quant 第 %d 列：scale %g（%s）、xu[0] = %d、xu[%d] = %d、xu[%d] = %d\n", r, S[r], isfinite(S[r]) ? "有限" : "非有限",
                   U[(size_t)r * K] - 128, 1000, U[(size_t)r * K + 1000] - 128, K - 1, U[(size_t)r * K + K - 1] - 128);
    }
    printf(fail ? "❌ %d 項失敗\n" : "全部通過\n", fail);
    qw_gpu_close(g);
    return fail ? 1 : 0;
}
