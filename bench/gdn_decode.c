// Gated DeltaNet decode 單步量測：36 層各自的狀態（合計約 113 MB，大於 SLC），每個 token 依序更新
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char src_[] = {
#embed "../kernels/gdn_decode.metal"
};

enum { DK = 128, DV = 128, HK = 16, HV = 48, LAYERS = 36 };
static qw_gpu *g;
static int DVB = 32;

static void die(const char *m) { fprintf(stderr, "錯誤：%s\n", m); exit(1); }
static uint64_t rng = 0x8badf00ddeadbeefull;
static double rndu(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (double)(rng >> 11) / 9007199254740992.0; }
static uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float h2f(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float)h; }

static qw_kernel *K_(void) {
    char name[64], defs[256], err[8192];
    snprintf(name, sizeof name, "gdn_step_b%d", DVB);
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    int n = snprintf(defs, sizeof defs, "#define gdn_step %s\n#define DK %d\n#define DV %d\n#define HK %d\n#define HV %d\n#define DVB %d\n", name, DK, DV, HK, HV, DVB);
    char *s = malloc(n + sizeof src_);
    memcpy(s, defs, n); memcpy(s + n, src_, sizeof src_);
    if (!qw_gpu_compile(g, s, n + sizeof src_, err, sizeof err)) { fprintf(stderr, "%s\n", err); exit(1); }
    free(s);
    return qw_kernel_get(g, name);
}

static qw_buf q, k, v, a, be, S, o, ssp;

static void enc(qw_batch *b, int layer) {
    qw_batch_kernel(b, K_());
    qw_batch_buf(b, 0, &q, 0); qw_batch_buf(b, 1, &k, 0); qw_batch_buf(b, 2, &v, 0);
    qw_batch_buf(b, 3, &a, 0); qw_batch_buf(b, 4, &be, 0);
    qw_batch_buf(b, 5, &S, (size_t)layer * HV * DK * DV * 4); qw_batch_buf(b, 6, &o, 0); qw_batch_buf(b, 7, &ssp, 0); qw_batch_buf(b, 8, &S, (size_t)layer * HV * DK * DV * 4);
    qw_batch_dispatch(b, HV, DV / DVB, 1, 256, 1, 1);
}

int main(void) {
    if (getenv("GDN_DVB")) DVB = atoi(getenv("GDN_DVB"));
    g = qw_gpu_open();
    char pw[256];
    qw_power_desc(pw, sizeof pw);
    printf("# GDN decode 單步量測\n\n- GPU：%s\n- %s\n- DVB = %d\n", qw_gpu_name(g), pw, DVB);
    size_t ns = (size_t)HV * DK * DV;
    if (!qw_buf_new(g, HK * DK * 2, &q) || !qw_buf_new(g, HK * DK * 2, &k) || !qw_buf_new(g, HV * DV * 2, &v) ||
        !qw_buf_new(g, HV * 4, &a) || !qw_buf_new(g, HV * 4, &be) || !qw_buf_new(g, ns * 4 * LAYERS, &S) ||
        !qw_buf_new(g, HV * DV * 2, &o) || !qw_buf_new(g, HV * 16 * 4, &ssp)) die("配置失敗");
    uint16_t *pq = q.ptr, *pk = k.ptr, *pv = v.ptr;
    for (int h = 0; h < HK; h++) {
        double nq = 0, nk = 0, tq[DK], tk[DK];
        for (int i = 0; i < DK; i++) { tq[i] = rndu() * 2 - 1; tk[i] = rndu() * 2 - 1; nq += tq[i] * tq[i]; nk += tk[i] * tk[i]; }
        for (int i = 0; i < DK; i++) { pq[h * DK + i] = f2h((float)(tq[i] / sqrt(nq) / sqrt(DK))); pk[h * DK + i] = f2h((float)(tk[i] / sqrt(nk))); }
    }
    for (int i = 0; i < HV * DV; i++) pv[i] = f2h((float)(rndu() * 2 - 1));
    float *pa = a.ptr, *pb = be.ptr, *ps = S.ptr;
    for (int i = 0; i < HV; i++) { pa[i] = (float)log(0.9 + 0.1 * rndu()); pb[i] = (float)(0.1 + 0.8 * rndu()); }
    for (size_t i = 0; i < ns * LAYERS; i++) ps[i] = (float)(rndu() * 0.2 - 0.1);

    // 正確性：第 0 層，對照 CPU
    double *S0 = malloc(ns * sizeof(double));
    for (size_t i = 0; i < ns; i++) S0[i] = ps[i];
    qw_batch *b = qw_batch_begin(g, false);
    enc(b, 0);
    if (qw_batch_end(b) < 0) die("GPU 執行失敗");
    double eo = 0, es = 0;
    const uint16_t *po = o.ptr;
    for (int hv = 0; hv < HV; hv++) {
        int hk = hv / (HV / HK);
        double gg = exp(pa[hv]), bb = pb[hv];
        for (int j = 0; j < DV; j++) {
            double mem = 0, out = 0;
            for (int i = 0; i < DK; i++) { S0[((size_t)hv * DK + i) * DV + j] *= gg; mem += S0[((size_t)hv * DK + i) * DV + j] * h2f(pk[hk * DK + i]); }
            double d = bb * (h2f(pv[hv * DV + j]) - mem);
            for (int i = 0; i < DK; i++) { double *s = &S0[((size_t)hv * DK + i) * DV + j]; *s += h2f(pk[hk * DK + i]) * d; out += *s * h2f(pq[hk * DK + i]); }
            eo = fmax(eo, fabs(out - h2f(po[hv * DV + j])));
        }
    }
    for (size_t i = 0; i < ns; i++) es = fmax(es, fabs(S0[i] - ps[i]));
    printf("\n## 1. 正確性（對照 CPU double）\n\n- 輸出最大誤差 %.1e、狀態最大誤差 %.1e → %s\n", eo, es, (eo < 1e-3 && es < 1e-5) ? "✅ 通過" : "❌ 不通過");

    // 速度：每個 token 依序跑 36 層
    const int reps = 50;
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        b = qw_batch_begin(g, false);
        for (int r = 0; r < reps; r++) for (int l = 0; l < LAYERS; l++) enc(b, l);
        double tt = qw_batch_end(b);
        if (tt > 0 && tt < best) best = tt;
    }
    double per_tok = best / reps, bytes = (double)ns * 4 * 2 * LAYERS;
    printf("\n## 2. 速度\n\n- 36 層每個 token %.1f µs（每層 %.2f µs），狀態讀寫 %.0f MB → %.0f GB/s\n", per_tok * 1e6, per_tok / LAYERS * 1e6,
           bytes / 1e6, bytes / per_tok / 1e9);
    qw_gpu_close(g);
    return 0;
}
