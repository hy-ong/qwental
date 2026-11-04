// Gated DeltaNet prefill 量測
//   1. 正確性：GPU 循序版 vs CPU 參考（double）
//   2. 速度：不同序列長度，換算成 36 層的總時間
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char gdn_src[] = {
#embed "../kernels/gdn.metal"
};

enum { DK = 128, DV = 128, HK = 16, HV = 48, LAYERS = 36 };

static qw_gpu *g;

static void die(const char *m) {
    fprintf(stderr, "錯誤：%s\n", m);
    exit(1);
}

static uint64_t rng = 0x2545f4914f6cdd1dull;
static double rndu(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (double)(rng >> 11) * (1.0 / 9007199254740992.0);
}
static uint16_t f2h(float f) {
    _Float16 h = (_Float16)f;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
static float h2f(uint16_t u) {
    _Float16 h;
    memcpy(&h, &u, 2);
    return (float)h;
}

typedef struct {
    int T;
    qw_buf q, k, v, a, beta, S, o;
} gdn_io;

// 產生跟真實模型相近的輸入：q、k 做 L2 normalize（q 再乘 1/√dk），
// a = log 衰減（約 0.9–1.0 的 g），beta 介於 0–1
static void gdn_alloc(gdn_io *io, int T) {
    io->T = T;
    size_t nqk = (size_t)T * HK * DK, nv = (size_t)T * HV * DV, ng = (size_t)T * HV;
    if (!qw_buf_new(g, nqk * 2, &io->q) || !qw_buf_new(g, nqk * 2, &io->k) || !qw_buf_new(g, nv * 2, &io->v) ||
        !qw_buf_new(g, ng * 4, &io->a) || !qw_buf_new(g, ng * 4, &io->beta) ||
        !qw_buf_new(g, (size_t)HV * DK * DV * 4, &io->S) || !qw_buf_new(g, nv * 2, &io->o))
        die("配置失敗");
    uint16_t *q = io->q.ptr, *k = io->k.ptr, *v = io->v.ptr;
    for (size_t r = 0; r < (size_t)T * HK; r++) {
        double nq = 0, nk = 0, tq[DK], tk[DK];
        for (int i = 0; i < DK; i++) {
            tq[i] = rndu() * 2 - 1;
            tk[i] = rndu() * 2 - 1;
            nq += tq[i] * tq[i];
            nk += tk[i] * tk[i];
        }
        nq = 1 / sqrt(nq) / sqrt((double)DK);
        nk = 1 / sqrt(nk);
        for (int i = 0; i < DK; i++) {
            q[r * DK + i] = f2h((float)(tq[i] * nq));
            k[r * DK + i] = f2h((float)(tk[i] * nk));
        }
    }
    for (size_t i = 0; i < nv; i++) v[i] = f2h((float)(rndu() * 2 - 1));
    float *a = io->a.ptr, *b = io->beta.ptr;
    for (size_t i = 0; i < ng; i++) {
        a[i] = (float)log(0.9 + 0.1 * rndu());
        b[i] = (float)(0.1 + 0.8 * rndu());
    }
    memset(io->S.ptr, 0, io->S.size);
}

static void gdn_free(gdn_io *io) {
    qw_buf_free(&io->q); qw_buf_free(&io->k); qw_buf_free(&io->v); qw_buf_free(&io->a);
    qw_buf_free(&io->beta); qw_buf_free(&io->S); qw_buf_free(&io->o);
}

// CPU 參考實作（double），只算前 nh 個 value head
static void gdn_ref(const gdn_io *io, int nh, float *o_ref, float *S_ref) {
    const uint16_t *q = io->q.ptr, *k = io->k.ptr, *v = io->v.ptr;
    const float *a = io->a.ptr, *b = io->beta.ptr;
    static double S[DK][DV];
    for (int hv = 0; hv < nh; hv++) {
        int hk = hv / (HV / HK);
        memset(S, 0, sizeof S);
        for (int t = 0; t < io->T; t++) {
            const uint16_t *kt = k + ((size_t)t * HK + hk) * DK, *qt = q + ((size_t)t * HK + hk) * DK;
            const uint16_t *vt = v + ((size_t)t * HV + hv) * DV;
            double gg = exp(a[(size_t)t * HV + hv]), bb = b[(size_t)t * HV + hv];
            for (int j = 0; j < DV; j++) {
                double mem = 0;
                for (int i = 0; i < DK; i++) {
                    S[i][j] *= gg;
                    mem += S[i][j] * h2f(kt[i]);
                }
                double d = bb * (h2f(vt[j]) - mem), out = 0;
                for (int i = 0; i < DK; i++) {
                    S[i][j] += h2f(kt[i]) * d;
                    out += S[i][j] * h2f(qt[i]);
                }
                o_ref[((size_t)t * nh + hv) * DV + j] = (float)out;
            }
        }
        for (int i = 0; i < DK; i++)
            for (int j = 0; j < DV; j++) S_ref[((size_t)hv * DK + i) * DV + j] = (float)S[i][j];
    }
}

enum { CH = 64, DVB = 32, SEG = 16384 };
static int DVB2 = 64; // scan2 每個 threadgroup 的 dv 欄數（可用環境變數 GDN_DVB 覆寫）

static qw_kernel *build(const char *kname, int nv) {
    char name[64];
    snprintf(name, sizeof name, "%s_nv%d_c%d_b%d", kname, nv, CH, (strstr(kname, "2") || strstr(kname, "3")) ? DVB2 : DVB);
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    size_t cap = sizeof gdn_src + 256;
    char *src = malloc(cap), err[8192];
    int n = snprintf(src, cap, "#define %s %s\n#define DK %d\n#define DV %d\n#define HK %d\n#define HV %d\n#define NV %d\n"
                     "#define C %d\n#define DVB %d\n%s",
                     kname, name, DK, DV, HK, HV, nv, CH, (strstr(kname, "2") || strstr(kname, "3")) ? DVB2 : DVB, getenv("GDN_EXTRA") ? getenv("GDN_EXTRA") : "");
    memcpy(src + n, gdn_src, sizeof gdn_src);
    if (!qw_gpu_compile(g, src, n + sizeof gdn_src, err, sizeof err)) {
        fprintf(stderr, "編譯失敗：\n%s\n", err);
        exit(1);
    }
    free(src);
    return qw_kernel_get(g, name);
}

static double run_seq(gdn_io *io, int nv) {
    qw_kernel *k = build("gdn_seq", nv);
    uint32_t T = (uint32_t)io->T;
    qw_batch *b = qw_batch_begin(g, false);
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, &io->q, 0);
    qw_batch_buf(b, 1, &io->k, 0);
    qw_batch_buf(b, 2, &io->v, 0);
    qw_batch_buf(b, 3, &io->a, 0);
    qw_batch_buf(b, 4, &io->beta, 0);
    qw_batch_buf(b, 5, &io->S, 0);
    qw_batch_buf(b, 6, &io->o, 0);
    qw_batch_bytes(b, 7, &T, 4);
    qw_batch_dispatch(b, HV, DV / nv, 1, 32u * nv, 1, 1);
    double t = qw_batch_end(b);
    if (t < 0) die("GPU 執行失敗");
    return t;
}

// 分塊版：prep + scan，回傳 GPU 時間（兩個 kernel 在同一個 command buffer）
static qw_buf cW, cU, cA, cG;
static size_t c_cap = 0;

static double run_chunk(gdn_io *io, double *t_prep) {
    uint32_t nchunk = (uint32_t)(io->T / CH);
    size_t need = (size_t)nchunk * HV * CH * DK * 4;
    if (need > c_cap) {
        if (c_cap) { qw_buf_free(&cW); qw_buf_free(&cU); qw_buf_free(&cA); qw_buf_free(&cG); }
        if (!qw_buf_new(g, need, &cW) || !qw_buf_new(g, need, &cU) ||
            !qw_buf_new(g, (size_t)nchunk * HV * CH * CH * 4, &cA) || !qw_buf_new(g, (size_t)nchunk * HV * CH * 4, &cG))
            die("配置分塊中間結果失敗");
        c_cap = need;
    }
    uint32_t args[2] = {(uint32_t)io->T, nchunk};
    qw_kernel *kp = build("gdn_prep", 1), *ks = build("gdn_scan", 1);
    qw_batch *b = qw_batch_begin(g, false);
    qw_batch_kernel(b, kp);
    qw_batch_buf(b, 0, &io->q, 0); qw_batch_buf(b, 1, &io->k, 0); qw_batch_buf(b, 2, &io->v, 0);
    qw_batch_buf(b, 3, &io->a, 0); qw_batch_buf(b, 4, &io->beta, 0);
    qw_batch_buf(b, 5, &cW, 0); qw_batch_buf(b, 6, &cU, 0); qw_batch_buf(b, 7, &cA, 0); qw_batch_buf(b, 8, &cG, 0);
    qw_batch_bytes(b, 9, args, sizeof args);
    qw_batch_dispatch(b, nchunk, HV, 1, 256, 1, 1);
    double tp = qw_batch_end(b);
    b = qw_batch_begin(g, false);
    qw_batch_kernel(b, ks);
    qw_batch_buf(b, 0, &io->q, 0); qw_batch_buf(b, 1, &io->k, 0);
    qw_batch_buf(b, 2, &cW, 0); qw_batch_buf(b, 3, &cU, 0); qw_batch_buf(b, 4, &cA, 0); qw_batch_buf(b, 5, &cG, 0);
    qw_batch_buf(b, 6, &io->S, 0); qw_batch_buf(b, 7, &io->o, 0);
    qw_batch_bytes(b, 8, args, sizeof args);
    qw_batch_dispatch(b, HV, DV / DVB, 1, 256, 1, 1);
    double ts = qw_batch_end(b);
    if (tp < 0 || ts < 0) die("GPU 執行失敗");
    if (t_prep) *t_prep = tp;
    return tp + ts;
}

// 分塊版 v2：每段 SEG 個 token 做一次 prep2 + scan2，狀態留在 S 接著往下傳
static qw_buf v2W, v2U, v2QA, v2K, v2G;
static bool v2_ready = false;

static double run_chunk2v(gdn_io *io, double *t_prep, const char *prep_name);
static double run_chunk2(gdn_io *io, double *t_prep) { return run_chunk2v(io, t_prep, "gdn_prep2"); }
static double run_chunk3(gdn_io *io, double *t_prep) { return run_chunk2v(io, t_prep, "gdn_prep3"); }

static double run_chunk2v(gdn_io *io, double *t_prep, const char *prep_name) {
    if (!v2_ready) {
        size_t nch = SEG / CH, per = nch * HV * CH;
        if (!qw_buf_new(g, per * DK * 2, &v2W) || !qw_buf_new(g, per * DV * 4, &v2U) ||
            !qw_buf_new(g, per * (DK + CH) * 2, &v2QA) || !qw_buf_new(g, per * DK * 2, &v2K) ||
            !qw_buf_new(g, nch * HV * 4, &v2G))
            die("配置 v2 中間結果失敗");
        v2_ready = true;
    }
    qw_kernel *kp = build(prep_name, 1), *ks = build("gdn_scan2", 1);
    double tp = 0, ts = 0;
    for (int t0 = 0; t0 < io->T; t0 += SEG) {
        int len = io->T - t0 < SEG ? io->T - t0 : SEG;
        uint32_t nchunk = (uint32_t)(len / CH), args[2] = {(uint32_t)len, nchunk}, tb = (uint32_t)t0;
        size_t oq = (size_t)t0 * HK * DK * 2, ov = (size_t)t0 * HV * DV * 2, og = (size_t)t0 * HV * 4;
        qw_batch *b = qw_batch_begin(g, false);
        qw_batch_kernel(b, kp);
        qw_batch_buf(b, 0, &io->q, oq); qw_batch_buf(b, 1, &io->k, oq); qw_batch_buf(b, 2, &io->v, ov);
        qw_batch_buf(b, 3, &io->a, og); qw_batch_buf(b, 4, &io->beta, og);
        qw_batch_buf(b, 5, &v2W, 0); qw_batch_buf(b, 6, &v2U, 0); qw_batch_buf(b, 7, &v2QA, 0);
        qw_batch_buf(b, 8, &v2K, 0); qw_batch_buf(b, 9, &v2G, 0);
        qw_batch_bytes(b, 10, args, sizeof args);
        qw_batch_dispatch(b, nchunk, HV, 1, 256, 1, 1);
        double t1 = qw_batch_end(b);
        b = qw_batch_begin(g, false);
        qw_batch_kernel(b, ks);
        qw_batch_buf(b, 0, &v2W, 0); qw_batch_buf(b, 1, &v2U, 0); qw_batch_buf(b, 2, &v2QA, 0);
        qw_batch_buf(b, 3, &v2K, 0); qw_batch_buf(b, 4, &v2G, 0);
        qw_batch_buf(b, 5, &io->S, 0); qw_batch_buf(b, 6, &io->o, 0);
        qw_batch_bytes(b, 7, args, sizeof args);
        qw_batch_bytes(b, 8, &tb, 4);
        qw_batch_buf(b, 9, &io->S, 0);
        { const uint32_t nosnap[6] = {0}; qw_batch_bytes(b, 10, nosnap, sizeof nosnap); }
        qw_batch_dispatch(b, HV, DV / DVB2, 1, 256, 1, 1);
        double t2 = qw_batch_end(b);
        if (t1 < 0 || t2 < 0) die("GPU 執行失敗");
        tp += t1;
        ts += t2;
    }
    if (t_prep) *t_prep = tp;
    return tp + ts;
}

// v3 + 重疊：同一個 command buffer、concurrent dispatch；
// 第 s 段的 scan 與第 s+1 段的 prep 同時跑（中間結果兩份輪流用），每輪之間一個 barrier
static qw_buf ovW[2], ovU[2], ovQA[2], ovK[2], ovG[2];
static bool ov_ready = false;

static void enc_prep(qw_batch *b, qw_kernel *kp, gdn_io *io, int t0, int len, int set) {
    uint32_t nchunk = (uint32_t)(len / CH), args[2] = {(uint32_t)len, nchunk};
    size_t oq = (size_t)t0 * HK * DK * 2, ov = (size_t)t0 * HV * DV * 2, og = (size_t)t0 * HV * 4;
    qw_batch_kernel(b, kp);
    qw_batch_buf(b, 0, &io->q, oq); qw_batch_buf(b, 1, &io->k, oq); qw_batch_buf(b, 2, &io->v, ov);
    qw_batch_buf(b, 3, &io->a, og); qw_batch_buf(b, 4, &io->beta, og);
    qw_batch_buf(b, 5, &ovW[set], 0); qw_batch_buf(b, 6, &ovU[set], 0); qw_batch_buf(b, 7, &ovQA[set], 0);
    qw_batch_buf(b, 8, &ovK[set], 0); qw_batch_buf(b, 9, &ovG[set], 0);
    qw_batch_bytes(b, 10, args, sizeof args);
    qw_batch_dispatch(b, nchunk, HV, 1, 256, 1, 1);
}

static void enc_scan(qw_batch *b, qw_kernel *ks, gdn_io *io, int t0, int len, int set) {
    uint32_t nchunk = (uint32_t)(len / CH), args[2] = {(uint32_t)len, nchunk}, tb = (uint32_t)t0;
    qw_batch_kernel(b, ks);
    qw_batch_buf(b, 0, &ovW[set], 0); qw_batch_buf(b, 1, &ovU[set], 0); qw_batch_buf(b, 2, &ovQA[set], 0);
    qw_batch_buf(b, 3, &ovK[set], 0); qw_batch_buf(b, 4, &ovG[set], 0);
    qw_batch_buf(b, 5, &io->S, 0); qw_batch_buf(b, 6, &io->o, 0);
    qw_batch_bytes(b, 7, args, sizeof args);
    qw_batch_bytes(b, 8, &tb, 4);
    qw_batch_buf(b, 9, &io->S, 0);
    { const uint32_t nosnap[6] = {0}; qw_batch_bytes(b, 10, nosnap, sizeof nosnap); }
    qw_batch_dispatch(b, HV, DV / DVB2, 1, 256, 1, 1);
}

static double run_chunk3o(gdn_io *io) {
    if (!ov_ready) {
        size_t nch = SEG / CH, per = nch * HV * CH;
        for (int i = 0; i < 2; i++)
            if (!qw_buf_new(g, per * DK * 2, &ovW[i]) || !qw_buf_new(g, per * DV * 4, &ovU[i]) ||
                !qw_buf_new(g, per * (DK + CH) * 2, &ovQA[i]) || !qw_buf_new(g, per * DK * 2, &ovK[i]) ||
                !qw_buf_new(g, nch * HV * 4, &ovG[i]))
                die("配置重疊用中間結果失敗");
        ov_ready = true;
    }
    qw_kernel *kp = build("gdn_prep3", 1), *ks = build("gdn_scan2", 1);
    int nseg = (io->T + SEG - 1) / SEG;
#define SEGLEN(s) (io->T - (s) * SEG < SEG ? io->T - (s) * SEG : SEG)
    qw_batch *b = qw_batch_begin(g, true);
    enc_prep(b, kp, io, 0, SEGLEN(0), 0);
    qw_batch_barrier(b);
    for (int s = 0; s < nseg; s++) {
        enc_scan(b, ks, io, s * SEG, SEGLEN(s), s & 1);
        if (s + 1 < nseg) enc_prep(b, kp, io, (s + 1) * SEG, SEGLEN(s + 1), (s + 1) & 1);
        qw_batch_barrier(b);
    }
#undef SEGLEN
    double t = qw_batch_end(b);
    if (t < 0) die("GPU 執行失敗");
    return t;
}

static void check(void) {
    printf("\n## 1. 正確性（T = 512，前 6 個 value head，對照 CPU double）\n\n");
    gdn_io io;
    gdn_alloc(&io, 512);
    const int nh = 6;
    float *oref = malloc((size_t)512 * nh * DV * 4), *sref = malloc((size_t)nh * DK * DV * 4);
    gdn_ref(&io, nh, oref, sref);
    run_seq(&io, 4);
    const uint16_t *o = io.o.ptr;
    const float *S = io.S.ptr;
    double eo = 0, es = 0, mo = 0;
    for (int t = 0; t < 512; t++)
        for (int hv = 0; hv < nh; hv++)
            for (int j = 0; j < DV; j++) {
                double r = oref[((size_t)t * nh + hv) * DV + j], x = h2f(o[((size_t)t * HV + hv) * DV + j]);
                eo = fmax(eo, fabs(x - r));
                mo = fmax(mo, fabs(r));
            }
    for (size_t i = 0; i < (size_t)nh * DK * DV; i++) es = fmax(es, fabs(S[i] - sref[i]));
    printf("- 輸出最大誤差 %.2e（輸出最大值 %.2f）、最終狀態最大誤差 %.2e\n", eo, mo, es);
    printf("- 循序版：%s\n", (eo < 2e-2 && es < 1e-3) ? "✅ 通過" : "❌ 不通過");
    memset(io.S.ptr, 0, io.S.size);
    run_chunk(&io, NULL);
    eo = es = 0;
    for (int t = 0; t < 512; t++)
        for (int hv = 0; hv < nh; hv++)
            for (int j = 0; j < DV; j++) {
                double r = oref[((size_t)t * nh + hv) * DV + j], x = h2f(o[((size_t)t * HV + hv) * DV + j]);
                eo = fmax(eo, fabs(x - r));
            }
    for (size_t i = 0; i < (size_t)nh * DK * DV; i++) es = fmax(es, fabs(S[i] - sref[i]));
    printf("- 分塊版：輸出最大誤差 %.2e、最終狀態最大誤差 %.2e → %s\n", eo, es,
           (eo < 2e-2 && es < 1e-3) ? "✅ 通過" : "❌ 不通過");
    memset(io.S.ptr, 0, io.S.size);
    run_chunk2(&io, NULL);
    eo = es = 0;
    for (int t = 0; t < 512; t++)
        for (int hv = 0; hv < nh; hv++)
            for (int j = 0; j < DV; j++) {
                double r = oref[((size_t)t * nh + hv) * DV + j], x = h2f(o[((size_t)t * HV + hv) * DV + j]);
                eo = fmax(eo, fabs(x - r));
            }
    for (size_t i = 0; i < (size_t)nh * DK * DV; i++) es = fmax(es, fabs(S[i] - sref[i]));
    printf("- 分塊版 v2（tensor 單元、FP16 運算元）：輸出最大誤差 %.2e、最終狀態最大誤差 %.2e → %s\n", eo, es,
           (eo < 2e-2 && es < 1e-3) ? "✅ 通過" : "❌ 不通過");
    memset(io.S.ptr, 0, io.S.size);
    run_chunk3(&io, NULL);
    eo = es = 0;
    for (int t = 0; t < 512; t++)
        for (int hv = 0; hv < nh; hv++)
            for (int j = 0; j < DV; j++) {
                double r = oref[((size_t)t * nh + hv) * DV + j], x = h2f(o[((size_t)t * HV + hv) * DV + j]);
                eo = fmax(eo, fabs(x - r));
            }
    for (size_t i = 0; i < (size_t)nh * DK * DV; i++) es = fmax(es, fabs(S[i] - sref[i]));
    printf("- 分塊版 v3（prep 也用 tensor 單元）：輸出最大誤差 %.2e、最終狀態最大誤差 %.2e → %s\n", eo, es,
           (eo < 2e-2 && es < 1e-3) ? "✅ 通過" : "❌ 不通過");
    memset(io.S.ptr, 0, io.S.size);
    run_chunk3o(&io);
    eo = es = 0;
    for (int t = 0; t < 512; t++)
        for (int hv = 0; hv < nh; hv++)
            for (int j = 0; j < DV; j++) {
                double r = oref[((size_t)t * nh + hv) * DV + j], x = h2f(o[((size_t)t * HV + hv) * DV + j]);
                eo = fmax(eo, fabs(x - r));
            }
    for (size_t i = 0; i < (size_t)nh * DK * DV; i++) es = fmax(es, fabs(S[i] - sref[i]));
    printf("- 分塊版 v3 + prep/scan 重疊：輸出最大誤差 %.2e、最終狀態最大誤差 %.2e → %s\n", eo, es,
           (eo < 2e-2 && es < 1e-3) ? "✅ 通過" : "❌ 不通過");
    free(oref);
    free(sref);
    gdn_free(&io);
}

static void speed(void) {
    printf("\n## 2. 速度（48 個 value head）\n\n");
    printf("| T | 版本 | 單層時間 | 每秒 token | 換算 36 層 |\n|---|---|---|---|---|\n");
    const int Ts[] = {4096, 16384, 131072};
    for (int i = 0; i < 3; i++) {
        gdn_io io;
        gdn_alloc(&io, Ts[i]);
        const int nvs[] = {1, 2, 4, 8};
        double best = 1e30;
        int bnv = 0;
        for (int j = 0; j < 4; j++) {
            double t = run_seq(&io, nvs[j]);
            if (t < best) {
                best = t;
                bnv = nvs[j];
            }
        }
        printf("| %d | 循序（每組 %d 欄） | %.1f ms | %.0f | %.2f 秒 |\n", Ts[i], bnv, best * 1e3, Ts[i] / best, best * LAYERS);
        double tp = 0, tc = run_chunk(&io, &tp);
        printf("| %d | 分塊 v1（prep %.1f ms） | %.1f ms | %.0f | %.2f 秒 |\n", Ts[i], tp * 1e3, tc * 1e3, Ts[i] / tc, tc * LAYERS);
        tc = run_chunk2(&io, &tp);
        printf("| %d | 分塊 v2（prep %.1f ms） | %.1f ms | %.0f | %.2f 秒 |\n", Ts[i], tp * 1e3, tc * 1e3, Ts[i] / tc, tc * LAYERS);
        tc = run_chunk3(&io, &tp);
        printf("| %d | 分塊 v3（prep %.1f ms） | %.1f ms | %.0f | %.2f 秒 |\n", Ts[i], tp * 1e3, tc * 1e3, Ts[i] / tc, tc * LAYERS);
        tc = run_chunk3o(&io);
        printf("| %d | 分塊 v3 + 重疊 | %.1f ms | %.0f | %.2f 秒 |\n", Ts[i], tc * 1e3, Ts[i] / tc, tc * LAYERS);
        fflush(stdout);
        gdn_free(&io);
    }
}

int main(void) {
    if (getenv("GDN_DVB")) DVB2 = atoi(getenv("GDN_DVB"));
    g = qw_gpu_open();
    if (!g) die("找不到 Metal 裝置");
    char pw[256];
    qw_power_desc(pw, sizeof pw);
    printf("# Gated DeltaNet prefill 量測\n\n- GPU：%s\n- %s\n- 形狀：HK=%d、HV=%d、DK=%d、DV=%d\n", qw_gpu_name(g), pw, HK,
           HV, DK, DV);
    check();
    speed();
    qw_gpu_close(g);
    return 0;
}
