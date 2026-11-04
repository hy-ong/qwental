// Decode 合併 MoE 量測：router → 選取 → gate/up + SiLU → down + 加權合併
// 形狀：E = 512、TOPK = 10、hidden 2560、expert 中間維度 640；routed 4-bit、共用 expert 8-bit、router BF16
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char moe_src[] = {
#embed "../kernels/moe_decode.metal"
};
static const char mem_src[] = {
#embed "../kernels/ceiling_mem.metal"
};

enum { E = 512, TOPK = 10, D = 2560, I = 640, NX = 64 };

static qw_gpu *g;
static int RPS_GU = 1, RPS_DN = 2; // 掃描過的最佳設定

static void die(const char *m) {
    fprintf(stderr, "錯誤：%s\n", m);
    exit(1);
}
static uint64_t rng = 0x5851f42d4c957f2dull;
static uint32_t rnd32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}
static float rndf(void) { return (rnd32() >> 8) * (1.0f / 16777216.0f) * 2 - 1; }
static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
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

static qw_kernel *K_(const char *kname) {
    char name[64];
    snprintf(name, sizeof name, "%s_r%d_%d", kname, RPS_GU, RPS_DN);
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    char defs[512], err[8192];
    int rps = !strcmp(kname, "moe_down") ? RPS_DN : RPS_GU;
    int n = snprintf(defs, sizeof defs, "#define %s %s\n#define E %d\n#define TOPK %d\n#define D %d\n#define I %d\n#define RPS %d\n",
                     kname, name, E, TOPK, D, I, rps);
    char *src = malloc(n + sizeof moe_src);
    memcpy(src, defs, n);
    memcpy(src + n, moe_src, sizeof moe_src);
    if (!qw_gpu_compile(g, src, n + sizeof moe_src, err, sizeof err)) {
        fprintf(stderr, "編譯失敗：\n%s\n", err);
        exit(1);
    }
    free(src);
    k = qw_kernel_get(g, name);
    if (!k) die(name);
    return k;
}

// 量化矩陣（含多個 expert）
typedef struct { qw_buf W, S, Z; int bits, N, K, ne; } qm;

static void qm_new(qm *q, int bits, int N, int K, int ne) {
    *q = (qm){{0}, {0}, {0}, bits, N, K, ne};
    size_t w = (size_t)ne * N * K * bits / 8, s = (size_t)ne * N * (K / 64) * 2;
    if (!qw_buf_new(g, w, &q->W) || !qw_buf_new(g, s, &q->S) || !qw_buf_new(g, s, &q->Z)) die("配置權重失敗");
    uint32_t *pw = q->W.ptr;
    for (size_t i = 0; i < w / 4; i++) pw[i] = rnd32();
    uint16_t *ps = q->S.ptr, *pz = q->Z.ptr;
    float qmax = (float)((1 << bits) - 1);
    for (size_t i = 0; i < s / 2; i++) {
        float sc = (0.5f + 0.25f * (rndf() + 1)) * 0.06f / qmax;
        ps[i] = f2bf(sc);
        pz[i] = f2bf(-sc * qmax / 2);
    }
}
static size_t qm_row_bytes(const qm *q) { return (size_t)q->K * q->bits / 8 + (size_t)(q->K / 64) * 4; }
static float qm_w(const qm *q, size_t row, int k) {
    const uint32_t *w = q->W.ptr;
    const uint16_t *s = q->S.ptr, *z = q->Z.ptr;
    int vpw = 32 / q->bits;
    uint32_t word = w[row * (q->K * q->bits / 32) + k / vpw];
    int v = (int)((word >> (q->bits * (k % vpw))) & ((1u << q->bits) - 1));
    return v * bf2f(s[row * (q->K / 64) + k / 64]) + bf2f(z[row * (q->K / 64) + k / 64]);
}

static qm Wg, Wu, Wsg, Wsu, Wd, Wsd, Wr; // 照 checkpoint：gate/up 分開、router 8-bit
static qw_buf Wsgate, X, logits, sel, wts, H, out;

static void encode(qw_batch *b, size_t xoff) {
    qw_batch_kernel(b, K_("moe_router"));
    qw_batch_buf(b, 0, &Wr.W, 0); qw_batch_buf(b, 1, &Wr.S, 0); qw_batch_buf(b, 2, &Wr.Z, 0);
    qw_batch_buf(b, 3, &Wsgate, 0); qw_batch_buf(b, 4, &X, xoff); qw_batch_buf(b, 5, &logits, 0);
    qw_batch_dispatch(b, (E + 1 + 7) / 8, 1, 1, 256, 1, 1);
    qw_batch_kernel(b, K_("moe_select"));
    qw_batch_buf(b, 0, &logits, 0); qw_batch_buf(b, 1, &sel, 0); qw_batch_buf(b, 2, &wts, 0);
    qw_batch_dispatch(b, 1, 1, 1, 32, 1, 1);
    qw_batch_kernel(b, K_("moe_gateup"));
    qw_batch_buf(b, 0, &Wg.W, 0); qw_batch_buf(b, 1, &Wg.S, 0); qw_batch_buf(b, 2, &Wg.Z, 0);
    qw_batch_buf(b, 3, &Wu.W, 0); qw_batch_buf(b, 4, &Wu.S, 0); qw_batch_buf(b, 5, &Wu.Z, 0);
    qw_batch_buf(b, 6, &Wsg.W, 0); qw_batch_buf(b, 7, &Wsg.S, 0); qw_batch_buf(b, 8, &Wsg.Z, 0);
    qw_batch_buf(b, 9, &Wsu.W, 0); qw_batch_buf(b, 10, &Wsu.S, 0); qw_batch_buf(b, 11, &Wsu.Z, 0);
    qw_batch_buf(b, 12, &X, xoff); qw_batch_buf(b, 13, &sel, 0); qw_batch_buf(b, 14, &H, 0);
    qw_batch_dispatch(b, I / (8 * RPS_GU), TOPK + 1, 1, 256, 1, 1);
    qw_batch_kernel(b, K_("moe_down"));
    qw_batch_buf(b, 0, &Wd.W, 0); qw_batch_buf(b, 1, &Wd.S, 0); qw_batch_buf(b, 2, &Wd.Z, 0);
    qw_batch_buf(b, 3, &Wsd.W, 0); qw_batch_buf(b, 4, &Wsd.S, 0); qw_batch_buf(b, 5, &Wsd.Z, 0);
    qw_batch_buf(b, 6, &H, 0); qw_batch_buf(b, 7, &sel, 0); qw_batch_buf(b, 8, &wts, 0); qw_batch_buf(b, 9, &out, 0);
    qw_batch_dispatch(b, D / (8 * RPS_DN), 1, 1, 256, 1, 1);
}

static void check(void) {
    qw_batch *b = qw_batch_begin(g, false);
    encode(b, 0);
    if (qw_batch_end(b) < 0) die("GPU 執行失敗");
    const uint16_t *x = X.ptr, *wsg = Wsgate.ptr, *o = out.ptr;
    // router（8-bit）與共用 gate（BF16）
    static double lg[E + 1];
    for (int r = 0; r <= E; r++) {
        double a = 0;
        for (int k = 0; k < D; k++) a += (r < E ? (double)qm_w(&Wr, (size_t)r, k) : (double)bf2f(wsg[k])) * h2f(x[k]);
        lg[r] = a;
    }
    int ids[TOPK];
    char used[E] = {0};
    for (int t = 0; t < TOPK; t++) {
        int bi = -1;
        for (int e = 0; e < E; e++)
            if (!used[e] && (bi < 0 || lg[e] > lg[bi])) bi = e;
        ids[t] = bi;
        used[bi] = 1;
    }
    const uint32_t *gs = sel.ptr;
    int match = 0;
    for (int t = 0; t < TOPK; t++)
        for (int u = 0; u < TOPK; u++) match += (int)gs[u] == ids[t];
    double w[TOPK + 1], mx = lg[ids[0]], sum = 0;
    for (int t = 0; t < TOPK; t++) sum += exp(lg[ids[t]] - mx);
    for (int t = 0; t < TOPK; t++) w[t] = exp(lg[ids[t]] - mx) / sum;
    w[TOPK] = 1 / (1 + exp(-lg[E]));
    // experts（H 照 GPU 一樣先存成 FP16）
    static double ref[D];
    memset(ref, 0, sizeof ref);
    for (int s = 0; s <= TOPK; s++) {
        const qm *gq = s < TOPK ? &Wg : &Wsg, *uq = s < TOPK ? &Wu : &Wsu, *dn = s < TOPK ? &Wd : &Wsd;
        size_t e = s < TOPK ? (size_t)ids[s] : 0;
        static float h[I];
        for (int j = 0; j < I; j++) {
            double gt = 0, up = 0;
            for (int k = 0; k < D; k++) {
                gt += qm_w(gq, e * I + j, k) * h2f(x[k]);
                up += qm_w(uq, e * I + j, k) * h2f(x[k]);
            }
            h[j] = h2f(f2h((float)(gt / (1 + exp(-gt)) * up)));
        }
        for (int r = 0; r < D; r++) {
            double d = 0;
            for (int j = 0; j < I; j++) d += qm_w(dn, e * D + r, j) * h[j];
            ref[r] += w[s] * d;
        }
    }
    double mref = 0, err = 0;
    for (int r = 0; r < D; r++) mref = fmax(mref, fabs(ref[r]));
    for (int r = 0; r < D; r++) err = fmax(err, fabs(h2f(o[r]) - ref[r]));
    printf("\n## 1. 正確性（對照 CPU）\n\n- 選中的 expert 一致：%d / %d\n- 輸出最大誤差 %.1e（輸出最大值 %.3f，相對 %.1e）→ %s\n", match, TOPK,
           err, mref, err / mref, (match == TOPK && err / mref < 1e-2) ? "✅ 通過" : "❌ 不通過");
}

static double peak_bw(void) {
    char err[4096];
    if (!qw_gpu_compile(g, mem_src, sizeof mem_src, err, sizeof err)) die(err);
    qw_kernel *k = qw_kernel_get(g, "bw_block");
    qw_buf pool, sink;
    if (!qw_buf_new(g, (size_t)4 << 30, &pool) || !qw_buf_new(g, 4096 * 4, &sink)) die("配置失敗");
    memset(pool.ptr, 1, pool.size);
    unsigned n = (1u << 30) / 16;
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        qw_batch *b = qw_batch_begin(g, false);
        qw_batch_kernel(b, k);
        qw_batch_bytes(b, 2, &n, 4);
        qw_batch_buf(b, 1, &sink, 0);
        for (int r = 0; r < 8; r++) {
            qw_batch_buf(b, 0, &pool, (size_t)(r % 4) << 30);
            qw_batch_dispatch(b, 2560, 1, 1, 1024, 1, 1);
        }
        double tt = qw_batch_end(b);
        if (tt > 0 && tt < best) best = tt;
    }
    qw_buf_free(&pool);
    qw_buf_free(&sink);
    return 8.0 * (1u << 30) / best / 1e9;
}

static void speed(double peak) {
    // 每個 token 實際讀到的權重位元組
    double bytes = (double)E * qm_row_bytes(&Wr) + D * 2.0 + TOPK * 2.0 * I * qm_row_bytes(&Wg) + 2.0 * I * qm_row_bytes(&Wsg) +
                   TOPK * (double)D * qm_row_bytes(&Wd) + (double)D * qm_row_bytes(&Wsd);
    const int reps = 200;
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        qw_batch *b = qw_batch_begin(g, false);
        for (int r = 0; r < reps; r++) encode(b, (size_t)(r % NX) * D * 2); // 每次換輸入 → 選到不同 expert
        double tt = qw_batch_end(b);
        if (tt > 0 && tt < best) best = tt;
    }
    double per = best / reps;
    printf("| %d×%d | %.1f µs | %.2f MB | %.0f GB/s | %.0f%% | %.2f ms |\n", RPS_GU, RPS_DN, per * 1e6, bytes / 1e6,
           bytes / per / 1e9, bytes / per / 1e9 / peak * 100, per * 48e3);
    fflush(stdout);
}

int main(void) {
    g = qw_gpu_open();
    if (!g) die("找不到 Metal 裝置");
    char pw[256];
    qw_power_desc(pw, sizeof pw);
    printf("# Decode 合併 MoE 量測\n\n- GPU：%s\n- %s\n", qw_gpu_name(g), pw);
    qm_new(&Wg, 4, I, D, E);
    qm_new(&Wu, 4, I, D, E);
    qm_new(&Wsg, 8, I, D, 1);
    qm_new(&Wsu, 8, I, D, 1);
    qm_new(&Wr, 8, E, D, 1);
    qm_new(&Wd, 4, D, I, E);
    qm_new(&Wsd, 8, D, I, 1);
    if (!qw_buf_new(g, (size_t)D * 2, &Wsgate) || !qw_buf_new(g, (size_t)NX * D * 2, &X) ||
        !qw_buf_new(g, (E + 1) * 4, &logits) || !qw_buf_new(g, 64, &sel) || !qw_buf_new(g, 64, &wts) ||
        !qw_buf_new(g, (TOPK + 1) * I * 2, &H) || !qw_buf_new(g, D * 2, &out))
        die("配置失敗");
    uint16_t *p = Wsgate.ptr;
    for (size_t i = 0; i < (size_t)D; i++) p[i] = f2bf(rndf() * 0.05f);
    p = X.ptr;
    for (size_t i = 0; i < (size_t)NX * D; i++) p[i] = f2h(rndf());
    check();
    double peak = peak_bw();
    printf("\n## 2. 速度（每次換輸入向量，expert 跟著變；頻寬上限現場量測 %.0f GB/s）\n\n", peak);
    printf("| RPS（gate/up × down） | 每層 | 每層讀取 | 有效頻寬 | 達成率 | 換算 48 層 |\n|---|---|---|---|---|---|\n");
    const int cfgs[][2] = {{1, 1}, {2, 2}, {4, 2}, {2, 4}, {4, 4}, {1, 2}};
    for (size_t i = 0; i < sizeof cfgs / sizeof cfgs[0]; i++) {
        RPS_GU = cfgs[i][0];
        RPS_DN = cfgs[i][1];
        speed(peak);
    }
    qw_gpu_close(g);
    return 0;
}
