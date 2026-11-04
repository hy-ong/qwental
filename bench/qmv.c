// Decode matvec（qmv）量測：Qwen3.8-Flash-Next 實際的矩陣形狀
//   1. 每種形狀掃描 RPS（每個 SIMD group 算幾列）× SPT（每個 threadgroup 幾個 SIMD group），
//      驗證數值，回報頻寬與達成率
//   2. 用各形狀最好的參數，組出「只有 matvec 的一整步 decode」（48 層），量每個 token 的時間
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char qmv_src[] = {
#embed "../kernels/qmv.metal"
};
static const char mem_src[] = {
#embed "../kernels/ceiling_mem.metal"
};

// 頻寬上限：每次執行時現場量（區塊讀取 1 GiB），這樣不管插不插電，達成率都有意義
static double PEAK_GBS = 0;
#define GS 64

static qw_gpu *g;

static void die(const char *m) {
    fprintf(stderr, "錯誤：%s\n", m);
    exit(1);
}

// ---------------------------------------------------------------- 工具

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}
static float rndf(void) { return (rnd32() >> 8) * (1.0f / 16777216.0f) * 2 - 1; } // [-1, 1)

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

// ---------------------------------------------------------------- 量化線性層

typedef struct {
    int bits, K, N, nexp; // nexp：權重有幾份（expert 數，或為了避開快取而複製的份數）
    qw_buf W, S, Z;
} qlin;

static size_t row_bytes(const qlin *l) { return (size_t)l->K * l->bits / 8; }
static size_t mat_bytes(const qlin *l) { return (size_t)l->N * (row_bytes(l) + 2 * 2 * (l->K / GS)); }

static void qlin_new(qlin *l, int bits, int K, int N, int nexp) {
    *l = (qlin){bits, K, N, nexp, {0}, {0}, {0}};
    size_t w = (size_t)nexp * N * row_bytes(l), s = (size_t)nexp * N * (K / GS) * 2;
    if (!qw_buf_new(g, w, &l->W) || !qw_buf_new(g, s, &l->S) || !qw_buf_new(g, s, &l->Z)) die("配置權重失敗");
    uint32_t *pw = l->W.ptr;
    for (size_t i = 0; i < w / 4; i++) pw[i] = rnd32();
    uint16_t *ps = l->S.ptr, *pz = l->Z.ptr;
    float qmax = (float)((1 << bits) - 1);
    for (size_t i = 0; i < s / 2; i++) {
        float scale = (0.5f + 0.5f * (rndf() + 1) / 2) * 0.05f / qmax;
        ps[i] = f2bf(scale);
        pz[i] = f2bf(-scale * qmax / 2);
    }
}

static void qlin_free(qlin *l) {
    qw_buf_free(&l->W);
    qw_buf_free(&l->S);
    qw_buf_free(&l->Z);
}

// ---------------------------------------------------------------- kernel

typedef struct { int rps, spt; } qcfg;

static qw_kernel *build(int bits, int K, qcfg c) {
    char name[64];
    snprintf(name, sizeof name, "qmv_b%d_k%d_r%d_s%d", bits, K, c.rps, c.spt);
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    size_t cap = sizeof qmv_src + 256;
    char *src = malloc(cap), err[8192];
    int n = snprintf(src, cap, "#define qmv %s\n#define BITS %d\n#define K %d\n#define RPS %d\n#define SPT %d\n#define ST bfloat\n",
                     name, bits, K, c.rps, c.spt);
    memcpy(src + n, qmv_src, sizeof qmv_src);
    if (!qw_gpu_compile(g, src, n + sizeof qmv_src, err, sizeof err)) {
        fprintf(stderr, "編譯 %s 失敗：\n%s\n", name, err);
        exit(1);
    }
    free(src);
    return qw_kernel_get(g, name);
}

typedef struct { uint32_t N, x_stride, y_stride, pad; } qargs;

// 把一次 qmv 加進 batch。E 個 slot，eids 放在 ebuf 的 eoff 位置。
static void enc_qmv(qw_batch *b, qw_kernel *k, const qlin *l, qcfg c, const qw_buf *x, size_t xoff, uint32_t xstride,
                    const qw_buf *y, size_t yoff, const qw_buf *ebuf, size_t eoff, int E) {
    qargs a = {(uint32_t)l->N, xstride, (uint32_t)l->N, 0};
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, &l->W, 0);
    qw_batch_buf(b, 1, &l->S, 0);
    qw_batch_buf(b, 2, &l->Z, 0);
    qw_batch_buf(b, 3, x, xoff);
    qw_batch_buf(b, 4, y, yoff);
    qw_batch_buf(b, 5, ebuf, eoff);
    qw_batch_bytes(b, 6, &a, sizeof a);
    unsigned rows_per_tg = (unsigned)(c.rps * c.spt);
    qw_batch_dispatch(b, (l->N + rows_per_tg - 1) / rows_per_tg, (unsigned)E, 1, 32u * c.spt, 1, 1);
}

// CPU 參考：第 slot 個 expert（編號 e）的第 row 列
static float ref_row(const qlin *l, int e, int row, const uint16_t *x) {
    const uint8_t *w = (const uint8_t *)l->W.ptr + ((size_t)e * l->N + row) * row_bytes(l);
    const uint16_t *s = (const uint16_t *)l->S.ptr + ((size_t)e * l->N + row) * (l->K / GS);
    const uint16_t *z = (const uint16_t *)l->Z.ptr + ((size_t)e * l->N + row) * (l->K / GS);
    double acc = 0;
    for (int k = 0; k < l->K; k++) {
        int q = l->bits == 4 ? ((w[k / 2] >> (4 * (k & 1))) & 15) : w[k];
        double wf = q * (double)bf2f(s[k / GS]) + bf2f(z[k / GS]);
        acc += wf * h2f(x[k]);
    }
    return (float)acc;
}

// ---------------------------------------------------------------- 1. 單一形狀掃描

typedef struct {
    const char *name;
    int bits, K, N;
    int E;      // 每次 dispatch 幾個 expert（dense = 1）
    int pool;   // 權重有幾份（expert 池大小，或複製份數）
    int shared_x; // 1：所有 slot 共用同一個 x（gate/up）；0：每個 slot 自己的 x（down）
} shape;

static const shape SHAPES[] = {
    {"Routed expert gate+up（10 個，4-bit）", 4, 2560, 1280, 10, 512, 1},
    {"Routed expert down（10 個，4-bit）", 4, 640, 2560, 10, 512, 0},
    {"共用 expert gate+up（8-bit）", 8, 2560, 1280, 1, 96, 1},
    {"共用 expert down（8-bit）", 8, 640, 2560, 1, 96, 0},
    {"DeltaNet in_proj（8-bit）", 8, 2560, 16384, 1, 12, 1},
    {"DeltaNet out_proj（8-bit）", 8, 6144, 2560, 1, 24, 1},
    {"Attention q（含 gate，8-bit）", 8, 2560, 12288, 1, 12, 1},
    {"Attention k+v（8-bit）", 8, 2560, 1024, 1, 96, 1},
    {"Attention o（8-bit）", 8, 6144, 2560, 1, 24, 1},
    {"lm_head（8-bit）", 8, 2560, 248320, 1, 2, 1},
};
#define NSHAPES (int)(sizeof SHAPES / sizeof SHAPES[0])

static const qcfg CFGS[] = {{1, 2}, {1, 4}, {1, 8}, {2, 2}, {2, 4}, {2, 8}, {4, 2}, {4, 4}, {4, 8}, {8, 2}, {8, 4}};
#define NCFGS (int)(sizeof CFGS / sizeof CFGS[0])

static qcfg best_cfg[NSHAPES];

static void pick_eids(uint32_t *out, int E, int pool) {
    for (int i = 0; i < E; i++) {
        uint32_t v;
        bool dup;
        do {
            v = rnd32() % (uint32_t)pool;
            dup = false;
            for (int j = 0; j < i; j++) dup |= out[j] == v;
        } while (dup);
        out[i] = v;
    }
}

static void bench_shapes(void) {
    printf("\n## 1. 各形狀的 qmv\n\n");
    printf("每次 dispatch 從權重池裡挑不同的份，避免打中快取。頻寬上限以實測 %.0f GB/s 計。\n\n", PEAK_GBS);
    printf("| 形狀 | K×N × 個數 | 每次讀取 | 最佳 RPS×SPT | 時間 | GB/s | 達成率 | 數值誤差 |\n");
    printf("|---|---|---|---|---|---|---|---|\n");
    qw_buf x, y, eb;
    if (!qw_buf_new(g, 16 * 6144 * 2, &x) || !qw_buf_new(g, 16 * 248320 * 2, &y) || !qw_buf_new(g, 4096 * 4, &eb))
        die("配置失敗");
    uint16_t *xp = x.ptr;
    for (int i = 0; i < 16 * 6144; i++) xp[i] = f2h(rndf());
    const int reps = 200;
    uint32_t *ep = eb.ptr;

    for (int si = 0; si < NSHAPES; si++) {
        shape shv = SHAPES[si];
        const shape *sh = &shv;
        // 權重池至少 1 GiB，避免被快取吃掉而灌水
        {
            qlin one = {sh->bits, sh->K, sh->N, 1, {0}, {0}, {0}};
            int need = (int)((1u << 30) / mat_bytes(&one)) + 1;
            if (shv.pool < need) shv.pool = need;
        }
        qlin l;
        qlin_new(&l, sh->bits, sh->K, sh->N, sh->pool);
        // 預先為每次 dispatch 挑好 expert 編號
        for (int r = 0; r < reps; r++) {
            if (sh->E > 1) pick_eids(ep + r * 16, sh->E, sh->pool);
            else ep[r * 16] = (uint32_t)(r % sh->pool);
        }
        double best_t = 1e30;
        qcfg bc = CFGS[0];
        for (int ci = 0; ci < NCFGS; ci++) {
            qcfg c = CFGS[ci];
            qw_kernel *k = build(sh->bits, sh->K, c);
            if ((unsigned)(32 * c.spt) > qw_kernel_max_threads(k)) continue;
            double bt = 1e30;
            for (int trial = 0; trial < 3; trial++) {
                qw_batch *b = qw_batch_begin(g, false);
                for (int r = 0; r < reps; r++)
                    enc_qmv(b, k, &l, c, &x, 0, sh->shared_x ? 0 : (uint32_t)sh->K, &y, 0, &eb, (size_t)r * 16 * 4, sh->E);
                double t = qw_batch_end(b);
                if (t < 0) die("GPU 執行失敗");
                if (t < bt) bt = t;
            }
            if (bt < best_t) {
                best_t = bt;
                bc = c;
            }
        }
        best_cfg[si] = bc;
        // 用最佳設定跑一次，驗證第 0 個 slot 與最後一個 slot 的幾列
        qw_kernel *k = build(sh->bits, sh->K, bc);
        qw_batch *b = qw_batch_begin(g, false);
        enc_qmv(b, k, &l, bc, &x, 0, sh->shared_x ? 0 : (uint32_t)sh->K, &y, 0, &eb, 0, sh->E);
        qw_batch_end(b);
        double worst = 0;
        const uint16_t *yp = y.ptr;
        for (int t = 0; t < 32; t++) {
            int slot = (t & 1) ? sh->E - 1 : 0;
            int row = (int)(rnd32() % (uint32_t)sh->N);
            const uint16_t *xs = xp + (sh->shared_x ? 0 : (size_t)slot * sh->K);
            float ref = ref_row(&l, (int)ep[slot], row, xs);
            float got = h2f(yp[(size_t)slot * sh->N + row]);
            double err = fabs(got - ref) / fmax(fabs(ref), 1e-2);
            if (err > worst) worst = err;
        }
        double t1 = best_t / reps;
        double bytes = (double)mat_bytes(&l) * sh->E;
        double gbs = bytes / t1 / 1e9;
        printf("| %s | %d×%d × %d | %.2f MB | %d×%d | %.1f µs | %.0f | %.0f%% | %.1e |\n", sh->name, sh->K, sh->N, sh->E,
               bytes / 1e6, bc.rps, bc.spt, t1 * 1e6, gbs, gbs / PEAK_GBS * 100, worst);
        fflush(stdout);
        qlin_free(&l);
    }
    qw_buf_free(&x);
    qw_buf_free(&y);
    qw_buf_free(&eb);
}

// ---------------------------------------------------------------- 2. 一整步 decode（只有 matvec）

static void bench_step(void) {
    printf("\n## 2. 只有 matvec 的一整步 decode（48 層）\n\n");
    // 每層各自的權重，routed expert 每層一個 64 個 expert 的池子（總量遠大於快取）
    enum { L = 48, POOL = 64 };
    qlin gu[L], dn[L], sgu[L], sdn[L], inp[L], outp[L], lm;
    int nlin = 0;
    for (int i = 0; i < L; i++) {
        bool full = (i % 4) == 3;
        qlin_new(&gu[i], 4, 2560, 1280, POOL);
        qlin_new(&dn[i], 4, 640, 2560, POOL);
        qlin_new(&sgu[i], 8, 2560, 1280, 1);
        qlin_new(&sdn[i], 8, 640, 2560, 1);
        // DeltaNet：in_proj 2560→16384、out_proj 6144→2560
        // Attention：q+k+v 合併成 2560→13312、o 6144→2560
        qlin_new(&inp[i], 8, 2560, full ? 13312 : 16384, 1);
        qlin_new(&outp[i], 8, 6144, 2560, 1);
        nlin += 6;
    }
    qlin_new(&lm, 8, 2560, 248320, 1);
    // 形狀對應到第 1 部分找到的最佳設定
    qcfg c_gu = best_cfg[0], c_dn = best_cfg[1], c_sgu = best_cfg[2], c_sdn = best_cfg[3], c_in = best_cfg[4],
         c_out = best_cfg[5], c_lm = best_cfg[9];
    qw_kernel *k_gu = build(4, 2560, c_gu), *k_dn = build(4, 640, c_dn), *k_sgu = build(8, 2560, c_sgu),
              *k_sdn = build(8, 640, c_sdn), *k_in = build(8, 2560, c_in), *k_out = build(8, 6144, c_out),
              *k_lm = build(8, 2560, c_lm);

    qw_buf x, h, eb;
    if (!qw_buf_new(g, 64 * 1024 * 2, &x) || !qw_buf_new(g, 248320 * 2 * 2, &h) || !qw_buf_new(g, 64 * L * 16 * 4, &eb))
        die("配置失敗");
    uint16_t *xp = x.ptr;
    for (int i = 0; i < 64 * 1024; i++) xp[i] = f2h(rndf() * 0.1f);
    uint32_t *ep = eb.ptr;
    uint32_t zero = 0;
    qw_buf zb;
    if (!qw_buf_new(g, 16, &zb)) die("配置失敗");
    memcpy(zb.ptr, &zero, 4);

    double bytes = 0;
    for (int i = 0; i < L; i++)
        bytes += mat_bytes(&gu[i]) * 10.0 + mat_bytes(&dn[i]) * 10.0 + mat_bytes(&sgu[i]) + mat_bytes(&sdn[i]) +
                 mat_bytes(&inp[i]) + mat_bytes(&outp[i]);
    bytes += mat_bytes(&lm);

    const int steps = 20;
    for (int mode = 0; mode < 2; mode++) {
        double best = 1e30;
        for (int trial = 0; trial < 3; trial++) {
            for (int s = 0; s < steps * L; s++) pick_eids(ep + s * 16, 10, POOL);
            qw_batch *b = qw_batch_begin(g, mode == 1);
            for (int s = 0; s < steps; s++) {
                for (int i = 0; i < L; i++) {
                    size_t eo = (size_t)(s * L + i) * 16 * 4;
                    // 依實際的資料依賴順序：投影 → 輸出投影 → expert gate/up → expert down
                    enc_qmv(b, k_in, &inp[i], c_in, &x, 0, 0, &h, 0, &zb, 0, 1);
                    if (mode) qw_batch_barrier(b);
                    enc_qmv(b, k_out, &outp[i], c_out, &x, 0, 0, &h, 0, &zb, 0, 1);
                    if (mode) qw_batch_barrier(b);
                    enc_qmv(b, k_gu, &gu[i], c_gu, &x, 0, 0, &h, 0, &eb, eo, 10);
                    enc_qmv(b, k_sgu, &sgu[i], c_sgu, &x, 0, 0, &h, 32768, &zb, 0, 1);
                    if (mode) qw_batch_barrier(b);
                    enc_qmv(b, k_dn, &dn[i], c_dn, &x, 0, 640, &h, 0, &eb, eo, 10);
                    enc_qmv(b, k_sdn, &sdn[i], c_sdn, &x, 0, 0, &h, 32768, &zb, 0, 1);
                    if (mode) qw_batch_barrier(b);
                }
                enc_qmv(b, k_lm, &lm, c_lm, &x, 0, 0, &h, 0, &zb, 0, 1);
                if (mode) qw_batch_barrier(b);
            }
            double t = qw_batch_end(b);
            if (t < 0) die("GPU 執行失敗");
            if (t < best) best = t;
        }
        double per = best / steps;
        double gbs = bytes / per / 1e9;
        printf("- %s：每個 token %.2f ms → **%.0f tok/s**，讀取 %.2f GB，%.0f GB/s（達成率 %.0f%%）\n",
               mode ? "concurrent + barrier（同層可以重疊的不加 barrier）" : "serial（每個 dispatch 依序）", per * 1e3,
               1 / per, bytes / 1e9, gbs, gbs / PEAK_GBS * 100);
    }
    printf("\n只含權重的讀取量（%.2f GB）。實際 decode 還要加 DeltaNet 狀態（約 0.23 GB）、attention、norm 等。\n", bytes / 1e9);
    printf("理論上限（只算這些權重）：%.0f tok/s\n", PEAK_GBS * 1e9 / bytes);

    for (int i = 0; i < L; i++) {
        qlin_free(&gu[i]); qlin_free(&dn[i]); qlin_free(&sgu[i]);
        qlin_free(&sdn[i]); qlin_free(&inp[i]); qlin_free(&outp[i]);
    }
    qlin_free(&lm);
    (void)nlin;
}

static double measure_peak(void) {
    char err[4096];
    if (!qw_gpu_compile(g, mem_src, sizeof mem_src, err, sizeof err)) die(err);
    qw_kernel *k = qw_kernel_get(g, "bw_block");
    const size_t bytes = (size_t)1 << 30, poolsz = (size_t)4 << 30;
    qw_buf pool, sink;
    if (!qw_buf_new(g, poolsz, &pool) || !qw_buf_new(g, 4096 * 4, &sink)) die("配置失敗");
    memset(pool.ptr, 1, poolsz);
    unsigned n = (unsigned)(bytes / 16);
    double best = 1e30;
    for (int trial = 0; trial < 5; trial++) {
        qw_batch *b = qw_batch_begin(g, false);
        qw_batch_kernel(b, k);
        qw_batch_bytes(b, 2, &n, sizeof n);
        qw_batch_buf(b, 1, &sink, 0);
        for (int r = 0; r < 8; r++) {
            qw_batch_buf(b, 0, &pool, (size_t)(r % 4) * bytes);
            qw_batch_dispatch(b, 2560, 1, 1, 1024, 1, 1);
        }
        double t = qw_batch_end(b);
        if (t > 0 && t < best) best = t;
    }
    qw_buf_free(&pool);
    qw_buf_free(&sink);
    return (double)bytes * 8 / best / 1e9;
}

int main(void) {
    g = qw_gpu_open();
    if (!g) die("找不到 Metal 裝置");
    printf("# qmv（decode matvec）量測\n\n- GPU：%s\n", qw_gpu_name(g));
    char pw[256];
    qw_power_desc(pw, sizeof pw);
    printf("- %s\n", pw);
    PEAK_GBS = measure_peak();
    printf("- 現場量測的頻寬上限（區塊讀取 1 GiB）：%.0f GB/s，以下達成率以此為分母\n", PEAK_GBS);
    bench_shapes();
    bench_step();
    qw_gpu_close(g);
    return 0;
}
