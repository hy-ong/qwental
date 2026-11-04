// qmm（prefill 量化矩陣乘法）量測
//   正確性：對照 CPU（解量化後的 double 乘法）
//   速度：模型實際形狀 × token 數，掃描 TM×TN×SG×KC，報告 TFLOPS 與相對「不量化 FP16」的比例
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char qmm_src[] = {
#embed "../kernels/qmm.metal"
};

static qw_gpu *g;

static void die(const char *m) {
    fprintf(stderr, "錯誤：%s\n", m);
    exit(1);
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
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

typedef struct { int tm, tn, sg, kc; } cfg;

static qw_kernel *build(int bits, int K, cfg c) {
    char name[96];
    snprintf(name, sizeof name, "qmm_b%d_k%d_%d_%d_%d_%d", bits, K, c.tm, c.tn, c.sg, c.kc);
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    size_t cap = sizeof qmm_src + 256;
    char *src = malloc(cap), err[8192];
    int n = snprintf(src, cap, "#define qmm %s\n#define BITS %d\n#define KD %d\n#define TM %d\n#define TN %d\n#define SG %d\n#define KC %d\n#define MOE 0\n",
                     name, bits, K, c.tm, c.tn, c.sg, c.kc);
    memcpy(src + n, qmm_src, sizeof qmm_src);
    bool ok = qw_gpu_compile(g, src, n + sizeof qmm_src, err, sizeof err);
    free(src);
    if (!ok) {
        char *nl = strchr(err, '\n');
        if (nl) *nl = 0;
        fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
        return NULL;
    }
    return qw_kernel_get(g, name);
}

static qw_buf bX, bW, bS, bZ, bY;

static void fill(int M, int N, int K, int bits) {
    uint16_t *x = bX.ptr;
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = f2h(rndf());
    uint32_t *w = bW.ptr;
    for (size_t i = 0; i < (size_t)N * K * bits / 32; i++) w[i] = rnd32();
    uint16_t *s = bS.ptr, *z = bZ.ptr;
    float qmax = (float)((1 << bits) - 1);
    for (size_t i = 0; i < (size_t)N * (K / 64); i++) {
        float sc = (0.5f + 0.25f * (rndf() + 1)) * 0.05f / qmax;
        s[i] = f2bf(sc);
        z[i] = f2bf(-sc * qmax / 2);
    }
}

static double run(qw_kernel *k, cfg c, int M, int N, int reps) {
    uint32_t args[2] = {(uint32_t)M, (uint32_t)N};
    double best = 1e30;
    for (int trial = 0; trial < 3; trial++) {
        qw_batch *b = qw_batch_begin(g, false);
        qw_batch_kernel(b, k);
        qw_batch_buf(b, 0, &bX, 0);
        qw_batch_buf(b, 1, &bW, 0);
        qw_batch_buf(b, 2, &bS, 0);
        qw_batch_buf(b, 3, &bZ, 0);
        qw_batch_buf(b, 4, &bY, 0);
        qw_batch_bytes(b, 5, args, sizeof args);
        for (int r = 0; r < reps; r++) qw_batch_dispatch(b, N / c.tn, (M + c.tm - 1) / c.tm, 1, 32u * c.sg, 1, 1);
        double t = qw_batch_end(b);
        if (t < 0) return -1;
        if (t < best) best = t;
    }
    return best / reps;
}

static double verify(int M, int N, int K, int bits) {
    const uint16_t *x = bX.ptr, *s = bS.ptr, *z = bZ.ptr, *y = bY.ptr;
    const uint32_t *w = bW.ptr;
    const int vpw = 32 / bits, roww = K * bits / 32;
    double worst = 0;
    for (int t = 0; t < 48; t++) {
        int m = (int)(rnd32() % (uint32_t)M), n = (int)(rnd32() % (uint32_t)N);
        double ref = 0;
        for (int k = 0; k < K; k++) {
            uint32_t word = w[(size_t)n * roww + k / vpw];
            int q = (int)((word >> (bits * (k % vpw))) & ((1u << bits) - 1));
            double wf = q * (double)bf2f(s[(size_t)n * (K / 64) + k / 64]) + bf2f(z[(size_t)n * (K / 64) + k / 64]);
            ref += wf * h2f(x[(size_t)m * K + k]);
        }
        double e = fabs(h2f(y[(size_t)m * N + n]) - ref) / fmax(fabs(ref), 0.05);
        if (e > worst) worst = e;
    }
    return worst;
}

static const cfg CFGS[] = {
    {64, 64, 2, 64}, {64, 64, 4, 64}, {128, 64, 4, 64}, {64, 64, 4, 128}, {128, 64, 4, 128},
    {128, 128, 4, 64}, {64, 128, 4, 64}, {128, 64, 8, 64}, {128, 128, 8, 64},
};
#define NCFGS (int)(sizeof CFGS / sizeof CFGS[0])

int main(void) {
    g = qw_gpu_open();
    if (!g) die("找不到 Metal 裝置");
    char pw[256];
    qw_power_desc(pw, sizeof pw);
    printf("# qmm（prefill 量化矩陣乘法）量測\n\n- GPU：%s\n- %s\n", qw_gpu_name(g), pw);
    const size_t MAXM = 16384, MAXK = 6144, MAXN = 16384;
    if (!qw_buf_new(g, MAXM * MAXK * 2, &bX) || !qw_buf_new(g, MAXN * MAXK, &bW) ||
        !qw_buf_new(g, MAXN * (MAXK / 64) * 2, &bS) || !qw_buf_new(g, MAXN * (MAXK / 64) * 2, &bZ) ||
        !qw_buf_new(g, MAXM * MAXN * 2, &bY))
        die("配置失敗");

    const struct { int K, N, bits; const char *name; } SH[] = {
        {2560, 1280, 4, "expert gate+up（4-bit）"}, {640, 2560, 4, "expert down（4-bit）"},
        {2560, 16384, 8, "DeltaNet in_proj（8-bit）"}, {6144, 2560, 8, "out_proj（8-bit）"},
        {2560, 1280, 8, "共用 expert gate+up（8-bit）"},
    };
    const int Ms[] = {512, 2048, 16384};
    printf("\n| 形狀 | M | 最佳 TM×TN×SG×KC | 時間 | TFLOPS | 數值誤差 |\n|---|---|---|---|---|---|\n");
    for (size_t si = 0; si < sizeof SH / sizeof SH[0]; si++)
        for (int mi = 0; mi < 3; mi++) {
            int M = Ms[mi], K = SH[si].K, N = SH[si].N, bits = SH[si].bits;
            fill(M, N, K, bits);
            double flop = 2.0 * M * N * K;
            int reps = (int)(1e12 / flop);
            if (reps < 2) reps = 2;
            if (reps > 200) reps = 200;
            double best = 1e30;
            cfg bc = CFGS[0];
            for (int ci = 0; ci < NCFGS; ci++) {
                cfg c = CFGS[ci];
                if (N % c.tn || K % c.kc) continue;
                qw_kernel *k = build(bits, K, c);
                if (!k || (unsigned)(32 * c.sg) > qw_kernel_max_threads(k)) continue;
                double t = run(k, c, M, N, reps);
                if (t > 0 && t < best) { best = t; bc = c; }
            }
            run(build(bits, K, bc), bc, M, N, 1);
            double e = verify(M, N, K, bits);
            printf("| %s | %d | %d×%d×%d×%d | %.1f µs | %.1f | %.1e |\n", SH[si].name, M, bc.tm, bc.tn, bc.sg, bc.kc,
                   best * 1e6, flop / best / 1e12, e);
            fflush(stdout);
        }
    qw_gpu_close(g);
    return 0;
}
