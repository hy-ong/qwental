// qmm_i8（dense prefill 的 INT8 × INT8 矩陣乘法，kernels/qmm_i8.metal）量測
//   用法：build/qmm_i8 M TM TN SG KC [WT [形狀編號]]（KC = 0：整個 K 一次 run；WT = 1：權重 [K][N]）
//   每個形狀先用原本的設定（128×64，SG 4，KC 64，權重 [N][K]）算一次並抽查對照 CPU，再用指定設定算，輸出必須逐位元相同；
//   然後量指定設定的 TOPS（一次只量一種設定）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char src_i8[] = {
#embed "../kernels/qmm_i8.metal"
};

static qw_gpu *g;
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}

typedef struct { int tm, tn, sg, kc, wt; } cfg;

static qw_kernel *build(int K, cfg c) {
    char name[96], defs[512], err[8192];
    snprintf(name, sizeof name, "qmm_i8_k%d_%dx%dx%d_kc%d_t%d", K, c.tm, c.tn, c.sg, c.kc, c.wt);
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    int n = snprintf(defs, sizeof defs, "#define KD %d\n#define TM %d\n#define TN %d\n#define SG %d\n#define KC %d\n#define WT %d\n#define YT half\n"
                     "#define i8_quant i8_quant_%s\n#define qmm_i8 %s\n", K, c.tm, c.tn, c.sg, c.kc, c.wt, name, name);
    size_t len = n + sizeof src_i8;
    char *s = malloc(len);
    memcpy(s, defs, n);
    memcpy(s + n, src_i8, sizeof src_i8);
    if (!qw_gpu_compile(g, s, len, err, sizeof err)) { fprintf(stderr, "編譯失敗：%s\n", err); exit(1); }
    free(s);
    return qw_kernel_get(g, name);
}

static qw_buf bX, bW, bWt, bS, bY, bY0, bsa, bas, brs;

static double run(cfg c, int M, int N, int K, int reps, const qw_buf *Y) {
    qw_kernel *k = build(K, c);
    const uint32_t a[2] = {(uint32_t)M, (uint32_t)N};
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        qw_batch *b = qw_batch_begin(g, false);
        qw_batch_kernel(b, k);
        qw_batch_buf(b, 0, &bX, 0); qw_batch_buf(b, 1, c.wt ? &bWt : &bW, 0); qw_batch_buf(b, 2, &bS, 0); qw_batch_buf(b, 4, Y, 0);
        qw_batch_bytes(b, 5, a, sizeof a); qw_batch_buf(b, 6, &bsa, 0); qw_batch_buf(b, 7, &bas, 0); qw_batch_buf(b, 8, &brs, 0);
        for (int r = 0; r < reps; r++) qw_batch_dispatch(b, (unsigned)(N / c.tn), (unsigned)((M + c.tm - 1) / c.tm), 1, 32u * c.sg, 1, 1);
        const double tt = qw_batch_end(b);
        if (tt < 0) { fprintf(stderr, "GPU 執行失敗\n"); exit(1); }
        if (tt < best) best = tt;
    }
    return best / reps;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "用法：%s M TM TN SG KC [WT [形狀]]\n", argv[0]); return 1; }
    const int M = atoi(argv[1]);
    const cfg c = {atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argc > 6 ? atoi(argv[6]) : 0}, base = {128, 64, 4, 64, 0};
    const int only = argc > 7 ? atoi(argv[7]) : -1;
    const struct { int K, N; const char *name; int per; } SH[] = {
        {2560, 10240, "DeltaNet in_qkv", 36}, {2560, 6144, "DeltaNet in_z", 36}, {6144, 2560, "DeltaNet / attn out", 48},
        {2560, 12288, "attn q", 12},          {2560, 512, "attn k / v", 24},     {2560, 640, "idx_qk / shared gate,up", 108},
        {640, 2560, "shared down", 48},
    };
    g = qw_gpu_open();
    if (!g) { fprintf(stderr, "開不了 GPU\n"); return 1; }
    const size_t MX = (size_t)M + 256;
    if (!qw_buf_new(g, MX * 6144, &bX) || !qw_buf_new(g, (size_t)12288 * 6144, &bW) || !qw_buf_new(g, (size_t)12288 * 6144, &bWt) || !qw_buf_new(g, (size_t)12288 * 96 * 2, &bS) ||
        !qw_buf_new(g, MX * 12288 * 2, &bY) || !qw_buf_new(g, MX * 12288 * 2, &bY0) || !qw_buf_new(g, MX * 4, &bsa) ||
        !qw_buf_new(g, MX * 4, &bas) || !qw_buf_new(g, (size_t)12288 * 4, &brs)) { fprintf(stderr, "配置失敗\n"); return 1; }
    uint8_t *px = bX.ptr, *pw = bW.ptr;
    for (size_t i = 0; i < bX.size; i++) px[i] = (uint8_t)rnd32();
    for (size_t i = 0; i < bW.size; i++) pw[i] = (uint8_t)rnd32();
    uint16_t *ps = bS.ptr;
    for (size_t i = 0; i < bS.size / 2; i++) ps[i] = (uint16_t)(0x3a00 + (rnd32() & 0x3ff)); // bf16 約 0.0005–0.001
    float *psa = bsa.ptr;
    int *pas = bas.ptr, *prs = brs.ptr;
    for (size_t i = 0; i < MX; i++) { psa[i] = 0.001f + (rnd32() & 0xffff) * 1e-8f; pas[i] = (int)(rnd32() % 20001) - 10000; }
    for (size_t i = 0; i < 12288; i++) prs[i] = (int)(rnd32() % 20001) - 10000;
    printf("M = %d，設定 %d×%d，SG %d，KC %d，WT %d\n\n| 形狀（K→N） | 逐位元 | 時間 | TOPS |\n|---|---|---|---|\n", M, c.tm, c.tn, c.sg, c.kc, c.wt);
    double tot = 0;
    for (int s = 0; s < (int)(sizeof SH / sizeof SH[0]); s++) {
        if (only >= 0 && s != only) continue;
        const int K = SH[s].K, N = SH[s].N;
        if (N % c.tn || (c.kc && K % c.kc)) { printf("| %s（%d→%d） | 不支援 | | |\n", SH[s].name, K, N); continue; }
        if (c.wt) // 權重轉置的副本 [K][N]
            for (int n = 0; n < N; n++)
                for (int k = 0; k < K; k++) ((uint8_t *)bWt.ptr)[(size_t)k * N + n] = pw[(size_t)n * K + k];
        memset(bY.ptr, 0, (size_t)M * N * 2);
        memset(bY0.ptr, 0xff, (size_t)M * N * 2);
        run(base, M, N, K, 1, &bY0);
        const double t = run(c, M, N, K, 10, &bY);
        const bool same = !memcmp(bY.ptr, bY0.ptr, (size_t)M * N * 2);
        double worst = 0; // 基準設定對照 CPU（抽查）
        for (int q = 0; q < 64; q++) {
            const int m = (int)(rnd32() % (unsigned)M), n = (int)(rnd32() % (unsigned)N);
            long acc = 0;
            for (int k = 0; k < K; k++) acc += (long)px[(size_t)m * K + k] * pw[(size_t)n * K + k];
            const long v = acc - 128L * pas[m] - 128L * prs[n] - (long)K * 16384;
            uint32_t u = (uint32_t)ps[(size_t)n * (K / 64)] << 16;
            float sw;
            memcpy(&sw, &u, 4);
            const double ref = (double)v * psa[m] * sw, got = (double)((_Float16 *)bY0.ptr)[(size_t)m * N + n];
            const double e = __builtin_fabs(got - ref) / (__builtin_fabs(ref) + 1e-3);
            if (e > worst) worst = e;
        }
        if (worst > 2e-3) printf("基準對照 CPU 誤差過大：%.1e\n", worst);
        printf("| %s（%d→%d） | %s | %.1f µs | %.1f |\n", SH[s].name, K, N, same ? "相同" : "**不同**", t * 1e6, 2.0 * M * N * K / t / 1e12);
        tot += t * SH[s].per;
        fflush(stdout);
    }
    printf("\n每段（%d token）合計：%.1f ms\n", M, tot * 1e3);
    return 0;
}
