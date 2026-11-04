// 第 0 階段補充量測：把 M5 Max 測清楚
//   1. tensor 單元在模型實際形狀下的速度，以及「每 64 個 K 乘一次 scale」的額外成本
//   2. GPU 快取（SLC）大小：反覆讀同一塊記憶體時的頻寬
//   3. 長時間滿載：運算與頻寬能不能持續（筆電會不會降頻）
// 用法：build/ceiling2 [--shapes] [--cache] [--sustain 秒數]（不加參數 = 全部，滿載 60 秒）
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char mm2_src[] = {
#embed "../kernels/ceiling_mm2.metal"
};
static const char mem_src[] = {
#embed "../kernels/ceiling_mem.metal"
};

static qw_gpu *g;
static bool bench_split_flag = false, bench_chunk_flag = false;

static void die(const char *m) {
    fprintf(stderr, "錯誤：%s\n", m);
    exit(1);
}

static uint64_t rng = 0x243f6a8885a308d3ull;
static uint8_t rnd8(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint8_t)(rng >> 32);
}

// ---------------------------------------------------------------- 1. 實際形狀

typedef struct {
    const char *label;
    const char *ta, *tb, *pt; // 元素型別、部分積型別
    double ea, eb;            // 每元素位元組
    double peak;              // 4096³ 實測峰值（插電），當作達成率分母
} ty;

static const ty TYPES[] = {
    {"FP16×FP16", "half", "half", "float", 2, 2, 63},
    {"FP16×INT8", "half", "uint8_t", "float", 2, 1, 56},
    {"INT8×INT8", "int8_t", "int8_t", "int32_t", 1, 1, 122},
    {"FP16×INT4", "half", "uint4b_format", "float", 2, 0.5, 59},
    {"INT8×INT4", "int8_t", "int4b_format", "int32_t", 1, 0.5, 99},
};
#define NTYPES (int)(sizeof TYPES / sizeof TYPES[0])

typedef struct { int tm, tn, sg; } tile;
static const tile TILES[] = {{64, 64, 2}, {128, 64, 4}, {64, 32, 1}, {128, 128, 4}};
#define NTILES (int)(sizeof TILES / sizeof TILES[0])

static const char *ptr_of(const char *t) { return strstr(t, "format") ? "uchar" : t; }

static int GKV = 64; // 分段長度

static qw_kernel *build_mm2(const ty *t, tile tl, int grouped) {
    char name[128];
    snprintf(name, sizeof name, "mm2_%s_%s_%d_%d_%d_g%d_k%d", t->ta, t->tb, tl.tm, tl.tn, tl.sg, grouped, GKV);
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    size_t cap = sizeof mm2_src + 512;
    char *src = malloc(cap), err[8192];
    int n = snprintf(src, cap,
                     "#define mm2 %s\n#define TA %s\n#define TB %s\n#define PA %s\n#define PB %s\n#define PT %s\n"
                     "#define TM %d\n#define TN %d\n#define SG %d\n#define GK %d\n#define GROUPED %d\n",
                     name, t->ta, t->tb, ptr_of(t->ta), ptr_of(t->tb), t->pt, tl.tm, tl.tn, tl.sg, GKV, grouped);
    memcpy(src + n, mm2_src, sizeof mm2_src);
    bool ok = qw_gpu_compile(g, src, n + sizeof mm2_src, err, sizeof err);
    free(src);
    if (!ok) {
        char *nl = strchr(err, '\n');
        if (nl) *nl = 0;
        fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
        return NULL;
    }
    return qw_kernel_get(g, name);
}

static qw_buf bA, bB, bS, bC;

static double run_mm2(qw_kernel *k, tile tl, int M, int N, int K, int reps) {
    uint32_t mnk[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 0};
    double best = 1e30;
    for (int trial = 0; trial < 3; trial++) {
        qw_batch *b = qw_batch_begin(g, false);
        qw_batch_kernel(b, k);
        qw_batch_buf(b, 0, &bA, 0);
        qw_batch_buf(b, 1, &bB, 0);
        qw_batch_buf(b, 2, &bS, 0);
        qw_batch_buf(b, 3, &bC, 0);
        qw_batch_bytes(b, 4, mnk, sizeof mnk);
        for (int r = 0; r < reps; r++) qw_batch_dispatch(b, (N + tl.tn - 1) / tl.tn, (M + tl.tm - 1) / tl.tm, 1, 32u * tl.sg, 1, 1);
        double t = qw_batch_end(b);
        if (t < 0) return -1;
        if (t < best) best = t;
    }
    return 2.0 * M * N * K / (best / reps) / 1e12;
}

static float h2f(uint16_t u) {
    _Float16 h;
    memcpy(&h, &u, 2);
    return (float)h;
}

// 驗證分段縮放版的 FP16×FP16：C[m,n] = Σ_g S[n,g] Σ_{k∈g} A[m,k] B[n,k]
static double verify_f16_grouped(int M, int N, int K) {
    const uint16_t *A = bA.ptr, *B = bB.ptr;
    const float *S = bS.ptr, *C = bC.ptr;
    double worst = 0;
    for (int s = 0; s < 32; s++) {
        int m = (int)(rnd8() * 131u % (unsigned)M), n = (int)(rnd8() * 977u % (unsigned)N);
        double ref = 0;
        for (int gi = 0; gi < K / 64; gi++) {
            double p = 0;
            for (int k = gi * 64; k < gi * 64 + 64; k++) p += (double)h2f(A[(size_t)m * K + k]) * h2f(B[(size_t)n * K + k]);
            ref += p * S[(size_t)n * (K / 64) + gi];
        }
        double e = fabs(C[(size_t)m * N + n] - ref) / fmax(fabs(ref), 1.0);
        if (e > worst) worst = e;
    }
    return worst;
}

static void bench_shapes(void) {
    printf("\n## 1. 實際形狀下的 tensor 單元，以及每 64 個 K 乘 scale 的成本\n\n");
    printf("每格是「分段縮放 / 不分段」的 TFLOPS（或 TOPS），括號是分段版相對 4096³ 峰值的達成率。各取最好的區塊大小。\n\n");
    const int Ms[] = {256, 2048, 16384};
    const struct { int K, N; const char *name; } KN[] = {
        {2560, 1280, "expert gate+up"}, {640, 2560, "expert down"}, {2560, 16384, "DeltaNet in_proj"}, {6144, 2560, "out_proj"}};
    size_t maxA = (size_t)16384 * 6144 * 2, maxB = (size_t)16384 * 6144 * 2, maxC = (size_t)16384 * 16384 * 4;
    if (!qw_buf_new(g, maxA, &bA) || !qw_buf_new(g, maxB, &bB) || !qw_buf_new(g, (size_t)16384 * 96 * 4, &bS) ||
        !qw_buf_new(g, maxC, &bC))
        die("配置失敗");
    uint8_t *pa = bA.ptr, *pb = bB.ptr;
    for (size_t i = 0; i < maxA; i++) pa[i] = rnd8() & 0x3b; // 各種格式都保持有限值
    for (size_t i = 0; i < maxB; i++) pb[i] = rnd8() & 0x3b;
    float *ps = bS.ptr;
    for (size_t i = 0; i < bS.size / 4; i++) ps[i] = 0.5f + (rnd8() / 255.0f);

    // 先驗證分段版的正確性（FP16，最小的形狀）
    {
        qw_kernel *k = build_mm2(&TYPES[0], TILES[0], 1);
        if (!k) die("編譯失敗");
        run_mm2(k, TILES[0], 256, 1280, 2560, 1);
        printf("分段縮放版正確性（FP16，256×1280×2560，對照 CPU）：最大相對誤差 %.1e\n\n", verify_f16_grouped(256, 1280, 2560));
    }

    printf("| 形狀（K→N） | M（token 數） |");
    for (int t = 0; t < NTYPES; t++) printf(" %s |", TYPES[t].label);
    printf("\n|---|---|");
    for (int t = 0; t < NTYPES; t++) printf("---|");
    printf("\n");
    for (size_t s = 0; s < sizeof KN / sizeof KN[0]; s++)
        for (int mi = 0; mi < 3; mi++) {
            int M = Ms[mi], K = KN[s].K, N = KN[s].N;
            double flop = 2.0 * M * N * K;
            int reps = (int)(2e12 / flop);
            if (reps < 2) reps = 2;
            if (reps > 400) reps = 400;
            printf("| %s（%d→%d） | %d |", KN[s].name, K, N, M);
            for (int t = 0; t < NTYPES; t++) {
                double best[2] = {-1, -1};
                for (int gr = 0; gr < 2; gr++)
                    for (int ti = 0; ti < NTILES; ti++) {
                        qw_kernel *k = build_mm2(&TYPES[t], TILES[ti], gr);
                        if (!k || (unsigned)(32 * TILES[ti].sg) > qw_kernel_max_threads(k)) continue;
                        double v = run_mm2(k, TILES[ti], M, N, K, reps);
                        if (v > best[gr]) best[gr] = v;
                    }
                printf(" %.1f / %.1f（%.0f%%） |", best[1], best[0], best[1] / TYPES[t].peak * 100);
            }
            printf("\n");
            fflush(stdout);
        }
    qw_buf_free(&bA); qw_buf_free(&bB); qw_buf_free(&bS); qw_buf_free(&bC);
}

// ---------------------------------------------------------------- 2. 快取大小

// 分段成本拆解：同一個形狀，比較 不分段 / 分段+device scale / 分段不乘 / 分段+threadgroup scale
static void bench_split(void) {
    printf("\n## 1b. 分段縮放的成本拆解（expert gate+up，M=2048，K=2560→N=1280）\n\n");
    const int M = 2048, K = 2560, N = 1280;
    if (!qw_buf_new(g, (size_t)M * K * 2, &bA) || !qw_buf_new(g, (size_t)N * K * 2, &bB) ||
        !qw_buf_new(g, (size_t)N * (K / 64) * 4, &bS) || !qw_buf_new(g, (size_t)M * N * 4, &bC))
        die("配置失敗");
    uint8_t *pa = bA.ptr, *pb = bB.ptr;
    for (size_t i = 0; i < bA.size; i++) pa[i] = rnd8() & 0x3b;
    for (size_t i = 0; i < bB.size; i++) pb[i] = rnd8() & 0x3b;
    float *ps = bS.ptr;
    for (size_t i = 0; i < bS.size / 4; i++) ps[i] = 1.0f;
    const char *names[] = {"不分段", "分段 + scale 從 device 讀", "分段、不乘 scale", "分段 + scale 在 threadgroup memory"};
    printf("| 精度 |");
    for (int v = 0; v < 4; v++) printf(" %s |", names[v]);
    printf("\n|---|---|---|---|---|\n");
    const int tys[] = {0, 2, 3, 4};
    for (int ti = 0; ti < 4; ti++) {
        const ty *t = &TYPES[tys[ti]];
        printf("| %s |", t->label);
        for (int v = 0; v < 4; v++) {
            double best = -1;
            for (int tl = 0; tl < NTILES; tl++) {
                qw_kernel *k = build_mm2(t, TILES[tl], v);
                if (!k || (unsigned)(32 * TILES[tl].sg) > qw_kernel_max_threads(k)) continue;
                double x = run_mm2(k, TILES[tl], M, N, K, 50);
                if (x > best) best = x;
            }
            printf(" %.1f |", best);
        }
        printf("\n");
        fflush(stdout);
    }
    qw_buf_free(&bA); qw_buf_free(&bB); qw_buf_free(&bS); qw_buf_free(&bC);
}

// 用 multiply_accumulate 分段累加時，分段長度要多長才夠快
static void bench_chunk(void) {
    printf("\n## 1c. 分段累加（multiply_accumulate）的分段長度（expert gate+up，M=2048，K=2560→N=1280）\n\n");
    const int M = 2048, K = 2560, N = 1280;
    if (!qw_buf_new(g, (size_t)M * K * 2, &bA) || !qw_buf_new(g, (size_t)N * K * 2, &bB) ||
        !qw_buf_new(g, (size_t)N * (K / 64) * 4, &bS) || !qw_buf_new(g, (size_t)M * N * 4, &bC))
        die("配置失敗");
    uint8_t *pa = bA.ptr, *pb = bB.ptr;
    for (size_t i = 0; i < bA.size; i++) pa[i] = rnd8() & 0x3b;
    for (size_t i = 0; i < bB.size; i++) pb[i] = rnd8() & 0x3b;
    const int gks[] = {64, 128, 256, 512, 1280};
    printf("| 精度 | 不分段 |");
    for (int i = 0; i < 5; i++) printf(" K 每段 %d |", gks[i]);
    printf("\n|---|---|---|---|---|---|---|\n");
    const int tys[] = {0, 2, 3, 4};
    for (int ti = 0; ti < 4; ti++) {
        const ty *t = &TYPES[tys[ti]];
        printf("| %s |", t->label);
        for (int v = -1; v < 5; v++) {
            GKV = v < 0 ? 64 : gks[v];
            double best = -1;
            for (int tl = 0; tl < NTILES; tl++) {
                qw_kernel *k = build_mm2(t, TILES[tl], v < 0 ? 0 : 4);
                if (!k || (unsigned)(32 * TILES[tl].sg) > qw_kernel_max_threads(k)) continue;
                double x = run_mm2(k, TILES[tl], M, N, K, 50);
                if (x > best) best = x;
            }
            printf(" %.1f |", best);
        }
        printf("\n");
        fflush(stdout);
    }
    GKV = 64;
    qw_buf_free(&bA); qw_buf_free(&bB); qw_buf_free(&bS); qw_buf_free(&bC);
}

static void bench_cache(void) {
    printf("\n## 2. GPU 快取大小（反覆讀同一塊記憶體）\n\n");
    printf("工作集小於快取時，頻寬會明顯高於 DRAM 的約 580 GB/s。\n\n");
    printf("| 工作集 | GB/s |\n|---|---|\n");
    qw_kernel *k = qw_kernel_get(g, "bw_block");
    qw_buf pool, sink;
    if (!qw_buf_new(g, (size_t)1 << 30, &pool) || !qw_buf_new(g, 4096 * 4, &sink)) die("配置失敗");
    memset(pool.ptr, 1, pool.size);
    const size_t sizes[] = {1 << 20, 2 << 20, 4 << 20, 8 << 20, 12 << 20, 16 << 20, 24 << 20, 32 << 20, 48 << 20,
                            64 << 20, 96 << 20, 128 << 20, 256 << 20, 512 << 20};
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        size_t sz = sizes[i];
        unsigned n = (unsigned)(sz / 16);
        int reps = (int)(((size_t)8 << 30) / sz);
        if (reps > 4000) reps = 4000;
        unsigned ntg = sz >= (64u << 20) ? 2560 : 320;
        double best = 1e30;
        for (int trial = 0; trial < 3; trial++) {
            qw_batch *b = qw_batch_begin(g, false);
            qw_batch_kernel(b, k);
            qw_batch_bytes(b, 2, &n, sizeof n);
            qw_batch_buf(b, 1, &sink, 0);
            qw_batch_buf(b, 0, &pool, 0); // 永遠讀同一塊
            for (int r = 0; r < reps; r++) qw_batch_dispatch(b, ntg, 1, 1, 1024, 1, 1);
            double t = qw_batch_end(b);
            if (t > 0 && t < best) best = t;
        }
        printf("| %zu MiB | %.0f |\n", sz >> 20, (double)sz * reps / best / 1e9);
        fflush(stdout);
    }
    qw_buf_free(&pool);
    qw_buf_free(&sink);
}

// ---------------------------------------------------------------- 3. 長時間滿載

static void bench_sustain(int secs) {
    printf("\n## 3. 長時間滿載（各 %d 秒）\n\n", secs);
    char pw[256];
    // 運算：FP16 4096³
    {
        const int M = 4096, N = 4096, K = 4096;
        if (!qw_buf_new(g, (size_t)M * K * 2, &bA) || !qw_buf_new(g, (size_t)N * K * 2, &bB) ||
            !qw_buf_new(g, (size_t)N * 64 * 4, &bS) || !qw_buf_new(g, (size_t)M * N * 4, &bC))
            die("配置失敗");
        uint8_t *pa = bA.ptr, *pb = bB.ptr;
        for (size_t i = 0; i < bA.size; i++) pa[i] = rnd8() & 0x3b;
        for (size_t i = 0; i < bB.size; i++) pb[i] = rnd8() & 0x3b;
        qw_kernel *k = build_mm2(&TYPES[0], TILES[0], 0);
        printf("### 運算（FP16 4096³）\n\n| 經過時間 | TFLOPS | 狀態 |\n|---|---|---|\n");
        double t0 = qw_now(), next = 0;
        while (qw_now() - t0 < secs) {
            double v = run_mm2(k, TILES[0], M, N, K, 60); // 每次約 0.13 秒 × 3
            double el = qw_now() - t0;
            if (el >= next) {
                qw_power_desc(pw, sizeof pw);
                printf("| %.0f 秒 | %.1f | %s |\n", el, v, pw);
                fflush(stdout);
                next += secs / 10.0;
            }
        }
        qw_buf_free(&bA); qw_buf_free(&bB); qw_buf_free(&bS); qw_buf_free(&bC);
    }
    // 頻寬：區塊讀取 1 GiB
    {
        qw_kernel *k = qw_kernel_get(g, "bw_block");
        qw_buf pool, sink;
        if (!qw_buf_new(g, (size_t)4 << 30, &pool) || !qw_buf_new(g, 4096 * 4, &sink)) die("配置失敗");
        memset(pool.ptr, 1, pool.size);
        unsigned n = (unsigned)((1u << 30) / 16);
        printf("\n### 頻寬（區塊讀取，輪流讀 4 個 1 GiB）\n\n| 經過時間 | GB/s | 狀態 |\n|---|---|---|\n");
        double t0 = qw_now(), next = 0;
        while (qw_now() - t0 < secs) {
            qw_batch *b = qw_batch_begin(g, false);
            qw_batch_kernel(b, k);
            qw_batch_bytes(b, 2, &n, sizeof n);
            qw_batch_buf(b, 1, &sink, 0);
            for (int r = 0; r < 64; r++) {
                qw_batch_buf(b, 0, &pool, (size_t)(r % 4) << 30);
                qw_batch_dispatch(b, 2560, 1, 1, 1024, 1, 1);
            }
            double t = qw_batch_end(b);
            double el = qw_now() - t0;
            if (el >= next) {
                qw_power_desc(pw, sizeof pw);
                printf("| %.0f 秒 | %.0f | %s |\n", el, 64.0 * (1u << 30) / t / 1e9, pw);
                fflush(stdout);
                next += secs / 10.0;
            }
        }
        qw_buf_free(&pool);
        qw_buf_free(&sink);
    }
}

int main(int argc, char **argv) {
    bool shapes = false, cache = false, sustain = false;
    int secs = 60;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--shapes")) shapes = true;
        else if (!strcmp(argv[i], "--split")) { bench_split_flag = true; }
        else if (!strcmp(argv[i], "--chunk")) { bench_chunk_flag = true; }
        else if (!strcmp(argv[i], "--cache")) cache = true;
        else if (!strcmp(argv[i], "--sustain")) {
            sustain = true;
            if (i + 1 < argc) secs = atoi(argv[++i]);
        }
    }
    if (!shapes && !cache && !sustain) shapes = cache = sustain = true;
    g = qw_gpu_open();
    if (!g) die("找不到 Metal 裝置");
    char pw[256], err[4096];
    qw_power_desc(pw, sizeof pw);
    printf("# qwental 補充量測\n\n- GPU：%s\n- %s\n", qw_gpu_name(g), pw);
    if (!qw_gpu_compile(g, mem_src, sizeof mem_src, err, sizeof err)) die(err);
    if (bench_split_flag) { bench_split(); qw_gpu_close(g); return 0; }
    if (bench_chunk_flag) { bench_chunk(); qw_gpu_close(g); return 0; }
    if (shapes) bench_shapes();
    if (cache) bench_cache();
    if (sustain) bench_sustain(secs);
    qw_gpu_close(g);
    return 0;
}
