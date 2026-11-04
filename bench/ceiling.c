// 第 0 階段：量測 M5 Max 的實際硬體上限
//   - 記憶體頻寬（大小掃描，含小區塊）
//   - kernel 呼叫的固定開銷
//   - neural accelerator 的矩陣運算速度（各種精度）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char mem_src[] = {
#embed "../kernels/ceiling_mem.metal"
};
static const char mm_src[] = {
#embed "../kernels/ceiling_mm.metal"
};

#define GiB (1024.0 * 1024.0 * 1024.0)
#define MiB (1024.0 * 1024.0)

static qw_gpu *g;

static void die(const char *msg) {
    fprintf(stderr, "錯誤：%s\n", msg);
    exit(1);
}

static qw_kernel *need(const char *name) {
    qw_kernel *k = qw_kernel_get(g, name);
    if (!k) {
        fprintf(stderr, "找不到 kernel：%s\n", name);
        exit(1);
    }
    return k;
}

static void compile(const char *src, size_t len, const char *what) {
    char err[8192];
    if (!qw_gpu_compile(g, src, len, err, sizeof err)) {
        fprintf(stderr, "編譯 %s 失敗：\n%s\n", what, err);
        exit(1);
    }
}

// ---------------------------------------------------------------- 頻寬

// 用一個很大的資料池，每次 dispatch 讀不同位置，避免小區塊一直打中快取。
static qw_buf pool, pool2, sink;
static size_t pool_size;

static unsigned threads_for(size_t n_vec) {
    // 每個執行緒至少處理 4 個 uint4；上限約為 40 核 × 4096 個執行緒
    size_t t = n_vec / 4;
    const size_t cap = 40 * 4096;
    if (t > cap) t = cap;
    if (t < 256) t = 256;
    return (unsigned)((t + 255) / 256 * 256);
}

// 回傳 GB/s（以讀取位元組計，copy 則讀+寫）
static double run_bw(const char *kname, size_t bytes, int reps, bool copy) {
    qw_kernel *k = need(kname);
    unsigned n = (unsigned)(bytes / 16);
    unsigned threads = threads_for(n);
    size_t slots = pool_size / bytes;
    if (slots == 0) die("資料池太小");
    double best = 1e30;
    for (int trial = 0; trial < 3; trial++) {
        qw_batch *b = qw_batch_begin(g, false);
        qw_batch_kernel(b, k);
        qw_batch_bytes(b, 2, &n, sizeof n);
        for (int r = 0; r < reps; r++) {
            size_t off = ((size_t)(r + trial * reps) % slots) * bytes;
            qw_batch_buf(b, 0, &pool, off);
            if (copy) qw_batch_buf(b, 1, &pool2, off % (pool2.size - bytes + 1));
            else qw_batch_buf(b, 1, &sink, 0);
            qw_batch_dispatch(b, threads / 256, 1, 1, 256, 1, 1);
        }
        double t = qw_batch_end(b);
        if (t < 0) die("GPU 執行失敗");
        if (t < best) best = t;
    }
    double moved = (double)bytes * reps * (copy ? 2 : 1);
    return moved / best / 1e9;
}

// 區塊讀取：threadgroup 數與大小掃描，回傳最好的 GB/s
static double run_block(size_t bytes, int reps, unsigned *best_ntg, unsigned *best_tsz) {
    qw_kernel *k = need("bw_block");
    unsigned n = (unsigned)(bytes / 16);
    size_t slots = pool_size / bytes;
    double best_gbs = 0;
    const unsigned tszs[] = {256, 512, 1024};
    const unsigned ntgs[] = {80, 160, 320, 640, 1280, 2560, 5120};
    for (int a = 0; a < 3; a++)
        for (int c = 0; c < 7; c++) {
            unsigned tsz = tszs[a], ntg = ntgs[c];
            if ((size_t)ntg * tsz * 4 > n) continue;
            double best = 1e30;
            for (int trial = 0; trial < 3; trial++) {
                qw_batch *b = qw_batch_begin(g, false);
                qw_batch_kernel(b, k);
                qw_batch_bytes(b, 2, &n, sizeof n);
                qw_batch_buf(b, 1, &sink, 0);
                for (int r = 0; r < reps; r++) {
                    qw_batch_buf(b, 0, &pool, ((size_t)(r + trial * reps) % slots) * bytes);
                    qw_batch_dispatch(b, ntg, 1, 1, tsz, 1, 1);
                }
                double t = qw_batch_end(b);
                if (t > 0 && t < best) best = t;
            }
            double gbs = (double)bytes * reps / best / 1e9;
            if (gbs > best_gbs) {
                best_gbs = gbs;
                *best_ntg = ntg;
                *best_tsz = tsz;
            }
        }
    return best_gbs;
}

static void bench_bandwidth(void) {
    printf("\n## 記憶體頻寬\n\n");
    printf("每次 dispatch 讀不同位置（資料池 %.1f GiB），取 3 次最好的結果。\n\n", pool_size / GiB);
    printf("| 讀取大小 | 每次 dispatch 時間 | read (GB/s) | read4 (GB/s) | 規格 614 的 %% |\n");
    printf("|---|---|---|---|---|\n");
    size_t sizes[] = {256 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20, 16 << 20, 32 << 20,
                      64 << 20, 128 << 20, 256 << 20, 512 << 20, (size_t)1 << 30, (size_t)2 << 30};
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        size_t s = sizes[i];
        if (s * 2 > pool_size) break;
        int reps = (int)(8.0 * GiB / s);
        if (reps > 2000) reps = 2000;
        if (reps < 4) reps = 4;
        double r1 = run_bw("bw_read", s, reps, false);
        double r4 = run_bw("bw_read4", s, reps, false);
        double best = r1 > r4 ? r1 : r4;
        double us = s / (best * 1e9) * 1e6;
        if (s < (1 << 20))
            printf("| %zu KiB | %.1f µs | %.0f | %.0f | %.0f%% |\n", s >> 10, us, r1, r4, best / 614 * 100);
        else
            printf("| %zu MiB | %.1f µs | %.0f | %.0f | %.0f%% |\n", s >> 20, us, r1, r4, best / 614 * 100);
        fflush(stdout);
    }
    printf("\n### 區塊讀取（每個 threadgroup 讀一段連續記憶體，掃描 threadgroup 數與大小）\n\n");
    printf("| 讀取大小 | GB/s | 規格 614 的 %% | 最佳設定 |\n|---|---|---|---|\n");
    size_t bsizes[] = {1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20, (size_t)1 << 30};
    for (int i = 0; i < 6; i++) {
        unsigned ntg = 0, tsz = 0;
        int reps = (int)(8.0 * GiB / bsizes[i]);
        if (reps > 2000) reps = 2000;
        double v = run_block(bsizes[i], reps, &ntg, &tsz);
        printf("| %zu MiB | %.0f | %.0f%% | %u 個 threadgroup × %u |\n", bsizes[i] >> 20, v, v / 614 * 100, ntg, tsz);
        fflush(stdout);
    }
    double c = run_bw("bw_copy", (size_t)1 << 30, 8, true);
    printf("\n複製 1 GiB（讀 + 寫合計）：%.0f GB/s\n", c);
}

// ---------------------------------------------------------------- 呼叫開銷

static void bench_dispatch(void) {
    printf("\n## Kernel 呼叫的固定開銷\n\n");
    qw_kernel *k = need("empty_k");
    const int n = 20000;
    printf("| 情境 | 每次的成本 |\n|---|---|\n");
    for (int mode = 0; mode < 2; mode++) {
        double best = 1e30;
        for (int trial = 0; trial < 3; trial++) {
            qw_batch *b = qw_batch_begin(g, mode == 1);
            qw_batch_kernel(b, k);
            for (int i = 0; i < n; i++) qw_batch_dispatch(b, 1, 1, 1, 32, 1, 1);
            double t = qw_batch_end(b);
            if (t < best) best = t;
        }
        printf("| 同一個 command buffer 連續 dispatch（%s） | %.2f µs |\n",
               mode ? "concurrent" : "serial", best / n * 1e6);
    }
    // concurrent + 每次之後加 barrier：最接近 engine 實際的依賴關係
    {
        double best = 1e30;
        for (int trial = 0; trial < 3; trial++) {
            qw_batch *b = qw_batch_begin(g, true);
            qw_batch_kernel(b, k);
            for (int i = 0; i < n; i++) {
                qw_batch_dispatch(b, 1, 1, 1, 32, 1, 1);
                qw_batch_barrier(b);
            }
            double t = qw_batch_end(b);
            if (t < best) best = t;
        }
        printf("| concurrent + 每次之後加 barrier | %.2f µs |\n", best / n * 1e6);
    }
    double lat = qw_gpu_empty_commit_latency(g, k, 300);
    printf("| 每次都送出新的 command buffer 並等待（牆鐘時間） | %.1f µs |\n", lat * 1e6);
}


// ---------------------------------------------------------------- 矩陣運算

typedef struct {
    const char *label;    // 顯示名稱
    const char *ta, *tb, *tc; // MSL 型別
    double ea, eb, ec;    // 每個元素的位元組數
    int check;            // 0 不驗證；1 = half×half→float；2 = int8×int8→int32；3 = half×int4→float；4 = int8×int4→int32
} mm_type;

static const mm_type MM_TYPES[] = {
    {"FP16 × FP16 → FP32", "half", "half", "float", 2, 2, 4, 1},
    {"FP16 × FP16 → FP16", "half", "half", "half", 2, 2, 2, 0},
    {"BF16 × BF16 → FP32", "bfloat", "bfloat", "float", 2, 2, 4, 0},
    {"INT8 × INT8 → INT32", "int8_t", "int8_t", "int32_t", 1, 1, 4, 2},
    {"FP16 × INT8 → FP32", "half", "int8_t", "float", 2, 1, 4, 0},
    {"FP16 × INT4 → FP32", "half", "int4b_format", "float", 2, 0.5, 4, 3},
    {"INT8 × INT4 → INT32", "int8_t", "int4b_format", "int32_t", 1, 0.5, 4, 4},
    {"FP16 × FP4 → FP32", "half", "metal::metal_fp4_e2m1_format", "float", 2, 0.5, 4, 0},
    {"FP8 × FP8 → FP32", "metal::metal_fp8_e4m3_format", "metal::metal_fp8_e4m3_format", "float", 1, 1, 4, 0},
    {"FP4 × FP4 → FP32", "metal::metal_fp4_e2m1_format", "metal::metal_fp4_e2m1_format", "float", 0.5, 0.5, 4, 0},
};

typedef struct { int tm, tn, sg; } mm_tile;

static uint64_t rng = 0x243f6a8885a308d3ull;
static uint8_t rnd8(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint8_t)(rng >> 32);
}

// 填入隨機位元組，但讓每種浮點格式都保持有限值（避免 NaN / Inf）
static void fill_random(qw_buf *b) {
    uint8_t *p = b->ptr;
    for (size_t i = 0; i < b->size; i++) p[i] = rnd8() & 0x3b;
}

// 「格式型別」（int4b_format、fp8、fp4）不能直接當指標型別，資料用 uchar 指向
static const char *ptr_type(const char *t) { return strstr(t, "format") ? "uchar" : t; }

static qw_kernel *build_mm(const mm_type *t, mm_tile tile, char *err, size_t errlen) {
    char name[96];
    snprintf(name, sizeof name, "mm_%s_%s_%s_%dx%dx%d", t->ta, t->tb, t->tc, tile.tm, tile.tn, tile.sg);
    for (char *c = name; *c; c++)
        if (*c == ':') *c = '_'; // 「metal::」不能出現在函式名稱裡
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    size_t cap = sizeof mm_src + 512;
    char *src = malloc(cap);
    int n = snprintf(src, cap,
                     "#define mm %s\n#define TA %s\n#define TB %s\n#define TC %s\n#define PA %s\n#define PB %s\n"
                     "#define TM %d\n#define TN %d\n#define SG %d\n",
                     name, t->ta, t->tb, t->tc, ptr_type(t->ta), ptr_type(t->tb), tile.tm, tile.tn, tile.sg);
    memcpy(src + n, mm_src, sizeof mm_src);
    bool ok = qw_gpu_compile(g, src, n + sizeof mm_src, err, errlen);
    free(src);
    if (!ok) return NULL;
    return qw_kernel_get(g, name);
}

static float h2f(uint16_t h) {
    _Float16 x;
    memcpy(&x, &h, 2);
    return (float)x;
}

// 抽查幾個輸出元素，跟 CPU 算的結果比對。回傳最大相對誤差；不驗證則回傳 -1。
// 4-bit 打包資料的第 e 個元素。nib_order：0 = 偶數元素放低 4 位，1 = 放高 4 位
static int nib_order = -1;
static int get_i4(const uint8_t *p, size_t e, int order) {
    uint8_t byte = p[e / 2];
    int v = ((e & 1) ^ order) ? (byte >> 4) : (byte & 15);
    return v >= 8 ? v - 16 : v; // 有號
}

static double verify_order(const mm_type *t, qw_buf *A, qw_buf *B, qw_buf *C, int M, int N, int K, int order);

static double verify(const mm_type *t, qw_buf *A, qw_buf *B, qw_buf *C, int M, int N, int K) {
    if (!t->check) return -1;
    if (t->check >= 3) {
        // 兩種排列都試，取符合的那個
        double e0 = verify_order(t, A, B, C, M, N, K, 0);
        double e1 = verify_order(t, A, B, C, M, N, K, 1);
        nib_order = e0 <= e1 ? 0 : 1;
        return e0 <= e1 ? e0 : e1;
    }
    return verify_order(t, A, B, C, M, N, K, 0);
}

static double verify_order(const mm_type *t, qw_buf *A, qw_buf *B, qw_buf *C, int M, int N, int K, int order) {
    double worst = 0;
    for (int s = 0; s < 64; s++) {
        int i = (int)((s * 2654435761u) % (unsigned)M), j = (int)((s * 40503u + 17) % (unsigned)N);
        double ref = 0, got;
        if (t->check == 1) {
            const uint16_t *a = A->ptr, *b = B->ptr;
            for (int k = 0; k < K; k++) ref += (double)h2f(a[(size_t)i * K + k]) * h2f(b[(size_t)k * N + j]);
            got = ((const float *)C->ptr)[(size_t)i * N + j];
        } else if (t->check == 2) {
            const int8_t *a = A->ptr, *b = B->ptr;
            for (int k = 0; k < K; k++) ref += (double)a[(size_t)i * K + k] * b[(size_t)k * N + j];
            got = ((const int32_t *)C->ptr)[(size_t)i * N + j];
        } else if (t->check == 3) {
            const uint16_t *a = A->ptr;
            const uint8_t *b = B->ptr;
            for (int k = 0; k < K; k++) ref += (double)h2f(a[(size_t)i * K + k]) * get_i4(b, (size_t)k * N + j, order);
            got = ((const float *)C->ptr)[(size_t)i * N + j];
        } else {
            const int8_t *a = A->ptr;
            const uint8_t *b = B->ptr;
            for (int k = 0; k < K; k++) ref += (double)a[(size_t)i * K + k] * get_i4(b, (size_t)k * N + j, order);
            got = ((const int32_t *)C->ptr)[(size_t)i * N + j];
        }
        double denom = ref < 0 ? -ref : ref;
        if (denom < 1) denom = 1;
        double e = (got - ref) / denom;
        if (e < 0) e = -e;
        if (e > worst) worst = e;
    }
    return worst;
}

// 回傳 TFLOPS（或 TOPS）；失敗回傳負值，原因寫進 err
static double run_mm(const mm_type *t, mm_tile tile, int M, int N, int K, int reps,
                     qw_buf *A, qw_buf *B, qw_buf *C, double *err_out, char *err, size_t errlen) {
    qw_kernel *k = build_mm(t, tile, err, errlen);
    if (!k) return -1;
    unsigned threads = 32u * tile.sg;
    if (threads > qw_kernel_max_threads(k)) {
        snprintf(err, errlen, "threadgroup 太大");
        return -1;
    }
    uint32_t mnk[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 0};
    double best = 1e30;
    for (int trial = 0; trial < 3; trial++) {
        memset(C->ptr, 0, (size_t)(M * (double)N * t->ec));
        qw_batch *b = qw_batch_begin(g, false);
        qw_batch_kernel(b, k);
        qw_batch_buf(b, 0, A, 0);
        qw_batch_buf(b, 1, B, 0);
        qw_batch_buf(b, 2, C, 0);
        qw_batch_bytes(b, 3, mnk, sizeof mnk);
        for (int r = 0; r < (trial == 0 ? 1 : reps); r++)
            qw_batch_dispatch(b, N / tile.tn, M / tile.tm, 1, threads, 1, 1);
        double tt = qw_batch_end(b);
        if (tt < 0) {
            snprintf(err, errlen, "GPU 執行失敗");
            return -1;
        }
        if (trial == 0) {
            // 第一次只跑一次，拿來驗證數值（C 從 0 開始）
            *err_out = verify(t, A, B, C, M, N, K);
            continue;
        }
        tt /= reps;
        if (tt < best) best = tt;
    }
    return 2.0 * M * N * K / best / 1e12;
}

static void bench_matmul(void) {
    printf("\n## Neural accelerator 矩陣運算\n\n");
    const int M = 4096, N = 4096, K = 4096;
    qw_buf A, B, C;
    if (!qw_buf_new(g, (size_t)M * K * 2, &A) || !qw_buf_new(g, (size_t)K * N * 2, &B) ||
        !qw_buf_new(g, (size_t)M * N * 4, &C))
        die("配置矩陣失敗");
    fill_random(&A);
    fill_random(&B);
    char err[8192];

    // 1) 先用 FP16 找最好的區塊大小
    const mm_tile tiles[] = {{32, 32, 1}, {64, 32, 1}, {64, 32, 4}, {64, 64, 4}, {128, 64, 4},
                             {128, 128, 4}, {64, 64, 2}, {128, 64, 8}, {128, 128, 8}};
    const int ntiles = sizeof tiles / sizeof tiles[0];
    printf("### 區塊大小掃描（FP16 × FP16 → FP32，%d×%d×%d）\n\n", M, N, K);
    printf("| 區塊（M×N，SIMD group 數） | TFLOPS | 抽查誤差 |\n|---|---|---|\n");
    double tf[16] = {0};
    for (int i = 0; i < ntiles; i++) {
        double e = -1;
        double v = run_mm(&MM_TYPES[0], tiles[i], M, N, K, 20, &A, &B, &C, &e, err, sizeof err);
        tf[i] = v;
        if (v < 0) {
            char *nl = strchr(err, '\n');
            if (nl) *nl = 0;
            printf("| %d×%d，%d | 失敗：%.120s | |\n", tiles[i].tm, tiles[i].tn, tiles[i].sg, err);
        } else
            printf("| %d×%d，%d | %.1f | %.1e |\n", tiles[i].tm, tiles[i].tn, tiles[i].sg, v, e);
        fflush(stdout);
    }
    int bi = 0;
    for (int i = 1; i < ntiles; i++) if (tf[i] > tf[bi]) bi = i;
    int b2 = bi == 0 ? 1 : 0;
    for (int i = 0; i < ntiles; i++) if (i != bi && tf[i] > tf[b2]) b2 = i;
    if (tf[bi] <= 0) {
        printf("\n所有區塊都失敗，略過精度比較。\n");
        return;
    }

    // 2) 各種精度，用最好的兩種區塊大小
    printf("\n### 各種精度（取區塊 %d×%d/%d 和 %d×%d/%d 中較好的）\n\n",
           tiles[bi].tm, tiles[bi].tn, tiles[bi].sg, tiles[b2].tm, tiles[b2].tn, tiles[b2].sg);
    printf("| 精度 | TFLOPS / TOPS | 對 FP16→FP32 的倍數 | 抽查誤差 |\n|---|---|---|---|\n");
    double base = tf[bi];
    for (size_t i = 0; i < sizeof MM_TYPES / sizeof MM_TYPES[0]; i++) {
        const mm_type *t = &MM_TYPES[i];
        double best = -1, e = -1;
        char firsterr[256] = "";
        for (int j = 0; j < 2; j++) {
            double ee = -1;
            double v = run_mm(t, tiles[j ? b2 : bi], M, N, K, 20, &A, &B, &C, &ee, err, sizeof err);
            if (v < 0 && !firsterr[0]) {
                char *nl = strchr(err, '\n');
                if (nl) *nl = 0;
                snprintf(firsterr, sizeof firsterr, "%.200s", err);
            }
            if (v > best) { best = v; e = ee; }
        }
        if (best < 0)
            printf("| %s | 失敗：%s | | |\n", t->label, firsterr);
        else if (e >= 0 && t->check >= 3)
            printf("| %s | %.1f | %.2f× | %.1e（偶數元素在%s 4 位、有號） |\n", t->label, best, best / base, e,
                   nib_order ? "高" : "低");
        else if (e >= 0)
            printf("| %s | %.1f | %.2f× | %.1e |\n", t->label, best, best / base, e);
        else
            printf("| %s | %.1f | %.2f× | 未驗證 |\n", t->label, best, best / base);
        fflush(stdout);
    }
    qw_buf_free(&A);
    qw_buf_free(&B);
    qw_buf_free(&C);
}

// ---------------------------------------------------------------- main

int main(int argc, char **argv) {
    bool do_bw = true, do_disp = true, do_mm = true;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--only-bw")) do_disp = do_mm = false;
        if (!strcmp(argv[i], "--only-dispatch")) do_bw = do_mm = false;
        if (!strcmp(argv[i], "--only-mm")) do_bw = do_disp = false;
    }
    g = qw_gpu_open();
    if (!g) die("找不到 Metal 裝置");
    printf("# Qwental 硬體上限量測\n\n");
    printf("- GPU：%s\n", qw_gpu_name(g));
    printf("- 建議最大工作記憶體：%.1f GiB\n", qw_gpu_max_working_set(g) / GiB);
    char pw[256];
    qw_power_desc(pw, sizeof pw);
    printf("- %s\n", pw);
    if (qw_on_battery()) printf("- ⚠️ 目前用電池，GPU 可能降速，絕對數字請以插電時的結果為準\n");

    compile(mem_src, sizeof mem_src, "ceiling_mem.metal");

    if (do_bw) {
        pool_size = (size_t)6 << 30;
        if (!qw_buf_new(g, pool_size, &pool)) die("配置資料池失敗");
        if (!qw_buf_new(g, (size_t)2 << 30, &pool2)) die("配置複製目標失敗");
        if (!qw_buf_new(g, 4096 * 4, &sink)) die("配置 sink 失敗");
        // 先寫過一次，確保實體分頁都已配置
        memset(pool.ptr, 1, pool.size);
        memset(pool2.ptr, 2, pool2.size);
        bench_bandwidth();
        qw_buf_free(&pool);
        qw_buf_free(&pool2);
        qw_buf_free(&sink);
    }
    if (do_disp) bench_dispatch();
    if (do_mm) bench_matmul();

    qw_gpu_close(g);
    return 0;
}
