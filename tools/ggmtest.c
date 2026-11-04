// GGUF 多矩陣 matvec（kernels/qmv_ggm.metal）的 GPU 驗證與計時：
//   build/ggmtest <ggml type> <K> <列數> <NB> <權重原始區塊> <x：NB × K half> <輸出 y：NB × 列數 float> [重複次數]
// 比對由 tools/ggmtest.py 做（numpy 從 GGUF 區塊解碼的參考值）
#include "../src/gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char SRC_TAB[] = {
#embed "../kernels/iq_tables.metal"
};
static const char SRC_IQ[] = {
#embed "../kernels/moe_iq.metal"
};
static const char SRC_GGM[] = {
#embed "../kernels/qmv_ggm.metal"
};

static void *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *b = malloc((size_t)*n);
    if (fread(b, 1, (size_t)*n, f) != (size_t)*n) { fclose(f); free(b); return NULL; }
    fclose(f);
    return b;
}

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "用法：%s <type> <K> <列數> <NB> <W> <x> <y> [重複]\n", argv[0]); return 2; }
    const int type = atoi(argv[1]), K = atoi(argv[2]), rows = atoi(argv[3]), nb = atoi(argv[4]), reps = argc > 8 ? atoi(argv[8]) : 1;
    const int nball = getenv("NBALL") != NULL, nsg = getenv("TC") ? atoi(getenv("TC")) : 0; // TC=NSG：矩陣單元版
    int ptm = 0, ptn = 0, psg = 0; // PF=TM,TN,SG：prefill 版（NB 當作 token 數 M）
    if (getenv("PF") && sscanf(getenv("PF"), "%d,%d,%d", &ptm, &ptn, &psg) != 3) return 2;
    const int maxnb = ptm ? 4096 : 8;
    const int bb = type == 12 ? 144 : type == 13 ? 176 : type == 14 ? 210 : type == 23 ? 136 : type == 30 ? 512 : 0;
    if (!bb || K % 256 || nb < 1 || nb > maxnb || (ptm && (rows % ptn || (ptm != 64 && ptm != 128) || (ptn != 32 && ptn != 64) || (psg != 4 && psg != 8))) || rows < 1 || (nsg && (rows % 32 || K % (32 * nsg) || (nsg != 4 && nsg != 8)))) { fprintf(stderr, "不支援的形狀 / 格式\n"); return 2; }
    long nw, nx;
    void *wb = slurp(argv[5], &nw), *xb = slurp(argv[6], &nx);
    if (!wb || !xb || nw != (long)rows * (K / 256) * bb || nx != (long)nb * K * 2) { fprintf(stderr, "檔案大小不符\n"); return 1; }
    qw_gpu *g = qw_gpu_open();
    char defs[512], err[8192];
    snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY 1\n#define KD %d\n#define NB %d\n#define NMAT 1\n#define GF0 %d\n#define YT float\n%s#define NSG %d\n"
             "#define TM %d\n#define TN %d\n#define SG %d\n%s", K, nball ? 0 : nb > 8 ? 8 : nb, type, nsg ? "#define GGM_TC 1\n" : "", nsg ? nsg : 4,
             ptm, ptn, psg, ptm ? "#define GGM_PF 1\n" : "");
    const size_t dl = strlen(defs);
    char *src = malloc(dl + sizeof SRC_TAB + sizeof SRC_IQ + sizeof SRC_GGM + 3);
    size_t n = 0;
    memcpy(src, defs, dl); n += dl;
    memcpy(src + n, SRC_TAB, sizeof SRC_TAB); n += sizeof SRC_TAB; src[n++] = '\n';
    memcpy(src + n, SRC_IQ, sizeof SRC_IQ); n += sizeof SRC_IQ; src[n++] = '\n';
    memcpy(src + n, SRC_GGM, sizeof SRC_GGM); n += sizeof SRC_GGM; src[n++] = 0;
    if (!qw_gpu_compile(g, src, n, err, sizeof err)) { fprintf(stderr, "編譯失敗：%s\n", err); return 1; }
    qw_buf w, x, y;
    if (!qw_buf_new(g, (size_t)nw, &w) || !qw_buf_new(g, (size_t)(nb > 8 ? nb : 8) * K * 2, &x) || !qw_buf_new(g, (size_t)nb * rows * 4, &y)) return 1;
    memcpy(w.ptr, wb, (size_t)nw);
    memcpy(x.ptr, xb, (size_t)nx);
    struct { uint32_t N[4], off[4], total, xstride; } a = {{(uint32_t)rows, 0, 0, 0}, {0, 0, 0, 0}, (uint32_t)rows, (uint32_t)K};
    char kn[32];
    snprintf(kn, sizeof kn, "qmv_ggm_b%d", nb);
    qw_kernel *k = qw_kernel_get(g, ptm ? "qmm_ggm" : nsg ? "qmv_ggm_tc" : nball ? kn : "qmv_ggm");
    const uint32_t pa[2] = {(uint32_t)nb, (uint32_t)rows};
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        qw_batch *b = qw_batch_begin(g, false);
        for (int r = 0; r < reps; r++) {
            qw_batch_kernel(b, k);
            if (ptm) {
                qw_batch_buf(b, 0, &x, 0);
                qw_batch_buf(b, 1, &w, 0);
                qw_batch_buf(b, 4, &y, 0);
                qw_batch_bytes(b, 5, pa, sizeof pa);
                qw_batch_dispatch(b, (unsigned)(rows / ptn), (unsigned)((nb + ptm - 1) / ptm), 1, 32u * (unsigned)psg, 1, 1);
                continue;
            }
            for (int i = 0; i < 4; i++) qw_batch_buf(b, (unsigned)i, &w, 0);
            qw_batch_buf(b, 4, &x, 0);
            for (int i = 0; i < 4; i++) qw_batch_buf(b, 5u + (unsigned)i, &y, 0);
            qw_batch_bytes(b, 9, &a, sizeof a);
            if (nsg) qw_batch_dispatch(b, (unsigned)rows / 32, 1, 1, 32u * (unsigned)nsg, 1, 1);
            else qw_batch_dispatch(b, ((unsigned)rows + 7) / 8, 1, 1, 256, 1, 1);
        }
        const double tt = qw_batch_end(b);
        if (tt < 0) { fprintf(stderr, "GPU 失敗\n"); return 1; }
        if (tt < best) best = tt;
    }
    if (reps > 1) printf("%.1f µs／次，%.0f GB/s\n", best / reps * 1e6, (double)nw / (best / reps) / 1e9);
    if (reps > 1 && ptm) printf("%.1f TFLOPS\n", 2.0 * nb * rows * K / (best / reps) / 1e12);
    FILE *o = fopen(argv[7], "wb");
    fwrite(y.ptr, 4, (size_t)nb * rows, o);
    fclose(o);
    return 0;
}
