// GGUF 區塊解碼的 GPU 驗證：build/dqtest <ggml type> <K> <列數> <輸入原始區塊> <輸出 float>
// 用 kernels/moe_iq.metal 的 dq32<type>（與 iq_tables.metal）解碼；比對由 tools/dqtest.py 做
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

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "用法：%s <type> <K> <列數> <in> <out>\n", argv[0]); return 2; }
    const int type = atoi(argv[1]), K = atoi(argv[2]), rows = atoi(argv[3]);
    FILE *f = fopen(argv[4], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    const long nb = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (nb % rows || K % 32) { fprintf(stderr, "大小不符\n"); return 1; }
    qw_gpu *g = qw_gpu_open();
    char defs[256], err[8192];
    snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY 1\n#define DQ_TEST %d\n", type);
    const size_t dl = strlen(defs);
    char *src = malloc(dl + sizeof SRC_TAB + sizeof SRC_IQ + 2);
    size_t n = 0;
    memcpy(src, defs, dl); n += dl;
    memcpy(src + n, SRC_TAB, sizeof SRC_TAB - 1); n += sizeof SRC_TAB - 1; src[n++] = '\n';
    memcpy(src + n, SRC_IQ, sizeof SRC_IQ - 1); n += sizeof SRC_IQ - 1; src[n++] = 0;
    if (!qw_gpu_compile(g, src, n, err, sizeof err)) { fprintf(stderr, "編譯失敗：%s\n", err); return 1; }
    qw_buf w, y;
    qw_buf_new(g, (size_t)nb, &w);
    qw_buf_new(g, (size_t)rows * K * 4, &y);
    if (fread(w.ptr, 1, (size_t)nb, f) != (size_t)nb) return 1;
    fclose(f);
    uint32_t a[2] = {(uint32_t)K, (uint32_t)(nb / rows)};
    qw_batch *b = qw_batch_begin(g, false);
    qw_batch_kernel(b, qw_kernel_get(g, "dq_test"));
    qw_batch_buf(b, 0, &w, 0);
    qw_batch_buf(b, 1, &y, 0);
    qw_batch_bytes(b, 2, a, sizeof a);
    qw_batch_dispatch(b, (unsigned)rows, 1, 1, 32, 1, 1);
    if (qw_batch_end(b) < 0) { fprintf(stderr, "GPU 失敗\n"); return 1; }
    FILE *o = fopen(argv[5], "wb");
    fwrite(y.ptr, 4, (size_t)rows * K, o);
    fclose(o);
    return 0;
}
