// GGUF 區塊解碼的 CPU 驗證（src/ggml_dq.c）：介面同 build/dqtest，比對由 tools/dqtest.py 做（DQ_BIN=build/dqcpu）
#include "../src/ggml_dq.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "用法：%s <type> <K> <列數> <in> <out>\n", argv[0]); return 2; }
    const int type = atoi(argv[1]), K = atoi(argv[2]), rows = atoi(argv[3]);
    if (!qw_ggml_dq_ok(type)) { fprintf(stderr, "不支援 type %d\n", type); return 3; }
    FILE *f = fopen(argv[4], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *in = malloc((size_t)n);
    float *out = malloc(sizeof(float) * (size_t)K * rows);
    if (fread(in, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);
    const long rb = n / rows;
    for (int r = 0; r < rows; r++) qw_ggml_dq_row(type, in + r * rb, out + (size_t)r * K, K);
    f = fopen(argv[5], "wb");
    fwrite(out, sizeof(float), (size_t)K * rows, f);
    fclose(f);
    return 0;
}
