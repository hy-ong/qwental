// 在執行期編譯一個 MSL 檔案，印出完整的錯誤或警告訊息（不需要 Metal Toolchain）。
// 用法：build/mslc file.metal
#include <stdio.h>
#include <stdlib.h>
#include "gpu.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法：%s file.metal\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    char *src = malloc(n);
    if (fread(src, 1, n, f) != (size_t)n) { perror("read"); return 2; }
    fclose(f);
    qw_gpu *g = qw_gpu_open();
    static char err[1 << 20];
    bool ok = qw_gpu_compile(g, src, n, err, sizeof err);
    printf("%s\n%s\n", ok ? "編譯成功" : "編譯失敗", err);
    return ok ? 0 : 1;
}
