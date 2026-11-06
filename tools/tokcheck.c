// tokenizer 對照：把檔案編碼成 token id（逗號分隔印出），並檢查解碼後是否還原原文
// 用法：build/tokcheck 檔案 [-q]（QW_TOK_GGUF=GGUF 檔或目錄：改用 GGUF 的 tokenizer）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/gpu.h"
#include "../src/tok.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法：%s 檔案 [-q]\n", argv[0]); return 2; }
    char err[512];
    double t0 = qw_now();
    qw_tok *t = getenv("QW_TOK_GGUF") ? qw_tok_open_gguf(getenv("QW_TOK_GGUF"), err, sizeof err)
                                      : qw_tok_open("models/Qwen3.8-Flash-Next-mixed-4-8bit", err, sizeof err);
    if (!t) { fprintf(stderr, "%s\n", err); return 1; }
    const double tl = qw_now() - t0;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);
    t0 = qw_now();
    const int cnt = qw_tok_encode(t, s, (size_t)n, NULL, 0);
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)(cnt + 1));
    qw_tok_encode(t, s, (size_t)n, ids, cnt);
    const double te = qw_now() - t0;
    char *back = malloc((size_t)cnt * 256 + 1);
    size_t bl = 0;
    for (int i = 0; i < cnt; i++) bl += (size_t)qw_tok_decode1(t, ids[i], back + bl);
    const int same = bl == (size_t)n && !memcmp(back, s, (size_t)n);
    if (argc < 3)
        for (int i = 0; i < cnt; i++) printf("%d%s", ids[i], i + 1 < cnt ? "," : "\n");
    fprintf(stderr, "載入 %.2f 秒；%ld bytes → %d 個 token（%.1f ms，%.1f MB/s）；解碼還原：%s\n", tl, n, cnt, te * 1e3,
            n / te / 1e6, same ? "相同" : "不同");
    qw_tok_close(t);
    return 0;
}
