// chat template 對照：讀 OpenAI 格式的 request JSON（或 -a 表示 Anthropic 格式），印出排好的 prompt
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/chat.h"

int main(int argc, char **argv) {
    const int anth = argc > 2 && !strcmp(argv[2], "-a");
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);
    char err[256];
    js *b = js_parse(s, (size_t)n, err, sizeof err);
    if (!b) { fprintf(stderr, "%s\n", err); return 1; }
    chat_req r;
    if (!(anth ? chat_from_anthropic(b, &r, err, sizeof err) : chat_from_openai(b, &r, err, sizeof err))) { fprintf(stderr, "%s\n", err); return 1; }
    sbuf o = {0};
    chat_render(&r, &o);
    fwrite(o.p, 1, o.n, stdout);
    return 0;
}
