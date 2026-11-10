// chat template 對照：讀 request JSON（OpenAI 格式，或 -a 表示 Anthropic 格式），印出排好的 prompt
// 用法：build/chatcheck 檔案 [-a] [-m]（-m：保留特殊 token 字面前的 QW_TOK_PLAIN 標記；預設拿掉，方便和 Jinja 的結果比對）
// 檔案若是 {"body": {...}} 的擷取格式，取 body。CHAT_THINK_UNASKED=1：沒指定思考的 Anthropic 請求也思考
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/chat.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法：%s 檔案 [-a] [-m]\n", argv[0]); return 2; }
    bool anth = false, marks = false;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-a")) anth = true;
        else if (!strcmp(argv[i], "-m")) marks = true;
    }
    if (getenv("CHAT_THINK_UNASKED")) chat_think_unasked = true;
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
    const js *body = js_get(js_get(b, "body"), "messages") ? js_get(b, "body") : b;
    chat_req r;
    if (!(anth ? chat_from_anthropic(body, &r, err, sizeof err) : chat_from_openai(body, &r, err, sizeof err))) { fprintf(stderr, "%s\n", err); return 1; }
    sbuf o = {0};
    chat_render(&r, &o);
    if (!marks) chat_unmark(&o);
    fwrite(o.p, 1, o.n, stdout);
    fprintf(stderr, "thinking=%d effort=%s asked=%d deferred=%d\n", r.thinking, r.effort ? r.effort : "-", r.thinking_asked, r.deferred);
    sb_free(&o);
    chat_free(&r);
    js_free(b);
    free(s);
    return 0;
}
