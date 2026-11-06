// Qwental HTTP server：Anthropic Messages API（/v1/messages，含串流）與 OpenAI Chat Completions（/v1/chat/completions）
// 一次處理一個請求（單一使用者）。
// 用法：build/qwental-serve [--gguf models/iq3s-gguf] [--int8] [--int8-qk] [--int8-hyper] [--fast-head] [--exact-spec] [--port 8080] [--ctx 131072] [--host 127.0.0.1] [--slots 3] [--cache-dir cache] [--version]
// --gguf：整個模型改用該目錄的 GGUF（Strata 的 IQ3_S 等；等同 QW_GGUF=目錄 QW_GGUF_PARTS=all）。
//   這時 MLX 目錄（--model）是選用的：沒有時 tokenizer 讀 GGUF 的 metadata，dflash2/ 改找 GGUF 目錄的同層
//   （QW_GGUF_PARTS 只選一部分時其餘權重仍來自 MLX 目錄）
// --fast-head：GGUF 的 Q6_K lm_head 走矩陣單元（權重捨入成 FP16，同 llama.cpp 的 mul_mm；輸出與預設略有差異；等同 QW_LM_TC=2）
// --exact-spec：MLX profile 的推測解碼逐位元等於 greedy（kernel 改用 safe 數學模式；MLX 輸出與預設略有差異；GGUF 不受影響；等同 QW_EXACT_SPEC=1）
// --int8：INT8 prefill（routed expert 走 INT8 × INT8，GGUF 另外加上 dense 投影；decode 不變；等同 QW_INT8=1）
// --int8-qk：再加上 attention 的 Q·Kᵀ（indexer 打分與稀疏核心）走 INT8；--int8-hyper：再加上 GGUF hyper-connection 的 mix_down 走 INT8。
//   兩者都隱含 --int8，各自會讓輸出再多一點差異（等同 QW_INT8=lin,attn,shared,moe[,qk][,hyper]）
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include "chat.h"
#include "engine.h"
#include "gpu.h"
#include "json.h"
#include "tok.h"

static const char *MODEL_DIR = "models/Qwen3.8-Flash-Next-mixed-4-8bit";
static qw_engine *E;
static qw_tok *TK;
static int32_t IM_END, EOT;
static int CTX;

// ---------------------------------------------------------------- HTTP
typedef struct {
    int fd;
    bool dead;
} conn;

static bool wr(conn *c, const char *s, size_t n) {
    while (n && !c->dead) {
        const ssize_t k = write(c->fd, s, n);
        if (k <= 0) { c->dead = true; break; }
        s += k;
        n -= (size_t)k;
    }
    return !c->dead;
}

static void respond(conn *c, int code, const char *ctype, const char *body, size_t n) {
    char h[256];
    const int hn = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", code,
                            code == 200 ? "OK" : code == 404 ? "Not Found" : "Bad Request", ctype, n);
    wr(c, h, (size_t)hn);
    wr(c, body, n);
}

static void respond_err(conn *c, int code, const char *msg) {
    sbuf b = {0};
    sb_puts(&b, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":");
    sb_json_str(&b, msg, strlen(msg), false);
    sb_puts(&b, "}}");
    respond(c, code, "application/json", b.p, b.n);
    sb_free(&b);
}

// 讀一個請求：回傳 body（呼叫端 free），method / path 寫進參數
static char *read_req(int fd, char *method, char *path, size_t *blen) {
    size_t cap = 65536, n = 0;
    char *buf = malloc(cap + 1);
    char *he = NULL;
    while (!he) {
        if (n == cap) { cap *= 2; buf = realloc(buf, cap + 1); }
        const ssize_t k = read(fd, buf + n, cap - n);
        if (k <= 0) { free(buf); return NULL; }
        n += (size_t)k;
        buf[n] = 0;
        he = strstr(buf, "\r\n\r\n");
    }
    if (sscanf(buf, "%15s %1023s", method, path) != 2) { free(buf); return NULL; }
    size_t clen = 0;
    for (char *l = buf; l < he; l = strstr(l, "\r\n") + 2) {
        if (!strncasecmp(l, "Content-Length:", 15)) clen = strtoul(l + 15, NULL, 10);
        if (!strstr(l, "\r\n")) break;
    }
    const size_t hl = (size_t)(he + 4 - buf);
    if (hl + clen > cap) { cap = hl + clen; buf = realloc(buf, cap + 1); }
    while (n < hl + clen) {
        const ssize_t k = read(fd, buf + n, hl + clen - n);
        if (k <= 0) { free(buf); return NULL; }
        n += (size_t)k;
    }
    char *body = malloc(clen + 1);
    memcpy(body, buf + hl, clen);
    body[clen] = 0;
    *blen = clen;
    free(buf);
    return body;
}

// ---------------------------------------------------------------- 生成的共用部分
typedef struct {
    conn *c;
    bool stream, openai;
    const char *model;
    char id[48];
    chat_parser parser;
    char u8[8]; // 不完整的 UTF-8 尾巴
    int u8n;
    // Anthropic 串流：目前開著的區塊
    int block;      // 已開的區塊數
    int open_type;  // 0 無、1 thinking、2 text
    // 非串流：累積
    sbuf think, text, calls; // calls：Anthropic tool_use 區塊（逗號分隔）或 OpenAI tool_calls
    int ncalls;
    int out_tokens;
    const char *stops[4]; // 停止字串
    int nstops;
    sbuf tail;            // 最近輸出的文字（比對停止字串用）
    const char *hit;      // 命中的停止字串
} gen_ctx;

static void sse(gen_ctx *g, const char *event, sbuf *data) {
    sbuf b = {0};
    if (event) { sb_puts(&b, "event: "); sb_puts(&b, event); sb_puts(&b, "\n"); }
    sb_puts(&b, "data: ");
    sb_put(&b, data->p, data->n);
    sb_puts(&b, "\n\n");
    wr(g->c, b.p, b.n);
    sb_free(&b);
}

static void anth_close_block(gen_ctx *g) {
    if (!g->open_type) return;
    sbuf d = {0};
    if (g->open_type == 1) { // thinking 區塊需要 signature
        sb_printf(&d, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"qwental\"}}", g->block - 1);
        sse(g, "content_block_delta", &d);
        d.n = 0;
    }
    sb_printf(&d, "{\"type\":\"content_block_stop\",\"index\":%d}", g->block - 1);
    sse(g, "content_block_stop", &d);
    sb_free(&d);
    g->open_type = 0;
}

static void on_event(void *ctx, chat_ev_type t, const char *s, size_t n, const chat_call *call) {
    gen_ctx *g = ctx;
    if (!g->stream) {
        if (t == CEV_THINK) sb_put(&g->think, s, n);
        else if (t == CEV_TEXT) sb_put(&g->text, s, n);
        else {
            if (g->ncalls++) sb_puts(&g->calls, ",");
            sbuf a = {0};
            js_dump(&a, call->args, false);
            if (g->openai) {
                sb_puts(&g->calls, "{\"id\":");
                sb_json_str(&g->calls, call->id, strlen(call->id), false);
                sb_puts(&g->calls, ",\"type\":\"function\",\"function\":{\"name\":");
                sb_json_str(&g->calls, call->name, strlen(call->name), false);
                sb_puts(&g->calls, ",\"arguments\":");
                sb_json_str(&g->calls, a.p, a.n, false);
                sb_puts(&g->calls, "}}");
            } else {
                sb_puts(&g->calls, "{\"type\":\"tool_use\",\"id\":");
                sb_json_str(&g->calls, call->id, strlen(call->id), false);
                sb_puts(&g->calls, ",\"name\":");
                sb_json_str(&g->calls, call->name, strlen(call->name), false);
                sb_puts(&g->calls, ",\"input\":");
                sb_put(&g->calls, a.p, a.n);
                sb_puts(&g->calls, "}");
            }
            sb_free(&a);
        }
        return;
    }
    sbuf d = {0};
    if (g->openai) {
        sb_printf(&d, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", g->id, (long)time(NULL));
        sb_json_str(&d, g->model, strlen(g->model), false);
        sb_puts(&d, ",\"choices\":[{\"index\":0,\"delta\":{");
        if (t == CEV_THINK) { sb_puts(&d, "\"reasoning_content\":"); sb_json_str(&d, s, n, false); }
        else if (t == CEV_TEXT) { sb_puts(&d, "\"content\":"); sb_json_str(&d, s, n, false); }
        else {
            sbuf a = {0};
            js_dump(&a, call->args, false);
            sb_printf(&d, "\"tool_calls\":[{\"index\":%d,\"id\":", g->ncalls++);
            sb_json_str(&d, call->id, strlen(call->id), false);
            sb_puts(&d, ",\"type\":\"function\",\"function\":{\"name\":");
            sb_json_str(&d, call->name, strlen(call->name), false);
            sb_puts(&d, ",\"arguments\":");
            sb_json_str(&d, a.p, a.n, false);
            sb_puts(&d, "}}]");
            sb_free(&a);
        }
        sb_puts(&d, "},\"finish_reason\":null}]}");
        sse(g, NULL, &d);
        sb_free(&d);
        return;
    }
    // Anthropic：依類型開關區塊
    const int want = t == CEV_THINK ? 1 : t == CEV_TEXT ? 2 : 3;
    if (g->open_type != want || want == 3) {
        anth_close_block(g);
        if (want == 1) sb_printf(&d, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}", g->block);
        else if (want == 2) sb_printf(&d, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}", g->block);
        else {
            sb_printf(&d, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"tool_use\",\"id\":", g->block);
            sb_json_str(&d, call->id, strlen(call->id), false);
            sb_puts(&d, ",\"name\":");
            sb_json_str(&d, call->name, strlen(call->name), false);
            sb_puts(&d, ",\"input\":{}}}");
        }
        sse(g, "content_block_start", &d);
        d.n = 0;
        g->block++;
        g->open_type = want;
        g->ncalls += want == 3;
    }
    if (want == 3) {
        sbuf a = {0};
        js_dump(&a, call->args, false);
        sb_printf(&d, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":", g->block - 1);
        sb_json_str(&d, a.p, a.n, false);
        sb_puts(&d, "}}");
        sb_free(&a);
        sse(g, "content_block_delta", &d);
        d.n = 0;
        sb_printf(&d, "{\"type\":\"content_block_stop\",\"index\":%d}", g->block - 1);
        sse(g, "content_block_stop", &d);
        g->open_type = 0;
    } else {
        sb_printf(&d, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"%s\",\"%s\":", g->block - 1,
                  want == 1 ? "thinking_delta" : "text_delta", want == 1 ? "thinking" : "text");
        sb_json_str(&d, s, n, false);
        sb_puts(&d, "}}");
        sse(g, "content_block_delta", &d);
    }
    sb_free(&d);
}

// 每個 token：解碼成 bytes，湊滿完整的 UTF-8 字元才送進解析器
static bool on_token(void *ctx, int32_t t) {
    gen_ctx *g = ctx;
    g->out_tokens++;
    char buf[300];
    memcpy(buf, g->u8, (size_t)g->u8n);
    int n = g->u8n + qw_tok_decode1(TK, t, buf + g->u8n);
    // 尾巴的不完整字元留到下次
    int cut = n, k = n - 1;
    while (k >= 0 && k >= n - 4 && ((unsigned char)buf[k] & 0xC0) == 0x80) k--;
    if (k >= 0 && k >= n - 4) {
        const unsigned char c = (unsigned char)buf[k];
        const int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        if (n - k < need) cut = k;
    }
    g->u8n = n - cut;
    memcpy(g->u8, buf + cut, (size_t)g->u8n);
    if (cut && getenv("QW_LOG_RAW")) fwrite(buf, 1, (size_t)cut, stderr);
    if (cut && g->nstops) { // 停止字串：命中就只送出它之前的部分
        const size_t old = g->tail.n;
        sb_put(&g->tail, buf, (size_t)cut);
        for (int i = 0; i < g->nstops; i++) {
            const char *h = strstr(g->tail.p, g->stops[i]);
            if (!h) continue;
            const size_t upto = (size_t)(h - g->tail.p);
            if (upto > old) chat_parser_feed(&g->parser, g->tail.p + old, upto - old, on_event, g);
            g->hit = g->stops[i];
            return false;
        }
        if (g->tail.n > 4096) { // 只留最後一段
            memmove(g->tail.p, g->tail.p + g->tail.n - 256, 256);
            g->tail.n = 256;
            g->tail.p[256] = 0;
        }
    }
    if (cut) chat_parser_feed(&g->parser, buf, (size_t)cut, on_event, g);
    return !g->c->dead;
}

static void gen_id(char *id, const char *prefix) {
    snprintf(id, 48, "%s%08x%08x", prefix, (unsigned)arc4random(), (unsigned)arc4random());
}

static void handle_chat(conn *c, const char *body, size_t blen, bool openai) {
    char err[256];
    js *b = js_parse(body, blen, err, sizeof err);
    if (!b) { respond_err(c, 400, err); return; }
    chat_req r;
    if (!(openai ? chat_from_openai(b, &r, err, sizeof err) : chat_from_anthropic(b, &r, err, sizeof err))) {
        respond_err(c, 400, err);
        js_free(b);
        return;
    }
    if (getenv("QW_NO_THINK")) r.thinking = false;
    if (getenv("QW_LOG_REQ")) { // 除錯：請求的參數（不含訊息內容）
        fprintf(stderr, "[req]");
        for (const js *k = b->kid; k; k = k->next) {
            if (!strcmp(k->key, "messages") || !strcmp(k->key, "system") || !strcmp(k->key, "tools")) { fprintf(stderr, " %s(%d)", k->key, k->n); continue; }
            sbuf v = {0};
            js_dump(&v, k, false);
            fprintf(stderr, " %s=%.200s", k->key, v.p);
            sb_free(&v);
        }
        fprintf(stderr, "\n");
    }
    sbuf prompt = {0};
    chat_render(&r, &prompt);
    if (getenv("QW_LOG_PROMPT")) { // 除錯：每個請求的完整 prompt 存成檔案
        static int seq = 0;
        char pth[256];
        snprintf(pth, sizeof pth, "%s/req%03d.txt", getenv("QW_LOG_PROMPT"), seq++);
        FILE *f = fopen(pth, "wb");
        if (f) { fwrite(prompt.p, 1, prompt.n, f); fclose(f); }
    }
    const int np = qw_tok_encode(TK, prompt.p, prompt.n, NULL, 0);
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)np);
    qw_tok_encode(TK, prompt.p, prompt.n, ids, np);
    if (np + 64 > CTX) { // 超過 context：回 Anthropic 標準錯誤，Claude Code 會自動壓縮對話
        char m[160];
        snprintf(m, sizeof m, "prompt is too long: %d tokens > %d maximum", np, CTX - 64);
        respond_err(c, 400, m);
        free(ids);
        sb_free(&prompt);
        chat_free(&r);
        js_free(b);
        return;
    }
    const char *model = js_str(b, "model");
    gen_ctx g = {.c = c, .stream = js_bool(b, "stream", false), .openai = openai, .model = model ? model : "qwental"};
    gen_id(g.id, openai ? "chatcmpl-" : "msg_");
    chat_parser_init(&g.parser, r.thinking, r.tools);
    int max_tokens = (int)js_num(b, "max_tokens", openai ? js_num(b, "max_completion_tokens", 32768) : 4096);
    if (max_tokens > CTX - np - 16) max_tokens = CTX - np - 16;
    // stop_sequences（Anthropic）/ stop（OpenAI）：在輸出的文字裡比對
    const js *ss = js_get(b, openai ? "stop" : "stop_sequences");
    if (ss && ss->t == JS_STR) { g.stops[0] = ss->s; g.nstops = 1; }
    for (const js *x = ss && ss->t == JS_ARR ? ss->kid : NULL; x && g.nstops < 4; x = x->next)
        if (x->t == JS_STR && x->len) g.stops[g.nstops++] = x->s;
    const int32_t stops[2] = {IM_END, EOT};
    const char *ev = getenv("QW_TEMP");
    // 持久化點：工具定義結束前的 token 邊界、system 訊息結尾（第一個 <|im_end|> 之後）
    int persist[2], npersist = 0;
    {
        const char *te = strstr(prompt.p, "</tools>");
        if (te) {
            const size_t off = (size_t)(te - prompt.p);
            size_t acc = 0;
            char tb[300];
            for (int i = 0; i < np; i++) {
                const size_t l = (size_t)qw_tok_decode1(TK, ids[i], tb);
                if (acc + l > off) { if (i >= 1024) persist[npersist++] = i; break; }
                acc += l;
            }
        }
        for (int i = 0; i < np; i++)
            if (ids[i] == IM_END) { if (i + 1 >= 1024 && i + 1 < np) persist[npersist++] = i + 1; break; }
    }
    qw_gen gp = {.temperature = (float)js_num(b, "temperature", ev ? atof(ev) : 0.7),
                 .top_k = (int)js_num(b, "top_k", 20),
                 .top_p = (float)js_num(b, "top_p", 0.95),
                 .seed = (unsigned long long)js_num(b, "seed", (double)arc4random()),
                 .max_tokens = max_tokens,
                 .stop = stops,
                 .nstop = 2,
                 .spec = !getenv("QW_NO_SPEC"),
                 .persist = persist,
                 .npersist = npersist};
    if (g.stream) {
        const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        wr(c, h, strlen(h));
        if (!openai) {
            sbuf d = {0};
            sb_printf(&d, "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", g.id);
            sb_json_str(&d, g.model, strlen(g.model), false);
            sb_printf(&d, ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":%d,\"output_tokens\":1}}}", np);
            sse(&g, "message_start", &d);
            sb_free(&d);
        }
    }
    qw_gen_stats st;
    const double t0 = qw_now();
    const int fin = qw_engine_generate_ex(E, ids, np, &gp, on_token, &g, &st);
    chat_parser_end(&g.parser, on_event, &g);
    const double dt = qw_now() - t0;
    fprintf(stderr, "[%s] prompt %d（重用 %d）%.0f ms；生成 %d 個 token %.0f ms（%.1f tok/s）；%d 輪、接受 %d 個猜測（n-gram %d 輪）；結束 %d\n",
            openai ? "openai" : "anthropic", np, st.cached, st.prefill_ms, st.gen, st.gen_ms, st.gen ? st.gen / st.gen_ms * 1e3 : 0.0,
            st.rounds, st.accepted, st.ngram_rounds, fin);
    (void)dt;
    const bool tool = g.ncalls > 0;
    const char *sr = g.hit ? (openai ? "stop" : "stop_sequence") : fin == QW_FIN_LENGTH ? (openai ? "length" : "max_tokens")
                   : tool ? (openai ? "tool_calls" : "tool_use") : (openai ? "stop" : "end_turn");
    if (g.stream) {
        sbuf d = {0};
        if (openai) {
            sb_printf(&d, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", g.id, (long)time(NULL));
            sb_json_str(&d, g.model, strlen(g.model), false);
            sb_printf(&d, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                      sr, np, st.gen, np + st.gen);
            sse(&g, NULL, &d);
            wr(c, "data: [DONE]\n\n", 14);
        } else {
            anth_close_block(&g);
            sb_printf(&d, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":%s%s%s},\"usage\":{\"output_tokens\":%d}}", sr,
                      g.hit ? "\"" : "", g.hit ? g.hit : "null", g.hit ? "\"" : "", st.gen);
            sse(&g, "message_delta", &d);
            d.n = 0;
            sb_puts(&d, "{\"type\":\"message_stop\"}");
            sse(&g, "message_stop", &d);
        }
        sb_free(&d);
    } else {
        sbuf o = {0};
        if (openai) {
            sb_printf(&o, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", g.id, (long)time(NULL));
            sb_json_str(&o, g.model, strlen(g.model), false);
            sb_puts(&o, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
            sb_json_str(&o, g.text.p ? g.text.p : "", g.text.n, false);
            if (g.think.n) { sb_puts(&o, ",\"reasoning_content\":"); sb_json_str(&o, g.think.p, g.think.n, false); }
            if (tool) { sb_puts(&o, ",\"tool_calls\":["); sb_put(&o, g.calls.p, g.calls.n); sb_puts(&o, "]"); }
            sb_printf(&o, "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}", sr, np,
                      st.gen, np + st.gen);
        } else {
            sb_printf(&o, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", g.id);
            sb_json_str(&o, g.model, strlen(g.model), false);
            sb_puts(&o, ",\"content\":[");
            int k = 0;
            if (g.think.n) {
                sb_puts(&o, "{\"type\":\"thinking\",\"thinking\":");
                sb_json_str(&o, g.think.p, g.think.n, false);
                sb_puts(&o, ",\"signature\":\"qwental\"}");
                k++;
            }
            if (g.text.n) {
                if (k++) sb_puts(&o, ",");
                sb_puts(&o, "{\"type\":\"text\",\"text\":");
                sb_json_str(&o, g.text.p, g.text.n, false);
                sb_puts(&o, "}");
            }
            if (tool) { if (k++) sb_puts(&o, ","); sb_put(&o, g.calls.p, g.calls.n); }
            sb_printf(&o, "],\"stop_reason\":\"%s\",\"stop_sequence\":", sr);
            if (g.hit) sb_json_str(&o, g.hit, strlen(g.hit), false);
            else sb_puts(&o, "null");
            sb_printf(&o, ",\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}}", np, st.gen);
        }
        respond(c, 200, "application/json", o.p, o.n);
        sb_free(&o);
    }
    sb_free(&g.think);
    sb_free(&g.text);
    sb_free(&g.calls);
    sb_free(&g.tail);
    chat_parser_free(&g.parser);
    free(ids);
    sb_free(&prompt);
    chat_free(&r);
    js_free(b);
}

static void handle_count(conn *c, const char *body, size_t blen) {
    char err[256];
    js *b = js_parse(body, blen, err, sizeof err);
    if (!b) { respond_err(c, 400, err); return; }
    chat_req r;
    if (!chat_from_anthropic(b, &r, err, sizeof err)) { respond_err(c, 400, err); js_free(b); return; }
    sbuf prompt = {0};
    chat_render(&r, &prompt);
    if (getenv("QW_LOG_PROMPT")) { // 除錯：每個請求的完整 prompt 存成檔案
        static int seq = 0;
        char pth[256];
        snprintf(pth, sizeof pth, "%s/req%03d.txt", getenv("QW_LOG_PROMPT"), seq++);
        FILE *f = fopen(pth, "wb");
        if (f) { fwrite(prompt.p, 1, prompt.n, f); fclose(f); }
    }
    const int np = qw_tok_encode(TK, prompt.p, prompt.n, NULL, 0);
    char o[64];
    const int n = snprintf(o, sizeof o, "{\"input_tokens\":%d}", np);
    respond(c, 200, "application/json", o, (size_t)n);
    sb_free(&prompt);
    chat_free(&r);
    js_free(b);
}

int main(int argc, char **argv) {
    int port = 8080, ctx = 131072;
    setenv("QW_SLOTS", "3", 0); // 預設 3 組 KV slot（主對話、權限判斷等不同前綴的請求各自快取）
    const char *host = "127.0.0.1", *cache_dir = "cache"; // 持久化前綴快取（SSD），"" = 不用
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--version")) { printf("qwental %s\n", QW_VERSION); return 0; }
    bool i8 = false, i8qk = false, i8hy = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--int8")) { i8 = true; continue; }
        if (!strcmp(argv[i], "--int8-qk")) { i8qk = true; continue; }
        if (!strcmp(argv[i], "--int8-hyper")) { i8hy = true; continue; }
        if (!strcmp(argv[i], "--fast-head")) { setenv("QW_LM_TC", "2", 1); continue; }
        if (!strcmp(argv[i], "--exact-spec")) { setenv("QW_EXACT_SPEC", "1", 1); continue; }
        if (i + 1 >= argc) break;
        if (!strcmp(argv[i], "--gguf")) { setenv("QW_GGUF", argv[i + 1], 1); setenv("QW_GGUF_PARTS", "all", 0); }
        else if (!strcmp(argv[i], "--port")) port = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--ctx")) ctx = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--host")) host = argv[i + 1];
        else if (!strcmp(argv[i], "--model")) MODEL_DIR = argv[i + 1];
        else if (!strcmp(argv[i], "--slots")) setenv("QW_SLOTS", argv[i + 1], 1);
        else if (!strcmp(argv[i], "--cache-dir")) cache_dir = argv[i + 1];
        i++;
    }
    if (i8qk || i8hy) { // 類別清單：QW_INT8=1 的預設集合再加上 qk / hyper
        char l[64];
        snprintf(l, sizeof l, "lin,attn,shared,moe%s%s", i8qk ? ",qk" : "", i8hy ? ",hyper" : "");
        setenv("QW_INT8", l, 1);
    } else if (i8) setenv("QW_INT8", "1", 1);
    signal(SIGPIPE, SIG_IGN);
    char err[1024];
    TK = qw_tok_open_any(MODEL_DIR, getenv("QW_GGUF"), err, sizeof err); // 沒有 MLX 目錄時用 GGUF 的 tokenizer
    if (!TK) { fprintf(stderr, "❌ %s\n", err); return 1; }
    IM_END = qw_tok_special(TK, "<|im_end|>");
    EOT = qw_tok_special(TK, "<|endoftext|>");
    const double t0 = qw_now();
    CTX = ctx;
    E = qw_engine_open(MODEL_DIR, ctx, err, sizeof err);
    if (!E) { fprintf(stderr, "❌ %s\n", err); return 1; }
    qw_engine_set_cache_dir(E, cache_dir);
    // 先跑一次短生成：把 prefill / 驗證用的 kernel 都編好
    {
        const int32_t w[4] = {760, 6511, 314, 9338};
        qw_gen g = {.temperature = 0, .max_tokens = 4, .stop = NULL, .nstop = 0, .spec = !getenv("QW_NO_SPEC")};
        qw_engine_generate_ex(E, w, 4, &g, NULL, NULL, NULL);
    }
    // INT8 prefill（--int8 等）的 kernel 依段長（< 128 或 ≥ 128 個 token）與 context（> 2048 才走稀疏 attention 與 INT8 Q·Kᵀ）
    // 分別在第一次用到時編譯：再跑一段 WN 個 token 的 prefill 與接在後面的 WC 個 token（同一個 slot、從快照接續），
    // 第一個長請求就不必等編譯。dummy token 不會與真正的 prompt（以 <|im_start|> 開頭）共用前綴，輸出不受影響
    if (getenv("QW_INT8") && strcmp(getenv("QW_INT8"), "0") && !getenv("QW_NO_I8_WARM")) {
        enum { WN = 2304, WC = 64 };
        int32_t *w = malloc(sizeof(int32_t) * (WN + WC));
        for (int i = 0; i < WN + WC; i++) w[i] = 1000 + (int32_t)((i * 7919u) % 100000u);
        const double tw = qw_now();
        qw_gen g = {.temperature = 0, .max_tokens = 1, .stop = NULL, .nstop = 0, .spec = false};
        qw_engine_generate_ex(E, w, WN, &g, NULL, NULL, NULL);
        qw_engine_generate_ex(E, w, WN + WC, &g, NULL, NULL, NULL);
        free(w);
        fprintf(stderr, "INT8 prefill kernel 預熱 %.1f 秒\n", qw_now() - tw);
    }
    fprintf(stderr, "Qwental %s：模型載入與預熱 %.1f 秒，context %d%s%s%s，監聽 http://%s:%d\n", QW_VERSION, qw_now() - t0, ctx,
            getenv("QW_GGUF") ? "，GGUF " : "", getenv("QW_GGUF") ? getenv("QW_GGUF") : "",
            getenv("QW_INT8") && strcmp(getenv("QW_INT8"), "0") ? "，INT8 prefill" : "", host, port);
    if (i8qk || i8hy) fprintf(stderr, "INT8 prefill 另外包含：%s%s\n", i8qk ? "attention Q·Kᵀ（--int8-qk）" : "", i8hy ? (i8qk ? "、hyper mix_down（--int8-hyper）" : "hyper mix_down（--int8-hyper）") : "");
    if (getenv("QW_LM_TC") && atoi(getenv("QW_LM_TC"))) fprintf(stderr, "lm_head：矩陣單元版（--fast-head）\n");
    if (getenv("QW_EXACT_SPEC") && strcmp(getenv("QW_EXACT_SPEC"), "0"))
        fprintf(stderr, "推測解碼 = greedy（--exact-spec%s）\n", getenv("QW_GGUF") ? "；GGUF 本來就是，不影響" : "");
    const int ls = socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    inet_pton(AF_INET, host, &a.sin_addr);
    if (bind(ls, (struct sockaddr *)&a, sizeof a) < 0 || listen(ls, 16) < 0) { perror("bind/listen"); return 1; }
    for (;;) {
        const int fd = accept(ls, NULL, NULL);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        conn c = {fd, false};
        char method[16], path[1024];
        size_t blen = 0;
        char *body = read_req(fd, method, path, &blen);
        if (body) {
            char *q = strchr(path, '?');
            if (q) *q = 0;
            if (!strcmp(method, "POST") && !strcmp(path, "/v1/messages")) handle_chat(&c, body, blen, false);
            else if (!strcmp(method, "POST") && !strcmp(path, "/v1/chat/completions")) handle_chat(&c, body, blen, true);
            else if (!strcmp(method, "POST") && !strcmp(path, "/v1/messages/count_tokens")) handle_count(&c, body, blen);
            else if (!strcmp(method, "GET") && (!strcmp(path, "/v1/models") || !strcmp(path, "/models"))) {
                const char *m = "{\"object\":\"list\",\"data\":[{\"id\":\"qwen3.8-flash-next\",\"object\":\"model\",\"type\":\"model\",\"display_name\":\"Qwen3.8 Flash Next (Qwental)\"}]}";
                respond(&c, 200, "application/json", m, strlen(m));
            } else if (!strcmp(path, "/health")) respond(&c, 200, "application/json", "{\"ok\":true}", 11);
            else respond_err(&c, 404, "not found");
            free(body);
        }
        close(fd);
    }
}
