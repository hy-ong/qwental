// Qwental HTTP server：Anthropic Messages API（/v1/messages，含串流）與 OpenAI Chat Completions（/v1/chat/completions）
// 預設最多 16 個請求同時進行（--parallel N 改上限，最多 64；--parallel 1 = 一次一個）。MLX 權重一起 decode 時捨入與單獨跑略有不同
// （--exact-spec 時逐位元相同）。
// 照當下的請求數與實測速度自動選推測解碼（輪流）或一起 decode（超過 8 個時分組），greedy 輸出與單獨跑逐位元相同。
// 用法：build/qwental-serve [--gguf models/iq3s-gguf] [--int8] [--int8-qk] [--int8-hyper] [--fast-head] [--exact-spec] [--port 8080] [--ctx 131072] [--host 127.0.0.1] [--slots 3] [--cache-dir cache] [--parallel [N]] [--version]
// --gguf：整個模型改用該目錄的 GGUF（Strata 的 IQ3_S 等；等同 QW_GGUF=目錄 QW_GGUF_PARTS=all）。
//   這時 MLX 目錄（--model）是選用的：沒有時 tokenizer 讀 GGUF 的 metadata，dflash2/ 改找 GGUF 目錄的同層
//   （QW_GGUF_PARTS 只選一部分時其餘權重仍來自 MLX 目錄）
// --fast-head：GGUF 的 Q6_K lm_head 走矩陣單元（權重捨入成 FP16，同 llama.cpp 的 mul_mm；輸出與預設略有差異；等同 QW_LM_TC=2）
// --exact-spec：MLX profile 的推測解碼逐位元等於 greedy（kernel 改用 safe 數學模式；MLX 輸出與預設略有差異；GGUF 不受影響；等同 QW_EXACT_SPEC=1）
// --int8：INT8 prefill（routed expert 走 INT8 × INT8，GGUF 另外加上 dense 投影；decode 不變；等同 QW_INT8=1）
// --int8-qk：再加上 attention 的 Q·Kᵀ（indexer 打分與稀疏核心）走 INT8；--int8-hyper：再加上 GGUF hyper-connection 的 mix_down 走 INT8。
//   兩者都隱含 --int8，各自會讓輸出再多一點差異（等同 QW_INT8=lin,attn,shared,moe[,qk][,hyper]）
#include <arpa/inet.h>
#include <dispatch/dispatch.h>
#include <pthread.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>
#include "chat.h"
#include "engine.h"
#include "gpu.h"
#include "json.h"
#include "tok.h"

static const char *MODEL_DIR = "models/Qwen3.8-Flash-Next-mixed-4-8bit";
static pthread_mutex_t tok_mu = PTHREAD_MUTEX_INITIALIZER; // --parallel：連線執行緒 tokenize、engine 執行緒 decode 共用 tokenizer
static void tok_lock(void) { pthread_mutex_lock(&tok_mu); }
static void tok_unlock(void) { pthread_mutex_unlock(&tok_mu); }
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
    tok_lock();
    int n = g->u8n + qw_tok_decode1(TK, t, buf + g->u8n);
    tok_unlock();
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

// 一個生成請求：解析（chat_begin）→ 生成（單一請求：qw_engine_generate_ex；--parallel：engine 執行緒）→ 收尾（chat_end）
typedef struct job {
    conn *c;
    bool openai;
    js *b;
    chat_req r;
    sbuf prompt;
    int32_t *ids;
    int np;
    gen_ctx g;
    qw_gen gp;
    int32_t stops[2];
    int persist[2];
    int fin;
    qw_gen_stats st;
    // --parallel：engine 執行緒的狀態
    int sid, got, base;
    bool preempt, cbstop, inb; // inb：在一起 decode 裡（last 還沒餵）
    bool pfing; // 分段 prefill 中（別人在跑時，長 prompt 一次只算一段）
    int pfd;    // 分段 prefill 算到哪
    volatile bool yflag;
    double srate; // 這個對話推測解碼的實測速度（tok/s）
    int32_t last;
    int32_t *gen;
    unsigned long long rng;
    double t0, tg0;
    bool done;
    struct job *next;
} job;

static job *chat_begin(conn *c, const char *body, size_t blen, bool openai) {
    char err[256];
    js *b = js_parse(body, blen, err, sizeof err);
    if (!b) { respond_err(c, 400, err); return NULL; }
    job *j = calloc(1, sizeof *j);
    j->c = c;
    j->openai = openai;
    j->b = b;
    chat_req r;
    if (!(openai ? chat_from_openai(b, &r, err, sizeof err) : chat_from_anthropic(b, &r, err, sizeof err))) {
        respond_err(c, 400, err);
        js_free(b);
        free(j);
        return NULL;
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
    tok_lock();
    const int np = qw_tok_encode(TK, prompt.p, prompt.n, NULL, 0);
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)np);
    qw_tok_encode(TK, prompt.p, prompt.n, ids, np);
    tok_unlock();
    if (np + 64 > CTX) { // 超過 context：回 Anthropic 標準錯誤，Claude Code 會自動壓縮對話
        char m[160];
        snprintf(m, sizeof m, "prompt is too long: %d tokens > %d maximum", np, CTX - 64);
        respond_err(c, 400, m);
        free(ids);
        sb_free(&prompt);
        chat_free(&r);
        js_free(b);
        free(j);
        return NULL;
    }
    const char *model = js_str(b, "model");
    j->g = (gen_ctx){.c = c, .stream = js_bool(b, "stream", false), .openai = openai, .model = model ? model : "qwental"};
    gen_id(j->g.id, openai ? "chatcmpl-" : "msg_");
    chat_parser_init(&j->g.parser, r.thinking, r.tools);
    int max_tokens = (int)js_num(b, "max_tokens", openai ? js_num(b, "max_completion_tokens", 32768) : 4096);
    if (max_tokens > CTX - np - 16) max_tokens = CTX - np - 16;
    // stop_sequences（Anthropic）/ stop（OpenAI）：在輸出的文字裡比對
    const js *ss = js_get(b, openai ? "stop" : "stop_sequences");
    if (ss && ss->t == JS_STR) { j->g.stops[0] = ss->s; j->g.nstops = 1; }
    for (const js *x = ss && ss->t == JS_ARR ? ss->kid : NULL; x && j->g.nstops < 4; x = x->next)
        if (x->t == JS_STR && x->len) j->g.stops[j->g.nstops++] = x->s;
    j->stops[0] = IM_END;
    j->stops[1] = EOT;
    const char *ev = getenv("QW_TEMP");
    // 持久化點：工具定義結束前的 token 邊界、system 訊息結尾（第一個 <|im_end|> 之後）
    int *persist = j->persist, npersist = 0;
    {
        const char *te = strstr(prompt.p, "</tools>");
        if (te) {
            const size_t off = (size_t)(te - prompt.p);
            size_t acc = 0;
            char tb[300];
            for (int i = 0; i < np; i++) {
                tok_lock();
                const size_t l = (size_t)qw_tok_decode1(TK, ids[i], tb);
                tok_unlock();
                if (acc + l > off) { if (i >= 1024) persist[npersist++] = i; break; }
                acc += l;
            }
        }
        for (int i = 0; i < np; i++)
            if (ids[i] == IM_END) { if (i + 1 >= 1024 && i + 1 < np) persist[npersist++] = i + 1; break; }
    }
    j->gp = (qw_gen){.temperature = (float)js_num(b, "temperature", ev ? atof(ev) : 0.7),
                 .top_k = (int)js_num(b, "top_k", 20),
                 .top_p = (float)js_num(b, "top_p", 0.95),
                 .seed = (unsigned long long)js_num(b, "seed", (double)arc4random()),
                 .max_tokens = max_tokens,
                 .stop = j->stops,
                 .nstop = 2,
                 .spec = !getenv("QW_NO_SPEC"),
                 .persist = persist,
                 .npersist = npersist};
    if (j->g.stream) {
        const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        wr(c, h, strlen(h));
        if (!openai) {
            sbuf d = {0};
            sb_printf(&d, "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", j->g.id);
            sb_json_str(&d, j->g.model, strlen(j->g.model), false);
            sb_printf(&d, ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":%d,\"output_tokens\":1}}}", np);
            sse(&j->g, "message_start", &d);
            sb_free(&d);
        }
    }
    j->r = r;
    j->prompt = prompt;
    j->ids = ids;
    j->np = np;
    return j;
}

static void chat_end(job *j) {
    conn *c = j->c;
    const bool openai = j->openai;
    gen_ctx *gg = &j->g;
    const int np = j->np, fin = j->fin;
    const qw_gen_stats st = j->st;
    chat_req r = j->r;
    sbuf prompt = j->prompt;
    int32_t *ids = j->ids;
    js *b = j->b;
    chat_parser_end(&gg->parser, on_event, gg);
    fprintf(stderr, "[%s] prompt %d（重用 %d）%.0f ms；生成 %d 個 token %.0f ms（%.1f tok/s）；%d 輪、接受 %d 個猜測（n-gram %d 輪）；結束 %d\n",
            openai ? "openai" : "anthropic", np, st.cached, st.prefill_ms, st.gen, st.gen_ms, st.gen ? st.gen / st.gen_ms * 1e3 : 0.0,
            st.rounds, st.accepted, st.ngram_rounds, fin);
    const bool tool = gg->ncalls > 0;
    const char *sr = gg->hit ? (openai ? "stop" : "stop_sequence") : fin == QW_FIN_LENGTH ? (openai ? "length" : "max_tokens")
                   : tool ? (openai ? "tool_calls" : "tool_use") : (openai ? "stop" : "end_turn");
    if (gg->stream) {
        sbuf d = {0};
        if (openai) {
            sb_printf(&d, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", gg->id, (long)time(NULL));
            sb_json_str(&d, gg->model, strlen(gg->model), false);
            sb_printf(&d, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                      sr, np, st.gen, np + st.gen);
            sse(gg, NULL, &d);
            wr(c, "data: [DONE]\n\n", 14);
        } else {
            anth_close_block(gg);
            sb_printf(&d, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":%s%s%s},\"usage\":{\"output_tokens\":%d}}", sr,
                      gg->hit ? "\"" : "", gg->hit ? gg->hit : "null", gg->hit ? "\"" : "", st.gen);
            sse(gg, "message_delta", &d);
            d.n = 0;
            sb_puts(&d, "{\"type\":\"message_stop\"}");
            sse(gg, "message_stop", &d);
        }
        sb_free(&d);
    } else {
        sbuf o = {0};
        if (openai) {
            sb_printf(&o, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", gg->id, (long)time(NULL));
            sb_json_str(&o, gg->model, strlen(gg->model), false);
            sb_puts(&o, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
            sb_json_str(&o, gg->text.p ? gg->text.p : "", gg->text.n, false);
            if (gg->think.n) { sb_puts(&o, ",\"reasoning_content\":"); sb_json_str(&o, gg->think.p, gg->think.n, false); }
            if (tool) { sb_puts(&o, ",\"tool_calls\":["); sb_put(&o, gg->calls.p, gg->calls.n); sb_puts(&o, "]"); }
            sb_printf(&o, "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}", sr, np,
                      st.gen, np + st.gen);
        } else {
            sb_printf(&o, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", gg->id);
            sb_json_str(&o, gg->model, strlen(gg->model), false);
            sb_puts(&o, ",\"content\":[");
            int k = 0;
            if (gg->think.n) {
                sb_puts(&o, "{\"type\":\"thinking\",\"thinking\":");
                sb_json_str(&o, gg->think.p, gg->think.n, false);
                sb_puts(&o, ",\"signature\":\"qwental\"}");
                k++;
            }
            if (gg->text.n) {
                if (k++) sb_puts(&o, ",");
                sb_puts(&o, "{\"type\":\"text\",\"text\":");
                sb_json_str(&o, gg->text.p, gg->text.n, false);
                sb_puts(&o, "}");
            }
            if (tool) { if (k++) sb_puts(&o, ","); sb_put(&o, gg->calls.p, gg->calls.n); }
            sb_printf(&o, "],\"stop_reason\":\"%s\",\"stop_sequence\":", sr);
            if (gg->hit) sb_json_str(&o, gg->hit, strlen(gg->hit), false);
            else sb_puts(&o, "null");
            sb_printf(&o, ",\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}}", np, st.gen);
        }
        respond(c, 200, "application/json", o.p, o.n);
        sb_free(&o);
    }
    sb_free(&gg->think);
    sb_free(&gg->text);
    sb_free(&gg->calls);
    sb_free(&gg->tail);
    chat_parser_free(&gg->parser);
    free(ids);
    sb_free(&prompt);
    chat_free(&r);
    js_free(b);
    free(j);
}

static void handle_chat(conn *c, const char *body, size_t blen, bool openai) {
    job *j = chat_begin(c, body, blen, openai);
    if (!j) return;
    j->fin = qw_engine_generate_ex(E, j->ids, j->np, &j->gp, on_token, &j->g, &j->st);
    chat_end(j);
}

// ---------------------------------------------------------------- --parallel N：多個請求一起 decode
// 連線執行緒：讀請求、解析（chat_begin）→ 排進佇列 → 等 engine 執行緒做完 → 收尾（chat_end）
// engine 執行緒：新請求逐一 prefill（期間其他對話暫停），之後每步一起 decode（qw_engine_par_step），token 經 on_token 串流給各自的連線
static int PAR = -1; // 同時最多幾個請求；-1 = 自動（16；只有一部分是 GGUF 時 1）
static pthread_mutex_t q_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_cv = PTHREAD_COND_INITIALIZER, done_cv = PTHREAD_COND_INITIALIZER;
static job *q_head, *q_tail;

static bool q_pending(void) {
    pthread_mutex_lock(&q_mu);
    const bool p = q_head != NULL;
    pthread_mutex_unlock(&q_mu);
    return p;
}
// 生成的 token 接在 prompt 後面（j->ids[np + k]，j->gen = j->ids + np），切換單獨 / 一起時整段就是新的 prompt。
// j->base：slot 的 KV 內容目前記到哪（上一次 prefill 的長度），結束時從這裡之後記快照
static void par_done(job *j, int fin) {
    j->fin = fin;
    j->st.gen = j->got;
    j->st.gen_ms = (qw_now() - j->tg0) * 1e3;
    if (j->sid >= 0) qw_engine_par_finish(E, j->sid, j->ids + j->base, j->np + j->got - j->base);
    pthread_mutex_lock(&q_mu);
    j->done = true;
    pthread_cond_broadcast(&done_cv);
    pthread_mutex_unlock(&q_mu);
}
// 送出一個 token（同 qw_engine_generate_ex 的 EMIT）：回傳 false = 這個請求結束了
static bool par_emit(job *j, int32_t t) {
    for (int q = 0; q < j->gp.nstop; q++)
        if (j->gp.stop[q] == t) { par_done(j, QW_FIN_STOP); return false; }
    j->gen[j->got++] = t;
    if (!on_token(&j->g, t)) { par_done(j, QW_FIN_CALLBACK); return false; }
    if (j->got >= j->gp.max_tokens) { par_done(j, QW_FIN_LENGTH); return false; }
    j->last = t;
    return true;
}
// ---- 排程：每次都從實測速度選比較快的做法（不必指定幾路）
//   推測解碼輪流跑（一次一個對話、時間片 SLICE_MS）：總速度 ≈ 各對話推測解碼速度的平均（含切換成本，實測）
//   一起 decode：總速度 = B / 一步的時間（實測；還沒量過的 B 用 engine 的各組時間估）
// 兩種做法都和單獨 greedy 逐位元相同，只差快慢；只有一個請求時一定是推測解碼
enum { SLICE_MS = 300 };
static volatile int par_others; // 單獨跑的時候，另外還有幾個進行中的對話在等（> 0 才切時間片）
static volatile bool par_pfwait; // 有對話在分段 prefill：時間片縮短（prefill 多分一點時間）
static double seg_t0;
static double spec_rate;                 // 推測解碼的實測速度（tok/s，EMA，含切換）
static double bat_rate[QW_PAR_MAX + 1];  // 一起 decode B 個的實測總速度（tok/s，EMA）
static bool solo_cb(void *ctx, int32_t t) { // 要讓出時只設 yield：這一輪的 token 送完才停（模型狀態停在最後一個 token 之前）
    job *j = ctx;
    j->gen[j->got++] = t;
    if (!on_token(&j->g, t)) { j->cbstop = true; return false; }
    if (!j->yflag && (q_pending() || (par_others && (qw_now() - seg_t0) * 1e3 > (par_pfwait ? SLICE_MS * 2 / 3 : SLICE_MS)))) j->yflag = true;
    return true;
}
static double ema(double o, double x, double w) { return o > 0 ? (1 - w) * o + w * x : x; }
// 推測解碼跑一段（到結束、有新請求、或時間片用完）。回傳 true = 還沒結束（狀態留在 slot，下次從 prompt + 已生成接續）
static bool run_solo(job *j) {
    qw_gen g2 = j->gp;
    g2.max_tokens = j->gp.max_tokens - j->got;
    g2.seed = j->rng;
    j->preempt = j->cbstop = j->yflag = false;
    g2.yield = &j->yflag;
    qw_gen_stats s2;
    const int got0 = j->got;
    const bool first = !j->st.prompt;
    seg_t0 = qw_now();
    const int fin = qw_engine_par_solo(E, j->sid, j->ids, j->np + j->got, &g2, solo_cb, j, &s2);
    double ms = (qw_now() - seg_t0) * 1e3;
    if (first) { j->st.prompt = j->np; j->st.cached = s2.cached; j->st.prefill_ms = s2.prefill_ms; ms -= s2.prefill_ms; } // 第一段扣掉 prompt
    const int ng = j->got - got0;
    if (ng >= 8 && ms > 1) { j->srate = ema(j->srate, ng * 1e3 / ms, 0.5); spec_rate = ema(spec_rate, ng * 1e3 / ms, 0.3); }
    j->st.rounds += s2.rounds;
    j->st.accepted += s2.accepted;
    j->st.ngram_rounds += s2.ngram_rounds;
    j->base = j->np + j->got; // generate_ex 結束時記了快照（不一定到最後一個 token，par_prefill / par_solo 會找能用的）
    j->rng = j->rng * 6364136223846793005ull + 1442695040888963407ull;
    if (fin != QW_FIN_YIELD) { par_done(j, fin); return false; }
    if (j->got >= j->gp.max_tokens) { par_done(j, QW_FIN_LENGTH); return false; }
    return true;
}
// 加入一起 decode：從快照接續 prompt + 已生成的部分，取第一個 token（之後每步餵 j->last）
static bool join_batch(job *j) {
    int cached = 0;
    qw_engine_par_persist(E, j->gp.persist, j->gp.npersist);
    const float *lg = qw_engine_par_prefill(E, j->sid, j->ids, j->np + j->got, &cached);
    j->base = j->np + j->got;
    if (!lg) { par_done(j, QW_FIN_ERROR); return false; }
    if (!j->st.prompt) { j->st.prompt = j->np; j->st.cached = cached; j->st.prefill_ms = (qw_now() - j->t0) * 1e3; j->tg0 = qw_now(); }
    j->inb = true;
    return par_emit(j, qw_engine_sample(lg, &j->gp, &j->rng));
}
// 分段 prefill 一段：段長照實測速度調成約 QW_PAR_CHUNK_MS（預設 500 ms，其他對話最多停這麼久），64 的倍數、256…8192 個 token；
// 整段算完就取第一個 token、進入一起 decode。回傳 false = 這個請求結束了
static bool pf_chunk(job *j) {
    static double target = 0, rate = 0; // rate：最近的 prefill 速度（token / ms，context 越長越慢）
    if (!target) target = getenv("QW_PAR_CHUNK_MS") ? atof(getenv("QW_PAR_CHUNK_MS")) : 500;
    int ch = rate > 0 ? (int)(rate * target) / 64 * 64 : 512;
    ch = ch < 256 ? 256 : ch > 8192 ? 8192 : ch;
    int cached = 0;
    const bool first = j->pfd == 0;
    const int d0 = j->pfd;
    const double t = qw_now();
    qw_engine_par_persist(E, j->gp.persist, j->gp.npersist);
    const float *lg = qw_engine_par_prefill_to(E, j->sid, j->ids, j->np, ch, &j->pfd, &cached);
    if (lg && j->pfd - (first ? cached : d0) >= 256) rate = (j->pfd - (first ? cached : d0)) / ((qw_now() - t) * 1e3);
    if (!lg) { par_done(j, QW_FIN_ERROR); return false; }
    if (first) j->st.cached = cached;
    if (j->pfd < j->np) return true;
    j->pfing = false;
    j->base = j->np;
    j->st.prompt = j->np;
    j->st.prefill_ms = (qw_now() - j->t0) * 1e3;
    j->tg0 = qw_now();
    j->inb = true;
    return par_emit(j, qw_engine_sample(lg, &j->gp, &j->rng));
}
// 離開一起 decode：記快照（最後送出、還沒餵的 token 留給下一段當 prompt 結尾）
static void leave_batch(job *j) {
    qw_engine_par_finish(E, j->sid, j->ids + j->base, j->np + j->got - j->base);
    j->inb = false;
}
// 這 n 個對話：輪流推測解碼比一起 decode 快嗎（cur = 目前的做法，差不多時不換，免得來回切）
// ---- 記憶體保護：slot（KV、狀態、快照）合計不超過 SLOT_BUDGET；系統可用記憶體低於 10% 時再多放一組閒置的
static size_t SLOT_BUDGET;
static size_t dir_bytes(const char *dir, const char *ext) {
    size_t t = 0;
    DIR *d = dir ? opendir(dir) : NULL;
    if (!d) return 0;
    for (struct dirent *de; (de = readdir(d));) {
        const size_t n = strlen(de->d_name), m = strlen(ext);
        if (n < m || strcmp(de->d_name + n - m, ext)) continue;
        char p[1024];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
        if (!stat(p, &st)) t += (size_t)st.st_size;
    }
    closedir(d);
    return t;
}
static void mem_guard(job **act, int na) {
    unsigned long long busy = 0;
    for (int i = 0; i < na; i++) busy |= 1ull << act[i]->sid;
    const int n = qw_engine_par_trim(E, busy, SLOT_BUDGET, false);
    if (n) fprintf(stderr, "[mem] slot 超過 %.1f GB：釋放 %d 組閒置的\n", SLOT_BUDGET / 1e9, n);
    static double last;
    int lvl = 100;
    size_t len = sizeof lvl;
    if (qw_now() - last > 1 && !sysctlbyname("kern.memorystatus_level", &lvl, &len, NULL, 0) && lvl < 10) {
        last = qw_now();
        if (qw_engine_par_trim(E, busy, 0, true)) fprintf(stderr, "[mem] 系統可用記憶體 %d%%：釋放一組閒置的 slot\n", lvl);
    }
}
static bool want_slice(job **act, int n, bool cur) {
    if (n <= 1) return true;
    double rs = 0;
    for (int i = 0; i < n; i++) rs += act[i]->srate > 0 ? act[i]->srate : spec_rate;
    rs /= n;
    double rb = bat_rate[n];
    if (rb <= 0) { const double c = qw_engine_par_cost(E, n); rb = c > 0 ? n * 1e3 / c : 0; }
    if (rs <= 0 || rb <= 0) return false; // 沒數字：一起（2 個以上通常比較快）
    return rs > rb * (cur ? 0.95 : 1.05);
}
static void *par_loop(void *arg) {
    (void)arg;
    job *act[QW_PAR_MAX];
    int na = 0, rr = 0;
    bool slice = false;
    int sids[QW_PAR_MAX];
    int32_t toks[QW_PAR_MAX];
    static int prof = -1;
    if (prof < 0) prof = getenv("QW_PAR_PROF") != NULL;
    for (;;) {
        pthread_mutex_lock(&q_mu);
        while (!na && !q_head) pthread_cond_wait(&q_cv, &q_mu);
        job *adm[QW_PAR_MAX];
        int nadm = 0;
        while (q_head && na + nadm < PAR) {
            adm[nadm++] = q_head;
            q_head = q_head->next;
            if (!q_head) q_tail = NULL;
        }
        pthread_mutex_unlock(&q_mu);
        for (int k = 0; k < nadm; k++) { // 新請求：選 slot
            job *j = adm[k];
            unsigned long long busy = 0;
            for (int i = 0; i < na; i++) busy |= 1ull << act[i]->sid;
            j->t0 = j->tg0 = qw_now();
            j->rng = j->gp.seed ? j->gp.seed : 0x1234567ull;
            j->sid = qw_engine_par_pick(E, j->ids, j->np, busy);
            if (j->sid < 0) { par_done(j, QW_FIN_ERROR); continue; }
            if (j->gp.max_tokens <= 0) { par_done(j, QW_FIN_LENGTH); continue; }
            act[na++] = j;
        }
        if (!na) continue;
        mem_guard(act, na);
        // 新請求：有別人在跑時分段 prefill（每輪一段，其他對話照常前進）
        for (int i = 0; i < na; i++) {
            job *j = act[i];
            if (na > 1 && !j->inb && !j->got && !j->st.prompt && !j->pfing) j->pfing = true;
        }
        { // 兩段之間至少讓其他對話生成 SLICE_MS × 2/3（沒有別人可以生成時直接下一段）
            static double pf_last;
            int nready = 0;
            for (int i = 0; i < na; i++) nready += !act[i]->pfing;
            int keep = 0, done1 = !(nready == 0 || (qw_now() - pf_last) * 1e3 >= SLICE_MS * 2 / 3);
            for (int i = 0; i < na; i++) {
                job *j = act[i];
                if (j->pfing && !done1) { done1 = 1; const bool ok = pf_chunk(j); pf_last = qw_now(); if (!ok) continue; }
                act[keep++] = j;
            }
            na = keep;
        }
        job *rd[QW_PAR_MAX]; // 可以生成的（不在分段 prefill 中）
        int nr = 0;
        for (int i = 0; i < na; i++) if (!act[i]->pfing) rd[nr++] = act[i];
        if (!nr) continue;
        const bool s2 = want_slice(rd, nr, slice);
        if (s2 != slice && nr > 1) fprintf(stderr, "[par] %d 個對話：改成%s\n", nr, s2 ? "輪流推測解碼" : "一起 decode");
        slice = s2;
        if (slice) { // 推測解碼：輪流，一次一個（有人在分段 prefill 時也切時間片）
            for (int i = 0; i < nr; i++) if (rd[i]->inb) leave_batch(rd[i]);
            job *j = rd[rr % nr];
            rr = rr % nr + 1;
            par_others = na - 1;
            par_pfwait = na > nr;
            const bool alive = run_solo(j);
            par_others = 0;
            par_pfwait = false;
            if (!alive) {
                int keep = 0;
                for (int i = 0; i < na; i++) if (act[i] != j) act[keep++] = act[i];
                na = keep;
            }
            continue;
        }
        job *pf[QW_PAR_MAX], *bj[QW_PAR_MAX]; // 一起 decode：還沒加入的先接上（結束的請求之後不能再碰：連線執行緒會釋放）
        int npf = 0, nb = 0;
        for (int i = 0; i < na; i++) if (act[i]->pfing) pf[npf++] = act[i];
        for (int i = 0; i < nr; i++) {
            job *j = rd[i];
            if (j->inb || join_batch(j)) bj[nb++] = j;
        }
        na = 0;
        for (int i = 0; i < npf; i++) act[na++] = pf[i];
        if (!nb) continue;
        for (int i = 0; i < nb; i++) { sids[i] = bj[i]->sid; toks[i] = bj[i]->last; }
        const double ts0 = qw_now();
        double gms = 0;
        const float *lg = qw_engine_par_step(E, sids, toks, nb, &gms);
        const double ts1 = qw_now();
        int32_t nxa[QW_PAR_MAX], *nx = nxa;
        job **aj = bj;
        if (lg) // 各對話的取樣互不相關（rng 各自一份）：平行做
            dispatch_apply((size_t)nb, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
                nx[i] = qw_engine_sample(lg + i * QW_VOCAB, &aj[i]->gp, &aj[i]->rng);
            });
        const double dt = (qw_now() - ts0) * 1e3;
        if (lg && dt > 0) bat_rate[nb] = ema(bat_rate[nb], nb * 1e3 / dt, 0.1);
        if (prof) { static int c = 0; if (++c % 50 == 0) fprintf(stderr, "[par] B=%d：par_step %.2f ms（GPU %.2f）、取樣 %.2f ms；一起 %.0f tok/s、推測 %.0f tok/s\n", nb, (ts1 - ts0) * 1e3, gms, dt - (ts1 - ts0) * 1e3, bat_rate[nb], spec_rate); }
        for (int i = 0; i < nb; i++) {
            job *j = bj[i];
            if (!lg) { par_done(j, QW_FIN_ERROR); continue; }
            if (j->np + j->got + 2 >= CTX) { par_done(j, QW_FIN_CTX); continue; }
            if (par_emit(j, nx[i])) act[na++] = j;
        }
    }
    return NULL;
}
static void par_chat(conn *c, const char *body, size_t blen, bool openai) {
    job *j = chat_begin(c, body, blen, openai);
    if (!j) return;
    j->sid = -1;
    j->ids = realloc(j->ids, sizeof(int32_t) * (size_t)(j->np + j->gp.max_tokens + 2)); // 生成的 token 接在 prompt 後面
    j->gen = j->ids + j->np;
    pthread_mutex_lock(&q_mu);
    if (q_tail) q_tail->next = j;
    else q_head = j;
    q_tail = j;
    pthread_cond_signal(&q_cv);
    while (!j->done) pthread_cond_wait(&done_cv, &q_mu);
    pthread_mutex_unlock(&q_mu);
    chat_end(j);
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

// --parallel：一個連線（執行緒）
static void *conn_main(void *arg) {
    const int fd = *(int *)arg;
    free(arg);
    conn c = {fd, false};
    char method[16], path[1024];
    size_t blen = 0;
    char *body = read_req(fd, method, path, &blen);
    if (body) {
        char *q = strchr(path, '?');
        if (q) *q = 0;
        if (!strcmp(method, "POST") && !strcmp(path, "/v1/messages")) par_chat(&c, body, blen, false);
        else if (!strcmp(method, "POST") && !strcmp(path, "/v1/chat/completions")) par_chat(&c, body, blen, true);
        else if (!strcmp(method, "POST") && !strcmp(path, "/v1/messages/count_tokens")) handle_count(&c, body, blen);
        else if (!strcmp(method, "GET") && (!strcmp(path, "/v1/models") || !strcmp(path, "/models"))) {
            const char *m = "{\"object\":\"list\",\"data\":[{\"id\":\"qwen3.8-flash-next\",\"object\":\"model\",\"type\":\"model\",\"display_name\":\"Qwen3.8 Flash Next (Qwental)\"}]}";
            respond(&c, 200, "application/json", m, strlen(m));
        } else if (!strcmp(path, "/health")) respond(&c, 200, "application/json", "{\"ok\":true}", 11);
        else respond_err(&c, 404, "not found");
        free(body);
    }
    close(fd);
    return NULL;
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
        if (!strcmp(argv[i], "--parallel") && (i + 1 >= argc || argv[i + 1][0] < '0' || argv[i + 1][0] > '9')) { PAR = 16; continue; } // 不給數字：上限 16，實際幾路照當下請求數（要更多就給數字，最多 QW_PAR_MAX）
        if (i + 1 >= argc) break;
        if (!strcmp(argv[i], "--gguf")) { setenv("QW_GGUF", argv[i + 1], 1); setenv("QW_GGUF_PARTS", "all", 0); }
        else if (!strcmp(argv[i], "--port")) port = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--ctx")) ctx = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--host")) host = argv[i + 1];
        else if (!strcmp(argv[i], "--model")) MODEL_DIR = argv[i + 1];
        else if (!strcmp(argv[i], "--slots")) setenv("QW_SLOTS", argv[i + 1], 1);
        else if (!strcmp(argv[i], "--cache-dir")) cache_dir = argv[i + 1];
        else if (!strcmp(argv[i], "--parallel")) PAR = atoi(argv[i + 1]);
        i++;
    }
    if (i8qk || i8hy) { // 類別清單：QW_INT8=1 的預設集合再加上 qk / hyper
        char l[64];
        snprintf(l, sizeof l, "lin,attn,shared,moe%s%s", i8qk ? ",qk" : "", i8hy ? ",hyper" : "");
        setenv("QW_INT8", l, 1);
    } else if (i8) setenv("QW_INT8", "1", 1);
    // 預設：自動並發（上限 16）。GGUF 一起 decode 與單獨跑逐位元相同；MLX 只在 --exact-spec 時相同，
    // 預設的捨入略有不同（同 MLX 預設推測解碼的性質；單一請求不受影響）
    const bool mixed = getenv("QW_GGUF") && getenv("QW_GGUF_PARTS") && *getenv("QW_GGUF_PARTS") && strcmp(getenv("QW_GGUF_PARTS"), "all"); // 只有一部分是 GGUF：不支援並發
    if (PAR < 0) PAR = mixed ? 1 : 16;
    if (PAR > 1 && mixed) { fprintf(stderr, "❌ --parallel 不支援只有一部分是 GGUF 的權重（QW_GGUF_PARTS 要是 all）\n"); return 1; }
    if (PAR > QW_PAR_MAX) PAR = QW_PAR_MAX;
    if (PAR > 1 && atoi(getenv("QW_SLOTS")) < PAR) { char v[8]; snprintf(v, sizeof v, "%d", PAR); setenv("QW_SLOTS", v, 1); } // 每個並發的對話一個 slot
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
    { // slot 的記憶體預算：QW_SLOT_MEM_GB，或「實體記憶體 − 權重檔 − 16 GB」（至少 4 GB）
        unsigned long long mem = 0;
        size_t len = sizeof mem;
        sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
        const size_t w = getenv("QW_GGUF") ? dir_bytes(getenv("QW_GGUF"), ".gguf") : dir_bytes(MODEL_DIR, ".safetensors");
        const double gb = getenv("QW_SLOT_MEM_GB") ? atof(getenv("QW_SLOT_MEM_GB")) : ((double)mem - (double)w - 16e9) / 1e9;
        SLOT_BUDGET = (size_t)((gb < 4 ? 4 : gb) * 1e9);
    }
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
    if (PAR > 1) { // 並發：slot 至少要有並發數那麼多；各個並發數的 kernel 先編好
        const double tw = qw_now();
        bool ok = true;
        int32_t w[4];
        int sids[8];
        const int WB = PAR < 8 ? PAR : 8; // 每步超過 8 個時分組，kernel 只到 8
        for (int k = 0; k < WB && ok; k++) {
            for (int i = 0; i < 4; i++) w[i] = 760 + 13 * k + i;
            sids[k] = k;
            if (!qw_engine_par_prefill(E, k, w, 4, NULL)) { ok = false; break; }
        }
        int32_t t[8] = {0};
        for (int r = 0; r < 4 && ok; r++) // 第一次編 kernel，之後量各組大小一步的時間（排程與分組用）
            for (int k = 1; k <= WB; k++)
                if (!qw_engine_par_step(E, sids, t, k, NULL)) { ok = false; break; }
        if (!ok) { fprintf(stderr, "❌ 並發預熱失敗\n"); return 1; }
        fprintf(stderr, "並發最多 %d 個（自動選推測解碼或一起 decode）：kernel 預熱 %.1f 秒；一步 B=1 %.1f ms、B=%d %.1f ms；slot 記憶體上限 %.0f GB\n", PAR,
                qw_now() - tw, qw_engine_par_cost(E, 1), WB, qw_engine_par_cost(E, WB), SLOT_BUDGET / 1e9);
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
    if (PAR > 1) {
        pthread_t th;
        pthread_create(&th, NULL, par_loop, NULL);
        for (;;) { // 每個連線一個執行緒
            const int fd = accept(ls, NULL, NULL);
            if (fd < 0) continue;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            int *a2 = malloc(sizeof(int));
            *a2 = fd;
            pthread_t ct;
            pthread_attr_t at;
            pthread_attr_init(&at);
            pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
            pthread_attr_setstacksize(&at, 8 << 20);
            if (pthread_create(&ct, &at, conn_main, a2)) { close(fd); free(a2); }
            pthread_attr_destroy(&at);
        }
    }
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
