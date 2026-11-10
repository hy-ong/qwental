// Qwental HTTP server：Anthropic Messages API（/v1/messages，含串流）與 OpenAI Chat Completions（/v1/chat/completions）
// 預設最多 16 個請求同時進行（--parallel N 改上限，最多 64；--parallel 1 = 一次一個）。MLX 權重一起 decode 時捨入與單獨跑略有不同
// （--exact-spec 時逐位元相同）。
// 照當下的請求數與實測速度自動選推測解碼（輪流，≤ 3 個對話時才考慮；QW_PAR_SLICE_MAX）或一起 decode（超過 8 個時分組），
// greedy 輸出與單獨跑逐位元相同。
// 串流時每 10 秒沒有輸出就送 keepalive（ping）；用戶端斷線就取消請求（GPU 停下、slot 放出來）。思考太長時收尾、留空間給回答
// （同 Strata 的 answer_room；QW_NO_ANSWER_ROOM=1 關掉）。
// 用法：build/qwental-serve [--gguf models/iq3s-gguf] [--int8] [--int8-qk] [--int8-hyper] [--fast-head] [--exact-spec] [--port 8080] [--ctx 131072] [--host 127.0.0.1] [--slots 3] [--cache-dir cache] [--stats-file PATH] [--parallel [N]] [--version]
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
#include <errno.h>
#include <pthread.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include "chat.h"
#include "engine.h"
#include "gpu.h"
#include "json.h"
#include "tok.h"

static const char dashboard_html[] = { // GET /dashboard（與 /）：即時狀態頁（讀 /status）
#embed "dashboard.html"
};

static const char *MODEL_DIR = "models/Qwen3.8-Flash-Next-mixed-4-8bit";
static pthread_mutex_t tok_mu = PTHREAD_MUTEX_INITIALIZER; // --parallel：連線執行緒 tokenize、engine 執行緒 decode 共用 tokenizer
static void tok_lock(void) { pthread_mutex_lock(&tok_mu); }
static void tok_unlock(void) { pthread_mutex_unlock(&tok_mu); }
static qw_engine *E;
static qw_tok *TK;
static int32_t IM_END, EOT;
static int CTX;

// ---------------------------------------------------------------- HTTP
// 輸出一律先排進連線自己的緩衝（engine 執行緒只做記憶體複製，不碰 socket）；寫 socket 的只有連線執行緒（pump_step）。
// 用戶端不讀（Ctrl-Z、網路卡住）只會卡住它自己的連線：一次寫超過 SNDTIMEO_S 秒、或緩衝超過 OUT_MAX 就當成斷線（dead），
// engine 在下一個 token / prefill 段之間看到 dead 就結束這個請求（GPU 停下、slot 放出來）。
// 連線執行緒閒著時（排隊、prefill、tool call 還在緩衝）每 PING_S 秒送一個 keepalive，並檢查用戶端是否已經斷線
enum { PING_S = 10, SNDTIMEO_S = 30, RCVTIMEO_S = 30, REQ_DEADLINE_S = 300 };
#define BODY_MAX ((size_t)256 << 20) // request body 上限（百萬 token 的對話也還有餘裕）
#define HDR_MAX ((size_t)1 << 20)
#define OUT_MAX ((size_t)64 << 20)
typedef struct {
    int fd;
    atomic_bool dead;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    sbuf out;          // 還沒寫出去的（mu 保護）
    double last_q;     // 最近一次排進輸出的時間（mu 保護；keepalive 用）
    double last_probe; // 最近一次檢查斷線（只有寫 socket 的執行緒用）
    int sse;           // 串流的回應標頭已送出：1 Anthropic（ping 事件）、2 OpenAI（SSE 註解）
} conn;

static void conn_init(conn *c, int fd) {
    memset(c, 0, sizeof *c);
    c->fd = fd;
    atomic_init(&c->dead, false);
    pthread_mutex_init(&c->mu, NULL);
    pthread_cond_init(&c->cv, NULL);
    c->last_q = qw_now();
}
static void conn_free(conn *c) {
    sb_free(&c->out);
    pthread_cond_destroy(&c->cv);
    pthread_mutex_destroy(&c->mu);
}
static bool conn_dead(conn *c) { return atomic_load_explicit(&c->dead, memory_order_relaxed); }
static void conn_kill(conn *c) { atomic_store(&c->dead, true); }

static bool wr(conn *c, const char *s, size_t n) {
    if (conn_dead(c)) return false;
    pthread_mutex_lock(&c->mu);
    if (c->out.n + n > OUT_MAX) {
        conn_kill(c);
        fprintf(stderr, "[http] 用戶端一直沒讀取輸出（緩衝超過 %zu MiB）：當成斷線\n", OUT_MAX >> 20);
    } else if (n) {
        sb_put(&c->out, s, n);
        c->last_q = qw_now();
        pthread_cond_signal(&c->cv);
    }
    pthread_mutex_unlock(&c->mu);
    return !conn_dead(c);
}
// 直接寫 socket（只有連線執行緒）：SO_SNDTIMEO 內寫不出去就當成斷線
static void send_all(conn *c, const char *s, size_t n) {
    while (n && !conn_dead(c)) {
        const ssize_t k = write(c->fd, s, n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) { conn_kill(c); break; }
        s += k;
        n -= (size_t)k;
    }
}
// 用戶端關了連線（FIN 或 RST）？請求已經整個讀完，之後讀得到資料只會是下一個請求（不算斷線）
static void conn_probe(conn *c) {
    char b;
    const ssize_t k = recv(c->fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
    if (k == 0 || (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) conn_kill(c);
}
// 連線執行緒的一步：等輸出（最多 0.5 秒；*stop 變 true 時也會被叫醒）→ 寫出去；閒著時檢查斷線、太久沒輸出就送 keepalive。
// 緩衝裡永遠是完整的 SSE 事件（sse() 一次排一整個），ping 接在後面不會插進事件中間
static void pump_step(conn *c, atomic_bool *stop) {
    pthread_mutex_lock(&c->mu);
    if (!c->out.n && !(stop && atomic_load(stop))) { // 斷線了也照樣等（等 engine 結束這個請求，不要空轉）
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 500 * 1000000L;
        if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        pthread_cond_timedwait(&c->cv, &c->mu, &ts);
    }
    sbuf o = c->out;
    c->out = (sbuf){0};
    const double lq = c->last_q;
    pthread_mutex_unlock(&c->mu);
    if (o.n) send_all(c, o.p, o.n);
    sb_free(&o);
    if (conn_dead(c)) return;
    const double now = qw_now();
    if (now - c->last_probe >= 0.5) { c->last_probe = now; conn_probe(c); }
    if (c->sse && now - lq >= PING_S) {
        const char *pg = c->sse == 1 ? "event: ping\ndata: {\"type\": \"ping\"}\n\n" : ": ping\n\n";
        wr(c, pg, strlen(pg));
    }
}
// 把剩下的輸出全部寫出去（回應結束、關連線之前）
static void pump_drain(conn *c) {
    for (;;) {
        pthread_mutex_lock(&c->mu);
        sbuf o = c->out;
        c->out = (sbuf){0};
        pthread_mutex_unlock(&c->mu);
        const bool any = o.n > 0;
        if (any) send_all(c, o.p, o.n);
        sb_free(&o);
        if (!any) break;
    }
}

static const char *http_reason(int code) {
    switch (code) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 408: return "Request Timeout";
    case 411: return "Length Required";
    case 413: return "Content Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    case 529: return "Overloaded";
    default: return code < 500 ? "Bad Request" : "Internal Server Error";
    }
}
// Anthropic 的錯誤類型（Claude Code 對 529 / 500 會重試）
static const char *err_type(int code) {
    switch (code) {
    case 404: return "not_found_error";
    case 413: return "request_too_large";
    case 408: return "timeout_error";
    case 500: return "api_error";
    case 503:
    case 529: return "overloaded_error";
    default: return "invalid_request_error";
    }
}

static void respond(conn *c, int code, const char *ctype, const char *body, size_t n) {
    char h[256];
    const int hn = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", code,
                            http_reason(code), ctype, n);
    wr(c, h, (size_t)hn);
    wr(c, body, n);
}

static void respond_err(conn *c, int code, const char *msg) {
    sbuf b = {0};
    sb_printf(&b, "{\"type\":\"error\",\"error\":{\"type\":\"%s\",\"message\":", err_type(code));
    sb_json_str(&b, msg, strlen(msg), false);
    sb_puts(&b, "}}");
    respond(c, code, "application/json", b.p, b.n);
    sb_free(&b);
}

// 標頭行 l（到 \r 為止）的值裡有沒有 word（不分大小寫）
static bool hdr_has(const char *l, const char *word) {
    const char *e = strchr(l, '\r');
    const size_t n = e ? (size_t)(e - l) : strlen(l), m = strlen(word);
    for (size_t i = 0; i + m <= n; i++)
        if (!strncasecmp(l + i, word, m)) return true;
    return false;
}

// 讀一個請求：回傳 body（呼叫端 free），method / path 寫進參數。失敗回傳 NULL：*status = 要回的錯誤（emsg 為訊息），0 = 連線斷了
// 每次 read 最多等 RCVTIMEO_S 秒（SO_RCVTIMEO），整個請求最多 REQ_DEADLINE_S 秒；標頭最多 HDR_MAX、body 最多 BODY_MAX（超過回 413）
static char *read_req(conn *c, char *method, char *path, size_t *blen, int *status, char *emsg, size_t emlen) {
    const double t0 = qw_now();
    *status = 0;
    size_t cap = 65536, n = 0;
    char *buf = malloc(cap + 1), *he = NULL, *body = NULL;
    if (!buf) { *status = 500; snprintf(emsg, emlen, "out of memory"); return NULL; }
    while (!he) {
        if (n == cap) {
            if (cap >= HDR_MAX) { *status = 431; snprintf(emsg, emlen, "request headers too large"); goto fail; }
            char *nb = realloc(buf, cap * 2 + 1);
            if (!nb) { *status = 500; snprintf(emsg, emlen, "out of memory"); goto fail; }
            buf = nb;
            cap *= 2;
        }
        if (qw_now() - t0 > REQ_DEADLINE_S) { *status = 408; snprintf(emsg, emlen, "request timeout"); goto fail; }
        const ssize_t k = read(c->fd, buf + n, cap - n);
        if (k < 0 && errno == EINTR) continue;
        if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) && n) { *status = 408; snprintf(emsg, emlen, "request timeout"); goto fail; }
        if (k <= 0) goto fail; // 什麼都沒送就關了（或閒置太久）：不回應
        const size_t from = n > 3 ? n - 3 : 0;
        n += (size_t)k;
        buf[n] = 0;
        he = memmem(buf + from, n - from, "\r\n\r\n", 4);
    }
    const size_t hl = (size_t)(he + 4 - buf);
    he[2] = 0; // 標頭（每行都以 \r\n 結尾）變成一個字串
    if (sscanf(buf, "%15s %1023s", method, path) != 2) { *status = 400; snprintf(emsg, emlen, "malformed request line"); goto fail; }
    size_t clen = 0;
    bool chunked = false, expect = false;
    for (char *l = strstr(buf, "\r\n"); l && l[2]; l = strstr(l, "\r\n")) {
        l += 2;
        if (!strncasecmp(l, "Content-Length:", 15)) {
            char *e;
            const unsigned long long v = strtoull(l + 15, &e, 10);
            while (*e == ' ' || *e == '\t') e++;
            if (e == l + 15 || *e != '\r') { *status = 400; snprintf(emsg, emlen, "invalid Content-Length"); goto fail; }
            if (v > BODY_MAX) {
                *status = 413;
                snprintf(emsg, emlen, "request body too large: %llu bytes > %zu maximum", v, BODY_MAX);
                goto fail;
            }
            clen = (size_t)v;
        } else if (!strncasecmp(l, "Transfer-Encoding:", 18) && hdr_has(l + 18, "chunked")) chunked = true;
        else if (!strncasecmp(l, "Expect:", 7) && hdr_has(l + 7, "100-continue")) expect = true;
    }
    if (chunked) { *status = 411; snprintf(emsg, emlen, "chunked request bodies are not supported: send Content-Length"); goto fail; }
    body = malloc(clen + 1);
    if (!body) { *status = 500; snprintf(emsg, emlen, "out of memory"); goto fail; }
    size_t got = n - hl > clen ? clen : n - hl;
    memcpy(body, buf + hl, got);
    free(buf);
    buf = NULL;
    if (expect && got < clen) send_all(c, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    while (got < clen) {
        if (qw_now() - t0 > REQ_DEADLINE_S) { *status = 408; snprintf(emsg, emlen, "request timeout"); goto fail; }
        const ssize_t k = read(c->fd, body + got, clen - got);
        if (k < 0 && errno == EINTR) continue;
        if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { *status = 408; snprintf(emsg, emlen, "request timeout"); goto fail; }
        if (k <= 0) goto fail;
        got += (size_t)k;
    }
    body[clen] = 0;
    *blen = clen;
    return body;
fail:
    free(buf);
    free(body);
    return NULL;
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
    const char *stops[4]; // 停止字串（只比對回答的文字，不含思考與 tool call）
    int nstops;
    size_t maxstop;       // 最長的停止字串
    sbuf hold;            // 還沒送出的文字尾巴（最長停止字串 − 1 個位元組：跨事件的停止字串也不會先送出一半）
    const char *hit;      // 命中的停止字串
    // 思考上限（Strata 的 answer_room）：思考到 think_cap 個 token 就收尾，留空間給回答
    int think_cap, think_n;
    int wrap;             // 要插入的收尾：1 = 到上限（WRAP_IDS）、2 = 還在思考就停了（CLOSE_IDS）
    bool wrapped, closed; // 各只做一次
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

static void emit_event(gen_ctx *g, chat_ev_type t, const char *s, size_t n, const chat_call *call) {
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

// 停止字串：只在回答的文字（CEV_TEXT）裡比對。文字留最長停止字串 − 1 個位元組不送（不切開 UTF-8 字元），
// 命中時只送出它之前的部分；之後的事件（含 chat_parser_end 吐出來的）都不送
static void stop_flush(gen_ctx *g) {
    if (g->hold.n && !g->hit) emit_event(g, CEV_TEXT, g->hold.p, g->hold.n, NULL);
    g->hold.n = 0;
}
static void stop_text(gen_ctx *g, const char *s, size_t n) {
    sb_put(&g->hold, s, n);
    const char *best = NULL;
    int bi = -1;
    for (int i = 0; i < g->nstops; i++) {
        const char *h = memmem(g->hold.p, g->hold.n, g->stops[i], strlen(g->stops[i]));
        if (h && (!best || h < best)) { best = h; bi = i; }
    }
    if (best) {
        const size_t upto = (size_t)(best - g->hold.p);
        if (upto) emit_event(g, CEV_TEXT, g->hold.p, upto, NULL);
        g->hit = g->stops[bi];
        g->hold.n = 0;
        return;
    }
    if (g->hold.n < g->maxstop) return;
    size_t cut = g->hold.n - (g->maxstop - 1);
    while (cut > 0 && cut < g->hold.n && ((unsigned char)g->hold.p[cut] & 0xC0) == 0x80) cut--;
    if (!cut) return;
    emit_event(g, CEV_TEXT, g->hold.p, cut, NULL);
    memmove(g->hold.p, g->hold.p + cut, g->hold.n - cut);
    g->hold.n -= cut;
}
static void on_event(void *ctx, chat_ev_type t, const char *s, size_t n, const chat_call *call) {
    gen_ctx *g = ctx;
    if (g->hit) return;
    if (g->nstops) {
        if (t == CEV_TEXT) { stop_text(g, s, n); return; }
        stop_flush(g);
    }
    emit_event(g, t, s, n, call);
}

// 解析器在 <think> 的一般文字裡、可以插入收尾（chat.h：chat_parser.state 0 = 思考；3 = 思考裡的 tool call 不算：
// 收尾會插進 call 中間；pend 留著完整的 </think> 時已經算結束）
static bool parser_thinking(const chat_parser *p) {
    return p->state == 0 && !(p->pend.n && memmem(p->pend.p, p->pend.n, "</think>", 8));
}

// 每個 token：解碼成 bytes，湊滿完整的 UTF-8 字元才送進解析器。回傳 false = 停（用戶端斷線、命中停止字串）
static bool on_token(void *ctx, int32_t t) {
    gen_ctx *g = ctx;
    g->out_tokens++;
    if (g->think_cap && !g->wrapped) { // 思考 token 數（含思考裡的 tool call）；到上限且在可以插入的地方才收尾
        if (g->parser.state == 0 || g->parser.state == 3) g->think_n++;
        if (g->think_n >= g->think_cap && parser_thinking(&g->parser)) { g->wrap = 1; g->wrapped = true; }
    }
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
    if (cut) chat_parser_feed(&g->parser, buf, (size_t)cut, on_event, g);
    return !g->hit && !conn_dead(g->c);
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
    int32_t *ids; // prompt，生成的 token 接在後面（容量 np + max_tokens + 2）
    int np;
    gen_ctx g;
    qw_gen gp;
    int32_t stops[2];
    int persist[2];
    int fin;
    int ecode;        // fin = QW_FIN_ERROR 時：529（暫時沒辦法，重試）或 500（engine 失敗）
    const char *emsg; // 同上：錯誤訊息
    qw_gen_stats st;
    // --parallel：engine 執行緒的狀態
    int sid, got, base;
    bool preempt, cbstop, inb; // inb：在一起 decode 裡（last 還沒餵）
    bool pfing; // 分段 prefill 中（別人在跑時，長 prompt 一次只算一段）
    int pfd;    // 分段 prefill 算到哪
    bool pfy;   // 單獨跑時 prompt 只 prefill 一部分就讓出了（之後分段接著算）
    bool hasd;  // 一起推測解碼：engine 裡有從 last 猜好的一塊
    volatile bool yflag;
    double srate; // 這個對話推測解碼的實測速度（tok/s）
    int32_t last;
    int32_t *gen;
    unsigned long long rng;
    double t0, tg0;
    int id;    // 流水號（/status）
    double tc; // 收到請求的時間（/status 的 age_s）
    atomic_bool done; // engine 做完了（在 c->mu 裡設，之後 engine 不再碰這個 job）
    struct job *next;
} job;

// 思考收尾的 token（main 裡 tokenize 好）：到上限時同 Strata 的 REASONING_WRAP_UP，還在思考就停了時同 REASONING_CLOSE
static const char *WRAP_TEXT = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n";
static const char *CLOSE_TEXT = "\n</think>\n\n";
static int32_t WRAP_IDS[64], CLOSE_IDS[16];
static int NWRAP, NCLOSE;

// 除錯（QW_LOG_PROMPT=目錄）：每個請求的完整 prompt 存成檔案。存的是人讀的版本（chat_unmark 拿掉 QW_TOK_PLAIN 標記）；
// encode 一定用帶標記的原版
static void log_prompt(const sbuf *prompt) {
    const char *dir = getenv("QW_LOG_PROMPT");
    if (!dir) return;
    static atomic_int seq;
    char pth[1024];
    snprintf(pth, sizeof pth, "%s/req%03d.txt", dir, atomic_fetch_add(&seq, 1));
    sbuf u = {0};
    sb_put(&u, prompt->p ? prompt->p : "", prompt->n);
    chat_unmark(&u);
    FILE *f = fopen(pth, "wb");
    if (f) { fwrite(u.p, 1, u.n, f); fclose(f); }
    sb_free(&u);
}

static void job_free(job *j) {
    gen_ctx *gg = &j->g;
    sb_free(&gg->think);
    sb_free(&gg->text);
    sb_free(&gg->calls);
    sb_free(&gg->hold);
    chat_parser_free(&gg->parser);
    free(j->ids);
    sb_free(&j->prompt);
    chat_free(&j->r);
    js_free(j->b);
    free(j);
}

static job *chat_begin(conn *c, const char *body, size_t blen, bool openai) {
    char err[256];
    js *b = js_parse(body, blen, err, sizeof err);
    if (!b) { respond_err(c, 400, err); return NULL; }
    job *j = calloc(1, sizeof *j);
    if (!j) { respond_err(c, 500, "out of memory"); js_free(b); return NULL; }
    j->c = c;
    j->openai = openai;
    j->b = b;
    static atomic_int job_seq;
    j->id = atomic_fetch_add(&job_seq, 1) + 1;
    j->tc = qw_now();
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
    log_prompt(&prompt);
    tok_lock();
    const int np = qw_tok_encode(TK, prompt.p, prompt.n, NULL, 0);
    int32_t *ids = np > 0 ? malloc(sizeof(int32_t) * (size_t)np) : NULL;
    if (ids) qw_tok_encode(TK, prompt.p, prompt.n, ids, np);
    tok_unlock();
    if (!ids || np + 64 > CTX) { // 超過 context：回 Anthropic 標準錯誤，Claude Code 會自動壓縮對話
        char m[160];
        if (ids) snprintf(m, sizeof m, "prompt is too long: %d tokens > %d maximum", np, CTX - 64);
        else snprintf(m, sizeof m, np > 0 ? "out of memory" : "empty prompt");
        respond_err(c, ids ? 400 : np > 0 ? 500 : 400, m);
        free(ids);
        sb_free(&prompt);
        chat_free(&r);
        js_free(b);
        free(j);
        return NULL;
    }
    const char *model = js_str(b, "model"); // 回應裡照抄請求的模型名稱（claude-haiku-… 等都接受）
    j->g = (gen_ctx){.c = c, .stream = js_bool(b, "stream", false), .openai = openai, .model = model && *model ? model : "qwental"};
    gen_id(j->g.id, openai ? "chatcmpl-" : "msg_");
    chat_parser_init(&j->g.parser, r.thinking, r.tools);
    int max_tokens = (int)js_num(b, "max_tokens", openai ? js_num(b, "max_completion_tokens", 32768) : 4096);
    if (max_tokens > CTX - np - 16) max_tokens = CTX - np - 16;
    // 生成的 token 接在 prompt 後面（切換單獨 / 一起、思考收尾後接續時整段就是新的 prompt）
    int32_t *ni = realloc(ids, sizeof(int32_t) * (size_t)(np + (max_tokens > 0 ? max_tokens : 0) + 2));
    if (!ni) {
        respond_err(c, 500, "out of memory");
        chat_parser_free(&j->g.parser);
        free(ids);
        sb_free(&prompt);
        chat_free(&r);
        js_free(b);
        free(j);
        return NULL;
    }
    ids = ni;
    // 思考上限（Strata 的 answer_room / reasoning budget）：max_tokens 也算思考，思考太長會沒有回答。
    // 留 max(512, max_tokens / 4)（最多一半）給回答：有 thinking.budget_tokens 時取兩者小的；沒有時 max_tokens ≥ 1024 才限制
    // （QW_NO_ANSWER_ROOM=1 關掉自動的部分）
    if (r.thinking && max_tokens > 1) {
        const js *th = openai ? NULL : js_get(b, "thinking");
        const int budget = th && th->t == JS_OBJ ? (int)js_num(th, "budget_tokens", 0) : 0;
        int reserve = max_tokens / 4 > 512 ? max_tokens / 4 : 512;
        if (reserve > max_tokens / 2) reserve = max_tokens / 2;
        int cap = 0;
        if (budget > 0) cap = budget < max_tokens - reserve ? budget : max_tokens - reserve;
        else if (max_tokens >= 1024 && !getenv("QW_NO_ANSWER_ROOM")) cap = max_tokens - reserve;
        j->g.think_cap = cap > 0 ? cap : 0;
    }
    // stop_sequences（Anthropic）/ stop（OpenAI）：在回答的文字裡比對
    const js *ss = js_get(b, openai ? "stop" : "stop_sequences");
    if (ss && ss->t == JS_STR && ss->len) { j->g.stops[0] = ss->s; j->g.nstops = 1; }
    for (const js *x = ss && ss->t == JS_ARR ? ss->kid : NULL; x && j->g.nstops < 4; x = x->next)
        if (x->t == JS_STR && x->len) j->g.stops[j->g.nstops++] = x->s;
    for (int i = 0; i < j->g.nstops; i++)
        if (strlen(j->g.stops[i]) > j->g.maxstop) j->g.maxstop = strlen(j->g.stops[i]);
    j->stops[0] = IM_END;
    j->stops[1] = EOT;
    const char *ev = getenv("QW_TEMP");
    // 持久化點：工具定義結束前的 token 邊界、system 訊息結尾（第一個 <|im_end|> 之後）
    int *persist = j->persist, npersist = 0;
    {
        sbuf u = {0}; // 位元組位置要對得上 decode 出來的文字：拿掉 QW_TOK_PLAIN 標記再找
        sb_put(&u, prompt.p, prompt.n);
        chat_unmark(&u);
        const char *te = u.p ? memmem(u.p, u.n, "</tools>", 8) : NULL;
        const bool has_te = te != NULL;
        const size_t off = has_te ? (size_t)(te - u.p) : 0;
        sb_free(&u);
        if (has_te) {
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
    j->r = r;
    j->prompt = prompt;
    j->ids = ids;
    j->gen = ids + np;
    j->np = np;
    return j;
}

// 串流：送出回應標頭（與 Anthropic 的 message_start）。之後連線執行緒閒著時就會送 keepalive
static void stream_start(job *j) {
    if (!j->g.stream) return;
    conn *c = j->c;
    const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    wr(c, h, strlen(h));
    if (!j->openai) { // usage 在這裡只能說整個 prompt；最後的數字在 message_delta（同 Strata）
        sbuf d = {0};
        sb_printf(&d, "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", j->g.id);
        sb_json_str(&d, j->g.model, strlen(j->g.model), false);
        sb_printf(&d, ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":%d,\"output_tokens\":0}}}", j->np);
        sse(&j->g, "message_start", &d);
        sb_free(&d);
    }
    c->sse = j->openai ? 2 : 1;
}

// ---- 思考收尾（answer_room）：思考到上限、或還在 <think> 裡就停了（只做一次）時，把收尾的 token 當成輸出接在後面（照樣串流），
// 之後從 prompt + 已生成接著生成（slot 的快照 / 前綴快取：只多算這幾個 token）
static bool close_ok(job *j) { // 模型還在思考就停了：收尾一次、讓它回答（同 Strata 的 reasoning_close_retry）
    return NCLOSE && !j->g.closed && !j->g.hit && !conn_dead(j->c) && parser_thinking(&j->g.parser) && !getenv("QW_NO_ANSWER_ROOM") &&
           j->got + NCLOSE + 1 <= j->gp.max_tokens;
}
static bool wrap_fits(const job *j) {
    const int k = j->g.wrap == 2 ? NCLOSE : NWRAP;
    return k && j->got + k + 1 <= j->gp.max_tokens;
}
// 插入 j->g.wrap 要的收尾（並清掉 wrap）。回傳 false = 沒插（已經不在思考裡、或沒空間了：照常生成到 max_tokens）
static bool wrap_inject(job *j) {
    const int kind = j->g.wrap;
    j->g.wrap = 0;
    if (!kind || !parser_thinking(&j->g.parser) || conn_dead(j->c) || j->g.hit) return false;
    j->g.wrap = kind;
    const bool fits = wrap_fits(j);
    j->g.wrap = 0;
    if (!fits) return false;
    const int32_t *t = kind == 2 ? CLOSE_IDS : WRAP_IDS;
    const int k = kind == 2 ? NCLOSE : NWRAP;
    if (kind == 2) j->g.closed = true;
    j->g.wrapped = true; // 收尾的 token 不再算思考上限
    fprintf(stderr, kind == 2 ? "[think] 還在思考就停了（%d 個 token）：收尾一次、接著回答\n"
                              : "[think] 思考到上限（%d 個 token，max_tokens %d）：收尾、留空間給回答\n",
            j->g.think_n, j->gp.max_tokens);
    for (int i = 0; i < k; i++) {
        j->gen[j->got++] = t[i];
        on_token(&j->g, t[i]);
    }
    return true;
}

// ---- /status：累計數字（兩種模式都有）
static double T_START;
static long long st_done, st_in, st_cached, st_out, st_cancel, st_err; // 完成的請求、prompt token、重用的 token、生成的 token、用戶端斷線、錯誤
static void lt_count(int np, int cached, int gen, bool dead, bool err); // 跨重啟的累計：加到今天的 bucket（下面）
static void chat_end(job *j) {
    conn *c = j->c;
    const bool openai = j->openai, dead = conn_dead(c);
    gen_ctx *gg = &j->g;
    const int np = j->np, fin = j->fin;
    const qw_gen_stats st = j->st;
    __atomic_fetch_add(&st_done, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&st_in, np, __ATOMIC_RELAXED);
    __atomic_fetch_add(&st_cached, st.cached, __ATOMIC_RELAXED);
    __atomic_fetch_add(&st_out, st.gen, __ATOMIC_RELAXED);
    if (dead) __atomic_fetch_add(&st_cancel, 1, __ATOMIC_RELAXED);
    if (fin == QW_FIN_ERROR) __atomic_fetch_add(&st_err, 1, __ATOMIC_RELAXED);
    lt_count(np, st.cached, st.gen, dead, fin == QW_FIN_ERROR);
    // 被截斷（max_tokens、context 滿、取消）：沒寫完的 tool call 當成文字輸出、不變成 tool_use（parser.cut_call），stop_reason 是 max_tokens
    chat_parser_finish(&gg->parser, fin == QW_FIN_LENGTH || fin == QW_FIN_CTX || dead, on_event, gg);
    if (gg->nstops) stop_flush(gg);
    fprintf(stderr, "[%s] prompt %d（重用 %d）%.0f ms；生成 %d 個 token %.0f ms（%.1f tok/s）；%d 輪、接受 %d 個猜測（n-gram %d 輪）；結束 %d%s%s%s\n",
            openai ? "openai" : "anthropic", np, st.cached, st.prefill_ms, st.gen, st.gen_ms, st.gen && st.gen_ms > 0 ? st.gen / st.gen_ms * 1e3 : 0.0,
            st.rounds, st.accepted, st.ngram_rounds, fin, dead ? "；用戶端已斷線（取消）" : "", fin == QW_FIN_ERROR ? "；錯誤" : "",
            gg->parser.cut_call ? "；tool call 沒寫完（max_tokens）" : "");
    if (gg->parser.rescued || gg->parser.unread)
        fprintf(stderr, "[%s] 從思考裡救出 %d 個 tool call；%d 個沒宣告的工具當成文字\n", openai ? "openai" : "anthropic", gg->parser.rescued, gg->parser.unread);
    const bool tool = gg->ncalls > 0;
    const bool cut = fin == QW_FIN_LENGTH || fin == QW_FIN_CTX;
    const char *sr = gg->hit ? (openai ? "stop" : "stop_sequence") : cut ? (openai ? "length" : "max_tokens")
                   : tool ? (openai ? "tool_calls" : "tool_use") : (openai ? "stop" : "end_turn");
    // usage（同 Strata）：這次新讀的 prompt 算 cache 寫入、重用的算 cache 讀取；input_tokens 留 1（0 會讓 Claude Code 把 prompt 算兩次）
    const int reused = st.cached < 0 ? 0 : st.cached > np - 1 ? np - 1 : st.cached, fresh = np - reused;
    char usage[192];
    snprintf(usage, sizeof usage, "{\"input_tokens\":1,\"cache_creation_input_tokens\":%d,\"cache_read_input_tokens\":%d,\"output_tokens\":%d}",
             fresh - 1, reused, st.gen);
    char ousage[192];
    snprintf(ousage, sizeof ousage, "{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,\"prompt_tokens_details\":{\"cached_tokens\":%d}}", np,
             st.gen, np + st.gen, reused);
    if (fin == QW_FIN_ERROR) { // 已經開始串流：SSE error 事件（Claude Code 對 overloaded_error 會重試）；還沒：HTTP 529 / 500
        const char *m = j->emsg ? j->emsg : "inference engine error";
        if (gg->stream) {
            sbuf d = {0};
            if (openai) {
                sb_puts(&d, "{\"error\":{\"message\":");
                sb_json_str(&d, m, strlen(m), false);
                sb_puts(&d, ",\"type\":\"server_error\"}}");
                sse(gg, NULL, &d);
            } else {
                sb_puts(&d, "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":");
                sb_json_str(&d, m, strlen(m), false);
                sb_puts(&d, "}}");
                sse(gg, "error", &d);
            }
            sb_free(&d);
        } else respond_err(c, j->ecode ? j->ecode : 500, m);
    } else if (gg->stream) {
        sbuf d = {0};
        if (openai) {
            sb_printf(&d, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", gg->id, (long)time(NULL));
            sb_json_str(&d, gg->model, strlen(gg->model), false);
            sb_printf(&d, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],\"usage\":%s}", sr, ousage);
            sse(gg, NULL, &d);
            wr(c, "data: [DONE]\n\n", 14);
        } else {
            anth_close_block(gg);
            sb_printf(&d, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":", sr);
            if (gg->hit) sb_json_str(&d, gg->hit, strlen(gg->hit), false);
            else sb_puts(&d, "null");
            sb_printf(&d, "},\"usage\":%s}", usage);
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
            sb_printf(&o, "},\"finish_reason\":\"%s\"}],\"usage\":%s}", sr, ousage);
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
            sb_printf(&o, ",\"usage\":%s}", usage);
        }
        respond(c, 200, "application/json", o.p, o.n);
        sb_free(&o);
    }
    job_free(j);
}

// ---- 單一請求（--parallel 1）：生成在這個執行緒，另一個執行緒負責寫 socket（keepalive、斷線偵測）
typedef struct {
    conn *c;
    atomic_bool stop;
} pump_arg;
static void *pump_main(void *a) {
    pump_arg *p = a;
    while (!atomic_load(&p->stop)) pump_step(p->c, &p->stop);
    return NULL;
}
static bool serial_cb(void *ctx, int32_t t) {
    job *j = ctx;
    j->gen[j->got++] = t;
    return on_token(&j->g, t) && !j->g.wrap; // 思考到上限：停下來插入收尾再接著生成
}
static bool serial_pfy(void *ctx) { return conn_dead(((job *)ctx)->c); } // 長 prompt 的 prefill 段之間：用戶端斷線就停（快照留著，重試時接續）

static void handle_chat(conn *c, const char *body, size_t blen, bool openai) {
    job *j = chat_begin(c, body, blen, openai);
    if (!j) return;
    stream_start(j);
    pump_arg pa = {.c = c};
    atomic_init(&pa.stop, false);
    pthread_t pt;
    const bool pumping = !pthread_create(&pt, NULL, pump_main, &pa);
    j->rng = j->gp.seed;
    int fin = QW_FIN_LENGTH;
    for (bool first = true;; first = false) {
        if (j->got >= j->gp.max_tokens) { fin = QW_FIN_LENGTH; break; }
        qw_gen g2 = j->gp;
        g2.max_tokens = j->gp.max_tokens - j->got;
        g2.seed = j->rng; // 第一段 = 請求的 seed（與以前相同）
        g2.pf_yield = serial_pfy;
        qw_gen_stats s2 = {0};
        fin = qw_engine_generate_ex(E, j->ids, j->np + j->got, &g2, serial_cb, j, &s2);
        if (first) j->st = s2;
        else {
            j->st.rounds += s2.rounds;
            j->st.accepted += s2.accepted;
            j->st.ngram_rounds += s2.ngram_rounds;
            j->st.gen_ms += s2.gen_ms + s2.prefill_ms;
        }
        j->rng = j->rng * 6364136223846793005ull + 1442695040888963407ull;
        if (fin == QW_FIN_YIELD) { fin = QW_FIN_CALLBACK; break; } // 只有 serial_pfy 會讓出：用戶端斷線
        if (conn_dead(c) || j->g.hit) break;
        if (fin == QW_FIN_CALLBACK && j->g.wrap) { wrap_inject(j); continue; } // 插不進去也接著生成（思考到 max_tokens）
        if (fin == QW_FIN_STOP && close_ok(j)) { j->g.wrap = 2; if (wrap_inject(j)) continue; }
        break;
    }
    j->fin = fin;
    j->st.gen = j->got;
    if (fin == QW_FIN_ERROR) { j->ecode = 500; j->emsg = "inference engine error"; }
    if (pumping) {
        pthread_mutex_lock(&c->mu);
        atomic_store(&pa.stop, true);
        pthread_cond_signal(&c->cv);
        pthread_mutex_unlock(&c->mu);
        pthread_join(pt, NULL);
    }
    chat_end(j);
}

// ---------------------------------------------------------------- --parallel N：多個請求一起 decode
// 連線執行緒：讀請求、解析（chat_begin）→ 排進佇列 → 等 engine 執行緒做完 → 收尾（chat_end）
// engine 執行緒：新請求逐一 prefill（期間其他對話暫停），之後每步一起 decode（qw_engine_par_step），token 經 on_token 串流給各自的連線
static int PAR = -1; // 同時最多幾個請求；-1 = 自動（16；只有一部分是 GGUF 時 1）
static pthread_mutex_t q_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_cv = PTHREAD_COND_INITIALIZER;
static job *q_head, *q_tail;
static int q_len;                 // 佇列裡幾個（q_mu）
enum { QUEUE_MAX = 64 };          // 排隊超過這麼多就回 529（Claude Code 會退避重試）
static volatile int par_na;       // engine 執行緒：目前進行中的對話數（讓出前看新請求進不進得來）

// 有排隊、而且進得來（還沒到 PAR 個）：16 個都在跑時讓出沒有用，只會白白切換
static bool q_admit(void) {
    if (par_na >= PAR) return false;
    pthread_mutex_lock(&q_mu);
    const bool p = q_head != NULL;
    pthread_mutex_unlock(&q_mu);
    return p;
}
// 生成的 token 接在 prompt 後面（j->ids[np + k]，j->gen = j->ids + np），切換單獨 / 一起時整段就是新的 prompt。
// j->base：slot 的 KV 內容目前記到哪（上一次 prefill 的長度），結束時從這裡之後記快照。
// 取消（用戶端斷線，fin = QW_FIN_CALLBACK）也走這裡：分段 prefill 中途停下時 slot 停在段界（kvlen = pos = 算到的位置），
// 狀態一致；單獨跑的 prefill 從 pf_yield 讓出時 engine 已在那裡記了快照，重試的請求直接接續
static void par_done(job *j, int fin) {
    j->fin = fin;
    j->st.gen = j->got;
    j->st.gen_ms = (qw_now() - j->tg0) * 1e3;
    if (j->sid >= 0) qw_engine_par_finish(E, j->sid, j->ids + j->base, j->np + j->got - j->base);
    conn *c = j->c;
    pthread_mutex_lock(&c->mu); // 在 c->mu 裡設 done：連線執行緒只會在這之後釋放 job / conn
    atomic_store(&j->done, true);
    pthread_cond_signal(&c->cv);
    pthread_mutex_unlock(&c->mu);
}
static void par_fail(job *j, int code, const char *msg) {
    j->ecode = code;
    j->emsg = msg;
    fprintf(stderr, "[par] 請求失敗（%d）：%s\n", code, msg);
    par_done(j, QW_FIN_ERROR);
}
// 思考收尾之後要重新加入的對話（離開了一起 decode，下一輪從 prompt + 已生成接續）
static job *rejoin[QW_PAR_MAX];
static int nrejoin;
static void leave_batch(job *j);
// 一起 decode 中要插入思考收尾：離開 batch（記快照）→ 插入 → 下一輪重新加入。回傳 true = 不用插（照常留在 batch）
static bool par_wrap(job *j) {
    if (!parser_thinking(&j->g.parser) || !wrap_fits(j)) { j->g.wrap = 0; return true; }
    if (j->inb) leave_batch(j);
    wrap_inject(j);
    // 快照停在插入之前（leave_batch 記的）：之後取消時 par_finish 不能把插入的 token 當成已經進模型的（ngen = 0 → 不記）
    j->base = j->np + j->got;
    if (j->g.hit || conn_dead(j->c)) { par_done(j, QW_FIN_CALLBACK); return false; }
    if (j->got >= j->gp.max_tokens) { par_done(j, QW_FIN_LENGTH); return false; }
    rejoin[nrejoin++] = j;
    return false;
}
// 送出一個 token（同 qw_engine_generate_ex 的 EMIT）：回傳 false = 這個請求不在這一輪了（結束、或等著重新加入）
static bool par_emit(job *j, int32_t t) {
    for (int q = 0; q < j->gp.nstop; q++)
        if (j->gp.stop[q] == t) {
            if (close_ok(j)) { j->g.wrap = 2; return par_wrap(j); } // 還在思考就停了：收尾、接著回答（停止 token 不留）
            par_done(j, QW_FIN_STOP);
            return false;
        }
    j->gen[j->got++] = t;
    if (!on_token(&j->g, t)) { par_done(j, QW_FIN_CALLBACK); return false; }
    if (j->got >= j->gp.max_tokens) { par_done(j, QW_FIN_LENGTH); return false; }
    j->last = t;
    if (j->g.wrap) return par_wrap(j);
    return true;
}
// ---- 排程：從實測速度選比較快的做法（不必指定幾路）
//   推測解碼輪流跑（一次一個對話、時間片 SLICE_MS）：總速度用牆鐘實測（rr_rate：一段時間內輪流生成的 token ÷ 經過的時間，
//     含 slot 切換、快照存取、排程本身；分段 prefill 的時間扣掉）。還沒量過時用各對話推測解碼速度的平均估（偏樂觀）
//   一起 decode：總速度 = B / 一步的時間（實測；還沒量過的 B 用 engine 的各組時間估）
// 輪流只在 ≤ SLICE_MAX_N 個對話時考慮（QW_PAR_SLICE_MAX；更多時切換成本一定吃掉推測解碼的好處：實測 15 個輪流總共只有約 4 tok/s，
// 一起 decode 約 160 tok/s）；換做法之後至少維持 MODE_HOLD_S 秒（免得來回切）；輪流時每個對話至少跑 MIN_SLICE_MS 才讓出。
// 兩種做法都和單獨 greedy 逐位元相同，只差快慢；只有一個請求時一定是推測解碼
enum { SLICE_MS = 300, MIN_SLICE_MS = 250, MODE_HOLD_S = 5, PF_AGE_S = 30 };
static int SLICE_MAX_N = 3;
static volatile int par_others; // 單獨跑的時候，另外還有幾個進行中的對話在等（> 0 才切時間片）
static volatile bool par_pfwait; // 有對話在分段 prefill：時間片縮短（prefill 多分一點時間）
static double seg_t0;
static double spec_rate;                 // 推測解碼的實測速度（tok/s，EMA；一段之內，不含切換）
static double rr_rate[QW_PAR_MAX + 1];   // n 個對話輪流推測解碼的實測總速度（tok/s，EMA，牆鐘、含切換）
static double bat_rate[QW_PAR_MAX + 1];  // 一起 decode B 個的實測總速度（tok/s，EMA；兩種一起的做法取快的）
static double pl_rate[QW_PAR_MAX + 1], vs_rate[QW_PAR_MAX + 1]; // 一起 decode / 一起推測解碼 各自的實測（tok/s，EMA）
static bool solo_cb(void *ctx, int32_t t) { // 要讓出時只設 yield：這一輪的 token 送完才停（模型狀態停在最後一個 token 之前）
    job *j = ctx;
    j->gen[j->got++] = t;
    if (!on_token(&j->g, t)) { j->cbstop = true; return false; }
    if (j->g.wrap) j->yflag = true; // 思考到上限：這一輪送完就停、插入收尾
    const double ms = (qw_now() - seg_t0) * 1e3;
    if (!j->yflag && ms >= MIN_SLICE_MS && ((par_others && ms > (par_pfwait ? SLICE_MS * 2 / 3 : SLICE_MS)) || q_admit())) j->yflag = true;
    return true;
}
static double ema(double o, double x, double w) { return o > 0 ? (1 - w) * o + w * x : x; }
// 單獨跑的長 prompt：每段 prefill 之間，用戶端斷線就停；有進得來的新請求（且這段至少跑了 MIN_SLICE_MS）就讓出。
// 讓出時 engine 在段界記快照（st->pf_done），之後接著算；取消的請求重試時也從這裡接續
static bool solo_pfy(void *ctx) {
    job *j = ctx;
    if (conn_dead(j->c)) return true;
    return (qw_now() - seg_t0) * 1e3 >= MIN_SLICE_MS && q_admit();
}
// 推測解碼跑一段（到結束、有新請求、或時間片用完）。回傳 true = 還沒結束（狀態留在 slot，下次從 prompt + 已生成接續）
// *ng = 這一段生成了幾個 token、*pf = 其中花在 prefill 的 ms（呼叫端量輪流速度用；j 結束後就不能再碰，所以從這裡帶出來）
static bool run_solo(job *j, int *ng_out, double *pf_out) {
    qw_gen g2 = j->gp;
    g2.max_tokens = j->gp.max_tokens - j->got;
    g2.seed = j->rng;
    j->preempt = j->cbstop = j->yflag = false;
    g2.yield = &j->yflag;
    g2.pf_yield = solo_pfy;
    qw_gen_stats s2;
    const int got0 = j->got;
    const bool first = !j->st.prompt;
    seg_t0 = qw_now();
    const int fin = qw_engine_par_solo(E, j->sid, j->ids, j->np + j->got, &g2, solo_cb, j, &s2);
    double ms = (qw_now() - seg_t0) * 1e3;
    *ng_out = 0;
    *pf_out = 0;
    if (fin == QW_FIN_YIELD && s2.pf_done > 0) { // prompt 還沒算完就讓出：之後分段 prefill 接著算（快照在 pf_done）
        j->pfy = true;
        j->st.cached = s2.cached;
        *pf_out = ms;
        return true;
    }
    if (first) { j->st.prompt = j->np; j->st.cached = s2.cached; j->st.prefill_ms = s2.prefill_ms; ms -= s2.prefill_ms; *pf_out = s2.prefill_ms; } // 第一段扣掉 prompt
    // 之後每段開頭從快照接續的部分是切換成本：不扣（輪流的實測速度要含它）
    const int ng = j->got - got0;
    *ng_out = ng;
    if (ng >= 8 && ms > 1) { j->srate = ema(j->srate, ng * 1e3 / ms, 0.5); spec_rate = ema(spec_rate, ng * 1e3 / ms, 0.3); }
    j->st.rounds += s2.rounds;
    j->st.accepted += s2.accepted;
    j->st.ngram_rounds += s2.ngram_rounds;
    j->base = j->np + j->got; // generate_ex 結束時記了快照（不一定到最後一個 token，par_prefill / par_solo 會找能用的）
    j->rng = j->rng * 6364136223846793005ull + 1442695040888963407ull;
    if (fin == QW_FIN_STOP && close_ok(j)) j->g.wrap = 2; // 還在思考就停了：收尾、接著回答（停止 token 不在 gen 裡）
    if (j->g.wrap && (fin == QW_FIN_YIELD || fin == QW_FIN_STOP)) { // 下一段從 prompt + 已生成 + 收尾接續（快照在 generate_ex 結束時記了）
        const bool injected = wrap_inject(j);
        j->base = j->np + j->got; // 同 par_wrap：插入的 token 還沒進模型，取消時不能記進快照
        if (fin == QW_FIN_YIELD || injected) {
            if (j->g.hit || conn_dead(j->c)) { par_done(j, QW_FIN_CALLBACK); return false; }
            if (j->got >= j->gp.max_tokens) { par_done(j, QW_FIN_LENGTH); return false; }
            return true;
        }
    }
    if (fin == QW_FIN_ERROR) { par_fail(j, 500, "inference engine error (solo decode)"); return false; }
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
    if (!lg) { par_fail(j, 500, "inference engine error (prefill)"); return false; }
    if (!j->st.prompt) { j->st.prompt = j->np; j->st.cached = cached; j->st.prefill_ms = (qw_now() - j->t0) * 1e3; j->tg0 = qw_now(); }
    j->inb = true;
    j->hasd = false;
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
    if (!lg) { par_fail(j, 500, "inference engine error (chunked prefill)"); return false; }
    if (first && !j->pfy) j->st.cached = cached;
    if (j->pfd < j->np) return true;
    j->pfing = false;
    j->base = j->np;
    j->st.prompt = j->np;
    j->st.prefill_ms = (qw_now() - j->t0) * 1e3;
    j->tg0 = qw_now();
    j->inb = true;
    j->hasd = false;
    return par_emit(j, qw_engine_sample(lg, &j->gp, &j->rng));
}
// 離開一起 decode：記快照（最後送出、還沒餵的 token 留給下一段當 prompt 結尾）
static void leave_batch(job *j) {
    qw_engine_par_finish(E, j->sid, j->ids + j->base, j->np + j->got - j->base);
    j->inb = false;
    j->hasd = false;
}
// ---- 一起推測解碼（實驗，QW_PAR_VSPEC=1；GGUF + DFlash2）：每個對話用自己的 DFlash2 猜一塊，2…VS_MAX 個對話時共 ≤ 8 列一起驗證。
// 實測與一起 decode 差不多快（多驗證一列約 3 ms、猜中率約 70%，加上每輪猜測 / 接受的成本），所以預設關
static int VS_MAX = 3; // 4 個以上一起 decode 已經夠快（實測推測解碼不會更快）
static double vs_conf = 0.8; // 猜測的信心低於這個就截斷（同單獨推測解碼的 QW_DF_CONF）
// 後綴猜測（同 qw_engine_generate_ex 的 prompt lookup）：最近 3 個 token 在前文出現過時，提出前文接在後面的（最多 7 個）。
// 回傳個數，*mlen = 匹配長度
static int suffix_draft(const int32_t *seq, int ns, int32_t *sf, int *mlen) {
    int best_i = -1, best_m = 0, ks = 0;
    if (ns >= 4)
        for (int i = ns - 2; i >= 2 && best_m < 32; i--) {
            if (seq[i] != seq[ns - 1] || seq[i - 1] != seq[ns - 2] || seq[i - 2] != seq[ns - 3]) continue;
            int m = 3;
            while (m < 32 && i - m >= 0 && seq[i - m] == seq[ns - 1 - m]) m++;
            if (m > best_m) { best_m = m; best_i = i; }
            if (best_m >= 8) break;
        }
    if (best_i >= 0) while (ks < 7 && best_i + 1 + ks < ns) { sf[ks] = seq[best_i + 1 + ks]; ks++; }
    *mlen = best_m;
    return ks;
}
// 一輪：沒有猜測的先猜 → 依信心分配列數 → 驗證 → 接受判定 → 狀態寫回並猜下一塊 → 送出 token。回傳還活著的放進 out
static int vspec_round(job **bj, int nb, job **out) {
    int sids[8], need[8], nn = 0, n[8], nd[8], cl[8];
    int32_t anc[8], dr[8][8], toks[8];
    float pp[8][8];
    for (int i = 0; i < nb; i++) if (!bj[i]->hasd) { need[nn] = bj[i]->sid; anc[nn++] = bj[i]->last; }
    if (nn && qw_engine_par_vdraft(E, need, anc, nn)) return -1;
    int rows = nb;
    for (int i = 0; i < nb; i++) {
        bj[i]->hasd = true;
        sids[i] = bj[i]->sid;
        const int m = qw_engine_par_draft(E, sids[i], dr[i], pp[i]);
        int k = 0;
        while (k < m && k < 7 && (k == 0 || pp[i][k] >= vs_conf)) k++;
        // 後綴：照抄（匹配 ≥ 20 且給得出 7 個）、或比 DFlash2 長且（匹配 ≥ 8、DFlash2 沒把握、或第一個一致）時改用（同單獨推測解碼）
        int32_t sf[8];
        int ml = 0;
        const int ks = suffix_draft(bj[i]->ids, bj[i]->np + bj[i]->got, sf, &ml);
        bool agree = true;
        for (int q = 0; q < k && agree; q++) agree = sf[q] == dr[i][q];
        const bool copy = ks == 7 && ml >= 20 && (k < 7 || agree);
        if (ks >= 2 && (copy || (ks > k && (ml >= 8 || k < 2 || sf[0] == dr[i][0])))) { memcpy(dr[i], sf, sizeof(int32_t) * (size_t)ks); k = ks; }
        cl[i] = k;
        nd[i] = 0;
    }
    for (bool more = true; more && rows < 8;) { // 輪流加一列，給還有把握的猜測
        more = false;
        for (int i = 0; i < nb && rows < 8; i++) if (nd[i] < cl[i]) { nd[i]++; rows++; more = true; }
    }
    int r = 0;
    for (int i = 0; i < nb; i++) {
        n[i] = 1 + nd[i];
        toks[r++] = bj[i]->last;
        for (int t = 0; t < nd[i]; t++) toks[r++] = dr[i][t];
    }
    const double t0 = qw_now();
    double gms = 0;
    const float *lg = qw_engine_par_vstep(E, sids, n, toks, nb, &gms);
    if (!lg) return -1;
    int acc[8], r0[8];
    int32_t nx[8];
    for (int i = 0, q = 0; i < nb; i++) { r0[i] = q; q += n[i]; }
    int *accp = acc, *r0p = r0, *ndp = nd;
    int32_t *nxp = nx, (*drp)[8] = dr;
    job **bjp = bj;
    dispatch_apply((size_t)nb, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        accp[i] = qw_engine_spec_accept(lg + (size_t)r0p[i] * QW_VOCAB, drp[i], ndp[i], &bjp[i]->gp, &bjp[i]->rng, &nxp[i]);
    });
    if (qw_engine_par_vaccept(E, acc, nx)) return -1;
    const double dt = (qw_now() - t0) * 1e3;
    int ntok = 0, no = 0;
    for (int i = 0; i < nb; i++) ntok += acc[i] + 1;
    if (dt > 0) vs_rate[nb] = ema(vs_rate[nb], ntok * 1e3 / dt, 0.1);
    bat_rate[nb] = vs_rate[nb] > pl_rate[nb] ? vs_rate[nb] : pl_rate[nb];
    static int prof = -1;
    if (prof < 0) prof = getenv("QW_PAR_PROF") != NULL;
    if (prof) { static int c = 0; if (++c % 20 == 0) fprintf(stderr, "[parv] B=%d %d 列：%.1f ms、%d 個 token（%.0f tok/s，GPU 驗證 %.1f ms）\n", nb, r, dt, ntok, ntok * 1e3 / dt, gms); }
    for (int i = 0; i < nb; i++) {
        job *j = bj[i];
        bool alive = true;
        for (int t = 0; t < acc[i] && alive; t++) alive = par_emit(j, dr[i][t]);
        if (alive) alive = par_emit(j, nx[i]);
        if (alive && j->np + j->got + 10 >= CTX) { par_done(j, QW_FIN_CTX); alive = false; }
        if (alive) out[no++] = j;
    }
    return no;
}
// 這 n 個對話：輪流推測解碼比一起 decode 快嗎（cur = 目前的做法，差不多時不換，免得來回切）
// ---- 記憶體保護：slot（KV、狀態、快照）合計不超過 SLOT_BUDGET；系統可用記憶體低於 10% 時再多放一組閒置的
static size_t SLOT_BUDGET;
static size_t st_mem; // /status：slot 目前佔的記憶體（mem_guard 更新）
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
    st_mem = qw_engine_par_mem(E);
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
    if (n > SLICE_MAX_N) return false;
    double rs = rr_rate[n]; // 實測（含切換）；還沒量過：各對話一段之內的速度平均
    if (rs <= 0) {
        for (int i = 0; i < n; i++) rs += act[i]->srate > 0 ? act[i]->srate : spec_rate;
        rs /= n;
    }
    double rb = bat_rate[n];
    if (rb <= 0) { const double c = qw_engine_par_cost(E, n); rb = c > 0 ? n * 1e3 / c : 0; }
    if (rs <= 0 || rb <= 0) return false; // 沒數字：一起（2 個以上通常比較快）
    return rs > rb * (cur ? 0.95 : 1.05);
}
// /status 的即時狀態（par_loop 每輪更新）
static pthread_mutex_t st_mu = PTHREAD_MUTEX_INITIALIZER;
static struct {
    int id, np, cached, got, prefilled;
    char state;
    double tps, t0;
    char model[64]; // 複製（job 結束後就被釋放）
} st_job[QW_PAR_MAX];
static int st_n;
static const char *st_mode = "idle";
static void st_publish(job **act, int na, const char *mode) {
    int ninb = 0;
    for (int i = 0; i < na; i++) ninb += act[i]->inb;
    pthread_mutex_lock(&st_mu);
    st_n = na;
    for (int i = 0; i < na; i++) {
        job *j = act[i];
        st_job[i].id = j->id;
        st_job[i].np = j->np;
        st_job[i].cached = j->st.cached;
        st_job[i].got = j->got;
        st_job[i].prefilled = j->pfing ? j->pfd : j->st.prompt ? j->np : 0;
        st_job[i].state = j->pfing ? 'p' : j->inb ? 'b' : 's';
        // 每個對話的速度：一起 decode 時 = 實測總速度 ÷ 人數；輪流時 = 它自己推測解碼的實測（一段之內）
        st_job[i].tps = j->inb ? (ninb > 0 && bat_rate[ninb] > 0 ? bat_rate[ninb] / ninb : 0) : j->srate;
        st_job[i].t0 = j->tc;
        snprintf(st_job[i].model, sizeof st_job[i].model, "%s", j->g.model);
    }
    if (mode) st_mode = mode;
    pthread_mutex_unlock(&st_mu);
}
// ---- 累計數字跨重啟保留（--cache-dir 裡的 stats.json，或 --stats-file）
// 檔案存「之前幾次」的總和（base）與每天（本地時區的日期）的 bucket；這次的總數仍是 st_*，base + st_* 就是 lifetime。
// 每個請求完成時加到當天的 bucket；服務時間（開始監聽之後）每 30 秒、/status 與存檔時記到當天，跨過午夜就拆到兩天。
// 只留最近 LT_KEEP_DAYS 天的 bucket。每 30 秒檢查一次：數字變了（或距上次存超過 10 分鐘）就寫；SIGTERM / SIGINT 時先寫再結束。
// 寫法：暫存檔 + fsync + rename
enum { LT_KEEP_DAYS = 400, LT_DAYS_MAX = 512 };
typedef struct {
    long long since, done, in, cached, out, cancel, err; // since = 第一次記錄的 unix 時間
    double uptime;                                        // 累計服務秒數
} lt_stats;
typedef struct {
    char date[11]; // YYYY-MM-DD（本地時區）
    long long done, in, cached, out, cancel, err;
    double uptime;
} lt_day;
static char *LT_PATH;    // NULL = 不保留
static lt_stats lt_base; // 載入時的值（之前幾次的總和）
static pthread_mutex_t lt_mu = PTHREAD_MUTEX_INITIALIZER; // 保護 lt_days / lt_nd / lt_up_t / 存檔
static lt_day lt_days[LT_DAYS_MAX];                       // 依日期排序（舊 → 新）
static int lt_nd;
static double lt_up_t;       // 服務時間記到哪個時間點（unix 秒；0 = 還沒開始服務）
static long long lt_last[6]; // 上次存檔時的 done / in / cached / out / cancel / err
static double lt_last_t;     // 上次存檔的時間（qw_now）
static double wall_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}
static void lt_date(double t, char out[11]) {
    const time_t s = (time_t)t;
    struct tm tm;
    localtime_r(&s, &tm);
    strftime(out, 11, "%Y-%m-%d", &tm);
}
static double lt_next_midnight(double t) { // t 之後的下一個本地午夜
    const time_t s = (time_t)t;
    struct tm tm;
    localtime_r(&s, &tm);
    tm.tm_hour = tm.tm_min = tm.tm_sec = 0;
    tm.tm_mday++;
    tm.tm_isdst = -1;
    return (double)mktime(&tm);
}
static bool lt_date_ok(const char *s) {
    if (!s || strlen(s) != 10 || s[4] != '-' || s[7] != '-') return false;
    for (int i = 0; i < 10; i++)
        if (i != 4 && i != 7 && (s[i] < '0' || s[i] > '9')) return false;
    return true;
}
static void lt_prune(double now) { // 刪掉 LT_KEEP_DAYS 天以前的 bucket（lt_mu 持有）
    char cut[11];
    lt_date(now - LT_KEEP_DAYS * 86400.0, cut);
    int k = 0;
    while (k < lt_nd && strcmp(lt_days[k].date, cut) < 0) k++;
    if (k) { memmove(lt_days, lt_days + k, sizeof *lt_days * (size_t)(lt_nd - k)); lt_nd -= k; }
}
static lt_day *lt_day_get(const char *date) { // date 的 bucket，沒有就照日期順序插入（lt_mu 持有）
    int i = lt_nd;
    while (i > 0 && strcmp(lt_days[i - 1].date, date) > 0) i--;
    if (i > 0 && !strcmp(lt_days[i - 1].date, date)) return &lt_days[i - 1];
    if (lt_nd == LT_DAYS_MAX) { // 滿了：丟最舊的（比全部都舊的就不收）
        if (i == 0) return NULL;
        memmove(lt_days, lt_days + 1, sizeof *lt_days * (size_t)(lt_nd - 1));
        lt_nd--;
        i--;
    }
    memmove(lt_days + i + 1, lt_days + i, sizeof *lt_days * (size_t)(lt_nd - i));
    lt_days[i] = (lt_day){0};
    memcpy(lt_days[i].date, date, 11);
    lt_nd++;
    return &lt_days[i];
}
static void lt_up_flush(double now) { // 服務時間記到各天（跨過午夜就拆開）（lt_mu 持有）
    if (lt_up_t <= 0 || now <= lt_up_t) return;
    for (double t = lt_up_t; t < now;) {
        const double m = lt_next_midnight(t), e = m > t && m < now ? m : now;
        char d[11];
        lt_date(t, d);
        lt_day *b = lt_day_get(d);
        if (b) b->uptime += e - t;
        t = e;
    }
    lt_up_t = now;
}
static void lt_serving(void) { // 開始監聽：服務時間從這裡起算
    if (!LT_PATH) return;
    pthread_mutex_lock(&lt_mu);
    lt_up_t = wall_now();
    pthread_mutex_unlock(&lt_mu);
}
static void lt_count(int np, int cached, int gen, bool dead, bool err) { // 一個請求完成：加到今天的 bucket
    if (!LT_PATH) return;
    char d[11];
    lt_date(wall_now(), d);
    pthread_mutex_lock(&lt_mu);
    lt_day *b = lt_day_get(d);
    if (b) { b->done++; b->in += np; b->cached += cached; b->out += gen; b->cancel += dead; b->err += err; }
    pthread_mutex_unlock(&lt_mu);
}
static lt_stats lt_now(void) { // base + 這次的數字
    lt_stats s = lt_base;
    s.done += __atomic_load_n(&st_done, __ATOMIC_RELAXED);
    s.in += __atomic_load_n(&st_in, __ATOMIC_RELAXED);
    s.cached += __atomic_load_n(&st_cached, __ATOMIC_RELAXED);
    s.out += __atomic_load_n(&st_out, __ATOMIC_RELAXED);
    s.cancel += __atomic_load_n(&st_cancel, __ATOMIC_RELAXED);
    s.err += __atomic_load_n(&st_err, __ATOMIC_RELAXED);
    if (T_START > 0) s.uptime += qw_now() - T_START;
    return s;
}
static bool lt_num(const js *v, const char *k, long long *o) { // 非負整數欄位
    const double x = js_num(v, k, -1);
    if (!(x >= 0 && x < 9e15)) return false;
    *o = (long long)x;
    return true;
}
// 讀 path（總數與每天的 bucket）：沒有檔案 → 從零開始；壞掉（不是 JSON、總數欄位缺少或是負的）→ 改名成 .bad、從零開始。
// 都只記一行。不合格的單日 bucket 略過
static bool lt_load(const char *path, lt_stats *out) {
    *out = (lt_stats){.since = (long long)time(NULL)};
    lt_nd = 0;
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (errno == ENOENT) fprintf(stderr, "[stats] %s 不存在：累計數字從零開始\n", path);
        else fprintf(stderr, "[stats] 讀不到 %s（%s）：累計數字從零開始\n", path, strerror(errno));
        return false;
    }
    enum { CAP = 8 << 20 };
    char *buf = malloc(CAP);
    const size_t n = buf ? fread(buf, 1, CAP - 1, f) : 0;
    fclose(f);
    char err[128] = "";
    js *v = buf && n < CAP - 1 ? js_parse(buf, n, err, sizeof err) : NULL;
    free(buf);
    bool ok = v && v->t == JS_OBJ;
    lt_stats s = {0};
    if (ok) {
        ok = lt_num(v, "since", &s.since) && lt_num(v, "done", &s.done) && lt_num(v, "prompt_tokens", &s.in) &&
             lt_num(v, "cached_tokens", &s.cached) && lt_num(v, "output_tokens", &s.out) && lt_num(v, "cancelled", &s.cancel) &&
             lt_num(v, "errors", &s.err);
        s.uptime = js_num(v, "uptime_s", -1);
        if (!(s.uptime >= 0 && s.uptime < 9e15)) ok = false;
    }
    if (ok) {
        const js *days = js_get(v, "days");
        if (days && days->t == JS_OBJ)
            for (const js *k = days->kid; k; k = k->next) {
                lt_day b = {0};
                const double up = js_num(k, "uptime_s", 0);
                if (k->t != JS_OBJ || !lt_date_ok(k->key) || !(up >= 0 && up < 9e15)) continue;
                if (!(lt_num(k, "done", &b.done) && lt_num(k, "prompt_tokens", &b.in) && lt_num(k, "cached_tokens", &b.cached) &&
                      lt_num(k, "output_tokens", &b.out)))
                    continue;
                lt_num(k, "cancelled", &b.cancel);
                lt_num(k, "errors", &b.err);
                lt_day *d = lt_day_get(k->key);
                if (d) { d->done += b.done; d->in += b.in; d->cached += b.cached; d->out += b.out; d->cancel += b.cancel; d->err += b.err; d->uptime += up; }
            }
        lt_prune(wall_now());
    }
    js_free(v);
    if (!ok) {
        lt_nd = 0;
        char bad[1100];
        snprintf(bad, sizeof bad, "%s.bad", path);
        rename(path, bad);
        fprintf(stderr, "[stats] %s 格式不對（%s）：改名成 .bad，累計數字從零開始\n", path, *err ? err : "欄位缺少或不合理");
        return false;
    }
    if (s.since <= 0) s.since = (long long)time(NULL);
    *out = s;
    return true;
}
// 總數；days > 0 時加上最近 days 天（含今天）的 bucket，days < 0 = 全部（lt_mu 持有）
static void lt_json(sbuf *o, const lt_stats *s, int days) {
    sb_printf(o, "{\"since\":%lld,\"uptime_s\":%.0f,\"done\":%lld,\"prompt_tokens\":%lld,\"cached_tokens\":%lld,\"output_tokens\":%lld,"
                 "\"cancelled\":%lld,\"errors\":%lld",
              s->since, s->uptime, s->done, s->in, s->cached, s->out, s->cancel, s->err);
    if (days) {
        char cut[11] = "";
        if (days > 0) lt_date(wall_now() - (days - 1) * 86400.0, cut);
        sb_puts(o, ",\"days\":{");
        bool first = true;
        for (int i = 0; i < lt_nd; i++) {
            const lt_day *b = &lt_days[i];
            if (strcmp(b->date, cut) < 0) continue;
            sb_printf(o, "%s\"%s\":{\"done\":%lld,\"prompt_tokens\":%lld,\"cached_tokens\":%lld,\"output_tokens\":%lld,\"cancelled\":%lld,"
                         "\"errors\":%lld,\"uptime_s\":%.0f}",
                      first ? "" : ",", b->date, b->done, b->in, b->cached, b->out, b->cancel, b->err, b->uptime);
            first = false;
        }
        sb_puts(o, "}");
    }
    sb_puts(o, "}");
}
// 寫到 path.tmp.<pid>、fsync、rename（不會留下寫一半的檔）。目錄不在時建一層（lt_mu 持有）
static bool lt_write(const char *path, const lt_stats *s) {
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp.%d", path, (int)getpid());
    FILE *f = fopen(tmp, "wb");
    if (!f && errno == ENOENT) {
        char dir[1100];
        snprintf(dir, sizeof dir, "%s", path);
        char *sl = strrchr(dir, '/');
        if (sl && sl != dir) { *sl = 0; mkdir(dir, 0755); f = fopen(tmp, "wb"); }
    }
    if (!f) return false;
    sbuf o = {0};
    lt_json(&o, s, -1);
    sb_puts(&o, "\n");
    bool ok = fwrite(o.p, 1, o.n, f) == o.n && fflush(f) == 0 && fsync(fileno(f)) == 0;
    ok = fclose(f) == 0 && ok;
    sb_free(&o);
    if (ok && rename(tmp, path) == 0) return true;
    const int e = errno;
    unlink(tmp);
    errno = e;
    return false;
}
// force = 一定寫（結束時）；否則數字變了、或距上次存超過 600 秒（只有服務時間變）才寫
static void lt_save(bool force) {
    if (!LT_PATH) return;
    pthread_mutex_lock(&lt_mu);
    const double w = wall_now();
    lt_up_flush(w);
    lt_prune(w);
    const lt_stats s = lt_now();
    const long long cur[6] = {s.done, s.in, s.cached, s.out, s.cancel, s.err};
    const double now = qw_now();
    if (force || memcmp(cur, lt_last, sizeof cur) || now - lt_last_t >= 600) {
        if (lt_write(LT_PATH, &s)) { memcpy(lt_last, cur, sizeof cur); lt_last_t = now; }
        else fprintf(stderr, "[stats] 寫不進 %s（%s）\n", LT_PATH, strerror(errno));
    }
    pthread_mutex_unlock(&lt_mu);
}
static void lt_init(const char *path) { // 載入；path NULL = 不保留
    if (!path) return;
    LT_PATH = strdup(path);
    pthread_mutex_lock(&lt_mu);
    const bool had = lt_load(LT_PATH, &lt_base);
    pthread_mutex_unlock(&lt_mu);
    const lt_stats b = lt_base;
    const long long l[6] = {b.done, b.in, b.cached, b.out, b.cancel, b.err};
    memcpy(lt_last, l, sizeof lt_last);
    lt_last_t = qw_now();
    if (had)
        fprintf(stderr, "[stats] 累計（%s）：%lld 個請求、prompt %lld 個 token、生成 %lld 個 token；%d 天的紀錄\n", LT_PATH, b.done, b.in, b.out, lt_nd);
}
// 每 30 秒存一次；SIGTERM / SIGINT 先存再照預設行為結束（launcher 用 SIGTERM 停 server）。沒有保留時什麼都不裝
static void lt_start(void) {
    if (!LT_PATH) return;
    dispatch_queue_t q = dispatch_queue_create("qwental.stats", DISPATCH_QUEUE_SERIAL);
    dispatch_source_t t = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, q);
    dispatch_source_set_timer(t, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC), 30 * NSEC_PER_SEC, 2 * NSEC_PER_SEC);
    dispatch_source_set_event_handler(t, ^{ lt_save(false); });
    dispatch_resume(t);
    const int sigs[2] = {SIGTERM, SIGINT};
    for (int i = 0; i < 2; i++) {
        const int sig = sigs[i];
        signal(sig, SIG_IGN); // 改由 dispatch source 接（I/O 不在 signal handler 裡做）
        dispatch_source_t s = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, (uintptr_t)sig, 0, q);
        dispatch_source_set_event_handler(s, ^{
            lt_save(true);
            fprintf(stderr, "[stats] 收到 %s：累計數字已存到 %s，結束\n", sig == SIGTERM ? "SIGTERM" : "SIGINT", LT_PATH);
            signal(sig, SIG_DFL); // 照預設行為結束（結束狀態同以前：被 sig 終止）
            sigset_t m;
            sigemptyset(&m);
            sigaddset(&m, sig);
            pthread_sigmask(SIG_UNBLOCK, &m, NULL); // dispatch 的執行緒可能擋著 signal
            raise(sig);
            usleep(200 * 1000); // macOS 的 raise 不一定在回傳前送達：等一下；還在就用 shell 慣例的 128 + sig 結束
            _exit(128 + sig);
        });
        dispatch_resume(s);
    }
}
static void handle_status(conn *c, const char *qs) { // qs：查詢字串（days=N：lifetime 加上最近 N 天的 bucket）
    sbuf o = {0};
    int queued = 0;
    pthread_mutex_lock(&q_mu);
    for (job *q = q_head; q; q = q->next) queued++;
    pthread_mutex_unlock(&q_mu);
    sb_printf(&o, "{\"version\":\"%s\",\"profile\":\"%s\",\"uptime_s\":%.0f,\"parallel\":%d,\"done\":%lld,\"prompt_tokens\":%lld,"
                  "\"cached_tokens\":%lld,\"output_tokens\":%lld,\"queued\":%d,\"cancelled\":%lld,\"errors\":%lld",
              QW_VERSION, getenv("QW_GGUF") ? "gguf" : "mlx", qw_now() - T_START, PAR, st_done, st_in, st_cached, st_out, queued, st_cancel, st_err);
    if (LT_PATH) { // 跨重啟的累計（含這次）；沒有保留時不送
        const char *dq = qs ? strstr(qs, "days=") : NULL;
        int nd = dq && (dq == qs || dq[-1] == '&') ? atoi(dq + 5) : 0;
        if (nd < 0) nd = 0;
        if (nd > LT_KEEP_DAYS + 1) nd = LT_KEEP_DAYS + 1;
        pthread_mutex_lock(&lt_mu);
        lt_up_flush(wall_now());
        const lt_stats l = lt_now();
        sb_puts(&o, ",\"lifetime\":");
        lt_json(&o, &l, nd);
        pthread_mutex_unlock(&lt_mu);
    }
    if (PAR > 1) {
        pthread_mutex_lock(&st_mu);
        sb_printf(&o, ",\"mode\":\"%s\",\"slot_mem_gb\":%.2f,\"slot_budget_gb\":%.1f,\"active\":[", st_n ? st_mode : "idle", st_mem / 1e9,
                  SLOT_BUDGET / 1e9);
        const double now = qw_now();
        for (int i = 0; i < st_n; i++) {
            sb_printf(&o, "%s{\"id\":%d,\"prompt\":%d,\"cached\":%d,\"prefilled\":%d,\"generated\":%d,\"state\":\"%s\",\"tps\":%.1f,\"age_s\":%.1f,\"model\":",
                      i ? "," : "", st_job[i].id, st_job[i].np, st_job[i].cached, st_job[i].prefilled, st_job[i].got,
                      st_job[i].state == 'p' ? "prefill" : st_job[i].state == 'b' ? "batch" : "solo", st_job[i].tps, now - st_job[i].t0);
            sb_json_str(&o, st_job[i].model, strlen(st_job[i].model), false);
            sb_puts(&o, "}");
        }
        pthread_mutex_unlock(&st_mu);
        sb_printf(&o, "],\"rate\":{\"spec\":%.1f", spec_rate);
        for (int n = 2; n <= QW_PAR_MAX; n++) if (bat_rate[n] > 0) sb_printf(&o, ",\"batch%d\":%.1f", n, bat_rate[n]);
        for (int n = 2; n <= QW_PAR_MAX; n++) if (rr_rate[n] > 0) sb_printf(&o, ",\"slice%d\":%.1f", n, rr_rate[n]);
        sb_printf(&o, "}");
    }
    sb_printf(&o, "}");
    respond(c, 200, "application/json", o.p, o.n);
    sb_free(&o);
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
    double mode_t = 0;              // 上一次換做法的時間（MODE_HOLD_S）
    int rr_n = 0, rr_tok = 0;       // 輪流推測解碼的牆鐘實測：目前在量幾個對話、累計 token 與時間
    double rr_ms = 0;
    for (;;) {
        for (int i = 0; i < nrejoin; i++) act[na++] = rejoin[i]; // 思考收尾後重新加入
        nrejoin = 0;
        { // 用戶端斷線（連線執行緒偵測到）：結束這個請求，GPU 不再為它算、slot 放出來
            int keep = 0;
            for (int i = 0; i < na; i++) {
                job *j = act[i];
                if (!conn_dead(j->c)) { act[keep++] = j; continue; }
                fprintf(stderr, "[par] 用戶端斷線：取消請求（prompt %d，%s，已生成 %d 個 token）\n", j->np,
                        j->pfing ? "分段 prefill 中" : j->pfy ? "prefill 讓出中" : j->inb ? "一起 decode 中" : "輪流中", j->got);
                par_done(j, QW_FIN_CALLBACK);
            }
            na = keep;
        }
        par_na = na;
        if (!na) st_publish(act, 0, "idle");
        pthread_mutex_lock(&q_mu);
        while (!na && !q_head) pthread_cond_wait(&q_cv, &q_mu);
        job *adm[QW_PAR_MAX];
        int nadm = 0;
        while (q_head && na + nadm < PAR) {
            adm[nadm++] = q_head;
            q_head = q_head->next;
            q_len--;
            if (!q_head) q_tail = NULL;
        }
        pthread_mutex_unlock(&q_mu);
        const double t_it = qw_now();
        double pf_ms = 0; // 這一輪花在 prefill 的時間（輪流速度的實測扣掉）
        for (int k = 0; k < nadm; k++) { // 新請求：選 slot
            job *j = adm[k];
            unsigned long long busy = 0;
            for (int i = 0; i < na; i++) busy |= 1ull << act[i]->sid;
            j->t0 = j->tg0 = qw_now();
            j->rng = j->gp.seed ? j->gp.seed : 0x1234567ull;
            if (conn_dead(j->c)) { par_done(j, QW_FIN_CALLBACK); continue; } // 排隊時就斷線了
            j->sid = qw_engine_par_pick(E, j->ids, j->np, busy);
            if (j->sid < 0) { par_fail(j, 529, "overloaded: no free KV slot"); continue; }
            if (j->gp.max_tokens <= 0) { par_done(j, QW_FIN_LENGTH); continue; }
            act[na++] = j;
        }
        par_na = na;
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
            const bool go = nready == 0 || (qw_now() - pf_last) * 1e3 >= SLICE_MS * 2 / 3;
            job *pj = NULL, *old = NULL; // 剩最少的先算（短的新請求不必等長的算完）；等超過 PF_AGE_S 秒的最舊的每隔一段輪到一次（不會一直被插隊）
            const double now = qw_now();
            for (int i = 0; go && i < na; i++) {
                if (!act[i]->pfing) continue;
                if (!pj || act[i]->np - act[i]->pfd < pj->np - pj->pfd) pj = act[i];
                if (now - act[i]->t0 > PF_AGE_S && (!old || act[i]->t0 < old->t0)) old = act[i];
            }
            static bool age_turn;
            if (old && old != pj && (age_turn = !age_turn)) pj = old;
            const bool ok = !pj || pf_chunk(pj);
            if (pj) { pf_last = qw_now(); pf_ms += (pf_last - now) * 1e3; }
            int keep = 0;
            for (int i = 0; i < na; i++)
                if (ok || act[i] != pj) act[keep++] = act[i];
            na = keep;
        }
        job *rd[QW_PAR_MAX]; // 可以生成的（不在分段 prefill 中）
        int nr = 0;
        for (int i = 0; i < na; i++) if (!act[i]->pfing) rd[nr++] = act[i];
        if (!nr) continue;
        bool s2 = want_slice(rd, nr, slice);
        // 換做法之後至少維持 MODE_HOLD_S 秒（只有一個對話、或超過 SLICE_MAX_N 個時照規則直接換）
        if (s2 != slice && nr > 1 && nr <= SLICE_MAX_N && qw_now() - mode_t < MODE_HOLD_S) s2 = slice;
        st_publish(act, na, s2 ? "spec" : "batch");
        if (s2 != slice) {
            mode_t = qw_now();
            if (nr > 1) fprintf(stderr, "[par] %d 個對話：改成%s\n", nr, s2 ? "輪流推測解碼" : "一起 decode");
        }
        slice = s2;
        if (!slice || nr != rr_n) { rr_n = slice ? nr : 0; rr_tok = 0; rr_ms = 0; }
        if (slice) { // 推測解碼：輪流，一次一個（有人在分段 prefill 時也切時間片）
            for (int i = 0; i < nr; i++) if (rd[i]->inb) leave_batch(rd[i]);
            job *j = rd[rr % nr];
            rr = rr % nr + 1;
            par_others = na - 1;
            par_pfwait = na > nr;
            int ng = 0;
            double spf = 0;
            const bool alive = run_solo(j, &ng, &spf);
            par_others = 0;
            par_pfwait = false;
            if (nr > 1) { // 輪流的總速度：牆鐘（含切換、快照、排程），扣掉 prefill；每量滿 1 秒更新一次
                rr_tok += ng;
                rr_ms += (qw_now() - t_it) * 1e3 - pf_ms - spf;
                if (rr_ms >= 1000) {
                    rr_rate[nr] = ema(rr_rate[nr], rr_tok * 1e3 / rr_ms, 0.3);
                    if (prof) fprintf(stderr, "[par] 輪流 %d 個：實測 %.1f tok/s（%d 個 token / %.0f ms）\n", nr, rr_tok * 1e3 / rr_ms, rr_tok, rr_ms);
                    rr_tok = 0;
                    rr_ms = 0;
                }
            }
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
        static int vs_on = -1;
        if (vs_on < 0) {
            vs_on = getenv("QW_PAR_VSPEC") && atoi(getenv("QW_PAR_VSPEC")); // 預設關：目前與一起 decode 差不多快（猜測、接受各多一次 GPU 往返）
            if (getenv("QW_PAR_VSMAX")) VS_MAX = atoi(getenv("QW_PAR_VSMAX"));
            if (getenv("QW_DF_CONF")) vs_conf = atof(getenv("QW_DF_CONF"));
        }
        // 一起推測解碼 vs 一起 decode：照實測選快的（差不到 5% 不換）；每 300 輪試另一種 8 輪，內容變了（例如改寫程式碼）也跟得上
        static int vs_cnt[QW_PAR_MAX + 1], vs_cur[QW_PAR_MAX + 1];
        bool use_vs = false;
        if (vs_on > 0 && nb >= 2 && nb <= VS_MAX) {
            const double a = vs_rate[nb], p2 = pl_rate[nb];
            const int c = vs_cnt[nb]++ % 300;
            if (a <= 0) use_vs = true;
            else if (p2 <= 0) use_vs = c >= 30 && c < 38 ? false : true;
            else {
                const bool best = vs_cur[nb] ? a > p2 * 0.95 : a > p2 * 1.05;
                use_vs = c >= 292 ? !best : best;
                if (c < 292) { if (best != (bool)vs_cur[nb]) fprintf(stderr, "[par] %d 個對話：改成%s（%.0f vs %.0f tok/s）\n", nb, best ? "一起推測解碼" : "一起 decode", a, p2); vs_cur[nb] = best; }
            }
        }
        if (use_vs) { // 一起推測解碼
            job *alive[8];
            const int no = vspec_round(bj, nb, alive);
            if (no >= 0) { for (int i = 0; i < no; i++) act[na++] = alive[i]; continue; }
            fprintf(stderr, "[par] 一起推測解碼不能用（不是 GGUF 或沒有 DFlash2）：改回一起 decode\n");
            vs_on = 0; // 失敗只會在第一次（engine 不支援）發生，狀態沒有動過
        }
        for (int i = 0; i < nb; i++) { sids[i] = bj[i]->sid; toks[i] = bj[i]->last; bj[i]->hasd = false; }
        qw_engine_par_keepdf(E, vs_on > 0 && nb >= 2 && nb <= VS_MAX);
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
        if (lg && dt > 0) {
            pl_rate[nb] = ema(pl_rate[nb], nb * 1e3 / dt, 0.1);
            bat_rate[nb] = vs_rate[nb] > pl_rate[nb] ? vs_rate[nb] : pl_rate[nb];
        }
        if (prof) { static int c = 0; if (++c % 50 == 0) fprintf(stderr, "[par] B=%d：par_step %.2f ms（GPU %.2f）、取樣 %.2f ms；一起 %.0f tok/s、推測 %.0f tok/s\n", nb, (ts1 - ts0) * 1e3, gms, dt - (ts1 - ts0) * 1e3, bat_rate[nb], spec_rate); }
        for (int i = 0; i < nb; i++) {
            job *j = bj[i];
            if (!lg) { par_fail(j, 500, "inference engine error (batch decode)"); continue; }
            if (j->np + j->got + 2 >= CTX) { par_done(j, QW_FIN_CTX); continue; }
            if (par_emit(j, nx[i])) act[na++] = j;
        }
    }
    return NULL;
}
// 排隊中就斷線了：直接從佇列拿掉（engine 不必碰它）。回傳 true = 拿掉了
static bool q_cancel(job *j) {
    bool got = false;
    pthread_mutex_lock(&q_mu);
    for (job **p = &q_head, *prev = NULL; *p; prev = *p, p = &(*p)->next)
        if (*p == j) {
            *p = j->next;
            if (q_tail == j) q_tail = prev;
            q_len--;
            got = true;
            break;
        }
    pthread_mutex_unlock(&q_mu);
    if (got) {
        j->fin = QW_FIN_CALLBACK;
        atomic_store(&j->done, true);
        fprintf(stderr, "[par] 用戶端斷線：取消排隊中的請求（prompt %d）\n", j->np);
    }
    return got;
}
static void par_chat(conn *c, const char *body, size_t blen, bool openai) {
    job *j = chat_begin(c, body, blen, openai);
    if (!j) return;
    j->sid = -1;
    pthread_mutex_lock(&q_mu);
    const bool full = q_len >= QUEUE_MAX;
    pthread_mutex_unlock(&q_mu);
    if (full) { // 還沒送任何東西：回 HTTP 529，Claude Code 會退避重試
        respond_err(c, 529, "overloaded: too many queued requests");
        fprintf(stderr, "[par] 排隊已滿（%d 個）：回 529\n", QUEUE_MAX);
        job_free(j);
        return;
    }
    stream_start(j);
    pthread_mutex_lock(&q_mu);
    if (q_tail) q_tail->next = j;
    else q_head = j;
    q_tail = j;
    q_len++;
    pthread_cond_signal(&q_cv);
    pthread_mutex_unlock(&q_mu);
    // 等 engine 做完：這段時間輸出在這裡寫出去（keepalive、斷線偵測也在這裡）
    while (!atomic_load(&j->done)) {
        pump_step(c, &j->done);
        if (conn_dead(c) && !atomic_load(&j->done) && q_cancel(j)) break;
    }
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
    log_prompt(&prompt);
    tok_lock(); // engine 執行緒同時在 decode（tokenizer 有共用的快取）
    const int np = qw_tok_encode(TK, prompt.p, prompt.n, NULL, 0);
    tok_unlock();
    char o[64];
    const int n = snprintf(o, sizeof o, "{\"input_tokens\":%d}", np);
    respond(c, 200, "application/json", o, (size_t)n);
    sb_free(&prompt);
    chat_free(&r);
    js_free(b);
}

// 一個連線：讀請求 → 處理 → 把輸出寫完 → 關掉。par：--parallel（生成交給 engine 執行緒），否則在這個執行緒生成
static void serve_conn(int fd, bool par) {
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    const struct timeval rt = {.tv_sec = RCVTIMEO_S}, st = {.tv_sec = SNDTIMEO_S};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rt, sizeof rt);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &st, sizeof st);
    conn c;
    conn_init(&c, fd);
    char method[16], path[1024], emsg[160];
    size_t blen = 0;
    int status = 0;
    char *body = read_req(&c, method, path, &blen, &status, emsg, sizeof emsg);
    if (body) {
        char *q = strchr(path, '?');
        if (q) *q = 0;
        const char *qs = q ? q + 1 : "";
        const char *p = path;
        if (!strncmp(p, "/v1/v1/", 7)) p += 3; // base URL 多寫了一次 /v1（同 Strata）
        const bool post = !strcmp(method, "POST"), get = !strcmp(method, "GET");
        if (post && !strcmp(p, "/v1/messages")) (par ? par_chat : handle_chat)(&c, body, blen, false);
        else if (post && !strcmp(p, "/v1/chat/completions")) (par ? par_chat : handle_chat)(&c, body, blen, true);
        else if (post && !strcmp(p, "/v1/messages/count_tokens")) handle_count(&c, body, blen);
        else if (get && (!strcmp(p, "/v1/models") || !strcmp(p, "/models"))) {
            const char *m = "{\"object\":\"list\",\"data\":[{\"id\":\"qwen3.8-flash-next\",\"object\":\"model\",\"type\":\"model\",\"display_name\":\"Qwen3.8 Flash Next (Qwental)\"}]}";
            respond(&c, 200, "application/json", m, strlen(m));
        } else if (!strcmp(p, "/health")) respond(&c, 200, "application/json", "{\"ok\":true}", 11);
        else if (get && (!strcmp(p, "/dashboard") || !strcmp(p, "/"))) respond(&c, 200, "text/html; charset=utf-8", dashboard_html, sizeof dashboard_html);
        else if (get && !strcmp(p, "/status")) handle_status(&c, qs);
        else {
            char m[1100];
            snprintf(m, sizeof m, "not found: %s %s", method, path);
            respond_err(&c, 404, m);
        }
        free(body);
    } else if (status) {
        fprintf(stderr, "[http] 請求有問題（%d）：%s\n", status, emsg);
        respond_err(&c, status, emsg);
    }
    pump_drain(&c);
    close(fd);
    conn_free(&c);
}

// --parallel：一個連線（執行緒）
static void *conn_main(void *arg) {
    const int fd = *(int *)arg;
    free(arg);
    serve_conn(fd, true);
    return NULL;
}

int main(int argc, char **argv) {
    int port = 8080, ctx = 131072;
    setenv("QW_SLOTS", "3", 0); // 預設 3 組 KV slot（主對話、權限判斷等不同前綴的請求各自快取）
    const char *host = "127.0.0.1", *cache_dir = "cache"; // 持久化前綴快取（SSD），"" = 不用
    const char *stats_file = NULL; // 累計數字的檔案：NULL = cache_dir/stats.json（cache_dir 是 "" 時不保留），"" = 不保留
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--version")) { printf("qwental %s\n", QW_VERSION); return 0; }
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("qwental %s: Anthropic / OpenAI compatible server for Qwen3.8-Flash-Next on Apple M5 Max\n\n"
                   "usage: qwental-serve [--gguf DIR] [--model DIR] [--port 8080] [--host 127.0.0.1] [--ctx 131072]\n"
                   "                     [--cache-dir cache] [--stats-file PATH] [--parallel [N]] [--slots 3] [--anthropic-thinking on_request|always]\n"
                   "                     [--int8] [--int8-qk] [--int8-hyper] [--fast-head] [--exact-spec] [--version]\n\n"
                   "  --gguf DIR      run the IQ3_S GGUF in DIR (DFlash2 pack in dflash2/ next to it)\n"
                   "  --model DIR     MLX weights (default models/Qwen3.8-Flash-Next-mixed-4-8bit)\n"
                   "  --cache-dir D   SSD prefix cache (\"\" = off); also holds stats.json (lifetime totals)\n"
                   "  --stats-file F  lifetime totals + per-day usage (default D/stats.json, \"\" = off); saved every 30 s and on SIGTERM/SIGINT\n"
                   "  --parallel N    concurrent requests (default 16, 1 = one at a time)\n"
                   "  --anthropic-thinking M  Anthropic requests that do not ask for thinking: on_request (default, no thinking) or always\n"
                   "  --int8          INT8 prefill (faster long prompts; outputs differ slightly)\n\n"
                   "endpoints: POST /v1/messages, /v1/messages/count_tokens, /v1/chat/completions; GET /v1/models, /health, /status[?days=N], /dashboard\n",
                   QW_VERSION);
            return 0;
        }
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
        else if (!strcmp(argv[i], "--stats-file")) stats_file = argv[i + 1];
        else if (!strcmp(argv[i], "--parallel")) PAR = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--anthropic-thinking")) { // Anthropic 請求沒提思考時：on_request（預設，不思考）/ always（思考）
            if (strcmp(argv[i + 1], "always") && strcmp(argv[i + 1], "on_request")) { fprintf(stderr, "❌ --anthropic-thinking 要是 on_request 或 always\n"); return 1; }
            chat_think_unasked = !strcmp(argv[i + 1], "always");
        }
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
    { // 累計數字（跨重啟）：--stats-file，或 cache_dir/stats.json
        char sp[1100];
        if (!stats_file) { if (*cache_dir) snprintf(sp, sizeof sp, "%s/stats.json", cache_dir); else sp[0] = 0; }
        else snprintf(sp, sizeof sp, "%s", stats_file);
        lt_init(*sp ? sp : NULL);
        lt_start();
    }
    char err[1024];
    TK = qw_tok_open_any(MODEL_DIR, getenv("QW_GGUF"), err, sizeof err); // 沒有 MLX 目錄時用 GGUF 的 tokenizer
    if (!TK) { fprintf(stderr, "❌ %s\n", err); return 1; }
    IM_END = qw_tok_special(TK, "<|im_end|>");
    EOT = qw_tok_special(TK, "<|endoftext|>");
    { // 思考收尾的 token（</think> 是特殊 token，encode 會認出來）
        const int nw = qw_tok_encode(TK, WRAP_TEXT, strlen(WRAP_TEXT), NULL, 0), nc = qw_tok_encode(TK, CLOSE_TEXT, strlen(CLOSE_TEXT), NULL, 0);
        if (nw > 0 && nw <= (int)(sizeof WRAP_IDS / sizeof *WRAP_IDS)) NWRAP = qw_tok_encode(TK, WRAP_TEXT, strlen(WRAP_TEXT), WRAP_IDS, nw);
        if (nc > 0 && nc <= (int)(sizeof CLOSE_IDS / sizeof *CLOSE_IDS)) NCLOSE = qw_tok_encode(TK, CLOSE_TEXT, strlen(CLOSE_TEXT), CLOSE_IDS, nc);
    }
    if (getenv("QW_PAR_SLICE_MAX")) SLICE_MAX_N = atoi(getenv("QW_PAR_SLICE_MAX"));
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
    T_START = qw_now();
    lt_serving();
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
            int *a2 = malloc(sizeof(int));
            if (!a2) { close(fd); continue; }
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
        serve_conn(fd, false);
    }
}
