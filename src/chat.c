#include "chat.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tok.h"

bool chat_think_unasked = false;

// ---------------------------------------------------------------- 基本
chat_msg *chat_add(chat_req *r, int role) {
    r->m = realloc(r->m, sizeof(chat_msg) * (size_t)(r->n + 1));
    chat_msg *m = &r->m[r->n++];
    memset(m, 0, sizeof *m);
    m->role = role;
    return m;
}

void chat_free(chat_req *r) {
    for (int i = 0; i < r->n; i++) {
        chat_msg *m = &r->m[i];
        sb_free(&m->content);
        sb_free(&m->reasoning);
        for (int k = 0; k < m->ncalls; k++) {
            free(m->calls[k].name);
            free(m->calls[k].id);
            js_free(m->calls[k].args);
        }
        free(m->calls);
        free(m->tool_id);
    }
    free(r->m);
    js_free(r->tools);
    memset(r, 0, sizeof *r);
}

static js *js_copy(const js *v) {
    sbuf b = {0};
    js_dump(&b, v, false);
    js *c = js_parse(b.p, b.n, NULL, 0);
    sb_free(&b);
    if (c) { free(c->key); c->key = NULL; }
    return c;
}

static void add_call(chat_msg *m, const char *name, js *args, const char *id) {
    m->calls = realloc(m->calls, sizeof(chat_call) * (size_t)(m->ncalls + 1));
    chat_call *c = &m->calls[m->ncalls++];
    c->name = strdup(name ? name : "");
    c->args = args ? args : js_parse("{}", 2, NULL, 0);
    c->id = strdup(id ? id : "");
}

// 呼叫的參數（客戶端的歷史）→ 物件：JSON 物件照用；空的是 {}；其他（截斷或壞掉的 JSON、陣列、數字）以文字放在 "arguments"（Strata #510）
static js *call_args(const js *raw) {
    if (!raw || raw->t == JS_NULL) return NULL;
    if (raw->t == JS_OBJ) return js_copy(raw);
    sbuf b = {0};
    js *v = NULL;
    if (raw->t == JS_STR) {
        size_t k = 0;
        while (k < raw->len && strchr(" \t\r\n", raw->s[k])) k++;
        if (k == raw->len) return NULL;
        v = js_parse(raw->s, raw->len, NULL, 0);
        if (v && v->t == JS_OBJ) return v;
        js_free(v);
        sb_puts(&b, "{\"arguments\":");
        sb_json_str(&b, raw->s, raw->len, false);
    } else {
        sbuf d = {0};
        js_dump(&d, raw, true);
        sb_puts(&b, "{\"arguments\":");
        sb_json_str(&b, d.p, d.n, false);
        sb_free(&d);
    }
    sb_puts(&b, "}");
    v = js_parse(b.p, b.n, NULL, 0);
    sb_free(&b);
    return v;
}

static const char *IMG_NOTE = "[image omitted: this model cannot see images]";
static const char *DOC_NOTE = "[document omitted: this model cannot read attached documents]";

// 內容：字串，或區塊陣列（text 區塊直接相接，同 template 的 list content；圖片以說明文字代替）
static void content_text(const js *c, sbuf *out) {
    if (!c) return;
    if (c->t == JS_STR) { sb_put(out, c->s, c->len); return; }
    if (c->t != JS_ARR) return;
    for (const js *b = c->kid; b; b = b->next) {
        const char *ty = js_str(b, "type"), *tx = js_str(b, "text");
        if (tx && (!ty || !strcmp(ty, "text") || !strcmp(ty, "input_text"))) sb_puts(out, tx);
        else if (ty && !strcmp(ty, "refusal") && js_str(b, "refusal")) sb_puts(out, js_str(b, "refusal"));
        else if (ty && (!strcmp(ty, "image") || !strcmp(ty, "image_url") || !strcmp(ty, "input_image"))) sb_puts(out, IMG_NOTE);
    }
}

// Anthropic 的 document 區塊：純文字來源就放內容，其他（PDF 等）放說明
static void document_text(const js *b, sbuf *out) {
    const js *src = js_get(b, "source");
    const char *st = js_str(src, "type"), *data = js_str(src, "data");
    if (st && !strcmp(st, "text") && data) { sb_puts(out, data); return; }
    if (st && !strcmp(st, "content")) {
        const js *c = js_get(src, "content");
        const size_t n0 = out->n;
        content_text(c, out);
        if (out->n > n0) return;
    }
    sb_puts(out, DOC_NOTE);
}

static void trim(const char *s, size_t n, const char **o, size_t *on) {
    while (n && strchr(" \t\n\r\v\f", s[0])) { s++; n--; }
    while (n && strchr(" \t\n\r\v\f", s[n - 1])) n--;
    *o = s;
    *on = n;
}

// ---------------------------------------------------------------- 思考強度
// Qwen 的 template 認得 low、medium、xhigh（預設；high 當 xhigh）與 enable_thinking=false。客戶端的各種寫法對到這四種。
// 回傳 1 = 開（*effort 設好）、0 = 關、-1 = 不認得
static int map_effort(const char *s, const char **effort) {
    if (!s) return -1;
    char k[16];
    size_t n = 0;
    while (*s == ' ') s++;
    for (; s[n] && n + 1 < sizeof k; n++) k[n] = (char)(s[n] >= 'A' && s[n] <= 'Z' ? s[n] + 32 : s[n]);
    while (n && k[n - 1] == ' ') n--;
    k[n] = 0;
    if (!strcmp(k, "none") || !strcmp(k, "off") || !strcmp(k, "minimal") || !strcmp(k, "disabled") || !strcmp(k, "false")) return 0;
    if (!strcmp(k, "low")) { *effort = "low"; return 1; }
    if (!strcmp(k, "medium")) { *effort = "medium"; return 1; }
    if (!strcmp(k, "high") || !strcmp(k, "xhigh") || !strcmp(k, "max") || !strcmp(k, "maximum")) { *effort = "xhigh"; return 1; }
    return -1;
}

// ---------------------------------------------------------------- Claude Code 的 billing header
// system prompt 開頭的 "x-anthropic-billing-header: cc_version=2.1.322.cbc; cc_entrypoint=sdk-cli; cch=b145e;"：
// cch 每個請求都變、版本的第四段每個 session 都變，prompt 的前綴快取就每次都斷。兩個都固定成 f（同 Strata pin_billing_stamp、
// llama.cpp #21793）；只動 system 最開頭的 header，而且只看前 160 個字元。
static void pin_billing(sbuf *s) {
    static const char H[] = "x-anthropic-billing-header:";
    if (s->n < sizeof H - 1 || memcmp(s->p, H, sizeof H - 1)) return;
    const size_t lim = s->n < 200 ? s->n : 200;
    for (size_t i = sizeof H - 1; i + 4 <= lim && i <= 160; i++)
        if (!memcmp(s->p + i, "cch=", 4)) {
            const size_t v = i + 4;
            size_t e = v;
            while (e < s->n && s->p[e] != ';') e++;
            if (e < s->n && e > v && e - v <= 16) memset(s->p + v, 'f', e - v);
            break;
        }
    for (size_t i = 0; i + 11 <= lim && i <= 160; i++)
        if (!memcmp(s->p + i, "cc_version=", 11)) {
            const size_t v = i + 11;
            size_t e = v;
            while (e < s->n && s->p[e] != ';') e++;
            if (e >= s->n || e - v > 64) break;
            int dots = 0;
            size_t tail = 0;
            for (size_t k = v; k < e; k++)
                if (s->p[k] == '.' && ++dots == 3) tail = k + 1;
            if (tail) memset(s->p + tail, 'f', e - tail);
            break;
        }
}

// ---------------------------------------------------------------- 工具順序
// 同一組工具，之後的請求都照第一次看到的順序排（Strata _stable_tool_order）：MCP 伺服器註冊完成的先後會變，
// 而 <tools> 在 system prompt 前面，換一次順序整段前綴快取就沒了。名稱重複或只有一個工具：照原樣。
#define ORDER_MAX 64
static struct { uint64_t key; int n; char **names; } orders[ORDER_MAX];
static int norders, order_next;
static pthread_mutex_t order_mu = PTHREAD_MUTEX_INITIALIZER;

static const char *tool_name(const js *t) {
    const js *f = js_get(t, "function");
    const char *n = js_str(f ? f : t, "name");
    return n ? n : "";
}

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static void stable_order(js *tools) {
    const int n = tools ? tools->n : 0;
    if (n < 2) return;
    const char **sorted = malloc(sizeof(char *) * (size_t)n);
    js **items = malloc(sizeof(js *) * (size_t)n);
    int k = 0;
    for (js *t = tools->kid; t && k < n; t = t->next) { items[k] = t; sorted[k++] = tool_name(t); }
    qsort(sorted, (size_t)k, sizeof(char *), cmp_str);
    uint64_t h = 1469598103934665603ull;
    bool dup = false;
    for (int i = 0; i < k; i++) {
        if (i && !strcmp(sorted[i], sorted[i - 1])) dup = true;
        for (const char *c = sorted[i]; *c; c++) h = (h ^ (uint8_t)*c) * 1099511628211ull;
        h = (h ^ 0xff) * 1099511628211ull;
    }
    free(sorted);
    if (dup || k != n) { free(items); return; }
    pthread_mutex_lock(&order_mu);
    int hit = -1;
    for (int i = 0; i < norders; i++)
        if (orders[i].key == h && orders[i].n == n) { hit = i; break; }
    if (hit < 0) {
        const int slot = norders < ORDER_MAX ? norders++ : (order_next++ % ORDER_MAX);
        for (int i = 0; i < orders[slot].n; i++) free(orders[slot].names[i]);
        free(orders[slot].names);
        orders[slot].key = h;
        orders[slot].n = n;
        orders[slot].names = malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) orders[slot].names[i] = strdup(tool_name(items[i]));
        pthread_mutex_unlock(&order_mu);
        free(items);
        return;
    }
    js **out = malloc(sizeof(js *) * (size_t)n);
    int m = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (items[j] && !strcmp(tool_name(items[j]), orders[hit].names[i])) { out[m++] = items[j]; items[j] = NULL; break; }
    pthread_mutex_unlock(&order_mu);
    if (m == n) {
        tools->kid = out[0];
        for (int i = 0; i < n; i++) out[i]->next = i + 1 < n ? out[i + 1] : NULL;
    }
    free(out);
    free(items);
}

// ---------------------------------------------------------------- system 訊息
// 開頭連續的 system 訊息：留著（chat_render 合併）；之後的：轉成 user（在原位置）
static int sys_role(const chat_req *r) {
    for (int i = 0; i < r->n; i++)
        if (r->m[i].role != CHAT_SYSTEM) return CHAT_USER;
    return CHAT_SYSTEM;
}

// ---------------------------------------------------------------- Anthropic → chat_req
static void anth_tool(sbuf *b, const js *t, bool deferred) {
    const char *name = js_str(t, "name"), *desc = js_str(t, "description");
    sb_puts(b, "{\"type\":\"function\",\"function\":{\"name\":");
    sb_json_str(b, name, strlen(name), false);
    if (desc) { sb_puts(b, ",\"description\":"); sb_json_str(b, desc, strlen(desc), false); }
    sb_puts(b, ",\"parameters\":");
    js_dump(b, js_get(t, "input_schema"), false);
    sb_puts(b, "}");
    if (deferred) sb_puts(b, ",\"defer_loading\":true");
    sb_puts(b, "}");
}

// tool_result 的內容：字串或區塊陣列（text；image → 說明；tool_reference → 延後載入的工具定義，同 Strata #4663）
static void tool_result_text(const js *c, const js *tools, sbuf *out) {
    if (!c) return;
    if (c->t != JS_ARR) { content_text(c, out); return; }
    for (const js *b = c->kid; b; b = b->next) {
        const char *ty = js_str(b, "type");
        if (ty && !strcmp(ty, "tool_reference")) {
            const char *tn = js_str(b, "tool_name");
            for (const js *t = tools ? tools->kid : NULL; t && tn; t = t->next)
                if (js_bool(t, "defer_loading", false) && !strcmp(tool_name(t), tn)) {
                    sb_puts(out, "<tools>\n");
                    js_dump(out, js_get(t, "function"), true);
                    sb_puts(out, "\n</tools>");
                    break;
                }
        } else if (ty && !strcmp(ty, "document")) document_text(b, out);
        else {
            js tmp = *b;
            tmp.next = NULL;
            js arr = {.t = JS_ARR, .kid = &tmp, .n = 1};
            content_text(&arr, out);
        }
    }
}

bool chat_from_anthropic(const js *body, chat_req *r, char *err, size_t errlen) {
    memset(r, 0, sizeof *r);
    r->effort = "xhigh";
    // 思考：thinking.type disabled → 關；output_config.effort → low / medium / xhigh（high、max 都是 xhigh）；
    // budget_tokens → < 2K low、< 8K medium、其他 xhigh；都沒有 → chat_think_unasked（預設不思考）
    const js *th = js_get(body, "thinking");
    const char *tt = js_str(th, "type");
    const char *eff = js_str(js_get(body, "output_config"), "effort");
    const double budget = th && (!tt || strcmp(tt, "disabled")) ? js_num(th, "budget_tokens", 0) : 0;
    r->thinking_asked = th || eff || budget > 0;
    if (tt && !strcmp(tt, "disabled")) r->thinking = false;
    else if (eff) {
        const int k = map_effort(eff, &r->effort);
        r->thinking = k != 0;
    } else if (budget > 0) {
        r->thinking = true;
        r->effort = budget < 2048 ? "low" : budget < 8192 ? "medium" : "xhigh";
    } else
        r->thinking = th ? true : chat_think_unasked;
    // 工具：{name, description, input_schema} → OpenAI 形式；defer_loading 的加上標記（不排進 <tools>）
    const js *tools = js_get(body, "tools");
    if (tools && tools->t == JS_ARR && tools->n > 0) {
        sbuf b = {0};
        sb_puts(&b, "[");
        int k = 0;
        for (const js *t = tools->kid; t; t = t->next) {
            const char *name = js_str(t, "name");
            const js *schema = js_get(t, "input_schema");
            if (!name || !*name || !schema) continue; // 伺服器端工具（web_search 等）略過
            const bool d = js_bool(t, "defer_loading", false);
            if (d) r->deferred = true;
            if (k++) sb_puts(&b, ",");
            anth_tool(&b, t, d);
        }
        sb_puts(&b, "]");
        if (k) r->tools = js_parse(b.p, b.n, NULL, 0);
        sb_free(&b);
        stable_order(r->tools);
    }
    const js *sys = js_get(body, "system");
    if (sys) {
        chat_msg *s = chat_add(r, CHAT_SYSTEM);
        content_text(sys, &s->content);
        pin_billing(&s->content);
    }
    const js *msgs = js_get(body, "messages");
    if (!msgs || msgs->t != JS_ARR) { snprintf(err, errlen, "messages 必須是陣列"); return false; }
    for (const js *m = msgs->kid; m; m = m->next) {
        const char *role = js_str(m, "role");
        const js *c = js_get(m, "content");
        if (!role) continue;
        if (!strcmp(role, "system") || !strcmp(role, "developer")) {
            // Claude Code 把環境、語言設定、日期、延後載入的工具清單放在對話中的 role:system 訊息
            content_text(c, &chat_add(r, sys_role(r))->content);
        } else if (!strcmp(role, "user")) {
            if (c && c->t == JS_STR) { sb_put(&chat_add(r, CHAT_USER)->content, c->s, c->len); continue; }
            sbuf text = {0};
            bool any = false; // 有 text 區塊（即使是空字串）
            for (const js *b = c ? c->kid : NULL; b; b = b->next) {
                const char *ty = js_str(b, "type");
                if (ty && !strcmp(ty, "tool_result")) {
                    if (any) { chat_msg *u = chat_add(r, CHAT_USER); u->content = text; text = (sbuf){0}; any = false; }
                    chat_msg *t = chat_add(r, CHAT_TOOL);
                    const char *id = js_str(b, "tool_use_id");
                    t->tool_id = strdup(id ? id : "");
                    if (js_bool(b, "is_error", false)) sb_puts(&t->content, "Error: ");
                    tool_result_text(js_get(b, "content"), r->tools, &t->content);
                } else if (ty && !strcmp(ty, "text")) {
                    const char *tx = js_str(b, "text");
                    if (tx) { sb_puts(&text, tx); any = true; }
                } else if (ty && !strcmp(ty, "image")) { sb_puts(&text, IMG_NOTE); any = true; }
                else if (ty && !strcmp(ty, "document")) { document_text(b, &text); any = true; }
            }
            if (any) { chat_msg *u = chat_add(r, CHAT_USER); u->content = text; }
            else sb_free(&text);
        } else if (!strcmp(role, "assistant")) {
            chat_msg *a = chat_add(r, CHAT_ASSISTANT);
            if (c && c->t == JS_STR) { sb_put(&a->content, c->s, c->len); continue; }
            for (const js *b = c ? c->kid : NULL; b; b = b->next) {
                const char *ty = js_str(b, "type");
                if (!ty) continue;
                if (!strcmp(ty, "thinking") && js_str(b, "thinking")) sb_puts(&a->reasoning, js_str(b, "thinking"));
                else if (!strcmp(ty, "text") && js_str(b, "text")) sb_puts(&a->content, js_str(b, "text"));
                else if (!strcmp(ty, "tool_use")) add_call(a, js_str(b, "name"), call_args(js_get(b, "input")), js_str(b, "id"));
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------- OpenAI → chat_req
bool chat_from_openai(const js *body, chat_req *r, char *err, size_t errlen) {
    memset(r, 0, sizeof *r);
    r->thinking = true;
    r->effort = "xhigh";
    // reasoning_effort（或 Responses 形式的 reasoning.effort）；false / "none" / "off" → 不思考
    const js *re = js_get(body, "reasoning_effort");
    if (!re || (re->t == JS_STR && !re->len)) re = js_get(js_get(body, "reasoning"), "effort");
    if (re && re->t == JS_FALSE) { r->thinking = false; r->thinking_asked = true; }
    else if (re && re->t == JS_STR && re->len) {
        const int k = map_effort(re->s, &r->effort);
        if (k == 0) r->thinking = false;
        r->thinking_asked = true;
    }
    const js *ctk = js_get(body, "chat_template_kwargs");
    if (ctk && js_get(ctk, "enable_thinking") && !js_bool(ctk, "enable_thinking", true)) { r->thinking = false; r->thinking_asked = true; }
    else if (ctk && js_str(ctk, "reasoning_effort")) {
        const int k = map_effort(js_str(ctk, "reasoning_effort"), &r->effort);
        if (k >= 0) r->thinking = k == 1;
        r->thinking_asked = true;
    }
    const js *msgs = js_get(body, "messages");
    if (!msgs || msgs->t != JS_ARR) { snprintf(err, errlen, "messages 必須是陣列"); return false; }
    for (const js *m = msgs->kid; m; m = m->next) {
        const char *role = js_str(m, "role");
        if (!role) continue;
        int ro = !strcmp(role, "system") || !strcmp(role, "developer") ? CHAT_SYSTEM : !strcmp(role, "user") ? CHAT_USER
               : !strcmp(role, "assistant") ? CHAT_ASSISTANT : !strcmp(role, "tool") ? CHAT_TOOL : -1;
        if (ro < 0) continue;
        if (ro == CHAT_SYSTEM) ro = sys_role(r);
        chat_msg *x = chat_add(r, ro);
        content_text(js_get(m, "content"), &x->content);
        const char *rc = js_str(m, "reasoning_content");
        if (!rc) rc = js_str(m, "reasoning");
        if (rc && ro == CHAT_ASSISTANT) sb_puts(&x->reasoning, rc);
        if (ro == CHAT_TOOL && js_str(m, "tool_call_id")) x->tool_id = strdup(js_str(m, "tool_call_id"));
        const js *tc = js_get(m, "tool_calls");
        for (const js *c = tc && tc->t == JS_ARR && ro == CHAT_ASSISTANT ? tc->kid : NULL; c; c = c->next) {
            const js *f = js_get(c, "function");
            if (!f) f = c;
            add_call(x, js_str(f, "name"), call_args(js_get(f, "arguments")), js_str(c, "id"));
        }
    }
    const js *tools = js_get(body, "tools");
    if (tools && tools->t == JS_ARR && tools->n > 0) {
        r->tools = js_copy(tools);
        stable_order(r->tools);
    }
    return true;
}

// ---------------------------------------------------------------- 特殊 token 字面
// 訊息內容裡的 "<|...|>"、<think>、</think>、<tts_…>（全部）與 <tool_call>、</tool_call>、<tool_response>、</tool_response>
// （assistant 自己的訊息除外：客戶端回放模型寫的原文時那是模型自己的 token，同 Strata #2026）前面插 QW_TOK_PLAIN。
enum { ESC_ALL, ESC_OWN };

static bool lit_at(const char *s, size_t n, const char *lit) {
    const size_t l = strlen(lit);
    return n >= l && !memcmp(s, lit, l);
}

static bool needs_mark(const char *s, size_t n, int mode) {
    if (n >= 4 && s[1] == '|') { // <|name|>
        size_t k = 2;
        while (k < n && (s[k] == '_' || (s[k] >= 'a' && s[k] <= 'z') || (s[k] >= 'A' && s[k] <= 'Z') || (s[k] >= '0' && s[k] <= '9'))) k++;
        return k > 2 && k + 1 < n && s[k] == '|' && s[k + 1] == '>';
    }
    if (lit_at(s, n, "<think>") || lit_at(s, n, "</think>") || lit_at(s, n, "<tts_")) return true;
    if (mode == ESC_ALL)
        return lit_at(s, n, "<tool_call>") || lit_at(s, n, "</tool_call>") || lit_at(s, n, "<tool_response>") || lit_at(s, n, "</tool_response>");
    return false;
}

static void put_esc(sbuf *o, const char *s, size_t n, int mode) {
    size_t st = 0;
    for (size_t i = 0; i < n; i++)
        if (s[i] == '<' && needs_mark(s + i, n - i, mode)) {
            sb_put(o, s + st, i - st);
            sb_puts(o, QW_TOK_PLAIN);
            st = i;
        }
    sb_put(o, s + st, n - st);
}

void chat_unmark(sbuf *s) {
    size_t o = 0;
    for (size_t i = 0; i < s->n;) {
        if (i + 4 <= s->n && !memcmp(s->p + i, QW_TOK_PLAIN, 4)) { i += 4; continue; }
        s->p[o++] = s->p[i++];
    }
    s->n = o;
    if (s->p) s->p[o] = 0;
}

// ---------------------------------------------------------------- 排出 prompt（chat_template.jinja）
static const char *TOOL_INSTR =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
    "specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required "
    "parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function "
    "call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n</IMPORTANT>";

static bool msg_text(const chat_msg *m, const char **c, size_t *cn) {
    trim(m->content.p ? m->content.p : "", m->content.n, c, cn);
    return *cn > 0;
}

void chat_render(const chat_req *r, sbuf *o) {
    const char *ri = "";
    if (r->thinking) {
        if (!r->effort || !strcmp(r->effort, "xhigh") || !strcmp(r->effort, "high"))
            ri = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible "
                 "alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
        else if (!strcmp(r->effort, "low"))
            ri = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
                 "unnecessary elaboration.";
    }
    // 開頭連續的 system 訊息：各自 trim，非空的以 "\n" 相接
    int num_sys = 0;
    sbuf sys = {0};
    for (; num_sys < r->n && r->m[num_sys].role == CHAT_SYSTEM; num_sys++) {
        const char *c;
        size_t cn;
        if (!msg_text(&r->m[num_sys], &c, &cn)) continue;
        if (sys.n) sb_puts(&sys, "\n");
        put_esc(&sys, c, cn, ESC_ALL);
    }
    int ntools = 0;
    for (const js *t = r->tools ? r->tools->kid : NULL; t; t = t->next)
        if (!js_bool(t, "defer_loading", false)) ntools++;
    if (ntools) {
        sb_puts(o, "<|im_start|>system\n");
        if (*ri) { sb_puts(o, ri); sb_puts(o, "\n\n"); }
        sb_puts(o, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
        sbuf tj = {0};
        for (const js *t = r->tools->kid; t; t = t->next) {
            if (js_bool(t, "defer_loading", false)) continue;
            sb_puts(o, "\n");
            tj.n = 0;
            js_dump(&tj, t, true);
            put_esc(o, tj.p, tj.n, ESC_ALL);
        }
        sb_free(&tj);
        sb_puts(o, "\n</tools>");
        sb_puts(o, TOOL_INSTR);
        if (sys.n) { sb_puts(o, "\n\n"); sb_put(o, sys.p, sys.n); }
        sb_puts(o, "<|im_end|>\n");
    } else if (sys.n) {
        sb_puts(o, "<|im_start|>system\n");
        if (*ri) { sb_puts(o, ri); sb_puts(o, "\n\n"); }
        sb_put(o, sys.p, sys.n);
        sb_puts(o, "<|im_end|>\n");
    } else if (*ri) {
        sb_puts(o, "<|im_start|>system\n");
        sb_puts(o, ri);
        sb_puts(o, "<|im_end|>\n");
    }
    sb_free(&sys);
    // 要排的訊息：完成的 assistant 回合若沒有文字也沒有 tool call 就不排（不當成下一次回覆的範例；最後一個訊息保留，Strata #843）
    int *idx = malloc(sizeof(int) * (size_t)(r->n + 1));
    int ni = 0;
    for (int i = num_sys; i < r->n; i++) {
        const chat_msg *m = &r->m[i];
        const char *c;
        size_t cn;
        if (m->role == CHAT_ASSISTANT && i + 1 < r->n && !m->ncalls && !msg_text(m, &c, &cn)) continue;
        idx[ni++] = i;
    }
    // preserve_thinking 預設為 true：每個 assistant 回合都帶思考（last_query_index 只在 preserve_thinking=false 時有用）
    for (int k = 0; k < ni; k++) {
        const chat_msg *m = &r->m[idx[k]];
        const int prev = k ? r->m[idx[k - 1]].role : num_sys ? CHAT_SYSTEM : -1;
        const int next = k + 1 < ni ? r->m[idx[k + 1]].role : -1;
        const char *c;
        size_t cn;
        msg_text(m, &c, &cn);
        if (m->role == CHAT_USER || m->role == CHAT_SYSTEM) { // 中途的 system 當 user
            sb_puts(o, "<|im_start|>user\n");
            put_esc(o, c, cn, ESC_ALL);
            sb_puts(o, "<|im_end|>\n");
        } else if (m->role == CHAT_ASSISTANT) {
            const char *rc;
            size_t rn;
            trim(m->reasoning.p ? m->reasoning.p : "", m->reasoning.n, &rc, &rn);
            sb_puts(o, "<|im_start|>assistant\n<think>\n");
            put_esc(o, rc, rn, ESC_OWN);
            sb_puts(o, "\n</think>\n\n");
            put_esc(o, c, cn, ESC_OWN);
            sbuf v = {0};
            for (int q = 0; q < m->ncalls; q++) {
                const chat_call *tc = &m->calls[q];
                if (q == 0) sb_puts(o, cn ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=");
                else sb_puts(o, "\n<tool_call>\n<function=");
                sb_puts(o, tc->name);
                sb_puts(o, ">\n");
                for (const js *a = tc->args && tc->args->t == JS_OBJ ? tc->args->kid : NULL; a; a = a->next) {
                    sb_puts(o, "<parameter=");
                    sb_puts(o, a->key);
                    sb_puts(o, ">\n");
                    if (a->t == JS_STR) put_esc(o, a->s, a->len, ESC_OWN);
                    else {
                        v.n = 0;
                        js_dump(&v, a, true);
                        put_esc(o, v.p, v.n, ESC_OWN);
                    }
                    sb_puts(o, "\n</parameter>\n");
                }
                sb_puts(o, "</function>\n</tool_call>");
            }
            sb_free(&v);
            sb_puts(o, "<|im_end|>\n");
        } else if (m->role == CHAT_TOOL) {
            if (prev >= 0 && prev != CHAT_TOOL) sb_puts(o, "<|im_start|>user");
            sb_puts(o, "\n<tool_response>\n");
            put_esc(o, c, cn, ESC_ALL);
            sb_puts(o, "\n</tool_response>");
            if (next != CHAT_TOOL) sb_puts(o, "<|im_end|>\n");
        }
    }
    free(idx);
    sb_puts(o, "<|im_start|>assistant\n");
    sb_puts(o, r->thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
}

// ================================================================ 輸出解析
// 依 Strata 的 OutputParser：tool call 只認宣告過的工具（有延後載入的工具時任何名稱都算）；參數值裡的 </parameter> 不會截斷
// （#210/#4701）；同一個 <tool_call> 裡多個 <function=>；沒有 <tool_call> 的行首 <function=NAME>；思考裡的 tool call
// 等思考結束或回合自然結束才算（#804）；code fence / inline code 裡的標籤是文字（#1058）。
static const char CALL_START[] = "<tool_call>", CALL_END[] = "</tool_call>", THINK_END[] = "</think>";
static const char FUNC_START[] = "<function=", FUNC_END[] = "</function>", PARAM_START[] = "<parameter=", PARAM_END[] = "</parameter>";
#define L(x) (sizeof(x) - 1)
#define CLOSE_LOOKAHEAD 1500
#define RCALL_MAX 32768

struct chat_pcall {
    sbuf raw;
    chat_call c;
    bool ok;
};

static bool isws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
static size_t skip_ws(const char *s, size_t n, size_t i) {
    while (i < n && isws(s[i])) i++;
    return i;
}
static bool starts(const char *s, size_t n, const char *lit, size_t l) { return n >= l && !memcmp(s, lit, l); }
// s 是 lit 的前綴（含空字串、不含整個 lit）
static bool partial(const char *s, size_t n, const char *lit, size_t l) { return n < l && !memcmp(s, lit, n); }
static long find(const char *s, size_t n, const char *lit, size_t l, size_t from) {
    if (from > n || n - from < l) return -1;
    const char *h = memmem(s + from, n - from, lit, l);
    return h ? h - s : -1;
}
static int count_in(const char *s, size_t n, const char *lit, size_t l) {
    int c = 0;
    for (long k = find(s, n, lit, l, 0); k >= 0; k = find(s, n, lit, l, (size_t)k + l)) c++;
    return c;
}

// 值裡引用的 call 開頭：`<function=NAME>` 後面（空白後）接 `<parameter=` 或 `</function>`（Strata _QUOTED_OPEN）
static int quoted_opens(const char *s, size_t n) {
    int c = 0;
    for (long k = find(s, n, FUNC_START, L(FUNC_START), 0); k >= 0;) {
        size_t j = (size_t)k + L(FUNC_START), j0 = j;
        while (j < n && s[j] != '<' && s[j] != '>' && !isws(s[j])) j++;
        if (j > j0 && j < n && s[j] == '>') {
            const size_t a = skip_ws(s, n, j + 1);
            if (starts(s + a, n - a, PARAM_START, L(PARAM_START)) || starts(s + a, n - a, FUNC_END, L(FUNC_END))) {
                c++;
                k = find(s, n, FUNC_START, L(FUNC_START), a);
                continue;
            }
        }
        k = find(s, n, FUNC_START, L(FUNC_START), (size_t)k + 1);
    }
    return c;
}

enum { V_END, V_CONTINUE, V_HOLD };
// a 以 </function> 和/或 </tool_call> 開頭（緊接在 </parameter> 後）：值是不是在這裡結束（Strata _close_verdict，#4701）
static int close_verdict(const char *a, size_t n, bool final) {
    size_t i = 0;
    if (starts(a, n, FUNC_END, L(FUNC_END))) i = L(FUNC_END);
    i = skip_ws(a, n, i);
    if (!starts(a + i, n - i, CALL_END, L(CALL_END))) return V_END;
    i = skip_ws(a, n, i + L(CALL_END));
    const char *r = a + i;
    const size_t rn = n - i;
    if (!rn) return final ? V_END : V_HOLD;
    if (starts(r, rn, CALL_START, L(CALL_START))) return V_END;
    if (!final && partial(r, rn, CALL_START, L(CALL_START))) return V_HOLD;
    if (find(r, rn, CALL_START, L(CALL_START), 0) >= 0) return V_END;
    if (find(r, rn, PARAM_END, L(PARAM_END), 0) >= 0 || find(r, rn, CALL_END, L(CALL_END), 0) >= 0 || find(r, rn, FUNC_END, L(FUNC_END), 0) >= 0)
        return V_CONTINUE;
    if (final || rn > CLOSE_LOOKAHEAD) return V_END;
    return V_HOLD;
}

// 參數值（s 從值開始）在哪裡結束：-1 還沒有、-2 有候選但後面還沒到（串流中）（Strata param_end）
static long param_end(const char *s, size_t n, bool final, const char *name, size_t namen, bool hold) {
    for (long at = find(s, n, PARAM_END, L(PARAM_END), 0); at >= 0; at = find(s, n, PARAM_END, L(PARAM_END), (size_t)at + 1)) {
        if (quoted_opens(s, (size_t)at) > count_in(s, (size_t)at, FUNC_END, L(FUNC_END))) continue;
        const size_t ai = skip_ws(s, n, (size_t)at + L(PARAM_END));
        const char *a = s + ai;
        const size_t an = n - ai;
        if (starts(a, an, PARAM_START, L(PARAM_START))) {
            const char *gt = memchr(a, '>', an);
            if (!gt) return final ? at : -2;
            const char *pn;
            size_t pl;
            trim(a + L(PARAM_START), (size_t)(gt - a) - L(PARAM_START), &pn, &pl);
            if (!name || pl != namen || memcmp(pn, name, pl)) return at;
        } else if (starts(a, an, FUNC_END, L(FUNC_END)) || starts(a, an, CALL_END, L(CALL_END))) {
            const int v = hold ? close_verdict(a, an, final) : V_END;
            if (v == V_END) return at;
            if (v == V_HOLD) return -2;
        } else if (!an || partial(a, an, PARAM_START, L(PARAM_START)) || partial(a, an, FUNC_END, L(FUNC_END)) ||
                   partial(a, an, CALL_END, L(CALL_END)))
            return final ? at : -2;
    }
    return -1;
}

// 走過 <function=NAME> 的參數：回傳下一個位置（</function> 或其他東西的開頭），-1 = 還沒完整
static long walk_params(const char *s, size_t n, size_t pos, bool final, bool hold) {
    for (;;) {
        pos = skip_ws(s, n, pos);
        if (!starts(s + pos, n - pos, PARAM_START, L(PARAM_START))) return (long)pos;
        const char *gt = memchr(s + pos, '>', n - pos);
        if (!gt) return -1;
        const char *pn;
        size_t pl;
        trim(s + pos + L(PARAM_START), (size_t)(gt - s - (long)pos) - L(PARAM_START), &pn, &pl);
        const size_t v = (size_t)(gt - s) + 1;
        const long e = param_end(s + v, n - v, final, pn, pl, hold);
        if (e < 0) return -1;
        pos = v + (size_t)e + L(PARAM_END);
    }
}

// s 開頭（空白後）的 <function=…>…</function> 在哪裡結束（</function> 之後），-1 = 不完整或不是這個格式（Strata function_end）
static long function_end(const char *s, size_t n) {
    size_t pos = skip_ws(s, n, 0);
    if (!starts(s + pos, n - pos, FUNC_START, L(FUNC_START))) return -1;
    const char *gt = memchr(s + pos, '>', n - pos);
    if (!gt) return -1;
    const long p = walk_params(s, n, (size_t)(gt - s) + 1, false, false);
    if (p < 0 || !starts(s + p, n - (size_t)p, FUNC_END, L(FUNC_END))) return -1;
    return p + (long)L(FUNC_END);
}

// tool call 的內容（<tool_call> 之後）在哪個 </tool_call> 結束（Strata call_end）；-1 = 還沒完整
static long call_end(const char *s, size_t n, bool final, bool hold) {
    size_t pos = skip_ws(s, n, 0);
    if (!starts(s + pos, n - pos, FUNC_START, L(FUNC_START)))
        return partial(s + pos, n - pos, FUNC_START, L(FUNC_START)) ? -1 : find(s, n, CALL_END, L(CALL_END), 0);
    const char *gt = memchr(s + pos, '>', n - pos);
    if (!gt) return -1;
    const long p = walk_params(s, n, (size_t)(gt - s) + 1, final, hold);
    if (p < 0) return -1;
    const char *a = s + p;
    const size_t an = n - (size_t)p;
    if (starts(a, an, FUNC_END, L(FUNC_END))) {
        const size_t q = skip_ws(s, n, (size_t)p + L(FUNC_END));
        if (starts(s + q, n - q, CALL_END, L(CALL_END))) return (long)q;
        return partial(s + q, n - q, CALL_END, L(CALL_END)) ? -1 : find(s, n, CALL_END, L(CALL_END), (size_t)p);
    }
    if (!an || partial(a, an, PARAM_START, L(PARAM_START)) || partial(a, an, FUNC_END, L(FUNC_END))) return -1;
    return find(s, n, CALL_END, L(CALL_END), (size_t)p);
}

// ---------------------------------------------------------------- 工具名稱與參數型別
static const js *find_tool(const js *tools, const char *name, size_t nl) {
    for (const js *t = tools ? tools->kid : NULL; t; t = t->next) {
        const char *tn = tool_name(t);
        if (strlen(tn) == nl && !memcmp(tn, name, nl)) return t;
    }
    return NULL;
}

// 模型寫的名稱 → 宣告的工具：完全相同；去掉 functions. / tools. / default_api. 前綴；`__` 被寫成 `_`（mcp_x_y → mcp__x__y，
// 只認唯一對得上的，Strata #3796）。沒有回傳 NULL
static const js *resolve_tool(const js *tools, const char *name, size_t nl) {
    const js *t = find_tool(tools, name, nl);
    if (t) return t;
    static const char *pre[] = {"functions.", "tools.", "default_api."};
    for (int i = 0; i < 3; i++) {
        const size_t pl = strlen(pre[i]);
        if (nl > pl && !memcmp(name, pre[i], pl) && (t = find_tool(tools, name + pl, nl - pl))) return t;
    }
    const js *hit = NULL;
    for (const js *u = tools ? tools->kid : NULL; u; u = u->next) {
        const char *tn = tool_name(u);
        if (!strstr(tn, "__")) continue;
        size_t k = 0, j = 0;
        bool same = true;
        while (tn[k] && same) {
            if (tn[k] == '_' && tn[k + 1] == '_') { same = j < nl && name[j] == '_'; k += 2; j++; }
            else { same = j < nl && name[j] == tn[k]; k++; j++; }
        }
        if (same && j == nl) {
            if (hit) return NULL; // 不唯一
            hit = u;
        }
    }
    return hit;
}

static bool callable(const chat_parser *p, const char *name, size_t nl, const js **tool) {
    *tool = p->has_tools ? resolve_tool(p->tools, name, nl) : NULL;
    if (*tool) return true;
    return p->has_tools && p->deferred && nl > 0;
}

enum { T_STR = 1, T_NUM = 2, T_BOOL = 4, T_NULL = 8, T_ARR = 16, T_OBJ = 32 };
static int type_bit(const char *s) {
    if (!strcmp(s, "string")) return T_STR;
    if (!strcmp(s, "number") || !strcmp(s, "integer")) return T_NUM;
    if (!strcmp(s, "boolean")) return T_BOOL;
    if (!strcmp(s, "null")) return T_NULL;
    if (!strcmp(s, "array")) return T_ARR;
    if (!strcmp(s, "object")) return T_OBJ;
    return 0;
}
static int schema_types(const js *sc) {
    if (!sc) return 0;
    const js *ty = js_get(sc, "type");
    int m = 0;
    if (ty && ty->t == JS_STR) return type_bit(ty->s);
    if (ty && ty->t == JS_ARR) {
        for (const js *x = ty->kid; x; x = x->next)
            if (x->t == JS_STR) m |= type_bit(x->s);
        return m;
    }
    const char *alts[] = {"anyOf", "oneOf"};
    for (int i = 0; i < 2; i++) {
        const js *a = js_get(sc, alts[i]);
        for (const js *x = a && a->t == JS_ARR ? a->kid : NULL; x; x = x->next) m |= schema_types(x);
    }
    return m;
}
static int val_bit(const js *v) {
    switch (v->t) {
    case JS_STR: return T_STR;
    case JS_NUM: return T_NUM;
    case JS_TRUE: case JS_FALSE: return T_BOOL;
    case JS_NULL: return T_NULL;
    case JS_ARR: return T_ARR;
    default: return T_OBJ;
    }
}

// 參數的文字 → JSON 值。型別只有 string：原文。型別宣告了且不含 string：能解析成 JSON 就用。
// 含 string 的多種型別（anyOf 等）或沒有型別：只有解析出允許的陣列／物件才用，其他（數字、布林）保留文字
static void put_value(sbuf *a, const char *v, size_t n, const js *tool, const char *key) {
    const js *props = js_get(js_get(js_get(tool, "function"), "parameters"), "properties");
    const int m = schema_types(js_get(props, key));
    if (m != T_STR) {
        js *val = js_parse(v, n, NULL, 0);
        if (val && val->t == JS_STR && (m & (T_ARR | T_OBJ)) && val->len && (val->s[0] == '[' || val->s[0] == '{')) {
            js *in = js_parse(val->s, val->len, NULL, 0); // 陣列／物件被寫成自己的 JSON 字串：再解一次
            if (in) { js_free(val); val = in; }
        }
        const int b = val ? val_bit(val) : 0;
        const bool ok = val && (m == 0 ? (b & (T_ARR | T_OBJ)) : (m & T_STR) ? (b & m & (T_ARR | T_OBJ)) : 1);
        if (ok) { js_dump(a, val, false); js_free(val); return; }
        js_free(val);
    }
    sb_json_str(a, v, n, false);
}

static void call_id(chat_parser *p, char *id, size_t n) {
    snprintf(id, n, "toolu_%08x%08x%04d", (unsigned)arc4random(), (unsigned)arc4random(), p->ncalls++);
}

// <function=NAME>…</function> → chat_call（呼叫端 free name、args）；不是這個格式回傳 false
static bool parse_call(const chat_parser *p, const char *s, size_t n, chat_call *c) {
    size_t pos = skip_ws(s, n, 0);
    if (!starts(s + pos, n - pos, FUNC_START, L(FUNC_START))) return false;
    const char *gt = memchr(s + pos, '>', n - pos);
    if (!gt) return false;
    const char *nm;
    size_t nl;
    trim(s + pos + L(FUNC_START), (size_t)(gt - s - (long)pos) - L(FUNC_START), &nm, &nl);
    const js *tool = NULL;
    callable(p, nm, nl, &tool);
    c->name = tool ? strdup(tool_name(tool)) : strndup(nm, nl);
    sbuf a = {0};
    sb_puts(&a, "{");
    int k = 0;
    size_t q = (size_t)(gt - s) + 1;
    for (long f; (f = find(s, n, PARAM_START, L(PARAM_START), q)) >= 0;) {
        const size_t ks = (size_t)f + L(PARAM_START);
        const char *ke = memchr(s + ks, '>', n - ks);
        if (!ke) break;
        const char *pn;
        size_t pl;
        trim(s + ks, (size_t)(ke - s) - ks, &pn, &pl);
        const size_t v0 = (size_t)(ke - s) + 1;
        const long e = param_end(s + v0, n - v0, true, pn, pl, true);
        size_t vs = v0, ve = e >= 0 ? v0 + (size_t)e : n;
        if (vs < ve && s[vs] == '\n') vs++;
        if (ve > vs && s[ve - 1] == '\n') ve--;
        char *key = strndup(pn, pl);
        if (k++) sb_puts(&a, ",");
        sb_json_str(&a, key, pl, false);
        sb_puts(&a, ":");
        put_value(&a, s + vs, ve - vs, tool, key);
        free(key);
        if (e < 0) break;
        q = v0 + (size_t)e + L(PARAM_END);
    }
    sb_puts(&a, "}");
    c->args = js_parse(a.p, a.n, NULL, 0);
    if (!c->args) c->args = js_parse("{}", 2, NULL, 0); // 不會發生（鍵值都經過跳脫）
    sb_free(&a);
    return true;
}

// ---------------------------------------------------------------- 程式碼區塊追蹤（行首的 tag 才算；fence 與 inline code 裡的是引用）
static void tr_reset(chat_track *t) {
    memset(t, 0, sizeof *t);
    t->blank = true;
}

static void tr_eol(chat_track *t) {
    if (t->fence_ch) {
        if (t->run_ch == t->fence_ch && t->run_len >= t->fence_len) t->fence_ch = 0;
    } else if (t->run_len >= 3 && !t->rest3) {
        t->fence_ch = t->run_ch;
        t->fence_len = t->run_len;
    } else if (t->blank) t->ticks = 0;
    else t->ticks += t->lticks;
    t->blank = true;
    t->lticks = t->run_ch = t->run_len = t->rest_seq = 0;
    t->in_run = t->rest3 = false;
}

static void tr_feed(chat_track *t, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const char c = s[i];
        t->last = c;
        if (c == '\n') { tr_eol(t); continue; }
        if (c == '`') t->lticks++;
        if (t->blank) {
            if (c == ' ' || c == '\t' || c == '\r') continue;
            t->blank = false;
            if (c == '`' || c == '~') { t->run_ch = c; t->run_len = 1; t->in_run = true; }
            continue;
        }
        if (t->in_run && c == t->run_ch) { t->run_len++; continue; }
        t->in_run = false;
        if (t->run_ch && c == t->run_ch) { if (++t->rest_seq >= 3) t->rest3 = true; }
        else t->rest_seq = 0;
    }
}

static bool tr_in_code(const chat_track *t) { return t->fence_ch || (t->ticks + t->lticks) % 2 == 1; }
static bool tr_opener_ok(const chat_track *t) { return !t->fence_ch && t->blank && t->ticks % 2 == 0; }

// ---------------------------------------------------------------- 解析器
void chat_parser_init(chat_parser *p, bool thinking, const js *tools) {
    memset(p, 0, sizeof *p);
    p->state = thinking ? 0 : 1;
    p->tools = tools;
    for (const js *t = tools ? tools->kid : NULL; t; t = t->next) {
        p->has_tools = true;
        if (js_bool(t, "defer_loading", false)) p->deferred = true;
    }
    tr_reset(&p->tr);
}

static void pc_clear(chat_parser *p) {
    for (int i = 0; i < p->npc; i++) {
        sb_free(&p->pc[i].raw);
        free(p->pc[i].c.name);
        js_free(p->pc[i].c.args);
    }
    free(p->pc);
    p->pc = NULL;
    p->npc = 0;
}

void chat_parser_free(chat_parser *p) {
    sb_free(&p->pend);
    pc_clear(p);
}

static void consume(chat_parser *p, size_t k) {
    p->ckd = p->ckn = 0;
    memmove(p->pend.p, p->pend.p + k, p->pend.n - k);
    p->pend.n -= k;
    if (p->pend.p) p->pend.p[p->pend.n] = 0;
}

static void emit_think(chat_parser *p, const char *s, size_t n, chat_ev_cb cb, void *ctx) {
    if (!n) return;
    tr_feed(&p->tr, s, n);
    cb(ctx, CEV_THINK, s, n, NULL);
}

static void emit_text(chat_parser *p, const char *s, size_t n, chat_ev_cb cb, void *ctx) {
    if (!p->text_started)
        while (n && isws(*s)) { s++; n--; }
    if (!n) return;
    p->text_started = true;
    tr_feed(&p->tr, s, n);
    cb(ctx, CEV_TEXT, s, n, NULL);
}

static void emit_call(chat_parser *p, const char *s, size_t n, chat_ev_cb cb, void *ctx) {
    chat_call c = {0};
    if (!parse_call(p, s, n, &c)) { emit_text(p, s, n, cb, ctx); return; } // 不會發生：進來前已確認 <function=NAME>
    char id[40];
    call_id(p, id, sizeof id);
    c.id = id;
    cb(ctx, CEV_CALL, NULL, 0, &c);
    free(c.name);
    js_free(c.args);
}

// 思考裡等待中的 call：deliver = 變成真的 tool call；否則全部當思考文字
static void release(chat_parser *p, bool deliver, chat_ev_cb cb, void *ctx) {
    chat_pcall *pc = p->pc;
    const int n = p->npc;
    p->pc = NULL;
    p->npc = 0;
    for (int i = 0; i < n; i++) {
        if (pc[i].ok && deliver) {
            char id[40];
            call_id(p, id, sizeof id);
            pc[i].c.id = id;
            cb(ctx, CEV_CALL, NULL, 0, &pc[i].c);
            pc[i].c.id = NULL;
            p->rescued++;
        } else
            emit_think(p, pc[i].raw.p, pc[i].raw.n, cb, ctx);
        sb_free(&pc[i].raw);
        free(pc[i].c.name);
        js_free(pc[i].c.args);
    }
    free(pc);
}

// call 狀態的快速路徑：上次看過之後新來的文字裡沒有任何結尾標籤，就不用整段重走（大檔案的 Write 不會變成 O(n²)）
static bool no_closer(chat_parser *p, const char *s, size_t n, bool think) {
    if (p->ckd < 0) { // 出現過結尾標籤：新文字裡有 '>'（標籤寫完了）或多了 256 byte 才重走
        if ((size_t)p->ckn <= n && n - (size_t)p->ckn < 256 && !memchr(s + p->ckn, '>', n - (size_t)p->ckn)) return true;
        p->ckn = (long)n;
        return false;
    }
    const size_t from = p->ckd > (long)L(CALL_END) ? (size_t)p->ckd - L(CALL_END) : 0;
    if (find(s, n, FUNC_END, L(FUNC_END), from) >= 0 || find(s, n, CALL_END, L(CALL_END), from) >= 0 ||
        (think && find(s, n, THINK_END, L(THINK_END), from) >= 0)) {
        p->ckd = -1;
        p->ckn = (long)n;
        return false;
    }
    p->ckd = (long)n;
    return true;
}

// s 最長的、是某個 tag 的真前綴的尾巴長度
static size_t hold_len(const char *s, size_t n, const char *const *tags, int nt) {
    size_t best = 0;
    for (int k = 0; k < nt; k++) {
        const size_t tl = strlen(tags[k]);
        for (size_t l = tl - 1; l > best; l--)
            if (n >= l && !memcmp(s + n - l, tags[k], l)) { best = l; break; }
    }
    return best;
}

static size_t rstrip_nl(const char *s, size_t j) {
    while (j > 0 && (s[j - 1] == '\n' || s[j - 1] == '\r')) j--;
    return j;
}

// 文字裡行首、code 外、宣告過的工具的 <function=NAME>（沒有 <tool_call>）。*decided = false：名稱還沒寫完
static long bare_opener(const chat_parser *p, const char *s, size_t n, bool *decided) {
    for (long k = find(s, n, FUNC_START, L(FUNC_START), 0); k >= 0; k = find(s, n, FUNC_START, L(FUNC_START), (size_t)k + 1)) {
        chat_track t = p->tr;
        tr_feed(&t, s, (size_t)k);
        if (!tr_opener_ok(&t)) continue;
        const size_t ns = (size_t)k + L(FUNC_START);
        const char *gt = memchr(s + ns, '>', n - ns);
        if (!gt) {
            bool clean = true;
            for (size_t j = ns; j < n; j++)
                if (s[j] == '<' || s[j] == '\n') clean = false;
            if (clean) { *decided = false; return k; }
            continue;
        }
        const char *nm;
        size_t nl;
        trim(s + ns, (size_t)(gt - s) - ns, &nm, &nl);
        if (resolve_tool(p->tools, nm, nl)) { *decided = true; return k; }
    }
    return -1;
}

// 答案裡行首的 </think>（思考已經結束過一次；Strata #4077）
static long stray_think(const chat_parser *p, const char *s, size_t n) {
    for (long k = find(s, n, THINK_END, L(THINK_END), 0); k >= 0; k = find(s, n, THINK_END, L(THINK_END), (size_t)k + 1)) {
        long j = k;
        while (j > 0 && s[j - 1] != '\n' && isws(s[j - 1])) j--;
        if (j == 0 ? p->tr.blank : s[j - 1] == '\n') return k;
    }
    return -1;
}

static void pump(chat_parser *p, chat_ev_cb cb, void *ctx, bool final, bool cut) {
    for (;;) {
        char *s = p->pend.p ? p->pend.p : "";
        const size_t n = p->pend.n;
        if (p->state == 0) { // ---- 思考
            if (p->npc) {
                const size_t k = skip_ws(s, n, 0);
                sb_put(&p->pc[p->npc - 1].raw, s, k);
                consume(p, k);
                s = p->pend.p;
                const size_t m = p->pend.n;
                if (!m) return;
                if (starts(s, m, CALL_START, L(CALL_START))) { p->state = 3; continue; }
                if (starts(s, m, THINK_END, L(THINK_END))) { release(p, true, cb, ctx); continue; }
                if (!final && (partial(s, m, CALL_START, L(CALL_START)) || partial(s, m, THINK_END, L(THINK_END)))) return;
                release(p, false, cb, ctx);
                continue;
            }
            const long i = find(s, n, THINK_END, L(THINK_END), 0);
            const long t = p->has_tools ? find(s, n, CALL_START, L(CALL_START), 0) : -1;
            if (t >= 0 && (i < 0 || t < i)) {
                emit_think(p, s, (size_t)t, cb, ctx);
                consume(p, (size_t)t);
                if (tr_opener_ok(&p->tr)) p->state = 3;
                else { emit_think(p, p->pend.p, L(CALL_START), cb, ctx); consume(p, L(CALL_START)); }
                continue;
            }
            if (i < 0) {
                static const char *const tg[] = {THINK_END, CALL_START};
                const size_t keep = final ? 0 : hold_len(s, n, tg, p->has_tools ? 2 : 1);
                emit_think(p, s, n - keep, cb, ctx);
                consume(p, n - keep);
                return;
            }
            // 引號或反引號後面、後面不是換行的 </think> 是在引用這個標籤（Strata #537）
            const char before = i ? s[i - 1] : p->tr.last;
            if (before == '"' || before == '\'' || before == '`') {
                const size_t after = (size_t)i + L(THINK_END);
                if (after == n && !final) {
                    emit_think(p, s, (size_t)i, cb, ctx);
                    consume(p, (size_t)i);
                    return;
                }
                if (after < n && s[after] != '\n' && s[after] != '\r') {
                    emit_think(p, s, after, cb, ctx);
                    consume(p, after);
                    continue;
                }
            }
            emit_think(p, s, (size_t)i, cb, ctx);
            consume(p, (size_t)i + L(THINK_END));
            p->state = 1;
            p->after_think = true;
            p->text_started = false;
            tr_reset(&p->tr);
        } else if (p->state == 3) { // ---- 思考裡的 <tool_call>：整段留著，結束了才決定
            const char *b = s + L(CALL_START);
            const size_t bn = n - L(CALL_START);
            if (!final && bn <= RCALL_MAX && no_closer(p, s, n, true)) return;
            long e = call_end(b, bn, false, false);
            const long th = find(b, bn, THINK_END, L(THINK_END), 0);
            size_t drop = L(CALL_END);
            const long f = function_end(b, bn);
            if (f >= 0 && (e < 0 || f < e)) {
                const size_t a = skip_ws(b, bn, (size_t)f);
                if (starts(b + a, bn - a, CALL_START, L(CALL_START))) { e = f; drop = 0; }
            }
            if (e >= 0 && (th < 0 || th >= e)) {
                const size_t ns = skip_ws(b, bn, 0) + L(FUNC_START);
                const char *gt = ns <= (size_t)e ? memchr(b + ns, '>', (size_t)e - ns) : NULL;
                chat_pcall pc = {0};
                if (gt) {
                    const char *nm;
                    size_t nl;
                    trim(b + ns, (size_t)(gt - b) - ns, &nm, &nl);
                    const js *tool;
                    if (callable(p, nm, nl, &tool)) pc.ok = parse_call(p, b, (size_t)e, &pc.c);
                }
                const size_t rawn = L(CALL_START) + (size_t)e + drop;
                if (pc.ok || p->npc) {
                    sb_put(&pc.raw, s, rawn);
                    p->pc = realloc(p->pc, sizeof(chat_pcall) * (size_t)(p->npc + 1));
                    p->pc[p->npc++] = pc;
                } else
                    emit_think(p, s, rawn, cb, ctx);
                consume(p, rawn);
                p->state = 0;
            } else if (th >= 0) { // 思考在裡面結束了：本來就不是 call
                release(p, false, cb, ctx);
                emit_think(p, s, L(CALL_START) + (size_t)th, cb, ctx);
                consume(p, L(CALL_START) + (size_t)th);
                p->state = 0;
            } else if (final || bn > RCALL_MAX) {
                bool kept = false;
                if (final && !cut && f >= 0 && skip_ws(b, bn, (size_t)f) == bn) { // </tool_call> 漏掉、輸出就在 </function> 結束
                    const size_t ns = skip_ws(b, bn, 0) + L(FUNC_START);
                    const char *gt = memchr(b + ns, '>', (size_t)f - ns);
                    chat_pcall pc = {0};
                    const js *tool;
                    const char *nm;
                    size_t nl;
                    if (gt) {
                        trim(b + ns, (size_t)(gt - b) - ns, &nm, &nl);
                        if (callable(p, nm, nl, &tool) && parse_call(p, b, (size_t)f, &pc.c)) {
                            pc.ok = true;
                            sb_put(&pc.raw, s, n);
                            p->pc = realloc(p->pc, sizeof(chat_pcall) * (size_t)(p->npc + 1));
                            p->pc[p->npc++] = pc;
                            consume(p, n);
                            kept = true;
                        }
                    }
                }
                if (!kept) {
                    release(p, false, cb, ctx);
                    emit_think(p, s, n, cb, ctx);
                    consume(p, n);
                }
                p->state = 0;
                return;
            } else
                return;
        } else if (p->state == 1) { // ---- 文字
            if (!p->text_started) {
                const size_t k = skip_ws(s, n, 0);
                consume(p, k);
                s = p->pend.p ? p->pend.p : "";
                if (!p->pend.n) return;
            }
            const size_t m = p->pend.n;
            if (p->drop_close) {
                const size_t k = skip_ws(s, m, 0);
                if (starts(s + k, m - k, CALL_END, L(CALL_END))) { consume(p, k + L(CALL_END)); p->drop_close = false; continue; }
                if (!final && partial(s + k, m - k, CALL_END, L(CALL_END))) return;
                p->drop_close = false;
            }
            const long i = find(s, m, CALL_START, L(CALL_START), 0);
            const long st = p->after_think ? stray_think(p, s, m) : -1;
            bool decided = true;
            const long bo = p->has_tools ? bare_opener(p, s, m, &decided) : -1;
            long best = -1;
            int kind = 0; // 1 <tool_call>、2 </think>、3 bare
            if (i >= 0) { best = i; kind = 1; }
            if (st >= 0 && (best < 0 || st < best)) { best = st; kind = 2; }
            if (bo >= 0 && (best < 0 || bo < best)) { best = bo; kind = 3; }
            if (kind == 2) {
                emit_text(p, s, (size_t)st, cb, ctx);
                consume(p, (size_t)st + L(THINK_END));
                p->text_started = false;
                tr_reset(&p->tr);
                continue;
            }
            if (kind == 3) {
                const size_t j = rstrip_nl(s, (size_t)bo);
                if (!decided) {
                    if (final) { emit_text(p, s, m, cb, ctx); consume(p, m); return; }
                    emit_text(p, s, j, cb, ctx);
                    consume(p, j);
                    return;
                }
                emit_text(p, s, j, cb, ctx);
                consume(p, (size_t)bo);
                p->state = 2;
                p->bare = true;
                continue;
            }
            if (kind == 0) {
                if (final) { emit_text(p, s, m, cb, ctx); consume(p, m); return; }
                static const char *const tg[] = {CALL_START, THINK_END};
                size_t j = m - hold_len(s, m, tg, p->after_think ? 2 : 1);
                if (p->has_tools) { // 正在變成行首 <function= 的最後一行
                    size_t k = m;
                    while (k > 0 && s[k - 1] != '\n') k--;
                    if (k < m && partial(s + k, m - k, FUNC_START, L(FUNC_START))) {
                        chat_track t = p->tr;
                        tr_feed(&t, s, k);
                        if (tr_opener_ok(&t) && k < j) j = k;
                    }
                }
                j = rstrip_nl(s, j);
                emit_text(p, s, j, cb, ctx);
                consume(p, j);
                return;
            }
            // <tool_call>：code 裡的是引用；後面（空白後）要接宣告過的工具的 <function=NAME> 才是 call
            chat_track t = p->tr;
            tr_feed(&t, s, (size_t)i);
            const size_t ai = skip_ws(s, m, (size_t)i + L(CALL_START));
            const char *a = s + ai;
            const size_t an = m - ai;
            bool text = tr_in_code(&t), wait = false;
            if (!text) {
                if (partial(a, an, FUNC_START, L(FUNC_START))) wait = true;
                else if (!starts(a, an, FUNC_START, L(FUNC_START))) text = true;
                else {
                    const char *gt = memchr(a, '>', an), *nl = memchr(a, '\n', an);
                    if (!gt && !nl) wait = true;
                    else if (!gt || (nl && nl < gt)) text = true;
                    else {
                        const char *nm;
                        size_t nn;
                        trim(a + L(FUNC_START), (size_t)(gt - a) - L(FUNC_START), &nm, &nn);
                        const js *tool;
                        if (!callable(p, nm, nn, &tool)) { text = true; p->unread++; }
                    }
                }
            }
            if (wait && final) text = true;
            if (wait && !text) {
                const size_t j = rstrip_nl(s, (size_t)i);
                emit_text(p, s, j, cb, ctx);
                consume(p, j);
                return;
            }
            if (text) {
                emit_text(p, s, (size_t)i + L(CALL_START), cb, ctx);
                consume(p, (size_t)i + L(CALL_START));
                continue;
            }
            emit_text(p, s, rstrip_nl(s, (size_t)i), cb, ctx);
            consume(p, (size_t)i + L(CALL_START));
            p->state = 2;
            p->bare = false;
        } else { // ---- tool call（pend 是 <tool_call> 之後的內容；bare 時從 <function= 開始）
            if (!final && no_closer(p, s, n, false)) return;
            const long f = function_end(s, n);
            if (p->bare) {
                if (f < 0) {
                    if (!final) return;
                    emit_text(p, s, n, cb, ctx);
                    consume(p, n);
                    p->cut_call = true;
                    p->state = 1;
                    return;
                }
                emit_call(p, s, (size_t)f, cb, ctx);
                consume(p, (size_t)f);
                p->state = 1;
                p->bare = false;
                p->drop_close = true;
                p->text_started = false;
                continue;
            }
            const long e = call_end(s, n, final, true);
            if (f >= 0 && (e < 0 || f < e)) {
                const size_t a = skip_ws(s, n, (size_t)f);
                if (starts(s + a, n - a, FUNC_START, L(FUNC_START))) { // 同一個 <tool_call> 裡的下一個 call
                    emit_call(p, s, (size_t)f, cb, ctx);
                    consume(p, a);
                    continue;
                }
                if (starts(s + a, n - a, CALL_START, L(CALL_START))) { // </tool_call> 漏掉，下一個 call 接著來
                    emit_call(p, s, (size_t)f, cb, ctx);
                    consume(p, a);
                    p->state = 1;
                    p->text_started = false;
                    continue;
                }
            }
            if (e >= 0) {
                emit_call(p, s, (size_t)e, cb, ctx);
                consume(p, (size_t)e + L(CALL_END));
                p->state = 1;
                p->text_started = false;
                continue;
            }
            if (!final) return;
            if (f >= 0 && !cut) { // 自然結束在 </function>（</tool_call> 漏掉）：還是 call
                emit_call(p, s, (size_t)f, cb, ctx);
                const size_t a = skip_ws(s, n, (size_t)f);
                if (a < n && !partial(s + a, n - a, CALL_END, L(CALL_END))) {
                    p->text_started = true;
                    emit_text(p, s + a, n - a, cb, ctx);
                }
            } else { // 沒寫完的 call：不變成 tool_use，原文當文字
                p->text_started = true;
                emit_text(p, CALL_START, L(CALL_START), cb, ctx);
                emit_text(p, s, n, cb, ctx);
                p->cut_call = true;
            }
            consume(p, n);
            p->state = 1;
            return;
        }
    }
}

void chat_parser_feed(chat_parser *p, const char *s, size_t n, chat_ev_cb cb, void *ctx) {
    sb_put(&p->pend, s, n);
    pump(p, cb, ctx, false, false);
}

void chat_parser_finish(chat_parser *p, bool cut, chat_ev_cb cb, void *ctx) {
    if (p->state == 0 && p->pend.n == L(THINK_END) && !memcmp(p->pend.p, THINK_END, L(THINK_END))) { consume(p, p->pend.n); p->state = 1; }
    pump(p, cb, ctx, true, cut);
    // 思考裡等著的 call：回合自然結束、思考也沒有別的文字接在後面才算數（被截斷就當思考文字，Strata #1058）
    if (p->npc) release(p, !cut && p->state == 0 && p->pend.n == 0, cb, ctx);
    if (p->pend.n) { // 理論上 pump(final) 已經清空
        if (p->state == 0 || p->state == 3) emit_think(p, p->pend.p, p->pend.n, cb, ctx);
        else { p->text_started = true; emit_text(p, p->pend.p, p->pend.n, cb, ctx); }
        consume(p, p->pend.n);
    }
}

void chat_parser_end(chat_parser *p, chat_ev_cb cb, void *ctx) { chat_parser_finish(p, false, cb, ctx); }
