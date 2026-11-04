#include "chat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// 內容：字串，或 [{type: text, text}] 區塊陣列（其他類型以說明文字代替）
static void content_text(const js *c, sbuf *out) {
    if (!c) return;
    if (c->t == JS_STR) { sb_put(out, c->s, c->len); return; }
    if (c->t != JS_ARR) return;
    for (const js *b = c->kid; b; b = b->next) {
        const char *ty = js_str(b, "type"), *tx = js_str(b, "text");
        if (tx && (!ty || !strcmp(ty, "text"))) sb_puts(out, tx);
        else if (ty && (!strcmp(ty, "image") || !strcmp(ty, "image_url"))) sb_puts(out, "[image not supported]");
    }
}

static void trim(const char *s, size_t n, const char **o, size_t *on) {
    while (n && strchr(" \t\n\r\v\f", s[0])) { s++; n--; }
    while (n && strchr(" \t\n\r\v\f", s[n - 1])) n--;
    *o = s;
    *on = n;
}

// ---------------------------------------------------------------- Anthropic → chat_req
bool chat_from_anthropic(const js *body, chat_req *r, char *err, size_t errlen) {
    memset(r, 0, sizeof *r);
    r->thinking = true;
    r->effort = "xhigh";
    const js *th = js_get(body, "thinking");
    if (th && js_str(th, "type") && !strcmp(js_str(th, "type"), "disabled")) r->thinking = false;
    // 思考強度：Anthropic 的 output_config.effort → Qwen 的 low / medium / xhigh（high 對應 medium：不加額外指示）
    const char *eff = js_str(js_get(body, "output_config"), "effort");
    if (eff) r->effort = !strcmp(eff, "low") ? "low" : (!strcmp(eff, "xhigh") || !strcmp(eff, "max")) ? "xhigh" : "medium";
    const js *sys = js_get(body, "system");
    if (sys) content_text(sys, &chat_add(r, CHAT_SYSTEM)->content);
    const js *msgs = js_get(body, "messages");
    if (!msgs || msgs->t != JS_ARR) { snprintf(err, errlen, "messages 必須是陣列"); return false; }
    for (const js *m = msgs->kid; m; m = m->next) {
        const char *role = js_str(m, "role");
        const js *c = js_get(m, "content");
        if (!role) continue;
        if (!strcmp(role, "user")) {
            if (c && c->t == JS_STR) { sb_put(&chat_add(r, CHAT_USER)->content, c->s, c->len); continue; }
            sbuf text = {0};
            for (const js *b = c ? c->kid : NULL; b; b = b->next) {
                const char *ty = js_str(b, "type");
                if (ty && !strcmp(ty, "tool_result")) {
                    if (text.n) { chat_msg *u = chat_add(r, CHAT_USER); u->content = text; text = (sbuf){0}; }
                    chat_msg *t = chat_add(r, CHAT_TOOL);
                    const char *id = js_str(b, "tool_use_id");
                    t->tool_id = strdup(id ? id : "");
                    if (js_bool(b, "is_error", false)) sb_puts(&t->content, "Error: ");
                    content_text(js_get(b, "content"), &t->content);
                } else if (ty && !strcmp(ty, "text")) {
                    const char *tx = js_str(b, "text");
                    if (tx) sb_puts(&text, tx);
                } else if (ty && (!strcmp(ty, "image") || !strcmp(ty, "document")))
                    sb_puts(&text, "[attachment not supported]");
            }
            if (text.n) { chat_msg *u = chat_add(r, CHAT_USER); u->content = text; }
            else sb_free(&text);
        } else if (!strcmp(role, "assistant")) {
            chat_msg *a = chat_add(r, CHAT_ASSISTANT);
            if (c && c->t == JS_STR) { sb_put(&a->content, c->s, c->len); continue; }
            for (const js *b = c ? c->kid : NULL; b; b = b->next) {
                const char *ty = js_str(b, "type");
                if (!ty) continue;
                if (!strcmp(ty, "thinking") && js_str(b, "thinking")) sb_puts(&a->reasoning, js_str(b, "thinking"));
                else if (!strcmp(ty, "text") && js_str(b, "text")) sb_puts(&a->content, js_str(b, "text"));
                else if (!strcmp(ty, "tool_use")) {
                    const js *in = js_get(b, "input");
                    add_call(a, js_str(b, "name"), in ? js_copy(in) : NULL, js_str(b, "id"));
                }
            }
        }
    }
    // 工具：{name, description, input_schema} → OpenAI 形式
    const js *tools = js_get(body, "tools");
    if (tools && tools->t == JS_ARR && tools->n > 0) {
        sbuf b = {0};
        sb_puts(&b, "[");
        int k = 0;
        for (const js *t = tools->kid; t; t = t->next) {
            const char *name = js_str(t, "name"), *desc = js_str(t, "description");
            const js *schema = js_get(t, "input_schema");
            if (!name || !schema) continue; // 伺服器端工具（web_search 等）略過
            if (k++) sb_puts(&b, ",");
            sb_puts(&b, "{\"type\":\"function\",\"function\":{\"name\":");
            sb_json_str(&b, name, strlen(name), false);
            if (desc) { sb_puts(&b, ",\"description\":"); sb_json_str(&b, desc, strlen(desc), false); }
            sb_puts(&b, ",\"parameters\":");
            js_dump(&b, schema, false);
            sb_puts(&b, "}}");
        }
        sb_puts(&b, "]");
        if (k) r->tools = js_parse(b.p, b.n, NULL, 0);
        sb_free(&b);
    }
    return true;
}

// ---------------------------------------------------------------- OpenAI → chat_req
bool chat_from_openai(const js *body, chat_req *r, char *err, size_t errlen) {
    memset(r, 0, sizeof *r);
    r->thinking = true;
    r->effort = "xhigh";
    const js *ctk = js_get(body, "chat_template_kwargs");
    if (ctk && js_get(ctk, "enable_thinking")) r->thinking = js_bool(ctk, "enable_thinking", true);
    const char *eff = js_str(body, "reasoning_effort");
    if (eff && (!strcmp(eff, "low") || !strcmp(eff, "medium"))) r->effort = !strcmp(eff, "low") ? "low" : "medium";
    const js *msgs = js_get(body, "messages");
    if (!msgs || msgs->t != JS_ARR) { snprintf(err, errlen, "messages 必須是陣列"); return false; }
    for (const js *m = msgs->kid; m; m = m->next) {
        const char *role = js_str(m, "role");
        if (!role) continue;
        const int ro = !strcmp(role, "system") || !strcmp(role, "developer") ? CHAT_SYSTEM : !strcmp(role, "user") ? CHAT_USER
                     : !strcmp(role, "assistant") ? CHAT_ASSISTANT : !strcmp(role, "tool") ? CHAT_TOOL : -1;
        if (ro < 0) continue;
        chat_msg *x = chat_add(r, ro);
        content_text(js_get(m, "content"), &x->content);
        const char *rc = js_str(m, "reasoning_content");
        if (!rc) rc = js_str(m, "reasoning");
        if (rc) sb_puts(&x->reasoning, rc);
        if (ro == CHAT_TOOL && js_str(m, "tool_call_id")) x->tool_id = strdup(js_str(m, "tool_call_id"));
        const js *tc = js_get(m, "tool_calls");
        for (const js *c = tc && tc->t == JS_ARR ? tc->kid : NULL; c; c = c->next) {
            const js *f = js_get(c, "function");
            const char *args = js_str(f, "arguments");
            js *a = args ? js_parse(args, strlen(args), NULL, 0) : NULL;
            if (!a && js_get(f, "arguments")) a = js_copy(js_get(f, "arguments"));
            add_call(x, js_str(f, "name"), a, js_str(c, "id"));
        }
    }
    const js *tools = js_get(body, "tools");
    if (tools && tools->t == JS_ARR && tools->n > 0) r->tools = js_copy(tools);
    return true;
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

void chat_render(const chat_req *r, sbuf *o) {
    const char *ri = "";
    if (r->thinking) {
        if (!r->effort || !strcmp(r->effort, "xhigh"))
            ri = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible "
                 "alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
        else if (!strcmp(r->effort, "low"))
            ri = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
                 "unnecessary elaboration.";
    }
    const bool has_sys = r->n > 0 && r->m[0].role == CHAT_SYSTEM;
    const char *sc = "";
    size_t scn = 0;
    if (has_sys) trim(r->m[0].content.p ? r->m[0].content.p : "", r->m[0].content.n, &sc, &scn);
    if (r->tools && r->tools->n > 0) {
        sb_puts(o, "<|im_start|>system\n");
        if (*ri) { sb_puts(o, ri); sb_puts(o, "\n\n"); }
        sb_puts(o, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (const js *t = r->tools->kid; t; t = t->next) {
            sb_puts(o, "\n");
            js_dump(o, t, true);
        }
        sb_puts(o, "\n</tools>");
        sb_puts(o, TOOL_INSTR);
        if (scn) { sb_puts(o, "\n\n"); sb_put(o, sc, scn); }
        sb_puts(o, "<|im_end|>\n");
    } else if (scn) {
        sb_puts(o, "<|im_start|>system\n");
        if (*ri) { sb_puts(o, ri); sb_puts(o, "\n\n"); }
        sb_put(o, sc, scn);
        sb_puts(o, "<|im_end|>\n");
    } else if (*ri) {
        sb_puts(o, "<|im_start|>system\n");
        sb_puts(o, ri);
        sb_puts(o, "<|im_end|>\n");
    }
    // 最後一個「真正的」user 訊息（不是只有 tool_response 的）
    int last_q = r->n - 1;
    for (int i = r->n - 1; i >= 0; i--)
        if (r->m[i].role == CHAT_USER) {
            const char *c; size_t cn;
            trim(r->m[i].content.p ? r->m[i].content.p : "", r->m[i].content.n, &c, &cn);
            const bool tr = cn >= 31 && !strncmp(c, "<tool_response>", 15) && !strncmp(c + cn - 16, "</tool_response>", 16);
            if (!tr) { last_q = i; break; }
        }
    for (int i = has_sys ? 1 : 0; i < r->n; i++) {
        const chat_msg *m = &r->m[i];
        const char *c; size_t cn;
        trim(m->content.p ? m->content.p : "", m->content.n, &c, &cn);
        if (m->role == CHAT_USER) {
            sb_puts(o, "<|im_start|>user\n");
            sb_put(o, c, cn);
            sb_puts(o, "<|im_end|>\n");
        } else if (m->role == CHAT_ASSISTANT) {
            const char *rc; size_t rn;
            trim(m->reasoning.p ? m->reasoning.p : "", m->reasoning.n, &rc, &rn);
            sb_puts(o, "<|im_start|>assistant\n<think>\n"); // preserve_thinking 預設為 true
            sb_put(o, rc, rn);
            sb_puts(o, "\n</think>\n\n");
            sb_put(o, c, cn);
            (void)last_q;
            for (int k = 0; k < m->ncalls; k++) {
                const chat_call *tc = &m->calls[k];
                if (k == 0) sb_puts(o, cn ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=");
                else sb_puts(o, "\n<tool_call>\n<function=");
                sb_puts(o, tc->name);
                sb_puts(o, ">\n");
                for (const js *a = tc->args && tc->args->t == JS_OBJ ? tc->args->kid : NULL; a; a = a->next) {
                    sb_puts(o, "<parameter=");
                    sb_puts(o, a->key);
                    sb_puts(o, ">\n");
                    if (a->t == JS_STR) sb_put(o, a->s, a->len);
                    else js_dump(o, a, true);
                    sb_puts(o, "\n</parameter>\n");
                }
                sb_puts(o, "</function>\n</tool_call>");
            }
            sb_puts(o, "<|im_end|>\n");
        } else if (m->role == CHAT_TOOL) {
            if (i > 0 && r->m[i - 1].role != CHAT_TOOL) sb_puts(o, "<|im_start|>user");
            sb_puts(o, "\n<tool_response>\n");
            sb_put(o, c, cn);
            sb_puts(o, "\n</tool_response>");
            if (i + 1 == r->n || r->m[i + 1].role != CHAT_TOOL) sb_puts(o, "<|im_end|>\n");
        }
    }
    sb_puts(o, "<|im_start|>assistant\n");
    sb_puts(o, r->thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
}

// ---------------------------------------------------------------- 輸出解析
void chat_parser_init(chat_parser *p, bool thinking, const js *tools) {
    memset(p, 0, sizeof *p);
    p->state = thinking ? 0 : 1;
    p->tools = tools;
}

void chat_parser_free(chat_parser *p) {
    sb_free(&p->pend);
    sb_free(&p->call);
}

// 參數的 schema 型別（找不到回傳 NULL）
static const char *param_type(const js *tools, const char *fn, const char *param) {
    for (const js *t = tools ? tools->kid : NULL; t; t = t->next) {
        const js *f = js_get(t, "function");
        const char *n = js_str(f, "name");
        if (!n || strcmp(n, fn)) continue;
        const js *props = js_get(js_get(f, "parameters"), "properties");
        const js *pp = js_get(props, param);
        const js *ty = js_get(pp, "type");
        if (ty && ty->t == JS_STR) return ty->s;
        if (ty && ty->t == JS_ARR && ty->kid && ty->kid->t == JS_STR) return ty->kid->s;
        return pp ? "" : NULL;
    }
    return NULL;
}

static void emit_call(chat_parser *p, const char *s, size_t n, chat_ev_cb cb, void *ctx) {
    char *t = malloc(n + 1);
    memcpy(t, s, n);
    t[n] = 0;
    const char *f = strstr(t, "<function=");
    if (!f) { free(t); return; }
    f += 10;
    const char *fe = strchr(f, '>');
    if (!fe) { free(t); return; }
    chat_call c = {0};
    c.name = strndup(f, (size_t)(fe - f));
    sbuf a = {0};
    sb_puts(&a, "{");
    int k = 0;
    for (const char *q = fe; (q = strstr(q, "<parameter=")) != NULL;) {
        q += 11;
        const char *ke = strchr(q, '>');
        if (!ke) break;
        char *key = strndup(q, (size_t)(ke - q));
        const char *v = ke + 1;
        if (*v == '\n') v++;
        const char *ve = strstr(v, "</parameter>");
        if (!ve) ve = v + strlen(v);
        const char *vend = ve;
        if (vend > v && vend[-1] == '\n') vend--;
        const char *ty = param_type(p->tools, c.name, key);
        js *val = NULL;
        const bool want_str = ty && !strcmp(ty, "string");
        if (!want_str) val = js_parse(v, (size_t)(vend - v), NULL, 0);
        if (k++) sb_puts(&a, ",");
        sb_json_str(&a, key, strlen(key), false);
        sb_puts(&a, ":");
        if (val) { js_dump(&a, val, false); js_free(val); }
        else sb_json_str(&a, v, (size_t)(vend - v), false);
        free(key);
        q = ve;
    }
    sb_puts(&a, "}");
    c.args = js_parse(a.p, a.n, NULL, 0);
    sb_free(&a);
    char id[40];
    snprintf(id, sizeof id, "toolu_%08x%08x%04d", (unsigned)arc4random(), (unsigned)arc4random(), p->ncalls++);
    c.id = id;
    cb(ctx, CEV_CALL, NULL, 0, &c);
    free(c.name);
    js_free(c.args);
    free(t);
}

// pend 裡找標籤；找到前的部分送出。保留可能是標籤開頭的尾巴。
static void pump(chat_parser *p, chat_ev_cb cb, void *ctx, bool final) {
    for (;;) {
        const char *tag = p->state == 0 ? "</think>" : p->state == 1 ? "<tool_call>" : "</tool_call>";
        const size_t tl = strlen(tag);
        char *s = p->pend.p ? p->pend.p : "";
        const size_t n = p->pend.n;
        char *hit = n ? strstr(s, tag) : NULL;
        size_t safe;
        if (hit) safe = (size_t)(hit - s);
        else if (final) safe = n;
        else { // 尾巴若是標籤的前綴就先留著
            safe = n;
            for (size_t k = tl - 1; k > 0; k--)
                if (n >= k && !memcmp(s + n - k, tag, k)) { safe = n - k; break; }
        }
        if (safe) {
            if (p->state == 0) cb(ctx, CEV_THINK, s, safe, NULL);
            else if (p->state == 1) {
                const char *t = s;
                size_t tn = safe;
                if (!p->text_started) { while (tn && strchr(" \t\r\n", *t)) { t++; tn--; } }
                if (tn) { p->text_started = true; cb(ctx, CEV_TEXT, t, tn, NULL); }
            } else
                sb_put(&p->call, s, safe);
        }
        if (!hit) {
            memmove(p->pend.p, s + safe, n - safe);
            p->pend.n = n - safe;
            if (p->pend.p) p->pend.p[p->pend.n] = 0;
            if (final && p->state == 2 && p->call.n) { emit_call(p, p->call.p, p->call.n, cb, ctx); p->call.n = 0; }
            return;
        }
        const size_t rest = n - safe - tl;
        memmove(p->pend.p, hit + tl, rest);
        p->pend.n = rest;
        p->pend.p[rest] = 0;
        if (p->state == 0) { p->state = 1; p->text_started = false; }
        else if (p->state == 1) { p->state = 2; p->call.n = 0; }
        else { emit_call(p, p->call.p ? p->call.p : "", p->call.n, cb, ctx); p->call.n = 0; p->state = 1; p->text_started = false; }
    }
}

void chat_parser_feed(chat_parser *p, const char *s, size_t n, chat_ev_cb cb, void *ctx) {
    sb_put(&p->pend, s, n);
    pump(p, cb, ctx, false);
}

void chat_parser_end(chat_parser *p, chat_ev_cb cb, void *ctx) { pump(p, cb, ctx, true); }
