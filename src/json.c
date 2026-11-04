#include "json.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p, *end;
    int depth;
} js_in;

static void ws(js_in *in) {
    while (in->p < in->end && (*in->p == ' ' || *in->p == '\t' || *in->p == '\n' || *in->p == '\r')) in->p++;
}

static int put_utf8(char *o, uint32_t c) {
    if (c < 0x80) { o[0] = (char)c; return 1; }
    if (c < 0x800) { o[0] = (char)(0xC0 | c >> 6); o[1] = (char)(0x80 | (c & 63)); return 2; }
    if (c < 0x10000) { o[0] = (char)(0xE0 | c >> 12); o[1] = (char)(0x80 | (c >> 6 & 63)); o[2] = (char)(0x80 | (c & 63)); return 3; }
    o[0] = (char)(0xF0 | c >> 18); o[1] = (char)(0x80 | (c >> 12 & 63)); o[2] = (char)(0x80 | (c >> 6 & 63)); o[3] = (char)(0x80 | (c & 63));
    return 4;
}

static int hexv(char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

static char *parse_str(js_in *in, size_t *len) {
    if (in->p >= in->end || *in->p != '"') return NULL;
    in->p++;
    const char *s = in->p;
    size_t cap = 16;
    while (s < in->end && *s != '"') { if (*s == '\\') s++; s++; cap++; }
    char *o = malloc(cap * 2 + 8);
    size_t n = 0;
    while (in->p < in->end && *in->p != '"') {
        char c = *in->p++;
        if (c != '\\') { o[n++] = c; continue; }
        if (in->p >= in->end) break;
        c = *in->p++;
        switch (c) {
        case 'n': o[n++] = '\n'; break;
        case 't': o[n++] = '\t'; break;
        case 'r': o[n++] = '\r'; break;
        case 'b': o[n++] = '\b'; break;
        case 'f': o[n++] = '\f'; break;
        case 'u': {
            if (in->end - in->p < 4) { free(o); return NULL; }
            uint32_t v = 0;
            for (int i = 0; i < 4; i++) { const int h = hexv(in->p[i]); if (h < 0) { free(o); return NULL; } v = v * 16 + (uint32_t)h; }
            in->p += 4;
            if (v >= 0xD800 && v < 0xDC00 && in->end - in->p >= 6 && in->p[0] == '\\' && in->p[1] == 'u') {
                uint32_t w = 0;
                for (int i = 0; i < 4; i++) w = w * 16 + (uint32_t)hexv(in->p[2 + i]);
                if (w >= 0xDC00 && w < 0xE000) { v = 0x10000 + ((v - 0xD800) << 10) + (w - 0xDC00); in->p += 6; }
            }
            n += (size_t)put_utf8(o + n, v);
            break;
        }
        default: o[n++] = c; break;
        }
    }
    if (in->p >= in->end) { free(o); return NULL; }
    in->p++;
    o[n] = 0;
    *len = n;
    return o;
}

static js *parse_val(js_in *in);

static js *newv(js_type t) {
    js *v = calloc(1, sizeof *v);
    v->t = t;
    return v;
}

static js *parse_val(js_in *in) {
    ws(in);
    if (in->p >= in->end || ++in->depth > 256) return NULL;
    js *v = NULL;
    const char c = *in->p;
    if (c == '{' || c == '[') {
        const bool obj = c == '{';
        v = newv(obj ? JS_OBJ : JS_ARR);
        in->p++;
        js **tail = &v->kid;
        ws(in);
        if (in->p < in->end && *in->p == (obj ? '}' : ']')) { in->p++; in->depth--; return v; }
        for (;;) {
            char *key = NULL;
            size_t kl;
            if (obj) {
                ws(in);
                key = parse_str(in, &kl);
                ws(in);
                if (!key || in->p >= in->end || *in->p != ':') { free(key); js_free(v); return NULL; }
                in->p++;
            }
            js *k = parse_val(in);
            if (!k) { free(key); js_free(v); return NULL; }
            k->key = key;
            *tail = k;
            tail = &k->next;
            v->n++;
            ws(in);
            if (in->p < in->end && *in->p == ',') { in->p++; continue; }
            if (in->p < in->end && *in->p == (obj ? '}' : ']')) { in->p++; break; }
            js_free(v);
            return NULL;
        }
    } else if (c == '"') {
        v = newv(JS_STR);
        v->s = parse_str(in, &v->len);
        if (!v->s) { free(v); return NULL; }
    } else if (!strncmp(in->p, "true", 4)) { v = newv(JS_TRUE); in->p += 4; }
    else if (!strncmp(in->p, "false", 5)) { v = newv(JS_FALSE); in->p += 5; }
    else if (!strncmp(in->p, "null", 4)) { v = newv(JS_NULL); in->p += 4; }
    else if (c == '-' || (c >= '0' && c <= '9')) {
        const char *s = in->p;
        while (in->p < in->end && strchr("+-0123456789.eE", *in->p)) in->p++;
        v = newv(JS_NUM);
        v->len = (size_t)(in->p - s);
        v->s = malloc(v->len + 1);
        memcpy(v->s, s, v->len);
        v->s[v->len] = 0;
        v->num = strtod(v->s, NULL);
    }
    in->depth--;
    return v;
}

js *js_parse(const char *text, size_t len, char *err, size_t errlen) {
    js_in in = {text, text + len, 0};
    js *v = parse_val(&in);
    ws(&in);
    if (!v || in.p != in.end) {
        if (err) snprintf(err, errlen, "JSON 解析失敗（位置 %ld）", (long)(in.p - text));
        js_free(v);
        return NULL;
    }
    return v;
}

void js_free(js *v) {
    while (v) {
        js *nx = v->next;
        js_free(v->kid);
        free(v->key);
        free(v->s);
        free(v);
        v = nx;
    }
}

js *js_get(const js *o, const char *key) {
    if (!o || o->t != JS_OBJ) return NULL;
    for (js *k = o->kid; k; k = k->next)
        if (!strcmp(k->key, key)) return k;
    return NULL;
}

const char *js_str(const js *o, const char *key) {
    const js *v = js_get(o, key);
    return v && v->t == JS_STR ? v->s : NULL;
}

double js_num(const js *o, const char *key, double def) {
    const js *v = js_get(o, key);
    return v && v->t == JS_NUM ? v->num : def;
}

bool js_bool(const js *o, const char *key, bool def) {
    const js *v = js_get(o, key);
    return v ? v->t == JS_TRUE : def;
}

// ---------------------------------------------------------------- 輸出
void sb_put(sbuf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t c = b->cap ? b->cap : 256;
        while (c < b->n + n + 1) c *= 2;
        b->p = realloc(b->p, c);
        b->cap = c;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

void sb_puts(sbuf *b, const char *s) { sb_put(b, s, strlen(s)); }

void sb_printf(sbuf *b, const char *fmt, ...) {
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < (int)sizeof tmp) { sb_put(b, tmp, (size_t)n); return; }
    char *big = malloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    sb_put(b, big, (size_t)n);
    free(big);
}

void sb_free(sbuf *b) {
    free(b->p);
    *b = (sbuf){0};
}

void sb_json_str(sbuf *b, const char *s, size_t n, bool ascii) {
    sb_put(b, "\"", 1);
    for (size_t i = 0; i < n;) {
        const unsigned char c = (unsigned char)s[i];
        char e[16];
        if (c == '"') { sb_put(b, "\\\"", 2); i++; }
        else if (c == '\\') { sb_put(b, "\\\\", 2); i++; }
        else if (c == '\n') { sb_put(b, "\\n", 2); i++; }
        else if (c == '\r') { sb_put(b, "\\r", 2); i++; }
        else if (c == '\t') { sb_put(b, "\\t", 2); i++; }
        else if (c == '\b') { sb_put(b, "\\b", 2); i++; }
        else if (c == '\f') { sb_put(b, "\\f", 2); i++; }
        else if (c < 0x20 || c == 0x7F) { snprintf(e, sizeof e, "\\u%04x", c); sb_put(b, e, 6); i++; }
        else if (c < 0x80 || !ascii) { sb_put(b, s + i, 1); i++; }
        else { // 非 ASCII → \uXXXX（含 surrogate pair）
            int k = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            if (i + (size_t)k > n) k = 1;
            uint32_t v = k == 1 ? 0xFFFD : (uint32_t)(c & (0x7F >> k));
            for (int j = 1; j < k; j++) v = v << 6 | ((unsigned char)s[i + j] & 63);
            if (v >= 0x10000) {
                v -= 0x10000;
                snprintf(e, sizeof e, "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
                sb_put(b, e, 12);
            } else {
                snprintf(e, sizeof e, "\\u%04x", v);
                sb_put(b, e, 6);
            }
            i += (size_t)k;
        }
    }
    sb_put(b, "\"", 1);
}

void js_dump(sbuf *b, const js *v, bool py) {
    switch (v->t) {
    case JS_NULL: sb_puts(b, "null"); break;
    case JS_TRUE: sb_puts(b, "true"); break;
    case JS_FALSE: sb_puts(b, "false"); break;
    case JS_NUM: sb_put(b, v->s, v->len); break;
    case JS_STR: sb_json_str(b, v->s, v->len, false); break;
    case JS_ARR:
    case JS_OBJ: {
        const bool o = v->t == JS_OBJ;
        sb_puts(b, o ? "{" : "[");
        for (js *k = v->kid; k; k = k->next) {
            if (o) {
                sb_json_str(b, k->key, strlen(k->key), false);
                sb_puts(b, py ? ": " : ":");
            }
            js_dump(b, k, py);
            if (k->next) sb_puts(b, py ? ", " : ",");
        }
        sb_puts(b, o ? "}" : "]");
        break;
    }
    }
}
