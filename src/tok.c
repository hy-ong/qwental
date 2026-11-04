// Qwen byte-level BPE tokenizer。
// 流程：特殊 token 整段比對 → 正規表示式切詞（手寫，等同 tokenizer.json 的 pattern）→ 每段 bytes 轉 byte-level 字元 → BPE 合併
#include "tok.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "unicode_tab.h"

typedef struct {
    uint64_t key; // (左 id << 32) | 右 id，0 = 空
    int32_t rank, id;
} merge_ent;

typedef struct {
    char *s;
    int len;
    int32_t id;
} special;

struct qw_tok {
    int nvocab;
    char **str;    // id → byte-level 字串（UTF-8）
    int *slen;
    // 字串 → id：開放定址
    int32_t *sh;
    size_t shcap;
    merge_ent *mh;
    size_t mhcap;
    int32_t byte_id[256]; // 單一 byte 的 token id
    uint8_t ch2byte[512]; // byte-level 字元（碼位 < 512）→ byte
    special sp[64];
    int nsp;
};

// ---------------------------------------------------------------- 雜湊
static uint64_t fnv(const char *s, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 1099511628211ull;
    return h;
}

static int32_t str_find(const qw_tok *t, const char *s, int n) {
    size_t i = fnv(s, n) & (t->shcap - 1);
    for (;; i = (i + 1) & (t->shcap - 1)) {
        const int32_t id = t->sh[i];
        if (id < 0) return -1;
        if (t->slen[id] == n && !memcmp(t->str[id], s, (size_t)n)) return id;
    }
}

static void str_put(qw_tok *t, int32_t id) {
    size_t i = fnv(t->str[id], t->slen[id]) & (t->shcap - 1);
    while (t->sh[i] >= 0) i = (i + 1) & (t->shcap - 1);
    t->sh[i] = id;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return x;
}

static const merge_ent *merge_find(const qw_tok *t, int32_t a, int32_t b) {
    const uint64_t key = ((uint64_t)(uint32_t)a << 32 | (uint32_t)b) + 1;
    size_t i = mix64(key) & (t->mhcap - 1);
    for (;; i = (i + 1) & (t->mhcap - 1)) {
        if (t->mh[i].key == 0) return NULL;
        if (t->mh[i].key == key) return &t->mh[i];
    }
}

// ---------------------------------------------------------------- 讀檔
static char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *p = malloc((size_t)n + 1);
    if (p && fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); p = NULL; }
    fclose(f);
    if (p) { p[n] = 0; if (len) *len = (size_t)n; }
    return p;
}

static int utf8_put(char *o, uint32_t c) {
    if (c < 0x80) { o[0] = (char)c; return 1; }
    if (c < 0x800) { o[0] = (char)(0xC0 | c >> 6); o[1] = (char)(0x80 | (c & 63)); return 2; }
    if (c < 0x10000) { o[0] = (char)(0xE0 | c >> 12); o[1] = (char)(0x80 | (c >> 6 & 63)); o[2] = (char)(0x80 | (c & 63)); return 3; }
    o[0] = (char)(0xF0 | c >> 18); o[1] = (char)(0x80 | (c >> 12 & 63)); o[2] = (char)(0x80 | (c >> 6 & 63)); o[3] = (char)(0x80 | (c & 63));
    return 4;
}

static uint32_t hex4(const char *p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        const char c = p[i];
        v = v * 16 + (uint32_t)(c >= 'a' ? c - 'a' + 10 : c >= 'A' ? c - 'A' + 10 : c - '0');
    }
    return v;
}

// 解析 JSON 字串（p 指向開頭的 "），寫到 o，回傳結尾 " 之後的位置
static const char *json_str(const char *p, char *o, int *olen) {
    int n = 0;
    p++;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
            case 'n': o[n++] = '\n'; p++; break;
            case 't': o[n++] = '\t'; p++; break;
            case 'r': o[n++] = '\r'; p++; break;
            case 'b': o[n++] = '\b'; p++; break;
            case 'f': o[n++] = '\f'; p++; break;
            case 'u': {
                uint32_t c = hex4(p + 1);
                p += 5;
                if (c >= 0xD800 && c < 0xDC00 && p[0] == '\\' && p[1] == 'u') {
                    c = 0x10000 + ((c - 0xD800) << 10) + (hex4(p + 2) - 0xDC00);
                    p += 6;
                }
                n += utf8_put(o + n, c);
                break;
            }
            default: o[n++] = *p++; break;
            }
        } else
            o[n++] = *p++;
    }
    *olen = n;
    return *p ? p + 1 : p;
}

// GPT-2 的 bytes_to_unicode：可見的 byte 對應自己，其他依序對應 256 起的碼位
static void byte_map(uint32_t *b2c) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        const bool vis = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        b2c[b] = vis ? (uint32_t)b : (uint32_t)(256 + n++);
    }
}

qw_tok *qw_tok_open(const char *dir, char *err, size_t errlen) {
    char path[1024];
    qw_tok *t = calloc(1, sizeof *t);
    // vocab.json
    snprintf(path, sizeof path, "%s/vocab.json", dir);
    size_t vlen;
    char *vj = slurp(path, &vlen);
    if (!vj) { snprintf(err, errlen, "讀不到 %s", path); free(t); return NULL; }
    int cap = 1 << 18;
    t->str = calloc((size_t)cap, sizeof(char *));
    t->slen = calloc((size_t)cap, sizeof(int));
    char buf[1024];
    for (const char *p = vj; *p;) {
        while (*p && *p != '"') p++;
        if (!*p) break;
        int n;
        p = json_str(p, buf, &n);
        while (*p == ' ' || *p == ':') p++;
        const long id = strtol(p, (char **)&p, 10);
        if (id < 0 || id >= cap) continue;
        t->str[id] = malloc((size_t)n + 1);
        memcpy(t->str[id], buf, (size_t)n);
        t->str[id][n] = 0;
        t->slen[id] = n;
        if (id + 1 > t->nvocab) t->nvocab = (int)id + 1;
    }
    free(vj);
    // 特殊 token（tokenizer.json 的 added_tokens）
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    char *tj = slurp(path, NULL);
    if (!tj) { snprintf(err, errlen, "讀不到 %s", path); qw_tok_close(t); return NULL; }
    const char *at = strstr(tj, "\"added_tokens\"");
    const char *end = at ? strchr(at, ']') : NULL;
    for (const char *p = at; p && p < end;) {
        const char *idp = strstr(p, "\"id\"");
        if (!idp || idp > end) break;
        const long id = strtol(strchr(idp, ':') + 1, NULL, 10);
        const char *cp = strstr(idp, "\"content\"");
        cp = strchr(cp + 9, '"');
        int n;
        p = json_str(cp, buf, &n);
        if (t->nsp < 64 && id >= 0 && id < cap) {
            special *s = &t->sp[t->nsp++];
            s->s = malloc((size_t)n + 1);
            memcpy(s->s, buf, (size_t)n);
            s->s[n] = 0;
            s->len = n;
            s->id = (int32_t)id;
            if (!t->str[id]) {
                t->str[id] = strdup(s->s);
                t->slen[id] = n;
            }
            if (id + 1 > t->nvocab) t->nvocab = (int)id + 1;
        }
    }
    free(tj);
    // 字串 → id
    t->shcap = 1;
    while (t->shcap < (size_t)t->nvocab * 2) t->shcap <<= 1;
    t->sh = malloc(t->shcap * sizeof(int32_t));
    memset(t->sh, 0xff, t->shcap * sizeof(int32_t));
    for (int i = 0; i < t->nvocab; i++)
        if (t->str[i]) str_put(t, i);
    // byte 對應
    uint32_t b2c[256];
    byte_map(b2c);
    for (int b = 0; b < 256; b++) {
        char o[4];
        const int n = utf8_put(o, b2c[b]);
        t->byte_id[b] = str_find(t, o, n);
        t->ch2byte[b2c[b]] = (uint8_t)b;
        if (t->byte_id[b] < 0) { snprintf(err, errlen, "vocab 缺少 byte %d", b); qw_tok_close(t); return NULL; }
    }
    // merges.txt：第 r 行「a b」→ rank r
    snprintf(path, sizeof path, "%s/merges.txt", dir);
    size_t mlen;
    char *mt = slurp(path, &mlen);
    if (!mt) { snprintf(err, errlen, "讀不到 %s", path); qw_tok_close(t); return NULL; }
    size_t nlines = 0;
    for (size_t i = 0; i < mlen; i++) nlines += mt[i] == '\n';
    t->mhcap = 1;
    while (t->mhcap < (nlines + 1) * 2) t->mhcap <<= 1;
    t->mh = calloc(t->mhcap, sizeof(merge_ent));
    int rank = 0;
    for (char *p = mt; *p;) {
        char *nl = strchr(p, '\n');
        if (!nl) nl = p + strlen(p);
        if (p[0] == '#' && rank == 0 && strncmp(p, "#version", 8) == 0) { p = *nl ? nl + 1 : nl; continue; }
        char *sp = memchr(p, ' ', (size_t)(nl - p));
        if (sp) {
            const int32_t a = str_find(t, p, (int)(sp - p)), b = str_find(t, sp + 1, (int)(nl - sp - 1));
            char cat[1024];
            const int la = (int)(sp - p), lb = (int)(nl - sp - 1);
            if (a >= 0 && b >= 0 && la + lb < (int)sizeof cat) {
                memcpy(cat, p, (size_t)la);
                memcpy(cat + la, sp + 1, (size_t)lb);
                const int32_t m = str_find(t, cat, la + lb);
                if (m >= 0) {
                    const uint64_t key = ((uint64_t)(uint32_t)a << 32 | (uint32_t)b) + 1;
                    size_t i = mix64(key) & (t->mhcap - 1);
                    while (t->mh[i].key && t->mh[i].key != key) i = (i + 1) & (t->mhcap - 1);
                    if (!t->mh[i].key) t->mh[i] = (merge_ent){key, rank, m};
                }
            }
            rank++;
        }
        p = *nl ? nl + 1 : nl;
    }
    free(mt);
    return t;
}

void qw_tok_close(qw_tok *t) {
    if (!t) return;
    if (t->str) for (int i = 0; i < t->nvocab; i++) free(t->str[i]);
    free(t->str);
    free(t->slen);
    free(t->sh);
    free(t->mh);
    for (int i = 0; i < t->nsp; i++) free(t->sp[i].s);
    free(t);
}

// ---------------------------------------------------------------- Unicode
static bool in_tab(const uint32_t (*tab)[2], int n, uint32_t c) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        const int m = (lo + hi) / 2;
        if (c < tab[m][0]) hi = m - 1;
        else if (c > tab[m][1]) lo = m + 1;
        else return true;
    }
    return false;
}
static bool isL(uint32_t c) { return c < 128 ? ((c | 32) >= 'a' && (c | 32) <= 'z') : in_tab(uni_L, uni_L_n, c); }
static bool isM(uint32_t c) { return c >= 0x300 && in_tab(uni_M, uni_M_n, c); }
static bool isN(uint32_t c) { return c < 128 ? (c >= '0' && c <= '9') : in_tab(uni_N, uni_N_n, c); }
static bool isWS(uint32_t c) { return c < 128 ? (c == ' ' || (c >= 9 && c <= 13)) : in_tab(uni_WS, uni_WS_n, c); }
static bool isNL(uint32_t c) { return c == '\r' || c == '\n'; }

// 解一個 UTF-8 字元（不合法的 byte 當成單一碼位 0xFFFD 但長度 1）
static uint32_t u8(const uint8_t *s, size_t n, size_t i, int *len) {
    const uint8_t c = s[i];
    if (c < 0x80) { *len = 1; return c; }
    int k = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
    if (k == 0 || i + (size_t)k > n) { *len = 1; return 0xFFFD; }
    uint32_t v = c & (0x7F >> k);
    for (int j = 1; j < k; j++) {
        if ((s[i + j] & 0xC0) != 0x80) { *len = 1; return 0xFFFD; }
        v = v << 6 | (s[i + j] & 63);
    }
    *len = k;
    return v;
}

// 切詞：回傳從 i 開始這一段的結尾（byte 位置）。依 pattern 的選項順序：
// (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} | ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]* |
// \s*[\r\n]+ | \s+(?!\S) | \s+
static size_t pretok(const uint8_t *s, size_t n, size_t i) {
    int l0, l1;
    const uint32_t c0 = u8(s, n, i, &l0);
    // 1. 縮寫
    if (c0 == '\'' && i + 1 < n) {
        const char a = (char)(s[i + 1] | 32), b = i + 2 < n ? (char)(s[i + 2] | 32) : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
    // 2. 可選的一個非字母數字（非換行）字元 + 字母/標記
    {
        size_t j = i;
        if (!isNL(c0) && !isL(c0) && !isN(c0)) {
            const size_t k = i + (size_t)l0;
            if (k < n) {
                const uint32_t c1 = u8(s, n, k, &l1);
                if (isL(c1) || isM(c1)) j = k;
            }
        }
        size_t k = j;
        int l;
        while (k < n) {
            const uint32_t c = u8(s, n, k, &l);
            if (!(isL(c) || isM(c))) break;
            k += (size_t)l;
        }
        if (k > j) return k;
    }
    // 3. 一個數字
    if (isN(c0)) return i + (size_t)l0;
    // 4. 可選空格 + 標點等 + 換行
    {
        size_t j = i;
        if (c0 == ' ') j = i + 1;
        size_t k = j;
        int l;
        while (k < n) {
            const uint32_t c = u8(s, n, k, &l);
            if (isWS(c) || isL(c) || isM(c) || isN(c)) break;
            k += (size_t)l;
        }
        if (k > j) {
            while (k < n && isNL(s[k])) k++;
            return k;
        }
    }
    // 空白：先找整段空白
    size_t k = i, lastnl = 0;
    bool hasnl = false;
    int l;
    while (k < n) {
        const uint32_t c = u8(s, n, k, &l);
        if (!isWS(c)) break;
        k += (size_t)l;
        if (isNL(c)) { hasnl = true; lastnl = k; }
    }
    if (k > i) {
        // 5. \s*[\r\n]+：到最後一個換行為止
        if (hasnl) return lastnl;
        // 6. \s+(?!\S)：到文字結尾就全拿；否則少拿最後一個字元（至少要剩 1 個）
        if (k == n) return k;
        size_t prev = i, q = i;
        while (q < k) { prev = q; u8(s, n, q, &l); q += (size_t)l; }
        if (prev > i) return prev;
        // 7. \s+
        return k;
    }
    // 其他（不應該發生）：單一字元
    return i + (size_t)l0;
}

// ---------------------------------------------------------------- NFC 正規化（分解 → 依組合類別排序 → 合成）
// 不合法的 UTF-8 byte 以 0x110000 + byte 表示，原樣寫回。
enum { SB = 0xAC00, LB = 0x1100, VB0 = 0x1161, TB = 0x11A7, LC = 19, VC = 21, TC = 28, NC = VC * TC, SC = LC * NC };

static int ccc_of(uint32_t c) {
    if (c < 0x300 || c >= 0x110000) return 0;
    int lo = 0, hi = uni_ccc_n - 1;
    while (lo <= hi) {
        const int m = (lo + hi) / 2;
        if (uni_ccc[m][0] == c) return (int)uni_ccc[m][1];
        if (uni_ccc[m][0] < c) lo = m + 1; else hi = m - 1;
    }
    return 0;
}

static const uint32_t *decomp_of(uint32_t c) {
    if (c < 0xC0 || c >= 0x110000) return NULL;
    int lo = 0, hi = uni_decomp_n - 1;
    while (lo <= hi) {
        const int m = (lo + hi) / 2;
        if (uni_decomp[m][0] == c) return uni_decomp[m];
        if (uni_decomp[m][0] < c) lo = m + 1; else hi = m - 1;
    }
    return NULL;
}

static uint32_t compose(uint32_t a, uint32_t b) {
    if (a >= LB && a < LB + LC && b >= VB0 && b < VB0 + VC) return SB + ((a - LB) * VC + (b - VB0)) * TC;
    if (a >= SB && a < SB + SC && (a - SB) % TC == 0 && b > TB && b < TB + TC) return a + (b - TB);
    int lo = 0, hi = uni_comp_n - 1;
    while (lo <= hi) {
        const int m = (lo + hi) / 2;
        const uint32_t x = uni_comp[m][0], y = uni_comp[m][1];
        if (x == a && y == b) return uni_comp[m][2];
        if (x < a || (x == a && y < b)) lo = m + 1; else hi = m - 1;
    }
    return 0;
}

// 回傳 NFC 後的新字串（呼叫端 free），長度寫進 *olen
static char *nfc(const uint8_t *s, size_t n, size_t *olen) {
    uint32_t *d = malloc(sizeof(uint32_t) * (n * 4 + 4));
    size_t m = 0;
    for (size_t i = 0; i < n;) {
        int l;
        uint32_t c = u8(s, n, i, &l);
        if (c == 0xFFFD && !(l == 3 && s[i] == 0xEF && s[i + 1] == 0xBF && s[i + 2] == 0xBD)) c = 0x110000 + s[i];
        i += (size_t)l;
        if (c >= SB && c < SB + SC) {
            const uint32_t si = c - SB;
            d[m++] = LB + si / NC;
            d[m++] = VB0 + (si % NC) / TC;
            if (si % TC) d[m++] = TB + si % TC;
        } else if (decomp_of(c)) {
            const uint32_t *e = decomp_of(c);
            for (uint32_t k = 0; k < e[1]; k++) d[m++] = e[2 + k];
        } else
            d[m++] = c;
    }
    // 組合類別排序（穩定）
    for (size_t i = 1; i < m; i++) {
        const int cb = ccc_of(d[i]);
        if (!cb) continue;
        for (size_t j = i; j > 0; j--) {
            const int ca = ccc_of(d[j - 1]);
            if (ca <= cb || ca == 0) break;
            const uint32_t t = d[j]; d[j] = d[j - 1]; d[j - 1] = t;
        }
    }
    // 合成
    size_t o = 0, st = SIZE_MAX;
    int last = 0;
    bool after = false;
    for (size_t i = 0; i < m; i++) {
        const uint32_t c = d[i];
        const int cc = ccc_of(c);
        if (st != SIZE_MAX && (!after || (last != 0 && last < cc))) {
            const uint32_t x = compose(d[st], c);
            if (x) { d[st] = x; continue; }
        }
        d[o++] = c;
        if (cc == 0) { st = o - 1; after = false; last = 0; }
        else { after = true; last = cc; }
    }
    char *out = malloc(o * 4 + 1);
    size_t k = 0;
    for (size_t i = 0; i < o; i++) {
        if (d[i] >= 0x110000) out[k++] = (char)(d[i] - 0x110000);
        else k += (size_t)utf8_put(out + k, d[i]);
    }
    free(d);
    *olen = k;
    return out;
}

// 一段 bytes 做 BPE，寫到 out（cap 不夠時只計數）
static int bpe(const qw_tok *t, const uint8_t *s, size_t n, int32_t *out, int cap, int pos) {
    // 整段剛好是一個 token：直接用（常見情況）
    uint32_t b2c[256];
    static uint32_t map_ready = 0;
    static uint32_t b2c_s[256];
    if (!map_ready) { byte_map(b2c_s); map_ready = 1; }
    memcpy(b2c, b2c_s, sizeof b2c);
    {
        char tmp[1024];
        int m = 0;
        if (n * 2 < sizeof tmp) {
            for (size_t i = 0; i < n; i++) m += utf8_put(tmp + m, b2c[s[i]]);
            const int32_t id = str_find(t, tmp, m);
            if (id >= 0) {
                if (out && pos < cap) out[pos] = id;
                return 1;
            }
        }
    }
    int32_t *sym = malloc(n * sizeof(int32_t));
    size_t m = n;
    for (size_t i = 0; i < n; i++) sym[i] = t->byte_id[s[i]];
    for (;;) {
        int32_t best = INT32_MAX, bi = -1, bid = -1;
        for (size_t i = 0; i + 1 < m; i++) {
            const merge_ent *e = merge_find(t, sym[i], sym[i + 1]);
            if (e && e->rank < best) { best = e->rank; bi = (int32_t)i; bid = e->id; }
        }
        if (bi < 0) break;
        sym[bi] = bid;
        memmove(sym + bi + 1, sym + bi + 2, (m - (size_t)bi - 2) * sizeof(int32_t));
        m--;
    }
    for (size_t i = 0; i < m; i++)
        if (out && pos + (int)i < cap) out[pos + (int)i] = sym[i];
    free(sym);
    return (int)m;
}

int qw_tok_encode(const qw_tok *t, const char *text, size_t len, int32_t *out, int cap) {
    // 有非 ASCII 字元時先做 NFC
    char *norm = NULL;
    for (size_t i = 0; i < len; i++)
        if ((uint8_t)text[i] >= 0x80) {
            norm = nfc((const uint8_t *)text, len, &len);
            text = norm;
            break;
        }
    const uint8_t *s = (const uint8_t *)text;
    int pos = 0;
    size_t i = 0, seg = 0;
    while (i <= len) {
        // 特殊 token
        int hit = -1;
        if (i < len && s[i] == '<')
            for (int k = 0; k < t->nsp; k++)
                if ((size_t)t->sp[k].len <= len - i && !memcmp(s + i, t->sp[k].s, (size_t)t->sp[k].len)) {
                    if (hit < 0 || t->sp[k].len > t->sp[hit].len) hit = k;
                }
        if (hit >= 0 || i == len) {
            // 先處理前面的一般文字
            for (size_t j = seg; j < i;) {
                const size_t e = pretok(s, i, j);
                pos += bpe(t, s + j, e - j, out, cap, pos);
                j = e;
            }
            if (i == len) break;
            if (out && pos < cap) out[pos] = t->sp[hit].id;
            pos++;
            i += (size_t)t->sp[hit].len;
            seg = i;
            continue;
        }
        i++;
    }
    free(norm);
    return pos;
}

int qw_tok_decode1(const qw_tok *t, int32_t id, char *buf) {
    if (id < 0 || id >= t->nvocab || !t->str[id]) return 0;
    for (int k = 0; k < t->nsp; k++)
        if (t->sp[k].id == id) {
            memcpy(buf, t->sp[k].s, (size_t)t->sp[k].len);
            return t->sp[k].len;
        }
    const uint8_t *p = (const uint8_t *)t->str[id];
    const size_t n = (size_t)t->slen[id];
    int o = 0;
    for (size_t i = 0; i < n && o < 255;) {
        int l;
        const uint32_t c = u8(p, n, i, &l);
        buf[o++] = c < 512 ? (char)t->ch2byte[c] : '?';
        i += (size_t)l;
    }
    return o;
}

int32_t qw_tok_special(const qw_tok *t, const char *content) {
    for (int k = 0; k < t->nsp; k++)
        if (!strcmp(t->sp[k].s, content)) return t->sp[k].id;
    return -1;
}

int qw_tok_vocab(const qw_tok *t) { return t->nvocab; }
