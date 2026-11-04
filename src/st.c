// safetensors 讀取器實作（說明見 st.h）
#include "st.h"
#include <ctype.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------- 迷你 JSON 解析器
// 只處理 safetensors header 會出現的東西：物件、陣列、字串、整數、以及要跳過的任意值。

typedef struct {
    const char *p, *end;
    char *err;
    size_t errlen;
} js;

static bool fail(js *j, const char *msg) {
    if (j->err && j->errlen) snprintf(j->err, j->errlen, "JSON 解析失敗：%s（剩 %td 個字元）", msg, j->end - j->p);
    return false;
}

static void ws(js *j) {
    while (j->p < j->end && isspace((unsigned char)*j->p)) j->p++;
}

static bool expect(js *j, char c) {
    ws(j);
    if (j->p >= j->end || *j->p != c) return fail(j, "缺少預期的字元");
    j->p++;
    return true;
}

// 讀字串（處理基本跳脫字元；\u 只保留原樣，tensor 名稱不會用到）
static bool str(js *j, char **out) {
    ws(j);
    if (j->p >= j->end || *j->p != '"') return fail(j, "預期字串");
    j->p++;
    const char *s = j->p;
    size_t n = 0;
    while (j->p < j->end && *j->p != '"') {
        if (*j->p == '\\') j->p++;
        j->p++;
        n++;
    }
    if (j->p >= j->end) return fail(j, "字串沒有結尾");
    if (out) {
        char *r = malloc(n + 1), *w = r;
        for (const char *q = s; q < j->p; q++) {
            if (*q == '\\') {
                q++;
                *w++ = *q == 'n' ? '\n' : *q == 't' ? '\t' : *q;
            } else
                *w++ = *q;
        }
        *w = 0;
        *out = r;
    }
    j->p++;
    return true;
}

static bool integer(js *j, int64_t *out) {
    ws(j);
    char *e;
    long long v = strtoll(j->p, &e, 10);
    if (e == j->p) return fail(j, "預期整數");
    j->p = e;
    *out = v;
    return true;
}

static bool skip(js *j) {
    ws(j);
    if (j->p >= j->end) return fail(j, "提前結束");
    char c = *j->p;
    if (c == '"') return str(j, NULL);
    if (c == '{' || c == '[') {
        char close = c == '{' ? '}' : ']';
        j->p++;
        ws(j);
        if (j->p < j->end && *j->p == close) {
            j->p++;
            return true;
        }
        for (;;) {
            if (c == '{') {
                if (!str(j, NULL) || !expect(j, ':')) return false;
            }
            if (!skip(j)) return false;
            ws(j);
            if (j->p < j->end && *j->p == ',') {
                j->p++;
                continue;
            }
            return expect(j, close);
        }
    }
    // 數字、true、false、null
    while (j->p < j->end && !strchr(",}] \t\r\n", *j->p)) j->p++;
    return true;
}

static st_dtype parse_dtype(const char *s) {
    static const struct { const char *n; st_dtype d; } map[] = {
        {"F32", ST_F32}, {"F16", ST_F16}, {"BF16", ST_BF16}, {"F64", ST_F64}, {"I8", ST_I8},
        {"U8", ST_U8}, {"I16", ST_I16}, {"U16", ST_U16}, {"I32", ST_I32}, {"U32", ST_U32},
        {"I64", ST_I64}, {"U64", ST_U64}, {"BOOL", ST_BOOL}, {"F8_E4M3", ST_F8_E4M3}, {"F8_E5M2", ST_F8_E5M2},
    };
    for (size_t i = 0; i < sizeof map / sizeof map[0]; i++)
        if (!strcmp(s, map[i].n)) return map[i].d;
    return ST_UNKNOWN;
}

size_t st_dtype_size(st_dtype d) {
    switch (d) {
    case ST_F64: case ST_I64: case ST_U64: return 8;
    case ST_F32: case ST_I32: case ST_U32: return 4;
    case ST_F16: case ST_BF16: case ST_I16: case ST_U16: return 2;
    case ST_I8: case ST_U8: case ST_BOOL: case ST_F8_E4M3: case ST_F8_E5M2: return 1;
    default: return 0;
    }
}

const char *st_dtype_name(st_dtype d) {
    static const char *n[] = {"?", "F32", "F16", "BF16", "F64", "I8", "U8", "I16", "U16",
                              "I32", "U32", "I64", "U64", "BOOL", "F8_E4M3", "F8_E5M2"};
    return (unsigned)d < sizeof n / sizeof n[0] ? n[d] : "?";
}

// 解析一個 tensor 的描述物件
static bool tensor_obj(js *j, st_tensor *t) {
    if (!expect(j, '{')) return false;
    bool have_off = false;
    for (;;) {
        char *key;
        if (!str(j, &key) || !expect(j, ':')) return false;
        if (!strcmp(key, "dtype")) {
            char *d;
            if (!str(j, &d)) return false;
            t->dtype = parse_dtype(d);
            free(d);
        } else if (!strcmp(key, "shape")) {
            if (!expect(j, '[')) return false;
            t->ndim = 0;
            ws(j);
            if (*j->p == ']') j->p++;
            else
                for (;;) {
                    int64_t v;
                    if (!integer(j, &v)) return false;
                    if (t->ndim < 8) t->shape[t->ndim++] = v;
                    ws(j);
                    if (*j->p == ',') { j->p++; continue; }
                    if (!expect(j, ']')) return false;
                    break;
                }
        } else if (!strcmp(key, "data_offsets")) {
            int64_t a, b;
            if (!expect(j, '[') || !integer(j, &a) || !expect(j, ',') || !integer(j, &b) || !expect(j, ']')) return false;
            t->offset = (uint64_t)a;
            t->nbytes = (uint64_t)(b - a);
            have_off = true;
        } else if (!skip(j))
            return false;
        free(key);
        ws(j);
        if (*j->p == ',') { j->p++; continue; }
        if (!expect(j, '}')) return false;
        break;
    }
    if (!have_off) return fail(j, "tensor 缺少 data_offsets");
    return true;
}

static int cmp_name(const void *a, const void *b) {
    return strcmp(((const st_tensor *)a)->name, ((const st_tensor *)b)->name);
}

st_file *st_parse_header(const void *buf, size_t len, char *err, size_t errlen) {
    if (len < 8) {
        snprintf(err, errlen, "檔案太短");
        return NULL;
    }
    uint64_t n;
    memcpy(&n, buf, 8); // 小端序（Apple Silicon 也是小端序）
    if (n > len - 8 || n > (1ull << 31)) {
        snprintf(err, errlen, "header 長度不合理：%llu", (unsigned long long)n);
        return NULL;
    }
    st_file *f = calloc(1, sizeof *f);
    f->fd = -1;
    f->data_start = 8 + n;
    js j = {(const char *)buf + 8, (const char *)buf + 8 + n, err, errlen};
    int cap = 256;
    f->tensors = calloc(cap, sizeof *f->tensors);
    if (!expect(&j, '{')) goto bad;
    ws(&j);
    if (*j.p == '}') goto done;
    for (;;) {
        char *name;
        if (!str(&j, &name) || !expect(&j, ':')) goto bad;
        if (!strcmp(name, "__metadata__")) {
            free(name);
            if (!skip(&j)) goto bad;
        } else {
            if (f->ntensors == cap) {
                cap *= 2;
                f->tensors = realloc(f->tensors, cap * sizeof *f->tensors);
            }
            st_tensor *t = &f->tensors[f->ntensors];
            memset(t, 0, sizeof *t);
            t->name = name;
            if (!tensor_obj(&j, t)) goto bad;
            t->offset += f->data_start;
            f->ntensors++;
        }
        ws(&j);
        if (*j.p == ',') { j.p++; continue; }
        if (!expect(&j, '}')) goto bad;
        break;
    }
done:
    qsort(f->tensors, f->ntensors, sizeof *f->tensors, cmp_name);
    return f;
bad:
    st_close(f);
    return NULL;
}

st_file *st_open(const char *path, char *err, size_t errlen) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        snprintf(err, errlen, "無法開啟 %s", path);
        return NULL;
    }
    struct stat sb;
    fstat(fd, &sb);
    void *map = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        snprintf(err, errlen, "mmap 失敗：%s", path);
        return NULL;
    }
    st_file *f = st_parse_header(map, (size_t)sb.st_size, err, errlen);
    if (!f) {
        munmap(map, (size_t)sb.st_size);
        close(fd);
        return NULL;
    }
    f->fd = fd;
    f->map = map;
    f->size = (size_t)sb.st_size;
    f->path = strdup(path);
    for (int i = 0; i < f->ntensors; i++) {
        st_tensor *t = &f->tensors[i];
        if (t->offset + t->nbytes > f->size) {
            snprintf(err, errlen, "%s：tensor %s 超出檔案範圍", path, t->name);
            st_close(f);
            return NULL;
        }
        t->data = (const char *)map + t->offset;
    }
    return f;
}

void st_close(st_file *f) {
    if (!f) return;
    for (int i = 0; i < f->ntensors; i++) free(f->tensors[i].name);
    free(f->tensors);
    if (f->map) munmap(f->map, f->size);
    if (f->fd >= 0) close(f->fd);
    free(f->path);
    free(f);
}

const st_tensor *st_find(const st_file *f, const char *name) {
    st_tensor key = {.name = (char *)name};
    return bsearch(&key, f->tensors, f->ntensors, sizeof *f->tensors, cmp_name);
}

// ---------------------------------------------------------------- 多分片

static char *read_all(const char *path, size_t *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    rewind(fp);
    char *b = malloc(n + 1);
    if (fread(b, 1, n, fp) != (size_t)n) {
        fclose(fp);
        free(b);
        return NULL;
    }
    b[n] = 0;
    fclose(fp);
    *len = (size_t)n;
    return b;
}

st_model *st_model_open(const char *dir, char *err, size_t errlen) {
    st_model *m = calloc(1, sizeof *m);
    m->dir = strdup(dir);
    char path[4096];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    size_t len;
    char *idx = read_all(path, &len);
    char **names = NULL;
    int nn = 0, cap = 0;
    if (idx) {
        // 收集 weight_map 裡出現過的分片檔名（去重）
        js j = {idx, idx + len, err, errlen};
        if (!expect(&j, '{')) goto bad;
        for (;;) {
            char *key;
            if (!str(&j, &key) || !expect(&j, ':')) goto bad;
            if (!strcmp(key, "weight_map")) {
                if (!expect(&j, '{')) goto bad;
                for (;;) {
                    char *w, *file;
                    if (!str(&j, &w) || !expect(&j, ':') || !str(&j, &file)) goto bad;
                    free(w);
                    bool seen = false;
                    for (int i = 0; i < nn; i++) seen |= !strcmp(names[i], file);
                    if (seen) free(file);
                    else {
                        if (nn == cap) names = realloc(names, (cap = cap ? cap * 2 : 16) * sizeof *names);
                        names[nn++] = file;
                    }
                    ws(&j);
                    if (*j.p == ',') { j.p++; continue; }
                    if (!expect(&j, '}')) goto bad;
                    break;
                }
            } else if (!skip(&j))
                goto bad;
            free(key);
            ws(&j);
            if (*j.p == ',') { j.p++; continue; }
            break;
        }
        free(idx);
        idx = NULL;
    } else {
        names = malloc(sizeof *names);
        names[nn++] = strdup("model.safetensors");
    }
    m->files = calloc(nn, sizeof *m->files);
    for (int i = 0; i < nn; i++) {
        snprintf(path, sizeof path, "%s/%s", dir, names[i]);
        m->files[i] = st_open(path, err, errlen);
        if (!m->files[i]) goto bad;
        m->nfiles++;
    }
    for (int i = 0; i < nn; i++) free(names[i]);
    free(names);
    return m;
bad:
    free(idx);
    for (int i = 0; i < nn; i++) free(names[i]);
    free(names);
    st_model_close(m);
    return NULL;
}

void st_model_close(st_model *m) {
    if (!m) return;
    for (int i = 0; i < m->nfiles; i++) st_close(m->files[i]);
    free(m->files);
    free(m->dir);
    free(m);
}

const st_tensor *st_model_find(const st_model *m, const char *name, const st_file **file_out) {
    for (int i = 0; i < m->nfiles; i++) {
        const st_tensor *t = st_find(m->files[i], name);
        if (t) {
            if (file_out) *file_out = m->files[i];
            return t;
        }
    }
    return NULL;
}
