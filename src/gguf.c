// GGUF profile 載入（說明見 gguf.h）
#include "gguf.h"
#include <dirent.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { GT_F32 = 0, GT_F16 = 1, GT_BF16 = 30 };

bool qw_ggml_block(int type, int *qb, int *bb) {
    static const struct { int t, qb, bb; } tab[] = {
        {0, 1, 4},     {1, 1, 2},     {30, 1, 2},    {2, 32, 18},   {3, 32, 20},   {6, 32, 22},   {7, 32, 24},
        {8, 32, 34},   {10, 256, 84}, {11, 256, 110}, {12, 256, 144}, {13, 256, 176}, {14, 256, 210}, {16, 256, 66},
        {17, 256, 74}, {18, 256, 98}, {19, 256, 50}, {20, 32, 18},  {21, 256, 110}, {22, 256, 82}, {23, 256, 136},
        {29, 256, 56}, {42, 64, 18},
    };
    for (size_t i = 0; i < sizeof tab / sizeof *tab; i++)
        if (tab[i].t == type) { *qb = tab[i].qb; *bb = tab[i].bb; return true; }
    return false;
}

const char *qw_ggml_name(int type) {
    switch (type) {
    case 0: return "F32"; case 1: return "F16"; case 30: return "BF16"; case 2: return "Q4_0"; case 3: return "Q4_1";
    case 6: return "Q5_0"; case 7: return "Q5_1"; case 8: return "Q8_0"; case 10: return "Q2_K"; case 11: return "Q3_K";
    case 12: return "Q4_K"; case 13: return "Q5_K"; case 14: return "Q6_K"; case 16: return "IQ2_XXS"; case 17: return "IQ2_XS";
    case 18: return "IQ3_XXS"; case 19: return "IQ1_S"; case 20: return "IQ4_NL"; case 21: return "IQ3_S"; case 22: return "IQ2_S";
    case 23: return "IQ4_XS"; case 29: return "IQ1_M"; case 42: return "Q2_0(42)";
    default: return "?";
    }
}

unsigned qw_gguf_parse_parts(const char *s) {
    if (!s || !*s || !strcmp(s, "all")) return QW_GG_ALL;
    static const struct { const char *n; unsigned b; } tab[] = {
        {"moe", QW_GG_MOE},     {"shared", QW_GG_SHARED}, {"router", QW_GG_ROUTER}, {"attn", QW_GG_ATTN}, {"lin", QW_GG_LIN},
        {"hyper", QW_GG_HYPER}, {"lm", QW_GG_LM},         {"embed", QW_GG_EMBED},   {"ple", QW_GG_PLE},
    };
    unsigned r = 0;
    char buf[256];
    snprintf(buf, sizeof buf, "%s", s);
    for (char *t = strtok(buf, ","); t; t = strtok(NULL, ","))
        for (size_t i = 0; i < sizeof tab / sizeof *tab; i++)
            if (!strcmp(t, tab[i].n)) r |= tab[i].b;
    return r;
}

// ---- 解析
typedef struct {
    char *name;
    uint32_t nd, type;
    uint64_t dims[4], off; // off：相對於資料區
} gtensor;

typedef struct {
    char *path;
    int fd;
    uint64_t fsize, data0;
    gtensor *t;
    int nt;
} gfile;

static bool rdn(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }
static char *rstr(FILE *f) {
    uint64_t n;
    if (!rdn(f, &n, 8) || n > (1u << 24)) return NULL;
    char *s = malloc(n + 1);
    if (!rdn(f, s, n)) { free(s); return NULL; }
    s[n] = 0;
    return s;
}
static bool skipv(FILE *f, uint32_t t, uint32_t *align, const char *key) {
    static const int sz[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (t == 8) { char *s = rstr(f); free(s); return s != NULL; }
    if (t == 9) {
        uint32_t at;
        uint64_t n;
        if (!rdn(f, &at, 4) || !rdn(f, &n, 8)) return false;
        if (at == 8) {
            for (uint64_t i = 0; i < n; i++) if (!skipv(f, 8, NULL, NULL)) return false;
            return true;
        }
        if (at > 12 || at == 9) return false;
        return fseeko(f, (off_t)(n * (uint64_t)sz[at]), SEEK_CUR) == 0;
    }
    if (t > 12) return false;
    uint64_t v = 0;
    if (!rdn(f, &v, (size_t)sz[t])) return false;
    if (align && key && !strcmp(key, "general.alignment")) *align = (uint32_t)v;
    return true;
}

static bool gparse(const char *path, gfile *g, char *err, size_t errlen) {
    memset(g, 0, sizeof *g);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "GGUF：開不了 %s", path); return false; }
    char magic[4];
    uint32_t ver, align = 32;
    uint64_t nt, nkv;
    if (!rdn(f, magic, 4) || memcmp(magic, "GGUF", 4) || !rdn(f, &ver, 4) || !rdn(f, &nt, 8) || !rdn(f, &nkv, 8) || ver < 2) {
        fclose(f);
        snprintf(err, errlen, "GGUF：%s 不是 GGUF v2+", path);
        return false;
    }
    for (uint64_t i = 0; i < nkv; i++) {
        char *k = rstr(f);
        uint32_t t;
        if (!k || !rdn(f, &t, 4) || !skipv(f, t, &align, k)) { free(k); fclose(f); snprintf(err, errlen, "GGUF：%s 的 metadata 壞掉", path); return false; }
        free(k);
    }
    g->t = calloc(nt, sizeof *g->t);
    g->nt = (int)nt;
    for (uint64_t i = 0; i < nt; i++) {
        gtensor *t = &g->t[i];
        t->name = rstr(f);
        if (!t->name || !rdn(f, &t->nd, 4) || t->nd > 4) { fclose(f); snprintf(err, errlen, "GGUF：%s 的 tensor 表壞掉", path); return false; }
        for (uint32_t d = 0; d < t->nd; d++) if (!rdn(f, &t->dims[d], 8)) { fclose(f); return false; }
        if (!rdn(f, &t->type, 4) || !rdn(f, &t->off, 8)) { fclose(f); return false; }
    }
    g->data0 = ((uint64_t)ftello(f) + align - 1) / align * align;
    fclose(f);
    g->path = strdup(path);
    g->fd = open(path, O_RDONLY);
    struct stat sb;
    if (g->fd < 0 || fstat(g->fd, &sb)) { snprintf(err, errlen, "GGUF：開不了 %s", path); return false; }
    g->fsize = (uint64_t)sb.st_size;
    return true;
}

static uint64_t tbytes(const gtensor *t) {
    int qb, bb;
    if (!qw_ggml_block((int)t->type, &qb, &bb)) return 0;
    uint64_t rows = 1;
    for (uint32_t d = 1; d < t->nd; d++) rows *= t->dims[d];
    return rows * (t->dims[0] / (uint64_t)qb) * (uint64_t)bb;
}

// ---- 名稱對應：GGUF → MLX（部分、MLX 名稱、種類）
enum { K_LIN, K_EXP, K_SMALL, K_ANEG, K_IDXQ, K_IDXK, K_PLETAB, K_F32 }; // K_F32：小 tensor 一律保留 F32（hyper 的 hc_norm、attention norm）
typedef struct { unsigned part; char mlx[200]; int kind; } gmap;

static bool map_name(const char *g, gmap *o) {
    int l = -1, n = 0;
    char rest[160];
    o->kind = K_LIN;
#define P(fmt, ...) snprintf(o->mlx, sizeof o->mlx, "language_model.model.layers.%d." fmt, l, ##__VA_ARGS__)
    if (sscanf(g, "blk.%d.%n", &l, &n) == 1 && n > 0) {
        snprintf(rest, sizeof rest, "%s", g + n);
        static const struct { const char *g; unsigned part; const char *m; int kind; } tab[] = {
            {"attn_qkv.weight", QW_GG_LIN, "linear_attn.in_proj_qkv.weight", K_LIN},
            {"attn_gate.weight", QW_GG_LIN, "linear_attn.in_proj_z.weight", K_LIN},
            {"ssm_alpha.weight", QW_GG_LIN, "linear_attn.in_proj_a.weight", K_LIN},
            {"ssm_beta.weight", QW_GG_LIN, "linear_attn.in_proj_b.weight", K_LIN},
            {"ssm_out.weight", QW_GG_LIN, "linear_attn.out_proj.weight", K_LIN},
            {"ssm_a", QW_GG_LIN, "linear_attn.A_neg", K_ANEG},
            {"ssm_dt.bias", QW_GG_LIN, "linear_attn.dt_bias", K_SMALL},
            {"ssm_conv1d.weight", QW_GG_LIN, "linear_attn.conv1d.weight", K_SMALL},
            {"ssm_norm.weight", QW_GG_LIN, "linear_attn.norm.weight", K_SMALL},
            {"attn_q.weight", QW_GG_ATTN, "self_attn.q_proj.weight", K_LIN},
            {"attn_k.weight", QW_GG_ATTN, "self_attn.k_proj.weight", K_LIN},
            {"attn_v.weight", QW_GG_ATTN, "self_attn.v_proj.weight", K_LIN},
            {"attn_output.weight", QW_GG_ATTN, "self_attn.o_proj.weight", K_LIN},
            {"attn_q_norm.weight", QW_GG_ATTN, "self_attn.q_norm.weight", K_F32},
            {"attn_k_norm.weight", QW_GG_ATTN, "self_attn.k_norm.weight", K_F32},
            {"indexer.q_proj.weight", QW_GG_ATTN, "self_attn.indexer.index_qk_proj.weight", K_IDXQ},
            {"indexer.k_proj.weight", QW_GG_ATTN, "self_attn.indexer.index_qk_proj.weight", K_IDXK},
            {"indexer.q_norm.weight", QW_GG_ATTN, "self_attn.indexer.q_layernorm.weight", K_F32},
            {"indexer.k_norm.weight", QW_GG_ATTN, "self_attn.indexer.k_layernorm.weight", K_F32},
            {"ffn_gate_inp.weight", QW_GG_ROUTER, "mlp.gate.weight", K_LIN},
            {"ffn_gate_inp_shexp.weight", QW_GG_ROUTER, "mlp.shared_expert_gate.weight", K_SMALL},
            {"ffn_gate_shexp.weight", QW_GG_SHARED, "mlp.shared_expert.gate_proj.weight", K_LIN},
            {"ffn_up_shexp.weight", QW_GG_SHARED, "mlp.shared_expert.up_proj.weight", K_LIN},
            {"ffn_down_shexp.weight", QW_GG_SHARED, "mlp.shared_expert.down_proj.weight", K_LIN},
            {"ffn_gate_exps.weight", QW_GG_MOE, "mlp.switch_mlp.gate_proj.iq", K_EXP},
            {"ffn_up_exps.weight", QW_GG_MOE, "mlp.switch_mlp.up_proj.iq", K_EXP},
            {"ffn_down_exps.weight", QW_GG_MOE, "mlp.switch_mlp.down_proj.iq", K_EXP},
            {"hc_attn_down.weight", QW_GG_HYPER, "attn_hyper_connection.input_mix_weight_down.weight", K_LIN},
            {"hc_attn_up.weight", QW_GG_HYPER, "attn_hyper_connection.input_mix_weight_up.weight", K_LIN},
            {"hc_attn_inject.weight", QW_GG_HYPER, "attn_hyper_connection.block_inject_weight.weight", K_SMALL},
            {"hc_attn_norm.weight", QW_GG_HYPER, "attn_hyper_connection.hc_norm.weight", K_F32},
            {"hc_ffn_down.weight", QW_GG_HYPER, "mlp_hyper_connection.input_mix_weight_down.weight", K_LIN},
            {"hc_ffn_up.weight", QW_GG_HYPER, "mlp_hyper_connection.input_mix_weight_up.weight", K_LIN},
            {"hc_ffn_inject.weight", QW_GG_HYPER, "mlp_hyper_connection.block_inject_weight.weight", K_SMALL},
            {"hc_ffn_norm.weight", QW_GG_HYPER, "mlp_hyper_connection.hc_norm.weight", K_F32},
            {"ple_key.weight", QW_GG_PLE, "ple.key_proj.weight", K_LIN},
            {"ple_value.weight", QW_GG_PLE, "ple.value_proj.weight", K_LIN},
            {"ple_conv1d.weight", QW_GG_PLE, "ple.conv1d.weight", K_SMALL},
            {"ple_norm_conv.weight", QW_GG_PLE, "ple.norm_conv.weight", K_SMALL},
            {"ple_norm_key.weight", QW_GG_PLE, "ple.norm_key.weight", K_SMALL},
            {"ple_norm_query.weight", QW_GG_PLE, "ple.norm_query.weight", K_SMALL},
        };
        for (size_t i = 0; i < sizeof tab / sizeof *tab; i++)
            if (!strcmp(rest, tab[i].g)) {
                o->part = tab[i].part;
                o->kind = tab[i].kind;
                P("%s", tab[i].m);
                return true;
            }
        return false;
    }
#undef P
    static const struct { const char *g; unsigned part; const char *m; int kind; } top[] = {
        {"output.weight", QW_GG_LM, "language_model.lm_head.weight", K_LIN},
        {"token_embd.weight", QW_GG_EMBED, "language_model.model.embed_tokens.weight", K_LIN},
        {"output_hc_down.weight", QW_GG_HYPER, "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight", K_LIN},
        {"output_hc_up.weight", QW_GG_HYPER, "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight", K_LIN},
        {"output_hc_norm.weight", QW_GG_HYPER, "language_model.model.hyper_connection_mixer.hc_norm.weight", K_F32},
        {"per_layer_token_embd.weight", QW_GG_PLE, "", K_PLETAB},
    };
    for (size_t i = 0; i < sizeof top / sizeof *top; i++)
        if (!strcmp(g, top[i].g)) {
            o->part = top[i].part;
            o->kind = top[i].kind;
            snprintf(o->mlx, sizeof o->mlx, "%s", top[i].m);
            return true;
        }
    return false;
}

static uint16_t f2bf_exact(float x, bool *ok) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if (u & 0xffff) *ok = false;
    return (uint16_t)(u >> 16);
}
static float h2f(uint16_t h) {
    _Float16 x;
    memcpy(&x, &h, 2);
    return (float)x;
}

static int cmpstr(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static int cmp_tname(const void *a, const void *b) { return strcmp(((const st_tensor *)a)->name, ((const st_tensor *)b)->name); }

// 一段檔案區域（page 對齊）做成 st_file
static st_file *region(const gfile *g, uint64_t lo, uint64_t hi, const char *tag) {
    const uint64_t pg = (uint64_t)getpagesize();
    lo = lo / pg * pg;
    if (hi > g->fsize) hi = g->fsize;
    void *map = mmap(NULL, hi - lo, PROT_READ, MAP_SHARED, g->fd, (off_t)lo);
    if (map == MAP_FAILED) return NULL;
    st_file *f = calloc(1, sizeof *f);
    char p[1200];
    snprintf(p, sizeof p, "%s%s", g->path, tag);
    f->path = strdup(p);
    f->fd = -1;
    f->map = map;
    f->size = hi - lo;
    f->data_start = lo; // 借用：這個區域在原檔的起點
    return f;
}

bool qw_gguf_attach(st_model *st, qw_model *m, const char *dir, unsigned parts, char *err, size_t errlen) {
    DIR *d = opendir(dir);
    if (!d) { snprintf(err, errlen, "GGUF：開不了目錄 %s", dir); return false; }
    char *names[16];
    int nn = 0;
    for (struct dirent *de; (de = readdir(d)) && nn < 16;) {
        const size_t l = strlen(de->d_name);
        if (l > 5 && !strcmp(de->d_name + l - 5, ".gguf")) {
            char p[2048];
            snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
            names[nn++] = strdup(p);
        }
    }
    closedir(d);
    if (!nn) { snprintf(err, errlen, "GGUF：%s 裡沒有 .gguf", dir); return false; }
    qsort(names, (size_t)nn, sizeof *names, cmpstr);

    // 轉換用的記憶體：小 tensor 轉 BF16 / 保留 F32、indexer q + k 拼起來（先算總量）
    typedef struct { const gfile *g; const gtensor *t; gmap mp; } job;
    job *jobs = NULL;
    int nj = 0, cj = 0;
    gfile *gf = calloc((size_t)nn, sizeof *gf);
    st_file **newf = NULL;
    int nnew = 0;
    uint64_t conv_bytes = 0;
    for (int i = 0; i < nn; i++) {
        if (!gparse(names[i], &gf[i], err, errlen)) return false;
        const gfile *g = &gf[i];
        // PLE 表所在的位元組範圍：不包進 Metal buffer
        uint64_t tlo = 0, thi = 0;
        for (int k = 0; k < g->nt; k++) {
            gmap mp;
            if (!map_name(g->t[k].name, &mp) || !(mp.part & parts)) continue;
            if (mp.kind == K_PLETAB) {
                tlo = g->data0 + g->t[k].off;
                thi = tlo + tbytes(&g->t[k]);
                const uint64_t pg = (uint64_t)getpagesize(), mlo = tlo / pg * pg;
                void *tab = mmap(NULL, thi - mlo, PROT_READ, MAP_SHARED, g->fd, (off_t)mlo);
                if (tab == MAP_FAILED) { snprintf(err, errlen, "GGUF：PLE 表 mmap 失敗"); return false; }
                madvise(tab, thi - mlo, MADV_RANDOM); // 隨機讀幾列：不要預讀（見 engine.c 的 ngram_table.bin）
                int qb, bb;
                qw_ggml_block((int)g->t[k].type, &qb, &bb);
                m->ple_tab = (const uint8_t *)tab + (tlo - mlo);
                m->ple_tab_gq = (int)g->t[k].type + 1;
                m->ple_row_bytes = g->t[k].dims[0] / (uint64_t)qb * (uint64_t)bb;
                continue;
            }
            if (nj == cj) jobs = realloc(jobs, sizeof *jobs * (size_t)(cj = cj ? cj * 2 : 256));
            jobs[nj++] = (job){g, &g->t[k], mp};
            if (mp.kind == K_SMALL || mp.kind == K_F32 || mp.kind == K_ANEG || mp.kind == K_IDXQ || mp.kind == K_IDXK)
                conv_bytes += (tbytes(&g->t[k]) * 2 + 16383) / 16384 * 16384 + 16384;
        }
        // 區域：[0, PLE 表) 與 [PLE 表結束, 檔尾)；沒有表就整個檔
        newf = realloc(newf, sizeof *newf * (size_t)(nnew + 2));
        if (thi > tlo) {
            st_file *a = region(g, 0, tlo, "（前段）"), *b = region(g, thi, g->fsize, "（後段）");
            if (!a || !b) { snprintf(err, errlen, "GGUF：mmap 失敗"); return false; }
            newf[nnew++] = a;
            newf[nnew++] = b;
        } else {
            st_file *a = region(g, 0, g->fsize, "");
            if (!a) { snprintf(err, errlen, "GGUF：mmap 失敗"); return false; }
            newf[nnew++] = a;
        }
    }
    void *cmap = mmap(NULL, conv_bytes + 16384, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (cmap == MAP_FAILED) { snprintf(err, errlen, "GGUF：配置轉換記憶體失敗"); return false; }
    st_file *cf = calloc(1, sizeof *cf);
    cf->path = strdup("(GGUF 轉換)");
    cf->fd = -1;
    cf->map = cmap;
    cf->size = conv_bytes + 16384;
    // 每個區域的 tensor 表（最多 nj 個）
    for (int i = 0; i < nnew; i++) newf[i]->tensors = calloc((size_t)nj + 1, sizeof(st_tensor));
    cf->tensors = calloc((size_t)nj + 1, sizeof(st_tensor));
    uint64_t cofs = 0;
    int nconv_f32 = 0;
    st_tensor *idxq[QW_LAYERS] = {0};
    const gtensor *idxk[QW_LAYERS] = {0};
    const gfile *idxkg[QW_LAYERS] = {0};
    for (int j = 0; j < nj; j++) {
        const gtensor *t = jobs[j].t;
        const gfile *g = jobs[j].g;
        const gmap *mp = &jobs[j].mp;
        int qb, bb;
        if (!qw_ggml_block((int)t->type, &qb, &bb)) {
            snprintf(err, errlen, "GGUF：%s 的格式 %u 不支援", t->name, t->type);
            return false;
        }
        const uint64_t abs0 = g->data0 + t->off, nb = tbytes(t);
        uint64_t rows = 1;
        for (uint32_t k = 1; k < t->nd; k++) rows *= t->dims[k];
        const bool is_float = t->type == GT_F32 || t->type == GT_F16 || t->type == GT_BF16;
        if (mp->kind == K_LIN || mp->kind == K_EXP) {
            // 放在原檔所在的區域
            st_file *rf = NULL;
            for (int i = 0; i < nnew; i++) {
                const st_file *f = newf[i];
                if (strncmp(f->path, g->path, strlen(g->path))) continue;
                if (abs0 >= f->data_start && abs0 + nb <= f->data_start + f->size) rf = newf[i];
            }
            if (!rf) { snprintf(err, errlen, "GGUF：%s 不在任何區域", t->name); return false; }
            st_tensor *o = &rf->tensors[rf->ntensors++];
            o->name = strdup(mp->mlx);
            o->offset = abs0 - rf->data_start;
            o->data = (const char *)rf->map + o->offset;
            o->nbytes = nb;
            if (t->type == GT_BF16) {
                o->dtype = ST_BF16;
                o->gq = GT_BF16 + 1;
                o->ndim = 2;
                o->shape[0] = (int64_t)rows;
                o->shape[1] = (int64_t)t->dims[0];
            } else if (is_float) {
                snprintf(err, errlen, "GGUF：%s 是 %s 矩陣，尚未支援", t->name, qw_ggml_name((int)t->type));
                return false;
            } else {
                o->dtype = ST_U8;
                o->gq = (int)t->type + 1;
                const uint64_t rb = t->dims[0] / (uint64_t)qb * (uint64_t)bb;
                if (mp->kind == K_EXP) {
                    o->ndim = 3;
                    o->shape[0] = (int64_t)t->dims[2];
                    o->shape[1] = (int64_t)t->dims[1];
                    o->shape[2] = (int64_t)rb;
                } else {
                    o->ndim = 2;
                    o->shape[0] = (int64_t)rows;
                    o->shape[1] = (int64_t)rb;
                }
            }
            continue;
        }
        if (!is_float) { snprintf(err, errlen, "GGUF：%s 預期是浮點，實際 %s", t->name, qw_ggml_name((int)t->type)); return false; }
        const uint64_t n = rows * t->dims[0];
        const uint8_t *src = (const uint8_t *)newf[0]->map; // 下面用 pread 讀，避免跨區域
        (void)src;
        void *buf = malloc(nb);
        if (pread(g->fd, buf, nb, (off_t)abs0) != (ssize_t)nb) { free(buf); snprintf(err, errlen, "GGUF：讀 %s 失敗", t->name); return false; }
        float *fv = malloc(sizeof(float) * n);
        for (uint64_t i = 0; i < n; i++) {
            if (t->type == GT_F32) fv[i] = ((const float *)buf)[i];
            else if (t->type == GT_F16) fv[i] = h2f(((const uint16_t *)buf)[i]);
            else { uint32_t u = (uint32_t)((const uint16_t *)buf)[i] << 16; memcpy(&fv[i], &u, 4); }
        }
        free(buf);
        if (mp->kind == K_IDXK) { // 等 q 那半處理時再拼
            int l = -1;
            sscanf(t->name, "blk.%d.", &l);
            idxk[l] = t;
            idxkg[l] = g;
            free(fv);
            continue;
        }
        st_tensor *o = &cf->tensors[cf->ntensors++];
        o->name = strdup(mp->mlx);
        o->offset = cofs;
        o->data = (char *)cmap + cofs;
        if (mp->kind == K_IDXQ) { // BF16 [512 + 128, 2560]：q 在前、k 在後（與 MLX 的 index_qk_proj 相同）
            int l = -1;
            sscanf(t->name, "blk.%d.", &l);
            idxq[l] = o;
            o->dtype = ST_BF16;
            o->gq = GT_BF16 + 1;
            o->ndim = 2;
            o->shape[0] = (int64_t)rows; // 先填 q 的部分，k 稍後補
            o->shape[1] = (int64_t)t->dims[0];
            bool ok = true;
            for (uint64_t i = 0; i < n; i++) ((uint16_t *)o->data)[i] = f2bf_exact(fv[i], &ok);
            o->nbytes = n * 2;
            cofs += (n * 2 + 16383) / 16384 * 16384 + 4 * 16384; // 留 k 的空間（128 × 2560 × 2 = 640 KB 以內另外加）
            cofs += (uint64_t)128 * t->dims[0] * 2;
            cofs = (cofs + 16383) / 16384 * 16384;
            free(fv);
            continue;
        }
        bool exact = true;
        for (uint64_t i = 0; i < n; i++) (void)f2bf_exact(fv[i], &exact);
        o->ndim = (int)t->nd;
        for (uint32_t k = 0; k < t->nd; k++) o->shape[k] = (int64_t)t->dims[t->nd - 1 - k];
        if (exact && mp->kind != K_ANEG && mp->kind != K_F32) {
            o->dtype = ST_BF16;
            for (uint64_t i = 0; i < n; i++) ((uint16_t *)o->data)[i] = f2bf_exact(fv[i], &exact);
            o->nbytes = n * 2;
        } else {
            o->dtype = ST_F32;
            memcpy((void *)o->data, fv, n * 4);
            o->nbytes = n * 4;
            nconv_f32++;
        }
        cofs += (o->nbytes + 16383) / 16384 * 16384;
        free(fv);
    }
    for (int l = 0; l < QW_LAYERS; l++) { // indexer：k 接在 q 後面
        if (!idxq[l] && !idxk[l]) continue;
        if (!idxq[l] || !idxk[l]) { snprintf(err, errlen, "GGUF：第 %d 層 indexer 只有一半", l); return false; }
        const gtensor *t = idxk[l];
        const uint64_t n = t->dims[0] * t->dims[1], nb = tbytes(t);
        if (t->type != GT_BF16 || t->dims[0] != (uint64_t)idxq[l]->shape[1]) { snprintf(err, errlen, "GGUF：第 %d 層 indexer k 格式不符", l); return false; }
        uint16_t *dst = (uint16_t *)idxq[l]->data + (uint64_t)idxq[l]->shape[0] * (uint64_t)idxq[l]->shape[1];
        if (pread(idxkg[l]->fd, dst, nb, (off_t)(idxkg[l]->data0 + t->off)) != (ssize_t)nb) { snprintf(err, errlen, "GGUF：讀 indexer k 失敗"); return false; }
        idxq[l]->shape[0] += (int64_t)t->dims[1];
        idxq[l]->nbytes += n * 2;
    }
    if (cofs > cf->size) { snprintf(err, errlen, "GGUF：轉換記憶體不足（%llu > %zu）", (unsigned long long)cofs, cf->size); return false; }
    // st_find 用 bsearch：每個檔的 tensor 表要依名稱排序（GGUF 的 blk.2 在 blk.10 前面，換成 MLX 名稱後不是字典序）
    for (int i = 0; i < nnew; i++) qsort(newf[i]->tensors, (size_t)newf[i]->ntensors, sizeof(st_tensor), cmp_tname);
    qsort(cf->tensors, (size_t)cf->ntensors, sizeof(st_tensor), cmp_tname);
    // 放到 st_model 最前面（蓋過 MLX 同名 tensor）
    const int add = nnew + 1;
    st->files = realloc(st->files, sizeof(st_file *) * (size_t)(st->nfiles + add));
    memmove(st->files + add, st->files, sizeof(st_file *) * (size_t)st->nfiles);
    for (int i = 0; i < nnew; i++) st->files[i] = newf[i];
    st->files[nnew] = cf;
    st->nfiles += add;
    m->gguf_parts = parts;
    m->lin_tiled = (parts & QW_GG_LIN) != 0;
    fprintf(stderr, "GGUF profile：%d 個分片、%d 個 tensor（%d 個保留 F32）、部分 0x%x%s\n", nn, nj, nconv_f32, parts,
            m->ple_tab ? "、PLE 表只給 CPU" : "");
    free(jobs);
    for (int i = 0; i < nn; i++) free(names[i]);
    return true;
}
