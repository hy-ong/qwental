#include "dflash.h"
#include "gguf.h"
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

typedef struct {
    char name[96], dtype[8];
    int rows, cols;
    uint64_t off, bytes;
} df_ent;

static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}
static float bf2f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float h2f(uint16_t h) {
    _Float16 x;
    memcpy(&x, &h, 2);
    return (float)x;
}

// Q4_0 一列 → MLX 8-bit affine（group 64）：先還原（d·(q − 8)），每 64 個取 min / max
static void q40_row_to_q8(const uint8_t *src, int cols, uint32_t *w, uint16_t *s, uint16_t *b) {
    for (int g = 0; g < cols / 64; g++) {
        float v[64], mn = INFINITY, mx = -INFINITY;
        for (int h = 0; h < 2; h++) {
            const uint8_t *blk = src + (size_t)(g * 2 + h) * 18;
            uint16_t dh;
            memcpy(&dh, blk, 2);
            const float d = h2f(dh);
            for (int i = 0; i < 16; i++) {
                v[h * 32 + i] = d * (float)((blk[2 + i] & 15) - 8);
                v[h * 32 + 16 + i] = d * (float)((blk[2 + i] >> 4) - 8);
            }
        }
        for (int i = 0; i < 64; i++) { mn = fminf(mn, v[i]); mx = fmaxf(mx, v[i]); }
        float sc = (mx - mn) / 255.0f;
        if (!(sc > 0)) sc = 1e-8f;
        const uint16_t sb = f2bf(sc), bb = f2bf(mn);
        const float S = bf2f(sb), B = bf2f(bb);
        s[g] = sb;
        b[g] = bb;
        for (int i = 0; i < 64; i += 4) {
            uint32_t word = 0;
            for (int j = 0; j < 4; j++) {
                int q = (int)lrintf((v[i + j] - B) / S);
                q = q < 0 ? 0 : q > 255 ? 255 : q;
                word |= (uint32_t)q << (8 * j);
            }
            w[(g * 64 + i) / 4] = word;
        }
    }
}

// Q4_0 一列 → 拆成 nibble 陣列（每區塊 16 位元組）與 FP16 d 陣列：數值與位元組數都不變，只是對齊（kernels/df_q4.metal）
static void q40_row_split(const uint8_t *src, int cols, uint8_t *qs, uint16_t *dd) {
    for (int g = 0; g < cols / 32; g++) {
        memcpy(&dd[g], src + (size_t)g * 18, 2);
        memcpy(qs + (size_t)g * 16, src + (size_t)g * 18 + 2, 16);
    }
}

static uint64_t al(uint64_t x) { return (x + 16383) / 16384 * 16384; }

bool qw_dflash_load(qw_model *m, const char *pack, const char *vocab, qw_dflash *d, char *err, size_t errlen) {
    memset(d, 0, sizeof *d);
    FILE *f = fopen(pack, "rb");
    if (!f) { snprintf(err, errlen, "DFlash2：開不了 %s", pack); return false; }
    char magic[4];
    uint32_t ver, ilen;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "DF2P", 4) || fread(&ver, 4, 1, f) != 1 || fread(&ilen, 4, 1, f) != 1) {
        fclose(f);
        snprintf(err, errlen, "DFlash2：%s 不是 DF2P 檔", pack);
        return false;
    }
    char *idx = malloc(ilen + 1);
    if (fread(idx, 1, ilen, f) != ilen) { fclose(f); free(idx); snprintf(err, errlen, "DFlash2：索引讀取失敗"); return false; }
    idx[ilen] = 0;
    fseek(f, 0, SEEK_END);
    const long fsize = ftell(f);
    fclose(f);
    const uint64_t base = 12 + ilen;
    df_ent *ent = calloc(512, sizeof *ent);
    int ne = 0;
    for (char *line = strtok(idx, "\n"); line; line = strtok(NULL, "\n")) {
        char k[64], v[256];
        if (sscanf(line, "cfg %63s %255s", k, v) == 2) {
            if (!strcmp(k, "hidden")) d->H = atoi(v);
            else if (!strcmp(k, "layers")) d->L = atoi(v);
            else if (!strcmp(k, "heads")) d->NH = atoi(v);
            else if (!strcmp(k, "kv_heads")) d->NKV = atoi(v);
            else if (!strcmp(k, "head_dim")) d->HD = atoi(v);
            else if (!strcmp(k, "inter")) d->inter = atoi(v);
            else if (!strcmp(k, "window")) d->win = atoi(v);
            else if (!strcmp(k, "rope_theta")) d->theta = (float)atof(v);
            else if (!strcmp(k, "eps")) d->eps = (float)atof(v);
            else if (!strcmp(k, "block")) d->block = atoi(v);
            else if (!strcmp(k, "conv_group")) d->gs = atoi(v);
            else if (!strcmp(k, "mask_token")) d->mask = atoi(v);
            else if (!strcmp(k, "rank")) d->rank = atoi(v);
            else if (!strcmp(k, "top_k")) d->topk = atoi(v);
            else if (!strcmp(k, "taps")) {
                for (char *p = v; *p && d->ntaps < 8;) { d->taps[d->ntaps++] = (int)strtol(p, &p, 10); if (*p == ',') p++; }
            }
        } else if (ne < 512 && sscanf(line, "t %95s %7s %d %d %llu %llu", ent[ne].name, ent[ne].dtype, &ent[ne].rows, &ent[ne].cols,
                                      (unsigned long long *)&ent[ne].off, (unsigned long long *)&ent[ne].bytes) == 6)
            ne++;
    }
    free(idx);
    if (d->L <= 0 || d->L > DF_MAXL || d->H != 2560 || d->HD != 128 || d->ntaps != 5) {
        free(ent);
        snprintf(err, errlen, "DFlash2：不支援的設定（layers %d、hidden %d）", d->L, d->H);
        return false;
    }
    // 讀整個檔（約 0.5 GB）
    int fd_ok = 0;
    FILE *g = fopen(pack, "rb");
    uint8_t *raw = malloc((size_t)fsize);
    if (g) { fd_ok = fread(raw, 1, (size_t)fsize, g) == (size_t)fsize; fclose(g); }
    if (!fd_ok) { free(raw); free(ent); snprintf(err, errlen, "DFlash2：讀取 %s 失敗", pack); return false; }
    // draft vocab
    FILE *vf = fopen(vocab, "rb");
    if (!vf) { free(raw); free(ent); snprintf(err, errlen, "DFlash2：開不了 %s", vocab); return false; }
    fseek(vf, 0, SEEK_END);
    d->nv = (int)(ftell(vf) / 4);
    fseek(vf, 0, SEEK_SET);
    uint32_t *vids = malloc((size_t)d->nv * 4);
    const bool vok = fread(vids, 4, (size_t)d->nv, vf) == (size_t)d->nv;
    fclose(vf);
    if (!vok) { free(vids); free(raw); free(ent); snprintf(err, errlen, "DFlash2：vocab 讀取失敗"); return false; }
    // 版面
    const qw_qlin *LH = &m->lm_head;
    const int hb = LH->bits, hrw = LH->in * hb / 32;
    // GGUF 的 lm_head（例如 Q6_K）：整列原始區塊照抄（每列 grb 位元組），head 也是同格式的 gq 矩陣
    int qb = 1, bb = 2;
    if (LH->gq && LH->gq != 31 && !qw_ggml_block(LH->gq - 1, &qb, &bb)) {
        free(vids); free(raw); free(ent);
        snprintf(err, errlen, "DFlash2：lm_head 的 ggml type %d 不認得", LH->gq - 1);
        return false;
    }
    const uint64_t grb = LH->gq ? (uint64_t)LH->in / qb * bb : 0;
    // QW_DF_Q8=1：Q4_0 照舊展開成 MLX 8-bit（重新量化，數值與 Q4_0 略有出入；對照用）
    const bool q8 = getenv("QW_DF_Q8") != NULL;
    uint64_t tot = 0;
    uint64_t *where = calloc((size_t)ne, sizeof *where);
    for (int i = 0; i < ne; i++) {
        where[i] = tot;
        const bool cbk = !strncmp(ent[i].name, "candidate_selector.", 19) && strstr(ent[i].name, "_codebook");
        if (cbk && !strcmp(ent[i].dtype, "q4_0")) tot += al((uint64_t)ent[i].rows * (ent[i].cols / 32) * 18); // selq4：codebook 保留 Q4_0 區塊
        else if (!strcmp(ent[i].dtype, "q4_0")) {
            const uint64_t r = (uint64_t)ent[i].rows;
            if (ent[i].cols % 512) { // df_q4 的 kernel 要求
                free(where); free(vids); free(raw); free(ent);
                snprintf(err, errlen, "DFlash2：%s 的欄數 %d 不是 512 的倍數", ent[i].name, ent[i].cols);
                return false;
            }
            if (q8) tot += al(r * ent[i].cols) + 2 * al(r * (ent[i].cols / 64) * 2);
            else tot += al(r * ent[i].cols / 2) + al(r * (ent[i].cols / 32) * 2);
        } else
            tot += al(ent[i].bytes);
    }
    const uint64_t head_off = tot;
    if (LH->gq) tot += al((uint64_t)d->nv * grb);
    else tot += al((uint64_t)d->nv * hrw * 4) + 2 * al((uint64_t)d->nv * (LH->in / 64) * 2);
    const uint64_t vocab_off = tot;
    tot += al((uint64_t)d->nv * 4);
    uint8_t *map = mmap(NULL, tot, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (map == MAP_FAILED) { free(where); free(vids); free(raw); free(ent); snprintf(err, errlen, "DFlash2：記憶體不足"); return false; }
    st_file *sf = calloc(1, sizeof *sf);
    sf->path = strdup("(DFlash2 drafter)");
    sf->fd = -1;
    sf->map = map;
    sf->size = tot;
    sf->tensors = calloc((size_t)ne * 3 + 8, sizeof(st_tensor));
    int nt = 0;
    #define T_(nm, dt, r, c, o, nb) (sf->tensors[nt] = (st_tensor){.name = strdup(nm), .dtype = dt, .ndim = 2, .shape = {r, c}, \
                                     .offset = o, .nbytes = nb, .data = map + (o)}, &sf->tensors[nt++])
    // 名稱 → 結果
    for (int i = 0; i < ne; i++) {
        const df_ent *E = &ent[i];
        const uint8_t *src = raw + base + E->off;
        qw_qlin *Q = NULL;
        const st_tensor **V = NULL;
        int l = -1;
        char sub[96] = "";
        if (sscanf(E->name, "layers.%d.%95s", &l, sub) == 2 && l >= 0 && l < d->L) {
            qw_df_layer *Ly = &d->layer[l];
            if (!strcmp(sub, "self_attn.q_proj.weight")) Q = &Ly->q;
            else if (!strcmp(sub, "self_attn.k_proj.weight")) Q = &Ly->k;
            else if (!strcmp(sub, "self_attn.v_proj.weight")) Q = &Ly->v;
            else if (!strcmp(sub, "self_attn.o_proj.weight")) Q = &Ly->o;
            else if (!strcmp(sub, "mlp.gate_proj.weight")) Q = &Ly->gate;
            else if (!strcmp(sub, "mlp.up_proj.weight")) Q = &Ly->up;
            else if (!strcmp(sub, "mlp.down_proj.weight")) Q = &Ly->down;
            else if (!strcmp(sub, "attention_conv.kernel_projection.weight")) Q = &Ly->akp;
            else if (!strcmp(sub, "mlp_conv.kernel_projection.weight")) Q = &Ly->mkp;
            else if (!strcmp(sub, "input_layernorm.weight")) V = &Ly->in_norm;
            else if (!strcmp(sub, "post_attention_layernorm.weight")) V = &Ly->post_norm;
            else if (!strcmp(sub, "self_attn.q_norm.weight")) V = &Ly->q_norm;
            else if (!strcmp(sub, "self_attn.k_norm.weight")) V = &Ly->k_norm;
            else if (!strcmp(sub, "attention_conv.base_kernel")) V = &Ly->abase;
            else if (!strcmp(sub, "mlp_conv.base_kernel")) V = &Ly->mbase;
        } else if (!strcmp(E->name, "fc.weight")) Q = &d->fc;
        else if (!strcmp(E->name, "candidate_selector.hidden_projection.weight")) Q = &d->hp;
        else if (!strcmp(E->name, "hidden_norm.weight")) V = &d->hidden_norm;
        else if (!strcmp(E->name, "norm.weight")) V = &d->norm;
        else if (!strcmp(E->name, "candidate_selector.predecessor_codebook")) V = &d->pred_cb;
        else if (!strcmp(E->name, "candidate_selector.successor_codebook")) V = &d->succ_cb;
        if (V && !strcmp(E->dtype, "q4_0")) { // selq4 打包檔：selector 的 codebook 是 Q4_0（只讀被選中的列，df_select 逐值解碼）
            const uint64_t nb = (uint64_t)E->rows * (E->cols / 32) * 18;
            st_tensor *t = T_(E->name, ST_U8, E->rows, (E->cols / 32) * 18, where[i], nb);
            memcpy(map + where[i], src, nb);
            *V = t;
            d->cb_q4 = true;
        } else if (!strcmp(E->dtype, "q4_0") && !q8) {
            const int rows = E->rows, cols = E->cols;
            const uint64_t o = where[i], od = o + al((uint64_t)rows * cols / 2);
            st_tensor *tw = T_(E->name, ST_U8, rows, cols / 2, o, (uint64_t)rows * cols / 2);
            st_tensor *td = T_(E->name, ST_F16, rows, cols / 32, od, (uint64_t)rows * (cols / 32) * 2);
            dispatch_apply((size_t)rows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
                q40_row_split(src + r * (size_t)(cols / 32) * 18, cols, (uint8_t *)tw->data + r * (size_t)(cols / 2),
                              (uint16_t *)td->data + r * (size_t)(cols / 32));
            });
            if (Q) *Q = (qw_qlin){tw, td, NULL, 4, rows, cols, 1, false, NULL, 0}; // b = NULL、bits = 4：Q4_0 拆開版（df_q4）
        } else if (!strcmp(E->dtype, "q4_0")) {
            const int rows = E->rows, cols = E->cols;
            const uint64_t o = where[i], os = o + al((uint64_t)rows * cols), ob = os + al((uint64_t)rows * (cols / 64) * 2);
            st_tensor *tw = T_(E->name, ST_U32, rows, cols / 4, o, (uint64_t)rows * cols);
            st_tensor *ts = T_(E->name, ST_BF16, rows, cols / 64, os, (uint64_t)rows * (cols / 64) * 2);
            st_tensor *tb = T_(E->name, ST_BF16, rows, cols / 64, ob, (uint64_t)rows * (cols / 64) * 2);
            dispatch_apply((size_t)rows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
                q40_row_to_q8(src + r * (size_t)(cols / 32) * 18, cols, (uint32_t *)tw->data + r * (cols / 4),
                              (uint16_t *)ts->data + r * (cols / 64), (uint16_t *)tb->data + r * (cols / 64));
            });
            if (Q) *Q = (qw_qlin){tw, ts, tb, 8, rows, cols, 1, false, NULL, 0};
        } else {
            st_tensor *t = T_(E->name, !strcmp(E->dtype, "bf16") ? ST_BF16 : ST_F32, E->rows, E->cols, where[i], E->bytes);
            memcpy(map + where[i], src, E->bytes);
            if (V) *V = t;
        }
    }
    // draft vocab 的 lm_head 列
    for (int i = 0; i < d->nv; i++)
        if (vids[i] >= (uint32_t)LH->out) {
            free(where); free(vids); free(raw); free(ent);
            snprintf(err, errlen, "DFlash2：draft vocab 第 %d 個 token %u 超出 lm_head", i, vids[i]);
            return false;
        }
    if (LH->gq) {
        st_tensor *tw = T_("df.head.weight", ST_U8, d->nv, (int64_t)grb, head_off, (uint64_t)d->nv * grb);
        tw->gq = LH->gq;
        dispatch_apply((size_t)d->nv, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            memcpy((uint8_t *)tw->data + i * grb, (const uint8_t *)LH->w->data + (size_t)vids[i] * grb, grb);
        });
        d->head = (qw_qlin){tw, NULL, NULL, 0, d->nv, LH->in, 1, false, NULL, LH->gq};
        st_tensor *tv = T_("df.vocab", ST_U32, d->nv, 1, vocab_off, (uint64_t)d->nv * 4);
        memcpy((void *)tv->data, vids, (size_t)d->nv * 4);
        d->vocab = tv;
    } else {
        const uint64_t wb = (uint64_t)d->nv * hrw * 4, sb = (uint64_t)d->nv * (LH->in / 64) * 2;
        st_tensor *tw = T_("df.head.weight", ST_U32, d->nv, hrw, head_off, wb);
        st_tensor *ts = T_("df.head.scales", ST_BF16, d->nv, LH->in / 64, head_off + al(wb), sb);
        st_tensor *tb = T_("df.head.biases", ST_BF16, d->nv, LH->in / 64, head_off + al(wb) + al(sb), sb);
        const int ng = LH->in / 64;
        for (int i = 0; i < d->nv; i++) {
            const uint32_t id = vids[i];
            memcpy((uint32_t *)tw->data + (size_t)i * hrw, (const uint32_t *)LH->w->data + (size_t)id * hrw, (size_t)hrw * 4);
            memcpy((uint16_t *)ts->data + (size_t)i * ng, (const uint16_t *)LH->s->data + (size_t)id * ng, (size_t)ng * 2);
            memcpy((uint16_t *)tb->data + (size_t)i * ng, (const uint16_t *)LH->b->data + (size_t)id * ng, (size_t)ng * 2);
        }
        d->head = (qw_qlin){tw, ts, tb, hb, d->nv, LH->in, 1, false, NULL, 0};
        st_tensor *tv = T_("df.vocab", ST_U32, d->nv, 1, vocab_off, (uint64_t)d->nv * 4);
        memcpy(tv->data ? (void *)tv->data : NULL, vids, (size_t)d->nv * 4);
        d->vocab = tv;
    }
    #undef T_
    sf->ntensors = nt;
    st_model *st = m->st;
    st->files = realloc(st->files, sizeof(st_file *) * (size_t)(st->nfiles + 1));
    st->files[st->nfiles++] = sf;
    free(where);
    free(vids);
    free(raw);
    free(ent);
    for (int l = 0; l < d->L; l++)
        if (!d->layer[l].q.w || !d->layer[l].down.w || !d->layer[l].abase || !d->layer[l].k_norm) {
            snprintf(err, errlen, "DFlash2：第 %d 層缺 tensor", l);
            return false;
        }
    if (!d->fc.w || !d->hp.w || !d->pred_cb || !d->succ_cb || !d->norm || !d->hidden_norm) {
        snprintf(err, errlen, "DFlash2：缺 tensor");
        return false;
    }
    fprintf(stderr, "DFlash2 drafter：%d 層、draft vocab %d、%.2f GB\n", d->L, d->nv, tot / 1e9);
    return true;
}
