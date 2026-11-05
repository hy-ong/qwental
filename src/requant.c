// 載入時把 8-bit 的 dense 權重重新量化成 4-bit（MLX affine、group 64），省一半讀取量。
// 新權重放在一個記憶體裡的「虛擬分片」（加在 st_model 的分片清單最後），engine 照常把它包成 Metal buffer。
// 每組 64 個值：先用原本的 8-bit 還原，再取 min / max 定 scale 與 bias（都存成 BF16），q = round((w − bias) / scale)。
// 量化誤差會稍大一點：用環境變數 QW_DENSE4 選哪些類別要轉（見 requant.h）。
#include "requant.h"
#include "ggml_dq.h"
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static float bf2f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

typedef struct {
    qw_qlin *L;
    uint64_t off; // 在虛擬分片裡的位置（w、s、b 依序）
} job;

// 一列：8-bit → 4-bit
static void requant_row(const qw_qlin *L, int64_t r, uint32_t *w4, uint16_t *s4, uint16_t *b4) {
    static int minmax = -1; // QW_RQ_MINMAX：只用 min / max（比較用）
    if (minmax < 0) minmax = getenv("QW_RQ_MINMAX") != NULL;
    const int in = L->in, ng = in / 64;
    const uint32_t *w8 = (const uint32_t *)L->w->data + r * (in / 4);
    const uint16_t *s8 = (const uint16_t *)L->s->data + r * ng, *b8 = (const uint16_t *)L->b->data + r * ng;
    for (int g = 0; g < ng; g++) {
        float v[64], mn = INFINITY, mx = -INFINITY;
        const float sc = bf2f(s8[g]), bi = bf2f(b8[g]);
        for (int i = 0; i < 64; i++) {
            const int k = g * 64 + i;
            v[i] = (float)((w8[k / 4] >> (8 * (k % 4))) & 255u) * sc + bi;
            mn = fminf(mn, v[i]);
            mx = fmaxf(mx, v[i]);
        }
        // 找誤差最小的 (scale, bias)：先試截掉兩端各 0–16% 的範圍，再用最小平方法依目前的 q 重算 scale 與 bias
        float bestE = INFINITY, S = 1, B = 0;
        uint16_t sb = 0, bb = 0;
        const float rg = mx - mn;
        const int na = minmax ? 1 : 5, nit = minmax ? 1 : 3;
        for (int ia = 0; ia < na; ia++)
            for (int ib = 0; ib < na; ib++) {
                float lo = mn + rg * 0.04f * ia, hi = mx - rg * 0.04f * ib;
                float sc0 = (hi - lo) / 15.0f;
                if (!(sc0 > 0)) { sc0 = 1e-8f; lo = mn; }
                for (int it = 0; it < nit; it++) {
                    const uint16_t s16 = f2bf(sc0), b16 = f2bf(lo);
                    const float Sv = bf2f(s16), Bv = bf2f(b16);
                    float err = 0, sq = 0, sv = 0, sqq = 0, sqv = 0;
                    for (int i = 0; i < 64; i++) {
                        int q = (int)lrintf((v[i] - Bv) / Sv);
                        q = q < 0 ? 0 : q > 15 ? 15 : q;
                        const float d = Sv * q + Bv - v[i];
                        err += d * d;
                        sq += q; sv += v[i]; sqq += (float)q * q; sqv += q * v[i];
                    }
                    if (err < bestE) { bestE = err; sb = s16; bb = b16; S = Sv; B = Bv; }
                    const float den = 64 * sqq - sq * sq;
                    if (den <= 0) break;
                    sc0 = (64 * sqv - sq * sv) / den;
                    lo = (sv - sc0 * sq) / 64;
                    if (!(sc0 > 0)) break;
                }
            }
        s4[g] = sb;
        b4[g] = bb;
        for (int i = 0; i < 64; i += 8) {
            uint32_t word = 0;
            for (int j = 0; j < 8; j++) {
                int q = (int)lrintf((v[i + j] - B) / S);
                q = q < 0 ? 0 : q > 15 ? 15 : q;
                word |= (uint32_t)q << (4 * j);
            }
            w4[(g * 64 + i) / 8] = word;
        }
    }
}

// 一列 float → 每列對稱 8-bit：s = max|w| / 127，q = round(w / s) + 128（u32 打包）；so 每組都存同一個 s，
// bo（可為 NULL）存 −128·s，rs = Σ(q − 128)
static void c8_row(const float *v, int in, uint32_t *w8o, uint16_t *so, uint16_t *bo, int32_t *rs) {
    float amax = 0;
    for (int k = 0; k < in; k++) amax = fmaxf(amax, fabsf(v[k]));
    const uint16_t sb = f2bf(amax > 0 ? amax / 127.0f : 1e-8f);
    const float S = bf2f(sb);
    for (int g = 0; g < in / 64; g++) {
        so[g] = sb;
        if (bo) bo[g] = f2bf(-128.0f * S);
    }
    int32_t sum = 0;
    for (int k = 0; k < in; k += 4) {
        uint32_t word = 0;
        for (int j = 0; j < 4; j++) {
            int q = (int)lrintf(v[k + j] / S);
            q = q < -127 ? -127 : q > 127 ? 127 : q;
            word |= (uint32_t)(q + 128) << (8 * j);
            sum += q;
        }
        w8o[k / 4] = word;
    }
    *rs = sum;
}

// 一列：8-bit（group 64）→ 每列對稱 8-bit：s = max|w| / 127，q = round(w / s) + 128；每組都存同一個 s、bias = −128·s
// （仍是 MLX 8-bit 格式，decode kernel 不用改；prefill 可以整列只乘一次 scale，走 INT8 × INT8）
static void requant_row_c8(const qw_qlin *L, int64_t r, uint32_t *w8o, uint16_t *so, uint16_t *bo, int32_t *rs) {
    const int in = L->in, ng = in / 64, src_bits = L->bits;
    const uint32_t *wi = (const uint32_t *)L->w->data + r * (in * src_bits / 32);
    const uint16_t *si = (const uint16_t *)L->s->data + r * ng, *bi = (const uint16_t *)L->b->data + r * ng;
    float *v = malloc(sizeof(float) * (size_t)in);
    const int per = 32 / src_bits;
    for (int k = 0; k < in; k++) {
        const uint32_t q = (wi[k / per] >> (src_bits * (k % per))) & ((1u << src_bits) - 1);
        v[k] = (float)q * bf2f(si[k / 64]) + bf2f(bi[k / 64]);
    }
    static int simb = -1; // 品質評估：QW_SIM_BITS=N → 先模擬 N-bit（group 64、affine，MLX 格式）再存成 8-bit
    if (simb < 0) simb = getenv("QW_SIM_BITS") ? atoi(getenv("QW_SIM_BITS")) : 0;
    if (simb >= 2 && simb < 8)
        for (int g = 0; g < ng; g++) {
            float lo = v[g * 64], hi = lo;
            for (int k = 1; k < 64; k++) { lo = fminf(lo, v[g * 64 + k]); hi = fmaxf(hi, v[g * 64 + k]); }
            const float qs = bf2f(f2bf((hi - lo) / (float)((1 << simb) - 1))), qb = bf2f(f2bf(lo));
            for (int k = 0; k < 64; k++) {
                float q = qs > 0 ? rintf((v[g * 64 + k] - qb) / qs) : 0;
                q = q < 0 ? 0 : q > (float)((1 << simb) - 1) ? (float)((1 << simb) - 1) : q;
                v[g * 64 + k] = q * qs + qb;
            }
        }
    c8_row(v, in, w8o, so, bo, rs);
    free(v);
}

static bool want(const char *sel, const char *cat) {
    if (!strcmp(sel, "1") || !strcmp(sel, "all")) return strcmp(cat, "shared") != 0; // 共用 expert 的 decode kernel 寫死 8-bit
    return strstr(sel, cat) != NULL;
}

bool qw_requant4(qw_model *m, const char *sel, char *err, size_t errlen) {
    const bool c8 = !strncmp(sel, "c8:", 3); // 每列對稱 8-bit（QW_DENSE4=c8:類別…）
    if (c8) sel += 3;
    const int ob = c8 ? 8 : 4;
    job jobs[2048];
    int nj = 0;
    #define ADD(cat, q) do { if (want(sel, cat) && (q).bits == 8 && (q).w) jobs[nj++] = (job){&(q), 0}; } while (0)
    for (int l = 0; l < QW_LAYERS; l++) {
        qw_layer *L = &m->layers[l];
        ADD("mixdown", L->attn_hc.mix_down);
        ADD("mixdown", L->mlp_hc.mix_down);
        if (L->full_attn) {
            ADD("attn", L->att.q); ADD("attn", L->att.k); ADD("attn", L->att.v); ADD("attn", L->att.o); ADD("attn", L->att.idx_qk);
        } else {
            ADD("lin", L->lin.in_qkv); ADD("lin", L->lin.in_z); ADD("lin", L->lin.in_a); ADD("lin", L->lin.in_b); ADD("lin", L->lin.out);
        }
        ADD("shared", L->moe.sh_gate); ADD("shared", L->moe.sh_up); ADD("shared", L->moe.sh_down);
    }
    ADD("mixdown", m->mixer.mix_down);
    ADD("lm", m->lm_head);
    if (m->has_mtp) {
        ADD("mtp", m->mtp.att.q); ADD("mtp", m->mtp.att.k); ADD("mtp", m->mtp.att.v); ADD("mtp", m->mtp.att.o); ADD("mtp", m->mtp.att.idx_qk);
        ADD("mtp", m->mtp_fc_emb); ADD("mtp", m->mtp_fc_hidden);
        ADD("mtp", m->mtp.attn_hc.mix_down); ADD("mtp", m->mtp.mlp_hc.mix_down); ADD("mtp", m->mtp_mixer.mix_down);
    }
    #undef ADD
    if (!nj) return true;
    // 版面：每個 tensor 16 KB 對齊
    uint64_t tot = 0;
    for (int i = 0; i < nj; i++) {
        const qw_qlin *L = jobs[i].L;
        const uint64_t rows = (uint64_t)L->ne * L->out;
        jobs[i].off = tot;
        tot += (rows * (L->in * ob / 32) * 4 + 16383) / 16384 * 16384;
        tot += 2 * ((rows * (L->in / 64) * 2 + 16383) / 16384 * 16384);
        if (c8) tot += (rows * 4 + 16383) / 16384 * 16384;
    }
    void *map = mmap(NULL, tot, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (map == MAP_FAILED) { snprintf(err, errlen, "重新量化：配置 %.1f GB 失敗", tot / 1e9); return false; }
    st_file *f = calloc(1, sizeof *f);
    f->path = strdup("(4-bit 重新量化)");
    f->fd = -1;
    f->map = map;
    f->size = tot;
    f->tensors = calloc((size_t)nj * 4, sizeof(st_tensor));
    f->ntensors = nj * (c8 ? 4 : 3);
    for (int i = 0; i < nj; i++) {
        qw_qlin *L = jobs[i].L;
        const uint64_t rows = (uint64_t)L->ne * L->out;
        const uint64_t wb = rows * (L->in * ob / 32) * 4, sbn = rows * (L->in / 64) * 2;
        const uint64_t ow = jobs[i].off, os = ow + (wb + 16383) / 16384 * 16384, obo = os + (sbn + 16383) / 16384 * 16384;
        st_tensor *t = &f->tensors[(c8 ? 4 : 3) * i];
        t[0] = (st_tensor){.name = strdup(L->w->name), .dtype = ST_U32, .ndim = 2, .shape = {(int64_t)rows, L->in * ob / 32}, .offset = ow, .nbytes = wb, .data = (char *)map + ow};
        t[1] = (st_tensor){.name = strdup(L->s->name), .dtype = ST_BF16, .ndim = 2, .shape = {(int64_t)rows, L->in / 64}, .offset = os, .nbytes = sbn, .data = (char *)map + os};
        t[2] = (st_tensor){.name = strdup(L->b->name), .dtype = ST_BF16, .ndim = 2, .shape = {(int64_t)rows, L->in / 64}, .offset = obo, .nbytes = sbn, .data = (char *)map + obo};
        if (c8) {
            const uint64_t orr = obo + (sbn + 16383) / 16384 * 16384;
            t[3] = (st_tensor){.name = strdup(L->w->name), .dtype = ST_I32, .ndim = 1, .shape = {(int64_t)rows}, .offset = orr, .nbytes = rows * 4, .data = (char *)map + orr};
        }
        const qw_qlin src = *L;
        dispatch_apply((size_t)rows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
            if (c8) requant_row_c8(&src, (int64_t)r, (uint32_t *)t[0].data + r * (src.in / 4), (uint16_t *)t[1].data + r * (src.in / 64),
                                   (uint16_t *)t[2].data + r * (src.in / 64), (int32_t *)t[3].data + r);
            else requant_row(&src, (int64_t)r, (uint32_t *)t[0].data + r * (src.in / 8), (uint16_t *)t[1].data + r * (src.in / 64),
                             (uint16_t *)t[2].data + r * (src.in / 64));
        });
        L->w = &t[0];
        L->s = &t[1];
        L->b = &t[2];
        L->bits = ob;
        L->chan = c8;
        L->rsum_t = c8 ? &t[3] : NULL;
    }
    st_model *st = m->st;
    st->files = realloc(st->files, sizeof(st_file *) * (size_t)(st->nfiles + 1));
    st->files[st->nfiles++] = f;
    fprintf(stderr, "重新量化成 %s：%d 個矩陣，%.2f GB\n", c8 ? "每列對稱 8-bit" : "4-bit", nj, tot / 1e9);
    return true;
}

// GGUF dense 權重的 INT8 影子（QW_INT8=1）：CPU 解碼每一列（ggml_dq.c），再轉成每列對稱 8-bit（同 c8_row）。
// 只給 prefill 的 pf_qmm_i8 用：原本的 GGUF 區塊不動（decode / 驗證照舊），影子只存 W（[in][out]，kn）、S、rsum（沒有 bias）
static bool gg_c8_ok(const qw_qlin *L) {
    return L->gq && L->w && L->ne == 1 && L->in % 64 == 0 && L->out % 64 == 0 && L->in <= 10240 && qw_ggml_dq_ok(L->gq - 1) &&
           L->w->nbytes % (uint64_t)L->out == 0;
}

bool qw_gguf_c8(qw_model *m, const char *sel, char *err, size_t errlen) {
    qw_qlin *jobs[512];
    int nj = 0;
    #define ADD(cat, q) do { if (strstr(sel, cat) && gg_c8_ok(&(q)) && nj < 512) jobs[nj++] = &(q); } while (0)
    for (int l = 0; l < QW_LAYERS; l++) {
        qw_layer *L = &m->layers[l];
        if (L->full_attn) {
            ADD("attn", L->att.q); ADD("attn", L->att.k); ADD("attn", L->att.v); ADD("attn", L->att.o); ADD("attn", L->att.idx_qk);
        } else {
            ADD("lin", L->lin.in_qkv); ADD("lin", L->lin.in_z); ADD("lin", L->lin.out);
        }
        ADD("shared", L->moe.sh_gate); ADD("shared", L->moe.sh_up); ADD("shared", L->moe.sh_down);
        // hyper-connection 的 mix_down（BF16）：只在清單明確含 hyper 時（QW_INT8=1 的預設集合不含：會改變精度）。
        // mix_up 不轉（輸入有離群值，INT8 品質差太多；見 kernels/hyper_gg.metal）
        ADD("hyper", L->attn_hc.mix_down); ADD("hyper", L->mlp_hc.mix_down);
    }
    #undef ADD
    if (!nj) return true;
    #define AL(x) (((uint64_t)(x) + 16383) / 16384 * 16384)
    uint64_t tot = 0, off[512];
    for (int i = 0; i < nj; i++) {
        const qw_qlin *L = jobs[i];
        off[i] = tot;
        tot += AL((uint64_t)L->out * L->in) + AL((uint64_t)L->out * (L->in / 64) * 2) + AL((uint64_t)L->out * 4);
    }
    void *map = mmap(NULL, tot, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (map == MAP_FAILED) { snprintf(err, errlen, "GGUF INT8：配置 %.1f GB 失敗", tot / 1e9); return false; }
    st_file *f = calloc(1, sizeof *f);
    f->path = strdup("(GGUF INT8 影子)");
    f->fd = -1;
    f->map = map;
    f->size = tot;
    f->tensors = calloc((size_t)nj * 3, sizeof(st_tensor));
    f->ntensors = nj * 3;
    qw_qlin *sh = calloc((size_t)nj, sizeof *sh);
    for (int i = 0; i < nj; i++) {
        qw_qlin *L = jobs[i];
        const uint64_t rows = (uint64_t)L->out, wb = rows * L->in, sbn = rows * (L->in / 64) * 2;
        const uint64_t ow = off[i], os = ow + AL(wb), orr = os + AL(sbn);
        st_tensor *t = &f->tensors[3 * i];
        t[0] = (st_tensor){.name = strdup(L->w->name), .dtype = ST_U32, .ndim = 2, .shape = {L->in, (int64_t)rows / 4}, .offset = ow, .nbytes = wb, .data = (char *)map + ow};
        t[1] = (st_tensor){.name = strdup(L->w->name), .dtype = ST_BF16, .ndim = 2, .shape = {(int64_t)rows, L->in / 64}, .offset = os, .nbytes = sbn, .data = (char *)map + os};
        t[2] = (st_tensor){.name = strdup(L->w->name), .dtype = ST_I32, .ndim = 1, .shape = {(int64_t)rows}, .offset = orr, .nbytes = rows * 4, .data = (char *)map + orr};
        const int type = L->gq - 1, in = L->in;
        const uint64_t rb = L->w->nbytes / rows; // 每列位元組
        const uint8_t *src = (const uint8_t *)L->w->data;
        __block bool bad = false;
        // W 存成 [in][out]（K 主序、輸出連續）：prefill 的右運算元不用轉置，tensor 運算快約 12%（kernels/qmm_i8.metal 的 WT）。
        // 先照列解碼成 [out][in] 的暫存（每列一個工作），再每 64 個 k 一個工作轉置（各工作寫不同的輸出列）
        uint8_t *rw = malloc(wb), *wt = (uint8_t *)t[0].data;
        dispatch_apply((size_t)rows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
            float *v = malloc(sizeof(float) * (size_t)in);
            if (!qw_ggml_dq_row(type, src + r * rb, v, in)) bad = true;
            else c8_row(v, in, (uint32_t *)(rw + r * in), (uint16_t *)t[1].data + r * (in / 64), NULL, (int32_t *)t[2].data + r);
            free(v);
        });
        dispatch_apply((size_t)(in / 64), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t kb) {
            for (size_t n = 0; n < rows; n++)
                for (size_t k = kb * 64; k < kb * 64 + 64; k++) wt[k * rows + n] = rw[n * in + k];
        });
        free(rw);
        if (bad) { snprintf(err, errlen, "GGUF INT8：%s 解碼失敗（type %d）", L->w->name, type); return false; }
        sh[i] = (qw_qlin){.w = &t[0], .s = &t[1], .b = NULL, .bits = 8, .out = L->out, .in = L->in, .ne = 1, .chan = true, .rsum_t = &t[2], .kn = true};
        L->i8 = &sh[i];
    }
    #undef AL
    st_model *st = m->st;
    st->files = realloc(st->files, sizeof(st_file *) * (size_t)(st->nfiles + 1));
    st->files[st->nfiles++] = f;
    fprintf(stderr, "GGUF INT8 影子（prefill 用）：%d 個矩陣，%.2f GB\n", nj, tot / 1e9);
    return true;
}
