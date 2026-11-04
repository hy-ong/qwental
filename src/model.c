// 模型權重對照：依層號組出 tensor 名稱，找到並檢查形狀（說明見 model.h）
#include "model.h"
#include "gguf.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PFX "language_model.model."

typedef struct {
    qw_model *m;
    char *err;
    size_t errlen;
    bool ok;
} ctx;

static void fail(ctx *c, const char *fmt, ...) {
    if (!c->ok) return; // 只記第一個錯誤
    c->ok = false;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->err, c->errlen, fmt, ap);
    va_end(ap);
}

static const st_tensor *find(ctx *c, const char *name) {
    const st_tensor *t = st_model_find(c->m->st, name, NULL);
    if (!t) {
        fail(c, "找不到 tensor：%s", name);
        return NULL;
    }
    c->m->total_bytes += t->nbytes;
    c->m->ntensors_used++;
    return t;
}

// 不量化的 tensor，檢查 dtype 與元素總數
static qw_t raw(ctx *c, st_dtype dt, int64_t numel, const char *fmt, ...) {
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    const st_tensor *t = find(c, name);
    if (!t) return NULL;
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    if (t->dtype != dt || n != numel)
        fail(c, "%s：預期 %s × %lld，實際 %s × %lld", name, st_dtype_name(dt), (long long)numel, st_dtype_name(t->dtype),
             (long long)n);
    return t;
}

// BF16 或 F32（GGUF profile 的 F32 小 tensor 只有恰為 BF16 時才轉成 BF16，否則保留 F32，例如 PLE 的 norm）
static qw_t raw_bf_f32(ctx *c, int64_t numel, const char *fmt, ...) {
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    const st_tensor *t = st_model_find(c->m->st, name, NULL);
    return raw(c, t && t->dtype == ST_F32 ? ST_F32 : ST_BF16, numel, "%s", name);
}

// 量化線性層：weight [ne?, out, in·bits/32] U32、scales/biases [ne?, out, in/64] BF16
static qw_qlin qlin(ctx *c, int bits, int out, int in, int ne, const char *fmt, ...) {
    char base[200], name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(base, sizeof base, fmt, ap);
    va_end(ap);
    qw_qlin q = {NULL, NULL, NULL, bits, out, in, ne, false, NULL, 0};
    snprintf(name, sizeof name, "%s.weight", base);
    q.w = find(c, name);
    if (q.w && q.w->gq) { // GGUF 格式：原始區塊（或 BF16），沒有 scales / biases
        q.gq = q.w->gq;
        q.bits = 0;
        int qb = 1, bb = 2;
        if (q.gq != 31 && !qw_ggml_block(q.gq - 1, &qb, &bb)) fail(c, "%s：不認得的 ggml type %d", base, q.gq - 1);
        const int64_t rb = q.gq == 31 ? (int64_t)in : (int64_t)in / qb * bb;
        if (q.w->ndim != 2 || q.w->shape[0] != (int64_t)ne * out || q.w->shape[1] != rb || in % qb)
            fail(c, "%s：GGUF 形狀不符（預期 [%d, %lld]，實際 [%lld, %lld]）", base, ne * out, (long long)rb, (long long)q.w->shape[0],
                 (long long)(q.w->ndim > 1 ? q.w->shape[1] : 0));
        return q;
    }
    snprintf(name, sizeof name, "%s.scales", base);
    q.s = find(c, name);
    snprintf(name, sizeof name, "%s.biases", base);
    q.b = find(c, name);
    if (!q.w || !q.s || !q.b) return q;
    const int64_t want_w = (int64_t)ne * out * (in * bits / 32), want_s = (int64_t)ne * out * (in / 64);
    int64_t nw = 1, ns = 1;
    for (int i = 0; i < q.w->ndim; i++) nw *= q.w->shape[i];
    for (int i = 0; i < q.s->ndim; i++) ns *= q.s->shape[i];
    if (q.w->dtype != ST_U32 || nw != want_w)
        fail(c, "%s.weight：預期 U32 × %lld（%d-bit、%d×%d×%d），實際 %s × %lld", base, (long long)want_w, bits, ne, out, in,
             st_dtype_name(q.w->dtype), (long long)nw);
    if (q.s->dtype != ST_BF16 || ns != want_s || q.b->dtype != ST_BF16)
        fail(c, "%s.scales/biases：預期 BF16 × %lld，實際 %s × %lld", base, (long long)want_s, st_dtype_name(q.s->dtype),
             (long long)ns);
    return q;
}

static void hyper(ctx *c, qw_hyper *h, bool inject, const char *base) {
    const int hcd = QW_HC * QW_HIDDEN;
    h->mix_down = qlin(c, 8, QW_HC_LOWRANK, hcd, 1, "%s.input_mix_weight_down", base);
    h->mix_up = qlin(c, 8, hcd, QW_HC_LOWRANK, 1, "%s.input_mix_weight_up", base);
    // GGUF：mix_down / mix_up 是 BF16、hc_norm 是 F32（不捨入成 BF16），kernel 只認這個組合
    const bool gg = h->mix_down.gq || h->mix_up.gq;
    if (gg && (h->mix_down.gq != 31 || h->mix_up.gq != 31)) fail(c, "%s：GGUF 的 mix_down / mix_up 必須都是 BF16", base);
    h->hc_norm = raw(c, gg ? ST_F32 : ST_BF16, hcd, "%s.hc_norm.weight", base);
    h->inject = inject ? raw(c, ST_BF16, (int64_t)QW_HC * hcd, "%s.block_inject_weight.weight", base) : NULL;
}

// attention norm 的型別：看 q_norm 是 F32 還是 BF16（其他三個由 raw() 檢查要相同）
static st_dtype attn_norm_dtype(ctx *c, const char *p) {
    char name[256];
    snprintf(name, sizeof name, "%s.self_attn.q_norm.weight", p);
    const st_tensor *t = st_model_find(c->m->st, name, NULL);
    return t && t->dtype == ST_F32 ? ST_F32 : ST_BF16;
}

// iq_experts.json（{"<名稱>": ggml type}）裡某層某個投影的格式；沒有就 0
static int iq_fmt(const char *dir, const char *layer, const char *proj) {
    static char *txt = NULL;
    static char loaded_dir[1024] = "";
    if (strcmp(loaded_dir, dir)) {
        free(txt);
        txt = NULL;
        snprintf(loaded_dir, sizeof loaded_dir, "%s", dir);
        char path[1100];
        snprintf(path, sizeof path, "%s/iq_experts.json", dir);
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            const long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            txt = calloc((size_t)n + 1, 1);
            if (fread(txt, 1, (size_t)n, f) != (size_t)n) { free(txt); txt = NULL; }
            fclose(f);
        }
    }
    if (!txt) return 0;
    char key[300];
    snprintf(key, sizeof key, "\"%s.mlp.switch_mlp.%s_proj.iq\"", layer, proj);
    const char *k = strstr(txt, key);
    if (!k) return 0;
    k = strchr(k + strlen(key), ':');
    return k ? atoi(k + 1) : 0;
}

qw_model *qw_model_open(const char *dir, char *err, size_t errlen) {
    st_model *st = st_model_open(dir, err, errlen);
    if (!st) return NULL;
    qw_model *m = calloc(1, sizeof *m);
    m->st = st;
    if (getenv("QW_GGUF") && !qw_gguf_attach(st, m, getenv("QW_GGUF"), qw_gguf_parse_parts(getenv("QW_GGUF_PARTS")), err, errlen)) {
        st_model_close(st);
        free(m);
        return NULL;
    }
    ctx c = {m, err, errlen, true};
    const int D = QW_HIDDEN, hcd = QW_HC * QW_HIDDEN;

    // embed_tokens 是量化的（位元數由形狀推回）
    m->embed_w = find(&c, PFX "embed_tokens.weight");
    if (m->embed_w && m->embed_w->gq) { // GGUF：原始區塊 [VOCAB, 每列位元組]
        m->embed_gq = m->embed_w->gq;
        if (m->embed_w->shape[0] != QW_VOCAB) fail(&c, "embed_tokens（GGUF）列數不符：%lld", (long long)m->embed_w->shape[0]);
    } else {
    m->embed_s = find(&c, PFX "embed_tokens.scales");
    m->embed_b = find(&c, PFX "embed_tokens.biases");
    if (m->embed_w && m->embed_w->ndim == 2) m->embed_bits = (int)(m->embed_w->shape[1] * 32 / D);
    }
    if (!m->embed_gq && m->embed_w && (m->embed_w->shape[0] != QW_VOCAB || (m->embed_bits != 4 && m->embed_bits != 8)))
        fail(&c, "embed_tokens 形狀不符：[%lld, %lld]", (long long)m->embed_w->shape[0], (long long)m->embed_w->shape[1]);

    for (int i = 0; i < QW_LAYERS && c.ok; i++) {
        qw_layer *L = &m->layers[i];
        char p[96];
        snprintf(p, sizeof p, PFX "layers.%d", i);
        L->full_attn = (i % 4) == 3;
        char hb[128];
        snprintf(hb, sizeof hb, "%s.attn_hyper_connection", p);
        hyper(&c, &L->attn_hc, true, hb);
        snprintf(hb, sizeof hb, "%s.mlp_hyper_connection", p);
        hyper(&c, &L->mlp_hc, true, hb);
        if (L->full_attn) {
            qw_attn *A = &L->att;
            A->q = qlin(&c, 8, QW_ATT_QH * QW_ATT_HD * 2, D, 1, "%s.self_attn.q_proj", p);
            A->k = qlin(&c, 8, QW_ATT_KVH * QW_ATT_HD, D, 1, "%s.self_attn.k_proj", p);
            A->v = qlin(&c, 8, QW_ATT_KVH * QW_ATT_HD, D, 1, "%s.self_attn.v_proj", p);
            A->o = qlin(&c, 8, D, QW_ATT_QH * QW_ATT_HD, 1, "%s.self_attn.o_proj", p);
            A->idx_qk = qlin(&c, 8, (QW_IDX_H + 1) * QW_IDX_D, D, 1, "%s.self_attn.indexer.index_qk_proj", p);
            // norm 的型別：MLX 是 BF16；GGUF profile 保留 F32（四個要一致，kernel 依 nf32 選版本）
            const st_dtype nt = attn_norm_dtype(&c, p);
            A->nf32 = nt == ST_F32;
            A->q_norm = raw(&c, nt, QW_ATT_HD, "%s.self_attn.q_norm.weight", p);
            A->k_norm = raw(&c, nt, QW_ATT_HD, "%s.self_attn.k_norm.weight", p);
            A->idx_q_norm = raw(&c, nt, QW_IDX_D, "%s.self_attn.indexer.q_layernorm.weight", p);
            A->idx_k_norm = raw(&c, nt, QW_IDX_D, "%s.self_attn.indexer.k_layernorm.weight", p);
            if ((m->gguf_parts & QW_GG_ATTN) && (!A->q.gq || !A->k.gq || !A->v.gq || !A->o.gq || !A->idx_qk.gq || !A->nf32))
                fail(&c, "%s：GGUF profile 的 attention 有 tensor 不是來自 GGUF", p);
        } else {
            qw_deltanet *N = &L->lin;
            const int kd = QW_LIN_KH * QW_LIN_DK, vd = QW_LIN_VH * QW_LIN_DV;
            N->in_qkv = qlin(&c, 8, 2 * kd + vd, D, 1, "%s.linear_attn.in_proj_qkv", p);
            N->in_z = qlin(&c, 8, vd, D, 1, "%s.linear_attn.in_proj_z", p);
            N->in_a = qlin(&c, 8, QW_LIN_VH, D, 1, "%s.linear_attn.in_proj_a", p);
            N->in_b = qlin(&c, 8, QW_LIN_VH, D, 1, "%s.linear_attn.in_proj_b", p);
            N->out = qlin(&c, 8, D, vd, 1, "%s.linear_attn.out_proj", p);
            N->conv1d = raw(&c, ST_BF16, (int64_t)(2 * kd + vd) * QW_LIN_CONV, "%s.linear_attn.conv1d.weight", p);
            if (m->lin_tiled) N->A_neg = raw(&c, ST_F32, QW_LIN_VH, "%s.linear_attn.A_neg", p); // GGUF：−exp(A_log)，F32
            else N->A_log = raw(&c, ST_BF16, QW_LIN_VH, "%s.linear_attn.A_log", p);
            N->dt_bias = raw(&c, ST_BF16, QW_LIN_VH, "%s.linear_attn.dt_bias", p);
            N->norm = raw(&c, ST_BF16, QW_LIN_DV, "%s.linear_attn.norm.weight", p);
        }
        qw_moe *E = &L->moe;
        E->router = qlin(&c, 8, QW_EXPERTS, D, 1, "%s.mlp.gate", p);
        char iqn[256];
        snprintf(iqn, sizeof iqn, "%s.mlp.switch_mlp.gate_proj.iq", p);
        if (st_model_find(m->st, iqn, NULL)) { // 混合模型：expert 是 GGUF 區塊
            E->giq = find(&c, iqn);
            snprintf(iqn, sizeof iqn, "%s.mlp.switch_mlp.up_proj.iq", p);
            E->uiq = find(&c, iqn);
            snprintf(iqn, sizeof iqn, "%s.mlp.switch_mlp.down_proj.iq", p);
            E->diq = find(&c, iqn);
            if (E->giq && E->giq->gq) { // GGUF profile：格式記在 tensor 上
                E->fgu = E->giq->gq - 1;
                E->fd = E->diq ? E->diq->gq - 1 : 0;
                if (!E->uiq || E->uiq->gq - 1 != E->fgu) fail(&c, "%s：gate 與 up 的格式不同", p);
            } else {
            E->fgu = iq_fmt(dir, p, "gate");
            E->fd = iq_fmt(dir, p, "down");
            if (E->fgu != iq_fmt(dir, p, "up") || E->fgu <= 0 || E->fd <= 0) fail(&c, "%s：iq_experts.json 的格式不符", p);
            }
        } else {
            E->gate = qlin(&c, 4, QW_MOE_INTER, D, QW_EXPERTS, "%s.mlp.switch_mlp.gate_proj", p);
            E->up = qlin(&c, 4, QW_MOE_INTER, D, QW_EXPERTS, "%s.mlp.switch_mlp.up_proj", p);
            E->down = qlin(&c, 4, D, QW_MOE_INTER, QW_EXPERTS, "%s.mlp.switch_mlp.down_proj", p);
        }
        E->sh_gate = qlin(&c, 8, QW_SHARED_INTER, D, 1, "%s.mlp.shared_expert.gate_proj", p);
        E->sh_up = qlin(&c, 8, QW_SHARED_INTER, D, 1, "%s.mlp.shared_expert.up_proj", p);
        E->sh_down = qlin(&c, 8, D, QW_SHARED_INTER, 1, "%s.mlp.shared_expert.down_proj", p);
        E->sh_gate_w = raw(&c, ST_BF16, D, "%s.mlp.shared_expert_gate.weight", p);
        L->has_ple = (i == 1); // ple_layer_ids = [2]（從 1 開始數）
        if (L->has_ple) {
            qw_ple *P = &L->ple;
            P->key_proj = qlin(&c, 8, hcd, D, 1, "%s.ple.key_proj", p);
            P->value_proj = qlin(&c, 8, D, D, 1, "%s.ple.value_proj", p);
            P->norm_key = raw_bf_f32(&c, hcd, "%s.ple.norm_key.weight", p);
            P->norm_query = raw_bf_f32(&c, hcd, "%s.ple.norm_query.weight", p);
            P->norm_conv = raw_bf_f32(&c, hcd, "%s.ple.norm_conv.weight", p);
            P->conv1d = raw(&c, ST_BF16, (int64_t)hcd * 4, "%s.ple.conv1d.weight", p);
            P->multipliers = raw(&c, ST_I64, 3, "%s.ple.ple_embedding.layer_multipliers", p);
            P->head_offsets = raw(&c, ST_I64, 16, "%s.ple.ple_embedding.ngram_heads_offsets", p);
            P->head_vocab = raw(&c, ST_I64, 16, "%s.ple.ple_embedding.ngram_heads_vocab_sizes", p);
        }
    }
    if (c.ok) hyper(&c, &m->mixer, false, PFX "hyper_connection_mixer");
    if (c.ok) m->lm_head = qlin(&c, 8, QW_VOCAB, D, 1, "language_model.lm_head");
    // MTP
    // MTP 共用 embed 與 lm_head；GGUF profile 換掉這兩個時不用（Strata 的 GGUF 本來就沒有 MTP）
    if (c.ok && !(m->gguf_parts & (QW_GG_LM | QW_GG_EMBED)) && st_model_find(m->st, "language_model.mtp.fc_embedding.weight", NULL)) {
        const char *p = "language_model.mtp.layers.0";
        qw_layer *L = &m->mtp;
        L->full_attn = true;
        char hb[128];
        snprintf(hb, sizeof hb, "%s.attn_hyper_connection", p);
        hyper(&c, &L->attn_hc, true, hb);
        snprintf(hb, sizeof hb, "%s.mlp_hyper_connection", p);
        hyper(&c, &L->mlp_hc, true, hb);
        qw_attn *A = &L->att;
        A->q = qlin(&c, 8, QW_ATT_QH * QW_ATT_HD * 2, D, 1, "%s.self_attn.q_proj", p);
        A->k = qlin(&c, 8, QW_ATT_KVH * QW_ATT_HD, D, 1, "%s.self_attn.k_proj", p);
        A->v = qlin(&c, 8, QW_ATT_KVH * QW_ATT_HD, D, 1, "%s.self_attn.v_proj", p);
        A->o = qlin(&c, 8, D, QW_ATT_QH * QW_ATT_HD, 1, "%s.self_attn.o_proj", p);
        A->idx_qk = qlin(&c, 8, (QW_IDX_H + 1) * QW_IDX_D, D, 1, "%s.self_attn.indexer.index_qk_proj", p);
        A->q_norm = raw(&c, ST_BF16, QW_ATT_HD, "%s.self_attn.q_norm.weight", p);
        A->k_norm = raw(&c, ST_BF16, QW_ATT_HD, "%s.self_attn.k_norm.weight", p);
        A->idx_q_norm = raw(&c, ST_BF16, QW_IDX_D, "%s.self_attn.indexer.q_layernorm.weight", p);
        A->idx_k_norm = raw(&c, ST_BF16, QW_IDX_D, "%s.self_attn.indexer.k_layernorm.weight", p);
        qw_moe *E = &L->moe;
        E->router = qlin(&c, 8, QW_EXPERTS, D, 1, "%s.mlp.gate", p);
        E->gate = qlin(&c, 4, QW_MOE_INTER, D, QW_EXPERTS, "%s.mlp.switch_mlp.gate_proj", p);
        E->up = qlin(&c, 4, QW_MOE_INTER, D, QW_EXPERTS, "%s.mlp.switch_mlp.up_proj", p);
        E->down = qlin(&c, 4, D, QW_MOE_INTER, QW_EXPERTS, "%s.mlp.switch_mlp.down_proj", p);
        E->sh_gate = qlin(&c, 8, QW_SHARED_INTER, D, 1, "%s.mlp.shared_expert.gate_proj", p);
        E->sh_up = qlin(&c, 8, QW_SHARED_INTER, D, 1, "%s.mlp.shared_expert.up_proj", p);
        E->sh_down = qlin(&c, 8, D, QW_SHARED_INTER, 1, "%s.mlp.shared_expert.down_proj", p);
        E->sh_gate_w = raw(&c, ST_BF16, D, "%s.mlp.shared_expert_gate.weight", p);
        m->mtp_fc_emb = qlin(&c, 8, D, D, 1, "language_model.mtp.fc_embedding");
        m->mtp_fc_hidden = qlin(&c, 8, D, D, 1, "language_model.mtp.fc_hidden");
        m->mtp_norm_emb = raw(&c, ST_BF16, D, "language_model.mtp.pre_fc_norm_embedding.weight");
        m->mtp_norm_hidden = raw(&c, ST_BF16, hcd, "language_model.mtp.pre_fc_norm_hidden.weight");
        hyper(&c, &m->mtp_mixer, false, "language_model.mtp.hyper_connection_mixer");
        m->has_mtp = c.ok;
    }
    if (!c.ok) {
        qw_model_close(m);
        return NULL;
    }
    return m;
}

void qw_model_close(qw_model *m) {
    if (!m) return;
    st_model_close(m->st);
    free(m);
}
