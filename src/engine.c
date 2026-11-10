// Qwental 推論 engine：decode 路徑（一次一個 token）
// 每一步把 48 層的所有 kernel 編進同一個 command buffer（serial dispatch，依序執行），最後等一次。
// 參考實作：mlx-lm PR #1788 的模型實作。
#include "engine.h"
#include "gguf.h"
#include <os/lock.h>
#include <stdatomic.h>
#include "requant.h"
#include "dflash.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <sys/mman.h>
#include <time.h>

static const char src_engine[] = {
#embed "../kernels/engine.metal"
};
static const char src_qmv[] = {
#embed "../kernels/qmv.metal"
};
static const char src_moe[] = {
#embed "../kernels/moe_decode.metal"
};
static const char src_gdn[] = {
#embed "../kernels/gdn_decode.metal"
};
static const char src_qmm_i8[] = {
#embed "../kernels/qmm_i8.metal"
    , 0};
static const char src_qk_i8[] = {
#embed "../kernels/qk_i8.metal"
    , 0};
static const char src_qmm_a8[] = {
#embed "../kernels/qmm_a8.metal"
    , 0};
static const char src_iq_tables[] = {
#embed "../kernels/iq_tables.metal"
    , 0};
static const char src_moe_iq[] = {
#embed "../kernels/moe_iq.metal"
    , 0};
static const char src_dflash[] = {
#embed "../kernels/dflash.metal"
    , 0};
static const char src_df_q4[] = {
#embed "../kernels/df_q4.metal"
    , 0};
static const char src_qsa_pf[] = {
#embed "../kernels/qsa_pf.metal"
    , 0};
static const char src_qsa_dec[] = {
#embed "../kernels/qsa_dec.metal"
    , 0};
static const char src_qsa[] = {
#embed "../kernels/qsa.metal"
};
static const char src_prefill[] = {
#embed "../kernels/prefill.metal"
};
static const char src_qmm[] = {
#embed "../kernels/qmm.metal"
};
static const char src_gdnp[] = {
#embed "../kernels/gdn.metal"
};
static const char src_moep[] = {
#embed "../kernels/moe_prefill.metal"
};
static const char src_qmvb[] = {
#embed "../kernels/qmvb.metal"
};
static const char src_qmvb_tc[] = {
#embed "../kernels/qmvb_tc.metal"
};
static const char src_qmv_gg[] = {
#embed "../kernels/qmv_gg.metal"
};
static const char src_dn_proj[] = {
#embed "../kernels/dn_proj.metal"
};
static const char src_qmv_ggm[] = {
#embed "../kernels/qmv_ggm.metal"
    , 0};
static const char src_moe_gg[] = {
#embed "../kernels/moe_gg.metal"
    , 0};
static const char src_moe_tc[] = {
#embed "../kernels/moe_tc.metal"
};
static const char src_gg_ple[] = {
#embed "../kernels/gg_ple.metal"
};
static const char src_hyper_gg[] = {
#embed "../kernels/hyper_gg.metal"
};

enum { VBMAX = 8 };
enum { DF_Q4_NSG = 8, DF_Q6B_NSG = 4 }; // kernels/df_q4.metal 矩陣單元版每個 threadgroup 的 SIMD group 數（df_q6tcb 另計）
enum { CKMAX = 8 }; // 每組 KV slot 最多保留幾個狀態檢查點
enum { PF_NCH = 8 }; // 稀疏 prefill 每個 query 的 key 最多切幾段平行算
enum { D = QW_HIDDEN, HCD = QW_HC * QW_HIDDEN, NLIN = 36, NATT = 12, QSA_NCH = 32, EOS_ID = 248044 };
static int pf_size(void) { // prefill 每段最多幾個 token（QW_PF，64 的倍數）
    static int v = 0;
    // 4096：MoE 每個 expert 平均約 64 列，權重解量化攤到兩倍的 token（比 2048 快約 15%）；8192 反而變慢
    if (!v) v = getenv("QW_PF") ? atoi(getenv("QW_PF")) / 64 * 64 : 4096;
    return v;
}
#define PF (pf_size())

struct qw_engine {
    qw_gpu *g;
    qw_model *m;
    qw_buf *fbuf; // 每個分片一個（mmap 直接包成 buffer）
    qw_buf trace;   // 除錯（QW_TRACE）：每個 token、每層兩個點（attention / DeltaNet 之後、MoE 之後）的 h 與該部分輸出
    int trace_tok;  // decode 時寫到第幾個 token 的位置
    int nfiles;
    st_file *ngram; // ngram_table.bin（safetensors 格式）
    int max_ctx, pos;
    // 即時狀態「與從 0 一次 prefill 逐位元相同」到哪個位置：只由起點在 GDN 分塊倍數上的 prefill 段算出時 = pos，decode 過就不是
    // （decode 的 kernel 與 prefill 不同；從不在分塊倍數的位置接續，分塊的切法也不同）。快照記下 exact，跨請求只從這種快照接續
    int st_exact;
    bool seg_exact; // 目前這段 prefill 的起點符合上面的條件（段中檢查點用）
    // 段中檢查點：prefill 時在這些位置（絕對位置）順便存狀態，不必為了檢查點切段
    int pf_snap[16], npf_snap;
    uint32_t seg_q[4]; // 目前這段裡的檢查點（相對位置，GDN 分塊大小的倍數）
    int seg_n;
    qw_buf snS, snConv, snPle; // 暫存：[4] 份 DeltaNet / conv / PLE 狀態
    // DFlash2 drafter（models/dflash2 有打包檔時載入）
    bool df_on, df_tapping, df_mode; // df_mode：這次生成用 DFlash2 猜（不用 MTP）
    int df_ctx_from, df_slot; // df_tapping：這次的 prefill / 驗證要輸出特徵
    qw_dflash df;
    int df_lo, df_done; // 環狀 KV 有效的最小位置、已寫入的位置上限（不含）
    qw_buf dfKr, dfVr, dftap, dfids, dfx, dfxn, dfxh, dfc, dfdyn, dfq, dfk, dfv, dfqr, dfkb, dfvb, dfo, dfof, dfg, dfu, dfgu, dfm,
        dflog, dftv, dfti, dfhp, dfpath, dfprob, dfch, dfcv, dfci, dfatp;
    uint64_t vb_flush; // vb_encode 在這些層（位元）之後先送出已編好的部分（qw_batch_flush），CPU 編碼與 GPU 執行重疊；0 = 不分
    qw_buf dfprm; // df_accept 寫的參數（GPU 決定接受幾個時，猜下一塊的位置與 anchor 由這裡讀）
    qw_kernel *df_k[16];
    int kvcap; // 目前 KV（Kc / Vc / rawk / kblk）每份 attention 的容量（token 數），不夠時 kv_reserve 擴大（kv_grow_cap）
    double last_ms;
    // 殘差流與共用暫存
    qw_buf h, nh, nf, d, d2, u, x, inj, inj2, out, dump; // inj / inj2：inject 部分和 [HC][HC]，輪流使用
    // DeltaNet
    qw_buf qkv, z, ain, bin, conv, gq, gk, la, be, go, gy, conv_st, S;
    // attention
    qw_buf qg, kk, vv, iqk, aq, iq, ao, ay, Kc, Vc, rawk, kblk, part;
    // MoE
    qw_buf rlog, msel, mw, H, eid0;
    qw_buf ggsel; // GGUF MoE：decode 選出的 expert（kernels/moe_gg.metal 的 gg_sel）
    qw_buf ggtab; // GGUF MoE 的解碼查表（gg_tab_init 產生）
    qw_buf ggH;          // GGUF MoE 的 SiLU(gate)·up：[VBMAX][TOPK + 1][I] float（不捨入成 half）
    qw_buf ggcnt, ggpsh; // GGUF MoE：前段 kernel 的完成計數器（用完歸零）、shared down 的結果 [VBMAX][D] float
    qw_buf ggpd;         // GGUF MoE：各 token 各名次的 routed down 結果 [VBMAX][TOPK][D] float
    bool ggtab_ready;
    qw_kernel *ggk[QW_LAYERS][4][5]; // gg_k 的快取：[層][token 數 1、2、4、8][kernel]
    // PLE
    qw_buf emb, keyh, keyf, kn, qn, val, gated, cn, ple_st;
    qw_buf logits;
    // n-gram 用的 token 歷史：hist[0..1] 是起始的兩個 EOS，hist[2 + t] 是第 t 個 token
    int32_t *hist;
    int last_eos; // hist 中、目前 token 之前最近一個 EOS 的位置
    // kernel
    qw_kernel *k_embed, *k_hcnorm, *k_silu_div, *k_hcmix, *k_inject, *k_resid, *k_conv4, *k_gdn_qk, *k_normgate,
        *k_ple_gate, *k_ple_conv, *k_hcnorm_f, *k_h2f, *k_copyf, *k_hc_in, *k_mixup, *k_gdn_in;
    qw_kernel *moe_gateup_f, *moe_down_f, *moe_gateup_f2, *moe_down_f1, *k_embed_g, *k_ngram, *k_argmax1, *k_argmax2;
    qw_buf tok;      // GPU 上的 token 歷史（int32）：tok[0..1] = 起始的兩個 EOS，tok[2 + t] = 第 t 個 token
    qw_buf amv, ami; // argmax 第一階段的部分結果
    qw_buf pa8q, pa8s, pa8r;
    qw_buf pi8x, pi8s, pi8a;
    qw_buf pfq8, pfqs, pfk8, pfks; // attention 核心 Q·Kᵀ 的 INT8 prefill（QW_INT8 含 qk）：q [PF][24][256]、scale；K [max_ctx][2][256]、scale
    qw_buf pqk8, pqks, pkb8, pkbs; // QSA indexer 打分的 INT8 prefill（QW_INT8 含 qk）：iq [IH][PF + 64][ID]、scale；區塊 key [max_ctx / 4 + 65][ID]、scale
    const qw_buf *i8q_src; // pf_i8_share：目前共用量化的輸入；i8q_m、i8q_k ≠ 0 表示 pi8x 已是它的量化結果
    uint32_t i8q_m;
    int i8q_k;
    qw_buf pq8x, pq8s, pq8h, pq8hs; // GGUF routed expert 的 INT8 prefill（QW_INT8）：gate/up 的輸入 xq [P + 64][D]、sx [P + 64]；down 的輸入 [P + 64][I]、[P + 64]
    qw_buf plog2; // router 兩階段的近似 logits
    qw_buf pul, pum, puc; // attention 4 個 query 的區塊聯集：清單、可見遮罩、個數 // dense INT8 prefill：xu [PF][K]、sa [PF]、ΣQa [PF] // prefill MoE 的 INT8 輸入（QW_A8）：Qa [P][D]、Sa [P]、Ra [P][D / 64]
    qw_buf ngbuf;    // ngram_table.bin 整個包成 Metal buffer（QW_NG_GPU=1 才用；否則 Metal 會把 32 GB 整個釘在記憶體裡）
    bool ng_gpu;
    qw_buf ngW, ngS, ngB; // 預設：CPU 依雜湊把需要的列抄到這裡（[位置][16 head] 列）
    // GGUF 的 n-gram 表（m->ple_tab，IQ4_NL）：CPU 解成 float 再拆 hi / lo 兩個 FP16 → ngX [位置][2][D]（見 kernels/gg_ple.metal）
    qw_buf ngX, pleYk, pleYv, valf; // pleYk / pleYv：hi、lo 兩份的 key / value 投影 [2][N] FP32；valf：value FP32 [D]
    int ng_base;
    // prefill 的 n-gram 列預取：GPU 跑第 i 段時，背景把第 i + 1 段的列抄（或解碼）進另一份 buffer（ngW2… / ngX2），
    // 下一段開始時互換。ng_pre_p / ng_pre_n：已預取的位置（tok 的索引）與個數（0 = 沒有）
    qw_buf ngW2, ngS2, ngB2, ngX2;
    int ng_pre_p, ng_pre_n;
    dispatch_group_t ng_grp;
    const int32_t *pf_nx_ids; // qw_engine_prefill：下一段的 token（預取用），pf_nx_T = 0 表示這是最後一段
    int pf_nx_T;
    qw_buf vbuf; // DeltaNet 的 v（合併版 conv 的輸出）
    qw_buf ssp;  // gdn_step 輸出的部分平方和 [48][4]
    // MTP
    qw_buf mh, memb, mnh, mfe, mfh, mlog;
    // MTP 驗證（一次 2 個 token）
    bool vb_ready;
    qw_buf bh, bnh, bnf, bd, bx, binj, binj2, bout, bqkv, bz, ba, bb, bgy, bqg, bkk, bvv, biqk, bay, blog, bmsel, bHu, bH, blg, bres;
    qw_buf Sx[VBMAX], convx[VBMAX], plex[VBMAX]; // 第 t 個 token（t ≥ 1）之後的狀態寫到 [t − 1]（猜中才換過去）
    int vbn;
    qw_buf bgq, bgk, bgv, bla, bbe, bgo, bssp;
    qw_buf bhpd; // GGUF hyper 驗證三段版：各段部分和 pv、sx、r（gh_pv_* 的位移）
    qw_buf baq, biq, bpart, bao;
    qw_buf bmlg;
    qw_buf bmh, bme, bmn, bmfe, bmnh, bmfh; // MTP（多個 token 一起）
    bool bpending;
    double vb_ms;
    // prefix cache：KV 裡目前是哪些 token、兩個狀態快照（prompt 結束、生成結束）
    int32_t *kvtok;
    int kvlen;
    struct qw_snap { int len; unsigned long long age; qw_buf S, conv, ple; int why; bool exact; int tag; } snap[CKMAX]; // why：誰存的（SW_*，QW_VERIFY_REUSE 的紀錄用）
    // 多組 KV slot（不同對話 / 不同用途的請求各自保留快取）：目前用的那組放在上面的欄位，其他的存在這裡
    struct qw_slot {
        bool init;
        qw_buf Kc, Vc, rawk, kblk, tok;
        int kvcap;
        int32_t *hist, *kvtok;
        int kvlen, last_eos;
        struct qw_snap snap[CKMAX];
        unsigned long long used;
        double tuse; // 並發：最後一次用到的時間（qw_now 秒；選 slot / 釋放時估算對話還會不會回來）
        qw_buf S, conv, ple; int pos, exact; // 並發：這個對話自己的 DeltaNet / conv / PLE 狀態與位置（qw_engine_par_*）、st_exact
        qw_buf dfK, dfV;  // 並發（GGUF）：這個對話自己的 DFlash2 環狀 KV
        int dflo, dfdone; // 環狀 KV 的有效範圍 [dflo, dfdone)（一起推測解碼用；dfdone ≤ 0 = 沒有）
    } slots[64]; // 平常最多 8 組（QW_SLOTS）；並發模式最多 64 組（每個同時進行的對話一組）
    int nslots, cur;
    unsigned long long clock;
    char cache_dir[512]; // 持久化前綴快取（SSD）；空字串 = 不用
    long spec_tok;
    qw_kernel *k_norm1, *k_mtp_combine;
    // ---- prefill（每段最多 PF 個 token）
    bool pf_ready;
    qw_buf ph, pnh, pnf, pd, pd2, pu, px, pinj, pinj2, pout;          // hyper-connection / 殘差
    qw_buf pqkv, pz, pa, pb, pq, pk, pv, pla, pbe, po, py;            // DeltaNet
    qw_buf gW, gU, gQA, gK, gG;                                       // 分塊 GDN 中間結果
    qw_buf pqg, pkk, pvv, piqk, paq, pay;                             // attention
    qw_buf piq, pscore, psel, pnsel, ppart, pao, pqa;                 // QSA 稀疏 prefill
    qw_buf prcs; // RoPE 的 (cos, sin) 表 [PF][32]（p_rope_tab），內容是位置 rcs_pos0 起 rcs_T 個
    uint32_t rcs_pos0, rcs_T;
    qw_buf pme, pmn, pmfe, pmnh, pmfh, pmh;                           // MTP 的 prompt 部分（只填 KV cache）
    bool pf_mtp;
    int mtp_T; // 這一段有幾個位置要填 MTP 的 KV（下一個 token 已知的）
    int32_t pf_next; // 呼叫 prefill 時，ids 之後的下一個 token（不知道時 −1）
    qw_buf plog, psgl, pids, pwts, psg, pcnt, poffs, pcur, ptiles, pXs, ppos, pGg, pGu, pH, pDm, pSg, pSu, pSH, pDs; // MoE
    qw_buf pemb, pkeyh, pkeyf, pkn, pqn, pval, pgated, pcn;           // PLE
    qw_buf pvalf;                                                     // PLE（GGUF）：value FP32 [T][D]
    uint32_t max_tiles; // ptiles 的容量（最小的區塊列數算）
    uint32_t ptm, ntiles; // 目前區塊表的區塊列數、區塊數（grouped GEMM 的 grid）
    bool ppending;
    qw_kernel *qmv_xf1, *qmv_xf2, *qmv_xf1_4, *qmv_xf2_4; // 8-bit、4-bit 權重各一份
    bool pending; // 上一個區塊的殘差（out · inj）還沒加進 h（合併到下一個 norm 裡做）
    qw_kernel *moe_router, *moe_select, *moe_gateup, *moe_down, *gdn_step;
    qw_kernel *gdn_step_t, *k_gdn_in_t; // GGUF 的 DeltaNet（V head 交錯順序、A_neg）
    qw_buf dnK, dnG, dnD; // GGUF DeltaNet 驗證延後寫回狀態：每層兩組（輪流）× [VBMAX] 份的 k、log 衰減、δ（gdn_commit 重放用）
    uint64_t dn_pmask;    // 哪些 DeltaNet 層還有待重放的狀態更新（重放 dn_lJ[層] 步）
    uint8_t dn_lJ[64];    // 每層待重放幾步（驗證接受的 token，或 decode 延後寫回累積的 token）
    uint8_t dn_lp[64];    // 每層最近一次驗證寫在哪一組（待重放的從這組讀；下一次驗證寫另一組，合併版重放與寫入才不會互相覆蓋）
    qw_buf dncnt;         // gdn_fstep 每個 head 的完成計數（最後一個 threadgroup 算 gated RMSNorm 並歸零）
    bool dn_vdef;         // 上一輪驗證用了延後寫回
    qw_kernel *gdn_commit_t;
    qw_buf dnP;    // GGUF DeltaNet 投影（tensor 單元版）K 分段的部分和
    qw_buf dntap;  // 除錯：QW_DN_TAP 記錄某一層 DeltaNet 的輸入 / 輸出
    int dntap_n;   // 這個 batch 記了幾列
    qw_kernel *qsa_attn, *qsa_merge; // MLX 的 attention 核心（qsa_attn2 / qsa_merge2）
    // QSA v3（kernels/qsa_dec.metal）：decode 與驗證共用
    qw_kernel *q3_prep[2], *q3_score, *q3_score4, *q3_topk[3], *q4_attn, *q4_merge, *q5_attn;
    qw_buf q3sc, q3sel, q3nsel, q3hist, q4part, q5stat, q5chk;
    qw_kernel *q5_cmp;
};

// ---------------------------------------------------------------- 編譯

static qw_kernel *gf_kernel(qw_engine *e, int nb);
static bool dn_alloc_defer(qw_engine *e);

// QSA attention 核心分幾段（kernels/qsa_dec.metal 的 NCH4；QW_NCH4 量測用）
static int q4_nch(void) { const int v = getenv("QW_NCH4") ? atoi(getenv("QW_NCH4")) : 32; return v >= 8 && v <= 128 ? v : 32; }
// qsa_attn5 一個 threadgroup 幾個 token（kernels/qsa_dec.metal 的 TT；QW_ATT_TT 量測用）
static int q5_tt(void) { const int v = getenv("QW_ATT_TT") ? atoi(getenv("QW_ATT_TT")) : 1; return v >= 1 && v <= 8 ? v : 1; }
// 稀疏 token 的核心：qsa_attn5（QW_ATT4：沿用 qsa_attn4，對照用）
static bool q5_on(void) { static int v = -1; if (v < 0) v = !getenv("QW_ATT4"); return v; }

// QW_EXACT_SPEC=1（server 的 --exact-spec）：MLX profile 的推測解碼逐位元等於 greedy。
// MLX profile 的 kernel 加 QW_STRICT（decode / 驗證用到的 .metal 檔改用 safe 數學模式、qmv / qmvb 共用 chunk 寫法，見 engine.metal 開頭），
// 驗證的 gated norm / output gate 用與 decode 相同的融合算式（vb_xform）。代價：MLX 的 greedy 輸出也跟著變（捨入不同，q300 PPL 1.337 → 1.340）。
// 預設（不開）與 GGUF profile 不受影響：kernel 原始碼前處理後與 v1.0.0 相同（GGUF 的 kernel 在 fast 模式下本來就逐位元一致）
static bool exact_on(const qw_engine *e) {
    static int v = -1;
    if (v < 0) v = getenv("QW_EXACT_SPEC") && strcmp(getenv("QW_EXACT_SPEC"), "0") != 0;
    return v && e->m && !e->m->gguf_parts;
}
static const char *strict_def(const qw_engine *e) { return exact_on(e) && !getenv("QW_NO_STRICT") ? "#define QW_STRICT 1\n" : ""; }
static bool compile(qw_engine *e, const char *defs, const char *src, size_t len, char *err, size_t errlen) {
    const char *sd = strict_def(e);
    size_t sl = strlen(sd), dl = strlen(defs);
    char *buf = malloc(sl + dl + len);
    memcpy(buf, sd, sl);
    memcpy(buf + sl, defs, dl);
    memcpy(buf + sl + dl, src, len);
    bool ok = qw_gpu_compile(e->g, buf, sl + dl + len, err, errlen);
    free(buf);
    return ok;
}

// 多份原始碼接在一起編譯（最後一份的結尾 0 之前都要去掉）
static bool compile3(qw_engine *e, const char *defs, const char *a, size_t al, const char *b, size_t bl, const char *c, size_t cl,
                     char *err, size_t errlen) {
    const char *sd = strict_def(e);
    const size_t sl = strlen(sd), dl = strlen(defs);
    char *buf = malloc(sl + dl + al + bl + cl + 3);
    size_t n = 0;
    memcpy(buf + n, sd, sl); n += sl;
    memcpy(buf + n, defs, dl); n += dl;
    memcpy(buf + n, a, al - 1); n += al - 1; buf[n++] = '\n';
    memcpy(buf + n, b, bl - 1); n += bl - 1; buf[n++] = '\n';
    if (c) { memcpy(buf + n, c, cl - 1); n += cl - 1; }
    buf[n++] = 0;
    const bool ok = qw_gpu_compile(e->g, buf, n, err, errlen);
    free(buf);
    return ok;
}

// ggml 區塊：每塊值數、位元組（BF16 在 dq32 裡以 32 個值 / 64 位元組為一段）
static int iq_qb(int f) { int qb = 32, bb = 64; if (f != 30) qw_ggml_block(f, &qb, &bb); return qb; }
static int iq_bb(int f) { int qb = 32, bb = 64; if (f != 30) qw_ggml_block(f, &qb, &bb); return bb; }
static int gg_rb(const qw_qlin *L) { return L->in / iq_qb(L->gq - 1) * iq_bb(L->gq - 1); } // 每列位元組

// IQ expert 的 MoE kernel（依 gate/up 與 down 的格式、NB 編譯並改名）
static qw_kernel *iq_kernel(qw_engine *e, const char *base, int fgu, int fd, int nb) {
    char name[96], defs[1024], err[8192];
    if (nb) snprintf(name, sizeof name, "%s_%d_%d_n%d", base, fgu, fd, nb);
    else snprintf(name, sizeof name, "%s_%d_%d", base, fgu, fd);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    char sfx[32], nbd[32] = "";
    if (nb) { snprintf(sfx, sizeof sfx, "_%d_%d_n%d", fgu, fd, nb); snprintf(nbd, sizeof nbd, "#define NB %d\n", nb); }
    else snprintf(sfx, sizeof sfx, "_%d_%d", fgu, fd);
    snprintf(defs, sizeof defs,
             "#define E %d\n#define TOPK %d\n#define D %d\n#define I %d\n#define FGU %d\n#define FD %d\n%s"
             "#define moe_gateup_iq moe_gateup_iq%s\n#define moe_down_iq moe_down_iq%s\n#define iq_dequant_row iq_dequant_row%s\n"
             "#define moe_gateup_iqb moe_gateup_iqb%s\n#define moe_down_iqb moe_down_iqb%s\n",
             QW_EXPERTS, QW_TOPK, D, QW_MOE_INTER, fgu, fd, nbd, sfx, sfx, sfx, sfx, sfx);
    if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, NULL, 0, err, sizeof err)) {
        fprintf(stderr, "編譯 IQ MoE（%d / %d）失敗：%s\n", fgu, fd, err);
        exit(1);
    }
    return qw_kernel_get(e->g, name);
}

static qw_kernel *need(qw_engine *e, const char *name, char *err, size_t errlen) {
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) snprintf(err, errlen, "找不到 kernel：%s", name);
    return k;
}

// qmv 變體：qmv_b{bits}_k{K}[_f]
static qw_kernel *qmv_kernel(qw_engine *e, int bits, int K, bool fout) {
    char name[64], defs[1024], err[4096];
    snprintf(name, sizeof name, "qmv_b%d_k%d%s", bits, K, fout ? "_f" : "");
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs,
             "#define qmv %s\n#define qmv_splitk qmv_splitk_b%d_k%d%s\n#define qmv_halfsg qmv_halfsg_b%d_k%d%s\n"
             "#define qmv_multi qmv_multi_b%d_k%d%s\n#define qmv_splitk_inj qmv_splitk_inj_b%d_k%d%s\n"
             "#define BITS %d\n#define K %d\n#define RPS 1\n#define SPT 4\n#define ST bfloat\n%s",
             name, bits, K, fout ? "_f" : "", bits, K, fout ? "_f" : "", bits, K, fout ? "_f" : "", bits, K, fout ? "_f" : "",
             bits, K, fout ? "#define YT float\n" : "");
    if (!compile(e, defs, src_qmv, sizeof src_qmv, err, sizeof err)) {
        fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
        exit(1);
    }
    return qw_kernel_get(e->g, name);
}

// ---------------------------------------------------------------- 權重綁定

static int file_index(qw_engine *e, const st_tensor *t) {
    for (int i = 0; i < e->nfiles; i++) {
        const st_file *f = e->m->st->files[i];
        if (t >= f->tensors && t < f->tensors + f->ntensors) return i;
    }
    fprintf(stderr, "tensor 不屬於任何分片：%s\n", t->name);
    exit(1);
}

// 權重 tensor → Metal buffer：第一次綁定時才把它所在的頁面包成 buffer（每個 tensor 一個）。
// 整個分片包成一個 buffer 時，只要用到其中一個 tensor，Metal 就會把整個分片釘在記憶體裡（GGUF 與 MLX 混用時可達 100 GB）。
// 表固定大小、不重新配置：新增時上鎖，查詢不上鎖（先寫好內容再發布 key）
enum { TB_CAP = 1 << 16 };
typedef struct { _Atomic(const st_tensor *) key; qw_buf buf; size_t off; } tb_ent;
static tb_ent *tb_tab;
static os_unfair_lock tb_lock = OS_UNFAIR_LOCK_INIT;
static void bind_t(qw_engine *e, qw_batch *b, unsigned idx, const st_tensor *t) {
    if (e->fbuf) { qw_batch_buf(b, idx, &e->fbuf[file_index(e, t)], t->offset); return; } // QW_WRAP_FILE=1：整個分片一個 buffer
    if (!tb_tab) {
        os_unfair_lock_lock(&tb_lock);
        if (!tb_tab) tb_tab = calloc(TB_CAP, sizeof *tb_tab);
        os_unfair_lock_unlock(&tb_lock);
    }
    size_t h = ((uintptr_t)t >> 4) * 0x9E3779B97F4A7C15ull >> 48;
    for (;; h = (h + 1) & (TB_CAP - 1)) {
        const st_tensor *k = atomic_load_explicit(&tb_tab[h].key, memory_order_acquire);
        if (k == t) { qw_batch_buf(b, idx, &tb_tab[h].buf, tb_tab[h].off); return; }
        if (k) continue;
        os_unfair_lock_lock(&tb_lock);
        k = atomic_load_explicit(&tb_tab[h].key, memory_order_relaxed);
        if (k) { os_unfair_lock_unlock(&tb_lock); h = (h - 1) & (TB_CAP - 1); continue; } // 別人剛放進來：重看這一格
        static int copy_small = -1; // 除錯：QW_COPY_SMALL=1 → 64 MB 以下的 tensor 複製到長度剛好的 buffer（配合 MTL_SHADER_VALIDATION 抓越界讀取）
        if (copy_small < 0) copy_small = getenv("QW_COPY_SMALL") ? 1 : 0;
        if (copy_small && t->nbytes < (64u << 20)) {
            if (!qw_buf_new(e->g, t->nbytes, &tb_tab[h].buf)) { fprintf(stderr, "配置 %s 失敗\n", t->name); exit(1); }
            memcpy(tb_tab[h].buf.ptr, t->data, t->nbytes);
            tb_tab[h].off = 0;
            atomic_store_explicit(&tb_tab[h].key, t, memory_order_release);
            os_unfair_lock_unlock(&tb_lock);
            qw_batch_buf(b, idx, &tb_tab[h].buf, 0);
            return;
        }
        const size_t pg = (size_t)getpagesize();
        char *p0 = (char *)((uintptr_t)t->data / pg * pg);
        const size_t len = (size_t)((const char *)t->data + t->nbytes - p0);
        if (!qw_buf_wrap(e->g, p0, len, &tb_tab[h].buf)) {
            fprintf(stderr, "無法把 tensor %s 包成 Metal buffer\n", t->name);
            exit(1);
        }
        tb_tab[h].off = (size_t)((const char *)t->data - p0);
        atomic_store_explicit(&tb_tab[h].key, t, memory_order_release);
        os_unfair_lock_unlock(&tb_lock);
        qw_batch_buf(b, idx, &tb_tab[h].buf, tb_tab[h].off);
        return;
    }
}

// ---- GGUF profile 的 MoE（kernels/moe_gg.metal）
// 調參用（kernel 的編譯參數）：環境變數或預設值
static int gg_par(const char *name, int def) { const char *v = getenv(name); return v ? atoi(v) : def; }

// shared expert 的格式代號：ggml type，或 0 = MLX 8-bit affine
static int gg_sfmt(const qw_qlin *L) { return L->gq ? L->gq - 1 : 0; }

// 驗證的 token 數向上取到 1、2、4、8 編譯（實際個數執行時給）：少編幾份，NB 小時又不會因為暫存器用量變慢
static int gg_nbv(int nb) { return nb <= 1 ? 1 : nb <= 2 ? 2 : nb <= 4 ? 4 : 8; }
_Static_assert(VBMAX <= 8, "gg_nbv：驗證最多 8 個 token（kernels/moe_gg.metal 的 GNB）");
// routed gate/up 每個 SIMD group 幾列（同一個單位的 x 只讀一次，給這幾列共用）
static int gg_rg(int nb) { (void)nb; return gg_par("QW_RG", 1); } // 量過：2 列在 NB = 8 時變慢（暫存器）
// routed down 每個 threadgroup 幾個 SIMD group（各 4 列、同一個 expert；量過 4 比 8、16 稍快）
static int gg_dsg(void) { return gg_par("QW_DNSG", 4); }
static int gg_gsg(void) { return gg_par("QW_GSG", 8); }
// moe_gu_gg 的 grid x：routed（I / (GSG·RG)）與 shared down（D / (4·GSG)）取大者
static int gg_gux(int nb) { const int a = QW_MOE_INTER / (gg_gsg() * gg_rg(nb)), c = D / (4 * gg_gsg()); return a > c ? a : c; }

// GGUF MoE 的 kernel：依用到的格式分開編譯（減少組合數），token 數見 gg_nbv
//   part 0：gg_tab_init、moe_sel_gg；1：moe_pre_gg（shared gate / up 的格式、router 是否 BF16）；2：moe_gu_gg（routed gate/up、shared down）；
//   3：moe_dn_gg（routed down）；4：BF16 router（moe_router_bf、p_router_bf*，NB = 實際 token 數）
static qw_kernel *gg_kernel(qw_engine *e, const char *base, const qw_moe *M, int nb) {
    const int part = !strcmp(base, "moe_sel_gg") || !strcmp(base, "gg_tab_init") ? 0 : !strcmp(base, "moe_pre_gg") ? 1 : !strcmp(base, "moe_gu_gg") ? 2
                     : !strcmp(base, "moe_dn_gg") || !strcmp(base, "moe_cmb_gg") ? 3 : 4;
    const int nbv = part == 4 ? nb : gg_nbv(nb);
    char name[128], sfx[96], defs[2048], err[8192];
    switch (part) {
    case 0: snprintf(sfx, sizeof sfx, "_n%d", nbv); break;
    case 1: snprintf(sfx, sizeof sfx, "_%d_%d_r%d_n%d", gg_sfmt(&M->sh_gate), gg_sfmt(&M->sh_up), M->router.gq != 0, nbv); break;
    case 2: snprintf(sfx, sizeof sfx, "_%d_%d_g%d_n%d", M->fgu, gg_sfmt(&M->sh_down), gg_rg(nb), nbv); break;
    case 3: snprintf(sfx, sizeof sfx, "_%d_s%d_n%d", M->fd, gg_dsg(), nbv); break;
    default: snprintf(sfx, sizeof sfx, "_n%d", nbv);
    }
    snprintf(name, sizeof name, "%s%s", base, sfx);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    if (nb < 1 || nb > VBMAX || (part == 3 && (gg_dsg() < 1 || gg_dsg() > 32 || D % (4 * gg_dsg()))) ||
        (part == 2 && QW_MOE_INTER % (gg_gsg() * gg_rg(nb)))) {
        fprintf(stderr, "%s：不支援的參數（nb %d）\n", name, nb);
        exit(1);
    }
    int n = snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY\n#define E %d\n#define TOPK %d\n#define D %d\n#define I %d\n#define NB %d\n", QW_EXPERTS, QW_TOPK, D, QW_MOE_INTER, nbv);
    char dk[16];
    snprintf(dk, sizeof dk, "QW_GG_DEFS%d", part);
    for (int i = 0; i < 2; i++) { // 實驗用：額外的巨集（QW_GG_DEFS 全部、QW_GG_DEFS<part> 只給那一組；例如 "#define X 1;#define Y 2"）
        const char *v = getenv(i ? dk : "QW_GG_DEFS");
        if (!v) continue;
        const int n0 = n;
        n += snprintf(defs + n, sizeof defs - n, "%s\n", v);
        for (char *c = defs + n0; *c; c++) if (*c == ';') *c = '\n';
    }
    switch (part) {
    case 0: n += snprintf(defs + n, sizeof defs - n, "#define GG_PART 0\n#define moe_sel_gg moe_sel_gg%s\n#define gg_tab_init gg_tab_init%s\n", sfx, sfx); break;
    case 1:
        n += snprintf(defs + n, sizeof defs - n, "#define GG_PART 1\n#define FSG %d\n#define FSU %d\n#define RBF %d\n#define moe_pre_gg moe_pre_gg%s\n",
                      gg_sfmt(&M->sh_gate), gg_sfmt(&M->sh_up), M->router.gq != 0, sfx);
        break;
    case 2:
        n += snprintf(defs + n, sizeof defs - n, "#define GG_PART 2\n#define GSG %d\n#define RG %d\n#define FGU %d\n#define FSD %d\n#define moe_gu_gg moe_gu_gg%s\n", gg_gsg(),
                      gg_rg(nb), M->fgu, gg_sfmt(&M->sh_down), sfx);
        break;
    case 3:
        n += snprintf(defs + n, sizeof defs - n, "#define GG_PART 3\n#define FD %d\n#define DSGN %d\n#define moe_dn_gg moe_dn_gg%s\n#define moe_cmb_gg moe_cmb_gg%s\n", M->fd,
                      gg_dsg(), sfx, sfx);
        break;
    default:
        n += snprintf(defs + n, sizeof defs - n, "#define GG_ROUTER\n#define moe_router_bf moe_router_bf%s\n#define p_router_bf p_router_bf%s\n"
                      "#define p_router_bf_refine p_router_bf_refine%s\n", sfx, sfx, sfx);
    }
    // moe_iq.metal（IQ_DQ_ONLY）提供 ldh 等
    if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_moe_gg, sizeof src_moe_gg, err, sizeof err)) {
        fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
        exit(1);
    }
    k = qw_kernel_get(e->g, name);
    if (!k) { fprintf(stderr, "找不到 %s\n", name); exit(1); }
    if (getenv("QW_GG_VERBOSE")) fprintf(stderr, "%s：每個 threadgroup 最多 %u 個執行緒\n", name, qw_kernel_max_threads(k));
    return k;
}

// 每層的 MoE kernel 快取（不必每次 dispatch 都組名字、查表）
static qw_kernel *gg_k(qw_engine *e, const qw_moe *M, int nb, int which) {
    static const char *const nm[5] = {"moe_pre_gg", "moe_gu_gg", "moe_dn_gg", "moe_sel_gg", "moe_cmb_gg"};
    const ptrdiff_t l = ((const char *)M - (const char *)&e->m->layers[0].moe) / (ptrdiff_t)sizeof(qw_layer);
    if (l < 0 || l >= QW_LAYERS || &e->m->layers[l].moe != M) return gg_kernel(e, nm[which], M, nb);
    const int v = gg_nbv(nb);
    qw_kernel **c = &e->ggk[l][v == 1 ? 0 : v == 2 ? 1 : v == 4 ? 2 : 3][which];
    if (!*c) *c = gg_kernel(e, nm[which], M, nb);
    return *c;
}

static bool gg_old(void) { // QW_MOE_OLD=1：routed expert 走舊的 moe_iq.metal kernel（對照用）
    static int v = -1;
    if (v < 0) v = getenv("QW_MOE_OLD") != NULL;
    return v;
}

// 量測用：QW_MOE_SKIP 含 r / s / g / d → 不送 router / 選取 / gate-up / down（輸出會錯，只用來拆時間）
static const char *gg_skipv = (const char *)1;
static bool gg_skip(char c) {
    if (gg_skipv == (const char *)1) gg_skipv = getenv("QW_MOE_SKIP");
    return gg_skipv && strchr(gg_skipv, c);
}

// BF16 router（decode / 驗證）：logits [nb][E + 1]
static void gg_router(qw_engine *e, qw_batch *b, const qw_moe *M, int nb, const qw_buf *x, const qw_buf *lg) {
    if (gg_skip('r')) return;
    qw_batch_kernel(b, gg_kernel(e, "moe_router_bf", M, nb));
    bind_t(e, b, 0, M->router.w);
    bind_t(e, b, 1, M->sh_gate_w);
    qw_batch_buf(b, 2, x, 0);
    qw_batch_buf(b, 3, lg, 0);
    qw_batch_dispatch(b, (QW_EXPERTS + 1 + 7) / 8, 1, 1, 256, 1, 1);
}

// 先把各層 nb 個 token 用到的 GGUF MoE kernel 編好（不讓編譯時間算進生成）
static void gg_precompile(qw_engine *e, int nb) {
    for (int l = 0; l < QW_LAYERS; l++) {
        const qw_moe *M = &e->m->layers[l].moe;
        if (!M->giq) continue;
        for (int w = 0; w < (M->router.gq ? 3 : 4); w++) gg_k(e, M, nb, w);
        gg_k(e, M, nb, 4);
    }
}

static qw_kernel *vk(qw_engine *e, const char *base);
// GGUF MoE（routed expert 是 GGUF；shared 是 GGUF 或 MLX 8-bit；router 是 BF16 或 MLX 8-bit）。nb 個 token：
//   x [nb][D]、lg [nb][E + 1]（router 的 logits）、sel（gg_sel）、out [nb][D]
// 四個 dispatch（kernels/moe_gg.metal）：前段（BF16 router + shared gate/up + 選 expert）→ routed gate/up + shared down → routed down → 加權合併
static void gg_moe(qw_engine *e, qw_batch *b, const qw_moe *M, int nb, const qw_buf *x, const qw_buf *lg, const qw_buf *sel,
                   const qw_buf *out) {
    const qw_buf *H = &e->ggH;
    if (!e->ggtab_ready) { // 第一次：產生查表（同一個 command buffer、在用到之前）
        qw_batch_kernel(b, gg_kernel(e, "gg_tab_init", M, 1));
        qw_batch_buf(b, 0, &e->ggtab, 0);
        qw_batch_dispatch(b, 32, 1, 1, 256, 1, 1); // 8192 個執行緒（kernels/moe_gg.metal 的 gg_tab_init）
        e->ggtab_ready = true;
    }
    const bool rbf = M->router.gq != 0;
    const uint32_t nbr = (uint32_t)nb;
    const qw_qlin *S[3] = {&M->sh_gate, &M->sh_up, &M->sh_down};
    if (!rbf && !gg_skip('r')) { // MLX 8-bit router
        qw_batch_kernel(b, nb == 1 ? e->moe_router : vk(e, "moe_router_b"));
        bind_t(e, b, 0, M->router.w);
        bind_t(e, b, 1, M->router.s);
        bind_t(e, b, 2, M->router.b);
        bind_t(e, b, 3, M->sh_gate_w);
        qw_batch_buf(b, 4, x, 0);
        qw_batch_buf(b, 5, lg, 0);
        qw_batch_dispatch(b, (QW_EXPERTS + 1 + 7) / 8, (unsigned)nb, 1, 256, 1, 1);
    }
    if (!gg_skip('p')) {
        qw_batch_kernel(b, gg_k(e, M, nb, 0));
        bind_t(e, b, 0, rbf ? M->router.w : M->sh_gate_w);
        bind_t(e, b, 1, M->sh_gate_w);
        qw_batch_buf(b, 2, x, 0);
        qw_batch_buf(b, 3, lg, 0);
        qw_batch_buf(b, 4, &e->ggcnt, 0);
        qw_batch_buf(b, 5, sel, 0);
        bind_t(e, b, 6, S[0]->w);
        bind_t(e, b, 7, S[1]->w);
        qw_batch_buf(b, 8, H, 0);
        for (int i = 0; i < 2; i++) { // MLX 8-bit 的 scale / bias；GGUF 用不到（綁權重本身佔位）
            bind_t(e, b, 9 + 2 * i, S[i]->gq ? S[i]->w : S[i]->s);
            bind_t(e, b, 10 + 2 * i, S[i]->gq ? S[i]->w : S[i]->b);
        }
        qw_batch_buf(b, 13, &e->ggtab, 0);
        qw_batch_bytes(b, 14, &nbr, 4);
        qw_batch_dispatch(b, (rbf ? (QW_EXPERTS + 1 + 7) / 8 : 0) + QW_MOE_INTER / 8, 1, 1, 256, 1, 1);
    }
    if (!rbf && !gg_skip('s')) {
        qw_batch_kernel(b, gg_k(e, M, nb, 3));
        qw_batch_buf(b, 0, lg, 0);
        qw_batch_buf(b, 1, sel, 0);
        qw_batch_bytes(b, 2, &nbr, 4);
        qw_batch_dispatch(b, 1, 1, 1, 32u * nb, 1, 1);
    }
    if (!gg_skip('g')) {
        qw_batch_kernel(b, gg_k(e, M, nb, 1));
        bind_t(e, b, 0, M->giq);
        bind_t(e, b, 1, M->uiq);
        bind_t(e, b, 2, S[2]->w);
        qw_batch_buf(b, 3, &e->ggpsh, 0);
        qw_batch_buf(b, 4, x, 0);
        qw_batch_buf(b, 5, sel, 0);
        qw_batch_buf(b, 6, H, 0);
        bind_t(e, b, 7, S[2]->gq ? S[2]->w : S[2]->s);
        bind_t(e, b, 8, S[2]->gq ? S[2]->w : S[2]->b);
        qw_batch_buf(b, 11, &e->ggtab, 0);
        qw_batch_bytes(b, 12, &nbr, 4);
        qw_batch_dispatch(b, (unsigned)gg_gux(nb), (unsigned)(nb * QW_TOPK + 1), 1, 32u * gg_gsg(), 1, 1);
    }
    if (gg_skip('d')) return;
    // routed down（每個 threadgroup 一個 expert 的 4·DSGN 列）→ Pg，再加權合併
    qw_batch_kernel(b, gg_k(e, M, nb, 2));
    bind_t(e, b, 0, M->diq);
    qw_batch_buf(b, 2, H, 0);
    qw_batch_buf(b, 3, sel, 0);
    qw_batch_buf(b, 7, &e->ggtab, 0);
    qw_batch_buf(b, 9, &e->ggpd, 0);
    qw_batch_dispatch(b, D / (4 * gg_dsg()), (unsigned)(nb * QW_TOPK), 1, 32u * gg_dsg(), 1, 1);
    qw_batch_kernel(b, gg_k(e, M, nb, 4));
    qw_batch_buf(b, 1, &e->ggpsh, 0);
    qw_batch_buf(b, 3, sel, 0);
    qw_batch_buf(b, 4, out, 0);
    qw_batch_bytes(b, 8, &nbr, 4);
    qw_batch_buf(b, 9, &e->ggpd, 0);
    qw_batch_dispatch(b, D / 256, 1, 1, 256, 1, 1);
}

typedef struct { uint32_t N, x_stride, y_stride, pad; } qmv_args;

// y = W·x（dense、量化）。依形狀選 kernel：列少且 K 長 → split-K；K 短 → 每個 SIMD group 兩列
static void qmv_y(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *x, size_t xoff, const qw_buf *y, size_t yoff, bool fout);
static void qmv(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *x, size_t xoff, const qw_buf *y, bool fout) {
    qmv_y(e, b, L, x, xoff, y, 0, fout);
}

static qw_kernel *df_q4_kernel(qw_engine *e, const char *kind, int K, int nb, bool fout);
// lm_head（Q6_K、列數多）走矩陣單元版（kernels/df_q4.metal 的 df_q6tc，DFlash2 的 draft head 也用它）：
// 純量版每個權重要對每個 token 各乘一次，驗證 8 個 token 時被算力卡住（3.5 ms，單 token 1.0 ms）；矩陣單元版 8 個 token 也接近頻寬上限。
// decode（nb = 1）、驗證、prefill 的最後一個 token 都走這裡：各列輸入互不影響，每個 token 的算式相同，驗證與 decode 逐位元一致。
// 權重捨入成 FP16 當運算元（同 llama.cpp 的 mul_mm），數值與純量版略有不同，所以預設關閉：QW_LM_TC=2（qwental-serve --fast-head）開啟
// 用哪一版：0 純量（qmv_gg，預設）、1 df_q6tc（K 方向一次 64 個值）、2 df_q6tcb（一次 128 個值，讀取少；矩陣單元排列核對過才用）。
// 開啟時決定一次（lm_tc_init），之後 decode 與驗證都用同一版。DFlash2 的 draft head 只影響猜測，排列核對過就一律用 df_q6tcb（q6b_ok）
static int lm_tc_mode = 0;
static bool q6b_ok = false;
static bool lm_tc_shape(const qw_qlin *L) { return L->gq == 15 && L->out >= 65536 && L->in % 512 == 0; }
static bool lm_tc(const qw_qlin *L) { return lm_tc_mode && lm_tc_shape(L); }
static void lm_tc_init(qw_engine *e) {
    const qw_qlin *L = &e->m->lm_head;
    int want = getenv("QW_LM_TC") ? atoi(getenv("QW_LM_TC")) : 0;
    if (want < 0 || want > 2) want = 2;
    lm_tc_mode = want;
    if (!lm_tc_shape(L)) return;
    qw_buf chk; // df_q6tcb 假設 cooperative tensor 的排列：先在這台機器上核對，不符就用 df_q6tc
    if (!qw_buf_new(e->g, 16, &chk)) { if (want == 2) lm_tc_mode = 1; return; }
    memset(chk.ptr, 0, 16);
    qw_batch *b = qw_batch_begin(e->g, false);
    qw_batch_kernel(b, df_q4_kernel(e, "q6bl", L->in, 1, true));
    qw_batch_buf(b, 0, &chk, 0);
    qw_batch_dispatch(b, 1, 1, 1, 32, 1, 1);
    const bool ran = qw_batch_end(b) >= 0;
    q6b_ok = ran && !*(const uint32_t *)chk.ptr;
    if (!q6b_ok) fprintf(stderr, "lm_head：矩陣單元的排列與 df_q6tcb 的假設不符，改用 df_q6tc\n");
    if (want == 2 && !q6b_ok) lm_tc_mode = 1;
    qw_buf_free(&chk);
}

// GGUF profile 的 dense matvec（kernels/qmv_gg.metal）：nb 個 token，x 的列距 = K、y 的列距 = N
static void qmv_gg(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *x, size_t xoff, const qw_buf *y, size_t yoff, int nb, bool fout) {
    char name[96], defs[512], err[8192];
    if (lm_tc(L)) {
        if (xoff + (size_t)8 * L->in * 2 > x->size || yoff || !fout || nb < 1 || nb > 8) { // 矩陣單元一次讀 8 列輸入：不能退回純量版（decode 與驗證會不一致）
            fprintf(stderr, "lm_head 矩陣單元版：輸入緩衝區不到 8 列或參數不符\n");
            exit(1);
        }
        struct { uint32_t N[4], off[4], total, xstride; } a = {{(uint32_t)L->out}, {0}, (uint32_t)L->out, (uint32_t)L->in};
        qw_batch_kernel(b, df_q4_kernel(e, lm_tc_mode == 2 ? "q6b" : "q6", L->in, nb, true));
        bind_t(e, b, 0, L->w);
        qw_batch_buf(b, 8, x, xoff);
        qw_batch_buf(b, 9, y, 0);
        qw_batch_bytes(b, 13, &a, sizeof a);
        qw_batch_dispatch(b, ((unsigned)L->out + 31) / 32, 1, 1, 32 * (lm_tc_mode == 2 ? DF_Q6B_NSG : DF_Q4_NSG), 1, 1);
        return;
    }
    const int f = L->gq - 1;
    snprintf(name, sizeof name, "qmv_gg_f%d_k%d_n%d%s", f, L->in, nb, fout ? "_f" : "");
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) {
        snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY 1\n#define qmv_gg %s\n#define GF %d\n#define KD %d\n#define RB %d\n#define NB %d\n%s",
                 name, f, L->in, gg_rb(L), nb, fout ? "#define YT float\n" : "");
        if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_qmv_gg, sizeof src_qmv_gg, err, sizeof err)) {
            fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
            exit(1);
        }
        k = qw_kernel_get(e->g, name);
    }
    struct { uint32_t N, xs, ys, pad; } a = {(uint32_t)L->out, (uint32_t)L->in, (uint32_t)L->out, 0};
    qw_batch_kernel(b, k);
    bind_t(e, b, 0, L->w);
    qw_batch_buf(b, 3, x, xoff);
    qw_batch_buf(b, 4, y, yoff);
    qw_batch_bytes(b, 5, &a, sizeof a);
    qw_batch_dispatch(b, ((unsigned)L->out + 7) / 8, 1, 1, 256, 1, 1);
}

// GGUF attention 的多矩陣 matvec（kernels/qmv_ggm.metal）：同一個 x、1–4 個矩陣（K 相同、256 的倍數；格式可不同）、nb 個 token，
// 一次 dispatch。x 的列距 = K、第 i 個矩陣輸出的列距 = 它的 N。不支援的組合回傳 false（呼叫端改走通用版）
static bool ggm_fmt_ok(int gq) { const int f = gq - 1; return f == 12 || f == 13 || f == 14 || f == 23 || f == 30; }
// 每個格式組合一次編出 NB 1…8 的 8 個 kernel（<名稱>_b{NB}）；組合不多，開啟時先編好（ggm_warm），不在生成途中編譯
static qw_kernel *qmv_ggm_kernel(qw_engine *e, const qw_qlin *const *L, int n, int nb) {
    char base[128], name[140], defs[512], err[8192];
    int f[4];
    for (int i = 0; i < 4; i++) f[i] = L[i < n ? i : 0]->gq - 1;
    snprintf(base, sizeof base, "qmv_ggm_k%d_m%d_%d_%d_%d_%d", L[0]->in, n, f[0], f[1], f[2], f[3]);
    snprintf(name, sizeof name, "%s_b%d", base, nb);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY 1\n#define qmv_ggm %s\n#define KD %d\n#define NB 0\n#define NMAT %d\n"
             "#define GF0 %d\n#define GF1 %d\n#define GF2 %d\n#define GF3 %d\n", base, L[0]->in, n, f[0], f[1], f[2], f[3]);
    if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_qmv_ggm, sizeof src_qmv_ggm, err, sizeof err)) {
        fprintf(stderr, "編譯 %s 失敗：%s\n", base, err);
        exit(1);
    }
    return qw_kernel_get(e->g, name);
}
static bool qmv_ggm_ok(const qw_qlin *const *L, int n) {
    if (n < 1 || n > 4 || getenv("QW_NO_GGM")) return false;
    for (int i = 0; i < n; i++)
        if (!ggm_fmt_ok(L[i]->gq) || L[i]->in != L[0]->in || L[i]->in % 256 || L[i]->ne != 1) return false;
    return true;
}
static bool qmv_ggm(qw_engine *e, qw_batch *b, const qw_qlin *const *L, const qw_buf *const *y, int n, const qw_buf *x, int nb) {
    if (nb < 1 || nb > 8 || !qmv_ggm_ok(L, n)) return false;
    qw_kernel *k = qmv_ggm_kernel(e, L, n, nb);
    struct { uint32_t N[4], off[4], total, xstride; } a = {{0}, {0}, 0, (uint32_t)L[0]->in};
    qw_batch_kernel(b, k);
    for (int i = 0; i < 4; i++) {
        const int j = i < n ? i : 0;
        bind_t(e, b, (unsigned)i, L[j]->w);
        qw_batch_buf(b, 5u + (unsigned)i, y[j], 0);
        a.N[i] = i < n ? (uint32_t)L[i]->out : 0;
        a.off[i] = a.total;
        if (i < n) a.total += (uint32_t)L[i]->out;
    }
    qw_batch_buf(b, 4, x, 0);
    qw_batch_bytes(b, 9, &a, sizeof a);
    qw_batch_dispatch(b, (a.total + 7) / 8, 1, 1, 256, 1, 1);
    return true;
}

static void qmv_y(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *x, size_t xoff, const qw_buf *y, size_t yoff, bool fout) {
    if (L->gq) { qmv_gg(e, b, L, x, xoff, y, yoff, 1, fout); return; }
    qw_kernel *k = qmv_kernel(e, L->bits, L->in, fout);
    qmv_args a = {(uint32_t)L->out, 0, (uint32_t)L->out, 0};
    const char *variant = NULL;
    unsigned gx = 0;
    if (L->out <= 1024 && L->in >= 4096 && !getenv("QW_NO_SPLITK")) { variant = "qmv_splitk"; gx = (unsigned)L->out; }
    else if (L->in <= 640 && getenv("QW_HALFSG")) { variant = "qmv_halfsg"; gx = ((unsigned)L->out + 15) / 16; } // 量過反而略慢，預設不用
    if (variant) {
        char name[80];
        snprintf(name, sizeof name, "%s_b%d_k%d%s", variant, L->bits, L->in, fout ? "_f" : "");
        k = qw_kernel_get(e->g, name);
        if (!k) { fprintf(stderr, "找不到 %s\n", name); exit(1); }
        qw_batch_kernel(b, k);
        bind_t(e, b, 0, L->w);
        bind_t(e, b, 1, L->s);
        bind_t(e, b, 2, L->b);
        qw_batch_buf(b, 3, x, xoff);
        qw_batch_buf(b, 4, y, yoff);
        qw_batch_buf(b, 5, &e->eid0, 0);
        qw_batch_bytes(b, 6, &a, sizeof a);
        qw_batch_dispatch(b, gx, 1, 1, 256, 1, 1);
        return;
    }
    qw_batch_kernel(b, k);
    bind_t(e, b, 0, L->w);
    bind_t(e, b, 1, L->s);
    bind_t(e, b, 2, L->b);
    qw_batch_buf(b, 3, x, xoff);
    qw_batch_buf(b, 4, y, yoff);
    qw_batch_buf(b, 5, &e->eid0, 0);
    qw_batch_bytes(b, 6, &a, sizeof a);
    qw_batch_dispatch(b, ((unsigned)L->out + 3) / 4, 1, 1, 128, 1, 1);
}

// 多矩陣 qmv：同一個 x、最多 4 個矩陣（相同 K、位元數），一次 dispatch
typedef struct { uint32_t N[4], off[4], total; } qmv_multi_args;

static void qmv_multi(qw_engine *e, qw_batch *b, const qw_qlin *const *L, const qw_buf *const *y, int n, const qw_buf *x) {
    bool gg = false;
    for (int i = 0; i < n; i++) gg |= L[i]->gq != 0;
    if (getenv("QW_NO_MULTI") || gg) { // GGUF：先逐個矩陣算（各部分再合併特化）
        for (int i = 0; i < n; i++) qmv(e, b, L[i], x, 0, y[i], false);
        return;
    }
    qmv_kernel(e, L[0]->bits, L[0]->in, false); // 確保已編譯
    char name[80];
    snprintf(name, sizeof name, "qmv_multi_b%d_k%d", L[0]->bits, L[0]->in);
    qw_kernel *k = qw_kernel_get(e->g, name);
    qmv_multi_args a = {{0}, {0}, 0};
    qw_batch_kernel(b, k);
    for (int i = 0; i < 4; i++) {
        const qw_qlin *Q = L[i < n ? i : 0];
        bind_t(e, b, 3 * i, Q->w);
        bind_t(e, b, 3 * i + 1, Q->s);
        bind_t(e, b, 3 * i + 2, Q->b);
        qw_batch_buf(b, 13 + i, y[i < n ? i : 0], 0);
        a.N[i] = i < n ? (uint32_t)L[i]->out : 0;
        a.off[i] = a.total;
        if (i < n) a.total += (uint32_t)L[i]->out;
    }
    qw_batch_buf(b, 12, x, 0);
    qw_batch_bytes(b, 17, &a, sizeof a);
    qw_batch_dispatch(b, (a.total + 7) / 8, 1, 1, 256, 1, 1);
}

static void d1(qw_batch *b, unsigned n, unsigned tg) { qw_batch_dispatch(b, (n + tg - 1) / tg, 1, 1, tg, 1, 1); }

// ---------------------------------------------------------------- 開啟

static void snap_save(qw_engine *e, int len);
// QW_VERIFY_REUSE（重用狀態的自我檢查，見 qw_engine_par_verify）
static int vr_on(void);
static void vr_note(int sid, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void vr_capture(qw_engine *e, int sid, const int32_t *ids, int n, const float *lg);
static void vr_mark(int sid); // 這個 prompt 用到了別處來的狀態
static qw_kernel *dn_kernel(qw_engine *e, const char *base, int nb);
static void snap_save_from(qw_engine *e, int len, int stage);
static int gdn_chunk(void);

static bool alloc(qw_engine *e, qw_buf *b, size_t bytes) {
    if (!qw_buf_new(e->g, bytes < 16 ? 16 : bytes, b)) return false;
    memset(b->ptr, 0, b->size);
    return true;
}

// KV 依需要配置：一開始小，不夠時以 1.25 倍（至少 KV_STEP）成長（記憶體吃緊時，3 組 slot 都照最大 context 配置會讓 GPU 一直換頁）
enum { DF_RING_ = 2064, KV_STEP = 4096 }; // DFlash2 環狀 KV 的槽數（與下方 DF_RING 相同）

static size_t kv_bytes(int cap, int which) {
    const size_t L = (size_t)cap;
    return which < 2 ? (size_t)(NATT + 1) * L * 2 * 256 * 2 : which == 2 ? (size_t)(NATT + 1) * L * 128 * 4 : (size_t)(NATT + 1) * (L / 4 + 1) * 128 * 2;
}
static size_t kv_row(int which) { return which < 2 ? 2 * 256 * 2 : which == 2 ? 128 * 4 : 128 * 2; }
static size_t kv_rows(int cap, int which) { return which == 3 ? (size_t)cap / 4 + 1 : (size_t)cap; }

static int kv_init_cap(const qw_engine *e) {
    const int c = getenv("QW_KV_INIT") ? atoi(getenv("QW_KV_INIT")) : 16384;
    return c < e->max_ctx ? c : e->max_ctx;
}

static bool kv_alloc(qw_engine *e, qw_buf *K, qw_buf *V, qw_buf *R, qw_buf *B, int cap) {
    qw_buf *b[4] = {K, V, R, B};
    for (int w = 0; w < 4; w++)
        if (!alloc(e, b[w], kv_bytes(cap, w))) return false;
    return true;
}

// KV 換成容量 cap 的新 buffer：前 min(舊, 新) 列照抄、其餘清零（擴大與縮小共用；要在 GPU 閒置時呼叫，直接在 CPU 端搬資料）。
// 每份 attention 的資料在 buffer 裡的位移照容量算（ai × rows），所以換容量一定要整份重排，不能只截尾
static bool kv_resize(qw_engine *e, qw_buf *K, qw_buf *V, qw_buf *R, qw_buf *B, int *kvcap, int cap) {
    qw_buf nb[4];
    qw_buf *ob[4] = {K, V, R, B};
    for (int w = 0; w < 4; w++) {
        if (!qw_buf_new(e->g, kv_bytes(cap, w), &nb[w])) {
            for (int q = 0; q < w; q++) qw_buf_free(&nb[q]);
            return false;
        }
    }
    for (int w = 0; w < 4; w++) {
        const size_t row = kv_row(w), on = kv_rows(*kvcap, w), nn = kv_rows(cap, w), cp = on < nn ? on : nn;
        for (int ai = 0; ai <= NATT; ai++) {
            memcpy((char *)nb[w].ptr + ai * nn * row, (char *)ob[w]->ptr + ai * on * row, cp * row);
            memset((char *)nb[w].ptr + (ai * nn + cp) * row, 0, (nn - cp) * row);
        }
        qw_buf_free(ob[w]);
        *ob[w] = nb[w];
    }
    *kvcap = cap;
    return true;
}

// KV 成長：每次加 max(4096, 目前容量 / 4)（至少到 need），對齊 4096。原本是加倍（16K → 32K → 64K），33K 的對話要佔 64K 的容量；
// 改成 1.25 倍後浪費 ≤ 25%（閒置時 qw_engine_par_shrink 再縮到實際長度），而重配次數只多一點：16K → 100K 約 8 次，
// 累計搬動量 ≈ 4 × 最終大小（CPU memcpy，100K 時累計約 13 GB、0.3 s 量級，分散在幾次；prefill 前先照整個 prompt 預留，通常一次到位）。
// 每次擴大是先配好四份新的再搬（失敗時原本的不動），尖峰暫時多一份新容量
// QW_KV_DOUBLE=1：照舊加倍
static int kv_grow_cap(const qw_engine *e, int need) {
    static int dbl = -1;
    if (dbl < 0) dbl = getenv("QW_KV_DOUBLE") && atoi(getenv("QW_KV_DOUBLE")) ? 1 : 0;
    int cap = e->kvcap;
    if (dbl) while (cap < need) cap *= 2;
    else {
        const int step = cap / 4 > KV_STEP ? cap / 4 : KV_STEP;
        cap = cap + step > need ? cap + step : need;
    }
    cap = (cap + KV_STEP - 1) / KV_STEP * KV_STEP;
    return cap > e->max_ctx ? e->max_ctx : cap;
}

// 確保目前的 KV 至少容納 need 個 token；要在 GPU 閒置時呼叫（直接在 CPU 端搬資料）
static bool kv_reserve(qw_engine *e, int need) {
    if (need <= e->kvcap) return true;
    const int cap = kv_grow_cap(e, need), old = e->kvcap;
    if (cap <= e->kvcap) return need <= e->kvcap;
    if (!kv_resize(e, &e->Kc, &e->Vc, &e->rawk, &e->kblk, &e->kvcap, cap)) return false;
    if (getenv("QW_CACHE_DBG")) fprintf(stderr, "[cache] KV 容量 %d → %d\n", old, cap);
    if (vr_on()) vr_note(e->cur, "grow kv %d→%d; ", old, cap);
    return true;
}

// qsa_score4（驗證 5–8 個 token 的 indexer 打分數，reduce-scatter）的前提是 simd_sum 的加法順序與它相同：
// 啟動時拿亂數與 qsa_score3 比（分數與直方圖逐位元，nt = 5…VBMAX），不同（或跑不了）就一律用 qsa_score3。QW_SCORE3=1：一律用 qsa_score3
static bool score4_ok(qw_engine *e, qw_kernel *k4) {
    typedef struct { uint32_t L, pos, nvis, ntok, sparse; } args;
    enum { NBK = 2048, SST = NBK + 8, NBIN_ = 4096 };
    if (!k4) return false;
    qw_buf iq = {0}, kb = {0}, sc[2] = {{0}}, hs[2] = {{0}};
    bool ok = qw_buf_new(e->g, (size_t)VBMAX * 4 * 128 * 2, &iq) && qw_buf_new(e->g, (size_t)NBK * 128 * 2, &kb);
    for (int v = 0; v < 2 && ok; v++) ok = qw_buf_new(e->g, (size_t)VBMAX * SST * 4, &sc[v]) && qw_buf_new(e->g, (size_t)VBMAX * NBIN_ * 4, &hs[v]);
    uint64_t r = 0x9e3779b97f4a7c15ull;
    for (int i = 0; ok && i < VBMAX * 4 * 128; i++) {
        r = r * 6364136223846793005ull + 1442695040888963407ull;
        ((_Float16 *)iq.ptr)[i] = (_Float16)((double)(r >> 11) / 9007199254740992.0 * 4 - 2);
    }
    for (int i = 0; ok && i < NBK * 128; i++) { // 大小不一，讓加法順序造成的捨入差異現形
        r = r * 6364136223846793005ull + 1442695040888963407ull;
        ((_Float16 *)kb.ptr)[i] = (_Float16)(((double)(r >> 11) / 9007199254740992.0 * 2 - 1) * (double)(1 + (r >> 60)));
    }
    for (int nt = 5; nt <= VBMAX && ok; nt++) {
        args aa[VBMAX];
        for (int t = 0; t < nt; t++) aa[t] = (args){0, 0, (uint32_t)(NBK - 37 * t), 0, t != 1}; // token 1 不稀疏：不寫
        const uint32_t unt = (uint32_t)nt, sst = SST;
        for (int v = 0; v < 2 && ok; v++) {
            memset(sc[v].ptr, 0, sc[v].size);
            memset(hs[v].ptr, 0, hs[v].size);
            qw_batch *b = qw_batch_begin(e->g, false);
            qw_batch_kernel(b, v ? k4 : e->q3_score);
            qw_batch_buf(b, 0, &iq, 0);
            qw_batch_buf(b, 1, &kb, 0);
            qw_batch_buf(b, 2, &sc[v], 0);
            qw_batch_bytes(b, 3, aa, sizeof(args) * (size_t)nt);
            qw_batch_bytes(b, 4, &unt, 4);
            qw_batch_bytes(b, 5, &sst, 4);
            qw_batch_buf(b, 6, &hs[v], 0);
            qw_batch_dispatch(b, NBK / 8, 1, 1, 256, 1, 1);
            ok = qw_batch_end(b) >= 0;
        }
        ok = ok && !memcmp(sc[0].ptr, sc[1].ptr, sc[0].size) && !memcmp(hs[0].ptr, hs[1].ptr, hs[0].size);
    }
    qw_buf *fb[6] = {&iq, &kb, &sc[0], &sc[1], &hs[0], &hs[1]};
    for (int i = 0; i < 6; i++) if (fb[i]->handle) qw_buf_free(fb[i]);
    if (!ok) fprintf(stderr, "qsa_score4 與 qsa_score3 的結果不同（simd_sum 的加法順序與假設不同），indexer 打分數一律用 qsa_score3\n");
    return ok;
}

static void ggm_warm(qw_engine *e);
qw_engine *qw_engine_open(const char *dir, int max_ctx, char *err, size_t errlen) {
    qw_engine *e = calloc(1, sizeof *e);
    e->max_ctx = max_ctx;
    e->g = qw_gpu_open();
    if (!e->g) { snprintf(err, errlen, "找不到 Metal 裝置"); return NULL; }
    e->m = qw_model_open(dir, err, errlen);
    if (!e->m) return NULL;
    { // GGUF profile：只有 kernel 已經支援的部分才能用；其餘一律拒絕（不讓不支援的格式跑上 GPU、算出錯的結果）
        // 支援一個部分時，在這裡加上它的位元（每個部分一行）
        unsigned ok = 0;
        ok |= QW_GG_MOE; // routed expert：kernels/moe_iq.metal（IQ2_S、IQ3_XXS、IQ3_S、IQ4_XS、IQ4_NL、Q2_0）
        ok |= QW_GG_LM;  // lm_head：通用 dense matvec（kernels/qmv_gg.metal、qmm.metal 的 IQF）；DFlash2 的 draft head 抄 GGUF 列（src/dflash.c）
        ok |= QW_GG_EMBED; // embed_tokens：kernels/gg_ple.metal 的 gg_embed（decode、驗證、prefill、DFlash2）
        ok |= QW_GG_PLE;   // PLE：n-gram 表 CPU 精確解碼（ng_gather）、F32 norm、FP32 value（kernels/gg_ple.metal）
        ok |= QW_GG_HYPER; // hyper-connection：kernels/hyper_gg.metal（hc_norm F32、mix_down / mix_up BF16）
        ok |= QW_GG_LIN; // DeltaNet：V head 交錯順序（h % 16）、A_neg 的 kernel 變體；投影 kernels/dn_proj.metal
        ok |= QW_GG_ATTN; // attention：F32 norm（*_f kernel）、q / k / v / indexer 一次 dispatch（kernels/qmv_ggm.metal）、prefill qmm_ggm
        ok |= QW_GG_SHARED; // shared expert：kernels/moe_gg.metal（decode / 驗證）、通用 dense qmm（prefill）；須同時選 moe
        ok |= QW_GG_ROUTER; // router：BF16（kernels/moe_gg.metal 的 moe_router_bf、p_router_bf*）
        if ((e->m->gguf_parts & QW_GG_SHARED) && (!(e->m->gguf_parts & QW_GG_MOE) || getenv("QW_MOE_OLD"))) {
            snprintf(err, errlen, "GGUF profile：shared 要和 moe 一起選（GGUF 的 shared expert 只接在 GGUF routed expert 的 kernel 上）");
            return NULL;
        }
        const unsigned bad = e->m->gguf_parts & ~ok;
        if (bad) {
            static const char *nm[] = {"moe", "shared", "router", "attn", "lin", "hyper", "lm", "embed", "ple"};
            char lst[200] = "";
            for (int i = 0; i < 9; i++)
                if (bad & (1u << i)) { strcat(lst, lst[0] ? "," : ""); strcat(lst, nm[i]); }
            snprintf(err, errlen, "GGUF profile：這些部分的 kernel 尚未支援：%s（QW_GGUF_PARTS 先只選已支援的）", lst);
            return NULL;
        }
    }
    // dense 權重：預設把 DeltaNet / attention / 共用 expert 轉成每列對稱 8-bit（prefill 走 INT8 × INT8，困惑度不變）
    const char *rq = getenv("QW_DENSE4") ? getenv("QW_DENSE4") : getenv("QW_NO_C8") ? NULL : "c8:lin,attn,shared";
    if (rq && !qw_requant4(e->m, rq, err, errlen)) return NULL;
    // GGUF dense 投影的 INT8 影子（QW_INT8=1 或類別清單）：prefill 走 INT8 × INT8，decode / 驗證不受影響
    if (e->m->gguf_parts && getenv("QW_INT8") && strcmp(getenv("QW_INT8"), "0") &&
        !qw_gguf_c8(e->m, strcmp(getenv("QW_INT8"), "1") ? getenv("QW_INT8") : "lin,attn,shared", err, errlen))
        return NULL;
    { // DFlash2 drafter：要在包成 Metal buffer 之前（會加一個虛擬分片）
        char dp[4200], vp[4200];
        // dflash2/ 與模型目錄同層；沒有 MLX 目錄（GGUF profile）時改找 GGUF 目錄的同層
        const char *ddir = access(dir, F_OK) && getenv("QW_GGUF") ? getenv("QW_GGUF") : dir;
        // 打包檔：QW_DF_PACK 指定；否則優先用 selq4 版（selector codebook 為 Q4_0，與 Strata 預設相同），沒有才用一般版
        if (getenv("QW_DF_PACK")) snprintf(dp, sizeof dp, "%s", getenv("QW_DF_PACK"));
        else {
            snprintf(dp, sizeof dp, "%s/../dflash2/qwen3.8-flash-next-dflash2.q4_0-selq4.pack", ddir);
            if (access(dp, R_OK)) snprintf(dp, sizeof dp, "%s/../dflash2/qwen3.8-flash-next-dflash2.q4_0.pack", ddir);
        }
        snprintf(vp, sizeof vp, "%s/../dflash2/draft_vocab.bin", ddir);
        if (!getenv("QW_NO_DFLASH") && access(dp, R_OK) == 0) {
            if (!qw_dflash_load(e->m, dp, vp, &e->df, err, errlen)) return NULL;
            e->df_on = true;
        }
    }
    e->nfiles = e->m->st->nfiles;
    if (getenv("QW_WRAP_FILE")) e->fbuf = calloc(e->nfiles, sizeof *e->fbuf);
    for (int i = 0; e->fbuf && i < e->nfiles; i++) {
        st_file *f = e->m->st->files[i];
        if (!qw_buf_wrap(e->g, f->map, f->size, &e->fbuf[i])) {
            snprintf(err, errlen, "無法把 %s 包成 Metal buffer", f->path);
            return NULL;
        }
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/ngram_table.bin", dir);
    if (!e->m->ple_tab && !(e->ngram = st_open(path, err, errlen))) return NULL; // GGUF 的 n-gram 表：不開 MLX 的
    // 每個 token 只讀幾列、散在整個 32 GB 表裡：關掉 page fault 的預讀（冷的時候一段 4096 個 token 的抄列約快 2–3 倍，也少擠掉 page cache）
    if (e->ngram) madvise(e->ngram->map, e->ngram->size, MADV_RANDOM);

    // ---- kernel
    char defs[1024];
    snprintf(defs, sizeof defs, "#define D %d\n#define HC %d\n", D, QW_HC);
    if (!compile(e, defs, src_engine, sizeof src_engine, err, errlen)) return NULL;
    if (e->m->embed_gq || e->m->ple_tab) { // GGUF 的 embed / PLE（kernels/gg_ple.metal）
        const int gf = e->m->embed_gq ? e->m->embed_gq - 1 : 20;
        snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY 1\n#define D %d\n#define HC %d\n#define GF %d\n#define ERB %d\n", D, QW_HC, gf,
                 D / iq_qb(gf) * iq_bb(gf));
        if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_gg_ple, sizeof src_gg_ple, err, errlen))
            return NULL;
    }
    // MoE：gate/up 用 RPS 1、down 用 RPS 2（量測過的最佳設定）→ 編兩次、改名
    snprintf(defs, sizeof defs, "#define moe_down moe_down_r1\n#define moe_down_f moe_down_f_r1\n#define E %d\n#define TOPK %d\n#define D %d\n#define I %d\n#define RPS 1\n",
             QW_EXPERTS, QW_TOPK, D, QW_MOE_INTER);
    if (!compile(e, defs, src_moe, sizeof src_moe, err, errlen)) return NULL;
    snprintf(defs, sizeof defs,
             "#define moe_router moe_router_r2\n#define moe_select moe_select_r2\n#define moe_gateup moe_gateup_r2\n"
             "#define moe_down moe_down_r2\n#define moe_gateup_f moe_gateup_f_r2\n#define moe_down_f moe_down_f_r2\n"
             "#define E %d\n#define TOPK %d\n#define D %d\n#define I %d\n#define RPS 2\n",
             QW_EXPERTS, QW_TOPK, D, QW_MOE_INTER);
    if (!compile(e, defs, src_moe, sizeof src_moe, err, errlen)) return NULL;
    if (!compile(e, "#define DK 128\n#define DV 128\n#define HK 16\n#define HV 48\n#define DVB 32\n", src_gdn, sizeof src_gdn, err, errlen))
        return NULL;
    if (e->m->lin_tiled && // GGUF：V head 交錯順序（v head h 對應 key head h % 16）的另一份
        !compile(e, "#define GDN_TILED 1\n#define GDN_COMMIT 1\n#define gdn_step gdn_step_t\n#define gdn_commit gdn_commit_t\n#define DK 128\n#define DV 128\n#define HK 16\n#define HV 48\n#define DVB 32\n",
                 src_gdn, sizeof src_gdn, err, errlen))
        return NULL;
    for (int bits = 4; bits <= 8; bits += 4) // 權重可能被重新量化成 4-bit（QW_DENSE4）：兩種都編
        for (int xf = 1; xf <= 2; xf++) {
            snprintf(defs, sizeof defs,
                     "#define qmv qmv_xfbase%d_b%d\n#define qmv_splitk qmv_splitk_xf%d_b%d\n#define qmv_halfsg qmv_halfsg_xf%d_b%d\n"
                     "#define qmv_multi qmv_multi_xf%d_b%d\n#define qmv_splitk_inj qmv_splitk_inj_xf%d_b%d\n#define qmv_xf qmv_xf%d_b%d\n#define XF %d\n"
                     "#define BITS %d\n#define K 6144\n#define RPS 1\n#define SPT 4\n#define ST bfloat\n",
                     xf, bits, xf, bits, xf, bits, xf, bits, xf, bits, xf, bits, xf, bits);
            if (!compile(e, defs, src_qmv, sizeof src_qmv, err, errlen)) return NULL;
        }
    if (!(e->qmv_xf1 = need(e, "qmv_xf1_b8", err, errlen)) || !(e->qmv_xf2 = need(e, "qmv_xf2_b8", err, errlen)) ||
        !(e->qmv_xf1_4 = need(e, "qmv_xf1_b4", err, errlen)) || !(e->qmv_xf2_4 = need(e, "qmv_xf2_b4", err, errlen)))
        return NULL;
    // QSA：選取 kernel 依 context 長度有 3 種 MAXPER；其他 kernel 只用第一份
    const int maxper[3] = {8, 16, 32};
    for (int i = 0; i < 3; i++) {
        snprintf(defs, sizeof defs,
                 "#define qsa_score qsa_score_m%d\n#define qsa_topk qsa_topk_m%d\n#define qsa_attn qsa_attn_m%d\n"
                 "#define qsa_merge qsa_merge_m%d\n#define qsa_score2 qsa_score2_m%d\n#define qsa_topk2 qsa_topk2_m%d\n"
                 "#define qsa_attn2 qsa_attn2_m%d\n#define qsa_merge2 qsa_merge2_m%d\n#define qsa_topk2b qsa_topk2b_m%d\n"
                 "#define IH 4\n#define ID 128\n#define QH 24\n#define KVH 2\n#define HD 256\n#define SEL 512\n#define CR 4\n#define MAXPER %d\n",
                 maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], maxper[i]);
        if (!compile(e, defs, src_qsa, sizeof src_qsa, err, errlen)) return NULL;
    }
    // QSA v3：選取 kernel 3 種 MAXPER；QW_Q3X 可以多注入一段定義（實驗用）
    for (int i = 0; i < 3; i++) {
        snprintf(defs, sizeof defs,
                 "#define qsa_prep_bf qsa_prep_bf_m%d\n#define qsa_prep_f qsa_prep_f_m%d\n#define qsa_score3 qsa_score3_m%d\n"
                 "#define qsa_topk3 qsa_topk3_m%d\n#define qsa_attn4 qsa_attn4_m%d\n#define qsa_merge4 qsa_merge4_m%d\n"
                 "#define qsa_attn5 qsa_attn5_m%d\n#define qsa_cmp5 qsa_cmp5_m%d\n#define qsa_score4 qsa_score4_m%d\n"
                 "#define IH 4\n#define ID 128\n#define QH 24\n#define KVH 2\n#define HD 256\n#define SEL 512\n#define CR 4\n#define MAXPER %d\n"
                 "#define NCH4 %d\n#define TT %d\n%s",
                 maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], maxper[i], q4_nch(), q5_tt(),
                 getenv("QW_Q3X") ? getenv("QW_Q3X") : "");
        if (!compile(e, defs, src_qsa_dec, sizeof src_qsa_dec, err, errlen)) return NULL;
    }
    // K4：M5 的 cooperative tensor 排列（每個 lane 連續 4 個元素 = 同一個 token 的連續 4 維），K 一次讀 half4；
    // QW_QSA_K1 改回逐個讀（其他排列的 GPU 用）
    if (!compile(e, getenv("QW_QSA_K1") ? "#define QH 24\n#define KVH 2\n#define HD 256\n#define SEL 512\n#define CR 4\n"
                                        : "#define QH 24\n#define KVH 2\n#define HD 256\n#define SEL 512\n#define CR 4\n#define K4 1\n#define QK8_K4 1\n#define QK8_KC 64\n",
                 src_qsa_pf, sizeof src_qsa_pf, err, errlen))
        return NULL;
#define K_(f) if (!(e->f = need(e, #f, err, errlen))) return NULL;
    K_(k_embed) K_(k_hcnorm) K_(k_silu_div) K_(k_hcmix) K_(k_inject) K_(k_resid) K_(k_conv4) K_(k_gdn_qk) K_(k_normgate)
    K_(k_ple_gate) K_(k_ple_conv) K_(k_hcnorm_f) K_(k_h2f) K_(k_copyf)
    K_(k_hc_in) K_(k_mixup) K_(k_gdn_in) K_(moe_gateup_f) K_(k_embed_g) K_(k_ngram) K_(k_argmax1) K_(k_argmax2)
    K_(k_norm1) K_(k_mtp_combine)
    K_(moe_router) K_(moe_select) K_(moe_gateup) K_(gdn_step)
#undef K_
    if (!(e->moe_down_f = need(e, "moe_down_f_r2", err, errlen))) return NULL;
    if (e->m->lin_tiled && (!(e->gdn_step_t = need(e, "gdn_step_t", err, errlen)) || !(e->k_gdn_in_t = need(e, "k_gdn_in_t", err, errlen)) ||
                            !(e->gdn_commit_t = need(e, "gdn_commit_t", err, errlen))))
        return NULL;
    if (e->m->lin_tiled) dn_kernel(e, "dn_ksum", 1); // GGUF DeltaNet 投影的 decode 版（先編好）
    if (e->m->lin_tiled && (!alloc(e, &e->dncnt, 64 * 4) || !gf_kernel(e, 1) || !dn_alloc_defer(e))) return NULL;
    if (!(e->moe_down_f1 = need(e, "moe_down_f_r1", err, errlen)) || !(e->moe_gateup_f2 = need(e, "moe_gateup_f_r2", err, errlen)))
        return NULL;
    if (!(e->moe_down = need(e, "moe_down_r2", err, errlen)) ||
        !(e->qsa_attn = need(e, "qsa_attn2_m8", err, errlen)) || !(e->qsa_merge = need(e, "qsa_merge2_m8", err, errlen)))
        return NULL;
    for (int i = 0; i < 3; i++) {
        char n[32];
        snprintf(n, sizeof n, "qsa_topk3_m%d", maxper[i]);
        if (!(e->q3_topk[i] = need(e, n, err, errlen))) return NULL;
    }
    if (!(e->q3_prep[0] = need(e, "qsa_prep_bf_m8", err, errlen)) || !(e->q3_prep[1] = need(e, "qsa_prep_f_m8", err, errlen)) ||
        !(e->q3_score = need(e, "qsa_score3_m8", err, errlen)) || !(e->q4_attn = need(e, "qsa_attn4_m8", err, errlen)) ||
        !(e->q4_merge = need(e, "qsa_merge4_m8", err, errlen)) || !(e->q5_attn = need(e, "qsa_attn5_m8", err, errlen)) ||
        !(e->q5_cmp = need(e, "qsa_cmp5_m8", err, errlen)))
        return NULL;

    if (!getenv("QW_SCORE3") && score4_ok(e, need(e, "qsa_score4_m8", err, errlen))) e->q3_score4 = need(e, "qsa_score4_m8", err, errlen);
    if (e->m->gguf_parts & QW_GG_ATTN) ggm_warm(e);

    // ---- 緩衝區
    const size_t L = (size_t)max_ctx;
    bool ok = alloc(e, &e->h, HCD * 4) && alloc(e, &e->nh, HCD * 2) && alloc(e, &e->nf, HCD * 4) && alloc(e, &e->d, 320 * 2) &&
              alloc(e, &e->d2, 320 * 2) && alloc(e, &e->u, HCD * 2) && alloc(e, &e->x, 8 * D * 2) && alloc(e, &e->inj, 64) && alloc(e, &e->inj2, 64) &&
              alloc(e, &e->out, D * 2) && alloc(e, &e->dump, (size_t)QW_LAYERS * HCD * 4) &&
              alloc(e, &e->qkv, 10240 * 2) && alloc(e, &e->z, 6144 * 2) && alloc(e, &e->ain, 64 * 2) && alloc(e, &e->bin, 64 * 2) &&
              alloc(e, &e->conv, 10240 * 2) && alloc(e, &e->gq, 2048 * 2) && alloc(e, &e->gk, 2048 * 2) && alloc(e, &e->la, 64 * 4) &&
              alloc(e, &e->be, 64 * 4) && alloc(e, &e->go, 6144 * 2) && alloc(e, &e->gy, 6144 * 2) &&
              alloc(e, &e->conv_st, (size_t)NLIN * 3 * 10240 * 4) && alloc(e, &e->S, (size_t)NLIN * 48 * 128 * 128 * 4) &&
              alloc(e, &e->qg, 12288 * 2) && alloc(e, &e->kk, 512 * 2) && alloc(e, &e->vv, 512 * 2) && alloc(e, &e->iqk, 640 * 2) &&
              alloc(e, &e->aq, 6144 * 2) && alloc(e, &e->iq, 512 * 2) && alloc(e, &e->ao, 6144 * 2) && alloc(e, &e->ay, 6144 * 2) &&
              kv_alloc(e, &e->Kc, &e->Vc, &e->rawk, &e->kblk, e->kvcap = kv_init_cap(e)) &&
              alloc(e, &e->part, (size_t)QSA_NCH * 24 * 258 * 4) &&
              alloc(e, &e->q3sc, (size_t)VBMAX * (L / 4 + 1) * 4) && alloc(e, &e->q3sel, (size_t)VBMAX * 512 * 4) &&
              alloc(e, &e->q3nsel, VBMAX * 4) && alloc(e, &e->q3hist, (size_t)VBMAX * 4096 * 4) && alloc(e, &e->q4part, (size_t)VBMAX * q4_nch() * 24 * 258 * 4) && alloc(e, &e->q5stat, 64) &&
              (!getenv("QW_ATT5_VCHECK") || alloc(e, &e->q5chk, (size_t)VBMAX * q4_nch() * 24 * 258 * 4)) &&
              alloc(e, &e->rlog, 520 * 4) && alloc(e, &e->msel, 64) && alloc(e, &e->mw, 64) && alloc(e, &e->H, 11 * 640 * 2) && alloc(e, &e->ggsel, 4096) && alloc(e, &e->ggtab, 131072) && alloc(e, &e->ggcnt, 16) && alloc(e, &e->ggH, (size_t)VBMAX * (QW_TOPK + 1) * QW_MOE_INTER * 4) && alloc(e, &e->ggpsh, (size_t)VBMAX * D * 4) && alloc(e, &e->ggpd, (size_t)VBMAX * QW_TOPK * D * 4) &&
              alloc(e, &e->eid0, 16) &&
              alloc(e, &e->emb, D * 2) && alloc(e, &e->keyh, HCD * 2) && alloc(e, &e->keyf, HCD * 4) && alloc(e, &e->kn, HCD * 4) &&
              alloc(e, &e->qn, HCD * 4) && alloc(e, &e->val, D * 2) && alloc(e, &e->gated, HCD * 4) && alloc(e, &e->cn, HCD * 4) &&
              alloc(e, &e->ple_st, 9 * HCD * 4) && alloc(e, &e->logits, (size_t)QW_VOCAB * 4) && alloc(e, &e->vbuf, 6144 * 2) &&
              alloc(e, &e->tok, (L + 8) * 4) && alloc(e, &e->amv, 256 * 4) && alloc(e, &e->ami, 256 * 4) &&
              alloc(e, &e->ssp, 48 * 4 * 4) && alloc(e, &e->mh, HCD * 4) && alloc(e, &e->memb, HCD * 4) &&
              alloc(e, &e->mnh, HCD * 2) && alloc(e, &e->mfe, D * 2) && alloc(e, &e->mfh, HCD * 2) &&
              alloc(e, &e->mlog, (size_t)QW_VOCAB * 4);
    if (!ok) {
        snprintf(err, errlen, "配置緩衝區失敗");
        return NULL;
    }
    e->hist = calloc(L + 2, sizeof *e->hist);
    e->hist[0] = e->hist[1] = EOS_ID;
    e->last_eos = 1;
    e->pf_next = -1;
    ((int32_t *)e->tok.ptr)[0] = ((int32_t *)e->tok.ptr)[1] = EOS_ID;
    e->ng_gpu = getenv("QW_NG_GPU") != NULL && !e->m->ple_tab; // GGUF 的表一律 CPU 抄列（28.8 GB 不交給 GPU）
    if (e->m->ple_tab) {
        if (e->m->ple_tab_gq != 20 + 1 || e->m->ple_row_bytes != 90) {
            snprintf(err, errlen, "GGUF：PLE 表格式不符（%s、每列 %zu 位元組；預期 IQ4_NL、90）", qw_ggml_name(e->m->ple_tab_gq - 1), e->m->ple_row_bytes);
            return NULL;
        }
        const size_t pos = (size_t)(PF > VBMAX ? PF : VBMAX);
        if (!alloc(e, &e->ngX, pos * 2 * D * 2) || !alloc(e, &e->ngX2, pos * 2 * D * 2) || !alloc(e, &e->pleYk, 2 * VBMAX * HCD * 4) || !alloc(e, &e->pleYv, 2 * VBMAX * D * 4) ||
            !alloc(e, &e->valf, D * 4)) {
            snprintf(err, errlen, "n-gram（GGUF）buffer 配置失敗");
            return NULL;
        }
    } else if (e->ng_gpu) {
        if (!qw_buf_wrap(e->g, e->ngram->map, e->ngram->size, &e->ngbuf)) {
            snprintf(err, errlen, "無法把 ngram_table.bin 包成 Metal buffer");
            return NULL;
        }
    } else {
        const size_t rows = (size_t)(PF > VBMAX ? PF : VBMAX) * 16;
        if (!alloc(e, &e->ngW, rows * 80) || !alloc(e, &e->ngS, rows * 10) || !alloc(e, &e->ngB, rows * 10) ||
            !alloc(e, &e->ngW2, rows * 80) || !alloc(e, &e->ngS2, rows * 10) || !alloc(e, &e->ngB2, rows * 10)) {
            snprintf(err, errlen, "n-gram 抄列 buffer 配置失敗");
            return NULL;
        }
    }
    gg_precompile(e, 1); // GGUF MoE 的 decode kernel
    lm_tc_init(e);
    return e;
}

void qw_engine_close(qw_engine *e) {
    if (!e) return;
    if (e->ngram) st_close(e->ngram);
    qw_model_close(e->m);
    qw_gpu_close(e->g);
    free(e->hist);
    free(e->fbuf);
    free(e);
}

int qw_engine_pos(const qw_engine *e) { return e->pos; }
qw_gpu *qw_engine_gpu(qw_engine *e) { return e->g; }
double qw_engine_last_gpu_ms(const qw_engine *e) { return e->last_ms; }

// ---------------------------------------------------------------- PLE：n-gram 雜湊（CPU）

static int64_t floor_mod(int64_t a, int64_t m) {
    int64_t r = a % m;
    return r < 0 ? r + m : r;
}

static float bf16f(uint16_t u) {
    uint32_t v = (uint32_t)u << 16;
    float f;
    memcpy(&f, &v, 4);
    return f;
}

static float round_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u;
    memcpy(&f, &u, 4);
    return f;
}

// 依參考實作 NGramEmbedding：
//   shifted[s] = 往回 s 個 token；若中間跨過 EOS（同一段內不足 s 個）則用 EOS
//   2-gram（head 0–7）與 3-gram（head 8–15）：mixed = XOR_s shifted[s]·mult[s]（int64 溢位）
//   gid = floor_mod(mixed, size[head]) + offset[head]；查表、解量化（4-bit、group 32）、轉 BF16
static void ple_embed(qw_engine *e, const qw_ple *P, uint16_t *out_half) {
    const int64_t *mult = P->multipliers->data, *off = P->head_offsets->data, *size = P->head_vocab->data;
    const int p = e->pos + 2;
    const int in_seg = p - (e->last_eos + 1);
    int64_t sh[3];
    for (int s = 0; s < 3; s++) sh[s] = in_seg >= s ? e->hist[p - s] : EOS_ID;
    const st_tensor *tw = st_find(e->ngram, "weight"), *ts = st_find(e->ngram, "scales"), *tb = st_find(e->ngram, "biases");
    const uint32_t *W = tw->data;
    const uint16_t *S = ts->data, *B = tb->data;
    const int words = (int)tw->shape[1], dim = words * 8, ngs = (int)ts->shape[1], gsz = dim / ngs;
    for (int n = 2; n <= 3; n++) {
        uint64_t mixed = (uint64_t)sh[0] * (uint64_t)mult[0];
        for (int q = 1; q < n; q++) mixed ^= (uint64_t)sh[q] * (uint64_t)mult[q];
        for (int hh = 0; hh < 8; hh++) {
            const int head = (n - 2) * 8 + hh;
            const int64_t row = floor_mod((int64_t)mixed, size[head]) + off[head];
            for (int i = 0; i < dim; i++) {
                const uint32_t w = W[row * words + i / 8];
                const float q = (float)((w >> (4 * (i % 8))) & 15u);
                const float v = round_bf16(q * bf16f(S[row * ngs + i / gsz]) + bf16f(B[row * ngs + i / gsz]));
                _Float16 hv = (_Float16)v;
                memcpy(&out_half[head * dim + i], &hv, 2);
            }
        }
    }
}

// ---------------------------------------------------------------- 一步 decode

// GatedResidual 的前半：x = mix(norm(h))；inject 非 NULL 時另算 inject 係數
static bool skip(const char *what);
static void resid(qw_engine *e, qw_batch *b);
static void df_tap_after(qw_engine *e, qw_batch *b, int l, const qw_buf *h, int rows);
static void df_ctx(qw_engine *e, qw_batch *b, int row0, int n, int pos0);
static void df_draft(qw_engine *e, qw_batch *b, int s, const qw_buf *prm);
static void df_accept_enc(qw_engine *e, qw_batch *b, int p0, int nd);
static void df_set_ids(qw_engine *e, int32_t anchor);
static bool df_init(qw_engine *e, char *err, size_t errlen);


// ghc_down / ghc_up 的參數（依 token 數 nb）：which 0 = GH_R（每個 threadgroup 幾列）、1 = GH_G（每組幾列先加總，
// 累加器 GH_G × nb 個）、2 = GU_SG（ghc_up 每個 threadgroup 幾個 SIMD group）。QW_GH_R / QW_GH_G / QW_GU_SG 可改
static int gh_cfg(int nb, int which) {
    static const char *const env[3] = {"QW_GH_R", "QW_GH_G", "QW_GU_SG"};
    if (getenv(env[which])) return atoi(getenv(env[which]));
    if (which == 0) return nb <= 4 ? 16 : 32; // 量過（kbench）：token 多時列多一點，h / out 少重讀
    if (which == 1) return nb <= 2 ? 16 : 8;
    return 8;
}

// 驗證的三段版（ghc_dsk：GH_GK 段 × HC 條流 × 列分 gh_vcfg(0) 份、每個 threadgroup gh_vcfg(1) 個 SIMD group）；bhpd 裡的位移（PDS = 328）
// QW_GH_VRP / QW_GH_VSG 可改（列份數 × SIMD group × 4 必須整除 320）
static int gh_vcfg(int which) {
    static int v[2] = {-1, -1};
    static const char *const env[2] = {"QW_GH_VRP", "QW_GH_VSG"};
    static const int def[2] = {10, 8};
    if (v[which] < 0) v[which] = getenv(env[which]) ? atoi(getenv(env[which])) : def[which];
    return v[which];
}
enum { GH_GK = 10, GH_PDS = QW_HC_LOWRANK + 2 * QW_HC,
       GH_PV_SX = VBMAX * QW_HC * GH_GK * GH_PDS * 4, GH_PV_RS = GH_PV_SX + VBMAX * QW_HC_LOWRANK * 4, GH_PV_SIZE = GH_PV_RS + VBMAX * QW_HC * 4 };
// 驗證是否走三段版（QW_GH_SPLIT=0 關閉，回到與 decode 相同的兩段版）
static bool gh_split(int nb) {
    (void)nb;
    static int v = -1;
    if (v < 0) v = getenv("QW_GH_SPLIT") ? atoi(getenv("QW_GH_SPLIT")) : 1;
    return v != 0;
}

// GGUF 的 hyper-connection kernel（kernels/hyper_gg.metal）：ghc_down / ghc_up 依 token 數 nb 編譯（名稱加 _n{nb}）
static qw_kernel *gh_kernel(qw_engine *e, const char *base, int nb) {
    char name[64], defs[512], err[8192];
    snprintf(name, sizeof name, "%s_n%d", base, nb);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs, "#define ghc_down ghc_down_n%d\n#define ghc_up ghc_up_n%d\n#define ghc_dsk ghc_dsk_n%d\n#define ghc_mid ghc_mid_n%d\n"
             "#define ghc_upv ghc_upv_n%d\n#define D %d\n#define HC %d\n#define NB %d\n"
             "#define GH_R %d\n#define GH_G %d\n#define GU_SG %d\n#define GV_SG %d\n#define GV_RP %d\n#define GK %d\n%s\n", nb, nb, nb, nb, nb, D, QW_HC, nb,
             gh_cfg(nb, 0), gh_cfg(nb, 1), gh_cfg(nb, 2), gh_vcfg(1), gh_vcfg(0), GH_GK,
             getenv("QW_GH_DEFS") ? getenv("QW_GH_DEFS") : "");
    if (!compile(e, defs, src_hyper_gg, sizeof src_hyper_gg, err, sizeof err)) {
        fprintf(stderr, "編譯 hyper_gg（NB=%d）失敗：%s\n", nb, err);
        exit(1);
    }
    k = qw_kernel_get(e->g, name);
    if (!k) { fprintf(stderr, "找不到 %s\n", name); exit(1); }
    return k;
}

// GGUF 的 hyper-connection：nb 個 token。h [nb][HC][D]、out [nb][D]、inj（舊的 inject 部分和）→ inj_out（新的）、x [nb][D]；
// pd 是部分和暫存（[nb][HC][328] float）
static void hyper_gg(qw_engine *e, qw_batch *b, const qw_hyper *H, int nb, uint32_t apply, const qw_buf *h, const qw_buf *out,
                     const qw_buf *inj, const qw_buf *inj_out, const qw_buf *pd, const qw_buf *x) {
    const uint32_t fl[2] = {apply, H->inject ? 1u : 0u};
    if (nb > 1 && gh_split(nb)) { // 驗證：三段版（kernels/hyper_gg.metal 的 ghc_dsk / ghc_mid / ghc_upv，與 decode 逐位元相同）
        const qw_buf *pv = &e->bhpd;
        static int part = -1; // 量測用：QW_GH_PART 位元遮罩（1 ghc_dsk、2 ghc_mid、4 ghc_upv），只編碼部分 kernel
        if (part < 0) part = getenv("QW_GH_PART") ? atoi(getenv("QW_GH_PART")) : 7;
        if (part & 1) {
            qw_batch_kernel(b, gh_kernel(e, "ghc_dsk", nb));
            qw_batch_buf(b, 0, h, 0);
            qw_batch_buf(b, 1, out, 0);
            qw_batch_buf(b, 2, inj, 0);
            bind_t(e, b, 3, H->hc_norm);
            bind_t(e, b, 4, H->mix_down.w);
            if (H->inject) bind_t(e, b, 5, H->inject);
            else bind_t(e, b, 5, H->mix_down.w);
            qw_batch_buf(b, 6, pv, 0);
            qw_batch_bytes(b, 7, fl, sizeof fl);
            qw_batch_dispatch(b, GH_GK, QW_HC, gh_vcfg(0), 32 * gh_vcfg(1), 1, 1);
        }
        if (part & 2) {
            qw_batch_kernel(b, gh_kernel(e, "ghc_mid", nb));
            qw_batch_buf(b, 0, pv, 0);
            qw_batch_buf(b, 1, pv, GH_PV_SX);
            qw_batch_buf(b, 2, pv, GH_PV_RS);
            qw_batch_buf(b, 3, inj_out, 0);
            qw_batch_bytes(b, 4, fl, sizeof fl);
            qw_batch_dispatch(b, nb, 1, 1, QW_HC_LOWRANK, 1, 1);
        }
        if (!(part & 4)) return;
        const int ips = gh_cfg(nb, 2);
        qw_batch_kernel(b, gh_kernel(e, "ghc_upv", nb));
        qw_batch_buf(b, 0, h, 0);
        qw_batch_buf(b, 1, out, 0);
        qw_batch_buf(b, 2, inj, 0);
        bind_t(e, b, 3, H->hc_norm);
        bind_t(e, b, 4, H->mix_up.w);
        qw_batch_buf(b, 5, pv, GH_PV_SX);
        qw_batch_buf(b, 6, pv, GH_PV_RS);
        qw_batch_buf(b, 7, x, 0);
        qw_batch_bytes(b, 8, fl, sizeof fl);
        qw_batch_dispatch(b, D / ips, 1, 1, 32 * ips, 1, 1);
        return;
    }
    qw_kernel *kd = gh_kernel(e, "ghc_down", nb), *ku = gh_kernel(e, "ghc_up", nb);
    const int r = gh_cfg(nb, 0), ips = gh_cfg(nb, 2);
    qw_batch_kernel(b, kd);
    qw_batch_buf(b, 0, h, 0);
    qw_batch_buf(b, 1, out, 0);
    qw_batch_buf(b, 2, inj, 0);
    bind_t(e, b, 3, H->hc_norm);
    bind_t(e, b, 4, H->mix_down.w);
    if (H->inject) bind_t(e, b, 5, H->inject);
    else bind_t(e, b, 5, H->mix_down.w);
    qw_batch_buf(b, 6, pd, 0);
    qw_batch_bytes(b, 7, fl, sizeof fl);
    qw_batch_dispatch(b, QW_HC_LOWRANK / r + 1, QW_HC, 1, D / 8, 1, 1);
    qw_batch_kernel(b, ku);
    qw_batch_buf(b, 0, h, 0);
    qw_batch_buf(b, 1, out, 0);
    qw_batch_buf(b, 2, inj, 0);
    bind_t(e, b, 3, H->hc_norm);
    bind_t(e, b, 4, H->mix_up.w);
    qw_batch_buf(b, 5, pd, 0);
    qw_batch_buf(b, 6, x, 0);
    qw_batch_buf(b, 7, inj_out, 0);
    qw_batch_bytes(b, 8, fl, sizeof fl);
    qw_batch_dispatch(b, D / ips, 1, 1, 32 * ips, 1, 1);
}

// 合併版：殘差更新 + norm → mix_down（split-K）→ SiLU + mix_up + mix（一個 kernel）→ inject
static void hyper_in(qw_engine *e, qw_batch *b, const qw_hyper *H) {
    uint32_t apply = e->pending ? 1 : 0, want = H->inject ? 1 : 0;
    e->pending = false;
    if (H->mix_down.gq) { // GGUF：兩個 kernel（部分和暫存借用 nf：這條路不需要 FP32 的正規化結果）
        hyper_gg(e, b, H, 1, apply, &e->h, &e->out, &e->inj, &e->inj2, &e->nf, &e->x);
        if (want) { qw_buf t = e->inj; e->inj = e->inj2; e->inj2 = t; }
        return;
    }
    // inject 的部分和寫到另一份（舊的那份可能正被這次的 apply 讀取）
    qw_buf *src = &e->inj, *dst = &e->inj2;
    qw_batch_kernel(b, e->k_hc_in);
    qw_batch_buf(b, 0, &e->h, 0);
    qw_batch_buf(b, 1, &e->out, 0);
    qw_batch_buf(b, 2, src, 0);
    bind_t(e, b, 3, H->hc_norm);
    qw_batch_buf(b, 4, &e->nh, 0);
    qw_batch_buf(b, 5, &e->nf, 0);
    qw_batch_bytes(b, 6, &apply, 4);
    if (want) bind_t(e, b, 7, H->inject);
    else qw_batch_buf(b, 7, &e->nf, 0);
    qw_batch_buf(b, 8, dst, 0);
    qw_batch_bytes(b, 9, &want, 4);
    qw_batch_dispatch(b, QW_HC, 1, 1, 1024, 1, 1);
    if (want) { qw_buf t = e->inj; e->inj = e->inj2; e->inj2 = t; }
    qmv(e, b, &H->mix_down, &e->nh, 0, &e->d, false);
    qw_batch_kernel(b, e->k_mixup);
    bind_t(e, b, 0, H->mix_up.w);
    bind_t(e, b, 1, H->mix_up.s);
    bind_t(e, b, 2, H->mix_up.b);
    qw_batch_buf(b, 3, &e->d, 0);
    qw_batch_buf(b, 4, &e->nf, 0);
    qw_batch_buf(b, 5, &e->x, 0);
    qw_batch_dispatch(b, D / 8, 1, 1, 256, 1, 1);
}

static void resid(qw_engine *e, qw_batch *b) {
    qw_batch_kernel(b, e->k_resid);
    qw_batch_buf(b, 0, &e->h, 0);
    qw_batch_buf(b, 1, &e->out, 0);
    qw_batch_buf(b, 2, &e->inj, 0);
    d1(b, HCD, 256);
}

static void hcnorm_f(qw_engine *e, qw_batch *b, const qw_buf *x, qw_t w, const qw_buf *y) {
    qw_batch_kernel(b, e->k_hcnorm_f);
    qw_batch_buf(b, 0, x, 0);
    bind_t(e, b, 1, w);
    qw_batch_buf(b, 2, y, 0);
    qw_batch_dispatch(b, QW_HC, 1, 1, 1024, 1, 1);
}

// n-gram 查表的列（同 k_ngram 的雜湊）：位置 p_first .. p_first + n − 1（tok 的索引）各 16 列抄到 ngW / ngS / ngB 的第 0.. 列
static int32_t argmax_cpu(const float *lg);
static void ng_gather_gg(qw_engine *e, const qw_ple *P, int p_first, int n);
static void ng_gather0(qw_engine *e, const qw_ple *P, int p_first, int n);
static void ng_gather(qw_engine *e, const qw_ple *P, int p_first, int n) {
    const double t0 = qw_now();
    ng_gather0(e, P, p_first, n);
    if (getenv("QW_CACHE_DBG") && n > 64) fprintf(stderr, "[t] ng_gather %d 個位置：%.1f ms\n", n, (qw_now() - t0) * 1e3);
}
static int ng_row0;
static void ng_gather0(qw_engine *e, const qw_ple *P, int p_first, int n) {
    if (e->m->ple_tab) { ng_gather_gg(e, P, p_first, n); return; }
    if (e->ng_gpu) return;
    const int32_t *tk = e->tok.ptr;
    const int64_t *mult = P->multipliers->data, *off = P->head_offsets->data, *size = P->head_vocab->data;
    const uint8_t *TW = (const uint8_t *)e->ngram->map + st_find(e->ngram, "weight")->offset;
    const uint8_t *TS = (const uint8_t *)e->ngram->map + st_find(e->ngram, "scales")->offset;
    const uint8_t *TB = (const uint8_t *)e->ngram->map + st_find(e->ngram, "biases")->offset;
    uint8_t *gw = e->ngW.ptr, *gs = e->ngS.ptr, *gb = e->ngB.ptr;
    dispatch_apply((size_t)n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
        const int p = p_first + (int)t;
        const int64_t t0 = tk[p], t1 = tk[p - 1], t2 = tk[p - 2], eos = EOS_ID;
        const int64_t s1 = t1 != eos ? t1 : eos, s2 = (t1 != eos && t2 != eos) ? t2 : eos;
        const uint64_t m01 = (uint64_t)t0 * (uint64_t)mult[0] ^ (uint64_t)s1 * (uint64_t)mult[1];
        for (int h = 0; h < 16; h++) {
            const uint64_t mixed = h >= 8 ? m01 ^ (uint64_t)s2 * (uint64_t)mult[2] : m01;
            int64_t r = (int64_t)mixed % size[h];
            if (r < 0) r += size[h];
            const uint64_t row = (uint64_t)(r + off[h]);
            const size_t d = ((size_t)t + (size_t)ng_row0) * 16 + (size_t)h;
            memcpy(gw + d * 80, TW + row * 80, 80);
            memcpy(gs + d * 10, TS + row * 10, 10);
            memcpy(gb + d * 10, TB + row * 10, 10);
        }
    });
}

// prefill 用：這段的列若已在上一段時預取好就直接用，否則現在抄
static void ng_gather_pf(qw_engine *e, const qw_ple *P, int p_first, int n) {
    if (e->ng_pre_n == n && e->ng_pre_p == p_first) {
        e->ng_pre_n = 0;
        return;
    }
    e->ng_pre_n = 0;
    ng_gather(e, P, p_first, n);
}

// 第 i 段已送出、GPU 執行中：把第 i + 1 段（位置 p_first … p_first + n − 1）的列在背景抄進另一份 buffer。
// 互換後 e->ngW… / ngX 是新的那份（第 i 段的 command buffer 綁的是舊的那份，不受影響）。下一段編碼前要 ng_prefetch_wait
static void ng_prefetch(qw_engine *e, const qw_ple *P, int p_first, int n) {
    if (e->ng_gpu) return;
    qw_buf t;
    if (e->m->ple_tab) { t = e->ngX; e->ngX = e->ngX2; e->ngX2 = t; }
    else {
        t = e->ngW; e->ngW = e->ngW2; e->ngW2 = t;
        t = e->ngS; e->ngS = e->ngS2; e->ngS2 = t;
        t = e->ngB; e->ngB = e->ngB2; e->ngB2 = t;
    }
    if (!e->ng_grp) e->ng_grp = dispatch_group_create();
    e->ng_pre_p = p_first;
    e->ng_pre_n = n;
    dispatch_group_async(e->ng_grp, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{ ng_gather(e, P, p_first, n); });
}
static void ng_prefetch_wait(qw_engine *e) {
    if (e->ng_grp) dispatch_group_wait(e->ng_grp, DISPATCH_TIME_FOREVER);
}

static void ng_bind(qw_engine *e, qw_batch *b, int gbase) {
    if (e->ng_gpu) {
        qw_batch_buf(b, 5, &e->ngbuf, st_find(e->ngram, "weight")->offset);
        qw_batch_buf(b, 6, &e->ngbuf, st_find(e->ngram, "scales")->offset);
        qw_batch_buf(b, 7, &e->ngbuf, st_find(e->ngram, "biases")->offset);
        gbase = -1;
    } else {
        qw_batch_buf(b, 5, &e->ngW, 0);
        qw_batch_buf(b, 6, &e->ngS, 0);
        qw_batch_buf(b, 7, &e->ngB, 0);
    }
    qw_batch_bytes(b, 10, &gbase, 4);
}

// ---------------------------------------------------------------- GGUF 的 embed 與 PLE（kernels/gg_ple.metal）

// n-gram 表第 h 個 head 的列（同 ng_gather / k_ngram 的雜湊）：位置 p（tok 的索引）
static inline uint64_t ng_row(const qw_ple *P, const int32_t *tk, int p, int h) {
    const int64_t *mult = P->multipliers->data, *off = P->head_offsets->data, *size = P->head_vocab->data;
    const int64_t t0 = tk[p], t1 = tk[p - 1], t2 = tk[p - 2], eos = EOS_ID;
    const int64_t s1 = t1 != eos ? t1 : eos, s2 = (t1 != eos && t2 != eos) ? t2 : eos;
    uint64_t mixed = (uint64_t)t0 * (uint64_t)mult[0] ^ (uint64_t)s1 * (uint64_t)mult[1];
    if (h >= 8) mixed ^= (uint64_t)s2 * (uint64_t)mult[2];
    int64_t r = (int64_t)mixed % size[h];
    if (r < 0) r += size[h];
    return (uint64_t)(r + off[h]);
}

// GGUF 的 n-gram 表（IQ4_NL：每列 160 = 5 塊 × [fp16 d、16 位元組 4-bit 碼]）：位置 p_first .. p_first + n − 1 各 16 列，
// 照 ggml 的 dequantize_row_iq4_nl 精確解成 float（d·碼本值，18 位元以內，float 不捨入），再拆成 hi + lo 兩個 FP16 寫進 ngX[t][0 / 1]
static const int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static int ng_row0 = 0; // 並發（qw_engine_par_step）：n-gram 列寫到 ngX 的第 ng_row0 個位置起（各對話一列）；平常是 0
static void ng_gather_gg(qw_engine *e, const qw_ple *P, int p_first, int n) {
    const int32_t *tk = e->tok.ptr;
    const uint8_t *TAB = e->m->ple_tab;
    const size_t RB = e->m->ple_row_bytes;
    _Float16 *X = (_Float16 *)e->ngX.ptr + (size_t)ng_row0 * 2 * D;
    static _Atomic int bad = 0;
    dispatch_apply((size_t)n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
        _Float16 *hi = X + t * 2 * D, *lo = hi + D;
        int nbad = 0;
        for (int h = 0; h < 16; h++) {
            const uint8_t *row = TAB + ng_row(P, tk, p_first + (int)t, h) * RB;
            for (int k = 0; k < 5; k++) {
                const uint8_t *blk = row + k * 18;
                _Float16 dh;
                memcpy(&dh, blk, 2);
                const float d = (float)dh;
                for (int j = 0; j < 32; j++) {
                    const int q = j < 16 ? blk[2 + j] & 15 : blk[2 + j - 16] >> 4;
                    const float v = d * (float)kv_iq4nl[q];
                    const int i = h * 160 + k * 32 + j;
                    hi[i] = (_Float16)v;
                    lo[i] = (_Float16)(v - (float)hi[i]);
                    nbad += (float)hi[i] + (float)lo[i] != v;
                }
            }
        }
        if (nbad) atomic_fetch_add(&bad, nbad);
    });
    if (atomic_load(&bad)) { fprintf(stderr, "PLE（GGUF）：%d 個 n-gram 值無法拆成 hi + lo\n", atomic_load(&bad)); exit(1); }
    const char *dp = getenv("QW_PLE_DUMP"); // 除錯：每個位置寫 16 個列號（u64）+ 解出的 2560 個 float（tools/gguf_ple_check.py 對照）
    if (dp) {
        FILE *f = fopen(dp, "ab");
        for (int t = 0; t < n && f; t++) {
            uint64_t rows[16];
            float v[D];
            for (int h = 0; h < 16; h++) rows[h] = ng_row(P, tk, p_first + t, h);
            for (int i = 0; i < D; i++) v[i] = (float)X[(size_t)t * 2 * D + i] + (float)X[(size_t)t * 2 * D + D + i];
            fwrite(rows, 8, 16, f);
            fwrite(v, 4, D, f);
        }
        if (f) fclose(f);
    }
}

static qw_kernel *gp_need(qw_engine *e, const char *name) {
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) { fprintf(stderr, "找不到 kernel：%s\n", name); exit(1); }
    return k;
}

// embed（GGUF）：ids[p + r] 那列 → out[r][c·D + i]（c < ncopy，列距 ystride 個 float）
static void gg_embed_rows(qw_engine *e, qw_batch *b, const qw_buf *ids, uint32_t p, uint32_t rows, uint32_t ncopy, uint32_t ystride,
                          const qw_buf *out) {
    const struct { uint32_t p, ncopy, ystride, pad; } a = {p, ncopy, ystride, 0};
    qw_batch_kernel(b, gp_need(e, "gg_embed"));
    bind_t(e, b, 0, e->m->embed_w);
    qw_batch_buf(b, 1, ids, 0);
    qw_batch_bytes(b, 2, &a, sizeof a);
    qw_batch_buf(b, 3, out, 0);
    qw_batch_dispatch(b, rows, 1, 1, D / 32, 1, 1);
}

// y[r][n] = Y[2r][n] + Y[2r + 1][n]（hi、lo 兩份的投影相加）
static void gp_pair_sum_o(qw_engine *e, qw_batch *b, const qw_buf *Y, size_t Yoff, const qw_buf *y, size_t yoff, uint32_t N, uint32_t rows) {
    qw_batch_kernel(b, gp_need(e, "gp_pair_sum"));
    qw_batch_buf(b, 0, Y, Yoff);
    qw_batch_buf(b, 1, y, yoff);
    qw_batch_bytes(b, 2, &N, 4);
    qw_batch_bytes(b, 3, &rows, 4);
    d1(b, N * rows, 256);
}
static void gp_pair_sum(qw_engine *e, qw_batch *b, const qw_buf *Y, const qw_buf *y, size_t yoff, uint32_t N, uint32_t rows) {
    gp_pair_sum_o(e, b, Y, 0, y, yoff, N, rows);
}

// 分流 RMSNorm（T 個 token）：權重是 F32（GGUF）用 gp_hcnorm_w32，BF16 用原本的 kernel
static void hcnorm_f(qw_engine *e, qw_batch *b, const qw_buf *x, qw_t w, const qw_buf *y);
static void pf_hcnorm_f(qw_engine *e, qw_batch *b, const qw_buf *x, qw_t w, const qw_buf *y, uint32_t T);
static void gp_hcnorm(qw_engine *e, qw_batch *b, const qw_buf *x, qw_t w, const qw_buf *y, uint32_t T, bool pf) {
    if (w->dtype != ST_F32) {
        if (pf) pf_hcnorm_f(e, b, x, w, y, T);
        else hcnorm_f(e, b, x, w, y);
        return;
    }
    qw_batch_kernel(b, gp_need(e, "gp_hcnorm_w32"));
    qw_batch_buf(b, 0, x, 0);
    bind_t(e, b, 1, w);
    qw_batch_buf(b, 2, y, 0);
    qw_batch_dispatch(b, QW_HC, T, 1, 1024, 1, 1);
}

static void gp_gate(qw_engine *e, qw_batch *b, const qw_buf *kn, const qw_buf *qn, const qw_buf *val, const qw_buf *gated, uint32_t T) {
    qw_batch_kernel(b, gp_need(e, "gp_ple_gate"));
    qw_batch_buf(b, 0, kn, 0);
    qw_batch_buf(b, 1, qn, 0);
    qw_batch_buf(b, 2, val, 0);
    qw_batch_buf(b, 3, gated, 0);
    qw_batch_dispatch(b, QW_HC, T, 1, 1024, 1, 1);
}

static void qmv_gg(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *x, size_t xoff, const qw_buf *y, size_t yoff, int nb, bool fout);

// PLE 的 key / value 投影（GGUF）：ntok 個 token 的 n-gram 列（從 ngX 的第 ng_base / 16 個位置起，每個 token hi、lo 兩列）
// → pleYk / pleYv 的第 0…2·ntok − 1 列（FP32）。投影不依賴 h：驗證時 VB 個 token 一次算，BF16 權重（65 MB）只讀一次
// （原本逐 token 各讀一次：NB = 8 時多讀約 450 MB）。qmv_gg 每一列的算式與列數無關，結果與逐 token 算逐位元相同
static void ple_gg_proj(qw_engine *e, qw_batch *b, const qw_ple *P, int ntok) {
    const size_t xo = (size_t)(e->ng_base / 16) * 2 * D * 2;
    qmv_gg(e, b, &P->key_proj, &e->ngX, xo, &e->pleYk, 0, 2 * ntok, true);
    qmv_gg(e, b, &P->value_proj, &e->ngX, xo, &e->pleYv, 0, 2 * ntok, true);
}
static int ple_pre_t = -1; // ≥ 0：投影已由 ple_gg_proj 一次算好，這個 token 用第 ple_pre_t 組（驗證）

// PLE 單一 token（GGUF）：n-gram 列在 ngX 的第 ng_base / 16 個位置（hi、lo 兩列）；key / value 都是 FP32（不經過 FP16）
static void ple_gg(qw_engine *e, qw_batch *b, const qw_ple *P) {
    int t = ple_pre_t;
    if (t < 0) {
        ple_gg_proj(e, b, P, 1);
        t = 0;
    }
    gp_pair_sum_o(e, b, &e->pleYk, (size_t)t * 2 * HCD * 4, &e->keyf, 0, HCD, 1);
    gp_pair_sum_o(e, b, &e->pleYv, (size_t)t * 2 * D * 4, &e->valf, 0, D, 1);
    gp_hcnorm(e, b, &e->keyf, P->norm_key, &e->kn, 1, false);
    gp_hcnorm(e, b, &e->h, P->norm_query, &e->qn, 1, false);
    gp_gate(e, b, &e->kn, &e->qn, &e->valf, &e->gated, 1);
    gp_hcnorm(e, b, &e->gated, P->norm_conv, &e->cn, 1, false);
    qw_batch_kernel(b, e->k_ple_conv); // conv1d：GGUF 是 F16，恰為 BF16（gguf.c 轉成 BF16，不失真）
    qw_batch_buf(b, 0, &e->h, 0);
    qw_batch_buf(b, 1, &e->gated, 0);
    qw_batch_buf(b, 2, &e->cn, 0);
    bind_t(e, b, 3, P->conv1d);
    qw_batch_buf(b, 4, &e->ple_st, 0);
    d1(b, HCD, 256);
}

static void ple(qw_engine *e, qw_batch *b, const qw_ple *P) {
    if (e->m->ple_tab) { ple_gg(e, b, P); return; }
    if (getenv("QW_CPU_NGRAM")) ple_embed(e, P, (uint16_t *)e->emb.ptr); // 舊的 CPU 查表（對照用）
    else {
        uint32_t p = (uint32_t)e->pos + 2, eos = EOS_ID;
        qw_batch_kernel(b, e->k_ngram);
        qw_batch_buf(b, 0, &e->tok, 0);
        qw_batch_bytes(b, 1, &p, 4);
        bind_t(e, b, 2, P->multipliers);
        bind_t(e, b, 3, P->head_offsets);
        bind_t(e, b, 4, P->head_vocab);
        ng_bind(e, b, e->ng_base);
        qw_batch_buf(b, 8, &e->emb, 0);
        qw_batch_bytes(b, 9, &eos, 4);
        qw_batch_dispatch(b, 16, 1, 1, 160, 1, 1);
    }
    {
        const qw_qlin *Ls[2] = {&P->key_proj, &P->value_proj};
        const qw_buf *Ys[2] = {&e->keyh, &e->val};
        qmv_multi(e, b, Ls, Ys, 2, &e->emb);
    }
    uint32_t n = HCD;
    qw_batch_kernel(b, e->k_h2f);
    qw_batch_buf(b, 0, &e->keyh, 0);
    qw_batch_buf(b, 1, &e->keyf, 0);
    qw_batch_bytes(b, 2, &n, 4);
    d1(b, n, 256);
    hcnorm_f(e, b, &e->keyf, P->norm_key, &e->kn);
    hcnorm_f(e, b, &e->h, P->norm_query, &e->qn);
    qw_batch_kernel(b, e->k_ple_gate);
    qw_batch_buf(b, 0, &e->kn, 0);
    qw_batch_buf(b, 1, &e->qn, 0);
    qw_batch_buf(b, 2, &e->val, 0);
    qw_batch_buf(b, 3, &e->gated, 0);
    qw_batch_dispatch(b, QW_HC, 1, 1, 1024, 1, 1);
    hcnorm_f(e, b, &e->gated, P->norm_conv, &e->cn);
    qw_batch_kernel(b, e->k_ple_conv);
    qw_batch_buf(b, 0, &e->h, 0);
    qw_batch_buf(b, 1, &e->gated, 0);
    qw_batch_buf(b, 2, &e->cn, 0);
    bind_t(e, b, 3, P->conv1d);
    qw_batch_buf(b, 4, &e->ple_st, 0);
    d1(b, HCD, 256);
}

// 除錯：QW_DN_TAP=檔案:li → 第 li 個 DeltaNet 層（0…35）的輸入 x 與輸出 out（各 D 個 half）記下來，batch 結束後附加到檔案：
// 每筆 [int32 n][n 列 x][n 列 out]。decode（qw_engine_step）與 prefill 都會記；給 numpy 參考實作比對（tools/dn_ref.py）
static int dn_tap_li(void) {
    static int li = -2;
    if (li == -2) { const char *v = getenv("QW_DN_TAP"), *c = v ? strrchr(v, ':') : NULL; li = c ? atoi(c + 1) : -1; }
    return li;
}
enum { DN_TAP_CAP = 4096 };
static void dn_tap_copy(qw_engine *e, qw_batch *b, const qw_buf *src, bool out, uint32_t rows) {
    if (!e->dntap.ptr && !alloc(e, &e->dntap, (size_t)2 * DN_TAP_CAP * D * 2)) { fprintf(stderr, "QW_DN_TAP：配置失敗\n"); exit(1); }
    if (e->dntap_n + rows > DN_TAP_CAP) { fprintf(stderr, "QW_DN_TAP：超過 %d 列\n", DN_TAP_CAP); exit(1); }
    const uint32_t nf = rows * D / 2;
    qw_batch_kernel(b, e->k_copyf);
    qw_batch_buf(b, 0, src, 0);
    qw_batch_buf(b, 1, &e->dntap, ((size_t)(out ? DN_TAP_CAP : 0) + e->dntap_n) * D * 2);
    qw_batch_bytes(b, 2, &nf, 4);
    d1(b, nf, 256);
}
static void dn_tap_flush(qw_engine *e) {
    if (!e->dntap_n) return;
    const char *v = getenv("QW_DN_TAP");
    char path[1024];
    snprintf(path, sizeof path, "%.*s", (int)(strrchr(v, ':') - v), v);
    FILE *f = fopen(path, "ab");
    const int32_t n = e->dntap_n;
    fwrite(&n, 4, 1, f);
    fwrite(e->dntap.ptr, 2, (size_t)n * D, f);
    fwrite((char *)e->dntap.ptr + (size_t)DN_TAP_CAP * D * 2, 2, (size_t)n * D, f);
    fclose(f);
    e->dntap_n = 0;
}

static void deltanet_core(qw_engine *e, qw_batch *b, const qw_deltanet *N, int li);
static bool mlx_xf(const qw_qlin *L);

// ---- GGUF DeltaNet 投影（kernels/dn_proj.metal）：qkv / z（Q4_K、Q5_K、Q6_K、IQ4_XS）+ a / b（BF16）一次 dispatch，out_proj 另一個
enum { DN_NSG = 8 }; // 純量版：每個 threadgroup 幾個 SIMD group
static int dn_r(int nb) { // 純量版每個 SIMD group 幾列（dn_proj.metal 的 R）：token 多時暫存器不夠，改 1 列
    static int th = -1;
    if (th < 0) th = getenv("QW_DN_R1") ? atoi(getenv("QW_DN_R1")) : 1;
    return nb >= th ? 1 : 2;
}
enum { DN_TSG = 8 }; // tensor 單元版：每個 threadgroup 32 列、8 個 SIMD group 分 K（dn_proj.metal 的 DN_TSG）
static bool dn_tc(int nb) { // 驗證 token 數 ≥ QW_DN_TC 時走 tensor 單元版（x 緩衝區要有 8 列：只用在驗證）
    static int v = -1;
    // 預設關閉（99）：tensor 版的捨入與 decode 不同，推測輸出就不再等於 greedy；而且真實生成時每輪 GPU 時間並沒有比較短
    // （kbench 單層 NB = 8 較快，但端到端 prose 17.46 → 16.99 ms、code 29.13 → 29.07 ms）。QW_DN_TC=7 恢復
    if (v < 0) v = getenv("QW_DN_TC") ? atoi(getenv("QW_DN_TC")) : 99;
    return nb >= 2 && nb >= v;
}
static bool dn_fmt_ok(int gq) { return gq == 12 + 1 || gq == 13 + 1 || gq == 14 + 1 || gq == 23 + 1; }
static bool dn_in_ok(const qw_deltanet *N) {
    return !getenv("QW_DN_GENERIC") && dn_fmt_ok(N->in_qkv.gq) && dn_fmt_ok(N->in_z.gq) && N->in_a.gq == 30 + 1 && N->in_b.gq == 30 + 1 &&
           N->in_qkv.in == 2560 && N->in_qkv.out == 10240 && N->in_z.out == 6144 && N->in_a.out == 48 && N->in_b.out == 48;
}
static bool dn_out_ok(const qw_deltanet *N) { // out_proj 另外接受 Q8_0（dn_out_8）
    return !getenv("QW_DN_GENERIC") && (dn_fmt_ok(N->out.gq) || N->out.gq == 8 + 1) && N->out.in == 6144 && N->out.out == 2560;
}
static qw_kernel *dn_kernel(qw_engine *e, const char *base, int nb) { // base 例："dn_in_14_12"；每個 NB 編一個 library（含所有格式組合）
    char name[64], defs[160], err[8192];
    snprintf(name, sizeof name, "%s_n%d", base, nb);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs, "#define NB %d\n#define NBS \"%d\"\n#define R %d\n", nb, nb, dn_r(nb));
    if (!compile(e, defs, src_dn_proj, sizeof src_dn_proj, err, sizeof err)) { fprintf(stderr, "編譯 dn_proj（NB=%d）失敗：%s\n", nb, err); exit(1); }
    if (!(k = qw_kernel_get(e->g, name))) { fprintf(stderr, "找不到 %s\n", name); exit(1); }
    return k;
}
// 並發版（kernels/dn_proj.metal 的 DN_MS）：第 t 個 token 是另一個對話（conv 狀態各自一份）。名稱 <base>_n{nb}ms
static qw_kernel *dn_kernel_ms(qw_engine *e, const char *base, int nb) {
    char name[64], defs[200], err[8192];
    snprintf(name, sizeof name, "%s_n%dms", base, nb);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs, "#define NB %d\n#define NBS \"%dms\"\n#define R %d\n#define DN_MS 1\n", nb, nb, dn_r(nb));
    if (!compile(e, defs, src_dn_proj, sizeof src_dn_proj, err, sizeof err)) { fprintf(stderr, "編譯 dn_proj（NB=%d，並發）失敗：%s\n", nb, err); exit(1); }
    if (!(k = qw_kernel_get(e->g, name))) { fprintf(stderr, "找不到 %s\n", name); exit(1); }
    return k;
}
typedef struct { uint32_t xs, ys[4], ks, pr; } dn_in_args;
typedef struct { uint32_t ks, pr, b0, b1, b2, ys[4]; } dn_ksum_args;
enum { DN_KSMAX = 8 };
static uint32_t dn_ks(bool out) { // tensor 單元版 K 分幾段（QW_DN_KS、QW_DN_KSO）：threadgroup 不夠多時補平行度
    static int v[2] = {0, 0};
    if (!v[out]) {
        const char *s = getenv(out ? "QW_DN_KSO" : "QW_DN_KS");
        v[out] = s ? atoi(s) : out ? 4 : 1;
        if (v[out] < 1 || v[out] > DN_KSMAX || (out ? 6144 : 2560) % (32 * v[out])) { fprintf(stderr, "QW_DN_KS / KSO 不合法\n"); exit(1); }
    }
    return (uint32_t)v[out];
}
// 部分和 → 輸出（最多 4 段列）
static void dn_ksum(qw_engine *e, qw_batch *b, const dn_ksum_args *a, const qw_buf *const *y, int nb) {
    qw_batch_kernel(b, dn_kernel(e, "dn_ksum", nb));
    qw_batch_buf(b, 0, &e->dnP, 0);
    for (int i = 0; i < 4; i++) qw_batch_buf(b, 1 + i, y[i], 0);
    qw_batch_bytes(b, 5, a, sizeof *a);
    d1(b, a->pr * (uint32_t)nb, 256);
}
static void dn_need_p(qw_engine *e) {
    if (!e->dnP.ptr && !alloc(e, &e->dnP, (size_t)DN_KSMAX * 8 * 16480 * 4)) { fprintf(stderr, "dnP 配置失敗\n"); exit(1); }
}
// x [nb][2560] → qkv [nb][10240]、z [nb][6144]、a / b [nb][48]
static void dn_in(qw_engine *e, qw_batch *b, const qw_deltanet *N, const qw_buf *x, const qw_buf *yq, const qw_buf *yz, const qw_buf *ya,
                  const qw_buf *yb, int nb) {
    char base[32];
    const bool tc = dn_tc(nb);
    snprintf(base, sizeof base, "dn_in%s_%d_%d", tc ? "_tc" : "", N->in_qkv.gq - 1, N->in_z.gq - 1);
    const uint32_t ks = tc ? dn_ks(false) : 1;
    const dn_in_args a = {2560, {10240, 6144, 48, 48}, ks, 16480};
    if (tc) dn_need_p(e);
    qw_batch_kernel(b, dn_kernel(e, base, nb));
    bind_t(e, b, 0, N->in_qkv.w);
    bind_t(e, b, 1, N->in_z.w);
    bind_t(e, b, 2, N->in_a.w);
    bind_t(e, b, 3, N->in_b.w);
    qw_batch_buf(b, 4, x, 0);
    qw_batch_buf(b, 5, yq, 0);
    qw_batch_buf(b, 6, yz, 0);
    qw_batch_buf(b, 7, ya, 0);
    qw_batch_buf(b, 8, yb, 0);
    qw_batch_bytes(b, 9, &a, sizeof a);
    if (tc) {
        qw_batch_buf(b, 10, &e->dnP, 0);
        qw_batch_dispatch(b, (16480 + 31) / 32, ks, 1, 32 * DN_TSG, 1, 1);
        if (ks > 1) {
            const dn_ksum_args k = {ks, 16480, 10240, 16384, 16432, {10240, 6144, 48, 48}};
            const qw_buf *ys[4] = {yq, yz, ya, yb};
            dn_ksum(e, b, &k, ys, nb);
        }
    }
    else qw_batch_dispatch(b, 16480 / (DN_NSG * dn_r(nb)), 1, 1, 32 * DN_NSG, 1, 1);
}
// out_proj：x [nb][6144] → y [nb][2560]
static void dn_out(qw_engine *e, qw_batch *b, const qw_deltanet *N, const qw_buf *x, const qw_buf *y, int nb) {
    char base[32];
    const bool tc = dn_tc(nb);
    snprintf(base, sizeof base, "dn_out%s_%d", tc ? "_tc" : "", N->out.gq - 1);
    const uint32_t ks = tc ? dn_ks(true) : 1;
    const dn_in_args a = {6144, {2560, 0, 0, 0}, ks, 2560};
    if (tc) dn_need_p(e);
    qw_batch_kernel(b, dn_kernel(e, base, nb));
    bind_t(e, b, 0, N->out.w);
    qw_batch_buf(b, 4, x, 0);
    qw_batch_buf(b, 5, y, 0);
    qw_batch_bytes(b, 9, &a, sizeof a);
    if (tc) {
        qw_batch_buf(b, 10, &e->dnP, 0);
        qw_batch_dispatch(b, 2560 / 32, ks, 1, 32 * DN_TSG, 1, 1);
        if (ks > 1) {
            const dn_ksum_args k = {ks, 2560, 2560, 2560, 2560, {2560, 0, 0, 0}};
            const qw_buf *ys[4] = {y, y, y, y};
            dn_ksum(e, b, &k, ys, nb);
        }
    }
    else qw_batch_dispatch(b, 2560 / (DN_NSG * dn_r(nb)), 1, 1, 32 * DN_NSG, 1, 1);
}
// ---- GGUF DeltaNet 驗證的延後寫回（QW_DN_DEFER=0 關閉）：驗證時狀態只讀、每步的 δ 存起來；接受 J 個 token 後，
// 下次用到這一層（或要讀整份狀態）之前才用 gdn_commit 從原狀態重放 J 步。省掉每個猜測 token 各寫一份 3 MB 狀態
static bool dn_defer(const qw_engine *e) {
    static int v = -1;
    if (v < 0) v = getenv("QW_DN_DEFER") ? atoi(getenv("QW_DN_DEFER")) : 1;
    return v && e->m->lin_tiled && e->dnD.ptr;
}
// 每層兩組 × VBMAX 份的 k、衰減、δ（驗證與 decode 的延後寫回共用；約 17 MB）
static bool dn_alloc_defer(qw_engine *e) {
    return alloc(e, &e->dnK, (size_t)NLIN * 2 * VBMAX * 2048 * 2) && alloc(e, &e->dnG, (size_t)NLIN * 2 * VBMAX * 64 * 4) &&
           alloc(e, &e->dnD, (size_t)NLIN * 2 * VBMAX * 48 * 128 * 4);
}
static size_t dn_koff(int li, int s) { return (size_t)(li * 2 + s) * VBMAX * 2048 * 2; }
static size_t dn_goff(int li, int s) { return (size_t)(li * 2 + s) * VBMAX * 64 * 4; }
static size_t dn_doff(int li, int s) { return (size_t)(li * 2 + s) * VBMAX * 48 * 128 * 4; }
static void dn_apply(qw_engine *e, qw_batch *b, int li) {
    if (!(e->dn_pmask >> li & 1)) return;
    e->dn_pmask &= ~(1ull << li);
    const uint32_t J = e->dn_lJ[li];
    const int ps = e->dn_lp[li];
    qw_batch_kernel(b, e->gdn_commit_t);
    qw_batch_buf(b, 0, &e->dnK, dn_koff(li, ps));
    qw_batch_buf(b, 1, &e->dnG, dn_goff(li, ps));
    qw_batch_buf(b, 2, &e->dnD, dn_doff(li, ps));
    qw_batch_buf(b, 3, &e->S, (size_t)li * 48 * 128 * 128 * 4);
    qw_batch_bytes(b, 4, &J, 4);
    qw_batch_dispatch(b, 48, 128 / 32, 1, 256, 1, 1);
}
static void dn_flush(qw_engine *e) { // 要讀整份狀態之前
    if (!e->dn_pmask) return;
    qw_batch *b = qw_batch_begin(e->g, false);
    for (int li = 0; li < NLIN; li++) dn_apply(e, b, li);
    qw_batch_end(b);
}
// 驗證後接受了 j 個猜測（共 j + 1 個 token）：延後模式記下待重放的步數，否則把狀態換成第 j 個 token 之後的那份
static void dn_accept(qw_engine *e, int j) {
    if (e->dn_vdef) { memset(e->dn_lJ, j + 1, sizeof e->dn_lJ); e->dn_pmask = (1ull << NLIN) - 1; }
    else if (j >= 1) { qw_buf t1 = e->S; e->S = e->Sx[j - 1]; e->Sx[j - 1] = t1; }
}

// ---- GGUF DeltaNet 合併版（QW_DN_FUSE=0 關閉）：每層 3 個 dispatch
//   dn_inf（投影 + conv + 衰減 / β）→ gdn_fstep（待重放的步數 + q/k 正規化 + 遞迴 + gated RMSNorm）→ dn_out
// 原本是 dn_in → k_gdn_in → (gdn_commit) → gdn_step → k_normgate → dn_out；結果逐位元相同
static bool dn_fuse(const qw_engine *e, const qw_deltanet *N) {
    static int v = -1;
    if (v < 0) v = getenv("QW_DN_FUSE") ? atoi(getenv("QW_DN_FUSE")) : 1;
    return v && e->m->lin_tiled && e->dncnt.ptr && dn_in_ok(N) && dn_out_ok(N);
}
static int gf_dvb(void) { // gdn_fstep 每個 threadgroup 幾欄（QW_GF_DVB：16、32、64、128）
    static int v = 0;
    if (!v) v = getenv("QW_GF_DVB") ? atoi(getenv("QW_GF_DVB")) : 32;
    if (v != 16 && v != 32 && v != 64 && v != 128) { fprintf(stderr, "QW_GF_DVB 不合法\n"); exit(1); }
    return v;
}
static qw_kernel *gf_kernel(qw_engine *e, int nb) {
    char name[64], defs[512], err[8192];
    snprintf(name, sizeof name, "gdn_fstep_n%d", nb);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs,
             "#define GDN_TILED 1\n#define GDN_FUSED 1\n#define gdn_step gdn_step_fu%d\n#define gdn_stepn gdn_stepn_fu%d\n"
             "#define gdn_fstep gdn_fstep_n%d\n#define DK 128\n#define DV 128\n#define HK 16\n#define HV 48\n#define DVB 32\n#define NB %d\n#define GF_DVB %d\n",
             nb, nb, nb, nb, gf_dvb());
    if (!compile(e, defs, src_gdn, sizeof src_gdn, err, sizeof err)) { fprintf(stderr, "編譯 gdn_fstep（NB=%d）失敗：%s\n", nb, err); exit(1); }
    if (!(k = qw_kernel_get(e->g, name))) { fprintf(stderr, "找不到 %s\n", name); exit(1); }
    return k;
}
// 投影 + 每列收尾：yq ← conv 輸出、yz ← z、la / be ← 衰減 / β（列距 64，可以帶偏移）；so[t − 1] 是第 t 個 token 之後的 conv 狀態
static void dn_inf(qw_engine *e, qw_batch *b, const qw_deltanet *N, int li, const qw_buf *x, const qw_buf *yq, const qw_buf *yz,
                   const qw_buf *la, size_t laoff, const qw_buf *be, int nb, const qw_buf *so) {
    char base[32];
    const bool tc = dn_tc(nb);
    snprintf(base, sizeof base, "dn_inf%s_%d_%d", tc ? "_tc" : "", N->in_qkv.gq - 1, N->in_z.gq - 1);
    const dn_in_args a = {2560, {10240, 6144, 64, 64}, 1, 16480};
    const size_t cst = (size_t)li * 3 * 10240 * 4;
    qw_batch_kernel(b, dn_kernel(e, base, nb));
    bind_t(e, b, 0, N->in_qkv.w);
    bind_t(e, b, 1, N->in_z.w);
    bind_t(e, b, 2, N->in_a.w);
    bind_t(e, b, 3, N->in_b.w);
    qw_batch_buf(b, 4, x, 0);
    qw_batch_buf(b, 5, yq, 0);
    qw_batch_buf(b, 6, yz, 0);
    qw_batch_buf(b, 7, la, laoff);
    qw_batch_buf(b, 8, be, 0);
    qw_batch_bytes(b, 9, &a, sizeof a);
    qw_batch_buf(b, 10, yq, 0); // 只有 K 分段時用到（合併版不分段）
    bind_t(e, b, 11, N->conv1d);
    qw_batch_buf(b, 12, &e->conv_st, cst);
    bind_t(e, b, 13, N->A_neg);
    bind_t(e, b, 14, N->dt_bias);
    for (int t = 1; t < VBMAX; t++) qw_batch_buf(b, 14 + t, t < nb ? &so[t - 1] : &e->conv_st, cst);
    if (tc) qw_batch_dispatch(b, (16480 + 31) / 32, 1, 1, 32 * DN_TSG, 1, 1);
    else qw_batch_dispatch(b, 16480 / (DN_NSG * dn_r(nb)), 1, 1, 32 * DN_NSG, 1, 1);
}
// gdn_fstep：mode 0 = decode（寫回狀態）、1 = 延後寫回的驗證（δ、k 寫到第 ws 組第 0… 份）、
// 2 = decode 延後寫回（δ、k 接在第 dn_lp[li] 組待重放的後面，狀態不寫）；有待重放的步數時從第 dn_lp[li] 組讀
typedef struct { uint32_t J, defer; } gf_args;
static void gdn_fstep(qw_engine *e, qw_batch *b, const qw_deltanet *N, int li, int nb, int mode, const qw_buf *cv, const qw_buf *la,
                      size_t laoff, const qw_buf *be, const qw_buf *o, const qw_buf *z, const qw_buf *gy, int ws) {
    gf_args ga = {0, (uint32_t)mode};
    const int ps = e->dn_lp[li];
    if (e->dn_pmask >> li & 1) ga.J = e->dn_lJ[li];
    size_t kd = 0, dd = 0; // 寫入位置（mode 2：接在待重放的 J 份後面）
    if (mode == 2) { ws = ps; kd = (size_t)ga.J * 2048 * 2; dd = (size_t)ga.J * 48 * 128 * 4; }
    if (mode == 2) { e->dn_lJ[li] = (uint8_t)(ga.J + 1); e->dn_pmask |= 1ull << li; }
    else e->dn_pmask &= ~(1ull << li);
    qw_batch_kernel(b, gf_kernel(e, nb));
    qw_batch_buf(b, 0, cv, 0);
    qw_batch_buf(b, 1, la, laoff);
    qw_batch_buf(b, 2, be, 0);
    qw_batch_buf(b, 3, &e->S, (size_t)li * 48 * 128 * 128 * 4);
    qw_batch_buf(b, 4, o, 0);
    qw_batch_buf(b, 5, z, 0);
    bind_t(e, b, 6, N->norm);
    qw_batch_buf(b, 7, gy, 0);
    qw_batch_buf(b, 8, &e->dncnt, 0);
    const bool hasd = e->dnK.ptr != NULL; // 驗證緩衝區還沒配置時沒有重放也沒有延後寫回
    qw_batch_buf(b, 9, hasd ? &e->dnK : gy, hasd ? dn_koff(li, ws) + kd : 0);
    qw_batch_buf(b, 10, hasd ? &e->dnD : gy, hasd ? dn_doff(li, ws) + dd : 0);
    qw_batch_buf(b, 11, hasd ? &e->dnK : gy, hasd ? dn_koff(li, ps) : 0);
    qw_batch_buf(b, 12, hasd ? &e->dnG : gy, hasd ? dn_goff(li, ps) : 0);
    qw_batch_buf(b, 13, hasd ? &e->dnD : gy, hasd ? dn_doff(li, ps) : 0);
    qw_batch_bytes(b, 14, &ga, sizeof ga);
    qw_batch_dispatch(b, 48, 128 / gf_dvb(), 1, 8 * gf_dvb(), 1, 1);
}

static void deltanet(qw_engine *e, qw_batch *b, const qw_deltanet *N, int li) {
    if (!dn_fuse(e, N)) dn_apply(e, b, li); // 合併版在 gdn_fstep 裡重放
    const bool tap = li == dn_tap_li();
    if (tap) dn_tap_copy(e, b, &e->x, false, 1);
    deltanet_core(e, b, N, li);
    if (tap) { dn_tap_copy(e, b, &e->out, true, 1); e->dntap_n++; }
}

// decode 的延後寫回（QW_DN_DDEFER = 最多累積幾步，0 關閉；預設 VBMAX）：狀態只讀、δ / k / 衰減接在待重放的後面，
// 累積到上限的下一個 token 才重放並寫回一次。重放的捨入與遞迴相同，結果與每步寫回逐位元相同；省下大部分 3 MB 狀態的寫入
static int dn_ddefer(const qw_engine *e) {
    static int v = -1;
    if (v < 0) v = getenv("QW_DN_DDEFER") ? atoi(getenv("QW_DN_DDEFER")) : VBMAX;
    if (v > VBMAX) v = VBMAX;
    return dn_defer(e) ? v : 0;
}
static void deltanet_core(qw_engine *e, qw_batch *b, const qw_deltanet *N, int li) {
    if (dn_fuse(e, N)) {
        const int J = e->dn_pmask >> li & 1 ? e->dn_lJ[li] : 0, p = e->dn_lp[li];
        if (J < dn_ddefer(e)) { // 延後：衰減寫到第 p 組第 J 份（重放用），狀態不寫
            const size_t lo = dn_goff(li, p) + (size_t)J * 64 * 4;
            dn_inf(e, b, N, li, &e->x, &e->qkv, &e->z, &e->dnG, lo, &e->be, 1, NULL);
            gdn_fstep(e, b, N, li, 1, 2, &e->qkv, &e->dnG, lo, &e->be, &e->go, &e->z, &e->gy, p);
        } else {
            dn_inf(e, b, N, li, &e->x, &e->qkv, &e->z, &e->la, 0, &e->be, 1, NULL);
            gdn_fstep(e, b, N, li, 1, 0, &e->qkv, &e->la, 0, &e->be, &e->go, &e->z, &e->gy, 0);
        }
        dn_out(e, b, N, &e->gy, &e->out, 1);
        return;
    }
    if (dn_in_ok(N)) dn_in(e, b, N, &e->x, &e->qkv, &e->z, &e->ain, &e->bin, 1);
    else {
        const qw_qlin *Ls[4] = {&N->in_qkv, &N->in_z, &N->in_a, &N->in_b};
        const qw_buf *Ys[4] = {&e->qkv, &e->z, &e->ain, &e->bin};
        qmv_multi(e, b, Ls, Ys, 4, &e->x);
    }
    const bool tl = e->m->lin_tiled; // GGUF：V head 交錯順序、A_neg（F32）
    if (tl || !getenv("QW_NO_FUSE_GDN")) {
        qw_batch_kernel(b, tl ? e->k_gdn_in_t : e->k_gdn_in);
        qw_batch_buf(b, 0, &e->qkv, 0);
        bind_t(e, b, 1, N->conv1d);
        qw_batch_buf(b, 2, &e->conv_st, (size_t)li * 3 * 10240 * 4);
        qw_batch_buf(b, 3, &e->ain, 0);
        qw_batch_buf(b, 4, &e->bin, 0);
        bind_t(e, b, 5, tl ? N->A_neg : N->A_log);
        bind_t(e, b, 6, N->dt_bias);
        qw_batch_buf(b, 7, &e->gq, 0);
        qw_batch_buf(b, 8, &e->gk, 0);
        qw_batch_buf(b, 9, &e->vbuf, 0);
        qw_batch_buf(b, 10, &e->la, 0);
        qw_batch_buf(b, 11, &e->be, 0);
        qw_batch_buf(b, 12, &e->conv_st, (size_t)li * 3 * 10240 * 4);
        qw_batch_dispatch(b, 16 + 48, 1, 1, 128, 1, 1);
    } else {
    uint32_t C = 10240;
    qw_batch_kernel(b, e->k_conv4);
    qw_batch_buf(b, 0, &e->qkv, 0);
    bind_t(e, b, 1, N->conv1d);
    qw_batch_buf(b, 2, &e->conv_st, (size_t)li * 3 * C * 4);
    qw_batch_buf(b, 3, &e->conv, 0);
    qw_batch_bytes(b, 4, &C, 4);
    d1(b, C, 256);
    qw_batch_kernel(b, e->k_gdn_qk);
    qw_batch_buf(b, 0, &e->conv, 0);
    qw_batch_buf(b, 1, &e->ain, 0);
    qw_batch_buf(b, 2, &e->bin, 0);
    bind_t(e, b, 3, N->A_log);
    bind_t(e, b, 4, N->dt_bias);
    qw_batch_buf(b, 5, &e->gq, 0);
    qw_batch_buf(b, 6, &e->gk, 0);
    qw_batch_buf(b, 7, &e->la, 0);
    qw_batch_buf(b, 8, &e->be, 0);
    qw_batch_dispatch(b, 16, 1, 1, 128, 1, 1);
    }
    qw_batch_kernel(b, tl ? e->gdn_step_t : e->gdn_step);
    qw_batch_buf(b, 0, &e->gq, 0);
    qw_batch_buf(b, 1, &e->gk, 0);
    if (tl || !getenv("QW_NO_FUSE_GDN")) qw_batch_buf(b, 2, &e->vbuf, 0);
    else qw_batch_buf(b, 2, &e->conv, 4096 * 2); // v 在 conv 輸出的 [4096, 10240)
    qw_batch_buf(b, 3, &e->la, 0);
    qw_batch_buf(b, 4, &e->be, 0);
    qw_batch_buf(b, 5, &e->S, (size_t)li * 48 * 128 * 128 * 4);
    qw_batch_buf(b, 6, &e->go, 0);
    qw_batch_buf(b, 7, &e->ssp, 0);
    qw_batch_buf(b, 8, &e->S, (size_t)li * 48 * 128 * 128 * 4);
    qw_batch_dispatch(b, 48, 128 / 32, 1, 256, 1, 1);
    if (mlx_xf(&N->out)) { // gated RMSNorm 併進 out_proj 的輸入（--exact-spec 時驗證用 vb_xform，算式相同）
        qmv_args a = {(uint32_t)N->out.out, 0, (uint32_t)N->out.out, 0};
        qw_batch_kernel(b, N->out.bits == 4 ? e->qmv_xf1_4 : e->qmv_xf1);
        bind_t(e, b, 0, N->out.w); bind_t(e, b, 1, N->out.s); bind_t(e, b, 2, N->out.b);
        qw_batch_buf(b, 3, &e->go, 0);
        qw_batch_buf(b, 4, &e->out, 0);
        qw_batch_buf(b, 5, &e->z, 0);
        qw_batch_bytes(b, 6, &a, sizeof a);
        qw_batch_buf(b, 7, &e->ssp, 0);
        bind_t(e, b, 8, N->norm);
        qw_batch_dispatch(b, (a.N + 7) / 8, 1, 1, 256, 1, 1);
        return;
    }
    qw_batch_kernel(b, e->k_normgate);
    qw_batch_buf(b, 0, &e->go, 0);
    qw_batch_buf(b, 1, &e->z, 0);
    bind_t(e, b, 2, N->norm);
    qw_batch_buf(b, 3, &e->gy, 0);
    qw_batch_dispatch(b, 48, 1, 1, 128, 1, 1);
    if (dn_out_ok(N)) dn_out(e, b, N, &e->gy, &e->out, 1);
    else qmv(e, b, &N->out, &e->gy, 0, &e->out, false);
}

typedef struct { uint32_t L, pos, nvis, ntok, sparse; } qsa_args;

// 量測用（QW_ATT_TEST）：只編碼部分步驟：1 投影、2 準備（norm / RoPE / KV、indexer）、4 打分數、8 選取、16 attention 核心、
// 32 合併、64 gate、128 o_proj
static unsigned at_mask = ~0u;

typedef struct {
    const qw_buf *x, *qg, *kk, *vv, *iqk, *aq, *iq, *ao, *ay, *out;
    size_t oqg, okk, ovv, oiqk, oay, oao; // 並發（GGUF 核心）：qg / kk / vv / iqk / ay 的位移（批次緩衝區的第 t 欄）；平常是 0
} qsa_io;
static void qsa_layer(qw_engine *e, qw_batch *b, const qw_attn *A, int ai, uint32_t p0, int nb, const qsa_io *io);

static void attention(qw_engine *e, qw_batch *b, const qw_attn *A, int ai) {
    const qsa_io io = {&e->x, &e->qg, &e->kk, &e->vv, &e->iqk, &e->aq, &e->iq, &e->ao, &e->ay, &e->out};
    qsa_layer(e, b, A, ai, (uint32_t)e->pos, 1, &io);
}

static void moe_old(qw_engine *e, qw_batch *b, const qw_moe *M);

// router（MLX 8-bit 或 BF16）→ rlog [E + 1]
static void moe_router1(qw_engine *e, qw_batch *b, const qw_moe *M) {
    if (M->router.gq) { gg_router(e, b, M, 1, &e->x, &e->rlog); return; }
    qw_batch_kernel(b, e->moe_router);
    bind_t(e, b, 0, M->router.w);
    bind_t(e, b, 1, M->router.s);
    bind_t(e, b, 2, M->router.b);
    bind_t(e, b, 3, M->sh_gate_w);
    qw_batch_buf(b, 4, &e->x, 0);
    qw_batch_buf(b, 5, &e->rlog, 0);
    qw_batch_dispatch(b, (QW_EXPERTS + 1 + 7) / 8, 1, 1, 256, 1, 1);
}

static void moe(qw_engine *e, qw_batch *b, const qw_moe *M) {
    if (M->giq && !gg_old()) { // GGUF：kernels/moe_gg.metal（router 也在裡面）
        gg_moe(e, b, M, 1, &e->x, &e->rlog, &e->ggsel, &e->out);
        return;
    }
    if (M->giq) { // routed expert 是 GGUF 區塊（舊版 kernel，QW_MOE_OLD；shared 須為 MLX、router 可為 BF16）
        if (M->router.gq) moe_router1(e, b, M); else {
        qw_batch_kernel(b, e->moe_router);
        bind_t(e, b, 0, M->router.w);
        bind_t(e, b, 1, M->router.s);
        bind_t(e, b, 2, M->router.b);
        bind_t(e, b, 3, M->sh_gate_w);
        qw_batch_buf(b, 4, &e->x, 0);
        qw_batch_buf(b, 5, &e->rlog, 0);
        qw_batch_dispatch(b, (QW_EXPERTS + 1 + 7) / 8, 1, 1, 256, 1, 1);
        }
        qw_batch_kernel(b, iq_kernel(e, "moe_gateup_iq", M->fgu, M->fd, 0));
        bind_t(e, b, 0, M->giq);
        bind_t(e, b, 3, M->uiq);
        bind_t(e, b, 6, M->sh_gate.w); bind_t(e, b, 7, M->sh_gate.s); bind_t(e, b, 8, M->sh_gate.b);
        bind_t(e, b, 9, M->sh_up.w); bind_t(e, b, 10, M->sh_up.s); bind_t(e, b, 11, M->sh_up.b);
        qw_batch_buf(b, 12, &e->x, 0);
        qw_batch_buf(b, 13, &e->rlog, 0);
        qw_batch_buf(b, 14, &e->H, 0);
        qw_batch_dispatch(b, QW_MOE_INTER / 8, QW_TOPK + 1, 1, 256, 1, 1);
        qw_batch_kernel(b, iq_kernel(e, "moe_down_iq", M->fgu, M->fd, 0));
        bind_t(e, b, 0, M->diq);
        bind_t(e, b, 3, M->sh_down.w); bind_t(e, b, 4, M->sh_down.s); bind_t(e, b, 5, M->sh_down.b);
        qw_batch_buf(b, 6, &e->H, 0);
        qw_batch_buf(b, 7, &e->rlog, 0);
        qw_batch_buf(b, 9, &e->out, 0);
        qw_batch_dispatch(b, D / 8, 1, 1, 256, 1, 1);
        return;
    }
    if (getenv("QW_NO_FUSE_MOE") && !M->router.gq) { moe_old(e, b, M); return; }
    moe_router1(e, b, M);
    const bool gu2 = getenv("QW_MOE_GU2") != NULL, dn1 = getenv("QW_MOE_DN2") == NULL; // down 每組 1 列（真實權重量過較快）
    qw_batch_kernel(b, gu2 ? e->moe_gateup_f2 : e->moe_gateup_f);
    bind_t(e, b, 0, M->gate.w); bind_t(e, b, 1, M->gate.s); bind_t(e, b, 2, M->gate.b);
    bind_t(e, b, 3, M->up.w); bind_t(e, b, 4, M->up.s); bind_t(e, b, 5, M->up.b);
    bind_t(e, b, 6, M->sh_gate.w); bind_t(e, b, 7, M->sh_gate.s); bind_t(e, b, 8, M->sh_gate.b);
    bind_t(e, b, 9, M->sh_up.w); bind_t(e, b, 10, M->sh_up.s); bind_t(e, b, 11, M->sh_up.b);
    qw_batch_buf(b, 12, &e->x, 0);
    qw_batch_buf(b, 13, &e->rlog, 0);
    qw_batch_buf(b, 14, &e->H, 0);
    qw_batch_dispatch(b, QW_MOE_INTER / (8 * (gu2 ? 2 : 1)), QW_TOPK + 1, 1, 256, 1, 1);
    qw_batch_kernel(b, dn1 ? e->moe_down_f1 : e->moe_down_f);
    bind_t(e, b, 0, M->down.w); bind_t(e, b, 1, M->down.s); bind_t(e, b, 2, M->down.b);
    bind_t(e, b, 3, M->sh_down.w); bind_t(e, b, 4, M->sh_down.s); bind_t(e, b, 5, M->sh_down.b);
    qw_batch_buf(b, 6, &e->H, 0);
    qw_batch_buf(b, 7, &e->rlog, 0);
    qw_batch_buf(b, 9, &e->out, 0);
    qw_batch_dispatch(b, D / (8 * (dn1 ? 1 : 2)), 1, 1, 256, 1, 1);
}

static void moe_old(qw_engine *e, qw_batch *b, const qw_moe *M) {
    qw_batch_kernel(b, e->moe_router);
    bind_t(e, b, 0, M->router.w);
    bind_t(e, b, 1, M->router.s);
    bind_t(e, b, 2, M->router.b);
    bind_t(e, b, 3, M->sh_gate_w);
    qw_batch_buf(b, 4, &e->x, 0);
    qw_batch_buf(b, 5, &e->rlog, 0);
    qw_batch_dispatch(b, (QW_EXPERTS + 1 + 7) / 8, 1, 1, 256, 1, 1);
    qw_batch_kernel(b, e->moe_select);
    qw_batch_buf(b, 0, &e->rlog, 0);
    qw_batch_buf(b, 1, &e->msel, 0);
    qw_batch_buf(b, 2, &e->mw, 0);
    qw_batch_dispatch(b, 1, 1, 1, 32, 1, 1);
    qw_batch_kernel(b, e->moe_gateup);
    bind_t(e, b, 0, M->gate.w); bind_t(e, b, 1, M->gate.s); bind_t(e, b, 2, M->gate.b);
    bind_t(e, b, 3, M->up.w); bind_t(e, b, 4, M->up.s); bind_t(e, b, 5, M->up.b);
    bind_t(e, b, 6, M->sh_gate.w); bind_t(e, b, 7, M->sh_gate.s); bind_t(e, b, 8, M->sh_gate.b);
    bind_t(e, b, 9, M->sh_up.w); bind_t(e, b, 10, M->sh_up.s); bind_t(e, b, 11, M->sh_up.b);
    qw_batch_buf(b, 12, &e->x, 0);
    qw_batch_buf(b, 13, &e->msel, 0);
    qw_batch_buf(b, 14, &e->H, 0);
    qw_batch_dispatch(b, QW_MOE_INTER / 8, QW_TOPK + 1, 1, 256, 1, 1);
    qw_batch_kernel(b, e->moe_down);
    bind_t(e, b, 0, M->down.w); bind_t(e, b, 1, M->down.s); bind_t(e, b, 2, M->down.b);
    bind_t(e, b, 3, M->sh_down.w); bind_t(e, b, 4, M->sh_down.s); bind_t(e, b, 5, M->sh_down.b);
    qw_batch_buf(b, 6, &e->H, 0);
    qw_batch_buf(b, 7, &e->msel, 0);
    qw_batch_buf(b, 8, &e->mw, 0);
    qw_batch_buf(b, 9, &e->out, 0);
    qw_batch_dispatch(b, D / (8 * 2), 1, 1, 256, 1, 1);
}

// 除錯用：QW_SKIP=moe,hyper,lin,attn,ple,lm,resid 跳過指定部分（輸出會錯，只用來拆時間）
static bool skip(const char *what) {
    static const char *s = (const char *)1;
    if (s == (const char *)1) s = getenv("QW_SKIP");
    return s && strstr(s, what);
}

// 編碼一步（目前位置 e->pos 的 token 已在 GPU 的 token 歷史 tok[pos + 2]）。argmax_out：在最後加上 GPU argmax，
// 把下一個 token 寫進 tok[pos + 3]（流水線生成用）
// QW_TRACE：把 h（HCD 個 float）與 out（D 個 half，原樣位元）抄到 trace[token][layer][point]，驗證與 decode 逐層比對用
enum { TR_REC = (QW_HC * QW_HIDDEN * 4 + QW_HIDDEN * 2) };
static void trace_copy(qw_engine *e, qw_batch *b, const qw_buf *h, size_t hoff, const qw_buf *out, size_t ooff, int tok, int l, int pt) {
    if (!e->trace.ptr && !alloc(e, &e->trace, (size_t)8 * QW_LAYERS * 2 * TR_REC)) return;
    const size_t base = (((size_t)tok * QW_LAYERS + l) * 2 + pt) * TR_REC;
    uint32_t n = HCD;
    qw_batch_kernel(b, e->k_copyf);
    qw_batch_buf(b, 0, h, hoff);
    qw_batch_buf(b, 1, &e->trace, base);
    qw_batch_bytes(b, 2, &n, 4);
    d1(b, n, 256);
    n = D / 2;
    qw_batch_kernel(b, e->k_copyf);
    qw_batch_buf(b, 0, out, ooff);
    qw_batch_buf(b, 1, &e->trace, base + (size_t)HCD * 4);
    qw_batch_bytes(b, 2, &n, 4);
    d1(b, n, 256);
}
static bool trace_on(void) { static int v = -1; if (v < 0) v = getenv("QW_TRACE") ? 1 : 0; return v; }

static void encode_step(qw_engine *e, qw_batch *b, bool argmax_out, bool dump) {
    uint32_t p = (uint32_t)e->pos + 2;
    if (e->m->embed_gq) gg_embed_rows(e, b, &e->tok, p, 1, QW_HC, HCD, &e->h);
    else {
    qw_batch_kernel(b, e->k_embed_g);
    bind_t(e, b, 0, e->m->embed_w);
    bind_t(e, b, 1, e->m->embed_s);
    bind_t(e, b, 2, e->m->embed_b);
    qw_batch_buf(b, 3, &e->tok, 0);
    qw_batch_bytes(b, 4, &p, 4);
    qw_batch_buf(b, 5, &e->h, 0);
    d1(b, D, 256);
    }
    int li = 0, ai = 0;
    for (int l = 0; l < QW_LAYERS; l++) {
        const qw_layer *L = &e->m->layers[l];
        if (L->has_ple && !skip("ple")) {
            if (e->pending) { resid(e, b); e->pending = false; } // PLE 要讀更新後的 h
            ng_gather(e, &L->ple, e->pos + 2, 1);
            e->ng_base = 0;
            ple(e, b, &L->ple);
        }
        if (!skip("hyper")) hyper_in(e, b, &L->attn_hc);
        if (L->full_attn) { if (!skip("attn")) attention(e, b, &L->att, ai); ai++; }
        else { if (!skip("lin")) deltanet(e, b, &L->lin, li); li++; }
        e->pending = true;
        if (trace_on()) trace_copy(e, b, &e->h, 0, &e->out, 0, e->trace_tok, l, 0);
        if (!skip("hyper")) hyper_in(e, b, &L->mlp_hc);
        if (!skip("moe")) moe(e, b, &L->moe);
        e->pending = true;
        if (trace_on()) trace_copy(e, b, &e->h, 0, &e->out, 0, e->trace_tok, l, 1);
        if (dump) {
            resid(e, b);
            e->pending = false;
            uint32_t n = HCD;
            qw_batch_kernel(b, e->k_copyf);
            qw_batch_buf(b, 0, &e->h, 0);
            qw_batch_buf(b, 1, &e->dump, (size_t)l * HCD * 4);
            qw_batch_bytes(b, 2, &n, 4);
            d1(b, n, 256);
        }
    }
    if (!skip("hyper")) hyper_in(e, b, &e->m->mixer);
    e->pending = false;
    if (!skip("lm")) qmv(e, b, &e->m->lm_head, &e->x, 0, &e->logits, true);
    if (argmax_out) {
        uint32_t n = QW_VOCAB, m = (QW_VOCAB + 1023) / 1024, q = p + 1;
        qw_batch_kernel(b, e->k_argmax1);
        qw_batch_buf(b, 0, &e->logits, 0);
        qw_batch_buf(b, 1, &e->amv, 0);
        qw_batch_buf(b, 2, &e->ami, 0);
        qw_batch_bytes(b, 3, &n, 4);
        qw_batch_dispatch(b, m, 1, 1, 1024, 1, 1);
        qw_batch_kernel(b, e->k_argmax2);
        qw_batch_buf(b, 0, &e->amv, 0);
        qw_batch_buf(b, 1, &e->ami, 0);
        qw_batch_bytes(b, 2, &m, 4);
        qw_batch_buf(b, 3, &e->tok, 0);
        qw_batch_bytes(b, 4, &q, 4);
        qw_batch_dispatch(b, 1, 1, 1, 1024, 1, 1);
    }
    e->pos++;
}

static bool vb_init(qw_engine *e, char *err, size_t errlen);
static void vb_encode(qw_engine *e, qw_batch *b, int p0);
// 除錯（QW_TRACE）：從目前位置一次驗證 k 個 token（同生成時的驗證），回傳 trace 緩衝區（[k][層][2][h + out]）
const void *qw_engine_trace_vb(qw_engine *e, const int32_t *toks, int k) {
    char err[256];
    if (!vb_init(e, err, sizeof err) || k < 2 || k > VBMAX) return NULL;
    const int p0 = e->pos;
    if (!kv_reserve(e, p0 + k + 2)) return NULL;
    int32_t *tk = e->tok.ptr;
    for (int i = 0; i < k; i++) { tk[p0 + 2 + i] = toks[i]; e->hist[p0 + 2 + i] = toks[i]; }
    e->vbn = k;
    qw_batch *b = qw_batch_begin(e->g, false);
    vb_encode(e, b, p0);
    qw_batch_end(b);
    return e->trace.ptr;
}
// 量驗證成本（實驗）：從目前位置用 toks 的前 k 個 token 驗證 reps 次（不提交），回傳最快一次的 GPU ms（機器上有別的程式偶爾搶 GPU）。
// k = 1 用一般 decode 一步（之後退回位置）。QW_VB_CONC=1：全部重疊、不放 barrier（結果不對，只看沒有 barrier 空檔時的下限）
double qw_engine_vb_bench(qw_engine *e, const int32_t *toks, int k, int reps) {
    char err[256];
    if (k > 1 && (!vb_init(e, err, sizeof err) || k > VBMAX)) return -1;
    const int p0 = e->pos;
    if (!kv_reserve(e, p0 + VBMAX + 4)) return -1;
    int32_t *tk = e->tok.ptr;
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t;
        if (k == 1) {
            tk[p0 + 2] = toks[0]; e->hist[p0 + 2] = toks[0];
            const bool pend = e->pending;
            qw_batch *b = qw_batch_begin(e->g, false);
            encode_step(e, b, false, false);
            t = qw_batch_end(b);
            e->pos = p0; e->pending = pend;
        } else {
            for (int i = 0; i < k; i++) { tk[p0 + 2 + i] = toks[i]; e->hist[p0 + 2 + i] = toks[i]; }
            e->vbn = k;
            qw_batch *b = qw_batch_begin(e->g, getenv("QW_VB_CONC") != NULL);
            vb_encode(e, b, p0);
            t = qw_batch_end(b);
        }
        if (t > 0 && t < best) best = t;
    }
    return best * 1e3;
}
const void *qw_engine_trace_buf(qw_engine *e, int tok) { e->trace_tok = tok; return e->trace.ptr; }

const float *qw_engine_step(qw_engine *e, int32_t token, float *dump_layers) {
    if (e->pos >= e->max_ctx || !kv_reserve(e, e->pos + 2)) return NULL;
    e->hist[e->pos + 2] = token;
    ((int32_t *)e->tok.ptr)[e->pos + 2] = token; // 前一步已完成，CPU 直接寫
    qw_batch *b = qw_batch_begin(e->g, false);
    encode_step(e, b, false, dump_layers != NULL);
    double t = qw_batch_end(b);
    dn_tap_flush(e);
    e->last_ms = t * 1e3;
    if (t < 0) return NULL;
    if (token == EOS_ID) e->last_eos = e->pos + 1;
    if (dump_layers) memcpy(dump_layers, e->dump.ptr, (size_t)QW_LAYERS * HCD * 4);
    return e->logits.ptr;
}

// MTP 猜一個 token：用主模型剛算完位置 pos 的「mixer 之前」殘差流（e->h）與下一個 token next，
// 在位置 pos 跑 MTP 層，回傳 MTP 預測的再下一個 token。必須緊接在主模型那一步之後呼叫。
static int32_t mtp_draft_from(qw_engine *e, int32_t next, int pos, bool chain, double *gpu_ms);
int32_t qw_engine_mtp_draft(qw_engine *e, int32_t next, int pos, double *gpu_ms) { return mtp_draft_from(e, next, pos, false, gpu_ms); }
// 連續猜（MTP 鏈）：輸入改用上一次 MTP 層自己的殘差流（e->mh），位置 pos 接在上一次之後。
// 寫進 MTP cache 位置 pos 的 K/V 來自猜測的殘差流；主模型之後處理到 pos 時再用真實殘差流呼叫 qw_engine_mtp_draft 蓋過去
int32_t qw_engine_mtp_chain(qw_engine *e, int32_t next, int pos, double *gpu_ms) { return mtp_draft_from(e, next, pos, true, gpu_ms); }
static int32_t mtp_draft_from(qw_engine *e, int32_t next, int pos, bool chain, double *gpu_ms) {
    if (!e->m->has_mtp) return -1;
    if (chain && !kv_reserve(e, pos + 2)) return -1;
    const qw_model *M = e->m;
    qw_batch *b = qw_batch_begin(e->g, false);
    uint32_t id = (uint32_t)next, n1 = D, n2 = HCD;
    // 1) e = fc_embedding(norm(embed(next)))
    qw_batch_kernel(b, e->k_embed);
    bind_t(e, b, 0, M->embed_w);
    bind_t(e, b, 1, M->embed_s);
    bind_t(e, b, 2, M->embed_b);
    qw_batch_bytes(b, 3, &id, 4);
    qw_batch_buf(b, 4, &e->memb, 0);
    d1(b, D, 256);
    qw_batch_kernel(b, e->k_norm1);
    qw_batch_buf(b, 0, &e->memb, 0);
    bind_t(e, b, 1, M->mtp_norm_emb);
    qw_batch_buf(b, 2, &e->mnh, 0);
    qw_batch_bytes(b, 3, &n1, 4);
    qw_batch_dispatch(b, 1, 1, 1, 1024, 1, 1);
    qmv(e, b, &M->mtp_fc_emb, &e->mnh, 0, &e->mfe, false);
    // 2) 每條流：fc_hidden(norm(h))（4 條流一起正規化）；chain 時 h 是上一次 MTP 層的輸出（下面第 3 步才覆寫 mh）
    qw_batch_kernel(b, e->k_norm1);
    qw_batch_buf(b, 0, chain ? &e->mh : &e->h, 0);
    bind_t(e, b, 1, M->mtp_norm_hidden);
    qw_batch_buf(b, 2, &e->mnh, 0);
    qw_batch_bytes(b, 3, &n2, 4);
    qw_batch_dispatch(b, 1, 1, 1, 1024, 1, 1);
    for (int s = 0; s < QW_HC; s++) qmv_y(e, b, &M->mtp_fc_hidden, &e->mnh, (size_t)s * D * 2, &e->mfh, (size_t)s * D * 2, false);
    // 3) MTP 殘差流 = 每條流 + e
    qw_batch_kernel(b, e->k_mtp_combine);
    qw_batch_buf(b, 0, &e->mfh, 0);
    qw_batch_buf(b, 1, &e->mfe, 0);
    qw_batch_buf(b, 2, &e->mh, 0);
    d1(b, HCD, 256);
    // 4) 一層 QSA decoder（用 MTP 自己的殘差流與第 NATT 份 attention cache）
    qw_buf hsave = e->h;
    e->h = e->mh;
    const int psave = e->pos;
    const bool pend = e->pending;
    e->pos = pos;
    e->pending = false;
    hyper_in(e, b, &M->mtp.attn_hc);
    attention(e, b, &M->mtp.att, NATT);
    e->pending = true;
    hyper_in(e, b, &M->mtp.mlp_hc);
    moe(e, b, &M->mtp.moe);
    e->pending = true;
    hyper_in(e, b, &M->mtp_mixer);
    e->pending = false;
    qmv(e, b, &M->lm_head, &e->x, 0, &e->mlog, true);
    e->mh = e->h;
    e->h = hsave;
    e->pos = psave;
    e->pending = pend;
    double t = qw_batch_end(b);
    if (gpu_ms) *gpu_ms = t * 1e3;
    const float *lg = e->mlog.ptr;
    int best = 0;
    for (int v = 1; v < QW_VOCAB; v++)
        if (lg[v] > lg[best]) best = v;
    return best;
}

// 流水線生成：最後一次 qw_engine_step 的 logits 的 argmax 是第一個生成的 token。
// 之後每一步在 GPU 上 argmax、寫進 token 歷史；CPU 在 GPU 跑第 t 步時就編碼第 t+1 步（最多 2 步在途）。
int qw_engine_generate(qw_engine *e, int n, int32_t *out, double *gpu_ms) {
    if (e->pos + n + 1 > e->max_ctx) n = e->max_ctx - e->pos - 1;
    if (!kv_reserve(e, e->pos + n + 2)) return 0;
    if (!e->ng_gpu) { // n-gram 列要 CPU 先抄：下一個 token 必須先回到 CPU，不能流水線
        double g = 0;
        int32_t t = argmax_cpu(e->logits.ptr);
        for (int i = 0; i < n; i++) {
            out[i] = t;
            const float *lg = qw_engine_step(e, t, NULL);
            if (!lg) return i;
            g += e->last_ms;
            t = argmax_cpu(lg);
        }
        if (gpu_ms) *gpu_ms = g;
        return n;
    }
    const int start = e->pos;
    // 第 0 個：對上一步的 logits 做 argmax，寫到 tok[pos + 2]
    qw_batch *first = qw_batch_begin(e->g, false);
    {
        uint32_t nn = QW_VOCAB, m = (QW_VOCAB + 1023) / 1024, q = (uint32_t)e->pos + 2;
        qw_batch_kernel(first, e->k_argmax1);
        qw_batch_buf(first, 0, &e->logits, 0);
        qw_batch_buf(first, 1, &e->amv, 0);
        qw_batch_buf(first, 2, &e->ami, 0);
        qw_batch_bytes(first, 3, &nn, 4);
        qw_batch_dispatch(first, m, 1, 1, 1024, 1, 1);
        qw_batch_kernel(first, e->k_argmax2);
        qw_batch_buf(first, 0, &e->amv, 0);
        qw_batch_buf(first, 1, &e->ami, 0);
        qw_batch_bytes(first, 2, &m, 4);
        qw_batch_buf(first, 3, &e->tok, 0);
        qw_batch_bytes(first, 4, &q, 4);
        qw_batch_dispatch(first, 1, 1, 1, 1024, 1, 1);
    }
    qw_batch_submit(first);
    qw_batch *inflight[2] = {first, NULL};
    double gt = 0;
    for (int i = 0; i < n; i++) {
        qw_batch *b = qw_batch_begin(e->g, false);
        encode_step(e, b, true, false);
        qw_batch_submit(b);
        // 等最舊的那一步，讓在途的最多 2 步
        if (inflight[0]) gt += qw_batch_wait(inflight[0]) * 1e3;
        inflight[0] = inflight[1];
        inflight[1] = b;
    }
    for (int k = 0; k < 2; k++)
        if (inflight[k]) gt += qw_batch_wait(inflight[k]) * 1e3;
    const int32_t *tk = e->tok.ptr;
    for (int i = 0; i < n; i++) out[i] = tk[start + 2 + i];
    if (gpu_ms) *gpu_ms = gt;
    return n;
}

// ---------------------------------------------------------------- 各段的穩態時間（QW_KBENCH）
// 用真實權重，同一段運算在同一個 command buffer 連續跑 reps 次再平均（GPU 維持正常時脈）
static double bytes_q(const qw_qlin *L) { return !L->w ? 0 : L->gq ? (double)L->w->nbytes : (double)L->w->nbytes + L->s->nbytes + L->b->nbytes; }

static void kbench_vb(qw_engine *e, int reps);
static bool vb_init(qw_engine *e, char *err, size_t errlen);
static void moe_test(qw_engine *e);
static void moe_bench(qw_engine *e, int reps);
void qw_engine_kbench(qw_engine *e, int reps) {
    if (getenv("QW_MOE_BENCH")) { moe_bench(e, 4 * reps); return; }
    if (getenv("QW_MOE_TEST")) { moe_test(e); return; }
    if (getenv("QW_KBENCH_VB")) { kbench_vb(e, reps); return; }
    const qw_layer *Ll = &e->m->layers[0], *La = &e->m->layers[3];
    struct { const char *name; int kind; double bytes; } parts[] = {
        {"hyper-connection（1 次）", 0, bytes_q(&Ll->attn_hc.mix_down) + bytes_q(&Ll->attn_hc.mix_up) + 4.0 * HCD * 2 + HCD * 2.0},
        {"DeltaNet 層核心", 1, bytes_q(&Ll->lin.in_qkv) + bytes_q(&Ll->lin.in_z) + bytes_q(&Ll->lin.in_a) + bytes_q(&Ll->lin.in_b) +
                                bytes_q(&Ll->lin.out) + 48.0 * 128 * 128 * 4 * 2},
        {"QSA 層核心（短 context）", 2, 0}, // 讀取量：12 層平均（下面算）
        {"MoE", 3, bytes_q(&Ll->moe.router) + 10.0 / 512 * (bytes_q(&Ll->moe.gate) + bytes_q(&Ll->moe.up) + bytes_q(&Ll->moe.down)) +
                       (Ll->moe.giq ? 10.0 / 512 * ((double)Ll->moe.giq->nbytes + Ll->moe.uiq->nbytes + Ll->moe.diq->nbytes) : 0) +
                       bytes_q(&Ll->moe.sh_gate) + bytes_q(&Ll->moe.sh_up) + bytes_q(&Ll->moe.sh_down)},
        {"lm_head", 4, bytes_q(&e->m->lm_head)},
    };
    for (int l = 0; l < NATT; l++) { // QSA 層 12 層輪流跑，讀取量用 12 層平均（各層格式不同）
        const qw_attn *A = &e->m->layers[l * 4 + 3].att;
        parts[2].bytes += (bytes_q(&A->q) + bytes_q(&A->k) + bytes_q(&A->v) + bytes_q(&A->idx_qk) + bytes_q(&A->o)) / NATT;
    }
    (void)La;
    printf("| 段 | 每次 | 讀取量 | 有效頻寬 | 次數 / token | 每 token 合計 |\n|---|---|---|---|---|---|\n");
    const int per_tok[] = {97, 36, 12, 48, 1};
    double total = 0;
    for (size_t p = 0; p < sizeof parts / sizeof parts[0]; p++) {
        double best = 1e30;
        for (int t = 0; t < 3; t++) {
            qw_batch *b = qw_batch_begin(e->g, false);
            for (int r = 0; r < reps; r++) {
                switch (parts[p].kind) {
                case 0: hyper_in(e, b, &e->m->layers[r % 48].attn_hc); break;
                case 1: deltanet(e, b, &e->m->layers[(r % 12) * 4].lin, (r % 12) * 3); break;
                case 2: attention(e, b, &e->m->layers[(r % 12) * 4 + 3].att, r % 12); break;
                case 3: moe(e, b, &e->m->layers[r % 48].moe); break;
                case 4: qmv(e, b, &e->m->lm_head, &e->x, 0, &e->logits, true); break;
                }
            }
            double tt = qw_batch_end(b);
            if (tt > 0 && tt < best) best = tt;
        }
        const double us = best / reps * 1e6;
        total += us * per_tok[p];
        printf("| %s | %.1f µs | %.1f MB | %.0f GB/s | %d | %.2f ms |\n", parts[p].name, us, parts[p].bytes / 1e6, parts[p].bytes / us / 1e3,
               per_tok[p], us * per_tok[p] / 1e3);
    }
    printf("\n各段合計：%.2f ms / token\n", total / 1e3);
    { // QSA 層的投影（q / k / v / indexer 一起 + o），12 層輪流；讀取量是 12 層平均
        double bytes = 0, best = 1e30;
        for (int l = 0; l < NATT; l++) {
            const qw_attn *A = &e->m->layers[l * 4 + 3].att;
            bytes += (bytes_q(&A->q) + bytes_q(&A->k) + bytes_q(&A->v) + bytes_q(&A->idx_qk) + bytes_q(&A->o)) / NATT;
        }
        for (int t = 0; t < 3; t++) {
            qw_batch *b = qw_batch_begin(e->g, false);
            for (int r = 0; r < reps; r++) {
                const qw_attn *A = &e->m->layers[(r % NATT) * 4 + 3].att;
                const qw_qlin *Ls[4] = {&A->q, &A->k, &A->v, &A->idx_qk}, *Lo = &A->o;
                const qw_buf *Ys[4] = {&e->qg, &e->kk, &e->vv, &e->iqk}, *Yo = &e->out;
                if (!qmv_ggm(e, b, Ls, Ys, 4, &e->x, 1)) qmv_multi(e, b, Ls, Ys, 4, &e->x);
                if (!qmv_ggm(e, b, &Lo, &Yo, 1, &e->ay, 1)) qmv(e, b, Lo, &e->ay, 0, &e->out, false);
            }
            const double tt = qw_batch_end(b);
            if (tt > 0 && tt < best) best = tt;
        }
        const double us = best / reps * 1e6;
        printf("QSA 投影（12 層平均）：%.1f µs，%.1f MB，%.0f GB/s\n", us, bytes / 1e6, bytes / us / 1e3);
    }

    if (getenv("QW_DN_KB1") && dn_fuse(e, &Ll->lin)) { // GGUF DeltaNet 合併版的各 dispatch（12 層輪流；NB = QW_KBENCH_NB，> 1 是延後寫回的驗證，含重放 NB 步）
        const int nb = getenv("QW_KBENCH_NB") ? atoi(getenv("QW_KBENCH_NB")) : 1;
        char err[256];
        if (nb > 1 && !vb_init(e, err, sizeof err)) return;
        if (nb > 1) e->vbn = nb;
        const bool v = nb > 1;
        const qw_buf *x = v ? &e->bx : &e->x, *cv = v ? &e->bqkv : &e->qkv, *z = v ? &e->bz : &e->z, *be = v ? &e->bbe : &e->be,
                     *o = v ? &e->bgo : &e->go, *gy = v ? &e->bgy : &e->gy, *out = v ? &e->bout : &e->out;
        static const char *nm[] = {"dn_inf（投影 + conv + 衰減）", "gdn_fstep（遞迴 + normgate）", "dn_out", "全部", "dn_inf + dn_out", "dn_inf + gdn_fstep"};
        printf("\n| DeltaNet 合併版，NB = %d | 時間 |\n|---|---|\n", nb);
        for (int k = 0; k < 6; k++) {
            double best = 1e30;
            for (int t = 0; t < 3; t++) {
                qw_batch *b = qw_batch_begin(e->g, false);
                for (int r = 0; r < reps; r++) {
                    const qw_deltanet *N = &e->m->layers[(r % 12) * 4].lin;
                    const int li = (r % 12) * 3, ws = v ? e->dn_lp[li] ^ 1 : 0;
                    const qw_buf *la = v ? &e->dnG : &e->la;
                    const size_t lo = v ? dn_goff(li, ws) : 0;
                    if (k == 0 || k >= 3) dn_inf(e, b, N, li, x, cv, z, la, lo, be, nb, e->convx);
                    if (k == 1 || k == 3 || k == 5) {
                        if (v && !getenv("QW_DN_KB1_NOREPLAY")) { e->dn_lJ[li] = (uint8_t)nb; e->dn_pmask |= 1ull << li; } // 重放算進來（同一般的驗證輪）
                        gdn_fstep(e, b, N, li, nb, v ? 1 : 0, cv, la, lo, be, o, z, gy, ws);
                        if (v) e->dn_lp[li] = (uint8_t)ws;
                    }
                    if (k == 2 || k == 3 || k == 4) dn_out(e, b, N, gy, out, nb);
                }
                const double tt = qw_batch_end(b);
                if (tt > 0 && tt < best) best = tt;
            }
            printf("| %s | %.1f µs |\n", nm[k], best / reps * 1e6);
        }
        e->dn_pmask = 0;
        { // 同樣的權重位元組，純讀取的上限（threadgroup 數 tpb = 每個 threadgroup 讀多少 KB，QW_DN_RDKB，預設 32）
            const int kb = getenv("QW_DN_RDKB") ? atoi(getenv("QW_DN_RDKB")) : 32;
            double tb = 0, best[2] = {1e30, 1e30}, by[2] = {0, 0};
            for (int l = 0; l < 12; l++) {
                const qw_deltanet *N = &e->m->layers[l * 4].lin;
                by[0] += (bytes_q(&N->in_qkv) + bytes_q(&N->in_z) + bytes_q(&N->in_a) + bytes_q(&N->in_b)) / 12;
                by[1] += bytes_q(&N->out) / 12;
            }
            for (int k = 0; k < 2; k++)
                for (int t = 0; t < 3; t++) {
                    qw_batch *b = qw_batch_begin(e->g, false);
                    for (int r = 0; r < reps; r++) {
                        const qw_deltanet *N = &e->m->layers[(r % 12) * 4].lin;
                        const qw_qlin *L[4] = {&N->in_qkv, &N->in_z, &N->in_a, &N->in_b};
                        struct { uint32_t n[4], tb[5]; } a = {{0}, {0}};
                        for (int i = 0; i < 4; i++) {
                            const qw_qlin *Li = k ? &N->out : L[i];
                            a.n[i] = k && i ? 0 : (uint32_t)(bytes_q(Li) / 16);
                            a.tb[i + 1] = a.tb[i] + (a.n[i] ? (uint32_t)((a.n[i] * 16 + kb * 1024 - 1) / (kb * 1024)) : 0);
                            bind_t(e, b, i, Li->w);
                        }
                        qw_batch_kernel(b, dn_kernel(e, "dn_rd", 1));
                        qw_batch_buf(b, 4, &e->gy, 0);
                        qw_batch_bytes(b, 5, &a, sizeof a);
                        qw_batch_dispatch(b, a.tb[4], 1, 1, 256, 1, 1);
                    }
                    const double tt = qw_batch_end(b);
                    if (tt > 0 && tt < best[k]) best[k] = tt;
                }
            (void)tb;
            for (int k = 0; k < 2; k++)
                printf("| 純讀取上限：%s（%.1f MB） | %.1f µs（%.0f GB/s） |\n", k ? "out" : "qkv + z + a + b", by[k] / 1e6, best[k] / reps * 1e6,
                       by[k] / (best[k] / reps) / 1e9);
        }
        return;
    }
    if (Ll->attn_hc.mix_down.gq) { // GGUF：ghc_down、ghc_up 各自的時間（NB = QW_KBENCH_NB，預設 1）
        const int nb = getenv("QW_KBENCH_NB") ? atoi(getenv("QW_KBENCH_NB")) : 1;
        char err[256];
        if (nb > 1 && !vb_init(e, err, sizeof err)) return;
        printf("\n### hyper-connection 內部（GGUF，NB = %d，每次）\n\n| kernel | 時間 | 讀取量 | 有效頻寬 |\n|---|---|---|---|\n", nb);
        const qw_hyper *H0 = &Ll->attn_hc;
        const double by[2] = {bytes_q(&H0->mix_down) + (double)H0->inject->nbytes, bytes_q(&H0->mix_up)};
        const int r = gh_cfg(nb, 0), ips = gh_cfg(nb, 2);
        const qw_buf *hb = nb > 1 ? &e->bh : &e->h, *ob = nb > 1 ? &e->bout : &e->out, *ib = nb > 1 ? &e->binj : &e->inj,
                     *ib2 = nb > 1 ? &e->binj2 : &e->inj2, *pdb = nb > 1 ? &e->bnf : &e->nf, *xb = nb > 1 ? &e->bx : &e->x;
        for (int k = 0; k < 3; k++) {
            double best = 1e30;
            for (int t = 0; t < 3; t++) {
                qw_batch *b = qw_batch_begin(e->g, false);
                for (int rr = 0; rr < reps; rr++) {
                    const qw_hyper *H = &e->m->layers[rr % 48].attn_hc;
                    const uint32_t fl[2] = {0, 1};
                    if (k == 2) { hyper_gg(e, b, H, nb, 0, hb, ob, ib, ib2, pdb, xb); continue; }
                    qw_batch_kernel(b, gh_kernel(e, k ? "ghc_up" : "ghc_down", nb));
                    qw_batch_buf(b, 0, hb, 0); qw_batch_buf(b, 1, ob, 0); qw_batch_buf(b, 2, ib, 0); bind_t(e, b, 3, H->hc_norm);
                    if (k == 0) {
                        bind_t(e, b, 4, H->mix_down.w); bind_t(e, b, 5, H->inject); qw_batch_buf(b, 6, pdb, 0); qw_batch_bytes(b, 7, fl, 8);
                        qw_batch_dispatch(b, QW_HC_LOWRANK / r + 1, QW_HC, 1, D / 8, 1, 1);
                    } else {
                        bind_t(e, b, 4, H->mix_up.w); qw_batch_buf(b, 5, pdb, 0); qw_batch_buf(b, 6, xb, 0); qw_batch_buf(b, 7, ib2, 0);
                        qw_batch_bytes(b, 8, fl, 8);
                        qw_batch_dispatch(b, D / ips, 1, 1, 32 * ips, 1, 1);
                    }
                }
                const double tt = qw_batch_end(b);
                if (tt > 0 && tt < best) best = tt;
            }
            const double us = best / reps * 1e6, bb = k < 2 ? by[k] : by[0] + by[1];
            printf("| %s | %.1f µs | %.1f MB | %.0f GB/s |\n", k == 0 ? "ghc_down（殘差 + 平方和 + mix_down + inject）" : k == 1 ? "ghc_up（SiLU + mix_up + mix + 寫回 h）" : "合計（兩個連著跑）",
                   us, bb / 1e6, bb / us / 1e3);
        }
        { // 同樣的權重位元組，純讀取的上限（dn_rd：每個 threadgroup 讀 QW_DN_RDKB KB，預設 8）
            const int kb = getenv("QW_DN_RDKB") ? atoi(getenv("QW_DN_RDKB")) : 8;
            for (int k = 0; k < 2; k++) {
                double best = 1e30;
                for (int t = 0; t < 3; t++) {
                    qw_batch *b = qw_batch_begin(e->g, false);
                    for (int rr = 0; rr < reps; rr++) {
                        const qw_hyper *H = &e->m->layers[rr % 48].attn_hc;
                        const qw_t W[2] = {k ? H->mix_up.w : H->mix_down.w, k ? NULL : H->inject};
                        struct { uint32_t n[4], tb[5]; } a = {{0}, {0}};
                        for (int i = 0; i < 4; i++) {
                            a.n[i] = i < 2 && W[i] ? (uint32_t)(W[i]->nbytes / 16) : 0;
                            a.tb[i + 1] = a.tb[i] + (a.n[i] ? (uint32_t)((a.n[i] * 16 + kb * 1024 - 1) / (kb * 1024)) : 0);
                            bind_t(e, b, i, W[i < 2 && W[i] ? i : 0]);
                        }
                        qw_batch_kernel(b, dn_kernel(e, "dn_rd", 1));
                        qw_batch_buf(b, 4, &e->gy, 0);
                        qw_batch_bytes(b, 5, &a, sizeof a);
                        qw_batch_dispatch(b, a.tb[4], 1, 1, 256, 1, 1);
                    }
                    const double tt = qw_batch_end(b);
                    if (tt > 0 && tt < best) best = tt;
                }
                printf("| 純讀取上限：%s | %.1f µs | %.1f MB | %.0f GB/s |\n", k ? "mix_up" : "mix_down + inject", best / reps * 1e6, by[k] / 1e6,
                       by[k] / (best / reps) / 1e9);
            }
        }
        return;
    }

    // hyper-connection 內部
    printf("\n### hyper-connection 內部（每次）\n\n| kernel | 時間 |\n|---|---|\n");
    const char *names[3] = {"k_hc_in（殘差 + norm + inject 部分和）", "mix_down（split-K，10240→320）", "k_mixup（SiLU + 320→10240 + mix）"};
    for (int k = 0; k < 3; k++) {
        double best = 1e30;
        for (int t = 0; t < 3; t++) {
            qw_batch *b = qw_batch_begin(e->g, false);
            for (int r = 0; r < reps; r++) {
                const qw_hyper *H = &e->m->layers[r % 48].attn_hc;
                if (k == 0) {
                    uint32_t apply = 1, want = 1;
                    qw_batch_kernel(b, e->k_hc_in);
                    qw_batch_buf(b, 0, &e->h, 0); qw_batch_buf(b, 1, &e->out, 0); qw_batch_buf(b, 2, &e->inj, 0);
                    bind_t(e, b, 3, H->hc_norm); qw_batch_buf(b, 4, &e->nh, 0); qw_batch_buf(b, 5, &e->nf, 0);
                    qw_batch_bytes(b, 6, &apply, 4); bind_t(e, b, 7, H->inject); qw_batch_buf(b, 8, &e->inj2, 0);
                    qw_batch_bytes(b, 9, &want, 4);
                    qw_batch_dispatch(b, QW_HC, 1, 1, 1024, 1, 1);
                } else if (k == 1) qmv(e, b, &H->mix_down, &e->nh, 0, &e->d, false);
                else {
                    qw_batch_kernel(b, e->k_mixup);
                    bind_t(e, b, 0, H->mix_up.w); bind_t(e, b, 1, H->mix_up.s); bind_t(e, b, 2, H->mix_up.b);
                    qw_batch_buf(b, 3, &e->d, 0); qw_batch_buf(b, 4, &e->nf, 0); qw_batch_buf(b, 5, &e->x, 0);
                    qw_batch_dispatch(b, D / 8, 1, 1, 256, 1, 1);
                }
            }
            double tt = qw_batch_end(b);
            if (tt > 0 && tt < best) best = tt;
        }
        printf("| %s | %.1f µs |\n", names[k], best / reps * 1e6);
    }
}

// ================================================================ Prefill（一次多個 token）

static qw_kernel *pf_need(qw_engine *e, const char *name) {
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) { fprintf(stderr, "找不到 prefill kernel：%s\n", name); exit(1); }
    return k;
}

// qmm 變體：qmm_b{bits}_k{K}_{TM}x{TN}x{SG}_m{moe}{_f}
static qw_kernel *qmm_kernel(qw_engine *e, int bits, int K, int tm, int tn, int sg, bool moe, bool fout) {
    char name[96], defs[512], err[8192];
    snprintf(name, sizeof name, "qmm_b%d_k%d_%dx%dx%d_m%d%s", bits, K, tm, tn, sg, moe, fout ? "_f" : "");
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs, "#define qmm %s\n#define BITS %d\n#define KD %d\n#define TM %d\n#define TN %d\n#define SG %d\n#define KC 64\n#define MOE %d\n%s",
             name, bits, K, tm, tn, sg, moe ? 1 : 0, fout ? "#define YT float\n" : "");
    if (!compile(e, defs, src_qmm, sizeof src_qmm, err, sizeof err)) {
        fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
        exit(1);
    }
    return qw_kernel_get(e->g, name);
}

typedef struct { uint32_t M, N; } qmm_args;

// Y[M, N] = X[M, K] · Wᵀ（dense）
// 每列對稱 8-bit 的 dense 權重：輸入量化成 8-bit，uint8 × uint8 → int32 的 tensor 運算（kernels/qmm_i8.metal）
static int i8q_nt(int K) { return K / 4 >= 256 ? 256 : (K / 4 + 31) / 32 * 32; } // i8_quant 每列的執行緒數
static void pf_i8_kernels(qw_engine *e, const qw_qlin *L, uint32_t M, bool fout, qw_kernel **kq, qw_kernel **km) {
    // KC = 0：整個 K 一次 op.run（比每 64 一次快約 7%）；kn：權重 [K][N]，右運算元不轉置（再快約 12%）。整數累加是精確的，結果逐位元不變
    const int K = L->in, tm = M >= 128 ? 128 : 64, tn = 64, sg = 4, wt = L->kn;
    int kc = getenv("QW_I8_KC") ? atoi(getenv("QW_I8_KC")) : 0;
    if (!e->pi8x.ptr && !(alloc(e, &e->pi8x, ((size_t)PF + 128) * 10240) && alloc(e, &e->pi8s, ((size_t)PF + 128) * 4) &&
                          alloc(e, &e->pi8a, ((size_t)PF + 128) * 4))) {
        fprintf(stderr, "INT8 prefill buffer 配置失敗\n");
        exit(1);
    }
    if (kc && K % kc) kc = 0;
    char qn[64], mn[96], defs[512], err[8192];
    snprintf(qn, sizeof qn, "i8_quant_k%d", K);
    snprintf(mn, sizeof mn, "qmm_i8_k%d_%dx%dx%d_kc%d%s%s", K, tm, tn, sg, kc, wt ? "_t" : "", fout ? "_f" : "");
    *kq = qw_kernel_get(e->g, qn), *km = qw_kernel_get(e->g, mn);
    if (!*km) {
        snprintf(defs, sizeof defs, "#define KD %d\n#define TM %d\n#define TN %d\n#define SG %d\n#define KC %d\n#define WT %d\n#define YT %s\n"
                 "#define I8Q_NT %d\n#define i8_quant %s\n#define qmm_i8 %s\n", K, tm, tn, sg, kc, wt, fout ? "float" : "half", i8q_nt(K),
                 *kq ? "i8_quant_unused" : qn, mn);
        if (!compile(e, defs, src_qmm_i8, sizeof src_qmm_i8, err, sizeof err)) {
            fprintf(stderr, "編譯 %s 失敗：%s\n", mn, err);
            exit(1);
        }
        *kq = qw_kernel_get(e->g, qn);
        *km = qw_kernel_get(e->g, mn);
    }
}

// 同一個輸入給好幾個 INT8 投影（DeltaNet 的 qkv / z、attention 的 q / k / v / idx_qk、shared expert 的 gate / up）：
// pf_i8_share(e, X) 之後到 pf_i8_share(e, NULL) 之前，X 不會被改寫，pi8x / pi8s / pi8a 只量化一次。
// 量化結果與每次重算逐位元相同（同一個 kernel、同樣的輸入）；batch 是 serial dispatch，量化一定在讀它的 qmm 之前跑完
static void pf_i8_share(qw_engine *e, const qw_buf *X) {
    e->i8q_src = X;
    e->i8q_m = 0;
    e->i8q_k = 0;
}

// X [M][K] → pi8x（uint8，Qa + 128）、pi8s（scale）、pi8a（ΣQa）；共用中的輸入已量化過就跳過
static void pf_i8_quant(qw_engine *e, qw_batch *b, qw_kernel *kq, const qw_buf *X, uint32_t M, int K) {
    if (e->i8q_src == X && e->i8q_m == M && e->i8q_k == K) return;
    qw_batch_kernel(b, kq);
    qw_batch_buf(b, 0, X, 0);
    qw_batch_buf(b, 1, &e->pi8x, 0);
    qw_batch_buf(b, 2, &e->pi8s, 0);
    qw_batch_buf(b, 3, &e->pi8a, 0);
    qw_batch_dispatch(b, M, 1, 1, (unsigned)i8q_nt(K), 1, 1);
    if (e->i8q_src == X) { e->i8q_m = M; e->i8q_k = K; }
    else e->i8q_m = 0, e->i8q_k = 0; // 別的輸入覆寫了 pi8x
}

// pf_qmm 會不會把 L 交給 pf_qmm_i8（順序同 pf_qmm）：是的話傳回實際用的 INT8 權重，否則 NULL
static const qw_qlin *pf_i8_of(const qw_qlin *L) {
    if (L->i8) return L->i8;
    if (L->gq) return NULL;
    return L->chan && L->out % 64 == 0 && L->in % 64 == 0 && !getenv("QW_NO_I8") ? L : NULL;
}

static void pf_qmm_i8(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *X, const qw_buf *Y, uint32_t M, bool fout) {
    const int tm = M >= 128 ? 128 : 64, tn = 64, sg = 4;
    qw_kernel *kq, *km;
    pf_i8_kernels(e, L, M, fout, &kq, &km);
    pf_i8_quant(e, b, kq, X, M, L->in);
    const uint32_t a[2] = {M, (uint32_t)L->out};
    qw_batch_kernel(b, km);
    qw_batch_buf(b, 0, &e->pi8x, 0);
    bind_t(e, b, 1, L->w);
    bind_t(e, b, 2, L->s);
    qw_batch_buf(b, 4, Y, 0);
    qw_batch_bytes(b, 5, a, sizeof a);
    qw_batch_buf(b, 6, &e->pi8s, 0);
    qw_batch_buf(b, 7, &e->pi8a, 0);
    bind_t(e, b, 8, (const st_tensor *)L->rsum_t);
    qw_batch_dispatch(b, (unsigned)L->out / tn, (M + tm - 1) / tm, 1, 32u * sg, 1, 1);
}

static void pf_qmm_gg(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *X, const qw_buf *Y, uint32_t M, bool fout);
static void pf_qmm(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *X, const qw_buf *Y, uint32_t M, bool fout) {
    if (L->i8) { pf_qmm_i8(e, b, L->i8, X, Y, M, fout); return; } // GGUF 的 INT8 影子（QW_INT8）
    if (L->gq) { pf_qmm_gg(e, b, L, X, Y, M, fout); return; }
    if (L->chan && L->out % 64 == 0 && L->in % 64 == 0 && !getenv("QW_NO_I8")) { pf_qmm_i8(e, b, L, X, Y, M, fout); return; }
    // 輸出很窄的矩陣（例如 hyper-connection 的 mix_down，320 列）用窄 tile，否則 threadgroup 太少
    const int narrow = getenv("QW_PF_NARROW") ? atoi(getenv("QW_PF_NARROW")) : 1024;
    const int wide = getenv("QW_PF_TN") ? atoi(getenv("QW_PF_TN")) : 64;
    const int tn = (L->out % wide == 0 && L->out > narrow) ? wide : (L->out % 64 == 0 && L->out > narrow) ? 64 : 16;
    const int sg = tn >= 128 ? 8 : tn == 64 ? 4 : 2;
    const int tm = M >= 128 ? (getenv("QW_PF_TM") ? atoi(getenv("QW_PF_TM")) : 128) : 64; // 128 列一個 tile：反量化的成本分攤到更多 token
    qw_kernel *k = qmm_kernel(e, L->bits, L->in, tm, tn, sg, false, fout);
    qmm_args a = {M, (uint32_t)L->out};
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, X, 0);
    bind_t(e, b, 1, L->w);
    bind_t(e, b, 2, L->s);
    bind_t(e, b, 3, L->b);
    qw_batch_buf(b, 4, Y, 0);
    qw_batch_bytes(b, 5, &a, sizeof a);
    qw_batch_dispatch(b, (unsigned)L->out / tn, (M + tm - 1) / tm, 1, 32u * sg, 1, 1);
}

// GGUF profile 的 dense prefill：kernels/qmm.metal 的 IQF 分支（權重區塊解碼進 threadgroup memory），MOE 0
static void pf_qmm_gg(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *X, const qw_buf *Y, uint32_t M, bool fout) {
    char name[96], defs[1024], err[8192];
    // 窄 tile（16 欄）只給真的很窄的矩陣（DeltaNet 的 a / b 等）；N ≥ 512（shared expert 640、router 512）用 64 欄，否則 threadgroup 內的 tensor 運算太小
    const int wide = getenv("QW_PFGG_WIDE") ? atoi(getenv("QW_PFGG_WIDE")) : 512;
    const int f = L->gq - 1, tn = (L->out % 64 == 0 && L->out >= wide) ? 64 : 16, sg = tn == 64 ? 4 : 2, tm = M >= 128 ? 128 : 64;
    snprintf(name, sizeof name, "qmm_gg%d_k%d_%dx%dx%d%s", f, L->in, tm, tn, sg, fout ? "_f" : "");
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) {
        snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY\n#define qmm %s\n#define BITS 8\n#define KD %d\n#define TM %d\n#define TN %d\n#define SG %d\n"
                 "#define KC 64\n#define MOE 0\n#define IQF %d\n#define IQRB %d\n%s", name, L->in, tm, tn, sg, f, gg_rb(L), fout ? "#define YT float\n" : "");
        if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_qmm, sizeof src_qmm, err, sizeof err)) {
            fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
            exit(1);
        }
        k = qw_kernel_get(e->g, name);
    }
    qmm_args a = {M, (uint32_t)L->out};
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, X, 0);
    bind_t(e, b, 1, L->w);
    bind_t(e, b, 2, L->w); // S、Z 用不到
    bind_t(e, b, 3, L->w);
    qw_batch_buf(b, 4, Y, 0);
    qw_batch_bytes(b, 5, &a, sizeof a);
    qw_batch_dispatch(b, (unsigned)L->out / tn, (M + tm - 1) / tm, 1, 32u * sg, 1, 1);
}

// GGUF attention 的 dense prefill（kernels/qmv_ggm.metal 的 GGM_PF）：超區塊用 deq16 解碼進 threadgroup memory，
// 128 × 32 的 tile（量過 64×32、128×64、8 個 SIMD group 都較慢）。格式或形狀不支援時改走 pf_qmm（通用版）
enum { GGM_TM = 128, GGM_TN = 32, GGM_SG = 4 };
static bool qmm_ggm_ok(const qw_qlin *L) {
    return L->gq && ggm_fmt_ok(L->gq) && L->in % 256 == 0 && L->out % GGM_TN == 0 && L->ne == 1 && !getenv("QW_NO_GGM");
}
static qw_kernel *qmm_ggm_kernel(qw_engine *e, const qw_qlin *L) {
    enum { TM = GGM_TM, TN = GGM_TN, SG = GGM_SG };
    char name[96], defs[512], err[8192];
    snprintf(name, sizeof name, "qmm_ggm_f%d_k%d", L->gq - 1, L->in);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) {
        snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY 1\n#define GGM_PF 1\n#define qmm_ggm %s\n#define KD %d\n#define GF0 %d\n"
                 "#define TM %d\n#define TN %d\n#define SG %d\n", name, L->in, L->gq - 1, TM, TN, SG);
        if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_qmv_ggm, sizeof src_qmv_ggm, err, sizeof err)) {
            fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
            exit(1);
        }
        k = qw_kernel_get(e->g, name);
    }
    return k;
}
static bool dn_ggm_on(void) { static int v = -1; if (v < 0) v = getenv("QW_DN_GGM") ? atoi(getenv("QW_DN_GGM")) : 1; return v; }
static void pf_attn_mm(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *X, const qw_buf *Y, uint32_t M) {
    enum { TM = GGM_TM, TN = GGM_TN, SG = GGM_SG };
    if (L->i8) { pf_qmm_i8(e, b, L->i8, X, Y, M, false); return; } // GGUF 的 INT8 影子（QW_INT8）
    if (!qmm_ggm_ok(L)) {
        pf_qmm(e, b, L, X, Y, M, false);
        return;
    }
    qw_kernel *k = qmm_ggm_kernel(e, L);
    const uint32_t a[2] = {M, (uint32_t)L->out};
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, X, 0);
    bind_t(e, b, 1, L->w);
    qw_batch_buf(b, 4, Y, 0);
    qw_batch_bytes(b, 5, a, sizeof a);
    qw_batch_dispatch(b, (unsigned)L->out / TN, (M + TM - 1) / TM, 1, 32u * SG, 1, 1);
}

// GGUF attention 的 kernel 開啟時先編好（每層的格式組合：decode、驗證、prefill），不要在生成途中編譯
static void ggm_warm(qw_engine *e) {
    for (int l = 3; l < QW_LAYERS; l += 4) {
        const qw_attn *A = &e->m->layers[l].att;
        const qw_qlin *Ls[4] = {&A->q, &A->k, &A->v, &A->idx_qk}, *Lo = &A->o;
        if (qmv_ggm_ok(Ls, 4)) qmv_ggm_kernel(e, Ls, 4, 1);
        if (qmv_ggm_ok(&Lo, 1)) qmv_ggm_kernel(e, &Lo, 1, 1);
        for (int i = 0; i < 4; i++)
            if (qmm_ggm_ok(Ls[i])) qmm_ggm_kernel(e, Ls[i]);
        if (qmm_ggm_ok(Lo)) qmm_ggm_kernel(e, Lo);
    }
}

// MoE grouped：X 是依 expert 排好的 P 列，區塊表在 e->ptiles
static int moe_tm(void) {
    static int t = 0;
    // 每個 expert 平均約 PF × 8 / 512 = PF / 64 列：區塊太高一半是空的，太矮則重複解量化太多次
    if (!t) t = getenv("QW_MOE_TM") ? atoi(getenv("QW_MOE_TM")) : (PF / 64 < 32 ? 32 : PF / 64 > 128 ? 128 : PF / 64);
    return t;
}

// GGUF 的 routed expert（moe_pf_gg）的區塊列數：QW_PFG_TM（每次讀環境變數，量測時可在同一個行程切換）
static int moe_gg_tm(void) { return gg_par("QW_PFG_TM", moe_tm()); }

static bool a8_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("QW_A8") ? atoi(getenv("QW_A8")) : 0;
    return v;
}

// MoE gate / up 的 INT8 路徑：X（pXs，P 列）量化一次，gate、up 各一次 INT8 GEMM
static void pf_moe_gateup_a8(qw_engine *e, qw_batch *b, const qw_moe *M, uint32_t P) {
    const size_t rows = (size_t)PF * QW_TOPK + 64;
    if (!e->pa8q.ptr && !(alloc(e, &e->pa8q, rows * D) && alloc(e, &e->pa8s, rows * 4) && alloc(e, &e->pa8r, rows * (D / 64) * 2))) {
        fprintf(stderr, "A8 buffer 配置失敗\n");
        exit(1);
    }
    qw_kernel *kq = qw_kernel_get(e->g, "a8_quant_k2560"), *km = qw_kernel_get(e->g, "qmm_a8_k2560");
    if (!kq) {
        char err[8192];
        if (!compile(e, "#define KD 2560\n#define a8_quant a8_quant_k2560\n#define qmm_a8 qmm_a8_k2560\n#define u8_quant u8_quant_k2560\n#define qmm_u4 qmm_u4_k2560\n", src_qmm_a8, sizeof src_qmm_a8, err, sizeof err)) {
            fprintf(stderr, "編譯 qmm_a8 失敗：%s\n", err);
            exit(1);
        }
        kq = qw_kernel_get(e->g, "a8_quant_k2560");
        km = qw_kernel_get(e->g, "qmm_a8_k2560");
    }
    if (a8_on() == 2) { // uint8 × uint4b：權重直接當 4-bit 運算元
        kq = qw_kernel_get(e->g, "u8_quant_k2560");
        km = qw_kernel_get(e->g, "qmm_u4_k2560");
    }
    qw_batch_kernel(b, kq);
    qw_batch_buf(b, 0, &e->pXs, 0);
    qw_batch_buf(b, 1, &e->pa8q, 0);
    qw_batch_buf(b, 2, &e->pa8s, 0);
    qw_batch_buf(b, 3, &e->pa8r, 0);
    qw_batch_bytes(b, 4, &P, 4);
    qw_batch_dispatch(b, P, 1, 1, 256, 1, 1);
    const qw_qlin *L2[2] = {&M->gate, &M->up};
    const qw_buf *Y2[2] = {&e->pGg, &e->pGu};
    for (int k = 0; k < 2; k++) {
        const uint32_t mn[2] = {P, (uint32_t)L2[k]->out};
        qw_batch_kernel(b, km);
        qw_batch_buf(b, 0, &e->pa8q, 0);
        bind_t(e, b, 1, L2[k]->w);
        bind_t(e, b, 2, L2[k]->s);
        bind_t(e, b, 3, L2[k]->b);
        qw_batch_buf(b, 4, Y2[k], 0);
        qw_batch_bytes(b, 5, mn, sizeof mn);
        qw_batch_buf(b, 6, &e->ptiles, 0);
        qw_batch_buf(b, 7, &e->pa8s, 0);
        qw_batch_buf(b, 8, &e->pa8r, 0);
        qw_batch_dispatch(b, (unsigned)L2[k]->out / 64, e->ntiles, 1, 128, 1, 1);
    }
}

// GGUF routed expert 的 INT8 prefill（kernels/moe_gg.metal 的 GG_PART 6）：QW_INT8=1 或類別清單含 "moe"（例如 QW_INT8=moe、
// QW_INT8=lin,attn,shared,moe）；QW_INT8_MOE=0 / 1 可單獨蓋過。量測時用 QW_PFG_I8（每次讀環境變數，QW_MOE_KB 可在同一個行程切換）。
// 每 QW_I8_RK 個 K 一個 scale（預設 128；32 時除了 IQ2_S 以外權重不失真）。只支援 IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL / Q2_0
static int i8moe_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("QW_INT8_MOE"), *l = getenv("QW_INT8");
        v = s ? atoi(s) : l ? (!strcmp(l, "1") || strstr(l, "moe") != NULL) : 0;
    }
    return gg_par("QW_PFG_I8", v);
}
static bool i8moe_fmt(int f) { return f == 22 || f == 18 || f == 21 || f == 23 || f == 20 || f == 42; }
// MLX affine 4-bit 的 routed expert（moe_pf_i8m，kernel 的格式代碼 100）：權重每列每 QW_I8_RK 個 K 重新量化成對稱 INT8
enum { I8_MLX4 = 100 };
static bool i8moe_mlx(const qw_moe *M) {
    return !M->giq && M->gate.bits == 4 && M->up.bits == 4 && M->down.bits == 4 && M->gate.w && M->down.w;
}
static bool i8moe_use(const qw_moe *M) {
    return i8moe_on() && (i8moe_mlx(M) || (M->giq && !gg_old() && i8moe_fmt(M->fgu) && i8moe_fmt(M->fd)));
}
static int i8moe_fgu(const qw_moe *M) { return M->giq ? M->fgu : I8_MLX4; } // kernel 的格式代碼（scatter、SiLU 的量化與格式無關）
static int i8moe_fd(const qw_moe *M) { return M->giq ? M->fd : I8_MLX4; }

static qw_kernel *i8moe_kernel(qw_engine *e, int f, int K, const char *kind) {
    char name[128], defs[1024], err[8192];
    const int tm = (int)e->ptm, tn = gg_par("QW_PFG_TN", 64), sgn = gg_par("QW_PFG_SG", 4), kc = gg_par("QW_I8_KC", 128),
              rk = gg_par("QW_I8_RK", 128), fb = f == 22 ? 0 : f == 18 ? 1 : f == 21 ? 2 : 3,
              // QW_I8_GC（位元同 QW_PFG_T）：這個格式的 grid 表改從 constant 讀、不放 threadgroup memory（預設只有 IQ2_S，見 moe_gg.metal）
              gc = fb < 3 && (gg_par("QW_I8_GC", 1) >> fb) & 1 && strcmp(kind, "moe_pf_i8m") != 0,
              pft = gc ? 0 : (gg_par("QW_PFG_T", 7) >> fb) & 1, dbg = gg_par("QW_PFG_DBG", 0),
              // QW_I8_PAIR：每個執行緒一次解 2 或 4 個相鄰單位（moe_gg.metal 的 pf_dq8p；0 = 一次一個）
              p0 = strcmp(kind, "moe_pf_i8") ? 0 : gg_par("QW_I8_PAIR", 2),
              pair = (p0 == 2 || p0 == 4) && kc % (16 * p0) == 0 && rk % (16 * p0) == 0 && (tn * (kc / 16 / p0)) % 32 == 0 ? p0 : 0; // 不適用時一次一個
    if (K % kc || kc % rk || rk % 32 || tn % 32 || (tn * (kc / 16)) % (32 * sgn) || sgn * 32 > 1024 || tm % sgn || (tm / sgn) % 8 ||
        tn * kc + kc / rk * tn * 4 + (pft ? 8192 : 0) > 32768) {
        fprintf(stderr, "moe_pf_i8：不支援的 tile\n");
        exit(1);
    }
    snprintf(name, sizeof name, "%s_%d_k%d_tm%d_%dx%dx%d_r%d_t%d_d%d_c%d_p%d", kind, f, K, tm, tn, sgn, kc, rk, pft, dbg, gc, pair);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) {
        snprintf(defs, sizeof defs, "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                 "#define IQ_DQ_ONLY\n#define E %d\n#define TOPK %d\n#define D %d\n#define I %d\n#define NB 1\n#define GG_PART 6\n"
                 "#define PFF %d\n#define KD %d\n#define TM %d\n#define TN %d\n#define SG %d\n#define KC %d\n#define PFS 1\n#define PFT %d\n"
                 "#define PFDBG %d\n#define RK %d\n#define I8_GC %d\n#define I8_PAIR %d\n#define %s %s\n",
                 QW_EXPERTS, QW_TOPK, D, QW_MOE_INTER, f, K, tm, tn, sgn, kc, pft, dbg, rk, gc, pair, kind, name);
        if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_moe_gg, sizeof src_moe_gg, err, sizeof err)) {
            fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
            exit(1);
        }
        k = qw_kernel_get(e->g, name);
    }
    return k;
}

// INT8 輸入的 buffer（第一次用時配置）；P 超出就停
static void i8moe_bufs(qw_engine *e, uint32_t P) {
    const size_t rows = (size_t)PF * QW_TOPK + 64;
    if (!e->pq8x.ptr && !(alloc(e, &e->pq8x, rows * D) && alloc(e, &e->pq8s, rows * 4) && alloc(e, &e->pq8h, rows * QW_MOE_INTER) &&
                          alloc(e, &e->pq8hs, rows * 4))) {
        fprintf(stderr, "INT8 MoE buffer 配置失敗\n");
        exit(1);
    }
    if ((size_t)P > rows) { fprintf(stderr, "INT8 MoE：列數 %u 太大\n", P); exit(1); }
}

// X [P][K] half → Xq、sx（每列一個 scale）。prefill 用合併版（moe_scatter_i8、p_act2_i8），這個給量測與比較
static void pf_i8q(qw_engine *e, qw_batch *b, int f, int K, const qw_buf *X, const qw_buf *Xq, const qw_buf *sx, uint32_t P) {
    i8moe_bufs(e, P);
    qw_batch_kernel(b, i8moe_kernel(e, f, K, "pf_i8q"));
    qw_batch_buf(b, 0, X, 0);
    qw_batch_buf(b, 1, Xq, 0);
    qw_batch_buf(b, 2, sx, 0);
    qw_batch_bytes(b, 3, &P, 4);
    qw_batch_dispatch(b, P, 1, 1, 256, 1, 1);
}

// moe_scatter 的 INT8 版：px 依 expert 排好、直接量化進 pq8x / pq8s（不寫 pXs）
static void pf_scatter_i8(qw_engine *e, qw_batch *b, const qw_moe *M, uint32_t P) {
    i8moe_bufs(e, P);
    qw_batch_kernel(b, i8moe_kernel(e, i8moe_fgu(M), D, "moe_scatter_i8"));
    qw_batch_buf(b, 0, &e->px, 0);
    qw_batch_buf(b, 1, &e->pids, 0);
    qw_batch_buf(b, 2, &e->pcur, 0);
    qw_batch_buf(b, 3, &e->pq8x, 0);
    qw_batch_buf(b, 4, &e->ppos, 0);
    qw_batch_buf(b, 5, &e->pq8s, 0);
    qw_batch_dispatch(b, P, 1, 1, 256, 1, 1);
}

// shared expert 也走 dense INT8（K = D）時：px 先量化進 pi8x（shared gate / up 之後直接用），scatter 只搬整數列（moe_scatter_i8u）。
// 呼叫端要先 pf_i8_share(e, &e->px)；不適用時傳回 false（改用 pf_scatter_i8）
static bool pf_scatter_i8u(qw_engine *e, qw_batch *b, const qw_moe *M, uint32_t T, uint32_t P) {
    const qw_qlin *sh = pf_i8_of(&M->sh_gate);
    if (!sh || sh->in != D || e->i8q_src != &e->px || getenv("QW_I8_SCATTER0")) return false;
    qw_kernel *kq, *km;
    pf_i8_kernels(e, sh, T, false, &kq, &km);
    pf_i8_quant(e, b, kq, &e->px, T, D);
    i8moe_bufs(e, P);
    qw_batch_kernel(b, i8moe_kernel(e, i8moe_fgu(M), D, "moe_scatter_i8u"));
    qw_batch_buf(b, 0, &e->pi8x, 0);
    qw_batch_buf(b, 1, &e->pids, 0);
    qw_batch_buf(b, 2, &e->pcur, 0);
    qw_batch_buf(b, 3, &e->pq8x, 0);
    qw_batch_buf(b, 4, &e->ppos, 0);
    qw_batch_buf(b, 5, &e->pq8s, 0);
    qw_batch_buf(b, 6, &e->pi8s, 0);
    qw_batch_dispatch(b, P, 1, 1, 256, 1, 1);
    return true;
}

// p_act2 的 INT8 版：SiLU(pGg)·pGu 直接量化進 pq8h / pq8hs（不寫 pH）
static void pf_act_i8(qw_engine *e, qw_batch *b, const qw_moe *M, uint32_t P) {
    _Static_assert(QW_MOE_INTER % 128 == 0 && QW_MOE_INTER / 4 <= 1024, "p_act2_i8：一列 I / 4 個執行緒");
    i8moe_bufs(e, P);
    qw_batch_kernel(b, i8moe_kernel(e, i8moe_fd(M), QW_MOE_INTER, "p_act2_i8"));
    qw_batch_buf(b, 0, &e->pGg, 0);
    qw_batch_buf(b, 1, &e->pGu, 0);
    qw_batch_buf(b, 2, &e->pq8h, 0);
    qw_batch_buf(b, 3, &e->pq8hs, 0);
    qw_batch_dispatch(b, P, 1, 1, QW_MOE_INTER / 4, 1, 1);
}

// grouped qmm（INT8）：Y [P][N] = (Xq · W_eᵀ) · sx；Xq、sx 是量化好的輸入（K 個值一列）
static void pf_qmm_moe_i8(qw_engine *e, qw_batch *b, const st_tensor *W, int f, int N, int K, const qw_buf *Xq, const qw_buf *sx,
                          const qw_buf *Y, uint32_t P) {
    i8moe_bufs(e, P);
    qw_kernel *k = i8moe_kernel(e, f, K, "moe_pf_i8");
    if (N % gg_par("QW_PFG_TN", 64)) { fprintf(stderr, "moe_pf_i8：N\n"); exit(1); }
    if (!e->ggtab_ready) {
        qw_batch_kernel(b, gg_kernel(e, "gg_tab_init", NULL, 1));
        qw_batch_buf(b, 0, &e->ggtab, 0);
        qw_batch_dispatch(b, 32, 1, 1, 256, 1, 1);
        e->ggtab_ready = true;
    }
    const uint32_t a[2] = {P, (uint32_t)N};
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, Xq, 0);
    bind_t(e, b, 1, W);
    qw_batch_buf(b, 4, Y, 0);
    qw_batch_bytes(b, 5, a, sizeof a);
    qw_batch_buf(b, 6, &e->ptiles, 0);
    qw_batch_buf(b, 7, &e->ggtab, 0);
    qw_batch_buf(b, 8, sx, 0);
    qw_batch_dispatch(b, (unsigned)N / gg_par("QW_PFG_TN", 64), e->ntiles, 1, 32u * gg_par("QW_PFG_SG", 4), 1, 1);
}

// 同上，權重是 MLX affine 4-bit（L：[E][N][K / 8] U32 與 [E][N][K / 64] 的 scales / biases）
static void pf_qmm_moe_i8m(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *Xq, const qw_buf *sx, const qw_buf *Y, uint32_t P) {
    i8moe_bufs(e, P);
    if (L->bits != 4 || L->in % 128 || L->out % gg_par("QW_PFG_TN", 64)) { fprintf(stderr, "moe_pf_i8m：形狀\n"); exit(1); }
    qw_kernel *k = i8moe_kernel(e, I8_MLX4, L->in, "moe_pf_i8m");
    const uint32_t a[2] = {P, (uint32_t)L->out};
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, Xq, 0);
    bind_t(e, b, 1, L->w);
    bind_t(e, b, 2, L->s);
    bind_t(e, b, 3, L->b);
    qw_batch_buf(b, 4, Y, 0);
    qw_batch_bytes(b, 5, a, sizeof a);
    qw_batch_buf(b, 6, &e->ptiles, 0);
    qw_batch_buf(b, 8, sx, 0);
    qw_batch_dispatch(b, (unsigned)L->out / gg_par("QW_PFG_TN", 64), e->ntiles, 1, 32u * gg_par("QW_PFG_SG", 4), 1, 1);
}
// routed expert 的 INT8 grouped GEMM：gate + up（輸入 pq8x）或 down（輸入 pq8h），GGUF 與 MLX 各走自己的 kernel
static void pf_moe_i8_gu(qw_engine *e, qw_batch *b, const qw_moe *M, uint32_t P) {
    if (!M->giq) {
        pf_qmm_moe_i8m(e, b, &M->gate, &e->pq8x, &e->pq8s, &e->pGg, P);
        pf_qmm_moe_i8m(e, b, &M->up, &e->pq8x, &e->pq8s, &e->pGu, P);
        return;
    }
    pf_qmm_moe_i8(e, b, M->giq, M->fgu, QW_MOE_INTER, D, &e->pq8x, &e->pq8s, &e->pGg, P);
    pf_qmm_moe_i8(e, b, M->uiq, M->fgu, QW_MOE_INTER, D, &e->pq8x, &e->pq8s, &e->pGu, P);
}
static void pf_moe_i8_dn(qw_engine *e, qw_batch *b, const qw_moe *M, uint32_t P) {
    if (!M->giq) pf_qmm_moe_i8m(e, b, &M->down, &e->pq8h, &e->pq8hs, &e->pDm, P);
    else pf_qmm_moe_i8(e, b, M->diq, M->fd, D, QW_MOE_INTER, &e->pq8h, &e->pq8hs, &e->pDm, P);
}

// grouped qmm，權重是 GGUF 區塊（type f）：Y [P][N] = X [P][K] · W_eᵀ
static void pf_qmm_moe_iq(qw_engine *e, qw_batch *b, const st_tensor *W, int f, int N, int K, const qw_buf *X, const qw_buf *Y, uint32_t P) {
    char name[96], defs[1024], err[8192];
    if (!gg_old()) { // kernels/moe_gg.metal 的 moe_pf_gg（查表解碼）
        // 預設：條帶版（PFS 1，16 列一條）、IQ2_S / IQ3_XXS / IQ3_S 的 grid 表放 threadgroup memory（QW_PFG_T 的位元 0 / 1 / 2）
        const int tm = (int)e->ptm, tn = gg_par("QW_PFG_TN", 64), sgn = gg_par("QW_PFG_SG", 4), kc = gg_par("QW_PFG_KC", 64), pfs = gg_par("QW_PFG_S", 1),
                  pft = (gg_par("QW_PFG_T", 7) >> (f == 22 ? 0 : f == 18 ? 1 : f == 21 ? 2 : 3)) & 1, dbg = gg_par("QW_PFG_DBG", 0);
        if (N % tn || K % kc || kc % 32 || tn % 32 || sgn * 32 > 1024 || tn * kc * 2 + (pft ? 8192 : 0) > 32768 || (pfs && (tm % sgn || tm / sgn % 8))) {
            fprintf(stderr, "moe_pf_gg：不支援的 tile\n");
            exit(1);
        }
        snprintf(name, sizeof name, "moe_pf_gg_%d_k%d_tm%d_%dx%dx%d_s%d_t%d_d%d", f, K, tm, tn, sgn, kc, pfs, pft, dbg);
        qw_kernel *k = qw_kernel_get(e->g, name);
        if (!k) {
            snprintf(defs, sizeof defs, "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                     "#define IQ_DQ_ONLY\n#define E %d\n#define TOPK %d\n#define D %d\n#define I %d\n#define NB 1\n#define GG_PART 5\n"
                     "#define PFF %d\n#define KD %d\n#define TM %d\n#define TN %d\n#define SG %d\n#define KC %d\n#define PFS %d\n#define PFT %d\n#define PFDBG %d\n"
                     "#define moe_pf_gg %s\n",
                     QW_EXPERTS, QW_TOPK, D, QW_MOE_INTER, f, K, tm, tn, sgn, kc, pfs, pft, dbg, name);
            if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_moe_gg, sizeof src_moe_gg, err, sizeof err)) {
                fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
                exit(1);
            }
            k = qw_kernel_get(e->g, name);
        }
        if (!e->ggtab_ready) { // 查表（同 gg_moe）
            qw_batch_kernel(b, gg_kernel(e, "gg_tab_init", NULL, 1));
            qw_batch_buf(b, 0, &e->ggtab, 0);
            qw_batch_dispatch(b, 32, 1, 1, 256, 1, 1); // 8192 個執行緒（kernels/moe_gg.metal 的 gg_tab_init）
            e->ggtab_ready = true;
        }
        const uint32_t a[2] = {P, (uint32_t)N};
        qw_batch_kernel(b, k);
        qw_batch_buf(b, 0, X, 0);
        bind_t(e, b, 1, W);
        qw_batch_buf(b, 4, Y, 0);
        qw_batch_bytes(b, 5, a, sizeof a);
        qw_batch_buf(b, 6, &e->ptiles, 0);
        qw_batch_buf(b, 7, &e->ggtab, 0);
        qw_batch_dispatch(b, (unsigned)N / tn, e->ntiles, 1, 32u * sgn, 1, 1);
        return;
    }
    const int tm = moe_tm(), rb = K / iq_qb(f) * iq_bb(f);
    snprintf(name, sizeof name, "qmm_iq%d_k%d_%dx64x4", f, K, tm);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) {
        snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY\n#define qmm %s\n#define BITS 4\n#define KD %d\n#define TM %d\n#define TN 64\n#define SG 4\n"
                 "#define KC 64\n#define MOE 1\n#define IQF %d\n#define IQRB %d\n", name, K, tm, f, rb);
        if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_qmm, sizeof src_qmm, err, sizeof err)) {
            fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
            exit(1);
        }
        k = qw_kernel_get(e->g, name);
    }
    qmm_args a = {P, (uint32_t)N};
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, X, 0);
    bind_t(e, b, 1, W);
    bind_t(e, b, 2, W); // S、Z 用不到
    bind_t(e, b, 3, W);
    qw_batch_buf(b, 4, Y, 0);
    qw_batch_bytes(b, 5, &a, sizeof a);
    qw_batch_buf(b, 6, &e->ptiles, 0);
    qw_batch_dispatch(b, (unsigned)N / 64, e->ntiles, 1, 128, 1, 1);
}

// MoE gate + up + SiLU 合併（kernels/qmm.metal 的 GU）：X 只讀一次、不寫 gate / up 的中間結果
static void pf_qmm_moe_gu(qw_engine *e, qw_batch *b, const qw_moe *M, uint32_t P) {
    const int tm = moe_tm(), K = M->gate.in;
    char name[96], defs[512], err[8192];
    snprintf(name, sizeof name, "qmm_gu_b4_k%d_%dx64x4", K, tm);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) {
        snprintf(defs, sizeof defs, "#define qmm %s\n#define BITS 4\n#define KD %d\n#define TM %d\n#define TN 64\n#define SG 4\n#define KC 64\n#define MOE 1\n#define GU 1\n",
                 name, K, tm);
        if (!compile(e, defs, src_qmm, sizeof src_qmm, err, sizeof err)) {
            fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
            exit(1);
        }
        k = qw_kernel_get(e->g, name);
    }
    qmm_args a = {P, (uint32_t)M->gate.out};
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, &e->pXs, 0);
    bind_t(e, b, 1, M->gate.w); bind_t(e, b, 2, M->gate.s); bind_t(e, b, 3, M->gate.b);
    qw_batch_buf(b, 4, &e->pH, 0);
    qw_batch_bytes(b, 5, &a, sizeof a);
    qw_batch_buf(b, 6, &e->ptiles, 0);
    bind_t(e, b, 7, M->up.w); bind_t(e, b, 8, M->up.s); bind_t(e, b, 9, M->up.b);
    qw_batch_dispatch(b, (unsigned)M->gate.out / 64, e->ntiles, 1, 128, 1, 1);
}

static void pf_qmm_moe(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *X, const qw_buf *Y, uint32_t P) {
    // 權重直接解碼進 SIMD group 的 tensor 運算元（kernels/qmm.metal 的 REGDQ）：輸出與一般版完全相同、快約 4%。
    // 只用驗證過的尺寸：每個 SIMD group 32 欄、K 一次 32；不符合就走一般版
    if (L->bits == 4 && L->out % 128 == 0 && L->in % 64 == 0 && !getenv("QW_NO_MOE_REG")) {
        char name[96], defs[512], err[8192];
        snprintf(name, sizeof name, "qmm_reg_k%d_tm%d", L->in, moe_tm());
        qw_kernel *k = qw_kernel_get(e->g, name);
        if (!k) {
            snprintf(defs, sizeof defs, "#define qmm qmm_unused_%s\n#define qmm_reg %s\n#define BITS 4\n#define KD %d\n#define TM %d\n#define TN 64\n#define SG 4\n#define KC 64\n#define MOE 1\n#define REGDQ 1\n",
                     name, name, L->in, moe_tm());
            if (!compile(e, defs, src_qmm, sizeof src_qmm, err, sizeof err)) { fprintf(stderr, "編譯 %s 失敗：%s\n", name, err); exit(1); }
            k = qw_kernel_get(e->g, name);
        }
        qmm_args a = {P, (uint32_t)L->out};
        qw_batch_kernel(b, k);
        qw_batch_buf(b, 0, X, 0);
        bind_t(e, b, 1, L->w); bind_t(e, b, 2, L->s); bind_t(e, b, 3, L->b);
        qw_batch_buf(b, 4, Y, 0);
        qw_batch_bytes(b, 5, &a, sizeof a);
        qw_batch_buf(b, 6, &e->ptiles, 0);
        qw_batch_dispatch(b, (unsigned)L->out / 128, e->ntiles, 1, 128, 1, 1);
        return;
    }
    const int mtn = getenv("QW_MOE_TN") ? atoi(getenv("QW_MOE_TN")) : 64, msg = mtn >= 128 ? 8 : 4;
    qw_kernel *k = qmm_kernel(e, L->bits, L->in, moe_tm(), mtn, msg, true, false);
    qmm_args a = {P, (uint32_t)L->out};
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, X, 0);
    bind_t(e, b, 1, L->w);
    bind_t(e, b, 2, L->s);
    bind_t(e, b, 3, L->b);
    qw_batch_buf(b, 4, Y, 0);
    qw_batch_bytes(b, 5, &a, sizeof a);
    qw_batch_buf(b, 6, &e->ptiles, 0);
    qw_batch_dispatch(b, (unsigned)L->out / mtn, e->ntiles, 1, 32u * msg, 1, 1);
}

// 分塊 GDN 的塊大小（QW_GDN_C 可改，預設 64）
static int gdn_chunk(void) {
    static int c = 0;
    if (!c) c = getenv("QW_GDN_C") ? atoi(getenv("QW_GDN_C")) : 64;
    return c;
}

// prefill 的分塊 DeltaNet 走 gdn_prep3s / gdn_scan2s（每個 SIMD group 各算一塊，不用 execution_simdgroups<8>；遞迴約快 26%）。
// 輸出與原本的 gdn_prep3 / gdn_scan2 逐位元相同（check_exact 驗證）；QW_GDN_CHUNK=0 改回原本的 kernel
static int gdn_fast_ovr = -1; // 量測用（QW_GDN_KB 交錯比較）
static bool gdn_fast(void) {
    static int v = -1;
    if (v < 0) v = !getenv("QW_GDN_CHUNK") || atoi(getenv("QW_GDN_CHUNK")) > 0;
    return gdn_fast_ovr >= 0 ? gdn_fast_ovr : v;
}

static bool pf_init(qw_engine *e, char *err, size_t errlen) {
    if (e->pf_ready) return true;
    char defs[512];
    snprintf(defs, sizeof defs, "#define D %d\n#define HC %d\n", D, QW_HC);
    if (!compile(e, defs, src_prefill, sizeof src_prefill, err, errlen)) return false;
    {
        char gd[256];
        snprintf(gd, sizeof gd, "#define DK 128\n#define DV 128\n#define HK 16\n#define HV 48\n#define NV 1\n#define C %d\n#define DVB 64\n", gdn_chunk());
        if (!compile(e, gd, src_gdnp, sizeof src_gdnp, err, errlen)) return false;
        if (e->m->lin_tiled) { // GGUF：V head 交錯順序的另一份（只有 prep 用到 key head；scan 共用）
            char gt[512];
            snprintf(gt, sizeof gt, "#define GDN_TILED 1\n#define gdn_seq gdn_seq_t\n#define gdn_prep gdn_prep_t\n#define gdn_scan gdn_scan_t\n"
                     "#define gdn_prep2 gdn_prep2_t\n#define gdn_prep3 gdn_prep3_t\n#define gdn_scan2 gdn_scan2_t\n#define gdn_scan2s gdn_scan2s_t\n#define gdn_prep3s gdn_prep3s_t\n%s", gd);
            if (!compile(e, gt, src_gdnp, sizeof src_gdnp, err, errlen)) return false;
        }
    }
    snprintf(defs, sizeof defs, "#define E %d\n#define TOPK %d\n#define KD %d\n#define I %d\n#define TM %d\n", QW_EXPERTS, QW_TOPK, D,
             QW_MOE_INTER, moe_tm());
    if (!compile(e, defs, src_moep, sizeof src_moep, err, errlen)) return false;
    const size_t T = PF, P = (size_t)PF * QW_TOPK;
    const int tmin = moe_tm() < 32 ? moe_tm() : 32; // GGUF 的 moe_pf_gg 可用 32 列以上的區塊
    e->max_tiles = (uint32_t)((P + tmin - 1) / tmin + QW_EXPERTS);
    bool ok = alloc(e, &e->ph, T * HCD * 4) && alloc(e, &e->pnh, T * HCD * 2) && alloc(e, &e->pnf, T * HCD * 4) &&
              alloc(e, &e->pd, T * 320 * 2) && alloc(e, &e->pd2, T * 320 * 2) && alloc(e, &e->pu, T * HCD * 2) &&
              alloc(e, &e->px, T * D * 2) && alloc(e, &e->pinj, T * 64) && alloc(e, &e->pinj2, T * 64) && alloc(e, &e->pout, T * D * 2) &&
              alloc(e, &e->pqkv, T * 10240 * 2) && alloc(e, &e->pz, T * 6144 * 2) && alloc(e, &e->pa, T * 48 * 2) &&
              alloc(e, &e->pb, T * 48 * 2) && alloc(e, &e->pq, T * 2048 * 2) && alloc(e, &e->pk, T * 2048 * 2) &&
              alloc(e, &e->pv, T * 6144 * 2) && alloc(e, &e->pla, T * 48 * 4) && alloc(e, &e->pbe, T * 48 * 4) &&
              alloc(e, &e->po, T * 6144 * 2) && alloc(e, &e->py, T * 6144 * 2) &&
              alloc(e, &e->gW, (T / 64) * 48 * 64 * 128 * 2) && alloc(e, &e->gU, (T / 64) * 48 * 64 * 128 * 4) &&
              alloc(e, &e->gQA, (T / 64) * 48 * 64 * (128 + 64) * 2) && alloc(e, &e->gK, (T / 64) * 48 * 64 * 128 * 2) &&
              alloc(e, &e->gG, (T / 64) * 48 * 4) &&
              alloc(e, &e->pqg, T * 12288 * 2) && alloc(e, &e->pkk, T * 512 * 2) && alloc(e, &e->pvv, T * 512 * 2) &&
              alloc(e, &e->piqk, T * 640 * 2) && alloc(e, &e->paq, T * 6144 * 2) && alloc(e, &e->pay, T * 6144 * 2) &&
              alloc(e, &e->plog, T * QW_EXPERTS * 4) && alloc(e, &e->psgl, T * 4) && alloc(e, &e->pids, P * 4) &&
              alloc(e, &e->pwts, P * 4) && alloc(e, &e->psg, T * 4) && alloc(e, &e->pcnt, QW_EXPERTS * 4) &&
              alloc(e, &e->poffs, QW_EXPERTS * 4) && alloc(e, &e->pcur, QW_EXPERTS * 4) && alloc(e, &e->ptiles, (size_t)e->max_tiles * 16) &&
              alloc(e, &e->pXs, P * D * 2) && alloc(e, &e->ppos, P * 4) && alloc(e, &e->pGg, P * 640 * 2) &&
              alloc(e, &e->pGu, P * 640 * 2) && alloc(e, &e->pH, P * 640 * 2) && alloc(e, &e->pDm, P * D * 2) &&
              alloc(e, &e->pSg, T * 640 * 2) && alloc(e, &e->pSu, T * 640 * 2) && alloc(e, &e->pSH, T * 640 * 2) &&
              alloc(e, &e->pDs, T * D * 2) &&
              alloc(e, &e->pemb, T * D * 2) && alloc(e, &e->pkeyh, T * HCD * 2) && alloc(e, &e->pkeyf, T * HCD * 4) &&
              alloc(e, &e->pkn, T * HCD * 4) && alloc(e, &e->pqn, T * HCD * 4) && alloc(e, &e->pval, T * D * 2) &&
              alloc(e, &e->pgated, T * HCD * 4) && alloc(e, &e->pcn, T * HCD * 4) &&
              (!e->m->ple_tab || alloc(e, &e->pvalf, T * D * 4)) &&
              alloc(e, &e->piq, T * 512 * 2) && alloc(e, &e->pscore, T * (size_t)(e->max_ctx / 4 + 1) * 4) &&
              alloc(e, &e->psel, T * 512 * 4) && alloc(e, &e->pnsel, T * 4) && alloc(e, &e->ppart, T * PF_NCH * 24 * 258 * 4) &&
              alloc(e, &e->pao, T * 6144 * 2) && alloc(e, &e->pqa, T * 32) && alloc(e, &e->prcs, T * 32 * 8) &&
              alloc(e, &e->pme, T * HCD * 4) && alloc(e, &e->pmn, T * D * 2) && alloc(e, &e->pmfe, T * D * 2) &&
              alloc(e, &e->pmnh, T * HCD * 2) && alloc(e, &e->pmfh, T * HCD * 2) && alloc(e, &e->pmh, T * HCD * 4);
    if (!ok) {
        snprintf(err, errlen, "配置 prefill 緩衝區失敗");
        return false;
    }
    e->pf_ready = true;
    return true;
}

static void pf_resid(qw_engine *e, qw_batch *b, uint32_t T) {
    qw_batch_kernel(b, pf_need(e, "p_resid"));
    qw_batch_buf(b, 0, &e->ph, 0);
    qw_batch_buf(b, 1, &e->pout, 0);
    qw_batch_buf(b, 2, &e->pinj, 0);
    qw_batch_bytes(b, 3, &T, 4);
    d1(b, T * HCD, 256);
}

// 量測用：QW_XREP=名稱:N → 該步驟多跑 N 次（結果不變），看總時間增加多少
static int xrep(const char *name) {
    static const char *v = (const char *)1;
    if (v == (const char *)1) v = getenv("QW_XREP");
    if (!v) return 1;
    const size_t n = strlen(name);
    return !strncmp(v, name, n) && v[n] == ':' ? 1 + atoi(v + n + 1) : 1;
}

// ghp_up 的 tile：GTM（token）、GTN（i）、GSG（SIMD group 數）；QW_GHP=GTMxGTNxGSG 可改（只用 qmm 驗證過的組合）
// 3 = GH_BFT（BF16 權重直接當 tensor 運算元，QW_GHP_BFT）；4–6 = ghp_down 的 GDM、GDN、GDS（QW_GHPD=GDMxGDNxGDS）
static int gh_pf_cfg(int which) {
    static int c[7] = {0};
    if (!c[0]) {
        c[0] = 64; c[1] = 16; c[2] = 2;
        c[3] = getenv("QW_GHP_BFT") ? atoi(getenv("QW_GHP_BFT")) : 1; // 量過：比轉 FP16 放 threadgroup memory 快 23%
        c[4] = 128; c[5] = 64; c[6] = 4;
        if (getenv("QW_GHP")) sscanf(getenv("QW_GHP"), "%dx%dx%d", &c[0], &c[1], &c[2]);
        if (getenv("QW_GHPD")) sscanf(getenv("QW_GHPD"), "%dx%dx%d", &c[4], &c[5], &c[6]);
    }
    return c[which];
}

// GGUF 的 prefill hyper kernel（kernels/hyper_gg.metal 的 GH_PF 部分）
static qw_kernel *ghp_kernel(qw_engine *e, const char *name) {
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    char defs[256], err[8192];
    snprintf(defs, sizeof defs, "#define GH_PF 1\n#define D %d\n#define HC %d\n#define GTM %d\n#define GTN %d\n#define GSG %d\n#define GH_BFT %d\n"
             "#define GDM %d\n#define GDN %d\n#define GDS %d\n", D, QW_HC, gh_pf_cfg(0), gh_pf_cfg(1), gh_pf_cfg(2), gh_pf_cfg(3),
             gh_pf_cfg(4), gh_pf_cfg(5), gh_pf_cfg(6));
    if (!compile(e, defs, src_hyper_gg, sizeof src_hyper_gg, err, sizeof err)) {
        fprintf(stderr, "編譯 hyper_gg（prefill）失敗：%s\n", err);
        exit(1);
    }
    return pf_need(e, name);
}

// GGUF 的 prefill hyper-connection：ghp_hc_in（殘差 + norm → FP16、r[t][s] 借放 pnf）→ ghp_down（mix_down + SiLU）
// → ghp_up（mix_up + 合流一起，u 不寫出）
// 1（預設）：ghp_hc_in_t（一個 token 一個 threadgroup，與原本逐位元相同）；QW_GHP_HCT=0 用原本的 ghp_hc_in（kbench 會切換來比較）
static int ghp_hct = -1;
static void ghp_in16(qw_engine *e, qw_batch *b, const qw_hyper *H, uint32_t T, uint32_t apply, uint32_t want) {
    if (ghp_hct < 0) ghp_hct = getenv("QW_GHP_HCT") ? atoi(getenv("QW_GHP_HCT")) : 1;
    int hct = ghp_hct;
    qw_kernel *k = ghp_kernel(e, hct ? "ghp_hc_in_t" : "ghp_hc_in");
    if (hct && qw_kernel_max_threads(k) < D / 4) k = ghp_kernel(e, "ghp_hc_in"), hct = ghp_hct = 0;
    for (int rp = 0; rp < xrep("hcin"); rp++) {
    if (rp) apply = 0;
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, &e->ph, 0);
    qw_batch_buf(b, 1, &e->pout, 0);
    qw_batch_buf(b, 2, &e->pinj, 0);
    bind_t(e, b, 3, H->hc_norm);
    qw_batch_buf(b, 4, &e->pnh, 0);
    qw_batch_buf(b, 5, &e->pnf, 0);
    qw_batch_bytes(b, 6, &apply, 4);
    if (want) bind_t(e, b, 7, H->inject);
    else qw_batch_buf(b, 7, &e->pnf, 0);
    qw_batch_buf(b, 8, &e->pinj2, 0);
    qw_batch_bytes(b, 9, &want, 4);
    if (hct) qw_batch_dispatch(b, T, 1, 1, D / 4, 1, 1);
    else qw_batch_dispatch(b, QW_HC, T, 1, D / 4, 1, 1);
    }
    if (want) { qw_buf t = e->pinj; e->pinj = e->pinj2; e->pinj2 = t; }
}
static void ghp_down16(qw_engine *e, qw_batch *b, const qw_hyper *H, uint32_t T) {
    for (int rp = 0; rp < xrep("hcdown"); rp++) { // mix_down + SiLU（ghp_down）→ pd2
    qw_batch_kernel(b, ghp_kernel(e, "ghp_down"));
    qw_batch_buf(b, 0, &e->pnh, 0);
    bind_t(e, b, 1, H->mix_down.w);
    qw_batch_buf(b, 2, &e->pd2, 0);
    qw_batch_bytes(b, 3, &T, 4);
    qw_batch_dispatch(b, QW_HC_LOWRANK / gh_pf_cfg(5), (T + gh_pf_cfg(4) - 1) / gh_pf_cfg(4), 1, 32 * gh_pf_cfg(6), 1, 1);
    }
}
static void ghp_up16(qw_engine *e, qw_batch *b, const qw_hyper *H, uint32_t T) {
    for (int rp = 0; rp < xrep("hcup"); rp++) {
    qw_batch_kernel(b, ghp_kernel(e, "ghp_up"));
    qw_batch_buf(b, 0, &e->pd2, 0);
    bind_t(e, b, 1, H->mix_up.w);
    qw_batch_buf(b, 2, &e->ph, 0);
    qw_batch_buf(b, 3, &e->pnf, 0);
    bind_t(e, b, 4, H->hc_norm);
    qw_batch_buf(b, 5, &e->px, 0);
    qw_batch_bytes(b, 6, &T, 4);
    qw_batch_dispatch(b, D / gh_pf_cfg(1), (T + gh_pf_cfg(0) - 1) / gh_pf_cfg(0), 1, 32 * gh_pf_cfg(2), 1, 1);
    }
}

// INT8 版（QW_INT8 清單含 hyper，mix_down 有 INT8 影子；會改變精度）：ghp_hc_in8（殘差 + norm → uint8 進 pi8x）
// → qmm_i8（mix_down → pd）→ ghp_silu（→ pd2）→ 原本的 ghp_up。mix_up 不走 INT8（見 kernels/hyper_gg.metal）
static void ghp_in8(qw_engine *e, qw_batch *b, const qw_hyper *H, uint32_t T, uint32_t apply, uint32_t want) {
    qw_kernel *kin = ghp_kernel(e, "ghp_hc_in8");
    if (qw_kernel_max_threads(kin) < D / 4) { fprintf(stderr, "ghp_hc_in8：每個 threadgroup 最多 %u 個執行緒（要 %d）\n", qw_kernel_max_threads(kin), D / 4); exit(1); }
    for (int rp = 0; rp < xrep("hcin"); rp++) { // 量測用（QW_XREP，同 ghp_in16）
    if (rp) apply = 0;
    qw_batch_kernel(b, kin);
    qw_batch_buf(b, 0, &e->ph, 0);
    qw_batch_buf(b, 1, &e->pout, 0);
    qw_batch_buf(b, 2, &e->pinj, 0);
    bind_t(e, b, 3, H->hc_norm);
    qw_batch_buf(b, 4, &e->pi8x, 0);
    qw_batch_buf(b, 5, &e->pnf, 0);
    qw_batch_bytes(b, 6, &apply, 4);
    if (want) bind_t(e, b, 7, H->inject);
    else qw_batch_buf(b, 7, &e->pnf, 0);
    qw_batch_buf(b, 8, &e->pinj2, 0);
    qw_batch_bytes(b, 9, &want, 4);
    qw_batch_buf(b, 10, &e->pi8s, 0);
    qw_batch_buf(b, 11, &e->pi8a, 0);
    qw_batch_dispatch(b, T, 1, 1, D / 4, 1, 1);
    }
    if (want) { qw_buf t = e->pinj; e->pinj = e->pinj2; e->pinj2 = t; }
}
static void ghp_down8(qw_engine *e, qw_batch *b, const qw_hyper *H, uint32_t T) { // mix_down：qmm_i8（同 pf_qmm_i8，輸入已量化）→ pd（FP16）
    const qw_qlin *Ld = H->mix_down.i8;
    qw_kernel *kq, *km;
    pf_i8_kernels(e, Ld, T, false, &kq, &km);
    const int tm = T >= 128 ? 128 : 64, tn = 64, sg = 4;
    const uint32_t a[2] = {T, (uint32_t)Ld->out};
    for (int rp = 0; rp < xrep("hcdown"); rp++) {
    qw_batch_kernel(b, km);
    qw_batch_buf(b, 0, &e->pi8x, 0);
    bind_t(e, b, 1, Ld->w);
    bind_t(e, b, 2, Ld->s);
    qw_batch_buf(b, 4, &e->pd, 0);
    qw_batch_bytes(b, 5, a, sizeof a);
    qw_batch_buf(b, 6, &e->pi8s, 0);
    qw_batch_buf(b, 7, &e->pi8a, 0);
    bind_t(e, b, 8, (const st_tensor *)Ld->rsum_t);
    qw_batch_dispatch(b, (unsigned)Ld->out / tn, (T + tm - 1) / tm, 1, 32u * sg, 1, 1);
    }
}
static void pf_hyper_gg(qw_engine *e, qw_batch *b, const qw_hyper *H, uint32_t T, uint32_t apply, uint32_t want) {
    if (H->mix_down.i8) {
        qw_kernel *kq, *km;
        pf_i8_kernels(e, H->mix_down.i8, T, false, &kq, &km); // 配置 pi8x / pi8s / pi8a（[PF + 128] 列，夠放 HCD 欄）
        e->i8q_m = 0, e->i8q_k = 0; // pi8x 會被改寫：共用中的量化結果失效
        ghp_in8(e, b, H, T, apply, want);
        ghp_down8(e, b, H, T);
        qw_batch_kernel(b, ghp_kernel(e, "ghp_silu"));
        qw_batch_buf(b, 0, &e->pd, 0);
        qw_batch_buf(b, 1, &e->pd2, 0);
        qw_batch_dispatch(b, T, 1, 1, QW_HC_LOWRANK, 1, 1);
    } else {
        ghp_in16(e, b, H, T, apply, want);
        ghp_down16(e, b, H, T);
    }
    ghp_up16(e, b, H, T);
}

static void pf_hyper(qw_engine *e, qw_batch *b, const qw_hyper *H, uint32_t T) {
    uint32_t apply = e->ppending ? 1 : 0, want = H->inject ? 1 : 0;
    e->ppending = false;
    if (H->mix_down.gq) { pf_hyper_gg(e, b, H, T, apply, want); return; }
    for (int rp = 0; rp < xrep("hcin"); rp++) {
    if (rp) apply = 0;
    qw_batch_kernel(b, H->mix_down.gq ? ghp_kernel(e, "ghp_hc_in") : pf_need(e, "p_hc_in")); // GGUF：hc_norm 是 F32
    qw_batch_buf(b, 0, &e->ph, 0);
    qw_batch_buf(b, 1, &e->pout, 0);
    qw_batch_buf(b, 2, &e->pinj, 0);
    bind_t(e, b, 3, H->hc_norm);
    qw_batch_buf(b, 4, &e->pnh, 0);
    qw_batch_buf(b, 5, &e->pnf, 0);
    qw_batch_bytes(b, 6, &apply, 4);
    if (want) bind_t(e, b, 7, H->inject);
    else qw_batch_buf(b, 7, &e->pnf, 0);
    qw_batch_buf(b, 8, &e->pinj2, 0);
    qw_batch_bytes(b, 9, &want, 4);
    qw_batch_dispatch(b, QW_HC, T, 1, 1024, 1, 1);
    }
    if (want) { qw_buf t = e->pinj; e->pinj = e->pinj2; e->pinj2 = t; }
    for (int rp = 0; rp < xrep("hcdown"); rp++) pf_qmm(e, b, &H->mix_down, &e->pnh, &e->pd, T, false);
    uint32_t n = T * QW_HC_LOWRANK;
    qw_batch_kernel(b, pf_need(e, "p_silu_div"));
    qw_batch_buf(b, 0, &e->pd, 0);
    qw_batch_buf(b, 1, &e->pd2, 0);
    qw_batch_bytes(b, 2, &n, 4);
    d1(b, n, 256);
    for (int rp = 0; rp < xrep("hcup"); rp++) pf_qmm(e, b, &H->mix_up, &e->pd2, &e->pu, T, false);
    for (int rp = 0; rp < xrep("hcmix"); rp++) {
    qw_batch_kernel(b, pf_need(e, "p_hcmix"));
    qw_batch_buf(b, 0, &e->pu, 0);
    qw_batch_buf(b, 1, &e->pnf, 0);
    qw_batch_buf(b, 2, &e->px, 0);
    qw_batch_bytes(b, 3, &T, 4);
    d1(b, T * D, 256);
    }
}

typedef struct { uint32_t T, nchunk; } gdn_chunk_args;

static void pf_deltanet_core(qw_engine *e, qw_batch *b, const qw_deltanet *N, int li, uint32_t T);
static void pf_deltanet(qw_engine *e, qw_batch *b, const qw_deltanet *N, int li, uint32_t T) {
    dn_apply(e, b, li);
    const bool tap = li == dn_tap_li();
    if (tap) dn_tap_copy(e, b, &e->px, false, T);
    pf_deltanet_core(e, b, N, li, T);
    if (tap) { dn_tap_copy(e, b, &e->pout, true, T); e->dntap_n += (int)T; }
}

// 量測用（QW_PF_KBENCH + QW_GDN_KB）：只編碼 pf_deltanet_core 的部分步驟：
// 1 投影（qkv / z / a / b）、2 p_gdn_in + conv 狀態、4 分塊 prep、8 分塊 scan（含尾巴）、16 gated norm、32 out_proj
static unsigned pfdn_mask = ~0u;

// DeltaNet 的輸入投影 px → pqkv / pz / pa / pb；parts：1 qkv、2 z、4 a + b（量測用可只編碼一部分，正式路徑 7）
static void pf_dn_proj(qw_engine *e, qw_batch *b, const qw_deltanet *N, uint32_t T, unsigned parts) {
    // GGUF DeltaNet 投影改走 attention 的 qmm_ggm（每 256 個值解一次 FP16）；QW_DN_GGM=0 用通用的 IQF qmm
    pf_i8_share(e, &e->px); // INT8：px 只量化一次
    if (dn_ggm_on() && N->in_qkv.gq) {
        if (parts & 1) pf_attn_mm(e, b, &N->in_qkv, &e->px, &e->pqkv, T);
        if (parts & 2) pf_attn_mm(e, b, &N->in_z, &e->px, &e->pz, T);
    } else {
        if (parts & 1) pf_qmm(e, b, &N->in_qkv, &e->px, &e->pqkv, T, false);
        if (parts & 2) pf_qmm(e, b, &N->in_z, &e->px, &e->pz, T, false);
    }
    if (parts & 4) {
        pf_qmm(e, b, &N->in_a, &e->px, &e->pa, T, false);
        pf_qmm(e, b, &N->in_b, &e->px, &e->pb, T, false);
    }
    pf_i8_share(e, NULL);
}

// p_gdn_in 的 token 區塊版（kernels/prefill.metal 的 p_gdn_in_tb：每個 threadgroup 連續 GDN_TB 個 token，逐位元相同）；
// QW_GDN_TB=0 改回每個 token 一個 threadgroup
enum { GDN_TB = 16 }; // 同 prefill.metal 的 GDN_TB
static int gdn_tb_ovr = -1; // 量測用（QW_DN_KB 交錯比較）
static bool gdn_tb_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("QW_GDN_TB") ? atoi(getenv("QW_GDN_TB")) : 1;
    return gdn_tb_ovr >= 0 ? gdn_tb_ovr : v;
}

static void pf_deltanet_core(qw_engine *e, qw_batch *b, const qw_deltanet *N, int li, uint32_t T) {
    if (pfdn_mask & 1) pf_dn_proj(e, b, N, T, 7);
    const size_t cst = (size_t)li * 3 * 10240 * 4;
    const bool tl = e->m->lin_tiled; // GGUF：V head 交錯順序、A_neg（F32）
    if (pfdn_mask & 2) {
    for (int rp = 0; rp < xrep("gdnin"); rp++) { // 量測用：QW_XREP=gdnin:n
    const bool tb = gdn_tb_on();
    qw_batch_kernel(b, pf_need(e, tb ? (tl ? "p_gdn_in_t_tb" : "p_gdn_in_tb") : (tl ? "p_gdn_in_t" : "p_gdn_in")));
    qw_batch_buf(b, 0, &e->pqkv, 0);
    bind_t(e, b, 1, N->conv1d);
    qw_batch_buf(b, 2, &e->conv_st, cst);
    qw_batch_buf(b, 3, &e->pa, 0);
    qw_batch_buf(b, 4, &e->pb, 0);
    bind_t(e, b, 5, tl ? N->A_neg : N->A_log);
    bind_t(e, b, 6, N->dt_bias);
    qw_batch_buf(b, 7, &e->pq, 0);
    qw_batch_buf(b, 8, &e->pk, 0);
    qw_batch_buf(b, 9, &e->pv, 0);
    qw_batch_buf(b, 10, &e->pla, 0);
    qw_batch_buf(b, 11, &e->pbe, 0);
    if (tb) {
        qw_batch_bytes(b, 12, &T, 4);
        qw_batch_dispatch(b, 16 + 48, (T + GDN_TB - 1) / GDN_TB, 1, 128, 1, 1);
    } else qw_batch_dispatch(b, 16 + 48, T, 1, 128, 1, 1);
    }
    uint32_t C = 10240;
    qw_batch_kernel(b, pf_need(e, "p_conv_state"));
    qw_batch_buf(b, 0, &e->pqkv, 0);
    qw_batch_buf(b, 1, &e->conv_st, cst);
    qw_batch_bytes(b, 2, &T, 4);
    qw_batch_bytes(b, 3, &C, 4);
    d1(b, C, 256);
    for (int k = 0; k < e->seg_n; k++) { // 段中檢查點的 conv 狀態：前 q 個輸入的最後 3 個
        qw_batch_kernel(b, pf_need(e, "p_conv_state"));
        qw_batch_buf(b, 0, &e->pqkv, 0);
        qw_batch_buf(b, 1, &e->snConv, (size_t)k * e->conv_st.size + cst);
        qw_batch_bytes(b, 2, &e->seg_q[k], 4);
        qw_batch_bytes(b, 3, &C, 4);
        d1(b, C, 256);
    }
    }
    const size_t soff = (size_t)li * 48 * 128 * 128 * 4;
    // 分塊 GDN：前 T64 個 token
    const uint32_t CC = (uint32_t)gdn_chunk();
    const uint32_t T64 = getenv("QW_PF_GDN_SEQ") ? 0 : T / CC * CC; // 除錯：QW_PF_GDN_SEQ=1 全部逐 token
    if (T64) {
        gdn_chunk_args a = {T64, T64 / CC};
        uint32_t tb = 0;
        if (pfdn_mask & 4) {
        qw_batch_kernel(b, pf_need(e, gdn_fast() ? (tl ? "gdn_prep3s_t" : "gdn_prep3s") : (tl ? "gdn_prep3_t" : "gdn_prep3")));
        qw_batch_buf(b, 0, &e->pq, 0);
        qw_batch_buf(b, 1, &e->pk, 0);
        qw_batch_buf(b, 2, &e->pv, 0);
        qw_batch_buf(b, 3, &e->pla, 0);
        qw_batch_buf(b, 4, &e->pbe, 0);
        qw_batch_buf(b, 5, &e->gW, 0);
        qw_batch_buf(b, 6, &e->gU, 0);
        qw_batch_buf(b, 7, &e->gQA, 0);
        qw_batch_buf(b, 8, &e->gK, 0);
        qw_batch_buf(b, 9, &e->gG, 0);
        qw_batch_bytes(b, 10, &a, sizeof a);
        qw_batch_dispatch(b, a.nchunk, 48, 1, 256, 1, 1);
        }
        if (pfdn_mask & 8) {
        qw_batch_kernel(b, pf_need(e, gdn_fast() ? "gdn_scan2s" : "gdn_scan2"));
        qw_batch_buf(b, 0, &e->gW, 0);
        qw_batch_buf(b, 1, &e->gU, 0);
        qw_batch_buf(b, 2, &e->gQA, 0);
        qw_batch_buf(b, 3, &e->gK, 0);
        qw_batch_buf(b, 4, &e->gG, 0);
        qw_batch_buf(b, 5, &e->S, soff);
        qw_batch_buf(b, 6, &e->po, 0);
        qw_batch_bytes(b, 7, &a, sizeof a);
        qw_batch_bytes(b, 8, &tb, 4);
        struct { uint32_t n, stride, chunk[4]; } sa = {(uint32_t)e->seg_n, (uint32_t)(e->S.size / 4), {0}};
        for (int k = 0; k < e->seg_n; k++) sa.chunk[k] = e->seg_q[k] / CC;
        qw_batch_buf(b, 9, e->seg_n ? &e->snS : &e->S, soff);
        qw_batch_bytes(b, 10, &sa, sizeof sa);
        qw_batch_dispatch(b, 48, 128 / 64, 1, 256, 1, 1);
        }
    }
    // 不滿 64 的尾巴：逐 token 用 decode 的單步 kernel
    for (uint32_t t = T64; t < T && (pfdn_mask & 8); t++) {
        qw_batch_kernel(b, tl ? e->gdn_step_t : e->gdn_step);
        qw_batch_buf(b, 0, &e->pq, (size_t)t * 2048 * 2);
        qw_batch_buf(b, 1, &e->pk, (size_t)t * 2048 * 2);
        qw_batch_buf(b, 2, &e->pv, (size_t)t * 6144 * 2);
        qw_batch_buf(b, 3, &e->pla, (size_t)t * 48 * 4);
        qw_batch_buf(b, 4, &e->pbe, (size_t)t * 48 * 4);
        qw_batch_buf(b, 5, &e->S, soff);
        qw_batch_buf(b, 6, &e->po, (size_t)t * 6144 * 2);
        qw_batch_buf(b, 7, &e->ssp, 0);
        qw_batch_buf(b, 8, &e->S, soff);
        qw_batch_dispatch(b, 48, 128 / 32, 1, 256, 1, 1);
    }
    if (pfdn_mask & 16) {
    qw_batch_kernel(b, pf_need(e, "p_normgate"));
    qw_batch_buf(b, 0, &e->po, 0);
    qw_batch_buf(b, 1, &e->pz, 0);
    bind_t(e, b, 2, N->norm);
    qw_batch_buf(b, 3, &e->py, 0);
    qw_batch_dispatch(b, 48, T, 1, 128, 1, 1);
    }
    if (!(pfdn_mask & 32)) return;
    if (dn_ggm_on() && N->out.gq) pf_attn_mm(e, b, &N->out, &e->py, &e->pout, T);
    else pf_qmm(e, b, &N->out, &e->py, &e->pout, T, false);
}

// attention prefill 的 INT8 Q·Kᵀ（QW_INT8 的類別清單明確含 "qk" 才用，例如 QW_INT8=lin,attn,shared,moe,qk；QW_INT8=1 不含，
// 會改變精度）：1 = indexer 打分（qk_score_i8）、2 = 核心的 S = Q·Kᵀ（qsa_flash_tr8）。QW_QK_PARTS=位元 可只開一部分（量測用）。
// MLX / GGUF 共用（prefill 的 attention 核心兩者相同）；decode / 驗證不受影響
static bool i8qk_on(void) {
    static int v = -1;
    if (v < 0) {
        v = 0;
        const char *l = getenv("QW_INT8");
        if (l && strcmp(l, "1") && strcmp(l, "0")) {
            char buf[256];
            snprintf(buf, sizeof buf, "%s", l);
            for (char *sv = NULL, *t = strtok_r(buf, ",", &sv); t; t = strtok_r(NULL, ",", &sv))
                if (!strcmp(t, "qk")) v = 1;
        }
    }
    return v;
}
static int qk_ovr = -1; // 量測用（QW_KB_POS + QW_QK_AB）：同一個行程裡切換
static int i8qk_parts(void) {
    static int v = -1;
    if (qk_ovr >= 0) return qk_ovr;
    if (v < 0) v = !i8qk_on() ? 0 : getenv("QW_QK_PARTS") ? atoi(getenv("QW_QK_PARTS")) : 3;
    return v;
}

// qsa_flash_tr8 的輸入：q（paq，T × 24 列）與 K cache 的前 npos 個位置（× 2 個 kv head）各列量化成 INT8。K 每層、每段重新量化
// （decode 會寫新的 K，不另外維護影子）。緩衝區照 max_ctx 一次配置（command buffer 不保留參照，不能中途換）
static void pf_flash_q8(qw_engine *e, qw_batch *b, size_t koff, uint32_t T, uint32_t npos) {
    if (!e->pfq8.ptr && !(alloc(e, &e->pfq8, (size_t)PF * 24 * 256) && alloc(e, &e->pfqs, (size_t)PF * 24 * 4) &&
                          alloc(e, &e->pfk8, (size_t)e->max_ctx * 2 * 256) && alloc(e, &e->pfks, (size_t)e->max_ctx * 2 * 4))) {
        fprintf(stderr, "INT8 attention buffer 配置失敗\n");
        exit(1);
    }
    if (T > (uint32_t)PF || npos > (uint32_t)e->max_ctx || npos > (uint32_t)e->kvcap) {
        fprintf(stderr, "pf_flash_q8：T %u / 位置 %u 超出範圍\n", T, npos);
        exit(1);
    }
    const uint32_t nq = T * 24, nk = npos * 2;
    qw_batch_kernel(b, pf_need(e, "qk8_quant"));
    qw_batch_buf(b, 0, &e->paq, 0);
    qw_batch_buf(b, 1, &e->pfq8, 0);
    qw_batch_buf(b, 2, &e->pfqs, 0);
    qw_batch_bytes(b, 3, &nq, 4);
    qw_batch_dispatch(b, (nq + 7) / 8, 1, 1, 256, 1, 1);
    qw_batch_kernel(b, pf_need(e, "qk8_quant"));
    qw_batch_buf(b, 0, &e->Kc, koff);
    qw_batch_buf(b, 1, &e->pfk8, 0);
    qw_batch_buf(b, 2, &e->pfks, 0);
    qw_batch_bytes(b, 3, &nk, 4);
    qw_batch_dispatch(b, (nk + 7) / 8, 1, 1, 256, 1, 1);
}

typedef struct { uint32_t L, pos, nvis, ntok, sparse; } qsa_args_t;
static uint32_t pf_nch(void) {
    const char *v = getenv("QW_PF_NCH");
    const int n = v ? atoi(v) : 2;
    return (uint32_t)(n < 1 ? 1 : n > PF_NCH ? PF_NCH : n);
}

static bool pf_attn_old(void) { return getenv("QW_PF_ATTN_OLD") != NULL; }

// tensor 單元版 attention：每個 (kv head, query) 一個 threadgroup，輸出 pao（之後乘 gate）。pqa 要先填好
// 版本：1 = qsa_flash_tr（預設）、0 = qsa_flash_pf（舊版，QW_PF_FLASH=0）、2 = qsa_union4 + qsa_flash4（實驗，較慢；
// 舊開關 QW_ATTN4 也是這個）。QW_PF_FLASH=n 指定
static int pf_flash_mode = -1;
static int pf_flash_variant(void) {
    if (pf_flash_mode < 0) pf_flash_mode = getenv("QW_PF_FLASH") ? atoi(getenv("QW_PF_FLASH")) : getenv("QW_ATTN4") ? 2 : 1;
    return pf_flash_mode;
}

// gate = true：可以的話（qsa_flash_tr）順便乘上 gate、直接寫 pay，回傳 true；否則寫 pao，要另外跑 kb_attn_gate
static bool pf_flash(qw_engine *e, qw_batch *b, size_t koff, uint32_t T, bool gate) {
    const uint32_t G = (T + 3) / 4;
    const qsa_args_t *qa = e->pqa.ptr;
    const int mode = pf_flash_variant();
    if (mode == 2 && qa[T - 1].pos / 4 + 1 <= 32768) { // 實驗：4 個相鄰 query 共用 K/V（目前比較慢，預設關）
        const size_t MAXU = 2064;
        if (!e->pul.ptr && !(alloc(e, &e->pul, ((size_t)PF / 4 + 1) * MAXU * 4) && alloc(e, &e->pum, ((size_t)PF / 4 + 1) * MAXU) &&
                             alloc(e, &e->puc, ((size_t)PF / 4 + 1) * 4))) {
            fprintf(stderr, "attention 聯集 buffer 配置失敗\n");
            exit(1);
        }
        qw_batch_kernel(b, pf_need(e, "qsa_union4"));
        qw_batch_buf(b, 0, &e->psel, 0);
        qw_batch_buf(b, 1, &e->pnsel, 0);
        qw_batch_buf(b, 2, &e->pqa, 0);
        qw_batch_buf(b, 3, &e->pul, 0);
        qw_batch_buf(b, 4, &e->pum, 0);
        qw_batch_buf(b, 5, &e->puc, 0);
        qw_batch_bytes(b, 6, &T, 4);
        qw_batch_dispatch(b, G, 1, 1, 1024, 1, 1);
        qw_batch_kernel(b, pf_need(e, "qsa_flash4"));
        qw_batch_buf(b, 0, &e->paq, 0);
        qw_batch_buf(b, 1, &e->Kc, koff);
        qw_batch_buf(b, 2, &e->Vc, koff);
        qw_batch_buf(b, 3, &e->pul, 0);
        qw_batch_buf(b, 4, &e->pum, 0);
        qw_batch_buf(b, 5, &e->puc, 0);
        qw_batch_buf(b, 6, &e->pao, 0);
        qw_batch_buf(b, 7, &e->pqa, 0);
        qw_batch_bytes(b, 8, &T, 4);
        qw_batch_dispatch(b, 2, G, 1, 256, 1, 1);
        return false;
    }
    static int fuse = -1;
    if (fuse < 0) fuse = getenv("QW_PF_TRGATE") ? atoi(getenv("QW_PF_TRGATE")) : 1;
    const uint32_t gated = mode != 0 && gate && fuse;
    if (mode == 8 || (mode == 1 && (i8qk_parts() & 2))) { // INT8 Q·Kᵀ（qsa_flash_tr8）
        pf_flash_q8(e, b, koff, T, qa[T - 1].pos + 1);
        qw_batch_kernel(b, pf_need(e, "qsa_flash_tr8"));
        qw_batch_buf(b, 0, &e->pfq8, 0);
        qw_batch_buf(b, 1, &e->pfk8, 0);
        qw_batch_buf(b, 2, &e->Vc, koff);
        qw_batch_buf(b, 3, &e->psel, 0);
        qw_batch_buf(b, 4, &e->pnsel, 0);
        qw_batch_buf(b, 5, gated ? &e->pay : &e->pao, 0);
        qw_batch_buf(b, 6, &e->pqa, 0);
        qw_batch_buf(b, 7, &e->pqg, 0);
        qw_batch_bytes(b, 8, &gated, 4);
        qw_batch_buf(b, 9, &e->pfqs, 0);
        qw_batch_buf(b, 10, &e->pfks, 0);
        qw_batch_dispatch(b, 2, T, 1, 32 * 8, 1, 1);
        return gated;
    }
    qw_batch_kernel(b, pf_need(e, mode == 0 ? "qsa_flash_pf" : "qsa_flash_tr"));
    qw_batch_buf(b, 0, &e->paq, 0);
    qw_batch_buf(b, 1, &e->Kc, koff);
    qw_batch_buf(b, 2, &e->Vc, koff);
    qw_batch_buf(b, 3, &e->psel, 0);
    qw_batch_buf(b, 4, &e->pnsel, 0);
    qw_batch_buf(b, 5, gated ? &e->pay : &e->pao, 0);
    qw_batch_buf(b, 6, &e->pqa, 0);
    if (mode != 0) {
        qw_batch_buf(b, 7, &e->pqg, 0);
        qw_batch_bytes(b, 8, &gated, 4);
    }
    qw_batch_dispatch(b, 2, T, 1, mode == 0 ? 128 : 32 * 8, 1, 1); // qsa_flash_tr：TR_NSG = 8
    return gated;
}

// RoPE 表（位置 pos0 … pos0 + T − 1）：同一段的 12 個 attention 層（與 MTP 層）共用，起點不同才重算
static void pf_rope_tab(qw_engine *e, qw_batch *b, uint32_t pos0, uint32_t T) {
    if (e->rcs_T >= T && e->rcs_pos0 == pos0) return;
    if (T > (uint32_t)PF) { fprintf(stderr, "pf_rope_tab：T = %u 超過 PF\n", T); exit(1); }
    qw_batch_kernel(b, pf_need(e, "p_rope_tab"));
    qw_batch_buf(b, 0, &e->prcs, 0);
    qw_batch_bytes(b, 1, &pos0, 4);
    qw_batch_dispatch(b, 1, T, 1, 32, 1, 1);
    e->rcs_pos0 = pos0;
    e->rcs_T = T;
}

// QSA indexer 打分的 INT8 prefill（kernels/qk_i8.metal，QW_INT8 含 qk）：iq 每個 (token, head)、區塊 key 每個區塊對稱量化成
// INT8，tensor 單元算 INT8 × INT8 → INT32，再乘兩邊的 scale。區塊 key 每層、每段重新量化（decode 會新增區塊，不另外維護影子）
enum { QK_TM = 32, QK_TN = 64, QK_SG = 4, QK_PAD = 64 }; // QK_PAD ≥ QK_TM、QK_TN：最後一塊超出的列讀得到
static void pf_qk_score_i8(qw_engine *e, qw_batch *b, size_t boff, uint32_t T, uint32_t pos0, uint32_t sstride, uint32_t nbmax) {
    const uint32_t qp = (uint32_t)PF + QK_PAD, nbp = (uint32_t)(e->max_ctx / 4 + 1) + QK_PAD;
    if (nbmax + QK_TN > nbp || T > (uint32_t)PF) { fprintf(stderr, "pf_qk_score_i8：nbmax %u / T %u 超出範圍\n", nbmax, T); exit(1); }
    if (!e->pqk8.ptr && !(alloc(e, &e->pqk8, (size_t)4 * qp * 128) && alloc(e, &e->pqks, (size_t)4 * qp * 4) &&
                          alloc(e, &e->pkb8, (size_t)nbp * 128) && alloc(e, &e->pkbs, (size_t)nbp * 4))) {
        fprintf(stderr, "INT8 indexer buffer 配置失敗\n");
        exit(1);
    }
    char kn[64];
    snprintf(kn, sizeof kn, "qk_score_i8_%dx%dx%d", QK_TM, QK_TN, QK_SG);
    if (!qw_kernel_get(e->g, kn)) {
        char defs[512], err[8192];
        snprintf(defs, sizeof defs, "#define IH 4\n#define ID 128\n#define CR %d\n#define TM %d\n#define TN %d\n#define SG %d\n"
                 "#define qk_score_i8 %s\n", QW_IDX_CR, QK_TM, QK_TN, QK_SG, kn);
        if (!compile(e, defs, src_qk_i8, sizeof src_qk_i8, err, sizeof err)) {
            fprintf(stderr, "編譯 %s 失敗：%s\n", kn, err);
            exit(1);
        }
    }
    const uint32_t aq[4] = {T * 4, 4, qp, 0}, ak[4] = {nbmax, 1, 0, 0};
    qw_batch_kernel(b, pf_need(e, "qk_quant"));
    qw_batch_buf(b, 0, &e->piq, 0);
    qw_batch_buf(b, 1, &e->pqk8, 0);
    qw_batch_buf(b, 2, &e->pqks, 0);
    qw_batch_bytes(b, 3, aq, sizeof aq);
    qw_batch_dispatch(b, (T * 4 + 7) / 8, 1, 1, 256, 1, 1);
    if (nbmax) {
        qw_batch_kernel(b, pf_need(e, "qk_quant"));
        qw_batch_buf(b, 0, &e->kblk, boff);
        qw_batch_buf(b, 1, &e->pkb8, 0);
        qw_batch_buf(b, 2, &e->pkbs, 0);
        qw_batch_bytes(b, 3, ak, sizeof ak);
        qw_batch_dispatch(b, (nbmax + 7) / 8, 1, 1, 256, 1, 1);
    }
    const uint32_t sa[5] = {pos0, T, sstride, qp, nbp};
    qw_batch_kernel(b, pf_need(e, kn));
    qw_batch_buf(b, 0, &e->pqk8, 0);
    qw_batch_buf(b, 1, &e->pqks, 0);
    qw_batch_buf(b, 2, &e->pkb8, 0);
    qw_batch_buf(b, 3, &e->pkbs, 0);
    qw_batch_buf(b, 4, &e->pscore, 0);
    qw_batch_bytes(b, 5, sa, sizeof sa);
    qw_batch_dispatch(b, (nbmax + QK_TN - 1) / QK_TN, (T + QK_TM - 1) / QK_TM, 1, 32 * QK_SG, 1, 1);
}

// QSA 稀疏 attention（prefill）：indexer 分數矩陣 → 每個 query 選前 512 個區塊 → 每個 query 一個 z 的 attention
static void pf_qsa_sparse(qw_engine *e, qw_batch *b, const qw_attn *A, int ai, uint32_t T, uint32_t pos0) {
    const size_t Lc = (size_t)e->kvcap;
    const size_t koff = (size_t)ai * Lc * 2 * 256 * 2, boff = (size_t)ai * (Lc / 4 + 1) * 128 * 2;
    const uint32_t sstride = (uint32_t)(Lc / 4 + 1), nbmax = (pos0 + T) / 4;
    // 每個 query 的參數（CPU 算好放 buffer）
    qsa_args_t *qa = e->pqa.ptr;
    for (uint32_t t = 0; t < T; t++) {
        const uint32_t pos = pos0 + t;
        qsa_args_t a = {pos + 1, pos, (pos + 1) / QW_IDX_CR, pos + 1, pos + 1 > QW_IDX_BUDGET};
        if (a.sparse) {
            const uint32_t ns = a.nvis < 512 ? a.nvis : 512;
            a.ntok = ns * QW_IDX_CR + (pos + 1 - a.nvis * QW_IDX_CR);
        }
        qa[t] = a;
    }
    pf_rope_tab(e, b, pos0, T);
    qw_batch_kernel(b, pf_need(e, A->nf32 ? "p_idx_q_tf" : "p_idx_q_t"));
    qw_batch_buf(b, 0, &e->piqk, 0);
    bind_t(e, b, 1, A->idx_q_norm);
    qw_batch_buf(b, 2, &e->piq, 0);
    qw_batch_buf(b, 3, &e->prcs, 0);
    qw_batch_dispatch(b, T, 1, 1, 128, 1, 1);
    if (i8qk_parts() & 1) pf_qk_score_i8(e, b, boff, T, pos0, sstride, nbmax);
    else {
    qw_batch_kernel(b, pf_need(e, "p_qsa_score"));
    qw_batch_buf(b, 0, &e->piq, 0);
    qw_batch_buf(b, 1, &e->kblk, boff);
    qw_batch_buf(b, 2, &e->pscore, 0);
    qw_batch_bytes(b, 3, &pos0, 4);
    qw_batch_bytes(b, 4, &T, 4);
    qw_batch_bytes(b, 5, &sstride, 4);
    qw_batch_dispatch(b, (nbmax + 255) / 256, (T + 7) / 8, 1, 256, 1, 1);
    }
    const int need = (int)((nbmax + 1023) / 1024), mi = need <= 8 ? 8 : need <= 16 ? 16 : 32;
    char tk[32];
    snprintf(tk, sizeof tk, "qsa_topk2b_m%d", mi);
    qw_batch_kernel(b, pf_need(e, tk));
    qw_batch_buf(b, 0, &e->pscore, 0);
    qw_batch_buf(b, 1, &e->psel, 0);
    qw_batch_buf(b, 2, &e->pnsel, 0);
    qw_batch_buf(b, 3, &e->pqa, 0);
    qw_batch_bytes(b, 4, &sstride, 4);
    qw_batch_dispatch(b, T, 1, 1, 1024, 1, 1);
    bool gated = false;
    if (!pf_attn_old()) {
        for (int rp = 0; rp < xrep("flash"); rp++) gated = pf_flash(e, b, koff, T, true); // 量測用：QW_XREP=flash:n
    } else {
    qw_batch_kernel(b, pf_need(e, "qsa_attn2b"));
    qw_batch_buf(b, 0, &e->paq, 0);
    qw_batch_buf(b, 1, &e->Kc, koff);
    qw_batch_buf(b, 2, &e->Vc, koff);
    qw_batch_buf(b, 3, &e->psel, 0);
    qw_batch_buf(b, 4, &e->pnsel, 0);
    qw_batch_buf(b, 5, &e->ppart, 0);
    qw_batch_buf(b, 6, &e->pqa, 0);
    const uint32_t nch = pf_nch();
    qw_batch_dispatch(b, nch, 2, T, 32 * 12, 1, 1);
    qw_batch_kernel(b, pf_need(e, "qsa_merge2b"));
    qw_batch_buf(b, 0, &e->ppart, 0);
    qw_batch_buf(b, 1, &e->pao, 0);
    qw_batch_bytes(b, 2, &nch, 4);
    qw_batch_dispatch(b, 24, T, 1, 256, 1, 1);
    }
    for (int rp = 0; rp < (gated ? 0 : xrep("agate")); rp++) { // 量測用：QW_XREP=agate:n
    qw_batch_kernel(b, pf_need(e, "kb_attn_gate"));
    qw_batch_buf(b, 0, &e->pao, 0);
    qw_batch_buf(b, 1, &e->pqg, 0);
    qw_batch_buf(b, 2, &e->pay, 0);
    qw_batch_bytes(b, 3, &T, 4);
    d1(b, T * 6144, 256);
    }
}

static void pf_attention(qw_engine *e, qw_batch *b, const qw_attn *A, int ai, uint32_t T, uint32_t pos0) {
    const size_t Lc = (size_t)e->kvcap;
    const size_t koff = (size_t)ai * Lc * 2 * 256 * 2, roff = (size_t)ai * Lc * 128 * 4, boff = (size_t)ai * (Lc / 4 + 1) * 128 * 2;
    pf_i8_share(e, &e->px); // INT8：px 只量化一次
    pf_attn_mm(e, b, &A->q, &e->px, &e->pqg, T);
    pf_attn_mm(e, b, &A->k, &e->px, &e->pkk, T);
    pf_attn_mm(e, b, &A->v, &e->px, &e->pvv, T);
    pf_attn_mm(e, b, &A->idx_qk, &e->px, &e->piqk, T);
    pf_i8_share(e, NULL);
    pf_rope_tab(e, b, pos0, T);
    for (int rp = 0; rp < xrep("aprep"); rp++) { // 量測用：QW_XREP=aprep:n
    qw_batch_kernel(b, pf_need(e, A->nf32 ? "p_attn_prep_f" : "p_attn_prep"));
    qw_batch_buf(b, 0, &e->pqg, 0);
    qw_batch_buf(b, 1, &e->pkk, 0);
    qw_batch_buf(b, 2, &e->pvv, 0);
    bind_t(e, b, 3, A->q_norm);
    bind_t(e, b, 4, A->k_norm);
    qw_batch_buf(b, 5, &e->paq, 0);
    qw_batch_buf(b, 6, &e->Kc, koff);
    qw_batch_buf(b, 7, &e->Vc, koff);
    qw_batch_bytes(b, 8, &pos0, 4);
    qw_batch_buf(b, 9, &e->prcs, 0);
    qw_batch_dispatch(b, 24 + 2, T, 1, 256, 1, 1);
    }
    // indexer：raw_k 與新完成的區塊 key（之後 decode 會用到）
    qw_batch_kernel(b, pf_need(e, "p_idx_rawk"));
    qw_batch_buf(b, 0, &e->piqk, 0);
    qw_batch_buf(b, 1, &e->rawk, roff);
    qw_batch_bytes(b, 2, &pos0, 4);
    qw_batch_dispatch(b, T, 1, 1, 128, 1, 1);
    const uint32_t nb0 = pos0 / 4, nb1 = (pos0 + T) / 4; // 區塊 nb 完成的條件：(nb + 1)·4 ≤ pos0 + T
    if (nb1 > nb0) {
        qw_batch_kernel(b, pf_need(e, A->nf32 ? "p_idx_blocks_f" : "p_idx_blocks"));
        qw_batch_buf(b, 0, &e->rawk, roff);
        bind_t(e, b, 1, A->idx_k_norm);
        qw_batch_buf(b, 2, &e->kblk, boff);
        qw_batch_bytes(b, 3, &nb0, 4);
        qw_batch_dispatch(b, nb1 - nb0, 1, 1, 128, 1, 1);
    }
    if (pos0 + T > QW_IDX_BUDGET && !getenv("QW_PF_DENSE")) { // 有 query 的 context 超過 2048：QSA 稀疏
        pf_qsa_sparse(e, b, A, ai, T, pos0);
        pf_attn_mm(e, b, &A->o, &e->pay, &e->pout, T);
        return;
    }
    if (!pf_attn_old()) { // context ≤ 預算：全部都看（sparse = 0）
        qsa_args_t *qa = e->pqa.ptr;
        for (uint32_t t = 0; t < T; t++) {
            const uint32_t pos = pos0 + t;
            qa[t] = (qsa_args_t){pos + 1, pos, (pos + 1) / QW_IDX_CR, pos + 1, 0};
        }
        bool gated = false;
        for (int rp = 0; rp < xrep("flash"); rp++) gated = pf_flash(e, b, koff, T, true); // 量測用：QW_XREP=flash:n
        if (!gated) {
            qw_batch_kernel(b, pf_need(e, "kb_attn_gate"));
            qw_batch_buf(b, 0, &e->pao, 0);
            qw_batch_buf(b, 1, &e->pqg, 0);
            qw_batch_buf(b, 2, &e->pay, 0);
            qw_batch_bytes(b, 3, &T, 4);
            d1(b, T * 6144, 256);
        }
        pf_attn_mm(e, b, &A->o, &e->pay, &e->pout, T);
        return;
    }
    qw_batch_kernel(b, pf_need(e, "p_attn_dense"));
    qw_batch_buf(b, 0, &e->paq, 0);
    qw_batch_buf(b, 1, &e->Kc, koff);
    qw_batch_buf(b, 2, &e->Vc, koff);
    qw_batch_buf(b, 3, &e->pqg, 0);
    qw_batch_buf(b, 4, &e->pay, 0);
    qw_batch_bytes(b, 5, &pos0, 4);
    qw_batch_dispatch(b, T, 2, 1, 32 * 12, 1, 1);
    pf_attn_mm(e, b, &A->o, &e->pay, &e->pout, T);
}

// 區塊表（kernels/moe_prefill.metal 的 moe_plan_t）：每個 expert 的列切成 tm 列一塊
static void moe_plan(qw_engine *e, qw_batch *b, int tm) {
    e->ptm = (uint32_t)tm;
    e->ntiles = (uint32_t)(((size_t)PF * QW_TOPK + tm - 1) / tm + QW_EXPERTS);
    if (e->ntiles > e->max_tiles) { fprintf(stderr, "moe_plan：區塊列數 %d 太小\n", tm); exit(1); }
    qw_batch_kernel(b, pf_need(e, "moe_plan_t"));
    qw_batch_buf(b, 0, &e->pcnt, 0);
    qw_batch_buf(b, 1, &e->poffs, 0);
    qw_batch_buf(b, 2, &e->pcur, 0);
    qw_batch_buf(b, 3, &e->ptiles, 0);
    qw_batch_bytes(b, 4, &e->ntiles, 4);
    qw_batch_bytes(b, 5, &e->ptm, 4);
    qw_batch_dispatch(b, 1, 1, 1, QW_EXPERTS, 1, 1);
}

static void pf_moe(qw_engine *e, qw_batch *b, const qw_moe *M, uint32_t T) {
    const uint32_t P = T * QW_TOPK;
    for (int rp = 0; rp < xrep("router"); rp++)
    if (getenv("QW_PF_ROUTER_QMM")) pf_qmm(e, b, &M->router, &e->px, &e->plog, T, true);
    else if (!getenv("QW_ROUTER_EXACT")) { // 兩階段：tensor 版近似 → 前 16 名用精確公式重算（結果同精確版）
        if (!e->plog2.ptr && !alloc(e, &e->plog2, (size_t)PF * QW_EXPERTS * 4)) { fprintf(stderr, "router buffer 配置失敗\n"); exit(1); }
        pf_qmm(e, b, &M->router, &e->px, &e->plog2, T, true);
        if (M->router.gq) { // BF16 router
            qw_batch_kernel(b, gg_kernel(e, "p_router_bf_refine", M, 1));
            bind_t(e, b, 0, M->router.w);
            qw_batch_buf(b, 1, &e->px, 0);
            qw_batch_buf(b, 2, &e->plog, 0);
            qw_batch_buf(b, 3, &e->plog2, 0);
            qw_batch_dispatch(b, T, 1, 1, 32, 1, 1);
        } else {
        qw_batch_kernel(b, pf_need(e, "p_router_refine"));
        bind_t(e, b, 0, M->router.w);
        bind_t(e, b, 1, M->router.s);
        bind_t(e, b, 2, M->router.b);
        qw_batch_buf(b, 3, &e->px, 0);
        qw_batch_buf(b, 4, &e->plog, 0);
        qw_batch_buf(b, 5, &e->plog2, 0);
        qw_batch_dispatch(b, T, 1, 1, 32, 1, 1);
        }
    } else if (M->router.gq) { // 精確的 FP32 router，BF16 權重
        qw_batch_kernel(b, gg_kernel(e, "p_router_bf", M, 1));
        bind_t(e, b, 0, M->router.w);
        qw_batch_buf(b, 1, &e->px, 0);
        qw_batch_buf(b, 2, &e->plog, 0);
        qw_batch_bytes(b, 3, &T, 4);
        qw_batch_dispatch(b, QW_EXPERTS / 8, (T + 7) / 8, 1, 256, 1, 1);
    } else { // 精確的 FP32 router（避免接近平手的 expert 排名對調）
        qw_batch_kernel(b, pf_need(e, "p_router"));
        bind_t(e, b, 0, M->router.w);
        bind_t(e, b, 1, M->router.s);
        bind_t(e, b, 2, M->router.b);
        qw_batch_buf(b, 3, &e->px, 0);
        qw_batch_buf(b, 4, &e->plog, 0);
        qw_batch_bytes(b, 5, &T, 4);
        qw_batch_dispatch(b, QW_EXPERTS / 8, (T + 7) / 8, 1, 256, 1, 1);
    }
    qw_batch_kernel(b, pf_need(e, "p_dot_bf16"));
    bind_t(e, b, 0, M->sh_gate_w);
    qw_batch_buf(b, 1, &e->px, 0);
    qw_batch_buf(b, 2, &e->psgl, 0);
    qw_batch_dispatch(b, T, 1, 1, 256, 1, 1);
    qw_batch_kernel(b, pf_need(e, "p_moe_select"));
    qw_batch_buf(b, 0, &e->plog, 0);
    qw_batch_buf(b, 1, &e->psgl, 0);
    qw_batch_buf(b, 2, &e->pids, 0);
    qw_batch_buf(b, 3, &e->pwts, 0);
    qw_batch_buf(b, 4, &e->psg, 0);
    qw_batch_dispatch(b, T, 1, 1, 32, 1, 1);
    // 路由：計數（先清零）→ 規劃 → 分散
    {
        uint32_t ne = QW_EXPERTS; // 每層都要在 GPU 上清零計數器（CPU 編碼時清零的話，48 層會疊加）
        qw_batch_kernel(b, pf_need(e, "p_zero"));
        qw_batch_buf(b, 0, &e->pcnt, 0);
        qw_batch_bytes(b, 1, &ne, 4);
        d1(b, ne, 256);
    }
    qw_batch_kernel(b, pf_need(e, "moe_count"));
    qw_batch_buf(b, 0, &e->pids, 0);
    qw_batch_buf(b, 1, &e->pcnt, 0);
    qw_batch_bytes(b, 2, &P, 4);
    d1(b, P, 256);
    const bool i8 = i8moe_use(M);
    moe_plan(e, b, M->giq || i8 ? moe_gg_tm() : moe_tm());
    for (int rp = 0; rp < xrep("scatter"); rp++) { // 量測用：QW_XREP=scatter:n（重複時 pcur 會繼續累加，結果不對，只看時間）
    if (i8) {
        pf_i8_share(e, &e->px); // px 量化一次：scatter 與 shared gate / up 共用（到 shared expert 之後才結束）
        if (!pf_scatter_i8u(e, b, M, T, P)) pf_scatter_i8(e, b, M, P);
        continue;
    }
    qw_batch_kernel(b, pf_need(e, "moe_scatter"));
    qw_batch_buf(b, 0, &e->px, 0);
    qw_batch_buf(b, 1, &e->pids, 0);
    qw_batch_buf(b, 2, &e->pcur, 0);
    qw_batch_buf(b, 3, &e->pXs, 0);
    qw_batch_buf(b, 4, &e->ppos, 0);
    qw_batch_dispatch(b, P, 1, 1, 256, 1, 1);
    }
    // routed experts
    const bool gu = !M->giq && !a8_on() && M->gate.bits == 4 && getenv("QW_GU") && !i8; // 實驗：合併版目前比較慢（threadgroup memory 加倍）
    for (int rp = 0; rp < xrep("gateup"); rp++) {
    if (gu) pf_qmm_moe_gu(e, b, M, P); // gate + up + SiLU 一次算完，直接寫 pH
    else if (i8) pf_moe_i8_gu(e, b, M, P);
    else if (M->giq) {
        pf_qmm_moe_iq(e, b, M->giq, M->fgu, QW_MOE_INTER, D, &e->pXs, &e->pGg, P);
        pf_qmm_moe_iq(e, b, M->uiq, M->fgu, QW_MOE_INTER, D, &e->pXs, &e->pGu, P);
    } else if (a8_on() && M->gate.bits == 4) pf_moe_gateup_a8(e, b, M, P);
    else {
        pf_qmm_moe(e, b, &M->gate, &e->pXs, &e->pGg, P);
        pf_qmm_moe(e, b, &M->up, &e->pXs, &e->pGu, P);
    }
    }
    uint32_t n = P * QW_MOE_INTER;
    if (!gu) for (int rp = 0; rp < xrep("act"); rp++) {
    if (i8) { pf_act_i8(e, b, M, P); continue; }
    qw_batch_kernel(b, pf_need(e, "p_act2"));
    qw_batch_buf(b, 0, &e->pGg, 0);
    qw_batch_buf(b, 1, &e->pGu, 0);
    qw_batch_buf(b, 2, &e->pH, 0);
    qw_batch_bytes(b, 3, &n, 4);
    d1(b, n, 256);
    }
    for (int rp = 0; rp < xrep("down"); rp++) if (i8) pf_moe_i8_dn(e, b, M, P);
    else if (M->diq) pf_qmm_moe_iq(e, b, M->diq, M->fd, D, QW_MOE_INTER, &e->pH, &e->pDm, P); else pf_qmm_moe(e, b, &M->down, &e->pH, &e->pDm, P);
    // 共用 expert
    uint32_t n2 = T * QW_SHARED_INTER;
    for (int rp = 0; rp < xrep("shared"); rp++) {
    if (e->i8q_src != &e->px) pf_i8_share(e, &e->px); // INT8：px 只量化一次（routed scatter 已量化的話直接用）
    pf_qmm(e, b, &M->sh_gate, &e->px, &e->pSg, T, false);
    pf_qmm(e, b, &M->sh_up, &e->px, &e->pSu, T, false);
    qw_batch_kernel(b, pf_need(e, "p_act2"));
    qw_batch_buf(b, 0, &e->pSg, 0);
    qw_batch_buf(b, 1, &e->pSu, 0);
    qw_batch_buf(b, 2, &e->pSH, 0);
    qw_batch_bytes(b, 3, &n2, 4);
    d1(b, n2, 256);
    pf_qmm(e, b, &M->sh_down, &e->pSH, &e->pDs, T, false);
    }
    pf_i8_share(e, NULL);
    for (int rp = 0; rp < xrep("combine"); rp++) {
    qw_batch_kernel(b, pf_need(e, "p_moe_combine"));
    qw_batch_buf(b, 0, &e->pDm, 0);
    qw_batch_buf(b, 1, &e->ppos, 0);
    qw_batch_buf(b, 2, &e->pwts, 0);
    qw_batch_buf(b, 3, &e->pDs, 0);
    qw_batch_buf(b, 4, &e->psg, 0);
    qw_batch_buf(b, 5, &e->pout, 0);
    qw_batch_dispatch(b, D / 4 / 64, T, 1, 64, 1, 1);
    }
}

static void pf_hcnorm_f(qw_engine *e, qw_batch *b, const qw_buf *x, qw_t w, const qw_buf *y, uint32_t T) {
    qw_batch_kernel(b, pf_need(e, "p_hcnorm_f"));
    qw_batch_buf(b, 0, x, 0);
    bind_t(e, b, 1, w);
    qw_batch_buf(b, 2, y, 0);
    qw_batch_dispatch(b, QW_HC, T, 1, 1024, 1, 1);
}

// PLE 的後半（prefill）：h += gated + silu(conv(cn))，conv 狀態留最後 9 個 token（含段中檢查點）
static void pf_ple_tail(qw_engine *e, qw_batch *b, const qw_ple *P, uint32_t T) {
    qw_batch_kernel(b, pf_need(e, "p_ple_conv"));
    qw_batch_buf(b, 0, &e->ph, 0);
    qw_batch_buf(b, 1, &e->pgated, 0);
    qw_batch_buf(b, 2, &e->pcn, 0);
    bind_t(e, b, 3, P->conv1d);
    qw_batch_buf(b, 4, &e->ple_st, 0);
    qw_batch_dispatch(b, HCD / 256, T, 1, 256, 1, 1);
    qw_batch_kernel(b, pf_need(e, "p_ple_state"));
    qw_batch_buf(b, 0, &e->pcn, 0);
    qw_batch_buf(b, 1, &e->ple_st, 0);
    qw_batch_bytes(b, 2, &T, 4);
    d1(b, HCD, 256);
    for (int k = 0; k < e->seg_n; k++) {
        qw_batch_kernel(b, pf_need(e, "p_ple_state"));
        qw_batch_buf(b, 0, &e->pcn, 0);
        qw_batch_buf(b, 1, &e->snPle, (size_t)k * e->ple_st.size);
        qw_batch_bytes(b, 2, &e->seg_q[k], 4);
        d1(b, HCD, 256);
    }
}

// GGUF 的 dense prefill（同 pf_qmm_gg 的 kernel），X / Y 可指定位移、輸出 FP32
static void ple_qmm_gg(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *X, size_t xoff, const qw_buf *Y, size_t yoff, uint32_t M) {
    char name[96], defs[1024], err[8192];
    // 窄 tile（16 欄）只給真的很窄的矩陣（DeltaNet 的 a / b 等）；N ≥ 512（shared expert 640、router 512）用 64 欄，否則 threadgroup 內的 tensor 運算太小
    const int wide = getenv("QW_PFGG_WIDE") ? atoi(getenv("QW_PFGG_WIDE")) : 512;
    const int f = L->gq - 1, tn = (L->out % 64 == 0 && L->out >= wide) ? 64 : 16, sg = tn == 64 ? 4 : 2, tm = M >= 128 ? 128 : 64;
    snprintf(name, sizeof name, "qmm_gg%d_k%d_%dx%dx%d_f", f, L->in, tm, tn, sg);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) {
        snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY\n#define qmm %s\n#define BITS 8\n#define KD %d\n#define TM %d\n#define TN %d\n#define SG %d\n"
                 "#define KC 64\n#define MOE 0\n#define IQF %d\n#define IQRB %d\n#define YT float\n", name, L->in, tm, tn, sg, f, gg_rb(L));
        if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_qmm, sizeof src_qmm, err, sizeof err)) {
            fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
            exit(1);
        }
        k = qw_kernel_get(e->g, name);
    }
    qmm_args a = {M, (uint32_t)L->out};
    qw_batch_kernel(b, k);
    qw_batch_buf(b, 0, X, xoff);
    bind_t(e, b, 1, L->w);
    bind_t(e, b, 2, L->w); // S、Z 用不到
    bind_t(e, b, 3, L->w);
    qw_batch_buf(b, 4, Y, yoff);
    qw_batch_bytes(b, 5, &a, sizeof a);
    qw_batch_dispatch(b, (unsigned)L->out / tn, (M + tm - 1) / tm, 1, 32u * sg, 1, 1);
}

// PLE（prefill，GGUF）：n-gram 列 CPU 解碼成 ngX [T][2][D]（hi、lo），key / value 投影把 2T 列一起乘再兩兩相加（FP32）。
// 投影的 FP32 暫存借用 pkn / pqn（[PF][HCD] FP32）：每次最多 PF / 2 個 token（2 倍列數）
static void pf_ple_gg(qw_engine *e, qw_batch *b, const qw_ple *P, uint32_t T, uint32_t pos0) {
    ng_gather_pf(e, P, (int)(pos0 + 2), (int)T);
    const uint32_t C = (uint32_t)PF / 2;
    for (uint32_t c0 = 0; c0 < T; c0 += C) {
        const uint32_t m = T - c0 < C ? T - c0 : C;
        const size_t xo = (size_t)c0 * 2 * D * 2;
        ple_qmm_gg(e, b, &P->key_proj, &e->ngX, xo, &e->pkn, 0, 2 * m);
        gp_pair_sum(e, b, &e->pkn, &e->pkeyf, (size_t)c0 * HCD * 4, HCD, m);
        ple_qmm_gg(e, b, &P->value_proj, &e->ngX, xo, &e->pqn, 0, 2 * m);
        gp_pair_sum(e, b, &e->pqn, &e->pvalf, (size_t)c0 * D * 4, D, m);
    }
    gp_hcnorm(e, b, &e->pkeyf, P->norm_key, &e->pkn, T, true);
    gp_hcnorm(e, b, &e->ph, P->norm_query, &e->pqn, T, true);
    gp_gate(e, b, &e->pkn, &e->pqn, &e->pvalf, &e->pgated, T);
    gp_hcnorm(e, b, &e->pgated, P->norm_conv, &e->pcn, T, true);
    pf_ple_tail(e, b, P, T);
}

static void pf_ple(qw_engine *e, qw_batch *b, const qw_ple *P, uint32_t T, uint32_t pos0) {
    if (e->m->ple_tab) { pf_ple_gg(e, b, P, T, pos0); return; }
    uint32_t p0 = pos0 + 2, eos = EOS_ID;
    qw_batch_kernel(b, pf_need(e, "p_ngram"));
    qw_batch_buf(b, 0, &e->tok, 0);
    qw_batch_bytes(b, 1, &p0, 4);
    bind_t(e, b, 2, P->multipliers);
    bind_t(e, b, 3, P->head_offsets);
    bind_t(e, b, 4, P->head_vocab);
    ng_gather_pf(e, P, (int)p0, (int)T);
    ng_bind(e, b, 0);
    qw_batch_buf(b, 8, &e->pemb, 0);
    qw_batch_bytes(b, 9, &eos, 4);
    qw_batch_dispatch(b, 16, T, 1, 160, 1, 1);
    pf_qmm(e, b, &P->key_proj, &e->pemb, &e->pkeyh, T, false);
    pf_qmm(e, b, &P->value_proj, &e->pemb, &e->pval, T, false);
    uint32_t n = T * HCD;
    qw_batch_kernel(b, pf_need(e, "p_h2f"));
    qw_batch_buf(b, 0, &e->pkeyh, 0);
    qw_batch_buf(b, 1, &e->pkeyf, 0);
    qw_batch_bytes(b, 2, &n, 4);
    d1(b, n, 256);
    pf_hcnorm_f(e, b, &e->pkeyf, P->norm_key, &e->pkn, T);
    pf_hcnorm_f(e, b, &e->ph, P->norm_query, &e->pqn, T);
    qw_batch_kernel(b, pf_need(e, "p_ple_gate"));
    qw_batch_buf(b, 0, &e->pkn, 0);
    qw_batch_buf(b, 1, &e->pqn, 0);
    qw_batch_buf(b, 2, &e->pval, 0);
    qw_batch_buf(b, 3, &e->pgated, 0);
    qw_batch_dispatch(b, QW_HC, T, 1, 1024, 1, 1);
    pf_hcnorm_f(e, b, &e->pgated, P->norm_conv, &e->pcn, T);
    pf_ple_tail(e, b, P, T);
}

// 一段（T ≤ PF 個 token、context ≤ 預算）的 prefill；算完把最後一個 token 的 logits 放進 e->logits

// MTP 的 prompt 部分：位置 pos0 … pos0+T−1 各自的「下一個 token」已在 tok[pos + 3]，主模型 mixer 前的殘差流在 ph。
// 只需要把 MTP 層的 attention KV（含 indexer 的 raw_k / 區塊 key）寫進第 NATT 份 cache，輸出用不到。
static void pf_mtp_fill(qw_engine *e, qw_batch *b, uint32_t T, uint32_t pos0) {
    const qw_model *M = e->m;
    uint32_t p3 = pos0 + 3, n1 = D, n2 = HCD, hcd = HCD, d = D;
    qw_batch_kernel(b, pf_need(e, "p_embed"));
    bind_t(e, b, 0, M->embed_w);
    bind_t(e, b, 1, M->embed_s);
    bind_t(e, b, 2, M->embed_b);
    qw_batch_buf(b, 3, &e->tok, 0);
    qw_batch_bytes(b, 4, &p3, 4);
    qw_batch_buf(b, 5, &e->pme, 0);
    qw_batch_dispatch(b, D / 256, T, 1, 256, 1, 1);
    qw_batch_kernel(b, pf_need(e, "k_norm_rows")); // 每個 token 只取第 0 條流（4 條相同）
    qw_batch_buf(b, 0, &e->pme, 0);
    bind_t(e, b, 1, M->mtp_norm_emb);
    qw_batch_buf(b, 2, &e->pmn, 0);
    qw_batch_bytes(b, 3, &n1, 4);
    qw_batch_bytes(b, 4, &hcd, 4);
    qw_batch_bytes(b, 5, &d, 4);
    qw_batch_dispatch(b, T, 1, 1, 1024, 1, 1);
    pf_qmm(e, b, &M->mtp_fc_emb, &e->pmn, &e->pmfe, T, false);
    qw_batch_kernel(b, pf_need(e, "k_norm_rows")); // 4 條流一起正規化
    qw_batch_buf(b, 0, &e->ph, 0);
    bind_t(e, b, 1, M->mtp_norm_hidden);
    qw_batch_buf(b, 2, &e->pmnh, 0);
    qw_batch_bytes(b, 3, &n2, 4);
    qw_batch_bytes(b, 4, &hcd, 4);
    qw_batch_bytes(b, 5, &hcd, 4);
    qw_batch_dispatch(b, T, 1, 1, 1024, 1, 1);
    pf_qmm(e, b, &M->mtp_fc_hidden, &e->pmnh, &e->pmfh, T * QW_HC, false); // [T·4][D]
    qw_batch_kernel(b, pf_need(e, "p_mtp_combine"));
    qw_batch_buf(b, 0, &e->pmfh, 0);
    qw_batch_buf(b, 1, &e->pmfe, 0);
    qw_batch_buf(b, 2, &e->pmh, 0);
    qw_batch_bytes(b, 3, &T, 4);
    d1(b, T * HCD, 256);
    (void)n1;
    qw_buf hs = e->ph;
    e->ph = e->pmh;
    e->ppending = false;
    pf_hyper(e, b, &M->mtp.attn_hc, T);
    // 只寫 KV：k、v、indexer 投影 + 準備（q 不需要，p_attn_prep 的 q 輸出忽略）
    const qw_attn *A = &M->mtp.att;
    const size_t Lc = (size_t)e->kvcap;
    const size_t koff = (size_t)NATT * Lc * 2 * 256 * 2, roff = (size_t)NATT * Lc * 128 * 4, boff = (size_t)NATT * (Lc / 4 + 1) * 128 * 2;
    pf_qmm(e, b, &A->k, &e->px, &e->pkk, T, false);
    pf_qmm(e, b, &A->v, &e->px, &e->pvv, T, false);
    pf_qmm(e, b, &A->idx_qk, &e->px, &e->piqk, T, false);
    pf_rope_tab(e, b, pos0, T);
    qw_batch_kernel(b, pf_need(e, A->nf32 ? "p_attn_prep_f" : "p_attn_prep"));
    qw_batch_buf(b, 0, &e->pqg, 0);
    qw_batch_buf(b, 1, &e->pkk, 0);
    qw_batch_buf(b, 2, &e->pvv, 0);
    bind_t(e, b, 3, A->q_norm);
    bind_t(e, b, 4, A->k_norm);
    qw_batch_buf(b, 5, &e->paq, 0);
    qw_batch_buf(b, 6, &e->Kc, koff);
    qw_batch_buf(b, 7, &e->Vc, koff);
    qw_batch_bytes(b, 8, &pos0, 4);
    qw_batch_buf(b, 9, &e->prcs, 0);
    qw_batch_dispatch(b, 24 + 2, T, 1, 256, 1, 1);
    qw_batch_kernel(b, pf_need(e, "p_idx_rawk"));
    qw_batch_buf(b, 0, &e->piqk, 0);
    qw_batch_buf(b, 1, &e->rawk, roff);
    qw_batch_bytes(b, 2, &pos0, 4);
    qw_batch_dispatch(b, T, 1, 1, 128, 1, 1);
    const uint32_t nb0 = pos0 / 4, nb1 = (pos0 + T) / 4;
    if (nb1 > nb0) {
        qw_batch_kernel(b, pf_need(e, A->nf32 ? "p_idx_blocks_f" : "p_idx_blocks"));
        qw_batch_buf(b, 0, &e->rawk, roff);
        bind_t(e, b, 1, A->idx_k_norm);
        qw_batch_buf(b, 2, &e->kblk, boff);
        qw_batch_bytes(b, 3, &nb0, 4);
        qw_batch_dispatch(b, nb1 - nb0, 1, 1, 128, 1, 1);
    }
    e->ph = hs;
}

// profile 結果依 kernel 名稱加總，印到 stderr（QW_*_PROF 用）
static void prof_dump(qw_engine *e, const char *title) {
    const char *nm[160];
    double us[160] = {0}, tot = 0;
    int cnt[160] = {0}, na = 0;
    for (int i = 0; i < qw_profile_count(e->g); i++) {
        double u;
        const char *s = qw_profile_get(e->g, i, &u);
        int j = 0;
        while (j < na && strcmp(nm[j], s)) j++;
        if (j == na) { if (na == 160) continue; nm[na++] = s; }
        us[j] += u, cnt[j]++, tot += u;
    }
    for (int i = 0; i < na; i++)
        for (int j = i + 1; j < na; j++)
            if (us[j] > us[i]) {
                const char *tn = nm[i]; nm[i] = nm[j]; nm[j] = tn;
                double tu = us[i]; us[i] = us[j]; us[j] = tu;
                int tc = cnt[i]; cnt[i] = cnt[j]; cnt[j] = tc;
            }
    fprintf(stderr, "%s：%d 個 dispatch，合計 %.2f ms\n", title, qw_profile_count(e->g), tot / 1e3);
    for (int i = 0; i < na; i++) fprintf(stderr, "  %-34s %5d  %8.3f ms  %5.1f%%\n", nm[i], cnt[i], us[i] / 1e3, us[i] / tot * 100);
}

static bool mix_on(const qw_engine *e);              // mixed step（qw_engine_par_mix）：這段 prefill 跑的同時讓其他對話 decode
static bool mix_busy;                                // mixed step 的 decode 迴圈中（不能再開始 prefill：p* 暫存正在用）
static int mix_sid = -1;                             // mixed step 中正在 prefill 的 slot（decode 不能碰）
static double mix_wait(qw_engine *e, qw_batch *b); // 已 submit 的 b 跑完之前反覆 decode；回傳 b 的 GPU 秒數
static double pf_segment(qw_engine *e, const int32_t *ids, uint32_t T, float *dump) {
    if (mix_busy) { fprintf(stderr, "mixed step：decode 迴圈裡不能 prefill\n"); return -1; }
    const double tseg0 = qw_now();
    const uint32_t pos0 = (uint32_t)e->pos;
    e->seg_exact = e->st_exact == (int)pos0 && pos0 % (uint32_t)gdn_chunk() == 0;
    const bool prof = getenv("QW_PF_PROF") && (int)pos0 + (int)T >= atoi(getenv("QW_PF_PROF"));
    if (prof) qw_profile_enable(e->g, true);
    int32_t *tk = e->tok.ptr;
    for (uint32_t t = 0; t < T; t++) tk[pos0 + 2 + t] = ids[t];
    e->seg_n = 0;
    for (int k = 0; k < e->npf_snap && e->seg_n < 4; k++) {
        const int q = e->pf_snap[k] - (int)pos0;
        if (q > 0 && q < (int)T && q % gdn_chunk() == 0) e->seg_q[e->seg_n++] = (uint32_t)q;
    }
    if (e->seg_n && !e->snS.ptr &&
        !(alloc(e, &e->snS, 4 * e->S.size) && alloc(e, &e->snConv, 4 * e->conv_st.size) && alloc(e, &e->snPle, 4 * e->ple_st.size)))
        e->seg_n = 0;
    qw_batch *b = qw_batch_begin(e->g, false);
    uint32_t p0 = pos0 + 2;
    if (e->m->embed_gq) gg_embed_rows(e, b, &e->tok, p0, T, QW_HC, HCD, &e->ph);
    else {
    qw_batch_kernel(b, pf_need(e, "p_embed"));
    bind_t(e, b, 0, e->m->embed_w);
    bind_t(e, b, 1, e->m->embed_s);
    bind_t(e, b, 2, e->m->embed_b);
    qw_batch_buf(b, 3, &e->tok, 0);
    qw_batch_bytes(b, 4, &p0, 4);
    qw_batch_buf(b, 5, &e->ph, 0);
    qw_batch_dispatch(b, D / 256, T, 1, 256, 1, 1);
    }
    e->ppending = false;
    int li = 0, ai = 0;
    const int stop = getenv("QW_STOP_LAYER") ? atoi(getenv("QW_STOP_LAYER")) : QW_LAYERS;
    for (int l = 0; l < stop; l++) {
        const qw_layer *L = &e->m->layers[l];
        if (L->has_ple && !skip("ple")) {
            if (e->ppending) { pf_resid(e, b, T); e->ppending = false; }
            pf_ple(e, b, &L->ple, T, pos0);
        }
        pf_hyper(e, b, &L->attn_hc, T);
        if (!getenv("QW_DF_NOTAP_PF")) df_tap_after(e, b, l, &e->ph, (int)T);
        if (L->full_attn) pf_attention(e, b, &L->att, ai++, T, pos0);
        else pf_deltanet(e, b, &L->lin, li++, T);
        e->ppending = true;
        pf_hyper(e, b, &L->mlp_hc, T);
        pf_moe(e, b, &L->moe, T);
        e->ppending = true;
        if (dump) {
            pf_resid(e, b, T);
            e->ppending = false;
            uint32_t n = T * HCD;
            qw_batch_kernel(b, e->k_copyf);
            qw_batch_buf(b, 0, &e->ph, 0);
            qw_batch_buf(b, 1, &e->dump, (size_t)l * PF * HCD * 4);
            qw_batch_bytes(b, 2, &n, 4);
            d1(b, n, 256);
        }
    }
    if (e->ppending) { pf_resid(e, b, T); e->ppending = false; }
    if (e->pf_mtp && e->mtp_T > 0) pf_mtp_fill(e, b, (uint32_t)e->mtp_T, pos0);
    if (e->df_tapping) { // DFlash2 的上下文：只需要 prompt 最後一個視窗內的位置
        const int first = (int)pos0 > e->df_ctx_from ? (int)pos0 : e->df_ctx_from;
        if (first < (int)(pos0 + T)) df_ctx(e, b, first - (int)pos0, (int)(pos0 + T) - first, first);
    }
    // 最後一個 token：複製到 decode 的 h，跑 mixer + lm_head
    uint32_t n = HCD;
    qw_batch_kernel(b, e->k_copyf);
    qw_batch_buf(b, 0, &e->ph, (size_t)(T - 1) * HCD * 4);
    qw_batch_buf(b, 1, &e->h, 0);
    qw_batch_bytes(b, 2, &n, 4);
    d1(b, n, 256);
    e->pending = false;
    hyper_in(e, b, &e->m->mixer);
    qmv(e, b, &e->m->lm_head, &e->x, 0, &e->logits, true);
    const double te0 = qw_now();
    double t;
    const qw_ple *Pn = NULL;
    if (e->pf_nx_T > 0 && !prof && !skip("ple"))
        for (int l = 0; l < stop && !Pn; l++) if (e->m->layers[l].has_ple) Pn = &e->m->layers[l].ple;
    const bool mix = !prof && mix_on(e);
    if (Pn) { // 下一段的 n-gram 列在 GPU 跑這段時抄（列在 32 GB 的 mmap 表裡，冷的時候一段要 1–2 秒）
        qw_batch_submit(b);
        int32_t *tkn = e->tok.ptr; // 下一段的 token 先放進 tok（這段的 kernel 只讀到 pos0 + T + 2 為止，該位置的值相同）
        for (int t2 = 0; t2 < e->pf_nx_T; t2++) tkn[pos0 + T + 2 + t2] = e->pf_nx_ids[t2];
        ng_prefetch(e, Pn, (int)(pos0 + T + 2), e->pf_nx_T);
        if (mix) ng_prefetch_wait(e); // 背景抄列時才讀 e->tok / e->ngX：mixed step 換 slot、換暫存之前先等它抄完
        t = mix ? mix_wait(e, b) : qw_batch_wait(b);
        ng_prefetch_wait(e);
    } else if (mix) {
        qw_batch_submit(b);
        t = mix_wait(e, b);
    } else t = qw_batch_end(b);
    dn_tap_flush(e);
    if (getenv("QW_CACHE_DBG")) fprintf(stderr, "[t] pf_segment T=%u：編碼 %.1f ms、等待 %.1f ms、GPU %.1f ms\n", T, (te0 - tseg0) * 1e3, (qw_now() - te0) * 1e3, t * 1e3);
    if (prof) {
        char ti[64];
        snprintf(ti, sizeof ti, "prefill 段 pos %u + %u", pos0, T);
        prof_dump(e, ti);
        qw_profile_enable(e->g, false);
    }
    e->pos += (int)T;
    e->st_exact = e->seg_exact ? e->pos : -1;
    for (uint32_t i = 0; i < T; i++) e->hist[pos0 + 2 + i] = ids[i];
    if (getenv("QW_UNION_CHECK") && pos0 + T > QW_IDX_BUDGET && e->pul.ptr) { // 除錯：GPU 的區塊聯集 vs CPU 重算（最後一層）
        const uint32_t *sel = e->psel.ptr, *nsel = e->pnsel.ptr, *ul = e->pul.ptr, *uc = e->puc.ptr;
        const uint8_t *um = e->pum.ptr;
        const qsa_args_t *qa = e->pqa.ptr;
        int bad = 0;
        for (uint32_t grp = 0; grp < (T + 3) / 4 && bad < 5; grp++) {
            static uint8_t mk[1 << 16];
            memset(mk, 0, sizeof mk);
            for (int qq = 0; qq < 4; qq++) {
                const uint32_t t = grp * 4 + qq;
                if (t >= T) break;
                if (qa[t].sparse) {
                    for (uint32_t i = 0; i < nsel[t]; i++) mk[sel[t * 512 + i]] |= 1 << qq;
                    if ((qa[t].pos + 1) % 4) mk[qa[t].nvis] |= 1 << qq;
                } else for (uint32_t bb = 0; bb <= qa[t].pos / 4; bb++) mk[bb] |= 1 << qq;
            }
            uint32_t n = 0;
            for (int bb = 0; bb < (1 << 16); bb++) if (mk[bb]) {
                if (n < uc[grp] && (ul[grp * 2064 + n] != (uint32_t)bb || um[grp * 2064 + n] != mk[bb])) {
                    if (bad++ < 5) fprintf(stderr, "[union] 組 %u 第 %u 個：GPU %u/%x，CPU %d/%x\n", grp, n, ul[grp * 2064 + n], um[grp * 2064 + n], bb, mk[bb]);
                }
                n++;
            }
            if (n != uc[grp] && bad++ < 5) fprintf(stderr, "[union] 組 %u：GPU %u 個，CPU %u 個\n", grp, uc[grp], n);
        }
        if (!bad) fprintf(stderr, "[union] 一致（%u 組）\n", (T + 3) / 4);
        uint32_t mn = 1u << 30, mx = 0, nshort = 0;
        for (uint32_t t = 0; t < T; t++) if (qa[t].sparse) {
            const uint32_t want = qa[t].nvis < 512 ? qa[t].nvis : 512;
            mn = nsel[t] < mn ? nsel[t] : mn; mx = nsel[t] > mx ? nsel[t] : mx;
            if (nsel[t] != want) nshort++;
        }
        fprintf(stderr, "[union] nsel 範圍 %u..%u，與預估不同的 query：%u\n", mn, mx, nshort);
    }
    if (getenv("QW_SEL_STATS") && pos0 + T > QW_IDX_BUDGET) { // 除錯：相鄰 query 選中區塊的重疊（最後一個 attention 層）
        const uint32_t *sel = e->psel.ptr;
        for (int W = 2; W <= 32; W *= 2) {
            double uni = 0;
            int cnt = 0;
            static uint8_t mark[1 << 16];
            for (uint32_t t0 = T > 512 ? T - 512 : 0; t0 + W <= T; t0 += W) {
                memset(mark, 0, sizeof mark);
                int u = 0;
                for (int j = 0; j < W; j++)
                    for (int i = 0; i < 512; i++) { const uint32_t bk = sel[(size_t)(t0 + j) * 512 + i]; if (!mark[bk & 0xffff]) { mark[bk & 0xffff] = 1; u++; } }
                uni += u;
                cnt++;
            }
            fprintf(stderr, "[sel] %d 個相鄰 query 的區塊聯集平均 %.0f（每個 512）\n", W, uni / cnt);
        }
    }
    for (int k = 0; k < e->seg_n; k++) snap_save_from(e, (int)(pos0 + e->seg_q[k]), k);
    if (e->npf_snap)
        for (int k = 0; k < e->npf_snap; k++)
            if (e->pf_snap[k] == e->pos) snap_save(e, e->pos); // 剛好在段尾
    e->seg_n = 0;
    if (getenv("QW_DUMP_PAO")) { // 除錯：最後一個 attention 層的輸出（gate 之前），[T][24][256] half
        FILE *f = fopen(getenv("QW_DUMP_PAO"), "wb");
        fwrite(e->pao.ptr, 2, (size_t)T * 6144, f);
        fclose(f);
    }
    if (getenv("QW_DUMP_GDN0")) { // 除錯：最後一層 DeltaNet 的輸入（pla、pbe 是最後寫入的那一層）
        FILE *f = fopen("build/gdn_la.bin", "wb"); fwrite(e->pla.ptr, 4, (size_t)T * 48, f); fclose(f);
        f = fopen("build/gdn_be.bin", "wb"); fwrite(e->pbe.ptr, 4, (size_t)T * 48, f); fclose(f);
        f = fopen(getenv("QW_PF_GDN_SEQ") ? "build/gdn_o_seq.bin" : "build/gdn_o_chunk.bin", "wb");
        fwrite(e->po.ptr, 2, (size_t)T * 6144, f);
        fclose(f);
    }
    return t;
}

// attention 核心（pf_flash）的 A/B：用 kbench 剛跑完的第一個 attention 層的 q、K/V、選取結果（pqa、psel），
// 各版本（QW_FLASH_KB=模式,模式,…，預設 0,1；見 pf_flash_variant）輪流跑 reps 次取最好，輸出與第一個版本比對。
// QW_SEL_DUMP=檔名：另存 T、pqa、pnsel、psel（離線分析選取的重疊）
// 量測用（QW_PF_KBENCH + QW_QK_KB）：indexer 打分 FP（p_qsa_score）vs INT8（qk_score_i8），用最後一段最後一個 attention 層的
// iq 與區塊 key。印出時間、分數的相對 RMS 差，以及每 16 個 query 抽一個的前 512 個區塊重疊率（選取真正會受影響的部分）
static int qk_cmpf(const void *a, const void *b) {
    const float x = *(const float *)a, y = *(const float *)b;
    return x < y ? 1 : x > y ? -1 : 0;
}
static void qk_kb(qw_engine *e, uint32_t T, uint32_t pos0, int reps) {
    const size_t Lc = (size_t)e->kvcap;
    const size_t boff = (size_t)(NATT - 1) * (Lc / 4 + 1) * 128 * 2;
    const uint32_t sstride = (uint32_t)(Lc / 4 + 1), nbmax = (pos0 + T) / 4;
    const size_t ns = (size_t)T * sstride;
    float *ref = malloc(ns * 4), *row = malloc((size_t)sstride * 4);
    double best[2] = {1e9, 1e9};
    for (int trial = 0; trial < 3; trial++)
        for (int m = 0; m < 2; m++) {
            memset(e->pscore.ptr, 0, ns * 4);
            qw_batch *b = qw_batch_begin(e->g, false);
            for (int r = 0; r < reps; r++) {
                if (m) pf_qk_score_i8(e, b, boff, T, pos0, sstride, nbmax);
                else {
                    qw_batch_kernel(b, pf_need(e, "p_qsa_score"));
                    qw_batch_buf(b, 0, &e->piq, 0);
                    qw_batch_buf(b, 1, &e->kblk, boff);
                    qw_batch_buf(b, 2, &e->pscore, 0);
                    qw_batch_bytes(b, 3, &pos0, 4);
                    qw_batch_bytes(b, 4, &T, 4);
                    qw_batch_bytes(b, 5, &sstride, 4);
                    qw_batch_dispatch(b, (nbmax + 255) / 256, (T + 7) / 8, 1, 256, 1, 1);
                }
            }
            const double t = qw_batch_end(b) * 1e3 / reps;
            if (t < best[m]) best[m] = t;
            if (trial) continue;
            if (!m) { memcpy(ref, e->pscore.ptr, ns * 4); continue; }
            const float *o = e->pscore.ptr;
            double s2 = 0, r2 = 0, ov = 0;
            int nov = 0;
            for (uint32_t q = 0; q < T; q++) {
                const uint32_t nv = (pos0 + q + 1) / 4;
                for (uint32_t n = 0; n < nv; n++) {
                    const double d = (double)o[(size_t)q * sstride + n] - ref[(size_t)q * sstride + n];
                    s2 += d * d;
                    r2 += (double)ref[(size_t)q * sstride + n] * ref[(size_t)q * sstride + n];
                }
                if (q % 16 || nv <= 512) continue;
                memcpy(row, ref + (size_t)q * sstride, nv * 4);
                qsort(row, nv, 4, qk_cmpf);
                const float tr = row[511]; // FP 的第 512 名
                memcpy(row, o + (size_t)q * sstride, nv * 4);
                qsort(row, nv, 4, qk_cmpf);
                const float ti = row[511];
                int c = 0, ci = 0;
                for (uint32_t n = 0; n < nv; n++) {
                    const bool a = ref[(size_t)q * sstride + n] >= tr, bb = o[(size_t)q * sstride + n] >= ti;
                    c += a && bb;
                    ci += a;
                }
                ov += (double)c / (ci ? ci : 1);
                nov++;
            }
            printf("  indexer 打分 INT8 vs FP：相對 RMS %.4g、前 512 區塊重疊率 %.4f（%d 個 query）\n", sqrt(s2 / (r2 > 0 ? r2 : 1)),
                   nov ? ov / nov : 1.0, nov);
        }
    printf("  indexer 打分：FP %.3f ms、INT8（含量化）%.3f ms／層（T %u、pos0 %u、區塊 %u）\n", best[0], best[1], T, pos0, nbmax);
    free(ref);
    free(row);
}

static void pf_flash_kb(qw_engine *e, uint32_t T, int reps) {
    if (getenv("QW_SEL_DUMP")) {
        FILE *f = fopen(getenv("QW_SEL_DUMP"), "wb");
        if (f) {
            fwrite(&T, 4, 1, f);
            fwrite(e->pqa.ptr, sizeof(qsa_args_t), T, f);
            fwrite(e->pnsel.ptr, 4, T, f);
            fwrite(e->psel.ptr, 4, (size_t)T * 512, f);
            fclose(f);
        }
    }
    int modes[8], nm = 0;
    {
        const char *s = getenv("QW_FLASH_KB");
        while (s && *s && nm < 8) {
            if (*s >= '0' && *s <= '9') { modes[nm++] = (int)strtol(s, (char **)&s, 10); continue; }
            s++;
        }
        if (nm < 2) { modes[0] = 0; modes[1] = 1; nm = 2; }
    }
    const size_t n = (size_t)T * 6144;
    _Float16 *ref = malloc(n * 2);
    double best[8];
    for (int m = 0; m < nm; m++) best[m] = 1e9;
    const int saved = pf_flash_mode;
    for (int trial = 0; trial < 3; trial++)
        for (int m = 0; m < nm; m++) {
            pf_flash_mode = modes[m];
            memset(e->pao.ptr, 0xff, n * 2); // NaN：沒寫到的地方會被抓出來
            qw_batch *b = qw_batch_begin(e->g, false);
            for (int r = 0; r < reps; r++) pf_flash(e, b, 0, T, false);
            const double t = qw_batch_end(b) * 1e3 / reps;
            if (t < best[m]) best[m] = t;
            if (trial) continue;
            const _Float16 *o = e->pao.ptr;
            if (m == 0) { memcpy(ref, o, n * 2); continue; }
            double mad = 0, mref = 0, s2 = 0, r2 = 0;
            size_t bad = 0;
            for (size_t i = 0; i < n; i++) {
                const double x = (double)o[i], y = (double)ref[i];
                if (x != x) { bad++; continue; }
                const double d = fabs(x - y);
                if (d > mad) mad = d;
                if (fabs(y) > mref) mref = fabs(y);
                s2 += d * d;
                r2 += y * y;
            }
            printf("  模式 %d vs %d：最大絕對差 %.3g（輸出最大 %.3g）、相對 RMS %.3g、NaN %zu\n", modes[m], modes[0], mad, mref,
                   sqrt(s2 / (r2 > 0 ? r2 : 1)), bad);
        }
    pf_flash_mode = saved;
    for (int m = 0; m < nm; m++) printf("  attention 核心 模式 %d：%.3f ms／層\n", modes[m], best[m]);
    free(ref);
}

// prefill 各部分的微基準：在最後一段 prefill 的實際資料上，同一個 command buffer 連續跑 reps 次，算每次的 GPU 時間
// （profile 模式每個 dispatch 一個 encoder，小 kernel 會被放大十幾倍，不能用）
// 量測用（QW_PF_KBENCH + QW_MOE_KB）：routed expert 的 grouped GEMM，gate + up、down 各自，48 層輪流。
// 路由是 prefill 最後一段最後一層的（真實分布）。QW_MOE_KB="設定;設定;…"，設定是空白分隔的 名稱=值（"-" 不改），
// 每輪依序換設定（setenv、重新規劃區塊表）交錯量，各取最好的一次
static void moe_kb(qw_engine *e, int T) {
    const uint32_t P = (uint32_t)T * QW_TOPK;
    const qw_layer *Ll0 = &e->m->layers[0];
    const uint32_t *cnt = e->pcnt.ptr;
    long pad[4] = {0};
    int mx = 0;
    for (int x = 0; x < QW_EXPERTS; x++) {
        for (int s = 0; s < 4; s++) pad[s] += ((cnt[x] + (16 << s) - 1) / (16 << s)) * (16 << s);
        if ((int)cnt[x] > mx) mx = (int)cnt[x];
    }
    printf("每個 expert 的列數：平均 %.1f、最多 %d；補齊後 TM16 %.3f、32 %.3f、64 %.3f、128 %.3f 倍\n", (double)P / QW_EXPERTS, mx,
           pad[0] / (double)P, pad[1] / (double)P, pad[2] / (double)P, pad[3] / (double)P);
    char cfgs[1024], *cfg[8];
    snprintf(cfgs, sizeof cfgs, "%s", getenv("QW_MOE_KB"));
    int nc = 0;
    for (char *t = strtok(cfgs, ";"); t && nc < 8; t = strtok(NULL, ";")) cfg[nc++] = t;
    char set[8][256];
    double best[8][2];
    const double fl[2] = {2.0 * 2 * P * D * QW_MOE_INTER, 2.0 * P * D * QW_MOE_INTER};
    for (int c = 0; c < nc; c++) { snprintf(set[c], sizeof set[c], "%s", cfg[c]); best[c][0] = best[c][1] = 1e9; }
    const int ntr = gg_par("QW_KB_TRIALS", 5); // 輪數（機器雜訊大時多量幾輪）
    for (int trial = 0; trial < ntr; trial++)
        for (int c = 0; c < nc; c++) {
            char tmp[256];
            snprintf(tmp, sizeof tmp, "%s", set[c]);
            static const char *knobs[] = {"QW_PFG_TM", "QW_PFG_TN", "QW_PFG_SG", "QW_PFG_KC", "QW_PFG_S", "QW_PFG_T", "QW_PFG_DBG", "QW_PFG_I8", "QW_I8_RK", "QW_I8_KC", "QW_I8_GC", "QW_I8_PAIR"};
            for (size_t i = 0; i < sizeof knobs / sizeof knobs[0]; i++) unsetenv(knobs[i]);
            for (char *sv, *t = strtok_r(tmp, " ", &sv); t; t = strtok_r(NULL, " ", &sv)) {
                char *eq = strchr(t, '=');
                if (eq) { *eq = 0; setenv(t, eq + 1, 1); }
            }
            qw_batch *b = qw_batch_begin(e->g, false);
            moe_plan(e, b, Ll0->moe.giq || i8moe_on() ? moe_gg_tm() : moe_tm()); // 同 pf_moe：MLX 的 FP16 版用 moe_tm()
            if (i8moe_on()) { // INT8：輸入先量化好（不計時；prefill 裡併在 scatter 與 SiLU 裡）
                const qw_moe *M0 = &e->m->layers[0].moe;
                pf_i8q(e, b, i8moe_fgu(M0), D, &e->pXs, &e->pq8x, &e->pq8s, P);
                pf_i8q(e, b, i8moe_fd(M0), QW_MOE_INTER, &e->pH, &e->pq8h, &e->pq8hs, P);
            }
            qw_batch_end(b);
            for (int k = 0; k < 2; k++) {
                b = qw_batch_begin(e->g, false);
                int nl = 0;
                for (int r = 0; r < QW_LAYERS; r++) {
                    const qw_moe *M = &e->m->layers[r].moe;
                    const int ff = gg_par("QW_KB_FMT", -1); // 只量某個格式的層（gate/up 看 fgu、down 看 fd）
                    if (ff >= 0 && (k == 0 ? M->fgu : M->fd) != ff) continue;
                    nl++;
                    if (i8moe_use(M)) {
                        if (k == 0) pf_moe_i8_gu(e, b, M, P);
                        else pf_moe_i8_dn(e, b, M, P);
                    } else if (!M->giq) { // MLX：FP16 的 grouped qmm（預設路徑）
                        if (k == 0) {
                            pf_qmm_moe(e, b, &M->gate, &e->pXs, &e->pGg, P);
                            pf_qmm_moe(e, b, &M->up, &e->pXs, &e->pGu, P);
                        } else pf_qmm_moe(e, b, &M->down, &e->pH, &e->pDm, P);
                    } else if (k == 0) {
                        pf_qmm_moe_iq(e, b, M->giq, M->fgu, QW_MOE_INTER, D, &e->pXs, &e->pGg, P);
                        pf_qmm_moe_iq(e, b, M->uiq, M->fgu, QW_MOE_INTER, D, &e->pXs, &e->pGu, P);
                    } else pf_qmm_moe_iq(e, b, M->diq, M->fd, D, QW_MOE_INTER, &e->pH, &e->pDm, P);
                }
                const double t = qw_batch_end(b) * 1e3 / (nl ? nl : 1);
                if (t < best[c][k]) best[c][k] = t;
            }
        }
    // QW_KB_CMP：兩種設定（QW_KB_CMP_A、QW_KB_CMP_B：INT8 路徑的 QW_PFG_DBG；-1 = 非 INT8 的預設路徑）逐層、逐元素比較（gate 與 down），
    // 另外印相對 RMS 誤差 sqrt(Σ(b − a)² / Σa²)
    if (getenv("QW_KB_CMP")) {
        double wsum[2] = {0, 0}, wref[2] = {0, 0};
        for (int lr = 0; lr < QW_LAYERS * 2; lr++) {
        const qw_moe *M = &e->m->layers[lr / 2].moe;
        const bool dn = lr & 1;
        const int NN = dn ? D : QW_MOE_INTER;
        const size_t ne = (size_t)P * NN;
        _Float16 *ref = malloc(ne * 2);
        for (int v = 0; v < 2; v++) {
            const char *dv = getenv(v ? "QW_KB_CMP_B" : "QW_KB_CMP_A");
            const bool fp = dv && atoi(dv) < 0;
            setenv("QW_PFG_I8", fp ? "0" : "1", 1);
            setenv("QW_PFG_DBG", dv && !fp ? dv : "0", 1);
            qw_batch *b = qw_batch_begin(e->g, false);
            moe_plan(e, b, M->giq || !fp ? moe_gg_tm() : moe_tm());
            if (fp) {
                if (!M->giq) pf_qmm_moe(e, b, dn ? &M->down : &M->gate, dn ? &e->pH : &e->pXs, dn ? &e->pDm : &e->pGg, P);
                else if (dn) pf_qmm_moe_iq(e, b, M->diq, M->fd, D, QW_MOE_INTER, &e->pH, &e->pDm, P);
                else pf_qmm_moe_iq(e, b, M->giq, M->fgu, QW_MOE_INTER, D, &e->pXs, &e->pGg, P);
            } else if (dn) {
                pf_i8q(e, b, i8moe_fd(M), QW_MOE_INTER, &e->pH, &e->pq8h, &e->pq8hs, P);
                pf_moe_i8_dn(e, b, M, P);
            } else {
                pf_i8q(e, b, i8moe_fgu(M), D, &e->pXs, &e->pq8x, &e->pq8s, P);
                if (M->giq) pf_qmm_moe_i8(e, b, M->giq, M->fgu, QW_MOE_INTER, D, &e->pq8x, &e->pq8s, &e->pGg, P);
                else pf_qmm_moe_i8m(e, b, &M->gate, &e->pq8x, &e->pq8s, &e->pGg, P);
            }
            qw_batch_end(b);
            const _Float16 *y = dn ? e->pDm.ptr : e->pGg.ptr;
            if (!v) { memcpy(ref, y, ne * 2); continue; }
            size_t bad = 0; double mx = 0, s2 = 0, r2 = 0;
            for (size_t i = 0; i < ne; i++) {
                const double d = fabs((double)y[i] - (double)ref[i]);
                s2 += d * d;
                r2 += (double)ref[i] * ref[i];
                if (d > 0) { if (bad < 2) printf("  差異 列 %zu 欄 %zu：%g vs %g\n", i / NN, i % NN, (double)ref[i], (double)y[i]); bad++; }
                if (d > mx) mx = d;
            }
            if (bad) printf("比較 層 %d %s（fmt %d）：%zu / %zu 個不同，最大差 %g，相對 RMS %.3g\n", lr / 2, dn ? "down" : "gate", dn ? i8moe_fd(M) : i8moe_fgu(M), bad, ne, mx,
                            r2 > 0 ? sqrt(s2 / r2) : 0.0);
            wsum[dn] += s2;
            wref[dn] += r2;
        }
        free(ref);
        }
        printf("全部層的相對 RMS：gate %.4g、down %.4g\n", wref[0] > 0 ? sqrt(wsum[0] / wref[0]) : 0.0, wref[1] > 0 ? sqrt(wsum[1] / wref[1]) : 0.0);
    }
    for (int c = 0; c < nc; c++)
        printf("[%s] gate+up %.3f ms（%.1f TFLOPS）、down %.3f ms（%.1f TFLOPS）、合計 %.3f ms\n", set[c], best[c][0], fl[0] / best[c][0] / 1e9,
               best[c][1], fl[1] / best[c][1] / 1e9, best[c][0] + best[c][1]);
}

void qw_engine_pf_kbench(qw_engine *e, int T, int reps) {
    const qw_layer *Ll = &e->m->layers[0], *La = NULL;
    for (int l = 0; l < QW_LAYERS && !La; l++) if (e->m->layers[l].full_attn) La = &e->m->layers[l];
    const uint32_t pos0 = (uint32_t)(e->pos - T);
    if (getenv("QW_MOE_KB")) { moe_kb(e, T); return; }
    if (getenv("QW_QK_KB")) { qk_kb(e, (uint32_t)T, pos0, reps); return; }
    const char *names[4] = {"hyper", "deltanet", "attention", "moe"};
    const int per[4] = {2 * QW_LAYERS, NLIN, NATT, QW_LAYERS};
    double tot = 0;
    for (int k = 0; k < 4; k++) {
        double best = 1e9;
        for (int trial = 0; trial < 3; trial++) {
            qw_batch *b = qw_batch_begin(e->g, false);
            for (int r = 0; r < reps; r++) {
                e->ppending = false;
                if (k == 0) pf_hyper(e, b, &Ll->attn_hc, (uint32_t)T);
                else if (k == 1) pf_deltanet(e, b, &Ll->lin, 0, (uint32_t)T);
                else if (k == 2) pf_attention(e, b, &La->att, 0, (uint32_t)T, pos0);
                else pf_moe(e, b, &Ll->moe, (uint32_t)T);
            }
            const double t = qw_batch_end(b) * 1e3 / reps;
            if (t < best) best = t;
        }
        printf("%-10s %8.3f ms／次 × %3d = %7.1f ms\n", names[k], best, per[k], best * per[k]);
        tot += best * per[k];
    }
    printf("合計 %.1f ms（%d 個 token，pos %u）\n", tot, T, pos0);
    if (Ll->attn_hc.mix_down.i8) { // hyper 的 INT8 版 vs 原本（同一個行程交錯量）
        qw_hyper H0 = Ll->attn_hc;
        H0.mix_down.i8 = NULL;
        double bh[2] = {1e9, 1e9};
        for (int trial = 0; trial < 5; trial++)
            for (int v = 0; v < 2; v++) {
                qw_batch *b = qw_batch_begin(e->g, false);
                for (int r = 0; r < reps; r++) {
                    e->ppending = true; // 含殘差更新（同實際 prefill）
                    pf_hyper(e, b, v ? &H0 : &Ll->attn_hc, (uint32_t)T);
                }
                const double t = qw_batch_end(b) * 1e3 / reps;
                if (t < bh[v]) bh[v] = t;
            }
        printf("hyper INT8 %.3f ms、原本 %.3f ms（每次，含殘差更新）\n", bh[0], bh[1]);
    }
    if (!Ll->attn_hc.mix_down.i8 && Ll->attn_hc.mix_down.gq) { // ghp_hc_in_t vs ghp_hc_in（同一個行程交錯量）
        const int keep = ghp_hct;
        double bh[2] = {1e9, 1e9};
        for (int trial = 0; trial < 5; trial++)
            for (int v = 0; v < 2; v++) {
                ghp_hct = !v;
                qw_batch *b = qw_batch_begin(e->g, false);
                for (int r = 0; r < reps; r++) {
                    e->ppending = true; // 含殘差更新（同實際 prefill）
                    pf_hyper(e, b, &Ll->attn_hc, (uint32_t)T);
                }
                const double t = qw_batch_end(b) * 1e3 / reps;
                if (t < bh[v]) bh[v] = t;
            }
        ghp_hct = keep;
        printf("hyper：ghp_hc_in_t %.3f ms、ghp_hc_in %.3f ms（每次，含殘差更新）\n", bh[0], bh[1]);
    }
    if (getenv("QW_KB_POS")) { // attention 在不同起點的成本（同一個行程裡交錯量，避開降頻的差異）：QW_KB_POS=4096,16384,…（≤ pos0）
        // 注意：會用最後一段的輸入覆寫這些位置的 KV（量完就結束，不影響結果檢查）
        uint32_t ps[8];
        int np = 0;
        for (const char *s = getenv("QW_KB_POS"); *s && np < 8;) {
            const long v = strtol(s, (char **)&s, 10);
            if (v >= 0 && (uint32_t)v <= pos0) ps[np++] = (uint32_t)v / 64 * 64;
            if (*s == ',') s++;
            else if (*s) break;
        }
        if (getenv("QW_KB_PREP")) { // 只量 p_attn_prep（q / k 的 norm + RoPE、KV 寫入）在各位置的時間；QW_KB_PREP=層號（KV 寫到哪一層，預設 0）
            const int ai = atoi(getenv("QW_KB_PREP"));
            const size_t Lc = (size_t)e->kvcap, koff = (size_t)(ai % NATT) * Lc * 2 * 256 * 2;
            double bp[8];
            for (int i = 0; i < np; i++) bp[i] = 1e9;
            for (int trial = 0; trial < 3; trial++)
                for (int i = np - 1; i >= 0; i--) {
                    qw_batch *b = qw_batch_begin(e->g, false);
                    pf_rope_tab(e, b, ps[i], (uint32_t)T);
                    qw_batch_end(b);
                    b = qw_batch_begin(e->g, false);
                    for (int r = 0; r < reps; r++) {
                        qw_batch_kernel(b, pf_need(e, La->att.nf32 ? "p_attn_prep_f" : "p_attn_prep"));
                        qw_batch_buf(b, 0, &e->pqg, 0);
                        qw_batch_buf(b, 1, &e->pkk, 0);
                        qw_batch_buf(b, 2, &e->pvv, 0);
                        bind_t(e, b, 3, La->att.q_norm);
                        bind_t(e, b, 4, La->att.k_norm);
                        qw_batch_buf(b, 5, &e->paq, 0);
                        qw_batch_buf(b, 6, &e->Kc, koff);
                        qw_batch_buf(b, 7, &e->Vc, koff);
                        qw_batch_bytes(b, 8, &ps[i], 4);
                        qw_batch_buf(b, 9, &e->prcs, 0);
                        qw_batch_dispatch(b, 24 + 2, (unsigned)T, 1, 256, 1, 1);
                    }
                    const double t = qw_batch_end(b) * 1e3 / reps;
                    if (t < bp[i]) bp[i] = t;
                }
            for (int i = 0; i < np; i++) printf("p_attn_prep @pos %6u（KV 第 %d 層）：%.3f ms\n", ps[i], ai % NATT, bp[i]);
            return;
        }
        if (getenv("QW_QK_AB")) { // INT8 Q·Kᵀ 的交錯 A/B：每個位置 QW_QK_AB=位元（預設 3）vs 不開，各取 3 次中最好的
            const int on = atoi(getenv("QW_QK_AB")) > 0 ? atoi(getenv("QW_QK_AB")) : 3;
            double bq[8][2];
            for (int i = 0; i < np; i++) bq[i][0] = bq[i][1] = 1e9;
            for (int trial = 0; trial < 3; trial++)
                for (int i = np - 1; i >= 0; i--)
                    for (int k = 0; k < 2; k++) {
                        const int m = (trial + k) & 1; // 每輪交換先後
                        qk_ovr = m ? on : 0;
                        qw_batch *b = qw_batch_begin(e->g, false);
                        for (int r = 0; r < reps; r++) {
                            e->ppending = false;
                            pf_attention(e, b, &La->att, 0, (uint32_t)T, ps[i]);
                        }
                        const double t = qw_batch_end(b) * 1e3 / reps;
                        if (t < bq[i][m]) bq[i][m] = t;
                    }
            qk_ovr = -1;
            for (int i = 0; i < np; i++)
                printf("attention @pos %6u：不開 %8.3f ms、qk(%d) %8.3f ms／層（%+.1f%%）× %d = %+.1f ms\n", ps[i], bq[i][0], on, bq[i][1],
                       (bq[i][1] / bq[i][0] - 1) * 100, NATT, (bq[i][1] - bq[i][0]) * NATT);
            return;
        }
        double best[8];
        for (int i = 0; i < np; i++) best[i] = 1e9;
        for (int trial = 0; trial < 3; trial++)
            for (int i = np - 1; i >= 0; i--) { // 由大到小：較小的位置會蓋掉較大位置之前的 KV（比較大的位置要先量）
                qw_batch *b = qw_batch_begin(e->g, false);
                for (int r = 0; r < reps; r++) {
                    e->ppending = false;
                    pf_attention(e, b, &La->att, 0, (uint32_t)T, ps[i]);
                }
                const double t = qw_batch_end(b) * 1e3 / reps;
                if (t < best[i]) best[i] = t;
            }
        for (int i = 0; i < np; i++) printf("attention @pos %6u：%8.3f ms／次 × %d = %6.1f ms\n", ps[i], best[i], NATT, best[i] * NATT);
        if (getenv("QW_PF_PROF")) // 由大到小、再由小到大各一次（看得出暖機 / 降頻造成的順序偏差）
            for (int k = 0; k < 2 * np; k++) {
                const int i = k < np ? np - 1 - k : k - np;
                qw_profile_enable(e->g, true);
                qw_batch *b = qw_batch_begin(e->g, false);
                e->ppending = false;
                pf_attention(e, b, &La->att, 0, (uint32_t)T, ps[i]);
                qw_batch_end(b);
                char ti[64];
                snprintf(ti, sizeof ti, "attention @pos %u", ps[i]);
                prof_dump(e, ti);
                qw_profile_enable(e->g, false);
            }
        return;
    }
    if (getenv("QW_GDN_KB")) { // DeltaNet prefill 各步驟（第 0 層，pfdn_mask 只編碼一部分；每次的時間 × NLIN）
        // 每一步各量預設與 QW_GDN_CHUNK 版（同一個行程裡交錯，每輪交換先後）
        static const char *gn[] = {"投影 qkv/z/a/b", "p_gdn_in + conv", "分塊 prep", "分塊 scan", "分塊 prep + scan", "gated norm", "out_proj", "全部"};
        static const unsigned gm[] = {1, 2, 4, 8, 12, 16, 32, ~0u};
        for (int k = 0; k < 8; k++) {
            double best[2] = {1e9, 1e9};
            for (int trial = 0; trial < 3; trial++)
                for (int v2 = 0; v2 < 2; v2++) {
                    const int f = (trial + v2) & 1;
                    pfdn_mask = gm[k];
                    gdn_fast_ovr = f;
                    qw_batch *b = qw_batch_begin(e->g, false);
                    for (int r = 0; r < reps; r++) pf_deltanet_core(e, b, &Ll->lin, 0, (uint32_t)T);
                    const double t = qw_batch_end(b) * 1e3 / reps;
                    if (t < best[f]) best[f] = t;
                }
            pfdn_mask = ~0u;
            gdn_fast_ovr = -1;
            printf("  DeltaNet %-18s 預設 %8.3f ms、fast %8.3f ms／次 × %d = %6.1f / %6.1f ms\n", gn[k], best[0], best[1], NLIN,
                   best[0] * NLIN, best[1] * NLIN);
        }
    }
    if (getenv("QW_FLASH_KB") || getenv("QW_SEL_DUMP")) pf_flash_kb(e, (uint32_t)T, reps);
    if (getenv("QW_PF_PROF")) { // 各部分一次（第 0 層 / 第一個 attention 層）的逐 kernel 時間（profiling 會把 dispatch 拆開，總和比實際略大）
        for (int k = 0; k < 4; k++) {
            qw_profile_enable(e->g, true);
            qw_batch *b = qw_batch_begin(e->g, false);
            e->ppending = false;
            if (k == 0) pf_hyper(e, b, &Ll->attn_hc, (uint32_t)T);
            else if (k == 1) pf_deltanet(e, b, &Ll->lin, 0, (uint32_t)T);
            else if (k == 2) pf_attention(e, b, &La->att, 0, (uint32_t)T, pos0);
            else pf_moe(e, b, &Ll->moe, (uint32_t)T);
            qw_batch_end(b);
            prof_dump(e, names[k]);
            qw_profile_enable(e->g, false);
        }
    }
    if (getenv("QW_DN_KB")) { // DeltaNet 的投影（第 0 層）：qkv、z、a + b、out
        const qw_deltanet *N = &Ll->lin;
        const char *pn[4] = {"in_qkv", "in_z", "in_a + in_b", "out_proj"};
        for (int k = 0; k < 4; k++) {
            double best = 1e9;
            for (int trial = 0; trial < 3; trial++) {
                qw_batch *b = qw_batch_begin(e->g, false);
                for (int r = 0; r < reps; r++) { // 與 pf_deltanet_core 同一條路徑（INT8 時每次都含 px 的量化）
                    if (k < 3) pf_dn_proj(e, b, N, (uint32_t)T, 1u << k);
                    else if (dn_ggm_on() && N->out.gq) pf_attn_mm(e, b, &N->out, &e->py, &e->pout, (uint32_t)T);
                    else pf_qmm(e, b, &N->out, &e->py, &e->pout, (uint32_t)T, false);
                }
                const double t = qw_batch_end(b) * 1e3 / reps;
                if (t < best) best = t;
            }
            printf("  DeltaNet %-12s %8.3f ms／次\n", pn[k], best);
        }
        // p_gdn_in：每個 token 一個 threadgroup（gdn_tb_ovr = 0）vs token 區塊版（1）；先逐位元比對 q / k / v / la / be
        const size_t sq = (size_t)T * 2048 * 2, sv = (size_t)T * 6144 * 2, sl = (size_t)T * 48 * 4;
        char *r2 = malloc(2 * sq + sv + 2 * sl), *cs = malloc(3 * 10240 * 4);
        memcpy(cs, e->conv_st.ptr, 3 * 10240 * 4); // 第 0 層的 conv 狀態：兩次都從同一個狀態開始（p_conv_state 會改寫）
        const qw_buf *ob[5] = {&e->pq, &e->pk, &e->pv, &e->pla, &e->pbe};
        const size_t os[5] = {sq, sq, sv, sl, sl};
        for (int v = 0; v < 2; v++) {
            gdn_tb_ovr = v;
            for (int i = 0; i < 5; i++) memset(ob[i]->ptr, 0xff, os[i]);
            memcpy(e->conv_st.ptr, cs, 3 * 10240 * 4);
            pfdn_mask = 2;
            qw_batch *b = qw_batch_begin(e->g, false);
            pf_deltanet_core(e, b, N, 0, (uint32_t)T);
            qw_batch_end(b);
            size_t o = 0;
            bool same = true;
            for (int i = 0; i < 5; i++) {
                if (!v) memcpy(r2 + o, ob[i]->ptr, os[i]);
                else if (memcmp(r2 + o, ob[i]->ptr, os[i])) {
                    same = false;
                    size_t j = 0, nd = 0;
                    while (j < os[i] && r2[o + j] == ((char *)ob[i]->ptr)[j]) j++;
                    for (size_t u = 0; u < os[i]; u++) nd += r2[o + u] != ((char *)ob[i]->ptr)[u];
                    printf("    緩衝區 %d：第一個不同的位元組 %zu / %zu，共 %zu 個\n", i, j, os[i], nd);
                }
                o += os[i];
            }
            if (v) printf("  DeltaNet p_gdn_in 區塊版 vs 原版：%s\n", same ? "相同" : "不同");
        }
        free(r2), free(cs);
        double bt[2] = {1e9, 1e9};
        for (int trial = 0; trial < 4; trial++)
            for (int v2 = 0; v2 < 2; v2++) {
                const int v = (trial + v2) & 1;
                gdn_tb_ovr = v;
                qw_batch *b = qw_batch_begin(e->g, false);
                for (int r = 0; r < reps; r++) pf_deltanet_core(e, b, N, 0, (uint32_t)T);
                const double t = qw_batch_end(b) * 1e3 / reps;
                if (t < bt[v]) bt[v] = t;
            }
        gdn_tb_ovr = -1;
        pfdn_mask = ~0u;
        printf("  DeltaNet p_gdn_in + conv：原版 %8.3f ms、區塊版 %8.3f ms／次\n", bt[0], bt[1]);
    }
    { // attention 的投影（q / k / v / indexer / o），12 層輪流（格式依層不同）
        double best = 1e9;
        for (int trial = 0; trial < 3; trial++) {
            qw_batch *b = qw_batch_begin(e->g, false);
            for (int r = 0; r < reps; r++) {
                const qw_attn *A = &e->m->layers[(r % NATT) * 4 + 3].att;
                pf_attn_mm(e, b, &A->q, &e->px, &e->pqg, (uint32_t)T);
                pf_attn_mm(e, b, &A->k, &e->px, &e->pkk, (uint32_t)T);
                pf_attn_mm(e, b, &A->v, &e->px, &e->pvv, (uint32_t)T);
                pf_attn_mm(e, b, &A->idx_qk, &e->px, &e->piqk, (uint32_t)T);
                pf_attn_mm(e, b, &A->o, &e->pay, &e->pout, (uint32_t)T);
            }
            const double t = qw_batch_end(b) * 1e3 / reps;
            if (t < best) best = t;
        }
        printf("（attention 投影 %.3f ms／次 × %d = %.1f ms）\n", best, NATT, best * NATT);
    }
}

const float *qw_engine_prefill(qw_engine *e, const int32_t *ids, int n, float *dump, double *gpu_ms) {
    char err[1024];
    if (!pf_init(e, err, sizeof err)) {
        fprintf(stderr, "prefill 初始化失敗：%s\n", err);
        return NULL;
    }
    if (dump && n > PF) return NULL;
    // dump 用的空間：[48][PF][HCD]
    if (dump && e->dump.size < (size_t)QW_LAYERS * PF * HCD * 4) {
        qw_buf_free(&e->dump);
        if (!alloc(e, &e->dump, (size_t)QW_LAYERS * PF * HCD * 4)) return NULL;
    }
    double gt = 0;
    int i = 0;
    e->ng_pre_n = 0;
    { // KV 一次配置到整個 prompt 需要的大小：逐段加倍會多搬幾次整份 KV（CPU memcpy，GPU 閒著）
        int need = e->pos + n + VBMAX + 2;
        if (need > e->max_ctx) need = e->max_ctx;
        kv_reserve(e, need);
    }
    while (i < n) {
        int T = n - i;
        if (T > PF) T = PF;
        if (e->pos + T > e->max_ctx || !kv_reserve(e, e->pos + T + VBMAX + 2)) break;
        if (e->pf_mtp) { // 段尾的下一個 token 先放進 tok；不知道下一個時（整個 prompt 的最後一個位置）留給 decode 的 MTP
            const int32_t nx = i + T < n ? ids[i + T] : e->pf_next;
            if (nx >= 0) ((int32_t *)e->tok.ptr)[e->pos + 2 + T] = nx;
            e->mtp_T = nx >= 0 ? T : T - 1;
        }
        e->pf_nx_T = n - i - T < PF ? n - i - T : PF;
        e->pf_nx_ids = ids + i + T;
        if (e->pos + T + e->pf_nx_T > e->max_ctx) e->pf_nx_T = 0;
        double t = pf_segment(e, ids + i, (uint32_t)T, dump);
        e->pf_nx_T = 0;
        if (t < 0) { e->ng_pre_n = 0; return NULL; }
        gt += t * 1e3;
        if (dump) {
            for (int l = 0; l < QW_LAYERS; l++)
                memcpy(dump + (size_t)l * n * HCD, (char *)e->dump.ptr + (size_t)l * PF * HCD * 4, (size_t)T * HCD * 4);
        }
        i += T;
    }
    const float *lg = e->logits.ptr;
    for (; i < n; i++) {
        lg = qw_engine_step(e, ids[i], NULL);
        gt += e->last_ms;
    }
    if (gpu_ms) *gpu_ms = gt;
    return lg;
}


// ================================================================ MTP 驗證：一次 2 個 token
#define VB (e->vbn) // 這一輪驗證幾個 token（2…VBMAX）

// 多 token kernel 依 NB 編譯並改名（名稱加 _n{NB}）；第一次用到某個 NB 時編譯
static const char *const vb_names[] = {"kb_hc_in", "kb_mixup", "kb_embed", "kb_gdn_in", "kb_gdn_in_t", "kb_normgate",
                                       "moe_router_b", "moe_gateup_fb", "moe_down_fb", "gdn_stepn",
                                       "moe_sel_b", "moe_gateup_db", "moe_down_db"};
static qw_kernel *vk(qw_engine *e, const char *base) {
    char name[64];
    snprintf(name, sizeof name, "%s_n%d", base, VB);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    char ren[1024] = "", defs[2048], err[8192];
    for (size_t i = 0; i < sizeof vb_names / sizeof *vb_names; i++) {
        char l[96];
        snprintf(l, sizeof l, "#define %s %s_n%d\n", vb_names[i], vb_names[i], VB);
        strcat(ren, l);
    }
    snprintf(defs, sizeof defs, "%s#define D %d\n#define HC %d\n#define NB %d\n", ren, D, QW_HC, VB);
    bool ok = compile(e, defs, src_engine, sizeof src_engine, err, sizeof err);
    snprintf(defs, sizeof defs, "%s#define E %d\n#define TOPK %d\n#define D %d\n#define I %d\n#define RPS 1\n#define NB %d\n"
             "#define moe_router mr_unused\n#define moe_select ms_unused\n#define moe_gateup mg_unused\n#define moe_down md_unused\n"
             "#define moe_gateup_f mgf_unused\n#define moe_down_f mdf_unused\n", ren, QW_EXPERTS, QW_TOPK, D, QW_MOE_INTER, VB);
    ok = ok && compile(e, defs, src_moe, sizeof src_moe, err, sizeof err);
    snprintf(defs, sizeof defs, "%s#define gdn_step gdn_step_unused\n#define DK 128\n#define DV 128\n#define HK 16\n#define HV 48\n#define DVB 32\n#define NB %d\n",
             ren, VB);
    ok = ok && compile(e, defs, src_gdn, sizeof src_gdn, err, sizeof err);
    if (e->m->lin_tiled) { // GGUF：V head 交錯順序
        snprintf(defs, sizeof defs, "#define GDN_TILED 1\n#define GDN_DEFER 1\n#define gdn_step gdn_step_t_unused\n#define gdn_stepn gdn_stepn_t_n%d\n"
                 "#define gdn_stepn_d gdn_stepn_dt_n%d\n#define DK 128\n#define DV 128\n#define HK 16\n#define HV 48\n#define DVB 32\n#define NB %d\n",
                 VB, VB, VB);
        ok = ok && compile(e, defs, src_gdn, sizeof src_gdn, err, sizeof err);
    }
    if (!ok) { fprintf(stderr, "編譯 NB=%d 的 kernel 失敗：%s\n", VB, err); exit(1); }
    return pf_need(e, name);
}

static qw_kernel *qmvb_kernel(qw_engine *e, const char *kind, int bits, int K, bool fout) {
    char name[96], defs[512], err[8192];
    snprintf(name, sizeof name, "%s_b%d_k%d_n%d%s", kind, bits, K, VB, fout ? "_f" : "");
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs,
             "#define qmvb qmvb_b%d_k%d_n%d%s\n#define qmvb_splitk qmvb_splitk_b%d_k%d_n%d%s\n#define qmvb_multi qmvb_multi_b%d_k%d_n%d%s\n"
             "#define BITS %d\n#define K %d\n#define NB %d\n#define ST bfloat\n%s",
             bits, K, VB, fout ? "_f" : "", bits, K, VB, fout ? "_f" : "", bits, K, VB, fout ? "_f" : "", bits, K, VB,
             fout ? "#define YT float\n" : "");
    if (!compile(e, defs, src_qmvb, sizeof src_qmvb, err, sizeof err)) {
        fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
        exit(1);
    }
    return qw_kernel_get(e->g, name);
}

typedef struct { uint32_t N, xstride, ystride, pad; } qmvb_args;
// 驗證用的輸入轉換（kernels/qmvb.metal 的 qmvb_xform；XF = 1 DeltaNet gated RMSNorm、2 attention output gate）。
// decode 把同樣的轉換併在 qmv_xf 裡（MLX），驗證要用逐字相同的算式（kb_normgate / kb_attn_gate 的捨入不同），推測解碼才會等於 greedy
static qw_kernel *xform_kernel(qw_engine *e, int xf) {
    char name[64], defs[512], err[8192];
    snprintf(name, sizeof name, "qmvb_xform%d", xf);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs,
             "#define qmvb_xform %s\n#define qmvb qmvb_xfu%d\n#define qmvb_splitk qmvb_splitk_xfu%d\n#define qmvb_multi qmvb_multi_xfu%d\n"
             "#define XF %d\n#define BITS 8\n#define K 6144\n#define NB 1\n#define ST bfloat\n",
             name, xf, xf, xf, xf);
    if (!compile(e, defs, src_qmvb, sizeof src_qmvb, err, sizeof err) || !(k = qw_kernel_get(e->g, name))) {
        fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
        exit(1);
    }
    return k;
}
// decode 的 MLX DeltaNet / attention 是否把轉換併進 out_proj / o_proj（qmv_xf）；--exact-spec 時驗證跟著用 xform_kernel
static bool mlx_xf(const qw_qlin *L) { return !L->gq && !getenv("QW_NO_XF"); }
static bool vb_xf(const qw_engine *e, const qw_qlin *L) { return exact_on(e) && mlx_xf(L); }
static void vb_xform(qw_engine *e, qw_batch *b, int xf, const qw_buf *o, const qw_buf *aux, const qw_buf *ssp, const st_tensor *nw, const qw_buf *y) {
    _Static_assert(6144 % 256 == 0, "qmvb_xform：每個 token 6144 個元素、256 個一組");
    qw_batch_kernel(b, xform_kernel(e, xf));
    qw_batch_buf(b, 0, o, 0);
    qw_batch_buf(b, 1, aux, 0);
    qw_batch_buf(b, 2, ssp, 0);
    bind_t(e, b, 3, nw);
    qw_batch_buf(b, 4, y, 0);
    qw_batch_dispatch(b, 6144 / 256, (unsigned)VB, 1, 256, 1, 1);
}

typedef struct { uint32_t N[4], off[4], total, xstride; } qmvb_multi_args;

// 驗證用 matvec 改走矩陣單元（kernels/qmvb_tc.metal）的條件：token 數夠多（QW_VB_TC=最少幾個，0 = 關）、
// 4 / 8-bit、K 是 32·NSG 的倍數、N 是 32 的倍數、x 的列距 = K 且緩衝區從 xoff 起至少 8 列
static int vb_tc_min(void) {
    static int v = -1;
    if (v < 0) v = getenv("QW_VB_TC") ? atoi(getenv("QW_VB_TC")) : 0; // 實驗：NB ≥ 6 時 lm_head / QSA 較快，真實生成打平
    return v;
}
static int tc_nsg(int K) { return K % 256 == 0 ? 8 : 4; }
static bool vb_tc_ok(const qw_engine *e, const qw_qlin *L, const qw_buf *x, size_t xoff) {
    const int mn = vb_tc_min();
    return mn > 0 && VB >= mn && (L->bits == 4 || L->bits == 8) && L->in % (32 * tc_nsg(L->in)) == 0 && L->out % 32 == 0 &&
           xoff + (size_t)VBMAX * L->in * 2 <= x->size;
}
static qw_kernel *qmvb_tc_kernel(qw_engine *e, int bits, int K, bool fout) {
    char name[96], defs[600], err[8192];
    snprintf(name, sizeof name, "qmvb_tc_b%d_k%d_n%d%s", bits, K, VB, fout ? "_f" : "");
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs, "#define qmvb_tc %s\n#define qmvb_tc_multi %s_m\n#define BITS %d\n#define K %d\n#define NB %d\n#define NSG %d\n%s",
             name, name, bits, K, VB, tc_nsg(K), fout ? "#define YT float\n" : "");
    if (!compile(e, defs, src_qmvb_tc, sizeof src_qmvb_tc, err, sizeof err)) {
        fprintf(stderr, "編譯 %s 失敗：%s\n", name, err);
        exit(1);
    }
    return qw_kernel_get(e->g, name);
}

static void vb_qmv_o(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *x, size_t xoff, const qw_buf *y, size_t yoff,
                     bool fout) {
    if (L->gq) { qmv_gg(e, b, L, x, xoff, y, yoff, VB, fout); return; }
    const bool splitk = L->out <= 1024 && L->in >= 4096;
    if (!splitk && vb_tc_ok(e, L, x, xoff)) {
        qmvb_args a = {(uint32_t)L->out, (uint32_t)L->in, (uint32_t)L->out, 0};
        qw_batch_kernel(b, qmvb_tc_kernel(e, L->bits, L->in, fout));
        bind_t(e, b, 0, L->w);
        bind_t(e, b, 1, L->s);
        bind_t(e, b, 2, L->b);
        qw_batch_buf(b, 3, x, xoff);
        qw_batch_buf(b, 4, y, yoff);
        qw_batch_bytes(b, 5, &a, sizeof a);
        qw_batch_dispatch(b, a.N / 32, 1, 1, 32 * tc_nsg(L->in), 1, 1);
        return;
    }
    qw_kernel *k = qmvb_kernel(e, splitk ? "qmvb_splitk" : "qmvb", L->bits, L->in, fout);
    qmvb_args a = {(uint32_t)L->out, (uint32_t)L->in, (uint32_t)L->out, 0};
    qw_batch_kernel(b, k);
    bind_t(e, b, 0, L->w);
    bind_t(e, b, 1, L->s);
    bind_t(e, b, 2, L->b);
    qw_batch_buf(b, 3, x, xoff);
    qw_batch_buf(b, 4, y, yoff);
    qw_batch_bytes(b, 5, &a, sizeof a);
    if (splitk) qw_batch_dispatch(b, a.N, 1, 1, 256, 1, 1);
    else qw_batch_dispatch(b, (a.N + 15) / 16, 1, 1, 256, 1, 1);
}

static void vb_qmv(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *x, const qw_buf *y, bool fout) {
    vb_qmv_o(e, b, L, x, 0, y, 0, fout);
}

static void vb_qmv_multi(qw_engine *e, qw_batch *b, const qw_qlin *const *L, const qw_buf *const *y, int n, const qw_buf *x) {
    bool anygg = false;
    for (int i = 0; i < n; i++) anygg |= L[i]->gq != 0;
    if (anygg) { // GGUF：逐個矩陣（輸出列距 = 各自的 N）
        for (int i = 0; i < n; i++) vb_qmv_o(e, b, L[i], x, 0, y[i], 0, false);
        return;
    }
    bool tc = true;
    for (int i = 0; i < n; i++) tc = tc && vb_tc_ok(e, L[i], x, 0) && L[i]->bits == L[0]->bits && L[i]->in == L[0]->in;
    qw_kernel *k = tc ? qw_kernel_get(e->g, (qmvb_tc_kernel(e, L[0]->bits, L[0]->in, false), ({ static char nm[96]; snprintf(nm, sizeof nm, "qmvb_tc_b%d_k%d_n%d_m", L[0]->bits, L[0]->in, VB); nm; })))
                      : qmvb_kernel(e, "qmvb_multi", L[0]->bits, L[0]->in, false);
    qmvb_multi_args a = {{0}, {0}, 0, (uint32_t)L[0]->in};
    qw_batch_kernel(b, k);
    for (int i = 0; i < 4; i++) {
        const qw_qlin *Q = L[i < n ? i : 0];
        bind_t(e, b, 3 * i, Q->w);
        bind_t(e, b, 3 * i + 1, Q->s);
        bind_t(e, b, 3 * i + 2, Q->b);
        qw_batch_buf(b, 13 + i, y[i < n ? i : 0], 0);
        a.N[i] = i < n ? (uint32_t)L[i]->out : 0;
        a.off[i] = a.total;
        if (i < n) a.total += (uint32_t)L[i]->out;
    }
    qw_batch_buf(b, 12, x, 0);
    qw_batch_bytes(b, 17, &a, sizeof a);
    if (tc) qw_batch_dispatch(b, a.total / 32, 1, 1, 32 * tc_nsg(L[0]->in), 1, 1);
    else qw_batch_dispatch(b, (a.total + 7) / 8, 1, 1, 256, 1, 1);
}

static void vb_copy(qw_engine *e, qw_batch *b, const qw_buf *src, size_t soff, const qw_buf *dst, size_t doff, uint32_t nfloat) {
    qw_batch_kernel(b, e->k_copyf);
    qw_batch_buf(b, 0, src, soff);
    qw_batch_buf(b, 1, dst, doff);
    qw_batch_bytes(b, 2, &nfloat, 4);
    d1(b, nfloat, 256);
}

// 每輪最多驗證幾個 token（含確定的那個）
static int spec_kmax(void) {
    const char *v = getenv("QW_SPEC_K");
    int k = v ? atoi(v) : 6;
    return k < 2 ? 2 : k > VBMAX ? VBMAX : k;
}

// 驗證一輪最多幾個 token：DFlash2 有信心截斷，一律準備到 VBMAX；否則照 QW_SPEC_K
static int vb_maxnb(const qw_engine *e) { return e->df_on ? VBMAX : spec_kmax(); }

static bool vb_init(qw_engine *e, char *err, size_t errlen) {
    if (e->vb_ready) return true;
    const size_t B = VBMAX;
    bool ok = alloc(e, &e->bh, B * HCD * 4) && alloc(e, &e->bnh, B * HCD * 2) && alloc(e, &e->bnf, B * HCD * 4) &&
              alloc(e, &e->bd, B * 320 * 2) && alloc(e, &e->bx, B * D * 2) && alloc(e, &e->binj, B * 64) && alloc(e, &e->binj2, B * 64) &&
              alloc(e, &e->bout, B * D * 2) && alloc(e, &e->bqkv, B * 10240 * 2) && alloc(e, &e->bz, B * 6144 * 2) &&
              alloc(e, &e->ba, B * 64 * 2) && alloc(e, &e->bb, B * 64 * 2) && alloc(e, &e->bgy, B * 6144 * 2) &&
              alloc(e, &e->bqg, B * 12288 * 2) && alloc(e, &e->bkk, B * 512 * 2) && alloc(e, &e->bvv, B * 512 * 2) &&
              alloc(e, &e->biqk, B * 640 * 2) && alloc(e, &e->bay, B * 6144 * 2) && alloc(e, &e->blog, B * 520 * 4) && alloc(e, &e->bmsel, 4096) && alloc(e, &e->bHu, (size_t)(VBMAX * QW_TOPK + 1) * 8 * QW_MOE_INTER * 2) &&
              alloc(e, &e->bH, B * 11 * 640 * 2) && alloc(e, &e->blg, B * (size_t)QW_VOCAB * 4) && alloc(e, &e->bmlg, B * (size_t)QW_VOCAB * 4) && alloc(e, &e->bres, 4 * 2 * VBMAX) && alloc(e, &e->baq, B * 6144 * 2) && alloc(e, &e->biq, B * 512 * 2) &&
              alloc(e, &e->bpart, B * QSA_NCH * 24 * 258 * 4) &&
              alloc(e, &e->bao, B * 6144 * 2) && alloc(e, &e->bgq, B * 2048 * 2) && alloc(e, &e->bgk, B * 2048 * 2) &&
              alloc(e, &e->bgv, B * 6144 * 2) && alloc(e, &e->bla, B * 64 * 4) && alloc(e, &e->bbe, B * 64 * 4) &&
              alloc(e, &e->bgo, B * 6144 * 2) && alloc(e, &e->bssp, B * 48 * 4 * 4) && alloc(e, &e->bhpd, GH_PV_SIZE) &&
              alloc(e, &e->bmh, B * HCD * 4) && alloc(e, &e->bme, B * HCD * 4) && alloc(e, &e->bmn, B * D * 2) &&
              alloc(e, &e->bmfe, B * D * 2) && alloc(e, &e->bmnh, B * HCD * 2) && alloc(e, &e->bmfh, B * HCD * 2) &&
              true;
    if (ok && e->m->lin_tiled && !e->dnK.ptr) ok = dn_alloc_defer(e); // GGUF DeltaNet 的延後寫回（通常開啟時已配置）
    for (int i = 0; ok && i < vb_maxnb(e) - 1; i++)
        ok = alloc(e, &e->Sx[i], e->S.size) && alloc(e, &e->convx[i], e->conv_st.size) && alloc(e, &e->plex[i], e->ple_st.size);
    for (int i = 0; ok && i < vb_maxnb(e) - 1; i++) { // 先碰一次（避免第一次用到時的配置成本算進生成）
        memset(e->Sx[i].ptr, 0, e->S.size);
        memset(e->convx[i].ptr, 0, e->conv_st.size);
        memset(e->plex[i].ptr, 0, e->ple_st.size);
    }
    if (!ok) {
        snprintf(err, errlen, "配置 MTP 驗證緩衝區失敗");
        return false;
    }
    // 先把 NB = 2…kmax 用到的 kernel 都編好（不讓編譯時間算進生成）
    for (int nb = 2; nb <= vb_maxnb(e); nb++) {
    e->vbn = nb;
    for (int l = 0; l < QW_LAYERS; l++) {
        const qw_layer *L = &e->m->layers[l];
        const qw_qlin *ql[] = {&L->attn_hc.mix_down, &L->mlp_hc.mix_down, L->full_attn ? &L->att.o : &L->lin.out};
        for (int i = 0; i < 3; i++)
            if (!ql[i]->gq) qmvb_kernel(e, ql[i]->out <= 1024 && ql[i]->in >= 4096 ? "qmvb_splitk" : "qmvb", ql[i]->bits, ql[i]->in, false);
        const qw_qlin *q0 = L->full_attn ? &L->att.q : &L->lin.in_qkv;
        if (!q0->gq) qmvb_kernel(e, "qmvb_multi", q0->bits, D, false);
    }
    const qw_qlin *lm = &e->m->lm_head, *mx = &e->m->mixer.mix_down;
    qmvb_kernel(e, lm->out <= 1024 && lm->in >= 4096 ? "qmvb_splitk" : "qmvb", lm->bits, lm->in, true);
    qmvb_kernel(e, mx->out <= 1024 && mx->in >= 4096 ? "qmvb_splitk" : "qmvb", mx->bits, mx->in, false);
    if (e->m->has_mtp) {
        const qw_qlin *mq[] = {&e->m->mtp_fc_emb, &e->m->mtp_fc_hidden, &e->m->mtp.att.o, &e->m->mtp.attn_hc.mix_down,
                               &e->m->mtp.mlp_hc.mix_down, &e->m->mtp_mixer.mix_down};
        for (int i = 0; i < 6; i++) qmvb_kernel(e, mq[i]->out <= 1024 && mq[i]->in >= 4096 ? "qmvb_splitk" : "qmvb", mq[i]->bits, mq[i]->in, false);
        qmvb_kernel(e, "qmvb_multi", e->m->mtp.att.q.bits, D, false);
    }
    vk(e, "kb_hc_in");
    if (exact_on(e)) { xform_kernel(e, 1); xform_kernel(e, 2); }
    if (e->m->lin_tiled) { dn_kernel(e, "dn_ksum", nb); gf_kernel(e, nb); } // GGUF DeltaNet 投影（一個 NB 一個 library）、合併版 GDN
    if (e->m->mixer.mix_down.gq) gh_kernel(e, "ghc_dsk", nb); // GGUF hyper（同一個 library 有兩段版與三段版）
    if (nb == 2 || nb == 3 || nb == 5) gg_precompile(e, nb); // 驗證的 MoE kernel 以 2、4、8 個 token 編譯（gg_nbv）
    }
    e->vbn = 2;
    e->vb_ready = true;
    return true;
}

static void vb_hyper(qw_engine *e, qw_batch *b, const qw_hyper *H) {
    uint32_t apply = e->bpending ? 1 : 0, want = H->inject ? 1 : 0;
    e->bpending = false;
    if (H->mix_down.gq) { // GGUF：同 decode（部分和暫存借用 bnf）
        hyper_gg(e, b, H, VB, apply, &e->bh, &e->bout, &e->binj, &e->binj2, &e->bnf, &e->bx);
        if (want) { qw_buf t = e->binj; e->binj = e->binj2; e->binj2 = t; }
        return;
    }
    qw_batch_kernel(b, vk(e, "kb_hc_in"));
    qw_batch_buf(b, 0, &e->bh, 0);
    qw_batch_buf(b, 1, &e->bout, 0);
    qw_batch_buf(b, 2, &e->binj, 0);
    bind_t(e, b, 3, H->hc_norm);
    qw_batch_buf(b, 4, &e->bnh, 0);
    qw_batch_buf(b, 5, &e->bnf, 0);
    qw_batch_bytes(b, 6, &apply, 4);
    if (want) bind_t(e, b, 7, H->inject);
    else qw_batch_buf(b, 7, &e->bnf, 0);
    qw_batch_buf(b, 8, &e->binj2, 0);
    qw_batch_bytes(b, 9, &want, 4);
    qw_batch_dispatch(b, QW_HC, VB, 1, 1024, 1, 1);
    if (want) { qw_buf t = e->binj; e->binj = e->binj2; e->binj2 = t; }
    vb_qmv(e, b, &H->mix_down, &e->bnh, &e->bd, false);
    qw_batch_kernel(b, vk(e, "kb_mixup"));
    bind_t(e, b, 0, H->mix_up.w);
    bind_t(e, b, 1, H->mix_up.s);
    bind_t(e, b, 2, H->mix_up.b);
    qw_batch_buf(b, 3, &e->bd, 0);
    qw_batch_buf(b, 4, &e->bnf, 0);
    qw_batch_buf(b, 5, &e->bx, 0);
    qw_batch_dispatch(b, D / 8, 1, 1, 256, 1, 1);
}

static unsigned dn_mask = ~0u; // 量測用（kbench）：只編碼部分步驟：1 投影、2 conv / 正規化、4 GDN、8 gated norm、16 out_proj
static void vb_deltanet(qw_engine *e, qw_batch *b, const qw_deltanet *N, int li) {
    const bool df = dn_defer(e); // 延後寫回：k、log 衰減寫到這一層自己的位置，狀態只讀
    e->dn_vdef = df;
    const int ws = e->dn_lp[li] ^ 1; // 這次的 k / 衰減 / δ 寫到另一組（待重放的那組還要讀）
    if (df && dn_fuse(e, N) && dn_mask == ~0u && VB >= 2) {
        dn_inf(e, b, N, li, &e->bx, &e->bqkv, &e->bz, &e->dnG, dn_goff(li, ws), &e->bbe, VB, e->convx);
        gdn_fstep(e, b, N, li, VB, 1, &e->bqkv, &e->dnG, dn_goff(li, ws), &e->bbe, &e->bgo, &e->bz, &e->bgy, ws);
        dn_out(e, b, N, &e->bgy, &e->bout, VB);
        e->dn_lp[li] = (uint8_t)ws;
        return;
    }
    dn_apply(e, b, li);
    if (df) e->dn_lp[li] = (uint8_t)ws;
    const qw_qlin *Ls[4] = {&N->in_qkv, &N->in_z, &N->in_a, &N->in_b};
    const qw_buf *Ys[4] = {&e->bqkv, &e->bz, &e->ba, &e->bb};
    if (!(dn_mask & 1)) {}
    else if (dn_in_ok(N)) dn_in(e, b, N, &e->bx, &e->bqkv, &e->bz, &e->ba, &e->bb, VB);
    else vb_qmv_multi(e, b, Ls, Ys, 4, &e->bx);
    const size_t cst = (size_t)li * 3 * 10240 * 4, soff = (size_t)li * 48 * 128 * 128 * 4;
    // 第 1 個 token 的狀態就地更新；第 2 個 token 的寫到另一份（猜中才換過去）
    const bool tl = e->m->lin_tiled; // GGUF：V head 交錯順序、A_neg（F32）
    if (dn_mask & 2) {
    qw_batch_kernel(b, vk(e, tl ? "kb_gdn_in_t" : "kb_gdn_in"));
    qw_batch_buf(b, 0, &e->bqkv, 0);
    bind_t(e, b, 1, N->conv1d);
    qw_batch_buf(b, 2, &e->conv_st, cst);
    qw_batch_buf(b, 3, &e->ba, 0);
    qw_batch_buf(b, 4, &e->bb, 0);
    bind_t(e, b, 5, tl ? N->A_neg : N->A_log);
    bind_t(e, b, 6, N->dt_bias);
    qw_batch_buf(b, 7, &e->bgq, 0);
    if (df) qw_batch_buf(b, 8, &e->dnK, dn_koff(li, ws));
    else qw_batch_buf(b, 8, &e->bgk, 0);
    qw_batch_buf(b, 9, &e->bgv, 0);
    if (df) qw_batch_buf(b, 10, &e->dnG, dn_goff(li, ws));
    else qw_batch_buf(b, 10, &e->bla, 0);
    qw_batch_buf(b, 11, &e->bbe, 0);
    for (int t = 1; t < VBMAX; t++) qw_batch_buf(b, 11 + t, t < VB ? &e->convx[t - 1] : &e->conv_st, cst);
    qw_batch_dispatch(b, 16 + 48, 1, 1, 128, 1, 1);
    }
    if ((dn_mask & 4) && df) {
        qw_batch_kernel(b, vk(e, "gdn_stepn_dt"));
        qw_batch_buf(b, 0, &e->bgq, 0);
        qw_batch_buf(b, 1, &e->dnK, dn_koff(li, ws));
        qw_batch_buf(b, 2, &e->bgv, 0);
        qw_batch_buf(b, 3, &e->dnG, dn_goff(li, ws));
        qw_batch_buf(b, 4, &e->bbe, 0);
        qw_batch_buf(b, 5, &e->S, soff);
        qw_batch_buf(b, 6, &e->bgo, 0);
        qw_batch_buf(b, 7, &e->bssp, 0);
        qw_batch_buf(b, 8, &e->dnD, dn_doff(li, ws));
        qw_batch_dispatch(b, 48, 128 / 32, 1, 256, 1, 1);
    } else if (dn_mask & 4) {
    qw_batch_kernel(b, vk(e, tl ? "gdn_stepn_t" : "gdn_stepn"));
    qw_batch_buf(b, 0, &e->bgq, 0);
    qw_batch_buf(b, 1, &e->bgk, 0);
    qw_batch_buf(b, 2, &e->bgv, 0);
    qw_batch_buf(b, 3, &e->bla, 0);
    qw_batch_buf(b, 4, &e->bbe, 0);
    qw_batch_buf(b, 5, &e->S, soff);
    qw_batch_buf(b, 6, &e->bgo, 0);
    qw_batch_buf(b, 7, &e->bssp, 0);
    for (int t = 1; t < VBMAX; t++) qw_batch_buf(b, 7 + t, t < VB ? &e->Sx[t - 1] : &e->S, soff);
    qw_batch_dispatch(b, 48, 128 / 32, 1, 256, 1, 1);
    }
    if ((dn_mask & 8) && vb_xf(e, &N->out)) vb_xform(e, b, 1, &e->bgo, &e->bz, &e->bssp, N->norm, &e->bgy);
    else if (dn_mask & 8) {
    qw_batch_kernel(b, vk(e, "kb_normgate"));
    qw_batch_buf(b, 0, &e->bgo, 0);
    qw_batch_buf(b, 1, &e->bz, 0);
    bind_t(e, b, 2, N->norm);
    qw_batch_buf(b, 3, &e->bgy, 0);
    qw_batch_dispatch(b, 48, VB, 1, 128, 1, 1);
    }
    if (!(dn_mask & 16)) {}
    else if (dn_out_ok(N)) dn_out(e, b, N, &e->bgy, &e->bout, VB);
    else vb_qmv(e, b, &N->out, &e->bgy, &e->bout, false);
}

// QSA 層（decode：nb = 1；驗證：nb = VB）。投影 → qsa_prep → [稀疏：qsa_score3 → qsa_topk3] → 核心 → o_proj。
// GGUF：核心是 qsa_attn5 → qsa_merge4（含 gate）；MLX：沿用 qsa_attn2 + 合併 + gate 併進 o_proj（qmv_xf2），輸出與之前逐位元相同。
// 每個 token 的計算不受 nb 影響，所以驗證與逐 token decode 逐位元相同（QW_ATT_TEST 檢查）
static void qsa_layer(qw_engine *e, qw_batch *b, const qw_attn *A, int ai, uint32_t p0, int nb, const qsa_io *io) {
    const size_t Lc = (size_t)e->kvcap;
    const size_t koff = (size_t)ai * Lc * 2 * 256 * 2, roff = (size_t)ai * Lc * 128 * 4, boff = (size_t)ai * (Lc / 4 + 1) * 128 * 2;
    if (at_mask & 1) {
        const qw_qlin *Ls[4] = {&A->q, &A->k, &A->v, &A->idx_qk};
        const qw_buf *Ys[4] = {io->qg, io->kk, io->vv, io->iqk};
        if (!qmv_ggm(e, b, Ls, Ys, 4, io->x, nb)) {
            if (nb == 1) qmv_multi(e, b, Ls, Ys, 4, io->x);
            else vb_qmv_multi(e, b, Ls, Ys, 4, io->x);
        }
    }
    if (at_mask & 2) {
        qw_batch_kernel(b, e->q3_prep[A->nf32 ? 1 : 0]);
        qw_batch_buf(b, 0, io->qg, io->oqg);
        qw_batch_buf(b, 1, io->kk, io->okk);
        qw_batch_buf(b, 2, io->vv, io->ovv);
        bind_t(e, b, 3, A->q_norm);
        bind_t(e, b, 4, A->k_norm);
        qw_batch_buf(b, 5, io->aq, 0);
        qw_batch_buf(b, 6, &e->Kc, koff);
        qw_batch_buf(b, 7, &e->Vc, koff);
        qw_batch_bytes(b, 8, &p0, 4);
        qw_batch_buf(b, 9, io->iqk, io->oiqk);
        bind_t(e, b, 10, A->idx_q_norm);
        bind_t(e, b, 11, A->idx_k_norm);
        qw_batch_buf(b, 12, io->iq, 0);
        qw_batch_buf(b, 13, &e->rawk, roff);
        qw_batch_buf(b, 14, &e->kblk, boff);
        qw_batch_dispatch(b, 24 + 2 + 4 + 1, (unsigned)nb, 1, 256, 1, 1);
    }
    qsa_args aa[VBMAX];
    uint32_t nvmax = 0, nt = (uint32_t)nb, sstride = (uint32_t)(e->max_ctx / 4 + 1);
    bool sparse = false;
    for (int t = 0; t < nb; t++) {
        const uint32_t pos = p0 + (uint32_t)t;
        qsa_args a = {pos + 1, pos, (pos + 1) / QW_IDX_CR, pos + 1, pos + 1 > QW_IDX_BUDGET};
        if (a.sparse) {
            const uint32_t ns = a.nvis < 512 ? a.nvis : 512;
            a.ntok = ns * QW_IDX_CR + (pos + 1 - a.nvis * QW_IDX_CR);
            sparse = true;
            nvmax = a.nvis > nvmax ? a.nvis : nvmax;
        }
        aa[t] = a;
    }
    if (sparse && (at_mask & 4)) { // 驗證 5–8 個 token：qsa_score4（同結果，交換次數少）
        qw_batch_kernel(b, e->q3_score4 && nb >= 5 && nb <= 8 ? e->q3_score4 : e->q3_score);
        qw_batch_buf(b, 0, io->iq, 0);
        qw_batch_buf(b, 1, &e->kblk, boff);
        qw_batch_buf(b, 2, &e->q3sc, 0);
        qw_batch_bytes(b, 3, aa, sizeof(qsa_args) * (size_t)nb);
        qw_batch_bytes(b, 4, &nt, 4);
        qw_batch_bytes(b, 5, &sstride, 4);
        qw_batch_buf(b, 6, &e->q3hist, 0);
        qw_batch_dispatch(b, (nvmax + 7) / 8, 1, 1, 256, 1, 1);
    }
    if (sparse && (at_mask & 8)) {
        const int need = (int)((nvmax + 1023) / 1024), mi = need <= 8 ? 0 : need <= 16 ? 1 : 2;
        qw_batch_kernel(b, e->q3_topk[mi]);
        qw_batch_buf(b, 0, &e->q3sc, 0);
        qw_batch_buf(b, 1, &e->q3sel, 0);
        qw_batch_buf(b, 2, &e->q3nsel, 0);
        qw_batch_bytes(b, 3, aa, sizeof(qsa_args) * (size_t)nb);
        qw_batch_bytes(b, 4, &sstride, 4);
        qw_batch_buf(b, 5, &e->q3hist, 0);
        qw_batch_dispatch(b, (unsigned)nb, 1, 1, 1024, 1, 1);
    }
    if (A->q.gq && !getenv("QW_ATT_OLDCORE")) { // GGUF：qsa_attn5 → qsa_merge4（含 gate）→ o_proj（QW_ATT_OLDCORE：MLX 的核心，對照用）
        // 核心：qsa_attn5（QW_ATT4=1 改用 qsa_attn4 對照）
        const bool use5 = q5_on();
        if ((at_mask & 16) && !use5) {
            const uint32_t donly = 0;
            qw_batch_kernel(b, e->q4_attn);
            qw_batch_buf(b, 0, io->aq, 0);
            qw_batch_buf(b, 1, &e->Kc, koff);
            qw_batch_buf(b, 2, &e->Vc, koff);
            qw_batch_buf(b, 3, &e->q3sel, 0);
            qw_batch_buf(b, 4, &e->q3nsel, 0);
            qw_batch_buf(b, 5, &e->q4part, 0);
            qw_batch_bytes(b, 6, aa, sizeof(qsa_args) * (size_t)nb);
            qw_batch_bytes(b, 7, &donly, 4);
            qw_batch_dispatch(b, (unsigned)q4_nch(), 2, (unsigned)nb, 32 * 12, 1, 1);
        }
        if ((at_mask & 16) && use5) {
            static int dostat = -1;
            if (dostat < 0) dostat = getenv("QW_ATT5_STATS") != NULL;
            const uint32_t ds = (uint32_t)dostat && nb > 1, tt = (uint32_t)q5_tt();
            qw_batch_kernel(b, e->q5_attn);
            qw_batch_buf(b, 0, io->aq, 0);
            qw_batch_buf(b, 1, &e->Kc, koff);
            qw_batch_buf(b, 2, &e->Vc, koff);
            qw_batch_buf(b, 3, &e->q3sel, 0);
            qw_batch_buf(b, 4, &e->q4part, 0);
            qw_batch_bytes(b, 5, aa, sizeof(qsa_args) * (size_t)nb);
            qw_batch_bytes(b, 6, &nt, 4);
            qw_batch_buf(b, 7, &e->q5stat, 0);
            qw_batch_bytes(b, 8, &ds, 4);
            qw_batch_dispatch(b, (unsigned)q4_nch(), 2, (nt + tt - 1) / tt, 32 * 12, 1, 1);
            if (nb > 1 && e->q5chk.ptr) { // 除錯：每個 token 單獨（像 decode）再算一次，與驗證的 part 逐位元比較
                const uint32_t one = 1, z0 = 0, npart = (uint32_t)nb * (uint32_t)q4_nch() * 24 * 258;
                for (int t = 0; t < nb; t++) {
                    qw_batch_kernel(b, e->q5_attn);
                    qw_batch_buf(b, 0, io->aq, (size_t)t * 24 * 256 * 2);
                    qw_batch_buf(b, 1, &e->Kc, koff);
                    qw_batch_buf(b, 2, &e->Vc, koff);
                    qw_batch_buf(b, 3, &e->q3sel, (size_t)t * 512 * 4);
                    qw_batch_buf(b, 4, &e->q5chk, (size_t)t * q4_nch() * 24 * 258 * 4);
                    qw_batch_bytes(b, 5, &aa[t], sizeof(qsa_args));
                    qw_batch_bytes(b, 6, &one, 4);
                    qw_batch_buf(b, 7, &e->q5stat, 0);
                    qw_batch_bytes(b, 8, &z0, 4);
                    qw_batch_dispatch(b, (unsigned)q4_nch(), 2, 1, 32 * 12, 1, 1);
                }
                qw_batch_kernel(b, e->q5_cmp);
                qw_batch_buf(b, 0, &e->q4part, 0);
                qw_batch_buf(b, 1, &e->q5chk, 0);
                qw_batch_bytes(b, 2, &npart, 4);
                qw_batch_buf(b, 3, &e->q5stat, 0);
                d1(b, npart, 256);
            }
        }
        if (at_mask & 32) {
            qw_batch_kernel(b, e->q4_merge);
            qw_batch_buf(b, 0, &e->q4part, 0);
            qw_batch_buf(b, 1, io->qg, io->oqg);
            qw_batch_buf(b, 2, io->ay, io->oay);
            qw_batch_dispatch(b, 24, (unsigned)nb, 1, 256, 1, 1);
        }
        if (!(at_mask & 128)) return;
        const qw_qlin *Lo = &A->o;
        const qw_buf *Yo = io->out;
        if (!qmv_ggm(e, b, &Lo, &Yo, 1, io->ay, nb)) {
            if (nb == 1) qmv(e, b, &A->o, io->ay, 0, io->out, false);
            else vb_qmv(e, b, &A->o, io->ay, io->out, false);
        }
        return;
    }
    // MLX：舊的核心（qsa_attn2 / 2b）
    const qw_buf *part = nb == 1 ? &e->part : &e->bpart;
    if (at_mask & 16) {
        qw_batch_kernel(b, nb == 1 ? e->qsa_attn : pf_need(e, "qsa_attn2b"));
        qw_batch_buf(b, 0, io->aq, 0);
        qw_batch_buf(b, 1, &e->Kc, koff);
        qw_batch_buf(b, 2, &e->Vc, koff);
        qw_batch_buf(b, 3, &e->q3sel, 0);
        qw_batch_buf(b, 4, &e->q3nsel, 0);
        qw_batch_buf(b, 5, part, 0);
        qw_batch_bytes(b, 6, aa, sizeof(qsa_args) * (size_t)nb);
        qw_batch_dispatch(b, QSA_NCH, 2, (unsigned)nb, 32 * 12, 1, 1);
    }
    uint32_t nch = QSA_NCH;
    if (at_mask & 32) {
        qw_batch_kernel(b, nb == 1 ? e->qsa_merge : pf_need(e, "qsa_merge2b"));
        qw_batch_buf(b, 0, part, 0);
        qw_batch_buf(b, 1, io->ao, io->oao);
        qw_batch_bytes(b, 2, &nch, 4);
        qw_batch_dispatch(b, 24, (unsigned)nb, 1, 256, 1, 1);
    }
    if (nb == 1 && mlx_xf(&A->o)) { // output gate 併進 o_proj 的輸入（--exact-spec 時驗證用 vb_xform，算式相同）
        if (!(at_mask & 128)) return;
        qmv_args a2 = {(uint32_t)A->o.out, 0, (uint32_t)A->o.out, 0};
        qw_batch_kernel(b, A->o.bits == 4 ? e->qmv_xf2_4 : e->qmv_xf2);
        bind_t(e, b, 0, A->o.w); bind_t(e, b, 1, A->o.s); bind_t(e, b, 2, A->o.b);
        qw_batch_buf(b, 3, io->ao, 0);
        qw_batch_buf(b, 4, io->out, 0);
        qw_batch_buf(b, 5, io->qg, 0);
        qw_batch_bytes(b, 6, &a2, sizeof a2);
        qw_batch_buf(b, 7, &e->ssp, 0);
        bind_t(e, b, 8, A->q_norm);
        qw_batch_dispatch(b, (a2.N + 7) / 8, 1, 1, 256, 1, 1);
        return;
    }
    if ((at_mask & 64) && nb > 1 && vb_xf(e, &A->o)) vb_xform(e, b, 2, io->ao, io->qg, &e->bssp, A->q_norm, io->ay); // 同 decode 的 qmv_xf2
    else if (at_mask & 64) {
        qw_batch_kernel(b, pf_need(e, "kb_attn_gate"));
        qw_batch_buf(b, 0, io->ao, 0);
        qw_batch_buf(b, 1, io->qg, 0);
        qw_batch_buf(b, 2, io->ay, 0);
        qw_batch_bytes(b, 3, &nt, 4);
        d1(b, nt * 6144, 256);
    }
    if (!(at_mask & 128)) return;
    const qw_qlin *Lo = &A->o;
    const qw_buf *Yo = io->out;
    if (qmv_ggm(e, b, &Lo, &Yo, 1, io->ay, nb)) return;
    if (nb == 1) qmv(e, b, &A->o, io->ay, 0, io->out, false);
    else vb_qmv(e, b, &A->o, io->ay, io->out, false);
}

static void vb_attention(qw_engine *e, qw_batch *b, const qw_attn *A, int ai, int p0) {
    const qsa_io io = {&e->bx, &e->bqg, &e->bkk, &e->bvv, &e->biqk, &e->baq, &e->biq, &e->bao, &e->bay, &e->bout};
    qsa_layer(e, b, A, ai, (uint32_t)p0, VB, &io);
}

static void vb_moe(qw_engine *e, qw_batch *b, const qw_moe *M) {
    if (M->giq && !gg_old()) { gg_moe(e, b, M, VB, &e->bx, &e->blog, &e->bmsel, &e->bout); return; }
    if (M->router.gq) gg_router(e, b, M, VB, &e->bx, &e->blog);
    else {
        qw_batch_kernel(b, vk(e, "moe_router_b"));
        bind_t(e, b, 0, M->router.w);
        bind_t(e, b, 1, M->router.s);
        bind_t(e, b, 2, M->router.b);
        bind_t(e, b, 3, M->sh_gate_w);
        qw_batch_buf(b, 4, &e->bx, 0);
        qw_batch_buf(b, 5, &e->blog, 0);
        qw_batch_dispatch(b, (QW_EXPERTS + 1 + 7) / 8, VB, 1, 256, 1, 1);
    }
    if (M->giq) {
        qw_batch_kernel(b, iq_kernel(e, "moe_gateup_iqb", M->fgu, M->fd, VB));
        bind_t(e, b, 0, M->giq);
        bind_t(e, b, 3, M->uiq);
        bind_t(e, b, 6, M->sh_gate.w); bind_t(e, b, 7, M->sh_gate.s); bind_t(e, b, 8, M->sh_gate.b);
        bind_t(e, b, 9, M->sh_up.w); bind_t(e, b, 10, M->sh_up.s); bind_t(e, b, 11, M->sh_up.b);
        qw_batch_buf(b, 12, &e->bx, 0);
        qw_batch_buf(b, 13, &e->blog, 0);
        qw_batch_buf(b, 14, &e->bH, 0);
        qw_batch_dispatch(b, QW_MOE_INTER / 8, VB * (QW_TOPK + 1), 1, 256, 1, 1);
        qw_batch_kernel(b, iq_kernel(e, "moe_down_iqb", M->fgu, M->fd, VB));
        bind_t(e, b, 0, M->diq);
        bind_t(e, b, 3, M->sh_down.w); bind_t(e, b, 4, M->sh_down.s); bind_t(e, b, 5, M->sh_down.b);
        qw_batch_buf(b, 6, &e->bH, 0);
        qw_batch_buf(b, 7, &e->blog, 0);
        qw_batch_buf(b, 9, &e->bout, 0);
        qw_batch_dispatch(b, D / 8, VB, 1, 256, 1, 1);
        return;
    }
    static int moe_tc_min = -1; // 一輪驗證至少幾個 token 才走矩陣單元版（QW_MOE_TC=最少幾個，預設 0 = 關）
    if (moe_tc_min < 0) moe_tc_min = getenv("QW_MOE_TC") ? atoi(getenv("QW_MOE_TC")) : 0; // 實驗：kbench（8 個相同 token）339→182 µs，真實生成（不重複 expert 約 54/80）反而較慢
    if (moe_tc_min > 0 && VB >= moe_tc_min && !M->giq) { // 矩陣單元版（kernels/moe_tc.metal）：expert 去重、權重解一次跟 8 列一起乘
        char nm[2][64], defs[512], err[8192];
        snprintf(nm[0], sizeof nm[0], "moe_gu_tc_n%d", VB);
        snprintf(nm[1], sizeof nm[1], "moe_dn_tc_n%d", VB);
        if (!qw_kernel_get(e->g, nm[0])) {
            snprintf(defs, sizeof defs, "#define moe_gu_tc %s\n#define moe_dn_tc %s\n#define NB %d\n#define MTOPK %d\n#define MD %d\n#define MI %d\n#define NSG 8\n",
                     nm[0], nm[1], VB, QW_TOPK, D, QW_MOE_INTER);
            if (!compile(e, defs, src_moe_tc, sizeof src_moe_tc, err, sizeof err)) { fprintf(stderr, "編譯 %s 失敗：%s\n", nm[0], err); exit(1); }
        }
        qw_batch_kernel(b, vk(e, "moe_sel_b"));
        qw_batch_buf(b, 0, &e->blog, 0);
        qw_batch_buf(b, 1, &e->bmsel, 0);
        qw_batch_dispatch(b, 1, 1, 1, 32 * VB, 1, 1);
        qw_batch_kernel(b, qw_kernel_get(e->g, nm[0]));
        bind_t(e, b, 0, M->gate.w); bind_t(e, b, 1, M->gate.s); bind_t(e, b, 2, M->gate.b);
        bind_t(e, b, 3, M->up.w); bind_t(e, b, 4, M->up.s); bind_t(e, b, 5, M->up.b);
        bind_t(e, b, 6, M->sh_gate.w); bind_t(e, b, 7, M->sh_gate.s); bind_t(e, b, 8, M->sh_gate.b);
        bind_t(e, b, 9, M->sh_up.w); bind_t(e, b, 10, M->sh_up.s); bind_t(e, b, 11, M->sh_up.b);
        qw_batch_buf(b, 12, &e->bx, 0);
        qw_batch_buf(b, 13, &e->bmsel, 0);
        qw_batch_buf(b, 14, &e->bHu, 0);
        qw_batch_dispatch(b, QW_MOE_INTER / 32, VB * QW_TOPK + 1, 1, 256, 1, 1);
        qw_batch_kernel(b, qw_kernel_get(e->g, nm[1]));
        bind_t(e, b, 0, M->down.w); bind_t(e, b, 1, M->down.s); bind_t(e, b, 2, M->down.b);
        bind_t(e, b, 3, M->sh_down.w); bind_t(e, b, 4, M->sh_down.s); bind_t(e, b, 5, M->sh_down.b);
        qw_batch_buf(b, 6, &e->bHu, 0);
        qw_batch_buf(b, 7, &e->bmsel, 0);
        qw_batch_buf(b, 9, &e->bout, 0);
        qw_batch_dispatch(b, D / 32, 1, 1, 256, 1, 1);
        return;
    }
    if (getenv("QW_MOE_DD")) { // expert 去重（kernels/moe_decode.metal 的 *_db）；純量版，比原本慢，保留作對照
        qw_batch_kernel(b, vk(e, "moe_sel_b"));
        qw_batch_buf(b, 0, &e->blog, 0);
        qw_batch_buf(b, 1, &e->bmsel, 0);
        qw_batch_dispatch(b, 1, 1, 1, 32 * VB, 1, 1);
        qw_batch_kernel(b, vk(e, "moe_gateup_db"));
        bind_t(e, b, 0, M->gate.w); bind_t(e, b, 1, M->gate.s); bind_t(e, b, 2, M->gate.b);
        bind_t(e, b, 3, M->up.w); bind_t(e, b, 4, M->up.s); bind_t(e, b, 5, M->up.b);
        bind_t(e, b, 6, M->sh_gate.w); bind_t(e, b, 7, M->sh_gate.s); bind_t(e, b, 8, M->sh_gate.b);
        bind_t(e, b, 9, M->sh_up.w); bind_t(e, b, 10, M->sh_up.s); bind_t(e, b, 11, M->sh_up.b);
        qw_batch_buf(b, 12, &e->bx, 0);
        qw_batch_buf(b, 13, &e->bmsel, 0);
        qw_batch_buf(b, 14, &e->bH, 0);
        qw_batch_dispatch(b, QW_MOE_INTER / 8, VB * QW_TOPK + 1, 1, 256, 1, 1);
        qw_batch_kernel(b, vk(e, "moe_down_db"));
        bind_t(e, b, 0, M->down.w); bind_t(e, b, 1, M->down.s); bind_t(e, b, 2, M->down.b);
        bind_t(e, b, 3, M->sh_down.w); bind_t(e, b, 4, M->sh_down.s); bind_t(e, b, 5, M->sh_down.b);
        qw_batch_buf(b, 6, &e->bH, 0);
        qw_batch_buf(b, 7, &e->bmsel, 0);
        qw_batch_buf(b, 9, &e->bout, 0);
        qw_batch_dispatch(b, D / 8, 1, 1, 256, 1, 1);
        return;
    }
    qw_batch_kernel(b, vk(e, "moe_gateup_fb"));
    bind_t(e, b, 0, M->gate.w); bind_t(e, b, 1, M->gate.s); bind_t(e, b, 2, M->gate.b);
    bind_t(e, b, 3, M->up.w); bind_t(e, b, 4, M->up.s); bind_t(e, b, 5, M->up.b);
    bind_t(e, b, 6, M->sh_gate.w); bind_t(e, b, 7, M->sh_gate.s); bind_t(e, b, 8, M->sh_gate.b);
    bind_t(e, b, 9, M->sh_up.w); bind_t(e, b, 10, M->sh_up.s); bind_t(e, b, 11, M->sh_up.b);
    qw_batch_buf(b, 12, &e->bx, 0);
    qw_batch_buf(b, 13, &e->blog, 0);
    qw_batch_buf(b, 14, &e->bH, 0);
    qw_batch_dispatch(b, QW_MOE_INTER / 8, VB * (QW_TOPK + 1), 1, 256, 1, 1);
    qw_batch_kernel(b, vk(e, "moe_down_fb"));
    bind_t(e, b, 0, M->down.w); bind_t(e, b, 1, M->down.s); bind_t(e, b, 2, M->down.b);
    bind_t(e, b, 3, M->sh_down.w); bind_t(e, b, 4, M->sh_down.s); bind_t(e, b, 5, M->sh_down.b);
    qw_batch_buf(b, 6, &e->bH, 0);
    qw_batch_buf(b, 7, &e->blog, 0);
    qw_batch_buf(b, 9, &e->bout, 0);
    qw_batch_dispatch(b, D / 8, VB, 1, 256, 1, 1);
}

// PLE（只有第 2 層）：逐 token，用單 token 的流程；第 2 個 token 的 conv 狀態寫到另一份
static void vb_ple(qw_engine *e, qw_batch *b, const qw_ple *P, int p0) {
    ng_gather(e, P, p0 + 2, VB);
    if (e->m->ple_tab) { // GGUF：key / value 投影不依賴 h，VB 個 token 一次算
        e->ng_base = 0;
        ple_gg_proj(e, b, P, VB);
    }
    for (int t = 0; t < VB; t++) {
        e->ng_base = t * 16;
        vb_copy(e, b, &e->bh, (size_t)t * HCD * 4, &e->h, 0, HCD);
        // token t（t ≥ 1）從前一個 token 之後的狀態出發，寫到自己那份
        if (t >= 1) vb_copy(e, b, t == 1 ? &e->ple_st : &e->plex[t - 2], 0, &e->plex[t - 1], 0, 9 * HCD);
        qw_buf keep = e->ple_st;
        if (t >= 1) e->ple_st = e->plex[t - 1];
        const int ps = e->pos;
        e->pos = p0 + t;
        if (e->m->ple_tab) ple_pre_t = t;
        ple(e, b, P);
        ple_pre_t = -1;
        e->pos = ps;
        e->ple_st = keep;
        vb_copy(e, b, &e->h, 0, &e->bh, (size_t)t * HCD * 4, HCD);
    }
}

static void vb_argmax(qw_engine *e, qw_batch *b, int ridx);
static void vb_mtp(qw_engine *e, qw_batch *b, int p0);

// 驗證：位置 p0、p0+1 的 token 已在 tok[p0+2]、tok[p0+3]。算完 res[t] = argmax（位置 p0+t 的下一個 token）
static void vb_encode(qw_engine *e, qw_batch *b, int p0) {
    uint32_t p = (uint32_t)p0 + 2;
    if (e->m->embed_gq) gg_embed_rows(e, b, &e->tok, p, (uint32_t)VB, QW_HC, HCD, &e->bh);
    else {
    qw_batch_kernel(b, vk(e, "kb_embed"));
    bind_t(e, b, 0, e->m->embed_w);
    bind_t(e, b, 1, e->m->embed_s);
    bind_t(e, b, 2, e->m->embed_b);
    qw_batch_buf(b, 3, &e->tok, 0);
    qw_batch_bytes(b, 4, &p, 4);
    qw_batch_buf(b, 5, &e->bh, 0);
    qw_batch_dispatch(b, D / 256, VB, 1, 256, 1, 1);
    }
    e->bpending = false;
    int li = 0, ai = 0;
    for (int l = 0; l < QW_LAYERS; l++) {
        const qw_layer *L = &e->m->layers[l];
        if (L->has_ple) {
            if (e->bpending) { // 先把殘差加進去：借用 hyper 的 apply（不需要 inject 的輸出）
                uint32_t n = 1, z = 0;
                (void)n; (void)z;
                qw_batch_kernel(b, vk(e, "kb_hc_in"));
                uint32_t apply = 1, want = 0;
                qw_batch_buf(b, 0, &e->bh, 0);
                qw_batch_buf(b, 1, &e->bout, 0);
                qw_batch_buf(b, 2, &e->binj, 0);
                bind_t(e, b, 3, L->attn_hc.hc_norm);
                qw_batch_buf(b, 4, &e->bnh, 0);
                qw_batch_buf(b, 5, &e->bnf, 0);
                qw_batch_bytes(b, 6, &apply, 4);
                qw_batch_buf(b, 7, &e->bnf, 0);
                qw_batch_buf(b, 8, &e->binj2, 0);
                qw_batch_bytes(b, 9, &want, 4);
                qw_batch_dispatch(b, QW_HC, VB, 1, 1024, 1, 1);
                e->bpending = false;
            }
            vb_ple(e, b, &L->ple, p0);
        }
        static int vskip = -1; // QW_VB_SKIP（只給 qw_engine_vb_bench 量時間用，結果不對）：1 moe、2 deltanet、4 attention、8 hyper、16 head、64 mtp
        if (vskip < 0) vskip = getenv("QW_VB_SKIP") ? atoi(getenv("QW_VB_SKIP")) : 0;
        if (!(vskip & 8)) vb_hyper(e, b, &L->attn_hc);
        if (!getenv("QW_DF_NOTAP_VB")) df_tap_after(e, b, l, &e->bh, VB);
        if (L->full_attn) { if (!(vskip & 4)) vb_attention(e, b, &L->att, ai, p0); ai++; }
        else { if (!(vskip & 2)) vb_deltanet(e, b, &L->lin, li); li++; }
        e->bpending = true;
        if (trace_on()) for (int t = 0; t < VB; t++) trace_copy(e, b, &e->bh, (size_t)t * HCD * 4, &e->bout, (size_t)t * D * 2, t, l, 0);
        if (!(vskip & 8)) vb_hyper(e, b, &L->mlp_hc);
        if (!(vskip & 1)) vb_moe(e, b, &L->moe);
        e->bpending = true;
        if (trace_on()) for (int t = 0; t < VB; t++) trace_copy(e, b, &e->bh, (size_t)t * HCD * 4, &e->bout, (size_t)t * D * 2, t, l, 1);
        if (e->vb_flush >> l & 1) qw_batch_flush(b);
    }
    vb_hyper(e, b, &e->m->mixer); // 殘差加進 bh（= mixer 之前的 4 條流，給 MTP 用），並算出 x
    e->bpending = false;
    static int vskip2 = -1;
    if (vskip2 < 0) vskip2 = getenv("QW_VB_SKIP") ? atoi(getenv("QW_VB_SKIP")) : 0;
    if (!(vskip2 & 16)) { vb_qmv(e, b, &e->m->lm_head, &e->bx, &e->blg, true); vb_argmax(e, b, 0); }
    if (!(vskip2 & 64) && e->m->has_mtp && (!e->df_mode || getenv("QW_DF_MTP"))) vb_mtp(e, b, p0);
}

static void vb_argmax_from(qw_engine *e, qw_batch *b, int ridx, const qw_buf *lgb) {
    for (int t = 0; t < VB; t++) {
        uint32_t n = QW_VOCAB, m = (QW_VOCAB + 1023) / 1024, q = (uint32_t)(ridx + t);
        qw_batch_kernel(b, e->k_argmax1);
        qw_batch_buf(b, 0, lgb, (size_t)t * QW_VOCAB * 4);
        qw_batch_buf(b, 1, &e->amv, 0);
        qw_batch_buf(b, 2, &e->ami, 0);
        qw_batch_bytes(b, 3, &n, 4);
        qw_batch_dispatch(b, m, 1, 1, 1024, 1, 1);
        qw_batch_kernel(b, e->k_argmax2);
        qw_batch_buf(b, 0, &e->amv, 0);
        qw_batch_buf(b, 1, &e->ami, 0);
        qw_batch_bytes(b, 2, &m, 4);
        qw_batch_buf(b, 3, &e->bres, 0);
        qw_batch_bytes(b, 4, &q, 4);
        qw_batch_dispatch(b, 1, 1, 1, 1024, 1, 1);
    }
}

static void vb_argmax(qw_engine *e, qw_batch *b, int ridx) { vb_argmax_from(e, b, ridx, &e->blg); }

// MTP 兩個位置一起跑（接在驗證之後、同一個 command buffer）：
// token t（= 驗證結果 bres[t]）配主模型位置 p0+t 的 mixer 前殘差流（bh[t]），在位置 p0+t 跑 MTP 層。
// 結果 bres[2 + t]：猜錯時用 bres[2]（位置 p0 的猜測），猜中時用 bres[3]。
static void vb_mtp(qw_engine *e, qw_batch *b, int p0) {
    const qw_model *M = e->m;
    uint32_t z = 0, n1 = D, n2 = HCD;
    qw_batch_kernel(b, vk(e, "kb_embed"));
    bind_t(e, b, 0, M->embed_w);
    bind_t(e, b, 1, M->embed_s);
    bind_t(e, b, 2, M->embed_b);
    qw_batch_buf(b, 3, &e->bres, 0);
    qw_batch_bytes(b, 4, &z, 4);
    qw_batch_buf(b, 5, &e->bme, 0);
    qw_batch_dispatch(b, D / 256, VB, 1, 256, 1, 1);
    for (int t = 0; t < VB; t++) {
        qw_batch_kernel(b, e->k_norm1);
        qw_batch_buf(b, 0, &e->bme, (size_t)t * HCD * 4);
        bind_t(e, b, 1, M->mtp_norm_emb);
        qw_batch_buf(b, 2, &e->bmn, (size_t)t * D * 2);
        qw_batch_bytes(b, 3, &n1, 4);
        qw_batch_dispatch(b, 1, 1, 1, 1024, 1, 1);
        qw_batch_kernel(b, e->k_norm1);
        qw_batch_buf(b, 0, &e->bh, (size_t)t * HCD * 4);
        bind_t(e, b, 1, M->mtp_norm_hidden);
        qw_batch_buf(b, 2, &e->bmnh, (size_t)t * HCD * 2);
        qw_batch_bytes(b, 3, &n2, 4);
        qw_batch_dispatch(b, 1, 1, 1, 1024, 1, 1);
    }
    vb_qmv(e, b, &M->mtp_fc_emb, &e->bmn, &e->bmfe, false);
    // fc_hidden：[2][4][D] 共 8 個向量，兩兩一組
    for (int q = 0; q < QW_HC; q++) // VB·HC 個向量，每次 VB 個
        vb_qmv_o(e, b, &M->mtp_fc_hidden, &e->bmnh, (size_t)q * VB * D * 2, &e->bmfh, (size_t)q * VB * D * 2, false);
    for (int t = 0; t < VB; t++) {
        qw_batch_kernel(b, e->k_mtp_combine);
        qw_batch_buf(b, 0, &e->bmfh, (size_t)t * HCD * 2);
        qw_batch_buf(b, 1, &e->bmfe, (size_t)t * D * 2);
        qw_batch_buf(b, 2, &e->bmh, (size_t)t * HCD * 4);
        d1(b, HCD, 256);
    }
    qw_buf hs = e->bh;
    e->bh = e->bmh;
    e->bpending = false;
    vb_hyper(e, b, &M->mtp.attn_hc);
    vb_attention(e, b, &M->mtp.att, NATT, p0);
    e->bpending = true;
    vb_hyper(e, b, &M->mtp.mlp_hc);
    vb_moe(e, b, &M->mtp.moe);
    e->bpending = true;
    vb_hyper(e, b, &M->mtp_mixer);
    e->bpending = false;
    vb_qmv(e, b, &M->lm_head, &e->bx, &e->bmlg, true); // 不能蓋掉主模型的 logits（取樣要用）
    vb_argmax_from(e, b, VB, &e->bmlg);
    e->bh = hs;
}

// MTP 推測解碼：先逐 token 處理 prompt（每步同時填 MTP 的 cache），之後每輪驗證 [確定的 token, MTP 猜的 token]。
// 回傳生成的 token 數；stats：[輪數, 猜中數]
// ---------------------------------------------------------------- 取樣
static uint64_t rng_next(uint64_t *x) { // splitmix64
    uint64_t z = (*x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static double rng_unif(uint64_t *x) { return (double)(rng_next(x) >> 11) * (1.0 / 9007199254740992.0); }

enum { DIST_MAX = 256 };
typedef struct { int n; int32_t id[DIST_MAX]; float p[DIST_MAX]; } dist;

// 候選分佈：前 k 個（k ≤ 256）→ softmax(logit / T) → top-p 截斷 → 重新正規化
static void make_dist(const float *lg, const qw_gen *g, dist *d) {
    const int k = g->top_k > 0 && g->top_k < DIST_MAX ? g->top_k : DIST_MAX;
    int n = 0;
    for (int v = 0; v < QW_VOCAB; v++) { // 插入排序維持前 k 大（大多數 logit 一比就跳過）
        const float x = lg[v];
        if (n == k && x <= lg[d->id[n - 1]]) continue;
        int i = n < k ? n++ : n - 1;
        while (i > 0 && lg[d->id[i - 1]] < x) { d->id[i] = d->id[i - 1]; i--; }
        d->id[i] = v;
    }
    const float T = g->temperature, mx = lg[d->id[0]];
    double sum = 0;
    for (int i = 0; i < n; i++) { d->p[i] = expf((lg[d->id[i]] - mx) / T); sum += d->p[i]; }
    double c = 0;
    int keep = n;
    for (int i = 0; i < n; i++) {
        d->p[i] = (float)(d->p[i] / sum);
        c += d->p[i];
        if (g->top_p > 0 && g->top_p < 1 && c >= g->top_p) { keep = i + 1; break; }
    }
    double s2 = 0;
    for (int i = 0; i < keep; i++) s2 += d->p[i];
    for (int i = 0; i < keep; i++) d->p[i] = (float)(d->p[i] / s2);
    d->n = keep;
}

static float dist_p(const dist *d, int32_t x) {
    for (int i = 0; i < d->n; i++) if (d->id[i] == x) return d->p[i];
    return 0;
}

// 抽一個；exclude ≥ 0 時排除該 token（推測取樣拒絕後的重抽，等同 (p − q)+ 正規化，q 是猜測的 one-hot）
static int32_t dist_draw(const dist *d, uint64_t *rng, int32_t exclude) {
    double tot = 0;
    for (int i = 0; i < d->n; i++) if (d->id[i] != exclude) tot += d->p[i];
    if (tot <= 0) return d->id[0];
    double u = rng_unif(rng) * tot;
    for (int i = 0; i < d->n; i++) {
        if (d->id[i] == exclude) continue;
        u -= d->p[i];
        if (u <= 0) return d->id[i];
    }
    for (int i = d->n - 1; i >= 0; i--) if (d->id[i] != exclude) return d->id[i];
    return d->id[0];
}

static int32_t argmax_cpu(const float *lg) {
    int best = 0;
    for (int v = 1; v < QW_VOCAB; v++) if (lg[v] > lg[best]) best = v;
    return best;
}

// 狀態檢查點：GPU 上複製 DeltaNet / conv / PLE 狀態（to_snap = true：目前狀態 → 檢查點）
// stage ≥ 0：來源改為段中檢查點的第 stage 份暫存
static void snap_copy_k(qw_engine *e, struct qw_snap *c, bool to_snap, int stage) {
    if (to_snap) dn_flush(e); // DeltaNet 延後寫回的狀態先補上
    else e->dn_pmask = 0;     // 狀態整份被蓋掉
    qw_batch *b = qw_batch_begin(e->g, false);
    const qw_buf *src[3] = {&e->S, &e->conv_st, &e->ple_st}, *dst[3] = {&c->S, &c->conv, &c->ple};
    const qw_buf *stg[3] = {&e->snS, &e->snConv, &e->snPle};
    for (int i = 0; i < 3; i++) {
        const uint32_t n = (uint32_t)(src[i]->size / 4);
        const size_t so = stage >= 0 ? (size_t)stage * src[i]->size : 0;
        if (stage >= 0) src[i] = stg[i];
        qw_batch_kernel(b, e->k_copyf);
        qw_batch_buf(b, 0, to_snap ? src[i] : dst[i], to_snap ? so : 0);
        qw_batch_buf(b, 1, to_snap ? dst[i] : src[i], to_snap ? 0 : so);
        qw_batch_bytes(b, 2, &n, 4);
        d1(b, n, 256);
    }
    qw_batch_end(b);
    if (stage >= 0 && to_snap && getenv("QW_SNAP_DUMP")) { // 除錯：段中檢查點的 DeltaNet 狀態寫到檔案（比較 scan 版本用）
        char p[512];
        snprintf(p, sizeof p, "%s.%d", getenv("QW_SNAP_DUMP"), stage);
        FILE *f = fopen(p, "wb");
        if (f) { fwrite((const char *)e->snS.ptr + (size_t)stage * e->S.size, 1, e->S.size, f); fclose(f); }
    }
}

// 存一個長度為 len 的檢查點：同長度的覆寫，否則用空的或最舊的
static void snap_copy(qw_engine *e, struct qw_snap *c, bool to_snap) { snap_copy_k(e, c, to_snap, -1); }

// 快照是誰存的（只給 QW_VERIFY_REUSE 的紀錄用，不影響數值）：呼叫 snap_save 前設 snap_why，存完回到 SW_CKPT
enum { SW_CKPT, SW_PROMPT, SW_FINISH, SW_PARSNAP, SW_XS, SW_SSD, SW_YIELD };
static const char *const SW_NAME[] = {"ckpt", "prompt-end", "finish", "par_snap", "xs_copy", "ssd", "yield"};
static int snap_why;
static int par_cur; // 定義在下面（並發模式目前載入的 slot）
static int slot_tag[QW_PAR_MAX]; // 每組 slot 目前的請求編號（qw_engine_par_tag）：快照記下是哪個請求存的
// 能不能從快照 s 接續長度 np 的 prompt（tag = 這個請求的編號，-1 = 不是並發模式）：
//   跨請求：只用 exact 而且長度是 GDN 分塊倍數的快照 —— 接續的結果與從 0 一次 prefill 逐位元相同（不同分段點、或 decode 算出的狀態，
//   最後 logits 的 KL 可到 0.01–0.4，greedy 常走不同的路）
//   同一個請求接著生成（只差最後送出、還沒餵的那一個 token）：用 decode 算到的狀態（與沒中斷時逐位元相同）
static bool snap_cross(const struct qw_snap *s) { return s->S.ptr && s->len > 0 && s->exact && s->len % gdn_chunk() == 0; }
static bool snap_cont(const struct qw_snap *s, int np, int tag) { return s->S.ptr && tag >= 0 && s->tag == tag && s->len == np - 1; }
static bool snap_ok(const struct qw_snap *s, int np, int tag) { return s->len > 0 && s->len <= np - 1 && (snap_cross(s) || snap_cont(s, np, tag)); }
// 重用狀態的自我檢查（QW_VERIFY_REUSE=N，見 qw_engine_par_verify）：prompt 算完時記下最後位置的 logits，閒置時在空的 slot 從頭重算比對
static bool vr_fresh; // 正在從頭重算（不接快照、不 xs_copy、不 dc_load）
static bool no_reuse(void) { return vr_fresh || getenv("QW_NO_CACHE") || getenv("QW_NO_REUSE"); } // QW_NO_REUSE = QW_NO_CACHE：一律從 0 算
static int snap_cap(qw_engine *e); // 每組 slot 最多幾份快照（並發模式較少：每份含 113 MB 的 DeltaNet 狀態，slot 多時會佔太多記憶體）
static void snap_save_k(qw_engine *e, int len, int stage) {
    const int why = snap_why;
    snap_why = SW_CKPT;
    int k = -1;
    const int CK = snap_cap(e);
    for (int i = 0; i < CK && k < 0; i++) if (e->snap[i].len == len) k = i;
    for (int i = 0; i < CK && k < 0; i++) if (e->snap[i].len == 0 && e->snap[i].S.ptr) k = i;
    for (int i = 0; i < CK && k < 0; i++)
        if (!e->snap[i].S.ptr) {
            if (!alloc(e, &e->snap[i].S, e->S.size) || !alloc(e, &e->snap[i].conv, e->conv_st.size) || !alloc(e, &e->snap[i].ple, e->ple_st.size)) return;
            k = i;
        }
    if (k < 0) {
        k = 0;
        for (int i = 1; i < CK; i++) if (e->snap[i].age < e->snap[k].age) k = i;
    }
    snap_copy_k(e, &e->snap[k], true, stage);
    e->snap[k].len = len;
    e->snap[k].age = ++e->clock;
    e->snap[k].why = why;
    e->snap[k].exact = stage >= 0 ? e->seg_exact && len % gdn_chunk() == 0 : e->st_exact == len && e->pos == len;
    e->snap[k].tag = par_cur >= 0 ? slot_tag[e->cur] : -1;
}
static void snap_save(qw_engine *e, int len) { snap_save_k(e, len, -1); }
static void snap_save_from(qw_engine *e, int len, int stage) { snap_save_k(e, len, stage); }

// 目前的 slot 欄位 ↔ slots[i]
static void slot_save(qw_engine *e, int i) {
    struct qw_slot *s = &e->slots[i];
    s->init = true;
    s->Kc = e->Kc; s->Vc = e->Vc; s->rawk = e->rawk; s->kblk = e->kblk; s->tok = e->tok; s->kvcap = e->kvcap;
    s->hist = e->hist; s->kvtok = e->kvtok; s->kvlen = e->kvlen; s->last_eos = e->last_eos;
    memcpy(s->snap, e->snap, sizeof e->snap);
}

static void slot_load(qw_engine *e, int i) {
    struct qw_slot *s = &e->slots[i];
    e->Kc = s->Kc; e->Vc = s->Vc; e->rawk = s->rawk; e->kblk = s->kblk; e->tok = s->tok; e->kvcap = s->kvcap;
    e->hist = s->hist; e->kvtok = s->kvtok; e->kvlen = s->kvlen; e->last_eos = s->last_eos;
    memcpy(e->snap, s->snap, sizeof e->snap);
    e->cur = i;
}

static bool slot_alloc(qw_engine *e, int i) {
    struct qw_slot *s = &e->slots[i];
    const size_t L = (size_t)e->max_ctx;
    memset(s, 0, sizeof *s);
    if (!kv_alloc(e, &s->Kc, &s->Vc, &s->rawk, &s->kblk, s->kvcap = kv_init_cap(e)) || !alloc(e, &s->tok, e->tok.size))
        return false;
    s->hist = calloc(L + 2, sizeof(int32_t));
    s->hist[0] = s->hist[1] = EOS_ID;
    ((int32_t *)s->tok.ptr)[0] = ((int32_t *)s->tok.ptr)[1] = EOS_ID;
    s->kvtok = malloc(sizeof(int32_t) * L);
    s->last_eos = 1;
    s->init = true;
    return true;
}

static int par_cur = -1; // 並發（qw_engine_par_*）：目前載入的 slot（-1：還沒進並發模式）
// 依 prompt 選 slot：共同前綴最長的；太短（< 1024 且不到一半）時改用空的或最久沒用的
static void slot_pick(qw_engine *e, const int32_t *prompt, int np) {
    if (par_cur >= 0) return; // 並發模式：slot 由 qw_engine_par_solo 先選好（換 slot 要連同 DeltaNet 狀態一起換）
    if (!e->nslots) {
        const char *v = getenv("QW_SLOTS");
        e->nslots = v ? atoi(v) : 1;
        if (e->nslots < 1) e->nslots = 1;
        if (e->nslots > QW_PAR_MAX) e->nslots = QW_PAR_MAX;
    }
    if (e->nslots == 1) return;
    slot_save(e, e->cur);
    int best = -1, bc = -1;
    for (int i = 0; i < e->nslots; i++) {
        if (!e->slots[i].init) continue;
        int c = 0;
        while (c < e->slots[i].kvlen && c < np && e->slots[i].kvtok[c] == prompt[c]) c++;
        if (c > bc) { bc = c; best = i; }
    }
    int pick = best;
    if (bc < 1024 && bc < np / 2) {
        pick = -1;
        for (int i = 0; i < e->nslots && pick < 0; i++)
            if (!e->slots[i].init && slot_alloc(e, i)) pick = i;
        if (pick < 0) { // 最久沒用的
            pick = 0;
            for (int i = 1; i < e->nslots; i++)
                if (e->slots[i].used < e->slots[pick].used) pick = i;
        }
    }
    e->slots[pick].used = ++e->clock;
    if (pick != e->cur) slot_load(e, pick);
}

// ---------------------------------------------------------------- 持久化前綴快取（SSD）
// 一個檔案 = 某段 token 前綴處理完的完整狀態：DeltaNet / conv / PLE 狀態 + 13 份 attention 的 KV、indexer raw_k / 區塊 key + token 歷史
typedef struct { char magic[4]; uint32_t len, last_eos, pad; uint64_t hash; } dc_head;

static uint64_t tok_hash(const int32_t *t, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; i++) h = (h ^ (uint32_t)t[i]) * 1099511628211ull;
    return h;
}

// 檔案的指紋：模型（GGUF 目錄或 MLX 目錄）與影響數值的設定（QW_INT8 等）。換了模型 / 設定，舊檔案不會被讀進來
static uint32_t dc_fp(void) {
    static uint32_t fp;
    if (!fp) {
        char k[2048];
        snprintf(k, sizeof k, "%s|%s|%s|%s|%s", getenv("QW_GGUF") ? getenv("QW_GGUF") : "mlx", getenv("QW_GGUF_PARTS") ? getenv("QW_GGUF_PARTS") : "",
                 getenv("QW_INT8") ? getenv("QW_INT8") : "", getenv("QW_EXACT_SPEC") ? getenv("QW_EXACT_SPEC") : "", getenv("QW_DENSE4") ? getenv("QW_DENSE4") : "");
        uint64_t h = 1469598103934665603ull;
        for (const char *c = k; *c; c++) h = (h ^ (uint8_t)*c) * 1099511628211ull;
        fp = (uint32_t)(h ^ h >> 32) | 1;
    }
    return fp;
}
static void dc_path(const qw_engine *e, uint64_t h, int len, char *out, size_t n) { // QWC2：只存 exact 的狀態（snap_cross），檔名帶指紋
    snprintf(out, n, "%s/%016llx_%08x_%d.qwc", e->cache_dir, (unsigned long long)h, dc_fp(), len);
}

// 各段資料：{目前 slot 的 buffer, 每份 attention 的位移, 每份的位元組數}
static void dc_parts_of(const qw_buf *K, const qw_buf *V, const qw_buf *R, const qw_buf *B, int cap, int len, int ai, const qw_buf **buf,
                        size_t *off, size_t *bytes) { // 同 dc_parts，來源是指定的 buffer（例如沒載入的 slot）
    const size_t Lc = (size_t)cap;
    buf[0] = K; off[0] = (size_t)ai * Lc * 2 * 256 * 2; bytes[0] = (size_t)len * 2 * 256 * 2;
    buf[1] = V; off[1] = off[0]; bytes[1] = bytes[0];
    buf[2] = R; off[2] = (size_t)ai * Lc * 128 * 4; bytes[2] = (size_t)len * 128 * 4;
    buf[3] = B; off[3] = (size_t)ai * (Lc / 4 + 1) * 128 * 2; bytes[3] = (size_t)(len / 4) * 128 * 2;
}
static void dc_parts(const qw_engine *e, int len, int ai, const qw_buf **buf, size_t *off, size_t *bytes) {
    dc_parts_of(&e->Kc, &e->Vc, &e->rawk, &e->kblk, e->kvcap, len, ai, buf, off, bytes);
}

static void dc_trim_dir(qw_engine *e);
static void dc_write_from(qw_engine *e, const int32_t *toks, int len, const struct qw_snap *src) {
    if (!e->cache_dir[0] || len < 256) return;
    const uint64_t h = tok_hash(toks, len);
    char path[700], tmp[720];
    dc_path(e, h, len, path, sizeof path);
    if (access(path, F_OK) == 0) return;
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return;
    const double t0 = qw_now();
    if (!snap_cross(src) || src->len != len) return; // 只存跨請求接得上的狀態（與從 0 一次 prefill 逐位元相同）
    dc_head hd = {{'Q', 'W', 'C', '2'}, (uint32_t)len, (uint32_t)e->last_eos, dc_fp(), h};
    bool ok = fwrite(&hd, sizeof hd, 1, f) == 1 && fwrite(toks, 4, (size_t)len, f) == (size_t)len &&
              fwrite(e->hist, 4, (size_t)len + 2, f) == (size_t)len + 2 && fwrite(e->tok.ptr, 4, (size_t)len + 2, f) == (size_t)len + 2 &&
              fwrite(src->S.ptr, 1, e->S.size, f) == e->S.size && fwrite(src->conv.ptr, 1, e->conv_st.size, f) == e->conv_st.size &&
              fwrite(src->ple.ptr, 1, e->ple_st.size, f) == e->ple_st.size;
    for (int ai = 0; ok && ai <= NATT; ai++) {
        const qw_buf *b[4];
        size_t off[4], by[4];
        dc_parts(e, len, ai, b, off, by);
        for (int k = 0; k < 4 && ok; k++) ok = fwrite((char *)b[k]->ptr + off[k], 1, by[k], f) == by[k];
    }
    ok = fclose(f) == 0 && ok;
    if (ok) rename(tmp, path);
    else remove(tmp);
    dc_trim_dir(e);
    if (getenv("QW_CACHE_DBG")) fprintf(stderr, "[cache] 寫入 SSD：%d 個 token，%.0f ms\n", len, (qw_now() - t0) * 1e3);
}
static double dc_cap(void) { return (getenv("QW_CACHE_GB") ? atof(getenv("QW_CACHE_GB")) : 20.0) * 1e9; }
static void dc_trim_dir(qw_engine *e) { // 容量上限（QW_CACHE_GB，預設 20）：超過就刪最舊的
    const double cap = dc_cap();
    for (;;) {
        DIR *d = opendir(e->cache_dir);
        if (!d) break;
        double tot = 0;
        time_t oldest = 0;
        char opath[700] = "";
        struct dirent *de;
        while ((de = readdir(d))) {
            if (!strstr(de->d_name, ".qwc") || strstr(de->d_name, ".tmp")) continue;
            char pth[700];
            snprintf(pth, sizeof pth, "%s/%s", e->cache_dir, de->d_name);
            struct stat sb;
            if (stat(pth, &sb)) continue;
            tot += (double)sb.st_size;
            if (!opath[0] || sb.st_atime < oldest) { oldest = sb.st_atime; snprintf(opath, sizeof opath, "%s", pth); }
        }
        closedir(d);
        if (tot <= cap || !opath[0]) break;
        remove(opath);
    }
}
// 沒載入的 slot s 的前 len 個位置（快照 sn 的狀態；len ≤ s->kvlen）寫到 SSD，格式同 dc_write_from（dc_load 讀得回來）。
// 只讀 CPU 看得到的 buffer：呼叫時這個 slot 不能是目前載入的、也不能有還沒完成的 GPU 工作（閒置的 slot）
static bool dc_write_slot(qw_engine *e, const struct qw_slot *s, const struct qw_snap *sn, int len) {
    if (!e->cache_dir[0] || len < 256 || len > s->kvlen || !snap_cross(sn) || sn->len != len || !s->hist || !s->kvtok || !s->tok.ptr) return false;
    const uint64_t h = tok_hash(s->kvtok, len);
    char path[700], tmp[720];
    dc_path(e, h, len, path, sizeof path);
    if (access(path, F_OK) == 0) return true;
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return false;
    const double t0 = qw_now();
    int last_eos = 1; // 同 xs_copy：照 token 歷史重算到 len 為止
    for (int p = 0; p < len; p++) if (s->hist[p + 2] == EOS_ID) last_eos = p + 1;
    dc_head hd = {{'Q', 'W', 'C', '2'}, (uint32_t)len, (uint32_t)last_eos, dc_fp(), h};
    bool ok = fwrite(&hd, sizeof hd, 1, f) == 1 && fwrite(s->kvtok, 4, (size_t)len, f) == (size_t)len &&
              fwrite(s->hist, 4, (size_t)len + 2, f) == (size_t)len + 2 && fwrite(s->tok.ptr, 4, (size_t)len + 2, f) == (size_t)len + 2 &&
              sn->S.size == e->S.size && sn->conv.size == e->conv_st.size && sn->ple.size == e->ple_st.size &&
              fwrite(sn->S.ptr, 1, sn->S.size, f) == sn->S.size && fwrite(sn->conv.ptr, 1, sn->conv.size, f) == sn->conv.size &&
              fwrite(sn->ple.ptr, 1, sn->ple.size, f) == sn->ple.size;
    for (int ai = 0; ok && ai <= NATT; ai++) {
        const qw_buf *b[4];
        size_t off[4], by[4];
        dc_parts_of(&s->Kc, &s->Vc, &s->rawk, &s->kblk, s->kvcap, len, ai, b, off, by);
        for (int k = 0; k < 4 && ok; k++) ok = fwrite((const char *)b[k]->ptr + off[k], 1, by[k], f) == by[k];
    }
    ok = fclose(f) == 0 && ok;
    if (ok) rename(tmp, path);
    else remove(tmp);
    dc_trim_dir(e);
    fprintf(stderr, "[cache] 釋放前寫入 SSD：%d 個 token，%.0f ms%s\n", len, (qw_now() - t0) * 1e3, ok ? "" : "（失敗）");
    return ok;
}

// 在 SSD 找與 prompt 前綴相同、最長（> min_len、≤ np − 1）的檔案，載入目前的 slot。回傳載入的長度（0 = 沒有）
static int dc_load(qw_engine *e, const int32_t *prompt, int np, int min_len) {
    if (!e->cache_dir[0]) return 0;
    DIR *d = opendir(e->cache_dir);
    if (!d) return 0;
    int best = 0;
    char bpath[700] = "";
    struct dirent *de;
    while ((de = readdir(d))) {
        unsigned long long h;
        unsigned fp;
        int len;
        if (sscanf(de->d_name, "%16llx_%8x_%d.qwc", &h, &fp, &len) != 3 || fp != dc_fp() || !strstr(de->d_name, ".qwc") || strstr(de->d_name, ".tmp")) continue;
        if (len <= min_len || len <= best || len > np - 1 || len % gdn_chunk()) continue;
        if (tok_hash(prompt, len) != h) continue;
        best = len;
        dc_path(e, h, len, bpath, sizeof bpath);
    }
    closedir(d);
    if (!best) return 0;
    const double t0 = qw_now();
    FILE *f = fopen(bpath, "rb");
    if (!f) return 0;
    dc_head hd;
    bool ok = fread(&hd, sizeof hd, 1, f) == 1 && !memcmp(hd.magic, "QWC2", 4) && (int)hd.len == best && hd.pad == dc_fp();
    int32_t *chk = malloc(sizeof(int32_t) * (size_t)best);
    ok = ok && fread(chk, 4, (size_t)best, f) == (size_t)best && !memcmp(chk, prompt, sizeof(int32_t) * (size_t)best);
    free(chk);
    ok = ok && fread(e->hist, 4, (size_t)best + 2, f) == (size_t)best + 2 && fread(e->tok.ptr, 4, (size_t)best + 2, f) == (size_t)best + 2 &&
         (e->dn_pmask = 0, fread(e->S.ptr, 1, e->S.size, f) == e->S.size) && fread(e->conv_st.ptr, 1, e->conv_st.size, f) == e->conv_st.size &&
         fread(e->ple_st.ptr, 1, e->ple_st.size, f) == e->ple_st.size;
    for (int ai = 0; ok && ai <= NATT; ai++) {
        const qw_buf *b[4];
        size_t off[4], by[4];
        dc_parts(e, best, ai, b, off, by);
        for (int k = 0; k < 4 && ok; k++) ok = fread((char *)b[k]->ptr + off[k], 1, by[k], f) == by[k];
    }
    fclose(f);
    if (!ok) { // KV / 歷史可能已經讀了一半：這組 slot 的快照都不能再用
        qw_engine_reset(e);
        e->kvlen = 0;
        for (int i = 0; i < CKMAX; i++) e->snap[i].len = 0;
        return 0;
    }
    e->last_eos = (int)hd.last_eos;
    e->pos = best;
    e->st_exact = best; // 只存 exact 的（dc_write_from / dc_write_slot）
    e->pending = false;
    for (int i = 0; i < CKMAX; i++) e->snap[i].len = 0; // KV 被覆寫，記憶體裡的檢查點都失效
    memcpy(e->kvtok, prompt, sizeof(int32_t) * (size_t)best);
    e->kvlen = best;
    snap_why = SW_SSD, snap_save(e, best);
    if (getenv("QW_CACHE_DBG")) fprintf(stderr, "[cache] 從 SSD 載入：%d 個 token，%.0f ms\n", best, (qw_now() - t0) * 1e3);
    return best;
}

void qw_engine_set_cache_dir(qw_engine *e, const char *dir) {
    snprintf(e->cache_dir, sizeof e->cache_dir, "%s", dir ? dir : "");
    if (dir && *dir) mkdir(dir, 0755);
}

void qw_engine_reset(qw_engine *e) {
    e->pos = 0;
    e->st_exact = 0;
    e->dn_pmask = 0;
    memset(e->S.ptr, 0, e->S.size);
    memset(e->conv_st.ptr, 0, e->conv_st.size);
    memset(e->ple_st.ptr, 0, e->ple_st.size);
    e->pending = false;
}

int qw_engine_generate_ex(qw_engine *e, const int32_t *prompt, int np, const qw_gen *g, qw_tok_cb cb, void *ctx, qw_gen_stats *st) {
    char err[1024];
    qw_gen_stats S0 = {0};
    if (!st) st = &S0;
    *st = (qw_gen_stats){0};
    st->prompt = np;
    const bool greedy = g->temperature <= 0;
    // 猜測來源：MTP 或 DFlash2（GGUF profile 沒有 MTP，只能靠 DFlash2）
    const bool want_mtp = getenv("QW_DRAFT") && !strcmp(getenv("QW_DRAFT"), "mtp");
    const bool spec = g->spec && (e->m->has_mtp || (e->df_on && !want_mtp)) && vb_init(e, err, sizeof err);
    uint64_t rng = g->seed ? g->seed : 0x1234567ull;
    const int n = g->max_tokens;
    if (np <= 0 || np + 2 >= e->max_ctx) return QW_FIN_CTX;
    if (n <= 0) return QW_FIN_LENGTH;
    double ms = 0;
    const double tp0 = qw_now();
    // prefix cache：找共同前綴，從最長的可用快照恢復，只算新增的部分
    if (!e->kvtok) {
        e->kvtok = malloc(sizeof(int32_t) * (size_t)e->max_ctx);
        slot_save(e, 0);
        e->slots[0].used = ++e->clock;
    }
    slot_pick(e, prompt, np);
    if (!kv_reserve(e, np + n + 2 * VBMAX + 8 < e->max_ctx ? np + n + 2 * VBMAX + 8 : e->max_ctx)) return QW_FIN_ERROR;
    const bool tdbg = getenv("QW_CACHE_DBG");
    if (tdbg) fprintf(stderr, "[t] slot_pick %.1f ms\n", (qw_now() - tp0) * 1e3);
    int common = 0;
    while (common < e->kvlen && common < np && e->kvtok[common] == prompt[common]) common++;
    const int tag = par_cur >= 0 ? slot_tag[e->cur] : -1; // 同一個請求接著生成時可以用 decode 的狀態（snap_ok）
    int best = -1;
    for (int i = 0; i < CKMAX; i++) {
        if (e->snap[i].len > common) e->snap[i].len = 0; // KV 會從 common 起被覆寫，比它長的快照失效
        if (snap_ok(&e->snap[i], np, tag) && (best < 0 || e->snap[i].len > e->snap[best].len)) best = i;
    }
    // 記憶體裡的快取不夠長時，看 SSD 上有沒有更長的
    if (!no_reuse() && e->cache_dir[0]) {
        const int have = best >= 0 ? e->snap[best].len : 0;
        if (have < np / 2 && dc_load(e, prompt, np, have) > 0) {
            if (par_cur >= 0 && vr_on()) vr_note(e->cur, "solo: SSD %d; ", e->kvlen);
            common = e->kvlen;
            best = -1;
            for (int i = 0; i < CKMAX; i++)
                if (snap_ok(&e->snap[i], np, tag) && (best < 0 || e->snap[i].len > e->snap[best].len)) best = i;
        }
    }
    if (tdbg) fprintf(stderr, "[t] dc_load %.1f ms\n", (qw_now() - tp0) * 1e3);
    int start = 0;
    bool cont = false; // 同一個請求接著生成：剩下的 token 用 decode 餵（與沒中斷時相同）
    if (best >= 0 && !no_reuse()) {
        start = e->snap[best].len;
        cont = snap_cont(&e->snap[best], np, tag);
        snap_copy(e, &e->snap[best], false);
        e->pos = start;
        e->st_exact = e->snap[best].exact ? start : -1;
        e->pending = false;
    } else
        qw_engine_reset(e);
    if (par_cur >= 0 && vr_on())
        vr_note(e->cur, "solo %d→%d (%s); ", start, np, start ? SW_NAME[e->snap[best].why] : "from 0");
    if (par_cur >= 0 && start > 0) vr_mark(e->cur);
    st->cached = start;
    if (getenv("QW_CACHE_DBG"))
        fprintf(stderr, "[cache] slot %d：共同前綴 %d / %d，從 %d 開始\n", e->cur, common, np, start);
    memcpy(e->kvtok, prompt, sizeof(int32_t) * (size_t)np);
    e->kvlen = np;
    // DFlash2：有載入就用（QW_DRAFT=mtp 改回 MTP）；環狀 KV 在同一個 slot 且位置連續時，start 之前的部分還能用
    const bool dfm = spec && e->df_on && !want_mtp && df_init(e, err, sizeof err);
    if (spec && !dfm && !e->m->has_mtp) { fprintf(stderr, "DFlash2 初始化失敗（又沒有 MTP）：%s\n", err); return QW_FIN_ERROR; }
    e->df_mode = dfm;
    e->df_tapping = dfm;
    if (dfm) {
        int lo = start;
        if (e->df_slot == e->cur && e->df_done >= start) lo = e->df_done - DF_RING_ + 1 > 0 ? e->df_done - DF_RING_ + 1 : 0;
        if (lo > start) lo = start;
        e->df_lo = lo;
        e->df_ctx_from = np - e->df.win > start ? np - e->df.win : start;
        e->df_done = start;
        e->df_slot = e->cur;
    }
    e->pf_mtp = spec && !dfm;
    // 檢查點：prompt 結尾前 2048、512、128 個 token 與結尾（下次請求常在結尾附近才不同，例如 Claude Code 的權限判斷）
    int cks[20], nck = 0;
    const int back[3] = {2048, 512, 128};
    // 每多切一段就多一次 prefill 的固定開銷：新增量大時才切 −2048 / −512，−128 在新增量 > 512 時才切
    const int minnew[3] = {4096, 4096, 512};
    for (int k = 0; k < 3; k++)
        if (np - start > minnew[k] && np - back[k] > start + 64 && (nck == 0 || np - back[k] > cks[nck - 1] + 32)) cks[nck++] = np - back[k];
    if (np / gdn_chunk() * gdn_chunk() > start && np % gdn_chunk()) cks[nck++] = np / gdn_chunk() * gdn_chunk(); // 下一輪能接續的最長位置（snap_ok）
    // 持久化點（例如 system 訊息結尾）：也切在這裡，處理完寫到 SSD
    for (int k = 0; k < g->npersist; k++) {
        const int pp = g->persist[k];
        if (pp <= start || pp >= np || nck >= 15) continue;
        bool dup = false;
        for (int q = 0; q < nck; q++) dup |= cks[q] == pp;
        if (!dup) cks[nck++] = pp;
    }
    for (int a = 0; a < nck; a++) // 排序
        for (int b2 = a + 1; b2 < nck; b2++)
            if (cks[b2] < cks[a]) { const int t = cks[a]; cks[a] = cks[b2]; cks[b2] = t; }
    // 檢查點對齊到 GDN 分塊邊界（相對於 start；每段 PF 個 token 也是分塊的倍數），就能在同一段裡順便存狀態、不必切段
    const int CC = gdn_chunk();
    e->npf_snap = 0;
    for (int k = 0; k < nck; k++) {
        const int a = start + (cks[k] - start) / CC * CC;
        if (a > start && a < np && (e->npf_snap == 0 || e->pf_snap[e->npf_snap - 1] != a)) e->pf_snap[e->npf_snap++] = a;
    }
    double pms = 0;
    e->pf_next = -1;
    const float *lg;
    if (cont) { // 並發模式同一個請求接續（切換單獨 / 一起）：剩下的 token 用 decode 一步一步算，與沒中斷時逐位元相同
        e->npf_snap = 0;
        lg = NULL;
        for (int i = start; i < np; i++) if (!(lg = qw_engine_step(e, prompt[i], NULL))) break;
    } else if (g->pf_yield && np - start > PF) { // 可讓出：一段 PF 個（與一次 prefill 的分段相同，逐位元相同），段間問要不要讓
        int i = start;
        lg = NULL;
        while (i < np) {
            const int end = np - i > PF ? i + PF : np;
            e->pf_next = end < np ? prompt[end] : -1; // MTP 段尾要知道下一個 token（同一次 prefill 時的段界）
            double pm = 0;
            lg = qw_engine_prefill(e, prompt + i, end - i, NULL, &pm);
            pms += pm;
            if (!lg) break;
            i = end;
            if (i < np && g->pf_yield(ctx)) { // 讓出：KV / 狀態停在 i，記快照，之後從這裡接
                e->npf_snap = 0;
                e->pf_next = -1;
                e->pf_mtp = false;
                e->mtp_T = 0;
                e->kvlen = i;
                snap_why = SW_YIELD, snap_save(e, i);
                for (int q = 0; q < g->npersist; q++) {
                    const int pp = g->persist[q];
                    if (pp <= start || pp > i) continue;
                    const int a = start + (pp - start) / CC * CC;
                    for (int k = 0; k < CKMAX; k++)
                        if (e->snap[k].len == a && a > 0) { dc_write_from(e, prompt, a, &e->snap[k]); break; }
                }
                st->prefill_ms = (qw_now() - tp0) * 1e3;
                st->pf_done = i;
                return QW_FIN_YIELD;
            }
        }
    } else lg = qw_engine_prefill(e, prompt + start, np - start, NULL, &pms);
    e->npf_snap = 0;
    if (tdbg) fprintf(stderr, "[t] prefill %d..%d done %.1f ms (gpu %.1f)\n", start, np, (qw_now() - tp0) * 1e3, pms);
    if (lg) {
        snap_why = SW_PROMPT, snap_save(e, np);
        // 持久化：從對齊後的檢查點寫到 SSD
        for (int q = 0; q < g->npersist; q++) {
            const int pp = g->persist[q];
            if (pp <= start || pp >= np) continue;
            const int a = start + (pp - start) / CC * CC;
            for (int i = 0; i < CKMAX; i++)
                if (e->snap[i].len == a && a > 0) { dc_write_from(e, prompt, a, &e->snap[i]); break; }
        }
    }
    e->pf_next = -1;
    e->pf_mtp = false;
    e->mtp_T = 0;
    if (!lg) return QW_FIN_ERROR;
    if (par_cur >= 0) vr_capture(e, e->cur, prompt, np, lg);
    st->prefill_ms = (qw_now() - tp0) * 1e3;
    dist d;
    int32_t first;
    if (greedy) first = argmax_cpu(lg);
    else { make_dist(lg, g, &d); first = dist_draw(&d, &rng, -1); }
    // QW_SPEC_STATS=1：接受統計（依猜測來源：0 = DFlash2、1 = 後綴）與各 NB 的每輪 GPU 時間（gacc 時含下一塊的猜測）
    const bool sstat = getenv("QW_SPEC_STATS") != NULL;
    int ss_j[2][VBMAX] = {{0}}, ss_n[2] = {0}, ss_acc[2] = {0}, ss_nd[2] = {0}, ss_skip = 0, ss_lazy = 0;
    double ss_gnb[VBMAX + 1] = {0}, ss_lazy_ms = 0;
    const double tw0 = qw_now();
    int got = 0, fin = QW_FIN_LENGTH;
    // 送出一個 token：回傳 false 表示要停
    #define EMIT(tokv)                                                                     \
        do {                                                                               \
            const int32_t tv_ = (tokv);                                                    \
            bool stop_ = false;                                                            \
            for (int q_ = 0; q_ < g->nstop; q_++) stop_ |= g->stop[q_] == tv_;             \
            if (stop_) { fin = QW_FIN_STOP; goto done; }                                   \
            got++;                                                                         \
            if (cb && !cb(ctx, tv_)) { fin = QW_FIN_CALLBACK; goto done; }                 \
            if (got >= n) { fin = QW_FIN_LENGTH; goto done; }                              \
        } while (0)
    int32_t *seq = malloc(sizeof(int32_t) * (size_t)(np + n + 16));
    memcpy(seq, prompt, sizeof(int32_t) * (size_t)np);
    int ns = np;
    seq[ns++] = first;
    if (!spec && getenv("QW_DF_DBG")) fprintf(stderr, "[t] %d %d\n", ns - 1, first);
    int32_t last = first; // 已決定、模型還沒處理的 token
    EMIT(first);
    if (!spec) { // 不推測：逐 token
        for (;;) {
            if (e->pos + 2 >= e->max_ctx) { fin = QW_FIN_CTX; goto done; }
            lg = qw_engine_step(e, last, NULL);
            if (!lg) { fin = QW_FIN_ERROR; goto done; }
            int32_t t;
            if (greedy) t = argmax_cpu(lg);
            else { make_dist(lg, g, &d); t = dist_draw(&d, &rng, -1); }
            seq[ns++] = t;
            last = t;
            if (getenv("QW_DF_DBG")) fprintf(stderr, "[t] %d %d\n", ns - 1, t);
            EMIT(t);
            if (g->yield && *g->yield) { fin = QW_FIN_YIELD; goto done; }
        }
    }
    double sp_gpu = 0, sp_wall = 0;
    int sp_nb[VBMAX + 1] = {0};
    int src = 0;
    int32_t draft = 0;
    const int32_t *dpath = dfm ? e->dfpath.ptr : NULL;
    const float *dprob = dfm ? e->dfprob.ptr : NULL;
    const float dconf = getenv("QW_DF_CONF") ? (float)atof(getenv("QW_DF_CONF")) : 0.8f; // 候選挑選的機率低於這個就截斷（同 Strata 的 --draft-conf 0.8）
    // df_have：dpath / dprob 是這一輪的猜測；df_skip：這輪的 command buffer 不猜下一塊（照抄中，QW_DF_NOSKIP=1 關）
    bool df_have = true, df_skip = false;
    const bool df_skip_on = !getenv("QW_DF_NOSKIP");
    // greedy + DFlash2：接受判定在 GPU 上做，猜下一塊接在驗證後面（一輪只等一次 CPU）。QW_DF_CPU_ACCEPT=1 改回兩個 command buffer。
    // 驗證過但沒接受的位置也會寫進環狀 KV（下一輪寫上正確的之前不會被讀到），所以視窗加一輪不能超過環的大小
    const bool gacc = dfm && greedy && !getenv("QW_DF_CPU_ACCEPT") && !getenv("QW_DF_NOCTX") && !getenv("QW_DF_NODRAFT") &&
                      !getenv("QW_DF_ORACLE") && e->df.win + VBMAX <= DF_RING_;
    double df_gms = 0;
    if (dfm) { // 第一塊：anchor = 第一個生成的 token（位置 np）
        df_set_ids(e, first);
        qw_batch *b0 = qw_batch_begin(e->g, false);
        df_draft(e, b0, e->pos, NULL);
        df_gms += qw_batch_end(b0) * 1e3;
    } else
        draft = qw_engine_mtp_draft(e, first, e->pos - 1, &ms);
    int32_t *tk = e->tok.ptr;
    const int kmax = spec_kmax();
    // 驗證分段送出：第一段很短（GPU 盡早開始），之後 CPU 編碼的同時 GPU 在跑。QW_VB_FLUSH=層號,層號…（none = 不分）
    e->vb_flush = 0;
    {
        const char *fs = getenv("QW_VB_FLUSH") ? getenv("QW_VB_FLUSH") : "0,8,24";
        for (const char *q = fs; *q;) {
            char *end;
            const long v = strtol(q, &end, 10);
            if (end == q) break;
            if (v >= 0 && v < QW_LAYERS - 1) e->vb_flush |= 1ull << v;
            q = *end == ',' ? end + 1 : end;
        }
    }
    const bool use_ng = !getenv("QW_NO_NGRAM");
    for (;;) {
        const int p0 = e->pos;
        if (p0 + VBMAX + 2 >= e->max_ctx) { fin = QW_FIN_CTX; goto done; }
        // 候選：MTP 的 1 個；前文有相同結尾（3 個 token）且下一個與 MTP 一致時，改用前文接下來的一串
        int32_t dr[VBMAX];
        int nd = 1;
        dr[0] = draft;
        if (dfm) {
            nd = 0;
            src = 0;
            // 後綴猜測（prompt lookup，同 Strata 的 suffix drafter）：最近 ≥ 3 個 token 在前文出現過時，提出前文接在後面的一串。
            // 改程式碼、照抄前文時這比 DFlash2 準得多。先找（只用 CPU、很便宜），下面再決定用哪個（QW_SUFFIX=0 關）
            static int suf_on = -1, suf_m = 8, suf_long = 20;
            if (suf_on < 0) {
                suf_on = getenv("QW_SUFFIX") ? atoi(getenv("QW_SUFFIX")) : 1;
                if (getenv("QW_SUF_M")) suf_m = atoi(getenv("QW_SUF_M"));
                if (getenv("QW_SUF_LONG")) suf_long = atoi(getenv("QW_SUF_LONG"));
            }
            int best_i = -1, best_m = 0, ks = 0;
            int32_t sf[VBMAX];
            if (suf_on && ns >= 4 && !getenv("QW_DF_ORACLE")) {
                for (int i = ns - 2; i >= 2 && best_m < 32; i--) { // 最近的匹配優先；匹配長度最多量到 32
                    if (seq[i] != seq[ns - 1] || seq[i - 1] != seq[ns - 2] || seq[i - 2] != seq[ns - 3]) continue;
                    int m = 3;
                    while (m < 32 && i - m >= 0 && seq[i - m] == seq[ns - 1 - m]) m++;
                    if (m > best_m) { best_m = m; best_i = i; }
                    if (best_m >= suf_m) break;
                }
                if (best_i >= 0)
                    while (ks < VBMAX - 1 && best_i + 1 + ks < ns) { sf[ks] = seq[best_i + 1 + ks]; ks++; }
            }
            // 照抄：匹配很長（≥ QW_SUF_LONG，預設 20）且後綴給得出一整輪。離線量過（code / edit）這時後綴猜中的比 DFlash2 多
            const bool copy = ks == VBMAX - 1 && best_m >= suf_long;
            if (!df_have && !copy) { // 上一輪省掉了猜測，但這輪不是照抄：補猜（接受判定在 GPU 上寫好的 dfids、dfprm）
                const double tl0 = qw_now();
                qw_batch *bl = qw_batch_begin(e->g, false);
                df_draft(e, bl, 0, &e->dfprm);
                df_gms += qw_batch_end(bl) * 1e3;
                ss_lazy++;
                ss_lazy_ms += (qw_now() - tl0) * 1e3;
                df_have = true;
            }
            if (df_have) { // DFlash2 的路徑；信心低於 QW_DF_CONF 就截斷
                const int dk = getenv("QW_SPEC_K") ? kmax : VBMAX; // 有信心截斷，預設一輪最多驗證 8 個
                while (nd < dk - 1 && nd < e->df.block - 1 && (nd == 0 || dprob[nd] >= dconf)) { dr[nd] = dpath[nd]; nd++; }
                static int32_t *orc = NULL; // 除錯：QW_DF_ORACLE=檔案（每行「位置 token」）→ 用正確答案當猜測
                if (getenv("QW_DF_ORACLE")) {
                    if (!orc) {
                        orc = calloc(1 << 20, 4);
                        FILE *of = fopen(getenv("QW_DF_ORACLE"), "r");
                        int ps, tv;
                        while (of && fscanf(of, "%d %d", &ps, &tv) == 2) if (ps >= 0 && ps < (1 << 20)) orc[ps] = tv;
                        if (of) fclose(of);
                    }
                    for (int i = 0; i < nd; i++) dr[i] = orc[p0 + 1 + i];
                }
                if (getenv("QW_DF_DBG"))
                    for (int i = 0; i < e->df.block - 1; i++)
                        if (dpath[i] < 0 || dpath[i] >= QW_VOCAB) fprintf(stderr, "[df] 第 %d 輪 path[%d] = %d 超出範圍\n", st->rounds, i, dpath[i]);
            }
            // 照抄時 DFlash2 給滿一整輪又與後綴不同，多半是要改的地方（離線量過 edit：DFlash2 平均接受 6.4、後綴 3.4），用 DFlash2
            bool agree = true;
            for (int i = 0; copy && i < nd && agree; i++) agree = sf[i] == dr[i];
            const bool copy_ok = copy && (nd < VBMAX - 1 || agree);
            // 用後綴：照抄、或後綴提出的比 DFlash2 多且（匹配 ≥ QW_SUF_M、DFlash2 這輪沒把握（< 2 個）、或兩者第一個 token 一致）
            if (ks >= 2 && (copy_ok || (ks > nd && (best_m >= suf_m || nd < 2 || sf[0] == dr[0])))) {
                for (int i = 0; i < ks; i++) dr[i] = sf[i];
                nd = ks;
                st->ngram_rounds++;
                src = 1;
            }
            // 照抄中：下一輪多半還是照抄，先不猜下一塊（省一次 drafter，約 2 ms）；沒全中或不再照抄時下一輪開頭補猜
            df_skip = gacc && df_skip_on && src == 1 && copy;
        } else if (use_ng && ns >= 4) {
            const int need = 3;
            for (int i = ns - 2; i >= need; i--) {
                bool m = true;
                for (int q = 0; q < need && m; q++) m = seq[i - q] == seq[ns - 1 - q];
                if (!m) continue;
                if (seq[i + 1] != draft) break;
                int c = 0;
                while (c < kmax - 1 && i + 1 + c < ns) { dr[c] = seq[i + 1 + c]; c++; }
                if (c > nd) { nd = c; st->ngram_rounds++; }
                break;
            }
        }
        e->vbn = 1 + nd;
        tk[p0 + 2] = last;
        for (int i = 0; i < nd; i++) tk[p0 + 3 + i] = dr[i];
        qw_batch *b = qw_batch_begin(e->g, false);
        const double tr0 = qw_now();
        const bool sprof = getenv("QW_SPEC_PROF") && atoi(getenv("QW_SPEC_PROF")) == 2 && st->rounds == 20;
        if (sprof) { qw_batch_end(b); qw_profile_enable(e->g, true); b = qw_batch_begin(e->g, false); }
        vb_encode(e, b, p0);
        static double sp_enc = 0, sp_denc = 0, sp_dwall = 0; // QW_SPEC_PROF：CPU 編碼時間（驗證、猜測）與猜測的送出到完成
        sp_enc += (qw_now() - tr0) * 1e3;
        if (gacc) { // 接受判定 → 上下文（驗證的 VB 列都寫）→ 猜下一塊，同一個 command buffer
            const double td0 = qw_now();
            df_accept_enc(e, b, p0, nd);
            df_ctx(e, b, 0, VB, p0);
            if (!df_skip) df_draft(e, b, 0, &e->dfprm);
            ss_skip += df_skip;
            sp_denc += (qw_now() - td0) * 1e3;
        }
        const double tg_ = qw_batch_end(b);
        if (sprof) { prof_dump(e, "推測解碼一輪"); qw_profile_enable(e->g, false); }
        if (tg_ < 0) { fin = QW_FIN_ERROR; goto done; }
        sp_gpu += tg_ * 1e3;
        sp_wall += (qw_now() - tr0) * 1e3;
        sp_nb[e->vbn]++;
        const int32_t *res = e->bres.ptr;
        st->rounds++;
        e->hist[p0 + 2] = last;
        // 決定接受幾個（j）與下一個 token（nx）
        int j = 0;
        int32_t nx;
        if (greedy) {
            while (j < nd && res[j] == dr[j]) j++;
            nx = res[j];
        } else {
            const float *blg = e->blg.ptr;
            for (;;) {
                make_dist(blg + (size_t)j * QW_VOCAB, g, &d);
                if (j == nd) { nx = dist_draw(&d, &rng, -1); break; }
                if (rng_unif(&rng) < dist_p(&d, dr[j])) { j++; continue; }
                nx = dist_draw(&d, &rng, dr[j]);
                break;
            }
        }
        st->accepted += j;
        if (sstat) { ss_j[src][j]++; ss_n[src]++; ss_acc[src] += j; ss_nd[src] += nd; ss_gnb[e->vbn] += tg_ * 1e3; }
        if (getenv("QW_DF_DBG")) {
            fprintf(stderr, "[r] p0=%d nd=%d j=%d dr=", p0, nd, j);
            for (int i = 0; i < nd; i++) fprintf(stderr, "%d,", dr[i]);
            fprintf(stderr, " res=");
            for (int i = 0; i <= nd; i++) fprintf(stderr, "%d,", res[i]);
            fprintf(stderr, "\n");
        }
        dn_accept(e, j); // DeltaNet 狀態：換成第 j 個 token 之後的那份，或記下待重放的步數
        if (j >= 1) { // 狀態換成第 j 個 token 之後的那份
            qw_buf t2 = e->conv_st; e->conv_st = e->convx[j - 1]; e->convx[j - 1] = t2;
            qw_buf t3 = e->ple_st; e->ple_st = e->plex[j - 1]; e->plex[j - 1] = t3;
        }
        e->pos = p0 + 1 + j;
        if (gacc) { // GPU 已經猜好下一塊；核對 GPU 的接受判定，環狀 KV 的有效範圍只到接受的位置
            const uint32_t *pr = e->dfprm.ptr;
            if (pr[12] != (uint32_t)j || ((const int32_t *)e->dfids.ptr)[0] != nx) {
                fprintf(stderr, "DFlash2：GPU 的接受判定（j = %u）與 CPU（j = %d）不同\n", pr[12], j);
                fin = QW_FIN_ERROR;
                goto done;
            }
            e->df_done = p0 + j + 1;
            df_have = !df_skip;
            if (getenv("QW_SPEC_PROF") && st->rounds && st->rounds % 50 == 0)
                fprintf(stderr, "[spec] 前 %d 輪平均：驗證 CPU 編碼 %.2f ms、猜測 CPU 編碼 %.2f ms（同一個 command buffer）\n", st->rounds,
                        sp_enc / st->rounds, sp_denc / st->rounds);
        } else if (dfm) { // 接受的 j + 1 個位置寫進 drafter 的上下文，再從 nx（位置 p0 + j + 1）猜下一塊
            const double td0 = qw_now();
            df_set_ids(e, nx);
            qw_batch *b2 = qw_batch_begin(e->g, false);
            if (!getenv("QW_DF_NOCTX")) df_ctx(e, b2, 0, j + 1, p0);
            if (!getenv("QW_DF_NODRAFT")) df_draft(e, b2, p0 + j + 1, NULL);
            sp_denc += (qw_now() - td0) * 1e3;
            df_gms += qw_batch_end(b2) * 1e3;
            sp_dwall += (qw_now() - td0) * 1e3;
            if (getenv("QW_SPEC_PROF") && st->rounds && st->rounds % 50 == 0)
                fprintf(stderr, "[spec] 前 %d 輪平均：驗證 CPU 編碼 %.2f ms、猜測 CPU 編碼 %.2f ms、猜測送出到完成 %.2f ms\n", st->rounds,
                        sp_enc / st->rounds, sp_denc / st->rounds, sp_dwall / st->rounds);
        } else if (nx == res[j]) draft = res[VB + j];
        else {
            memcpy(e->h.ptr, (char *)e->bh.ptr + (size_t)j * HCD * 4, HCD * 4);
            draft = qw_engine_mtp_draft(e, nx, p0 + j, &ms);
        }
        for (int i = 0; i < j; i++) {
            e->hist[p0 + 3 + i] = dr[i];
            seq[ns++] = dr[i];
            EMIT(dr[i]);
        }
        seq[ns++] = nx;
        last = nx;
        EMIT(nx);
        if (g->yield && *g->yield) { fin = QW_FIN_YIELD; goto done; } // 這一輪的 token 都送出了：模型處理到 nx 之前
    }
done:
    #undef EMIT
    e->vb_flush = 0;
    e->df_tapping = false;
    e->df_mode = false;
    if (getenv("QW_SPEC_PROF") && st->rounds && dfm) fprintf(stderr, "[spec] DFlash2：每輪猜測 GPU %.2f ms\n", df_gms / st->rounds);
    if (getenv("QW_SPEC_PROF") && st->rounds)
        fprintf(stderr, "[spec] %d 輪：每輪 GPU %.2f ms、送出到完成 %.2f ms、整輪 %.2f ms；驗證數 2:%d 3:%d 4:%d 5:%d 6:%d 7:%d 8:%d\n",
                st->rounds, sp_gpu / st->rounds, sp_wall / st->rounds, (qw_now() - tw0) * 1e3 / st->rounds, sp_nb[2], sp_nb[3],
                sp_nb[4], sp_nb[5], sp_nb[6], sp_nb[7], sp_nb[8]);
    if (sstat && st->rounds) {
        fprintf(stderr, "[stats] %d 輪、生成 %d、每輪平均 %.2f 個 token\n", st->rounds, got, (double)got / st->rounds);
        for (int k = 0; k < 2; k++) {
            if (!ss_n[k]) continue;
            fprintf(stderr, "[stats] %s：%d 輪，平均猜 %.2f、接受 %.2f；接受數分布", k ? "後綴" : "DFlash2", ss_n[k], (double)ss_nd[k] / ss_n[k],
                    (double)ss_acc[k] / ss_n[k]);
            for (int i = 0; i < VBMAX; i++) fprintf(stderr, " %d:%d", i, ss_j[k][i]);
            fprintf(stderr, "\n");
        }
        if (ss_skip) fprintf(stderr, "[stats] 照抄省掉猜測 %d 輪、補猜 %d 輪（補猜每次 %.2f ms）\n", ss_skip, ss_lazy, ss_lazy ? ss_lazy_ms / ss_lazy : 0.0);
        fprintf(stderr, "[stats] 各 NB 每輪 GPU ms：");
        for (int i = 2; i <= VBMAX; i++)
            if (sp_nb[i]) fprintf(stderr, " %d:%.2f(%d)", i, ss_gnb[i] / sp_nb[i], sp_nb[i]);
        fprintf(stderr, "\n");
    }
    if (getenv("QW_ATT5_STATS") && e->q5stat.ptr) { // 驗證時 qsa_attn5 合併讀了幾個區塊 vs 各 token 各讀
        const uint32_t *s5 = e->q5stat.ptr;
        fprintf(stderr, "[spec] 稀疏 attention 合併讀取：%u / %u 個區塊（%.1f%%，一組 %d 個 token）；驗證 vs 單獨不同 %u\n", s5[0], s5[1],
                s5[1] ? 100.0 * s5[0] / s5[1] : 0.0, q5_tt(), s5[2]);
    }
    // 快照 1：生成結束時（模型已處理到 e->pos；seq 前 e->pos 個 token 就是 KV 的內容）
    if (e->pos > np && e->pos <= ns) {
        memcpy(e->kvtok, seq, sizeof(int32_t) * (size_t)e->pos);
        e->kvlen = e->pos;
        snap_why = SW_FINISH, snap_save(e, e->pos);
    }
    free(seq);
    st->gen = got;
    st->gen_ms = (qw_now() - tw0) * 1e3;
    return fin;
}

// 舊介面（greedy、固定長度），給 run 的 QW_MTP 測試用
typedef struct { int32_t *out; int n; } collect_ctx;
static bool collect_cb(void *c, int32_t t) {
    collect_ctx *x = c;
    x->out[x->n++] = t;
    return true;
}

int qw_engine_generate_mtp(qw_engine *e, const int32_t *prompt, int np, int n, int32_t *out, int *stats, double *times) {
    qw_gen g = {.temperature = 0, .max_tokens = n, .spec = true};
    collect_ctx c = {out, 0};
    qw_gen_stats st;
    qw_engine_generate_ex(e, prompt, np, &g, collect_cb, &c, &st);
    if (getenv("QW_MTP_DBG")) fprintf(stderr, "n-gram 候選 %d 輪\n", st.ngram_rounds);
    if (stats) { stats[0] = st.rounds; stats[1] = st.accepted; }
    if (times) { times[0] = st.prefill_ms; times[1] = st.gen_ms; times[2] = st.gen_ms; }
    return c.n;
}

// 2-token 驗證路徑各段的穩態時間（同一段連續跑 reps 次），對照單 token 版
// DeltaNet 驗證路徑 vs 逐 token decode（QW_KBENCH=1 QW_KBENCH_VB=1 QW_KBENCH_NB=n QW_DN_VTEST=li）：
// 同樣的 NB 個輸入（隨機），逐 token 跑 deltanet() 與一次 vb_deltanet()，比較每個 token 的輸出與最後的狀態（應逐位元相同）
static void dn_vtest(qw_engine *e, int li) {
    const int l = li + li / 3, nb = VB;
    const qw_deltanet *N = &e->m->layers[l].lin;
    const size_t soff = (size_t)li * 48 * 128 * 128 * 4, cst = (size_t)li * 3 * 10240 * 4, sb = 48 * 128 * 128 * 4, cb = 3 * 10240 * 4;
    uint64_t r = 12345;
    _Float16 *bx = e->bx.ptr;
    for (int i = 0; i < nb * D; i++) { r = r * 6364136223846793005ull + 1442695040888963407ull; bx[i] = (_Float16)(((double)(r >> 11) / 9007199254740992.0 - 0.5) * 2.0); }
    // 起始狀態：先跑幾個 token 讓狀態非零
    float *S0 = malloc(sb), *C0 = malloc(cb), *Sd = malloc(sb);
    _Float16 *od = malloc((size_t)nb * D * 2);
    const int nw = getenv("QW_DN_VW") ? atoi(getenv("QW_DN_VW")) : 3;
    // 起始狀態清零（之前的 prompt 步驟經過所有層，不同版本的捨入會讓這一層的起點不同）
    memset((char *)e->S.ptr + soff, 0, sb);
    memset((char *)e->conv_st.ptr + cst, 0, cb);
    e->dn_pmask &= ~(1ull << li);
    for (int w = 0; w < nw; w++) {
        memcpy(e->x.ptr, bx + (size_t)((w + 1) % nb) * D, D * 2);
        qw_batch *b = qw_batch_begin(e->g, false); deltanet(e, b, N, li); qw_batch_end(b);
    }
    dn_flush(e); // decode 延後寫回的狀態先補上
    memcpy(S0, (char *)e->S.ptr + soff, sb);
    memcpy(C0, (char *)e->conv_st.ptr + cst, cb);
    for (int t = 0; t < nb; t++) {
        memcpy(e->x.ptr, bx + (size_t)t * D, D * 2);
        qw_batch *b = qw_batch_begin(e->g, false); deltanet(e, b, N, li); qw_batch_end(b);
        memcpy(od + (size_t)t * D, e->out.ptr, D * 2);
    }
    dn_flush(e);
    memcpy(Sd, (char *)e->S.ptr + soff, sb);
    memcpy((char *)e->S.ptr + soff, S0, sb);
    memcpy((char *)e->conv_st.ptr + cst, C0, cb);
    qw_batch *b = qw_batch_begin(e->g, false); vb_deltanet(e, b, N, li); qw_batch_end(b);
    const _Float16 *ov = e->bout.ptr;
    double md = 0, mx = 0;
    int neq = 0;
    for (int i = 0; i < nb * D; i++) {
        const double d = fabs((double)ov[i] - (double)od[i]);
        md = d > md ? d : md;
        mx = fabs((double)od[i]) > mx ? fabs((double)od[i]) : mx;
        neq += ov[i] != od[i];
    }
    if (e->dn_vdef) { dn_accept(e, nb - 1); e->dn_pmask = 1ull << li; dn_flush(e); } // 延後寫回：重放 nb 步
    const float *Sv = nb > 1 && !e->dn_vdef ? (const float *)((char *)e->Sx[nb - 2].ptr + soff) : (const float *)((char *)e->S.ptr + soff);
    double sd = 0;
    int sneq = 0;
    for (size_t i = 0; i < sb / 4; i++) { const double d = fabs((double)Sv[i] - Sd[i]); sd = d > sd ? d : sd; sneq += Sv[i] != Sd[i]; }
    printf("DeltaNet 第 %d 層、NB = %d：輸出不同 %d / %d 個（最大差 %.3g，|out| 最大 %.3g）；最後狀態不同 %d 個（最大差 %.3g）\n", l, nb, neq, nb * D,
           md, mx, sneq, sd);
    if (getenv("QW_DN_VDUMP")) { // decode 的輸出與狀態、驗證的輸出與（重放後的）狀態、conv 狀態寫到檔案（不同版本逐位元比對用）
        FILE *f = fopen(getenv("QW_DN_VDUMP"), "wb");
        if (f) {
            fwrite(od, 2, (size_t)nb * D, f); fwrite(Sd, 1, sb, f); fwrite(ov, 2, (size_t)nb * D, f); fwrite(Sv, 1, sb, f);
            fwrite((char *)e->conv_st.ptr + cst, 1, cb, f);
            fclose(f);
        }
    }
    free(S0); free(C0); free(Sd); free(od);
}

// QSA 層：驗證路徑 vs 逐 token decode（應逐位元相同），以及各步驟的累加時間
// QW_KBENCH=1 QW_KBENCH_VB=1 QW_ATT_TEST=位置[,NB,NB…]（預設 NB 2,3,8；位置 > 2048 時走稀疏）。KV cache 填固定亂數
int cmpu(const void *x, const void *y) { const uint32_t a = *(const uint32_t *)x, b = *(const uint32_t *)y; return a < b ? -1 : a > b; }
static uint64_t at_rng(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return *s >> 11; }
static float at_rnd(uint64_t *s) { return (float)((double)at_rng(s) / 9007199254740992.0 * 2.0 - 1.0); }
static void att_test(qw_engine *e, int reps) {
    const char *arg = getenv("QW_ATT_TEST");
    const int P = atoi(arg);
    int nbs[8] = {1, 2, 3, 8}, nn = 4;
    if (strchr(arg, ',')) {
        nn = 1;
        for (const char *q = strchr(arg, ','); q && nn < 8; q = strchr(q + 1, ',')) nbs[nn++] = atoi(q + 1);
    }
    if (P < 8 || P + VBMAX + 8 > e->max_ctx || !kv_reserve(e, P + VBMAX + 8)) { fprintf(stderr, "QW_ATT_TEST：位置不合（QW_CTX 要 > 位置 + 16）\n"); return; }
    const size_t Lc = (size_t)e->kvcap;
    uint64_t s = 777;
    if (!getenv("QW_ATT_REAL")) { // KV：每層 4096 個 token 的亂數重複（時間量測不受影響）；區塊 key、raw_k 全部亂數（QW_ATT_REAL=1：沿用 prefill 寫好的真實 KV）
        const size_t rows = 4096, rowh = 2 * 256;
        _Float16 *pat = malloc(rows * rowh * 2 * 2);
        for (size_t i = 0; i < rows * rowh * 2; i++) pat[i] = (_Float16)at_rnd(&s);
        for (int ai = 0; ai < NATT; ai++)
            for (size_t t0 = 0; t0 < (size_t)P + VBMAX; t0 += rows) {
                const size_t n = t0 + rows <= Lc ? rows : Lc - t0;
                memcpy((_Float16 *)e->Kc.ptr + ((size_t)ai * Lc + t0) * rowh, pat, n * rowh * 2);
                memcpy((_Float16 *)e->Vc.ptr + ((size_t)ai * Lc + t0) * rowh, pat + rows * rowh, n * rowh * 2);
            }
        free(pat);
        for (int ai = 0; ai < NATT; ai++) {
            _Float16 *kb = (_Float16 *)e->kblk.ptr + (size_t)ai * (Lc / 4 + 1) * 128;
            for (size_t i = 0; i < ((size_t)P / 4 + 4) * 128; i++) kb[i] = (_Float16)(at_rnd(&s) * 1.5f);
            float *rk = (float *)e->rawk.ptr + (size_t)ai * Lc * 128;
            for (size_t i = 0; i < ((size_t)P + VBMAX) * 128; i++) rk[i] = at_rnd(&s);
        }
    }
    _Float16 *bx = e->bx.ptr;
    for (int i = 0; i < VBMAX * D; i++) bx[i] = (_Float16)(at_rnd(&s) * 2.0f);
    if (getenv("QW_ATT_SIM")) { // 驗證的 token 輸入相近（選到的區塊大多重疊，像真實的連續 token）：x_t = x_0 + f·雜訊
        const float f = (float)atof(getenv("QW_ATT_SIM"));
        for (int t = 1; t < VBMAX; t++)
            for (int i = 0; i < D; i++) bx[(size_t)t * D + i] = (_Float16)((float)bx[i] + f * (float)bx[(size_t)t * D + i]);
    }
    // 1) 正確性：每個 NB，第 3、47 層
    _Float16 *od = malloc((size_t)VBMAX * D * 2), *oa = malloc((size_t)VBMAX * 6144 * 2);
    const size_t sst = (size_t)(e->max_ctx / 4 + 1);
    float *osc = malloc((size_t)VBMAX * sst * 4); // decode 時每個 token 的 indexer 分數（q3sc 第 0 列）
    for (int k = 1; k < nn; k++) {
        const int nb = nbs[k];
        if (nb < 2 || nb > VBMAX) continue;
        e->vbn = nb;
        for (int ai = 0; ai < NATT; ai += getenv("QW_ATT_ALL") ? 1 : NATT - 1) { // QW_ATT_ALL：每一層都比
            const qw_attn *A = &e->m->layers[ai * 4 + 3].att;
            for (int t = 0; t < nb; t++) {
                memcpy(e->x.ptr, bx + (size_t)t * D, D * 2);
                e->pos = P + t;
                qw_batch *b = qw_batch_begin(e->g, false); attention(e, b, A, ai); qw_batch_end(b);
                memcpy(od + (size_t)t * D, e->out.ptr, D * 2);
                memcpy(oa + (size_t)t * 6144, e->ay.ptr, 6144 * 2);
                memcpy(osc + (size_t)t * sst, e->q3sc.ptr, sst * 4);
            }
            if (P + nb > QW_IDX_BUDGET) { // 選取 vs CPU（最後一個 decode token：q3sc / q3sel 的第 0 列）
                const uint32_t pos = (uint32_t)(P + nb - 1), nv = (pos + 1) / QW_IDX_CR;
                const uint32_t *sc = e->q3sc.ptr, *sl = e->q3sel.ptr;
                uint32_t *srt = malloc(nv * 4);
                memcpy(srt, sc, nv * 4);
                int cmpu(const void *x, const void *y);
                qsort(srt, nv, 4, cmpu);
                const uint32_t thr = nv > 512 ? srt[nv - 512] : 0; // 第 512 大
                uint32_t k = 0, ngt = 0, neq = 0;
                int bad = 0;
                for (uint32_t i = 0; i < nv && nv > 512; i++) ngt += sc[i] > thr;
                for (uint32_t i = 0; i < nv && nv > 512; i++) // 依位置排序：> thr 的，加上前 512 − ngt 個 = thr 的
                    if (sc[i] > thr || (sc[i] == thr && neq++ < 512 - ngt)) bad += sl[k++] != i;
                printf("選取 vs CPU（位置 %u、%u 個區塊）：%d 個不同\n", pos, nv, bad);
                free(srt);
            }
            e->pos = P;
            memset(e->q5stat.ptr, 0, 8);
            qw_batch *b = qw_batch_begin(e->g, false); vb_attention(e, b, A, ai, P); qw_batch_end(b);
            if (getenv("QW_ATT5_STATS")) {
                const uint32_t *st = e->q5stat.ptr;
                printf("合併讀取：%u / %u 個區塊（%.1f%%）\n", st[0], st[1], st[1] ? 100.0 * st[0] / st[1] : 0.0);
            }
            int bad = 0, bada = 0, bads = 0;
            double md = 0;
            for (int t = 0; t < nb; t++) { // indexer 分數：驗證的第 t 列 vs decode
                const uint32_t pos = (uint32_t)(P + t), nv = (pos + 1) / QW_IDX_CR;
                if (pos + 1 > QW_IDX_BUDGET) bads += memcmp((const float *)e->q3sc.ptr + (size_t)t * sst, osc + (size_t)t * sst, nv * 4) != 0;
            }
            for (int t = 0; t < nb; t++) {
                const _Float16 *v = (const _Float16 *)e->bout.ptr + (size_t)t * D;
                for (int i = 0; i < D; i++) {
                    bad += v[i] != od[(size_t)t * D + i];
                    const double d = fabs((double)v[i] - (double)od[(size_t)t * D + i]);
                    md = d > md ? d : md;
                }
                bada += memcmp((const _Float16 *)e->bay.ptr + (size_t)t * 6144, oa + (size_t)t * 6144, 6144 * 2) != 0;
                if (getenv("QW_ATT_DBG")) {
                    int nd = 0, f = -1;
                    for (int i = 0; i < 6144; i++)
                        if (oa[(size_t)t * 6144 + i] != ((const _Float16 *)e->bay.ptr)[(size_t)t * 6144 + i]) { nd++; if (f < 0) f = i; }
                    printf("  t%d：ay 不同 %d 個，第一個 %d（decode %g、verify %g）\n", t, nd, f, f < 0 ? 0.0 : (double)oa[(size_t)t * 6144 + f],
                           f < 0 ? 0.0 : (double)((const _Float16 *)e->bay.ptr)[(size_t)t * 6144 + f]);
                }
            }
            printf("第 %d 層、位置 %d、NB = %d：輸出不同 %d / %d 個（最大差 %.3g）；gate 後 attention 輸出不同的 token %d / %d；分數不同的 token %d\n", ai * 4 + 3, P, nb, bad,
                   nb * D, md, bada, nb, bads);
        }
    }
    free(od); free(oa); free(osc);
    // 2) 各步驟累加時間（12 層輪流）。QW_ATT_PS=位置,位置…（≤ P，KV 已填到 P）：同一個行程裡交錯量各位置（避開降頻造成的差異）
    static const char *nm[] = {"投影 q/k/v/idx", "+ 準備", "+ 打分數", "+ 選取", "+ attention", "+ 合併", "+ gate", "+ o_proj（全部）"};
    int ps[8] = {P}, np = 1;
    if (getenv("QW_ATT_PS")) {
        np = 0;
        for (const char *q = getenv("QW_ATT_PS"); *q && np < 8;) {
            char *end;
            const long v = strtol(q, &end, 10);
            if (end == q) break;
            if (v >= 8 && v <= P) ps[np++] = (int)v;
            q = *end == ',' ? end + 1 : end;
        }
        if (!np) ps[np++] = P;
    }
    printf("\n| 步驟 |");
    for (int pi = 0; pi < np; pi++)
        for (int k = 0; k < nn; k++) printf(" %d（%s）NB = %d |", ps[pi], ps[pi] + 1 > QW_IDX_BUDGET ? "稀疏" : "全看", nbs[k]);
    printf("\n|---|");
    for (int k = 0; k < nn * np; k++) printf("---|");
    printf("\n");
    const int nst = getenv("QW_ATT_MASK") ? 1 : 8; // QW_ATT_MASK=遮罩：只量這些步驟
    for (int st = 0; st < nst; st++) {
        at_mask = getenv("QW_ATT_MASK") ? (unsigned)strtoul(getenv("QW_ATT_MASK"), NULL, 0) : (2u << st) - 1;
        if (nst > 1) printf("| %s |", nm[st]);
        else printf("| 遮罩 0x%x |", at_mask);
        for (int pi = 0; pi < np; pi++)
            for (int k = 0; k < nn; k++) {
                const int nb = nbs[k], P1 = ps[pi];
                if (nb > 1) e->vbn = nb;
                double best = 1e30;
                for (int t = 0; t < 3; t++) {
                    qw_batch *b = qw_batch_begin(e->g, false);
                    for (int r = 0; r < reps; r++) {
                        const int ai = r % NATT;
                        e->pos = P1;
                        if (nb == 1) attention(e, b, &e->m->layers[ai * 4 + 3].att, ai);
                        else vb_attention(e, b, &e->m->layers[ai * 4 + 3].att, ai, P1);
                        if (getenv("QW_ATT_SERIAL")) qw_batch_barrier(b); // 層與層之間不重疊（像真正的 decode）
                    }
                    const double tt = qw_batch_end(b);
                    if (tt > 0 && tt < best) best = tt;
                }
                printf(" %.1f µs |", best / reps * 1e6);
            }
        printf("\n");
    }
    at_mask = ~0u;
    e->pos = P;
}

static void kbench_vb(qw_engine *e, int reps) {
    char err[1024];
    if (!vb_init(e, err, sizeof err)) { fprintf(stderr, "%s\n", err); return; }
    if (getenv("QW_KBENCH_NB")) e->vbn = atoi(getenv("QW_KBENCH_NB"));
    if (getenv("QW_DN_VTEST")) { dn_vtest(e, atoi(getenv("QW_DN_VTEST"))); return; }
    if (getenv("QW_ATT_TEST")) { att_test(e, reps); return; }
    if (getenv("QW_DN_KB")) { // DeltaNet 驗證路徑的各步驟（12 層輪流）：NB = 1 與 QW_KBENCH_NB
        static const char *nm[] = {"投影（qkv/z/a/b）", "conv + 正規化", "GDN 遞迴", "gated RMSNorm", "out_proj", "全部"};
        static const unsigned mk[] = {1, 2, 4, 8, 16, 31};
        const int nbs[2] = {1, e->vbn};
        printf("| 步驟 | NB = 1 | NB = %d |\n|---|---|---|\n", nbs[1]);
        for (int s = 0; s < 6; s++) {
            double best[2] = {1e30, 1e30};
            for (int v = 0; v < 2; v++) {
                e->vbn = nbs[v];
                dn_mask = mk[s];
                for (int t = 0; t < 3; t++) {
                    qw_batch *b = qw_batch_begin(e->g, false);
                    for (int r = 0; r < reps; r++) {
                        vb_deltanet(e, b, &e->m->layers[(r % 12) * 4].lin, (r % 12) * 3);
                        if (e->dn_vdef && (dn_mask & 4)) { e->dn_lJ[(r % 12) * 3] = (uint8_t)e->vbn; e->dn_pmask |= 1ull << ((r % 12) * 3); } // 延後寫回：重放算進下一次
                    }
                    const double tt = qw_batch_end(b);
                    if (tt > 0 && tt < best[v]) best[v] = tt;
                }
            }
            printf("| %s | %.1f µs | %.1f µs |\n", nm[s], best[0] / reps * 1e6, best[1] / reps * 1e6);
        }
        dn_mask = ~0u;
        e->vbn = nbs[1];
        return;
    }
    printf("NB = %d\n", e->vbn);
    const char *names[] = {"hyper-connection", "DeltaNet 層", "QSA 層", "MoE", "lm_head"};
    printf("| 段 | 單 token | 2 token | 比值 |\n|---|---|---|---|\n");
    for (int p = 0; p < 5; p++) {
        double best[2] = {1e30, 1e30};
        for (int v = 0; v < 2; v++)
            for (int t = 0; t < 3; t++) {
                qw_batch *b = qw_batch_begin(e->g, false);
                for (int r = 0; r < reps; r++) {
                    const qw_layer *L = &e->m->layers[r % 48];
                    switch (p * 2 + v) {
                    case 0: hyper_in(e, b, &L->attn_hc); break;
                    case 1: e->bpending = false; vb_hyper(e, b, &L->attn_hc); break;
                    case 2: deltanet(e, b, &e->m->layers[(r % 12) * 4].lin, (r % 12) * 3); break;
                    case 3:
                        vb_deltanet(e, b, &e->m->layers[(r % 12) * 4].lin, (r % 12) * 3);
                        if (e->dn_vdef) { e->dn_lJ[(r % 12) * 3] = (uint8_t)e->vbn; e->dn_pmask |= 1ull << ((r % 12) * 3); } // 延後寫回：重放算進下一次
                        break;
                    case 4: attention(e, b, &e->m->layers[(r % 12) * 4 + 3].att, r % 12); break;
                    case 5: vb_attention(e, b, &e->m->layers[(r % 12) * 4 + 3].att, r % 12, e->pos); break;
                    case 6: moe(e, b, &L->moe); break;
                    case 7: vb_moe(e, b, &L->moe); break;
                    case 8: qmv(e, b, &e->m->lm_head, &e->x, 0, &e->logits, true); break;
                    case 9: vb_qmv(e, b, &e->m->lm_head, &e->bx, &e->blg, true); break;
                    }
                }
                double tt = qw_batch_end(b);
                if (tt > 0 && tt < best[v]) best[v] = tt;
            }
        printf("| %s | %.1f µs | %.1f µs | %.2f |\n", names[p], best[0] / reps * 1e6, best[1] / reps * 1e6, best[1] / best[0]);
    }
    // tensor 單元版（prefill 的 qmm，M = NB）對照 SIMD 版
    const qw_qlin *Ls[3] = {&e->m->lm_head, &e->m->layers[0].lin.in_qkv, &e->m->layers[0].lin.out};
    const char *ln[3] = {"lm_head", "DeltaNet in_qkv", "DeltaNet out"};
    const int cfg[][3] = {{8, 32, 1}, {8, 64, 2}, {16, 64, 2}, {8, 128, 4}, {16, 128, 4}, {32, 64, 2}};
    printf("\n| 矩陣 | SIMD qmvb |");
    for (size_t c = 0; c < sizeof cfg / sizeof *cfg; c++) printf(" qmm %dx%dx%d |", cfg[c][0], cfg[c][1], cfg[c][2]);
    printf("\n");
    for (int i = 0; i < 3; i++) {
        const qw_qlin *L = Ls[i];
        if (L->gq) continue; // GGUF 格式：qmm 的 affine 版不適用
        const bool fo = i == 0;
        printf("| %s |", ln[i]);
        for (int c = -1; c < (int)(sizeof cfg / sizeof *cfg); c++) {
            double best = 1e30;
            for (int t = 0; t < 3; t++) {
                qw_batch *b = qw_batch_begin(e->g, false);
                for (int r = 0; r < reps; r++) {
                    if (c < 0) vb_qmv(e, b, L, i == 2 ? &e->bgy : &e->bx, &e->blg, fo);
                    else {
                        const int tm = cfg[c][0], tn = cfg[c][1], sg = cfg[c][2];
                        qw_kernel *k = qmm_kernel(e, L->bits, L->in, tm, tn, sg, false, fo);
                        qmm_args a = {(uint32_t)VB, (uint32_t)L->out};
                        qw_batch_kernel(b, k);
                        qw_batch_buf(b, 0, i == 2 ? &e->bgy : &e->bx, 0);
                        bind_t(e, b, 1, L->w);
                        bind_t(e, b, 2, L->s);
                        bind_t(e, b, 3, L->b);
                        qw_batch_buf(b, 4, &e->blg, 0);
                        qw_batch_bytes(b, 5, &a, sizeof a);
                        qw_batch_dispatch(b, (unsigned)L->out / tn, (VB + tm - 1) / tm, 1, 32u * sg, 1, 1);
                    }
                }
                double tt = qw_batch_end(b);
                if (tt > 0 && tt < best) best = tt;
            }
            printf(" %.1f µs |", best / reps * 1e6);
        }
        printf("\n");
    }
}

// ================================================================ DFlash2 drafter
// 流程（對照 Strata dflash2.inc）：
//   特徵   主模型第 taps[i] 層的完整輸出（4 條殘差流的平均）→ dftap [列][5·H] half
//   上下文 特徵 → fc → hidden_norm → 每層的 k（norm + RoPE）與 v → 環狀 KV（槽 = 位置 % DF_RING）
//   猜     anchor + 7 個 mask 一起過 5 層（動態卷積、windowed attention：環狀 KV + block 自己），final norm →
//          draft vocab 的 lm_head → 每列前 16 名 → 候選挑選由左到右走出 7 個 token
enum { DF_RING = 2064, DF_ROWS = 8, DF_TOPK = 16 };
enum { DK_TAP, DK_EMBED, DK_RMS, DK_CONV, DK_ATTN, DK_SWIGLU, DK_TOPK, DK_SEL, DK_TOPK1, DK_TOPK2, DK_ACCEPT, DK_NR3, DK_CONVRMS, DK_ATSPLIT, DK_ATMERGE, DK_N };
enum { DF_TKC = 256 }; // top-K 第一階段分幾段（每段一個 SIMD group，約 nv / 256 ≈ 415 個，每個 lane ≤ 24 個）
static_assert(DF_TOPK <= 16 && DF_ROWS <= 8, "df_select / df_topk2：最多 16 個候選、8 列（threadgroup memory 與暫存器陣列的大小）");
static_assert(DF_TKC * DF_TOPK <= 32 * 32 * 8, "df_topk2：第二階段的候選數上限（32 個 SIMD group × 32 lane × TK2_PL）");

static int df_q4_notc;

static bool df_init(qw_engine *e, char *err, size_t errlen) {
    if (e->df_k[0]) return true;
    const qw_dflash *d = &e->df;
    char defs[512];
    snprintf(defs, sizeof defs, "#define H %d\n#define HC %d\n#define NH %d\n#define NKV %d\n#define HD %d\n#define GS %d\n#define RING %d\n#define EB %d\n#define CBQ4 %d\n",
             d->H, QW_HC, d->NH, d->NKV, d->HD, d->gs, DF_RING,
             e->m->embed_bits ? e->m->embed_bits : 4, d->cb_q4 ? 1 : 0); // GGUF 的 embed 不用 df_embed（gg_embed），EB 只是讓它編得過
    if (!compile(e, defs, src_dflash, sizeof src_dflash, err, errlen)) return false;
    static const char *nm[] = {"df_tap", "df_embed", "df_rms", "df_conv", "df_attn", "df_swiglu", "df_topk", "df_select",
                               "df_topk1", "df_topk2", "df_accept", "df_nr3", "df_conv_rms", "df_attn_split", "df_attn_merge"};
    static_assert(sizeof nm / sizeof nm[0] == DK_N && DK_N <= 16, "df_k 的大小");
    for (int i = 0; i < DK_N; i++)
        if (!(e->df_k[i] = need(e, nm[i], err, errlen))) return false;
    const size_t R = DF_ROWS, H = (size_t)d->H, KV = (size_t)d->NKV * d->HD, Q = (size_t)d->NH * d->HD;
    const size_t taprows = (size_t)(PF > VBMAX ? PF : VBMAX);
    const bool ok = alloc(e, &e->dfKr, (size_t)d->L * DF_RING * KV * 2) && alloc(e, &e->dfVr, (size_t)d->L * DF_RING * KV * 2) &&
                    alloc(e, &e->dftap, taprows * d->ntaps * H * 2) && alloc(e, &e->dfids, R * 4) && alloc(e, &e->dfx, R * H * 4) &&
                    alloc(e, &e->dfxn, R * H * 4) && alloc(e, &e->dfxh, (R + 1) * H * 2) /* 多一列：draft head 從第 1 列起讀 8 列（矩陣單元版）*/ && alloc(e, &e->dfc, R * H * 4) &&
                    alloc(e, &e->dfdyn, R * 4 * (H / d->gs) * 4) && alloc(e, &e->dfq, R * Q * 2) && alloc(e, &e->dfk, R * KV * 2) &&
                    alloc(e, &e->dfv, R * KV * 2) && alloc(e, &e->dfqr, R * Q * 2) && alloc(e, &e->dfkb, R * KV * 2) &&
                    alloc(e, &e->dfvb, R * KV * 2) && alloc(e, &e->dfo, R * Q * 2) && alloc(e, &e->dfof, R * H * 4) &&
                    alloc(e, &e->dfg, R * d->inter * 4) && alloc(e, &e->dfu, R * d->inter * 4) && alloc(e, &e->dfgu, R * d->inter * 2) &&
                    alloc(e, &e->dfm, R * H * 4) && alloc(e, &e->dflog, R * (size_t)d->nv * 4) && alloc(e, &e->dftv, R * DF_TOPK * 4) &&
                    alloc(e, &e->dfti, R * DF_TOPK * 4) && alloc(e, &e->dfcv, R * DF_TKC * DF_TOPK * 4) &&
                    alloc(e, &e->dfci, R * DF_TKC * DF_TOPK * 4) && alloc(e, &e->dfhp, R * d->rank * 4) && alloc(e, &e->dfpath, R * 4) &&
                    alloc(e, &e->dfprob, R * 4) && alloc(e, &e->dfch, R * H * 4) && alloc(e, &e->dfprm, 64) &&
                    alloc(e, &e->dfatp, (size_t)d->NKV * ((d->win + R + 31) / 32) * 32 * (d->HD + 2) * 4); // df_attn_split 的各段結果
    if (!ok) { snprintf(err, errlen, "DFlash2：配置 buffer 失敗"); return false; }
    // 矩陣單元版 Q4_0 matvec 假設 cooperative tensor 的排列（kernels/df_q4.metal）：先在這台機器上核對，不符就一律走 ALU 版
    if (df_q4_notc < 0) df_q4_notc = getenv("QW_DF_NOTC") != NULL;
    if (!df_q4_notc) {
        qw_buf chk;
        if (!qw_buf_new(e->g, 16, &chk)) { snprintf(err, errlen, "DFlash2：配置 buffer 失敗"); return false; }
        memset(chk.ptr, 0, 16);
        qw_batch *b = qw_batch_begin(e->g, false);
        qw_batch_kernel(b, df_q4_kernel(e, "layout", d->H, DF_ROWS, false));
        qw_batch_buf(b, 0, &chk, 0);
        qw_batch_dispatch(b, 1, 1, 1, 32, 1, 1);
        const bool ran = qw_batch_end(b) >= 0;
        const uint32_t bad = *(const uint32_t *)chk.ptr;
        qw_buf_free(&chk);
        if (!ran || bad) {
            fprintf(stderr, "DFlash2：矩陣單元的排列與假設不符（%u 處），matvec 改走 ALU 版\n", bad);
            df_q4_notc = 1;
        }
    }
    // 預先編譯各列數的 kernel（上下文每輪 1–8 列；第一次用到才編譯會讓那一輪多等幾十 ms）
    e->df_lo = 0;
    for (int r = 1; r <= DF_ROWS; r++) {
        qw_batch *b = qw_batch_begin(e->g, false);
        df_ctx(e, b, 0, r, 0);
        qw_batch_end(b);
    }
    df_set_ids(e, 0);
    qw_batch *b = qw_batch_begin(e->g, false);
    df_draft(e, b, 8, NULL);
    qw_batch_end(b);
    e->df_done = 0;
    return true;
}

// Q4_0 拆開版（src/dflash.c：w = nibble、s = FP16 d、b = NULL）的 matvec（kernels/df_q4.metal）
static bool df_isq4(const qw_qlin *L) { return !L->gq && !L->b && L->bits == 4; }
static int df_q4_rps(void) { static int v = 0; if (!v) v = getenv("QW_DF_RPS") ? atoi(getenv("QW_DF_RPS")) : 2; return v; } // ALU 版每個 SIMD group 的列數
static int df_q4_notc = -1; // 1：TC 版不能用（QW_DF_NOTC=1，或 df_init 核對矩陣單元的排列不符）
// 一個 (K, 列數, 輸出型別) 編一份，裡面有 TC（_tc）、gate·up（_glu）、ALU（_mv）三個 kernel
static qw_kernel *df_q4_kernel(qw_engine *e, const char *kind, int K, int nb, bool fout) {
    char base[64], name[80], defs[512], err[8192];
    snprintf(base, sizeof base, "df_q4_k%d_n%d%s", K, nb, fout ? "_f" : "");
    snprintf(name, sizeof name, "%s_%s", base, kind);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs, "#define df_q4tc %s_tc\n#define df_q4tc_glu %s_glu\n#define df_q4mv %s_mv\n#define df_q6tc %s_q6\n#define df_q4_layout %s_layout\n"
             "#define df_q6tcb %s_q6b\n#define df_q6b_layout %s_q6bl\n#define Q6B_NSG %d\n#define K %d\n#define NB %d\n#define NSG %d\n%s"
             "#define Q4_LO %d\n#define Q6_LO %d\n#define KT %d\n#define RPS %d\n",
             base, base, base, base, base, base, base, DF_Q6B_NSG, K, nb, DF_Q4_NSG, fout ? "#define YT float\n" : "", getenv("QW_DF_Q4LO") ? atoi(getenv("QW_DF_Q4LO")) : 0,
             getenv("QW_DF_Q6LO") ? atoi(getenv("QW_DF_Q6LO")) : 0, getenv("QW_DF_KT") ? atoi(getenv("QW_DF_KT")) : 64, df_q4_rps());
    if (!compile(e, defs, src_df_q4, sizeof src_df_q4, err, sizeof err)) {
        fprintf(stderr, "編譯 %s 失敗：%s\n", base, err);
        exit(1);
    }
    return qw_kernel_get(e->g, name);
}
// TC 版的條件：每個矩陣的 N 是 32 的倍數、x 從 xoff 起至少 8 列（列距 = K）；QW_DF_NOTC=1 一律走 ALU 版
static bool df_q4_tc_ok(const qw_qlin *const *L, int n, const qw_buf *x, size_t xoff) {
    if (df_q4_notc < 0) df_q4_notc = getenv("QW_DF_NOTC") != NULL;
    if (df_q4_notc) return false;
    for (int i = 0; i < n; i++)
        if (L[i]->out % 32) return false;
    return xoff + (size_t)8 * L[0]->in * 2 <= x->size;
}
// n 個矩陣（K 相同）共用輸入 x（rows 列、列距 K），輸出 y[i]（列距 = 各自的 N）。glu：n = 2（gate、up），y[0] 寫 silu(g)·u（half）
static void df_q4(qw_engine *e, qw_batch *b, const qw_qlin *const *L, const qw_buf *const *y, int n, const qw_buf *x, size_t xoff, int rows,
                  bool fout, bool glu) {
    const int K = L[0]->in;
    for (int i = 1; i < n; i++)
        if (L[i]->in != K || !df_isq4(L[i])) { fprintf(stderr, "df_q4：矩陣的 K 不同\n"); exit(1); }
    const bool tc = df_q4_tc_ok(L, n, x, xoff);
    if (glu && !tc) { fprintf(stderr, "df_q4：gate·up 需要 TC 版\n"); exit(1); }
    struct { uint32_t N[4], off[4], total, xstride; } a = {{0}, {0}, 0, (uint32_t)K};
    for (int i = 0; i < n; i++) {
        a.N[i] = (uint32_t)L[i]->out;
        a.off[i] = a.total;
        a.total += (uint32_t)L[i]->out;
    }
    qw_batch_kernel(b, df_q4_kernel(e, glu ? "glu" : tc ? "tc" : "mv", K, rows, glu ? false : fout));
    for (int i = 0; i < 4; i++) {
        const int j = i < n ? i : 0;
        bind_t(e, b, 2u * (unsigned)i, L[j]->w);
        bind_t(e, b, 2u * (unsigned)i + 1, L[j]->s);
        qw_batch_buf(b, 9u + (unsigned)i, y[j], 0);
    }
    qw_batch_buf(b, 8, x, xoff);
    qw_batch_bytes(b, 13, &a, sizeof a);
    if (glu) qw_batch_dispatch(b, a.N[0] / 32, 1, 1, 32 * DF_Q4_NSG, 1, 1);
    else if (tc) qw_batch_dispatch(b, a.total / 32, 1, 1, 32 * DF_Q4_NSG, 1, 1);
    else {
        const unsigned rps = (unsigned)df_q4_rps();
        for (int i = 0; i < n; i++)
            if (a.N[i] % rps) { fprintf(stderr, "df_q4：N = %u 不是 %u 的倍數\n", a.N[i], rps); exit(1); }
        qw_batch_dispatch(b, (a.total + 8 * rps - 1) / (8 * rps), 1, 1, 256, 1, 1);
    }
}

static void df_mv(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *x, size_t xoff, const qw_buf *y, size_t yoff, int rows, bool fout);
// draft vocab 的 lm_head（GGUF 區塊，例如 Q6_K）：qmv_ggm（半個 SIMD group 一個 256 值超區塊，llama.cpp 的排列）比通用的
// qmv_gg（每個 lane 一段 32 個值，K = 2560 時 80 段分給 32 個 lane 不平均）快；不支援的格式退回 df_mv
static void df_head(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *x, size_t xoff, const qw_buf *y, int rows) {
    if (L->gq == 15 && !getenv("QW_DF_HEAD_ALU") && df_q4_notc == 0 && xoff + (size_t)8 * L->in * 2 <= x->size && L->in % (64 * DF_Q4_NSG) == 0) { // Q6_K：矩陣單元版
        struct { uint32_t N[4], off[4], total, xstride; } a = {{(uint32_t)L->out}, {0}, (uint32_t)L->out, (uint32_t)L->in};
        const bool kb = q6b_ok && L->in % (128 * DF_Q6B_NSG) == 0; // 排列核對過（lm_tc_init）：用 K 方向一次 128 個值的版本（讀取少）
        qw_batch_kernel(b, df_q4_kernel(e, kb ? "q6b" : "q6", L->in, rows, true));
        bind_t(e, b, 0, L->w);
        qw_batch_buf(b, 8, x, xoff);
        qw_batch_buf(b, 9, y, 0);
        qw_batch_bytes(b, 13, &a, sizeof a);
        qw_batch_dispatch(b, ((unsigned)L->out + 31) / 32, 1, 1, 32 * (kb ? DF_Q6B_NSG : DF_Q4_NSG), 1, 1);
        return;
    }
    if (!L->gq || !qmv_ggm_ok(&L, 1) || getenv("QW_DF_HEAD_GG")) { df_mv(e, b, L, x, xoff, y, 0, rows, true); return; }
    char base[128], name[140], defs[512], err[8192];
    snprintf(base, sizeof base, "df_head_k%d_%d", L->in, L->gq - 1);
    snprintf(name, sizeof name, "%s_b%d", base, rows); // 一次編出 NB 1…8（qmv_ggm 的 NB = 0），輸出 float
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (!k) {
        snprintf(defs, sizeof defs, "#define IQ_DQ_ONLY 1\n#define qmv_ggm %s\n#define KD %d\n#define NB 0\n#define NMAT 1\n#define GF0 %d\n#define YT float\n",
                 base, L->in, L->gq - 1);
        if (!compile3(e, defs, src_iq_tables, sizeof src_iq_tables, src_moe_iq, sizeof src_moe_iq, src_qmv_ggm, sizeof src_qmv_ggm, err, sizeof err)) {
            fprintf(stderr, "編譯 %s 失敗：%s\n", base, err);
            exit(1);
        }
        k = qw_kernel_get(e->g, name);
    }
    struct { uint32_t N[4], off[4], total, xstride; } a = {{(uint32_t)L->out}, {0}, (uint32_t)L->out, (uint32_t)L->in};
    qw_batch_kernel(b, k);
    for (unsigned i = 0; i < 4; i++) {
        bind_t(e, b, i, L->w);
        qw_batch_buf(b, 5 + i, y, 0);
    }
    qw_batch_buf(b, 4, x, xoff);
    qw_batch_bytes(b, 9, &a, sizeof a);
    qw_batch_dispatch(b, (a.total + 7) / 8, 1, 1, 256, 1, 1);
}

// qmvb：rows 列（≤ 8）一起，權重讀一次
static void df_mv(qw_engine *e, qw_batch *b, const qw_qlin *L, const qw_buf *x, size_t xoff, const qw_buf *y, size_t yoff, int rows, bool fout) {
    if (df_isq4(L)) {
        if (yoff) { fprintf(stderr, "df_mv：Q4_0 版不支援輸出位移\n"); exit(1); }
        df_q4(e, b, &L, &y, 1, x, xoff, rows, fout, false);
        return;
    }
    const int keep = e->vbn;
    e->vbn = rows;
    vb_qmv_o(e, b, L, x, xoff, y, yoff, fout);
    e->vbn = keep;
}

static void df_rmsk(qw_engine *e, qw_batch *b, const qw_buf *x, const st_tensor *w, const qw_buf *y, const qw_buf *yh, int rows) {
    qw_batch_kernel(b, e->df_k[DK_RMS]);
    qw_batch_buf(b, 0, x, 0);
    bind_t(e, b, 1, w);
    qw_batch_buf(b, 2, y, 0);
    qw_batch_buf(b, 3, yh, 0);
    qw_batch_bytes(b, 4, &e->df.eps, 4);
    qw_batch_dispatch(b, (unsigned)rows, 1, 1, 256, 1, 1);
}

static void df_convk(qw_engine *e, qw_batch *b, const qw_buf *x, const st_tensor *base, uint32_t part, const qw_buf *resid,
                     const qw_buf *out, const qw_buf *outh, int rows) {
    const uint32_t hr = resid ? 1 : 0;
    qw_batch_kernel(b, e->df_k[DK_CONV]);
    qw_batch_buf(b, 0, x, 0);
    qw_batch_buf(b, 1, &e->dfdyn, 0);
    bind_t(e, b, 2, base);
    qw_batch_buf(b, 3, resid ? resid : x, 0);
    qw_batch_buf(b, 4, out, 0);
    qw_batch_buf(b, 5, outh, 0);
    qw_batch_bytes(b, 6, &part, 4);
    qw_batch_bytes(b, 7, &hr, 4);
    qw_batch_dispatch(b, (unsigned)(e->df.H + 255) / 256, (unsigned)rows, 1, 256, 1, 1);
}

// q、k、v（nq = 0：只有 k、v）的 norm + RoPE 一次 dispatch（df_nr3）；ring：k、v 寫環狀 KV（kd / vd 的位移 koff）
// pb 非 NULL 時位置從 pb 的第 4 個 uint 讀（df_accept 寫的），pos0 不用
static void df_nr3k(qw_engine *e, qw_batch *b, const qw_df_layer *Ly, uint32_t nq, const qw_buf *qd, const qw_buf *kd, const qw_buf *vd,
                    size_t koff, uint32_t ring, uint32_t pos0, int rows, const qw_buf *pb) {
    const struct { uint32_t nq, nk, ring; float eps, theta; } a = {nq, (uint32_t)e->df.NKV, ring, e->df.eps, e->df.theta};
    qw_batch_kernel(b, e->df_k[DK_NR3]);
    qw_batch_buf(b, 0, &e->dfq, 0);
    qw_batch_buf(b, 1, &e->dfk, 0);
    qw_batch_buf(b, 2, &e->dfv, 0);
    bind_t(e, b, 3, Ly->q_norm);
    bind_t(e, b, 4, Ly->k_norm);
    qw_batch_buf(b, 5, qd ? qd : &e->dfqr, 0);
    qw_batch_buf(b, 6, kd, koff);
    qw_batch_buf(b, 7, vd, koff);
    if (pb) qw_batch_buf(b, 8, pb, 16);
    else qw_batch_bytes(b, 8, &pos0, 4);
    qw_batch_bytes(b, 9, &a, sizeof a);
    qw_batch_dispatch(b, nq + 2 * (uint32_t)e->df.NKV, (unsigned)rows, 1, (unsigned)e->df.HD, 1, 1);
}

// 卷積（finish，有殘差：dfx += …）接著 RMSNorm（w）→ dfxn、dfxh，一次 dispatch（df_conv_rms）
static void df_convrms(qw_engine *e, qw_batch *b, const qw_buf *x, const st_tensor *base, const st_tensor *w, int rows) {
    const uint32_t part = 1;
    qw_batch_kernel(b, e->df_k[DK_CONVRMS]);
    qw_batch_buf(b, 0, x, 0);
    qw_batch_buf(b, 1, &e->dfdyn, 0);
    bind_t(e, b, 2, base);
    qw_batch_buf(b, 3, &e->dfx, 0);
    qw_batch_buf(b, 4, &e->dfx, 0);
    qw_batch_bytes(b, 5, &part, 4);
    bind_t(e, b, 6, w);
    qw_batch_buf(b, 7, &e->dfxn, 0);
    qw_batch_buf(b, 8, &e->dfxh, 0);
    qw_batch_bytes(b, 9, &e->df.eps, 4);
    qw_batch_dispatch(b, (unsigned)rows, 1, 1, 256, 1, 1);
}

// 上下文：dftap 的第 row0.. 列（位置 pos0..）寫進環狀 KV，每次最多 8 列
static void df_ctx(qw_engine *e, qw_batch *b, int row0, int n, int pos0) {
    const qw_dflash *d = &e->df;
    const size_t KVs = (size_t)DF_RING * d->NKV * d->HD * 2, tapw = (size_t)d->ntaps * d->H;
    for (int c = 0; c < n; c += DF_ROWS) {
        const int r = n - c < DF_ROWS ? n - c : DF_ROWS;
        df_mv(e, b, &d->fc, &e->dftap, (size_t)(row0 + c) * tapw * 2, &e->dfch, 0, r, true);
        df_rmsk(e, b, &e->dfch, d->hidden_norm, &e->dfxn, &e->dfxh, r);
        for (int l = 0; l < d->L; l++) {
            const qw_df_layer *Ly = &d->layer[l];
            if (df_isq4(&Ly->k) && df_isq4(&Ly->v)) { // k、v 一次 dispatch
                const qw_qlin *L2[2] = {&Ly->k, &Ly->v};
                const qw_buf *Y2[2] = {&e->dfk, &e->dfv};
                df_q4(e, b, L2, Y2, 2, &e->dfxh, 0, r, false, false);
            } else {
                df_mv(e, b, &Ly->k, &e->dfxh, 0, &e->dfk, 0, r, false);
                df_mv(e, b, &Ly->v, &e->dfxh, 0, &e->dfv, 0, r, false);
            }
            df_nr3k(e, b, Ly, 0, NULL, &e->dfKr, &e->dfVr, (size_t)l * KVs, 1, (uint32_t)(pos0 + c), r, NULL);
        }
    }
    if (pos0 + n > e->df_done) e->df_done = pos0 + n;
}

// 猜一塊：anchor 在位置 s（ids 已寫入 dfids），結果在 dfpath / dfprob（7 個）。
// prm 非 NULL：s 與 anchor 由 GPU 決定（df_accept 寫進 dfids[0] 與 prm），參數改從 prm 讀，s 不用
// 一起推測解碼：各對話的 anchor / 結果放在自己的列（dfx_ids 第 dfx_irow 塊、dfx_out 第 dfx_orow 列）；NULL = 用 dfids / dfpath / dfprob
static const qw_buf *dfx_ids, *dfx_path, *dfx_prob;
static int dfx_irow, dfx_orow;
static void df_draft(qw_engine *e, qw_batch *b, int s, const qw_buf *prm) {
    const qw_dflash *d = &e->df;
    const int R = d->block, H = d->H;
    const size_t KVs = (size_t)DF_RING * d->NKV * d->HD * 2;
    const qw_buf *ids = dfx_ids ? dfx_ids : &e->dfids, *opath = dfx_ids ? dfx_path : &e->dfpath, *oprob = dfx_ids ? dfx_prob : &e->dfprob;
    const uint32_t i0 = dfx_ids ? (uint32_t)(dfx_irow * R) : 0;
    const size_t oo = dfx_ids ? (size_t)dfx_orow * DF_ROWS * 4 : 0;
    if (e->m->embed_gq) gg_embed_rows(e, b, ids, i0, (uint32_t)R, 1, (uint32_t)H, &e->dfx);
    else {
    qw_batch_kernel(b, e->df_k[DK_EMBED]);
    bind_t(e, b, 0, e->m->embed_w);
    bind_t(e, b, 1, e->m->embed_s);
    bind_t(e, b, 2, e->m->embed_b);
    qw_batch_buf(b, 3, ids, (size_t)i0 * 4);
    qw_batch_buf(b, 4, &e->dfx, 0);
    qw_batch_dispatch(b, (unsigned)(H + 255) / 256, (unsigned)R, 1, 256, 1, 1);
    }
    const int lo0 = s - d->win + 1 > e->df_lo ? s - d->win + 1 : e->df_lo;
    const uint32_t attn_a[4] = {(uint32_t)s, (uint32_t)(lo0 > 0 ? lo0 : 0), (uint32_t)d->win, (uint32_t)R};
    for (int l = 0; l < d->L; l++) {
        const qw_df_layer *Ly = &d->layer[l];
        if (l == 0) df_rmsk(e, b, &e->dfx, Ly->in_norm, &e->dfxn, &e->dfxh, R); // 之後各層的 input norm 併在上一層最後的卷積裡
        df_mv(e, b, &Ly->akp, &e->dfxh, 0, &e->dfdyn, 0, R, true);
        df_convk(e, b, &e->dfxn, Ly->abase, 0, NULL, &e->dfc, &e->dfxh, R);
        if (df_isq4(&Ly->q) && df_isq4(&Ly->k) && df_isq4(&Ly->v)) { // q、k、v 一次 dispatch
            const qw_qlin *L3[3] = {&Ly->q, &Ly->k, &Ly->v};
            const qw_buf *Y3[3] = {&e->dfq, &e->dfk, &e->dfv};
            df_q4(e, b, L3, Y3, 3, &e->dfxh, 0, R, false, false);
        } else {
            df_mv(e, b, &Ly->q, &e->dfxh, 0, &e->dfq, 0, R, false);
            df_mv(e, b, &Ly->k, &e->dfxh, 0, &e->dfk, 0, R, false);
            df_mv(e, b, &Ly->v, &e->dfxh, 0, &e->dfv, 0, R, false);
        }
        df_nr3k(e, b, Ly, (uint32_t)d->NH, &e->dfqr, &e->dfkb, &e->dfvb, 0, 0, (uint32_t)s, R, prm);
        // 上下文長時用分段版（df_attn_split + df_attn_merge）；短時（< 384）舊版較快。QW_DF_ATTN1=1：一律舊版
        const bool split = R == 8 && d->NH == 4 * d->NKV && d->HD == 128 && (d->win + R + 31) / 32 <= d->HD && !getenv("QW_DF_ATTN1") &&
                           (prm ? e->pos : s) >= 384;
        qw_batch_kernel(b, e->df_k[split ? DK_ATSPLIT : DK_ATTN]);
        qw_batch_buf(b, 0, &e->dfqr, 0);
        qw_batch_buf(b, 1, &e->dfKr, (size_t)l * KVs);
        qw_batch_buf(b, 2, &e->dfVr, (size_t)l * KVs);
        qw_batch_buf(b, 3, &e->dfkb, 0);
        qw_batch_buf(b, 4, &e->dfvb, 0);
        qw_batch_buf(b, 5, split ? &e->dfatp : &e->dfo, 0);
        if (prm) qw_batch_buf(b, 6, prm, 0);
        else qw_batch_bytes(b, 6, attn_a, sizeof attn_a);
        const uint32_t nch = (uint32_t)(d->win + R + 31) / 32; // 段數上限（key 最多 win − 1 個上下文 + R 個 block）
        if (split) {
            qw_batch_dispatch(b, nch, (unsigned)d->NKV, 1, 256, 1, 1);
            qw_batch_kernel(b, e->df_k[DK_ATMERGE]);
            qw_batch_buf(b, 0, &e->dfatp, 0);
            qw_batch_buf(b, 1, &e->dfo, 0);
            if (prm) qw_batch_buf(b, 2, prm, 0);
            else qw_batch_bytes(b, 2, attn_a, sizeof attn_a);
            qw_batch_bytes(b, 3, &nch, 4);
            qw_batch_dispatch(b, (unsigned)R, (unsigned)d->NH, 1, (unsigned)d->HD, 1, 1);
        } else qw_batch_dispatch(b, (unsigned)R, (unsigned)d->NH, 1, 256, 1, 1);
        df_mv(e, b, &Ly->o, &e->dfo, 0, &e->dfof, 0, R, true);
        df_convrms(e, b, &e->dfof, Ly->abase, Ly->post_norm, R);
        df_mv(e, b, &Ly->mkp, &e->dfxh, 0, &e->dfdyn, 0, R, true);
        df_convk(e, b, &e->dfxn, Ly->mbase, 0, NULL, &e->dfc, &e->dfxh, R);
        const qw_qlin *GU[2] = {&Ly->gate, &Ly->up};
        if (df_isq4(&Ly->gate) && df_isq4(&Ly->up) && df_q4_tc_ok(GU, 2, &e->dfxh, 0)) { // gate、up、silu(g)·u 一次 dispatch
            const qw_buf *Y2[2] = {&e->dfgu, &e->dfgu};
            df_q4(e, b, GU, Y2, 2, &e->dfxh, 0, R, false, true);
        } else {
            df_mv(e, b, &Ly->gate, &e->dfxh, 0, &e->dfg, 0, R, true);
            df_mv(e, b, &Ly->up, &e->dfxh, 0, &e->dfu, 0, R, true);
            uint32_t n = (uint32_t)(R * d->inter);
            qw_batch_kernel(b, e->df_k[DK_SWIGLU]);
            qw_batch_buf(b, 0, &e->dfg, 0);
            qw_batch_buf(b, 1, &e->dfu, 0);
            qw_batch_buf(b, 2, &e->dfgu, 0);
            qw_batch_bytes(b, 3, &n, 4);
            d1(b, n, 256);
        }
        df_mv(e, b, &Ly->down, &e->dfgu, 0, &e->dfm, 0, R, true);
        df_convrms(e, b, &e->dfm, Ly->mbase, l + 1 < d->L ? d->layer[l + 1].in_norm : d->norm, R); // 下一層的 input norm，或最後的 norm
    }
    const int rows = R - 1;
    df_head(e, b, &d->head, &e->dfxh, (size_t)H * 2, &e->dflog, rows);
    df_mv(e, b, &d->hp, &e->dfxh, (size_t)H * 2, &e->dfhp, 0, rows, true);
    uint32_t nv = (uint32_t)d->nv, K = DF_TOPK;
    qw_batch_kernel(b, e->df_k[DK_TOPK1]);
    qw_batch_buf(b, 0, &e->dflog, 0);
    qw_batch_buf(b, 1, &e->dfcv, 0);
    qw_batch_buf(b, 2, &e->dfci, 0);
    qw_batch_bytes(b, 3, &nv, 4);
    qw_batch_bytes(b, 4, &K, 4);
    const uint32_t tkc = DF_TKC;
    qw_batch_bytes(b, 5, &tkc, 4);
    qw_batch_dispatch(b, (unsigned)rows, DF_TKC / 8, 1, 256, 1, 1);
    uint32_t nc = DF_TKC * K;
    qw_batch_kernel(b, e->df_k[DK_TOPK2]);
    qw_batch_buf(b, 0, &e->dfcv, 0);
    qw_batch_buf(b, 1, &e->dfci, 0);
    qw_batch_buf(b, 2, &e->dftv, 0);
    qw_batch_buf(b, 3, &e->dfti, 0);
    qw_batch_bytes(b, 4, &nc, 4);
    qw_batch_bytes(b, 5, &K, 4);
    qw_batch_dispatch(b, (unsigned)rows, 1, 1, 1024, 1, 1);
    const uint32_t sa[4] = {(uint32_t)((int32_t *)ids->ptr)[i0], (uint32_t)rows, K, (uint32_t)d->rank};
    qw_batch_kernel(b, e->df_k[DK_SEL]);
    qw_batch_buf(b, 0, &e->dftv, 0);
    qw_batch_buf(b, 1, &e->dfti, 0);
    bind_t(e, b, 2, d->vocab);
    qw_batch_buf(b, 3, &e->dfhp, 0);
    bind_t(e, b, 4, d->pred_cb);
    bind_t(e, b, 5, d->succ_cb);
    qw_batch_buf(b, 6, opath, oo);
    qw_batch_buf(b, 7, oprob, oo);
    if (prm) qw_batch_buf(b, 8, prm, 32);
    else qw_batch_bytes(b, 8, sa, sizeof sa);
    qw_batch_dispatch(b, 1, 1, 1, 32 * K, 1, 1); // 每個候選一個 SIMD group
}

// 接受判定（greedy）編進驗證的 command buffer：df_accept 讀驗證的 argmax（bres）與猜測（tok[p0 + 3 + i]），
// 寫 dfids[0] 與 dfprm，之後的 df_ctx / df_draft(…, &dfprm) 就不必等 CPU
static void df_accept_enc(qw_engine *e, qw_batch *b, int p0, int nd) {
    const qw_dflash *d = &e->df;
    const uint32_t a[8] = {(uint32_t)nd, (uint32_t)p0, (uint32_t)e->df_lo, (uint32_t)d->win, (uint32_t)d->block, (uint32_t)(d->block - 1),
                           DF_TOPK, (uint32_t)d->rank};
    qw_batch_kernel(b, e->df_k[DK_ACCEPT]);
    qw_batch_buf(b, 0, &e->bres, 0);
    qw_batch_buf(b, 1, &e->tok, 0);
    qw_batch_buf(b, 2, &e->dfids, 0);
    qw_batch_buf(b, 3, &e->dfprm, 0);
    qw_batch_bytes(b, 4, a, sizeof a);
    qw_batch_dispatch(b, 1, 1, 1, 32, 1, 1);
}

static void df_set_ids(qw_engine *e, int32_t anchor) {
    int32_t *ids = e->dfids.ptr;
    ids[0] = anchor;
    for (int i = 1; i < e->df.block; i++) ids[i] = e->df.mask;
}

// 測試：feats [n][5·H]（float）、toks [n]，從 npr 起每個位置猜一塊，印出平均猜中長度
int qw_engine_df_test(qw_engine *e, const float *feats, const int32_t *toks, int n, int npr) {
    char err[1024];
    if (!e->df_on) { fprintf(stderr, "沒有載入 DFlash2\n"); return 1; }
    if (!df_init(e, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    const size_t tapw = (size_t)e->df.ntaps * e->df.H;
    e->df_lo = 0;
    e->df_done = 0;
    long acc = 0, cnt = 0;
    double gms = 0;
    for (int s = npr; s + e->df.block < n; s++) {
        while (e->df_done < s) { // 上下文補到 s − 1
            const int r = s - e->df_done < DF_ROWS ? s - e->df_done : DF_ROWS;
            _Float16 *tp = e->dftap.ptr;
            for (int i = 0; i < r; i++)
                for (size_t j = 0; j < tapw; j++) tp[i * tapw + j] = (_Float16)feats[(size_t)(e->df_done + i) * tapw + j];
            qw_batch *b = qw_batch_begin(e->g, false);
            df_ctx(e, b, 0, r, e->df_done);
            qw_batch_end(b);
        }
        df_set_ids(e, toks[s]);
        // QW_DF_PROF=N：第 npr + 5 塊起 N 塊各量一次（每個 dispatch 前後取時間戳，會比不量時慢），印每塊的平均
        const int pn = getenv("QW_DF_PROF") ? atoi(getenv("QW_DF_PROF")) : 0;
        const bool prof = pn > 0 && s >= npr + 5 && s < npr + 5 + pn;
        static const char *pnm[64];
        static double pus[64];
        static int pcnt[64], pna = 0;
        if (pn > 0 && s == npr) qw_profile_enable(e->g, true); // 開一次就好（中途開關會讓之後的 batch 結果不對）
        qw_batch *b = qw_batch_begin(e->g, false);
        df_draft(e, b, s, NULL);
        gms += qw_batch_end(b) * 1e3;
        if (prof) {
            for (int i = 0; i < qw_profile_count(e->g); i++) {
                double u;
                const char *nm = qw_profile_get(e->g, i, &u);
                int j = 0;
                while (j < pna && strcmp(pnm[j], nm)) j++;
                if (j == pna) { if (pna == 64) continue; pnm[pna++] = nm; }
                pus[j] += u / pn;
                pcnt[j]++;
            }
            if (s == npr + 4 + pn) {
                qw_profile_enable(e->g, false);
                double tot = 0;
                for (int i = 0; i < pna; i++) tot += pus[i];
                fprintf(stderr, "DFlash2 一塊（%d 塊平均，逐 dispatch 計時）：合計 %.3f ms\n", pn, tot / 1e3);
                for (int i = 0; i < pna; i++)
                    for (int j = i + 1; j < pna; j++)
                        if (pus[j] > pus[i]) {
                            const char *tn = pnm[i]; pnm[i] = pnm[j]; pnm[j] = tn;
                            double tu = pus[i]; pus[i] = pus[j]; pus[j] = tu;
                            int tc = pcnt[i]; pcnt[i] = pcnt[j]; pcnt[j] = tc;
                        }
                for (int i = 0; i < pna; i++) fprintf(stderr, "  %-34s %4d  %8.3f ms\n", pnm[i], pcnt[i] / pn, pus[i] / 1e3);
            }
        }
        if (getenv("QW_DF_N") && cnt + 1 >= atoi(getenv("QW_DF_N"))) { const int32_t *pp = e->dfpath.ptr; int kk = 0; while (kk < e->df.block - 1 && pp[kk] == toks[s + 1 + kk]) kk++; acc += kk; cnt++; break; }
        const int32_t *path = e->dfpath.ptr;
        static FILE *dumpf = NULL; // QW_DF_DUMP=檔案：每塊寫 path、prob、前 16 名的值與 id（逐位元比對用）
        if (getenv("QW_DF_DUMP")) {
            if (!dumpf) dumpf = fopen(getenv("QW_DF_DUMP"), "wb");
            const size_t r = (size_t)e->df.block - 1;
            if (dumpf) {
                fwrite(path, 4, r, dumpf);
                fwrite(e->dfprob.ptr, 4, r, dumpf);
                fwrite(e->dftv.ptr, 4, r * DF_TOPK, dumpf);
                fwrite(e->dfti.ptr, 4, r * DF_TOPK, dumpf);
                fflush(dumpf);
            }
        }
        int k = 0;
        while (k < e->df.block - 1 && path[k] == toks[s + 1 + k]) k++;
        acc += k;
        cnt++;
    }
    printf("DFlash2（GPU）：%ld 塊，平均猜中 %.2f 個（最多 %d），每塊 GPU %.2f ms\n", cnt, (double)acc / cnt, e->df.block - 1, gms / cnt);
    return 0;
}

// 第 l 層的 hyper-connection 套用完殘差之後，h 就是第 l − 1 層的完整輸出：是 drafter 要的層時寫進 dftap 的對應份
static void df_tap_after(qw_engine *e, qw_batch *b, int l, const qw_buf *h, int rows) {
    if (!e->df_on || !e->df_tapping || !e->df_k[0] || getenv("QW_DF_NOTAP")) return;
    for (int i = 0; i < e->df.ntaps; i++)
        if (e->df.taps[i] == l - 1) {
            const uint32_t os = (uint32_t)(e->df.ntaps * e->df.H), slot = (uint32_t)i;
            qw_batch_kernel(b, e->df_k[DK_TAP]);
            qw_batch_buf(b, 0, h, 0);
            qw_batch_buf(b, 1, &e->dftap, 0);
            qw_batch_bytes(b, 2, &os, 4);
            qw_batch_bytes(b, 3, &slot, 4);
            qw_batch_dispatch(b, (unsigned)(e->df.H + 255) / 256, (unsigned)rows, 1, 256, 1, 1);
        }
}

// ---------------------------------------------------------------- hyper-connection 的 kernel 測試（QW_HCTEST=目錄）
// 用固定的亂數輸入（h、out、舊 inject 部分和，各 8 個 token）跑 decode（NB = 1）、驗證（NB = 2、3、8）與 prefill（T = 8）
// 三條路，把輸入與輸出寫到目錄裡，給 tools/hc_ref.py 用 numpy 對照。層號：QW_HCTEST_L（預設 5），-1 = mixer
static void hct_write(const char *dir, const char *name, const void *p, size_t n) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(p, 1, n, f); fclose(f); }
}

static float hct_rnd(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((*s >> 40) & 0xffffff) / 16777216.0f - 0.5f;
}

void qw_engine_hctest(qw_engine *e, const char *dir) {
    const int T = getenv("QW_HCTEST_T") ? atoi(getenv("QW_HCTEST_T")) : 8; // prefill 的 token 數（decode / 驗證用前 8 個）
    if (T < 8 || T > PF) { fprintf(stderr, "hctest：QW_HCTEST_T 要在 8…%d\n", PF); return; }
    const int l = getenv("QW_HCTEST_L") ? atoi(getenv("QW_HCTEST_L")) : 5;
    const qw_hyper *H = l < 0 ? &e->m->mixer : &e->m->layers[l].attn_hc;
    float *h = malloc(sizeof(float) * T * HCD), *inj = malloc(sizeof(float) * T * 16);
    _Float16 *out = malloc(2 * T * D);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < T * HCD; i++) h[i] = hct_rnd(&s) * (1.0f + 20.0f * (float)((i / D) % QW_HC)); // 各條流的大小不同
    for (int i = 0; i < T * D; i++) out[i] = (_Float16)(hct_rnd(&s) * 4.0f);
    for (int i = 0; i < T * 16; i++) inj[i] = hct_rnd(&s) * 2.0f;
    hct_write(dir, "in_h", h, sizeof(float) * T * HCD);
    hct_write(dir, "in_out", out, 2 * T * D);
    hct_write(dir, "in_inj", inj, sizeof(float) * T * 16);
    char nm[64], err[256];
    // decode
    memcpy(e->h.ptr, h, sizeof(float) * HCD);
    memcpy(e->out.ptr, out, 2 * D);
    memcpy(e->inj.ptr, inj, 64);
    e->pending = true;
    qw_batch *b = qw_batch_begin(e->g, false);
    hyper_in(e, b, H);
    qw_batch_end(b);
    hct_write(dir, "n1_h", e->h.ptr, sizeof(float) * HCD);
    hct_write(dir, "n1_x", e->x.ptr, 2 * D);
    hct_write(dir, "n1_inj", e->inj.ptr, 64);
    { // 8 個 token 各自跑 decode（d1_*：驗證的第 t 個 token 應與這裡的第 t 個逐位元相同）
        float *dh = malloc(sizeof(float) * 8 * HCD), *dj = malloc(64 * 8);
        _Float16 *dx = malloc(2 * 8 * D);
        for (int t = 0; t < 8; t++) {
            memcpy(e->h.ptr, h + (size_t)t * HCD, sizeof(float) * HCD);
            memcpy(e->out.ptr, out + (size_t)t * D, 2 * D);
            memcpy(e->inj.ptr, inj + t * 16, 64);
            e->pending = true;
            b = qw_batch_begin(e->g, false);
            hyper_in(e, b, H);
            qw_batch_end(b);
            memcpy(dh + (size_t)t * HCD, e->h.ptr, sizeof(float) * HCD);
            memcpy(dx + (size_t)t * D, e->x.ptr, 2 * D);
            memcpy(dj + t * 16, e->inj.ptr, 64);
        }
        hct_write(dir, "d1_h", dh, sizeof(float) * 8 * HCD);
        hct_write(dir, "d1_x", dx, 2 * 8 * D);
        hct_write(dir, "d1_inj", dj, 64 * 8);
        free(dh); free(dj); free(dx);
    }
    // 驗證
    if (vb_init(e, err, sizeof err)) {
        const int nbs[] = {2, 3, 8, 4, 5, 6, 7};
        for (int k = 0; k < 7; k++) {
            const int nb = nbs[k];
            e->vbn = nb;
            memcpy(e->bh.ptr, h, sizeof(float) * HCD * nb);
            memcpy(e->bout.ptr, out, 2 * D * nb);
            memcpy(e->binj.ptr, inj, 64 * nb);
            e->bpending = true;
            b = qw_batch_begin(e->g, false);
            vb_hyper(e, b, H);
            qw_batch_end(b);
            snprintf(nm, sizeof nm, "n%d_h", nb); hct_write(dir, nm, e->bh.ptr, sizeof(float) * HCD * nb);
            snprintf(nm, sizeof nm, "n%d_x", nb); hct_write(dir, nm, e->bx.ptr, 2 * D * nb);
            snprintf(nm, sizeof nm, "n%d_inj", nb); hct_write(dir, nm, e->binj.ptr, 64 * nb);
        }
        e->vbn = 2;
    } else fprintf(stderr, "hctest：vb_init 失敗：%s\n", err);
    // prefill
    if (pf_init(e, err, sizeof err)) {
        memcpy(e->ph.ptr, h, sizeof(float) * HCD * T);
        memcpy(e->pout.ptr, out, 2 * D * T);
        memcpy(e->pinj.ptr, inj, 64 * T);
        e->ppending = true;
        b = qw_batch_begin(e->g, false);
        pf_hyper(e, b, H, T);
        qw_batch_end(b);
        hct_write(dir, "pf_h", e->ph.ptr, sizeof(float) * HCD * T);
        hct_write(dir, "pf_x", e->px.ptr, 2 * D * T);
        hct_write(dir, "pf_inj", e->pinj.ptr, 64 * T);
    } else fprintf(stderr, "hctest：pf_init 失敗：%s\n", err);
    free(h); free(inj); free(out);
    fprintf(stderr, "hctest：輸出寫到 %s（層 %d）\n", dir, l);
}

// ---------------------------------------------------------------- MoE 單層測試（QW_MOE_TEST=層[,dir]）
// x 用固定亂數（NB = QW_KBENCH_NB 個 token，token 0 也跑單 token decode），把 x、logits、decode / 驗證 /（若已 prefill）prefill 的輸出
// 寫到 dir（預設 build/moe_test），由 tools/moe_ref.py 依 GGUF 區塊算 FP64 參考比對
static void moe_test(qw_engine *e) {
    char err[1024], dir[512] = "build/moe_test", p[600];
    const char *a = getenv("QW_MOE_TEST");
    const int l = atoi(a);
    if (strchr(a, ',')) snprintf(dir, sizeof dir, "%s", strchr(a, ',') + 1);
    if (!vb_init(e, err, sizeof err)) { fprintf(stderr, "%s\n", err); return; }
    const int nb = getenv("QW_KBENCH_NB") ? atoi(getenv("QW_KBENCH_NB")) : 3;
    e->vbn = nb;
    const qw_moe *M = &e->m->layers[l].moe;
    uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t)l;
    _Float16 *bx = e->bx.ptr;
    for (int i = 0; i < nb * D; i++) { // 近似常態（4 個均勻和）
        float v = 0;
        for (int k = 0; k < 4; k++) { s = s * 6364136223846793005ull + 1442695040888963407ull; v += (float)(s >> 40) / (float)(1 << 24) - 0.5f; }
        bx[i] = (_Float16)(v * 1.7f);
    }
    if (getenv("QW_MOE_TEST_DUP")) { // token t 複製 token t % k（測去重：部分 expert 重複）
        const int k = atoi(getenv("QW_MOE_TEST_DUP"));
        for (int t = k; t < nb; t++) memcpy(bx + (size_t)t * D, bx + (size_t)(t % k) * D, D * 2);
    }
    memcpy(e->x.ptr, bx, D * 2);
    qw_batch *b = qw_batch_begin(e->g, false);
    moe(e, b, M);
    vb_moe(e, b, M);
    qw_batch_end(b);
    mkdir(dir, 0755);
    struct { const char *n; const void *ptr; size_t sz; } outs[] = {
        {"x", e->bx.ptr, (size_t)nb * D * 2}, {"dec", e->out.ptr, D * 2}, {"vb", e->bout.ptr, (size_t)nb * D * 2},
        {"vblog", e->blog.ptr, (size_t)nb * (QW_EXPERTS + 1) * 4}, {"declog", e->rlog.ptr, (QW_EXPERTS + 1) * 4}};
    for (size_t i = 0; i < sizeof outs / sizeof *outs; i++) {
        snprintf(p, sizeof p, "%s/%s.bin", dir, outs[i].n);
        FILE *f = fopen(p, "wb");
        fwrite(outs[i].ptr, 1, outs[i].sz, f);
        fclose(f);
    }
    { // 驗證的每個 token 都要和單獨 decode 逐位元相同（推測解碼 = 一般解碼）
        int bad = 0;
        for (int t = 0; t < nb; t++) {
            memcpy(e->x.ptr, bx + (size_t)t * D, D * 2);
            b = qw_batch_begin(e->g, false);
            moe(e, b, M);
            qw_batch_end(b);
            bad += memcmp(e->out.ptr, (const _Float16 *)e->bout.ptr + (size_t)t * D, D * 2) != 0;
        }
        printf("驗證 vs decode：%d / %d 個 token 不同\n", bad, nb);
    }
    if (e->pf_ready) { // prefill 的 MoE（T = nb）
        memcpy(e->px.ptr, bx, (size_t)nb * D * 2);
        b = qw_batch_begin(e->g, false);
        pf_moe(e, b, M, (uint32_t)nb);
        qw_batch_end(b);
        snprintf(p, sizeof p, "%s/pf.bin", dir);
        FILE *f = fopen(p, "wb");
        fwrite(e->pout.ptr, 2, (size_t)nb * D, f);
        fclose(f);
    }
    printf("MoE 測試：第 %d 層，%d 個 token → %s（prefill %s）\n", l, nb, dir, e->pf_ready ? "有" : "無");
}

// ---------------------------------------------------------------- MoE 速度（QW_MOE_BENCH=1；QW_MOE_BENCH_NB=2,3,8）
// 每層用自己的固定亂數 x（各 token 不同，最壞情況：幾乎沒有重複的 expert），48 層輪流、同一個 command buffer 跑 reps 次取最小
static void moe_bench(qw_engine *e, int reps) {
    char err[1024];
    if (!vb_init(e, err, sizeof err)) { fprintf(stderr, "%s\n", err); return; }
    uint64_t s = 12345;
    _Float16 *bx = e->bx.ptr;
    for (int i = 0; i < VBMAX * D; i++) {
        float v = 0;
        for (int k = 0; k < 4; k++) { s = s * 6364136223846793005ull + 1442695040888963407ull; v += (float)(s >> 40) / (float)(1 << 24) - 0.5f; }
        bx[i] = (_Float16)(v * 1.7f);
    }
    if (getenv("QW_MOE_BENCH_SAME")) for (int t = 1; t < VBMAX; t++) memcpy(bx + (size_t)t * D, bx, D * 2); // 各 token 相同（最好情況）
    if (getenv("QW_MOE_BENCH_CORR")) { // 各 token 與 token 0 相關（c·x0 + √(1−c²)·xt）：0.6 時 8 個 token 約 53 個不重複 expert，接近真實驗證（54 / 80）
        const float c = (float)atof(getenv("QW_MOE_BENCH_CORR")), r = sqrtf(1 - c * c);
        for (int t = 1; t < VBMAX; t++)
            for (int i = 0; i < D; i++) bx[(size_t)t * D + i] = (_Float16)(c * (float)bx[i] + r * (float)bx[(size_t)t * D + i]);
    }
    memcpy(e->x.ptr, bx, D * 2);
    double eb = 0; // routed expert 每層每個 token 的位元組（10 個 expert，48 層平均）＋ shared + router
    for (int l = 0; l < QW_LAYERS; l++) {
        const qw_moe *M = &e->m->layers[l].moe;
        if (M->giq) eb += 10.0 / QW_EXPERTS * ((double)M->giq->nbytes + M->uiq->nbytes + M->diq->nbytes);
        else eb += 10.0 / QW_EXPERTS * (bytes_q(&M->gate) + bytes_q(&M->up) + bytes_q(&M->down));
        eb += bytes_q(&M->sh_gate) + bytes_q(&M->sh_up) + bytes_q(&M->sh_down) + bytes_q(&M->router);
    }
    eb /= QW_LAYERS;
    char nbl[64] = "1,2,3,8";
    if (getenv("QW_MOE_BENCH_NB")) snprintf(nbl, sizeof nbl, "1,%s", getenv("QW_MOE_BENCH_NB"));
    { // 每個 routed expert 的 gate+up、down 位元組（48 層平均；乘上 U 就是驗證時 routed 部分的讀取量）
        double gu = 0, dn = 0;
        for (int l = 0; l < QW_LAYERS; l++) {
            const qw_moe *M = &e->m->layers[l].moe;
            if (!M->giq) continue;
            gu += (double)(M->giq->nbytes + M->uiq->nbytes) / QW_EXPERTS;
            dn += (double)M->diq->nbytes / QW_EXPERTS;
        }
        if (gu > 0) printf("每個 expert（48 層平均）：gate+up %.3f MB、down %.3f MB\n", gu / QW_LAYERS / 1e6, dn / QW_LAYERS / 1e6);
    }
    printf("MoE 每層平均讀取（1 個 token）%.1f MB\n| token 數 | 每層 µs | 有效頻寬（1 token 的讀取量） |\n|---|---|---|\n", eb / 1e6);
    const int bl = getenv("QW_MOE_BENCH_LAYER") ? atoi(getenv("QW_MOE_BENCH_LAYER")) : -1; // 只跑這一層
    const char *only = getenv("QW_MOE_BENCH_ONLY"); // 例如 "rsd"：先完整跑一次（選取結果留著），再只量沒列出的 kernel
    for (char *q = nbl; *q;) {
        const int nb = atoi(q);
        double best = 1e30;
        if (only) { // 沒指定層時用第 0 層選出的 expert（各層 expert 數相同），48 層輪流仍不會重讀同一份權重
            gg_skipv = NULL;
            qw_batch *b = qw_batch_begin(e->g, false);
            if (nb == 1) moe(e, b, &e->m->layers[bl >= 0 ? bl : 0].moe);
            else { e->vbn = nb; vb_moe(e, b, &e->m->layers[bl >= 0 ? bl : 0].moe); }
            qw_batch_end(b);
            gg_skipv = only;
        }
        for (int t = 0; t < 3; t++) {
            qw_batch *b = qw_batch_begin(e->g, false);
            for (int r = 0; r < reps; r++) {
                const qw_moe *M = &e->m->layers[bl >= 0 ? bl : r % QW_LAYERS].moe;
                if (nb == 1) moe(e, b, M);
                else { e->vbn = nb; vb_moe(e, b, M); }
            }
            const double tt = qw_batch_end(b);
            if (tt > 0 && tt < best) best = tt;
        }
        if (getenv("QW_MOE_BENCH_PROF")) { // 每個 dispatch 各量一次（48 層各一次；小 kernel 會被高估，只看相對大小）
            qw_profile_enable(e->g, true);
            qw_batch *b = qw_batch_begin(e->g, false);
            for (int r = 0; r < QW_LAYERS; r++) {
                const qw_moe *M = &e->m->layers[bl >= 0 ? bl : r].moe;
                if (nb == 1) moe(e, b, M);
                else { e->vbn = nb; vb_moe(e, b, M); }
            }
            qw_batch_end(b);
            prof_dump(e, "MoE 48 層");
            qw_profile_enable(e->g, false);
        }
        const double us = best / reps * 1e6;
        if (nb == 1) printf("| %d | %.1f | %.0f GB/s |\n", nb, us, eb / us / 1e3);
        else printf("| %d | %.1f | %.0f GB/s | U = %u |\n", nb, us, eb / us / 1e3, *(const uint32_t *)e->bmsel.ptr); // 最後一層的不重複 expert 數
        while (*q && *q != ',') q++;
        if (*q) q++;
    }
}

// ================================================================ 並發（--parallel N，GGUF）
// 每個對話用一個 slot；slot 另外保存自己的 DeltaNet / conv / PLE 狀態與位置（S、conv、ple、pos）。
// qw_engine_par_step 一次 decode B 個對話各一個 token：與權重有關的部分（embedding、hyper、投影、MoE、lm_head）B 欄一起算
// （驗證用的多 token kernel，GGUF 與 decode 逐位元相同），讀寫各自狀態的部分（PLE、conv / 遞迴、attention 核心）逐個對話。
// DeltaNet 走未合併版（dn_in → k_gdn_in_t → gdn_step_t → k_normgate → dn_out，與合併版逐位元相同），狀態每步寫回。

// 目前的欄位 ↔ slots[i]（含即時狀態）
// GGUF + DFlash2：並發時每個對話有自己的環狀 KV（一起推測解碼要各自猜）。單獨推測解碼照舊以 df_slot 判斷環狀 KV 能不能接著用
static bool par_dfslot(const qw_engine *e) { return e->df_on && e->m->lin_tiled && e->m->embed_gq && !getenv("QW_PAR_NODF"); }
static void par_save(qw_engine *e, int i) {
    struct qw_slot *s = &e->slots[i];
    slot_save(e, i);
    s->S = e->S; s->conv = e->conv_st; s->ple = e->ple_st; s->pos = e->pos; s->exact = e->st_exact;
    if (par_dfslot(e) && e->dfKr.ptr) { s->dfK = e->dfKr; s->dfV = e->dfVr; } // 環狀 KV 跟著對話走（GGUF）
}
static bool par_load(qw_engine *e, int i) {
    struct qw_slot *s = &e->slots[i];
    if (!s->init && !slot_alloc(e, i)) return false;
    if (!s->S.ptr) { // 第一次用：配置並清空自己的狀態
        if (!alloc(e, &s->S, e->S.size) || !alloc(e, &s->conv, e->conv_st.size) || !alloc(e, &s->ple, e->ple_st.size)) return false;
        memset(s->S.ptr, 0, s->S.size); memset(s->conv.ptr, 0, s->conv.size); memset(s->ple.ptr, 0, s->ple.size);
        s->pos = 0;
        s->exact = 0;
    }
    slot_load(e, i);
    e->S = s->S; e->conv_st = s->conv; e->ple_st = s->ple; e->pos = s->pos; e->st_exact = s->exact;
    e->pending = false;
    if (par_dfslot(e) && e->dfKr.ptr) { // 沒有自己的環狀 KV 時配一份（內容無效：dfdone = 0）
        if (!s->dfK.ptr && (!alloc(e, &s->dfK, e->dfKr.size) || !alloc(e, &s->dfV, e->dfVr.size))) return false;
        e->dfKr = s->dfK; e->dfVr = s->dfV;
    }
    return true;
}
static int snap_cap(qw_engine *e) { // 並發模式每組 slot 的快照數：總共 24 份（= 原本 3 組 × 8）平分給用到的 slot，每組 3…8；QW_PAR_SNAPS 固定
    static int v = -1;
    if (v < 0) { v = getenv("QW_PAR_SNAPS") ? atoi(getenv("QW_PAR_SNAPS")) : 0; if (v > CKMAX) v = CKMAX; }
    if (par_cur < 0) return CKMAX;
    if (v > 0) return v;
    int used = 0;
    for (int i = 0; i < QW_PAR_MAX; i++) used += e->slots[i].init;
    int c = used > 0 ? 3 * CKMAX / used : CKMAX;
    return c < 3 ? 3 : c > CKMAX ? CKMAX : c;
}
static bool par_use(qw_engine *e, int i) {
    if (par_cur == i) return true;
    if (par_cur < 0) { // 第一次：引擎原本的欄位（KV、狀態）歸 slot e->cur
        if (!e->kvtok) { e->kvtok = malloc(sizeof(int32_t) * (size_t)e->max_ctx); e->kvlen = 0; }
        par_save(e, e->cur);
        par_cur = e->cur;
        if (par_cur == i) return true;
    }
    dn_flush(e); // 延後寫回的 DeltaNet 狀態先補上（之後的並發 decode 每步寫回）
    par_save(e, par_cur);
    if (!par_load(e, i)) return false;
    par_cur = i;
    return true;
}

static const qw_buf *par_conv(const qw_engine *e, int i) { return i == par_cur ? &e->conv_st : &e->slots[i].conv; }
static const qw_buf *par_S(const qw_engine *e, int i) { return i == par_cur ? &e->S : &e->slots[i].S; }

int qw_engine_par_slots(qw_engine *e) {
    if (!e->nslots) {
        const char *v = getenv("QW_SLOTS");
        e->nslots = v ? atoi(v) : 1;
        if (e->nslots < 1) e->nslots = 1;
        if (e->nslots > QW_PAR_MAX) e->nslots = QW_PAR_MAX;
    }
    return e->nslots;
}

// ---- 記憶體：每組 slot 佔的位元組（KV、token 歷史、DeltaNet / conv / PLE 狀態、快照）；目前載入的 slot 看 engine 自己的欄位
static size_t kv_total(int cap) { size_t t = 0; for (int w = 0; w < 4; w++) t += kv_bytes(cap, w); return t; }
static size_t snaps_bytes(const struct qw_snap *sn) {
    size_t t = 0;
    for (int k = 0; k < CKMAX; k++) t += sn[k].S.size + sn[k].conv.size + sn[k].ple.size;
    return t;
}
static size_t slot_mem(const qw_engine *e, int i) {
    const struct qw_slot *s = &e->slots[i];
    if (!s->init) return 0;
    if (i == par_cur) return kv_total(e->kvcap) + e->tok.size + e->S.size + e->conv_st.size + e->ple_st.size + snaps_bytes(e->snap) + s->dfK.size + s->dfV.size;
    return kv_total(s->kvcap) + s->tok.size + s->S.size + s->conv.size + s->ple.size + snaps_bytes(s->snap) + s->dfK.size + s->dfV.size;
}
size_t qw_engine_par_mem(qw_engine *e) {
    size_t t = 0;
    if (par_cur >= 0) for (int i = 0; i < QW_PAR_MAX; i++) t += slot_mem(e, i);
    return t;
}
// 釋放 slot i 的全部記憶體（之後再用到時重新配置，等於空的 slot）
static void slot_evict(qw_engine *e, int i) {
    struct qw_slot *s = &e->slots[i];
    qw_buf *bs[] = {&s->Kc, &s->Vc, &s->rawk, &s->kblk, &s->tok, &s->S, &s->conv, &s->ple, &s->dfK, &s->dfV};
    for (size_t k = 0; k < sizeof bs / sizeof *bs; k++) qw_buf_free(bs[k]);
    for (int k = 0; k < CKMAX; k++) { qw_buf_free(&s->snap[k].S); qw_buf_free(&s->snap[k].conv); qw_buf_free(&s->snap[k].ple); }
    free(s->hist);
    free(s->kvtok);
    memset(s, 0, sizeof *s);
    if (e->df_slot == i) e->df_slot = -1;
    if (vr_on()) vr_note(i, "slot evicted; ");
}
// ---- 選 slot / 釋放 slot 的成本（只決定用哪一組 slot，不影響數值：接續點一律是逐 token 比對過的快照）
// keep(s)：這組 slot 能直接接續的 token 數（最長的有效快照；沒有快照的 KV 接不上 DeltaNet 狀態，不算）
static int slot_keep(const struct qw_slot *s) {
    int k = 0;
    for (int i = 0; i < CKMAX; i++)
        if (s->snap[i].len > k && s->snap[i].len <= s->kvlen && snap_cross(&s->snap[i])) k = s->snap[i].len; // 別的請求能接的
    return k;
}
// 對話還會回來的權重：剛用過 = 1，閒置 H 秒 = 1/2，2H = 1/3…（H = QW_SLOT_AGE，預設 900 秒；
// Claude Code 主對話等 subagent 時常閒置 10～30 分鐘，用 1/(1+t/H) 而不是指數衰減，長對話不會太快變得不值錢）
static double slot_recency(const struct qw_slot *s, double now) {
    static double H = -1;
    if (H < 0) { H = getenv("QW_SLOT_AGE") ? atof(getenv("QW_SLOT_AGE")) : 900; if (H < 1) H = 1; }
    const double age = s->tuse > 0 && now > s->tuse ? now - s->tuse : 0;
    return 1.0 / (1.0 + age / H);
}
static void slot_stamp(qw_engine *e, unsigned long long busy, double now) { // 正在用的 slot：記成剛用過
    for (int i = 0; i < QW_PAR_MAX; i++) if (busy >> i & 1) e->slots[i].tuse = now;
}
// ---- 縮小閒置的 slot（不影響數值：只動容量與用不到的快照，接續點仍是逐 token 比對過的快照）
// 閒置的 slot 保留幾份快照（QW_IDLE_SNAPS，預設 2，最少 1）：最長的（多半是生成結束 par_finish / 取消時 par_snap 存的）與次長的
// （多半是上一個 prompt 的結尾：下一輪的 prompt 把上一輪生成的部分重新排版、思考內容拿掉時，共同前綴停在這裡）。
// 更短的（prompt 結尾前 2048 / 512 / 128 的檢查點、持久化點）丟掉：持久化點在算的時候已寫到 SSD（par_pf_persist），
// 別的 slot 要共用前綴時由 dc_load 讀回
static int idle_snaps(void) {
    static int v = -1;
    if (v < 0) { v = getenv("QW_IDLE_SNAPS") ? atoi(getenv("QW_IDLE_SNAPS")) : 2; if (v < 1) v = 1; if (v > CKMAX) v = CKMAX; }
    return v;
}
static size_t snap_prune(struct qw_snap *sn, int kvlen, int keep) {
    size_t fr = 0;
    for (int k = 0; k < CKMAX; k++) { // 長度第 1…keep 名（有效的：0 < len ≤ kvlen）留下
        if (!sn[k].S.ptr) continue;
        int rank = 0;
        const bool ok = sn[k].len > 0 && sn[k].len <= kvlen && snap_cross(&sn[k]); // 閒置的 slot：之後只有別的請求會來接（snap_ok）
        if (ok)
            for (int q = 0; q < CKMAX; q++)
                if (q != k && sn[q].len > 0 && sn[q].len <= kvlen && snap_cross(&sn[q]) && (sn[q].len > sn[k].len || (sn[q].len == sn[k].len && q < k))) rank++;
        if (ok && rank < keep) continue;
        fr += sn[k].S.size + sn[k].conv.size + sn[k].ple.size;
        qw_buf_free(&sn[k].S); qw_buf_free(&sn[k].conv); qw_buf_free(&sn[k].ple);
        sn[k].len = 0;
        sn[k].age = 0;
    }
    return fr;
}
size_t qw_engine_par_shrink(qw_engine *e, unsigned long long busy) {
    if (par_cur < 0) return 0;
    const size_t m0 = qw_engine_par_mem(e);
    const int ks = idle_snaps();
    for (int i = 0; i < QW_PAR_MAX; i++) {
        struct qw_slot *s = &e->slots[i];
        if (!s->init || busy >> i & 1) continue;
        const bool cur = i == par_cur; // 目前載入的 slot：資料在 engine 自己的欄位
        qw_buf *K = cur ? &e->Kc : &s->Kc, *V = cur ? &e->Vc : &s->Vc, *R = cur ? &e->rawk : &s->rawk, *B = cur ? &e->kblk : &s->kblk;
        int *cap = cur ? &e->kvcap : &s->kvcap;
        const int kvlen = cur ? e->kvlen : s->kvlen, pos = cur ? e->pos : s->pos;
        const int used = kvlen > pos ? kvlen : pos;
        int want = (used + 2 * VBMAX + 8 + KV_STEP - 1) / KV_STEP * KV_STEP; // 同 kv_reserve 的餘量（驗證會寫到 pos + VBMAX 之後）
        if (want > e->max_ctx) want = e->max_ctx;
        const int old = *cap;
        const bool rs = want < *cap && kv_resize(e, K, V, R, B, cap, want);
        if (rs && getenv("QW_CACHE_DBG")) fprintf(stderr, "[mem] slot %d 的 KV 容量 %d → %d（用到 %d）\n", i, old, want, used);
        const size_t pr = snap_prune(cur ? e->snap : s->snap, kvlen, ks);
        if (vr_on() && (rs || pr)) vr_note(i, "shrink kv %d→%d%s (kvlen %d, pos %d); ", old, *cap, pr ? " + prune snaps" : "", kvlen, pos);
        if (cur) par_save(e, i); // slots[par_cur] 的副本跟著更新（buffer 換過）
    }
    const size_t m1 = qw_engine_par_mem(e), fr = m0 > m1 ? m0 - m1 : 0;
    if (fr >= (size_t)64 << 20) fprintf(stderr, "[mem] 縮小閒置的 slot：%.2f GB → %.2f GB\n", m0 / 1e9, m1 / 1e9);
    return fr;
}
// 分段 prefill 中途停下（例如客戶端取消）：slot sid 的即時狀態停在一段的段尾（pos == kvlen，KV / DeltaNet 狀態都只含
// kvtok[0..kvlen)），把它記成快照，之後同一個 prompt（或它的延伸）再來時 par_pick 選回這組、par_prefill_to 從這裡接續。
// 段尾本來就是 prefill 的分段點（chunk 是 GDN 分塊的倍數），接續與一次算完逐位元相同（同 pf_yield 讓出時記的快照）。
// 沒載入的 slot 也可以（先換進來）；回傳 true = 這組現在有長度 kvlen 的快照（剛存的或原本就有），false = 狀態不在段界或沒東西可存
bool qw_engine_par_snap(qw_engine *e, int sid) {
    if (sid < 0 || sid >= QW_PAR_MAX || par_cur < 0 || !e->slots[sid].init) return false;
    const struct qw_slot *s = &e->slots[sid];
    const int kvlen = sid == par_cur ? e->kvlen : s->kvlen, pos = sid == par_cur ? e->pos : s->pos;
    if (kvlen <= 0 || pos != kvlen || kvlen > e->max_ctx) return false; // 生成中（pos > kvlen）或還沒算：不是一致的段界
    const struct qw_snap *sn = sid == par_cur ? e->snap : s->snap;
    for (int k = 0; k < CKMAX; k++)
        if (sn[k].len == kvlen && sn[k].S.ptr) return true;
    if (!par_use(e, sid)) return false;
    snap_why = SW_PARSNAP, snap_save(e, kvlen); // 內含 dn_flush（延後寫回的 DeltaNet 狀態先補上）
    for (int k = 0; k < CKMAX; k++)
        if (e->snap[k].len == kvlen && e->snap[k].S.ptr) {
            if (getenv("QW_CACHE_DBG")) fprintf(stderr, "[cache] slot %d 停在 %d：記快照\n", sid, kvlen);
            if (vr_on()) vr_note(sid, "par_snap at %d; ", kvlen);
            return true;
        }
    return false;
}
// 超過 budget 位元組時釋放閒置的 slot（busy 位元 = 正在用；目前載入的 slot 不動）。回傳釋放了幾組
// 先放「重算成本 × 還會回來的權重」最小的：value = keep / (1 + 閒置秒數 / H)，同分時最久沒用的先放。
// 短的（權限判斷、標題）與早就結束的先走，等 subagent 回來的長主對話留到最後；真的要放長對話時先寫到 SSD（有 cache 目錄、
// keep ≥ QW_EVICT_PERSIST_MIN（預設 4096：寫一份約 0.4 s，重算 subagent 的幾千～幾萬 token 要數十秒）、檔案不超過 QW_CACHE_GB 的一半；
// 系統記憶體吃緊的 one 模式不寫，免得再加壓）
// 釋放之前先縮（qw_engine_par_shrink）：超過 budget 或 one 模式時縮全部閒置的 slot；另外閒置超過 QW_SHRINK_IDLE 秒（預設 60，
// 0 = 不主動縮）的 slot 平常也縮。one 模式縮出 ≥ 256 MB 時這次就不釋放 slot（回傳 0，下次再看）
int qw_engine_par_trim(qw_engine *e, unsigned long long busy, size_t budget, bool one) {
    int n = 0;
    const double now = qw_now();
    slot_stamp(e, busy, now);
    static int pmin = -1;
    if (pmin < 0) pmin = getenv("QW_EVICT_PERSIST_MIN") ? atoi(getenv("QW_EVICT_PERSIST_MIN")) : 4096;
    static double sidle = -1;
    if (sidle < 0) { sidle = getenv("QW_SHRINK_IDLE") ? atof(getenv("QW_SHRINK_IDLE")) : 60; if (sidle < 0) sidle = 0; }
    if (one || qw_engine_par_mem(e) > budget) {
        const size_t fr = qw_engine_par_shrink(e, busy);
        if (one && fr >= (size_t)256 << 20) return 0;
    } else if (sidle > 0) {
        unsigned long long keep = busy; // 最近用過的不縮（下一輪多半馬上回來）
        for (int i = 0; i < QW_PAR_MAX; i++)
            if (e->slots[i].init && !(e->slots[i].tuse > 0 && now - e->slots[i].tuse >= sidle)) keep |= 1ull << i;
        qw_engine_par_shrink(e, keep);
    }
    for (;;) {
        if (!one && qw_engine_par_mem(e) <= budget) return n;
        int pick = -1;
        double pv = 0;
        for (int i = 0; i < QW_PAR_MAX; i++) {
            if (!e->slots[i].init || i == par_cur || busy >> i & 1) continue;
            const double v = slot_keep(&e->slots[i]) * slot_recency(&e->slots[i], now);
            if (pick < 0 || v < pv || (v == pv && e->slots[i].used < e->slots[pick].used)) { pick = i; pv = v; }
        }
        if (pick < 0) return n;
        struct qw_slot *s = &e->slots[pick];
        const int keep = slot_keep(s);
        if (getenv("QW_CACHE_DBG"))
            fprintf(stderr, "[mem] 釋放 slot %d（%.1f GB，可接續 %d 個 token，閒置 %.0f 秒）\n", pick, slot_mem(e, pick) / 1e9, keep,
                    s->tuse > 0 ? now - s->tuse : 0);
        if (!one && e->cache_dir[0] && !getenv("QW_NO_CACHE") && !getenv("QW_NO_EVICT_PERSIST") && keep >= pmin && keep >= 256) {
            const double fb = (double)keep * (double)(kv_row(0) * 2 + kv_row(2) + kv_row(3) / 4) * (NATT + 1);
            for (int k = 0; k < CKMAX; k++)
                if (s->snap[k].len == keep && snap_cross(&s->snap[k])) {
                    if (fb <= dc_cap() / 2) dc_write_slot(e, s, &s->snap[k], keep);
                    break;
                }
        }
        slot_evict(e, pick);
        n++;
        if (one) return n;
    }
}
// 選一個沒在用的 slot（busy 的位元 = 正在用），回傳編號，-1 = 沒有。對每組閒置的 slot i：
//   c_i = 與 prompt 的共同前綴（逐 token 比對），keep_i = slot_keep，g_i = 不超過 min(c_i, np − 1) 的最長快照（拿它接續能省的）
//   loss_i = keep_i > c_i ? keep_i − g_i : 0（蓋掉它會丟掉、之後那段對話回來要重算的 token）
// 1. 可以接著用（c_i ≥ 1024 或 ≥ np/2，且 loss_i ≤ max(16384, c_i)）的裡面，c_i 最長的（同長時 loss 小的）：
//    同一段對話的下一輪（prompt 延伸 slot 的內容，loss ≈ 0）照舊回到自己的 slot
// 2. 否則用空的 slot：只共用 system + 工具前綴（例如 subagent 對主對話：c ≈ 15K、loss ≈ 200K）時不蓋掉長對話，
//    共同前綴由 prefill 的 xs_copy 從那組 slot 的快照複製過來（或 SSD），能接續的長度與直接蓋掉那組相同
// 3. 沒有空的：蓋掉成本 loss_i / (1 + 閒置秒數 / H) 最小的（同分時最久沒用的）；都一樣時就是原本的做法
int qw_engine_par_pick(qw_engine *e, const int32_t *prompt, int np, unsigned long long busy) {
    const int ns = qw_engine_par_slots(e);
    if (par_cur < 0) par_use(e, e->cur);
    par_save(e, par_cur); // slots[par_cur] 的 kvtok / kvlen 是最新的
    const double now = qw_now();
    slot_stamp(e, busy, now);
    int cs[QW_PAR_MAX], ls[QW_PAR_MAX];
    int best = -1;
    for (int i = 0; i < ns; i++) {
        cs[i] = ls[i] = 0;
        if (busy >> i & 1 || !e->slots[i].init) continue;
        const struct qw_slot *s = &e->slots[i];
        int c = 0;
        while (c < s->kvlen && c < np && s->kvtok[c] == prompt[c]) c++;
        int g = 0;
        for (int k = 0; k < CKMAX; k++)
            if (s->snap[k].len > g && s->snap[k].len <= c && s->snap[k].len <= np - 1 && snap_cross(&s->snap[k])) g = s->snap[k].len;
        const int keep = slot_keep(s);
        cs[i] = c;
        ls[i] = keep > c ? keep - g : 0;
        if ((c >= 1024 || c >= np / 2) && ls[i] <= (c > 16384 ? c : 16384) && (best < 0 || c > cs[best] || (c == cs[best] && ls[i] < ls[best])))
            best = i;
    }
    const bool dbg = getenv("QW_CACHE_DBG") != NULL;
    if (best >= 0) {
        if (dbg) fprintf(stderr, "[slot] 接著用 slot %d（共同前綴 %d，會丟掉 %d）\n", best, cs[best], ls[best]);
        return best;
    }
    for (int i = 0; i < ns; i++)
        if (!(busy >> i & 1) && !e->slots[i].init) {
            if (dbg) fprintf(stderr, "[slot] 用空的 slot %d\n", i);
            return i;
        }
    int pick = -1;
    double pc = 0;
    for (int i = 0; i < ns; i++) {
        if (busy >> i & 1) continue;
        const double c = ls[i] * slot_recency(&e->slots[i], now);
        if (pick < 0 || c < pc || (c == pc && e->slots[i].used < e->slots[pick].used)) { pick = i; pc = c; }
    }
    if (dbg && pick >= 0) fprintf(stderr, "[slot] 蓋掉 slot %d（共同前綴 %d，會丟掉 %d）\n", pick, cs[pick], ls[pick]);
    return pick;
}

// 在 slot sid 處理 prompt：從該 slot 最長的可用快照接續，只算新增的部分；回傳最後一個位置的 logits（失敗 NULL）
// 檢查點（同 qw_engine_generate_ex）：prompt 結尾前 2048、512、128 個 token，對齊到 GDN 分塊邊界，prefill 時順便存；
// 分段時照整段（第一段的起點 pf0）算，只留這一段 [start, end) 裡的
// 跨 slot 共用前綴（例如多個 subagent 的 system prompt 與工具定義）：別的 slot（可能正在用）有更長的可用快照時，
// 把前 len 個位置的 KV、token 歷史與快照的狀態複製到目前的 slot（KV 前 len 列只由這些 token 決定，與在那邊接續逐位元相同）
static int xs_src, xs_why; // 上一次 xs_copy 的來源 slot 與那份快照是誰存的（QW_VERIFY_REUSE 的紀錄）
static int xs_copy(qw_engine *e, int sid, const int32_t *prompt, int np, int have) {
    int bs = -1, bk = -1, bl = have;
    for (int i = 0; i < QW_PAR_MAX; i++) {
        if (i == sid || i == par_cur || !e->slots[i].init) continue;
        const struct qw_slot *o = &e->slots[i];
        int c = 0;
        while (c < o->kvlen && c < np && o->kvtok[c] == prompt[c]) c++;
        for (int k = 0; k < CKMAX; k++) {
            const int L = o->snap[k].len;
            if (L > bl && L <= c && L <= np - 1 && snap_cross(&o->snap[k])) { bl = L; bs = i; bk = k; }
        }
    }
    if (bs < 0 || bl - have < 256) return 0;
    xs_src = bs;
    xs_why = e->slots[bs].snap[bk].why;
    const double t0 = qw_now();
    if (!kv_reserve(e, np + 2 * VBMAX + 8 < e->max_ctx ? np + 2 * VBMAX + 8 : e->max_ctx)) return 0;
    struct qw_slot *o = &e->slots[bs];
    qw_buf *dst[4] = {&e->Kc, &e->Vc, &e->rawk, &e->kblk};
    const qw_buf *src[4] = {&o->Kc, &o->Vc, &o->rawk, &o->kblk};
    for (int ai = 0; ai <= NATT; ai++)
        for (int w = 0; w < 4; w++) {
            const size_t row = kv_row(w), rows = w == 3 ? (size_t)bl / 4 : (size_t)bl;
            memcpy((char *)dst[w]->ptr + (size_t)ai * kv_rows(e->kvcap, w) * row, (const char *)src[w]->ptr + (size_t)ai * kv_rows(o->kvcap, w) * row, rows * row);
        }
    memcpy(e->hist, o->hist, sizeof(int32_t) * (size_t)(bl + 2));
    memcpy(e->tok.ptr, o->tok.ptr, sizeof(int32_t) * (size_t)(bl + 2));
    e->last_eos = 1;
    for (int p = 0; p < bl; p++) if (e->hist[p + 2] == EOS_ID) e->last_eos = p + 1;
    snap_copy(e, &o->snap[bk], false);
    e->pos = bl;
    e->st_exact = bl; // 來源快照是 exact（snap_cross）
    e->pending = false;
    for (int i = 0; i < CKMAX; i++) e->snap[i].len = 0; // KV 被覆寫，這個 slot 原本的檢查點都失效
    memcpy(e->kvtok, prompt, sizeof(int32_t) * (size_t)bl);
    e->kvlen = bl;
    snap_why = SW_XS, snap_save(e, bl);
    if (e->df_slot == sid) e->df_slot = -1;
    if (getenv("QW_CACHE_DBG")) fprintf(stderr, "[cache] 從 slot %d 複製前綴：%d 個 token，%.1f ms\n", bs, bl, (qw_now() - t0) * 1e3);
    return bl;
}
static int pf0[QW_PAR_MAX]; // 分段 prefill：這個 slot 第一段從哪裡開始（檢查點位置照整段算）
static int par_pp[8], par_npp; // 持久化點（qw_engine_par_persist 設定，給下一個 prompt 用）
static int par_sp[QW_PAR_SNAPAT_MAX], par_nsp; // 共用前綴的交接點（qw_engine_par_snapat；只存記憶體裡的快照，不寫 SSD）
static int par_xs_n; // 上一次 par_prefill_to 從別的 slot 複製了幾個 token（qw_engine_par_copied）
void qw_engine_par_persist(qw_engine *e, const int *pos, int n) {
    (void)e;
    par_npp = 0;
    par_nsp = 0; // 交接點每次另外設（qw_engine_par_snapat 要在這之後呼叫）
    for (int i = 0; i < n && par_npp < 8; i++) par_pp[par_npp++] = pos[i];
}
void qw_engine_par_snapat(qw_engine *e, const int *pos, int n) {
    (void)e;
    par_nsp = 0;
    for (int i = 0; i < n && par_nsp < QW_PAR_SNAPAT_MAX; i++) par_sp[par_nsp++] = pos[i];
}
int qw_engine_par_copied(qw_engine *e) { (void)e; return par_xs_n; }
// 段尾停在第一個交接點：pts 裡的位置照 s0 起算對齊到 cc（GDN 分塊）的倍數 a，start < a < end 的最小的 a 當新的段尾
// （a 是合法的分段點：與一次算完逐位元相同）。a − start < cc 不會發生（start 也是從 s0 起算的分段點）
int qw_engine_par_clip(int s0, int start, int end, const int *pts, int n, int cc) {
    int out = end;
    for (int k = 0; k < n; k++) {
        if (pts[k] <= s0) continue;
        const int a = s0 + (pts[k] - s0) / cc * cc;
        if (a > start && a < out) out = a;
    }
    return out;
}
static bool par_pf_plan(qw_engine *e, int sid, int np, int start, int end) { // 回傳 true：end 本身是檢查點（段尾另外存）
    const int s0 = pf0[sid];
    int cks[3 + 8 + QW_PAR_SNAPAT_MAX], nck = 0;
    const int back[3] = {2048, 512, 128}, minnew[3] = {4096, 4096, 512};
    for (int k = 0; k < 3; k++)
        if (np - s0 > minnew[k] && np - back[k] > s0 + 64 && (nck == 0 || np - back[k] > cks[nck - 1] + 32)) cks[nck++] = np - back[k];
    { // prompt 結尾往下對齊到 GDN 分塊：下一輪（不同請求）能接續的最長位置（結尾本身多半不在分塊倍數上，snap_ok）
        const int CC = gdn_chunk(), a = np / CC * CC;
        if (a > s0 && a < np && (nck == 0 || cks[nck - 1] != a)) cks[nck++] = a;
    }
    for (int k = 0; k < par_npp + par_nsp; k++) { // 持久化點（例如 system 訊息結尾）與交接點：也存快照（持久化點算完寫到 SSD）
        const int pp = k < par_npp ? par_pp[k] : par_sp[k - par_npp];
        bool dup = pp <= s0 || pp >= np;
        for (int q = 0; q < nck; q++) dup |= cks[q] == pp;
        if (!dup) cks[nck++] = pp;
    }
    for (int a = 0; a < nck; a++)
        for (int b2 = a + 1; b2 < nck; b2++)
            if (cks[b2] < cks[a]) { const int t = cks[a]; cks[a] = cks[b2]; cks[b2] = t; }
    const int CC = gdn_chunk();
    bool at_end = false;
    e->npf_snap = 0;
    for (int k = 0; k < nck; k++) {
        const int a = s0 + (cks[k] - s0) / CC * CC;
        if (a > start && a < end && e->npf_snap < (int)(sizeof e->pf_snap / sizeof *e->pf_snap) && (e->npf_snap == 0 || e->pf_snap[e->npf_snap - 1] != a))
            e->pf_snap[e->npf_snap++] = a;
        if (a == end && end < np) at_end = true;
    }
    return at_end;
}
// 分段：*done > 0 時從上一段停下的位置（slot 的即時狀態）接著算；chunk > 0 時這次最多算 chunk 個 token（之後至少還剩 64 個才切），
// *done 回傳算到哪。分段與一次算完逐位元相同（prefill 本來就逐段算，與段長無關：build/run QW_PF_CHUNK 檢查）
static void par_pf_persist(qw_engine *e, int sid, const int32_t *prompt, int start, int end) { // 這一段 (start, end] 裡的持久化點寫到 SSD
    const int CC = gdn_chunk(), s0 = pf0[sid];
    for (int q = 0; q < par_npp; q++) {
        const int pp = par_pp[q], a = s0 + (pp - s0) / CC * CC;
        if (pp <= s0 || a <= start || a > end) continue;
        for (int i = 0; i < CKMAX; i++)
            if (e->snap[i].len == a && a > 0) { dc_write_from(e, prompt, a, &e->snap[i]); break; }
    }
}
const float *qw_engine_par_prefill_to(qw_engine *e, int sid, const int32_t *prompt, int np, int chunk, int *done, int *cached) {
    if (np <= 0 || np + 2 >= e->max_ctx || !par_use(e, sid)) return NULL;
    e->slots[sid].used = ++e->clock;
    e->slots[sid].tuse = qw_now();
    struct qw_slot *ss = &e->slots[sid];
    char derr[256];
    const bool tapdf = par_dfslot(e) && df_init(e, derr, sizeof derr) && e->dfKr.ptr; // GGUF：prompt 最後一個視窗的特徵寫進這個對話的環狀 KV
    if (*done > 0 && *done < np && e->pos == *done && e->kvlen == *done && !memcmp(e->kvtok, prompt, sizeof(int32_t) * (size_t)*done)) {
        const int start = *done;
        int end = chunk > 0 && np - start > chunk + 64 ? start + chunk : np;
        end = qw_engine_par_clip(pf0[sid], start, end, par_sp, par_nsp, gdn_chunk()); // 交接點：段尾停在那裡（快照在段尾存）
        par_xs_n = 0;
        memcpy(e->kvtok, prompt, sizeof(int32_t) * (size_t)end);
        e->kvlen = end;
        const bool ck = par_pf_plan(e, sid, np, start, end);
        double pms = 0;
        if (tapdf && ss->dfdone == start) { e->df_tapping = true; e->df_ctx_from = ss->dflo; }
        const float *lg = qw_engine_prefill(e, prompt + start, end - start, NULL, &pms);
        if (e->df_tapping) { ss->dfdone = end; if (e->df_slot == sid) e->df_slot = -1; }
        else ss->dfdone = 0;
        e->df_tapping = false;
        e->npf_snap = 0;
        if (!lg) return NULL;
        if (end == np || ck) { snap_why = end == np ? SW_PROMPT : SW_CKPT; snap_save(e, end); }
        par_pf_persist(e, sid, prompt, start, end);
        dn_flush(e);
        *done = end;
        if (end == np) vr_capture(e, sid, prompt, np, lg);
        return lg;
    }
    const int tag = slot_tag[sid];
    int common = 0;
    while (common < e->kvlen && common < np && e->kvtok[common] == prompt[common]) common++;
    int best = -1;
    for (int i = 0; i < CKMAX; i++) {
        if (e->snap[i].len > common) e->snap[i].len = 0;
        if (snap_ok(&e->snap[i], np, tag) && (best < 0 || e->snap[i].len > e->snap[best].len)) best = i;
    }
    par_xs_n = 0;
    if (!no_reuse() && (par_xs_n = xs_copy(e, sid, prompt, np, best >= 0 ? e->snap[best].len : 0)) > 0) { // 別的 slot 有更長的前綴
        best = -1;
        for (int i = 0; i < CKMAX; i++)
            if (snap_ok(&e->snap[i], np, tag) && (best < 0 || e->snap[i].len > e->snap[best].len)) best = i;
    }
    if (!no_reuse() && e->cache_dir[0]) { // 記憶體裡的不夠長：看 SSD（同 qw_engine_generate_ex）
        const int have = best >= 0 ? e->snap[best].len : 0;
        if (have < np / 2 && kv_reserve(e, np + 2 * VBMAX + 8 < e->max_ctx ? np + 2 * VBMAX + 8 : e->max_ctx) && dc_load(e, prompt, np, have) > 0) {
            if (vr_on()) vr_note(sid, "SSD %d; ", e->kvlen);
            best = -1;
            for (int i = 0; i < CKMAX; i++)
                if (snap_ok(&e->snap[i], np, tag) && (best < 0 || e->snap[i].len > e->snap[best].len)) best = i;
        }
    }
    int start = 0;
    bool cont = false; // 同一個請求接著生成（只差一個還沒餵的 token）：用 decode 餵（與沒中斷時相同）
    if (best >= 0 && !no_reuse()) {
        start = e->snap[best].len;
        cont = snap_cont(&e->snap[best], np, tag);
        snap_copy(e, &e->snap[best], false);
        e->pos = start;
        e->st_exact = e->snap[best].exact ? start : -1;
        e->pending = false;
    } else qw_engine_reset(e);
    if (cached) *cached = start;
    if (vr_on()) {
        if (par_xs_n > 0) vr_note(sid, "xs_copy %d from slot %d (%s); ", par_xs_n, xs_src, SW_NAME[xs_why]);
        vr_note(sid, "pf %d→%d (%s, kvcap %d); ", start, np, start ? SW_NAME[e->snap[best].why] : "from 0", e->kvcap);
        if (start > 0) vr_mark(sid);
    }
    // 分段時 qw_engine_prefill 每次只預留這一段：第一段就照整個 prompt 預留，KV 一次到位（1.25 倍成長下逐段擴大會多搬好幾次整份 KV）
    kv_reserve(e, np + 2 * VBMAX + 8 < e->max_ctx ? np + 2 * VBMAX + 8 : e->max_ctx);
    int end = chunk > 0 && np - start > chunk + 64 ? start + chunk : np;
    if (!cont) end = qw_engine_par_clip(start, start, end, par_sp, par_nsp, gdn_chunk()); // 交接點：段尾停在那裡
    memcpy(e->kvtok, prompt, sizeof(int32_t) * (size_t)end);
    e->kvlen = end;
    e->df_tapping = false;
    e->df_mode = false;
    e->pf_mtp = false;
    pf0[sid] = start;
    const bool ck = par_pf_plan(e, sid, np, start, end);
    e->pf_next = -1;
    double pms = 0;
    const float *lg;
    if (cont) { // 同一個請求接著生成：剩下的 token 用 decode 一步一步算（切換單獨 / 一起時與沒中斷逐位元相同）
        e->npf_snap = 0;
        lg = NULL;
        for (int i = start; i < np; i++) if (!(lg = qw_engine_step(e, prompt[i], NULL))) break;
        ss->dflo = ss->dfdone = 0; // decode 不寫特徵：環狀 KV 沒有這段
        if (tapdf && ss->dfdone == 0) { ss->dflo = np; ss->dfdone = np; } // 之後從 np 起累積（猜測一開始沒有上下文）
    } else {
        if (tapdf) {
            e->df_tapping = true;
            e->df_ctx_from = np - e->df.win > start ? np - e->df.win : start;
            ss->dflo = e->df_ctx_from;
        }
        lg = qw_engine_prefill(e, prompt + start, end - start, NULL, &pms);
        if (e->df_tapping) { ss->dfdone = end; if (e->df_slot == sid) e->df_slot = -1; }
        else ss->dfdone = 0;
        e->df_tapping = false;
    }
    e->npf_snap = 0;
    if (!lg) return NULL;
    if (end == np || ck) { snap_why = end == np ? SW_PROMPT : SW_CKPT; snap_save(e, end); }
    par_pf_persist(e, sid, prompt, start, end);
    dn_flush(e);
    *done = end;
    if (end == np) vr_capture(e, sid, prompt, np, lg);
    return lg;
}
const float *qw_engine_par_prefill(qw_engine *e, int sid, const int32_t *prompt, int np, int *cached) {
    int done = 0;
    return qw_engine_par_prefill_to(e, sid, prompt, np, 0, &done, cached);
}

// ---- 重用狀態的自我檢查（QW_VERIFY_REUSE=N；沒設時這段完全不動）
// 每次 prompt 算完（par_prefill_to 的最後一段、單獨跑 generate_ex 的 prefill）：這個 prompt 有用到別處來的狀態（從快照接續、xs_copy、
// SSD）時，每 N 次記一次「token + 最後位置的 logits + 這組 slot 這段期間發生過的事（vr_note）」。引擎閒置時 serve 呼叫
// qw_engine_par_verify：在沒用到的 slot（編號 ≥ par_slots）從 0 算同一段 token（不接任何快取），比對 logits，算完就釋放那組 slot。
// 從快照接續的 prefill 分段位置不同，與從頭算不一定逐位元相同（浮點加法順序），所以看分佈差多少：KL(ref‖重算) 與 argmax。
// QW_VERIFY_ALL=1：沒有重用的也抽（量正常的雜訊底線）；QW_VERIFY_KL：MISMATCH 的門檻（預設 1.0：沒有重用、只是分段位置不同時實測 KL 就到 0.01–0.4）；QW_VERIFY_MAXTOK：太長的不驗（預設 65536）
enum { VR_QMAX = 32 };
static struct vr_ev { int tag, sid, n, start; int32_t *ids; float *lg; char how[512]; } vr_q[VR_QMAX];
static int vr_nq, vr_cnt, vr_stat[4]; // vr_stat：OK、TIE（argmax 不同但分佈幾乎一樣）、MISMATCH、跳過
static char vr_how[QW_PAR_MAX][512];
static int vr_reused[QW_PAR_MAX];
static bool vr_running;
static void vr_mark(int sid) { if (sid >= 0 && sid < QW_PAR_MAX && !vr_running) vr_reused[sid] = 1; }
static int vr_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("QW_VERIFY_REUSE") ? atoi(getenv("QW_VERIFY_REUSE")) : 0;
    return v > 0 ? v : 0;
}
static void vr_note(int sid, const char *fmt, ...) {
    if (sid < 0 || sid >= QW_PAR_MAX || vr_running) return;
    char *h = vr_how[sid];
    const size_t l = strlen(h);
    if (l + 1 >= sizeof vr_how[sid]) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(h + l, sizeof vr_how[sid] - l, fmt, ap);
    va_end(ap);
}
void qw_engine_par_tag(qw_engine *e, int sid, int id) {
    (void)e;
    if (sid < 0 || sid >= QW_PAR_MAX) return;
    slot_tag[sid] = id;
}
static void vr_capture(qw_engine *e, int sid, const int32_t *ids, int n, const float *lg) {
    (void)e;
    if (!vr_on() || vr_running || sid < 0 || sid >= QW_PAR_MAX) return;
    static int vmax = -1, all = -1;
    if (vmax < 0) { vmax = getenv("QW_VERIFY_MAXTOK") ? atoi(getenv("QW_VERIFY_MAXTOK")) : 65536; all = getenv("QW_VERIFY_ALL") != NULL; }
    fprintf(stderr, "[reuse] #%d slot %d prompt %d%s：%s\n", slot_tag[sid], sid, n, vr_reused[sid] ? "" : "（沒有重用）", vr_how[sid][0] ? vr_how[sid] : "-");
    const bool use = (vr_reused[sid] || all) && n <= vmax && vr_cnt++ % vr_on() == 0;
    if (use && vr_nq < VR_QMAX) {
        struct vr_ev *v = &vr_q[vr_nq];
        v->ids = malloc(sizeof(int32_t) * (size_t)n);
        v->lg = malloc(sizeof(float) * QW_VOCAB);
        if (v->ids && v->lg) {
            memcpy(v->ids, ids, sizeof(int32_t) * (size_t)n);
            memcpy(v->lg, lg, sizeof(float) * QW_VOCAB);
            v->tag = slot_tag[sid];
            v->sid = sid;
            v->n = n;
            snprintf(v->how, sizeof v->how, "%s", vr_how[sid][0] ? vr_how[sid] : "(none) ");
            vr_nq++;
        } else { free(v->ids); free(v->lg); v->ids = NULL; v->lg = NULL; }
    } else if (use) vr_stat[3]++;
    vr_how[sid][0] = 0; // 下一個 prompt（例如同一個請求收尾後重新加入）重新記
    vr_reused[sid] = 0;
}
int qw_engine_par_verify_pending(qw_engine *e) { (void)e; return vr_nq; }
void qw_engine_par_verify_stats(qw_engine *e, int out[5]) { (void)e; for (int i = 0; i < 4; i++) out[i] = vr_stat[i]; out[4] = vr_nq; }
// 跑一筆等著的檢查（引擎執行緒、GPU 閒置時）。回傳還剩幾筆；-1 = 沒有可以跑的
int qw_engine_par_verify(qw_engine *e) {
    if (!vr_nq || par_cur < 0) return -1;
    struct vr_ev v = vr_q[0];
    memmove(&vr_q[0], &vr_q[1], sizeof vr_q[0] * (size_t)(vr_nq - 1));
    vr_nq--;
    const int ns = qw_engine_par_slots(e), prev = par_cur;
    int vs = -1;
    for (int i = QW_PAR_MAX - 1; i >= ns && vs < 0; i--) if (!e->slots[i].init) vs = i;
    if (vs < 0) {
        fprintf(stderr, "[verify] 跳過 #%d：沒有多的 slot\n", v.tag);
        vr_stat[3]++;
        free(v.ids); free(v.lg);
        return vr_nq;
    }
    const double t0 = qw_now();
    vr_running = vr_fresh = true;
    qw_engine_par_persist(e, NULL, 0); // 不寫 SSD、沒有交接點（serve 每次 prefill 前都會重設）
    int done = 0, cached = 0;
    const float *lg = qw_engine_par_prefill_to(e, vs, v.ids, v.n, 0, &done, &cached);
    double kl = 0, mx = 0, num = 0, den = 0, pr_ref = 0, pr_new = 0;
    int ar = 0, an = 0;
    if (lg) {
        for (int i = 1; i < QW_VOCAB; i++) { if (v.lg[i] > v.lg[ar]) ar = i; if (lg[i] > lg[an]) an = i; }
        double zr = 0, zn = 0;
        for (int i = 0; i < QW_VOCAB; i++) { zr += exp((double)v.lg[i] - v.lg[ar]); zn += exp((double)lg[i] - lg[an]); }
        for (int i = 0; i < QW_VOCAB; i++) {
            const double d = (double)v.lg[i] - lg[i];
            if (fabs(d) > mx) mx = fabs(d);
            num += d * d;
            den += (double)v.lg[i] * v.lg[i];
            const double lr = (double)v.lg[i] - v.lg[ar] - log(zr), ln = (double)lg[i] - lg[an] - log(zn);
            kl += exp(lr) * (lr - ln);
        }
        pr_ref = exp((double)lg[ar] - lg[an]) / zn; // 原本選的 token 在重算的分佈裡的機率
        pr_new = 1 / zn;                            // 重算的第一名的機率
    }
    vr_running = vr_fresh = false;
    par_use(e, prev); // 換回原本的 slot，再放掉檢查用的
    slot_evict(e, vs);
    vr_how[vs][0] = 0;
    static double tol = -1;
    if (tol < 0) tol = getenv("QW_VERIFY_KL") ? atof(getenv("QW_VERIFY_KL")) : 1.0;
    const int cls = !lg ? 3 : kl > tol ? 2 : ar != an ? 1 : 0;
    vr_stat[cls]++;
    static const char *const nm[] = {"OK", "TIE", "MISMATCH", "FAIL"};
    fprintf(stderr, "[verify] %s #%d slot %d：prompt %d，重算 %.1f s；argmax %d/%d（重算的機率 %.3f/%.3f）、KL %.2e、max|Δ| %.3f、rel %.2e；reuse path = %s\n",
            nm[cls], v.tag, v.sid, v.n, qw_now() - t0, ar, an, pr_ref, pr_new, kl, mx, den > 0 ? sqrt(num / den) : 0.0, v.how);
    free(v.ids);
    free(v.lg);
    return vr_nq;
}
// 並發模式下只有一個請求時：在 slot sid 用一般的生成（含推測解碼）；cb 回傳 false 可中途停下（之後可用 par_prefill 接續）
int qw_engine_par_solo(qw_engine *e, int sid, const int32_t *prompt, int np, const qw_gen *g, qw_tok_cb cb, void *ctx, qw_gen_stats *st) {
    if (!par_use(e, sid)) return QW_FIN_ERROR;
    e->slots[sid].used = ++e->clock;
    e->slots[sid].tuse = qw_now();
    const int fin = qw_engine_generate_ex(e, prompt, np, g, cb, ctx, st);
    struct qw_slot *ss = &e->slots[sid]; // 環狀 KV 的有效範圍（df_slot 是這個 slot 時才是這個對話的）
    if (par_dfslot(e) && e->df_slot == sid && e->df_done > e->df_lo) { ss->dflo = e->df_lo; ss->dfdone = e->df_done; }
    else ss->dfdone = 0;
    return fin;
}

// 生成結束：把目前內容（prompt + 生成的 token，KV 已到 pos）記成快照，下次同一段對話接續
void qw_engine_par_finish(qw_engine *e, int sid, const int32_t *gen, int ngen) {
    if (!par_use(e, sid)) return;
    e->slots[sid].tuse = qw_now();
    const int np = e->kvlen, fed = e->pos - np; // 已經進模型的生成 token 數（停在停止 token 時 = ngen，其他情況通常 = ngen − 1）
    if (fed > 0 && fed <= ngen && e->pos <= e->max_ctx) {
        memcpy(e->kvtok + np, gen, sizeof(int32_t) * (size_t)fed);
        e->kvlen = np + fed;
        snap_why = SW_FINISH, snap_save(e, e->kvlen);
    }
}

static const float *par_step8(qw_engine *e, const int *sids, const int32_t *toks, int B, double *gpu_ms);
// 並發 decode 一步：對話 sids[i]（目前位置 pos_i）餵入 toks[i]；回傳 [B][QW_VOCAB] 的 logits（下一個 token 的分佈），失敗 NULL。
// B > VBMAX 時分成幾組（每組最多 VBMAX 個，各自讀一次權重）依序算
// 各組大小（1…VBMAX）一步的牆鐘時間（ms，EMA；第一次含編譯不算）：決定 B 個對話怎麼分組最快
static double par_cost[VBMAX + 1];
static int par_seen[VBMAX + 1];
double qw_engine_par_cost(qw_engine *e, int B) { // B 個對話一步的預估 ms（最佳分組）；還沒量過的組大小用線性外插
    (void)e;
    if (B < 1) return 0;
    double c[VBMAX + 1], best[QW_PAR_MAX + 1];
    for (int b = 1; b <= VBMAX; b++) c[b] = par_cost[b] > 0 ? par_cost[b] : 0;
    double a = 0, k = 0; // 有量過的拿來外插：cost ≈ a + k·b
    int lo = 0, hi = 0;
    for (int b = 1; b <= VBMAX; b++) if (c[b] > 0) { if (!lo) lo = b; hi = b; }
    if (!lo) return 0;
    if (hi > lo) { k = (c[hi] - c[lo]) / (hi - lo); a = c[lo] - k * lo; } else a = c[lo];
    for (int b = 1; b <= VBMAX; b++) if (c[b] <= 0) c[b] = a + k * b;
    best[0] = 0;
    for (int n = 1; n <= B && n <= QW_PAR_MAX; n++) {
        best[n] = 1e30;
        for (int b = 1; b <= VBMAX && b <= n; b++) if (best[n - b] + c[b] < best[n]) best[n] = best[n - b] + c[b];
    }
    return best[B < QW_PAR_MAX ? B : QW_PAR_MAX];
}
// 分組：動態規劃取「各組時間加總」最小的切法（組大小 ≤ VBMAX；量到的時間會隨降頻等變化）
static int par_split(int B, int *g) {
    double c[VBMAX + 1], best[QW_PAR_MAX + 1];
    int arg[QW_PAR_MAX + 1], ng = 0;
    for (int b = 1; b <= VBMAX; b++) c[b] = par_cost[b] > 0 ? par_cost[b] : 0; // 沒量過：當成最便宜（先試、量到就有數字）
    best[0] = 0;
    for (int n = 1; n <= B; n++) {
        best[n] = 1e30; arg[n] = 1;
        for (int b = 1; b <= VBMAX && b <= n; b++) if (best[n - b] + c[b] <= best[n]) { best[n] = best[n - b] + c[b]; arg[n] = b; } // 一樣快時取大組
    }
    for (int n = B; n > 0; n -= arg[n]) g[ng++] = arg[n];
    return ng;
}
const float *qw_engine_par_step(qw_engine *e, const int *sids, const int32_t *toks, int B, double *gpu_ms) {
    if (B < 1 || B > QW_PAR_MAX) return NULL;
    for (int i = 0; mix_busy && i < B; i++) if (sids[i] == mix_sid) return NULL; // mixed step：prefill 中的 slot 不能 decode
    int grp[QW_PAR_MAX];
    const int ng = par_split(B, grp);
    if (ng == 1) return par_step8(e, sids, toks, B, gpu_ms);
    static qw_buf plg;
    if (!plg.ptr && !alloc(e, &plg, (size_t)QW_PAR_MAX * QW_VOCAB * 4)) return NULL;
    double tot = 0;
    for (int gi = 0, c = 0; gi < ng; c += grp[gi++]) {
        const int n = grp[gi];
        double g = 0;
        const float *lg = par_step8(e, sids + c, toks + c, n, &g);
        if (!lg) return NULL;
        memcpy((float *)plg.ptr + (size_t)c * QW_VOCAB, lg, (size_t)n * QW_VOCAB * 4);
        tot += g;
    }
    if (gpu_ms) *gpu_ms = tot;
    return plg.ptr;
}
// MLX 的 DeltaNet（並發）：投影 B 欄一起（同 vb_deltanet）；conv + 遞迴逐對話（NB = 1 的 kernel、自己的 conv / S 狀態，
// 中間結果用第 0 欄當暫存、輸出寫到第 t 欄）；gated norm 與 out_proj 再 B 欄一起
static void par_dn_mlx(qw_engine *e, qw_batch *b, const qw_deltanet *N, int li, const int *sids, int B) {
    const qw_qlin *Ls[4] = {&N->in_qkv, &N->in_z, &N->in_a, &N->in_b};
    const qw_buf *Ys[4] = {&e->bqkv, &e->bz, &e->ba, &e->bb};
    if (dn_in_ok(N)) dn_in(e, b, N, &e->bx, &e->bqkv, &e->bz, &e->ba, &e->bb, B);
    else vb_qmv_multi(e, b, Ls, Ys, 4, &e->bx);
    const size_t cst = (size_t)li * 3 * 10240 * 4, soff = (size_t)li * 48 * 128 * 128 * 4;
    e->vbn = 1;
    for (int t = 0; t < B; t++) {
        const qw_buf *cv = par_conv(e, sids[t]), *S = par_S(e, sids[t]);
        qw_batch_kernel(b, vk(e, "kb_gdn_in"));
        qw_batch_buf(b, 0, &e->bqkv, (size_t)t * 10240 * 2);
        bind_t(e, b, 1, N->conv1d);
        qw_batch_buf(b, 2, cv, cst);
        qw_batch_buf(b, 3, &e->ba, (size_t)t * 48 * 2);
        qw_batch_buf(b, 4, &e->bb, (size_t)t * 48 * 2);
        bind_t(e, b, 5, N->A_log);
        bind_t(e, b, 6, N->dt_bias);
        qw_batch_buf(b, 7, &e->bgq, 0);
        qw_batch_buf(b, 8, &e->bgk, 0);
        qw_batch_buf(b, 9, &e->bgv, 0);
        qw_batch_buf(b, 10, &e->bla, 0);
        qw_batch_buf(b, 11, &e->bbe, 0);
        for (int q = 1; q < VBMAX; q++) qw_batch_buf(b, 11 + q, cv, cst); // NB = 1：用不到
        qw_batch_dispatch(b, 16 + 48, 1, 1, 128, 1, 1);
        qw_batch_kernel(b, vk(e, "gdn_stepn"));
        qw_batch_buf(b, 0, &e->bgq, 0);
        qw_batch_buf(b, 1, &e->bgk, 0);
        qw_batch_buf(b, 2, &e->bgv, 0);
        qw_batch_buf(b, 3, &e->bla, 0);
        qw_batch_buf(b, 4, &e->bbe, 0);
        qw_batch_buf(b, 5, S, soff);
        qw_batch_buf(b, 6, &e->bgo, (size_t)t * 6144 * 2);
        qw_batch_buf(b, 7, &e->bssp, (size_t)t * 48 * 4 * 4); // 各 token 的平方和 [t][48][128 / 32]（vb_xform 要讀）
        for (int q = 1; q < VBMAX; q++) qw_batch_buf(b, 7 + q, S, soff);
        qw_batch_dispatch(b, 48, 128 / 32, 1, 256, 1, 1);
    }
    e->vbn = B;
    if (vb_xf(e, &N->out)) vb_xform(e, b, 1, &e->bgo, &e->bz, &e->bssp, N->norm, &e->bgy);
    else {
        qw_batch_kernel(b, vk(e, "kb_normgate"));
        qw_batch_buf(b, 0, &e->bgo, 0);
        qw_batch_buf(b, 1, &e->bz, 0);
        bind_t(e, b, 2, N->norm);
        qw_batch_buf(b, 3, &e->bgy, 0);
        qw_batch_dispatch(b, 48, VB, 1, 128, 1, 1);
    }
    if (dn_out_ok(N)) dn_out(e, b, N, &e->bgy, &e->bout, B);
    else vb_qmv(e, b, &N->out, &e->bgy, &e->bout, false);
}
static bool par_keepdf; // 一起 decode 時也把特徵寫進各自的環狀 KV（之後可能換成一起推測解碼）
void qw_engine_par_keepdf(qw_engine *e, bool on) { (void)e; par_keepdf = on; }
static const float *par_step8(qw_engine *e, const int *sids, const int32_t *toks, int B, double *gpu_ms) {
    char err[256];
    if (B < 1 || B > VBMAX || !vb_init(e, err, sizeof err)) return NULL;
    const bool tapdf = par_keepdf && par_dfslot(e) && df_init(e, err, sizeof err) && e->dfKr.ptr;
    const bool gg = e->m->lin_tiled && e->m->embed_gq; // GGUF；否則 MLX（批次 kernel 與 decode 只在 --exact-spec 時逐位元相同）
    if (!gg && e->m->gguf_parts) return NULL; // 只有一部分是 GGUF 的混合：不支援
    const double tst = qw_now();
    static qw_buf ptok;
    if (!ptok.ptr && !alloc(e, &ptok, 64)) return NULL;
    int pos[VBMAX];
    for (int i = 0; i < B; i++) { // 各對話：位置、token 歷史（PLE 的 n-gram 要讀）、KV 容量
        if (!par_use(e, sids[i]) || e->pos + 2 >= e->max_ctx || !kv_reserve(e, e->pos + 2)) return NULL;
        pos[i] = e->pos;
        e->hist[e->pos + 2] = toks[i];
        ((int32_t *)e->tok.ptr)[e->pos + 2] = toks[i];
        ((int32_t *)ptok.ptr)[i] = toks[i];
    }
    e->vbn = B;
    qw_batch *b = qw_batch_begin(e->g, false);
    if (gg) gg_embed_rows(e, b, &ptok, 0, (uint32_t)B, QW_HC, HCD, &e->bh);
    else {
        const uint32_t z = 0;
        qw_batch_kernel(b, vk(e, "kb_embed"));
        bind_t(e, b, 0, e->m->embed_w);
        bind_t(e, b, 1, e->m->embed_s);
        bind_t(e, b, 2, e->m->embed_b);
        qw_batch_buf(b, 3, &ptok, 0);
        qw_batch_bytes(b, 4, &z, 4);
        qw_batch_buf(b, 5, &e->bh, 0);
        qw_batch_dispatch(b, D / 256, VB, 1, 256, 1, 1);
    }
    e->bpending = false;
    const unsigned am = at_mask;
    int li = 0, ai = 0;
    for (int l = 0; l < QW_LAYERS; l++) {
        const qw_layer *L = &e->m->layers[l];
        if (L->has_ple) { // 逐對話（同 vb_ple，但每個 token 是不同對話：歷史、狀態、位置各自的）
            if (e->bpending) {
                uint32_t apply = 1, want = 0;
                qw_batch_kernel(b, vk(e, "kb_hc_in"));
                qw_batch_buf(b, 0, &e->bh, 0);
                qw_batch_buf(b, 1, &e->bout, 0);
                qw_batch_buf(b, 2, &e->binj, 0);
                bind_t(e, b, 3, L->attn_hc.hc_norm);
                qw_batch_buf(b, 4, &e->bnh, 0);
                qw_batch_buf(b, 5, &e->bnf, 0);
                qw_batch_bytes(b, 6, &apply, 4);
                qw_batch_buf(b, 7, &e->bnf, 0);
                qw_batch_buf(b, 8, &e->binj2, 0);
                qw_batch_bytes(b, 9, &want, 4);
                qw_batch_dispatch(b, QW_HC, VB, 1, 1024, 1, 1);
                e->bpending = false;
            }
            for (int t = 0; t < B; t++) {
                par_use(e, sids[t]);
                ng_row0 = t;
                ng_gather(e, &L->ple, pos[t] + 2, 1);
                ng_row0 = 0;
            }
            e->ng_base = 0;
            if (gg) ple_gg_proj(e, b, &L->ple, B); // MLX：投影在 ple() 裡逐對話算（同 vb_ple）
            for (int t = 0; t < B; t++) {
                par_use(e, sids[t]);
                e->ng_base = t * 16;
                vb_copy(e, b, &e->bh, (size_t)t * HCD * 4, &e->h, 0, HCD);
                e->pos = pos[t];
                ple_pre_t = gg ? t : -1;
                ple(e, b, &L->ple);
                ple_pre_t = -1;
                vb_copy(e, b, &e->h, 0, &e->bh, (size_t)t * HCD * 4, HCD);
            }
            e->ng_base = 0;
        }
        vb_hyper(e, b, &L->attn_hc);
        if (tapdf) { e->df_tapping = true; df_tap_after(e, b, l, &e->bh, B); e->df_tapping = false; }
        if (L->full_attn) {
            const qsa_io bio = {&e->bx, &e->bqg, &e->bkk, &e->bvv, &e->biqk, &e->baq, &e->biq, &e->bao, &e->bay, &e->bout};
            const qsa_io dio = {&e->x, &e->qg, &e->kk, &e->vv, &e->iqk, &e->aq, &e->iq, &e->ao, &e->ay, &e->out};
            at_mask = 1; // 投影：B 欄一起
            qsa_layer(e, b, &L->att, ai, 0, B, &bio);
            at_mask = ~0u & ~1u & ~64u & ~128u; // 核心：逐對話（自己的 KV、位置）
            for (int t = 0; t < B; t++) { // 讀寫批次緩衝區的第 t 欄（中間結果 aq / iq 用 decode 的）
                par_use(e, sids[t]);
                qsa_io tio = dio;
                tio.qg = &e->bqg; tio.oqg = (size_t)t * 12288 * 2;
                tio.kk = &e->bkk; tio.okk = (size_t)t * 512 * 2;
                tio.vv = &e->bvv; tio.ovv = (size_t)t * 512 * 2;
                tio.iqk = &e->biqk; tio.oiqk = (size_t)t * 640 * 2;
                tio.ay = &e->bay; tio.oay = (size_t)t * 6144 * 2;
                tio.ao = &e->bao; tio.oao = (size_t)t * 6144 * 2; // MLX 核心的輸出（gate 與 o_proj 之後一起算）
                qsa_layer(e, b, &L->att, ai, (uint32_t)pos[t], 1, &tio);
            }
            at_mask = 64 | 128; // （MLX：output gate）+ o_proj：B 欄一起
            qsa_layer(e, b, &L->att, ai, 0, B, &bio);
            at_mask = am;
            ai++;
        } else {
            const qw_deltanet *N = &L->lin;
            if (!gg) { par_dn_mlx(e, b, N, li, sids, B); li++; goto dn_done; }
            if (!dn_fuse(e, N)) return NULL; // 只做合併版（與預設 decode 逐位元相同）
            { // dn_inf（投影 + conv + 衰減 / β）：B 欄一起，第 t 欄用對話 t 自己的 conv 狀態
                char base[32];
                snprintf(base, sizeof base, "dn_inf_%d_%d", N->in_qkv.gq - 1, N->in_z.gq - 1);
                const dn_in_args a = {2560, {10240, 6144, 64, 64}, 1, 16480};
                const size_t cst = (size_t)li * 3 * 10240 * 4;
                qw_batch_kernel(b, B == 1 && getenv("QW_PAR_NOMS") ? dn_kernel(e, base, 1) : dn_kernel_ms(e, base, B)); // QW_PAR_NOMS：除錯（B = 1 用原本的）
                bind_t(e, b, 0, N->in_qkv.w);
                bind_t(e, b, 1, N->in_z.w);
                bind_t(e, b, 2, N->in_a.w);
                bind_t(e, b, 3, N->in_b.w);
                qw_batch_buf(b, 4, &e->bx, 0);
                qw_batch_buf(b, 5, &e->bqkv, 0);
                qw_batch_buf(b, 6, &e->bz, 0);
                qw_batch_buf(b, 7, &e->bla, 0);
                qw_batch_buf(b, 8, &e->bbe, 0);
                qw_batch_bytes(b, 9, &a, sizeof a);
                qw_batch_buf(b, 10, &e->bqkv, 0);
                bind_t(e, b, 11, N->conv1d);
                qw_batch_buf(b, 12, par_conv(e, sids[0]), cst);
                bind_t(e, b, 13, N->A_neg);
                bind_t(e, b, 14, N->dt_bias);
                for (int t = 1; t < VBMAX; t++) qw_batch_buf(b, 14 + t, par_conv(e, sids[t < B ? t : 0]), cst);
                qw_batch_dispatch(b, 16480 / (DN_NSG * dn_r(B)), 1, 1, 32 * DN_NSG, 1, 1);
            }
            for (int t = 0; t < B; t++) { // 遞迴 + gated RMSNorm：逐對話（自己的狀態，寫回）
                gf_args ga = {0, 0};
                qw_batch_kernel(b, gf_kernel(e, 1));
                qw_batch_buf(b, 0, &e->bqkv, (size_t)t * 10240 * 2);
                qw_batch_buf(b, 1, &e->bla, (size_t)t * 64 * 4);
                qw_batch_buf(b, 2, &e->bbe, (size_t)t * 64 * 4);
                qw_batch_buf(b, 3, par_S(e, sids[t]), (size_t)li * 48 * 128 * 128 * 4);
                qw_batch_buf(b, 4, &e->go, 0);
                qw_batch_buf(b, 5, &e->bz, (size_t)t * 6144 * 2);
                bind_t(e, b, 6, N->norm);
                qw_batch_buf(b, 7, &e->bgy, (size_t)t * 6144 * 2);
                qw_batch_buf(b, 8, &e->dncnt, 0);
                for (int q = 9; q <= 13; q++) qw_batch_buf(b, q, &e->bgy, (size_t)t * 6144 * 2); // 沒有重放、不延後：用不到
                qw_batch_bytes(b, 14, &ga, sizeof ga);
                qw_batch_dispatch(b, 48, 128 / gf_dvb(), 1, 8 * gf_dvb(), 1, 1);
            }
            dn_out(e, b, N, &e->bgy, &e->bout, B);
            li++;
        }
    dn_done:
        if (trace_on()) for (int t = 0; t < B; t++) trace_copy(e, b, &e->bh, (size_t)t * HCD * 4, &e->bout, (size_t)t * D * 2, t, l, 0);
        e->bpending = true;
        vb_hyper(e, b, &L->mlp_hc);
        vb_moe(e, b, &L->moe);
        if (trace_on()) for (int t = 0; t < B; t++) trace_copy(e, b, &e->bh, (size_t)t * HCD * 4, &e->bout, (size_t)t * D * 2, t, l, 1);
        e->bpending = true;
    }
    vb_hyper(e, b, &e->m->mixer);
    e->bpending = false;
    vb_qmv(e, b, &e->m->lm_head, &e->bx, &e->blg, true);
    if (tapdf)
        for (int i = 0; i < B; i++) { // 這一步的特徵寫進各自的環狀 KV（位置 pos[i]）
            par_use(e, sids[i]);
            struct qw_slot *ss = &e->slots[sids[i]];
            if (ss->dfdone != pos[i]) ss->dflo = pos[i];
            df_ctx(e, b, i, 1, pos[i]);
            ss->dfdone = pos[i] + 1;
            if (e->df_slot == sids[i]) e->df_slot = -1;
        }
    const double tenc = qw_now();
    const double t = qw_batch_end(b);
    if (getenv("QW_PAR_PROF")) fprintf(stderr, "[par] B=%d 編碼 %.2f ms、送出到完成 %.2f ms、GPU %.2f ms\n", B, (tenc - tst) * 1e3, (qw_now() - tenc) * 1e3, t * 1e3);
    if (gpu_ms) *gpu_ms = t * 1e3;
    if (t < 0) return NULL;
    const double wall = (qw_now() - tst) * 1e3;
    if (par_seen[B]++) par_cost[B] = par_cost[B] > 0 ? 0.9 * par_cost[B] + 0.1 * wall : wall;
    for (int i = 0; i < B; i++) {
        par_use(e, sids[i]);
        e->pos = pos[i] + 1;
        if (toks[i] == EOS_ID) e->last_eos = pos[i] + 1;
        if (!tapdf) e->slots[sids[i]].dfdone = 0; // 沒寫特徵：環狀 KV 接不上了
    }
    return e->blg.ptr;
}

// ---- mixed step（並發模式；serve 以 QW_PAR_MIX 開啟）：分段 prefill 的一段在 GPU 上跑的同時，其他對話照常一起 decode。
// prefill 算力受限（矩陣單元）、一起 decode 頻寬受限（每步讀一次全部權重），兩者在 GPU 上重疊。
// 不把 decode 的列併進 prefill 的矩陣（prefill 的 qmm 分塊 / 累加順序與 decode 的 qmv 不同 → 結果不再逐位元相同），
// 而是兩邊各用原本的 kernel、原本的算式：prefill 這段照常編碼、送到串流 0，decode 每一步（qw_engine_par_step）送到串流 1
// （另一個 command queue 與 event 鏈，兩條互不等待）。所以 prefill 與 decode 的結果都與分開跑逐位元相同，前提是同時在跑的兩邊
// 不寫同一個 buffer：
//   - 各 slot 的 KV、token 歷史、DeltaNet / conv / PLE 狀態本來就分開（par_use 換的是 slot 自己的 buffer；prefill 已綁定的不受影響）
//   - prefill 的暫存（p*、gW…、分塊 GDN 等）與一起 decode 的暫存（b*、q3*/q4*/q5*、gg*、dncnt…）本來就不重疊
//   - 兩邊共用的只有下面 mix_fields 這些單列暫存（prefill 段尾 hyper_in + lm_head 寫 h / x / logits…；decode 的 PLE 用 h、
//     n-gram 列 ngW / ngS / ngB / ngX 由 CPU 在編碼時寫入；ssp 兩邊的 DeltaNet / attention 都寫）：decode 迴圈期間換成另一份
//   - DFlash2 的暫存（dftap…）：decode 迴圈期間不寫特徵（par_keepdf 暫時關；只影響一起推測解碼的猜測上下文，不影響輸出）
// 換 slot 的 CPU 狀態（e->pos、kvlen、snap…）由 par_use 存回 / 載入；迴圈結束後換回 prefill 的 slot，pf_segment 照常收尾。
static qw_mix_fn mix_fn;
static void *mix_ctx;
void qw_engine_par_mix(qw_engine *e, qw_mix_fn fn, void *ctx) { (void)e; mix_fn = fn; mix_ctx = ctx; }
enum { MIX_NF = 16 };
static void mix_fields(qw_engine *e, qw_buf **f) {
    qw_buf *a[MIX_NF] = {&e->h, &e->nh, &e->nf, &e->d, &e->d2, &e->u, &e->x, &e->inj, &e->inj2, &e->out, &e->logits, &e->ssp,
                         &e->ngW, &e->ngS, &e->ngB, &e->ngX};
    memcpy(f, a, sizeof a);
}
static qw_buf mix_sh[MIX_NF]; // 另一份（與 mix_fields 一一對應）
static void mix_swap(qw_engine *e) {
    qw_buf *f[MIX_NF];
    mix_fields(e, f);
    for (int i = 0; i < MIX_NF; i++)
        if (f[i]->ptr && mix_sh[i].ptr) { const qw_buf t = *f[i]; *f[i] = mix_sh[i]; mix_sh[i] = t; }
}
static bool mix_alloc(qw_engine *e) { // 另一份單列暫存（至少與目前的一樣大；decode 迴圈期間 e 的欄位指向這一份）
    qw_buf *f[MIX_NF];
    mix_fields(e, f);
    for (int i = 0; i < MIX_NF; i++) {
        if (!f[i]->ptr || mix_sh[i].size >= f[i]->size) continue;
        if (mix_sh[i].ptr) qw_buf_free(&mix_sh[i]);
        if (!alloc(e, &mix_sh[i], f[i]->size)) { memset(&mix_sh[i], 0, sizeof mix_sh[i]); return false; }
    }
    return true;
}
static bool mix_on(const qw_engine *e) {
    return mix_fn && !mix_busy && par_cur >= 0 && e->m->lin_tiled && e->m->embed_gq && !e->dn_pmask && !trace_on();
}
static double mix_wait(qw_engine *e, qw_batch *b) {
    const int P = par_cur;
    if (!mix_alloc(e)) return qw_batch_wait(b);
    const double t0 = qw_now();
    const bool kd = par_keepdf, dft = e->df_tapping;
    const int dff = e->df_ctx_from;
    double pc[VBMAX + 1];
    int ps[VBMAX + 1];
    memcpy(pc, par_cost, sizeof pc); // 量到的一步時間含重疊（變慢）：不算進分組用的估計
    memcpy(ps, par_seen, sizeof ps);
    mix_swap(e);
    par_keepdf = false;
    mix_sid = P;
    mix_busy = true;
    qw_gpu_set_stream(e->g, 1);
    int n = 0;
    while (!qw_batch_poll(b) && mix_fn(mix_ctx)) n++;
    qw_gpu_set_stream(e->g, 0);
    mix_busy = false;
    mix_sid = -1;
    const double t1 = qw_now();
    const double t = qw_batch_wait(b);
    mix_swap(e);
    memcpy(par_cost, pc, sizeof pc);
    memcpy(par_seen, ps, sizeof ps);
    par_keepdf = kd;
    if (par_cur != P && !par_use(e, P)) { fprintf(stderr, "mixed step：換回 prefill 的 slot %d 失敗\n", P); return -1; }
    e->df_tapping = dft;
    e->df_ctx_from = dff;
    e->pending = false;
    if (getenv("QW_PAR_PROF"))
        fprintf(stderr, "[mix] prefill 一段 GPU %.1f ms；期間 decode %d 步（%.1f ms），之後再等 %.1f ms\n", t * 1e3, n, (t1 - t0) * 1e3,
                (qw_now() - t1) * 1e3);
    return t;
}

// ---- 並發推測解碼：B 個對話各驗證 n[i] 個 token（第一個是確定的、其餘是猜測），共 R = Σn[i] ≤ VBMAX 列，共用權重的部分一起讀
// 列 r0[i] … r0[i] + n[i] − 1 是對話 i 的 token。attention 核心、DeltaNet 遞迴、PLE 逐對話（各自 n[i] 個 token，同單一對話的驗證）。
// DeltaNet 狀態只讀（k、衰減、δ 寫到延後寫回的第 0 組、各自的列），接受後 qw_engine_par_vaccept 重放寫回。只支援 GGUF
static qw_kernel *dn_kernel_sg(qw_engine *e, const char *base, int nb) { // DN_SEG：列分段（各段一個對話）。名稱 <base>_n{nb}sg
    char name[64], defs[200], err[8192];
    snprintf(name, sizeof name, "%s_n%dsg", base, nb);
    qw_kernel *k = qw_kernel_get(e->g, name);
    if (k) return k;
    snprintf(defs, sizeof defs, "#define NB %d\n#define NBS \"%dsg\"\n#define R %d\n#define DN_SEG 1\n", nb, nb, dn_r(nb));
    if (!compile(e, defs, src_dn_proj, sizeof src_dn_proj, err, sizeof err)) { fprintf(stderr, "編譯 dn_proj（NB=%d，分段）失敗：%s\n", nb, err); exit(1); }
    if (!(k = qw_kernel_get(e->g, name))) { fprintf(stderr, "找不到 %s\n", name); exit(1); }
    return k;
}
static struct { int B, sid[VBMAX], n[VBMAX], r0[VBMAX], pos[VBMAX]; int32_t tok[VBMAX]; } pv; // 上一次 par_vstep（給 vaccept）
const float *qw_engine_par_vstep(qw_engine *e, const int *sids, const int *n, const int32_t *toks, int B, double *gpu_ms) {
    char err[256];
    int R = 0;
    for (int i = 0; i < B; i++) { if (n[i] < 1) return NULL; R += n[i]; }
    if (B < 1 || R > VBMAX || !vb_init(e, err, sizeof err)) return NULL;
    if (!(e->m->lin_tiled && e->m->embed_gq) || !dn_defer(e)) return NULL; // 只有 GGUF
    dn_flush(e); // 目前的 slot 若有待重放的（單獨推測解碼留下的），先寫回：這裡用第 0 組
    const bool tapdf = par_dfslot(e) && df_init(e, err, sizeof err) && e->dfKr.ptr; // 每列的特徵（接受後寫進各自的環狀 KV）
    const double tst = qw_now();
    static qw_buf ptok;
    if (!ptok.ptr && !alloc(e, &ptok, 64)) return NULL;
    int pos[VBMAX], r0[VBMAX], rowc[VBMAX]; // rowc[r]：第 r 列屬於哪個對話
    for (int i = 0, r = 0; i < B; i++) {
        if (!par_use(e, sids[i]) || e->pos + n[i] + 2 >= e->max_ctx || !kv_reserve(e, e->pos + n[i] + 2)) return NULL;
        pos[i] = e->pos;
        r0[i] = r;
        for (int t = 0; t < n[i]; t++, r++) {
            e->hist[e->pos + 2 + t] = toks[r];
            ((int32_t *)e->tok.ptr)[e->pos + 2 + t] = toks[r];
            ((int32_t *)ptok.ptr)[r] = toks[r];
            rowc[r] = i;
            pv.tok[r] = toks[r];
        }
    }
    pv.B = B;
    for (int i = 0; i < B; i++) { pv.sid[i] = sids[i]; pv.n[i] = n[i]; pv.r0[i] = r0[i]; pv.pos[i] = pos[i]; }
    e->vbn = R;
    qw_batch *b = qw_batch_begin(e->g, false);
    gg_embed_rows(e, b, &ptok, 0, (uint32_t)R, QW_HC, HCD, &e->bh);
    e->bpending = false;
    const unsigned am = at_mask;
    int li = 0, ai = 0;
    for (int l = 0; l < QW_LAYERS; l++) {
        const qw_layer *L = &e->m->layers[l];
        if (L->has_ple) {
            if (e->bpending) {
                uint32_t apply = 1, want = 0;
                qw_batch_kernel(b, vk(e, "kb_hc_in"));
                qw_batch_buf(b, 0, &e->bh, 0);
                qw_batch_buf(b, 1, &e->bout, 0);
                qw_batch_buf(b, 2, &e->binj, 0);
                bind_t(e, b, 3, L->attn_hc.hc_norm);
                qw_batch_buf(b, 4, &e->bnh, 0);
                qw_batch_buf(b, 5, &e->bnf, 0);
                qw_batch_bytes(b, 6, &apply, 4);
                qw_batch_buf(b, 7, &e->bnf, 0);
                qw_batch_buf(b, 8, &e->binj2, 0);
                qw_batch_bytes(b, 9, &want, 4);
                qw_batch_dispatch(b, QW_HC, VB, 1, 1024, 1, 1);
                e->bpending = false;
            }
            for (int i = 0; i < B; i++) { // n-gram：各對話自己的歷史，寫到自己的列
                par_use(e, sids[i]);
                ng_row0 = r0[i];
                ng_gather(e, &L->ple, pos[i] + 2, n[i]);
                ng_row0 = 0;
            }
            e->ng_base = 0;
            ple_gg_proj(e, b, &L->ple, R);
            for (int i = 0; i < B; i++) { // 同 vb_ple：第 t 個 token（t ≥ 1）從前一個之後的狀態出發，寫到 plex[r0 + t − 1]
                par_use(e, sids[i]);
                for (int t = 0; t < n[i]; t++) {
                    const int r = r0[i] + t;
                    e->ng_base = r * 16;
                    vb_copy(e, b, &e->bh, (size_t)r * HCD * 4, &e->h, 0, HCD);
                    if (t >= 1) vb_copy(e, b, t == 1 ? &e->ple_st : &e->plex[r - 2], 0, &e->plex[r - 1], 0, 9 * HCD);
                    qw_buf keep = e->ple_st;
                    if (t >= 1) e->ple_st = e->plex[r - 1];
                    e->pos = pos[i] + t;
                    ple_pre_t = r;
                    ple(e, b, &L->ple);
                    ple_pre_t = -1;
                    e->pos = pos[i];
                    e->ple_st = keep;
                    vb_copy(e, b, &e->h, 0, &e->bh, (size_t)r * HCD * 4, HCD);
                }
            }
            e->ng_base = 0;
        }
        vb_hyper(e, b, &L->attn_hc);
        if (tapdf) { e->df_tapping = true; df_tap_after(e, b, l, &e->bh, R); e->df_tapping = false; }
        if (L->full_attn) {
            const qsa_io bio = {&e->bx, &e->bqg, &e->bkk, &e->bvv, &e->biqk, &e->baq, &e->biq, &e->bao, &e->bay, &e->bout};
            at_mask = 1;
            qsa_layer(e, b, &L->att, ai, 0, R, &bio);
            at_mask = ~0u & ~1u & ~64u & ~128u;
            for (int i = 0; i < B; i++) { // 核心：各對話 n[i] 個 token（同單一對話的驗證），讀寫批次緩衝區自己的列
                par_use(e, sids[i]);
                const size_t r = (size_t)r0[i];
                qsa_io tio = bio;
                tio.oqg = r * 12288 * 2;
                tio.okk = r * 512 * 2;
                tio.ovv = r * 512 * 2;
                tio.oiqk = r * 640 * 2;
                tio.oay = r * 6144 * 2;
                tio.oao = r * 6144 * 2;
                qsa_layer(e, b, &L->att, ai, (uint32_t)pos[i], (uint32_t)n[i], &tio);
            }
            at_mask = 64 | 128;
            qsa_layer(e, b, &L->att, ai, 0, R, &bio);
            at_mask = am;
            ai++;
        } else {
            const qw_deltanet *N = &L->lin;
            if (!dn_fuse(e, N)) return NULL;
            { // dn_inf：R 列一起；各段（對話）從自己的 conv 狀態出發，段內接著前一列（DN_SEG）
                char base[32];
                snprintf(base, sizeof base, "dn_inf_%d_%d", N->in_qkv.gq - 1, N->in_z.gq - 1);
                const dn_in_args a = {2560, {10240, 6144, 64, 64}, 1, 16480};
                const size_t cst = (size_t)li * 3 * 10240 * 4;
                uint32_t chain = 0;
                for (int r = 1; r < R; r++) if (rowc[r] == rowc[r - 1]) chain |= 1u << r;
                qw_batch_kernel(b, dn_kernel_sg(e, base, R));
                bind_t(e, b, 0, N->in_qkv.w);
                bind_t(e, b, 1, N->in_z.w);
                bind_t(e, b, 2, N->in_a.w);
                bind_t(e, b, 3, N->in_b.w);
                qw_batch_buf(b, 4, &e->bx, 0);
                qw_batch_buf(b, 5, &e->bqkv, 0);
                qw_batch_buf(b, 6, &e->bz, 0);
                qw_batch_buf(b, 7, &e->dnG, dn_goff(li, 0));
                qw_batch_buf(b, 8, &e->bbe, 0);
                qw_batch_bytes(b, 9, &a, sizeof a);
                qw_batch_buf(b, 10, &e->bqkv, 0);
                bind_t(e, b, 11, N->conv1d);
                qw_batch_buf(b, 12, par_conv(e, sids[0]), cst);
                bind_t(e, b, 13, N->A_neg);
                bind_t(e, b, 14, N->dt_bias);
                for (int r = 1; r < VBMAX; r++) {
                    const qw_buf *cb = r >= R ? par_conv(e, sids[0]) : (chain >> r & 1) ? &e->convx[r - 1] : par_conv(e, sids[rowc[r]]);
                    qw_batch_buf(b, 14 + r, cb, cst);
                }
                qw_batch_bytes(b, 22, &chain, 4);
                qw_batch_dispatch(b, 16480 / (DN_NSG * dn_r(R)), 1, 1, 32 * DN_NSG, 1, 1);
            }
            for (int i = 0; i < B; i++) { // 遞迴 + gated RMSNorm：各對話 n[i] 步，狀態只讀（defer = 1），k / δ 寫到第 0 組自己的列
                const size_t r = (size_t)r0[i];
                gf_args ga = {0, 1};
                qw_batch_kernel(b, gf_kernel(e, n[i]));
                qw_batch_buf(b, 0, &e->bqkv, r * 10240 * 2);
                qw_batch_buf(b, 1, &e->dnG, dn_goff(li, 0) + r * 64 * 4);
                qw_batch_buf(b, 2, &e->bbe, r * 64 * 4);
                qw_batch_buf(b, 3, par_S(e, sids[i]), (size_t)li * 48 * 128 * 128 * 4);
                qw_batch_buf(b, 4, &e->bgo, r * 6144 * 2);
                qw_batch_buf(b, 5, &e->bz, r * 6144 * 2);
                bind_t(e, b, 6, N->norm);
                qw_batch_buf(b, 7, &e->bgy, r * 6144 * 2);
                qw_batch_buf(b, 8, &e->dncnt, 0);
                qw_batch_buf(b, 9, &e->dnK, dn_koff(li, 0) + r * 2048 * 2);
                qw_batch_buf(b, 10, &e->dnD, dn_doff(li, 0) + r * 48 * 128 * 4);
                qw_batch_buf(b, 11, &e->dnK, dn_koff(li, 0));
                qw_batch_buf(b, 12, &e->dnG, dn_goff(li, 0));
                qw_batch_buf(b, 13, &e->dnD, dn_doff(li, 0));
                qw_batch_bytes(b, 14, &ga, sizeof ga);
                qw_batch_dispatch(b, 48, 128 / gf_dvb(), 1, 8 * gf_dvb(), 1, 1);
            }
            dn_out(e, b, N, &e->bgy, &e->bout, R);
            li++;
        }
        if (trace_on()) for (int r = 0; r < R; r++) trace_copy(e, b, &e->bh, (size_t)r * HCD * 4, &e->bout, (size_t)r * D * 2, r, l, 0);
        e->bpending = true;
        vb_hyper(e, b, &L->mlp_hc);
        vb_moe(e, b, &L->moe);
        if (trace_on()) for (int r = 0; r < R; r++) trace_copy(e, b, &e->bh, (size_t)r * HCD * 4, &e->bout, (size_t)r * D * 2, r, l, 1);
        e->bpending = true;
    }
    vb_hyper(e, b, &e->m->mixer);
    e->bpending = false;
    vb_qmv(e, b, &e->m->lm_head, &e->bx, &e->blg, true);
    const double tenc = qw_now(), t = qw_batch_end(b);
    if (gpu_ms) *gpu_ms = t * 1e3;
    if (t < 0) return NULL;
    if (getenv("QW_PAR_PROF")) fprintf(stderr, "[parv] B=%d R=%d：編碼 %.2f ms、送出到完成 %.2f ms（GPU %.2f）\n", B, R, (tenc - tst) * 1e3, (qw_now() - tenc) * 1e3, t * 1e3);
    for (int i = 0; i < B; i++) { par_use(e, sids[i]); e->pos = pos[i]; }
    return e->blg.ptr;
}
// 上一次 par_vstep 之後：對話 i 接受了 acc[i] 個猜測（共 acc[i] + 1 個 token 進了模型）。DeltaNet 重放 acc[i] + 1 步寫回、
// conv / PLE 狀態換成第 acc[i] 個 token 之後的那份、位置前進
static qw_buf pd_ids, pd_path, pd_prob; // 各 slot 的猜測：anchor + mask [slot][block]、結果 [slot][DF_ROWS]
static bool pd_ok(qw_engine *e) {
    char err[256];
    if (!par_dfslot(e) || !df_init(e, err, sizeof err) || !e->dfKr.ptr) return false;
    return pd_ids.ptr || (alloc(e, &pd_ids, (size_t)QW_PAR_MAX * e->df.block * 4) && alloc(e, &pd_path, (size_t)QW_PAR_MAX * DF_ROWS * 4) &&
                          alloc(e, &pd_prob, (size_t)QW_PAR_MAX * DF_ROWS * 4));
}
static void pd_encode(qw_engine *e, qw_batch *b, int sid, int32_t anchor) { // 目前的 slot（sid）：從位置 pos、anchor 猜一塊
    struct qw_slot *ss = &e->slots[sid];
    int32_t *ids = (int32_t *)pd_ids.ptr + (size_t)sid * e->df.block;
    ids[0] = anchor;
    for (int i = 1; i < e->df.block; i++) ids[i] = e->df.mask;
    const int lo = e->df_lo;
    e->df_lo = ss->dfdone == e->pos && ss->dflo <= e->pos ? ss->dflo : e->pos; // 環狀 KV 接得上才用（否則沒有上下文）
    dfx_ids = &pd_ids; dfx_path = &pd_path; dfx_prob = &pd_prob; dfx_irow = sid; dfx_orow = sid;
    df_draft(e, b, e->pos, NULL);
    dfx_ids = dfx_path = dfx_prob = NULL;
    e->df_lo = lo;
}
int qw_engine_par_vdraft(qw_engine *e, const int *sids, const int32_t *anchors, int B) {
    if (!pd_ok(e)) return -1;
    qw_batch *b = qw_batch_begin(e->g, false);
    for (int i = 0; i < B; i++) { if (!par_use(e, sids[i])) { qw_batch_end(b); return -1; } pd_encode(e, b, sids[i], anchors[i]); }
    return qw_batch_end(b) < 0 ? -1 : 0;
}
int qw_engine_par_draft(qw_engine *e, int sid, int32_t *dr, float *pr) {
    if (!pd_path.ptr || sid < 0 || sid >= QW_PAR_MAX) return 0;
    const int n = e->df.block - 1 < DF_ROWS ? e->df.block - 1 : DF_ROWS;
    memcpy(dr, (int32_t *)pd_path.ptr + (size_t)sid * DF_ROWS, sizeof(int32_t) * (size_t)n);
    memcpy(pr, (float *)pd_prob.ptr + (size_t)sid * DF_ROWS, sizeof(float) * (size_t)n);
    return n;
}
int qw_engine_par_vaccept(qw_engine *e, const int *acc, const int32_t *nx) {
    if (pv.B < 1) return -1;
    const double tva = qw_now();
    const bool df = pd_ok(e);
    qw_batch *b = qw_batch_begin(e->g, false);
    for (int i = 0; i < pv.B; i++) {
        const int j = acc[i], r = pv.r0[i];
        if (j < 0 || j >= pv.n[i] || !par_use(e, pv.sid[i])) { qw_batch_end(b); return -1; }
        const uint32_t J = (uint32_t)j + 1;
        for (int li = 0; li < NLIN; li++) {
            qw_batch_kernel(b, e->gdn_commit_t);
            qw_batch_buf(b, 0, &e->dnK, dn_koff(li, 0) + (size_t)r * 2048 * 2);
            qw_batch_buf(b, 1, &e->dnG, dn_goff(li, 0) + (size_t)r * 64 * 4);
            qw_batch_buf(b, 2, &e->dnD, dn_doff(li, 0) + (size_t)r * 48 * 128 * 4);
            qw_batch_buf(b, 3, &e->S, (size_t)li * 48 * 128 * 128 * 4);
            qw_batch_bytes(b, 4, &J, 4);
            qw_batch_dispatch(b, 48, 128 / 32, 1, 256, 1, 1);
        }
        if (j >= 1) {
            qw_buf t2 = e->conv_st; e->conv_st = e->convx[r + j - 1]; e->convx[r + j - 1] = t2;
            qw_buf t3 = e->ple_st; e->ple_st = e->plex[r + j - 1]; e->plex[r + j - 1] = t3;
        }
        e->pos = pv.pos[i] + 1 + j;
        for (int t = 0; t <= j; t++) if (pv.tok[r + t] == EOS_ID) e->last_eos = pv.pos[i] + t + 1;
        if (df) { // 接受的 j + 1 個位置寫進這個對話的環狀 KV；接不上時從這裡重新累積
            struct qw_slot *ss = &e->slots[pv.sid[i]];
            if (ss->dfdone != pv.pos[i]) ss->dflo = pv.pos[i];
            df_ctx(e, b, r, j + 1, pv.pos[i]);
            ss->dfdone = e->pos;
            if (e->df_slot == pv.sid[i]) e->df_slot = -1;
            if (nx) pd_encode(e, b, pv.sid[i], nx[i]);
        }
    }
    pv.B = 0;
    const double ta = qw_now(), tg = qw_batch_end(b);
    if (getenv("QW_PAR_PROF")) fprintf(stderr, "[parv] accept：編碼 %.2f ms、送出到完成 %.2f ms（GPU %.2f）\n", (ta - tva) * 1e3, (qw_now() - ta) * 1e3, tg * 1e3);
    return tg < 0 ? -1 : 0;
}

// 推測解碼的接受判定（同 qw_engine_generate_ex）：lg 是 nd + 1 列 logits（第 t 列 = 餵了第 t 個 token 之後），dr 是 nd 個猜測。
// 回傳接受幾個（j），*nx = 下一個 token。greedy：argmax 比對；取樣：依機率接受、拒絕時從 (p − q)+ 重抽（rng 會前進）
int qw_engine_spec_accept(const float *lg, const int32_t *dr, int nd, const qw_gen *g, unsigned long long *rng, int32_t *nx) {
    int j = 0;
    if (g->temperature <= 0) {
        while (j < nd && argmax_cpu(lg + (size_t)j * QW_VOCAB) == dr[j]) j++;
        *nx = argmax_cpu(lg + (size_t)j * QW_VOCAB);
        return j;
    }
    uint64_t r = *rng;
    dist d;
    for (;;) {
        make_dist(lg + (size_t)j * QW_VOCAB, g, &d);
        if (j == nd) { *nx = dist_draw(&d, &r, -1); break; }
        if (rng_unif(&r) < dist_p(&d, dr[j])) { j++; continue; }
        *nx = dist_draw(&d, &r, dr[j]);
        break;
    }
    *rng = r;
    return j;
}

// 取樣（與 qw_engine_generate_ex 相同的規則）：temperature ≤ 0 → argmax；否則前 k 個 → softmax(logit / T) → top-p
int32_t qw_engine_sample(const float *lg, const qw_gen *g, unsigned long long *rng) {
    if (g->temperature <= 0) return argmax_cpu(lg);
    dist d;
    make_dist(lg, g, &d);
    uint64_t r = *rng;
    const int32_t t = dist_draw(&d, &r, -1);
    *rng = r;
    return t;
}

