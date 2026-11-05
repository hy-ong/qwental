// Qwen3.8-Flash-Next 的模型結構與權重對照
// 權重直接指向 mmap 的 safetensors（不複製）。
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "st.h"

enum {
    QW_HIDDEN = 2560,
    QW_LAYERS = 48,
    QW_HC = 4,               // hyper-connection 的流數
    QW_HC_LOWRANK = 320,
    QW_VOCAB = 248320,
    QW_EXPERTS = 512,
    QW_TOPK = 10,
    QW_MOE_INTER = 640,
    QW_SHARED_INTER = 640,
    // Gated DeltaNet
    QW_LIN_KH = 16, QW_LIN_VH = 48, QW_LIN_DK = 128, QW_LIN_DV = 128, QW_LIN_CONV = 4,
    // QSA attention
    QW_ATT_QH = 24, QW_ATT_KVH = 2, QW_ATT_HD = 256, QW_ROPE_DIM = 64,
    QW_IDX_H = 4, QW_IDX_D = 128, QW_IDX_CR = 4, QW_IDX_BUDGET = 2048,
};

// 一個量化線性層（MLX affine，group 64）。ne > 1 時是多個 expert 疊在一起。
typedef struct qw_qlin {
    const st_tensor *w, *s, *b; // weight（U32）、scales、biases（BF16）
    int bits, out, in, ne;
    bool chan; // 每列對稱 8-bit（每組 scale 相同、bias = −128·scale）：prefill 可走 INT8 × INT8
    const void *rsum_t; // chan 時：每列 Σ(q − 128)（int32 [列]，st_tensor）
    int gq;             // GGUF 格式：ggml type + 1（0 = MLX affine）。此時 w 是原始區塊 [ne?·out][每列位元組]、s / b 為 NULL、bits = 0；
                        // BF16 矩陣也走這裡（gq = 31）
    const struct qw_qlin *i8; // GGUF 的 INT8 影子（QW_INT8）：每列對稱 8-bit（chan、沒有 bias），只給 prefill 用；NULL = 沒有
    bool kn; // w 排成 [in][out]（只有 GGUF 的 INT8 影子：每列對稱 8-bit，prefill 的右運算元不轉置）
} qw_qlin;

// 一個不量化的 tensor（BF16 / F32 / I64）
typedef const st_tensor *qw_t;

typedef struct {
    qw_t hc_norm;              // [10240]（GGUF：F32）
    qw_qlin mix_down, mix_up;  // 10240→320、320→10240（GGUF：BF16，gq = 31）
    qw_t inject;               // [4, 10240]（mixer 沒有）
} qw_hyper;

typedef struct {
    qw_qlin in_qkv, in_z, in_a, in_b, out;
    qw_t conv1d, A_log, dt_bias, norm;
    qw_t A_neg; // GGUF profile：−exp(A_log)（F32，V head 交錯順序）；此時 A_log 為 NULL
} qw_deltanet;

typedef struct {
    qw_qlin q, k, v, o, idx_qk;
    qw_t q_norm, k_norm, idx_q_norm, idx_k_norm;
    bool nf32; // 四個 norm 是 F32（GGUF profile：保留原本的全精度值）；false = BF16（MLX）
} qw_attn;

typedef struct {
    qw_qlin router;                  // 8-bit [512, 2560]
    qw_qlin gate, up, down;          // routed experts（4-bit，ne = 512）
    qw_qlin sh_gate, sh_up, sh_down; // 共用 expert（8-bit）
    qw_t sh_gate_w;                  // shared_expert_gate [1, 2560] BF16
    // routed expert 也可能是 GGUF 的 IQ / Q2_0 區塊（GGUF profile 或混合模型目錄）：U8 [E, 列, 每列位元組]，type 是 ggml 編號
    const st_tensor *giq, *uiq, *diq;
    int fgu, fd;
} qw_moe;

typedef struct {
    qw_qlin key_proj, value_proj;
    qw_t norm_key, norm_query, norm_conv, conv1d;
    qw_t multipliers, head_offsets, head_vocab;
} qw_ple;

typedef struct {
    bool full_attn; // true：QSA attention；false：Gated DeltaNet
    qw_hyper attn_hc, mlp_hc;
    qw_deltanet lin;
    qw_attn att;
    qw_moe moe;
    bool has_ple;
    qw_ple ple;
} qw_layer;

typedef struct {
    st_model *st;
    qw_t embed_w, embed_s, embed_b; // embed_tokens（量化）
    int embed_bits;
    qw_layer layers[QW_LAYERS];
    qw_hyper mixer;
    qw_qlin lm_head;
    // MTP（multi-token prediction）：一層 QSA decoder + 輸入投影 + 自己的 mixer，共用 embed 與 lm_head
    bool has_mtp;
    qw_layer mtp;
    qw_qlin mtp_fc_emb, mtp_fc_hidden;  // 2560 → 2560（8-bit）
    qw_t mtp_norm_emb, mtp_norm_hidden; // [2560]、[10240]（整個 4 條流一起正規化）
    qw_hyper mtp_mixer;
    size_t total_bytes;  // 用到的 tensor 合計位元組
    int ntensors_used;
    // GGUF profile（QW_GGUF）：哪些部分改用 GGUF 的權重（QW_GG_* 位元），以及那邊的特殊處理
    unsigned gguf_parts;
    bool lin_tiled;            // DeltaNet 的 V head 是交錯順序（r·16 + k），對應的 k head 是 h % 16（不是 h / 3）
    int embed_gq;              // embed_tokens 的 ggml type + 1（0 = MLX）
    const uint8_t *ple_tab;    // PLE n-gram 表（GGUF，只給 CPU 讀）；NULL = 用 MLX 的 ngram_table.bin
    int ple_tab_gq;
    size_t ple_row_bytes;
} qw_model;

// GGUF profile 的部分（QW_GGUF_PARTS，逗號分隔；all = 全部）
enum {
    QW_GG_MOE = 1u << 0,    // routed expert
    QW_GG_SHARED = 1u << 1, // shared expert
    QW_GG_ROUTER = 1u << 2, // router + shared gate
    QW_GG_ATTN = 1u << 3,   // attention（q / k / v / o、norm、indexer）
    QW_GG_LIN = 1u << 4,    // DeltaNet（qkv / z / a / b / out、A、dt、conv、norm）
    QW_GG_HYPER = 1u << 5,  // hyper-connection（各層與 mixer）
    QW_GG_LM = 1u << 6,     // lm_head
    QW_GG_EMBED = 1u << 7,  // embed_tokens
    QW_GG_PLE = 1u << 8,    // PLE（第 1 層的 key / value / norm / conv 與 n-gram 表）
    QW_GG_ALL = (1u << 9) - 1,
};

// 開啟模型目錄並把所有權重對應好；任何缺漏或形狀不符都會失敗並寫進 err。
qw_model *qw_model_open(const char *dir, char *err, size_t errlen);
void qw_model_close(qw_model *m);
