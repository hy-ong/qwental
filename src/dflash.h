// DFlash2 drafter（incoai/Qwen3.8-Flash-Next-DFlash2，Strata 的 DF2P 打包格式）。
// 載入時把 Q4_0 矩陣拆成 nibble 與 FP16 d 兩個陣列（數值、位元組數都不變，kernels/df_q4.metal 直接讀；qw_qlin 的 b = NULL、bits = 4），
// 和 drafter 的其他 tensor 一起放進一個記憶體裡的虛擬分片。QW_DF_Q8=1 改回舊的做法：展開成 MLX 8-bit affine（group 64，
// 重新量化，數值略有出入、位元組數加倍），走 qmvb。另外從主模型的 lm_head 取出 draft vocab 的那些列，當 drafter 的輸出頭。
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include "model.h"

enum { DF_MAXL = 8 };

typedef struct {
    qw_qlin q, k, v, o, gate, up, down, akp, mkp; // akp / mkp：attention / mlp 動態卷積的 kernel_projection（2560 → 640）
    const st_tensor *in_norm, *post_norm, *q_norm, *k_norm, *abase, *mbase; // F32；base：[2][2][H]（prep、finish 各 2 個 tap）
} qw_df_layer;

typedef struct {
    int H, L, NH, NKV, HD, inter, win, block, gs, mask, rank, topk, ntaps, taps[8];
    float theta, eps;
    qw_qlin fc, hp; // fc：taps·H → H；hp：candidate_selector.hidden_projection（H → rank）
    const st_tensor *hidden_norm, *norm;
    qw_df_layer layer[DF_MAXL];
    const st_tensor *pred_cb, *succ_cb; // BF16 [V][rank]；cb_q4 時是 Q4_0 原始區塊 [V][rank / 32 · 18]（selq4 打包檔）
    bool cb_q4;
    qw_qlin head;                       // lm_head 的 draft vocab 列（位元數同主模型的 lm_head）
    const st_tensor *vocab;             // U32 [nv]：head 第 i 列對應的 token id
    int nv;
} qw_dflash;

bool qw_dflash_load(qw_model *m, const char *pack, const char *vocab, qw_dflash *d, char *err, size_t errlen);
