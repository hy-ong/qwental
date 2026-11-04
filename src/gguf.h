// GGUF profile：讀這個模型架構的 GGUF（例如 Strata 預設的 IQ3_S、huihui UD-Q4_K_XL），把選定部分的 tensor
// 以 MLX 的名稱加進 st_model（放在最前面，蓋過 MLX 的同名 tensor），model.c 照常依名稱對應。
//   量化矩陣：dtype ST_U8、gq = ggml type + 1、shape [列, 每列位元組]（expert：[512, 列, 每列位元組]，名稱 …switch_mlp.*_proj.iq）
//   BF16 矩陣：dtype ST_BF16、gq = 31（GGML BF16 + 1）
//   小 tensor：F32 / F16 的值剛好都是 BF16 時無損轉成 BF16；否則保留 F32（例如 norm），讓還沒支援的 kernel 在 model.c 型別檢查時失敗
//   語意不同的 tensor 換名字：ssm_a（= −exp(A_log)，F32）→ linear_attn.A_neg
//   DeltaNet 的 V head 是交錯順序：qw_model.lin_tiled = true
// PLE 的 n-gram 表（28.8 GB）不放進 st_model（不包成 Metal buffer），只留給 CPU 讀。
#pragma once
#include "model.h"
#include "st.h"
#include <stdbool.h>
#include <stddef.h>

// 解析 QW_GGUF_PARTS（"all"、"moe,attn,…"）成 QW_GG_* 位元
unsigned qw_gguf_parse_parts(const char *s);

// dir：放 GGUF 分片的目錄（*.gguf，依檔名排序）。成功後 m->gguf_parts 等欄位已設定
bool qw_gguf_attach(st_model *st, qw_model *m, const char *dir, unsigned parts, char *err, size_t errlen);

// ggml type → 每塊值數、每塊位元組；不認得回傳 false
bool qw_ggml_block(int type, int *qb, int *bb);
const char *qw_ggml_name(int type);
