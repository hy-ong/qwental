// GGUF 區塊的 CPU 解碼（照 ggml 的 dequantize_row_*）：載入時把 dense 權重轉成 INT8 用（requant.c 的 qw_gguf_c8）
#pragma once
#include <stdbool.h>

// 支援的 ggml type：F32、F16、Q8_0、Q4_K、Q5_K、Q6_K、IQ4_NL、IQ4_XS、BF16
bool qw_ggml_dq_ok(int type);
// 一列 K 個值（K 是區塊大小的倍數）；type 不支援時回傳 false
bool qw_ggml_dq_row(int type, const void *src, float *dst, int K);
