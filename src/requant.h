// 8-bit dense 權重 → 4-bit（載入時）。sel："1" / "all" 全部，或以逗號列出類別：
//   lin（DeltaNet 投影）、attn（attention 投影）、lm（lm_head）、mixdown（hyper-connection 的 mix_down）、
//   shared（共用 expert）、mtp（MTP 層）
// hyper 的 mix_up 與 router 由寫死 8-bit 的 kernel 使用，不轉。
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include "model.h"

bool qw_requant4(qw_model *m, const char *sel, char *err, size_t errlen);
