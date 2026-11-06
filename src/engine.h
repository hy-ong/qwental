// Qwental 推論 engine（目前：decode 路徑，一次一個 token；prompt 也逐個 token 餵入）
#pragma once

#define QW_VERSION "1.2.0"
#include <stdbool.h>
#include <stdint.h>
#include "gpu.h"
#include "model.h"

typedef struct qw_engine qw_engine;

// 開啟模型目錄（權重 mmap 後直接包成 Metal buffer，不複製）。max_ctx：最多幾個 token。
qw_engine *qw_engine_open(const char *model_dir, int max_ctx, char *err, size_t errlen);
void qw_engine_close(qw_engine *e);

// 餵入一個 token，算出下一個 token 的 logits（FP32，長度 QW_VOCAB），回傳指標（engine 內部的記憶體）。
// dump_layers 非 NULL 時，把每一層之後的殘差流（[48][HC·D] FP32）複製進去（除錯比對用）。
const float *qw_engine_step(qw_engine *e, int32_t token, float *dump_layers);

// 流水線 greedy 生成 n 個 token（第一個是最後一次 qw_engine_step 的 logits 的 argmax）。
// argmax 在 GPU 上做，CPU 在 GPU 執行第 t 步時編碼第 t+1 步。回傳實際生成數，gpu_ms 為 GPU 時間合計。
int qw_engine_generate(qw_engine *e, int n, int32_t *out, double *gpu_ms);

// 一次 prefill 多個 token（context ≤ 2048 的部分批次處理；更長的暫時逐 token）。
// 回傳最後一個 token 的 logits。dump 非 NULL 時（n ≤ 2048）存每層殘差流 [48][n][HC·D]。
const float *qw_engine_prefill(qw_engine *e, const int32_t *ids, int n, float *dump, double *gpu_ms);

// MTP 猜一個 token（必須緊接在主模型處理位置 pos 的那一步之後）：next 是主模型選出的下一個 token，
// 回傳 MTP 預測的再下一個 token（沒有 MTP 權重時回傳 −1）
int32_t qw_engine_mtp_draft(qw_engine *e, int32_t next, int pos, double *gpu_ms);

// MTP 推測解碼（greedy）：逐 token 處理 prompt（同時填 MTP 的 cache），之後每輪一次驗證 2 個 token。
// 回傳生成數；stats = [輪數, 猜中數]；times = [prompt 的 GPU ms, 生成的 GPU ms, 生成的牆鐘 ms]
int qw_engine_generate_mtp(qw_engine *e, const int32_t *prompt, int np, int n, int32_t *out, int *stats, double *times);

// 一般生成（給 server 用）：處理 prompt → 每生成一個 token 呼叫 cb（回傳 false 就停）
typedef struct {
    float temperature;     // ≤ 0：greedy
    int top_k;             // ≤ 0 或 > 256：取前 256
    float top_p;           // 1：不截斷
    unsigned long long seed;
    int max_tokens;
    const int32_t *stop;   // 遇到就停（不送出），例如 <|im_end|>
    int nstop;
    bool spec;             // 推測解碼（MTP + n-gram）
    const int *persist;    // 處理到這些位置時把狀態存到 SSD（例如 system 訊息結尾）；需要 qw_engine_set_cache_dir
    int npersist;
} qw_gen;
typedef bool (*qw_tok_cb)(void *ctx, int32_t tok);
typedef struct {
    int prompt, cached, gen, rounds, accepted, ngram_rounds;
    double prefill_ms, gen_ms;
} qw_gen_stats;
enum { QW_FIN_STOP, QW_FIN_LENGTH, QW_FIN_CALLBACK, QW_FIN_CTX, QW_FIN_ERROR = -1 };
int qw_engine_generate_ex(qw_engine *e, const int32_t *prompt, int np, const qw_gen *g, qw_tok_cb cb, void *ctx, qw_gen_stats *st);
void qw_engine_reset(qw_engine *e); // 回到空的 context
void qw_engine_set_cache_dir(qw_engine *e, const char *dir); // 持久化前綴快取的目錄

int qw_engine_pos(const qw_engine *e);
qw_gpu *qw_engine_gpu(qw_engine *e);
double qw_engine_last_gpu_ms(const qw_engine *e);
// 各段的穩態時間（真實權重、同一段連續跑 reps 次）
void qw_engine_kbench(qw_engine *e, int reps);
void qw_engine_pf_kbench(qw_engine *e, int T, int reps);
void qw_engine_hctest(qw_engine *e, const char *dir); // hyper-connection 的 kernel 測試（輸入輸出寫到 dir）
int qw_engine_df_test(qw_engine *e, const float *feats, const int32_t *toks, int n, int npr); // DFlash2 離線測試 // prefill 各部分的微基準（在最後一段的資料上）

// 除錯（QW_TRACE）：驗證與 decode 逐層比對
const void *qw_engine_trace_vb(qw_engine *e, const int32_t *toks, int k);
const void *qw_engine_trace_buf(qw_engine *e, int tok);
