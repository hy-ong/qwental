// Qwental 推論 engine（目前：decode 路徑，一次一個 token；prompt 也逐個 token 餵入）
#pragma once

#define QW_VERSION "1.7.1"
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
double qw_engine_vb_bench(qw_engine *e, const int32_t *toks, int k, int reps); // 驗證 k 個 token 的 GPU ms（實驗）
int32_t qw_engine_mtp_chain(qw_engine *e, int32_t next, int pos, double *gpu_ms); // 接著上一次猜下一個（實驗）

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
    volatile bool *yield; // 非 NULL 且為 true 時：在推測解碼一輪結束（或不推測時一個 token）後停下，回傳 QW_FIN_YIELD（並發模式切換用）
    bool (*pf_yield)(void *ctx); // 非 NULL：長 prompt 每 PF 個 token 一段，段與段之間問一次（ctx 同 cb），true 就存快照、回傳 QW_FIN_YIELD（st->pf_done = 算到哪）
} qw_gen;
typedef bool (*qw_tok_cb)(void *ctx, int32_t tok);
typedef struct {
    int prompt, cached, gen, rounds, accepted, ngram_rounds;
    int pf_done; // > 0：prompt 只 prefill 到這裡就讓出（pf_yield），還沒生成
    double prefill_ms, gen_ms;
} qw_gen_stats;
enum { QW_FIN_STOP, QW_FIN_LENGTH, QW_FIN_CALLBACK, QW_FIN_CTX, QW_FIN_YIELD, QW_FIN_ERROR = -1 };
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

// ---- 並發（--parallel N）：每個對話一個 slot（QW_SLOTS ≥ 並發數）。GGUF 與單獨跑逐位元相同；MLX 只在 QW_EXACT_SPEC 時相同（否則捨入略有不同）
int qw_engine_par_slots(qw_engine *e);
enum { QW_PAR_MAX = 64 }; // 最多幾個對話同時進行（每步超過 8 個時分組算）
int qw_engine_par_pick(qw_engine *e, const int32_t *prompt, int np, unsigned long long busy); // 選沒在用的 slot（busy 位元 = 正在用；不為了共用前綴蓋掉長對話），-1 = 沒有
const float *qw_engine_par_prefill(qw_engine *e, int sid, const int32_t *prompt, int np, int *cached); // 回傳最後位置的 logits
const float *qw_engine_par_prefill_to(qw_engine *e, int sid, const int32_t *prompt, int np, int chunk, int *done, int *cached); // 分段：一次最多 chunk 個，*done = 算到哪（> 0 從那裡接著算）
void qw_engine_par_persist(qw_engine *e, const int *pos, int n); // 接下來的 par_prefill 處理到這些位置時把狀態存到 SSD（同 qw_gen.persist）
// 共用前綴的交接點（serve 合併同時進來、前綴相同的 prompt）：接下來的 par_prefill_to 算到這些位置時段尾停在那裡、記快照（只在記憶體，不寫 SSD），
// 別的 slot 的 par_prefill_to 再由 xs_copy 複製過去。位置照這組 slot 第一段的起點對齊到 GDN 分塊（合法的分段點：結果與一次算完逐位元相同）。
// 每次 qw_engine_par_persist 都會清掉，要在它之後呼叫；最多 QW_PAR_SNAPAT_MAX 個
enum { QW_PAR_SNAPAT_MAX = 4 };
void qw_engine_par_snapat(qw_engine *e, const int *pos, int n);
int qw_engine_par_copied(qw_engine *e); // 上一次 par_prefill_to 從別的 slot 複製了幾個 token 的前綴（0 = 沒有）
int qw_engine_par_clip(int s0, int start, int end, const int *pts, int n, int cc); // 純函式（測試用）：pts 對齊後落在 (start, end) 的最小位置，沒有就是 end
const float *qw_engine_par_step(qw_engine *e, const int *sids, const int32_t *toks, int B, double *gpu_ms); // [B][vocab] logits
void qw_engine_par_finish(qw_engine *e, int sid, const int32_t *gen, int ngen); // 生成結束：存快照給下次接續
int qw_engine_par_solo(qw_engine *e, int sid, const int32_t *prompt, int np, const qw_gen *g, qw_tok_cb cb, void *ctx,
                       qw_gen_stats *st); // 只有一個請求：在 slot sid 走一般生成（推測解碼）；cb 回傳 false 中途停下
double qw_engine_par_cost(qw_engine *e, int B); // B 個對話一起 decode 一步的預估牆鐘 ms（最佳分組；0 = 還沒量過）
size_t qw_engine_par_mem(qw_engine *e); // 各 slot 佔的記憶體合計（位元組）
int qw_engine_par_trim(qw_engine *e, unsigned long long busy, size_t budget, bool one); // 超過 budget 就釋放閒置 slot（重算成本 × 近期度最小的先放；one：只放一組）；回傳放了幾組
// 縮小閒置的 slot（busy 位元 = 正在用，不動）：KV 容量縮到 max(kvlen, pos) + 餘量、對齊 4096；快照只留最長的 QW_IDLE_SNAPS 份（預設 2）。
// 要在 GPU 閒置時呼叫（引擎執行緒、兩步之間）。回傳省下的位元組。par_trim 在超過 budget / one 模式時會先呼叫它，
// 平常也會縮閒置超過 QW_SHRINK_IDLE 秒（預設 60）的；serve 一般不必自己叫
size_t qw_engine_par_shrink(qw_engine *e, unsigned long long busy);
// 分段 prefill（par_prefill_to）中途停下時（客戶端取消、逾時）呼叫：slot sid 停在段尾（pos == kvlen）就把狀態記成快照，
// 同一個 prompt 重送時 par_pick 會選回這組、par_prefill_to（*done = 0）從這裡接續，結果與一次算完逐位元相同。
// 沒載入的 slot 也可以。回傳 true = 這組現在有長度 kvlen 的快照；false = 不在段界（例如已在生成，改用 par_finish）或 sid 無效
bool qw_engine_par_snap(qw_engine *e, int sid);
const float *qw_engine_par_vstep(qw_engine *e, const int *sids, const int *n, const int32_t *toks, int B,
                                 double *gpu_ms); // 並發推測解碼：對話 i 驗證 n[i] 個 token（toks 依序排，Σn ≤ 8；GGUF）；回傳 [Σn][vocab] logits
int qw_engine_par_vaccept(qw_engine *e, const int *acc, const int32_t *nx); // 上一次 vstep 之後：對話 i 接受了 acc[i] 個猜測（狀態寫回、位置前進）；
                                                                          // nx 非 NULL：接著從 nx[i] 猜下一塊（DFlash2，qw_engine_par_draft 取）
int qw_engine_par_vdraft(qw_engine *e, const int *sids, const int32_t *anchors, int B); // 各對話從目前位置、anchor 猜一塊
void qw_engine_par_keepdf(qw_engine *e, bool on); // 一起 decode 時也維持各對話的 DFlash2 上下文（之後要換成一起推測解碼時用）
int qw_engine_par_draft(qw_engine *e, int sid, int32_t *dr, float *pr); // 這個對話最近一次的猜測（回傳個數；pr = 各位置的信心）
// mixed step（實驗）：設定之後，qw_engine_par_prefill_to 的每一段 prefill 送到 GPU 後、跑完之前，engine 反覆呼叫 fn(ctx)；
// fn 只能做一次 qw_engine_par_step（不含正在 prefill 的 slot）、取樣與送出 token（含 qw_engine_par_finish），不能 prefill / 驗證 / 猜測；
// 回傳 false = 沒有要 decode 的了。decode 走第二條 GPU 串流與 prefill 同時執行，兩邊結果都與分開跑逐位元相同（只支援 GGUF；
// 其他情況照舊不重疊）。fn = NULL 關閉
typedef bool (*qw_mix_fn)(void *ctx);
void qw_engine_par_mix(qw_engine *e, qw_mix_fn fn, void *ctx);
int32_t qw_engine_sample(const float *logits, const qw_gen *g, unsigned long long *rng);
int qw_engine_spec_accept(const float *lg, const int32_t *dr, int nd, const qw_gen *g, unsigned long long *rng,
                          int32_t *nx); // 推測解碼的接受判定：回傳接受幾個猜測，*nx = 下一個 token // 同 qw_engine_generate_ex 的取樣（rng 會前進）
