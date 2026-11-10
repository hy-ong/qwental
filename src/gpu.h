// Qwental GPU 介面：所有 Metal 呼叫都藏在 gpu_metal.m 裡，其他部分只用這個 C API。
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct qw_gpu qw_gpu;
typedef struct qw_kernel qw_kernel;
typedef struct qw_batch qw_batch;

typedef struct {
    void *handle; // id<MTLBuffer>（bridged）
    void *ptr;    // CPU 可直接讀寫的位址（統一記憶體）
    size_t size;
} qw_buf;

// 裝置
qw_gpu *qw_gpu_open(void);
void qw_gpu_close(qw_gpu *g);
const char *qw_gpu_name(const qw_gpu *g);
size_t qw_gpu_max_working_set(const qw_gpu *g);

// 在執行期編譯 MSL。成功回傳 true；失敗時把錯誤訊息寫進 err。
// 可以多次呼叫，每次編譯的 library 都會加入搜尋範圍。
bool qw_gpu_compile(qw_gpu *g, const char *src, size_t len, char *err, size_t errlen);

// 依名稱取得 kernel（第一次會建立 pipeline）。找不到回傳 NULL。
qw_kernel *qw_kernel_get(qw_gpu *g, const char *name);
unsigned qw_kernel_max_threads(const qw_kernel *k);
unsigned qw_kernel_simd_width(const qw_kernel *k);

// 緩衝區（shared storage，CPU 和 GPU 共用）
bool qw_buf_new(qw_gpu *g, size_t size, qw_buf *out);
void qw_buf_free(qw_buf *b);
// 把已經 mmap 的記憶體（page 對齊）直接包成 Metal buffer，不複製。len 會補到 page 的倍數。
bool qw_buf_wrap(qw_gpu *g, void *ptr, size_t len, qw_buf *out);

// 一批 GPU 工作 = 一個 command buffer + 一個 compute encoder。
// concurrent=false：每個 dispatch 依序執行（serial dispatch type）。
// concurrent=true ：dispatch 之間可以重疊，需要時自己呼叫 qw_batch_barrier。
qw_batch *qw_batch_begin(qw_gpu *g, bool concurrent);
void qw_batch_kernel(qw_batch *b, qw_kernel *k);
void qw_batch_buf(qw_batch *b, unsigned index, const qw_buf *buf, size_t offset);
void qw_batch_bytes(qw_batch *b, unsigned index, const void *data, size_t len);
void qw_batch_threadgroup_mem(qw_batch *b, unsigned index, size_t len);
// 以 threadgroup 為單位送出
void qw_batch_dispatch(qw_batch *b, unsigned gx, unsigned gy, unsigned gz,
                       unsigned tx, unsigned ty, unsigned tz);
void qw_batch_barrier(qw_batch *b);
// 把目前為止編好的部分先送出（GPU 先開始跑），之後的 dispatch 編進新的 command buffer（以 event 接在後面，順序不變）。
// 一個 batch 最多分 8 次；profiling 時不分。qw_batch_end 回傳的 GPU 時間是各段加總
void qw_batch_flush(qw_batch *b);
// 送出並等待完成，回傳 GPU 實際執行時間（秒）；失敗回傳負值。
double qw_batch_end(qw_batch *b);
// 分開的版本：submit 只送出（不等）；wait 等完成、回傳 GPU 時間並釋放 batch
void qw_batch_submit(qw_batch *b);
double qw_batch_wait(qw_batch *b);
// 已 submit 的 batch 是否已在 GPU 上跑完（不等待；之後仍要 qw_batch_wait 釋放）
bool qw_batch_poll(qw_batch *b);
// 之後 qw_batch_begin 用哪條串流：0（預設）或 1（另一個 command queue 與 event 鏈）。兩條串流的工作互不等待、GPU 可同時執行，
// 所以同時在跑的兩邊不能寫同一個 buffer（也不能一邊寫、一邊讀）；同一條串流裡照送出順序執行
void qw_gpu_set_stream(qw_gpu *g, int s);

// Profiling：之後開始的 batch 每個 dispatch 用獨立 encoder，前後各取一次 GPU 時間戳。
// qw_batch_end 後可用 qw_profile_count / qw_profile_get 取出每個 dispatch 的 kernel 名稱與 GPU 時間（µs）。
void qw_profile_enable(qw_gpu *g, bool on);
int qw_profile_count(const qw_gpu *g);
const char *qw_profile_get(const qw_gpu *g, int i, double *us);

// 一次送出 n 個空的 command buffer 並等待，回傳每個的平均牆鐘時間（秒）。
double qw_gpu_empty_commit_latency(qw_gpu *g, qw_kernel *k, int n);

// 電源與散熱狀態（寫進每份量測報告；電池供電時 GPU 可能降速）
// 例如「電源：電池（58%）、低耗電模式：關、散熱：正常」
void qw_power_desc(char *out, size_t len);
bool qw_on_battery(void);

// 牆鐘時間（秒）
double qw_now(void);
