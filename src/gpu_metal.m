// Qwental 的 Metal 實作。整個專案只有這個檔案直接碰 Objective-C / Metal。
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/sysctl.h>
#include "gpu.h"

#define QW_MAX_LIBS 1024 // GGUF profile 每部分依格式 / token 數各編一份 library
#define QW_MAX_KERNELS 4096 // 並發（--parallel）每種 token 數各編一批，512 不夠

struct qw_kernel {
    char name[128];
    id<MTLComputePipelineState> pso;
};

#define QW_PROF_MAX 2048 // counter sample buffer 最多 32 KB（4096 個樣本）
#define QW_MAX_BIND 32

struct qw_gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> libs[QW_MAX_LIBS];
    int nlibs;
    struct qw_kernel kernels[QW_MAX_KERNELS];
    int nkernels;
    char name[128];
    // 依序執行：每個 command buffer 開頭等前一個的 event 值、結尾發出新值
    // （buffer 不追蹤依賴，沒有這個的話相鄰的 command buffer 可能在 GPU 上重疊）
    id<MTLEvent> event;
    uint64_t evv;
    // 第二條串流（qw_gpu_set_stream(g, 1)）：自己的 queue 與 event 鏈，與串流 0 的工作互不等待，GPU 可以同時執行
    // （並發模式的 mixed step：prefill 一段在串流 0 跑的同時，decode 在串流 1 跑；兩邊寫的 buffer 不重疊由呼叫端保證）
    id<MTLCommandQueue> queue1;
    id<MTLEvent> event1;
    uint64_t evv1;
    int stream; // 之後 qw_batch_begin 用哪條（0 / 1）
    // profiling
    bool prof;
    id<MTLCounterSampleBuffer> sbuf;
    int nprof;
    const char *prof_name[QW_PROF_MAX];
    double prof_us[QW_PROF_MAX];
    // keep-alive（qw_gpu_keepalive_start）
    pthread_t ka_th;
    bool ka_on;
    _Atomic bool ka_stop;
    id<MTLCommandQueue> ka_q;
    id<MTLComputePipelineState> ka_pso;
    id<MTLBuffer> ka_buf;
    double ka_ms, ka_win;
    int ka_lvl_min;
};

enum { QW_FLUSH_MAX = 8 };
struct qw_batch {
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    id<MTLCommandBuffer> done[QW_FLUSH_MAX]; // qw_batch_flush 先送出的 command buffer（依序；最後一個 cb 完成時它們都已完成）
    int ndone;
    bool conc;
    int stream; // 這個 batch 在哪條串流（queue 與 event 鏈）
    bool ka; // 已算進 ka_inflight（第一次 commit 時；qw_batch_end / qw_batch_wait 扣回）
    // profiling：綁定先記下來，dispatch 時才開 encoder
    qw_gpu *g;
    bool prof;
    struct qw_kernel *k;
    id<MTLBuffer> bufs[QW_MAX_BIND];
    size_t offs[QW_MAX_BIND];
    uint8_t bytes[QW_MAX_BIND][256];
    size_t blen[QW_MAX_BIND];
    size_t tglen[QW_MAX_BIND];
    int n;
};

double qw_now(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

// ---- GPU keep-alive（qw_gpu_keepalive_start；想法來自 Strata #6078，MIT）
// macOS 在約 1 秒沒有 command buffer 之後放掉這個 process 的 GPU 配置，下一個 command buffer 要先等它們回來：
// M5 Max、SC117、21 token 的短請求，間隔 ≤ 1 秒 150 ms，間隔 2–4 秒 650 ms。閒置時每 QW_GPU_KEEPALIVE_MS 送一個
// 1 thread 的 kernel，useResource 列出所有登記的 buffer（qw_buf_new / qw_buf_wrap），配置就一直留著。
// 只在沒有工作在跑、距上次工作至少一個週期、且不超過 QW_GPU_KEEPALIVE_WINDOW_S 時送；系統記憶體吃緊時暫停（wired 的記憶體還給系統）。
// 用自己的 command queue：不排在真正的工作前面，也不碰 event 鏈；kernel 只寫自己的 16 bytes，數值完全不受影響。
// 只用 MTLResidencySet（requestResidency、addResidencySet 到兩條 queue）實測沒用：間隔 2–4 秒仍是 470–1090 ms（同 Strata 的結果）
static pthread_mutex_t reg_mu = PTHREAD_MUTEX_INITIALIZER;
static void **reg; // 登記的 MTLBuffer（不 retain；送 keep-alive 時在鎖裡 retain 一份，qw_buf_free 先移除再 release）
static size_t nreg, capreg;
static void reg_add(void *h) {
    pthread_mutex_lock(&reg_mu);
    if (nreg == capreg) {
        size_t c = capreg ? capreg * 2 : 1024;
        void **n = realloc(reg, c * sizeof *n);
        if (!n) { pthread_mutex_unlock(&reg_mu); return; } // 登記不了：只是 keep-alive 少列一個
        reg = n, capreg = c;
    }
    reg[nreg++] = h;
    pthread_mutex_unlock(&reg_mu);
}
static void reg_del(void *h) {
    pthread_mutex_lock(&reg_mu);
    for (size_t i = nreg; i-- > 0;)
        if (reg[i] == h) { reg[i] = reg[--nreg]; break; }
    pthread_mutex_unlock(&reg_mu);
}
static _Atomic int ka_inflight;      // 已 commit、還沒等完的 batch 數
static _Atomic uint64_t ka_last;     // 上次 commit 或完成的 mach_absolute_time
static void ka_touch(void) { atomic_store_explicit(&ka_last, mach_absolute_time(), memory_order_relaxed); }
static void ka_begin(void) { atomic_fetch_add_explicit(&ka_inflight, 1, memory_order_relaxed); ka_touch(); }
static void ka_end(void) { atomic_fetch_sub_explicit(&ka_inflight, 1, memory_order_relaxed); ka_touch(); }
// batch 的第一個 command buffer commit 時算一次（qw_batch_flush 先送出的那段也算在跑）
#define KA_COMMIT(b) do { if (!(b)->ka) { (b)->ka = true; ka_begin(); } else ka_touch(); } while (0)

qw_gpu *qw_gpu_open(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return NULL;
        qw_gpu *g = calloc(1, sizeof *g);
        g->dev = dev;
        g->queue = [dev newCommandQueue];
        g->event = [dev newEvent];
        g->evv = 0;
        snprintf(g->name, sizeof g->name, "%s", dev.name.UTF8String);
        return g;
    }
}

void qw_gpu_close(qw_gpu *g) {
    if (!g) return;
    if (g->ka_on) {
        atomic_store(&g->ka_stop, true);
        pthread_join(g->ka_th, NULL);
        g->ka_on = false;
        g->ka_q = nil;
        g->ka_pso = nil;
        g->ka_buf = nil;
    }
    for (int i = 0; i < g->nkernels; i++) g->kernels[i].pso = nil;
    for (int i = 0; i < g->nlibs; i++) g->libs[i] = nil;
    g->queue = nil;
    g->queue1 = nil;
    g->event1 = nil;
    g->dev = nil;
    free(g);
}

const char *qw_gpu_name(const qw_gpu *g) { return g->name; }
size_t qw_gpu_max_working_set(const qw_gpu *g) { return (size_t)g->dev.recommendedMaxWorkingSetSize; }

bool qw_gpu_compile(qw_gpu *g, const char *src, size_t len, char *err, size_t errlen) {
    @autoreleasepool {
        if (g->nlibs >= QW_MAX_LIBS) {
            snprintf(err, errlen, "too many libraries");
            return false;
        }
        NSString *s = [[NSString alloc] initWithBytes:src length:len encoding:NSUTF8StringEncoding];
        MTLCompileOptions *opt = [MTLCompileOptions new];
        opt.languageVersion = MTLLanguageVersion4_1;
        opt.mathMode = MTLMathModeFast;
        NSError *e = nil;
        id<MTLLibrary> lib = [g->dev newLibraryWithSource:s options:opt error:&e];
        if (!lib) {
            snprintf(err, errlen, "%s", e.localizedDescription.UTF8String);
            return false;
        }
        g->libs[g->nlibs++] = lib;
        if (e && err && errlen) snprintf(err, errlen, "%s", e.localizedDescription.UTF8String); // 警告
        else if (err && errlen) err[0] = 0;
        return true;
    }
}

qw_kernel *qw_kernel_get(qw_gpu *g, const char *name) {
    for (int i = 0; i < g->nkernels; i++)
        if (strcmp(g->kernels[i].name, name) == 0) return &g->kernels[i];
    if (g->nkernels >= QW_MAX_KERNELS) return NULL;
    @autoreleasepool {
        NSString *n = [NSString stringWithUTF8String:name];
        for (int i = g->nlibs - 1; i >= 0; i--) {
            id<MTLFunction> f = [g->libs[i] newFunctionWithName:n];
            if (!f) continue;
            NSError *e = nil;
            id<MTLComputePipelineState> pso = [g->dev newComputePipelineStateWithFunction:f error:&e];
            if (!pso) {
                fprintf(stderr, "pipeline %s: %s\n", name, e.localizedDescription.UTF8String);
                return NULL;
            }
            struct qw_kernel *k = &g->kernels[g->nkernels++];
            snprintf(k->name, sizeof k->name, "%s", name);
            k->pso = pso;
            return k;
        }
    }
    return NULL;
}

unsigned qw_kernel_max_threads(const qw_kernel *k) { return (unsigned)k->pso.maxTotalThreadsPerThreadgroup; }
unsigned qw_kernel_simd_width(const qw_kernel *k) { return (unsigned)k->pso.threadExecutionWidth; }

extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *);
void *qw_pool_push(void) { return objc_autoreleasePoolPush(); }
void qw_pool_pop(void *pool) { if (pool) objc_autoreleasePoolPop(pool); }

bool qw_buf_new(qw_gpu *g, size_t size, qw_buf *out) {
    id<MTLBuffer> b;
    @autoreleasepool {
        b = [g->dev newBufferWithLength:size options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    }
    if (!b) return false;
    static size_t tot;
    tot += size;
    if (getenv("QW_ALLOC_DBG") && size >= (64u << 20)) fprintf(stderr, "[alloc] %.2f GB（累計 %.2f GB）\n", size / 1e9, tot / 1e9);
    out->handle = (__bridge_retained void *)b;
    reg_add(out->handle);
    out->ptr = b.contents;
    out->size = size;
    return true;
}

bool qw_buf_wrap(qw_gpu *g, void *ptr, size_t len, qw_buf *out) {
    const size_t page = (size_t)getpagesize();
    if ((uintptr_t)ptr % page) return false;
    size_t plen = (len + page - 1) / page * page;
    id<MTLBuffer> b;
    @autoreleasepool {
        b = [g->dev newBufferWithBytesNoCopy:ptr
                                      length:plen
                                     options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                                 deallocator:nil];
    }
    if (!b) return false;
    out->handle = (__bridge_retained void *)b;
    reg_add(out->handle);
    out->ptr = ptr;
    out->size = len;
    return true;
}

void qw_buf_free(qw_buf *b) {
    if (b && b->handle) {
        reg_del(b->handle);
        @autoreleasepool {
            CFRelease(b->handle); // 對應 qw_buf_new 的 __bridge_retained
        }
        b->handle = NULL;
        b->ptr = NULL;
        b->size = 0;
    }
}

void qw_profile_enable(qw_gpu *g, bool on) {
    g->prof = on;
    if (on && !g->sbuf) {
        id<MTLCounterSet> ts = nil;
        for (id<MTLCounterSet> cs in g->dev.counterSets)
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = cs;
        if (!ts || ![g->dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            fprintf(stderr, "這個裝置不支援 GPU 時間戳取樣\n");
            g->prof = false;
            return;
        }
        MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = ts;
        d.storageMode = MTLStorageModeShared;
        d.sampleCount = 2 * QW_PROF_MAX;
        NSError *e = nil;
        g->sbuf = [g->dev newCounterSampleBufferWithDescriptor:d error:&e];
        if (!g->sbuf) {
            fprintf(stderr, "建立 counter sample buffer 失敗：%s\n", e.localizedDescription.UTF8String);
            g->prof = false;
        }
    }
}

int qw_profile_count(const qw_gpu *g) { return g->nprof; }
const char *qw_profile_get(const qw_gpu *g, int i, double *us) {
    *us = g->prof_us[i];
    return g->prof_name[i];
}

static id<MTLCommandQueue> st_queue(qw_gpu *g, int s) { return s ? g->queue1 : g->queue; }
static id<MTLEvent> st_event(qw_gpu *g, int s) { return s ? g->event1 : g->event; }
static uint64_t *st_evv(qw_gpu *g, int s) { return s ? &g->evv1 : &g->evv; }

void qw_gpu_set_stream(qw_gpu *g, int s) {
    if (s && !g->queue1) {
        g->queue1 = [g->dev newCommandQueue];
        g->event1 = [g->dev newEvent];
        g->evv1 = 0;
    }
    g->stream = s && g->queue1 ? 1 : 0;
}

qw_batch *qw_batch_begin(qw_gpu *g, bool concurrent) {
    qw_batch *b = calloc(1, sizeof *b);
    b->g = g;
    b->stream = g->prof ? 0 : g->stream; // profiling 只用串流 0
    @autoreleasepool { // cb / enc 是 strong 欄位，pool 清掉之後仍在；不包的話 autorelease 的那份要等執行緒結束才放
        b->cb = [st_queue(g, b->stream) commandBufferWithUnretainedReferences];
        [b->cb encodeWaitForEvent:st_event(g, b->stream) value:*st_evv(g, b->stream)];
        if (g->prof) {
            b->prof = true;
            g->nprof = 0;
            return b;
        }
        b->conc = concurrent;
        b->enc = [b->cb computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    }
    return b;
}

void qw_batch_flush(qw_batch *b) {
    if (b->prof || b->ndone >= QW_FLUSH_MAX) return;
    @autoreleasepool {
        qw_gpu *g = b->g;
        [b->enc endEncoding];
        [b->cb encodeSignalEvent:st_event(g, b->stream) value:++*st_evv(g, b->stream)];
        KA_COMMIT(b);
        [b->cb commit];
        b->done[b->ndone++] = b->cb;
        b->cb = [st_queue(g, b->stream) commandBufferWithUnretainedReferences];
        [b->cb encodeWaitForEvent:st_event(g, b->stream) value:*st_evv(g, b->stream)];
        b->enc = [b->cb computeCommandEncoderWithDispatchType:b->conc ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    }
}

// 先送出的 command buffer：GPU 時間加總、檢查狀態，然後釋放
static double batch_done_time(qw_batch *b) {
    double t = 0;
    for (int i = 0; i < b->ndone; i++) {
        id<MTLCommandBuffer> c = b->done[i];
        [c waitUntilCompleted];
        if (c.status == MTLCommandBufferStatusCompleted) t += c.GPUEndTime - c.GPUStartTime;
        else {
            fprintf(stderr, "command buffer error: %s\n", c.error.localizedDescription.UTF8String);
            t = -1e30;
        }
        b->done[i] = nil;
    }
    b->ndone = 0;
    return t;
}

void qw_batch_kernel(qw_batch *b, qw_kernel *k) {
    if (b->prof) { b->k = k; return; }
    [b->enc setComputePipelineState:k->pso];
}

void qw_batch_buf(qw_batch *b, unsigned index, const qw_buf *buf, size_t offset) {
    if (b->prof) {
        b->bufs[index] = (__bridge id<MTLBuffer>)buf->handle;
        b->offs[index] = offset;
        b->blen[index] = 0;
        return;
    }
    [b->enc setBuffer:(__bridge id<MTLBuffer>)buf->handle offset:offset atIndex:index];
}

void qw_batch_bytes(qw_batch *b, unsigned index, const void *data, size_t len) {
    if (b->prof) {
        memcpy(b->bytes[index], data, len);
        b->blen[index] = len;
        b->bufs[index] = nil;
        return;
    }
    [b->enc setBytes:data length:len atIndex:index];
}

void qw_batch_threadgroup_mem(qw_batch *b, unsigned index, size_t len) {
    if (b->prof) { b->tglen[index] = len; return; }
    [b->enc setThreadgroupMemoryLength:len atIndex:index];
}

void qw_batch_dispatch(qw_batch *b, unsigned gx, unsigned gy, unsigned gz,
                       unsigned tx, unsigned ty, unsigned tz) {
    if (b->prof) {
        qw_gpu *g = b->g;
        if (g->nprof >= QW_PROF_MAX) return;
        MTLComputePassDescriptor *pd = [MTLComputePassDescriptor computePassDescriptor];
        pd.dispatchType = MTLDispatchTypeSerial;
        pd.sampleBufferAttachments[0].sampleBuffer = g->sbuf;
        pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * g->nprof;
        pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * g->nprof + 1;
        id<MTLComputeCommandEncoder> enc = [b->cb computeCommandEncoderWithDescriptor:pd];
        [enc setComputePipelineState:b->k->pso];
        for (int i = 0; i < QW_MAX_BIND; i++) {
            if (b->bufs[i]) [enc setBuffer:b->bufs[i] offset:b->offs[i] atIndex:i];
            else if (b->blen[i]) [enc setBytes:b->bytes[i] length:b->blen[i] atIndex:i];
            if (b->tglen[i]) [enc setThreadgroupMemoryLength:b->tglen[i] atIndex:i];
        }
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
        [enc endEncoding];
        g->prof_name[g->nprof++] = b->k->name;
        return;
    }
    [b->enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
}

void qw_batch_barrier(qw_batch *b) {
    if (b->prof) return; // 每個 dispatch 都是獨立 encoder，本來就依序
    [b->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

double qw_batch_end(qw_batch *b) {
    double t = -1;
    @autoreleasepool {
        if (!b->prof) [b->enc endEncoding];
        [b->cb encodeSignalEvent:st_event(b->g, b->stream) value:++*st_evv(b->g, b->stream)];
        MTLTimestamp c0 = 0, g0 = 0, c1 = 0, g1 = 0;
        if (b->prof) [b->cb.device sampleTimestamps:&c0 gpuTimestamp:&g0];
        KA_COMMIT(b);
        [b->cb commit];
        [b->cb waitUntilCompleted];
        ka_end();
        if (b->cb.status == MTLCommandBufferStatusCompleted)
            t = b->cb.GPUEndTime - b->cb.GPUStartTime;
        else
            fprintf(stderr, "command buffer error: %s\n", b->cb.error.localizedDescription.UTF8String);
        if (b->ndone) {
            const double td = batch_done_time(b);
            t = t >= 0 && td >= 0 ? t + td : -1;
        }
        if (t >= 0 && b->prof && b->g->nprof > 0) {
            // GPU 時間戳換算成 ns：用前後兩次 CPU/GPU 對照
            [b->cb.device sampleTimestamps:&c1 gpuTimestamp:&g1];
            const double ns_per_tick = (g1 > g0) ? (double)(c1 - c0) / (double)(g1 - g0) : 1.0;
            NSData *d = [b->g->sbuf resolveCounterRange:NSMakeRange(0, 2 * b->g->nprof)];
            const MTLCounterResultTimestamp *ts = d.bytes;
            for (int i = 0; i < b->g->nprof; i++)
                b->g->prof_us[i] = (double)(ts[2 * i + 1].timestamp - ts[2 * i].timestamp) * ns_per_tick / 1e3;
        }
        b->enc = nil;
        b->cb = nil;
    }
    free(b);
    return t;
}

void qw_batch_submit(qw_batch *b) {
    @autoreleasepool {
        [b->enc endEncoding];
        b->enc = nil;
        [b->cb encodeSignalEvent:st_event(b->g, b->stream) value:++*st_evv(b->g, b->stream)];
        KA_COMMIT(b); // 對應 qw_batch_wait 的 ka_end
        [b->cb commit];
    }
}

bool qw_batch_poll(qw_batch *b) {
    const MTLCommandBufferStatus st = b->cb.status; // 最後一個 command buffer 完成時，先送出的（flush）都已完成（event 鏈）
    return st == MTLCommandBufferStatusCompleted || st == MTLCommandBufferStatusError;
}

double qw_batch_wait(qw_batch *b) {
    double t = -1;
    @autoreleasepool {
        [b->cb waitUntilCompleted];
        ka_end();
        if (b->cb.status == MTLCommandBufferStatusCompleted) t = b->cb.GPUEndTime - b->cb.GPUStartTime;
        else fprintf(stderr, "command buffer error: %s\n", b->cb.error.localizedDescription.UTF8String);
        if (b->ndone) {
            const double td = batch_done_time(b);
            t = t >= 0 && td >= 0 ? t + td : -1;
        }
        b->cb = nil;
    }
    free(b);
    return t;
}

double qw_gpu_empty_commit_latency(qw_gpu *g, qw_kernel *k, int n) {
    double t0 = qw_now();
    for (int i = 0; i < n; i++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g->queue commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:k->pso];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            ka_begin();
            [cb commit];
            [cb waitUntilCompleted];
            ka_end();
        }
    }
    return (qw_now() - t0) / n;
}

// 系統記憶體吃緊：VM pressure 高於 normal，或可用記憶體 %（kern.memorystatus_level，同 serve 的 mem_guard）低於下限
static bool ka_pressure(int lvl_min) {
    int v = 0;
    size_t len = sizeof v;
    if (!sysctlbyname("kern.memorystatus_vm_pressure_level", &v, &len, NULL, 0) && v > 1) return true;
    len = sizeof v;
    return !sysctlbyname("kern.memorystatus_level", &v, &len, NULL, 0) && v >= 0 && v < lvl_min;
}

static void *ka_loop(void *arg) {
    qw_gpu *g = arg;
    static mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    const useconds_t us = (useconds_t)(g->ka_ms * 1e3);
    const bool dbg = getenv("QW_GPU_KEEPALIVE_DBG") != NULL;
    bool live = false; // 目前在送（除錯 log 只印狀態變化）
    long ticks = 0;
    uint64_t held = 0; // 因為記憶體吃緊停下時的 ka_last：直到它變了（有新的工作）才再送
    while (!atomic_load(&g->ka_stop)) {
        usleep(us);
        if (atomic_load(&g->ka_stop)) break;
        const uint64_t last = atomic_load_explicit(&ka_last, memory_order_relaxed);
        const double idle = (double)(mach_absolute_time() - last) * tb.numer / tb.denom * 1e-6;
        const bool busy = atomic_load_explicit(&ka_inflight, memory_order_relaxed) > 0;
        bool go = !busy && idle >= g->ka_ms && idle <= g->ka_win * 1e3 && last != held;
        // 記憶體吃緊：這段閒置就不再送（直到下一次真正的工作）。不這樣的話放掉之後可用記憶體回升、又開始送、又 wire 回來，來回擺盪
        const bool pres = go && ka_pressure(g->ka_lvl_min);
        if (pres) {
            if (dbg && live) fprintf(stderr, "[keepalive] 系統記憶體吃緊：停到下一次工作（已送 %ld 次）\n", ticks);
            else if (dbg) fprintf(stderr, "[keepalive] 系統記憶體吃緊：這段閒置不送\n");
            held = last, go = false, live = false;
        }
        if (dbg && go != live) fprintf(stderr, "[keepalive] %s（閒置 %.1f 秒%s，已送 %ld 次）\n", go ? "開始" : "停止", idle / 1e3,
                                       busy ? "，有工作在跑" : "", ticks);
        live = go;
        if (!go) continue;
        @autoreleasepool { // 每一輪自己的 pool：command buffer、encoder、retain 的 buffer 清單這一輪就放掉
            NSMutableArray *bufs = [NSMutableArray new];
            pthread_mutex_lock(&reg_mu); // 在鎖裡 retain：qw_buf_free 先移除再 release，這一輪送完前 buffer 都還在
            for (size_t i = 0; i < nreg; i++) [bufs addObject:(__bridge id)reg[i]];
            pthread_mutex_unlock(&reg_mu);
            id<MTLCommandBuffer> cb = [g->ka_q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:g->ka_pso];
            [en setBuffer:g->ka_buf offset:0 atIndex:0];
            for (id<MTLBuffer> x in bufs) [en useResource:x usage:MTLResourceUsageRead];
            [en dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [en endEncoding];
            [cb commit]; // 不算真正的工作：ka_last 不動
            [cb waitUntilCompleted];
            ticks++;
        }
    }
    return NULL;
}

int qw_gpu_keepalive_start(qw_gpu *g) {
    if (g->ka_on) return 1;
    const char *e = getenv("QW_GPU_KEEPALIVE_MS");
    g->ka_ms = e && *e ? atof(e) : 250;
    e = getenv("QW_GPU_KEEPALIVE_WINDOW_S");
    g->ka_win = e && *e ? atof(e) : 120;
    e = getenv("QW_MEM_LEVEL_MIN");
    g->ka_lvl_min = e && *e ? atoi(e) : 20;
    if (!(g->ka_ms > 0) || !(g->ka_win > 0)) return 0;
    if (g->ka_ms < 10) g->ka_ms = 10; // 再短沒有意義，只是忙等
    @autoreleasepool {
        NSError *err = nil;
        id<MTLLibrary> l = [g->dev newLibraryWithSource:@"kernel void qw_keepalive(device uint *b [[buffer(0)]]) { b[0] += 1u; }" options:nil error:&err];
        id<MTLFunction> f = l ? [l newFunctionWithName:@"qw_keepalive"] : nil;
        g->ka_pso = f ? [g->dev newComputePipelineStateWithFunction:f error:&err] : nil;
        g->ka_buf = [g->dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        g->ka_q = [g->dev newCommandQueue];
    }
    if (!g->ka_pso || !g->ka_buf || !g->ka_q) {
        fprintf(stderr, "GPU keep-alive 的 kernel 建立失敗：關閉\n");
        return 0;
    }
    ka_touch();
    atomic_store(&g->ka_stop, false);
    if (pthread_create(&g->ka_th, NULL, ka_loop, g)) return 0;
    g->ka_on = true;
    return 1;
}

bool qw_on_battery(void) {
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    if (!info) return false;
    CFStringRef type = IOPSGetProvidingPowerSourceType(info);
    bool batt = type && CFStringCompare(type, CFSTR(kIOPMBatteryPowerKey), 0) == kCFCompareEqualTo;
    CFRelease(info);
    return batt;
}

void qw_power_desc(char *out, size_t len) {
    @autoreleasepool {
        int pct = -1;
        CFTypeRef info = IOPSCopyPowerSourcesInfo();
        if (info) {
            CFArrayRef list = IOPSCopyPowerSourcesList(info);
            if (list) {
                for (CFIndex i = 0; i < CFArrayGetCount(list); i++) {
                    CFDictionaryRef d = IOPSGetPowerSourceDescription(info, CFArrayGetValueAtIndex(list, i));
                    CFNumberRef cap = d ? CFDictionaryGetValue(d, CFSTR(kIOPSCurrentCapacityKey)) : NULL;
                    if (cap) CFNumberGetValue(cap, kCFNumberIntType, &pct);
                }
                CFRelease(list);
            }
            CFRelease(info);
        }
        NSProcessInfo *pi = NSProcessInfo.processInfo;
        const char *therm[] = {"正常", "偏熱", "嚴重", "臨界"};
        NSInteger ts = pi.thermalState;
        snprintf(out, len, "電源：%s（電量 %d%%）、低耗電模式：%s、散熱狀態：%s",
                 qw_on_battery() ? "電池" : "插電", pct, pi.lowPowerModeEnabled ? "開" : "關",
                 (ts >= 0 && ts < 4) ? therm[ts] : "未知");
    }
}
