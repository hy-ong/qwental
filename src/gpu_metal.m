// Qwental 的 Metal 實作。整個專案只有這個檔案直接碰 Objective-C / Metal。
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
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
    // profiling
    bool prof;
    id<MTLCounterSampleBuffer> sbuf;
    int nprof;
    const char *prof_name[QW_PROF_MAX];
    double prof_us[QW_PROF_MAX];
};

enum { QW_FLUSH_MAX = 8 };
struct qw_batch {
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    id<MTLCommandBuffer> done[QW_FLUSH_MAX]; // qw_batch_flush 先送出的 command buffer（依序；最後一個 cb 完成時它們都已完成）
    int ndone;
    bool conc;
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
    for (int i = 0; i < g->nkernels; i++) g->kernels[i].pso = nil;
    for (int i = 0; i < g->nlibs; i++) g->libs[i] = nil;
    g->queue = nil;
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

bool qw_buf_new(qw_gpu *g, size_t size, qw_buf *out) {
    id<MTLBuffer> b = [g->dev newBufferWithLength:size options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    if (!b) return false;
    static size_t tot;
    tot += size;
    if (getenv("QW_ALLOC_DBG") && size >= (64u << 20)) fprintf(stderr, "[alloc] %.2f GB（累計 %.2f GB）\n", size / 1e9, tot / 1e9);
    out->handle = (__bridge_retained void *)b;
    out->ptr = b.contents;
    out->size = size;
    return true;
}

bool qw_buf_wrap(qw_gpu *g, void *ptr, size_t len, qw_buf *out) {
    const size_t page = (size_t)getpagesize();
    if ((uintptr_t)ptr % page) return false;
    size_t plen = (len + page - 1) / page * page;
    id<MTLBuffer> b = [g->dev newBufferWithBytesNoCopy:ptr
                                                length:plen
                                               options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                                           deallocator:nil];
    if (!b) return false;
    out->handle = (__bridge_retained void *)b;
    out->ptr = ptr;
    out->size = len;
    return true;
}

void qw_buf_free(qw_buf *b) {
    if (b && b->handle) {
        CFRelease(b->handle); // 對應 qw_buf_new 的 __bridge_retained
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

qw_batch *qw_batch_begin(qw_gpu *g, bool concurrent) {
    qw_batch *b = calloc(1, sizeof *b);
    b->g = g;
    b->cb = [g->queue commandBufferWithUnretainedReferences];
    [b->cb encodeWaitForEvent:g->event value:g->evv];
    if (g->prof) {
        b->prof = true;
        g->nprof = 0;
        return b;
    }
    b->conc = concurrent;
    b->enc = [b->cb computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    return b;
}

void qw_batch_flush(qw_batch *b) {
    if (b->prof || b->ndone >= QW_FLUSH_MAX) return;
    @autoreleasepool {
        qw_gpu *g = b->g;
        [b->enc endEncoding];
        [b->cb encodeSignalEvent:g->event value:++g->evv];
        [b->cb commit];
        b->done[b->ndone++] = b->cb;
        b->cb = [g->queue commandBufferWithUnretainedReferences];
        [b->cb encodeWaitForEvent:g->event value:g->evv];
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
        [b->cb encodeSignalEvent:b->g->event value:++b->g->evv];
        MTLTimestamp c0 = 0, g0 = 0, c1 = 0, g1 = 0;
        if (b->prof) [b->cb.device sampleTimestamps:&c0 gpuTimestamp:&g0];
        [b->cb commit];
        [b->cb waitUntilCompleted];
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
        [b->cb encodeSignalEvent:b->g->event value:++b->g->evv];
        [b->cb commit];
    }
}

double qw_batch_wait(qw_batch *b) {
    double t = -1;
    @autoreleasepool {
        [b->cb waitUntilCompleted];
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
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    return (qw_now() - t0) / n;
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
