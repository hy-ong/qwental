// 第 0 階段：記憶體頻寬與 kernel 呼叫開銷的量測 kernel
#include <metal_stdlib>
using namespace metal;

// 什麼都不做，用來量 dispatch 的固定開銷
kernel void empty_k() {}

// 純讀取：每個執行緒以 grid stride 讀 uint4，結果 XOR 起來，
// 只有在幾乎不可能的條件下才寫出，讓編譯器不能把讀取刪掉。
kernel void bw_read(device const uint4 *src [[buffer(0)]],
                    device uint *out [[buffer(1)]],
                    constant uint &n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint gsz [[threads_per_grid]]) {
    uint4 acc = 0;
    for (uint i = gid; i < n; i += gsz) acc ^= src[i];
    uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0x9e3779b9u) out[gid & 1023] = r;
}

// 純讀取，每輪 4 個互相獨立的載入，讓記憶體請求更多同時在路上
kernel void bw_read4(device const uint4 *src [[buffer(0)]],
                     device uint *out [[buffer(1)]],
                     constant uint &n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]],
                     uint gsz [[threads_per_grid]]) {
    uint4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint i = gid;
    for (; i + 3 * gsz < n; i += 4 * gsz) {
        a0 ^= src[i];
        a1 ^= src[i + gsz];
        a2 ^= src[i + 2 * gsz];
        a3 ^= src[i + 3 * gsz];
    }
    for (; i < n; i += gsz) a0 ^= src[i];
    uint4 acc = a0 ^ a1 ^ a2 ^ a3;
    uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0x9e3779b9u) out[gid & 1023] = r;
}

// 複製：讀 + 寫
kernel void bw_copy(device const uint4 *src [[buffer(0)]],
                    device uint4 *dst [[buffer(1)]],
                    constant uint &n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint gsz [[threads_per_grid]]) {
    for (uint i = gid; i < n; i += gsz) dst[i] = src[i];
}

// 區塊讀取：每個 threadgroup 負責一段連續記憶體，組內以 uint4 合併讀取。
// 這是 qmv 實際的讀取方式（一個 SIMD group 讀一整列），比 grid stride 更接近真正的上限。
kernel void bw_block(device const uint4 *src [[buffer(0)]],
                     device uint *out [[buffer(1)]],
                     constant uint &n [[buffer(2)]],
                     uint tgid [[threadgroup_position_in_grid]],
                     uint ntg [[threadgroups_per_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint tsz [[threads_per_threadgroup]]) {
    uint per = (n + ntg - 1) / ntg;
    uint begin = tgid * per, end = min(begin + per, n);
    uint4 a0 = 0, a1 = 0;
    uint i = begin + tid;
    for (; i + tsz < end; i += 2 * tsz) {
        a0 ^= src[i];
        a1 ^= src[i + tsz];
    }
    for (; i < end; i += tsz) a0 ^= src[i];
    uint4 acc = a0 ^ a1;
    uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0x9e3779b9u) out[tgid & 1023] = r;
}
