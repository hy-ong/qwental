// MoE decode 讀取樣式的純頻寬量測（bench/moe_bw.c）：只讀、不解碼，知道這種樣式的上限
#include <metal_stdlib>
using namespace metal;

// 主程式注入：SL（每個 threadgroup 讀幾個 uint4）、LD（每個執行緒一輪幾個獨立載入）
// chunks[c] = {起點（uint4 單位）, 長度（uint4 單位）}；threadgroup (x, c) 讀第 c 段的第 x 片
kernel void bw_chunks(device const uint4 *src [[buffer(0)]], device const uint2 *chunks [[buffer(1)]], device uint *out [[buffer(2)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint2 nt2 [[threads_per_threadgroup]]) {
    const uint nt = nt2.x;
    const uint2 ch = chunks[tg.y];
    const uint b = tg.x * SL;
    if (b >= ch.y) return;
    const uint e = min(b + SL, ch.y);
    device const uint4 *p = src + ch.x;
    uint4 a[LD];
    for (uint k = 0; k < LD; k++) a[k] = 0;
    uint i = b + tid;
    for (; i + (LD - 1) * nt < e; i += LD * nt)
        for (uint k = 0; k < LD; k++) a[k] ^= p[i + k * nt];
    for (; i < e; i += nt) a[0] ^= p[i];
    uint4 acc = 0;
    for (uint k = 0; k < LD; k++) acc ^= a[k];
    const uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0x9e3779b9u) out[tid] = r;
}

// 現行 moe_gu_gg 的樣式：每個 SIMD group 一列（RB 位元組），lane 以 2 位元組讀取、每個單位 UB 位元組；
// threadgroup (x, c)：第 c 段的第 x·8 … x·8 + 7 列
kernel void bw_rows(device const uchar *src [[buffer(0)]], device const uint2 *chunks [[buffer(1)]], device uint *out [[buffer(2)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint2 ch = chunks[tg.y];
    device const uchar *row = src + (ulong)ch.x * 16 + (ulong)(tg.x * 8 + sg) * RB;
    uint acc = 0;
    for (uint u = lane; u < RB / max(UB, 1); u += 32)
        for (uint j = 0; j < UB; j += 2) acc ^= uint(*(device const ushort *)(row + u * UB + j));
    if (acc == 0x9e3779b9u) out[lane] = acc;
}
