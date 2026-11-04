// MoE prefill 的路由與合併（搭配 qmm.metal 的 MOE = 1 版本）
//
// 流程（P = T × TOPK 個「token、選中的 expert」配對）：
//   1. moe_count   ：每個 expert 被選了幾次
//   2. moe_plan    ：前綴和 → 每個 expert 在排序後的起點；並產生 qmm 的區塊表 {expert, 起始列, 結束列}
//   3. moe_scatter ：把每個配對的輸入列複製到排序後的位置，記下 pos_of[配對] = 位置
//   4. qmm（gate+up，N = 2·I）→ moe_act：H = SiLU(gate)·up → qmm（down）
//   5. moe_combine ：y[t] = Σ_s w[t,s] · D[pos_of[t,s]]（用反向對照表收集，不需要 atomic）
//
// 主程式注入：E（expert 數）、TOPK、KD（hidden）、I（expert 中間維度）、TM（qmm 的區塊列數）

#include <metal_stdlib>
using namespace metal;

kernel void moe_count(device const uint *ids [[buffer(0)]],
                      device atomic_uint *counts [[buffer(1)]],
                      constant uint &P [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < P) atomic_fetch_add_explicit(&counts[ids[gid]], 1u, memory_order_relaxed);
}

// 一個 threadgroup、E 個執行緒（E ≤ 1024）；tm 是區塊列數（moe_plan 用編譯時的 TM，moe_plan_t 執行時給）
inline void moe_plan_tm(device const uint *counts, device uint *offs, device atomic_uint *cursor, device uint4 *tiles, uint max_tiles, uint tm,
                        uint tid, uint sg, uint lane, threadgroup uint *s_rows, threadgroup uint *s_tiles) {
    const uint c = tid < E ? counts[tid] : 0;
    const uint nt = (c + tm - 1) / tm;
    // 兩個前綴和：列數與區塊數
    uint pr = simd_prefix_exclusive_sum(c), pt = simd_prefix_exclusive_sum(nt);
    if (lane == 31) {
        s_rows[sg] = pr + c;
        s_tiles[sg] = pt + nt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint vr = lane < (E + 31) / 32 ? s_rows[lane] : 0, vt = lane < (E + 31) / 32 ? s_tiles[lane] : 0;
        s_rows[lane] = simd_prefix_exclusive_sum(vr);
        s_tiles[lane] = simd_prefix_exclusive_sum(vt);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = pr + s_rows[sg], tile0 = pt + s_tiles[sg];
    if (tid < E) {
        offs[tid] = row0;
        atomic_store_explicit(&cursor[tid], row0, memory_order_relaxed);
        for (uint t = 0; t < nt; t++) {
            const uint a = row0 + t * tm;
            tiles[tile0 + t] = uint4(tid, a, min(a + tm, row0 + c), 0);
        }
    }
    // 沒用到的區塊標成空的
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == E - 1)
        for (uint t = tile0 + nt; t < max_tiles; t++) tiles[t] = uint4(0, 0, 0, 0);
}

kernel void moe_plan(device const uint *counts [[buffer(0)]],
                     device uint *offs [[buffer(1)]],      // [E] 起點
                     device atomic_uint *cursor [[buffer(2)]], // [E] 分散用的游標（初始化成起點）
                     device uint4 *tiles [[buffer(3)]],    // [max_tiles]
                     constant uint &max_tiles [[buffer(4)]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint s_rows[32], s_tiles[32];
    moe_plan_tm(counts, offs, cursor, tiles, max_tiles, TM, tid, sg, lane, s_rows, s_tiles);
}

kernel void moe_plan_t(device const uint *counts [[buffer(0)]], device uint *offs [[buffer(1)]], device atomic_uint *cursor [[buffer(2)]],
                       device uint4 *tiles [[buffer(3)]], constant uint &max_tiles [[buffer(4)]], constant uint &tm [[buffer(5)]],
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint s_rows[32], s_tiles[32];
    moe_plan_tm(counts, offs, cursor, tiles, max_tiles, tm, tid, sg, lane, s_rows, s_tiles);
}

// grid：P 個 threadgroup，每個 256 個執行緒，複製一列
kernel void moe_scatter(device const half *X [[buffer(0)]],
                        device const uint *ids [[buffer(1)]],
                        device atomic_uint *cursor [[buffer(2)]],
                        device half *Xs [[buffer(3)]],
                        device uint *pos_of [[buffer(4)]],
                        uint p [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint pos;
    if (tid == 0) {
        pos = atomic_fetch_add_explicit(&cursor[ids[p]], 1u, memory_order_relaxed);
        pos_of[p] = pos;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint t = p / TOPK;
    device const half4 *src = (device const half4 *)(X + (ulong)t * KD);
    device half4 *dst = (device half4 *)(Xs + (ulong)pos * KD);
    for (uint i = tid; i < KD / 4; i += 256) dst[i] = src[i];
}

// H[r, j] = SiLU(G[r, j]) · G[r, I + j]；grid 涵蓋 P × I
kernel void moe_act(device const half *G [[buffer(0)]],
                    device half *H [[buffer(1)]],
                    constant uint &P [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    const uint r = gid / I, j = gid % I;
    if (r >= P) return;
    const float gt = float(G[(ulong)r * 2 * I + j]), up = float(G[(ulong)r * 2 * I + I + j]);
    H[gid] = half(gt / (1.0f + exp(-gt)) * up);
}

// y[t, c] = Σ_s w[t,s] · D[pos_of[t,s], c]；grid：(KD / 4, T)
kernel void moe_combine(device const half *D [[buffer(0)]],
                        device const uint *pos_of [[buffer(1)]],
                        device const float *w [[buffer(2)]],
                        device half *Y [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const uint c4 = gid.x, t = gid.y;
    float4 acc = 0;
    for (uint s = 0; s < TOPK; s++) {
        const uint p = t * TOPK + s;
        acc += w[p] * float4(((device const half4 *)(D + (ulong)pos_of[p] * KD))[c4]);
    }
    ((device half4 *)(Y + (ulong)t * KD))[c4] = half4(acc);
}
