// MoE decode 一層的讀取樣式（IQ3_S 層：router BF16、shared gate/up/down、10 個 expert 的 gate/up/down）的純頻寬上限。
// 不解碼，只讀同樣的位元組：知道「這種樣式、這種 dispatch 切法」最快能多快，再去調真正的 kernel。
// 用法：build/moe_bw（每種設定 48 層 × reps，同一個 command buffer，取 3 次最小）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char src_bw[] = {
#embed "../kernels/moe_bw.metal"
};

static qw_gpu *g;
static qw_buf arena, out, chb;
#define ARENA ((size_t)2 << 30)
#define D 2560
#define I 640
// 位元組（都是 16 的倍數）
#define B_ROUTER ((size_t)513 * D * 2)
#define B_SG ((size_t)I * D / 256 * 144) // Q4_K
#define B_SU ((size_t)I * D / 256 * 136) // IQ4_XS
#define B_SD ((size_t)D * I / 32 * 18)   // IQ4_NL
#define B_GU ((size_t)I * D / 256 * 110) // IQ3_S
#define B_DN ((size_t)D * I / 32 * 18)   // IQ4_NL
#define NL 64                            // 一個 command buffer 幾層（各層的 chunk 位置不同）
#define MAXC 40

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static qw_kernel *kget(const char *base, int sl, int ld, int rb, int ub) {
    char name[96], defs[256], err[4096];
    snprintf(name, sizeof name, "%s_%d_%d_%d_%d", base, sl, ld, rb, ub);
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    int n = snprintf(defs, sizeof defs, "#define SL %d\n#define LD %d\n#define RB %d\n#define UB %d\n#define %s %s\n", sl, ld, rb, ub, base, name);
    char *s = malloc(n + sizeof src_bw);
    memcpy(s, defs, n);
    memcpy(s + n, src_bw, sizeof src_bw);
    if (!qw_gpu_compile(g, s, n + sizeof src_bw, err, sizeof err)) { fprintf(stderr, "%s\n", err); exit(1); }
    free(s);
    return qw_kernel_get(g, name);
}

// 一層的 chunk：c[0..2] router、shared gate、up；c[3] shared down；c[4..23] expert gate/up；c[24..33] expert down
typedef struct { uint32_t off, len; } ch; // uint4 單位
static ch lay[NL][34];
static const size_t LEN[34] = {B_ROUTER, B_SG, B_SU, B_SD, [4 ... 23] = B_GU, [24 ... 33] = B_DN};

static void place(void) {
    for (int l = 0; l < NL; l++) {
        if (getenv("FIXED")) rs = 88172645463325252ull; // 每層同一組位置：量 SLC 命中時的速度（對照單層重複的 kernel bench）
        for (int c = 0; c < 34; c++) {
            const size_t o = (rnd() % ((ARENA - LEN[c]) / 16384)) * 16384;
            lay[l][c] = (ch){(uint32_t)(o / 16), (uint32_t)(LEN[c] / 16)};
        }
    }
}

static double total(int c0, int c1) {
    double s = 0;
    for (int c = c0; c < c1; c++) s += LEN[c];
    return s;
}

// 依序送 groups 個 dispatch（每個 dispatch 讀 [cs[i], cs[i+1]) 的 chunk）；回傳每層 µs
static double run(int ngrp, const int *cs, int sl, int ld, int nt) {
    qw_kernel *k = kget("bw_chunks", sl, ld, 0, 0);
    double best = 1e30;
    for (int it = 0; it < 3; it++) {
        place();
        ch *cp = chb.ptr;
        for (int l = 0; l < NL; l++) memcpy(cp + l * MAXC, lay[l], sizeof lay[l]);
        qw_batch *b = qw_batch_begin(g, false);
        for (int l = 0; l < NL; l++)
            for (int gi = 0; gi < ngrp; gi++) {
                uint32_t mx = 0;
                for (int c = cs[gi]; c < cs[gi + 1]; c++) if (lay[l][c].len > mx) mx = lay[l][c].len;
                qw_batch_kernel(b, k);
                qw_batch_buf(b, 0, &arena, 0);
                qw_batch_buf(b, 1, &chb, (size_t)(l * MAXC + cs[gi]) * 8);
                qw_batch_buf(b, 2, &out, 0);
                qw_batch_dispatch(b, (mx + sl - 1) / sl, cs[gi + 1] - cs[gi], 1, nt, 1, 1);
            }
        const double t = qw_batch_end(b);
        if (t > 0 && t < best) best = t;
    }
    return best / NL * 1e6;
}

// 現行 gate/up 樣式（每個 SIMD group 一列、2 位元組讀取）：只讀 expert gate/up 的 20 段
static double run_rows(int rb, int ub) {
    qw_kernel *k = kget("bw_rows", 1, 1, rb, ub);
    double best = 1e30;
    for (int it = 0; it < 3; it++) {
        place();
        ch *cp = chb.ptr;
        for (int l = 0; l < NL; l++) memcpy(cp + l * MAXC, lay[l], sizeof lay[l]);
        qw_batch *b = qw_batch_begin(g, false);
        for (int l = 0; l < NL; l++) {
            qw_batch_kernel(b, k);
            qw_batch_buf(b, 0, &arena, 0);
            qw_batch_buf(b, 1, &chb, (size_t)(l * MAXC + 4) * 8);
            qw_batch_buf(b, 2, &out, 0);
            qw_batch_dispatch(b, (unsigned)(B_GU / rb / 8), 20, 1, 256, 1, 1);
        }
        const double t = qw_batch_end(b);
        if (t > 0 && t < best) best = t;
    }
    return best / NL * 1e6;
}

int main(void) {
    g = qw_gpu_open();
    if (!g || !qw_buf_new(g, ARENA, &arena) || !qw_buf_new(g, 4096, &out) || !qw_buf_new(g, NL * MAXC * 8, &chb)) { fprintf(stderr, "GPU 初始化失敗\n"); return 1; }
    for (size_t i = 0; i < ARENA / 8; i++) ((uint64_t *)arena.ptr)[i] = rnd();
    char pw[128];
    qw_power_desc(pw, sizeof pw);
    printf("%s\n一層 %.2f MB（router %.2f、shared %.2f、expert gate/up %.2f、down %.2f）\n", pw, total(0, 34) / 1e6, B_ROUTER / 1e6,
           total(1, 4) / 1e6, total(4, 24) / 1e6, total(24, 34) / 1e6);
    static const int one[] = {0, 34}, three[] = {0, 3, 24, 34}, gu[] = {4, 24}, dn[] = {24, 34}, pre[] = {0, 3};
    static const int SL[] = {1024, 2048, 4096, 8192}, LDv[] = {1, 2, 4, 8}, NT[] = {256, 512, 1024};
    printf("| 樣式 | SL（uint4） | LD | 執行緒 | µs / 層 | GB/s |\n|---|---|---|---|---|---|\n");
    if (getenv("SWEEP")) for (size_t a = 0; a < 4; a++)
        for (size_t c = 0; c < 4; c++)
            for (size_t n = 0; n < 3; n++) {
                if ((size_t)SL[a] < (size_t)NT[n] * LDv[c]) continue;
                const double us = run(1, one, SL[a], LDv[c], NT[n]);
                printf("| 一個 dispatch | %d | %d | %d | %.1f | %.0f |\n", SL[a], LDv[c], NT[n], us, total(0, 34) / us / 1e3);
            }
    const int sl = 2048, ld = 4, nt = 256;
    double us = run(3, three, sl, ld, nt);
    printf("| 三個 dispatch（前段／gate-up＋shared down／down） | %d | %d | %d | %.1f | %.0f |\n", sl, ld, nt, us, total(0, 34) / us / 1e3);
    us = run(1, pre, sl, ld, nt);
    printf("| 只有前段（router＋shared gate/up） | %d | %d | %d | %.1f | %.0f |\n", sl, ld, nt, us, total(0, 3) / us / 1e3);
    us = run(1, gu, sl, ld, nt);
    printf("| 只有 expert gate/up | %d | %d | %d | %.1f | %.0f |\n", sl, ld, nt, us, total(4, 24) / us / 1e3);
    us = run(1, dn, sl, ld, nt);
    printf("| 只有 expert down | %d | %d | %d | %.1f | %.0f |\n", sl, ld, nt, us, total(24, 34) / us / 1e3);
    us = run_rows(1100, 8);
    printf("| expert gate/up，SIMD group 一列、2 位元組讀取 | – | – | 256 | %.1f | %.0f |\n", us, total(4, 24) / us / 1e3);
    return 0;
}
