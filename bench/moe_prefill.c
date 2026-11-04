// MoE prefill 量測：路由（計數、規劃、分散）→ gate+up qmm → SiLU·up → down qmm → 合併
// 形狀照 Qwen3.8-Flash-Next：E = 512、TOPK = 10、hidden 2560、expert 中間維度 640、4-bit
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char qmm_src[] = {
#embed "../kernels/qmm.metal"
};
static const char moe_src[] = {
#embed "../kernels/moe_prefill.metal"
};

enum { E = 512, TOPK = 10, HID = 2560, INTER = 640, BITS = 4 };

static qw_gpu *g;
static int TMV = 64;

static void die(const char *m) {
    fprintf(stderr, "錯誤：%s\n", m);
    exit(1);
}

static uint64_t rng = 0x123456789abcdefull;
static uint32_t rnd32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}
static float rndf(void) { return (rnd32() >> 8) * (1.0f / 16777216.0f) * 2 - 1; }
static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static uint16_t f2h(float f) {
    _Float16 h = (_Float16)f;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
static float h2f(uint16_t u) {
    _Float16 h;
    memcpy(&h, &u, 2);
    return (float)h;
}

static qw_kernel *compile_named(const char *src, size_t len, const char *defs, const char *name) {
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    size_t dl = strlen(defs);
    char *buf = malloc(dl + len), err[8192];
    memcpy(buf, defs, dl);
    memcpy(buf + dl, src, len);
    if (!qw_gpu_compile(g, buf, dl + len, err, sizeof err)) {
        fprintf(stderr, "編譯 %s 失敗：\n%s\n", name, err);
        exit(1);
    }
    free(buf);
    k = qw_kernel_get(g, name);
    if (!k) die(name);
    return k;
}

typedef struct { int tn, sg; } qcfg;

static qw_kernel *qmm_moe(int K, qcfg c) {
    char name[64], defs[512];
    snprintf(name, sizeof name, "qmm_moe_k%d_%d_%d_%d", K, TMV, c.tn, c.sg);
    snprintf(defs, sizeof defs,
             "#define qmm %s\n#define BITS %d\n#define KD %d\n#define TM %d\n#define TN %d\n#define SG %d\n#define KC 64\n#define MOE 1\n",
             name, BITS, K, TMV, c.tn, c.sg);
    return compile_named(qmm_src, sizeof qmm_src, defs, name);
}

static qw_kernel *moe_k(const char *kname) {
    char name[64], defs[512];
    snprintf(name, sizeof name, "%s_tm%d", kname, TMV);
    snprintf(defs, sizeof defs, "#define %s %s\n#define E %d\n#define TOPK %d\n#define KD %d\n#define I %d\n#define TM %d\n", kname,
             name, E, TOPK, HID, INTER, TMV);
    return compile_named(moe_src, sizeof moe_src, defs, name);
}

// 權重
typedef struct { qw_buf W, S, Z; int N, K; } qw4;

static void q4_new(qw4 *q, int N, int K) {
    q->N = N;
    q->K = K;
    size_t w = (size_t)E * N * K / 2, s = (size_t)E * N * (K / 64) * 2;
    if (!qw_buf_new(g, w, &q->W) || !qw_buf_new(g, s, &q->S) || !qw_buf_new(g, s, &q->Z)) die("配置權重失敗");
    uint32_t *pw = q->W.ptr;
    for (size_t i = 0; i < w / 4; i++) pw[i] = rnd32();
    uint16_t *ps = q->S.ptr, *pz = q->Z.ptr;
    for (size_t i = 0; i < s / 2; i++) {
        float sc = (0.5f + 0.25f * (rndf() + 1)) * 0.04f / 15;
        ps[i] = f2bf(sc);
        pz[i] = f2bf(-sc * 7.5f);
    }
}

static float q4_w(const qw4 *q, int e, int n, int k) {
    const uint32_t *w = q->W.ptr;
    const uint16_t *s = q->S.ptr, *z = q->Z.ptr;
    size_t row = (size_t)e * q->N + n;
    uint32_t word = w[row * (q->K / 8) + k / 8];
    int v = (int)((word >> (4 * (k % 8))) & 15);
    return v * bf2f(s[row * (q->K / 64) + k / 64]) + bf2f(z[row * (q->K / 64) + k / 64]);
}

// 執行期緩衝區
static qw_buf bX, bIds, bW8, bCnt, bOffs, bCur, bTiles, bXs, bPos, bG, bH, bD, bY;
static uint32_t max_tiles;

static void alloc_rt(int T) {
    size_t P = (size_t)T * TOPK;
    max_tiles = (uint32_t)((P + TMV - 1) / TMV + E);
    if (!qw_buf_new(g, (size_t)T * HID * 2, &bX) || !qw_buf_new(g, P * 4, &bIds) || !qw_buf_new(g, P * 4, &bW8) ||
        !qw_buf_new(g, E * 4, &bCnt) || !qw_buf_new(g, E * 4, &bOffs) || !qw_buf_new(g, E * 4, &bCur) ||
        !qw_buf_new(g, (size_t)max_tiles * 16, &bTiles) || !qw_buf_new(g, P * HID * 2, &bXs) ||
        !qw_buf_new(g, P * 4, &bPos) || !qw_buf_new(g, P * 2 * INTER * 2, &bG) || !qw_buf_new(g, P * INTER * 2, &bH) ||
        !qw_buf_new(g, P * HID * 2, &bD) || !qw_buf_new(g, (size_t)T * HID * 2, &bY))
        die("配置執行期緩衝區失敗");
    uint16_t *x = bX.ptr;
    for (size_t i = 0; i < (size_t)T * HID; i++) x[i] = f2h(rndf());
    uint32_t *ids = bIds.ptr;
    float *w = bW8.ptr;
    for (int t = 0; t < T; t++) {
        float sum = 0;
        for (int s = 0; s < TOPK; s++) {
            uint32_t v;
            bool dup;
            do {
                v = rnd32() % E;
                dup = false;
                for (int j = 0; j < s; j++) dup |= ids[t * TOPK + j] == v;
            } while (dup);
            ids[t * TOPK + s] = v;
            w[t * TOPK + s] = 0.05f + (rnd32() % 1000) / 1000.0f;
            sum += w[t * TOPK + s];
        }
        for (int s = 0; s < TOPK; s++) w[t * TOPK + s] /= sum;
    }
}

static void free_rt(void) {
    qw_buf *bs[] = {&bX, &bIds, &bW8, &bCnt, &bOffs, &bCur, &bTiles, &bXs, &bPos, &bG, &bH, &bD, &bY};
    for (size_t i = 0; i < sizeof bs / sizeof bs[0]; i++) qw_buf_free(bs[i]);
}

static qw4 Wgu, Wd;
static qcfg CGU = {64, 4}, CDN = {64, 4}; // 掃描過的最佳設定（TM = 64）

// 把整個 MoE 編進 batch；stage >= 0 時只編那一段（分段計時用）
static void encode(qw_batch *b, int T, int stage) {
    uint32_t P = (uint32_t)T * TOPK;
    struct { uint32_t M, N; } agu = {P, 2 * INTER}, adn = {P, HID};
    if (stage < 0 || stage == 0) {
        memset(bCnt.ptr, 0, bCnt.size);
        qw_batch_kernel(b, moe_k("moe_count"));
        qw_batch_buf(b, 0, &bIds, 0);
        qw_batch_buf(b, 1, &bCnt, 0);
        qw_batch_bytes(b, 2, &P, 4);
        qw_batch_dispatch(b, (P + 255) / 256, 1, 1, 256, 1, 1);
        qw_batch_kernel(b, moe_k("moe_plan"));
        qw_batch_buf(b, 0, &bCnt, 0);
        qw_batch_buf(b, 1, &bOffs, 0);
        qw_batch_buf(b, 2, &bCur, 0);
        qw_batch_buf(b, 3, &bTiles, 0);
        qw_batch_bytes(b, 4, &max_tiles, 4);
        qw_batch_dispatch(b, 1, 1, 1, E, 1, 1);
        qw_batch_kernel(b, moe_k("moe_scatter"));
        qw_batch_buf(b, 0, &bX, 0);
        qw_batch_buf(b, 1, &bIds, 0);
        qw_batch_buf(b, 2, &bCur, 0);
        qw_batch_buf(b, 3, &bXs, 0);
        qw_batch_buf(b, 4, &bPos, 0);
        qw_batch_dispatch(b, P, 1, 1, 256, 1, 1);
    }
    if (stage < 0 || stage == 1) {
        qw_batch_kernel(b, qmm_moe(HID, CGU));
        qw_batch_buf(b, 0, &bXs, 0);
        qw_batch_buf(b, 1, &Wgu.W, 0);
        qw_batch_buf(b, 2, &Wgu.S, 0);
        qw_batch_buf(b, 3, &Wgu.Z, 0);
        qw_batch_buf(b, 4, &bG, 0);
        qw_batch_bytes(b, 5, &agu, sizeof agu);
        qw_batch_buf(b, 6, &bTiles, 0);
        qw_batch_dispatch(b, 2 * INTER / CGU.tn, max_tiles, 1, 32u * CGU.sg, 1, 1);
    }
    if (stage < 0 || stage == 2) {
        qw_batch_kernel(b, moe_k("moe_act"));
        qw_batch_buf(b, 0, &bG, 0);
        qw_batch_buf(b, 1, &bH, 0);
        qw_batch_bytes(b, 2, &P, 4);
        qw_batch_dispatch(b, (P * INTER + 255) / 256, 1, 1, 256, 1, 1);
    }
    if (stage < 0 || stage == 3) {
        qw_batch_kernel(b, qmm_moe(INTER, CDN));
        qw_batch_buf(b, 0, &bH, 0);
        qw_batch_buf(b, 1, &Wd.W, 0);
        qw_batch_buf(b, 2, &Wd.S, 0);
        qw_batch_buf(b, 3, &Wd.Z, 0);
        qw_batch_buf(b, 4, &bD, 0);
        qw_batch_bytes(b, 5, &adn, sizeof adn);
        qw_batch_buf(b, 6, &bTiles, 0);
        qw_batch_dispatch(b, HID / CDN.tn, max_tiles, 1, 32u * CDN.sg, 1, 1);
    }
    if (stage < 0 || stage == 4) {
        qw_batch_kernel(b, moe_k("moe_combine"));
        qw_batch_buf(b, 0, &bD, 0);
        qw_batch_buf(b, 1, &bPos, 0);
        qw_batch_buf(b, 2, &bW8, 0);
        qw_batch_buf(b, 3, &bY, 0);
        qw_batch_dispatch(b, HID / 4 / 64, T, 1, 64, 1, 1);
    }
}

static double run(int T, int stage) {
    qw_batch *b = qw_batch_begin(g, false);
    encode(b, T, stage);
    double t = qw_batch_end(b);
    if (t < 0) die("GPU 執行失敗");
    return t;
}

static void check(void) {
    const int T = 48;
    alloc_rt(T);
    run(T, -1);
    const uint16_t *x = bX.ptr, *y = bY.ptr;
    const uint32_t *ids = bIds.ptr;
    const float *w = bW8.ptr;
    double worst = 0;
    static float hbuf[INTER];
    for (int t = 0; t < T; t += 7) {
        static double acc[HID];
        memset(acc, 0, sizeof acc);
        for (int s = 0; s < TOPK; s++) {
            int e = (int)ids[t * TOPK + s];
            for (int j = 0; j < INTER; j++) {
                double gt = 0, up = 0;
                for (int k = 0; k < HID; k++) {
                    double xv = h2f(x[(size_t)t * HID + k]);
                    gt += q4_w(&Wgu, e, j, k) * xv;
                    up += q4_w(&Wgu, e, INTER + j, k) * xv;
                }
                hbuf[j] = (float)(gt / (1 + exp(-gt)) * up);
            }
            for (int c = 0; c < HID; c++) {
                double d = 0;
                for (int j = 0; j < INTER; j++) d += q4_w(&Wd, e, c, j) * hbuf[j];
                acc[c] += w[t * TOPK + s] * d;
            }
        }
        double mx = 0;
        for (int c = 0; c < HID; c++) mx = fmax(mx, fabs(acc[c]));
        for (int c = 0; c < HID; c++) worst = fmax(worst, fabs(h2f(y[(size_t)t * HID + c]) - acc[c]) / fmax(mx, 1e-6));
    }
    printf("\n## 1. 正確性（T = %d，對照 CPU 逐 token 直接計算）\n\n- 最大誤差（相對於該 token 輸出的最大值）：%.1e → %s\n", T, worst,
           worst < 1e-2 ? "✅ 通過" : "❌ 不通過");
    free_rt();
}

static void speed(void) {
    printf("\n## 2. 速度（TM = %d、gate+up TN×SG = %d×%d、down TN×SG = %d×%d）\n\n", TMV, CGU.tn, CGU.sg, CDN.tn, CDN.sg);
    printf("| T（token 數） | 路由 | gate+up | SiLU·up | down | 合併 | 合計（一個 command buffer） | 有效 TFLOPS |\n|---|---|---|---|---|---|---|---|\n");
    const int Ts[] = {2048, 4096, 16384};
    for (int i = 0; i < 3; i++) {
        int T = Ts[i];
        alloc_rt(T);
        run(T, -1); // 熱身，也讓區塊表就緒
        double st[5];
        for (int s = 0; s < 5; s++) {
            double best = 1e30;
            for (int r = 0; r < 3; r++) {
                double t = run(T, s);
                if (t < best) best = t;
            }
            st[s] = best;
        }
        double tot = 1e30;
        for (int r = 0; r < 3; r++) {
            double t = run(T, -1);
            if (t < tot) tot = t;
        }
        double flop = (double)T * TOPK * (2.0 * HID * 2 * INTER + 2.0 * INTER * HID);
        printf("| %d | %.2f ms | %.2f ms | %.2f ms | %.2f ms | %.2f ms | **%.2f ms** | %.1f |\n", T, st[0] * 1e3, st[1] * 1e3,
               st[2] * 1e3, st[3] * 1e3, st[4] * 1e3, tot * 1e3, flop / tot / 1e12);
        fflush(stdout);
        free_rt();
    }
}

int main(void) {
    if (getenv("MOE_TM")) TMV = atoi(getenv("MOE_TM"));
    if (getenv("MOE_GU")) sscanf(getenv("MOE_GU"), "%d,%d", &CGU.tn, &CGU.sg);
    if (getenv("MOE_DN")) sscanf(getenv("MOE_DN"), "%d,%d", &CDN.tn, &CDN.sg);
    g = qw_gpu_open();
    if (!g) die("找不到 Metal 裝置");
    char pw[256];
    qw_power_desc(pw, sizeof pw);
    printf("# MoE prefill 量測\n\n- GPU：%s\n- %s\n- E=%d、TOPK=%d、hidden=%d、中間維度=%d、%d-bit\n", qw_gpu_name(g), pw, E, TOPK, HID,
           INTER, BITS);
    q4_new(&Wgu, 2 * INTER, HID);
    q4_new(&Wd, HID, INTER);
    check();
    speed();
    qw_gpu_close(g);
    return 0;
}
