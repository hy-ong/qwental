// QSA decode 量測：打分數 → top-512 → 稀疏 attention → 合併
// 對照 CPU 參考實作（double），並量不同 context 長度的時間
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"

static const char qsa_src[] = {
#embed "../kernels/qsa.metal"
};

enum { IH = 4, ID = 128, QH = 24, KVH = 2, HD = 256, SEL = 512, CR = 4, BUDGET = 2048, GQA = QH / KVH };

static qw_gpu *g;
static int NCH = 32; // 掃描過：32 最好
static int V2 = 1; // 1：用 v2 kernel（QSA_V1=1 改回 v1）

static void die(const char *m) {
    fprintf(stderr, "錯誤：%s\n", m);
    exit(1);
}
static uint64_t rng = 0xdeadbeefcafef00dull;
static uint32_t rnd32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}
static float rndf(void) { return (rnd32() >> 8) * (1.0f / 16777216.0f) * 2 - 1; }
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

static int MAXPER = 32;

static qw_kernel *K_(const char *kname) {
    char name[64];
    snprintf(name, sizeof name, "%s_q%d", kname, MAXPER);
    qw_kernel *k = qw_kernel_get(g, name);
    if (k) return k;
    char defs[512], err[8192];
    int n = snprintf(defs, sizeof defs, "#define %s %s\n#define IH %d\n#define ID %d\n#define QH %d\n#define KVH %d\n#define HD %d\n#define SEL %d\n#define CR %d\n#define MAXPER %d\n",
                     kname, name, IH, ID, QH, KVH, HD, SEL, CR, MAXPER);
    char *src = malloc(n + sizeof qsa_src);
    memcpy(src, defs, n);
    memcpy(src + n, qsa_src, sizeof qsa_src);
    if (!qw_gpu_compile(g, src, n + sizeof qsa_src, err, sizeof err)) {
        fprintf(stderr, "編譯失敗：\n%s\n", err);
        exit(1);
    }
    free(src);
    return qw_kernel_get(g, name);
}

typedef struct { uint32_t L, pos, nvis, ntok, sparse; } qargs;

static qw_buf bIq, bKb, bQ, bK, bV, bScore, bSel, bNsel, bPart, bOut;

static void alloc_all(int L) {
    int nb = L / CR;
    if (!qw_buf_new(g, IH * ID * 2, &bIq) || !qw_buf_new(g, (size_t)(nb + 1) * ID * 2, &bKb) ||
        !qw_buf_new(g, QH * HD * 2, &bQ) || !qw_buf_new(g, (size_t)L * KVH * HD * 2, &bK) ||
        !qw_buf_new(g, (size_t)L * KVH * HD * 2, &bV) || !qw_buf_new(g, (size_t)(nb + 1) * 4, &bScore) ||
        !qw_buf_new(g, SEL * 4, &bSel) || !qw_buf_new(g, 16, &bNsel) ||
        !qw_buf_new(g, (size_t)256 * QH * (HD + 2) * 4, &bPart) || !qw_buf_new(g, QH * HD * 2, &bOut))
        die("配置失敗");
    uint16_t *p;
    p = bIq.ptr; for (int i = 0; i < IH * ID; i++) p[i] = f2h(rndf() * 0.3f);
    p = bKb.ptr; for (size_t i = 0; i < (size_t)(nb + 1) * ID; i++) p[i] = f2h(rndf() * 0.3f);
    p = bQ.ptr; for (int i = 0; i < QH * HD; i++) p[i] = f2h(rndf());
    p = bK.ptr; for (size_t i = 0; i < (size_t)L * KVH * HD; i++) p[i] = f2h(rndf());
    p = bV.ptr; for (size_t i = 0; i < (size_t)L * KVH * HD; i++) p[i] = f2h(rndf());
}

static void free_all(void) {
    qw_buf *b[] = {&bIq, &bKb, &bQ, &bK, &bV, &bScore, &bSel, &bNsel, &bPart, &bOut};
    for (size_t i = 0; i < sizeof b / sizeof b[0]; i++) qw_buf_free(b[i]);
}

static qargs make_args(int L) {
    qargs a = {(uint32_t)L, (uint32_t)(L - 1), (uint32_t)(L / CR), 0, L > BUDGET};
    if (a.sparse) {
        uint32_t ns = a.nvis < SEL ? a.nvis : SEL;
        a.ntok = ns * CR + (a.pos + 1 - a.nvis * CR);
    } else
        a.ntok = (uint32_t)L;
    return a;
}

// stage：-1 全部；0 打分數；1 選取；2 attention + 合併
static double run_reps(int L, int stage, int reps);
static double run(int L, int stage) { return run_reps(L, stage, 1); }

// 同一個 command buffer 重複 reps 次再平均：單獨送一個很小的 kernel 時 GPU 會停在低時脈，量不準
static double run_reps(int L, int stage, int reps) {
    qargs a = make_args(L);
    { int need = (int)((a.nvis + 1023) / 1024); MAXPER = need <= 8 ? 8 : need <= 16 ? 16 : need <= 32 ? 32 : 64; }
    uint32_t nch = (uint32_t)NCH;
    qw_batch *b = qw_batch_begin(g, false);
    for (int rep = 0; rep < reps; rep++) {
    if (a.sparse && (stage < 0 || stage == 0)) {
        qw_batch_kernel(b, K_(V2 ? "qsa_score2" : "qsa_score"));
        qw_batch_buf(b, 0, &bIq, 0); qw_batch_buf(b, 1, &bKb, 0); qw_batch_buf(b, 2, &bScore, 0);
        qw_batch_bytes(b, 3, &a, sizeof a);
        if (V2) qw_batch_dispatch(b, (a.nvis + 7) / 8, 1, 1, 256, 1, 1);
        else qw_batch_dispatch(b, (a.nvis + 255) / 256, 1, 1, 256, 1, 1);
    }
    if (a.sparse && (stage < 0 || stage == 1)) {
        qw_batch_kernel(b, K_(V2 ? "qsa_topk2" : "qsa_topk"));
        qw_batch_buf(b, 0, &bScore, 0); qw_batch_buf(b, 1, &bSel, 0); qw_batch_buf(b, 2, &bNsel, 0);
        qw_batch_bytes(b, 3, &a, sizeof a);
        qw_batch_dispatch(b, 1, 1, 1, 1024, 1, 1);
    }
    if (stage < 0 || stage == 2) {
        qw_batch_kernel(b, K_(V2 ? "qsa_attn2" : "qsa_attn"));
        qw_batch_buf(b, 0, &bQ, 0); qw_batch_buf(b, 1, &bK, 0); qw_batch_buf(b, 2, &bV, 0);
        qw_batch_buf(b, 3, &bSel, 0); qw_batch_buf(b, 4, &bNsel, 0); qw_batch_buf(b, 5, &bPart, 0);
        qw_batch_bytes(b, 6, &a, sizeof a);
        qw_batch_dispatch(b, nch, KVH, 1, 32 * GQA, 1, 1);
    }
    if (stage < 0 || stage == 3) {
        qw_batch_kernel(b, K_(V2 ? "qsa_merge2" : "qsa_merge"));
        qw_batch_buf(b, 0, &bPart, 0); qw_batch_buf(b, 1, &bOut, 0);
        qw_batch_bytes(b, 2, &nch, 4);
        qw_batch_dispatch(b, QH, 1, 1, V2 ? 256 : 32, 1, 1);
    }
    }
    double t = qw_batch_end(b);
    if (t < 0) die("GPU 執行失敗");
    return t / reps;
}

// CPU 參考
// macOS 的 qsort_r：比較函式的 context 在第一個參數
static int cmp_desc(void *ctx, const void *x, const void *y) {
    const double *s = ctx;
    double a = s[*(const int *)x], b = s[*(const int *)y];
    return a < b ? 1 : a > b ? -1 : 0;
}

static void check(int L) {
    alloc_all(L);
    run(L, -1);
    qargs a = make_args(L);
    const uint16_t *iq = bIq.ptr, *kb = bKb.ptr, *q = bQ.ptr, *K = bK.ptr, *V = bV.ptr, *out = bOut.ptr;
    int *toks = malloc(sizeof(int) * (L + 8)), nt = 0;
    int sel_match = -1;
    if (a.sparse) {
        double *sc = malloc(sizeof(double) * a.nvis);
        int *idx = malloc(sizeof(int) * a.nvis);
        for (uint32_t n = 0; n < a.nvis; n++) {
            double s = 0;
            for (int h = 0; h < IH; h++) {
                double d = 0;
                for (int i = 0; i < ID; i++) d += (double)h2f(iq[h * ID + i]) * h2f(kb[(size_t)n * ID + i]);
                s += d > 0 ? d : 0;
            }
            sc[n] = s / sqrt((double)ID);
            idx[n] = (int)n;
        }
        qsort_r(idx, a.nvis, sizeof(int), sc, cmp_desc);
        int ns = a.nvis < SEL ? (int)a.nvis : SEL;
        // GPU 選的集合要和 CPU 一樣（不在乎順序）
        char *mark = calloc(a.nvis, 1);
        for (int i = 0; i < ns; i++) mark[idx[i]] = 1;
        const uint32_t *gs = bSel.ptr;
        sel_match = 0;
        for (int i = 0; i < ns; i++) sel_match += mark[gs[i]];
        for (int i = 0; i < ns; i++)
            for (int r = 0; r < CR; r++) toks[nt++] = idx[i] * CR + r;
        for (uint32_t t = a.nvis * CR; t <= a.pos; t++) toks[nt++] = (int)t;
        free(sc); free(idx); free(mark);
    } else
        for (int t = 0; t < L; t++) toks[nt++] = t;
    double worst = 0;
    double *sv = malloc(sizeof(double) * nt);
    for (int h = 0; h < QH; h++) {
        int kvh = h / GQA;
        double mx = -1e300;
        for (int i = 0; i < nt; i++) {
            double d = 0;
            for (int j = 0; j < HD; j++) d += (double)h2f(q[h * HD + j]) * h2f(K[((size_t)toks[i] * KVH + kvh) * HD + j]);
            sv[i] = d / sqrt((double)HD);
            if (sv[i] > mx) mx = sv[i];
        }
        double den = 0;
        for (int i = 0; i < nt; i++) den += exp(sv[i] - mx);
        for (int j = 0; j < HD; j++) {
            double o = 0;
            for (int i = 0; i < nt; i++) o += exp(sv[i] - mx) * h2f(V[((size_t)toks[i] * KVH + kvh) * HD + j]);
            o /= den;
            worst = fmax(worst, fabs(o - h2f(out[h * HD + j])));
        }
    }
    if (a.sparse)
        printf("| %d | 稀疏 | %d / %d | %d | %.1e | %s |\n", L, sel_match, a.nvis < SEL ? a.nvis : SEL, nt, worst,
               (sel_match == (int)(a.nvis < SEL ? a.nvis : SEL) && worst < 2e-3) ? "✅" : "❌");
    else
        printf("| %d | 不稀疏 | — | %d | %.1e | %s |\n", L, nt, worst, worst < 2e-3 ? "✅" : "❌");
    free(sv); free(toks);
    free_all();
}

int main(void) {
    if (getenv("QSA_NCH")) NCH = atoi(getenv("QSA_NCH"));
    if (getenv("QSA_V1") && atoi(getenv("QSA_V1")) == 1) V2 = 0;
    g = qw_gpu_open();
    if (!g) die("找不到 Metal 裝置");
    char pw[256];
    qw_power_desc(pw, sizeof pw);
    printf("# QSA decode 量測\n\n- GPU：%s\n- %s\n- 分段數 NCH = %d、kernel 版本 v%d\n", qw_gpu_name(g), pw, NCH, V2 ? 2 : 1);
    printf("\n## 1. 正確性（對照 CPU double）\n\n| L | 模式 | 選取一致 | 看的 token 數 | 輸出最大誤差 | |\n|---|---|---|---|---|---|\n");
    const int cl[] = {1500, 2049, 3001, 8195, 20000};
    for (int i = 0; i < 5; i++) check(cl[i]);
    printf("\n## 2. 速度（單層、單一 token）\n\n| L | 打分數 | 選取 | attention | 合併 | 合計 | 換算 12 層 |\n|---|---|---|---|---|---|---|\n");
    const int Ls[] = {8192, 32768, 131072, 262144};
    for (int i = 0; i < 4; i++) {
        alloc_all(Ls[i]);
        run(Ls[i], -1);
        double st[4], tot = 1e30;
        for (int s = 0; s < 4; s++) {
            st[s] = 1e30;
            for (int r = 0; r < 3; r++) { double t = run_reps(Ls[i], s, 50); if (t < st[s]) st[s] = t; }
        }
        for (int r = 0; r < 3; r++) { double t = run_reps(Ls[i], -1, 50); if (t < tot) tot = t; }
        printf("| %d | %.1f µs | %.1f µs | %.1f µs | %.1f µs | **%.1f µs** | %.2f ms |\n", Ls[i], st[0] * 1e6, st[1] * 1e6,
               st[2] * 1e6, st[3] * 1e6, tot * 1e6, tot * 12e3);
        fflush(stdout);
        free_all();
    }
    qw_gpu_close(g);
    return 0;
}
