#include <math.h>
// 用 engine 跑一段 token：逐個餵入 prompt，再 greedy 生成。
// 用法：build/run "760,6511,314,9338,369" [生成數量] [dump 目錄]
//   dump 目錄：每個 prompt 位置存一份每層殘差流（layers_t{t}.bin，[48][10240] float），以及最後的 logits
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine.h"
#include "tok.h"

static bool print_cb(void *ctx, int32_t t) {
    char buf[256];
    const int n = qw_tok_decode1(ctx, t, buf);
    fwrite(buf, 1, (size_t)n, stdout);
    fflush(stdout);
    return true;
}

static uint64_t par_hash(const float *l) { uint64_t h = 1469598103934665603ull; const uint32_t *u = (const uint32_t *)l; for (int i = 0; i < 248320; i++) h = (h ^ u[i]) * 1099511628211ull; return h; }
typedef struct { int n; int32_t *o; } par_col;
static bool par_cb(void *c, int32_t t) { par_col *q = c; q->o[q->n++] = t; return true; }

// QW_PAR + QW_PAR_MIX=檔案：mixed step 的檢查。分段 prefill 這個長 prompt（每段 QW_PAR_MIX_CHUNK，預設 4096）的同時，
// QW_PAR 的對話在 engine 的回呼（mix_cb）裡一起 decode；decode 每步的 logits、長 prompt 最後的 logits 都要與單獨跑逐位元相同
enum { PM = 64 };
static struct {
    qw_engine *e;
    int B, N, k, steps;
    int *sid, *lgbad;
    int32_t *last, (*ref)[4096], (*got)[4096];
    uint64_t (*rh)[4096];
    bool skip_ref;
} mxs;
static bool mix_cb(void *c) {
    (void)c;
    if (mxs.k >= mxs.N) return false;
    const float *bl = qw_engine_par_step(mxs.e, mxs.sid, mxs.last, mxs.B, NULL);
    if (!bl) { fprintf(stderr, "mixed step：par_step 失敗\n"); return false; }
    const int k = mxs.k;
    for (int i = 0; i < mxs.B; i++) {
        const float *r = bl + (size_t)i * 248320;
        if (!mxs.skip_ref && mxs.lgbad[i] < 0 && mxs.got[i][k - 1] == mxs.ref[i][k - 1] && par_hash(r) != mxs.rh[i][k]) mxs.lgbad[i] = k;
        int bst = 0;
        for (int v = 1; v < 248320; v++) if (r[v] > r[bst]) bst = v;
        mxs.got[i][k] = mxs.last[i] = bst;
    }
    mxs.k++;
    mxs.steps++;
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "用法：%s token,token,...|@檔案 [生成數量] [dump 目錄]（QW_CTX=最大 context，預設 4096）\n", argv[0]);
        return 2;
    }
    int ngen = argc > 2 ? atoi(argv[2]) : 16;
    const char *dump = argc > 3 ? argv[3] : NULL;
    // token 可以直接給，或用 @檔案（逗號分隔）
    char *text = argv[1];
    if (text[0] == '@') {
        FILE *f = fopen(text + 1, "rb");
        if (!f) { fprintf(stderr, "開不了 %s\n", text + 1); return 1; }
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        text = malloc((size_t)len + 1);
        text[fread(text, 1, (size_t)len, f)] = 0;
        fclose(f);
    }
    const int maxn = 1 << 20;
    int32_t *ids = malloc(sizeof(int32_t) * maxn);
    int n = 0;
    for (char *s = text; *s && n < maxn;) {
        char *e0 = s;
        long v = strtol(s, &s, 10);
        if (s == e0) break;
        ids[n++] = (int32_t)v;
        if (*s == ',') s++;
    }
    const int ctx = getenv("QW_CTX") ? atoi(getenv("QW_CTX")) : (n + ngen + 64 > 4096 ? n + ngen + 64 : 4096);
    char err[1024];
    double t0 = qw_now();
    qw_engine *e = qw_engine_open(getenv("QW_MODEL") ? getenv("QW_MODEL") : "models/Qwen3.8-Flash-Next-mixed-4-8bit", ctx, err, sizeof err);
    if (!e) {
        fprintf(stderr, "❌ %s\n", err);
        return 1;
    }
    printf("載入 %.1f 秒\n", qw_now() - t0);
    static float layers[48 * 10240];
    const float *lg = NULL;
    if (getenv("QW_GEN")) { // 一般生成：取樣參數由環境變數設定，印出解碼後的文字
        qw_tok *tk = qw_tok_open_any(getenv("QW_MODEL") ? getenv("QW_MODEL") : "models/Qwen3.8-Flash-Next-mixed-4-8bit", getenv("QW_GGUF"), err,
                                     sizeof err);
        if (!tk) { fprintf(stderr, "%s\n", err); return 1; }
        const int32_t stops[2] = {248046, 248044};
        qw_gen g = {.temperature = getenv("QW_TEMP") ? (float)atof(getenv("QW_TEMP")) : 0.7f, .top_k = 20, .top_p = 0.95f,
                    .seed = getenv("QW_SEED") ? strtoull(getenv("QW_SEED"), NULL, 10) : 1, .max_tokens = ngen, .stop = stops, .nstop = 2,
                    .spec = !getenv("QW_NO_SPEC")};
        qw_gen_stats st;
        printf("----\n");
        int fin = qw_engine_generate_ex(e, ids, n, &g, print_cb, tk, &st);
        printf("\n----\n結束原因 %d；prompt %d（%.0f ms）；生成 %d 個 token %.0f ms（%.1f tok/s）；%d 輪、接受 %d 個猜測\n", fin, st.prompt,
               st.prefill_ms, st.gen, st.gen_ms, st.gen / st.gen_ms * 1e3, st.rounds, st.accepted);
        qw_tok_close(tk);
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_MTP")) {
        int32_t out[4096];
        int st[2] = {0, 0};
        double tm[3] = {0};
        int got = qw_engine_generate_mtp(e, ids, n, ngen, out, st, tm);
        printf("生成：");
        for (int i = 0; i < got; i++) printf("%d%s", out[i], i + 1 < got ? "," : "\n");
        printf("MTP：%d 輪、猜中 %d 次（%.1f%%），每輪平均產出 %.2f 個 token；每輪 GPU %.2f ms\n", st[0], st[1],
               100.0 * st[1] / (st[0] ? st[0] : 1), (double)(got - 1) / (st[0] ? st[0] : 1), tm[1] / (st[0] ? st[0] : 1));
        printf("生成 %d 個 token：GPU %.0f ms（%.1f tok/s）、牆鐘 %.0f ms（%.1f tok/s）\n", got - 1, tm[1], (got - 1) / tm[1] * 1e3, tm[2],
               (got - 1) / tm[2] * 1e3);
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_PF_CHUNK")) { // 分段 prefill（每段 QW_PF_CHUNK 個 token，接著上一段）與一次 prefill 的最後 logits 比
        const int C = atoi(getenv("QW_PF_CHUNK"));
        double pm = 0;
        qw_engine_reset(e);
        const float *a = qw_engine_prefill(e, ids, n, NULL, &pm);
        if (!a) return 1;
        float *ac = malloc(248320 * 4);
        memcpy(ac, a, 248320 * 4);
        const float *s1 = qw_engine_step(e, 760, NULL);
        float *sc = malloc(248320 * 4);
        memcpy(sc, s1, 248320 * 4);
        qw_engine_reset(e);
        const float *b = NULL;
        for (int i = 0; i < n; i += C) if (!(b = qw_engine_prefill(e, ids + i, n - i < C ? n - i : C, NULL, &pm))) return 1;
        printf("分段 %d：prefill logits %s", C, memcmp(ac, b, 248320 * 4) ? "不同" : "相同");
        const float *s2 = qw_engine_step(e, 760, NULL);
        printf("、下一步 %s\n", memcmp(sc, s2, 248320 * 4) ? "不同" : "相同");
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_PAR_TR")) { // 逐層比：一般 decode 一步 vs 並發（B = 1）一步（prompt 的最後一個 token；QW_TRACE 要設）
        enum { REC = 10240 * 4 + 2560 * 2 };
        double pm = 0;
        qw_engine_reset(e);
        if (!qw_engine_prefill(e, ids, n - 1, NULL, &pm)) return 1;
        qw_engine_trace_buf(e, 0);
        const float *dl = qw_engine_step(e, ids[n - 1], NULL);
        float *dlc = malloc(248320 * 4);
        memcpy(dlc, dl, 248320 * 4);
        uint8_t *a = malloc((size_t)48 * 2 * REC);
        memcpy(a, qw_engine_trace_buf(e, 0), (size_t)48 * 2 * REC);
        setenv("QW_SLOTS", "3", 1);
        const int TB = atoi(getenv("QW_PAR_TR")) > 1 ? atoi(getenv("QW_PAR_TR")) : 1; // QW_PAR_TR=k：同一個 prompt k 份一起，各欄都與 decode 比
        int sidv[8];
        int32_t tkv[8];
        unsigned long long busy = 0;
        for (int q = 0; q < TB; q++) {
            sidv[q] = qw_engine_par_pick(e, ids, n - 1, busy);
            busy |= 1ull << sidv[q];
            if (!qw_engine_par_prefill(e, sidv[q], ids, n - 1, NULL)) return 1;
            tkv[q] = ids[n - 1];
        }
        const float *plg = qw_engine_par_step(e, sidv, tkv, TB, NULL);
        if (!plg) return 1;
        for (int q = 0; q < TB; q++) printf("第 %d 欄 logits：%s\n", q, memcmp(plg + (size_t)q * 248320, dlc, 248320 * 4) ? "不同" : "相同");
        const uint8_t *c0 = qw_engine_trace_buf(e, 0);
        for (int q = 0; q < TB; q++) {
        const uint8_t *c = c0 + (size_t)q * 48 * 2 * REC;
        printf("第 %d 欄：", q);
        for (int l = 0; l < 48; l++)
            for (int pt = 0; pt < 2; pt++) {
                const size_t o = ((size_t)l * 2 + pt) * REC;
                const bool hs = !memcmp(a + o, c + o, 10240 * 4), os = !memcmp(a + o + 10240 * 4, c + o + 10240 * 4, 2560 * 2);
                if (!hs || !os) { printf("第 %d 層%s：h %s、out %s\n", l, pt ? "（MoE 之後）" : "（attention / DeltaNet 之後）", hs ? "同" : "不同", os ? "同" : "不同"); l = 48; break; }
            }
        }
        printf("比完\n");
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_PAR_LG")) { // 一個 prompt：一般 decode（prefill + qw_engine_step）與並發路徑（B = 1）逐步比 logits，印出第一個不同的步
        const int N = ngen;
        float *ref = malloc((size_t)N * 248320 * 4);
        int32_t *rt = malloc(sizeof(int32_t) * (size_t)N);
        double pm = 0;
        qw_engine_reset(e);
        const float *l0 = qw_engine_prefill(e, ids, n, NULL, &pm);
        for (int k = 0; k < N; k++) {
            memcpy(ref + (size_t)k * 248320, l0, 248320 * 4);
            int bst = 0;
            for (int v = 1; v < 248320; v++) if (l0[v] > l0[bst]) bst = v;
            rt[k] = bst;
            l0 = qw_engine_step(e, bst, NULL);
        }
        setenv("QW_SLOTS", "3", 1);
        const int sid = qw_engine_par_pick(e, ids, n, 0);
        const float *pl = qw_engine_par_prefill(e, sid, ids, n, NULL);
        int first = -1; double mx = 0;
        for (int k = 0; k < N; k++) {
            const float *r = ref + (size_t)k * 248320;
            double d = 0;
            for (int v = 0; v < 248320; v++) { const double q = fabs((double)pl[v] - r[v]); if (q > d) d = q; }
            if (d > 0 && first < 0) first = k;
            if (d > mx) mx = d;
            if (d > 0 && k < first + 3) printf("第 %d 步（位置 %d）：logits 最大差 %.3g\n", k, n + k - 1, d);
            int32_t tk = rt[k];
            pl = qw_engine_par_step(e, &sid, &tk, 1, NULL);
            if (!pl) return 1;
        }
        printf("%s；最大差 %.3g\n", first < 0 ? "全部逐位元相同" : "有差異", mx);
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_PAR")) { // QW_PAR=檔案,檔案,…（各一個 prompt 的 token id）：先各自單獨 greedy（原本的路徑），再一起並發 greedy，逐 token 比較
        int32_t *pid[PM]; int pn[PM], B = 0;
        char *lst = strdup(getenv("QW_PAR"));
        for (char *f = strtok(lst, ","); f && B < PM; f = strtok(NULL, ",")) {
            FILE *fp = fopen(f, "rb");
            if (!fp) { fprintf(stderr, "開不了 %s\n", f); return 1; }
            pid[B] = malloc(sizeof(int32_t) * 200000); pn[B] = 0;
            long v; char c;
            while (fscanf(fp, "%ld", &v) == 1) { pid[B][pn[B]++] = (int32_t)v; if (fscanf(fp, "%c", &c) != 1) break; }
            fclose(fp);
            B++;
        }
        const int N = ngen;
        static int32_t ref[PM][4096], got[PM][4096];
        static uint64_t rh[PM][4096];
        const bool skip_ref = getenv("QW_PAR_NOREF") != NULL;
        for (int i = 0; i < B && !skip_ref; i++) { // 基準：一般 decode（prefill + qw_engine_step，greedy），每步 logits 的雜湊
            double pm = 0;
            qw_engine_reset(e);
            const float *l0 = qw_engine_prefill(e, pid[i], pn[i], NULL, &pm);
            for (int k = 0; k < N && l0; k++) {
                rh[i][k] = par_hash(l0);
                int bst = 0;
                for (int v = 1; v < 248320; v++) if (l0[v] > l0[bst]) bst = v;
                ref[i][k] = bst;
                if (k + 1 < N) l0 = qw_engine_step(e, bst, NULL);
            }
        }
        int32_t *mid = NULL; // QW_PAR_MIX：長 prompt（單獨 prefill 一次當基準：最後的 logits 雜湊、時間）
        int mn = 0;
        uint64_t mref = 0;
        double mt_alone = 0;
        if (getenv("QW_PAR_MIX")) {
            FILE *fp = fopen(getenv("QW_PAR_MIX"), "rb");
            if (!fp) { fprintf(stderr, "開不了 %s\n", getenv("QW_PAR_MIX")); return 1; }
            mid = malloc(sizeof(int32_t) * 1000000);
            long v; char c;
            while (mn < 1000000 && fscanf(fp, "%ld", &v) == 1) { mid[mn++] = (int32_t)v; if (fscanf(fp, "%c", &c) != 1) break; }
            fclose(fp);
            qw_engine_reset(e);
            double pm = 0;
            const double t1 = qw_now();
            const float *l0 = qw_engine_prefill(e, mid, mn, NULL, &pm);
            mt_alone = qw_now() - t1;
            if (!l0) { fprintf(stderr, "長 prompt 單獨 prefill 失敗\n"); return 1; }
            mref = par_hash(l0);
            printf("長 prompt %d 個 token：單獨 prefill %.0f ms（%.0f tok/s）\n", mn, mt_alone * 1e3, mn / mt_alone);
        }
        int lgbad[PM];
        for (int i = 0; i < PM; i++) lgbad[i] = -1;
        int sid[PM]; int32_t last[PM]; unsigned long long busy = 0;
        double tp = qw_now();
        for (int i = 0; i < B; i++) {
            sid[i] = qw_engine_par_pick(e, pid[i], pn[i], busy);
            if (sid[i] < 0) { fprintf(stderr, "沒有空的 slot（QW_SLOTS）\n"); return 1; }
            busy |= 1ull << sid[i];
            const float *pl = qw_engine_par_prefill(e, sid[i], pid[i], pn[i], NULL);
            if (!pl) { fprintf(stderr, "prefill 失敗\n"); return 1; }
            int bst = 0;
            for (int v = 1; v < 248320; v++) if (pl[v] > pl[bst]) bst = v;
            got[i][0] = last[i] = bst;
            if (!skip_ref && par_hash(pl) != rh[i][0] && lgbad[i] < 0) lgbad[i] = 0;
        }
        tp = qw_now() - tp;
        if (getenv("QW_PARV")) { // QW_PARV=k：一起推測解碼（每個對話最多猜 k 個；猜測 = 正確答案，每 3 輪故意弄錯一個），比每列 logits
            const int K = atoi(getenv("QW_PARV")) > 0 ? atoi(getenv("QW_PARV")) : 1;
            int cnt[PM], vbad[PM], rows = 0, rounds = 0, gen = 0, dsum = 0, asum = 0;
            for (int i = 0; i < B; i++) { cnt[i] = 1; vbad[i] = -1; }
            const bool usedf = getenv("QW_PARV_DF") != NULL; // 猜測用各對話自己的 DFlash2（信心 < 0.8 截斷），不用正確答案
            if (usedf && qw_engine_par_vdraft(e, sid, last, B)) { fprintf(stderr, "par_vdraft 失敗\n"); return 1; }
            double gs = 0, t0 = qw_now();
            for (;;) {
                int as[PM], na = 0;
                for (int i = 0; i < B; i++) if (cnt[i] < N) as[na++] = i;
                if (!na) break;
                const int per = 8 / na - 1 < K ? 8 / na - 1 : K; // 每個對話最多猜幾個（共 ≤ 8 列）
                if (na > 8) { fprintf(stderr, "QW_PARV：最多 8 個對話\n"); return 1; }
                int s2[8], n2[8], nd[8];
                int32_t tk[8], dr[8][8];
                int r = 0;
                for (int q = 0; q < na; q++) {
                    const int i = as[q];
                    nd[q] = per < N - 1 - cnt[i] ? per : N - 1 - cnt[i];
                    if (nd[q] < 0) nd[q] = 0;
                    s2[q] = sid[i];
                    n2[q] = 1 + nd[q];
                    tk[r++] = last[i];
                    if (usedf) {
                        int32_t pd[8];
                        float pp[8];
                        const int m = qw_engine_par_draft(e, sid[i], pd, pp);
                        int k2 = 0;
                        while (k2 < nd[q] && k2 < m && (k2 == 0 || pp[k2] >= 0.8f)) { dr[q][k2] = pd[k2]; k2++; }
                        nd[q] = k2;
                        n2[q] = 1 + k2;
                        dsum += k2;
                    }
                    for (int t = 0; t < nd[q]; t++) {
                        if (!usedf) {
                            dr[q][t] = skip_ref ? last[i] : ref[i][cnt[i] + t];
                            if (t == 1 && (rounds + i) % 3 == 0) dr[q][t] = (dr[q][t] + 1) % 248320; // 故意錯
                        }
                        tk[r++] = dr[q][t];
                    }
                }
                if (rounds == 3) { t0 = qw_now(); gs = 0; gen = 0; } // 前 3 輪含 kernel 編譯，不算
                double gm = 0;
                const float *bl = qw_engine_par_vstep(e, s2, n2, tk, na, &gm);
                if (!bl) { fprintf(stderr, "par_vstep 失敗\n"); return 1; }
                gs += gm;
                rounds++;
                int acc[8];
                int32_t nxa[8];
                r = 0;
                for (int q = 0; q < na; q++) {
                    const int i = as[q];
                    int am[8];
                    for (int t = 0; t <= nd[q]; t++) {
                        const float *lr = bl + (size_t)(r + t) * 248320;
                        int bst = 0;
                        for (int v = 1; v < 248320; v++) if (lr[v] > lr[bst]) bst = v;
                        am[t] = bst;
                        bool pre = got[i][cnt[i] - 1] == ref[i][cnt[i] - 1]; // 這列之前餵的都正確：logits 應與單獨跑相同
                        for (int u = 0; u < t && pre; u++) pre = dr[q][u] == ref[i][cnt[i] + u];
                        if (!skip_ref && pre) { rows++; if (par_hash(lr) != rh[i][cnt[i] + t] && vbad[i] < 0) vbad[i] = cnt[i] + t; }
                    }
                    int j = 0;
                    while (j < nd[q] && am[j] == dr[q][j]) j++;
                    for (int t = 0; t < j; t++) got[i][cnt[i] + t] = dr[q][t];
                    got[i][cnt[i] + j] = am[j];
                    last[i] = am[j];
                    cnt[i] += j + 1;
                    gen += j + 1;
                    acc[q] = j;
                    nxa[q] = am[j];
                    asum += j;
                    r += nd[q] + 1;
                }
                if (qw_engine_par_vaccept(e, acc, usedf ? nxa : NULL)) { fprintf(stderr, "par_vaccept 失敗\n"); return 1; }
            }
            const double tt = qw_now() - t0;
            for (int i = 0; i < B; i++) {
                int same = 0;
                while (same < N && got[i][same] == ref[i][same]) same++;
                printf("對話 %d（slot %d，prompt %d）：前 %d / %d 個 token 相同；%s\n", i, sid[i], pn[i], same, N,
                       vbad[i] < 0 ? "比過的列 logits 都與單獨跑逐位元相同 ✓" : "logits 不同 ✗");
                if (vbad[i] >= 0) printf("  第一個不同：位置 %d\n", vbad[i]);
            }
            if (usedf) printf("DFlash2：每個對話每輪平均猜 %.2f、接受 %.2f\n", (double)dsum / (rounds * B), (double)asum / (rounds * B));
            printf("一起推測解碼 %d 個對話、每個最多猜 %d 個：%d 輪、比了 %d 列；GPU 每輪 %.2f ms；%.1f tok/s（含 CPU argmax）\n", B, K, rounds,
                   rows, gs / (rounds - 3), gen / tt);
            qw_engine_close(e);
            return 0;
        }
        int k0 = 1;
        if (mid) { // 分段 prefill 長 prompt；每段在 GPU 上跑的同時，engine 反覆呼叫 mix_cb（B 個對話一起 decode 一步）
            if (N > 4096) { fprintf(stderr, "QW_PAR_MIX：生成數最多 4096\n"); return 1; }
            const int msid = qw_engine_par_pick(e, mid, mn, busy);
            if (msid < 0) { fprintf(stderr, "沒有空的 slot 給長 prompt（QW_SLOTS 至少 %d）\n", B + 1); return 1; }
            const int ch = getenv("QW_PAR_MIX_CHUNK") ? atoi(getenv("QW_PAR_MIX_CHUNK")) / 64 * 64 : 4096;
            mxs = (typeof(mxs)){.e = e, .B = B, .N = N, .k = 1, .sid = sid, .lgbad = lgbad, .last = last, .ref = ref, .got = got, .rh = rh,
                                .skip_ref = skip_ref};
            int done = 0, cached = 0, segs = 0;
            const float *ml = NULL;
            const double t1 = qw_now();
            while (done < mn) {
                const int d0 = done;
                qw_engine_par_mix(e, getenv("QW_PAR_MIX_OFF") ? NULL : mix_cb, NULL); // QW_PAR_MIX_OFF：同樣的分段、不重疊（對照）
                ml = qw_engine_par_prefill_to(e, msid, mid, mn, ch > 0 ? ch : 4096, &done, segs ? NULL : &cached);
                qw_engine_par_mix(e, NULL, NULL);
                if (!ml || done <= d0) { fprintf(stderr, "長 prompt 分段 prefill 失敗\n"); return 1; }
                segs++;
            }
            const double tm = qw_now() - t1;
            printf("長 prompt（slot %d，重用 %d）：分段 %d 段、%.0f ms（%.0f tok/s，單獨 %.0f tok/s）；最後的 logits %s；期間 decode %d 步（%.1f 步/秒）\n",
                   msid, cached, segs, tm * 1e3, (mn - cached) / tm, mn / mt_alone,
                   par_hash(ml) == mref ? "與單獨跑逐位元相同 ✓" : "不同 ✗", mxs.steps, mxs.steps / tm);
            k0 = mxs.k;
        }
        double gsum = 0, tw = qw_now();
        for (int k = k0; k < N; k++) {
            if (k == k0 + 1) { tw = qw_now(); gsum = 0; } // 第一步含 kernel 編譯，不算
            double gm = 0;
            const float *bl = qw_engine_par_step(e, sid, last, B, &gm);
            if (!bl) { fprintf(stderr, "par_step 失敗\n"); return 1; }
            gsum += gm;
            for (int i = 0; i < B; i++) {
                const float *r = bl + (size_t)i * 248320;
                if (!skip_ref && lgbad[i] < 0 && got[i][k - 1] == ref[i][k - 1] && par_hash(r) != rh[i][k]) lgbad[i] = k;
                int bst = 0;
                for (int v = 1; v < 248320; v++) if (r[v] > r[bst]) bst = v;
                got[i][k] = last[i] = bst;
            }
        }
        tw = qw_now() - tw;
        if (getenv("QW_PAR_KPROF")) { // 再跑一步，逐 dispatch 計時、依 kernel 名稱加總
            qw_gpu *g = qw_engine_gpu(e);
            qw_profile_enable(g, true);
            qw_engine_par_step(e, sid, last, B, NULL);
            const int nd = qw_profile_count(g);
            typedef struct { const char *name; double us; int cnt; } agg;
            static agg ag[512];
            int na = 0;
            double tot = 0;
            for (int q = 0; q < nd; q++) {
                double us;
                const char *nm = qw_profile_get(g, q, &us);
                tot += us;
                int j = 0;
                while (j < na && strcmp(ag[j].name, nm)) j++;
                if (j == na && na < 512) ag[na++] = (agg){nm, 0, 0};
                if (j < na) { ag[j].us += us; ag[j].cnt++; }
            }
            for (int x = 0; x < na; x++)
                for (int y = x + 1; y < na; y++)
                    if (ag[y].us > ag[x].us) { agg t = ag[x]; ag[x] = ag[y]; ag[y] = t; }
            printf("\n## 並發 %d 一步的 kernel（%d 個 dispatch，合計 %.2f ms）\n", B, nd, tot / 1e3);
            for (int x = 0; x < na && x < 22; x++) printf("| %s | %d | %.3f ms |\n", ag[x].name, ag[x].cnt, ag[x].us / 1e3);
            qw_profile_enable(g, false);
        }
        for (int i = 0; i < B; i++) {
            int same = 0;
            while (same < N && got[i][same] == ref[i][same]) same++;
            if (skip_ref) printf("對話 %d（slot %d，prompt %d）\n", i, sid[i], pn[i]);
            else if (lgbad[i] >= 0) printf("對話 %d（slot %d，prompt %d）：前 %d / %d 個 token 相同，logits 從第 %d 步起不同 ✗\n", i, sid[i], pn[i], same, N, lgbad[i]);
            else printf("對話 %d（slot %d，prompt %d）：%d 個 token、每步 logits 都與單獨跑逐位元相同%s\n", i, sid[i], pn[i], N, same == N ? " ✓" : "（token 卻不同？）✗");
        }
        const int ns = N - k0 - 1 > 0 ? N - k0 - 1 : 1; // 計時的步數（第一步不算）
        printf("並發 %d：prefill 共 %.0f ms；decode %d 步，每步 GPU %.2f ms、牆鐘 %.2f ms → 總共 %.1f tok/s（每個對話 %.1f）\n", B, tp * 1e3, N - k0,
               gsum / ns, tw * 1e3 / ns, B * ns / tw, ns / tw);
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_VB_BENCH")) { // QW_VB_BENCH=次數：prefill 前 n−8 個 token，再用最後 8 個 token 量驗證 1…8 個 token 的 GPU 時間
        const int reps = atoi(getenv("QW_VB_BENCH")) > 0 ? atoi(getenv("QW_VB_BENCH")) : 10;
        double pm = 0;
        if (!qw_engine_prefill(e, ids, n - 8, NULL, &pm)) return 1;
        qw_engine_vb_bench(e, ids + n - 8, 1, 2);
        for (int k = 2; k <= 8; k++) qw_engine_vb_bench(e, ids + n - 8, k, 1); // 暖機（編譯）
        for (double tw = qw_now(); qw_now() - tw < 1.5;) qw_engine_vb_bench(e, ids + n - 8, 8, 1); // 讓 GPU 時脈拉起來
        if (getenv("QW_VB_PROF")) { // QW_VB_PROF=k：驗證 k 個 token 一次，逐 dispatch 計時，依 kernel 名稱加總
            const int k = atoi(getenv("QW_VB_PROF"));
            qw_gpu *g = qw_engine_gpu(e);
            qw_profile_enable(g, true);
            qw_engine_vb_bench(e, ids + n - 8, k, 1);
            const int nd = qw_profile_count(g);
            typedef struct { const char *name; double us; int cnt; } agg;
            agg a[256];
            int na = 0;
            double tot = 0;
            for (int i = 0; i < nd; i++) {
                double us;
                const char *nm = qw_profile_get(g, i, &us);
                tot += us;
                int j = 0;
                while (j < na && strcmp(a[j].name, nm)) j++;
                if (j == na && na < 256) a[na++] = (agg){nm, 0, 0};
                if (j < na) { a[j].us += us; a[j].cnt++; }
            }
            for (int i = 0; i < na; i++)
                for (int j = i + 1; j < na; j++)
                    if (a[j].us > a[i].us) { agg t = a[i]; a[i] = a[j]; a[j] = t; }
            printf("\n## 驗證 %d 個 token 的每個 kernel（%d 個 dispatch，合計 %.2f ms）\n\n| kernel | 次數 | 合計 | 每次 |\n|---|---|---|---|\n", k, nd, tot / 1e3);
            for (int i = 0; i < na; i++) printf("| %s | %d | %.3f ms | %.1f µs |\n", a[i].name, a[i].cnt, a[i].us / 1e3, a[i].us / a[i].cnt);
            qw_profile_enable(g, false);
            qw_engine_close(e);
            return 0;
        }
        double t1 = 0;
        printf("context %d；", n - 8);
        for (int k = 1; k <= 8; k++) {
            const double t = qw_engine_vb_bench(e, ids + n - 8, k, reps);
            if (k == 1) t1 = t;
            printf(" %d:%.2f", k, t);
        }
        printf(" ms（1 = 一般 decode %.2f）\n", t1);
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_MTP_CHAIN")) {
        // 量 MTP 鏈（一輪猜 K 個）：逐 token 跑主模型（greedy），每步之後用真實殘差流猜第 1 個，再用 MTP 自己的殘差流連續猜到第 K 個。
        // 下一步主模型用真實殘差流重跑 MTP，蓋掉鏈寫進 MTP cache 的位置（同 NInfer 的 history repair）
        const int K = atoi(getenv("QW_MTP_CHAIN")) < 1 ? 1 : atoi(getenv("QW_MTP_CHAIN")) > 8 ? 8 : atoi(getenv("QW_MTP_CHAIN"));
        const int total = n + ngen;
        int32_t *seq = malloc(sizeof(int32_t) * (size_t)(total + 16));
        int32_t (*dr)[8] = calloc((size_t)total + 16, sizeof *dr);
        double m1 = 0, mc = 0;
        int nc = 0;
        memcpy(seq, ids, sizeof(int32_t) * (size_t)n);
        for (int t = 0; t < total - 1; t++) {
            const float *l2 = qw_engine_step(e, seq[t], NULL);
            if (!l2) return 1;
            int best = 0;
            for (int v = 1; v < 248320; v++) if (l2[v] > l2[best]) best = v;
            if (t + 1 >= n) seq[t + 1] = best;
            double ms = 0;
            int32_t d = qw_engine_mtp_draft(e, seq[t + 1], t, &ms); // 猜 seq[t+2]
            m1 += ms;
            if (t + 1 >= n - 1) { // 生成區：dr[t+1] = 從 seq[t+1] 之後猜的 K 個
                dr[t + 1][0] = d;
                for (int k = 1; k < K; k++) {
                    d = qw_engine_mtp_chain(e, d, t + k, &ms);
                    mc += ms;
                    nc++;
                    dr[t + 1][k] = d;
                }
            }
        }
        // 生成遇到停止 token 就截斷（之後的內容沒有意義）
        int tot2 = total;
        for (int i = n; i < total; i++) if (seq[i] == 248046 || seq[i] == 248044) { tot2 = i + 1; break; }
        const int total_ = total;
        (void)total_;
#define total tot2
        // 核對：位置 a（已決定的 anchor）之後猜的 K 個 vs seq[a+1..a+K]
        int hist[9] = {0}, cnt = 0, hitk[8] = {0};
        for (int a = n - 1; a + K < total; a++) {
            int j = 0;
            while (j < K && dr[a][j] == seq[a + 1 + j]) j++;
            hist[j]++;
            cnt++;
            for (int k = 0; k < K; k++) hitk[k] += dr[a][k] == seq[a + 1 + k];
        }
        // 模擬實際推測解碼的輪次：從 anchor 開始，每輪前進 1 + 接受數
        int rounds = 0, adv = 0;
        for (int a = n - 1; a + K < total;) {
            int j = 0;
            while (j < K && dr[a][j] == seq[a + 1 + j]) j++;
            rounds++;
            adv += 1 + j;
            a += 1 + j;
        }
        if (getenv("QW_MTP_OUT")) { FILE *fo = fopen(getenv("QW_MTP_OUT"), "w"); for (int i = n; i < total; i++) fprintf(fo, "%d%s", seq[i], i + 1 < total ? "," : "\n"); fclose(fo); }
        printf("MTP 鏈 K=%d：%d 個 anchor；連續猜中數分布", K, cnt);
        for (int j = 0; j <= K; j++) printf(" %d:%d", j, hist[j]);
        printf("\n各步猜中率（不管前面）：");
        for (int k = 0; k < K; k++) printf(" 第%d個 %.1f%%", k + 1, 100.0 * hitk[k] / (cnt ? cnt : 1));
        printf("\n模擬推測：%d 輪前進 %d 個 token，每輪 %.2f 個；MTP 第 1 步平均 %.2f ms、鏈每步 %.2f ms\n", rounds, adv,
               (double)adv / (rounds ? rounds : 1), m1 / (total - 1), nc ? mc / nc : 0);
#undef total
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_MTP_TEST")) {
        // 量 MTP 的猜中率：逐 token 跑主模型，每步之後讓 MTP 猜「再下一個」token，下一步對答案
        int hit = 0, tot = 0;
        double mt = 0;
        int32_t draft = -1;
        const float *l2 = NULL;
        int32_t cur = ids[0];
        int total = n + ngen;
        printf("生成：");
        for (int t = 0; t < total - 1; t++) {
            l2 = qw_engine_step(e, cur, NULL);
            int best = 0;
            for (int v = 1; v < 248320; v++) if (l2[v] > l2[best]) best = v;
            const int32_t next = t + 1 < n ? ids[t + 1] : best; // prompt 內用真實 token，之後用 greedy
            if (t + 1 >= n) {
                if (draft >= 0) { tot++; hit += draft == best; }
                printf("%d%s", best, (draft == best) ? "✓," : ",");
            }
            double ms = 0;
            draft = getenv("QW_MTP_OFF") ? -1 : qw_engine_mtp_draft(e, next, t, &ms);
            mt += ms;
            cur = next;
        }
        printf("\nMTP 猜中率：%d / %d = %.1f%%；每次猜的 GPU 時間平均 %.2f ms\n", hit, tot, 100.0 * hit / (tot ? tot : 1), mt / (total - 1));
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_DF_TEST")) { // QW_DF_TEST=taps 檔（QW_TAPS 寫的）；位置 0..n−1 是 prompt（n 由 ids 長度決定）
        FILE *f = fopen(getenv("QW_DF_TEST"), "rb");
        if (!f) return 1;
        fseek(f, 0, SEEK_END);
        const long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        const int rec = 8 + 4 * 5 * 2560, m = (int)(sz / rec);
        float *feats = malloc((size_t)m * 5 * 2560 * 4);
        int32_t *tk = malloc((size_t)m * 4);
        for (int i = 0; i < m; i++) {
            int32_t hd[2];
            if (fread(hd, 4, 2, f) != 2 || fread(feats + (size_t)i * 5 * 2560, 4, 5 * 2560, f) != 5 * 2560) break;
            tk[i] = hd[1];
        }
        fclose(f);
        return qw_engine_df_test(e, feats, tk, m, n);
    }
    if (getenv("QW_TAPS")) { // DFlash2 用的特徵：逐 token（prompt + greedy 生成 ngen 個），每個位置寫 [pos, token, 5 × 2560 float]
        static const int tl[5] = {5, 15, 25, 35, 45};
        FILE *f = fopen(getenv("QW_TAPS"), "wb");
        static float all[48 * 10240];
        float row[5 * 2560];
        int32_t tok = ids[0];
        for (int i = 0; i < n + ngen; i++) {
            if (i < n) tok = ids[i];
            const float *L = qw_engine_step(e, tok, all);
            if (!L) break;
            for (int k = 0; k < 5; k++)
                for (int d = 0; d < 2560; d++) {
                    const float *h = all + (size_t)tl[k] * 10240;
                    row[k * 2560 + d] = (h[d] + h[2560 + d] + h[5120 + d] + h[7680 + d]) / 4;
                }
            const int32_t hd[2] = {i, tok};
            fwrite(hd, 4, 2, f);
            fwrite(row, 4, 5 * 2560, f);
            if (i + 1 >= n) { // 下一個 token：greedy
                int best = 0;
                for (int v = 1; v < 248320; v++) if (L[v] > L[best]) best = v;
                tok = best;
            }
        }
        fclose(f);
        printf("taps：%d 個位置\n", n + ngen);
        return 0;
    }
    if (getenv("QW_TRACE")) { // QW_TRACE=k、QW_TRACE_MODE=dec|vb、QW_TRACE_OUT=檔案：前 n−k 個 token 逐個 decode，最後 k 個用 decode 或一次驗證，寫出逐層 trace
        const int k = atoi(getenv("QW_TRACE"));
        const bool vb = getenv("QW_TRACE_MODE") && !strcmp(getenv("QW_TRACE_MODE"), "vb");
        for (int i = 0; i < n - k; i++) if (!qw_engine_step(e, ids[i], NULL)) return 1;
        const void *tr = NULL;
        if (vb) tr = qw_engine_trace_vb(e, ids + n - k, k);
        else
            for (int i = 0; i < k; i++) {
                qw_engine_trace_buf(e, i);
                if (!qw_engine_step(e, ids[n - k + i], NULL)) return 1;
                tr = qw_engine_trace_buf(e, i);
            }
        if (!tr) { fprintf(stderr, "trace 失敗\n"); return 1; }
        FILE *f = fopen(getenv("QW_TRACE_OUT"), "wb");
        fwrite(tr, 1, (size_t)k * 48 * 2 * (10240 * 4 + 2560 * 2), f);
        fclose(f);
        return 0;
    }
    if (getenv("QW_PPL")) { // 困惑度：逐 token（decode 路徑），第 i 個位置的 logits 對第 i+1 個 token 的 −log p
        // QW_PPL_PF=N：前 N 個 token 走批次 prefill（量 prefill 路徑對後續輸出的影響），從位置 N−1 起計分與 dump
        const int npf = getenv("QW_PPL_PF") ? atoi(getenv("QW_PPL_PF")) : 0;
        if (npf >= n) { fprintf(stderr, "QW_PPL_PF 必須小於 token 數\n"); return 1; }
        double nll = 0;
        int cnt = 0;
        for (int i = npf > 0 ? npf - 1 : 0; i + 1 < n; i++) {
            double pms;
            const float *L = npf > 0 && i == npf - 1 ? qw_engine_prefill(e, ids, npf, NULL, &pms) : qw_engine_step(e, ids[i], NULL);
            if (!L) break;
            static FILE *pd = NULL; // QW_PPL_DUMP=檔案：每個位置的 logits（float32 × 248320）依序寫出，給 tools/vs_strata_logits.py 對照
            if (!pd && getenv("QW_PPL_DUMP")) pd = fopen(getenv("QW_PPL_DUMP"), "wb");
            if (pd) fwrite(L, 4, 248320, pd);
            float mx = L[0];
            for (int v = 1; v < 248320; v++) if (L[v] > mx) mx = L[v];
            double z = 0;
            for (int v = 0; v < 248320; v++) z += exp((double)L[v] - mx);
            if (i >= (npf > 0 ? npf - 1 : 32)) { nll += log(z) + mx - L[ids[i + 1]]; cnt++; }
        }
        printf("困惑度：%d 個 token，平均 NLL %.4f，PPL %.3f\n", cnt, nll / cnt, exp(nll / cnt));
        return 0;
    }
    if (getenv("QW_PREFILL")) {
        // 批次 prefill；dump 時存成與逐 token 相同的檔案格式
        float *all = dump ? malloc((size_t)48 * n * 10240 * 4) : NULL;
        double pms = 0, tw0 = qw_now();
        lg = qw_engine_prefill(e, ids, n, all, &pms);
        double tw = qw_now() - tw0;
        for (int r = 1; r < (getenv("QW_PF_REP") ? atoi(getenv("QW_PF_REP")) : 1); r++) { // 重複量測（第一次含暖機）
            printf("prefill：%d 個 token，GPU %.1f ms，牆鐘 %.1f ms\n", n, pms, tw * 1e3);
            qw_engine_reset(e);
            tw0 = qw_now();
            lg = qw_engine_prefill(e, ids, n, NULL, &pms);
            tw = qw_now() - tw0;
        }
        if (!lg) { fprintf(stderr, "prefill 失敗\n"); return 1; }
        if (getenv("QW_LOGITS")) { // 最後位置的 logits 寫到檔案（精度比較用）
            FILE *f = fopen(getenv("QW_LOGITS"), "wb");
            fwrite(lg, sizeof(float), 248320, f);
            fclose(f);
        }
        if (getenv("QW_PF_KBENCH")) { int pfs = getenv("QW_PF") ? atoi(getenv("QW_PF")) : 4096; qw_engine_pf_kbench(e, n % pfs ? n % pfs : pfs, atoi(getenv("QW_PF_KBENCH"))); return 0; }
        printf("prefill：%d 個 token，GPU %.1f ms（%.0f tok/s），牆鐘 %.1f ms\n", n, pms, n / pms * 1e3, tw * 1e3);
        if (dump)
            for (int t = 0; t < n; t++) {
                for (int l = 0; l < 48; l++) memcpy(layers + (size_t)l * 10240, all + ((size_t)l * n + t) * 10240, 10240 * 4);
                char p[512];
                snprintf(p, sizeof p, "%s/layers_t%d.bin", dump, t);
                FILE *f = fopen(p, "wb");
                fwrite(layers, sizeof layers, 1, f);
                fclose(f);
            }
        free(all);
        n = 0; // 跳過逐 token 迴圈
    }
    for (int t = 0; t < n; t++) {
        lg = qw_engine_step(e, ids[t], dump ? layers : NULL);
        if (!lg) { fprintf(stderr, "step 失敗\n"); return 1; }
        if (dump) {
            char p[512];
            snprintf(p, sizeof p, "%s/layers_t%d.bin", dump, t);
            FILE *f = fopen(p, "wb");
            fwrite(layers, sizeof layers, 1, f);
            fclose(f);
        }
    }
    if (dump) {
        char p[512];
        snprintf(p, sizeof p, "%s/logits_last.bin", dump);
        FILE *f = fopen(p, "wb");
        fwrite(lg, sizeof(float), 248320, f);
        fclose(f);
    }
    if (getenv("QW_HCTEST")) {
        qw_engine_hctest(e, getenv("QW_HCTEST"));
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_KBENCH")) {
        qw_engine_kbench(e, 48);
        qw_engine_close(e);
        return 0;
    }
    if (getenv("QW_PROFILE")) {
        // 多跑幾步，取最後一步的每個 dispatch 時間，依 kernel 名稱加總
        qw_gpu *g = qw_engine_gpu(e);
        qw_profile_enable(g, true);
        lg = qw_engine_step(e, 11751, NULL);
        int n = qw_profile_count(g);
        typedef struct { const char *name; double us; int cnt; } agg;
        agg a[256];
        int na = 0;
        double tot = 0;
        for (int i = 0; i < n; i++) {
            double us;
            const char *nm = qw_profile_get(g, i, &us);
            tot += us;
            int j = 0;
            while (j < na && strcmp(a[j].name, nm)) j++;
            if (j == na) a[na++] = (agg){nm, 0, 0};
            a[j].us += us;
            a[j].cnt++;
        }
        for (int i = 0; i < na; i++)
            for (int j = i + 1; j < na; j++)
                if (a[j].us > a[i].us) { agg t = a[i]; a[i] = a[j]; a[j] = t; }
        printf("\n## 每個 kernel 的 GPU 時間（一個 token，%d 個 dispatch，合計 %.2f ms）\n\n| kernel | 次數 | 合計 | 每次 | 佔比 |\n|---|---|---|---|---|\n", n, tot / 1e3);
        for (int i = 0; i < na; i++)
            printf("| %s | %d | %.3f ms | %.1f µs | %.1f%% |\n", a[i].name, a[i].cnt, a[i].us / 1e3, a[i].us / a[i].cnt, a[i].us / tot * 100);
        qw_profile_enable(g, false);
        qw_engine_close(e);
        return 0;
    }
    if (!getenv("QW_NO_PIPE")) {
        int32_t out[4096];
        double gms = 0, tw0 = qw_now();
        int got = qw_engine_generate(e, ngen, out, &gms);
        double tw = qw_now() - tw0;
        printf("生成：");
        for (int i = 0; i < got; i++) printf("%d%s", out[i], i + 1 < got ? "," : "\n");
        printf("decode（流水線）：每個 token GPU 時間平均 %.2f ms（%.1f tok/s）；牆鐘時間 %.2f ms（%.1f tok/s）\n", gms / got,
               1000.0 * got / gms, tw / got * 1e3, got / tw);
        qw_engine_close(e);
        return 0;
    }
    printf("生成：");
    double tg = 0, tw0 = qw_now();
    for (int i = 0; i < ngen; i++) {
        int best = 0;
        for (int v = 1; v < 248320; v++)
            if (lg[v] > lg[best]) best = v;
        printf("%d%s", best, i + 1 < ngen ? "," : "\n");
        fflush(stdout);
        lg = qw_engine_step(e, best, NULL);
        tg += qw_engine_last_gpu_ms(e);
    }
    double tw = qw_now() - tw0;
    printf("decode：每個 token GPU 時間平均 %.2f ms（%.1f tok/s）；牆鐘時間 %.2f ms（%.1f tok/s）\n", tg / ngen, 1000.0 * ngen / tg,
           tw / ngen * 1e3, ngen / tw);
    qw_engine_close(e);
    return 0;
}
