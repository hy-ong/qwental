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
        qw_tok *tk = qw_tok_open(getenv("QW_MODEL") ? getenv("QW_MODEL") : "models/Qwen3.8-Flash-Next-mixed-4-8bit", err, sizeof err);
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
        double nll = 0;
        int cnt = 0;
        for (int i = 0; i + 1 < n; i++) {
            const float *L = qw_engine_step(e, ids[i], NULL);
            if (!L) break;
            static FILE *pd = NULL; // QW_PPL_DUMP=檔案：每個位置的 logits（float32 × 248320）依序寫出，給 tools/vs_strata_logits.py 對照
            if (!pd && getenv("QW_PPL_DUMP")) pd = fopen(getenv("QW_PPL_DUMP"), "wb");
            if (pd) fwrite(L, 4, 248320, pd);
            float mx = L[0];
            for (int v = 1; v < 248320; v++) if (L[v] > mx) mx = L[v];
            double z = 0;
            for (int v = 0; v < 248320; v++) z += exp((double)L[v] - mx);
            if (i >= 32) { nll += log(z) + mx - L[ids[i + 1]]; cnt++; }
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
