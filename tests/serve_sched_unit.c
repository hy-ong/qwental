// serve.c 排程 / 進場控制 / 持久化點 / /status 的單元測試（不載模型、不用 GPU）。
// 編譯（engine 只是連結進來，不會開）：
//   clang -std=c23 -O1 -Isrc -Itools tests/serve_sched_unit.c src/engine.c src/model.c src/st.c src/tok.c src/json.c src/requant.c \
//     src/dflash.c src/gguf.c src/ggml_dq.c src/chat.c build/gpu_metal.o -o build/serve_sched_unit -framework Metal -framework Foundation -framework IOKit
//   build/serve_sched_unit [ccapture 目錄]   （給 Claude Code 抓下來的請求目錄時另外檢查主對話 / subagent 的共用前綴）
#define main qwental_serve_main
#include "serve.c"
#undef main
#include <assert.h>
#include <math.h>

static int fails;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static void ts_reset(void) {
    for (int i = 0; i < TOOLSET_MAX; i++) { free(toolsets[i].p); toolsets[i].p = NULL; toolsets[i].n = 0; toolsets[i].used = 0; }
}

static void test_reminder_end(void) {
    const char *p = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\n<system-reminder>\nA\n</system-reminder>\n<system-reminder>B</system-reminder>\n"
                    "do the task<|im_end|>\n<|im_start|>assistant\n";
    const size_t r = reminder_end(p, strlen(p));
    const char *want = strstr(p, "B</system-reminder>") + strlen("B</system-reminder>");
    CHECK(r == (size_t)(want - p));
    // 沒有 reminder、reminder 不在開頭、沒關起來：0
    const char *q = "<|im_start|>user\nhello <system-reminder>x</system-reminder><|im_end|>\n";
    CHECK(reminder_end(q, strlen(q)) == 0);
    const char *q2 = "<|im_start|>user\n<system-reminder>never closed<|im_end|>\n<|im_start|>user\n</system-reminder>";
    CHECK(reminder_end(q2, strlen(q2)) == 0);
    CHECK(reminder_end("abc", 3) == 0);
}

static void test_toolset_diverge(void) {
    ts_reset();
    const char *main_t = "\n{\"name\": \"Agent\"}\n{\"name\": \"Bash\"}\n{\"name\": \"Sched\"}\n{\"name\": \"Write\"}";
    const char *sub_t = "\n{\"name\": \"Agent\"}\n{\"name\": \"Bash\"}\n{\"name\": \"Write\"}";
    CHECK(toolset_diverge(main_t, strlen(main_t)) == 0); // 第一次：沒有可以比的
    const size_t d = toolset_diverge(sub_t, strlen(sub_t));
    const size_t want = (size_t)(strstr(sub_t, "\n{\"name\": \"Write\"") - sub_t);
    CHECK(d == want);
    CHECK(!memcmp(main_t, sub_t, d)); // 之前的位元組兩邊都一樣
    // 主對話再來一次：與 subagent 的清單比，同一個位置（工具 "Sched" 之前）
    const size_t d2 = toolset_diverge(main_t, strlen(main_t));
    CHECK(d2 == (size_t)(strstr(main_t, "\n{\"name\": \"Sched\"") - main_t) && d2 == d);
    // 完全相同的清單：不加點（</tools> 那個就夠）
    ts_reset();
    toolset_diverge(sub_t, strlen(sub_t));
    CHECK(toolset_diverge(sub_t, strlen(sub_t)) == 0);
    // 第一個工具就不同：0
    const char *other = "\n{\"name\": \"Zed\"}";
    CHECK(toolset_diverge(other, strlen(other)) == 0);
    // 滿了：LRU 換掉，不會出錯
    char buf[64];
    for (int i = 0; i < TOOLSET_MAX * 3; i++) {
        snprintf(buf, sizeof buf, "\n{\"name\": \"A\"}\n{\"name\": \"T%d\"}", i);
        toolset_diverge(buf, strlen(buf));
    }
    int used = 0;
    for (int i = 0; i < TOOLSET_MAX; i++) used += toolsets[i].p != NULL;
    CHECK(used == TOOLSET_MAX);
    snprintf(buf, sizeof buf, "\n{\"name\": \"A\"}\n{\"name\": \"Q\"}");
    CHECK(toolset_diverge(buf, strlen(buf)) == strlen("\n{\"name\": \"A\"}"));
    ts_reset();
}

static void test_persist_offsets(void) {
    ts_reset();
    const char *a = "<|im_start|>system\n# Tools\n<tools>\n{\"name\": \"A\"}\n{\"name\": \"B\"}\n</tools>rest sysA<|im_end|>\n<|im_start|>user\n"
                    "<system-reminder>R</system-reminder>\ntask<|im_end|>\n";
    const char *b = "<|im_start|>system\n# Tools\n<tools>\n{\"name\": \"A\"}\n{\"name\": \"C\"}\n</tools>rest sysB<|im_end|>\n<|im_start|>user\n"
                    "<system-reminder>R</system-reminder>\ntask2<|im_end|>\n";
    size_t o[PERSIST_MAX];
    int n = persist_offsets(a, strlen(a), o);
    CHECK(n == 2); // </tools>、reminder 結尾
    CHECK(n == 2 && o[0] == (size_t)(strstr(a, "\n</tools>") + 1 - a) && o[1] == (size_t)(strstr(a, "</system-reminder>") + 18 - a));
    n = persist_offsets(b, strlen(b), o);
    CHECK(n == 3);
    CHECK(n == 3 && o[0] == (size_t)(strstr(b, "\n{\"name\": \"C\"") - b) && o[0] < o[1] && o[1] < o[2]);
    CHECK(n == 3 && !memcmp(a, b, o[0]));
    ts_reset();
}

// 抓下來的請求：主對話與 subagent 照 serve 的方式 render，報告共用前綴到哪、各持久化點在哪
static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long L = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *p = malloc((size_t)L + 1);
    *n = fread(p, 1, (size_t)L, f);
    p[*n] = 0;
    fclose(f);
    return p;
}
static bool render_capture(const char *path, sbuf *u) {
    size_t n;
    char *raw = slurp(path, &n);
    if (!raw) return false;
    char err[256];
    js *w = js_parse(raw, n, err, sizeof err);
    free(raw);
    if (!w) return false;
    const js *body = js_get(w, "body");
    js *pb = NULL;
    if (body && body->t == JS_STR) { pb = js_parse(body->s, body->len, err, sizeof err); body = pb; }
    chat_req r;
    bool ok = body && chat_from_anthropic(body, &r, err, sizeof err);
    if (ok) {
        sbuf pr = {0};
        chat_render(&r, &pr);
        sb_put(u, pr.p, pr.n);
        chat_unmark(u);
        sb_free(&pr);
        chat_free(&r);
    }
    js_free(pb);
    js_free(w);
    return ok;
}
static void test_capture(const char *dir) {
    char pm[1024], ps[1024], ps2[1024];
    snprintf(pm, sizeof pm, "%s/cap-c_agent/0001-POST-main.json", dir);
    snprintf(ps, sizeof ps, "%s/cap-c_agent/0002-POST-sub.json", dir);
    snprintf(ps2, sizeof ps2, "%s/cap-c2_agent_haiku/0002-POST-sub.json", dir);
    sbuf m = {0}, s = {0}, s2 = {0};
    if (!render_capture(pm, &m) || !render_capture(ps, &s)) { fprintf(stderr, "（沒有抓下來的請求：略過）\n"); return; }
    ts_reset();
    size_t om[PERSIST_MAX], os[PERSIST_MAX];
    const int nm = persist_offsets(m.p, m.n, om), ns = persist_offsets(s.p, s.n, os);
    size_t l = 0;
    while (l < m.n && l < s.n && m.p[l] == s.p[l]) l++;
    fprintf(stderr, "[capture] 主對話 %zu bytes、subagent %zu bytes；共用前綴 %zu bytes（之後：…%.40s…）\n", m.n, s.n, l, s.p + l);
    fprintf(stderr, "[capture] 主對話的點：");
    for (int i = 0; i < nm; i++) fprintf(stderr, " %zu", om[i]);
    fprintf(stderr, "；subagent 的點：");
    for (int i = 0; i < ns; i++) fprintf(stderr, " %zu", os[i]);
    fprintf(stderr, "\n");
    CHECK(ns >= 2 && os[0] <= l && l - os[0] < 6000); // 第一個點（與主對話共用的工具）在共用前綴裡、而且接近終點
    CHECK(ns >= 2 && !memcmp(s.p + os[0], "\n{", 2));
    const char *rm = strstr(s.p, "<|im_start|>user\n");
    CHECK(rm && os[ns - 1] > (size_t)(rm - s.p) && !memcmp(s.p + os[ns - 1] - 18, "</system-reminder>", 18));
    // 另一個 session 的 subagent（不同任務）：到 reminder 結尾都一樣嗎（同一種 subagent 平行派出時共用到這裡）
    if (render_capture(ps2, &s2)) {
        size_t l2 = 0;
        while (l2 < s.n && l2 < s2.n && s.p[l2] == s2.p[l2]) l2++;
        fprintf(stderr, "[capture] 兩個 subagent（不同 session）共用前綴 %zu bytes；reminder 結尾在 %zu\n", l2, os[ns - 1]);
    }
    sb_free(&m); sb_free(&s); sb_free(&s2);
    ts_reset();
}

static void test_wait_est(void) {
    double ah;
    // 沒有別人：只有自己（1000 tok/s）
    double t = wait_est(NULL, 0, 30000, false, 1.0, &ah);
    CHECK(fabs(t - 30.0) < 1e-6 && ah == 0);
    // 有人在生成：prefill 只分到一半
    t = wait_est(NULL, 0, 30000, true, 1.0, &ah);
    CHECK(fabs(t - 60.0) < 1e-6);
    // 剩最少的先算：比自己多的只算到和自己一樣多
    const int rem[3] = {500, 60000, 40000};
    t = wait_est(rem, 3, 1000, true, 1.0, &ah);
    CHECK(fabs(ah - (500 + 1000 + 1000) / 500.0) < 1e-6 && fabs(t - (2500 + 1000) / 500.0) < 1e-6);
    // 還沒量過速度：0.8 tok/ms
    t = wait_est(NULL, 0, 800, false, 0, &ah);
    CHECK(fabs(t - 1.0) < 1e-6);
}

static void test_rc(void) {
    int32_t a[2000], b[2000];
    for (int i = 0; i < 2000; i++) a[i] = b[i] = i;
    b[1500] = -1;
    CHECK(rc_lcp(a, 2000) == 0);
    rc_put(a, 100); // 太短：不記
    CHECK(rc_lcp(a, 2000) == 0);
    rc_put(a, 1200);
    CHECK(rc_lcp(a, 2000) == 1200 && rc_lcp(b, 2000) == 1200);
    rc_put(a, 2000); // 同一個對話更長：換掉（不另佔位置）
    int used = 0;
    for (int i = 0; i < RC_MAX; i++) used += rc[i].t != NULL;
    CHECK(used == 1);
    CHECK(rc_lcp(b, 2000) == 1500 && rc_lcp(a, 1800) == 1800);
    rc_put(a, 1000); // 已經有更長的：不變
    CHECK(rc_lcp(a, 2000) == 2000);
    int32_t c[300];
    for (int k = 0; k < RC_MAX + 5; k++) { // 滿了：LRU
        for (int i = 0; i < 300; i++) c[i] = 100000 * (k + 1) + i;
        rc_put(c, 300);
    }
    used = 0;
    for (int i = 0; i < RC_MAX; i++) used += rc[i].t != NULL;
    CHECK(used == RC_MAX);
    CHECK(rc_lcp(c, 300) == 300);
}

static void test_rej(void) {
    const unsigned long long h = 42;
    CHECK(!rej_overdue(h, 1000, 180)); // 第一次：記下
    CHECK(!rej_overdue(h, 1100, 180));
    CHECK(rej_overdue(h, 1181, 180)); // 超過 180 秒：收（並清掉）
    CHECK(!rej_overdue(h, 1182, 180)); // 又被拒絕：重新起算
    CHECK(!rej_overdue(h, 1300, 180));
    CHECK(!rej_overdue(h, 1182 + 4 * 180 + 1, 180)); // 很久以前的拒絕：重新算
    CHECK(!rej_overdue(h, 1182 + 4 * 180 + 100, 180));
}

static job *fake_job(int np, int est) {
    job *j = calloc(1, sizeof *j);
    j->np = np;
    j->ids = calloc((size_t)np, sizeof(int32_t));
    for (int i = 0; i < np; i++) j->ids[i] = 7 * i + np;
    j->est_new = est;
    return j;
}

static void test_adm(void) {
    MAX_WAIT_S = 90;
    pf_rate = 1.0; // 1000 tok/s
    // 進行中：兩個在生成、一個在分段 prefill（剩 60000）
    pthread_mutex_lock(&st_mu);
    st_n = 3;
    st_job[0] = (typeof(st_job[0])){.id = 1, .rem = 0, .state = 'b'};
    st_job[1] = (typeof(st_job[0])){.id = 2, .rem = 0, .state = 'b'};
    st_job[2] = (typeof(st_job[0])){.id = 3, .rem = 60000, .state = 'p'};
    pthread_mutex_unlock(&st_mu);
    int ra = 0;
    double est, ah;
    job *small = fake_job(20000, 300);
    CHECK(!adm_reject(small, &ra, &est, &ah)); // 短的：剩最少的先算，很快
    job *big = fake_job(40000, 40000);
    CHECK(adm_reject(big, &ra, &est, &ah)); // (40000 + 40000) / 500 = 160 s > 90，前面佔 80 s
    CHECK(ra == 60 && fabs(est - 160) < 1e-6 && fabs(ah - 80) < 1e-6);
    // 只有自己很長（前面沒有 prefill）：照收
    st_job[2].rem = 0;
    job *huge = fake_job(200000, 200000);
    CHECK(!adm_reject(huge, &ra, &est, &ah) && est > 90);
    // 排隊中的也算
    st_job[2].rem = 30000;
    job *q = fake_job(50000, 50000);
    q_head = q_tail = q;
    CHECK(adm_reject(big, &ra, &est, &ah) && fabs(ah - (30000 + 40000) / 500.0) < 1e-6);
    q_head = q_tail = NULL;
    // 關掉
    MAX_WAIT_S = 0;
    CHECK(!adm_reject(big, &ra, &est, &ah));
    MAX_WAIT_S = 90;
    // 同一個 prompt 一直被拒絕：超過 2 × MAX_WAIT_S 後照收（用真的時間：先把第一次拒絕的時間改早）
    st_job[2].rem = 60000;
    job *r = fake_job(41000, 41000);
    CHECK(adm_reject(r, &ra, &est, &ah));
    const unsigned long long h = ids_hash(r->ids, r->np);
    for (int i = 0; i < REJ_MAX; i++) if (rej[i].h == h) rej[i].t -= 200;
    CHECK(!adm_reject(r, &ra, &est, &ah));
    pthread_mutex_lock(&st_mu);
    st_n = 0;
    pthread_mutex_unlock(&st_mu);
    job *all[] = {small, big, huge, q, r};
    for (int i = 0; i < 5; i++) { free(all[i]->ids); free(all[i]); }
}

static void test_retry_after_header(void) {
    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    conn c;
    conn_init(&c, sv[0]);
    respond_err_x(&c, 529, "retry-after: 17\r\n", "overloaded: wait");
    pump_drain(&c);
    char buf[1024] = {0};
    read(sv[1], buf, sizeof buf - 1);
    CHECK(!strncmp(buf, "HTTP/1.1 529 Overloaded\r\n", 25));
    CHECK(strstr(buf, "\r\nretry-after: 17\r\nConnection: close\r\n\r\n"));
    CHECK(strstr(buf, "\"type\":\"overloaded_error\""));
    const char *body = strstr(buf, "\r\n\r\n") + 4;
    const char *cl = strstr(buf, "Content-Length: ");
    CHECK(cl && (size_t)atoi(cl + 16) == strlen(body));
    conn_free(&c);
    close(sv[0]);
    close(sv[1]);
}

static void test_sched_params(void) {
    PF_SEG_MS = 200;
    DEC_MIN_MS = 200;
    pf_seg_ms = 120;
    CHECK(pf_gap_ms() == 200); // 一段很短：生成至少 200 ms
    pf_seg_ms = 450;
    CHECK(pf_gap_ms() == 450); // 一段很長：生成至少一樣長（≥ 一半的時間）
    pf_rate = 0;
    pf_rate_note(100, 50); // 太少：不量
    CHECK(pf_rate == 0);
    pf_rate_note(1000, 500);
    CHECK(fabs(pf_rate - 2.0) < 1e-9);
    pf_rate_note(1000, 1000);
    CHECK(fabs(pf_rate - (0.7 * 2.0 + 0.3 * 1.0)) < 1e-9);
}

// st_publish 的 rem、st_live 即時更新 generated、/status 的新欄位
static void test_status(void) {
    PAR = 4;
    job *a = fake_job(1000, 900), *b = fake_job(5000, 5000), *d = fake_job(3000, 3000);
    a->id = 11; a->st.prompt = 1000; a->inb = true; a->got = 5;
    b->id = 12; b->pfing = true; b->pfd = 0;
    d->id = 13; d->pfing = true; d->pfd = 1024;
    a->g.model = b->g.model = d->g.model = "m";
    job *act[3] = {a, b, d};
    st_publish(act, 3, "batch");
    CHECK(st_job[0].rem == 0 && st_job[1].rem == 5000 && st_job[2].rem == 3000 - 1024);
    a->got = 40;
    st_live(a); // 第一次一定更新（last = 0）
    CHECK(st_job[0].got == 40);
    a->got = 41;
    st_live(a); // 0.2 秒內：不更新
    CHECK(st_job[0].got == 40);
    // 排隊中一個、還沒到上限 → busy
    job *q = fake_job(100, 100);
    q_head = q_tail = q;
    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    conn c;
    conn_init(&c, sv[0]);
    handle_status(&c, "");
    pump_drain(&c);
    char *buf = calloc(1, 1 << 16);
    size_t got = 0;
    shutdown(sv[0], SHUT_WR);
    for (ssize_t k; (k = read(sv[1], buf + got, (1 << 16) - 1 - got)) > 0;) got += (size_t)k;
    const char *body = strstr(buf, "\r\n\r\n");
    CHECK(body != NULL);
    char err[256];
    js *s = body ? js_parse(body + 4, strlen(body + 4), err, sizeof err) : NULL;
    CHECK(s != NULL);
    if (s) {
        const js *qr = js_get(s, "queue_reason");
        CHECK(qr && qr->t == JS_STR && !strcmp(qr->s, "busy"));
        CHECK(js_num(s, "live_output_tokens", -1) == 40);
        CHECK(js_num(s, "prefill_backlog_tokens", -1) == 5000 + 3000 - 1024);
        CHECK(js_get(s, "mem_level") && js_get(s, "swap_used_gb") && js_get(s, "rejected"));
        const js *ac = js_get(s, "active");
        CHECK(ac && ac->t == JS_ARR && ac->n == 3);
        if (ac && ac->kid) CHECK(js_num(ac->kid, "generated", -1) == 40 && js_num(ac->kid, "prefill_left", -1) == 0);
        const js *rt = js_get(s, "rate");
        CHECK(rt && js_get(rt, "prefill"));
        fprintf(stderr, "[status] mem_level %g、swap_used_gb %g\n", js_num(s, "mem_level", -1), js_num(s, "swap_used_gb", -1));
        js_free(s);
    }
    // 到上限 → limit；沒有排隊 → null
    job *e = fake_job(10, 10);
    e->id = 14; e->st.prompt = 10; e->g.model = "m";
    job *act4[4] = {a, b, d, e};
    st_publish(act4, 4, "batch");
    conn_free(&c);
    close(sv[0]);
    close(sv[1]);
    for (int pass = 0; pass < 2; pass++) {
        if (pass) q_head = q_tail = NULL;
        socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
        conn_init(&c, sv[0]);
        handle_status(&c, "");
        pump_drain(&c);
        shutdown(sv[0], SHUT_WR);
        memset(buf, 0, 1 << 16);
        got = 0;
        for (ssize_t k; (k = read(sv[1], buf + got, (1 << 16) - 1 - got)) > 0;) got += (size_t)k;
        CHECK(strstr(buf, pass ? "\"queue_reason\":null" : "\"queue_reason\":\"limit\""));
        conn_free(&c);
        close(sv[0]);
        close(sv[1]);
    }
    free(buf);
    pthread_mutex_lock(&st_mu);
    st_n = 0;
    pthread_mutex_unlock(&st_mu);
    job *all[] = {a, b, d, q, e};
    for (int i = 0; i < 5; i++) { free(all[i]->ids); free(all[i]); }
}

// ---- 合併前綴相同的 prefill（coal_plan / pf_pick / coal_after / qw_engine_par_clip）
// 前 shared 個 token 相同（family 決定內容），之後各自不同（seed）
static job *coal_job(int id, int np, int shared, int family, int seed) {
    job *j = calloc(1, sizeof *j);
    j->id = id;
    j->np = np;
    j->sid = -1; // 不碰 engine（qw_engine_par_snap 對無效的 sid 直接回 false）
    j->ids = calloc((size_t)np, sizeof(int32_t));
    for (int i = 0; i < np; i++) j->ids[i] = i < shared ? 1000003 * family + i : 7000000 + seed * 100003 + i;
    j->pfing = true;
    j->t0 = 100.0 + id; // id 小的先來
    return j;
}
static void coal_free(job **a, int n) { for (int i = 0; i < n; i++) { free(a[i]->ids); free(a[i]); } }
// 模擬 pf_chunk：engine 的段尾規則（剩 ≤ ch + 64 就算到底）+ 交接點截斷；複製時從交接點接續
static void coal_sim_chunk(job *j, int ch) {
    if (j->coal_copy) {
        j->pfd = 0;
        const int bl = j->lead_at; // 領頭的在交接點記了快照（測試裡假設還在）
        coal_after(j, bl);
        j->pfd = bl;
    }
    const int start = j->pfd;
    int end = j->np - start > ch + 64 ? start + ch : j->np;
    end = qw_engine_par_clip(0, start, end, j->coal_pts, j->ncoal_pts, 64);
    j->pfd = end;
    if (j->pfd >= j->np) j->pfing = false;
}

static void test_coal_clip(void) {
    const int pts[3] = {13952, 9000, 20000};
    CHECK(qw_engine_par_clip(0, 4096, 12288, pts, 3, 64) == 8960); // 9000 → 8960（64 的倍數），比 13952 近
    CHECK(qw_engine_par_clip(0, 8960, 17152, pts, 3, 64) == 13952);
    CHECK(qw_engine_par_clip(0, 13952, 19968, pts, 3, 64) == 19968); // 20000 對齊成 19968 = 段尾：不變
    CHECK(qw_engine_par_clip(0, 13952, 20032, pts, 3, 64) == 19968);
    CHECK(qw_engine_par_clip(0, 13952, 16000, pts, 0, 64) == 16000); // 沒有交接點
    CHECK(qw_engine_par_clip(100, 100, 9000, pts + 1, 1, 64) == 100 + (9000 - 100) / 64 * 64); // 照第一段的起點對齊
    CHECK(qw_engine_par_clip(9500, 9500, 12000, pts + 1, 1, 64) == 12000); // 交接點在起點之前：不管
}

static void test_coal_plan(void) {
    COAL_MIN = 2048;
    st_coal = st_coal_n = 0;
    enum { SH = 14000, NP = 16000, AT = SH / 64 * 64 };
    job *a[4];
    for (int i = 0; i < 4; i++) a[i] = coal_job(i + 1, NP, SH, 1, i);
    coal_plan(a, 4);
    CHECK(a[0]->lead == 0);
    for (int i = 1; i < 4; i++) CHECK(a[i]->lead == 1 && a[i]->lead_at == AT && !a[i]->coal_copy);
    CHECK(a[0]->ncoal_pts == 1 && a[0]->coal_pts[0] == AT);
    CHECK(pf_pick(a, 4, 200) == a[0]); // 只有領頭的可以算
    // 領頭的算到一半：還在等
    coal_sim_chunk(a[0], 8192);
    CHECK(a[0]->pfd == 8192);
    coal_plan(a, 4);
    for (int i = 1; i < 4; i++) CHECK(a[i]->lead == 1 && !a[i]->coal_copy);
    CHECK(pf_pick(a, 4, 200) == a[0]);
    coal_sim_chunk(a[0], 8192); // 8192 + 8192 > AT：段尾停在交接點
    CHECK(a[0]->pfd == AT);
    coal_plan(a, 4);
    for (int i = 1; i < 4; i++) CHECK(a[i]->coal_copy && a[i]->coal_done);
    CHECK(a[0]->ncoal_pts == 0);
    // 跟的先複製（最早來的先），領頭的暫停到它們都複製完
    for (int k = 1; k < 4; k++) {
        job *p = pf_pick(a, 4, 200);
        CHECK(p == a[k]);
        if (p != a[k]) break;
        coal_sim_chunk(p, 1024);
        CHECK(p->lead == 0 && !p->coal_copy && p->pfd == AT + 1024);
        coal_plan(a, 4);
    }
    CHECK(st_coal == 3LL * AT && st_coal_n == 3);
    CHECK(!coal_held(a[0], a, 4));
    coal_plan(a, 4); // 跟過一次就不再跟
    for (int i = 1; i < 4; i++) CHECK(a[i]->lead == 0);
    coal_free(a, 4);

    // 領頭的被取消：剩下的重新選（進度相同時 id 小的當領頭）
    job *b[3];
    for (int i = 0; i < 3; i++) b[i] = coal_job(10 + i, NP, SH, 2, i);
    coal_plan(b, 3);
    CHECK(b[1]->lead == 10 && b[2]->lead == 10);
    job *rest[2] = {b[1], b[2]};
    coal_plan(rest, 2);
    CHECK(b[1]->lead == 0 && b[2]->lead == 11 && b[2]->lead_at == AT);
    CHECK(pf_pick(rest, 2, 200) == b[1]);
    // 領頭的已經算完 prompt（在生成）：跟的那個下一段直接複製
    b[1]->pfd = NP;
    b[1]->pfing = false;
    coal_plan(rest, 2);
    CHECK(b[2]->coal_copy && pf_pick(rest, 2, 200) == b[2]);
    coal_free(b, 3);

    // 不跟：共用太短、領頭的已經過了交接點、自己的進度差不多、COAL_MIN = 0
    job *c[2] = {coal_job(20, NP, 1000, 3, 0), coal_job(21, NP, 1000, 3, 1)};
    coal_plan(c, 2);
    CHECK(c[1]->lead == 0);
    coal_free(c, 2);
    job *d[2] = {coal_job(30, NP, SH, 4, 0), coal_job(31, NP, SH, 4, 1)};
    d[0]->pfd = 14500;
    coal_plan(d, 2);
    CHECK(d[1]->lead == 0);
    d[0]->pfd = 8192;
    d[1]->pfd = 12800; // 進度比較多的當領頭：d[1]（12800）領、d[0] 跟（交接點 13952 ≥ 8192 + 2048）
    coal_plan(d, 2);
    CHECK(d[0]->lead == 31 && d[1]->lead == 0);
    coal_free(d, 2);
    job *f[2] = {coal_job(40, NP, SH, 5, 0), coal_job(41, NP, SH, 5, 1)};
    COAL_MIN = 0;
    coal_plan(f, 2);
    CHECK(f[1]->lead == 0 && pf_pick(f, 2, 200) == f[0]);
    COAL_MIN = 2048;
    // 自己是整個 prompt 都相同（重試）：交接點留最後一個 token 自己算
    job *g[2] = {coal_job(50, NP, NP, 6, 0), coal_job(51, NP, NP, 6, 0)};
    coal_plan(g, 2);
    CHECK(g[1]->lead == 50 && g[1]->lead_at == (NP - 1) / 64 * 64);
    coal_free(f, 2);
    coal_free(g, 2);
}

// 隨機模擬：幾組前綴相同的 prompt 同時進來、隨機取消。有人在 prefill 時一定選得出一個（不會卡死），每個都算完，
// 而且共用前綴只算一次左右
static void test_coal_sim(void) {
    COAL_MIN = 2048;
    unsigned rs = 12345;
    for (int round = 0; round < 200; round++) {
        st_coal = st_coal_n = 0;
        const int n = 2 + (int)((rs = rs * 1103515245u + 12345u) >> 16) % 14;
        job *a[16];
        int alive = n, id0 = 1000 * round;
        long long naive = 0;
        for (int i = 0; i < n; i++) {
            rs = rs * 1103515245u + 12345u;
            const int fam = (int)(rs >> 16) % 3, np = 6000 + (int)(rs >> 8) % 20000, sh = fam == 0 ? 0 : 3000 + fam * 5000;
            a[i] = coal_job(id0 + i, np, sh < np ? sh : np, fam, round * 100 + i);
            naive += np;
        }
        long long computed = 0;
        int it = 0;
        for (; alive && it < 10000; it++) {
            coal_plan(a, alive);
            job *p = pf_pick(a, alive, 200 + it);
            int npf = 0;
            for (int i = 0; i < alive; i++) npf += a[i]->pfing;
            if (npf && !p) { CHECK(!"pf_pick: 有人在 prefill 卻選不出來（卡死）"); break; }
            if (p) {
                CHECK(!coal_waiting(p));
                const int d0 = p->coal_copy ? p->lead_at : p->pfd;
                coal_sim_chunk(p, 1024 + 64 * (it % 7));
                computed += p->pfd - d0;
            }
            rs = rs * 1103515245u + 12345u;
            int keep = 0;
            for (int i = 0; i < alive; i++) { // 算完的離開；偶爾取消一個
                const bool cancel = (rs >> 20) % 97 == 0 && i == (int)(rs >> 8) % alive;
                if (!a[i]->pfing || cancel) { free(a[i]->ids); free(a[i]); }
                else a[keep++] = a[i];
            }
            alive = keep;
        }
        CHECK(!alive);
        CHECK(computed <= naive);
        if (round < 3) fprintf(stderr, "[coal] %d 個請求：直接算 %lld 個 token，合併後算 %lld 個（複製 %lld 個，%lld 個跟過）\n", n, naive, computed, st_coal, st_coal_n);
        coal_free(a, alive);
    }
}

// /status：請求結束時 output_tokens 與 active[].generated 同時換手（Σ 不跳）
static void test_status_handoff(void) {
    PAR = 4;
    job *a = fake_job(100, 100), *b = fake_job(100, 100);
    a->id = 61; b->id = 62;
    a->sid = b->sid = -1;
    a->st.prompt = b->st.prompt = 100;
    a->inb = b->inb = true;
    a->g.model = b->g.model = "m";
    conn c;
    conn_init(&c, -1);
    a->c = b->c = &c;
    a->got = 300; // 含思考與收尾的 token：生成的每個 token 都在 got 裡
    b->got = 20;
    job *act[2] = {a, b};
    st_publish(act, 2, "batch");
    pthread_mutex_lock(&st_mu);
    long long live0 = 0;
    for (int i = 0; i < st_n; i++) live0 += st_job[i].got;
    const long long sum0 = st_out + live0;
    pthread_mutex_unlock(&st_mu);
    par_done(a, QW_FIN_STOP); // 還沒有下一次 st_publish
    pthread_mutex_lock(&st_mu);
    long long live1 = 0;
    for (int i = 0; i < st_n; i++) live1 += st_job[i].got;
    CHECK(st_out + live1 == sum0);
    CHECK(st_n == 1 && st_job[0].id == 62);
    pthread_mutex_unlock(&st_mu);
    CHECK(a->out_moved && a->st.gen == 300);
    st_n = 0;
    conn_free(&c);
    free(a->ids); free(a); free(b->ids); free(b);
}

int main(int argc, char **argv) {
    test_reminder_end();
    test_toolset_diverge();
    test_persist_offsets();
    test_capture(argc > 1 ? argv[1] : ".");
    test_wait_est();
    test_rc();
    test_rej();
    test_adm();
    test_retry_after_header();
    test_sched_params();
    test_status();
    test_coal_clip();
    test_coal_plan();
    test_coal_sim();
    test_status_handoff();
    fprintf(stderr, fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
