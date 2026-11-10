// 診斷：程序結束前的「遺言」（qwental-serve 用；tools/diagcheck.c 單獨測）
// server 曾兩次無聲消失（log 沒有 SIGTERM、沒有錯誤、沒有 crash report）。這裡把能接的結束路徑都寫一行到 stderr（= server log）：
//   - SIGSEGV / SIGBUS / SIGILL / SIGFPE / SIGABRT / SIGTRAP（arm64 的 __builtin_trap、Swift / ObjC runtime 的 trap）：只用 write(2) 與預先格式化的文字（async-signal-safe）+ backtrace_symbols_fd，
//     然後恢復預設行為再送一次，讓系統照樣產生 crash report
//   - SIGHUP / SIGXCPU / SIGXFSZ：同上（寫一行、恢復預設、再送一次）。SIGPIPE 維持忽略（socket 斷線很常見，不是結束原因）
//   - SIGTERM / SIGINT：serve.c 的 lt_start 用 dispatch source 先存累計數字再結束，那裡呼叫 diag_signal_note；
//     沒有累計檔時（lt_start 不裝）由這裡的 handler 接
//   - exit()（含 engine.c 失敗時的 exit(1)、main 的 return）：atexit 寫一行
//   - 記憶體：每 60 秒看一次 phys_footprint 與 kern.memorystatus_level，有明顯變化才寫；[mem] 動作時 diag_mem("…") 也寫
// SIGKILL（jetsam / 記憶體壓力）接不到：那時什麼都不會寫，launcher 的 log（server 意外結束、wait status）就是證據。
#pragma once
#include <dispatch/dispatch.h>
#include <execinfo.h>
#include <mach/mach.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

static long diag_gmtoff;                  // 本地時區（安裝時算好；handler 裡不能呼叫 localtime）
static _Atomic long long diag_fp_last = -1; // 最近量到的 phys_footprint（bytes）
static _Atomic int diag_lvl_last = -1;      // 最近量到的 kern.memorystatus_level（%）
static _Atomic int diag_lvl_logged = -1;    // 上次寫進 log 的 level（判斷有沒有明顯變化）

// ---- async-signal-safe 的小工具：只寫進呼叫者的 buffer
static size_t diag_put(char *b, size_t n, size_t cap, const char *s) {
    while (*s && n + 1 < cap) b[n++] = *s++;
    b[n] = 0;
    return n;
}
static size_t diag_putnum(char *b, size_t n, size_t cap, long long v, int width) { // width：補 0 到幾位
    char t[24];
    int k = 0;
    const bool neg = v < 0;
    unsigned long long u = neg ? (unsigned long long)(-v) : (unsigned long long)v;
    do { t[k++] = (char)('0' + u % 10); u /= 10; } while (u && k < 23);
    while (k < width && k < 23) t[k++] = '0';
    if (neg && n + 1 < cap) b[n++] = '-';
    while (k && n + 1 < cap) b[n++] = t[--k];
    b[n] = 0;
    return n;
}
static size_t diag_stamp(char *b, size_t n, size_t cap) { // "2026-11-10 12:34:56 "（本地時間）
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    long long s = (long long)ts.tv_sec + diag_gmtoff;
    long long days = s / 86400, r = s % 86400;
    if (r < 0) { r += 86400; days--; }
    // civil_from_days（Howard Hinnant）
    days += 719468;
    const long long era = (days >= 0 ? days : days - 146096) / 146097;
    const long long doe = days - era * 146097;
    const long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const long long mp = (5 * doy + 2) / 153;
    const long long d = doy - (153 * mp + 2) / 5 + 1, m = mp < 10 ? mp + 3 : mp - 9, y = yoe + era * 400 + (m <= 2);
    n = diag_putnum(b, n, cap, y, 4); n = diag_put(b, n, cap, "-");
    n = diag_putnum(b, n, cap, m, 2); n = diag_put(b, n, cap, "-");
    n = diag_putnum(b, n, cap, d, 2); n = diag_put(b, n, cap, " ");
    n = diag_putnum(b, n, cap, r / 3600, 2); n = diag_put(b, n, cap, ":");
    n = diag_putnum(b, n, cap, r / 60 % 60, 2); n = diag_put(b, n, cap, ":");
    n = diag_putnum(b, n, cap, r % 60, 2);
    return diag_put(b, n, cap, " ");
}
static long long diag_footprint(void) { // phys_footprint（Activity Monitor 的「記憶體」）；mach trap，handler 裡也能用
    task_vm_info_data_t vi;
    mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &c) != KERN_SUCCESS) return -1;
    return (long long)vi.phys_footprint;
}
static int diag_level(void) {
    int lvl = -1;
    size_t len = sizeof lvl;
    return sysctlbyname("kern.memorystatus_level", &lvl, &len, NULL, 0) ? -1 : lvl;
}
static size_t diag_putmem(char *b, size_t n, size_t cap, long long fp, int lvl) { // "footprint 12.34 GB，系統可用記憶體 15%"
    n = diag_put(b, n, cap, "footprint ");
    if (fp < 0) n = diag_put(b, n, cap, "?");
    else {
        const long long c = fp / 10000000; // 0.01 GB
        n = diag_putnum(b, n, cap, c / 100, 1); n = diag_put(b, n, cap, ".");
        n = diag_putnum(b, n, cap, c % 100, 2);
    }
    n = diag_put(b, n, cap, " GB，系統可用記憶體 ");
    if (lvl < 0) n = diag_put(b, n, cap, "?");
    else n = diag_putnum(b, n, cap, lvl, 1);
    return diag_put(b, n, cap, "%");
}
static const char *diag_signame(int sig) {
    switch (sig) {
    case SIGSEGV: return "SIGSEGV";
    case SIGBUS: return "SIGBUS";
    case SIGILL: return "SIGILL";
    case SIGFPE: return "SIGFPE";
    case SIGABRT: return "SIGABRT";
    case SIGTRAP: return "SIGTRAP";
    case SIGTERM: return "SIGTERM";
    case SIGINT: return "SIGINT";
    case SIGHUP: return "SIGHUP";
    case SIGPIPE: return "SIGPIPE";
    case SIGXCPU: return "SIGXCPU";
    case SIGXFSZ: return "SIGXFSZ";
    default: return "signal";
    }
}
static void diag_write(const char *b, size_t n) {
    while (n) {
        const ssize_t w = write(STDERR_FILENO, b, n);
        if (w <= 0) return;
        b += w;
        n -= (size_t)w;
    }
}

// 收到 sig 的遺言（async-signal-safe；footprint 現量，可用記憶體用最近一次量到的）
static void diag_signal_note(int sig, const siginfo_t *si) {
    char b[512];
    size_t n = diag_stamp(b, 0, sizeof b);
    n = diag_put(b, n, sizeof b, "[diag] 收到 ");
    n = diag_put(b, n, sizeof b, diag_signame(sig));
    n = diag_put(b, n, sizeof b, "（");
    n = diag_putnum(b, n, sizeof b, sig, 1);
    if (si) {
        n = diag_put(b, n, sizeof b, "，code ");
        n = diag_putnum(b, n, sizeof b, si->si_code, 1);
        if (!(sig == SIGSEGV || sig == SIGBUS || sig == SIGILL || sig == SIGFPE || sig == SIGTRAP) || si->si_code == SI_USER) { // 送來的：寄件人
            n = diag_put(b, n, sizeof b, "，來自 pid ");
            n = diag_putnum(b, n, sizeof b, si->si_pid, 1);
        } else {
            n = diag_put(b, n, sizeof b, "，位址 0x");
            char h[17];
            unsigned long long a = (unsigned long long)(uintptr_t)si->si_addr;
            for (int i = 15; i >= 0; i--) { h[i] = "0123456789abcdef"[a & 15]; a >>= 4; }
            h[16] = 0;
            n = diag_put(b, n, sizeof b, h);
        }
    }
    n = diag_put(b, n, sizeof b, "）；pid ");
    n = diag_putnum(b, n, sizeof b, getpid(), 1);
    n = diag_put(b, n, sizeof b, "，");
    n = diag_putmem(b, n, sizeof b, diag_footprint(), atomic_load(&diag_lvl_last));
    n = diag_put(b, n, sizeof b, "\n");
    diag_write(b, n);
}

static void diag_fatal(int sig, siginfo_t *si, void *uc) {
    (void)uc;
    diag_signal_note(sig, si);
    if (sig == SIGSEGV || sig == SIGBUS || sig == SIGILL || sig == SIGFPE || sig == SIGABRT || sig == SIGTRAP) {
        static const char h[] = "[diag] 堆疊：\n";
        diag_write(h, sizeof h - 1);
        void *f[64];
        const int k = backtrace(f, 64);
        backtrace_symbols_fd(f, k, STDERR_FILENO);
    }
    // 恢復預設行為再送一次：handler 回傳後送達 → 照預設結束（硬體錯誤回傳後也會再發生一次），系統照樣寫 crash report
    signal(sig, SIG_DFL);
    sigset_t m;
    sigemptyset(&m);
    sigaddset(&m, sig);
    pthread_sigmask(SIG_UNBLOCK, &m, NULL);
    raise(sig);
}

static void diag_atexit(void) {
    char b[256];
    size_t n = diag_stamp(b, 0, sizeof b);
    n = diag_put(b, n, sizeof b, "[diag] 程序以 exit() 結束（原因見上一行；pid ");
    n = diag_putnum(b, n, sizeof b, getpid(), 1);
    n = diag_put(b, n, sizeof b, "，");
    n = diag_putmem(b, n, sizeof b, diag_footprint(), diag_level());
    n = diag_put(b, n, sizeof b, "）\n");
    diag_write(b, n);
}

// 記憶體紀錄：why = NULL（定時）只在明顯變化時寫；why 不是 NULL（例如 [mem] 動作）一定寫
static void diag_mem(const char *why) {
    const long long fp = diag_footprint();
    const int lvl = diag_level();
    const long long fp0 = atomic_load(&diag_fp_last);
    const int lvl0 = atomic_load(&diag_lvl_logged);
    const long long dfp = fp - fp0 < 0 ? fp0 - fp : fp - fp0;
    const bool changed = fp0 < 0 || dfp >= 2000000000LL || dfp * 10 >= fp0 || // ≥ 2 GB 或 ≥ 10%
                         (lvl >= 0 && (lvl0 < 0 || abs(lvl - lvl0) >= 5 || ((lvl < 20) != (lvl0 < 20))));
    atomic_store(&diag_lvl_last, lvl);
    if (!why && !changed) return;
    atomic_store(&diag_fp_last, fp);
    atomic_store(&diag_lvl_logged, lvl);
    char b[512];
    size_t n = diag_stamp(b, 0, sizeof b);
    n = diag_put(b, n, sizeof b, "[diag] 記憶體");
    if (why) { n = diag_put(b, n, sizeof b, "（"); n = diag_put(b, n, sizeof b, why); n = diag_put(b, n, sizeof b, "）"); }
    n = diag_put(b, n, sizeof b, "：");
    n = diag_putmem(b, n, sizeof b, fp, lvl);
    n = diag_put(b, n, sizeof b, "\n");
    diag_write(b, n);
}

// 安裝：main 一開始呼叫（在 lt_start 之前：lt_start 會把 SIGTERM / SIGINT 換成 dispatch source）
static void diag_install(int interval_s) {
    tzset();
    const time_t now = time(NULL);
    struct tm lt;
    localtime_r(&now, &lt);
    diag_gmtoff = lt.tm_gmtoff;
    static char altstack[64 * 1024]; // 主執行緒 stack overflow 時 handler 還有地方跑
    stack_t ss = {.ss_sp = altstack, .ss_size = sizeof altstack, .ss_flags = 0};
    sigaltstack(&ss, NULL);
    const int sigs[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT, SIGTRAP, SIGTERM, SIGINT, SIGHUP, SIGXCPU, SIGXFSZ};
    for (size_t i = 0; i < sizeof sigs / sizeof *sigs; i++) {
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_sigaction = diag_fatal;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;
        sigemptyset(&sa.sa_mask);
        sigaction(sigs[i], &sa, NULL);
    }
    { // 先呼叫一次 backtrace：第一次會載入 libunwind 等（handler 裡不要做初次載入）
        void *f[4];
        (void)backtrace(f, 4);
    }
    atexit(diag_atexit);
    diag_mem("啟動");
    if (interval_s > 0) {
        dispatch_queue_t q = dispatch_queue_create("qwental.diag", DISPATCH_QUEUE_SERIAL);
        dispatch_source_t t = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, q);
        dispatch_source_set_timer(t, dispatch_time(DISPATCH_TIME_NOW, (int64_t)interval_s * NSEC_PER_SEC),
                                  (uint64_t)interval_s * NSEC_PER_SEC, NSEC_PER_SEC);
        dispatch_source_set_event_handler(t, ^{ diag_mem(NULL); });
        dispatch_resume(t);
    }
}
