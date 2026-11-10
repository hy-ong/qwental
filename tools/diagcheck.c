// tools/diag.h 的測試：每個情境開一個子程序（diag_install 之後故意出事），收它的 stderr 與結束狀態
// 用法：build/diagcheck（全部通過回傳 0）
#include "diag.h"
#include <fcntl.h>
#include <sys/wait.h>

static volatile int *nullp;
static int stack_eat(int d) { // 遞迴到 stack 用完（測 sigaltstack）
    volatile char pad[4096];
    pad[0] = (char)d;
    return d > 1000000 ? pad[0] : stack_eat(d + 1) + pad[0];
}

static void scenario(const char *name) {
    diag_install(0);
    if (!strcmp(name, "segv")) { *nullp = 1; }
    else if (!strcmp(name, "abort")) abort();
    else if (!strcmp(name, "bus")) raise(SIGBUS);
    else if (!strcmp(name, "fpe")) raise(SIGFPE);
    else if (!strcmp(name, "ill")) raise(SIGILL);
    else if (!strcmp(name, "trap")) __builtin_trap();
    else if (!strcmp(name, "overflow")) (void)stack_eat(0);
    else if (!strcmp(name, "term")) { kill(getpid(), SIGTERM); pause(); }
    else if (!strcmp(name, "int")) { kill(getpid(), SIGINT); pause(); }
    else if (!strcmp(name, "hup")) { kill(getpid(), SIGHUP); pause(); }
    else if (!strcmp(name, "xcpu")) { kill(getpid(), SIGXCPU); pause(); }
    else if (!strcmp(name, "xfsz")) { kill(getpid(), SIGXFSZ); pause(); }
    else if (!strcmp(name, "exit")) { fprintf(stderr, "❌ 測試用的失敗\n"); exit(3); }
    else if (!strcmp(name, "mem")) { // 配置並碰 3 GB：定時紀錄應該寫一行（變化 ≥ 2 GB）；再 diag_mem("…") 一定寫
        const size_t sz = 3ull << 30;
        volatile char *p = malloc(sz); // volatile：不讓編譯器把配置與寫入整個拿掉
        for (size_t i = 0; i < sz; i += 16384) p[i] = 1;
        diag_mem(NULL);
        diag_mem(NULL); // 沒變：不寫
        diag_mem("[mem] 測試");
        free((void *)p);
        _exit(0);
    }
    _exit(99);
}

typedef struct { const char *name; int sig; int code; const char *want[3]; } tcase;

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--child")) { scenario(argv[2]); return 99; }
    const tcase T[] = {
        {"segv", SIGSEGV, 0, {"[diag] 收到 SIGSEGV", "[diag] 堆疊：", "main"}},
        {"abort", SIGABRT, 0, {"[diag] 收到 SIGABRT", "[diag] 堆疊：", "abort"}},
        {"bus", SIGBUS, 0, {"[diag] 收到 SIGBUS", "[diag] 堆疊：", NULL}},
        {"fpe", SIGFPE, 0, {"[diag] 收到 SIGFPE", "[diag] 堆疊：", NULL}},
        {"ill", SIGILL, 0, {"[diag] 收到 SIGILL", "[diag] 堆疊：", NULL}},
        {"trap", SIGTRAP, 0, {"[diag] 收到 SIGTRAP", "[diag] 堆疊：", "main"}}, // arm64 的 __builtin_trap 是 brk → SIGTRAP
        {"overflow", SIGSEGV, 0, {"[diag] 收到 SIGSEGV", "[diag] 堆疊：", NULL}},
        {"term", SIGTERM, 0, {"[diag] 收到 SIGTERM", "來自 pid", "footprint"}},
        {"int", SIGINT, 0, {"[diag] 收到 SIGINT", NULL, NULL}},
        {"hup", SIGHUP, 0, {"[diag] 收到 SIGHUP", NULL, NULL}},
        {"xcpu", SIGXCPU, 0, {"[diag] 收到 SIGXCPU", NULL, NULL}},
        {"xfsz", SIGXFSZ, 0, {"[diag] 收到 SIGXFSZ", NULL, NULL}},
        {"exit", 0, 3, {"❌ 測試用的失敗", "[diag] 程序以 exit() 結束", "系統可用記憶體"}},
        {"mem", 0, 0, {"[diag] 記憶體（啟動）", "[diag] 記憶體：footprint 3.", "[diag] 記憶體（[mem] 測試）"}},
    };
    int fail = 0;
    for (size_t t = 0; t < sizeof T / sizeof *T; t++) {
        char path[] = "/tmp/diagcheck.XXXXXX";
        const char *tmp = getenv("TMPDIR");
        char pbuf[1024];
        snprintf(pbuf, sizeof pbuf, "%s/diagcheck.XXXXXX", tmp ? tmp : "/tmp");
        const int fd = mkstemp(tmp ? pbuf : path);
        const char *fn = tmp ? pbuf : path;
        const pid_t pid = fork();
        if (!pid) {
            dup2(fd, STDERR_FILENO);
            execl(argv[0], argv[0], "--child", T[t].name, (char *)NULL);
            _exit(98);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        char out[16384] = {0};
        lseek(fd, 0, SEEK_SET);
        const ssize_t n = read(fd, out, sizeof out - 1);
        if (n > 0) out[n] = 0;
        close(fd);
        unlink(fn);
        bool ok = T[t].sig ? (WIFSIGNALED(st) && WTERMSIG(st) == T[t].sig) : (WIFEXITED(st) && WEXITSTATUS(st) == T[t].code);
        for (int k = 0; k < 3; k++)
            if (T[t].want[k] && !strstr(out, T[t].want[k])) { ok = false; printf("  缺少：%s\n", T[t].want[k]); }
        if (!strcmp(T[t].name, "mem")) { // 第二次 diag_mem(NULL) 沒變化：記憶體行應該剛好 3 行
            int lines = 0;
            for (const char *p = out; (p = strstr(p, "[diag] 記憶體")); p++) lines++;
            if (lines != 3) { ok = false; printf("  記憶體行 %d 行（應為 3）\n", lines); }
        }
        printf("%s %-8s 結束狀態：%s %d\n", ok ? "✅" : "❌", T[t].name, WIFSIGNALED(st) ? "signal" : "exit",
               WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
        if (!ok || getenv("DIAG_VERBOSE")) printf("----\n%s----\n", out);
        fail += !ok;
    }
    printf(fail ? "%d 個失敗\n" : "全部通過\n", fail);
    return fail != 0;
}
