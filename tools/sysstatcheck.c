// tools/sysstat.h 的測試：印兩次 "sys" 物件（間隔 1 秒），再連續呼叫 N 次看 RSS 有沒有長（IOKit / CF 漏掉的話會長）
// 用法：build/sysstatcheck [cache 目錄（預設 cache）] [N（預設 2000）]；通過回傳 0
#include "sysstat.h"

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "cache";
    const int n = argc > 2 ? atoi(argv[2]) : 2000;
    char b[512];
    for (int i = 0; i < 2; i++) {
        if (i) sleep(1);
        sys_json(b, sizeof b, dir);
        printf("%s\n", b + 1);
    }
    double t0 = sys_now();
    sys_json(b, sizeof b, dir);
    const double one = sys_now() - t0;
    for (int i = 0; i < 50; i++) sys_json(b, sizeof b, dir); // 暖身
    const unsigned long long fp0 = sys_proc_bytes();
    t0 = sys_now();
    for (int i = 0; i < n; i++) sys_json(b, sizeof b, dir);
    const double per = (sys_now() - t0) / (n ? n : 1);
    const unsigned long long fp1 = sys_proc_bytes();
    const long long grow = (long long)fp1 - (long long)fp0;
    printf("single call %.3f ms; %d calls: %.3f ms/call, phys_footprint %.2f MB -> %.2f MB (%+lld bytes)\n",
           one * 1e3, n, per * 1e3, fp0 / 1e6, fp1 / 1e6, grow);
    const bool ok = grow < 256 * 1024;
    printf("%s\n", ok ? "OK" : "FAIL: footprint grew");
    return ok ? 0 : 1;
}
