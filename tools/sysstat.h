// 系統即時數字（qwental-serve 的 /status "sys" 物件用；tools/sysstatcheck.c 單獨測）
// dashboard 每秒輪詢一次：全部在回 /status 的執行緒上算，每次 < 1 ms（cache 目錄的大小最多 30 秒重算一次），不開執行緒。
//   cpu_pct      整台機器所有核心的忙碌 %（0–100）：host_processor_info(PROCESSOR_CPU_LOAD_INFO) 與上次的 tick 差；
//                距上次取樣不到 0.5 秒時回上次的值；第一次呼叫和開機以來的 tick 比（= 開機以來的平均，不是亂數）
//   gpu_pct      IOAccelerator 的 PerformanceStatistics["Device Utilization %"]（多個取最大）
//   ram_used_gb  Activity Monitor 的「已使用記憶體」= (internal − purgeable + wired + compressor) 頁 × page size
//                （internal_page_count 是 0 時退回 active + wired + compressor）；ram_total_gb = hw.memsize
//   proc_gb      這個程序的 phys_footprint（Activity Monitor 程序列表的「記憶體」）
//   disk_free_gb / disk_total_gb  statfs(dir)（dir 沒有或還沒建時用目前目錄）：free = f_bavail × f_bsize（一般使用者可用的空間）
//   cache_gb     dir 裡一般檔案的大小合計（SSD 快取的 .qwc 都直接放在 dir 下，不往下走）；dir 是 NULL / "" 時 null
// GB 一律 = 1e9 bytes，和 serve.c 既有的 swap_used_gb、slot_mem_gb 一致（所以 128 GiB 的機器 ram_total_gb ≈ 137.44）。
// 讀不到的欄位輸出 null。
#pragma once
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <dirent.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

static pthread_mutex_t sys_mu = PTHREAD_MUTEX_INITIALIZER; // /status 可能同時被多個連線叫

static double sys_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

// CPU 忙碌 %（-1 = 讀不到）；呼叫者持有 sys_mu
static double sys_cpu_pct(void) {
    static unsigned long long prev_busy, prev_total;
    static double prev_t = -1, last = -1;
    const double t = sys_now();
    if (prev_t >= 0 && t - prev_t < 0.5) return last;
    natural_t ncpu = 0;
    processor_info_array_t info = NULL;
    mach_msg_type_number_t ninfo = 0;
    if (host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &ncpu, &info, &ninfo) != KERN_SUCCESS) return last;
    unsigned long long busy = 0, total = 0;
    const processor_cpu_load_info_t ld = (processor_cpu_load_info_t)info;
    for (natural_t i = 0; i < ncpu; i++) {
        const unsigned long long u = ld[i].cpu_ticks[CPU_STATE_USER], s = ld[i].cpu_ticks[CPU_STATE_SYSTEM],
                                 n = ld[i].cpu_ticks[CPU_STATE_NICE], id = ld[i].cpu_ticks[CPU_STATE_IDLE];
        busy += u + s + n;
        total += u + s + n + id;
    }
    vm_deallocate(mach_task_self(), (vm_address_t)info, (vm_size_t)ninfo * sizeof(integer_t));
    if (total > prev_total && busy >= prev_busy) { // 第一次 prev = 0：和開機以來比
        double p = 100.0 * (double)(busy - prev_busy) / (double)(total - prev_total);
        last = p < 0 ? 0 : p > 100 ? 100 : p;
    }
    prev_busy = busy;
    prev_total = total;
    prev_t = t;
    return last;
}

// GPU 使用率 %（-1 = 讀不到）：所有 IOAccelerator 的 "Device Utilization %" 取最大；CF / IO 物件全部釋放
static double sys_gpu_pct(void) {
    io_iterator_t it = IO_OBJECT_NULL;
    // IOServiceMatching 回傳的 dictionary 由 IOServiceGetMatchingServices 收走（不用 CFRelease）
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) != KERN_SUCCESS) return -1;
    double best = -1;
    for (io_object_t svc; (svc = IOIteratorNext(it));) {
        CFTypeRef ps = IORegistryEntryCreateCFProperty(svc, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
        if (ps) {
            if (CFGetTypeID(ps) == CFDictionaryGetTypeID()) {
                const CFTypeRef v = CFDictionaryGetValue((CFDictionaryRef)ps, CFSTR("Device Utilization %")); // get：不擁有
                long long u = 0;
                if (v && CFGetTypeID(v) == CFNumberGetTypeID() && CFNumberGetValue((CFNumberRef)v, kCFNumberLongLongType, &u) && u >= 0)
                    if ((double)u > best) best = (double)u;
            }
            CFRelease(ps);
        }
        IOObjectRelease(svc);
    }
    IOObjectRelease(it);
    return best > 100 ? 100 : best;
}

// 已使用 / 總記憶體（bytes；0 = 讀不到）
static void sys_ram(unsigned long long *used, unsigned long long *total) {
    *used = *total = 0;
    size_t len = sizeof *total;
    if (sysctlbyname("hw.memsize", total, &len, NULL, 0)) *total = 0;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    vm_size_t pg = 0;
    if (host_page_size(mach_host_self(), &pg) != KERN_SUCCESS || !pg) return;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &cnt) != KERN_SUCCESS) return;
    unsigned long long pages;
    if (vm.internal_page_count > vm.purgeable_count)
        pages = (unsigned long long)vm.internal_page_count - vm.purgeable_count + vm.wire_count + vm.compressor_page_count;
    else pages = (unsigned long long)vm.active_count + vm.wire_count + vm.compressor_page_count;
    *used = pages * (unsigned long long)pg;
}

// 這個程序的 phys_footprint（bytes；0 = 讀不到）
static unsigned long long sys_proc_bytes(void) {
    task_vm_info_data_t vi;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &cnt) != KERN_SUCCESS) return 0;
    return vi.phys_footprint;
}

// dir 裡一般檔案的大小合計（bytes；-1 = 開不了）；最多 30 秒重算一次。呼叫者持有 sys_mu
static long long sys_cache_bytes(const char *dir) {
    static char last_dir[1024];
    static double last_t = -1;
    static long long last_v = -1;
    const double t = sys_now();
    if (last_t >= 0 && t - last_t < 30 && !strcmp(last_dir, dir)) return last_v;
    long long sum = -1;
    const int dfd = open(dir, O_RDONLY | O_DIRECTORY);
    DIR *d = dfd >= 0 ? fdopendir(dfd) : NULL;
    if (d) {
        sum = 0;
        for (struct dirent *de; (de = readdir(d));) {
            struct stat st;
            if (de->d_name[0] == '.' && (!de->d_name[1] || (de->d_name[1] == '.' && !de->d_name[2]))) continue;
            if (!fstatat(dfd, de->d_name, &st, AT_SYMLINK_NOFOLLOW) && S_ISREG(st.st_mode)) sum += st.st_size;
        }
        closedir(d); // 也關掉 dfd
    } else if (dfd >= 0) close(dfd);
    snprintf(last_dir, sizeof last_dir, "%s", dir);
    last_t = t;
    last_v = sum;
    return sum;
}

// 把 ,"sys":{...} 寫進 buf（cap 至少 384）；dir = cache 目錄（NULL / "" = 沒有：磁碟看目前目錄，cache_gb 是 null）。回傳寫入長度
static int sys_json(char *buf, size_t cap, const char *dir) {
    const bool has_dir = dir && *dir;
    pthread_mutex_lock(&sys_mu);
    const double cpu = sys_cpu_pct();
    const long long cb = has_dir ? sys_cache_bytes(dir) : -1;
    pthread_mutex_unlock(&sys_mu);
    const double gpu = sys_gpu_pct();
    unsigned long long ru, rt;
    sys_ram(&ru, &rt);
    const unsigned long long pb = sys_proc_bytes();
    struct statfs sf;
    const bool disk = (has_dir && !statfs(dir, &sf) ? true : !statfs(".", &sf)) && sf.f_bsize; // dir 還沒建：看目前目錄
    char f[8][24];
#define SYS_FMT(i, ok, fmt, v) ((ok) ? snprintf(f[i], sizeof f[i], fmt, v) : snprintf(f[i], sizeof f[i], "null"))
    SYS_FMT(0, cpu >= 0, "%.1f", cpu);
    SYS_FMT(1, gpu >= 0, "%.1f", gpu);
    SYS_FMT(2, ru > 0, "%.2f", ru / 1e9);
    SYS_FMT(3, rt > 0, "%.2f", rt / 1e9);
    SYS_FMT(4, pb > 0, "%.2f", pb / 1e9);
    SYS_FMT(5, disk, "%.1f", disk ? (double)sf.f_bavail * sf.f_bsize / 1e9 : 0.0);
    SYS_FMT(6, disk, "%.1f", disk ? (double)sf.f_blocks * sf.f_bsize / 1e9 : 0.0);
    SYS_FMT(7, cb >= 0, "%.2f", cb / 1e9);
#undef SYS_FMT
    const int n = snprintf(buf, cap,
                           ",\"sys\":{\"cpu_pct\":%s,\"gpu_pct\":%s,\"ram_used_gb\":%s,\"ram_total_gb\":%s,\"proc_gb\":%s,"
                           "\"disk_free_gb\":%s,\"disk_total_gb\":%s,\"cache_gb\":%s}",
                           f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
    return n < 0 ? 0 : n >= (int)cap ? (int)cap - 1 : n;
}
