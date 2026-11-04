// 量 dependent dispatch 的固定成本：一串小 kernel（serial encoder），每個讀前一個的輸出
// 用法：build/dispatch
#include "../src/gpu.h"
#include <stdio.h>
#include <string.h>

static const char SRC[] =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void tiny(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i + 1024] = x[i] + 1.0f; x[i] = x[i + 1024]; }\n";

int main(void) {
    qw_gpu *g = qw_gpu_open();
    char err[4096];
    if (!g || !qw_gpu_compile(g, SRC, strlen(SRC), err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    qw_kernel *k = qw_kernel_get(g, "tiny");
    qw_buf b;
    qw_buf_new(g, 1 << 20, &b);
    memset(b.ptr, 0, 1 << 20);
    const int ns[] = {1, 100, 700};
    const unsigned tgs[] = {1, 64, 512};
    for (int ti = 0; ti < 3; ti++)
        for (int ni = 0; ni < 3; ni++) {
            double best = 1e9;
            for (int r = 0; r < 20; r++) {
                qw_batch *q = qw_batch_begin(g, false);
                for (int i = 0; i < ns[ni]; i++) {
                    qw_batch_kernel(q, k);
                    qw_batch_buf(q, 0, &b, 0);
                    qw_batch_dispatch(q, tgs[ti] < 4 ? 1 : tgs[ti] / 4, 1, 1, tgs[ti] < 4 ? 1 : 4 * 0 + (tgs[ti] < 4 ? 1 : 4), 1, 1);
                }
                const double t = qw_batch_end(q);
                if (t < best) best = t;
            }
            printf("每組 %4u 個 thread：%3d 個 dispatch，GPU %.1f µs，每個 %.2f µs\n", tgs[ti], ns[ni], best * 1e6, best * 1e6 / ns[ni]);
        }
    return 0;
}
