// 開啟模型並檢查所有權重的名稱與形狀
#include <stdio.h>
#include "model.h"

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "models/Qwen3.8-Flash-Next-mixed-4-8bit";
    char err[512];
    qw_model *m = qw_model_open(dir, err, sizeof err);
    if (!m) {
        fprintf(stderr, "❌ %s\n", err);
        return 1;
    }
    int nfull = 0;
    for (int i = 0; i < QW_LAYERS; i++) nfull += m->layers[i].full_attn;
    printf("✅ 模型對照完成：用到 %d 個 tensor、%.2f GB\n", m->ntensors_used, m->total_bytes / 1e9);
    printf("   %d 層（%d 層 QSA、%d 層 DeltaNet）、embed %d-bit、lm_head %d-bit\n", QW_LAYERS, nfull, QW_LAYERS - nfull,
           m->embed_bits, m->lm_head.bits);
    qw_model_close(m);
    return 0;
}
