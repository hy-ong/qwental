// 列出 safetensors 的 tensor 清單與統計。
// 用法：
//   build/stinspect file.safetensors          整個檔案（mmap）
//   build/stinspect --header file.part        只有開頭（例如用 HTTP Range 抓的 header）
//   build/stinspect --model 目錄              多分片模型
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "st.h"

static void list(const st_file *f, int limit, double *total) {
    for (int i = 0; i < f->ntensors; i++) {
        const st_tensor *t = &f->tensors[i];
        *total += (double)t->nbytes;
        if (i < limit) {
            printf("  %-70s %-5s [", t->name, st_dtype_name(t->dtype));
            for (int d = 0; d < t->ndim; d++) printf(d ? ", %lld" : "%lld", (long long)t->shape[d]);
            printf("] %.2f MB\n", t->nbytes / 1e6);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "用法：%s [--header|--model] 路徑 [顯示筆數]\n", argv[0]);
        return 2;
    }
    char err[512];
    int a = 1, mode = 0;
    if (!strcmp(argv[1], "--header")) { mode = 1; a++; }
    if (!strcmp(argv[1], "--model")) { mode = 2; a++; }
    int limit = argc > a + 1 ? atoi(argv[a + 1]) : 40;
    double total = 0;
    if (mode == 2) {
        st_model *m = st_model_open(argv[a], err, sizeof err);
        if (!m) { fprintf(stderr, "%s\n", err); return 1; }
        int n = 0;
        for (int i = 0; i < m->nfiles; i++) {
            printf("%s：%d 個 tensor\n", m->files[i]->path, m->files[i]->ntensors);
            list(m->files[i], limit, &total);
            n += m->files[i]->ntensors;
        }
        printf("共 %d 個分片、%d 個 tensor、%.2f GB\n", m->nfiles, n, total / 1e9);
        st_model_close(m);
        return 0;
    }
    st_file *f;
    if (mode == 1) {
        FILE *fp = fopen(argv[a], "rb");
        if (!fp) { perror(argv[a]); return 1; }
        fseek(fp, 0, SEEK_END);
        long n = ftell(fp);
        rewind(fp);
        char *b = malloc(n);
        if (fread(b, 1, n, fp) != (size_t)n) return 1;
        fclose(fp);
        f = st_parse_header(b, (size_t)n, err, sizeof err);
    } else
        f = st_open(argv[a], err, sizeof err);
    if (!f) { fprintf(stderr, "%s\n", err); return 1; }
    printf("%d 個 tensor，資料區從第 %llu byte 開始\n", f->ntensors, (unsigned long long)f->data_start);
    list(f, limit, &total);
    printf("合計 %.2f GB\n", total / 1e9);
    st_close(f);
    return 0;
}
