// safetensors 讀取器：整個檔案 mmap，只解析 JSON header，不複製資料。
//
// 檔案格式：
//   [8 bytes 小端序 header 長度 N][N bytes JSON header][資料]
//   header：{"名稱": {"dtype": "BF16", "shape": [..], "data_offsets": [起, 迄]}, ..., "__metadata__": {..}}
//   data_offsets 相對於資料區開頭。
// 多分片的模型另有 model.safetensors.index.json：{"weight_map": {"名稱": "分片檔名", ...}}
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    ST_UNKNOWN = 0,
    ST_F32, ST_F16, ST_BF16, ST_F64,
    ST_I8, ST_U8, ST_I16, ST_U16, ST_I32, ST_U32, ST_I64, ST_U64,
    ST_BOOL, ST_F8_E4M3, ST_F8_E5M2,
} st_dtype;

typedef struct {
    char *name;
    st_dtype dtype;
    int ndim;
    int64_t shape[8];
    uint64_t offset;  // 從檔案開頭算起的位元組位置
    uint64_t nbytes;
    const void *data; // mmap 後的位址
    int gq;           // GGUF 的量化區塊：ggml type + 1（0 = 一般 dtype）。資料是原始區塊位元組（dtype ST_U8）
} st_tensor;

typedef struct {
    char *path;
    int fd;
    void *map;        // 整個檔案（page 對齊，可以零複製包成 Metal buffer）
    size_t size;
    uint64_t data_start;
    st_tensor *tensors;
    int ntensors;
} st_file;

// 開啟並解析單一 safetensors 檔。失敗回傳 NULL，原因寫進 err。
st_file *st_open(const char *path, char *err, size_t errlen);
// 只解析 header（不需要完整檔案），給測試和 Range 下載的 header 用
st_file *st_parse_header(const void *buf, size_t len, char *err, size_t errlen);
void st_close(st_file *f);
const st_tensor *st_find(const st_file *f, const char *name);
size_t st_dtype_size(st_dtype d);
const char *st_dtype_name(st_dtype d);

// 多分片模型
typedef struct {
    char *dir;
    st_file **files;
    int nfiles;
} st_model;

// dir 裡有 model.safetensors.index.json 就照它開所有分片；沒有就開 model.safetensors
st_model *st_model_open(const char *dir, char *err, size_t errlen);
void st_model_close(st_model *m);
// 在所有分片裡找；file_out 可為 NULL
const st_tensor *st_model_find(const st_model *m, const char *name, const st_file **file_out);
