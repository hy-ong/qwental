// Qwen 的 byte-level BPE tokenizer（讀模型目錄的 vocab.json、merges.txt、tokenizer.json 的特殊 token，或 GGUF 的 tokenizer.ggml.*）
// 限制：不做 NFC 正規化（輸入假設已經是 NFC；一般程式碼、英文、中日韓文都是）
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct qw_tok qw_tok;

qw_tok *qw_tok_open(const char *model_dir, char *err, size_t errlen);
// 從 GGUF 的 tokenizer.ggml.*（tokens、token_type、merges）讀；path 是 .gguf 檔或放分片的目錄。結果與 tokenizer.json 版相同
qw_tok *qw_tok_open_gguf(const char *path, char *err, size_t errlen);
// model_dir 有 vocab.json（或沒給 gguf）就用 qw_tok_open，否則用 gguf 的 tokenizer
qw_tok *qw_tok_open_any(const char *model_dir, const char *gguf, char *err, size_t errlen);
void qw_tok_close(qw_tok *t);

// 「後面這個 '<' 是一般文字」的標記（U+F0E00，私用區）：chat_render 把它插在訊息內容裡的特殊 token 字面（"<|im_end|>"、
// "</think>"、"<tool_call>"…）前面；qw_tok_encode 會拿掉標記，並把那段字面當一般文字編碼（不變成控制 token）
#define QW_TOK_PLAIN "\xF3\xB0\xB8\x80"

// 文字 → token id。特殊 token（<|im_start|> 等）原樣出現在文字裡時直接對應；前面有 QW_TOK_PLAIN 的則當一般文字。
// 回傳 token 數；out 為 NULL 或 cap 不夠時只計數。
int qw_tok_encode(const qw_tok *t, const char *text, size_t len, int32_t *out, int cap);

// 一個 token → bytes（UTF-8 片段，可能是不完整的字元）。回傳長度；buf 至少 256 bytes。
int qw_tok_decode1(const qw_tok *t, int32_t id, char *buf);

// 特殊 token 的 id（找不到回傳 −1）
int32_t qw_tok_special(const qw_tok *t, const char *content);
int qw_tok_vocab(const qw_tok *t);
