// 對話格式：把訊息排成 Qwen 的 prompt（等同模型目錄的 chat_template.jinja），並解析模型輸出（思考、文字、tool call）
#pragma once
#include <stdbool.h>
#include "json.h"

typedef struct {
    char *name;
    js *args; // 物件（擁有）
    char *id;
} chat_call;

typedef struct {
    int role; // CHAT_SYSTEM / USER / ASSISTANT / TOOL
    sbuf content;
    sbuf reasoning; // assistant 的思考
    chat_call *calls;
    int ncalls;
    char *tool_id; // tool 訊息對應的 tool_use id
} chat_msg;

enum { CHAT_SYSTEM, CHAT_USER, CHAT_ASSISTANT, CHAT_TOOL };

typedef struct {
    chat_msg *m;
    int n;
    js *tools;          // OpenAI 形式的工具陣列 [{"type":"function","function":{...}}]（擁有），可為 NULL
    bool thinking;      // enable_thinking
    const char *effort; // "xhigh"（預設）、"medium"、"low"
} chat_req;

chat_msg *chat_add(chat_req *r, int role);
void chat_free(chat_req *r);

// Anthropic Messages API 的 request body → chat_req；err 為錯誤訊息
bool chat_from_anthropic(const js *body, chat_req *r, char *err, size_t errlen);
// OpenAI Chat Completions 的 request body → chat_req
bool chat_from_openai(const js *body, chat_req *r, char *err, size_t errlen);

// 依 chat template 排出 prompt（含 generation prompt）
void chat_render(const chat_req *r, sbuf *out);

// ---------------------------------------------------------------- 串流解析模型輸出
typedef enum { CEV_THINK, CEV_TEXT, CEV_CALL } chat_ev_type;
typedef void (*chat_ev_cb)(void *ctx, chat_ev_type t, const char *s, size_t n, const chat_call *call);

typedef struct {
    int state; // 0 思考、1 文字、2 tool call
    sbuf pend; // 還不能確定的尾巴（可能是標籤的開頭）
    sbuf call; // 目前 tool call 的內容
    const js *tools;
    int ncalls;
    bool text_started; // 文字段是否已輸出過非空白
} chat_parser;

void chat_parser_init(chat_parser *p, bool thinking, const js *tools);
void chat_parser_feed(chat_parser *p, const char *s, size_t n, chat_ev_cb cb, void *ctx);
void chat_parser_end(chat_parser *p, chat_ev_cb cb, void *ctx);
void chat_parser_free(chat_parser *p);
