// 對話格式：把訊息排成 Qwen 的 prompt（等同模型的 chat_template.jinja），並解析模型輸出（思考、文字、tool call）
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
    // OpenAI 形式的工具陣列 [{"type":"function","function":{...}}]（擁有），可為 NULL。
    // 外層帶 "defer_loading":true 的是延後載入的工具（Anthropic tool search）：不排進 prompt 的 <tools>，但解析器認得它
    js *tools;
    bool thinking;       // enable_thinking
    const char *effort;  // "xhigh"（預設）、"medium"、"low"
    bool thinking_asked; // 請求自己指定了思考（thinking / output_config.effort / budget_tokens；OpenAI：reasoning_effort 等）
    bool deferred;       // 有 defer_loading 的工具：解析器把沒宣告的工具名稱也當成 tool call（客戶端會自己執行）
} chat_req;

// Anthropic 請求完全沒提思考（沒有 thinking、effort、budget）時要不要思考。
// 預設 false（"on_request"：不思考——Claude Code 的標題、話題判斷等小請求不用花 token 想）；serve.c 可用旗標改成 true
extern bool chat_think_unasked;

// 前面幾個（連續的）CHAT_SYSTEM 訊息合併成 system prompt（各自 trim 後以 "\n" 相接，同 template）；
// 對話中途的 system / developer 訊息轉成 user 訊息（保留位置，同 Strata #56）。
chat_msg *chat_add(chat_req *r, int role);
void chat_free(chat_req *r);

// Anthropic Messages API 的 request body → chat_req；err 為錯誤訊息
bool chat_from_anthropic(const js *body, chat_req *r, char *err, size_t errlen);
// OpenAI Chat Completions 的 request body → chat_req
bool chat_from_openai(const js *body, chat_req *r, char *err, size_t errlen);

// 依 chat template 排出 prompt（含 generation prompt）。
// 訊息內容（user、tool 結果、system、工具定義、assistant 的文字）裡的特殊 token 字面前面會插 QW_TOK_PLAIN（tok.h），
// qw_tok_encode 會把它們當一般文字編碼；template 本身的標記（<|im_start|>、<tool_call>…）照常是特殊 token。
// 要看「人讀的」prompt（記錄、比對 template）用 chat_unmark 拿掉標記。
void chat_render(const chat_req *r, sbuf *out);
void chat_unmark(sbuf *s);

// ---------------------------------------------------------------- 串流解析模型輸出
typedef enum { CEV_THINK, CEV_TEXT, CEV_CALL } chat_ev_type;
typedef void (*chat_ev_cb)(void *ctx, chat_ev_type t, const char *s, size_t n, const chat_call *call);

typedef struct {
    int fence_ch, fence_len; // 開著的 code fence（``` 或 ~~~）
    int ticks, lticks;       // 這段落已完成的行裡的反引號數、目前這行的反引號數
    bool blank;              // 目前這行到目前為止只有空白
    int run_ch, run_len;     // 這行開頭連續的 ` 或 ~
    bool in_run, rest3;
    int rest_seq;
    char last;               // 最後一個輸出的字元
} chat_track;

typedef struct chat_pcall chat_pcall;

typedef struct {
    int state; // 0 思考、1 文字、2 tool call、3 思考裡的 tool call
    sbuf pend; // 還不能確定的尾巴（可能是標籤的開頭）
    const js *tools;
    bool has_tools, deferred;
    int ncalls;
    bool text_started; // 文字段是否已輸出過非空白
    bool after_think;  // 思考已在 </think> 結束（之後行首的 </think> 是多餘的，丟掉）
    bool bare;         // 目前的 call 沒有 <tool_call>，直接以 <function= 開頭
    bool drop_close;   // 文字開頭多出來的 </tool_call> 丟掉
    long ckd;          // call 狀態：pend 前面這麼多 byte 已確認沒有結尾標籤（-1 = 出現過：之後有新的 '>' 才完整走一遍）
    long ckn;          // ckd = -1 時上次完整走過的長度
    chat_track tr;
    chat_pcall *pc; // 思考裡的 tool call：等到思考結束（或回合自然結束）才算數
    int npc;
    // 結果（給 serve.c 記錄／回報）
    bool cut_call; // 輸出結束時還在沒寫完的 tool call 裡：當成文字輸出了，沒有變成 tool_use
    int rescued;   // 從思考裡救出來的 tool call 數
    int unread;    // 寫成 <tool_call> 但工具沒宣告、當成文字的次數
} chat_parser;

void chat_parser_init(chat_parser *p, bool thinking, const js *tools);
void chat_parser_feed(chat_parser *p, const char *s, size_t n, chat_ev_cb cb, void *ctx);
// 輸出結束。cut = 因 max_tokens（或取消）被截斷：沒寫完的 tool call 不會變成 tool_use，思考裡的 call 也不算數
void chat_parser_finish(chat_parser *p, bool cut, chat_ev_cb cb, void *ctx);
// 同 chat_parser_finish(p, false, …)（舊介面）
void chat_parser_end(chat_parser *p, chat_ev_cb cb, void *ctx);
void chat_parser_free(chat_parser *p);
