// 小型 JSON：解析成樹（保留物件 key 的順序）、輸出（緊湊或 Python json.dumps 的預設格式）
#pragma once
#include <stdbool.h>
#include <stddef.h>

typedef enum { JS_NULL, JS_FALSE, JS_TRUE, JS_NUM, JS_STR, JS_ARR, JS_OBJ } js_type;

typedef struct js js;
struct js {
    js_type t;
    char *key;   // 物件成員的 key（不是成員時為 NULL）
    char *s;     // 字串值（UTF-8，NUL 結尾；len 是長度）／數字的原始文字
    size_t len;
    double num;
    js *kid;     // 陣列 / 物件的第一個成員
    js *next;    // 下一個兄弟
    int n;       // 成員數
};

// 解析；失敗回傳 NULL，err 寫錯誤位置
js *js_parse(const char *text, size_t len, char *err, size_t errlen);
void js_free(js *v);

js *js_get(const js *obj, const char *key);       // 物件成員（沒有回傳 NULL）
const char *js_str(const js *obj, const char *key); // 字串成員（不是字串回傳 NULL）
double js_num(const js *obj, const char *key, double def);
bool js_bool(const js *obj, const char *key, bool def);

// 輸出用的可成長字串
typedef struct {
    char *p;
    size_t n, cap;
} sbuf;
void sb_put(sbuf *b, const char *s, size_t n);
void sb_puts(sbuf *b, const char *s);
void sb_printf(sbuf *b, const char *fmt, ...);
void sb_json_str(sbuf *b, const char *s, size_t n, bool ascii); // 加上引號與跳脫；ascii = 非 ASCII 寫成 \uXXXX
void sb_free(sbuf *b);

// 把值輸出成 JSON。pretty_py = Python json.dumps(ensure_ascii=False) 的預設格式（", "、": "），否則緊湊
void js_dump(sbuf *b, const js *v, bool pretty_py);
