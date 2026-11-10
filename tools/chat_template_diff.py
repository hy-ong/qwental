#!/usr/bin/env python3
"""chat_render（C）對照官方 chat_template.jinja：每個 request JSON 各排一次，比對結果。

用法：python tools/chat_template_diff.py TEMPLATE.jinja REQ.json... [--openai] [--chatcheck build/chatcheck]

REQ.json 可以是 Claude Code 擷取的 {"body": {...}}，或直接是 request body。Anthropic 格式（預設）或 OpenAI 格式（--openai）。
這裡把請求轉成 template 的輸入時，套用和 src/chat.c 同樣的「政策」（這些是刻意的，不是 template 的一部分）：
  - 開頭連續的 system 合併（template 自己做）；對話中途的 system / developer 轉成 user 訊息（template 會 raise）
  - Claude Code 的 billing header 版本 / cch 固定成 f
  - defer_loading 的工具不排進 <tools>；tool_reference → <tools>\\n{定義}\\n</tools>
  - 完成的 assistant 回合沒有文字也沒有 tool call 就不排（最後一個保留）
  - 圖片 / 文件換成英文說明文字（不渲染 vision token）
  - 思考：thinking / output_config.effort / budget_tokens → enable_thinking、reasoning_effort
template 負責其他一切（排版、trim、tool call XML、tojson）。只要 C 的結果和 Jinja 一樣，就代表 C 的排版正確。
C 端特殊 token 字面前的 QW_TOK_PLAIN 標記會先拿掉（chatcheck 預設）。
"""
import json
import subprocess
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

IMG = "[image omitted: this model cannot see images]"
DOC = "[document omitted: this model cannot read attached documents]"


def load_template(path):
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(open(path, encoding="utf-8").read())


def pin_billing(s):
    h = "x-anthropic-billing-header:"
    if not s.startswith(h):
        return s
    t = list(s)
    c = s.find("cch=", len(h))
    if 0 <= c <= 160:
        v, e = c + 4, s.find(";", c + 4)
        if e > v and e - v <= 16:
            t[v:e] = "f" * (e - v)
    c = s.find("cc_version=")
    if 0 <= c <= 160:
        v, e = c + 11, s.find(";", c)
        if v < e and e - v <= 64:
            parts = s[v:e].split(".")
            if len(parts) > 3:
                tail = v + len(".".join(parts[:3])) + 1
                t[tail:e] = "f" * (e - tail)
    return "".join(t)


def text_of(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    out = []
    for b in c:
        ty = b.get("type")
        if "text" in b and ty in (None, "text", "input_text"):
            out.append(b["text"])
        elif ty == "refusal":
            out.append(b.get("refusal") or "")
        elif ty in ("image", "image_url", "input_image"):
            out.append(IMG)
    return "".join(out)


def doc_text(b):
    src = b.get("source") or {}
    if src.get("type") == "text" and isinstance(src.get("data"), str):
        return src["data"]
    if src.get("type") == "content":
        t = text_of(src.get("content"))
        if t:
            return t
    return DOC


EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "maximum": "xhigh"}


def sys_or_user(msgs):
    return "system" if all(m["role"] == "system" for m in msgs) else "user"


def from_anthropic(b, think_unasked=False):
    tools, deferred = [], {}
    for t in b.get("tools") or []:
        if not t.get("name") or "input_schema" not in t:
            continue
        fn = {"name": t["name"]}
        if "description" in t:
            fn["description"] = t["description"]
        fn["parameters"] = t["input_schema"]
        if t.get("defer_loading"):
            deferred[t["name"]] = fn
        else:
            tools.append({"type": "function", "function": fn})
    msgs = []
    if b.get("system"):
        msgs.append({"role": "system", "content": pin_billing(text_of(b["system"]))})
    for m in b["messages"]:
        role, c = m["role"], m.get("content")
        if role in ("system", "developer"):
            msgs.append({"role": sys_or_user(msgs), "content": text_of(c)})
        elif role == "user":
            if isinstance(c, str):
                msgs.append({"role": "user", "content": c})
                continue
            text, any_ = [], False
            for blk in c or []:
                ty = blk.get("type")
                if ty == "tool_result":
                    if any_:
                        msgs.append({"role": "user", "content": "".join(text)})
                        text, any_ = [], False
                    body = "Error: " if blk.get("is_error") else ""
                    rc = blk.get("content")
                    if isinstance(rc, list):
                        for p in rc:
                            if p.get("type") == "tool_reference":
                                if p.get("tool_name") in deferred:
                                    body += "<tools>\n" + json.dumps(deferred[p["tool_name"]], ensure_ascii=False) + "\n</tools>"
                            elif p.get("type") == "document":
                                body += doc_text(p)
                            else:
                                body += text_of([p])
                    else:
                        body += text_of(rc)
                    msgs.append({"role": "tool", "content": body})
                elif ty == "text":
                    text.append(blk.get("text", ""))
                    any_ = True
                elif ty == "image":
                    text.append(IMG)
                    any_ = True
                elif ty == "document":
                    text.append(doc_text(blk))
                    any_ = True
            if any_:
                msgs.append({"role": "user", "content": "".join(text)})
        elif role == "assistant":
            if isinstance(c, str):
                msgs.append({"role": "assistant", "content": c})
                continue
            out = {"role": "assistant", "content": "".join(x["text"] for x in c if x.get("type") == "text" and "text" in x)}
            r = "".join(x["thinking"] for x in c if x.get("type") == "thinking" and isinstance(x.get("thinking"), str))
            if r:
                out["reasoning_content"] = r
            calls = [{"function": {"name": x.get("name"), "arguments": x.get("input") or {}}} for x in c if x.get("type") == "tool_use"]
            if calls:
                out["tool_calls"] = calls
            msgs.append(out)
    th = b.get("thinking")
    eff = (b.get("output_config") or {}).get("effort")
    kw = {}
    budget = th.get("budget_tokens") if isinstance(th, dict) and th.get("type") != "disabled" else None
    if isinstance(th, dict) and th.get("type") == "disabled":
        kw["enable_thinking"] = False
    elif eff:
        lv = EFFORT.get(eff.lower(), "xhigh")
        kw = {"enable_thinking": False} if lv is None else {"reasoning_effort": lv}
    elif budget:
        kw["reasoning_effort"] = "low" if budget < 2048 else "medium" if budget < 8192 else "xhigh"
    elif th is None and not think_unasked:
        kw["enable_thinking"] = False
    return msgs, tools or None, kw


def from_openai(b):
    msgs = []
    for m in b["messages"]:
        role = m["role"]
        if role == "developer":
            role = "system"
        if role == "system":
            role = sys_or_user(msgs)
        out = {"role": role, "content": text_of(m.get("content"))}
        rc = m.get("reasoning_content") or m.get("reasoning")
        if rc and role == "assistant":
            out["reasoning_content"] = rc
        if m.get("tool_calls") and role == "assistant":
            calls = []
            for c in m["tool_calls"]:
                f = c.get("function", c)
                a = f.get("arguments")
                if isinstance(a, str):
                    try:
                        v = json.loads(a) if a.strip() else {}
                    except ValueError:
                        v = {"arguments": a}
                    a = v if isinstance(v, dict) else {"arguments": a}
                calls.append({"function": {"name": f.get("name"), "arguments": a or {}}})
            out["tool_calls"] = calls
        msgs.append(out)
    kw = {}
    re_ = b.get("reasoning_effort")
    if re_ in (None, ""):
        re_ = (b.get("reasoning") or {}).get("effort")
    if re_ is False:
        kw = {"enable_thinking": False}
    elif isinstance(re_, str) and re_:
        lv = EFFORT.get(re_.lower(), "xhigh")
        kw = {"enable_thinking": False} if lv is None else {"reasoning_effort": lv}
    ctk = b.get("chat_template_kwargs") or {}
    if "enable_thinking" in ctk and not ctk["enable_thinking"]:
        kw = {"enable_thinking": False}
    elif isinstance(ctk.get("reasoning_effort"), str):
        lv = EFFORT.get(ctk["reasoning_effort"].lower())
        kw = {"enable_thinking": False} if lv is None else {"reasoning_effort": lv}
    return msgs, b.get("tools") or None, kw


def drop_empty(msgs):
    return [m for i, m in enumerate(msgs) if i == len(msgs) - 1 or not (
        m["role"] == "assistant" and not (m.get("content") or "").strip() and not m.get("tool_calls"))]


def main():
    args = sys.argv[1:]
    openai = "--openai" in args
    chk = "build/chatcheck"
    if "--chatcheck" in args:
        chk = args[args.index("--chatcheck") + 1]
        args.remove("--chatcheck")
        args.remove(chk)
    args = [a for a in args if a != "--openai"]
    tpl = load_template(args[0])
    bad = 0
    for path in args[1:]:
        d = json.load(open(path, encoding="utf-8"))
        b = d["body"] if isinstance(d.get("body"), dict) and "messages" in d["body"] else d
        msgs, tools, kw = from_openai(b) if openai else from_anthropic(b)
        want = tpl.render(messages=drop_empty(msgs), tools=tools, add_generation_prompt=True, **kw)
        got = subprocess.run([chk, path] + ([] if openai else ["-a"]), capture_output=True, check=True).stdout.decode("utf-8")
        if got == want:
            print(f"SAME  {path}  ({len(want)} chars)")
            continue
        bad += 1
        k = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
        print(f"DIFF  {path}  at {k}: C={got[max(0, k - 80):k + 80]!r}\n      jinja={want[max(0, k - 80):k + 80]!r}")
    print(f"{len(args) - 1 - bad}/{len(args) - 1} identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
