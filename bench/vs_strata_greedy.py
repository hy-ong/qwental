#!/usr/bin/env python3
# 同一份權重的 greedy 輸出對照：Strata（OpenAI 端點，附 top-5 logprobs）與 Qwental（Anthropic 端點），thinking 關閉、temperature 0。
# 用法：bench/vs_strata_greedy.py collect <url> <openai|anthropic> <輸出.json>；bench/vs_strata_greedy.py compare <strata.json> <qwental.json>
import json, os, sys, urllib.request
ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPTS = [
    ("prose", "Write a detailed essay about the history of the printing press and its effect on Europe."),
    ("code", "Write a complete Python implementation of an LRU cache class with get, put, delete and resize methods, with docstrings and type hints, followed by unittest tests."),
    ("edit", open(os.path.join(ROOT, "prompts/code_edit.txt"), encoding="utf-8").read() + "\nOutput the full modified module."),
]
N = 200

def post(url, body, hdr):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json", **hdr})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())

def collect(url, kind, out):
    res = {}
    for name, p in PROMPTS:
        msgs = [{"role": "user", "content": p}]
        if kind == "openai":
            j = post(url + "/v1/chat/completions", {"model": "m", "messages": msgs, "max_tokens": N, "temperature": 0, "logprobs": True,
                                                    "top_logprobs": 5, "chat_template_kwargs": {"enable_thinking": False}}, {})
            c = j["choices"][0]
            lp = [(t["token"], t["logprob"], [(x["token"], x["logprob"]) for x in t.get("top_logprobs", [])]) for t in (c.get("logprobs") or {}).get("content") or []]
            res[name] = {"text": c["message"]["content"], "lp": lp}
        else:
            j = post(url + "/v1/messages", {"model": "m", "messages": msgs, "max_tokens": N, "temperature": 0, "thinking": {"type": "disabled"}},
                     {"x-api-key": "x", "anthropic-version": "2023-06-01"})
            res[name] = {"text": "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")}
        print(name, len(res[name]["text"]), "字元", flush=True)
    json.dump(res, open(out, "w"), ensure_ascii=False)

def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    for name, _ in PROMPTS:
        s, q = A[name]["text"], B[name]["text"]
        i = next((k for k in range(min(len(s), len(q))) if s[k] != q[k]), min(len(s), len(q)))
        same = i == len(s) == len(q)
        line = "%-6s Strata %d 字元、Qwental %d 字元：%s" % (name, len(s), len(q), "完全相同" if same else "第 %d 個字元起不同" % i)
        lp = A[name].get("lp") or []
        if not same and lp: # 找出分歧點所在的 Strata token，看前兩名的差距
            pos = 0
            for t, l, top in lp:
                if pos + len(t) > i:
                    tops = sorted(top, key=lambda x: -x[1])
                    gap = tops[0][1] - tops[1][1] if len(tops) > 1 else float("nan")
                    line += "；該處 Strata 前兩名 %r %.3f / %r %.3f（差 %.3f）" % (tops[0][0], tops[0][1], tops[1][0] if len(tops) > 1 else "", tops[1][1] if len(tops) > 1 else 0, gap)
                    break
                pos += len(t)
        print(line)
        if not same: print("   Strata ：%r\n   Qwental：%r" % (s[max(0, i - 40):i + 40], q[max(0, i - 40):i + 40]))

if sys.argv[1] == "collect": collect(sys.argv[2], sys.argv[3], sys.argv[4])
else: compare(sys.argv[2], sys.argv[3])
