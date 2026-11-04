#!/usr/bin/env python3
# 對任何 Anthropic 相容 server 量測：首 token 時間（prefill）與解碼速度。用法：bench/vs_server.py <url> <標籤> [輪數]
# 每個 prompt 前面加隨機字串，避免 prefix cache 讓 prefill 數字失真
import json, os, random, sys, time, urllib.request

URL, TAG = sys.argv[1], sys.argv[2]
PASSES = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ROOT = os.path.dirname(os.path.abspath(__file__))


def src(paths, limit):
    s = ""
    for p in paths:
        s += f"\n// ===== {p}\n" + open(os.path.join(ROOT, "..", p), encoding="utf-8").read()
    return s[:limit]


PROMPTS = [
    ("prose", "Write a detailed essay about the history of the printing press and its effect on Europe.", 400),
    ("code", "Write a complete Python implementation of an LRU cache class with get, put, delete and resize methods, with docstrings and type hints, followed by unittest tests.", 400),
    ("edit", open(os.path.join(ROOT, "prompts/code_edit.txt"), encoding="utf-8").read() + "\nOutput the full modified module.", 400),
    ("long8k", src(["kernels/qsa.metal", "src/chat.c"], 30000) + "\n\nSummarize what this code does in 5 bullet points.", 200),
    ("long32k", src(["src/engine.c", "kernels/engine.metal", "tools/serve.c"], 110000) + "\n\nSummarize what this code does in 5 bullet points.", 200),
]


def run(prompt, max_tokens):
    body = {"model": os.environ.get("MODEL", "m"), "max_tokens": max_tokens, "temperature": 0, "stream": True, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": f"[{random.random():.12f}]\n" + prompt}]}
    req = urllib.request.Request(URL + "/v1/messages", json.dumps(body).encode(),
                                 {"content-type": "application/json", "x-api-key": "x", "anthropic-version": "2023-06-01"})
    t0 = time.time()
    first = last = None
    out = inp = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            ev = json.loads(line[5:])
            if ev["type"] == "message_start":
                inp = ev["message"]["usage"].get("input_tokens", 0)
            elif ev["type"] == "content_block_delta":
                now = time.time()
                if first is None:
                    first = now
                last = now
            elif ev["type"] == "message_delta":
                out = ev.get("usage", {}).get("output_tokens", out)
    ttft = first - t0
    dec = (out - 1) / (last - first) if last > first else 0
    return inp, out, ttft, dec


ONLY = os.environ.get("ONLY", "").split(",") if os.environ.get("ONLY") else None  # 例：ONLY=prose,code
res = {}
for p in range(PASSES):
    for name, prompt, mt in PROMPTS:
        if ONLY and name not in ONLY:
            continue
        inp, out, ttft, dec = run(prompt, mt)
        print(f"{TAG} {name} pass{p}: in={inp} out={out} ttft={ttft:.2f}s prefill={inp / ttft:.0f} tok/s decode={dec:.1f} tok/s", flush=True)
        res.setdefault(name, []).append((inp, out, ttft, dec))
print(json.dumps({TAG: res}))
