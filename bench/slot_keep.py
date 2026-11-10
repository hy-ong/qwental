#!/usr/bin/env python3
# 長主對話在 subagent 跑的時候留不留得住（qw_engine_par_pick / par_trim 的 slot 策略）。要 GPU：先停掉其他在跑的 Qwental，
# 再開 build/qwental-serve --parallel（建議 QW_CACHE_DBG=1，看 [slot] / [mem] / [cache] 紀錄），然後：
#   bench/slot_keep.py <url> <out 目錄> [main 的 token 數，預設 100000] [subagent 數，預設 3] [seq]
# 流程（temperature 0）：
#   1. 主對話：共用前綴（tools + system，約 15K token）+ 長文件，生成 64 個 token
#   2. N 個 subagent 同時送（seq：一個接一個），共用同一段前綴、各自的短任務，各生成 200 個 token
#   3. 主對話下一輪（帶第 1 步的回答）：看 /status 的 cached_tokens 增加多少 —— 主對話的 slot 留著時 ≈ 第 1 步的長度
# out 目錄存各請求的輸出與 stats.json；新舊版本、seq 與並發各跑一次後 diff -r 兩個目錄：輸出必須逐字相同（slot 選擇不改數值）
import json, os, sys, threading, time, urllib.request

URL, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
MAIN = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
NSUB = int(sys.argv[4]) if len(sys.argv) > 4 else 3
SEQ = len(sys.argv) > 5 and sys.argv[5] == "seq"
os.makedirs(OUT, exist_ok=True)


def post(path, body, timeout=3600):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def status():
    return json.loads(urllib.request.urlopen(URL + "/status", timeout=60).read())


def filler(tag, n):  # 確定性的假內容（每段不同，避免太好壓縮）
    words = ["alpha", "beta", "gamma", "delta", "kernel", "buffer", "slot", "prefix", "token", "state", "commit", "branch"]
    return "\n".join(f"{tag} {i}: " + " ".join(words[(i * 7 + k * 3) % len(words)] + str((i * 31 + k) % 997) for k in range(12)) for i in range(n))


def ntok(body):
    try: return post("/v1/messages/count_tokens", body)["input_tokens"]
    except Exception: return -1


TOOLS = [{"name": f"tool_{t}", "description": filler(f"tool {t} rule", 120), "input_schema": {"type": "object", "properties": {"arg": {"type": "string"}}}}
         for t in range(4)]
SYSTEM = "You are a coding agent. Follow the rules below.\n" + filler("system rule", 400)


def body(msgs, mt):
    return {"model": "q", "max_tokens": mt, "temperature": 0, "system": SYSTEM, "tools": TOOLS, "messages": msgs}


pre = ntok(body([{"role": "user", "content": "hi"}], 1))
lines = 200
doc = filler("doc line", lines)
while True:  # 主對話的長度調到 ≈ MAIN
    n = ntok(body([{"role": "user", "content": doc + "\nSummarize the document in one sentence."}], 1))
    if n < 0 or n >= MAIN * 0.97: break
    lines = int(lines * min(4.0, max(1.05, MAIN / max(n - pre, 1))))
    doc = filler("doc line", lines)
print(f"共用前綴 ≈ {pre} token，主對話 ≈ {n} token")
stats = {"prefix": pre, "main_prompt": n}


def text(d): return "".join(c.get("text", "") + c.get("thinking", "") for c in d["content"])


def run(name, b):
    s0, t0 = status(), time.time()
    d = post("/v1/messages", b)
    s1 = status()
    open(os.path.join(OUT, name + ".txt"), "w").write(text(d))
    r = {"sec": round(time.time() - t0, 1), "cached": s1["cached_tokens"] - s0["cached_tokens"], "prompt": s1["prompt_tokens"] - s0["prompt_tokens"]}
    print(f"{name}: {r}")
    return d, r


m1 = [{"role": "user", "content": doc + "\nSummarize the document in one sentence."}]
d1, stats["main1"] = run("main1", body(m1, 64))

subs = [[{"role": "user", "content": f"Task {i}: list five words that rhyme with '{w}' and explain each briefly."}] for i, w in enumerate(["cat", "light", "stone", "river", "glass", "tree", "song", "rain"][:NSUB])]
s0, t0 = status(), time.time()
if SEQ:
    for i, m in enumerate(subs): run(f"sub{i}", body(m, 200))
else:
    th = [threading.Thread(target=run, args=(f"sub{i}", body(m, 200))) for i, m in enumerate(subs)]
    [x.start() for x in th]
    [x.join() for x in th]
s1 = status()
stats["subs"] = {"sec": round(time.time() - t0, 1), "cached": s1["cached_tokens"] - s0["cached_tokens"], "slot_mem_gb": s1.get("slot_mem_gb")}
print(f"subagents: {stats['subs']}")

m2 = m1 + [{"role": "assistant", "content": d1["content"]}, {"role": "user", "content": "Now name the three most frequent words in it."}]
_, stats["main2"] = run("main2", body(m2, 64))
ok = stats["main2"]["cached"] >= 0.9 * stats["main1"]["prompt"]
stats["main_kept"] = ok
json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
print(("✅ 主對話的前綴留住了" if ok else "❌ 主對話重算了") + f"：cached {stats['main2']['cached']} / 第一輪 prompt {stats['main1']['prompt']}")
sys.exit(0 if ok else 1)
