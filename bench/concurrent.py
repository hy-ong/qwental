#!/usr/bin/env python3
# 同時多個請求的總吞吐量：K 個請求（8 個不同的 prompt 輪流用，temperature 0、各 400 個 token）一起送出（seq：一個接一個），
# 印出總 token / 牆鐘時間；out 目錄存各請求的輸出（比對用）。用法：bench/concurrent.py <url> <K> [seq|par] [out 目錄] [edit]
# edit：改用改寫程式碼的 prompt（bench/prompts/code_edit.txt，各請求要改的名稱不同），照抄前文的比例高
import json, os, sys, threading, time, urllib.request

URL, K = sys.argv[1].rstrip("/"), int(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else "par"
OUT = sys.argv[4] if len(sys.argv) > 4 else None
P = ["Explain how a hash map handles collisions, with an example in Python.",
     "用繁體中文說明為什麼天空是藍色的，約 300 字。",
     "Write a bash script that finds the 10 largest files under a directory, with comments.",
     "List ten practical tips for writing clear commit messages, with an example for each.",
     "Describe the differences between TCP and UDP and when to use each.",
     "Write a C function that reverses a singly linked list, and explain it step by step.",
     "用繁體中文介紹台灣的夜市文化，包括常見小吃。",
     "Explain gradient descent to a high school student, with a simple numeric example."]
if len(sys.argv) > 5 and sys.argv[5] == "edit":
    E = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts/code_edit.txt"), encoding="utf-8").read()
    P = [E.replace("`Stock`", f"`{n}`") + "\nOutput the full modified module." for n in ("Stock", "Store", "Depot", "Warehouse", "Pantry", "Cellar", "Vault", "Shelf")]
tot = [0] * K


def ask(i):
    b = json.dumps({"model": "q", "max_tokens": 400, "temperature": 0, "messages": [{"role": "user", "content": P[i % len(P)]}]}).encode()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/messages", b, {"Content-Type": "application/json"}), timeout=900).read())
    tot[i] = d["usage"]["output_tokens"]
    if OUT:
        os.makedirs(OUT, exist_ok=True)
        open(os.path.join(OUT, f"l{i}.txt"), "w").write("".join(c.get("text", "") + c.get("thinking", "") for c in d["content"]))


t0 = time.time()
if MODE == "seq":
    for i in range(K): ask(i)
else:
    th = [threading.Thread(target=ask, args=(i,)) for i in range(K)]
    [x.start() for x in th]
    [x.join() for x in th]
dt = time.time() - t0
print(f"{MODE} K={K}：{sum(tot)} 個 token、{dt:.1f} 秒 → 總共 {sum(tot) / dt:.1f} tok/s")
