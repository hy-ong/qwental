#!/usr/bin/env python3
# 主對話在 subagent 並發時，下一輪接不接得上自己的 prompt（snap_save_k 滿了換掉哪一份快照）。一個指令開 server、跑完、關掉：
#   bench/snap_evict.py [--serve build/qwental-serve] [--main 50000] [--rounds 3] [--subs 3] [--snaps N] [--out DIR] [--int8]
# 流程（temperature 0、不思考）：
#   1. 主對話 M：tools + system + 長文件（約 --main 個 token），串流生成到 --gen 個 token
#   2. M 開始生成後，跑 --rounds 輪、每輪 --subs 個同時送的短請求（各 5–15K token、前綴互不相同、max_tokens 64）：
#      M 一直在單獨推測解碼 / 一起 decode 之間切換，每次切換都在 M 的 slot 存一份跨請求接不上的快照（SW_FINISH / SW_PROMPT）
#   3. M2 = M + M 的回答 + 一個 user 回合：看 usage.cache_read_input_tokens（= log 的「重用 R」）
# 預期：M2 重用 ≥ len(M) − 64（prompt 結尾對齊 64 的檢查點還在）。舊版（只照 age 換）會被切換存的快照擠掉，重用很少或 0。
# --snaps N = QW_PAR_SNAPS（每組 slot 的快照數；預設不設，照 snap_cap 算）。SSD 快取用每次新開的目錄（避免從 SSD 接上蓋掉問題）。
# GPU 安全：server 用 lockf build/gpu.lock 包住、port 8081、結束（含例外 / Ctrl-C）一定關掉。
import argparse, http.client, json, os, re, signal, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--serve", default=os.path.join(ROOT, "build", "qwental-serve"))
ap.add_argument("--model", default=os.path.join(ROOT, "models", "sc117-gguf"))
ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--main", type=int, default=50000, help="主對話 prompt 的 token 數")
ap.add_argument("--gen", type=int, default=1500, help="主對話生成上限（要夠長，subagent 跑完前還在生成）")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--subs", type=int, default=3)
ap.add_argument("--snaps", type=int, default=0, help="QW_PAR_SNAPS（0 = 不設）")
ap.add_argument("--int8", action="store_true")
ap.add_argument("--env", action="append", default=[], help="另外的環境變數 K=V")
ap.add_argument("--out", default=None)
args = ap.parse_args()
OUT = args.out or os.path.join(ROOT, "build", "snap_evict", time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(OUT, exist_ok=True)
HOST, LOCK = "127.0.0.1", os.path.join(ROOT, "build", "gpu.lock")


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


class Server:
    def __init__(self):
        self.logp = os.path.join(OUT, "server.log")
        env = dict(os.environ, QW_CACHE_DBG="1")
        if args.snaps: env["QW_PAR_SNAPS"] = str(args.snaps)
        for kv in args.env: k, v = kv.split("=", 1); env[k] = v
        cache = os.path.join(OUT, "cache")
        cmd = ["lockf", LOCK, args.serve, "--gguf", args.model, "--port", str(args.port), "--ctx", "262144",
               "--cache-dir", cache, "--stats-file", ""] + (["--int8"] if args.int8 else [])
        log("server:", " ".join(cmd), f"QW_PAR_SNAPS={args.snaps or '-'}")
        self.f = open(self.logp, "w")
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
        t0 = time.time()
        while True:
            if self.p.poll() is not None: raise SystemExit(f"server 結束了（{self.p.returncode}），看 {self.logp}")
            try:
                c = http.client.HTTPConnection(HOST, args.port, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200: break
            except Exception: pass
            if time.time() - t0 > 900: self.stop(); raise SystemExit("server 15 分鐘還沒起來")
            time.sleep(1)
        log(f"server 起來了（{time.time() - t0:.0f} s）")

    def stop(self):
        if self.p.poll() is None:
            try: os.killpg(self.p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: self.p.wait(60)
            except subprocess.TimeoutExpired: os.killpg(self.p.pid, signal.SIGKILL); self.p.wait()
        self.f.close()
        while subprocess.run(["pgrep", "-f", f"qwental-serve.*--port {args.port}"], capture_output=True).stdout.strip(): time.sleep(0.5)
        log("server 停了")


def http_json(method, path, body=None, timeout=3600):
    c = http.client.HTTPConnection(HOST, args.port, timeout=timeout)
    c.request(method, path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    r = c.getresponse()
    d = r.read()
    c.close()
    return r.status, (json.loads(d) if d else None)


def post_msg(b):
    for _ in range(120):  # 529：退避重送
        st, d = http_json("POST", "/v1/messages", b)
        if st == 200: return d
        if st != 529: raise RuntimeError(f"HTTP {st}: {d}")
        time.sleep(2)
    raise RuntimeError("一直 529")


def ntok(b): return http_json("POST", "/v1/messages/count_tokens", b)[1]["input_tokens"]


def filler(tag, n):  # 確定性的假內容（同 slot_keep.py）
    words = ["alpha", "beta", "gamma", "delta", "kernel", "buffer", "slot", "prefix", "token", "state", "commit", "branch"]
    return "\n".join(f"{tag} {i}: " + " ".join(words[(i * 7 + k * 3) % len(words)] + str((i * 31 + k) % 997) for k in range(12)) for i in range(n))


TOOLS = [{"name": f"tool_{t}", "description": filler(f"tool {t} rule", 60), "input_schema": {"type": "object", "properties": {"arg": {"type": "string"}}}}
         for t in range(4)]
SYSTEM = "You are a coding agent. Follow the rules below.\n" + filler("system rule", 200)


def body(msgs, mt, system=SYSTEM, tools=TOOLS, stream=False):
    b = {"model": "q", "max_tokens": mt, "temperature": 0, "thinking": {"type": "disabled"}, "system": system, "messages": msgs}
    if tools: b["tools"] = tools
    if stream: b["stream"] = True
    return b


def text_of(d): return "".join(c.get("text", "") for c in d.get("content", []))


def fit(make, target):  # 行數調到 token 數 ≈ target
    lines = 200
    while True:
        n = ntok(make(lines))
        if n >= target * 0.97: return lines, n
        lines = int(lines * min(4.0, max(1.05, target / max(n, 1))))


srv = Server()
res = {"serve": args.serve, "snaps": args.snaps}
try:
    ASK = "\nNow write a numbered list of 200 short lines; line k quotes doc line k of the document verbatim. Do not stop early."
    lines, nm = fit(lambda n: body([{"role": "user", "content": filler("doc line", n) + ASK}], 1), args.main)
    m1 = [{"role": "user", "content": filler("doc line", lines) + ASK}]
    log(f"主對話 M ≈ {nm} token")
    res["main_prompt"] = nm

    started, mdone, mres = threading.Event(), threading.Event(), {}

    def run_main():  # 串流：第一個文字出現 = 開始生成
        c = http.client.HTTPConnection(HOST, args.port, timeout=3600)
        c.request("POST", "/v1/messages", json.dumps(body(m1, args.gen, stream=True)).encode(), {"Content-Type": "application/json"})
        r = c.getresponse()
        txt, usage = [], {}
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data: {"): continue
            ev = json.loads(ln[6:])
            if ev.get("type") == "content_block_delta":
                txt.append(ev["delta"].get("text", "")); started.set()
            elif ev.get("type") == "message_start": usage.update(ev["message"].get("usage", {}))
            elif ev.get("type") == "message_delta": usage.update(ev.get("usage", {}))
        c.close()
        mres["text"], mres["usage"] = "".join(txt), usage
        started.set(); mdone.set()

    t0 = time.time()
    tm = threading.Thread(target=run_main); tm.start()
    started.wait()
    log(f"M 開始生成（{time.time() - t0:.0f} s）")

    def sub(r, i, out):
        sysx = f"You are helper {r}-{i}.\n" + filler(f"helper {r}-{i} rule", 20 + 5 * i)
        doc = filler(f"note {r}-{i}", 60 + 70 * ((r + i) % 3))  # 約 5–15K token
        d = post_msg(body([{"role": "user", "content": doc + f"\nTask {r}-{i}: summarize the notes in two sentences."}], 64, system=sysx, tools=None))
        out.append((f"sub{r}-{i}", d["usage"]))

    subs = []
    for r in range(args.rounds):
        if mdone.is_set(): log(f"M 已經生成完（第 {r} 輪前），不再送 subagent"); break
        th = [threading.Thread(target=sub, args=(r, i, subs)) for i in range(args.subs)]
        [x.start() for x in th]; [x.join() for x in th]
        log(f"第 {r} 輪 subagent 完成；M {'已完成' if mdone.is_set() else '還在生成'}")
    res["rounds_done_while_main_ran"] = sum(1 for _ in subs) // max(args.subs, 1)
    tm.join()
    res["main1"] = mres["usage"]
    log(f"M 完成：{len(mres['text'])} 字元，usage {mres['usage']}")
    open(os.path.join(OUT, "main1.txt"), "w").write(mres["text"])

    m2 = m1 + [{"role": "assistant", "content": mres["text"]}, {"role": "user", "content": "Which doc line number did you stop at? Answer with the number only."}]
    n2 = ntok(body(m2, 1))
    d2 = post_msg(body(m2, 16))
    u = d2["usage"]
    res["main2"] = {"prompt": n2, **u}
    reuse = u.get("cache_read_input_tokens", 0)
    ok = reuse >= nm - 64
    res["ok"] = ok
    log(f"M2 prompt {n2}：重用 {reuse}（M 的 prompt {nm}）→ {'✅ 接上了' if ok else '❌ 重算了'}")
finally:
    srv.stop()

lg = open(os.path.join(OUT, "server.log"), errors="replace").read()
res["switches"] = sum(int(b) == int(a) + 1 for a, b in re.findall(r"共同前綴 (\d+) / (\d+)，從", lg))  # 生成中切換後接著算（每次都存快照）
res["prompt_lines"] = re.findall(r"\[anthropic\] prompt \d+（重用 \d+）", lg)
json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1, ensure_ascii=False)
log(f"生成中切換 {res['switches']} 次；log：{OUT}/server.log")
for p in res["prompt_lines"]: print("   ", p)
sys.exit(0 if res.get("ok") else 1)
