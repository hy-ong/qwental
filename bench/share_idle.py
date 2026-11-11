#!/usr/bin/env python3
# 平行派出的 subagent 共用前綴（tools + system + 同一段 user 文字），閒置縮減（qw_engine_par_shrink）之後再來一個同類 subagent，
# 接不接得上共用前綴。一個指令開 server、跑完、關掉：
#   bench/share_idle.py [--serve build/qwental-serve] [--shared 15000] [--subs 3] [--idle 5] [--out DIR] [--int8]
# 流程（temperature 0、不思考）：
#   1. --subs 個同時送的請求：tools + system + 同一段 user 文件（共約 --shared 個 token），後面各接不同的任務文字（max_tokens 32）
#      → 同時在 prefill，合併（log「共用前綴 … 等 #x 算到那裡再複製」），領頭的在交接點記快照
#   2. 閒置超過 QW_SHRINK_IDLE（--idle 秒）後送一個無關的短請求（縮減只在排程迴圈跑）：log 出現縮小閒置的 slot
#      （snap_prune 只留每組最長的幾份快照）
#   3. 再送一個同前綴、新任務文字的請求：看 usage.cache_read_input_tokens（= log 的「重用 R」）
# 預期：重用 ≥ 共用前綴（往下對齊 64）− 64。舊版的閒置縮減會丟掉交接點的快照，只剩 SSD 的持久化點（system 結尾等）能接。
# SSD 快取用每次新開的目錄。GPU 安全：server 用 lockf build/gpu.lock 包住、port 8081、結束（含例外 / Ctrl-C）一定關掉。
import argparse, http.client, json, os, re, signal, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--serve", default=os.path.join(ROOT, "build", "qwental-serve"))
ap.add_argument("--model", default=os.path.join(ROOT, "models", "sc117-gguf"))
ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--shared", type=int, default=15000, help="共用前綴的 token 數（tools + system + 共用的 user 文件）")
ap.add_argument("--subs", type=int, default=3)
ap.add_argument("--idle", type=float, default=5, help="QW_SHRINK_IDLE（秒）")
ap.add_argument("--wait", type=float, default=20, help="無關請求之後最多等幾秒看到縮小閒置 slot")
ap.add_argument("--int8", action="store_true")
ap.add_argument("--env", action="append", default=[], help="另外的環境變數 K=V")
ap.add_argument("--out", default=None)
args = ap.parse_args()
OUT = args.out or os.path.join(ROOT, "build", "share_idle", time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(OUT, exist_ok=True)
HOST, LOCK = "127.0.0.1", os.path.join(ROOT, "build", "gpu.lock")
LOGP = os.path.join(OUT, "server.log")


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


class Server:
    def __init__(self):
        env = dict(os.environ, QW_CACHE_DBG="1", QW_SHRINK_IDLE=str(args.idle))
        for kv in args.env: k, v = kv.split("=", 1); env[k] = v
        cmd = ["lockf", LOCK, args.serve, "--gguf", args.model, "--port", str(args.port), "--ctx", "262144",
               "--cache-dir", os.path.join(OUT, "cache"), "--stats-file", ""] + (["--int8"] if args.int8 else [])
        log("server:", " ".join(cmd), f"QW_SHRINK_IDLE={args.idle}")
        self.f = open(LOGP, "w")
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
        t0 = time.time()
        while True:
            if self.p.poll() is not None: raise SystemExit(f"server 結束了（{self.p.returncode}），看 {LOGP}")
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


def mem_gb():
    try: return http_json("GET", "/status", timeout=10)[1].get("slot_mem_gb")
    except Exception: return None


def filler(tag, n):  # 確定性的假內容（同 snap_evict.py）
    words = ["alpha", "beta", "gamma", "delta", "kernel", "buffer", "slot", "prefix", "token", "state", "commit", "branch"]
    return "\n".join(f"{tag} {i}: " + " ".join(words[(i * 7 + k * 3) % len(words)] + str((i * 31 + k) % 997) for k in range(12)) for i in range(n))


TOOLS = [{"name": f"tool_{t}", "description": filler(f"tool {t} rule", 30), "input_schema": {"type": "object", "properties": {"arg": {"type": "string"}}}}
         for t in range(2)]
SYSTEM = "You are a research subagent. Follow the rules below.\n" + filler("agent rule", 60)


def body(user, mt):
    return {"model": "q", "max_tokens": mt, "temperature": 0, "thinking": {"type": "disabled"}, "system": SYSTEM, "tools": TOOLS,
            "messages": [{"role": "user", "content": user}]}


def tail(i):  # 各 subagent 不同的任務（約 1K token）
    return f"\n\nTask {i}: the notes below belong to case {i}. Summarize them in one sentence.\n" + filler(f"case {i} note", 25)


def tot(u): return u.get('input_tokens', 0) + u.get('cache_creation_input_tokens', 0) + u.get('cache_read_input_tokens', 0)


def read_log(): return open(LOGP, errors="replace").read()


srv = Server()
res = {"serve": args.serve, "idle": args.idle}
try:
    lines = 50
    while True:  # 共用的 user 文件行數調到 tools + system + 文件 ≈ --shared
        n = ntok(body(filler("shared doc", lines), 1))
        if n >= args.shared * 0.98: break
        lines = int(lines * min(4.0, max(1.05, args.shared / max(n, 1))))
    DOC = filler("shared doc", lines)
    res["shared_body"] = n
    log(f"共用部分 ≈ {n} token（tools + system + 文件 {lines} 行）")
    res["mem0"] = mem_gb()

    out, th = {}, []
    for i in range(args.subs):
        def go(i=i): out[i] = post_msg(body(DOC + tail(i), 32))["usage"]
        th.append(threading.Thread(target=go))
    t0 = time.time()
    [x.start() for x in th]; [x.join() for x in th]
    res["subs"] = [out[i] for i in range(args.subs)]
    log(f"{args.subs} 個 subagent 完成（{time.time() - t0:.0f} s）：" + "；".join(f"prompt {tot(u)} 重用 {u.get('cache_read_input_tokens', 0)}" for u in res["subs"]))
    res["mem_after_subs"] = mem_gb()
    lg = read_log()
    co = [int(x) for x in re.findall(r"共用前綴 (\d+) 個 token（prompt", lg)]
    res["coalesce"] = re.findall(r"\[par\] #\d+ (?:與|跟著).*", lg)
    for c in res["coalesce"]: print("   ", c)
    if not co: log("❗ 沒有合併（log 沒有「共用前綴 … 等 #x 算到那裡再複製」）")
    shared = max(co) if co else n
    res["shared_prefix"] = shared
    res["shared_aligned"] = shared // 64 * 64

    # 閒置縮減只在排程迴圈裡跑（mem_guard → qw_engine_par_trim），沒有請求時不跑：閒置超過 --idle 秒後送一個無關的短請求
    # （像主對話在 subagent 回來之後繼續），讓它把閒置的 subagent slot 縮掉
    time.sleep(args.idle + 3)
    pos0 = len(read_log())
    post_msg({"model": "q", "max_tokens": 8, "temperature": 0, "thinking": {"type": "disabled"}, "system": "You are the main agent.",
              "messages": [{"role": "user", "content": filler("main chat", 40) + "\nReply OK."}]})
    t1 = time.time()
    while time.time() - t1 < args.wait:  # 等縮小的 log
        s = read_log()[pos0:]
        if "縮小閒置的 slot" in s: break
        time.sleep(1)
    s = read_log()[pos0:]
    res["shrink"] = re.findall(r"\[mem\].*", s)
    log("無關的請求之後" + ("看到縮小：" if "縮小閒置的 slot" in s else "❗ 沒看到縮小閒置的 slot"))
    for x in res["shrink"]: print("   ", x)
    res["mem_idle"] = mem_gb()

    pos1 = len(read_log())
    u4 = post_msg(body(DOC + tail(args.subs + 7), 32))["usage"]
    res["late"] = u4
    reuse = u4.get("cache_read_input_tokens", 0)
    res["mem_after_late"] = mem_gb()
    time.sleep(1)
    res["late_log"] = [x for x in read_log()[pos1:].splitlines() if re.search(r"重用|cache\]|SSD|ssd|複製|xs|dc_", x)]
    ok = reuse >= res["shared_aligned"] - 64
    res["ok"] = ok
    log(f"之後的 subagent：prompt {tot(u4)}，重用 {reuse}（共用前綴 {shared}，對齊 {res['shared_aligned']}）→ {'✅ 接上了' if ok else '❌ 沒接上共用前綴'}")
    for x in res["late_log"]: print("   ", x)
    log(f"slot_mem_gb：開始 {res['mem0']}、subagent 後 {res['mem_after_subs']}、閒置後 {res['mem_idle']}、最後 {res['mem_after_late']}")
finally:
    srv.stop()

json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1, ensure_ascii=False)
log(f"log：{LOGP}")
sys.exit(0 if res.get("ok") else 1)
