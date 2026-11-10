#!/usr/bin/env python3
# 快取 / 狀態重用的壓力測試（找「重用了錯的狀態 → 答非所問」）。一個指令跑完：
#   bench/cache_fuzz.py [--model iq3s|sc117|DIR] [--out DIR] [--scale 1.0] [--seed 1] [--verify 1] [--quick]
# Phase A：開一個所有功能都開著的 server（--parallel 16、SSD 快取、逼著常縮小 / 釋放 slot 的環境變數），跑像 Claude Code 的工作量：
#   主對話一輪一輪變長（上一輪的回答 + tool_use，再加 500–6000 token 的 tool_result）、幾波同時開始的 subagent（共用約 12K 的
#   tools + system 前綴，各自 2–3 輪）、旁邊的短請求、用戶端在 prefill 中途斷線再重送同一個請求、閒置超過 QW_SHRINK_IDLE 的空檔、
#   小的 slot 記憶體上限（QW_SLOT_MEM_GB）。temperature 0、不思考（少數請求開思考、budget 很小，走收尾後重新加入的路）、max_tokens 48。
#   server 開 QW_VERIFY_REUSE（閒置時從頭重算比對 logits）與 QW_CACHE_DBG，log 存在 OUT/serverA.log。
# Phase B：停掉 A，開 --parallel 1、沒有 SSD 快取、QW_NO_REUSE=1 的 server，逐一重送 A 的每個請求，比對輸出文字。
# 最後印摘要：不一致的請求、它在 A 用了哪些重用機制（[reuse] / [par] / [verify] 的 log）。
# GPU 安全：同一時間只載入一個模型；每個 server 都用 lockf build/gpu.lock 包住，A 完全停掉之後才開 B。port 預設 8081。
import argparse, http.client, json, os, random, re, signal, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main_repo():  # worktree 裡沒有 models/：用主 checkout 的
    try:
        c = subprocess.run(["git", "-C", ROOT, "rev-parse", "--git-common-dir"], capture_output=True, text=True).stdout.strip()
        if c: return os.path.dirname(os.path.abspath(os.path.join(ROOT, c)))
    except Exception: pass
    return ROOT


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="iq3s", help="iq3s（models/iq3s-gguf）、sc117（models/sc117-gguf）或 GGUF 目錄")
ap.add_argument("--out", default=None, help="輸出目錄（預設 build/cache_fuzz/<時間>）")
ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--scale", type=float, default=1.0, help="工作量倍數（輪數、subagent 數）")
ap.add_argument("--verify", type=int, default=1, help="QW_VERIFY_REUSE=N（0 = 不開）")
ap.add_argument("--quick", action="store_true", help="很小的版本（確認流程用，約 3–5 分鐘）")
ap.add_argument("--phase", default="AB", help="A、B 或 AB（B 只重送 OUT/phaseA.jsonl）")
ap.add_argument("--envA", action="append", default=[], help="Phase A 另外的環境變數 K=V（可多次；例如關掉某個機制做二分）")
ap.add_argument("--envB", action="append", default=[], help="Phase B 另外的環境變數 K=V（例如 QW_NO_SPEC=1）")
ap.add_argument("--only", default="", help="Phase B 只重送這些 seq（逗號分隔）")
ap.add_argument("--bname", default="phaseB", help="Phase B 結果的檔名（OUT/<bname>.jsonl），對照不同設定時用")
ap.add_argument("--int8", action="store_true", help="兩個 server 都加 --int8（使用者的 launcher 有開）")
ap.add_argument("--serve", default=os.path.join(ROOT, "build", "qwental-serve"))
args = ap.parse_args()

REPO = main_repo()
MODEL = {"iq3s": "models/iq3s-gguf", "sc117": "models/sc117-gguf"}.get(args.model, args.model)
if not os.path.isabs(MODEL):
    MODEL = os.path.join(ROOT, MODEL) if os.path.isdir(os.path.join(ROOT, MODEL)) else os.path.join(REPO, MODEL)
LOCK = os.path.join(REPO, "build", "gpu.lock")
OUT = os.path.abspath(args.out or os.path.join(ROOT, "build", "cache_fuzz", time.strftime("%m%d-%H%M%S")))
os.makedirs(OUT, exist_ok=True)
HOST = "127.0.0.1"
SHRINK_IDLE = 5
QUICK = args.quick
SC = args.scale * (0.35 if QUICK else 1.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- server
class Server:
    def __init__(self, name, extra_args, env):
        self.name, self.logp = name, os.path.join(OUT, f"server{name}.log")
        e = dict(os.environ)
        e.update({"QW_GGUF": MODEL, "QW_GGUF_PARTS": "all"})
        e.update(env)
        cmd = ["lockf", LOCK, args.serve, "--gguf", MODEL, "--port", str(args.port), "--stats-file", ""] + extra_args + (["--int8"] if args.int8 else [])
        log(f"server {name}: {' '.join(k + '=' + v for k, v in env.items())} {' '.join(cmd)}")
        self.f = open(self.logp, "w")
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT, env=e, cwd=ROOT, start_new_session=True)
        t0 = time.time()
        while True:
            if self.p.poll() is not None: raise SystemExit(f"server {name} 結束了（{self.p.returncode}），看 {self.logp}")
            try:
                c = http.client.HTTPConnection(HOST, args.port, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200: break
            except Exception: pass
            if time.time() - t0 > 900: self.stop(); raise SystemExit(f"server {name} 15 分鐘還沒起來")
            time.sleep(1)
        log(f"server {name} 起來了（{time.time() - t0:.0f} s）")

    def stop(self):
        if self.p.poll() is None:
            try: os.killpg(self.p.pid, signal.SIGTERM)  # lockf 與 server 同一個 process group
            except ProcessLookupError: pass
            try: self.p.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait()
        self.f.close()
        while subprocess.run(["pgrep", "-f", f"qwental-serve.*--port {args.port}"], capture_output=True).stdout.strip(): time.sleep(0.5)
        log(f"server {self.name} 停了")


def http_json(method, path, body=None, timeout=3600):
    c = http.client.HTTPConnection(HOST, args.port, timeout=timeout)
    c.request(method, path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    r = c.getresponse()
    d = r.read()
    c.close()
    return r.status, (json.loads(d) if d else None)


def status():
    try: return http_json("GET", "/status", timeout=30)[1] or {}
    except Exception: return {}


def post_msg(body):
    for k in range(60):  # 529（排隊滿、沒有 slot）：退避重送
        st, d = http_json("POST", "/v1/messages", body)
        if st == 200: return d
        if st != 529: raise RuntimeError(f"HTTP {st}: {d}")
        time.sleep(2)
    raise RuntimeError("一直 529")


def cancel_mid(body, delay):  # 串流送出，delay 秒後直接關掉連線（多半還在 prefill）
    b = dict(body, stream=True)
    c = http.client.HTTPConnection(HOST, args.port, timeout=600)
    c.request("POST", "/v1/messages", json.dumps(b).encode(), {"Content-Type": "application/json"})
    time.sleep(delay)
    try: c.sock.shutdown(2)
    except Exception: pass
    c.close()


def text_of(d):
    return "".join(x.get("text", "") + x.get("thinking", "") + (json.dumps(x.get("input")) if x.get("type") == "tool_use" else "")
                   for x in d.get("content", []))


# ---------------------------------------------------------------- 內容（確定性、彼此不同、混合各種文字）
EN = ("the a of to in is for on that with as by this from be are at or it an was which not has can will more data file function "
      "value return state buffer kernel slot prefix token cache layer model request server thread queue memory index offset "
      "length error result config build test commit branch merge update module package import export class struct").split()
ZH = "的一是在不了有和人這中大為上個國我以要他時來用們生到作地於出就分對成會可主發年動同工也能下過子說產種面而方後多定行學法所民得經十三之進著等部度家電力裡如水化高自二理起小物現實加量都兩體制機當使點從業本去把性好應開它合還因由其些然前外天政四日那社義事平形相全表間樣與關各重新線內數正心反你明看原又麼利比或但質氣第向道命此變條只沒結解問意建月公無系軍很情者最立代想已通並提直題黨程展五果料象員革位入常文總次品式活設及管特件長求老頭基資邊流路級少圖山統接知較將組見計別她手角期根論運農指幾九區強放決西被幹做必戰先回則任取據處隊南給色光門即保治北造百規熱領七海口東導器壓志世金增爭濟階油思術極交受聯什認六共權收證改清己美再採轉更單風切打白教速花帶安場身車例真務具萬每目至達走積示議聲報鬥完類八離華名確才科張信馬節話米整空元況今集溫傳土許步群廣石記需段研界拉林律叫且究觀越織裝影算低持音眾書布复容兒須際商非驗連斷深難近礦千週委素技備半辦青省列習響約支般史感勞便團往酸歷市克何除消構府稱太準精值號率族維劃選標寫存候毛親快效斯院查江型眼王按格養易置派層片始卻專狀育廠京識適屬圓包火住調滿縣局照參紅細引聽該鐵價嚴"


def gen_text(rng, ntok, tag):
    out, n = [], 0
    while n < ntok:
        k = rng.randrange(6)
        if k == 0:  # 英文段落
            w = [rng.choice(EN) for _ in range(rng.randint(40, 120))]
            out.append(" ".join(w).capitalize() + "."); n += len(w) * 1.15
        elif k == 1:  # 程式碼
            fn = f"{rng.choice(EN)}_{rng.choice(EN)}_{rng.randint(0, 999)}"
            body = "\n".join(f"    {rng.choice(EN)}{i} = {rng.choice(EN)}({rng.randint(0, 99999)}, '{rng.choice(EN)}')  # {rng.choice(EN)}" for i in range(rng.randint(4, 14)))
            s = f"def {fn}(x, y={rng.randint(0, 9)}):\n{body}\n    return {rng.choice(EN)}\n"
            out.append(s); n += len(s) / 3.2
        elif k == 2:  # 中文
            s = "".join(rng.choice(ZH) for _ in range(rng.randint(60, 200))) + "。"
            out.append(s); n += len(s) * 0.9
        elif k == 3:  # JSON 紀錄
            rows = [json.dumps({"id": rng.randint(0, 10**6), "op": rng.choice(EN), "ms": round(rng.random() * 100, 2), "ok": rng.random() < 0.8}) for _ in range(rng.randint(3, 10))]
            s = "\n".join(rows); out.append(s); n += len(s) / 3.0
        elif k == 4:  # 數字表
            s = "\n".join(" | ".join(str(rng.randint(0, 99999)) for _ in range(6)) for _ in range(rng.randint(3, 10)))
            out.append(s); n += len(s) / 2.6
        else:  # 檔案路徑與行號
            s = "\n".join(f"src/{rng.choice(EN)}/{rng.choice(EN)}.c:{rng.randint(1, 9000)}: {rng.choice(EN)} {rng.choice(EN)}" for _ in range(rng.randint(3, 12)))
            out.append(s); n += len(s) / 3.3
    return f"[{tag}]\n" + "\n\n".join(out)


R0 = random.Random(args.seed)
PREFIX_TOK = int(12000 * (0.6 if QUICK else 1.0))
TOOLS = [{"name": n, "description": gen_text(random.Random(f"tool{n}{args.seed}"), PREFIX_TOK * 0.55 / 8, f"tool {n}"),
          "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "arg": {"type": "string"}}, "required": ["path"]}}
         for n in ["Read", "Write", "Edit", "Bash", "Grep", "Glob", "Task", "WebFetch"]]
SYS_MAIN = "You are Claude Code, a coding agent. Answer briefly.\n" + gen_text(random.Random(f"sysmain{args.seed}"), PREFIX_TOK * 0.45, "main system")
SYS_SUB = {t: f"You are a {t} subagent. Answer briefly.\n" + gen_text(random.Random(f"sys{t}{args.seed}"), PREFIX_TOK * 0.45, f"{t} system") for t in ["explore", "review"]}
REMINDER = "<system-reminder>\n" + gen_text(random.Random(f"rem{args.seed}"), 600, "reminder") + "\n</system-reminder>\n"


def body(system, msgs, think=False):
    # messages 要深拷貝：之後的輪會往同一個 list 加東西（紀錄與重送的必須是送出當時的內容）
    b = {"model": "q", "max_tokens": 96 if think else 48, "temperature": 0, "system": system, "tools": TOOLS, "messages": json.loads(json.dumps(msgs))}
    if think: b["thinking"] = {"type": "enabled", "budget_tokens": 24}
    return b


# ---------------------------------------------------------------- 執行與紀錄
REC, rec_mu, seq = [], threading.Lock(), [0]
CANCEL_P = 0.18


def run_req(name, b, rng):
    with rec_mu:
        seq[0] += 1
        s = seq[0]
    cancels = []
    while rng.random() < CANCEL_P and len(cancels) < 2:  # 中途斷線，之後重送同一個請求
        d = round(rng.uniform(0.3, 6.0), 2)  # 長 prompt 要數十秒：多半落在分段 prefill 中途（par_snap）
        cancel_mid(b, d)
        cancels.append(d)
        time.sleep(rng.uniform(0, 0.5))
    t0 = time.time()
    d = post_msg(b)
    r = {"seq": s, "name": name, "body": b, "text": text_of(d), "stop": d.get("stop_reason"), "usage": d.get("usage"), "msg_id": d.get("id"),
         "t0": t0, "t1": time.time(), "cancels": cancels}
    with rec_mu: REC.append(r)
    u = d.get("usage") or {}
    log(f"  {name}: 新算 {u.get('cache_creation_input_tokens')}、重用 {u.get('cache_read_input_tokens')}、{time.time() - t0:.1f} s"
        + (f"、先取消 {cancels}" if cancels else "") + f"：{r['text'][:60]!r}")
    return d


def tool_round(rng, d, msgs, tag, lo, hi):  # 把回答 + 一個 tool_use 接上，再加一個 tool_result（lo–hi token）
    tid = f"toolu_{tag}_{rng.randint(0, 10**8):08d}"
    txt = text_of(d) or "ok"
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": txt},
                                                  {"type": "tool_use", "id": tid, "name": rng.choice(TOOLS)["name"], "input": {"path": f"src/{rng.choice(EN)}.c"}}]})
    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": gen_text(rng, rng.randint(lo, hi), f"{tag} result")}]})


def subagent(k, typ, turns, seed):
    rng = random.Random(seed)
    msgs = [{"role": "user", "content": REMINDER + f"Task {k}: " + gen_text(rng, rng.randint(80, 400), f"task {k}") + "\nWhat is the first word of the task? Reply in one line."}]
    for t in range(turns):
        d = run_req(f"sub{k}.{typ}.t{t}", body(SYS_SUB[typ], msgs, think=rng.random() < 0.12), rng)
        tool_round(rng, d, msgs, f"s{k}t{t}", 500, 4000)
        msgs[-1]["content"].append({"type": "text", "text": "Continue. Quote one number from the tool result."})
    return text_of(d)


def side(k, seed):  # 旁邊的短請求（權限判斷、標題）：沒有工具、短 system
    rng = random.Random(seed)
    run_req(f"side{k}", {"model": "q", "max_tokens": 32, "temperature": 0, "system": "Classify the request. Reply with one word.",
                         "messages": [{"role": "user", "content": gen_text(rng, rng.randint(100, 900), f"side {k}")}]}, rng)


def wave(rng, w, n_explore, n_review, turns):
    th, res = [], {}
    for i in range(n_explore + n_review):
        typ = "explore" if i < n_explore else "review"
        sd = rng.randint(0, 10**9)
        t = threading.Thread(target=lambda i=i, typ=typ, sd=sd: res.__setitem__(i, subagent(f"{w}{i}", typ, turns, sd)))
        th.append(t)
    for t in th: t.start(); time.sleep(rng.uniform(0, 0.3))  # 幾乎同時開始（coalescing）
    for t in th: t.join()
    return "\n".join(f"subagent {i}: {res.get(i, '')}" for i in sorted(res))


def phase_a():
    env = {"QW_SLOT_MEM_GB": "6", "QW_SHRINK_IDLE": str(SHRINK_IDLE), "QW_IDLE_SNAPS": "2", "QW_MAX_WAIT_S": "0", "QW_CACHE_DBG": "1",
           "QW_LOG_ID": "1", "QW_CACHE_GB": "8"}
    if args.verify > 0: env["QW_VERIFY_REUSE"] = str(args.verify)
    for kv in args.envA:
        k, _, v = kv.partition("=")
        env[k] = v
    cdir = os.path.join(OUT, "cacheA")
    os.makedirs(cdir, exist_ok=True)
    for f in os.listdir(cdir): os.remove(os.path.join(cdir, f))
    srv = Server("A", ["--parallel", "16", "--cache-dir", cdir], env)
    t0 = time.time()
    try:
        rng = random.Random(args.seed * 7919)
        msgs = [{"role": "user", "content": REMINDER + gen_text(rng, 3000, "user request") + "\nSummarize the request in one sentence."}]
        nturn = max(3, int(8 * SC))
        waves = {1: (max(2, int(4 * SC)), 0), max(2, nturn // 2 + 1): (max(2, int(3 * SC)), max(1, int(2 * SC)))}
        side_n = 0
        for t in range(nturn):
            if t in (2, nturn - 2):  # 閒置：超過 QW_SHRINK_IDLE，閒置的 slot 會被縮小
                log(f"閒置 {SHRINK_IDLE + 4} 秒"); time.sleep(SHRINK_IDLE + 4)
            sides = []
            if rng.random() < 0.6:  # 主對話的同時有短請求
                side_n += 1
                sides.append(threading.Thread(target=side, args=(side_n, rng.randint(0, 10**9)))); sides[-1].start()
            d = run_req(f"main.t{t}", body(SYS_MAIN, msgs, think=rng.random() < 0.15), rng)
            for s in sides: s.join()
            tool_round(rng, d, msgs, f"m{t}", 500, 6000)
            if t in waves:  # 主對話派出 subagent，等它們回來（結果當成 tool_result 的一部分）
                ne, nr = waves[t]
                log(f"subagent 一波：{ne} explore + {nr} review")
                summ = wave(rng, t, ne, nr, 2 if QUICK else 3)
                msgs[-1]["content"].append({"type": "text", "text": "Subagent results:\n" + summ})
            msgs[-1]["content"].append({"type": "text", "text": "Next step: reply with one short sentence."})
        log("等 [verify] 做完")
        tw = time.time()
        while time.time() - tw < 600:
            v = status().get("verify")
            if not v or v.get("pending", 0) == 0: break
            time.sleep(3)
        log(f"Phase A：{len(REC)} 個請求、{time.time() - t0:.0f} s；verify = {status().get('verify')}")
    finally:
        with open(os.path.join(OUT, "phaseA.jsonl"), "w") as f:
            for r in sorted(REC, key=lambda r: r["seq"]): f.write(json.dumps(r, ensure_ascii=False) + "\n")
        srv.stop()


def phase_b():
    recs = [json.loads(l) for l in open(os.path.join(OUT, "phaseA.jsonl"))]
    if args.only: recs = [r for r in recs if str(r["seq"]) in args.only.split(",")]
    envb = {"QW_NO_REUSE": "1", "QW_NO_CACHE": "1", "QW_LOG_ID": "1"}
    for kv in args.envB:
        k, _, v = kv.partition("=")
        envb[k] = v
    srv = Server("B" if args.bname == "phaseB" else args.bname, ["--parallel", "1", "--cache-dir", ""], envb)
    t0 = time.time()
    out = []
    try:
        for r in recs:
            d = post_msg(r["body"])
            out.append({"seq": r["seq"], "text": text_of(d), "stop": d.get("stop_reason"), "msg_id": d.get("id")})
            same = out[-1]["text"] == r["text"]
            log(f"  B {r['name']}: {'同' if same else '不同'}" + ("" if same else f"：{out[-1]['text'][:120]!r}"))
    finally:
        with open(os.path.join(OUT, args.bname + ".jsonl"), "w") as f:
            for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
        srv.stop()
    log(f"Phase B：{len(out)} 個請求、{time.time() - t0:.0f} s")


# ---------------------------------------------------------------- 比對與摘要
def summary():
    A = {json.loads(l)["seq"]: json.loads(l) for l in open(os.path.join(OUT, "phaseA.jsonl"))}
    pb = os.path.join(OUT, args.bname + ".jsonl")
    B = {json.loads(l)["seq"]: json.loads(l) for l in open(pb)} if os.path.exists(pb) else {}
    lines = open(os.path.join(OUT, "serverA.log"), errors="replace").read().splitlines() if os.path.exists(os.path.join(OUT, "serverA.log")) else []
    rid = {}
    for l in lines:
        m = re.match(r"\[req\] #(\d+) (\S+) slot (-?\d+) prompt (\d+) cached (\d+)", l)
        if m: rid[m.group(2)] = (int(m.group(1)), int(m.group(3)), int(m.group(4)), int(m.group(5)))

    def ctx(jid):
        pat = re.compile(rf"#{jid}\b")
        return [l for l in lines if pat.search(l) and not l.startswith("[req]")]

    verify = [l for l in lines if l.startswith("[verify]")]
    vbad = [l for l in verify if " MISMATCH " in l]
    vtie = [l for l in verify if " TIE " in l]
    print("\n================ 摘要 ================")
    print(f"輸出目錄：{OUT}")
    mech = {"跟著 # (coalescing)": "跟著 #", "共用前綴等待": "共用前綴", "記下分段 (cancel snap)": "記下分段", "縮小閒置 slot": "縮小閒置",
            "釋放前寫入 SSD": "釋放前寫入 SSD", "從 SSD 載入": "從 SSD 載入", "從 slot 複製前綴 (xs_copy)": "複製前綴", "KV 容量變更": "KV 容量",
            "收尾後重新加入 (think wrap)": "[think]", "用戶端斷線": "用戶端斷線"}
    print("Phase A 的機制觸發次數：" + "、".join(f"{k} {sum(v in l for l in lines)}" for k, v in mech.items()))
    print(f"[verify]：{len(verify)} 筆，MISMATCH {len(vbad)}、TIE {len(vtie)}、OK {len(verify) - len(vbad) - len(vtie)}")
    for l in vbad: print("  " + l[:600])
    nd, early = 0, []
    for s, a in sorted(A.items()):
        b = B.get(s)
        if not b: continue
        if a["text"] == b["text"]: continue
        nd += 1
        p = 0
        while p < min(len(a["text"]), len(b["text"])) and a["text"][p] == b["text"][p]: p += 1
        info = rid.get(a["msg_id"])
        tag = "早" if p < 24 else "晚"
        if p < 24: early.append(s)
        print(f"\n--- 不一致 #{s} {a['name']}（{tag}：第 {p} 個字元開始不同）server #{info[0] if info else '?'} slot {info[1] if info else '?'} "
              f"prompt {info[2] if info else '?'} 重用 {info[3] if info else '?'}；先取消 {a['cancels']}")
        print(f"   A: {a['text'][:200]!r}")
        print(f"   B: {b['text'][:200]!r}")
        if info:
            for l in ctx(info[0])[-14:]: print("   | " + l[:400])
    print(f"\n比對：{len(B)} / {len(A)} 個請求，不一致 {nd}（其中前 24 個字元就不同的 {len(early)}）")
    print("判讀：文字不同本身不代表錯 —— 從不在 64 倍數的位置接續 prefill，最後 logits 與一次算完的 KL 就有 0.01–0.4（build/run 的 QW_PF_SPLIT，"
          "跟「一次 prefill vs 逐 token decode」同量級），greedy 遇到接近平手就會走不同的路。看 [verify]：MISMATCH（KL > QW_VERIFY_KL，預設 1）"
          "才是重用到錯的狀態；xs_copy、從對齊的檢查點接續應該 KL = 0（逐位元相同）。")


if "A" in args.phase: phase_a()
if "B" in args.phase: phase_b()
summary()
