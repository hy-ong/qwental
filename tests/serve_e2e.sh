#!/bin/bash
# qwental-serve 的端到端檢查（需要一個已經在跑的 server；這個腳本不會啟動或載入模型）。
# 先在另一個終端機啟動（不要用正在服務的 8080）：
#   build/qwental-serve --gguf models/iq3s-gguf --port 8081
# 然後：tests/serve_e2e.sh [port]
# 每一項印 ok / FAIL；最後印 FAIL 數。需要 curl、python3。
set -u
PORT=${1:-8081}
U=http://127.0.0.1:$PORT
T=$(mktemp -d)
trap 'rm -rf "$T"; jobs -p | xargs kill 2>/dev/null' EXIT
fails=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
msg() { # msg <file> <json body>：串流請求，輸出存檔
    curl -sN "$U/v1/messages?beta=true" -H 'content-type: application/json' -d "$2" >"$1"
}
status() { curl -s "$U/status"; }
field() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

curl -sf "$U/health" >/dev/null || { echo "server not running on $PORT"; exit 1; }

# 1. 基本串流：message_start / message_delta（usage 同 Strata）/ message_stop；模型名稱照抄
msg "$T/1" '{"model":"claude-haiku-5-5","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"Say hi"}]}'
check "stream has message_start" "grep -q '^event: message_start' $T/1"
check "model name echoed" "grep -q '\"model\":\"claude-haiku-5-5\"' $T/1"
check "message_delta usage has cache fields and input_tokens 1" \
    "grep '^data: {\"type\":\"message_delta\"' $T/1 | grep -q '\"input_tokens\":1,\"cache_creation_input_tokens\":[0-9]*,\"cache_read_input_tokens\":[0-9]*'"
check "stream ends with message_stop" "grep -q '^event: message_stop' $T/1"
# 同一個 prompt 再一次：cache_read 應該 > 0
# （短 prompt 沒有可重用的快照點，所以用約 3K token 的 prompt 送兩次）
python3 -c "import json;print(json.dumps({'model':'m','max_tokens':16,'stream':True,'messages':[{'role':'user','content':' '.join(f'item{i}' for i in range(800))+' Say hi.'}]}))" >"$T/mid.json"
curl -sN "$U/v1/messages" -H 'content-type: application/json' --data-binary @"$T/mid.json" >/dev/null
curl -sN "$U/v1/messages" -H 'content-type: application/json' --data-binary @"$T/mid.json" >"$T/1b"
check "repeat request reads the prefix cache" "grep '^data: {\"type\":\"message_delta\"' $T/1b | grep -qv '\"cache_read_input_tokens\":0,'"

# 2. 非串流：usage 與 content
curl -s "$U/v1/messages" -H 'content-type: application/json' \
    -d '{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"Say hi"}]}' >"$T/2"
check "non-stream usage" "grep -q '\"usage\":{\"input_tokens\":1,\"cache_creation_input_tokens\"' $T/2"

# 3. stop_sequences：只比對回答文字，命中時不含停止字串、stop_reason = stop_sequence、stop_sequence 有跳脫
msg "$T/3" '{"model":"m","max_tokens":400,"stream":true,"thinking":{"type":"disabled"},"stop_sequences":["7\""],"messages":[{"role":"user","content":"Write the numbers 1 to 10, each followed by a double quote character, like 1\" 2\" 3\""}]}'
sse_py='import json,sys
ev=[json.loads(l[6:]) for l in open(sys.argv[1]) if l.startswith("data: {")]
d=[e for e in ev if e.get("type")=="message_delta"][0]["delta"]
text="".join(e["delta"].get("text","") for e in ev if e.get("type")=="content_block_delta")
ok=d["stop_reason"]=="stop_sequence" and d["stop_sequence"]=="7\"" and "7\"" not in text
sys.exit(0 if ok else 1)'
check "stop_sequence: reason, escaped value, not in text" "python3 -c '$sse_py' $T/3"

# 4. 錯誤格式：404 not_found_error、/v1/v1 別名、413、count_tokens
check "404 type" "curl -s $U/nope | grep -q not_found_error"
check "/v1/v1/messages alias" "curl -s $U/v1/v1/messages/count_tokens -H 'content-type: application/json' -d '{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}' | grep -q input_tokens"
check "413 request_too_large" "python3 -c \"import socket;s=socket.create_connection(('127.0.0.1',$PORT));s.sendall(b'POST /v1/messages HTTP/1.1\\r\\nContent-Length: 999999999999\\r\\n\\r\\n');print(s.recv(4096).decode())\" | grep -q 'HTTP/1.1 413.*' "
check "prompt too long wording" "python3 -c \"import json;print(json.dumps({'model':'m','max_tokens':10,'messages':[{'role':'user','content':'hello '*400000}]}))\" | curl -s $U/v1/messages -H 'content-type: application/json' --data-binary @- | grep -q 'prompt is too long: [0-9]* tokens > [0-9]* maximum'"
check "dashboard served" "curl -s $U/dashboard | grep -qi '<html'"

# 5. keepalive：長 prompt（約 60K token）prefill 期間每 10 秒有 ping
python3 -c "import json;print(json.dumps({'model':'m','max_tokens':16,'stream':True,'messages':[{'role':'user','content':' '.join(f'word{i}' for i in range(12000))+' Summarize in one word.'}]}))" >"$T/long.json"
curl -sN "$U/v1/messages" -H 'content-type: application/json' --data-binary @"$T/long.json" >"$T/5"
check "ping during long prefill (needs prefill > 10 s)" "grep -q '^event: ping' $T/5"

# 6. 取消：長 prompt 開始後 3 秒斷線 → /status 的 cancelled + 1、之後沒有進行中的對話
c0=$(status | field 'd.get("cancelled",0)')
python3 -c "import json;print(json.dumps({'model':'m','max_tokens':2000,'stream':True,'messages':[{'role':'user','content':' '.join(f'other{i}' for i in range(12000))+' Summarize.'}]}))" >"$T/long2.json"
curl -sN --max-time 3 "$U/v1/messages" -H 'content-type: application/json' --data-binary @"$T/long2.json" >/dev/null 2>&1
for _ in $(seq 20); do sleep 0.5; [ "$(status | field 'len(d.get("active",[]))')" = 0 ] && break; done
c1=$(status | field 'd.get("cancelled",0)')
check "disconnect cancels the request (cancelled $c0 -> $c1)" "[ $c1 -gt $c0 ]"
check "no active conversation after cancel" "[ \"\$(status | field 'len(d.get(\"active\",[]))')\" = 0 ]"
# 重試同一個 prompt：應該從取消時的位置接續（log 的「重用」接近取消前算到的地方；看 server 的 log）

# 7. 卡住的用戶端（Ctrl-Z）不影響別人：A 被 SIGSTOP，B 照常完成
curl -sN "$U/v1/messages" -H 'content-type: application/json' \
    -d '{"model":"m","max_tokens":4000,"stream":true,"messages":[{"role":"user","content":"Write a very long story about a lighthouse."}]}' >/dev/null &
A=$!
sleep 2
kill -STOP $A
t0=$(date +%s)
msg "$T/7" '{"model":"m","max_tokens":200,"stream":true,"messages":[{"role":"user","content":"Count to 20."}]}'
t1=$(date +%s)
kill -CONT $A 2>/dev/null; kill $A 2>/dev/null
check "other request finishes while one client is stopped ($((t1 - t0)) s)" "grep -q message_stop $T/7 && [ $((t1 - t0)) -lt 60 ]"

# 8. answer room：max_tokens 1024、會想很久的題目 → 還是有 text 區塊（server log 有 [think]）
msg "$T/8" '{"model":"m","max_tokens":1024,"stream":true,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"Prove carefully that there are infinitely many primes of the form 4k+3, considering every edge case, then state the result in one sentence."}]}'
check "answer room leaves a text block" "grep -q '\"type\":\"text_delta\"' $T/8"

# 9. 並發：20 個同時（> 16）都要完成；排隊的有 ping 或直接完成
for i in $(seq 20); do msg "$T/9.$i" "{\"model\":\"m\",\"max_tokens\":64,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Say the number $i\"}]}" & done
wait
n=$(grep -l message_stop "$T"/9.* | wc -l | tr -d ' ')
check "20 concurrent requests all complete ($n/20)" "[ $n = 20 ]"
status | python3 -m json.tool | head -40

echo "$fails FAILED"
exit $((fails != 0))
