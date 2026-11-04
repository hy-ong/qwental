#!/bin/sh
# 同一份權重對照：Strata IQ3_S + DFlash2 與 qwental（GGUF profile，同一個 IQ3_S GGUF），兩個模型不能同時載入。
# 順序交錯（S Q、Q S、S Q…）以排除溫度與先後的影響。用法：lockf build/gpu.lock bench/vs_iq3s.sh [輪數]；ONLY=prose,code… 只跑部分 prompt
cd "$(dirname "$0")/.."
S=${STRATA_DIR:-$HOME/.local/share/Strata} # Strata 的安裝目錄（有 run-iq3_s.sh 等啟動腳本）
R=${1:-2}
strata() {
  "$S/run-iq3_s.sh" > build/vs_iq3s_s.log 2>&1 &
  until curl -s -m2 -o /dev/null -w "%{http_code}" localhost:8080/v1/models 2>/dev/null | grep -q 200; do sleep 1; done
  python3 bench/vs_server.py http://127.0.0.1:8080 strata-iq3s 1 | grep -v '^{'
  pkill -f "serve/server.py --engine strata"; sleep 2; pkill -f build-metal/strata; while pgrep -f build-metal/strata >/dev/null; do sleep 0.5; done
}
qwental() {
  QW_GGUF=models/iq3s-gguf QW_GGUF_PARTS=all ./build/qwental-serve --port 18081 > build/vs_iq3s_q.log 2>&1 &
  until grep -q "監聽" build/vs_iq3s_q.log 2>/dev/null; do sleep 0.5; done
  python3 bench/vs_server.py http://127.0.0.1:18081 qwental-iq3s 1 | grep -v '^{'
  pkill -f "qwental-serve --port 18081"; while pgrep -f "qwental-serve --port 18081" >/dev/null; do sleep 0.3; done
}
for r in $(seq "$R"); do
  if [ $((r % 2)) = 1 ]; then strata; qwental; else qwental; strata; fi
done
