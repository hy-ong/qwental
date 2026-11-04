#!/bin/sh
# 交錯 A/B：兩個 build 輪流跑 n 次，各取每輪 GPU 時間最小值（排除其他程式搶 GPU 的干擾）
# 用法：bench/ab.sh build/run_A build/run_B "<ids>" <生成數> <次數>，環境變數照常傳（例如 QW_MTP=1 QW_SPEC_K=6）
A=$1; B=$2; IDS=$3; N=${4:-300}; R=${5:-3}
for i in $(seq "$R"); do
  for x in "$A" "$B"; do
    t=$(lockf build/gpu.lock "$x" "$IDS" "$N" 2>&1 | grep -oE '每輪 GPU [0-9.]+ ms|每個 token GPU 時間平均 [0-9.]+ ms' | grep -oE '[0-9.]+ ms' | head -1)
    echo "$x $t"
  done
done | awk '{v=$2+0; if(!($1 in m) || v<m[$1]) m[$1]=v} END {for (k in m) printf "%s 最小 %.2f ms\n", k, m[k]}'
