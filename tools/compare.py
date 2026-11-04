# 比對 engine 與參考答案（tools/dump_ref.py 產生的 build/ref/acts.npz）
# 用法：ref/venv/bin/python tools/compare.py [engine dump 目錄] [engine 生成的 token（逗號分隔）]
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build/eng"
GEN = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
ref = np.load(ROOT / "build/ref/acts.npz")
T = len(ref["ids"])

print("## 逐層殘差流（相對誤差 = ‖engine − ref‖ / ‖ref‖）\n")
print("| 層 | " + " | ".join(f"t={t}" for t in range(T)) + " |")
print("|---|" + "---|" * T)
worst = 0
for l in range(48):
    row = []
    for t in range(T):
        eng = np.fromfile(ENG / f"layers_t{t}.bin", dtype=np.float32).reshape(48, -1)[l]
        r = ref[f"layer{l:02d}"][t]
        e = np.linalg.norm(eng - r) / max(np.linalg.norm(r), 1e-9)
        worst = max(worst, e)
        row.append(f"{e:.1e}")
    if l < 4 or l % 8 == 7 or l == 47:
        print(f"| {l} | " + " | ".join(row) + " |")
print(f"\n最大相對誤差：{worst:.2e}")

lg = np.fromfile(ENG / "logits_last.bin", dtype=np.float32)
rl = ref["logits"][-1]
print(f"\n## 最後位置的 logits\n\n- 相對誤差 {np.linalg.norm(lg - rl) / np.linalg.norm(rl):.2e}，最大絕對差 {np.abs(lg - rl).max():.3f}")
te, tr = np.argsort(-lg)[:10], np.argsort(-rl)[:10]
print(f"- 前 10 名：engine {te.tolist()}")
print(f"           參考   {tr.tolist()}")
print(f"- 前 5 名重疊：{len(set(te[:5]) & set(tr[:5]))}/5，第一名{'相同 ✅' if te[0] == tr[0] else '不同 ❌'}")

g = json.loads((ROOT / "build/ref/greedy.json").read_text())
if GEN:
    ref_gen = g["ids"][T:T + len(GEN)]
    same = 0
    for a, b in zip(GEN, ref_gen):
        if a != b:
            break
        same += 1
    print(f"\n## Greedy 生成\n\n- 參考：{ref_gen}\n- engine：{GEN}\n- 從頭開始連續一致：{same}/{len(GEN)} 個 token")
    try:
        from mlx_lm.utils import load_tokenizer
        tok = load_tokenizer(ROOT / "models/Qwen3.8-Flash-Next-mixed-4-8bit")
        print(f"- engine 文字：{tok.decode(list(ref['ids']) + GEN)!r}")
        print(f"- 參考文字：  {g['text']!r}")
    except Exception as ex:
        print("（無法解碼文字：", ex, "）")
