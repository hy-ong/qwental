#!/usr/bin/env python3
# 逐位置對照：把 Strata 生成的文字（gc_strata.json）接在同一個 chat prompt 後面，qwental 逐 token 算 logits（QW_PPL_DUMP），
# 在每個輸出位置比較 Strata 回報的 top-5 logprob 與 qwental 的 log softmax。
# 用法：bench/vs_strata_lp.py prep <name>（寫 build/lp_<name>.ids 與對齊資訊）；bench/vs_strata_lp.py cmp <name>
import json, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
from mlx_lm.utils import load_tokenizer
from pathlib import Path
tok = load_tokenizer(Path(ROOT) / 'models/Qwen3.8-Flash-Next-mixed-4-8bit')
exec(open(os.path.join(ROOT, 'bench/vs_strata_greedy.py')).read().split('N = 200')[0].split('ROOT =')[1].join(['ROOT =', '']) if False else '')
import importlib.util
spec = importlib.util.spec_from_file_location('g', os.path.join(ROOT, 'bench/vs_strata_greedy.py')); 
src = open(os.path.join(ROOT, 'bench/vs_strata_greedy.py')).read().split('\ndef post')[0]
g = {'__file__': os.path.join(ROOT, 'bench/vs_strata_greedy.py')}; exec(src, g)
P = dict(g['PROMPTS'])
name = sys.argv[2]
A = json.load(open(os.path.join(ROOT, 'build/gc_strata.json')))[name]
prompt = '<|im_start|>user\n' + P[name] + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
pids = tok.encode(prompt)
toks = [t for t, _, _ in A['lp']]
# Strata 的 token 字串逐一轉回 id（每個字串單獨編碼必須剛好一個 token，才能對齊）
oids = []
for t in toks:
    e = tok.encode(t)
    if len(e) != 1: oids = None; break
    oids.append(e[0])
if sys.argv[1] == 'prep':
    if oids is None: print('無法逐 token 對齊'); sys.exit(1)
    open(os.path.join(ROOT, 'build/lp_%s.ids' % name), 'w').write(','.join(map(str, pids + oids)))
    print(name, 'prompt', len(pids), '輸出', len(oids))
else:
    V = 248320
    L = np.fromfile(os.path.join(ROOT, 'build/lp_%s.f32' % name), dtype=np.float32).reshape(-1, V)
    n0 = len(pids) - 1  # 第 i 個輸出 token 由位置 n0 + i 的 logits 預測
    d, agree, nlp = [], 0, 0
    for i, (t, l, top) in enumerate(A['lp']):
        if n0 + i >= L.shape[0]: break
        x = L[n0 + i].astype(np.float64); x -= x.max(); ls = x - np.log(np.exp(x).sum())
        for tt, tl in top:
            e = tok.encode(tt)
            if len(e) == 1: d.append(ls[e[0]] - tl)
        ts = sorted(top, key=lambda z: -z[1])[0][0]; e = tok.encode(ts)
        agree += len(e) == 1 and int(np.argmax(ls)) == e[0]
        nlp += 1
    d = np.abs(np.array(d))
    print('%-6s %d 個位置：top-1 與 Strata 一致 %d（%.1f%%）；top-5 logprob 差 中位 %.4f、90%% %.4f、最大 %.4f' % (name, nlp, agree, 100 * agree / nlp, np.median(d), np.percentile(d, 90), d.max()))
