# 每種 ggml 格式各取一個 GGUF tensor 的幾列，GPU 解碼（build/dqtest）與 gguf 套件的 numpy 參考比對。
# 用法：ref/venv/bin/python -I tools/dqtest.py <GGUF 目錄>
import sys, os, glob, struct, subprocess, tempfile, numpy as np
from gguf import quants, GGMLQuantizationType as Q
def rd(f, fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def rstr(f): n, = rd(f, 'Q'); return f.read(n).decode('utf-8', 'replace')
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        at, n = rd(f, 'IQ'); return [val(f, at) for _ in range(n)]
    f.read(SZ[t])
BLK = {2: (32, 18), 7: (32, 24), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 42: (64, 18)}
pick = {}
for fn in sorted(glob.glob(os.path.join(sys.argv[1], '*.gguf'))):
    with open(fn, 'rb') as f:
        f.read(4); ver, nt, nkv = rd(f, 'IQQ')
        for _ in range(nkv): rstr(f); t, = rd(f, 'I'); val(f, t)
        for _ in range(nt):
            name = rstr(f); nd, = rd(f, 'I'); dims = rd(f, 'Q' * nd); t, = rd(f, 'I'); off, = rd(f, 'Q')
            if t in BLK and t not in pick: pick[t] = [fn, name, dims, off]
        d0 = (f.tell() + 31) // 32 * 32
    for t in pick:
        if len(pick[t]) == 4: pick[t].append(d0)
ok = True
for t, (fn, name, dims, off, d0) in sorted(pick.items()):
    qb, bb = BLK[t]; K = dims[0]; rb = K // qb * bb; rows = 8
    raw = np.fromfile(fn, dtype=np.uint8, count=rows * rb, offset=d0 + off)
    with tempfile.TemporaryDirectory() as td:
        open(td + '/in', 'wb').write(raw.tobytes())
        r = subprocess.run([os.environ.get('DQ_BIN', 'build/dqtest'), str(t), str(K), str(rows), td + '/in', td + '/out'], capture_output=True, text=True)
        if r.returncode: print(t, '失敗', r.stderr[:300]); ok = False; continue
        gpu = np.fromfile(td + '/out', dtype=np.float32).reshape(rows, K)
    if t == 42:
        print('%-8s %-40s 參考：無（gguf 套件沒有 type 42），GPU 範圍 %.3g…%.3g' % ('T42', name, gpu.min(), gpu.max())); continue
    ref = quants.dequantize(raw, Q(t)).reshape(rows, K).astype(np.float32)
    err = np.abs(gpu - ref).max(); rel = err / (np.abs(ref).max() + 1e-30)
    good = rel < 1e-6
    ok &= good
    print('%-8s %-40s K=%-5d 最大誤差 %.3g（相對 %.2g）%s' % (Q(t).name, name, K, err, rel, '' if good else '  ← 不符'))
print('全部相符' if ok else '有不符')
