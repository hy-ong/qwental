# kernels/qmv_ggm.metal 對照 numpy 參考值（gguf 套件從 GGUF 區塊解碼，float64 內積）
# 用法：python tools/ggmtest.py <GGUF 目錄> <tensor 名稱> <列數> <NB> [GPU 指令前綴…]
#   例：ref/venv/bin/python -I tools/ggmtest.py models/iq3s-gguf blk.3.attn_q.weight 512 3 lockf build/gpu.lock build/tmo 60
import glob, os, struct, subprocess, sys, tempfile
import numpy as np
from gguf import quants, GGMLQuantizationType as Q

BLK = {12: (256, 144), 13: (256, 176), 14: (256, 210), 23: (256, 136), 30: (1, 2)}

def rd(f, fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def rstr(f): n, = rd(f, 'Q'); return f.read(n).decode('utf-8', 'replace')
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        at, n = rd(f, 'IQ'); return [val(f, at) for _ in range(n)]
    f.read(SZ[t])

def index(d):
    idx = {}
    for fn in sorted(glob.glob(d + '/*.gguf')):
        with open(fn, 'rb') as f:
            f.read(4); _, nt, nkv = rd(f, 'IQQ')
            for _ in range(nkv): rstr(f); t, = rd(f, 'I'); val(f, t)
            ts = []
            for _ in range(nt):
                name = rstr(f); nd, = rd(f, 'I'); dims = rd(f, 'Q' * nd); t, = rd(f, 'I'); off, = rd(f, 'Q'); ts.append((name, dims, t, off))
            d0 = (f.tell() + 31) // 32 * 32
        for name, dims, t, off in ts: idx[name] = (fn, d0 + off, t, dims)
    return idx

def main():
    gdir, name, rows, nb = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    pre = sys.argv[5:]
    fn, off, t, dims = index(gdir)[name]
    K = dims[0]
    qb, bb = BLK[t]
    rb = K // qb * bb
    raw = np.fromfile(fn, dtype=np.uint8, count=rows * rb, offset=off)
    if t == 30: W = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(rows, K)
    else: W = quants.dequantize(raw, Q(t)).reshape(rows, K)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((nb, K)).astype(np.float16)
    ref = x.astype(np.float64) @ W.astype(np.float64).T
    with tempfile.TemporaryDirectory() as td:
        pw, px, py = td + '/w.bin', td + '/x.bin', td + '/y.bin'
        raw.tofile(pw); x.tofile(px)
        r = subprocess.run(pre + ['build/ggmtest', str(t), str(K), str(rows), str(nb), pw, px, py, os.environ.get('REPS', '200')], capture_output=True, text=True)
        if r.returncode: print(r.stderr); sys.exit(1)
        y = np.fromfile(py, dtype=np.float32).reshape(nb, rows)
    err = np.abs(y - ref)
    print(f'{name} type {t} K {K} 列 {rows} NB {nb}：最大誤差 / 最大值 {err.max() / np.abs(ref).max():.2e}，'
          f'逐元素相對誤差中位數 {np.median(err / (np.abs(ref) + 1e-6)):.2e}；{r.stdout.strip()}')

main()
