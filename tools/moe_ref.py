# MoE 單層的 FP64 參考（GGUF 區塊用 gguf.quants 精確解碼），對照 engine 的 QW_MOE_TEST 輸出（build/run … 搭配 QW_KBENCH=1 QW_MOE_TEST=層）。
# 用法：ref/venv/bin/python -I tools/moe_ref.py <GGUF 目錄> <層> [測試目錄 build/moe_test]
# 假設 moe、shared、router 都用 GGUF（QW_GGUF_PARTS=moe,shared,router）
import sys, os, glob, struct, numpy as np
from gguf import quants, GGMLQuantizationType as Q

def rd(f, fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def rstr(f): n, = rd(f, 'Q'); return f.read(n).decode('utf-8', 'replace')
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        at, n = rd(f, 'IQ'); return [val(f, at) for _ in range(n)]
    f.read(SZ[t])
BLK = {8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82),
       23: (256, 136), 42: (64, 18), 30: (1, 2), 0: (1, 4)}

gdir, L = sys.argv[1], int(sys.argv[2])
tdir = sys.argv[3] if len(sys.argv) > 3 else 'build/moe_test'
T = {}
for fn in sorted(glob.glob(os.path.join(gdir, '*.gguf'))):
    with open(fn, 'rb') as f:
        f.read(4); ver, nt, nkv = rd(f, 'IQQ')
        for _ in range(nkv): rstr(f); t, = rd(f, 'I'); val(f, t)
        lst = []
        for _ in range(nt):
            name = rstr(f); nd, = rd(f, 'I'); dims = rd(f, 'Q' * nd); t, = rd(f, 'I'); off, = rd(f, 'Q')
            lst.append((name, dims, t, off))
        d0 = (f.tell() + 31) // 32 * 32
    for name, dims, t, off in lst: T[name] = (fn, dims, t, d0 + off)

def rows(name, r0, n, inner=None):
    """name 的第 r0…r0+n−1 列（2D 的列；3D 時先選 expert：inner = expert）→ float64 [n, K]"""
    fn, dims, t, off = T[name]
    qb, bb = BLK[t]; K = dims[0]; rb = K // qb * bb
    if inner is not None: off += inner * dims[1] * rb
    raw = np.fromfile(fn, dtype=np.uint8, count=n * rb, offset=off + r0 * rb).reshape(n, rb)
    if t == 30: return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if t == 0: return raw.view(np.float32).astype(np.float64)
    if t == 42: # Q2_0：64 值 / 18 B：d（f16）、qs[16]，第 v 個值 = d·(((qs[v/4] >> 2(v%4)) & 3) − 1)
        b = raw.reshape(n, -1, 18)
        d = b[:, :, :2].copy().view(np.float16).astype(np.float64)
        q = b[:, :, 2:]
        v = np.stack([(q >> (2 * k)) & 3 for k in range(4)], -1).reshape(n, -1, 64).astype(np.float64) - 1
        return (v * d).reshape(n, K)
    return quants.dequantize(raw, Q(t)).reshape(n, K).astype(np.float64)

def fmt(name): return T[name][2]

x = np.fromfile(tdir + '/x.bin', dtype=np.float16).astype(np.float64).reshape(-1, 2560)
NB = x.shape[0]
p = 'blk.%d.' % L
Wr = rows(p + 'ffn_gate_inp.weight', 0, 512)
wsg = rows(p + 'ffn_gate_inp_shexp.weight', 0, 1)[0]
print('層 %d：expert gate/up %s、down %s；shared %s / %s / %s；router %s' % (L, Q(fmt(p + 'ffn_gate_exps.weight')).name if fmt(p + 'ffn_gate_exps.weight') != 42 else 'T42',
      Q(fmt(p + 'ffn_down_exps.weight')).name if fmt(p + 'ffn_down_exps.weight') != 42 else 'T42',
      Q(fmt(p + 'ffn_gate_shexp.weight')).name, Q(fmt(p + 'ffn_up_shexp.weight')).name, Q(fmt(p + 'ffn_down_shexp.weight')).name,
      Q(fmt(p + 'ffn_gate_inp.weight')).name))
SG = rows(p + 'ffn_gate_shexp.weight', 0, 640); SU = rows(p + 'ffn_up_shexp.weight', 0, 640); SD = rows(p + 'ffn_down_shexp.weight', 0, 2560)
silu = lambda v: v / (1 + np.exp(-v))
vblog = np.fromfile(tdir + '/vblog.bin', dtype=np.float32).reshape(NB, 513)
declog = np.fromfile(tdir + '/declog.bin', dtype=np.float32)
outs = {'dec': np.fromfile(tdir + '/dec.bin', dtype=np.float16).astype(np.float64).reshape(1, 2560),
        'vb': np.fromfile(tdir + '/vb.bin', dtype=np.float16).astype(np.float64).reshape(NB, 2560)}
if os.path.exists(tdir + '/pf.bin'): outs['pf'] = np.fromfile(tdir + '/pf.bin', dtype=np.float16).astype(np.float64).reshape(NB, 2560)
cache = {}
def expert(e):
    if e not in cache:
        cache[e] = (rows(p + 'ffn_gate_exps.weight', 0, 640, e), rows(p + 'ffn_up_exps.weight', 0, 640, e), rows(p + 'ffn_down_exps.weight', 0, 2560, e))
    return cache[e]
ref = np.zeros((NB, 2560))
for t in range(NB):
    lg = Wr @ x[t]
    lerr = np.abs(vblog[t, :512] - lg).max() / np.abs(lg).max()
    ids = np.argsort(-lg, kind='stable')[:10]
    gids = np.argsort(-vblog[t, :512].astype(np.float64), kind='stable')[:10]
    w = np.exp(lg[ids] - lg[ids].max()); w /= w.sum()
    for e, we in zip(ids, w):
        G, U, Dn = expert(e)
        ref[t] += we * (Dn @ (silu(G @ x[t]) * (U @ x[t])))
    sgl = wsg @ x[t]
    ref[t] += 1 / (1 + np.exp(-sgl)) * (SD @ (silu(SG @ x[t]) * (SU @ x[t])))
    gap = np.sort(lg)[::-1]
    print('token %d：router logit 相對誤差 %.2g（decode %.2g），shared gate 誤差 %.2g；選取%s（第 10/11 名差 %.3g）' % (
        t, lerr, np.abs(declog[:512] - Wr @ x[0]).max() / np.abs(Wr @ x[0]).max() if t == 0 else 0, abs(vblog[t, 512] - sgl),
        '相同' if set(ids) == set(gids) else '不同！', gap[9] - gap[10]))
for k, o in outs.items():
    r = ref[:o.shape[0]]
    err = np.abs(o - r).max() / np.abs(r).max()
    rms = np.sqrt(((o - r) ** 2).mean() / (r ** 2).mean())
    print('%-4s 最大相對誤差 %.3g、RMS 相對誤差 %.3g' % (k, err, rms))
if NB >= 1:
    print('decode 與驗證 token 0 逐位元相同：%s' % np.array_equal(outs['dec'][0], outs['vb'][0]))
