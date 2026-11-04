# GGUF hyper-connection 的 numpy 參考（float64），對照 QW_HCTEST 的輸出（decode / 驗證 / prefill）
# 用法：ref/venv/bin/python -I tools/hc_ref.py <QW_HCTEST 的目錄> <GGUF 目錄> [層號（-1 = mixer）]
import glob, os, struct, sys
import numpy as np

D, HC, LR = 2560, 4, 320
dump, ggdir = sys.argv[1], sys.argv[2]
layer = int(sys.argv[3]) if len(sys.argv) > 3 else 5


def rd(f, fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def rstr(f): n, = rd(f, 'Q'); return f.read(n).decode('utf-8', 'replace')
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def skipval(f, t):
    if t == 8: rstr(f); return
    if t == 9:
        at, n = rd(f, 'IQ')
        for _ in range(n): skipval(f, at)
        return
    f.read(SZ[t])


idx = {}
for fn in sorted(glob.glob(ggdir + '/*.gguf')):
    with open(fn, 'rb') as f:
        f.read(4); ver, nt, nkv = rd(f, 'IQQ')
        for _ in range(nkv): rstr(f); t, = rd(f, 'I'); skipval(f, t)
        for _ in range(nt):
            name = rstr(f); nd, = rd(f, 'I'); dims = rd(f, 'Q' * nd); t, = rd(f, 'I'); off, = rd(f, 'Q')
            idx[name] = (fn, off, t, dims)
        d0 = (f.tell() + 31) // 32 * 32
    for k, v in idx.items():
        if v[0] == fn and len(v) == 4: idx[k] = v + (d0,)


def gg(name):
    fn, off, t, dims, d0 = idx[name]
    n = int(np.prod(dims))
    if t == 0: return np.fromfile(fn, np.float32, n, offset=d0 + off).astype(np.float64).reshape(dims[::-1])
    assert t == 30, (name, t)
    raw = np.fromfile(fn, np.uint16, n, offset=d0 + off)
    return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(dims[::-1])


pre = 'output_hc_' if layer < 0 else 'blk.%d.hc_attn_' % layer
w, Wd, Wu = gg(pre + 'norm.weight'), gg(pre + 'down.weight'), gg(pre + 'up.weight')
Wi = None if layer < 0 else gg(pre + 'inject.weight')
assert Wd.shape == (LR, HC * D) and Wu.shape == (HC * D, LR)

T = os.path.getsize(os.path.join(dump, 'in_h')) // (HC * D * 4)
def ld(name, dt, shape): return np.fromfile(os.path.join(dump, name), dt).astype(np.float64).reshape(shape)
h0, out, inj0 = ld('in_h', np.float32, (T, HC, D)), ld('in_out', np.float16, (T, D)), ld('in_inj', np.float32, (T, HC, HC))
sig = lambda v: 1 / (1 + np.exp(-v))
js = 2 * sig(inj0.sum(1) / HC)                                   # [T][r]
h1 = h0 + out[:, None, :] * js[:, :, None]                       # 殘差
r = 1 / np.sqrt((h1 ** 2).mean(2) + 1e-6)                        # [T][HC]
y = (h1 * r[:, :, None]).reshape(T, HC * D) * w                  # 正規化
d = y @ Wd.T
u = (d / HC) * sig(d / HC) @ Wu.T                                # SiLU → mix_up
x = (sig(u) * y).reshape(T, HC, D).mean(1)
injp = None if Wi is None else np.einsum('tsi,qsi->tsq', y.reshape(T, HC, D), Wi.reshape(HC, HC, D))

def rel(a, b): return np.abs(a - b).max() / np.abs(b).max()
ok = True
for tag, n in [('n1', 1), ('n2', 2), ('n3', 3), ('n8', 8), ('pf', T)]:
    p = os.path.join(dump, tag + '_x')
    if not os.path.exists(p): continue
    gx = ld(tag + '_x', np.float16, (n, D)); gh = ld(tag + '_h', np.float32, (n, HC, D))
    ex, eh = rel(gx, x[:n]), rel(gh, h1[:n])
    msg = '%s：x 最大相對誤差 %.2e、h %.2e' % (tag, ex, eh)
    if injp is not None:
        gi = ld(tag + '_inj', np.float32, (n, HC, HC)); ei = rel(gi, injp[:n]); msg += '、inject 部分和 %.2e' % ei
    else: ei = 0
    bad = ex > (4e-3 if tag == 'pf' else 1e-3) or eh > 1e-6 or ei > 1e-4
    ok &= not bad
    print(msg + ('  ← 超出' if bad else ''))
print('通過' if ok else '失敗')
