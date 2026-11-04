# DeltaNet 層的 numpy 參考實作（GGUF 權重、V head 交錯順序），比對 engine 的 QW_DN_TAP 記錄
# 用法：python -I tools/dn_ref.py <tap 檔> <li（第幾個 DeltaNet 層，0…35）> [GGUF 目錄]
# tap 檔：每筆 [int32 n][n 列 x（2560 half）][n 列 out（2560 half）]，依時間順序；參考實作從零狀態逐 token 算
# 交錯順序：v head h = r·16 + k_head → 對應的 key head 是 h % 16（MLX 是 h / 3）
import struct, sys, os, glob, numpy as np
from gguf import quants, GGMLQuantizationType as Q

tap, li = sys.argv[1], int(sys.argv[2])
gdir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'iq3s-gguf')
GG = sorted(glob.glob(os.path.join(gdir, '*.gguf')))
def rd(f, fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def rstr(f): n, = rd(f, 'Q'); return f.read(n).decode('utf-8', 'replace')
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        at, n = rd(f, 'IQ'); return [val(f, at) for _ in range(n)]
    f.read(SZ[t])
idx = {}
for fn in GG:
    with open(fn, 'rb') as f:
        f.read(4); ver, nt, nkv = rd(f, 'IQQ')
        for _ in range(nkv): rstr(f); t, = rd(f, 'I'); val(f, t)
        ts = []
        for _ in range(nt):
            name = rstr(f); nd, = rd(f, 'I'); dims = rd(f, 'Q' * nd); t, = rd(f, 'I'); off, = rd(f, 'Q'); ts.append([off, name, t, dims])
        d0 = (f.tell() + 31) // 32 * 32
    end = os.path.getsize(fn) - d0; ts.sort()
    for i, (off, name, t, dims) in enumerate(ts):
        idx[name] = (fn, d0 + off, (ts[i + 1][0] if i + 1 < len(ts) else end) - off, t, dims)
def gg(name):
    fn, off, nb, t, dims = idx[name]
    raw = np.fromfile(fn, dtype=np.uint8, count=nb, offset=off)
    if t == 30: return raw.view(np.uint16).astype(np.uint32).__lshift__(16).view(np.float32).reshape(dims[::-1])
    if t == 0: return raw.view(np.float32).reshape(dims[::-1])
    if t == 1: return raw.view(np.float16).astype(np.float32).reshape(dims[::-1])
    return quants.dequantize(raw, Q(t)).reshape(dims[::-1]).astype(np.float32)

l = li + li // 3 # DeltaNet 層 → 模型層（每 4 層第 4 個是 attention）
p = 'blk.%d.' % l
Wqkv, Wz, Wa, Wb, Wo = (gg(p + n) for n in ('attn_qkv.weight', 'attn_gate.weight', 'ssm_alpha.weight', 'ssm_beta.weight', 'ssm_out.weight'))
conv = gg(p + 'ssm_conv1d.weight').reshape(10240, 4)
A = gg(p + 'ssm_a').reshape(48); dt = gg(p + 'ssm_dt.bias').reshape(48); nw = gg(p + 'ssm_norm.weight').reshape(128)
print('layer %d：qkv %s z %s out %s a %s' % (l, idx[p + 'attn_qkv.weight'][3], idx[p + 'attn_gate.weight'][3], idx[p + 'ssm_out.weight'][3],
                                             idx[p + 'ssm_alpha.weight'][3]))

X, O = [], []
with open(tap, 'rb') as f:
    while True:
        h = f.read(4)
        if len(h) < 4: break
        n, = struct.unpack('<i', h)
        X.append(np.frombuffer(f.read(n * 5120), np.float16).reshape(n, 2560))
        O.append(np.frombuffer(f.read(n * 5120), np.float16).reshape(n, 2560))
X = np.concatenate(X).astype(np.float32); O = np.concatenate(O).astype(np.float32)
T = X.shape[0]
h16 = lambda a: a.astype(np.float16).astype(np.float32)
sig = lambda a: 1 / (1 + np.exp(-a))
qkv = h16(X @ Wqkv.T); z = h16(X @ Wz.T); a = h16(X @ Wa.T); b = h16(X @ Wb.T)
pad = np.concatenate([np.zeros((3, 10240), np.float32), qkv])
cv = sum(conv[:, j] * pad[j:j + T] for j in range(4))
cv = h16(cv * sig(cv))
q = cv[:, :2048].reshape(T, 16, 128); k = cv[:, 2048:4096].reshape(T, 16, 128); v = cv[:, 4096:].reshape(T, 48, 128)
q = h16(q / np.sqrt((q * q).mean(-1, keepdims=True) + 1e-6) / 128)
k = h16(k / np.sqrt((k * k).mean(-1, keepdims=True) + 1e-6) / np.sqrt(128))
xa = a + dt
la = A * np.where(xa > 20, xa, np.log1p(np.exp(np.minimum(xa, 20))))
be = sig(b)
kh = np.arange(48) // 3 if os.environ.get('DN_REF_GROUPED') else np.arange(48) % 16 # 交錯順序（DN_REF_GROUPED=1：故意用錯的分組順序，確認測試有分辨力）
S = np.zeros((48, 128, 128), np.float64)
o = np.zeros((T, 48, 128), np.float32)
for t in range(T):
    kk = k[t, kh].astype(np.float64); qq = q[t, kh].astype(np.float64)
    S *= np.exp(la[t])[:, None, None]
    mem = np.einsum('hkd,hk->hd', S, kk)
    d = be[t][:, None] * (v[t] - mem)
    S += kk[:, :, None] * d[:, None, :]
    o[t] = h16(np.einsum('hkd,hk->hd', S, qq))
r = 1 / np.sqrt((o * o).mean(-1, keepdims=True) + 1e-6)
y = h16(o * r * nw * sig(z.reshape(T, 48, 128))).reshape(T, 6144)
ref = h16(y @ Wo.T)
err = np.linalg.norm(ref - O, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)
cos = (ref * O).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(O, axis=1) + 1e-12)
print('T = %d：相對誤差 平均 %.2e、最大 %.2e（第 %d 個）；cos 最小 %.6f；|out| 平均 %.3f' % (T, err.mean(), err.max(), err.argmax(), cos.min(),
                                                                         np.linalg.norm(ref, axis=1).mean()))
