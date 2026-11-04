# GGUF embed / PLE 的 GPU 結果對 numpy 參考（float64）比對。
# 先跑（只留 embed 與 PLE，其餘部分跳過，殘差流 = embed）：
#   QW_GGUF=<GGUF 目錄> QW_GGUF_PARTS=embed,ple QW_SKIP=hyper,lin,attn,moe QW_PLE_DUMP=<ple.bin> QW_NO_DFLASH=1 \
#     build/run "<ids>" 0 <dump 目錄>
# 再跑：ref/venv/bin/python -I tools/gguf_ple_ref.py <GGUF 目錄> "<ids>" <dump 目錄> <ple.bin>
# 檢查：(1) 第 0 層輸出（= embed）與 token_embd 的 dequantize 逐位元相同
#       (2) n-gram 列號（雜湊）與解出的值與 per_layer_token_embd 的 dequantize 逐位元相同
#       (3) 第 1 層輸出 − 第 0 層輸出（= PLE 的貢獻）與 float64 參考的相對誤差
import sys, os, glob, struct, numpy as np
from gguf import quants, GGMLQuantizationType as Q

GDIR, IDS, DDIR, PLEF = sys.argv[1:5]
ids = [int(x) for x in IDS.split(',')]
EOS, D, HC = 248044, 2560, 4
def rd(f, fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def rstr(f): n, = rd(f, 'Q'); return f.read(n).decode('utf-8', 'replace')
FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
def val(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        at, n = rd(f, 'IQ'); return [val(f, at) for _ in range(n)]
    return rd(f, FMT[t])[0]
BLK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 20: (32, 18), 23: (256, 136)}
kv, idx = {}, {}
for fn in sorted(glob.glob(os.path.join(GDIR, '*.gguf'))):
    with open(fn, 'rb') as f:
        f.read(4); ver, nt, nkv = rd(f, 'IQQ')
        for _ in range(nkv):
            k = rstr(f); t, = rd(f, 'I'); kv[k] = val(f, t)
        ts = []
        for _ in range(nt):
            name = rstr(f); nd, = rd(f, 'I'); dims = rd(f, 'Q' * nd); t, = rd(f, 'I'); off, = rd(f, 'Q'); ts.append((name, dims, t, off))
        d0 = (f.tell() + 31) // 32 * 32
    for name, dims, t, off in ts: idx[name] = (fn, d0 + off, t, dims)
def g_rows(name, r0, n):
    fn, off, t, dims = idx[name]
    qb, bb = BLK[t]; rb = dims[0] // qb * bb
    raw = np.fromfile(fn, dtype=np.uint8, count=n * rb, offset=off + r0 * rb)
    if t == 0: return raw.view(np.float32).reshape(n, dims[0]).copy()
    if t == 1: return raw.view(np.float16).astype(np.float32).reshape(n, dims[0])
    if t == 30: return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(n, dims[0])
    return quants.dequantize(raw, Q(t)).reshape(n, dims[0]).astype(np.float32)
def g_all(name):
    dims = idx[name][3]
    return g_rows(name, 0, int(np.prod(dims[1:])) if len(dims) > 1 else 1).reshape(dims[::-1])

ok = True
def check(c, msg):
    global ok
    ok &= bool(c)
    print(('  ' if c else '✗ ') + msg)

n = len(ids)
L = [np.fromfile('%s/layers_t%d.bin' % (DDIR, t), dtype=np.float32).reshape(48, HC * D) for t in range(n)]
# (1) embed
emb = g_rows('token_embd.weight', 0, 1)  # 只為了取型別
bad = 0
for t, tok in enumerate(ids):
    e = g_rows('token_embd.weight', tok, 1)[0]
    bad += sum(int((L[t][0, s * D:(s + 1) * D] != e).sum()) for s in range(HC))
check(bad == 0, 'embed：%d 個位置 × %d 條流，不同的值 %d 個' % (n, HC, bad))
# (2) n-gram 列
ARCH = kv['general.architecture']  # GGUF metadata 的鍵值前綴（模型架構名稱）
mult = kv[ARCH + '.ple.layer_multipliers']; off = kv[ARCH + '.ple.head_offsets']; size = kv[ARCH + '.ple.head_vocab_sizes']
hist = [EOS, EOS] + ids
raw = open(PLEF, 'rb').read()
rec = 16 * 8 + D * 4
check(len(raw) == n * rec, 'ple dump：%d 筆（預期 %d）' % (len(raw) // rec, n))
M = (1 << 64) - 1
ngv = []
badrow = badval = 0
for t in range(n):
    p = t + 2
    t0, t1, t2 = hist[p], hist[p - 1], hist[p - 2]
    s1 = t1; s2 = t2 if t1 != EOS else EOS
    rows = np.frombuffer(raw, dtype=np.uint64, count=16, offset=t * rec)
    v = np.frombuffer(raw, dtype=np.float32, count=D, offset=t * rec + 128)
    m01 = ((t0 * mult[0]) & M) ^ ((s1 * mult[1]) & M)
    ref = np.zeros(D, np.float32)
    for h in range(16):
        mixed = m01 ^ ((s2 * mult[2]) & M) if h >= 8 else m01
        sm = mixed - (1 << 64) if mixed >= (1 << 63) else mixed  # int64
        r = sm % size[h]  # python 的 % 已是 floor_mod
        row = r + off[h]
        badrow += int(rows[h]) != row
        ref[h * 160:(h + 1) * 160] = g_rows('per_layer_token_embd.weight', row, 1)[0]
    badval += int((v != ref).sum())
    ngv.append(ref.astype(np.float64))
check(badrow == 0 and badval == 0, 'n-gram：%d 個位置 × 16 列，列號不符 %d、值不同 %d 個（值域 ±%.3g）' % (n, badrow, badval, np.abs(np.array(ngv)).max()))
# (3) PLE 區塊（float64 參考）
Wk = g_all('blk.1.ple_key.weight').astype(np.float64); Wv = g_all('blk.1.ple_value.weight').astype(np.float64)
nk = g_all('blk.1.ple_norm_key.weight').astype(np.float64).reshape(HC, D)
nq = g_all('blk.1.ple_norm_query.weight').astype(np.float64).reshape(HC, D)
nc = g_all('blk.1.ple_norm_conv.weight').astype(np.float64).reshape(HC * D)
cw = g_all('blk.1.ple_conv1d.weight').astype(np.float64).reshape(HC * D, 4)
def rms(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
cns = []
worst = 0
for t in range(n):
    h = L[t][0].astype(np.float64).reshape(HC, D)
    key = (Wk @ ngv[t]).reshape(HC, D); value = Wv @ ngv[t]
    g = (rms(key, nk) * rms(h, nq)).sum(-1) / np.sqrt(D)
    g = np.sqrt(np.maximum(np.abs(g), 1e-6)) * np.sign(g)
    gated = (1 / (1 + np.exp(-g)))[:, None] * value[None, :]
    cn = rms(gated, nc.reshape(HC, D)).reshape(-1)
    cns.append(cn)
    past = lambda k: cns[t - k] if t - k >= 0 else np.zeros_like(cn)
    o = cw[:, 0] * past(9) + cw[:, 1] * past(6) + cw[:, 2] * past(3) + cw[:, 3] * cn
    delta = gated.reshape(-1) + o / (1 + np.exp(-o))
    got = L[t][1].astype(np.float64) - L[t][0].astype(np.float64)
    rel = np.linalg.norm(got - delta) / np.linalg.norm(delta)
    worst = max(worst, rel)
check(worst < 1e-5, 'PLE 區塊：%d 個位置，GPU 對 float64 參考的相對誤差最大 %.2e' % (n, worst))
print('全部符合' if ok else '有不符的項目')
sys.exit(0 if ok else 1)
