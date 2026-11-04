# GGUF（IQ3_S）的 embed / PLE 與 MLX 版逐項比對（只用 CPU，不跑 GPU）：
#   PLE 的雜湊參數（GGUF KV <架構>.ple.* 對 MLX 的 tensor）、第 1 層 ple_* 小 tensor 的格式與數值、
#   n-gram 表抽樣列（GGUF IQ4_NL 對 MLX 4-bit group 32）、token_embd 抽樣列（IQ4_XS 對 MLX 4-bit group 64）
# 用法：ref/venv/bin/python -I tools/gguf_ple_check.py <GGUF 目錄> <MLX 模型目錄>
import sys, os, glob, struct, json, numpy as np
from gguf import quants, GGMLQuantizationType as Q

GDIR, MDIR = sys.argv[1], sys.argv[2]
def rd(f, fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def rstr(f): n, = rd(f, 'Q'); return f.read(n).decode('utf-8', 'replace')
FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
def val(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        at, n = rd(f, 'IQ'); return [val(f, at) for _ in range(n)]
    return rd(f, FMT[t])[0]
BLK = {0: (1, 4), 30: (1, 2), 1: (1, 2), 14: (256, 210), 20: (32, 18), 23: (256, 136)}
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

def g_rows(name, r0, n):  # GGUF tensor 的第 r0 .. r0+n−1 列（float32）
    fn, off, t, dims = idx[name]
    qb, bb = BLK[t]; rb = dims[0] // qb * bb
    raw = np.fromfile(fn, dtype=np.uint8, count=n * rb, offset=off + r0 * rb)
    if t == 0: return raw.view(np.float32).reshape(n, dims[0])
    if t == 30: return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(n, dims[0])
    return quants.dequantize(raw, Q(t)).reshape(n, dims[0]).astype(np.float32)
def g_all(name):
    fn, off, t, dims = idx[name]
    rows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
    return g_rows(name, 0, rows).reshape(dims[::-1]), t

wm = json.load(open(MDIR + '/model.safetensors.index.json'))['weight_map']
hdrs = {}
def st_hdr(fn):
    if fn not in hdrs:
        with open(fn, 'rb') as f: n, = struct.unpack('<Q', f.read(8)); hdrs[fn] = (json.loads(f.read(n)), 8 + n)
    return hdrs[fn]
def st_raw(fn, name, r0=None, n=None):
    h, base = st_hdr(fn); i = h[name]; lo, hi = i['data_offsets']; sh = i['shape']
    isz = {'BF16': 2, 'F32': 4, 'U32': 4, 'I64': 8}[i['dtype']]
    if r0 is None: return np.fromfile(fn, dtype=np.uint8, count=hi - lo, offset=base + lo), i['dtype'], sh
    rb = int(np.prod(sh[1:])) * isz
    return np.fromfile(fn, dtype=np.uint8, count=n * rb, offset=base + lo + r0 * rb), i['dtype'], [n] + sh[1:]
def bf(a): return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
def m_tensor(name):
    a, dt, sh = st_raw(MDIR + '/' + wm[name], name)
    if dt == 'BF16': return bf(a).reshape(sh)
    if dt == 'F32': return a.view(np.float32).reshape(sh)
    if dt == 'I64': return a.view(np.int64).reshape(sh)
    raise ValueError(dt)
def m_affine(fn, base, r0, n, gs, names=('.weight', '.scales', '.biases')):
    w, _, sh = st_raw(fn, base + names[0], r0, n); s, _, ssh = st_raw(fn, base + names[1], r0, n); b, _, _ = st_raw(fn, base + names[2], r0, n)
    w = w.view(np.uint32).reshape(n, -1); s = bf(s).reshape(n, -1); b = bf(b).reshape(n, -1)
    K = s.shape[1] * gs; bits = w.shape[1] * 32 // K; per = 32 // bits
    q = ((w[:, :, None] >> (bits * np.arange(per, dtype=np.uint32))) & ((1 << bits) - 1)).reshape(n, K).astype(np.float32)
    return q * np.repeat(s, gs, 1) + np.repeat(b, gs, 1)

ok = True
def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(('  ' if cond else '✗ ') + msg)

print('== PLE 雜湊參數（GGUF KV 對 MLX tensor）')
P = 'language_model.model.layers.1.ple.'
ARCH = kv['general.architecture']  # GGUF metadata 的鍵值前綴（模型架構名稱）
for k, m in [('layer_multipliers', 'ple_embedding.layer_multipliers'), ('head_offsets', 'ple_embedding.ngram_heads_offsets'),
             ('head_vocab_sizes', 'ple_embedding.ngram_heads_vocab_sizes')]:
    a = np.array(kv[ARCH + '.ple.' + k], dtype=np.int64); b = m_tensor(P + m).reshape(-1)
    check(a.shape == b.shape and (a == b).all(), '%-18s GGUF %s… MLX %s…' % (k, a[:3].tolist(), b[:3].tolist()))
check(kv[ARCH + '.ple.eos_token_id'] == 248044, 'eos_token_id %d（engine 的 EOS_ID 248044）' % kv[ARCH + '.ple.eos_token_id'])
check(kv[ARCH + '.ple.layers'] == [1] and kv[ARCH + '.ple.ngram_size'] == 3 and kv[ARCH + '.ple.heads_per_ngram'] == 8 and
      kv[ARCH + '.embedding_length_per_layer_input'] == 160 and kv[ARCH + '.ple.conv_kernel'] == 4,
      'layers %s、ngram %d、heads %d、每列 %d、conv %d' % (kv[ARCH + '.ple.layers'], kv[ARCH + '.ple.ngram_size'],
                                                 kv[ARCH + '.ple.heads_per_ngram'], kv[ARCH + '.embedding_length_per_layer_input'],
                                                 kv[ARCH + '.ple.conv_kernel']))
print('   （GGUF 另有 <架構>.ple.image_token_id = %s：只跟影像輸入有關）' % kv.get(ARCH + '.ple.image_token_id'))

print('== 第 1 層 ple_* 小 tensor')
for g, m in [('ple_norm_key.weight', 'norm_key.weight'), ('ple_norm_query.weight', 'norm_query.weight'),
             ('ple_norm_conv.weight', 'norm_conv.weight'), ('ple_conv1d.weight', 'conv1d.weight')]:
    A, t = g_all('blk.1.' + g); B = m_tensor(P + m)
    A = A.reshape(-1); B = B.reshape(-1)
    bfexact = (A.view(np.uint32) & 0xffff == 0).all()
    rnd = bf(((A.view(np.uint32) + 0x7fff + ((A.view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16))
    print('  %-22s %s %s 恰為 BF16：%s；MLX 相對差 %.2g；MLX == round_bf16(GGUF)：%s' % (
        g, {0: 'F32', 30: 'BF16'}.get(t, t), idx['blk.1.' + g][3], bfexact, np.abs(A - B).max() / np.abs(B).max(), (rnd == B).all()))
for g, m in [('ple_key.weight', 'key_proj'), ('ple_value.weight', 'value_proj')]:
    A, t = g_all('blk.1.' + g)
    B = m_affine(MDIR + '/' + wm[P + m + '.weight'], P + m, 0, A.shape[0], 64)
    An = A / np.linalg.norm(A, axis=1, keepdims=True); Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    print('  %-22s %s %s 同列相關最小 %.5f' % (g, {0: 'F32', 30: 'BF16'}.get(t, t), A.shape, np.einsum('ij,ij->i', An, Bn).min()))

print('== n-gram 表（GGUF per_layer_token_embd 對 MLX ngram_table.bin）')
fn, off, t, dims = idx['per_layer_token_embd.weight']
check(t == 20 and dims == (160, 320001536), '格式 %s、dims %s' % (Q(t).name, dims))
rng = np.random.default_rng(0)
rows = np.concatenate([[0, 1, 20000003, 320001445], rng.integers(0, 320001446, 60)])
cs, rel = [], []
for r in rows:
    A = g_rows('per_layer_token_embd.weight', int(r), 1)[0]
    B = m_affine(MDIR + '/ngram_table.bin', '', int(r), 1, 32, ('weight', 'scales', 'biases'))[0]
    cs.append(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-30)); rel.append(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-30))
check(min(cs) > 0.97, '%d 列：相關 最小 %.4f 中位 %.4f；相對誤差 中位 %.3f 最大 %.3f；值域 ±%.3g' % (len(rows), min(cs), np.median(cs),
                                                                         np.median(rel), max(rel), np.abs(A).max()))
print('== token_embd（GGUF 對 MLX embed_tokens）')
fn, off, t, dims = idx['token_embd.weight']
check(t == 23 and dims == (2560, 248320), '格式 %s、dims %s' % (Q(t).name, dims))
cs = []
for r in np.concatenate([[0, 248044, 248319], rng.integers(0, 248320, 40)]):
    A = g_rows('token_embd.weight', int(r), 1)[0]
    B = m_affine(MDIR + '/' + wm['language_model.model.embed_tokens.weight'], 'language_model.model.embed_tokens', int(r), 1, 64)[0]
    cs.append(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-30))
check(min(cs) > 0.97, '%d 列：相關 最小 %.4f 中位 %.4f' % (len(cs), min(cs), np.median(cs)))
print('全部符合' if ok else '有不符的項目')
sys.exit(0 if ok else 1)
