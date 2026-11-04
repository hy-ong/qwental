# GGUF 與 MLX 同名權重逐列比對（找列的排列，例如 DeltaNet V head 的重排）。需要 gguf 套件
# GGUF（IQ3_S）與 MLX 同名權重逐列比對：找出列的排列（V head 是否被重排）
import struct, sys, os, json, numpy as np
from gguf import quants, GGMLQuantizationType as Q
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
GG = sorted(__import__('glob').glob(os.path.join(os.environ.get('GGUF_DIR', os.path.join(_ROOT, 'models/iq3s-gguf')), '*.gguf')))
MLX = os.environ.get('MLX_DIR', os.path.join(_ROOT, 'models/Qwen3.8-Flash-Next-mixed-4-8bit'))
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
wm = json.load(open(MLX + '/model.safetensors.index.json'))['weight_map']
hdrs = {}
def raw_st(name):
    fn = MLX + '/' + wm[name]
    if fn not in hdrs:
        with open(fn, 'rb') as f: n, = struct.unpack('<Q', f.read(8)); hdrs[fn] = (json.loads(f.read(n)), 8 + n)
    h, base = hdrs[fn]; i = h[name]; lo, hi = i['data_offsets']
    a = np.fromfile(fn, dtype=np.uint8, count=hi - lo, offset=base + lo)
    return a, i['dtype'], i['shape']
def mlx(name):
    a, dt, sh = raw_st(name + '.weight')
    if dt == 'BF16': return a.view(np.uint16).astype(np.uint32).__lshift__(16).view(np.float32).reshape(sh)
    if dt == 'F32': return a.view(np.float32).reshape(sh)
    w = a.view(np.uint32).reshape(sh)
    s = raw_st(name + '.scales')[0].view(np.uint16).astype(np.uint32).__lshift__(16).view(np.float32).reshape(sh[0], -1)
    b = raw_st(name + '.biases')[0].view(np.uint16).astype(np.uint32).__lshift__(16).view(np.float32).reshape(sh[0], -1)
    K = s.shape[1] * 64; bits = sh[1] * 32 // K; per = 32 // bits
    q = ((w[:, :, None] >> (bits * np.arange(per, dtype=np.uint32))) & ((1 << bits) - 1)).reshape(sh[0], K).astype(np.float32)
    return q * np.repeat(s, 64, 1) + np.repeat(b, 64, 1)
def cmp(gname, mname, tr=False):
    A = gg(gname); B = mlx(mname)
    if tr: B = B.T
    if A.shape != B.shape: A = A.reshape(B.shape) if A.size == B.size else A
    print('%-34s gguf %s mlx %s' % (gname, A.shape, B.shape), end='  ')
    if A.ndim == 1: print('相關 %.4f' % np.corrcoef(A, B)[0, 1]); return
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12); Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    d = np.einsum('ij,ij->i', An, Bn)
    bad = np.where(d < 0.9)[0]
    print('同列相關 最小 %.4f 中位 %.4f  不符 %d 列' % (d.min(), np.median(d), len(bad)), end='')
    if len(bad):
        S = An[bad[:64]] @ Bn.T; print('  例：gguf 列 %s → mlx 列 %s' % (bad[:8].tolist(), S.argmax(1)[:8].tolist()), end='')
    print()
P = 'language_model.model.layers.%d.'
for g, m in [('blk.0.attn_qkv.weight', P % 0 + 'linear_attn.in_proj_qkv'), ('blk.0.attn_gate.weight', P % 0 + 'linear_attn.in_proj_z'),
             ('blk.0.ssm_alpha.weight', P % 0 + 'linear_attn.in_proj_a'), ('blk.0.ssm_beta.weight', P % 0 + 'linear_attn.in_proj_b'),
             ('blk.0.ssm_out.weight', P % 0 + 'linear_attn.out_proj'), ('blk.0.ssm_a', P % 0 + 'linear_attn.A_log'),
             ('blk.0.ssm_dt.bias', P % 0 + 'linear_attn.dt_bias'), ('blk.0.ssm_norm.weight', P % 0 + 'linear_attn.norm'),
             ('blk.3.attn_q.weight', P % 3 + 'self_attn.q_proj'), ('blk.3.attn_k.weight', P % 3 + 'self_attn.k_proj'),
             ('blk.3.attn_v.weight', P % 3 + 'self_attn.v_proj'), ('blk.3.attn_output.weight', P % 3 + 'self_attn.o_proj'),
             ('blk.0.ffn_gate_inp.weight', P % 0 + 'mlp.gate'), ('blk.0.ffn_gate_shexp.weight', P % 0 + 'mlp.shared_expert.gate_proj'),
             ('blk.0.ffn_down_shexp.weight', P % 0 + 'mlp.shared_expert.down_proj'),
             ('blk.0.hc_attn_down.weight', P % 0 + 'attn_hyper_connection.input_mix_weight_down'),
             ('blk.0.hc_attn_up.weight', P % 0 + 'attn_hyper_connection.input_mix_weight_up'),
             ('output.weight', 'language_model.lm_head')]:
    try: cmp(g, m)
    except Exception as ex: print(g, '錯誤', type(ex).__name__, str(ex)[:120])
