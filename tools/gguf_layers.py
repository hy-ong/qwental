# IQ3_S GGUF 拆解：各型別 bpw 與逐層格式。用法：ref/venv/bin/python -I tools/gguf_layers.py <分片.gguf>…
import struct, sys, os, collections, re, glob
def rd(f, fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def rstr(f): n, = rd(f, 'Q'); return f.read(n).decode('utf-8', 'replace')
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        at, n = rd(f, 'IQ'); return [val(f, at) for _ in range(n)]
    f.read(SZ[t])
TN = {0:'F32',1:'F16',2:'Q4_0',8:'Q8_0',12:'Q4_K',13:'Q5_K',14:'Q6_K',18:'IQ3_XXS',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',30:'BF16',42:'T42'}
info = {}
for fn in sys.argv[1:]:
    with open(fn, 'rb') as f:
        f.read(4); ver, nt, nkv = rd(f, 'IQQ')
        for _ in range(nkv): rstr(f); t, = rd(f, 'I'); val(f, t)
        ts = []
        for _ in range(nt):
            name = rstr(f); nd, = rd(f, 'I'); dims = rd(f, 'Q' * nd); t, = rd(f, 'I'); off, = rd(f, 'Q'); ts.append([off, name, t, dims])
        d0 = (f.tell() + 31) // 32 * 32
    end = os.path.getsize(fn) - d0; ts.sort()
    for i, (off, name, t, dims) in enumerate(ts):
        n = 1
        for d in dims: n *= d
        info[name] = (TN.get(t, str(t)), ((ts[i + 1][0] if i + 1 < len(ts) else end) - off), n)
# 各型別 bpw
bpw = collections.defaultdict(lambda: [0, 0])
for n, (t, b, c) in info.items(): bpw[t][0] += b; bpw[t][1] += c
print('型別  bpw  總量')
for t, (b, c) in sorted(bpw.items(), key=lambda x: -x[1][0]): print('%-8s %.3f %7.3f GB' % (t, b * 8 / c, b / 1e9))
print()
print('層 類型 | gate/up/down expert | DeltaNet qkv/z/out 或 attn q/k/v/o | shared g/u/d')
for l in range(48):
    g = lambda s: info.get('blk.%d.%s.weight' % (l, s), ('-',))[0]
    ex = '%s/%s/%s' % (g('ffn_gate_exps'), g('ffn_up_exps'), g('ffn_down_exps'))
    if 'blk.%d.attn_qkv.weight' % l in info: dn = 'DN  %s/%s/%s' % (g('attn_qkv'), g('attn_gate'), g('ssm_out'))
    else: dn = 'ATT %s/%s/%s/%s' % (g('attn_q'), g('attn_k'), g('attn_v'), g('attn_output'))
    sh = '%s/%s/%s' % (g('ffn_gate_shexp'), g('ffn_up_shexp'), g('ffn_down_shexp'))
    eb = sum(info['blk.%d.ffn_%s_exps.weight' % (l, k)][1] for k in ('gate', 'up', 'down')) / 512 * 10
    print('%2d %-26s %-34s %-22s 每 token expert %.1f MB' % (l, ex, dn, sh, eb / 1e6))
