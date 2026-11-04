# DFlash2 drafter 的 numpy 參考實作（對照 Strata tools/dflash2/dflash2_ref.py，MIT），用來離線驗證 Qwental 的特徵（taps）是否正確：
#   ref/venv/bin/python tools/dflash2_ref.py <drafter .pack> <taps 檔（build/run 的 QW_TAPS）> <prompt 長度>
# 對每個生成位置 s 猜一塊（anchor = 位置 s 的 token），和實際的 greedy 序列比對，印出平均猜中長度。
import json, struct, sys
import numpy as np

MODEL = "models/Qwen3.8-Flash-Next-mixed-4-8bit"


def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


class Pack:
    def __init__(self, path):
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        magic, ver, ilen = struct.unpack("<4sII", raw[:12].tobytes())
        assert magic == b"DF2P", magic
        idx = raw[12:12 + ilen].tobytes().decode()
        self.base = 12 + ilen
        self.cfg, self.t = {}, {}
        for line in idx.splitlines():
            p = line.split()
            if not p: continue
            if p[0] == "cfg": self.cfg[p[1]] = p[2]
            elif p[0] == "t": self.t[p[1]] = (p[2], int(p[3]), int(p[4]), int(p[5]), int(p[6]))
        self.raw = raw

    def get(self, name):
        dt, rows, cols, off, nb = self.t[name]
        b = self.raw[self.base + off:self.base + off + nb]
        if dt == "f32": return np.frombuffer(b, np.float32).reshape(rows, cols).copy()
        if dt == "bf16": return np.frombuffer(b, np.uint16).reshape(rows, cols)
        if dt == "q4_0":
            blk = np.frombuffer(b, np.uint8).reshape(rows, cols // 32, 18)
            d = blk[..., :2].copy().view(np.float16).astype(np.float32)          # [rows, nb, 1]
            q = blk[..., 2:]
            v = np.concatenate([q & 15, q >> 4], axis=-1).astype(np.float32) - 8  # 前 16 個在低 4 位
            return (v * d).reshape(rows, cols)
        if dt == "q8_0":
            blk = np.frombuffer(b, np.uint8).reshape(rows, cols // 34 * 34 // 34 if False else cols // 32, 34)
            d = blk[..., :2].copy().view(np.float16).astype(np.float32)
            return (blk[..., 2:].view(np.int8).astype(np.float32) * d).reshape(rows, cols)
        raise ValueError(dt)


def st_tensors(names):
    """從 MLX 的 safetensors 分片讀出指定 tensor（numpy）"""
    idx = json.load(open(MODEL + "/model.safetensors.index.json"))["weight_map"]
    out = {}
    for n in names:
        fn = MODEL + "/" + idx[n]
        with open(fn, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(hl))
        info = h[n]
        lo, hi = info["data_offsets"]
        dt = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32}[info["dtype"]]
        out[n] = np.memmap(fn, dtype=dt, mode="r", offset=8 + hl + lo, shape=tuple(info["shape"]))
    return out


def mlx_dequant(w, s, b, bits):
    rows = w.shape[0]
    per = 32 // bits
    q = np.empty((rows, w.shape[1] * per), np.float32)
    for i in range(per): q[:, i::per] = (w >> (bits * i)) & ((1 << bits) - 1)
    S, B = bf16(np.asarray(s)), bf16(np.asarray(b))
    g = q.shape[1] // S.shape[1]
    return (q.reshape(rows, -1, g) * S[..., None] + B[..., None]).reshape(rows, -1)


def rms(x, w, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rot_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


class Drafter:
    def __init__(self, pack, embed, head):
        c = pack.cfg
        self.H, self.L, self.NH, self.NKV, self.HD = int(c["hidden"]), int(c["layers"]), int(c["heads"]), int(c["kv_heads"]), int(c["head_dim"])
        self.win, self.eps, self.gs, self.mask = int(c["window"]), float(c["eps"]), int(c["conv_group"]), int(c["mask_token"])
        self.topk, self.block = int(c["top_k"]), int(c["block"])
        theta = float(c["rope_theta"])
        self.inv = 1.0 / theta ** (np.arange(0, self.HD, 2, dtype=np.float32) / self.HD)
        self.w = {n: pack.get(n) for n in pack.t}
        for n in self.w:
            if self.w[n].shape[1] == 1: self.w[n] = self.w[n][:, 0]
        self.embed, self.head = embed, head

    def cs(self, pos):
        f = pos[:, None].astype(np.float32) * self.inv[None]
        e = np.concatenate([f, f], -1)
        return np.cos(e), np.sin(e)

    def set_context(self, feats):
        w = self.w
        h = rms(feats @ w["fc.weight"].T, w["hidden_norm.weight"], self.eps)
        n = h.shape[0]
        cos, sin = self.cs(np.arange(n))
        self.K, self.V = [], []
        for l in range(self.L):
            p = "layers.%d.self_attn." % l
            k = rms((h @ w[p + "k_proj.weight"].T).reshape(n, self.NKV, self.HD), w[p + "k_norm.weight"], self.eps)
            k = k * cos[:, None] + rot_half(k) * sin[:, None]
            self.K.append(k)
            self.V.append((h @ w[p + "v_proj.weight"].T).reshape(n, self.NKV, self.HD))

    def conv(self, x, dyn, base):
        b, H = x.shape
        G = H // self.gs
        blk = x.reshape(b, G, self.gs)
        out = np.zeros_like(blk)
        for o in range(base.shape[0]):
            vals = blk if o == 0 else np.concatenate([np.zeros((o, G, self.gs), np.float32), blk[:-o]], 0)
            out += base[o].reshape(1, G, self.gs) * vals + dyn[:, o, :, None] * vals
        return out.reshape(b, H)

    def draft(self, s, anchor):
        w, b = self.w, self.block
        H, NH, NKV, HD = self.H, self.NH, self.NKV, self.HD
        ids = [anchor] + [self.mask] * (b - 1)
        h = self.embed(ids)
        pos = np.arange(s, s + b)
        cos, sin = self.cs(pos)
        lo = max(0, s - self.win + 1)
        G = H // self.gs
        for l in range(self.L):
            p = "layers.%d." % l
            r = h
            x = rms(h, w[p + "input_layernorm.weight"], self.eps)
            dyn = (x @ w[p + "attention_conv.kernel_projection.weight"].T).reshape(b, 2, 2, G)
            bk = w[p + "attention_conv.base_kernel"].reshape(2, 2, H)
            x = self.conv(x, dyn[:, 0], bk[0])
            q = rms((x @ w[p + "self_attn.q_proj.weight"].T).reshape(b, NH, HD), w[p + "self_attn.q_norm.weight"], self.eps)
            kn = rms((x @ w[p + "self_attn.k_proj.weight"].T).reshape(b, NKV, HD), w[p + "self_attn.k_norm.weight"], self.eps)
            vn = (x @ w[p + "self_attn.v_proj.weight"].T).reshape(b, NKV, HD)
            q = q * cos[:, None] + rot_half(q) * sin[:, None]
            kn = kn * cos[:, None] + rot_half(kn) * sin[:, None]
            K = np.concatenate([self.K[l][lo:s], kn], 0)     # [c+b, NKV, HD]
            V = np.concatenate([self.V[l][lo:s], vn], 0)
            kpos = np.concatenate([np.arange(lo, s), pos])
            vis = np.abs(pos[:, None] - kpos[None]) < self.win
            rep = NH // NKV
            Kx, Vx = np.repeat(K, rep, 1), np.repeat(V, rep, 1)  # [c+b, NH, HD]
            sc = np.einsum("bhd,khd->hbk", q, Kx) / np.sqrt(HD)
            sc = np.where(vis[None], sc, -np.inf)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            att = sc / sc.sum(-1, keepdims=True)
            o = np.einsum("hbk,khd->bhd", att, Vx).reshape(b, NH * HD) @ w[p + "self_attn.o_proj.weight"].T
            o = self.conv(o, dyn[:, 1], bk[1])
            h = r + o
            r = h
            x = rms(h, w[p + "post_attention_layernorm.weight"], self.eps)
            dyn = (x @ w[p + "mlp_conv.kernel_projection.weight"].T).reshape(b, 2, 2, G)
            bk = w[p + "mlp_conv.base_kernel"].reshape(2, 2, H)
            x = self.conv(x, dyn[:, 0], bk[0])
            g = x @ w[p + "mlp.gate_proj.weight"].T
            m = (g / (1 + np.exp(-g)) * (x @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
            m = self.conv(m, dyn[:, 1], bk[1])
            h = r + m
        out = rms(h, w["norm.weight"], self.eps)[1:]
        logits = out @ self.head.T
        cand = np.argpartition(-logits, self.topk, -1)[:, :self.topk]
        unary = np.take_along_axis(logits, cand, -1)
        hp = out @ w["candidate_selector.hidden_projection.weight"].T
        A, B = w["candidate_selector.predecessor_codebook"], w["candidate_selector.successor_codebook"]
        pred, path = anchor, []
        for i in range(b - 1):
            sc = unary[i] + bf16(B[cand[i]]) @ (bf16(A[pred]) * hp[i])
            pred = int(cand[i][int(np.argmax(sc))])
            path.append(pred)
        return path, logits.argmax(-1).tolist()


def main():
    pack = Pack(sys.argv[1])
    raw = np.fromfile(sys.argv[2], np.uint8)
    rec = 8 + 4 * 5 * 2560
    n = raw.size // rec
    r = raw[:n * rec].reshape(n, rec)
    toks = r[:, 4:8].copy().view(np.int32)[:, 0]
    feats = r[:, 8:].copy().view(np.float32).reshape(n, 5 * 2560)
    npr = int(sys.argv[3])
    P = "language_model.model."
    T = st_tensors([P + "embed_tokens.weight", P + "embed_tokens.scales", P + "embed_tokens.biases",
                    "language_model.lm_head.weight", "language_model.lm_head.scales", "language_model.lm_head.biases"])
    ew, es, eb = T[P + "embed_tokens.weight"], T[P + "embed_tokens.scales"], T[P + "embed_tokens.biases"]
    ebits = ew.shape[1] * 32 // 2560
    embed = lambda ids: mlx_dequant(np.asarray(ew[ids]), es[ids], eb[ids], ebits)
    print("lm_head 解量化 …", flush=True)
    head = mlx_dequant(np.asarray(T["language_model.lm_head.weight"]), T["language_model.lm_head.scales"],
                       T["language_model.lm_head.biases"], 8)
    import os
    if os.environ.get("DV"):  # 只看 draft vocab（GPU 版的做法）
        dv = np.fromfile(os.environ["DV"], np.uint32)
        mask = np.full(head.shape[0], True)
        mask[dv] = False
        head[mask] = 0
        head[mask, 0] = -1e4  # 讓不在 vocab 的 logit 極小（hidden 第 0 維通常不為 0，但只是近似）
    if os.environ.get("NBLK"): globals()["NBLK"] = int(os.environ["NBLK"])
    d = Drafter(pack, embed, head)
    d.set_context(feats)
    acc, acc1, cnt = 0, 0, 0
    for s in range(npr, min(n - d.block, npr + globals().get("NBLK", 10**9))):
        path, top1 = d.draft(s, int(toks[s]))
        k = 0
        while k < len(path) and path[k] == toks[s + 1 + k]: k += 1
        k1 = 0
        while k1 < len(top1) and top1[k1] == toks[s + 1 + k1]: k1 += 1
        acc += k; acc1 += k1; cnt += 1
        if cnt % 50 == 0: print(f"  {cnt} 塊：平均猜中 {acc / cnt:.2f}（selector）、{acc1 / cnt:.2f}（各位置 top-1）", flush=True)
    print(f"{cnt} 塊：平均猜中 {acc / cnt:.2f} 個（selector 路徑）、{acc1 / cnt:.2f} 個（各位置 top-1），每塊最多 {d.block - 1}")


if __name__ == "__main__":
    main()
