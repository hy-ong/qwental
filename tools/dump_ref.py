# 用 mlx-lm（PR #1788 的模型實作）產生參考答案：每一層的輸出、最後的 logits，以及 greedy 生成的文字。
#
# 這份 checkpoint（ddalcu mixed-4-8bit）有兩點 mlx-lm 不會自動處理：
#   1. config 只寫了 4-bit，但很多層其實是 8-bit → 從權重形狀反推每層的位元數
#   2. n-gram 表放在 ngram_table.bin（safetensors 格式，一個 [320001536, 160] 的 4-bit、group 32 tensor）
#      → 用 numpy memmap 只讀需要的列、在 CPU 解量化（跟 mlx-serve 一樣，不把 32 GB 讀進記憶體）
#
# 用法：ref/venv/bin/python tools/dump_ref.py [模型目錄] [輸出目錄] [prompt]
import glob
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import importlib, json as _json, pathlib as _pl
Q = importlib.import_module("mlx_lm.models." + _json.load(open(_pl.Path(__file__).resolve().parent.parent / "models/Qwen3.8-Flash-Next-mixed-4-8bit/config.json"))["model_type"])  # 模型實作依 config.json 的 model_type

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/Qwen3.8-Flash-Next-mixed-4-8bit"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "build/ref"
PROMPT = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"
FP32 = bool(int(__import__("os").environ.get("REF_FP32", "0")))  # 1：整個前向用 FP32 啟動值（比對精度用）
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- n-gram 表（memmap）
class NgramTable:
    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        def view(name, dtype, shape):
            a, b = hdr[name]["data_offsets"]
            return np.memmap(path, dtype=dtype, mode="r", offset=base + a, shape=tuple(shape))
        w = hdr["weight"]["shape"]
        self.w = view("weight", np.uint32, w)                       # [R, 20]
        self.s = view("scales", np.uint16, hdr["scales"]["shape"])  # [R, 5] bf16
        self.b = view("biases", np.uint16, hdr["biases"]["shape"])
        self.rows, self.words = w
        self.dim = self.words * 8                                    # 160
        self.gs = self.dim // self.s.shape[1]                        # 32

    @staticmethod
    def _bf16(u16):
        return (u16.astype(np.uint32) << 16).view(np.float32)

    def lookup(self, idx):
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        w = np.asarray(self.w[idx])                                  # [n, 20]
        q = ((w[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(len(idx), self.dim).astype(np.float32)
        s = np.repeat(self._bf16(np.asarray(self.s[idx])), self.gs, axis=1)
        b = np.repeat(self._bf16(np.asarray(self.b[idx])), self.gs, axis=1)
        return q * s + b


TABLE = NgramTable(MODEL / "ngram_table.bin")


def _sharded_call(self, gid):
    flat = np.array(gid.reshape(-1).astype(mx.int64))
    rows = TABLE.lookup(flat)
    return mx.array(rows).reshape(*gid.shape, self.dim)


Q._ShardedEmbedding.__call__ = _sharded_call


# ---------------------------------------------------------------- 載入模型
config = json.loads((MODEL / "config.json").read_text())
args = Q.ModelArgs.from_dict(config)
model = Q.Model(args)
# n-gram 表的 128 片佔位參數（FP32 約 205 GB）用不到：查表已改成直接讀 ngram_table.bin
for layer in model.model.layers:
    if layer.ple is not None:
        layer.ple.ple_embedding.ngram_embedding.shards = []

weights = {}
for f in sorted(glob.glob(str(MODEL / "model-*.safetensors"))):
    weights.update(mx.load(f))
weights = model.sanitize(weights)

# 每個量化層的位元數：weight 的最後一維 × 32 ÷ 輸入維度（= scales 最後一維 × 64）
bits = {}
for k, v in weights.items():
    if k.endswith(".scales"):
        p = k[: -len(".scales")]
        in_dim = v.shape[-1] * 64
        bits[p] = weights[p + ".weight"].shape[-1] * 32 // in_dim


def pred(p, m):
    if p in bits and hasattr(m, "to_quantized"):
        return {"group_size": 64, "bits": bits[p], "mode": "affine"}
    return False


nn.quantize(model, class_predicate=pred)
missing = set(k for k, _ in nn.utils.tree_flatten(model.parameters())) - set(weights)
missing = sorted(k for k in missing if "ngram_embedding.shards" not in k)
print(f"量化層 {len(bits)} 個（4-bit {sum(b == 4 for b in bits.values())}、8-bit {sum(b == 8 for b in bits.values())}）")
print(f"模型裡缺少的權重（不含 n-gram 表）：{missing[:10]}{' …' if len(missing) > 10 else ''}")
model.load_weights(list(weights.items()), strict=False)
mx.eval(model.parameters())

# ---------------------------------------------------------------- 逐層記錄
from mlx_lm.utils import load_tokenizer

tok = load_tokenizer(MODEL)
ids = tok.encode(PROMPT)
print(f"prompt：{PROMPT!r} → {len(ids)} 個 token：{ids}")

acts = {}
orig_layer_call = Q.DecoderLayer.__call__


def layer_call(self, h, *a, **kw):
    out = orig_layer_call(self, h, *a, **kw)
    acts[f"layer{len([k for k in acts if k.startswith('layer')]):02d}"] = out
    return out


Q.DecoderLayer.__call__ = layer_call
if FP32:
    _emb = model.model.embed_tokens
    class _F32Embed(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def __call__(self, x):
            return self.inner(x).astype(mx.float32)
        def as_linear(self, x):
            return self.inner.as_linear(x)
    model.model.embed_tokens = _F32Embed(_emb)
    print("（FP32 啟動值）")
x = mx.array([ids])
emb = model.model.embed_tokens(x)
logits = model(x)
mx.eval(logits, emb, *acts.values())
Q.DecoderLayer.__call__ = orig_layer_call

np.savez(OUT / "acts.npz", ids=np.array(ids, dtype=np.int32), embed=np.array(emb.astype(mx.float32))[0],
         logits=np.array(logits.astype(mx.float32))[0],
         **{k: np.array(v.astype(mx.float32))[0] for k, v in acts.items()})
last = np.array(logits[0, -1].astype(mx.float32))
top = np.argsort(-last)[:5]
print("最後一個位置的前 5 名：", [(int(t), tok.decode([int(t)]), float(last[t])) for t in top])

# ---------------------------------------------------------------- greedy 生成（確認參考實作本身正確）
cache = model.make_cache()
out_ids = list(ids)
y = model(mx.array([ids]), cache=cache)[:, -1]
for _ in range(24):
    nxt = int(mx.argmax(y, axis=-1).item())
    out_ids.append(nxt)
    y = model(mx.array([[nxt]]), cache=cache)[:, -1]
text = tok.decode(out_ids)
print("greedy 生成：", repr(text))
(OUT / "greedy.json").write_text(json.dumps({"prompt": PROMPT, "ids": out_ids, "text": text}, ensure_ascii=False, indent=1))
print(f"已存到 {OUT}")

# ---------------------------------------------------------------- 速度（REF_BENCH=1）：mlx-lm decode tok/s
import os, time
if os.environ.get("REF_BENCH"):
    n = int(os.environ.get("REF_BENCH"))
    cache = model.make_cache()
    y = model(mx.array([ids]), cache=cache)[:, -1]
    mx.eval(y)
    nxt = mx.argmax(y, axis=-1)
    for _ in range(4):  # 熱身
        y = model(nxt.reshape(1, 1), cache=cache)[:, -1]
        nxt = mx.argmax(y, axis=-1)
        mx.eval(nxt)
    t0 = time.perf_counter()
    for _ in range(n):
        y = model(nxt.reshape(1, 1), cache=cache)[:, -1]
        nxt = mx.argmax(y, axis=-1)
        mx.eval(nxt)
    dt = time.perf_counter() - t0
    print(f"mlx-lm decode：{n} 個 token，{dt / n * 1e3:.2f} ms/token → {n / dt:.1f} tok/s（牆鐘時間）")

# ---------------------------------------------------------------- prefill 速度（REF_PF=N）：mlx-lm 一次處理 N 個 token
if os.environ.get("REF_PF"):
    n = int(os.environ["REF_PF"])
    pf_ids = (ids * (n // len(ids) + 1))[:n]
    for rep in range(2):  # 第一次含編譯，取第二次
        cache = model.make_cache()
        t0 = time.perf_counter()
        y = model(mx.array([pf_ids]), cache=cache)
        mx.eval(y)
        dt = time.perf_counter() - t0
    print(f"mlx-lm prefill：{n} 個 token，{dt * 1e3:.0f} ms → {n / dt:.0f} tok/s（牆鐘時間，第二次）")
