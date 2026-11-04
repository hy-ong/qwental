# 把文字轉成 token id（逗號分隔），給 build/run 用：ref/venv/bin/python tools/tok.py 檔案
import sys
from mlx_lm.utils import load_tokenizer
from pathlib import Path
tok = load_tokenizer(Path(__file__).resolve().parent.parent / "models/Qwen3.8-Flash-Next-mixed-4-8bit")
if len(sys.argv) > 2 and sys.argv[1] == "-d":
    print(tok.decode([int(x) for x in sys.argv[2].split(",")]))
else:
    print(",".join(map(str, tok.encode(open(sys.argv[1], newline="").read()))))
