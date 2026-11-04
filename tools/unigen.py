# 產生 src/unicode_tab.h：tokenizer 切詞用的 Unicode 類別（字母 L、標記 M、數字 N、空白）區間表
# 用法：python3 tools/unigen.py > src/unicode_tab.h
import sys, unicodedata

def ranges(pred):
    out, start = [], None
    for cp in range(0x110000):
        ok = pred(cp)
        if ok and start is None: start = cp
        if not ok and start is not None:
            out.append((start, cp - 1)); start = None
    if start is not None: out.append((start, 0x10FFFF))
    return out

cat = lambda cp: unicodedata.category(chr(cp))
WS = {0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000}
tabs = {
    "uni_L": ranges(lambda c: cat(c)[0] == "L"),
    "uni_M": ranges(lambda c: cat(c)[0] == "M"),
    "uni_N": ranges(lambda c: cat(c)[0] == "N"),
    "uni_WS": ranges(lambda c: c in WS),
}
print(f"// 由 tools/unigen.py 產生（Python {sys.version.split()[0]}、Unicode {unicodedata.unidata_version}），不要手改")
print("#pragma once\n#include <stdint.h>\n")
for name, rs in tabs.items():
    print(f"static const uint32_t {name}[][2] = {{")
    for i in range(0, len(rs), 6):
        print("    " + " ".join(f"{{0x{a:X}, 0x{b:X}}}," for a, b in rs[i:i + 6]))
    print("};")
    print(f"enum {{ {name}_n = {len(rs)} }};\n")

# ---- NFC 用的表：完整正規分解（不含韓文，韓文用演算法）、組合類別（CCC ≠ 0）、合成對
def full_decomp(cp):
    d = unicodedata.decomposition(chr(cp))
    if not d or d.startswith("<"): return [cp]
    out = []
    for x in d.split(): out += full_decomp(int(x, 16))
    return out

decomp = []
for cp in range(0x110000):
    if 0xAC00 <= cp <= 0xD7A3: continue
    d = full_decomp(cp)
    if d != [cp]: decomp.append((cp, d))
assert max(len(d) for _, d in decomp) <= 4
ccc = [(cp, unicodedata.combining(chr(cp))) for cp in range(0x110000) if unicodedata.combining(chr(cp))]
comp = []
for cp in range(0x110000):
    if 0xAC00 <= cp <= 0xD7A3: continue
    d = unicodedata.decomposition(chr(cp))
    if not d or d.startswith("<"): continue
    parts = [int(x, 16) for x in d.split()]
    if len(parts) == 2 and unicodedata.normalize("NFC", chr(parts[0]) + chr(parts[1])) == chr(cp):
        comp.append((parts[0], parts[1], cp))
comp.sort()
print("// 正規分解：{碼位, 長度, 分解後最多 4 個}")
print("static const uint32_t uni_decomp[][6] = {")
for i in range(0, len(decomp), 4):
    print("    " + " ".join("{" + ", ".join(f"0x{v:X}" for v in [cp, len(d)] + d + [0] * (4 - len(d))) + "}," for cp, d in decomp[i:i + 4]))
print("};")
print(f"enum {{ uni_decomp_n = {len(decomp)} }};\n")
print("static const uint32_t uni_ccc[][2] = {")
for i in range(0, len(ccc), 8):
    print("    " + " ".join(f"{{0x{a:X}, {b}}}," for a, b in ccc[i:i + 8]))
print("};")
print(f"enum {{ uni_ccc_n = {len(ccc)} }};\n")
print("// 合成：{第一個, 第二個, 結果}，依 (第一個, 第二個) 排序")
print("static const uint32_t uni_comp[][3] = {")
for i in range(0, len(comp), 5):
    print("    " + " ".join(f"{{0x{a:X}, 0x{b:X}, 0x{c:X}}}," for a, b, c in comp[i:i + 5]))
print("};")
print(f"enum {{ uni_comp_n = {len(comp)} }};")
