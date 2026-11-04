# 從 llama.cpp 的 ggml-common.h 取出 IQ 格式的查表，寫成 kernels/iq_tables.metal
#   python3 tools/gen_iq_tables.py <ggml-common.h>
import re, sys
src = open(sys.argv[1], encoding="utf-8").read()
def grab(name):
    m = re.search(r'GGML_TABLE_BEGIN\(([^,]+),\s*' + name + r',\s*([^)]+)\)(.*?)GGML_TABLE_END\(\)', src, re.S)
    return m.group(2).strip(), m.group(3).strip()
out = ['// IQ 量化格式的查表（取自 llama.cpp ggml-common.h，MIT License，Copyright (c) 2023-2026 The ggml authors）',
       '// 由 tools/gen_iq_tables.py 產生，不要手改', '#pragma once']
for name, mt in [('iq2s_grid', 'ulong'), ('iq3xxs_grid', 'uint'), ('iq3s_grid', 'uint'), ('ksigns_iq2xs', 'uchar'),
                 ('kmask_iq2xs', 'uchar'), ('kvalues_iq4nl', 'char')]:
    n, body = grab(name)
    out.append(f'constant {mt} {name}[{n}] = {{{body}}};')
open('kernels/iq_tables.metal', 'w').write('\n'.join(out) + '\n')
