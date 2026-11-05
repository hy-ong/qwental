# 兩份逐位置 logits（run 的 QW_PPL_DUMP，float32 × 248320 依序）的對照：參考 vs 待測
# 用法：ref/venv/bin/python -I tools/logit_cmp.py <參考.bin> <待測.bin> [ids 檔（逗號分隔，給 NLL 用）] [起始位置 N（QW_PPL_PF）]
# 印出：平均與最大 KL(參考‖待測)、top-1 相同比例、兩邊的平均 NLL
import sys, numpy as np
V = 248320
a = np.memmap(sys.argv[1], dtype=np.float32, mode='r').reshape(-1, V)
b = np.memmap(sys.argv[2], dtype=np.float32, mode='r').reshape(-1, V)
n = min(len(a), len(b))
ids = [int(x) for x in open(sys.argv[3]).read().replace('\n', '').split(',') if x] if len(sys.argv) > 3 else None
p0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
start = p0 - 1 if p0 > 0 else 0 # dump 的第 0 列對應位置 start
def lsm(x):
    x = x.astype(np.float64); m = x.max(); return x - m - np.log(np.exp(x - m).sum())
kl, top, na, nb = [], 0, [], []
for i in range(n):
    la, lb = lsm(a[i]), lsm(b[i])
    kl.append(float((np.exp(la) * (la - lb)).sum()))
    top += int(la.argmax() == lb.argmax())
    if ids and start + i + 1 < len(ids):
        t = ids[start + i + 1]; na.append(-la[t]); nb.append(-lb[t])
kl = np.array(kl)
print('位置 %d 個：KL 平均 %.5f、中位數 %.5f、最大 %.4f（位置 %d）；top-1 相同 %d/%d（%.2f%%）'
      % (n, kl.mean(), np.median(kl), kl.max(), int(kl.argmax()), top, n, 100.0 * top / n))
if na: print('平均 NLL：參考 %.4f、待測 %.4f（差 %+.4f）' % (np.mean(na), np.mean(nb), np.mean(nb) - np.mean(na)))
