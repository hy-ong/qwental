# vs_strata_lp.py 的延伸：Qwental 的分布照 Strata 的取樣規則（top_k 20、top_p 0.95）截斷再正規化後，與 Strata 回報的 logprob 比較
import json, sys, os, numpy as np
from mlx_lm.utils import load_tokenizer
from pathlib import Path
R=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
tok=load_tokenizer(Path(R)/'models/Qwen3.8-Flash-Next-mixed-4-8bit')
src=open(R+'/bench/vs_strata_greedy.py').read().split('\ndef post')[0]; g={'__file__':R+'/bench/vs_strata_greedy.py'}; exec(src,g); P=dict(g['PROMPTS'])
for name in ('prose','code','edit'):
    A=json.load(open(R+'/build/gc_strata.json'))[name]
    pids=tok.encode('<|im_start|>user\n'+P[name]+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'); n0=len(pids)-1
    L=np.fromfile(R+'/build/lp_%s.f32'%name,dtype=np.float32).reshape(-1,248320)
    full=[]; tr=[]
    for i,(t,l,top) in enumerate(A['lp']):
        x=L[n0+i].astype(np.float64); x-=x.max(); p=np.exp(x); p/=p.sum()
        ls=np.log(p+1e-300)
        o=np.argsort(-p)[:20]; q=p[o]; c=np.cumsum(q); k=int(np.searchsorted(c,0.95))+1; keep=o[:k]
        pt=np.zeros_like(p); pt[keep]=p[keep]; pt/=pt.sum(); lt=np.log(pt+1e-300)
        for tt,tl in top:
            e=tok.encode(tt)
            if len(e)==1 and tl>-20:
                full.append(ls[e[0]]-tl)
                if pt[e[0]]>0: tr.append(lt[e[0]]-tl)
    full=np.array(full); tr=np.array(tr)
    print('%-6s 完整 softmax：平均差 %+.4f、|差| 中位 %.4f｜top-k20/top-p0.95 截斷後：平均差 %+.4f、|差| 中位 %.4f、90%% %.4f、最大 %.4f'%(name,full.mean(),np.median(abs(full)),tr.mean(),np.median(abs(tr)),np.percentile(abs(tr),90),abs(tr).max()))
