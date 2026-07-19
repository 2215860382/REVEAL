import json, os
from collections import defaultdict

def load(f):
    if not os.path.exists(f): return []
    return [json.loads(l) for l in open(f) if l.strip() and isinstance(json.loads(l),dict)]

fp=load("outputs/results/egor1/egor1_fp_pool_text.jsonl")               # 第一人称(建库后eval)
base=load("outputs/results/egor1/ablation/egor1_pool_text.jsonl")       # baseline 66.0%(第三人称库)

key=lambda r:(r.get("video_id"), r.get("sample_idx"))
bmap={key(r):r for r in base if "correct" in r}
fp_done=[r for r in fp if "correct" in r]

common=[r for r in fp_done if key(r) in bmap]
fc=sum(bool(r["correct"]) for r in common)
bc=sum(bool(bmap[key(r)]["correct"]) for r in common)
n=len(common)
print(f"=== egor1 第一人称 vs baseline(同题子集 {n}, 已完成 {len(fp_done)}/150) ===")
if n:
    print(f"  第一人称:  {fc}/{n} = {fc/n*100:.1f}%")
    print(f"  baseline:  {bc}/{n} = {bc/n*100:.1f}%  (同题, 全量66.0%)")
    print(f"  净差:      {(fc-bc)/n*100:+.1f}pt  {'✅领先' if fc>bc else ('⚠️落后' if fc<bc else '持平')}")

# 分人 + 分题型
byp=defaultdict(lambda:[0,0])
for r in fp_done:
    p=r.get("video_id","?"); byp[p][1]+=1; byp[p][0]+=bool(r["correct"])
print("  分人:", {p:f"{c}/{t}" for p,(c,t) in sorted(byp.items())})

# 报错
bad=[r for r in fp_done if not (r.get("pred_letter") or r.get("pred")) or r.get("error")]
print(f"  报错/空题: {len(bad)}")
