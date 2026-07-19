import json

def load(f):
    rows=[json.loads(l) for l in open(f) if l.strip()]
    return [r for r in rows if isinstance(r,dict)]

fp=load("outputs/results/egolife/egolife_fp_pool_text.jsonl")          # 第一人称(进行中)
base=load("outputs/results/egolife/ablation/egolife_pool_t07_text.jsonl")  # baseline 60%

def key(r):
    return (r.get("video_id"), r.get("sample_idx"))

bmap={key(r):r for r in base if "correct" in r}
fp_done=[r for r in fp if "correct" in r]

# 同题子集对比
common=[r for r in fp_done if key(r) in bmap]
fc=sum(bool(r["correct"]) for r in common)
bc=sum(bool(bmap[key(r)]["correct"]) for r in common)
n=len(common)
print(f"=== 中间对比(已完成 {len(fp_done)} 题, 同题子集 {n} 题) ===")
if n:
    print(f"  第一人称:  {fc}/{n} = {fc/n*100:.1f}%")
    print(f"  baseline:  {bc}/{n} = {bc/n*100:.1f}%  (同题子集)")
    print(f"  中间净差:  {(fc-bc)/n*100:+.1f}pt  {'✅第一人称领先' if fc>bc else ('⚠️落后' if fc<bc else '持平')}")
    print(f"  (最终以500题为准, baseline全量=60.0%)")

# 报错/异常扫描
def bad(r):
    p=str(r.get("pred_letter") or r.get("pred") or "")
    err=r.get("error")
    return (not p) or err or "400" in str(err)
errs=[r for r in fp_done if bad(r)]
print(f"\n=== 报错/空预测题: {len(errs)} ===")
for r in errs[:8]:
    print(f"  {key(r)} pred={r.get('pred_letter')!r} err={str(r.get('error'))[:60]}")
