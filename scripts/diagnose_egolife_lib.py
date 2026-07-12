#!/usr/bin/env python
"""EgoLife 记忆库全面诊断: 时间轴覆盖 + 每题证据精确状态。
用法: python scripts/diagnose_egolife_lib.py <person> <多层库dir> <结果jsonl> <QA json>
默认诊断 A1_JAKE 基线。只读, 不改库。"""
import json, sys, numpy as np
sys.path.insert(0,".")
from pathlib import Path
from src.dataloader.egolife import _clip_abs_start, list_clips, _CLIP_RE

person   = sys.argv[1] if len(sys.argv)>1 else "A1_JAKE"
ml_dir   = Path(sys.argv[2]) if len(sys.argv)>2 else Path("outputs/memory/egolife")/person
res_path = sys.argv[3] if len(sys.argv)>3 else "outputs/results/egolife/egolifeqa_a1jake.jsonl"
qa_path  = sys.argv[4] if len(sys.argv)>4 else "/home2/ycj/Datas/EgoLife_bench/EgoLifeQA/EgoLifeQA_A1_JAKE.json"
EGOROOT  = "/home2/ycj/Datas/EgoLife"

# ── 1) 源clip时间轴(ground truth) ──
src = list_clips(EGOROOT, person)   # [(abs_start, path)]
src_starts = np.array([s for s,_ in src])
print(f"源 clip: {len(src)}  时间轴 {src_starts.min():.0f}~{src_starts.max():.0f}s ({(src_starts.max()-src_starts.min())/3600:.1f}h跨度)")

# ── 2) 库 L1 节点(idx=chunk_id → 时间) ──
L1=[json.loads(l) for l in open(ml_dir/"L1.jsonl")]
l1_start=np.array([n["t_start"] for n in L1]); l1_end=np.array([n["t_end"] for n in L1])
print(f"库 L1 节点: {len(L1)}  覆盖 {l1_start.min():.0f}~{l1_end.max():.0f}s")

# ── 3) 时间轴覆盖: 源clip里有多少被库覆盖 ──
def hhmm(sec): 
    d=int(sec//86400)+1; r=sec%86400; return f"DAY{d} {int(r//3600):02d}:{int(r%3600//60):02d}"
covered_src=0; gap_src=[]
for s,_ in src:
    if ((l1_start<=s+15)&(l1_end>=s-15)).any(): covered_src+=1
    else: gap_src.append(s)
print(f"源clip被库覆盖: {covered_src}/{len(src)} = {covered_src/len(src)*100:.1f}%  缺口clip: {len(gap_src)}")
# 缺口按昼夜(6-22点=白天)
if gap_src:
    day_gap=sum(1 for s in gap_src if 6<=(s%86400)/3600<22)
    print(f"  缺口中 白天(6-22h): {day_gap}  夜间: {len(gap_src)-day_gap}")
    # 缺口时段聚类(连续)
    gs=sorted(gap_src); runs=[]; a=b=gs[0]
    for x in gs[1:]:
        if x-b<=40: b=x
        else: runs.append((a,b)); a=b=x
    runs.append((a,b))
    big=[(a,b) for a,b in runs if b-a>300]
    print(f"  连续缺口段: {len(runs)} 个, 其中>5min的 {len(big)} 段:")
    for a,b in sorted(big,key=lambda r:-(r[1]-r[0]))[:8]:
        print(f"    {hhmm(a)}~{hhmm(b)}  ({(b-a)/60:.0f}min)")

# ── 4) 每题证据精确状态 ──
qa=json.load(open(qa_path)); qa=list(qa.values()) if isinstance(qa,dict) else qa
res=[json.loads(l) for l in open(res_path) if l.strip() and json.loads(l).get("record_type")!="header"]
res=[d for d in res if d.get("pred_letter")]

def cls(d):
    si=d.get("sample_idx")
    if si is None or si>=len(qa): return None
    tt=qa[si].get("target_time",{})
    try: t=_clip_abs_start(int(str(tt["date"]).replace("DAY","")), tt["time"])
    except: return None
    # a) target时刻有源clip吗
    src_here = (np.abs(src_starts-t)<=30).any()
    # b) 库里有覆盖target的L1吗
    inlib = ((l1_start<=t+15)&(l1_end>=t-15)).any()
    # c) 检索到的L1里有靠近target的吗(±90s)
    ev=[e for e in d.get("evidence_chunk_ids",[]) if e<100000 and e<len(L1)]
    ret_hit = any(abs(l1_start[e]-t)<=90 for e in ev)
    n_sum = sum(1 for e in d.get("evidence_chunk_ids",[]) if e>=100000)
    return src_here, inlib, ret_hit, n_sum, d.get("correct") is True

from collections import defaultdict
buckets=defaultdict(lambda:[0,0])  # label:[correct,total]
allsum=[]
for d in res:
    c=cls(d)
    if not c: continue
    src_here,inlib,ret_hit,n_sum,ok=c
    allsum.append(n_sum)
    if not src_here: lab="1_源就没有clip(碎片/损坏)"
    elif not inlib:  lab="2_源有但库缺(该补)"
    elif not ret_hit:lab="3_库有但没检索到(检索问题)"
    else:            lab="4_库有且检索到(真可评判)"
    buckets[lab][1]+=1; buckets[lab][0]+=ok
print(f"\n=== 每题证据精确诊断 (共{sum(b[1] for b in buckets.values())}题) ===")
for lab in sorted(buckets):
    c,n=buckets[lab]; print(f"  {lab}: {n}题 判对{c} = {c/n*100:.1f}%")
tot_c=sum(b[0] for b in buckets.values()); tot_n=sum(b[1] for b in buckets.values())
print(f"  原始整体: {tot_c}/{tot_n} = {tot_c/tot_n*100:.2f}%")
真可评判=buckets["4_库有且检索到(真可评判)"]
print(f"  ★真可评判(库有证据且检索到): {真可评判[0]}/{真可评判[1]} = {真可评判[0]/真可评判[1]*100:.2f}%")
print(f"  证据里摘要节点(100000+)平均个数: {np.mean(allsum):.1f}")
