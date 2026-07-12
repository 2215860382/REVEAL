#!/usr/bin/env python
"""多层库完整性审计: 逐库验证 L1=单库chunk、L2≈L1/12、L3≈L2/12、向量行对齐、
无全零向量(漏embedding)、无stale。每次建库/重建多层后跑一遍。
用法: python scripts/audit_multilayer.py
注意: 只验"内部一致性", 不验单库是否含全部源clip(那是建库的事)。"""
import numpy as np, json, math, sys
from pathlib import Path

def single_chunks(sd):
    vp = sd/"vectors.npz"
    try: return len(np.load(vp)["chunk_ids"]) if vp.exists() else None
    except: return None

def rows(p):
    if not p.exists(): return -1
    try:
        z=np.load(p); return z[list(z.keys())[0]].shape[0]
    except: return -2

def zeros(p):
    if not p.exists(): return 0
    try:
        z=np.load(p); v=z[list(z.keys())[0]]; return int((np.abs(v).sum(axis=1)==0).sum())
    except: return -1

def audit(ml, sroot):
    meta = json.load(open(ml/"meta.json")) if (ml/"meta.json").exists() else {}
    cnt = lambda f: sum(1 for _ in open(ml/f)) if (ml/f).exists() else -1
    l1,l2,l3 = cnt("L1.jsonl"),cnt("L2.jsonl"),cnt("L3.jsonl")
    sc = single_chunks(sroot/ml.name) if (sroot/ml.name).exists() else None
    iss=[]
    if sc is not None and l1!=sc: iss.append(f"L1={l1}≠单库{sc}")
    if rows(ml/"L1_text.npz")!=l1: iss.append(f"L1_text行≠L1")
    if rows(ml/"L1_visual.npz")!=l1: iss.append(f"L1_visual行≠L1")
    if l1>0 and abs(l2-math.ceil(l1/12))>2: iss.append(f"L2={l2}≠≈{math.ceil(l1/12)}")
    if l2>0 and abs(l3-math.ceil(l2/12))>2: iss.append(f"L3={l3}≠≈{math.ceil(l2/12)}")
    for f in ["L1_text.npz","L2.npz","L3.npz"]:
        z=zeros(ml/f)
        if z>0: iss.append(f"{f}有{z}全零向量")
    if meta.get("status") and meta["status"]!="ok": iss.append(f"status={meta['status']}")
    return l1,l2,l3,sc,iss

def main():
    pairs=[("outputs/memory/egolife","outputs/memory/egolife_single","EGOLIFE"),
           ("outputs/memory/lvbench_fullres","outputs/memory/lvbench_single_fullres","LVBENCH")]
    total_bad=0
    for mlr,sr,label in pairs:
        mlr,sr=Path(mlr),Path(sr)
        if not mlr.exists(): continue
        dirs=sorted(d for d in mlr.iterdir() if d.is_dir())
        bad=0
        print(f"\n== {label}: {len(dirs)} 库 ==")
        for d in dirs:
            l1,l2,l3,sc,iss=audit(d,sr)
            if iss:
                bad+=1; print(f"  ✗ {d.name}: L1={l1} L2={l2} L3={l3} 单库={sc}  -> {'; '.join(iss)}")
        print(f"  {len(dirs)-bad}/{len(dirs)} 完整" + (f", {bad} 有问题" if bad else " ✓"))
        total_bad+=bad
    print(f"\n总计问题库: {total_bad}")
    sys.exit(1 if total_bad else 0)

if __name__=="__main__": main()
