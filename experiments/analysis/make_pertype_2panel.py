#!/usr/bin/env python
"""Figure (2-panel): per-type BASE accuracy (left) + rubric gain (right), Video-MME-L.
Shows ALL 12 Video-MME-Long question types (nothing filtered). Left: negative-gain
types are already high (saturated, no room); positive-gain types start low and the
rubric lifts them -> the +1.3 overall is the verifier helping exactly where evidence
verification matters. Small-sample types (n<40) are drawn faded and grouped visually
so the reader can discount their single-question swings without them being hidden."""
from __future__ import annotations
import json, collections, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("/home2/ycj/Project/REVEAL"); OUT=ROOT/"paper_extract/Figures"
plt.rcParams.update({"font.family":"serif","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200})
C_POS="#2ca25f"; C_NEG="#d95f5f"; C_BASE="#8fb3d9"
C_POS_S="#a9d9bf"; C_NEG_S="#eebcbc"; C_BASE_S="#cdddec"  # faded, small-n
NMIN=40
def load(f): return [r for r in (json.loads(l) for l in open(f) if l.strip()) if r.get("trace_iters")]
# pool-era, same bank videommeL_1344: base = 记忆库+隐式 (no verifier module),
# +rubric = 记忆库+rubric = Ours. The right panel is the per-type value of the
# rubric verifier on top of the constructed memory bank.
b3={(r["video_id"],r["question"]):r for r in load(ROOT/"outputs/results/videomme/ablation/vmme_memory_implicit.jsonl")}
b5={(r["video_id"],r["question"]):r for r in load(ROOT/"outputs/results/videomme/ablation/vmme_pool_text.jsonl")}
common=set(b3)&set(b5)
bt=collections.defaultdict(lambda:[0,0,0])
for k in common:
    t=b5[k]["question_type"]; bt[t][0]+=1; bt[t][1]+=b3[k]["correct"]; bt[t][2]+=b5[k]["correct"]
# ALL types, no n>=40 filter. Sort ascending by gain so barh puts largest gain on top.
rows=[(t,n,100*c3/n,100*c5/n,100*(c5-c3)/n) for t,(n,c3,c5) in bt.items()]
rows.sort(key=lambda x:x[4])
_TN=sum(n for _,(n,_,_) in bt.items())
overall_gain=100*sum(c5-c3 for _,(n,c3,c5) in bt.items())/_TN
overall_base=100*sum(c3 for _,(n,c3,c5) in bt.items())/_TN
overall_full=100*sum(c5 for _,(n,c3,c5) in bt.items())/_TN
print(f"overall: base={overall_base:.2f} full={overall_full:.2f} gain={overall_gain:+.2f} (n={_TN}, common={len(common)})")
labels=[f"{t}\n(n={n})" for t,n,_,_,_ in rows]
small=[n<NMIN for _,n,_,_,_ in rows]
base=[r[2] for r in rows]; gain=[r[4] for r in rows]
y=np.arange(len(rows))
fig,(axL,axR)=plt.subplots(1,2,figsize=(11.0,3.9),gridspec_kw={"width_ratios":[1.15,1]})
# left: base accuracy
bcol=[C_BASE_S if s else C_BASE for s in small]
axL.barh(y,base,0.66,color=bcol,zorder=3)
for i,ba in enumerate(base): axL.text(ba-1.5,i,f"{ba:.0f}",va="center",ha="right",fontsize=7.5,color="white")
axL.set_yticks(y); axL.set_yticklabels(labels,fontsize=8)
for tick,s in zip(axL.get_yticklabels(),small):
    if s: tick.set_color("0.5")
axL.set_xlabel("Base accuracy, no verifier (%)"); axL.set_xlim(0,100)
axL.grid(axis="x",ls=":",lw=0.6,color="0.85"); axL.set_title("Base difficulty",fontsize=9)
# right: gain
cols=[]
for g,s in zip(gain,small):
    if g>0: cols.append(C_POS_S if s else C_POS)
    else:   cols.append(C_NEG_S if s else C_NEG)
axR.barh(y,gain,0.66,color=cols,zorder=3)
for i,g in enumerate(gain):
    axR.text(g+(0.15 if g>=0 else -0.15),i,f"{g:+.1f}",va="center",
             ha="left" if g>=0 else "right",fontsize=7.6,
             color="0.55" if small[i] else "0.15")
axR.axvline(0,color="0.4",lw=0.8); axR.axvline(overall_gain,color="0.6",ls="--",lw=1)
axR.text(overall_gain+0.15,len(rows)-0.55,f"overall {overall_gain:+.1f}",fontsize=7.5,color="0.4")
axR.set_yticks(y); axR.set_yticklabels([]); axR.set_xlabel("Gain from rubric verifier (pts)")
_g=[r[4] for r in rows]
axR.set_xlim(min(_g)-2, max(max(_g),overall_gain)+2.5); axR.grid(axis="x",ls=":",lw=0.6,color="0.85"); axR.set_title("Rubric gain",fontsize=9)
# legend note for faded = small sample
axR.text(0.985,0.015,"faded: n<40 (single-question swings)",transform=axR.transAxes,
         ha="right",va="bottom",fontsize=6.6,color="0.55",style="italic")
fig.tight_layout()
fig.savefig(OUT/"pertype_2panel.pdf"); fig.savefig(OUT/"pertype_2panel.png")
print("wrote",OUT/"pertype_2panel.pdf", "types=",len(rows))
