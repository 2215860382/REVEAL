#!/usr/bin/env python
"""Figure 3 — verifier calibration x gold-inject oracle (egor1).
Groups: verifier verdict {Sufficient, Insufficient}.
Bars per group:
  pipeline (REVEAL retrieved) | oracle gold-caption | oracle raw-frames(no cap) | oracle cap+frames
Story: (1) verifier separates answerable (70) from not (47); (2) raw frames w/o
caption collapse (45<<76) -> caption is the backbone; (3) even the gold-caption
oracle caps ~76 -> the residual is caption-loss/model limit, and REVEAL's retrieval
sits only ~5pt under it -> near-optimal retrieval on certified questions.
  /home2/ycj/miniconda3/envs/videorag/bin/python experiments/analysis/make_verifier_oracle_fig.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
os.environ["NO_PROXY"]="127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1"
ROOT=Path(__file__).resolve().parents[2]; os.chdir(ROOT); sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments/analysis/gold_inject"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from probe import load_questions

OUT=ROOT/"paper_extract/Figures"; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200})
C_PIPE="#2c6fbb"; C_CAP="#7ba7d4"; C_FRM="#d95f5f"; C_CF="#2ca25f"

def main():
    key2q={r["key"]:r["question"] for r in load_questions("egor1")}
    rs=[r for r in (json.loads(l) for l in open(ROOT/"outputs/results/egor1/analysis/mq_holistic_k3.jsonl") if l.strip()) if r.get("trace_iters")]
    q2suf={r["question"].strip():any(it.get("verdict")=="SUFFICIENT" for it in r["trace_iters"]) for r in rs}
    gi=[r for r in (json.loads(l) for l in open(ROOT/"experiments/analysis/gold_inject/egor1_goldinject_dense.jsonl") if l.strip()) if r.get("preds") and r.get("gold_letter")]
    def pipe(w):
        s=[r for r in rs if q2suf.get(r["question"].strip())==w]; return 100*sum(x["correct"] for x in s)/len(s)
    def orc(w,arm):
        s=[r for r in gi if q2suf.get(key2q.get(r["key"],""),None)==w]
        return 100*sum(1 for r in s if r["preds"].get(arm)==r["gold_letter"])/len(s)
    groups=[("Sufficient",True),("Insufficient",False)]
    series=[("REVEAL (retrieved)",C_PIPE,lambda w:pipe(w)),
            ("Oracle: raw frames (no caption)",C_FRM,lambda w:orc(w,"F_denseNoCap")),
            ("Oracle: gold caption",C_CAP,lambda w:orc(w,"A_caption")),
            ("Oracle: caption + frames",C_CF,lambda w:orc(w,"E_densevideo"))]

    fig,ax=plt.subplots(figsize=(6.2,3.4))
    x=np.arange(len(groups)); w=0.2
    for i,(lab,col,fn) in enumerate(series):
        vals=[fn(g[1]) for g in groups]
        bars=ax.bar(x+(i-1.5)*w,vals,w,label=lab,color=col,edgecolor="white",lw=0.6)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,v+0.6,f"{v:.0f}",ha="center",va="bottom",fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{g[0]}\n(n={sum(1 for r in rs if q2suf.get(r['question'].strip())==g[1])})" for g in groups])
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0,90)
    ax.set_xlabel("Verifier sufficiency verdict")
    ax.legend(frameon=False,fontsize=7.6,ncol=2,loc="upper center",bbox_to_anchor=(0.5,1.19))
    ax.grid(axis="y",ls=":",lw=0.6,color="0.85")
    fig.tight_layout()
    fig.savefig(OUT/"verifier_oracle.pdf"); fig.savefig(OUT/"verifier_oracle.png")
    print("充足:",[f"{fn(True):.1f}" for _,_,fn in series])
    print("不足:",[f"{fn(False):.1f}" for _,_,fn in series])
    print("wrote",OUT/"verifier_oracle.pdf")

if __name__=="__main__": main()
