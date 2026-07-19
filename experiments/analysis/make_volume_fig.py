#!/usr/bin/env python
"""Figure: accuracy vs evidence VOLUME on Video-MME-L (quality beats quantity).

Volume curve  = single-query plain dense top-N over short L1 captions, same
                answerer, NO method (no L2/L3, no rerank, no rubric verify).
REVEAL star     = full pipeline operating point: mean #segments fed vs accuracy,
                read from the mainline ablation log abl_5_full.jsonl.

  /home2/ycj/miniconda3/envs/videorag/bin/python experiments/analysis/make_volume_fig.py
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
VOL  = ROOT / "experiments/analysis/evidence_budget/videomme_volume.jsonl"
ABL  = ROOT / "outputs/results/videomme/ablation/vmme_pool_text.jsonl"
OUT  = ROOT / "paper_extract/Figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})
C_VOL, C_REVEAL, C_REF = "#d95f5f", "#1b4f8a", "#9aa0a6"

CAPS = [8, 16, 32, 64, 128, 256, 384, 512]

def main():
    recs = [json.loads(l) for l in open(VOL)]
    n = len(recs)
    accs = []
    for N in CAPS:
        k = str(N)
        ok = sum(1 for r in recs if k in r["by_cap"] and r["by_cap"][k]["correct"])
        accs.append(100 * ok / n)
    print(f"volume curve (n={n}):")
    for N, a in zip(CAPS, accs):
        print(f"  top-{N:>3}: {a:.1f}")

    # REVEAL operating point from mainline
    ab = [json.loads(l) for l in open(ABL) if '"correct"' in l]
    ab = [r for r in ab if "correct" in r]
    reveal_acc = 100 * sum(r["correct"] for r in ab) / len(ab)
    reveal_nev = st.mean([r["n_evidence_to_answer"] for r in ab
                        if r.get("n_evidence_to_answer") is not None])
    print(f"REVEAL: n_ev={reveal_nev:.1f} acc={reveal_acc:.2f}")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(CAPS, accs, "-o", color=C_VOL, lw=1.8, ms=5, zorder=3,
            label="Volume only (dense top-$N$, no verification)")
    for x, a in zip(CAPS, accs):
        ax.annotate(f"{a:.0f}", (x, a), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7.5, color=C_VOL)
    # REVEAL star + reference line
    ax.axhline(reveal_acc, color=C_REF, ls="--", lw=1, zorder=1)
    ax.scatter([reveal_nev], [reveal_acc], marker="*", s=260, color=C_REVEAL,
               edgecolor="white", lw=1, zorder=5,
               label=f"REVEAL (verified, {reveal_nev:.0f} seg.)")
    ax.annotate(f"{reveal_acc:.1f}", (reveal_nev, reveal_acc),
                textcoords="offset points", xytext=(8, 6),
                fontsize=9, fontweight="bold", color=C_REVEAL)
    ax.set_xscale("log", base=2)
    # thin out the crowded high-N ticks (384/512 collide on a log axis)
    xt = [c for c in CAPS if c != 384]
    ax.set_xticks(xt); ax.set_xticklabels([str(c) for c in xt])
    ax.set_ylim(61, 80.5)
    ax.set_xlabel("# L1 evidence segments fed to the answerer")
    ax.set_ylabel("Video-MME-L accuracy (%)")
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9,
              edgecolor="none")
    fig.tight_layout()
    fig.savefig(OUT / "evidence_volume.pdf")
    fig.savefig(OUT / "evidence_volume.png")
    print("wrote evidence_volume.pdf/.png")

if __name__ == "__main__":
    main()
