#!/usr/bin/env python
"""Turn the analysis-batch logs into the B1/B2/B3/B5 experiment figures.

Reads whatever jsonls exist under outputs/results/videomme/analysis/ and emits
into paper_extract/Figures/ only the figures whose data is ready. Safe to run
repeatedly while the batch is still going.

  tau_sensitivity.pdf   B5  accuracy vs sufficiency threshold tau
  rounds_K.pdf          B3  accuracy vs max retrieval rounds K
  budget_l1cap.pdf      B2  accuracy vs L1 evidence budget
  verifier_vs_blind.pdf B1  guided vs blind repair, overall + per-type

  /home2/ycj/miniconda3/envs/videorag/bin/python experiments/analysis/make_exp_figs.py
"""
from __future__ import annotations
import json, collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ANA  = ROOT / "outputs/results/videomme/analysis"
OUT  = ROOT / "paper_extract/Figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})
C_MAIN, C_ALT, C_POS, C_NEG = "#2c6fbb", "#d9822b", "#2ca25f", "#d95f5f"


def load(name):
    """Return list of per-question records for a job, or None if <900 (incomplete)."""
    f = ANA / f"{name}.jsonl"
    if not f.exists():
        return None
    recs = [json.loads(l) for l in open(f) if '"correct"' in l]
    return recs if len(recs) >= 900 else recs or None


def acc(recs):
    return 100 * sum(r["correct"] for r in recs) / len(recs)


def complete(recs):
    return recs is not None and len(recs) >= 900


def bytype(recs):
    d = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        d[r["question_type"]][0] += 1
        d[r["question_type"]][1] += r["correct"]
    return d


def fig_tau():
    pts = [(0.6, "tau_th60"), (0.7, "tau_th70"), (0.8, "th80_K3_cap8"),
           (0.9, "tau_th90"), (0.95, "tau_th95")]
    xs, ys = [], []
    for t, name in pts:
        recs = load(name)
        if complete(recs):
            xs.append(t); ys.append(acc(recs))
    if len(xs) < 2:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(xs, ys, "o-", color=C_MAIN, lw=1.6, ms=6)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set_xlabel(r"Type-specific sufficiency threshold $\tau$")
    ax.set_ylabel("VideoMME accuracy (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "tau_sensitivity.pdf"); plt.close(fig)
    return dict(zip(xs, [round(y, 2) for y in ys]))


def fig_rounds():
    pts = [(1, "th80_K1"), (2, "th80_K2"), (3, "th80_K3_cap8")]
    xs, ys = [], []
    for k, name in pts:
        recs = load(name)
        if complete(recs):
            xs.append(k); ys.append(acc(recs))
    if len(xs) < 2:
        return None
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(xs, ys, "o-", color=C_MAIN, lw=1.6, ms=7)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.set_xlabel("Max retrieval rounds $K$")
    ax.set_ylabel("VideoMME accuracy (%)")
    ax.set_xticks(xs)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "rounds_K.pdf"); plt.close(fig)
    return dict(zip(xs, [round(y, 2) for y in ys]))


def fig_budget():
    pts = [(4, "th80_cap4"), (8, "th80_K3_cap8"), (16, "th80_cap16"), (32, "th80_cap32")]
    xs, ys = [], []
    for c, name in pts:
        recs = load(name)
        if complete(recs):
            xs.append(c); ys.append(acc(recs))
    if len(xs) < 2:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(xs, ys, "o-", color=C_MAIN, lw=1.6, ms=6)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xscale("log", base=2); ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("L1 evidence pool per round")
    ax.set_ylabel("VideoMME accuracy (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "budget_l1cap.pdf"); plt.close(fig)
    return dict(zip(xs, [round(y, 2) for y in ys]))


def fig_verifier_vs_blind():
    g = load("th80_K3_cap8"); b = load("blind_K3")
    if not (complete(g) and complete(b)):
        return None
    # overall + per-type (n>=40) grouped bars
    gt, bt = bytype(g), bytype(b)
    types = [t for t in gt if gt[t][0] >= 40]
    types.sort(key=lambda t: -(gt[t][1] / gt[t][0]))
    labels = ["Overall"] + types
    gacc = [acc(g)] + [100 * gt[t][1] / gt[t][0] for t in types]
    bacc = [acc(b)] + [100 * bt[t][1] / bt[t][0] for t in types]
    import numpy as np
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.bar(x - w / 2, gacc, w, label="Verifier-guided repair", color=C_MAIN, zorder=3)
    ax.bar(x + w / 2, bacc, w, label="Blind repair (equal compute)", color=C_ALT, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labels], fontsize=7.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(min(bacc + gacc) - 4, max(bacc + gacc) + 4)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout(); fig.savefig(OUT / "verifier_vs_blind.pdf"); plt.close(fig)
    return {"overall_guided": round(gacc[0], 2), "overall_blind": round(bacc[0], 2)}


def main():
    made = {}
    for fn in (fig_tau, fig_rounds, fig_budget, fig_verifier_vs_blind):
        try:
            r = fn()
            made[fn.__name__] = r if r is not None else "PENDING (data incomplete)"
        except Exception as e:
            made[fn.__name__] = f"ERROR: {e}"
    for k, v in made.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa
    main()
