#!/usr/bin/env python
"""Generate the offline (log-only) analysis figures for the REVEAL paper.

Reads the five VideoMME ablation logs and emits three PDFs into
paper_extract/Figures/:
  - abl_waterfall.pdf       component-ablation waterfall (accuracy vs added part)
  - pertype_rubric_gain.pdf per-question-type accuracy gain from the rubric verifier
  - acc_cost_pareto.pdf     accuracy vs mean wall-clock cost across variants

Run with the videorag env python (has matplotlib):
  /home2/ycj/miniconda3/envs/videorag/bin/python experiments/analysis/make_figs.py
"""
from __future__ import annotations
import json, collections, statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ABL  = ROOT / "outputs/results/videomme/ablation"
OUT  = ROOT / "paper_extract/Figures"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "naive":       "abl_1_naive",
    "multiround":  "abl_2_multiround",
    "multilayer":  "abl_3_multilayer",
    "single_rub":  "abl_4_singlelayer_rubric",
    "full":        "abl_5_full",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

# palette
C_MAIN = "#2c6fbb"
C_POS  = "#2ca25f"
C_NEG  = "#d95f5f"
C_GREY = "#9aa0a6"


def load(name):
    recs = []
    for line in open(ABL / f"{name}.jsonl"):
        r = json.loads(line)
        if "correct" in r:
            recs.append(r)
    return recs


def key(r):
    return (r["video_id"], r["sample_idx"])


def main():
    data = {k: load(v) for k, v in FILES.items()}
    stats = {}
    for k, recs in data.items():
        acc = 100 * sum(r["correct"] for r in recs) / len(recs)
        el  = st.mean([r["elapsed"] for r in recs if r.get("elapsed")])
        nev = st.mean([r["n_evidence_to_answer"] for r in recs
                       if r.get("n_evidence_to_answer") is not None])
        stats[k] = dict(acc=acc, elapsed=el, evid=nev, n=len(recs))
    print("stats:", {k: (round(v["acc"], 1), round(v["elapsed"])) for k, v in stats.items()})

    # ── Figure 1: component waterfall ─────────────────────────────────────────
    steps = [
        ("Naive\nretrieval",        stats["naive"]["acc"]),
        ("+ Iterative\nrepair",     stats["multiround"]["acc"]),
        ("+ Hierarchical\nmemory",  stats["multilayer"]["acc"]),
        ("+ Rubric\nverifier",      stats["full"]["acc"]),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = range(len(steps))
    accs = [a for _, a in steps]
    bars = ax.bar(xs, accs, width=0.62, color=C_MAIN, zorder=3)
    bars[-1].set_color("#1b4f8a")
    for i, (lbl, a) in enumerate(steps):
        ax.text(i, a + 0.5, f"{a:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        if i > 0:
            d = accs[i] - accs[i - 1]
            ax.annotate(f"+{d:.1f}", xy=(i - 0.5, (accs[i] + accs[i - 1]) / 2),
                        ha="center", va="center", fontsize=8, color=C_POS,
                        fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([s for s, _ in steps], fontsize=8.5)
    ax.set_ylim(60, 82)
    ax.set_ylabel("VideoMME accuracy (%)")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "abl_waterfall.pdf")
    plt.close(fig)

    # ── Figure 2: per-type rubric gain (abl3 no-rubric -> abl5 rubric) ────────
    b3 = {key(r): r for r in data["multilayer"]}
    b5 = {key(r): r for r in data["full"]}
    common = set(b3) & set(b5)
    bytype = collections.defaultdict(lambda: [0, 0, 0])  # n, c3, c5
    for k in common:
        t = b5[k]["question_type"]
        bytype[t][0] += 1
        bytype[t][1] += b3[k]["correct"]
        bytype[t][2] += b5[k]["correct"]
    rows = []
    for t, (n, c3, c5) in bytype.items():
        if n >= 40:  # only report types with enough support
            rows.append((t, n, c3, c5, 100 * (c5 - c3) / n, c5 - c3))
    rows.sort(key=lambda x: x[4])  # ascending so positives end up on top
    overall = 100 * (sum(r["correct"] for r in b5.values()) -
                     sum(r["correct"] for r in b3.values())) / len(common)

    # Grouped before/after bars: reviewers see both the baseline height and the
    # change, so the "losses" read as tiny moves on already-saturated (tall) bars
    # while the gains stand out on the lower-baseline discriminative types.
    import numpy as np
    rows_desc = list(reversed(rows))  # gains at top
    labels = [t for t, *_ in rows_desc]
    base_a = [100 * c3 / n for t, n, c3, c5, _, _ in rows_desc]
    full_a = [100 * c5 / n for t, n, c3, c5, _, _ in rows_desc]
    dqs    = [dq for *_, dq in rows_desc]
    ns     = [n for _, n, *_ in rows_desc]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    y = np.arange(len(rows_desc)); h = 0.38
    ax.barh(y + h / 2, base_a, h, color="#c3d4e8", zorder=3, label="no verifier")
    fcols = [C_POS if d > 0 else (C_GREY if d < 0 else "#b8b8b8") for d in dqs]
    ax.barh(y - h / 2, full_a, h, color=fcols, zorder=3, label="+ verifier")
    for i, (fa, d) in enumerate(zip(full_a, dqs)):
        tag = f"{d:+d} q" if d != 0 else "0 q"
        ax.text(fa + 1, i - h / 2, tag, va="center", ha="left", fontsize=7.5,
                color="#2f7d4f" if d > 0 else "#888")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}  (n={n})" for t, n in zip(labels, ns)], fontsize=8)
    ax.set_xlabel("VideoMME accuracy (%)")
    ax.set_xlim(50, 100)
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9,
              edgecolor="none")
    ax.grid(axis="x", alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "pertype_rubric_gain.pdf")
    plt.close(fig)

    # ── Figure 3: accuracy vs cost Pareto ─────────────────────────────────────
    order = ["naive", "multiround", "multilayer", "single_rub", "full"]
    names = {
        "naive": "Naive", "multiround": "Iter. repair",
        "multilayer": "Hier. memory", "single_rub": "Single-layer+rubric",
        "full": "REVEAL (full)",
    }
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = [stats[k]["elapsed"] for k in order]
    ys = [stats[k]["acc"] for k in order]
    # Pareto frontier: points not dominated (higher acc AND lower cost)
    front = []
    for k in order:
        x, y = stats[k]["elapsed"], stats[k]["acc"]
        dominated = any(stats[o]["elapsed"] <= x and stats[o]["acc"] >= y and o != k
                        for o in order)
        front.append(not dominated)
    ax.plot([x for x, f in zip(xs, front) if f],
            [y for y, f in zip(ys, front) if f],
            "-", color=C_GREY, lw=1.2, zorder=1)
    for k in order:
        x, y = stats[k]["elapsed"], stats[k]["acc"]
        on = front[order.index(k)]
        ax.scatter([x], [y], s=70, color=C_MAIN if on else C_GREY,
                   edgecolor="white", lw=1, zorder=3)
        dy = 0.5 if k != "single_rub" else -1.2
        ax.annotate(names[k], (x, y), xytext=(4, dy * 8), textcoords="offset points",
                    fontsize=8)
    ax.set_xlabel("Mean latency per question (s)")
    ax.set_ylabel("VideoMME accuracy (%)")
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "acc_cost_pareto.pdf")
    plt.close(fig)

    print("wrote:", *(str((OUT / f).relative_to(ROOT)) for f in
          ["abl_waterfall.pdf", "pertype_rubric_gain.pdf", "acc_cost_pareto.pdf"]))


if __name__ == "__main__":
    main()
