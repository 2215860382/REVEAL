#!/usr/bin/env python
"""Sufficiency <-> accuracy correlation for a REVEAL result jsonl.

The paper claim: the per-option verifier's SUFFICIENT judgment must POSITIVELY
correlate with answer accuracy. This reports, from the per-iteration trace:

  1. Early-stop group accuracy: questions whose FINAL verdict is SUFFICIENT
     (verifier let the pipeline stop) vs INSUFFICIENT (ran to max_iter).
     Claim holds iff SUFFICIENT acc > INSUFFICIENT acc.
  2. Score-bucket monotonicity: bucket the decisive rubric score (min over
     options = weakest-option decidability) into [0,.4)/[.4,.6)/[.6,.8)/[.8,1]
     and report accuracy per bucket. Claim holds iff acc is (weakly) monotone up.
  3. Same two views at ITER-0 (does the FIRST verification already predict it).

Usage: python scripts/analyze_sufficiency_correlation.py <result.jsonl> [more.jsonl ...]
"""
import json, sys
from collections import defaultdict

BUCKETS = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]


def bucket_of(s):
    for lo, hi in BUCKETS:
        if lo <= s < hi:
            return f"[{lo:.1f},{hi if hi <= 1 else 1.0:.1f})"
    return "?"


def min_option_score(it):
    # New trace stores the scalar decisive score directly (min over options).
    s = it.get("rubric_score")
    if s is not None:
        return float(s)
    # Fallback for old-format traces that only had the per-option dict.
    d = it.get("option_rubric_scores") or {}
    vals = [float(v) for v in d.values() if v is not None]
    return min(vals) if vals else None


def acc(rows):
    n = len(rows)
    c = sum(1 for r in rows if r)
    return (c, n, (100.0 * c / n if n else 0.0))


def analyze(path):
    recs = []
    for line in open(path):
        d = json.loads(line)
        if d.get("record_type") == "header":
            continue
        if d.get("trace_iters"):
            recs.append(d)
    if not recs:
        print(f"\n### {path}\n  (no scored records)")
        return
    print(f"\n{'='*70}\n### {path}   n={len(recs)}")
    overall = acc([r["correct"] for r in recs])
    print(f"  overall accuracy: {overall[0]}/{overall[1]} = {overall[2]:.2f}%")

    # ---- 1. early-stop (final verdict) group accuracy ----
    grp = defaultdict(list)
    for r in recs:
        fv = r["trace_iters"][-1].get("verdict")
        grp[fv].append(r["correct"])
    print("\n  [1] final-verdict group accuracy (SUFFICIENT=early stop):")
    for v in ("SUFFICIENT", "INSUFFICIENT"):
        if grp[v]:
            c, n, a = acc(grp[v])
            print(f"      {v:<12} {c:>4}/{n:<4} = {a:6.2f}%")
    if grp["SUFFICIENT"] and grp["INSUFFICIENT"]:
        gap = acc(grp["SUFFICIENT"])[2] - acc(grp["INSUFFICIENT"])[2]
        print(f"      >>> SUFFICIENT - INSUFFICIENT = {gap:+.2f} pt  "
              f"({'CORRELATED' if gap > 0 else 'INVERTED!'})")

    # ---- 2. decisive-score bucket monotonicity (final iter) ----
    print("\n  [2] decisive min-option-score bucket -> accuracy (final iter):")
    _bucket_report(recs, which="final")

    # ---- 3. iter-0 views ----
    print("\n  [3] ITER-0 verdict group accuracy:")
    g0 = defaultdict(list)
    for r in recs:
        g0[r["trace_iters"][0].get("verdict")].append(r["correct"])
    for v in ("SUFFICIENT", "INSUFFICIENT"):
        if g0[v]:
            c, n, a = acc(g0[v])
            print(f"      iter0 {v:<12} {c:>4}/{n:<4} = {a:6.2f}%")
    print("\n  [3b] ITER-0 min-option-score bucket -> accuracy:")
    _bucket_report(recs, which="iter0")


def _bucket_report(recs, which):
    buckets = defaultdict(list)
    for r in recs:
        it = r["trace_iters"][-1] if which == "final" else r["trace_iters"][0]
        s = min_option_score(it)
        if s is None:
            continue
        buckets[bucket_of(s)].append(r["correct"])
    order = [f"[{lo:.1f},{(hi if hi<=1 else 1.0):.1f})" for lo, hi in BUCKETS]
    prev = None
    mono = True
    for b in order:
        if buckets[b]:
            c, n, a = acc(buckets[b])
            flag = ""
            if prev is not None and a + 1e-9 < prev:
                flag = "  <-- drop"
                mono = False
            print(f"      {b:<12} {c:>4}/{n:<4} = {a:6.2f}%{flag}")
            prev = a
    print(f"      monotonic-up: {'YES' if mono else 'NO'}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
