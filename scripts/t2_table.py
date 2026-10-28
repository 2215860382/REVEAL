#!/usr/bin/env python
"""T2 (stop-rule aggregation) summary: max vs min, two datasets, paired same-qid.
Only env differing between the two columns is REVEAL_SUFFICIENCY_AGG (max=paper mainline).
Robust to partial files: pairs on the common set of completed (video_id, sample_idx)."""
import json, os

PAIRS = [
    ("LVBench",  "outputs/results/lvbench/rebuttal/ours_s1.jsonl",   # AGG=max (existing, 66.06 full)
                 "outputs/results/lvbench/t2/lvbench_library_min.jsonl"),  # AGG=min
    ("VideoMME-L","outputs/results/videomme/t2/vmme_library_max.jsonl",
                  "outputs/results/videomme/t2/vmme_library_min.jsonl"),
]

def load(p):
    if not os.path.exists(p): return None
    d = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("record_type"): continue
        if r.get("error"): continue
        k = (r.get("video_id"), r.get("sample_idx"))
        d[k] = r
    return d

def iters(r):
    ti = r.get("trace_iters")
    if isinstance(ti, list): return len(ti)
    return None

print(f"\n{'dataset':12} | {'agg':4} | {'acc (full)':16} | {'acc (paired)':14} | {'avg iters':9} | n_paired")
print("-"*90)
for name, fmax, fmin in PAIRS:
    A, B = load(fmax), load(fmin)
    def full(d):
        if not d: return None
        c = sum(1 for r in d.values() if r.get("correct"))
        return 100*c/len(d), c, len(d)
    fa, fb = full(A), full(B)
    common = sorted(set(A) & set(B)) if (A and B) else []
    def paired(d):
        if not common: return None
        c = sum(1 for k in common if d[k].get("correct"))
        return 100*c/len(common)
    def avgit(d):
        if not common: return None
        vs = [iters(d[k]) for k in common if iters(d[k]) is not None]
        return sum(vs)/len(vs) if vs else None
    for agg, d, f in (("max", A, fa), ("min", B, fb)):
        fs = f"{f[0]:.2f}% ({f[1]}/{f[2]})" if f else "pending"
        ps = f"{paired(d):.2f}%" if (d and common) else "-"
        it = f"{avgit(d):.2f}" if (d and avgit(d) is not None) else "-"
        print(f"{name:12} | {agg:4} | {fs:16} | {ps:14} | {it:9} | {len(common)}")
    if common:
        da, db = paired(A), paired(B)
        print(f"{'  -> Δ(min-max) paired':12} = {db-da:+.2f} pt   (iters {avgit(A):.2f}→{avgit(B):.2f})")
    print("-"*90)
print("max = paper-mainline aggregation (ρt = max_o ρt,o); min = stricter rule reviewers asked for.")
print("paired = same (video_id,sample_idx) completed in BOTH columns; difficulty controlled.")
