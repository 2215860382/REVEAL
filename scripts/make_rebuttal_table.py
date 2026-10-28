#!/usr/bin/env python
"""Assemble the 5-condition rebuttal table: separation (held-out vmme45) + LVBench acc.
Reads result jsonl directly; prints '(partial …/1267)' or 'pending' for unfinished rows."""
import json, os, statistics as st

SEP = "outputs/rubric/rebuttal_sep"
ACC = "outputs/results/lvbench/rebuttal"
THR = [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]

# label -> (separation scored file, [acc seed files])
ROWS = [
    ("Ours (library, old mainline)", f"{SEP}/ours_scored.jsonl",
        [f"{ACC}/ours_s1.jsonl", f"{ACC}/ours_s2.jsonl"]),
    ("Ours (RTD-new round5)",        f"{SEP}/newrubric_scored.jsonl",
        [f"{ACC}/newrubric_t07_s1.jsonl", f"{ACC}/newrubric_t07_s2.jsonl"]),
    ("no rubric (binary self-judge)", f"{SEP}/no_rubric_true_scored.jsonl",
        [f"{ACC}/no_rubric_true_s1.jsonl", f"{ACC}/no_rubric_true_s2.jsonl"]),
    ("minimal rubric (holistic)",    f"{SEP}/no_rubric_scored.jsonl",
        [f"{ACC}/no_rubric_s1.jsonl", f"{ACC}/no_rubric_s2.jsonl"]),
    ("generic rubric (relevance)",   f"{SEP}/generic_scored.jsonl",
        [f"{ACC}/generic_s1.jsonl", f"{ACC}/generic_s2.jsonl"]),
]

def sep_stats(p):
    if not os.path.exists(p): return None
    ps = [json.loads(l) for l in open(p)]
    if not ps: return None
    def rate(t): return sum(1 for q in ps if q['weak_score'] < t <= q['strong_score']) / len(ps)
    best = max(THR, key=rate)
    return 100*rate(best), best, 100*rate(0.7), len(ps)

def acc_of(p):
    if not os.path.exists(p): return None
    t=c=0
    for l in open(p):
        d=json.loads(l)
        if d.get("record_type"): continue
        t+=1; c+= 1 if d.get("correct") else 0
    return (100*c/t, c, t) if t else None

print(f"\n{'condition':32} | {'separation(vmme45)':26} | {'LVBench acc (tau=0.7)':30}")
print("-"*95)
for label, sepf, accfs in ROWS:
    s = sep_stats(sepf)
    if s: sepstr = f"{s[0]:.1f}%@best={s[1]} ({s[2]:.1f}%@0.7)"
    else: sepstr = "pending"
    accs=[]; partial=""
    for f in accfs:
        a = acc_of(f)
        if a is None: continue
        if a[2] >= 1260: accs.append(a[0])
        else: partial = f" [partial {a[1]}/{a[2]}→{a[0]:.1f}%]"
    if accs:
        m = sum(accs)/len(accs); sd = st.pstdev(accs) if len(accs)>1 else 0.0
        accstr = f"{m:.2f}% ±{sd:.2f} (n={len(accs)})" + partial
    elif partial:
        accstr = "running" + partial
    else:
        accstr = "pending"
    print(f"{label:32} | {sepstr:26} | {accstr:30}")
print("-"*95)
print("separation: held-out vmme45 (45 pairs, pair_id overlap with construction set = 0)")
print("acc: LVBench full 1267, tau=0.7 uniform; ± = pstdev over seeds")
