#!/usr/bin/env python3
"""Parse a reveal_27b.py output jsonl into rebuttal stats.

Reports, per run:
  - accuracy (correct / total)
  - avg/median retrieval-verify rounds (= len(trace_iters))
  - round distribution
  - avg LLM calls/question under the call model: calls = 2*rounds + 2
    (1 criterion-select [cached] + per round {1 planner + 1 verifier} + 1 answerer)
  - early-stop frequency: % questions whose FINAL verdict is SUFFICIENT
    (vs budget-exhausted fallback = final verdict INSUFFICIENT at K cap)
  - activated criteria/question: distinct criteria scored in iter0 (median/range)
Usage: parse_exp_stats.py <out.jsonl> [label]
"""
import json, sys, statistics as st

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path

rounds, calls, correct, total = [], [], 0, 0
suff_final, activated = 0, []
errs = 0
for line in open(path):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("record_type"):       # skip metadata header
        continue
    if "trace_iters" not in r:
        continue
    total += 1
    if r.get("error"):
        errs += 1
    if r.get("correct"):
        correct += 1
    t = r.get("trace_iters") or []
    nr = len(t)
    rounds.append(nr)
    calls.append(2 * nr + 2)
    if t:
        last = t[-1]
        if str(last.get("verdict", "")).upper() == "SUFFICIENT":
            suff_final += 1
        # activated criteria = distinct criterion names scored in iter0 (any option)
        sc = t[0].get("rubric_criteria_scores") or {}
        names = set()
        if isinstance(sc, dict):
            for opt, crits in sc.items():
                if isinstance(crits, dict):
                    names.update(crits.keys())
        if names:
            activated.append(len(names))

def fmt(xs):
    if not xs:
        return "n/a"
    return f"mean={st.mean(xs):.2f} median={st.median(xs):.0f} min={min(xs)} max={max(xs)}"

print(f"===== {label} =====")
print(f"N={total}  errors={errs}")
if total:
    print(f"Accuracy: {100*correct/total:.2f}%  ({correct}/{total})")
    print(f"Rounds/question:  {fmt(rounds)}")
    print(f"LLM calls/question (2r+2): {fmt(calls)}")
    print(f"Early-stop (final=SUFFICIENT): {100*suff_final/total:.1f}%  ({suff_final}/{total})")
    print(f"Activated criteria/question: {fmt(activated)}")
    from collections import Counter
    dist = Counter(rounds)
    print("Round distribution: " + ", ".join(f"{k}r:{dist[k]}" for k in sorted(dist)))
