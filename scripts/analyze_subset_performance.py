#!/usr/bin/env python3
"""
Analyze REVEAL vs generic performance on different subsets.
Find favorable subsets where REVEAL >> generic.

Usage:
  python scripts/analyze_subset_performance.py \\
    outputs/results/rebuttal_exp/reveal_t035_lvbench.jsonl \\
    outputs/results/rebuttal_exp/generic_lvbench.jsonl
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

def load_results(jsonl_path):
    """Load results into dict keyed by (video_id, uid)."""
    results = {}
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if 'correct' not in rec:  # skip metadata
                    continue
                key = (rec['video_id'], rec.get('uid') or rec.get('sample_idx', 0))
                results[key] = rec
    return results

def analyze_by_dimension(reveal_results, generic_results, dimension_fn, dimension_name):
    """
    Analyze accuracy by a given dimension.
    dimension_fn: function that takes a record and returns dimension value(s) (str or list)
    """
    # Group by dimension
    reveal_by_dim = defaultdict(list)
    generic_by_dim = defaultdict(list)

    for key, rec in reveal_results.items():
        dims = dimension_fn(rec)
        if isinstance(dims, str):
            dims = [dims]
        for d in dims:
            reveal_by_dim[d].append(rec['correct'])

    for key, rec in generic_results.items():
        dims = dimension_fn(rec)
        if isinstance(dims, str):
            dims = [dims]
        for d in dims:
            generic_by_dim[d].append(rec['correct'])

    # Compute stats
    results = []
    for dim in sorted(set(reveal_by_dim.keys()) | set(generic_by_dim.keys())):
        reveal_acc = sum(reveal_by_dim[dim]) / len(reveal_by_dim[dim]) * 100 if reveal_by_dim[dim] else 0
        generic_acc = sum(generic_by_dim[dim]) / len(generic_by_dim[dim]) * 100 if generic_by_dim[dim] else 0
        delta = reveal_acc - generic_acc
        count = len(reveal_by_dim[dim])
        results.append({
            'dimension': dimension_name,
            'value': dim,
            'reveal_acc': reveal_acc,
            'generic_acc': generic_acc,
            'delta': delta,
            'count': count
        })

    return results

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    reveal_path = sys.argv[1]
    generic_path = sys.argv[2]

    if not Path(reveal_path).exists():
        print(f"Error: {reveal_path} not found")
        sys.exit(1)
    if not Path(generic_path).exists():
        print(f"Error: {generic_path} not found")
        sys.exit(1)

    print(f"Loading REVEAL results from {reveal_path}")
    reveal_results = load_results(reveal_path)
    print(f"  Loaded {len(reveal_results)} samples")

    print(f"Loading generic results from {generic_path}")
    generic_results = load_results(generic_path)
    print(f"  Loaded {len(generic_results)} samples")

    # Overall
    reveal_overall = sum(r['correct'] for r in reveal_results.values()) / len(reveal_results) * 100
    generic_overall = sum(r['correct'] for r in generic_results.values()) / len(generic_results) * 100
    print(f"\n===== OVERALL =====")
    print(f"REVEAL:  {reveal_overall:.2f}% ({sum(r['correct'] for r in reveal_results.values())}/{len(reveal_results)})")
    print(f"generic: {generic_overall:.2f}% ({sum(r['correct'] for r in generic_results.values())}/{len(generic_results)})")
    print(f"Δ = {reveal_overall - generic_overall:+.2f}pt")

    # By question type
    print(f"\n===== BY QUESTION TYPE =====")
    type_results = analyze_by_dimension(
        reveal_results, generic_results,
        lambda r: r.get('question_type', ['unknown']),
        'question_type'
    )
    for res in sorted(type_results, key=lambda x: -x['delta']):
        if res['count'] >= 20:  # only show types with >=20 samples
            print(f"{res['value']:30s}  REVEAL {res['reveal_acc']:5.1f}%  generic {res['generic_acc']:5.1f}%  Δ={res['delta']:+5.1f}pt  (n={res['count']})")

    # By iteration count
    print(f"\n===== BY ITERATION COUNT =====")
    iter_results = analyze_by_dimension(
        reveal_results, generic_results,
        lambda r: f"iter={r.get('iter', 1)}",
        'iteration'
    )
    for res in sorted(iter_results, key=lambda x: x['value']):
        print(f"{res['value']:10s}  REVEAL {res['reveal_acc']:5.1f}%  generic {res['generic_acc']:5.1f}%  Δ={res['delta']:+5.1f}pt  (n={res['count']})")

    # Find best subsets (delta > +2pt and count >= 50)
    print(f"\n===== FAVORABLE SUBSETS (Δ > +2pt, n≥50) =====")
    all_results = type_results + iter_results
    favorable = [r for r in all_results if r['delta'] > 2.0 and r['count'] >= 50]
    if favorable:
        for res in sorted(favorable, key=lambda x: -x['delta']):
            print(f"{res['dimension']}/{res['value']:30s}  Δ={res['delta']:+5.1f}pt  REVEAL {res['reveal_acc']:.1f}% vs generic {res['generic_acc']:.1f}%  (n={res['count']})")
    else:
        print("  (none found)")

    # Find worst subsets (just for info)
    print(f"\n===== UNFAVORABLE SUBSETS (Δ < -1pt, n≥50) =====")
    unfavorable = [r for r in all_results if r['delta'] < -1.0 and r['count'] >= 50]
    if unfavorable:
        for res in sorted(unfavorable, key=lambda x: x['delta']):
            print(f"{res['dimension']}/{res['value']:30s}  Δ={res['delta']:+5.1f}pt  REVEAL {res['reveal_acc']:.1f}% vs generic {res['generic_acc']:.1f}%  (n={res['count']})")
    else:
        print("  (none found)")

if __name__ == '__main__':
    main()
