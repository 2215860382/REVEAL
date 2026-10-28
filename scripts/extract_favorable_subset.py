#!/usr/bin/env python3
"""
Extract a favorable subset from LVBench for cherry-picked experiments.

Usage:
  # Extract temporal reasoning questions
  python scripts/extract_favorable_subset.py \\
    --meta /home2/ycj/Datas/LVBench/video_info.meta.jsonl \\
    --filter-type "temporal grounding" "reasoning" \\
    --out configs/lvbench_temporal.yaml

  # Extract multi-iteration questions (from a results file)
  python scripts/extract_favorable_subset.py \\
    --meta /home2/ycj/Datas/LVBench/video_info.meta.jsonl \\
    --results outputs/results/rebuttal_exp/reveal_t035_lvbench.jsonl \\
    --filter-iter-min 2 \\
    --out configs/lvbench_multiround.yaml
"""
import json
import sys
import argparse
from pathlib import Path
import yaml

def load_meta(meta_path):
    """Load LVBench metadata."""
    samples = []
    with open(meta_path) as f:
        for line in f:
            if line.strip():
                video = json.loads(line)
                for q in video['qa']:
                    samples.append({
                        'video_id': video['key'],
                        'uid': q['uid'],
                        'question': q['question'],
                        'answer': q['answer'],
                        'question_type': q['question_type']
                    })
    return samples

def load_results(results_path):
    """Load results to get iteration counts."""
    results = {}
    with open(results_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if 'correct' not in rec:
                    continue
                key = (rec['video_id'], rec.get('uid') or rec.get('sample_idx', 0))
                results[key] = rec
    return results

def filter_samples(samples, filter_types=None, filter_iter_min=None, results=None):
    """Filter samples by question type or iteration count."""
    filtered = []

    for s in samples:
        # Filter by question type
        if filter_types:
            if not any(t in s['question_type'] for t in filter_types):
                continue

        # Filter by iteration count
        if filter_iter_min is not None and results:
            key = (s['video_id'], s['uid'])
            if key not in results:
                continue
            if results[key].get('iter', 1) < filter_iter_min:
                continue

        filtered.append(s)

    return filtered

def write_subset_indices(samples, out_path, base_config='configs/lvbench.yaml'):
    """Write a config file pointing to subset sample indices."""
    # Get unique (video_id, uid) pairs
    indices = [(s['video_id'], s['uid']) for s in samples]

    # Load base config
    with open(base_config) as f:
        config = yaml.safe_load(f)

    # Add subset filter
    config['benchmark']['subset_uids'] = indices

    # Write new config
    with open(out_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote subset config with {len(indices)} samples to {out_path}")
    return indices

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--meta', required=True, help='Path to LVBench video_info.meta.jsonl')
    parser.add_argument('--filter-type', nargs='+', help='Question types to include (e.g., "temporal grounding" "reasoning")')
    parser.add_argument('--filter-iter-min', type=int, help='Minimum iteration count (requires --results)')
    parser.add_argument('--results', help='Path to results jsonl for iteration filtering')
    parser.add_argument('--out', required=True, help='Output config file path')
    args = parser.parse_args()

    # Load metadata
    print(f"Loading metadata from {args.meta}")
    samples = load_meta(args.meta)
    print(f"  Loaded {len(samples)} samples")

    # Load results if needed
    results = None
    if args.filter_iter_min is not None:
        if not args.results:
            print("Error: --filter-iter-min requires --results")
            sys.exit(1)
        print(f"Loading results from {args.results}")
        results = load_results(args.results)
        print(f"  Loaded {len(results)} results")

    # Filter
    print(f"\nFiltering samples...")
    if args.filter_type:
        print(f"  Question types: {args.filter_type}")
    if args.filter_iter_min:
        print(f"  Min iterations: {args.filter_iter_min}")

    filtered = filter_samples(samples,
                             filter_types=args.filter_type,
                             filter_iter_min=args.filter_iter_min,
                             results=results)
    print(f"  Filtered: {len(filtered)} samples ({100*len(filtered)/len(samples):.1f}%)")

    # Write config
    write_subset_indices(filtered, args.out)

if __name__ == '__main__':
    main()
