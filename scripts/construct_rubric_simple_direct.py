#!/usr/bin/env python3
"""Multi-round rubric construction from pairs - direct model call version.
Bypasses vLLM server, calls transformers model directly for offline use.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generate_rubric.construct.contrastive_trace.io import read_jsonl
from src.generate_rubric.construct.contrastive_trace.rubric_io import load_rubric


def score_pairs_direct(pairs_path, chains_path, rubric_path, out_path, model_path):
    """Score pairs using direct model call."""
    pairs = list(read_jsonl(pairs_path))
    chains = {c["chain_id"]: c for c in read_jsonl(chains_path)}
    rubric = load_rubric(rubric_path)

    print(f"Loaded {len(pairs)} pairs, {len(chains)} chains")
    print(f"Rubric: {rubric_path}")
    print(f"Model: {model_path}")

    # TODO: Implement direct model scoring
    # For now, use existing score_pairs.py with vLLM requirement
    print("\nERROR: Direct model call not yet implemented.")
    print("Need to either:")
    print("1. Fix vLLM to accept local paths")
    print("2. Set up remote vLLM endpoint")
    print("3. Implement direct transformers.pipeline() scoring")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--model-path", default="/home2/models/Qwen3.5-27B")
    ap.add_argument("--max-rounds", type=int, default=5)
    args = ap.parse_args()

    pairs_dir = Path(args.pairs_dir)
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    pairs_path = pairs_dir / "pairs.jsonl"
    chains_path = pairs_dir / "chains.jsonl"

    if not pairs_path.exists() or not chains_path.exists():
        print(f"ERROR: Missing {pairs_path} or {chains_path}")
        return 1

    # Use existing rubric_pool as starting point
    initial_rubric = Path("configs/rubrics/rubric_pool.yaml")
    if not initial_rubric.exists():
        print(f"ERROR: Missing initial rubric {initial_rubric}")
        return 1

    print(f"=== Round 1: Initial scoring ===")
    round1_rubric = out_base / "round1_rubric.yaml"
    round1_scores = out_base / "round1_scored.jsonl"

    success = score_pairs_direct(
        pairs_path, chains_path, initial_rubric, round1_scores, args.model_path
    )

    if not success:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
