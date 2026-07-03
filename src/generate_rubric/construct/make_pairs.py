"""Pair evidence chains: one answered-wrong (insufficient) × one answered-right (sufficient).

Chains come from three sources (random / query / oracle) but pairing does not care —
it just needs, within a question, a wrong-evidence chain and a right-evidence chain.
To give RTD both coarse and fine contrasts, when a question yields more pairs than the
cap we keep the *tightest* pair first (smallest sufficient vs largest insufficient),
then fill with variety across sources/sizes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from src.generate_rubric.construct.io import read_jsonl, write_jsonl
from src.generate_rubric.construct.schema import ChainPair


def _is_sufficient(c: dict) -> bool:
    if c.get("sufficient") is not None:
        return bool(c["sufficient"])
    return c.get("quality") == "sufficient"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", required=True)
    ap.add_argument("--out", default="outputs/rubric/pairs.jsonl")
    ap.add_argument("--max-pairs-per-question", type=int, default=3)
    args = ap.parse_args()

    by_idx: dict[int, list[dict]] = defaultdict(list)
    for c in read_jsonl(args.chains):
        by_idx[int(c["sample_idx"])].append(c)

    pairs = []
    skipped = 0
    for idx, chains in by_idx.items():
        suff = [c for c in chains if _is_sufficient(c)]
        insuff = [c for c in chains if not _is_sufficient(c)]
        if not suff or not insuff:
            skipped += 1
            continue

        # size for tightness = number of evidence chunks
        def nchunks(c):
            return c.get("n_chunks") or len(c.get("evidence") or [])

        min_suff = min(suff, key=nchunks)          # smallest chain that still answers right
        max_insuff = max(insuff, key=nchunks)      # largest chain that still answers wrong

        ordered: list[tuple[dict, dict]] = [(max_insuff, min_suff)]  # tightest first
        # then variety: each other insufficient against min_suff, spread by source
        for w in insuff:
            if w is max_insuff:
                continue
            ordered.append((w, min_suff))
        # and min_insuff against other sufficient chains for cross-source contrast
        for st in suff:
            if st is min_suff:
                continue
            ordered.append((max_insuff, st))

        seen = set()
        for weaker, stronger in ordered:
            if len(seen) >= args.max_pairs_per_question:
                break
            key = (weaker["chain_id"], stronger["chain_id"])
            if key in seen:
                continue
            seen.add(key)
            pair_id = f"{idx}:{weaker['chain_id']}->{stronger['chain_id']}"
            pairs.append(ChainPair(
                sample_idx=idx,
                video_id=weaker["video_id"],
                question_type=weaker.get("question_type", ""),
                pair_id=pair_id,
                weaker_chain_id=weaker["chain_id"],
                stronger_chain_id=stronger["chain_id"],
                pair_type="insufficient_vs_sufficient",
            ).to_dict())

    write_jsonl(args.out, pairs)
    print(f"wrote {len(pairs)} pairs to {args.out} ({skipped} questions skipped: no wrong/right split)")


if __name__ == "__main__":
    main()
