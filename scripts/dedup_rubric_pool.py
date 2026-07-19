#!/usr/bin/env python
"""Conservative near-duplicate merge of the 310-criterion pool.

Embeds each criterion (name + description) with BGE-M3, greedily clusters at a
HIGH cosine threshold so only genuine near-duplicates collapse; distinct-but-
related criteria stay separate (meaning preserved). Keeps one representative per
cluster (highest weight, tie -> longest description). Prints every merge group so
the reduction is auditable, and writes the deduped pool.

Usage:
  python scripts/dedup_rubric_pool.py --in outputs/rubric/rubric_pool.yaml \
     --out outputs/rubric/rubric_pool_dedup.yaml \
     --emb http://127.0.0.1:9000 --thr 0.93
"""
import argparse, yaml, numpy as np
from src.clients.embedder import BGEM3Embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/rubric/rubric_pool.yaml")
    ap.add_argument("--out", default="outputs/rubric/rubric_pool_dedup.yaml")
    ap.add_argument("--emb", default="http://127.0.0.1:9000")
    ap.add_argument("--thr", type=float, default=0.93)
    ap.add_argument("--write", action="store_true", help="write output (else dry-run preview)")
    args = ap.parse_args()

    d = yaml.safe_load(open(args.inp))
    crits = d["general"]["rubric_criteria"]
    texts = [f"{c.get('name','')}. {c.get('description','')}" for c in crits]
    print(f"pool: {len(crits)} criteria; embedding via {args.emb} ...")

    emb = BGEM3Embedder(api_url=args.emb)
    V = np.asarray(emb.encode(texts), dtype=np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    S = V @ V.T

    # show reduction at several thresholds so the choice is informed
    for t in (0.90, 0.92, 0.93, 0.95, 0.97):
        n = _cluster_count(S, t)
        print(f"  thr={t:.2f} -> {n} clusters (merges {len(crits)-n})")

    labels = _greedy_clusters(S, args.thr)
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)

    def rep(idxs):  # representative: highest weight, tie -> longest description
        return max(idxs, key=lambda i: (float(crits[i].get("weight", 1.0)),
                                        len(crits[i].get("description", ""))))

    kept, merged_groups = [], []
    for _, idxs in sorted(groups.items()):
        r = rep(idxs)
        kept.append(crits[r])
        if len(idxs) > 1:
            merged_groups.append((crits[r]["name"], [crits[i]["name"] for i in idxs if i != r]))

    print(f"\n=== thr={args.thr}: {len(crits)} -> {len(kept)} criteria "
          f"({len(crits)-len(kept)} merged away) ===")
    print(f"merge groups ({len(merged_groups)}):")
    for keep, dropped in merged_groups:
        print(f"  KEEP {keep}\n       <- {', '.join(dropped)}")

    if args.write:
        out = {"types": [], "general": {
            "rubric_criteria": kept, "scoring_rule": "average",
            "sufficient_threshold": d["general"].get("sufficient_threshold", 0.6)}}
        with open(args.out, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True, width=1000)
        print(f"\nwrote {args.out}: {len(kept)} criteria")
    else:
        print("\n(dry-run; pass --write to save)")


def _greedy_clusters(S, thr):
    n = S.shape[0]
    labels = [-1] * n
    nxt = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = nxt
        for j in range(i + 1, n):
            if labels[j] == -1 and S[i, j] >= thr:
                labels[j] = nxt
        nxt += 1
    return labels


def _cluster_count(S, thr):
    return len(set(_greedy_clusters(S, thr)))


if __name__ == "__main__":
    main()
