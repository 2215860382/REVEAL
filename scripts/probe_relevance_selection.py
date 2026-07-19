#!/usr/bin/env python
"""Probe: does Pass-1 relevance selection pick sensible criteria from the 310-pool?

Loads the merged pool rubric, samples cross-dataset / cross-type questions from
existing result jsonls, runs Verifier._relevant_criteria (the evidence-independent
Pass-1) on each, and prints what it selected. Pure LLM — no embed/bank needed.

Usage:
  REVEAL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml \
  python scripts/probe_relevance_selection.py \
     --ep http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004 \
     --per-type 3 --cap 60
"""
import os, json, argparse, collections, random
os.environ.setdefault("REVEAL_RUBRIC_PATH", "outputs/rubric/rubric_pool.yaml")
for v in ("http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
    os.environ.pop(v, None)
os.environ["NO_PROXY"] = "127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]

from src.clients.llm_client import LLMClient
from src.agents.verifier import Verifier, get_rubric_dict

SOURCES = {
    "lvbench":  "outputs/results/lvbench/ablation/lvb_abl_iter_rubric.jsonl",
    "videomme": "outputs/results/videomme/ablation/refix_vmme_ours.jsonl",
    "egolife":  "outputs/results/egolife/egolife_peropt_thr07.jsonl",
}


def load_samples(path, per_type, seed=0):
    if not os.path.exists(path):
        return []
    recs = []
    for line in open(path):
        d = json.loads(line)
        if d.get("record_type") == "header":
            continue
        q = d.get("question"); cand = d.get("candidates"); qt = d.get("question_type")
        if q and cand:
            recs.append({"q": q, "cand": cand, "type": qt or "?"})
    by = collections.defaultdict(list)
    for r in recs:
        by[r["type"]].append(r)
    rng = random.Random(seed)
    out = []
    for t, rs in by.items():
        rng.shuffle(rs)
        out.extend(rs[:per_type])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True)
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--cap", type=int, default=60)
    args = ap.parse_args()

    llm = LLMClient(model_path="Qwen3.5-27B", api_url=args.ep, api_model="Qwen3.5-27B")
    v = Verifier(llm)
    pool = get_rubric_dict("x", None)
    n_pool = len(pool.get("rubric_criteria") or [])
    print(f"pool size = {n_pool} criteria\n")

    counts = []
    fam_hits = collections.Counter()
    for ds, path in SOURCES.items():
        samples = load_samples(path, args.per_type)[: args.cap]
        print(f"\n{'='*90}\n### {ds}  ({len(samples)} sampled from {os.path.basename(path)})")
        for s in samples:
            opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(s["cand"]))
            sel = v._relevant_criteria(s["q"], opts, pool)
            sel = sorted(sel)
            counts.append(len(sel))
            for n in sel:
                fam_hits[n] += 1
            qshort = (s["q"][:72] + "…") if len(s["q"]) > 72 else s["q"]
            print(f"\n  [{s['type']}] {qshort}")
            print(f"    -> {len(sel)} selected: {', '.join(sel)}")

    print(f"\n{'='*90}\n=== SUMMARY ===")
    if counts:
        import statistics
        print(f"n_selected: min={min(counts)} max={max(counts)} "
              f"mean={statistics.mean(counts):.1f} median={statistics.median(counts)}")
        print(f"selections in [3,8] target: {sum(1 for c in counts if 3<=c<=8)}/{len(counts)}")
        print(f"distinct criteria ever selected: {len(fam_hits)} / {n_pool}")
        print("top-15 most-selected criteria:")
        for n, c in fam_hits.most_common(15):
            print(f"   {c:>3}  {n}")


if __name__ == "__main__":
    main()
