"""Diagnostics for the online rubric-construction pipeline.

Reports, per question type: #chains, #pairs, #criteria in the final rubric, and — by
running the verifier with the final rubric on every pair — the SEPARATION SUCCESS RATE
(fraction of pairs judged wrong→insufficient AND right→sufficient). Flags weak types.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.verifier import Verifier
from src.generate_rubric.construct.io import dump_json, read_jsonl
from src.generate_rubric.construct.rubric_io import load_rubric, verify_rubric_dict


def _texts(chain: dict) -> list[str]:
    return [e.get("text", "") for e in (chain.get("evidence") or [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--config", default="configs/veil.yaml")
    ap.add_argument("--llm-api-url", default=None, help="if set, compute separation success rate")
    ap.add_argument("--llm-api-model", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/rubric/backtest_report.json")
    args = ap.parse_args()

    chains = list(read_jsonl(args.chains))
    chain_by_id = {c["chain_id"]: c for c in chains}
    pairs = list(read_jsonl(args.pairs))
    rubric = load_rubric(args.rubric)

    n_crit_by_type = {qt: len((tpl.get("rubric_criteria") or []))
                      for qt, tpl in (rubric.get("templates") or {}).items()}

    report = {
        "n_chains": len(chains),
        "n_pairs": len(pairs),
        "chain_label_counts": dict(Counter(c.get("quality", "") for c in chains)),
        "chain_source_counts": dict(Counter((c.get("notes") or "").replace("source=", "")
                                             for c in chains)),
        "pairs_by_type": dict(Counter(p.get("question_type", "") for p in pairs)),
        "n_general_criteria": len((rubric.get("general") or {}).get("rubric_criteria") or []),
        "criteria_by_type": n_crit_by_type,
        "separation_success_rate": None,
        "separation_by_type": {},
        "needs_more_work": [],
    }

    if args.llm_api_url:
        from src.config import load_config
        from experiments.tuning.veil_27b import load_samples
        samples = {s.sample_idx: s for s in load_samples(load_config(args.config))}
        from src.clients.llm_client import LLMClient
        llm = LLMClient(model_path=args.llm_api_model or "Qwen3.5-27B",
                        api_url=args.llm_api_url, api_model=args.llm_api_model)
        verifier = Verifier(llm)

        def sep(pair):
            s = samples.get(int(pair["sample_idx"]))
            w = chain_by_id.get(pair["weaker_chain_id"])
            st = chain_by_id.get(pair["stronger_chain_id"])
            if s is None or w is None or st is None:
                return None
            rd = verify_rubric_dict(rubric, getattr(s, "question_type", ""))
            wl = verifier.verify(s.question, s.candidates, _texts(w), rd)["label"].upper()
            sl = verifier.verify(s.question, s.candidates, _texts(st), rd)["label"].upper()
            ok = (wl == "INSUFFICIENT") and (sl == "SUFFICIENT")
            return (pair.get("question_type", ""), ok)

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(sep, p) for p in pairs]):
                r = f.result()
                if r is not None:
                    results.append(r)
        by_type_ok = defaultdict(lambda: [0, 0])
        for qt, ok in results:
            by_type_ok[qt][1] += 1
            by_type_ok[qt][0] += int(ok)
        total_ok = sum(v[0] for v in by_type_ok.values())
        total_n = sum(v[1] for v in by_type_ok.values()) or 1
        report["separation_success_rate"] = round(total_ok / total_n, 4)
        report["separation_by_type"] = {qt: {"ok": ok, "n": n, "rate": round(ok / n, 4) if n else 0.0}
                                        for qt, (ok, n) in by_type_ok.items()}

    for qt, n_pairs in report["pairs_by_type"].items():
        rate = report["separation_by_type"].get(qt, {}).get("rate", 1.0)
        if n_pairs < 5 or n_crit_by_type.get(qt, 0) < 2 or rate < 0.6:
            report["needs_more_work"].append({
                "question_type": qt, "pairs": n_pairs,
                "criteria": n_crit_by_type.get(qt, 0),
                "separation_rate": report["separation_by_type"].get(qt, {}).get("rate"),
            })

    dump_json(args.out, report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
