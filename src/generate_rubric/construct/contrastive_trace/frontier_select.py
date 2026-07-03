"""Frontier selection (rounds ≥2): keep only pairs the current rubric can't separate.

For each pair, run the verifier (with the current rubric for that question type) on the
weaker chain and the stronger chain. The rubric "handles" the pair iff it judges
    weaker  → INSUFFICIENT   AND   stronger → SUFFICIENT.
Any other outcome ⇒ the pair is on the frontier and is emitted for refinement.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.verifier import Verifier
from .io import read_jsonl, write_jsonl
from .rubric_io import load_rubric, verify_rubric_dict


def _chain_texts(chain: dict) -> list[str]:
    return [e.get("text", "") for e in (chain.get("evidence") or [])]


def _is_sufficient(verifier, question, candidates, ev_texts, rubric_dict) -> bool:
    verdict = verifier.verify(question, candidates, ev_texts, rubric_dict)
    return str(verdict.get("label", "")).upper() == "SUFFICIENT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--rubric", required=True, help="current rubric yaml")
    ap.add_argument("--config", default="configs/veil.yaml")
    ap.add_argument("--llm-api-url", required=True, help="verifier LLM endpoint(s)")
    ap.add_argument("--llm-api-model", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/rubric/frontier_pairs.jsonl")
    args = ap.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)
    from experiments.tuning.veil_27b import load_samples
    samples = {s.sample_idx: s for s in load_samples(cfg)}

    chains = {c["chain_id"]: c for c in read_jsonl(args.chains)}
    pairs = list(read_jsonl(args.pairs))
    rubric = load_rubric(args.rubric)

    from src.clients.llm_client import LLMClient
    llm = LLMClient(model_path=args.llm_api_model or "Qwen3.5-27B",
                    api_url=args.llm_api_url, api_model=args.llm_api_model)
    verifier = Verifier(llm)

    def judge(pair: dict):
        s = samples.get(int(pair["sample_idx"]))
        w = chains.get(pair["weaker_chain_id"])
        st = chains.get(pair["stronger_chain_id"])
        if s is None or w is None or st is None:
            return None
        rd = verify_rubric_dict(rubric, getattr(s, "question_type", ""))
        weak_suff = _is_sufficient(verifier, s.question, s.candidates, _chain_texts(w), rd)
        strong_suff = _is_sufficient(verifier, s.question, s.candidates, _chain_texts(st), rd)
        separated = (not weak_suff) and strong_suff
        if separated:
            return None
        out = dict(pair)
        out["frontier_reason"] = f"weak_suff={weak_suff},strong_suff={strong_suff}"
        return out

    frontier = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(judge, p) for p in pairs]
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                frontier.append(r)

    write_jsonl(args.out, frontier)
    print(f"frontier: {len(frontier)}/{len(pairs)} pairs still unseparated → {args.out}")


if __name__ == "__main__":
    main()
