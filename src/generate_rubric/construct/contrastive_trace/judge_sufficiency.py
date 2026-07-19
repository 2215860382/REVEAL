"""Re-label each evidence chain's sufficiency by GOLD-AWARE ENTAILMENT, replacing the
answerer-vote label from sample_chains.

Why: `sample_chains._label_chain` labels a chain sufficient iff a CLOSED-BOOK answerer
(which guesses / uses priors on a 4-way MCQ) majority-answers it correctly. That label is
anti-correlated with evidence content — a thin k=2 chain that the model luckily guessed gets
`sufficient`, while a rich k=16 superset it fluffed gets `insufficient`. Pairs built from
those labels are ~73% evidence-quantity-inverted and unseparable by any sufficiency rubric.

Fix: judge OPEN-BOOK. Show the judge the question, options, the CORRECT answer, and the
evidence, and ask ONE thing — does the evidence TEXT ITSELF contain the specific facts needed
to VERIFY the correct option and/or EXCLUDE the competitors, forcing the correct answer
without world knowledge or guessing. Because the gold is given, the judge never guesses, so
the label tracks evidence content and is monotone in evidence (k16 ⊇ k2 can't invert).

Construction-time only (sees gold); not used at inference. Text-only by default (captions are
rich); `--images` adds the same keyframes the sampler used.

Out: <out> jsonl = input chains + {sufficient, quality, judge_sufficient, judge_reason};
     original `sufficient`/`quality`/`sufficient_votes` preserved as `*_answerer` for diff.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.utils.extract_json import extract_json
from .io import read_jsonl, write_jsonl, dump_json, load_json
from .llm import LLMRouter

JUDGE_SYSTEM = """\
You judge whether a set of retrieved video evidence is SUFFICIENT to answer a multiple-choice
question — NOT whether you can guess the answer.

You are given the CORRECT answer. Do NOT try to answer the question. Your only job is to decide,
by reading the EVIDENCE TEXT ALONE, whether that evidence contains the specific facts a careful
reader would NEED to arrive at the correct option with certainty.

Call it SUFFICIENT only if the evidence text itself lets you either:
  (a) VERIFY the correct option — the evidence states / shows the fact the correct option asserts; OR
  (b) EXCLUDE every competing option — the evidence contains the facts that rule the others out,
      leaving the correct one as the only possibility.
A reader restricted to this evidence must be FORCED to the correct answer, not merely nudged.

Call it INSUFFICIENT if the discriminating fact is ABSENT from the evidence — even when the
correct answer is "obvious" from background/world knowledge. World knowledge, priors, and lucky
guesses do NOT count. Partial evidence that narrows but does not decide is INSUFFICIENT.

Judge only what the evidence says. More evidence is never less sufficient than a subset of it.

Return ONE strict JSON object, no prose, no fences:
{"sufficient": true|false, "reason": "<=25 words: which fact makes it (in)sufficient"}
"""

JUDGE_USER = """\
QUESTION:
{question}

OPTIONS:
{options}

CORRECT ANSWER: {gold}

EVIDENCE ({n} item(s)) — judge ONLY this text:
{evidence}

Does this evidence TEXT ALONE contain the facts needed to verify the correct option or exclude
the others (world knowledge does not count)? Return {{"sufficient": ..., "reason": "..."}}.
"""


def _fmt_options(cands: list[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(cands))


def _fmt_evidence(texts: list[str]) -> str:
    return "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts) if t)


def _judge_one(llm, sample, chain, max_tokens):
    ev_texts = [e.get("text", "") for e in (chain.get("evidence") or [])]
    user = JUDGE_USER.format(
        question=sample.question, options=_fmt_options(sample.candidates),
        gold=_gold_text(sample), n=len(ev_texts), evidence=_fmt_evidence(ev_texts),
    )
    try:
        raw = llm.chat(JUDGE_SYSTEM, user, max_tokens=max_tokens)
        obj = extract_json(raw) or {}
        suff = bool(obj.get("sufficient"))
        reason = str(obj.get("reason") or "")[:200]
        return chain["chain_id"], suff, reason
    except Exception as exc:
        print(f"[judge] {chain['chain_id']} failed: {exc}")
        return chain["chain_id"], None, "ERROR"


def _gold_text(sample) -> str:
    return getattr(sample, "answer", "") or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/videomme.yaml")
    ap.add_argument("--llm-api-url", required=True)
    ap.add_argument("--llm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--sample-limit", type=int, default=None,
                    help="only judge chains of the first N sample_idx (seed/debug)")
    ap.add_argument("--only-samples", default=None,
                    help="comma list of sample_idx to judge (seed validation)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)
    from experiments.tuning.reveal_27b import load_samples
    samples = {s.sample_idx: s for s in load_samples(cfg)}

    chains = list(read_jsonl(args.chains))
    if args.only_samples:
        keep = {int(x) for x in args.only_samples.split(",") if x.strip()}
        chains = [c for c in chains if int(c["sample_idx"]) in keep]
    elif args.sample_limit:
        keep = sorted({int(c["sample_idx"]) for c in chains})[:args.sample_limit]
        keep = set(keep)
        chains = [c for c in chains if int(c["sample_idx"]) in keep]

    state_path = Path(str(args.out) + ".state.json")
    done: dict[str, list] = {}
    if args.resume and state_path.is_file():
        done = load_json(state_path).get("done", {})
        print(f"[judge] resume: {len(done)} already judged")

    llm = LLMRouter.from_urls(args.llm_api_url, model=args.llm_api_model)
    todo = [c for c in chains if c["chain_id"] not in done]
    print(f"[judge] {len(todo)} chains to judge ({len(done)} cached) | {args.workers}-way")

    lock = threading.Lock()
    ct = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_judge_one, llm, samples[int(c["sample_idx"])], c, args.max_tokens)
                for c in todo if int(c["sample_idx"]) in samples]
        for f in as_completed(futs):
            cid, suff, reason = f.result()
            if suff is None:
                continue
            with lock:
                done[cid] = [suff, reason]
                ct += 1
                if ct % 200 == 0:
                    print(f"[judge] {ct}/{len(todo)}")
                    dump_json(state_path, {"done": done})

    # emit relabeled chains (preserve answerer label for diff)
    out_rows, flipped = [], 0
    for c in chains:
        res = done.get(c["chain_id"])
        if res is None:
            continue
        suff, reason = res
        old = bool(c.get("sufficient")) if c.get("sufficient") is not None else (c.get("quality") == "sufficient")
        if bool(suff) != old:
            flipped += 1
        c = dict(c)
        c["sufficient_answerer"] = c.get("sufficient")
        c["quality_answerer"] = c.get("quality")
        c["sufficient_votes_answerer"] = c.get("sufficient_votes")
        c["sufficient"] = bool(suff)
        c["quality"] = "sufficient" if suff else "insufficient"
        c["judge_sufficient"] = bool(suff)
        c["judge_reason"] = reason
        out_rows.append(c)

    write_jsonl(args.out, out_rows)
    dump_json(state_path, {"done": done})
    n = len(out_rows) or 1
    nsuff = sum(1 for c in out_rows if c["sufficient"])
    print(f"[judge] wrote {len(out_rows)} chains -> {args.out}")
    print(f"[judge] sufficient={nsuff} ({nsuff/n*100:.1f}%)  insufficient={n-nsuff}  "
          f"flipped-vs-answerer={flipped} ({flipped/n*100:.1f}%)")


if __name__ == "__main__":
    main()
