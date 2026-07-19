#!/usr/bin/env python3
"""Reframe a sufficiency rubric from COVERAGE anchors → DISCRIMINABILITY anchors.

Problem: criteria score anchors reward "evidence explicitly shows/states X" (mere
coverage), so topically-rich evidence from the WRONG moment scores high → rubric_score
decouples from answerability. The verifier prompt already asks for per-option decidability;
the anchors undercut it.

Fix (surgical): keep every criterion's name / weight / relevance structure untouched so
REVEAL_VERIFIER_RELEVANCE gating and denominators are byte-identical, but rewrite each
criterion's description + score_1/score_half/score_0 so they measure whether the evidence
pins the dimension precisely enough to CONFIRM or RULE OUT the option(s) that hinge on it —
generic/mismatched/wrong-moment values do NOT count as sufficient.

Output is a SEPARATE file so it can be A/B'd against the original via REVEAL_RUBRIC_PATH.
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clients.llm_client import LLMClient  # noqa: E402

REFRAME_SYS = """\
You rewrite one evidence-sufficiency criterion used to judge whether retrieved video
evidence is enough to answer a multiple-choice question.

The criterion currently rewards COVERAGE: it gives a high score whenever the evidence
merely *describes / shows / states* the dimension — regardless of whether that value is
the RIGHT one for the question. This makes topically-rich but wrong-moment evidence score
high. Rewrite it to reward DISCRIMINABILITY instead.

Keep the SAME dimension (do not change what the criterion is about). Rewrite the
description and the three score anchors so they measure whether the evidence pins this
dimension precisely enough to CONFIRM or RULE OUT the answer option(s) that hinge on it:
  - score_1  = the evidence establishes this dimension specifically and unambiguously
               enough to confirm OR exclude the option(s) that turn on it. Correctly
               ruling an option out counts as fully sufficient.
  - score_half = the evidence is partially decisive — narrows the option(s) but leaves a
               real ambiguity (wrong granularity, plausible alternatives not excluded).
  - score_0  = the evidence describes this dimension only generically, for the wrong
               moment/entity, or in a way that does not let you confirm or exclude any
               option — mere presence of the dimension is NOT sufficient.

Do NOT mention specific option letters or a specific question — stay general across all
questions of this type. Keep each field to 1-2 sentences.

Return ONE strict JSON object, no prose, no fences:
{"description": "...", "score_1": "...", "score_half": "...", "score_0": "..."}"""


def _reframe_one(llm, crit: dict) -> dict:
    user = "\n".join([
        f"Criterion name: {crit.get('name','')}",
        f"Current description: {crit.get('description','')}",
        f"Current score_1: {crit.get('score_1','')}",
        f"Current score_half: {crit.get('score_half','')}",
        f"Current score_0: {crit.get('score_0','')}",
        "",
        "Rewrite it now. Return the JSON.",
    ])
    raw = llm.chat(
        [{"role": "system", "content": REFRAME_SYS},
         {"role": "user", "content": user}],
        max_new_tokens=400, enable_thinking=False)
    # tolerant JSON extract
    s = raw.find("{"); e = raw.rfind("}")
    obj = json.loads(raw[s:e + 1]) if s >= 0 and e > s else {}
    out = deepcopy(crit)
    for k in ("description", "score_1", "score_half", "score_0"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/rubric/rubric.yaml")
    ap.add_argument("--out", default="outputs/rubric/egolife_tt/rubric_discriminative.yaml")
    ap.add_argument("--api-url", required=True)
    ap.add_argument("--api-model", default="Qwen3.5-27B")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    d = yaml.safe_load(open(args.inp))
    llm = LLMClient(api_url=args.api_url, api_model=args.api_model)

    # collect every criterion across general + all type nodes
    jobs = []  # (node_dict, index)
    for key, node in d.items():
        if isinstance(node, dict) and node.get("rubric_criteria"):
            for i in range(len(node["rubric_criteria"])):
                jobs.append((node["rubric_criteria"], i))
    print(f"reframing {len(jobs)} criteria from {args.inp}", flush=True)

    lock = threading.Lock()
    done = [0]

    def work(job):
        crit_list, i = job
        try:
            new = _reframe_one(llm, crit_list[i])
            crit_list[i] = new
        except Exception as ex:
            with lock:
                print(f"  keep-original (err): {crit_list[i].get('name')}: {ex}", flush=True)
        with lock:
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"  {done[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))

    with open(args.out, "w") as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
