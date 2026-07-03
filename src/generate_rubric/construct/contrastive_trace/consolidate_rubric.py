"""Consolidate an over-fragmented RTD rubric.

round1 RTD split one mechanism into per-topic twins (Object Reasoning grew to 108
criteria). This collapses them back, in the order the user specified:

  Step 1 (per type):   merge same-MECHANISM duplicates within each type  → <=30 / type
  Step 2a (global):    from all distilled types, induce the shared GENERAL library
                       (mechanisms that recur across MULTIPLE types)     → ~10-30
  Step 2b (per type):  drop each type criterion whose mechanism now lives in general
                       → each type keeps only its type-specific residue

Writes <out> (verifier-compatible: general + type_aliases + templates). Run backtest
separately afterwards.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.extract_json import extract_json
from .llm import LLMRouter
from .rubric_io import (
    load_rubric, save_rubric, inject_groundedness, REPAIR_ACTIONS, DEFAULT_THRESHOLD, as_whether,
)

_FIELDS = ("name", "description", "score_1", "score_half", "score_0")


def _clean(c: dict) -> dict | None:
    if not isinstance(c, dict):
        return None
    name = str(c.get("name") or "").strip()
    if not name:
        return None
    out = {k: str(c.get(k) or "").strip() for k in _FIELDS}
    out["description"] = as_whether(out["description"])
    try:
        out["weight"] = float(c.get("weight", 3.0))
    except (TypeError, ValueError):
        out["weight"] = 3.0
    ra = str(c.get("failure_repair_action") or c.get("repair_action") or "refine_query").strip()
    out["failure_repair_action"] = ra if ra in REPAIR_ACTIONS else "refine_query"
    return out


def _compact(crits: list[dict]) -> str:
    """name — description, one per line (for prompts where scores add noise)."""
    return "\n".join(f"- {c.get('name')}: {c.get('description','')}" for c in crits)


# ── Step 1: per-type mechanism merge ──────────────────────────────────────────
STEP1_SYSTEM = """\
You consolidate an OVER-FRAGMENTED evidence-sufficiency rubric for ONE question type of
long-video multiple-choice QA.

A criterion scores whether the current retrieved evidence is sufficient to VERIFY OR EXCLUDE
each option (NOT answer correctness).

The list you receive was grown one pair at a time and is badly duplicated: the SAME abstract
evidence MECHANISM was cloned once per topic. Example — `causal_attribution_explicitness`,
`business_model_shift_explicitness`, `team_affiliation_explicitness` are all ONE mechanism:
"the evidence explicitly states the proposition an option asserts". They must become ONE criterion.

A "mechanism" is an abstract, topic-INDEPENDENT property of the evidence (e.g. proposition is
explicitly stated vs. merely implied; the target moment/interval is temporally located; the
entity is visually resolved, not just named; a comparison/superlative is grounded across the
compared items; a set is enumerated completely; negation/absence is confirmed). The SUBJECT
MATTER (causal, business, sports, art…) is NOT a mechanism.

Task: merge every group of criteria that test the SAME mechanism into ONE criterion, phrased
abstractly and type-generally (NO topic words, NO specific entities). Keep genuinely distinct
mechanisms separate. Preserve the strongest scoring language when merging.

Output AT MOST 30 criteria; fewer is better. Return ONE strict JSON object, no prose, no fences.
"""

STEP1_USER = """\
Question type: {qtype}
Current criteria ({n}):
{criteria_json}

Merge same-mechanism duplicates into <=30 abstract, type-general criteria. Return JSON:
{{"criteria": [
  {{"name": "snake_case", "description": "abstract mechanism (no topic words)",
    "score_1": "...", "score_half": "...", "score_0": "...",
    "weight": 3, "failure_repair_action": "refine_query"}}
]}}
"""


# ── Step 2a: induce the shared general library ────────────────────────────────
STEP2A_SYSTEM = """\
You build the GENERAL (shared) layer of an evidence-sufficiency rubric for long-video
multiple-choice QA. You are given the already-consolidated criteria of EVERY question type.

Identify the abstract evidence MECHANISMS that RECUR ACROSS MULTIPLE question types — these are
the shared, reusable sufficiency dimensions that belong in the general library (they apply to any
question regardless of type). Synthesize each recurring mechanism into ONE general criterion,
phrased so it holds for any question type.

Do NOT include a mechanism that is specific to a single type (leave those with their type).
Output 10-30 general criteria. Return ONE strict JSON object, no prose, no fences.
"""

STEP2A_USER = """\
Consolidated criteria per question type:
{by_type}

Emit the shared general library: mechanisms recurring across MULTIPLE types, each ONE abstract
criterion. Return JSON:
{{"general": [
  {{"name": "snake_case", "description": "cross-type sufficiency mechanism",
    "score_1": "...", "score_half": "...", "score_0": "...",
    "weight": 3, "failure_repair_action": "refine_query"}}
]}}
"""


# ── Step 2b: subtract general from each type ──────────────────────────────────
STEP2B_SYSTEM = """\
You prune one question type's evidence-sufficiency criteria against the finalized GENERAL library.

Drop any type criterion whose MECHANISM is already covered by a general criterion (same abstract
mechanism, even if worded for a narrower case). Keep ONLY criteria whose mechanism is genuinely
SPECIFIC to this question type and absent from general. The result may be short or even empty.

Return ONE strict JSON object, no prose, no fences.
"""

STEP2B_USER = """\
GENERAL library (already covers these mechanisms — do not duplicate):
{general}

Question type: {qtype}
This type's criteria:
{criteria_json}

Keep only the type-SPECIFIC ones (mechanism NOT in general). Return JSON:
{{"criteria": [ {{"name": "...", "description": "...", "score_1": "...", "score_half": "...",
   "score_0": "...", "weight": 3, "failure_repair_action": "refine_query"}} ]}}
"""


def _call(llm, system, user, max_tokens, key):
    raw = llm.chat(system, user, max_tokens=max_tokens)
    obj = extract_json(raw) or {}
    out = [x for x in (_clean(c) for c in (obj.get(key) or [])) if x]
    # de-dup by name, preserve order
    seen, uniq = set(), []
    for c in out:
        if c["name"] not in seen:
            seen.add(c["name"]); uniq.append(c)
    return uniq


def _tpl(rubric, qtype):
    return (rubric.get("templates") or {}).get(qtype) or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric-in", required=True)
    ap.add_argument("--rubric-out", required=True)
    ap.add_argument("--llm-api-url", required=True)
    ap.add_argument("--llm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dump-stages", action="store_true",
                    help="also write <out>.step1.yaml after per-type merge")
    args = ap.parse_args()

    rubric = load_rubric(args.rubric_in)
    llm = LLMRouter.from_urls(args.llm_api_url, model=args.llm_api_model)
    templates = rubric.get("templates") or {}
    types = list(templates.keys())

    # ── Step 1: per-type mechanism merge (parallel across types) ──
    print(f"[step1] merging duplicates within {len(types)} types ...")
    distilled: dict[str, list[dict]] = {}

    def _step1(qt):
        crits = _tpl(rubric, qt).get("rubric_criteria") or []
        if len(crits) <= 8:            # already lean → keep as-is
            return qt, [c for c in (_clean(x) for x in crits) if c]
        user = STEP1_USER.format(qtype=qt, n=len(crits),
                                 criteria_json=json.dumps(crits, ensure_ascii=False, indent=2))
        try:
            return qt, _call(llm, STEP1_SYSTEM, user, args.max_tokens, "criteria")
        except Exception as exc:
            print(f"[step1] {qt} failed: {exc} — keeping original")
            return qt, [c for c in (_clean(x) for x in crits) if c]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(_step1, qt) for qt in types]):
            qt, cs = f.result()
            distilled[qt] = cs
            print(f"[step1] {qt}: {len(_tpl(rubric, qt).get('rubric_criteria') or [])} -> {len(cs)}")

    if args.dump_stages:
        stage = load_rubric(args.rubric_in)
        for qt in types:
            stage["templates"][qt]["rubric_criteria"] = distilled[qt]
        save_rubric(str(args.rubric_out) + ".step1.yaml", stage)

    # ── Step 2a: induce shared general library ──
    print("[step2a] inducing shared general library ...")
    by_type = "\n\n".join(f"## {qt} ({len(distilled[qt])})\n{_compact(distilled[qt])}"
                          for qt in types if distilled[qt])
    general = _call(llm, STEP2A_SYSTEM, STEP2A_USER.format(by_type=by_type),
                    args.max_tokens, "general")
    print(f"[step2a] general library: {len(general)} criteria")

    # ── Step 2b: subtract general from each type (parallel) ──
    print("[step2b] pruning type-specific residue against general ...")
    gcompact = _compact(general)
    reduced: dict[str, list[dict]] = {}

    def _step2b(qt):
        crits = distilled[qt]
        if not crits:
            return qt, []
        user = STEP2B_USER.format(general=gcompact, qtype=qt,
                                  criteria_json=json.dumps(crits, ensure_ascii=False, indent=2))
        try:
            return qt, _call(llm, STEP2B_SYSTEM, user, args.max_tokens, "criteria")
        except Exception as exc:
            print(f"[step2b] {qt} failed: {exc} — keeping distilled")
            return qt, crits

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(_step2b, qt) for qt in types]):
            qt, cs = f.result()
            reduced[qt] = cs
            print(f"[step2b] {qt}: {len(distilled[qt])} -> {len(cs)}")

    # ── Assemble verifier-compatible rubric ──
    out = {
        "general": {
            "rubric_criteria": general,
            "scoring_rule": "average",
            "sufficient_threshold": DEFAULT_THRESHOLD,
        },
        "type_aliases": {qt: qt for qt in types},
        "templates": {
            qt: {
                "rubric_criteria": reduced[qt],
                "scoring_rule": "average",
                "sufficient_threshold": DEFAULT_THRESHOLD,
            }
            for qt in types
        },
    }
    inject_groundedness(out)
    save_rubric(args.rubric_out, out)

    print("\n=== consolidated ===")
    print(f"general: {len(out['general']['rubric_criteria'])}")
    for qt in sorted(types, key=lambda q: -len(reduced[q])):
        print(f"{len(reduced[qt]):3d}  {qt}")
    print(f"wrote → {args.rubric_out}")


if __name__ == "__main__":
    main()
