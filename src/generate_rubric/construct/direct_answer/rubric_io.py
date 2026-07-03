"""Shared rubric representation for the online (RTD) construction loop.

Rubric on disk matches what the verifier loads (src/agents/verifier.py):
    general:
      rubric_criteria: [ {name, description, score_1, score_half, score_0,
                          weight, failure_repair_action}, ... ]
      scoring_rule: average
      sufficient_threshold: 0.80
    type_aliases: { <question_type>: <template_key> }
    templates:
      <question_type>:
        rubric_criteria: [ ... ]
        scoring_rule: average
        sufficient_threshold: 0.80

We key templates directly by question_type (alias = identity) so every type grows
its own criteria set. `verify_rubric_dict` returns the combined dict the verifier
consumes for a given question type.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

DEFAULT_THRESHOLD = 0.80

REPAIR_ACTIONS = {"refine_query", "time_sorted", "dense_sample", "broadcast", "asr_match"}


def as_whether(desc: str) -> str:
    """Normalise a criterion description to a yes/partial/no question form.

    Every criterion is a scoreable checkpoint, so its description reads as
    "Whether ..." and pairs naturally with score_1/score_half/score_0. Idempotent;
    "For ... claims, ..." openers become "Whether, for ... claims, ...".
    """
    desc = (desc or "").strip()
    if not desc or desc.startswith("Whether "):
        return desc
    if desc.startswith("For "):
        return "Whether, for " + desc[4:]
    return "Whether " + desc[0].lower() + desc[1:]

# Coarse seed criteria (rubric_0). Each type starts from general only; refinement
# grows a per-type template.
GENERAL_SEED = [
    {
        "name": "evidence_coverage",
        "description": ("Evidence covers the key event, time window, state change, comparison "
                        "target, or causal link needed to verify or exclude each option."),
        "score_1": "All aspects the options hinge on are present in the evidence.",
        "score_half": "Some required aspect present but at least one is missing.",
        "score_0": "A key aspect the options depend on is entirely absent.",
        "weight": 1.0,
        "failure_repair_action": "refine_query",
    },
    {
        "name": "evidence_specificity",
        "description": ("Evidence explicitly grounds the relevant person, object, action, scene, "
                        "speaker, on-screen text, or spatial relation rather than a generic scene."),
        "score_1": "The specific entity/relation an option asserts is explicitly grounded.",
        "score_half": "Related but generic evidence; the exact entity/relation is implicit.",
        "score_0": "Only unrelated or generic scene evidence; nothing specific to the options.",
        "weight": 1.0,
        "failure_repair_action": "dense_sample",
    },
    {
        "name": "evidence_consistency",
        "description": ("Evidence is internally consistent and contains enough contrary evidence "
                        "to exclude options that conflict with the video."),
        "score_1": "Evidence is consistent and lets conflicting options be ruled out.",
        "score_half": "Partly consistent; some conflicting options cannot be ruled out yet.",
        "score_0": "Evidence is contradictory or cannot exclude any wrong option.",
        "weight": 1.0,
        "failure_repair_action": "broadcast",
    },
]

GROUNDEDNESS_GUARD = {
    "name": "evidence_groundedness",
    "description": ("The sufficiency judgment is grounded in the retrieved evidence, not in world "
                    "knowledge, option wording echoes, or guesswork."),
    "score_1": "Judgment rests entirely on facts present in the retrieved evidence.",
    "score_half": "Judgment leans partly on outside knowledge or option phrasing.",
    "score_0": "Judgment is fabricated from world knowledge / echoes with no evidentiary basis.",
    "weight": 1.0,
    "failure_repair_action": "refine_query",
}


def _whetherise(c: dict) -> dict:
    c = dict(c)
    c["description"] = as_whether(c.get("description", ""))
    return c


def default_rubric() -> dict:
    return {
        "general": {
            "rubric_criteria": [_whetherise(c) for c in GENERAL_SEED],
            "scoring_rule": "average",
            "sufficient_threshold": DEFAULT_THRESHOLD,
        },
        "type_aliases": {},
        "templates": {},
    }


def ensure_template(rubric: dict, qtype: str) -> dict:
    """Create (if absent) and return the per-type template block."""
    templates = rubric.setdefault("templates", {})
    if qtype not in templates:
        templates[qtype] = {
            "rubric_criteria": [],
            "scoring_rule": "average",
            "sufficient_threshold": DEFAULT_THRESHOLD,
        }
        rubric.setdefault("type_aliases", {})[qtype] = qtype
    return templates[qtype]


def verify_rubric_dict(rubric: dict, question_type: Optional[str]) -> dict:
    """Combined dict the verifier consumes: general criteria + this type's criteria."""
    general = rubric.get("general") or {}
    crits = list(general.get("rubric_criteria") or [])
    tpl = (rubric.get("templates") or {}).get(question_type or "")
    if tpl:
        crits = crits + list(tpl.get("rubric_criteria") or [])
        rule = tpl.get("scoring_rule", general.get("scoring_rule", "average"))
        thr = tpl.get("sufficient_threshold", general.get("sufficient_threshold", DEFAULT_THRESHOLD))
    else:
        rule = general.get("scoring_rule", "average")
        thr = general.get("sufficient_threshold", DEFAULT_THRESHOLD)
    return {"rubric_criteria": crits, "scoring_rule": rule, "sufficient_threshold": thr}


def load_rubric(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        return default_rubric()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("general", default_rubric()["general"])
    data.setdefault("templates", {})
    data.setdefault("type_aliases", {})
    return data


def save_rubric(path: str | Path, rubric: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(rubric, allow_unicode=True, sort_keys=False), encoding="utf-8")


def inject_groundedness(rubric: dict) -> None:
    crits = rubric.setdefault("general", {}).setdefault("rubric_criteria", [])
    if not any(c.get("name") == GROUNDEDNESS_GUARD["name"] for c in crits):
        crits.append(_whetherise(GROUNDEDNESS_GUARD))
