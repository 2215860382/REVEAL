"""Verifier — judge whether retrieved evidence satisfies the rubric for this question type.

Rubric-guided judgment:
  1. Rubric criteria – explicit 0 / 0.5 / 1 per criterion per option; weighted-averaged
                       to a per-option score, thresholded into true / false / unknown.
  2. Label and gaps  – label follows the rubric threshold; gaps drive the next query.

Default rubric lives in ``outputs/rubric/rubric_contrastive.yaml``: a ``general`` block
(criteria added to every question) plus one top-level block per question type (keyed by
the type name). ``get_rubric_dict`` matches ``task_type`` to a same-named block and appends
its criteria to ``general``; unmatched types fall back to ``general`` only. (Legacy
``templates:`` / ``type_aliases:`` nesting is still parsed for backward compatibility.)
"""
from __future__ import annotations

import statistics
from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.utils.extract_json import as_str, extract_json


def _inject_images(message: dict, pil_images) -> dict:
    """Replace str content with [image_url…, text] for multimodal API calls."""
    import base64, io
    valid = [img for img in pil_images if img is not None]
    if not valid:
        return message
    content = []
    for img in valid:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": message["content"]})
    return {"role": message["role"], "content": content}


# ── YAML loading ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _rubric_config() -> Tuple[Dict, Dict, Dict[str, str], List[Tuple[List[str], str]]]:
    override = os.environ.get("VEIL_RUBRIC_PATH")
    path = (
        Path(override).expanduser()
        if override
        else Path(__file__).resolve().parents[2]
        / "outputs"
        / "rubric"
        / "rubric_contrastive.yaml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"VEIL rubric templates missing: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    general: Dict = data.get("general") or {}
    if not general.get("rubric_criteria"):
        raise ValueError(f"{path} must define general.rubric_criteria")

    # Type-specific rubrics. New format: each question type is a top-level key
    # (a dict with its own rubric_criteria). Legacy format: nested under
    # ``templates:`` with a redundant identity ``type_aliases:`` map. Reserved
    # top-level keys are never treated as types.
    RESERVED = {"general", "types", "type_aliases", "templates", "keyword_rules"}
    templates: Dict = {}
    if data.get("templates"):
        for k, v in data["templates"].items():
            templates[k] = v
    else:
        for k, v in data.items():
            if k not in RESERVED and isinstance(v, dict) and v.get("rubric_criteria"):
                templates[k] = v

    aliases: Dict[str, str] = dict(data.get("type_aliases") or {})
    rules_raw = data.get("keyword_rules") or []
    keyword_rules: List[Tuple[List[str], str]] = []
    for row in rules_raw:
        kws = row.get("keywords") or []
        qtype = row.get("qtype", "")
        if isinstance(kws, str):
            kws = [kws]
        keyword_rules.append((list(kws), str(qtype)))

    return general, templates, aliases, keyword_rules


def get_rubric_dict(question: str, task_type: Optional[str] = None) -> dict:
    """Return the combined rubric dict for the given question / task type.

    Always includes general criteria. If a type-specific template matches,
    its criteria are appended and its scoring_rule / sufficient_threshold apply.
    Falls back to general's scoring_rule / sufficient_threshold when no type matches.
    """
    general, templates, aliases, keyword_rules = _rubric_config()

    # Resolve type-specific template key
    key: Optional[str] = None
    if task_type:
        key = aliases.get(task_type, task_type)  # no alias table → match type name directly
        if key not in templates:
            key = None

    # keyword_rules override (question-text based routing)
    q_lower = (question or "").lower()
    for kws, qtype in keyword_rules:
        if any(kw in q_lower for kw in kws) and qtype in templates:
            key = qtype
            break

    if key is not None:
        tpl = templates[key]
        return {
            "rubric_criteria": general["rubric_criteria"] + (tpl.get("rubric_criteria") or []),
            "scoring_rule":         tpl.get("scoring_rule",         general.get("scoring_rule", "average")),
            "sufficient_threshold": tpl.get("sufficient_threshold", general.get("sufficient_threshold", 0.75)),
        }
    # No type match: general only
    return dict(general)


def get_rubric(question: str, task_type: Optional[str] = None) -> str:
    """Return the rubric as a formatted text string (backward-compat helper)."""
    d = get_rubric_dict(question, task_type)
    if "_legacy_text" in d:
        return d["_legacy_text"]
    return _format_rubric_as_text(d)


def _format_rubric_as_text(d: dict) -> str:
    lines = []
    for crit in d.get("rubric_criteria") or []:
        weight = crit.get("weight", 1.0)
        wtxt = f" [weight={weight}]" if float(weight) != 1.0 else ""
        lines.append(
            f"  Criterion [{crit['name']}] {wtxt}: {crit['description']} "
            f"(1={crit['score_1']}; 0.5={crit['score_half']}; 0={crit['score_0']})"
        )
    rule = d.get("scoring_rule", "average")
    thr  = d.get("sufficient_threshold", 0.5)
    lines.append(f"  Scoring: {rule} of criteria; sufficient if score >= {thr}")
    return "\n".join(lines)


# ── Verifier prompt ────────────────────────────────────────────────────────────

VERIFIER_SYS = """\
You evaluate whether retrieved video evidence is sufficient to judge each answer option.
Output ONE strict JSON object — no prose, no markdown fences.

## Step 1 — Per-Option Evidence Sufficiency Scoring
For each answer option, convert it into a proposition and score each rubric criterion:
  1.0 = evidence fully sufficient to judge this option on this criterion
        (clearly confirms OR clearly rules it out — both count as sufficient)
  0.5 = evidence partially sufficient — some relevant information present but incomplete
  0.0 = evidence insufficient to judge this option on this criterion

Score whether the evidence ALLOWS JUDGMENT of this option, not whether it supports it.
Evidence that clearly rules out an option scores as high as evidence that clearly confirms it.
When unsure between 1.0 and 0.5, choose 0.5; between 0.5 and 0.0, choose 0.0.

## Step 2 — Option Judgment
Based on the overall evidence for each option:
  true    : evidence clearly establishes this option is correct
  false   : evidence clearly establishes this option is incorrect
  unknown : evidence is insufficient to determine whether this option is correct or not

## Step 3 — Missing Evidence Analysis
If any option is "unknown", write ONE actionable sentence in "missing_evidence_analysis"
naming the specific fact, time range, or event still needed to resolve it.
Leave it empty string if no option is unknown.

Return ONLY this JSON (use actual criterion names from the Rubric):
{
  "option_criteria_scores": {
    "A": {"<criterion>": 0.0, ...},
    "B": {"<criterion>": 0.0, ...}
  },
  "option_judgment": {
    "A": "true|false|unknown",
    "B": "true|false|unknown"
  },
  "missing_evidence_analysis": "..."
}"""



VERIFIER_SYS_HOLISTIC = """\
You evaluate whether retrieved video evidence is SUFFICIENT to answer a multiple-choice
question, using a rubric of sufficiency dimensions.
Output ONE strict JSON object — no prose, no markdown fences.

Judge EACH rubric criterion INDEPENDENTLY, one at a time, against THIS evidence.
This is the whole point of the rubric: the score is computed by CODE from your
per-criterion answers, so you must actually check each criterion separately — never
paint them all the same value from a gut "is it enough" feeling.

Score EVERY criterion listed in the Rubric — do NOT omit any — with 0, 0.5, or 1:
  1   = the evidence EXPLICITLY and FULLY provides what this dimension requires, about the
        EXACT entity / event / moment the QUESTION asks about.
  0.5 = PARTIAL: related information is present but incomplete, indirect, or about a
        near-but-not-exact target — enough to lean, not enough to settle it.
  0   = the required fact is ABSENT, OR only a distractor / different entity is present.
        A look-alike fact that is NOT about the question's specific target is 0.
Use 0.5 deliberately to separate "somewhat supported" from "fully grounded" and from
"absent" — do not collapse everything to 0 or 1.

CRITICAL — same criteria for every evidence set of a question:
Whether a dimension "applies" is a property of the QUESTION, never of the evidence in
front of you. You are scoring a FIXED list of criteria; two evidence sets for the SAME
question are scored on the SAME criteria and differ ONLY where one contains the needed
fact and the other does not. Do NOT drop, skip, or up-score a criterion just because this
particular evidence is silent on it — silence about a required fact is 0.

Reality check before you answer: if you gave every criterion the same value (all 1 or all
0), you judged holistically instead of one-by-one. Re-read each against the evidence and
correct it.

If the evidence is insufficient overall, write ONE actionable sentence in
"missing_evidence_analysis" naming the specific fact, time range, or event still needed;
otherwise leave it "".

Return ONLY this JSON (use the actual criterion names from the Rubric; every value is 0, 0.5,
or 1, and include EVERY criterion):
{
  "criteria_scores": {"<criterion>": 1, "<criterion>": 0.5, "<criterion>": 0, ...},
  "missing_evidence_analysis": "..."
}"""


# ── Pass 1 of the two-pass holistic verifier: select the criteria RELEVANT to the
#    question, from the question + options ALONE (no evidence). Relevance is a property
#    of the question, so both evidence chains of a pair get the SAME selected set and
#    become directly comparable; irrelevant dimensions are dropped so they can't dilute
#    the score. Cached per question. ──
VERIFIER_SYS_RELEVANCE = """\
You are given a multiple-choice question about a video and a library of evidence-sufficiency
criteria (abstract dimensions of what evidence would be needed to verify or rule out the
options). Decide, from the QUESTION and OPTIONS ALONE, WHICH criteria are RELEVANT — i.e.
answering this question requires that dimension of evidence. Do NOT look at any evidence;
relevance depends only on what the question asks.

Select every criterion whose dimension is needed to confirm or exclude at least one option,
and ONLY those. Drop criteria about a mechanism this question does not turn on (e.g. skip a
counting/quantity criterion when nothing is being counted, skip a causal criterion when no
why/how is asked). Keep it tight — typically 3–8 criteria.

Output ONE strict JSON object — no prose, no fences:
{"relevant": ["<criterion_name>", "<criterion_name>", ...]}"""


# ── Two-pass rubric verifier (--verifier-two-pass) ───────────────────────────
# Pass 1 scores sufficiency ONLY (no true/false), so the model is not anchored by
# its own answer guess. Python then thresholds per option, and Pass 2 judges
# true/false conditioned on which options cleared the threshold.
VERIFIER_SYS_TWOPASS_SCORE = """\
You evaluate whether retrieved video evidence is SUFFICIENT to judge each answer option.
Do NOT decide which option is correct — only score evidence sufficiency.
Output ONE strict JSON object — no prose, no markdown fences.

## Per-Option Evidence Sufficiency Scoring
For each answer option, convert it into a proposition and score each rubric criterion:
  1.0 = evidence fully sufficient to judge this option on this criterion
        (clearly confirms OR clearly rules it out — both count as sufficient)
  0.5 = evidence partially sufficient — some relevant information present but incomplete
  0.0 = evidence insufficient to judge this option on this criterion
Score whether the evidence ALLOWS JUDGMENT of this option, not whether it supports it.
Evidence that clearly rules out an option scores as high as evidence that confirms it.
When unsure between 1.0 and 0.5, choose 0.5; between 0.5 and 0.0, choose 0.0.

Return ONLY this JSON (use actual criterion names from the Rubric):
{
  "option_criteria_scores": {
    "A": {"<criterion>": 0.0, ...},
    "B": {"<criterion>": 0.0, ...}
  }
}"""

VERIFIER_SYS_TWOPASS_JUDGE = """\
You judge multiple-choice answer options for a video question. A separate scoring step
already decided which options have SUFFICIENT evidence to judge and which do not; each
option below is tagged [SUFFICIENT] or [INSUFFICIENT].
Output ONE strict JSON object — no prose, no markdown fences.

## Step 1 — Judge the [SUFFICIENT] options
For each option tagged [SUFFICIENT], commit to:
  true   : evidence clearly establishes this option is correct
  false  : evidence clearly establishes this option is incorrect
Do NOT output a judgment for [INSUFFICIENT] options — they are already "unknown".

## Step 2 — Missing Evidence Analysis
If any option is [INSUFFICIENT], write ONE actionable sentence in "missing_evidence_analysis"
naming the specific fact, time range, or event still needed to resolve it.
Leave it empty string if every option is [SUFFICIENT].

Return ONLY this JSON:
{
  "option_judgment": { "A": "true|false", ... },
  "missing_evidence_analysis": "..."
}"""




def _format_rubric_for_user(rubric: dict) -> str:
    """Format rubric as a user-message section (criteria + scoring rule)."""
    crits = rubric.get("rubric_criteria") or []
    rule  = rubric.get("scoring_rule", "average")
    thr   = rubric.get("sufficient_threshold", 0.5)

    lines = [f"### Rubric Criteria  (scoring: {rule} of scores; sufficient if score ≥ {thr})"]
    for c in crits:
        weight = c.get("weight", 1.0)
        weight_suffix = f" [weight={weight}]" if float(weight) != 1.0 else ""
        lines.append(f"  [{c['name']}] {weight_suffix}  {c['description']}")
        lines.append(f"    1.0 → {c['score_1']}")
        lines.append(f"    0.5 → {c['score_half']}")
        lines.append(f"    0.0 → {c['score_0']}")
    return "\n".join(lines)


def _criterion_weights(rubric: dict) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for crit in rubric.get("rubric_criteria") or []:
        name = str(crit.get("name") or "").strip()
        if not name:
            continue
        try:
            weight = float(crit.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        weights[name] = max(weight, 0.0)
    return weights


def _format_evidence(evidence_texts: List[str]) -> str:
    if not evidence_texts:
        return "(no evidence retrieved yet)"
    return "\n".join(f"[E{i+1}] {t}" for i, t in enumerate(evidence_texts))


# ── Verifier class ─────────────────────────────────────────────────────────────

class Verifier:
    def __init__(self, llm):
        self.llm = llm
        self._relevance_cache: Dict[str, set] = {}
        self._relevance_lock = __import__("threading").Lock()

    def get_rubric(self, question: str, task_type: Optional[str] = None) -> str:
        return get_rubric(question, task_type)

    def _relevant_criteria(self, question: str, opts: str, rubric_dict: dict) -> set:
        """Pass 1: which criteria are relevant to THIS question (evidence-independent).
        Cached per question so both chains of a pair share one selection. Returns a set of
        criterion names; on any failure falls back to the full set (score-all behaviour)."""
        all_names = [str(c.get("name") or "").strip()
                     for c in (rubric_dict.get("rubric_criteria") or [])
                     if str(c.get("name") or "").strip()]
        all_set = set(all_names)
        if not all_set:
            return all_set
        key = question.strip()
        with self._relevance_lock:
            if key in self._relevance_cache:
                return self._relevance_cache[key]
        catalog = "\n".join(
            f"- {c['name']}: {c.get('description','')}"
            for c in (rubric_dict.get("rubric_criteria") or []) if c.get("name"))
        user = "\n\n".join([
            f"Question: {question}", f"Options:\n{opts}",
            f"Criteria library:\n{catalog}", "Return the JSON now.",
        ])
        sel = all_set
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": VERIFIER_SYS_RELEVANCE},
                 {"role": "user", "content": user}],
                max_new_tokens=200, enable_thinking=False)
            picked = extract_json(raw).get("relevant") or []
            picked = {str(x).strip() for x in picked if str(x).strip() in all_set}
            if picked:                       # keep only recognised names; ignore empties
                sel = picked
        except Exception:
            sel = all_set
        with self._relevance_lock:
            self._relevance_cache[key] = sel
        return sel

    def _parse_criteria_scores(self, parsed: dict) -> Dict[str, Dict[str, float]]:
        """Parse the LLM's option_criteria_scores into {opt: {criterion: float}}."""
        out: Dict[str, Dict[str, float]] = {}
        raw = parsed.get("option_criteria_scores") or {}
        if isinstance(raw, dict):
            for opt_key, crit_dict in raw.items():
                opt = as_str(opt_key).strip().upper()[:1]
                if not opt or not isinstance(crit_dict, dict):
                    continue
                scores: Dict[str, float] = {}
                for cname, cscore in crit_dict.items():
                    try:
                        scores[str(cname).strip()] = float(cscore)
                    except (TypeError, ValueError):
                        scores[str(cname).strip()] = 0.0
                if scores:
                    out[opt] = scores
        return out

    def _verify_twopass(
        self, question, candidates, opts, rubric_section, ev, rubric_dict,
        keyframe_images, sufficient_threshold_delta,
    ) -> Dict:
        """Two-pass rubric verify: pass 1 scores sufficiency, pass 2 judges t/f.

        Pass 1 outputs ONLY option_criteria_scores (no answer guess). Python
        thresholds per option, then pass 2 commits true/false on the options that
        cleared the threshold and writes missing_evidence_analysis for the rest.
        Returns the same 6-field verdict as the single-pass rubric path.
        """
        has_imgs = bool(keyframe_images) and bool(getattr(self.llm, '_api_endpoints', None))

        # ── Pass 1: sufficiency scoring only ──────────────────────────────────
        sys1 = os.environ.get("VEIL_VERIFIER_SYS_SCORE", "").strip() or VERIFIER_SYS_TWOPASS_SCORE
        user1 = "\n\n".join([
            f"Question: {question}", f"Options:\n{opts}",
            f"Rubric:\n{rubric_section}", f"Evidence Chain:\n{ev}",
            "Return the sufficiency-scores JSON now.",
        ])
        msg1 = [{"role": "system", "content": sys1}, {"role": "user", "content": user1}]
        if has_imgs:
            msg1[-1] = _inject_images(msg1[-1], keyframe_images)
        parsed1 = extract_json(self.llm.chat(msg1, max_new_tokens=600, enable_thinking=False))

        option_criteria_scores = self._parse_criteria_scores(parsed1)

        weights = _criterion_weights(rubric_dict)
        option_rubric_scores: Dict[str, float] = {}
        for opt, crit_scores in option_criteria_scores.items():
            total_w = weighted_sum = 0.0
            for name, val in crit_scores.items():
                w = weights.get(name, 1.0)
                total_w += w
                weighted_sum += w * val
            option_rubric_scores[opt] = (weighted_sum / total_w) if total_w > 0 else 0.0

        sufficient_threshold = min(
            1.0,
            float(rubric_dict.get("sufficient_threshold", 0.75)) + float(sufficient_threshold_delta),
        )
        all_opts   = [chr(ord('A') + i) for i in range(len(candidates))]
        suff_opts  = [o for o in all_opts if option_rubric_scores.get(o, 0.0) >= sufficient_threshold]
        insuff_set = {o for o in all_opts if o not in suff_opts}

        # ── Pass 2: judge the sufficient options, ask missing for the rest ────
        opts_tagged = "\n".join(
            f"  ({chr(ord('A') + i)}) {c} "
            + ("[INSUFFICIENT]" if chr(ord('A') + i) in insuff_set else "[SUFFICIENT]")
            for i, c in enumerate(candidates)
        )
        sys2 = os.environ.get("VEIL_VERIFIER_SYS_JUDGE", "").strip() or VERIFIER_SYS_TWOPASS_JUDGE
        user2 = "\n\n".join([
            f"Question: {question}", f"Options:\n{opts_tagged}",
            f"Evidence Chain:\n{ev}", "Return the judgment JSON now.",
        ])
        msg2 = [{"role": "system", "content": sys2}, {"role": "user", "content": user2}]
        if has_imgs:
            msg2[-1] = _inject_images(msg2[-1], keyframe_images)
        parsed2 = extract_json(self.llm.chat(msg2, max_new_tokens=400, enable_thinking=False))

        raw_judgment = parsed2.get("option_judgment") or {}
        llm_judgment: Dict[str, str] = {}
        if isinstance(raw_judgment, dict):
            for k, v in raw_judgment.items():
                key = as_str(k).strip().upper()[:1]
                val = as_str(v).strip().lower()
                if key and val in ("true", "false", "unknown"):
                    llm_judgment[key] = val

        option_judgment: Dict[str, str] = {}
        for opt in all_opts:
            option_judgment[opt] = "unknown" if opt in insuff_set else llm_judgment.get(opt, "unknown")

        n_opts       = len(candidates)
        true_opts    = [k for k, v in option_judgment.items() if v == "true"]
        false_opts   = [k for k, v in option_judgment.items() if v == "false"]
        unknown_opts = [k for k, v in option_judgment.items() if v == "unknown"]
        label = "SUFFICIENT" if len(true_opts) == 1 and len(false_opts) == n_opts - 1 else "INSUFFICIENT"

        missing: Optional[str] = None
        if label == "INSUFFICIENT" and unknown_opts:
            raw_missing = parsed2.get("missing_evidence_analysis")
            if isinstance(raw_missing, str) and raw_missing.strip():
                missing = raw_missing.strip()

        return {
            "label":                  label,
            "option_judgment":        option_judgment,
            "unknown_options":        unknown_opts,
            "option_rubric_scores":   {k: round(v, 4) for k, v in option_rubric_scores.items()},
            "option_criteria_scores": option_criteria_scores,
            "missing_evidence_analysis": missing,
        }

    def _verify_holistic(self, question, opts, rubric_section, ev, rubric_dict,
                         keyframe_images, sufficient_threshold_delta) -> Dict:
        """Two-pass holistic sufficiency:
        Pass 1 (evidence-independent, cached per question): select the criteria RELEVANT to
        this question → fixed denominator shared by both evidence chains of a pair.
        Pass 2 (per evidence): score ONLY those relevant criteria 0/1; weighted-average →
        threshold → SUFFICIENT/INSUFFICIENT. Set VEIL_VERIFIER_NO_RELEVANCE=1 to score the
        full set instead (single-pass fallback)."""
        all_crits = list(rubric_dict.get("rubric_criteria") or [])
        if os.environ.get("VEIL_VERIFIER_NO_RELEVANCE", "").strip():
            relevant = {str(c.get("name") or "").strip() for c in all_crits}
        else:
            relevant = self._relevant_criteria(question, opts, rubric_dict)
        sel_crits = [c for c in all_crits
                     if str(c.get("name") or "").strip() in relevant] or all_crits
        # Pass 2 rubric section holds ONLY the selected criteria (focuses the scorer).
        rubric_section = _format_rubric_for_user({
            **{k: v for k, v in rubric_dict.items() if k != "rubric_criteria"},
            "rubric_criteria": sel_crits,
        })

        binary = bool(os.environ.get("VEIL_VERIFIER_BINARY", "").strip())
        sys_h = os.environ.get("VEIL_VERIFIER_SYS", "").strip() or VERIFIER_SYS_HOLISTIC
        parts = [f"Question: {question}", f"Options:\n{opts}",
                 f"Rubric:\n{rubric_section}", f"Evidence Chain:\n{ev}"]
        if binary:
            parts.append("Score each criterion as EXACTLY 1 (the evidence fully meets it) or 0 "
                         "(it does not). Do NOT use 0.5 — make a hard yes/no decision per criterion.")
        parts.append("Return the JSON now.")
        user = "\n\n".join(parts)
        messages = [{"role": "system", "content": sys_h},
                    {"role": "user", "content": user}]
        if keyframe_images and getattr(self.llm, '_api_endpoints', None):
            messages[-1] = _inject_images(messages[-1], keyframe_images)
        raw = self.llm.chat(messages, max_new_tokens=600, enable_thinking=False)
        parsed = extract_json(raw)
        raw_scores = parsed.get("criteria_scores") or {}
        weights = _criterion_weights(rubric_dict)
        tot = wsum = 0.0
        scores: Dict[str, float] = {}
        # Denominator = the SELECTED relevant criteria (same for both chains of a question).
        # A selected criterion the model omitted / marked non-numeric counts as 0 (required
        # fact absent), never dropped — dropping was what made per-chain denominators diverge.
        for crit in sel_crits:
            name = str(crit.get("name") or "").strip()
            if not name:
                continue
            try:
                v = float(raw_scores.get(name))
            except (TypeError, ValueError):
                v = 0.0
            if binary:
                v = 1.0 if v >= 0.5 else 0.0                      # binary: snap to {0,1}
            else:
                v = 0.0 if v < 0.25 else (0.5 if v < 0.75 else 1.0)  # snap to {0,0.5,1}
            w = weights.get(name, 1.0)
            tot += w; wsum += w * v; scores[name] = v
        holistic = (wsum / tot) if tot > 0 else 0.0
        threshold = min(1.0, float(rubric_dict.get("sufficient_threshold", 0.75))
                        + float(sufficient_threshold_delta))
        label = "SUFFICIENT" if holistic >= threshold else "INSUFFICIENT"
        missing = None
        if label == "INSUFFICIENT":
            rm = parsed.get("missing_evidence_analysis")
            if isinstance(rm, str) and rm.strip():
                missing = rm.strip()
        return {
            "label":                  label,
            "option_judgment":        {},
            "unknown_options":        [],
            "option_rubric_scores":   {},
            "option_criteria_scores": {},
            "holistic_score":         round(holistic, 4),
            "criteria_scores":        scores,
            "missing_evidence_analysis": missing,
        }

    def verify(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        rubric: str | dict,
        keyframe_images=(),
        sufficient_threshold_delta: float = 0.0,
        two_pass: bool = False,
    ) -> Dict:
        """Judge evidence sufficiency with rubric-guided per-option scoring.

        LLM scores each rubric criterion per option; Python computes the weighted
        per-option score, thresholds it, and derives the label
        ("SUFFICIENT" | "INSUFFICIENT").
        """
        if isinstance(rubric, str):
            rubric_dict = {"_legacy_text": rubric, "scoring_rule": "average",
                           "sufficient_threshold": 0.5}
        else:
            rubric_dict = rubric

        # ── Rubric path ────────────────────────────────────────────────────────
        rubric_section = (
            rubric_dict["_legacy_text"]
            if "_legacy_text" in rubric_dict
            else _format_rubric_for_user(rubric_dict)
        )
        opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
        ev   = _format_evidence(evidence_texts)

        if os.environ.get("VEIL_VERIFIER_HOLISTIC", "").strip():
            return self._verify_holistic(
                question, opts, rubric_section, ev, rubric_dict,
                keyframe_images, sufficient_threshold_delta,
            )

        if two_pass:
            return self._verify_twopass(
                question, candidates, opts, rubric_section, ev, rubric_dict,
                keyframe_images, sufficient_threshold_delta,
            )

        user = "\n\n".join([
            f"Question: {question}", f"Options:\n{opts}",
            f"Rubric:\n{rubric_section}", f"Evidence Chain:\n{ev}",
            "Return the JSON now.",
        ])
        # Prompt-iteration override (no source edit needed); empty/unset → default.
        sys_for_rubric = os.environ.get("VEIL_VERIFIER_SYS", "").strip() or VERIFIER_SYS
        messages = [
            {"role": "system", "content": sys_for_rubric},
            {"role": "user",   "content": user},
        ]
        if keyframe_images and getattr(self.llm, '_api_endpoints', None):
            messages[-1] = _inject_images(messages[-1], keyframe_images)
        raw    = self.llm.chat(messages, max_new_tokens=800, enable_thinking=False)
        parsed = extract_json(raw)

        # ── Parse option_criteria_scores ───────────────────────────────────────
        raw_ocs = parsed.get("option_criteria_scores") or {}
        option_criteria_scores: Dict[str, Dict[str, float]] = {}
        if isinstance(raw_ocs, dict):
            for opt_key, crit_dict in raw_ocs.items():
                opt = as_str(opt_key).strip().upper()[:1]
                if not opt or not isinstance(crit_dict, dict):
                    continue
                scores: Dict[str, float] = {}
                for cname, cscore in crit_dict.items():
                    try:
                        scores[str(cname).strip()] = float(cscore)
                    except (TypeError, ValueError):
                        scores[str(cname).strip()] = 0.0
                if scores:
                    option_criteria_scores[opt] = scores

        # ── Compute option_rubric_scores (weighted average per option) ─────────
        weights = _criterion_weights(rubric_dict)
        option_rubric_scores: Dict[str, float] = {}
        for opt, crit_scores in option_criteria_scores.items():
            total_w = weighted_sum = 0.0
            for name, val in crit_scores.items():
                w = weights.get(name, 1.0)
                total_w += w
                weighted_sum += w * val
            option_rubric_scores[opt] = (weighted_sum / total_w) if total_w > 0 else 0.0

        # ── Parse LLM option_judgment, then enforce threshold ──────────────────
        # rubric_score measures evidence sufficiency to judge an option (not support direction).
        # Python overrides to "unknown" when rubric_score < sufficient_threshold.
        sufficient_threshold = min(
            1.0,
            float(rubric_dict.get("sufficient_threshold", 0.75)) + float(sufficient_threshold_delta),
        )
        raw_judgment = parsed.get("option_judgment") or {}
        llm_judgment: Dict[str, str] = {}
        if isinstance(raw_judgment, dict):
            for k, v in raw_judgment.items():
                key = as_str(k).strip().upper()[:1]
                val = as_str(v).strip().lower()
                if key and val in ("true", "false", "unknown"):
                    llm_judgment[key] = val

        option_judgment: Dict[str, str] = {}
        all_opts = [chr(ord('A') + i) for i in range(len(candidates))]
        for opt in all_opts:
            rscore = option_rubric_scores.get(opt)
            if rscore is None or rscore < sufficient_threshold:
                option_judgment[opt] = "unknown"
            else:
                # Threshold only gates whether evidence is sufficient to make a
                # judgment; the LLM may still leave the option unresolved.
                option_judgment[opt] = llm_judgment.get(opt, "unknown")

        # ── Compute label ──────────────────────────────────────────────────────
        n_opts      = len(candidates)
        true_opts   = [k for k, v in option_judgment.items() if v == "true"]
        false_opts  = [k for k, v in option_judgment.items() if v == "false"]
        unknown_opts = [k for k, v in option_judgment.items() if v == "unknown"]

        label = "SUFFICIENT" if len(true_opts) == 1 and len(false_opts) == n_opts - 1 else "INSUFFICIENT"

        # ── Missing evidence analysis (from unknown options only) ──────────────
        missing: Optional[str] = None
        if label == "INSUFFICIENT" and unknown_opts:
            raw_missing = parsed.get("missing_evidence_analysis")
            if isinstance(raw_missing, str) and raw_missing.strip():
                missing = raw_missing.strip()

        return {
            "label":                  label,
            "option_judgment":        option_judgment,
            "unknown_options":        unknown_opts,
            "option_rubric_scores":   {k: round(v, 4) for k, v in option_rubric_scores.items()},
            "option_criteria_scores": option_criteria_scores,
            "missing_evidence_analysis": missing,
        }
