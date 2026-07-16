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

import logging
import statistics
from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.utils.extract_json import as_str, extract_json

log = logging.getLogger(__name__)


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

## Step 2 — Unknown Options
List answer options whose evidence is still insufficient to confidently confirm OR rule out.
This field guides the next retrieval round.

## Step 3 — Missing Evidence Analysis
Write ONE actionable sentence in "missing_evidence_analysis" naming the specific fact,
time range, or event still needed to resolve the unknown options. If the evidence is
sufficient, honestly write an empty string or "Nothing missing."

Return ONLY this JSON (use actual criterion names from the Rubric):
{
  "rubric_criteria_scores": {
    "A": {"<criterion>": 0.0, ...},
    "B": {"<criterion>": 0.0, ...}
  },
  "unknown_options": ["A", "C"],
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

MODALITY-AGNOSTIC — score the FACT, not how it was captured:
A criterion may be phrased in visual terms ("visible", "on-screen", "visual grounding",
"clearly seen"). You are judging whether the EVIDENCE ESTABLISHES the required fact, NOT
whether a picture is present. A concrete TEXT description that states the needed detail —
an object's location, colour, count, who did what, the order of events — FULLY satisfies
such a criterion and scores 1, exactly as a frame showing it would. NEVER dock a criterion
merely because the evidence is text with no image, and never treat "no image" as the fact
being absent. Score 0 only when the FACT ITSELF is missing from the evidence (text or
image alike). "Visual"/"on-screen" wording in a criterion is descriptive, not a demand for
pixels.

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
  "rubric_criteria_scores": {"<criterion>": 1, "<criterion>": 0.5, "<criterion>": 0, ...},
  "unknown_options": ["<letters (A/B/C/D) of options the current evidence cannot yet confirm or rule out; empty if sufficient>"],
  "missing_evidence_analysis": "..."
}"""


# ── Holistic sufficiency + per-option tagging (VEIL_HOLISTIC_UNKNOWN_OPTS): keep the cheap
#    whole-question holistic scoring, but ALSO name which options remain undecidable so the
#    planner can anchor repair to a specific option. Isolates whether option-anchoring — not
#    the expensive per-option criterion scoring — is what makes repair fetch discriminating
#    evidence instead of topical noise.
VERIFIER_SYS_HOLISTIC_OPT = VERIFIER_SYS_HOLISTIC


# ── Implicit sufficiency (VEIL_VERIFIER_IMPLICIT): a rubric-FREE baseline that mirrors
#    the WorldMM-style retrieval agent, which judges "is the gathered evidence enough?"
#    with a plain LLM call. Used to isolate the value of the explicit rubric: same loop,
#    same options + evidence, only the sufficiency SIGNAL differs (rubric criteria vs. this).
VERIFIER_SYS_IMPLICIT = """\
You judge whether the retrieved video evidence is SUFFICIENT to confidently answer a
multiple-choice question. Read the question, the options, and the evidence, then decide:
  SUFFICIENT   = the evidence clearly determines which option is correct.
  INSUFFICIENT = the evidence is missing, ambiguous, or does not pin the answer down.
If INSUFFICIENT, write ONE actionable sentence in "missing_evidence_analysis" naming the
specific fact, time range, or event still needed to decide; otherwise leave it "".
Return ONLY this JSON — no prose, no markdown fences:
{"verdict": "SUFFICIENT" | "INSUFFICIENT", "missing_evidence_analysis": "..."}"""


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


def _weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total_w = weighted_sum = 0.0
    for name, val in scores.items():
        w = weights.get(name, 1.0)
        total_w += w
        weighted_sum += w * float(val)
    return (weighted_sum / total_w) if total_w > 0 else 0.0


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

    def _parse_per_option_scores(self, parsed: dict) -> Dict[str, Dict[str, float]]:
        """Parse per-option rubric_criteria_scores into {opt: {criterion: float}}."""
        out: Dict[str, Dict[str, float]] = {}
        raw = parsed.get("rubric_criteria_scores") or {}
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

    def _parse_flat_criteria_scores(self, parsed: dict) -> Dict[str, float]:
        """Parse holistic rubric_criteria_scores into {criterion: float}."""
        raw = parsed.get("rubric_criteria_scores") or {}
        scores: Dict[str, float] = {}
        if isinstance(raw, dict):
            for name, val in raw.items():
                if isinstance(val, dict):
                    continue
                try:
                    scores[str(name).strip()] = float(val)
                except (TypeError, ValueError):
                    scores[str(name).strip()] = 0.0
        return scores

    def _verify_implicit(self, question, opts, ev, keyframe_images) -> Dict:
        """Rubric-free sufficiency (VEIL_VERIFIER_IMPLICIT): the LLM decides whether the
        evidence is enough to answer and names what is missing. Returns the same verdict
        contract as the rubric path so the repair loop is byte-identical except for the
        sufficiency signal — this is the WorldMM-style implicit baseline for the rubric."""
        user = "\n\n".join([
            f"Question: {question}", f"Options:\n{opts}",
            f"Evidence:\n{ev}", "Return the JSON now.",
        ])
        messages = [{"role": "system", "content": VERIFIER_SYS_IMPLICIT},
                    {"role": "user", "content": user}]
        if keyframe_images and getattr(self.llm, '_api_endpoints', None):
            messages[-1] = _inject_images(messages[-1], keyframe_images)
        raw = self.llm.chat(messages, max_new_tokens=300, enable_thinking=False)
        parsed = extract_json(raw)
        v = str(parsed.get("verdict") or parsed.get("label") or "").upper()
        # Guard the classic 'SUFF' in 'INSUFFICIENT' trap.
        label = "SUFFICIENT" if ("INSUF" not in v and "SUFF" in v) else "INSUFFICIENT"
        missing = None
        if label == "INSUFFICIENT":
            rm = parsed.get("missing_evidence_analysis") or parsed.get("missing")
            if isinstance(rm, str) and rm.strip():
                missing = rm.strip()
        return {
            "label":                  label,
            "rubric_criteria_scores": {},
            "rubric_score":           None,
            "unknown_options":        [],
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
        want_opts = bool(os.environ.get("VEIL_HOLISTIC_UNKNOWN_OPTS", "").strip())
        _default_sys = VERIFIER_SYS_HOLISTIC_OPT if want_opts else VERIFIER_SYS_HOLISTIC
        sys_h = os.environ.get("VEIL_VERIFIER_SYS", "").strip() or _default_sys
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
        raw_scores = parsed.get("rubric_criteria_scores") or {}
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
        # Option tagging: holistic decides sufficiency, but unknown_options anchors repair.
        unknown_opts: List[str] = []
        if label == "INSUFFICIENT":
            raw_unres = parsed.get("unknown_options") or parsed.get("unresolved_options")
            if isinstance(raw_unres, list):
                unknown_opts = [u for u in
                                (str(x).strip().upper()[:1] for x in raw_unres)
                                if u in "ABCD"]
        return {
            "label":                  label,
            "unknown_options":        unknown_opts,
            "rubric_criteria_scores": scores,
            "rubric_score":           round(holistic, 4),
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

        if os.environ.get("VEIL_VERIFIER_IMPLICIT", "").strip():
            return self._verify_implicit(question, opts, ev, keyframe_images)

        if os.environ.get("VEIL_VERIFIER_HOLISTIC", "").strip():
            return self._verify_holistic(
                question, opts, rubric_section, ev, rubric_dict,
                keyframe_images, sufficient_threshold_delta,
            )

        # ── Relevance gating (mainline default; dataset-independent) ───────────
        # Pass 1 picks the criteria RELEVANT to THIS question so irrelevant
        # type-criteria don't dilute the per-option denominator (on egolife only
        # ~6 of 33 apply → scoring all → every option ~0.18 → always INSUFFICIENT).
        # This IS the paper's mainline sufficiency judgment and is the SAME across
        # datasets, so it defaults ON. Opt out with VEIL_VERIFIER_NO_RELEVANCE=1
        # (legacy score-all ablation) — mirrors the holistic path's opt-out (~L459).
        if (not os.environ.get("VEIL_VERIFIER_NO_RELEVANCE", "").strip()
                and "_legacy_text" not in rubric_dict):
            relevant = self._relevant_criteria(question, opts, rubric_dict)
            sel_crits = [c for c in (rubric_dict.get("rubric_criteria") or [])
                         if str(c.get("name") or "").strip() in relevant]
            if sel_crits:
                rubric_section = _format_rubric_for_user({
                    **{k: v for k, v in rubric_dict.items() if k != "rubric_criteria"},
                    "rubric_criteria": sel_crits,
                })

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
        # Output size scales with 4 options × N criteria. At ~30 criteria the JSON
        # runs ~1900 tokens; an 800 cap silently truncated it mid-scores, dropping
        # missing_evidence_analysis and scores. Give it real headroom.
        raw    = self.llm.chat(messages, max_new_tokens=2500, enable_thinking=False)
        parsed = extract_json(raw)
        if not parsed.get("rubric_criteria_scores") and raw.rstrip() and not raw.rstrip().endswith("}"):
            # Truncated/unparseable despite the higher cap — surface it instead of
            # silently degrading to an empty (all-unknown) verdict.
            log.warning("verifier: truncated/unparseable scoring JSON (%d chars); "
                        "rubric_criteria_scores empty", len(raw))

        # ── Parse per-option rubric_criteria_scores ───────────────────────────
        per_option_scores = self._parse_per_option_scores(parsed)

        # ── Compute weighted average per option ───────────────────────────────
        weights = _criterion_weights(rubric_dict)
        per_option_rubric_scores: Dict[str, float] = {}
        for opt, crit_scores in per_option_scores.items():
            total_w = weighted_sum = 0.0
            for name, val in crit_scores.items():
                w = weights.get(name, 1.0)
                total_w += w
                weighted_sum += w * val
            per_option_rubric_scores[opt] = (weighted_sum / total_w) if total_w > 0 else 0.0

        # ── Enforce threshold and derive unknown options ───────────────────────
        # rubric_score measures evidence sufficiency to judge an option (not support direction).
        # VEIL_SUFFICIENT_DELTA shifts the threshold globally (for threshold sweeps without
        # editing the rubric): e.g. -0.2 turns a rubric threshold of 0.8 into 0.6.
        env_delta = float(os.environ.get("VEIL_SUFFICIENT_DELTA", "0") or "0")
        sufficient_threshold = min(
            1.0,
            float(rubric_dict.get("sufficient_threshold", 0.75))
            + float(sufficient_threshold_delta) + env_delta,
        )
        all_opts = [chr(ord('A') + i) for i in range(len(candidates))]
        raw_unknown = parsed.get("unknown_options") or []
        model_unknown = {
            u for u in (str(x).strip().upper()[:1] for x in raw_unknown)
            if u in all_opts
        } if isinstance(raw_unknown, list) else set()
        score_unknown = {
            opt for opt in all_opts
            if per_option_rubric_scores.get(opt, 0.0) < sufficient_threshold
        }
        unknown_opts = sorted(model_unknown | score_unknown)
        # Aggregation of per-option scores → rubric_score (VEIL_VERIFIER_AGG).
        #   min  (default): sufficient only when EVERY option is decidable — high precision, but
        #                   in open-world egocentric QA distractors are rarely excludable, so it
        #                   fires ~5%. max: sufficient when the WINNING option is confidently
        #                   decidable — matches "answerable" far better (sim: ~5× fire at equal
        #                   correlation gap). mean: middle ground.
        agg = os.environ.get("VEIL_VERIFIER_AGG", "min").strip().lower()
        if per_option_rubric_scores:
            _vals = [per_option_rubric_scores.get(opt, 0.0) for opt in all_opts]
            rubric_score = (max(_vals) if agg == "max"
                            else sum(_vals) / len(_vals) if agg == "mean"
                            else min(_vals))
        else:
            rubric_score = 0.0
        # For min, min≥threshold already implies no score-unknown; keep the model_unknown veto.
        # For max/mean the point is that ONE confident option suffices, so the per-score unknown
        # set must NOT veto sufficiency — only the aggregated score gates the label.
        if agg == "min":
            label = "SUFFICIENT" if rubric_score >= sufficient_threshold and not unknown_opts else "INSUFFICIENT"
        else:
            label = "SUFFICIENT" if rubric_score >= sufficient_threshold else "INSUFFICIENT"

        # ── Missing evidence analysis ─────────────────────────────────────────
        missing: Optional[str] = None
        if label == "INSUFFICIENT":
            raw_missing = parsed.get("missing_evidence_analysis")
            if isinstance(raw_missing, str) and raw_missing.strip():
                missing = raw_missing.strip()

        return {
            "label":                  label,
            "unknown_options":        unknown_opts,
            "rubric_criteria_scores": per_option_scores,
            "rubric_score":           round(rubric_score, 4),
            "missing_evidence_analysis": missing,
        }
