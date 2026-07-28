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
    override = os.environ.get("REVEAL_RUBRIC_PATH")
    path = (
        Path(override).expanduser()
        if override
        else Path(__file__).resolve().parents[2]
        / "outputs"
        / "rubric"
        / "rubric_contrastive.yaml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"REVEAL rubric templates missing: {path}")
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
In "missing_evidence_analysis", name ONLY what is still MISSING for the unknown options —
the specific fact, event, or moment that has NOT been found yet and must be retrieved next,
phrased as the thing to search for. Do NOT restate or summarize evidence already found; that
is not a retrieval target. If nothing is missing, write an empty string — the rubric scores
alone decide sufficiency, so a sufficient answer routes to the answer stage on its own.

Return ONLY this JSON (use actual criterion names from the Rubric):
{
  "rubric_criteria_scores": {
    "A": {"<criterion>": 0.0, ...},
    "B": {"<criterion>": 0.0, ...}
  },
  "unknown_options": ["A", "C"],
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


def _relevance_response_format(names: List[str], *, max_items: int) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "relevant_criteria",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "relevant": {
                        "type": "array",
                        "items": {"type": "string", "enum": names},
                        "maxItems": min(max_items, len(names)),
                    },
                },
                "required": ["relevant"],
                "additionalProperties": False,
            },
        },
    }


# ── Verifier class ─────────────────────────────────────────────────────────────

class Verifier:
    def __init__(self, llm):
        self.llm = llm
        self._relevance_cache: Dict[str, set] = {}
        self._relevance_lock = __import__("threading").Lock()

    def get_rubric(self, question: str, task_type: Optional[str] = None) -> str:
        return get_rubric(question, task_type)

    def _select_relevant_strict(
        self,
        question: str,
        opts: str,
        criteria: List[dict],
        *,
        max_items: int = 8,
    ) -> set:
        names = [str(c.get("name") or "").strip() for c in criteria]
        names = [name for name in names if name]
        allowed = set(names)
        if not names:
            return set()
        catalog = "\n".join(
            f"- {c['name']}: {c.get('description', '')}"
            for c in criteria if c.get("name")
        )
        user = "\n\n".join([
            f"Question: {question}", f"Options:\n{opts}",
            f"Criteria library:\n{catalog}", "Return the JSON now.",
        ])
        raw = self.llm.chat(
            [{"role": "system", "content": VERIFIER_SYS_RELEVANCE},
             {"role": "user", "content": user}],
            max_new_tokens=500,
            enable_thinking=False,
            response_format=_relevance_response_format(names, max_items=max_items),
        )
        picked = extract_json(raw).get("relevant") or []
        return {str(item).strip() for item in picked if str(item).strip() in allowed}

    def _relevant_criteria_batched(
        self, question: str, opts: str, criteria: List[dict], batch_size: int,
    ) -> set:
        shortlisted = set()
        for start in range(0, len(criteria), batch_size):
            batch = criteria[start:start + batch_size]
            try:
                shortlisted.update(self._select_relevant_strict(
                    question, opts, batch, max_items=8,
                ))
            except Exception as exc:
                log.warning("criteria selection batch %d failed: %s", start // batch_size, exc)
        if not shortlisted:
            return set()
        finalists = [c for c in criteria if str(c.get("name") or "").strip() in shortlisted]
        try:
            selected = self._select_relevant_strict(
                question, opts, finalists, max_items=8,
            )
            return selected or shortlisted
        except Exception as exc:
            log.warning("criteria selection consolidation failed: %s", exc)
            return shortlisted

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
        criteria = rubric_dict.get("rubric_criteria") or []
        use_schema = os.environ.get("REVEAL_CRITERIA_SELECTION_SCHEMA", "").strip() == "1"
        if use_schema:
            try:
                sel = self._select_relevant_strict(question, opts, criteria)
            except Exception as exc:
                log.warning("strict criteria selection failed: %s", exc)
                sel = set()
            if not sel:
                batch_size = max(1, int(os.environ.get(
                    "REVEAL_CRITERIA_BATCH_SIZE", "48"
                )))
                sel = self._relevant_criteria_batched(
                    question, opts, criteria, batch_size,
                )
            if not sel:
                sel = all_set
            with self._relevance_lock:
                self._relevance_cache[key] = sel
            return sel

        catalog = "\n".join(
            f"- {c['name']}: {c.get('description','')}"
            for c in criteria if c.get("name"))
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

        # ── Relevance gating (mainline; dataset-independent, always on) ────────
        # Pass 1 picks the criteria RELEVANT to THIS question so irrelevant
        # type-criteria don't dilute the per-option denominator (on egolife only
        # ~6 of 33 apply → scoring all → every option ~0.18 → always INSUFFICIENT).
        # This IS the paper's mainline sufficiency judgment and is the SAME across
        # datasets — always on (the legacy score-all path has been removed).
        if "_legacy_text" not in rubric_dict:
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
        sys_for_rubric = VERIFIER_SYS
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
        # REVEAL_SUFFICIENT_THRESHOLD overrides the rubric YAML base (unset ⇒ YAML value,
        # byte-identical). A higher base ⇒ more INSUFFICIENT ⇒ the loop runs more query rounds.
        _env_thr = os.environ.get("REVEAL_SUFFICIENT_THRESHOLD", "").strip()
        _base_thr = float(_env_thr) if _env_thr else float(rubric_dict.get("sufficient_threshold", 0.75))
        sufficient_threshold = min(
            1.0,
            _base_thr + float(sufficient_threshold_delta),
        )
        all_opts = [chr(ord('A') + i) for i in range(len(candidates))]
        # Paper-mainline aggregation: evidence is sufficient when the winning option
        # is confidently decidable.
        if per_option_rubric_scores:
            _vals = [per_option_rubric_scores.get(opt, 0.0) for opt in all_opts]
            # Aggregation over per-option scores gates sufficiency. Default ``max``
            # (winning option decidable) trips early — a decent best option ≥ threshold
            # after iter0 stops the loop. ``min`` (REVEAL_SUFFICIENCY_AGG=min) requires
            # EVERY option to clear the threshold, so retrieval keeps grinding until all
            # distractors are also decidable — closer to the 78.9 "grind all rounds" behavior.
            _agg = os.environ.get("REVEAL_SUFFICIENCY_AGG", "max").strip().lower()
            rubric_score = min(_vals) if _agg == "min" else max(_vals)
        else:
            rubric_score = 0.0
        # The aggregated rubric_score alone gates sufficiency. ``unknown_options`` is not parsed
        # or carried as a verdict field; unresolved options only scaffold the model's
        # missing-evidence analysis inside the verifier prompt.
        label = "SUFFICIENT" if rubric_score >= sufficient_threshold else "INSUFFICIENT"

        # ── Missing evidence analysis ─────────────────────────────────────────
        missing: Optional[str] = None
        if label == "INSUFFICIENT":
            raw_missing = parsed.get("missing_evidence_analysis")
            if isinstance(raw_missing, str) and raw_missing.strip():
                missing = raw_missing.strip()

        return {
            "label":                  label,
            "rubric_criteria_scores": per_option_scores,
            "rubric_score":           round(rubric_score, 4),
            "missing_evidence_analysis": missing,
        }
