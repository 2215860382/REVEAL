"""Planner — generate retrieval queries for the REVEAL iterative loop.

Two-stage planning:
  iter-0: sub-question decomposition (LLM-decomposed or option-grounded).
  iter≥1: unified planner — given the verifier's missing_evidence_analysis,
          pick either query retrieval or time retrieval.

Safety nets on iter≥1 query generation (applied via ``filter_targeted_queries``):
  - Jaccard ≥ ``dedup_thresh`` vs prior queries → drop the query.
  - BGE cosine ≥ ``drift_threshold`` vs accumulated evidence → drop the query.
  - All query retrievals dropped → fall back to time retrieval for this iter.
"""
from __future__ import annotations

import json as _json
import os
import re
from typing import List, Optional

import numpy as np


# ── Iter-0: sub-question decomposition ────────────────────────────────────────

_SUBQ_SYS = """You are a question decomposer for a video question-answering system.
Decompose the given multiple-choice question into 2-4 atomic sub-questions, each targeting a \
specific piece of information needed to evaluate one or more answer options.

Rules:
- Each sub-question must be self-contained and retrievable from video evidence.
- For "which is NOT correct" / "which is incorrect" questions: generate one sub-question per option.
- For other questions: identify the key atomic facts needed and generate one sub-question per fact.
- Sub-questions must be SHORT (≤15 words), specific, and different from each other.
- Do NOT include option letters in the sub-questions.
- Return ONLY a JSON array of strings, e.g.: ["sub-question 1", "sub-question 2"]"""


def _decompose_into_subquestions(
    question: str,
    candidates: List[str],
    llm,
) -> List[str]:
    """Iter-0: decompose a question into 2-4 atomic sub-questions."""
    opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
    user = (
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
        "Decompose into atomic sub-questions. Return ONLY a JSON array of strings."
    )
    messages = [
        {"role": "system", "content": _SUBQ_SYS},
        {"role": "user",   "content": user},
    ]
    raw = llm.chat(messages, max_new_tokens=200, enable_thinking=False)
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            qs = _json.loads(m.group())
            return [str(q).strip() for q in qs if str(q).strip()][:4]
        except Exception:
            pass
    return [question]


# ── Iter ≥1: Unified Planner ─────────────────────────────────────────────────

_PLANNER_UNIFIED_SYS = """You are a retrieval strategy planner for a video question-answering system.
Given the current evidence and what is still missing, decide the BEST next retrieval strategy.

## Strategies

**query** — Semantic retrieval over the caption TEXT. Use when the missing evidence is a
  fact describable in words: a specific person, object, action, dialogue, place, event, or
  relation. Make each query pinpoint the exact fact still missing, and phrase it differently
  from queries already tried — rewording a past query just returns the same chunks.

**time** — Retrieve by timestamp or time range. Use when the question targets a specific
  moment or bounded span: the first/last time something happens, "the first minute", a named
  segment, or chronological order. ALWAYS output concrete time ranges in seconds — never leave
  them empty.

**scan** — Timeline coverage sweep. Retrieve segments sampled evenly across the WHOLE video
  that have not been retrieved yet. Use when the answer is spread across the entire video and no
  single query or time window captures it — "which of these statements is correct", a total
  count over the whole video, an overall synopsis — or when repeated queries keep returning the
  same chunks and you need to cover parts of the video not yet seen. No queries or ranges needed.

## Output format
Return ONLY a JSON object:
{
  "think": "<brief reason for the choice>",
  "strategy": "query" | "time" | "scan",
  "queries": ["..."],
  "time_ranges": [{"start_sec": 0, "end_sec": 60}]
}

For strategy="query", fill queries and set time_ranges to [].
For strategy="time", fill time_ranges with concrete second-windows and set queries to [].
For strategy="scan", leave both queries and time_ranges empty."""


# The planner's system prompt: offers all three retrieval strategies (query / time / image).
# image is a permanent capability; this is the only planner prompt used.
_PLANNER_UNIFIED_SYS_IMG = """You are a retrieval strategy planner for a video question-answering system.
Given the current evidence and what is still missing, pick the retrieval strategy that fits
what the missing evidence IS.

## Strategies

**query** — Semantic retrieval over the caption TEXT. Use when the missing evidence is a
  fact describable in words: a specific person, object, action, dialogue, place, event, or
  relation. Make each query pinpoint the exact fact still missing, and phrase it differently
  from queries already tried — rewording a past query just returns the same chunks.

**image** — Retrieve the actual video KEYFRAMES and look at them. Use when the answer depends
  on a visual detail — colour, texture, count, on-screen text, facial expression, spatial
  layout, who-holds-what — something you must SEE. Put the moment description in queries.

**time** — Retrieve by timestamp or time range. Use when the question targets a specific
  moment or bounded span: the first/last time something happens, "the first minute", a named
  segment, or chronological order. ALWAYS output concrete time ranges in seconds — never leave
  them empty.

**scan** — Timeline coverage sweep. Retrieve segments sampled evenly across the WHOLE video
  that have not been retrieved yet. Use when the answer is spread across the entire video and no
  single query or time window captures it — "which of these statements is correct", a total
  count over the whole video, an overall synopsis — or when repeated queries keep returning the
  same chunks and you need to cover parts of the video not yet seen. No queries or ranges needed.

## Output format
Return ONLY a JSON object:
{
  "think": "<brief reason for the choice>",
  "strategy": "query" | "image" | "time" | "scan",
  "queries": ["..."],
  "time_ranges": [{"start_sec": 0, "end_sec": 60}]
}

For strategy="query" or "image", fill queries and set time_ranges to [].
For strategy="time", fill time_ranges with concrete second-windows and set queries to [].
For strategy="scan", leave both queries and time_ranges empty."""


# Last-resort image variant (gated: REVEAL_IMAGE_LAST_RESORT=1). Same query/time strategies,
# but image is demoted to a genuine last resort: the captions already describe the visuals in
# text, so image is worth its cost only when a query for this exact fact has ALREADY been tried
# and returned nothing, AND the missing detail is purely perceptual (captions cannot phrase it).
# Empirically image is chosen far less often under this prompt, recovering the accuracy that
# blanket image rounds drag down on VideoMME.
_PLANNER_UNIFIED_SYS_IMG_LASTRESORT = """You are a retrieval strategy planner for a video question-answering system.
Given the current evidence and what is still missing, pick the retrieval strategy that fits
what the missing evidence IS. STRONGLY PREFER text retrieval (query/time): the caption memory
already describes what is on screen — colours, objects, actions, on-screen text — in words, so a
query almost always reaches a visual fact faster and more reliably than re-examining pixels.

## Strategies

**query** — Semantic retrieval over the caption TEXT. Your DEFAULT. Use whenever the missing
  evidence is a fact describable in words: a specific person, object, action, dialogue, place,
  event, or relation. Make each query pinpoint the exact fact still missing, and phrase it
  differently from queries already tried — rewording a past query just returns the same chunks.

**time** — Retrieve by timestamp or time range. Use when the question targets a specific
  moment or bounded span: the first/last time something happens, "the first minute", a named
  segment, or chronological order. ALWAYS output concrete time ranges in seconds — never leave
  them empty.

**scan** — Timeline coverage sweep. Retrieve segments sampled evenly across the WHOLE video
  that have not been retrieved yet. Use when the answer is spread across the entire video and no
  single query or time window captures it — "which of these statements is correct", a total
  count over the whole video, an overall synopsis — or when repeated queries keep returning the
  same chunks and you need to cover parts of the video not yet seen. No queries or ranges needed.

**image** — LAST RESORT ONLY. Retrieve the raw video KEYFRAMES and look at the pixels. This is
  expensive and usually WORSE than a good query, because the captions already narrate the visuals.
  Choose image ONLY when BOTH hold: (1) a query for this exact fact has already been tried and the
  captions genuinely do not contain it, AND (2) the answer hinges on a fine perceptual detail that
  words cannot recover from the captions — an exact colour/shade, a precise count of small items,
  unreadable on-screen text, a subtle facial expression, or fine spatial layout. If a query could
  plausibly surface the fact, choose query, NOT image. Put the moment description in queries.

## Output format
Return ONLY a JSON object:
{
  "think": "<brief reason for the choice>",
  "strategy": "query" | "image" | "time" | "scan",
  "queries": ["..."],
  "time_ranges": [{"start_sec": 0, "end_sec": 60}]
}

For strategy="query" or "image", fill queries and set time_ranges to [].
For strategy="time", fill time_ranges with concrete second-windows and set queries to [].
For strategy="scan", leave both queries and time_ranges empty."""


def _coerce_time_ranges(raw) -> list[dict]:
    ranges: list[dict] = []
    if not isinstance(raw, list):
        return ranges
    for item in raw:
        start = end = None
        if isinstance(item, dict):
            start = item.get("start_sec", item.get("start", item.get("from")))
            end = item.get("end_sec", item.get("end", item.get("to")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        try:
            a, b = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if b < a:
            a, b = b, a
        ranges.append({"start_sec": max(0.0, a), "end_sec": max(0.0, b)})
    return ranges


def _extract_covered_times(evidence_texts: List[str]) -> str:
    """Extract time ranges present in evidence texts to inform the planner."""
    pattern = r'\[(\d+)s[-–](\d+)s\]'
    secs = sorted({(int(m.group(1)), int(m.group(2)))
                   for t in evidence_texts
                   for m in re.finditer(pattern, t)})
    def fmt(s: int) -> str:
        return f"{s//60}:{s%60:02d}"
    return ", ".join(f"{fmt(a)}-{fmt(b)}" for a, b in secs) if secs else ""


def _plan_next(
    question: str,
    candidates: List[str],
    evidence_texts: List[str],
    missing_analysis: str,
    llm,
    plan_history: List[List[str]] = (),
    trajectory_notes: Optional[str] = None,
) -> dict:
    """Unified planner for iter ≥1: pick query or time retrieval.

    Returns ``{"think": "...", "strategy": "query"|"time", "queries": [...],
    "time_ranges": [...]}``.
    """
    opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))

    parts = [f"Question: {question}", f"Options:\n{opts}"]

    # history_query — the single record of what was already searched. Same source as the old
    # per-attempt trajectory <query> (both are the planner's past queries), collapsed to ONE
    # representation: the original plan_history "Already tried" block. With it visible the planner
    # never reissues a query (0% repeats vs 51% when hidden). The redundant trajectory <query> copy
    # was removed; the trajectory now carries only the per-attempt <reason>.
    if plan_history:
        lines = [
            "  " + " | ".join(f'"{q}"' for q in qs)
            for qs in plan_history if qs
        ]
        if lines:
            parts.append("Already tried (do NOT repeat):\n" + "\n".join(lines))
    # Per-attempt outcome from the online memory, in the same order: <reason> tells the planner to
    # abandon an exhausted direction rather than reword it. Empty unless REVEAL_OL_REASON is on.
    if trajectory_notes:
        parts.append(
            "Outcome of each attempt so far, in order, tagged <reason>:\n" + trajectory_notes)

    if evidence_texts:
        covered = _extract_covered_times(evidence_texts)
        if covered:
            parts.append(f"Evidence covers video segments: {covered}")

    if missing_analysis:
        parts.append(
            "Missing fact to retrieve:\n"
            f"<missing_evidence>{missing_analysis}</missing_evidence>\n"
            "Generate queries for the concrete visual/dialogue evidence needed to resolve this fact."
        )

    parts.append("Output the retrieval strategy JSON now.")

    # The planner's menu tracks whether image ROUNDS can actually execute
    # (REVEAL_IMAGE_MAX_ROUNDS): when image is disabled (=0), the model only sees the
    # two-strategy prompt (query / time) so it can't pick a strategy that would no-op;
    # otherwise it gets the three-strategy prompt and freely chooses image per question.
    # Mainline (default 2) → three-strategy prompt, byte-identical to before.
    _img_enabled = (os.environ.get("REVEAL_IMAGE_MAX_ROUNDS", "2") or "2") != "0"
    # REVEAL_IMAGE_LAST_RESORT=1 → image demoted to last resort (see prompt above); default
    # keeps image on equal footing (byte-identical to before when the flag is unset).
    _img_lastresort = os.environ.get("REVEAL_IMAGE_LAST_RESORT") == "1"
    if not _img_enabled:
        _default_sys = _PLANNER_UNIFIED_SYS
    elif _img_lastresort:
        _default_sys = _PLANNER_UNIFIED_SYS_IMG_LASTRESORT
    else:
        _default_sys = _PLANNER_UNIFIED_SYS_IMG
    messages = [
        {"role": "system", "content": _default_sys},
        {"role": "user",   "content": "\n\n".join(parts)},
    ]
    raw = llm.chat(messages, max_new_tokens=200, enable_thinking=False)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            d        = _json.loads(m.group())
            raw_strategy = str(d.get("strategy", "query")).lower()
            if raw_strategy in ("targeted", "semantic"):
                strategy = "query"
            elif raw_strategy in ("timeline", "temporal"):
                strategy = "time"
            elif raw_strategy in ("scan", "sweep", "coverage", "broadcast", "uniform"):
                strategy = "scan"
            elif raw_strategy in ("image", "visual", "frame", "keyframe", "keyframes"):
                strategy = "image" if _img_enabled else "query"   # image off → safe text fallback
            elif raw_strategy in ("query", "time"):
                strategy = raw_strategy
            else:
                strategy = "query"
            queries  = [str(q).strip() for q in (d.get("queries") or []) if str(q).strip()]
            # REVEAL_SINGLE_QUERY: iter≥1 also emits ONE query per round (matches the
            # single-query iter-0), so the +Iter ladder rung is single-query throughout.
            if os.environ.get("REVEAL_SINGLE_QUERY") == "1" and queries:
                queries = queries[:1]
            time_ranges = _coerce_time_ranges(d.get("time_ranges") or d.get("time_range") or d.get("times"))
            if strategy in ("query", "image") and not queries:
                queries  = [question]
            return {
                "think":     str(d.get("think") or d.get("reasoning") or ""),
                "strategy":  strategy,
                "queries":   queries,
                "time_ranges": time_ranges,
                "reasoning": str(d.get("think") or d.get("reasoning") or ""),
            }
        except Exception:
            pass
    return {"think": "parse_failed", "strategy": "query", "queries": [question],
            "time_ranges": [], "reasoning": "parse_failed"}


def _planner_repair_context(verdict: dict, rubric_judgment: bool) -> str:
    """Build the verifier-derived missing-evidence description for the planner."""
    missing = str(verdict.get("missing_evidence_analysis") or "").strip()

    if not rubric_judgment:
        return f"Missing evidence analysis: {missing}" if missing else ""

    return missing


# ── Query similarity safety nets ─────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _too_similar(new_q: str, prev_queries: List[str], threshold: float = 0.9) -> bool:
    return any(_jaccard(new_q, pq) >= threshold for pq in prev_queries)


# ── Public class ─────────────────────────────────────────────────────────────

class Planner:
    """Encapsulates iter-0 decomposition + iter≥1 unified planner + query filtering.

    Holds only the LLM client; embedder and evidence vectors are passed per-call
    so the Planner stays parallel to ``Verifier`` (one external service only).
    """

    def __init__(self, llm):
        self.llm = llm

    def decompose_iter0(
        self,
        question: str,
        candidates: List[str],
    ) -> dict:
        """Iter-0 plan: LLM splits the question into 2-4 atomic sub-queries.

        Ablation knob ``REVEAL_SINGLE_QUERY=1`` disables the decomposition and issues a
        SINGLE query (the raw question). Combined with the same cap on iter≥1 (see
        ``_plan_next``), EVERY round emits exactly one query, so retrieval grows one
        query at a time: the minimal-retrieval base (~8 chunks/round). The "+Iter"
        component (max-iter 1→3) then isolates the verifier-guided multi-round retrieval
        value alone. Mainline sets none of it (decomposition on, byte-identical to before).
        """
        if os.environ.get("REVEAL_SINGLE_QUERY") == "1":
            queries = [question]
        else:
            queries = _decompose_into_subquestions(question, candidates, self.llm)
        return {
            "think":     "iter0_single" if queries == [question] else "iter0_decompose",
            "strategy":  "query",
            "queries":   queries,
            "time_ranges": [],
            "reasoning": "iter0_decompose",
        }

    def plan_next(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        verdict: dict,
        plan_history: List[List[str]],
        *,
        rubric_judgment: bool = True,
        trajectory_notes: Optional[str] = None,
    ) -> dict:
        """Pick the next iter plan from the verifier verdict.

        ``trajectory_notes`` (the online memory's failure-diagnosed retrieval history) replaces the
        plain "already tried" block; mainline passes it every iter. Returns the plan dict. The
        verifier's missing-evidence text is consumed internally (built into the planner prompt);
        the trace reads it straight from the verdict, so it is not returned here.
        """
        m_desc = _planner_repair_context(verdict, rubric_judgment)

        return _plan_next(
            question, candidates, evidence_texts, m_desc, self.llm, plan_history,
            trajectory_notes=trajectory_notes,
        )

    def filter_targeted_queries(
        self,
        plan: dict,
        plan_history: List[List[str]],
        embedder,
        evidence_vecs: List[List[float]],
        *,
        dedup_thresh: float = 0.9,
        drift_threshold: Optional[float] = 0.70,
    ) -> dict:
        """Apply Jaccard + BGE-drift safety nets to a targeted plan.

        Returns ``strategy="scan"`` if every targeted query is dropped: the planner has
        run out of NEW targeted queries (each was a near-duplicate of one already tried),
        so the only productive move left is to sweep timeline regions not yet retrieved
        (see ``_timeline_scan_retrieve``).
        """
        if plan["strategy"] not in ("query", "targeted", "image"):
            return plan

        prev_queries = [q for qs in plan_history for q in qs]
        ev_mat = (np.array(evidence_vecs, dtype=np.float32)
                  if drift_threshold is not None and evidence_vecs else None)
        kept: List[str] = []
        for q in plan["queries"]:
            if _too_similar(q, prev_queries, dedup_thresh):
                continue
            # An image query can intentionally revisit a concept whose caption
            # evidence is already present: the missing fact may be visual.  It
            # still must not repeat a previously issued visual/text query, but
            # text-evidence drift filtering is only appropriate for text search.
            if plan["strategy"] != "image" and ev_mat is not None:
                qv    = embedder.encode([q])[0]
                drift = float(np.max(ev_mat @ qv))
                if drift >= drift_threshold:
                    continue
            kept.append(q)
        if kept:
            return {**plan, "queries": kept}
        return {"think": "all_query_retrievals_filtered", "strategy": "scan",
                "queries": [], "time_ranges": [], "reasoning": "all_query_retrievals_filtered"}
