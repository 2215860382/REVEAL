"""Planner — generate retrieval queries for the VEIL iterative loop.

Two-stage planning:
  iter-0: sub-question decomposition (LLM-decomposed or option-grounded).
  iter≥1: unified planner — given verifier signals (unknown_options /
          missing_evidence_analysis), pick either query retrieval or time retrieval.

Safety nets on iter≥1 query generation (applied via ``filter_targeted_queries``):
  - Jaccard ≥ ``dedup_thresh`` vs prior queries → drop the query.
  - BGE cosine ≥ ``drift_threshold`` vs accumulated evidence → drop the query.
  - All query retrievals dropped → fall back to time retrieval for this iter.
"""
from __future__ import annotations

import json as _json
import os
import re
from typing import List, Optional, Tuple

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

**query** — Issue focused semantic retrieval queries.
  Each query must be self-contained, target a single retrievable fact, and not repeat
  queries already tried. Use this when the missing evidence is best described by
  people, objects, actions, dialogue, places, or relations.

**time** — Retrieve by timestamp or time range.
  Use this when the missing evidence is tied to a concrete time span, timestamp,
  sequence position, or requires coverage across a broad interval. Output one or more
  time ranges in seconds.

## Output format
Return ONLY a JSON object:
{
  "think": "<brief reason for the retrieval choice>",
  "strategy": "query" | "time",
  "queries": ["..."],
  "time_ranges": [{"start_sec": 0, "end_sec": 60}]
}

For strategy="query", fill queries and set time_ranges to [].
For strategy="time", fill time_ranges and set queries to [].
If the needed interval is the whole currently available video but exact times are unknown,
use strategy="time" with time_ranges=[] so the system samples uncovered time uniformly.

Use time retrieval when coverage must span the ENTIRE video, including:
  - Main theme, topic overview, synopsis, or overall narrative of the video
  - Events spread throughout / across all segments / entire duration
  - Chronological order, sequence of events, or timeline across the full video
  - Query retrieval has repeatedly failed to find sufficient evidence"""


# Same as _PLANNER_UNIFIED_SYS but offers a third, VISUAL retrieval strategy. Gated by
# VEIL_VISUAL_STRATEGY=1 so the default (query/time) planner — and every in-flight
# experiment/threshold sweep — is byte-identical unless a run explicitly opts in.
_PLANNER_UNIFIED_SYS_IMG = """You are a retrieval strategy planner for a video question-answering system.
Given the current evidence and what is still missing, decide the BEST next retrieval strategy.

## Strategies

**query** — Issue focused semantic retrieval queries over the caption TEXT.
  Each query must be self-contained, target a single retrievable fact, and not repeat
  queries already tried. Use this when the missing evidence is best described by
  people, objects, actions, dialogue, places, or relations in words.

**image** — Retrieve KEYFRAMES (the actual video frames) for the moment your query
  describes and look at them directly. Use this when the missing evidence is a VISUAL
  detail captions tend to miss or get wrong — exact colour, count, on-screen text,
  facial expression, spatial layout, who-holds-what — something you must SEE to confirm
  or rule out an option. Put the moment description in queries.

**time** — Retrieve by timestamp or time range.
  Use this when the missing evidence is tied to a concrete time span, timestamp,
  sequence position, or requires coverage across a broad interval. Output one or more
  time ranges in seconds.

## Output format
Return ONLY a JSON object:
{
  "think": "<brief reason for the retrieval choice>",
  "strategy": "query" | "image" | "time",
  "queries": ["..."],
  "time_ranges": [{"start_sec": 0, "end_sec": 60}]
}

For strategy="query" or "image", fill queries and set time_ranges to [].
For strategy="time", fill time_ranges and set queries to [].
If the needed interval is the whole currently available video but exact times are unknown,
use strategy="time" with time_ranges=[] so the system samples uncovered time uniformly.

Use time retrieval when coverage must span the ENTIRE video, including:
  - Main theme, topic overview, synopsis, or overall narrative of the video
  - Events spread throughout / across all segments / entire duration
  - Chronological order, sequence of events, or timeline across the full video
  - Query retrieval has repeatedly failed to find sufficient evidence"""


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
    unknown_options: Optional[List[str]] = None,
    prune_satisfied: bool = False,
) -> dict:
    """Unified planner for iter ≥1: pick query or time retrieval.

    Returns ``{"think": "...", "strategy": "query"|"time", "queries": [...],
    "time_ranges": [...]}``.
    """
    unknown_set = set(unknown_options or [])
    opts_lines = []
    for i, c in enumerate(candidates):
        letter = chr(ord('A')+i)
        tag = " [UNRESOLVED]" if letter in unknown_set else ""
        opts_lines.append(f"  ({letter}) {c}{tag}")
    opts = "\n".join(opts_lines)

    parts = [f"Question: {question}", f"Options:\n{opts}"]
    if prune_satisfied and plan_history:
        parts.append(
            "NOTE: Review the previously tried queries against the CURRENT evidence. "
            "For each previous query, judge whether its target fact is now adequately covered by the evidence. "
            "Do NOT regenerate queries whose target is already satisfied. "
            "Only generate queries for facts that remain unresolved."
        )

    if plan_history:
        lines = []
        for qs in plan_history:
            if qs:
                lines.append("  " + " | ".join(f'"{q}"' for q in qs))
            else:
                lines.append("  [broadcast]")
        parts.append("Already tried (do NOT repeat):\n" + "\n".join(lines))

    if evidence_texts:
        covered = _extract_covered_times(evidence_texts)
        if covered:
            parts.append(f"Evidence covers video segments: {covered}")

    if missing_analysis:
        parts.append(
            "Missing fact to retrieve:\n"
            f"{missing_analysis}\n"
            "Generate queries for the concrete visual/dialogue evidence needed to resolve this fact."
        )

    parts.append("Output the retrieval strategy JSON now.")

    # Visual strategy (VEIL_VISUAL_STRATEGY=1) offers a third `image` retrieval mode;
    # off by default so the query/time planner and every in-flight sweep are unchanged.
    _visual = os.environ.get("VEIL_VISUAL_STRATEGY") == "1"
    _default_sys = _PLANNER_UNIFIED_SYS_IMG if _visual else _PLANNER_UNIFIED_SYS
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
            elif raw_strategy in ("broadcast", "timeline", "temporal"):
                strategy = "time"
            elif raw_strategy in ("image", "visual", "frame", "keyframe", "keyframes"):
                strategy = "image" if _visual else "query"   # falls back to text when gated off
            elif raw_strategy in ("query", "time"):
                strategy = raw_strategy
            else:
                strategy = "query"
            queries  = [str(q).strip() for q in (d.get("queries") or []) if str(q).strip()]
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

    if missing:
        return missing
    if list(verdict.get("unknown_options") or []):
        return ("Retrieve concrete evidence that can confirm or rule out the options "
                "marked [UNRESOLVED].")
    return ""


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
        """Iter-0 plan: LLM splits the question into 2-4 atomic sub-queries."""
        return {
            "think":     "iter0_decompose",
            "strategy":  "query",
            "queries":   _decompose_into_subquestions(question, candidates, self.llm),
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
        prune_satisfied: bool = False,
    ) -> Tuple[dict, str]:
        """Pick the next iter plan from the verifier verdict.

        Returns (plan, missing_description).
        """
        m_desc = _planner_repair_context(verdict, rubric_judgment)

        unk_opts = verdict.get("unknown_options") if rubric_judgment else None
        plan = _plan_next(
            question, candidates, evidence_texts, m_desc, self.llm, plan_history,
            unknown_options=unk_opts,
            prune_satisfied=prune_satisfied,
        )
        return plan, m_desc

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

        Falls back to broadcast if all targeted queries are dropped.
        """
        if plan["strategy"] not in ("query", "targeted"):
            return plan

        prev_queries = [q for qs in plan_history for q in qs]
        ev_mat = (np.array(evidence_vecs, dtype=np.float32)
                  if drift_threshold is not None and evidence_vecs else None)
        kept: List[str] = []
        for q in plan["queries"]:
            if _too_similar(q, prev_queries, dedup_thresh):
                continue
            if ev_mat is not None:
                qv    = embedder.encode([q])[0]
                drift = float(np.max(ev_mat @ qv))
                if drift >= drift_threshold:
                    continue
            kept.append(q)
        if kept:
            return {**plan, "queries": kept}
        return {"think": "all_query_retrievals_filtered", "strategy": "time",
                "queries": [], "time_ranges": [], "reasoning": "all_query_retrievals_filtered"}
