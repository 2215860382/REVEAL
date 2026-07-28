"""Planner — generate retrieval queries for the REVEAL iterative loop.

Two-stage planning:
  iter-0: sub-question decomposition (LLM-decomposed or option-grounded).
  iter≥1: unified planner — given the verifier's missing_evidence_analysis,
          pick text (targeted queries) or scan (whole-timeline sweep) retrieval.

Safety nets on iter≥1 query generation (applied via ``filter_targeted_queries``):
  - Jaccard ≥ ``dedup_thresh`` vs prior queries → drop the query.
  - BGE cosine ≥ ``drift_threshold`` vs accumulated evidence → drop the query.
  - All query retrievals dropped → code-only timeline scan fallback for this iter.
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

# Mainline planner prompt — the VEIL-79.00 targeted|broadcast design. The two
# strategies are surfaced to the model as **Text Retrieval** and **Uniform Sampling**
# (same behavior as the old targeted / broadcast-sweep). The JSON ``strategy`` value
# stays the compact ``text`` / ``scan`` token so the parser and the 79-run output
# distribution are unchanged. No time channel. (Visual Retrieval is a future add —
# see REVEAL_PLANNER_MODE=qsi + REVEAL_IMAGE_MAX_ROUNDS; when enabled it should cap
# at one image round.)
_PLANNER_MAINLINE_SYS = """
You are a retrieval strategy planner for a video question-answering system.
Given the current evidence and what is still missing, decide the BEST next retrieval strategy.

## Strategies

**Text Retrieval** — Issue 1–4 focused queries to retrieve specific missing evidence.
  Each query must be self-contained, target a single retrievable fact,
  and not repeat queries already tried.

**Uniform Sampling** — Sample frames uniformly across the uncovered video timeline (no queries needed).
  Use when coverage must span the ENTIRE video, including:
  - Main theme, topic overview, synopsis, or overall narrative of the video
  - Events spread throughout / across all segments / entire duration
  - Chronological order, sequence of events, or timeline across the full video
  - Targeted text queries have repeatedly failed to find sufficient evidence

## Output format
Return ONLY a JSON object:
{
  "strategy": "Text Retrieval" | "Uniform Sampling",
  "queries": ["..."],
  "reasoning": "<one sentence explaining the choice>"
}
Use "Text Retrieval" for Text Retrieval and "Uniform Sampling" for Uniform Sampling.
For Uniform Sampling, set queries to [].
"""

# # Mainline prompt plus exactly one additional retrieval action.  Keep this as a
# # separate literal (rather than mutating the paper-mainline prompt at runtime)
# # so ``REVEAL_PLANNER_MODE=visual`` is an explicit, reproducible visual arm.
# _PLANNER_MAINLINE_VISUAL_SYS = """You are a retrieval strategy planner for a video question-answering system.
# Given the current evidence and what is still missing, decide the BEST next retrieval strategy.

# ## Strategies

# **Text Retrieval** — Issue 1–4 focused queries to retrieve specific missing evidence.
#   Each query must be self-contained, target a single retrievable fact,
#   and not repeat queries already tried.

# **Uniform Sampling** — Sample frames uniformly across the uncovered video timeline (no queries needed).
#   Use when coverage must span the ENTIRE video, including:
#   - Main theme, topic overview, synopsis, or overall narrative of the video
#   - Events spread throughout / across all segments / entire duration
#   - Chronological order, sequence of events, or timeline across the full video
#   - Targeted text queries have repeatedly failed to find sufficient evidence

# **Visual Retrieval** — Issue ONE focused query for a fact that must be visually inspected
#   (for example an object's appearance, a displayed action, a scene detail, or on-screen text).
#   The system retrieves the single most visually similar video chunk and shows its representative
#   keyframe. Use this only when the missing fact is genuinely visual rather than recoverable from
#   captions or speech.

# ## Output format
# Return ONLY a JSON object:
# {
#   "strategy": "Text Retrieval" | "Uniform Sampling" | "Visual Retrieval",
#   "queries": ["..."],
#   "reasoning": "<one sentence explaining the choice>"
# }
# Use "Text Retrieval" for Text Retrieval, "Uniform Sampling" for Uniform Sampling, and
# "Visual Retrieval" for Visual Retrieval. For Uniform Sampling, set queries to []. For Visual
# Retrieval, provide exactly one query."""



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
    retrieval_quality: Optional[str] = None,
    coverage_gap: Optional[str] = None,
) -> dict:
    """Unified planner for iter ≥1: pick text-query or timeline-scan retrieval.

    Returns ``{"think": "...", "strategy": "query"|"scan"|"image", "queries": [...]}``.
    """
    opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))

    parts = [f"Question: {question}", f"Options:\n{opts}"]

    # history_query — the single record of what was already searched. Same source as the old
    # per-attempt trajectory <query> (both are the planner's past queries), collapsed to ONE
    # representation: the original plan_history "Already tried" block. With it visible the planner
    # never reissues a query (0% repeats vs 51% when hidden). The redundant trajectory <query> copy
    # was removed; the trajectory now carries only per-attempt retrieval quality.
    if plan_history:
        lines = [
            "  " + " | ".join(f'"{q}"' for q in qs)
            for qs in plan_history if qs
        ]
        if lines:
            parts.append("Already tried (do NOT repeat):\n" + "\n".join(lines))
    # Per-attempt retrieval quality tells the planner to abandon an exhausted
    # direction rather than merely rewording it.
    if retrieval_quality:
        parts.append(
            "Retrieval quality of each attempt so far, in order:\n" + retrieval_quality)

    if coverage_gap:
        parts.append(
            "Timeline coverage state (use it to target genuinely uncovered evidence):\n"
            f"<coverage_gap>{coverage_gap}</coverage_gap>")

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

    # Mainline planner = the VEIL-79 text|scan prompt. The low-scoring query|time and
    # query|image|time prompts were removed — the mainline no longer has a time/image
    # retrieval channel. scan is planner-selectable here AND the execution fallback via
    # filter_targeted_queries. REVEAL_PLANNER_MODE=qs/qsi still selects the separate
    # retrieval-design ablation prompts (left untouched).
    _mode = os.environ.get("REVEAL_PLANNER_MODE", "").strip().lower()
    # ``visual`` is the mainline-compatible single-keyframe arm.  The older
    # qsi ablation continues to use REVEAL_IMAGE_MAX_ROUNDS for its dense-frame
    # visual channel.
    _visual_mainline = _mode == "visual"
    _img_enabled = _visual_mainline or (
        (os.environ.get("REVEAL_IMAGE_MAX_ROUNDS", "0") or "0") != "0"
    )
    # Mainline default = the VEIL-79 text|scan prompt, where scan IS planner-selectable.
    _default_mainline = not _mode
    _scan_selectable = _mode in ("qs", "qsi") or _default_mainline
    if _mode == "qs":
        _default_sys = _PLANNER_SYS_QS
    elif _mode == "qsi":
        _default_sys = _PLANNER_SYS_QSI
    elif _visual_mainline:
        _default_sys = _PLANNER_MAINLINE_VISUAL_SYS
    else:
        _default_sys = _PLANNER_MAINLINE_SYS         # mainline: text | scan  (VEIL-79)
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
            if raw_strategy in ("text", "targeted", "semantic", "query"):
                strategy = "query"
            elif raw_strategy in ("scan", "sweep", "coverage", "broadcast", "uniform"):
                # Mainline: ``scan`` is an execution fallback, never planner-selectable —
                # collapse a scan-like generation to a normal text query. Under
                # REVEAL_PLANNER_MODE=qs/qsi it IS selectable, so keep it as "scan".
                strategy = "scan" if _scan_selectable else "query"
            elif raw_strategy in (
                "image", "visual", "visual retrieval", "frame", "keyframe", "keyframes"
            ):
                strategy = "image" if _img_enabled else "query"   # image off → safe text fallback
            else:
                strategy = "query"
            queries  = [str(q).strip() for q in (d.get("queries") or []) if str(q).strip()]
            if strategy in ("query", "image") and not queries:
                queries  = [question]
            return {
                "think":     str(d.get("think") or d.get("reasoning") or ""),
                "strategy":  strategy,
                "queries":   queries,
                "reasoning": str(d.get("think") or d.get("reasoning") or ""),
            }
        except Exception:
            pass
    return {"think": "parse_failed", "strategy": "query", "queries": [question],
            "reasoning": "parse_failed"}


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
        """Iter-0 plan: split the question into 2-4 atomic sub-queries."""
        queries = _decompose_into_subquestions(question, candidates, self.llm)
        return {
            "think":     "iter0_single" if queries == [question] else "iter0_decompose",
            "strategy":  "query",
            "queries":   queries,
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
        retrieval_quality: Optional[str] = None,
        coverage_gap: Optional[str] = None,
    ) -> dict:
        """Pick the next iter plan from the verifier verdict.

        ``retrieval_quality`` and ``coverage_gap`` are compact Online-Memory state passed every
        repair iteration. Returns the plan dict. The
        verifier's missing-evidence text is consumed internally (built into the planner prompt);
        the trace reads it straight from the verdict, so it is not returned here.
        """
        m_desc = _planner_repair_context(verdict, rubric_judgment)

        return _plan_next(
            question, candidates, evidence_texts, m_desc, self.llm, plan_history,
            retrieval_quality=retrieval_quality,
            coverage_gap=coverage_gap,
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

        If every text query is dropped, keeps the planner's ``strategy="query"`` intent
        and adds ``fallback="scan"``.  Scan is an execution fallback, not a fourth
        planner-selectable retrieval strategy.
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
        if plan["strategy"] == "query":
            return {**plan, "queries": [], "fallback": "scan",
                    "fallback_reason": "all_query_retrievals_filtered"}
        # An exhausted image query returns to text retrieval with the same
        # semantic targets; it must not silently become a timeline scan.
        return {**plan, "strategy": "query", "queries": list(plan["queries"]),
                "fallback_reason": "all_image_queries_filtered_to_query"}
