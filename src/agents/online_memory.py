"""Online memory — REVEAL's single, question-conditioned inference-time state.

The online memory unifies the three things the answer loop maintains while it reasons over the
persistent *offline* memory (multi-scale narrative + visual index):

  1. **evidence cache**   — the compact set of retrieved fragments admitted so far;
  2. **retrieval trajectory** — every retrieval attempt: which method ran, with what query, and
                            (if it produced nothing useful) *why it failed*;
  3. **reasoning state**  — the per-round verifier verdict (sufficiency + missing-evidence gap).

These are not independent buffers: the reasoning state drives the next retrieval, the trajectory
suppresses redundant/failed retrievals, and a fragment enters the evidence cache only after
redundancy (and, optionally, relevance) filtering. This module owns that state; the main loop in
``experiments/core/reveal.py`` drives it.

Behavioural contract: with the opt-in flags OFF, ``admit`` is exactly the legacy
``_dedup_by_similarity`` + accumulate, so results are byte-identical to the pre-refactor loop.
The active behaviours are gated (recall is the proven lever here, so filtering is proven via A/B
before it becomes mainline):

  - ``REVEAL_ONLINE_FILTER=1``   → per-retrieval keep-supporting / drop-misleading fragment filter
                                    (conservative; keep-if-uncertain).
  - trajectory *recording* is always on (additive; never changes the answer).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from src.utils.extract_json import extract_json


# ── Legacy chunk-level dedup (moved verbatim from reveal.py) ──────────────────

def _dedup_by_similarity(
    new_texts: List[str],
    new_ids:   List[int],
    new_vecs:  List[List[float]],
    existing_vecs: List[List[float]],
    threshold: float = 0.85,
) -> tuple[List[str], List[int], List[List[float]]]:
    """Drop new chunks whose v_text cosine sim ≥ threshold against any already-accumulated chunk."""
    if not new_vecs or all(not v for v in new_vecs):
        return new_texts, new_ids, new_vecs
    kept_t, kept_i, kept_v = [], [], []
    running = list(existing_vecs)
    for text, cid, vec in zip(new_texts, new_ids, new_vecs):
        if not vec:
            kept_t.append(text); kept_i.append(cid); kept_v.append(vec)
            continue
        v = np.array(vec, dtype=np.float32)
        if running:
            mat = np.array(running, dtype=np.float32)
            if float(np.max(mat @ v)) >= threshold:
                continue
        kept_t.append(text); kept_i.append(cid); kept_v.append(vec)
        running.append(vec)
    return kept_t, kept_i, kept_v


# ── Relevance / misleading filter (gated: REVEAL_ONLINE_FILTER) ───────────────

_FILTER_SYS = """\
You are curating retrieved video-evidence fragments for a multiple-choice question. For each
fragment decide whether it could help CONFIRM or RULE OUT at least one option. KEEP a fragment
if it is even plausibly on-topic; DROP it only when it is clearly off-topic or misleading with
respect to the question and options. When in doubt, KEEP.

Return ONLY this JSON, no prose, no fences:
  {"keep": [<0-based indices of fragments to keep>]}"""


def _filter_supporting(
    question: str,
    candidates: List[str],
    texts: List[str],
    llm,
) -> List[int]:
    """Return the sorted indices of ``texts`` to KEEP. Conservative: on any failure keep all."""
    n = len(texts)
    if n == 0:
        return []
    opts = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(candidates))
    frags = "\n".join(f"[{i}] {t[:300]}" for i, t in enumerate(texts))
    user = (f"Question: {question}\nOptions:\n{opts}\n\n"
            f"Fragments:\n{frags}\n\nReturn the JSON now.")
    try:
        raw = llm.chat(
            [{"role": "system", "content": _FILTER_SYS},
             {"role": "user", "content": user}],
            max_new_tokens=200, enable_thinking=False)
        keep = extract_json(raw).get("keep")
        if isinstance(keep, list):
            idx = sorted({int(i) for i in keep if isinstance(i, (int, float)) and 0 <= int(i) < n})
            # Guard against a degenerate empty selection wiping out all evidence.
            if idx:
                return idx
    except Exception:
        pass
    return list(range(n))


# ── Records ───────────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    chunk_id: int
    text: str
    v_text: List[float]
    source_query: str
    method: str
    round: int
    status: str = "kept"   # "kept" | "dropped_dup" | "dropped_irrelevant"


@dataclass
class RetrievalAttempt:
    round: int
    method: str
    query: str
    params: dict
    n_retrieved: int
    n_kept: int
    n_dropped_dup: int
    n_dropped_irrelevant: int
    outcome: str                       # productive | redundant | off_target | empty
    failure_reason: Optional[str]


@dataclass
class ReasoningState:
    round: int
    label: str
    rubric_score: Optional[float]
    per_option_scores: dict
    unknown_options: List[str]
    missing_evidence_analysis: Optional[str]


# ── Online memory ───────────────────────────────────────────────────────────

_OUTCOME_REASON = {
    "empty":     "no chunk matched this query / empty time range",
    "redundant": "re-retrieved already-seen evidence",
    "off_target": "retrieved fragments don't bear on any option",
}


class OnlineMemory:
    """Single question-conditioned inference-time state (see module docstring)."""

    def __init__(self, llm, question: str, candidates: List[str]):
        self.llm = llm
        self.question = question
        self.candidates = candidates
        self.evidence: List[EvidenceItem] = []
        self.trajectory: List[RetrievalAttempt] = []
        self.reasoning: List[ReasoningState] = []
        self.seen_ids: Set[int] = set()

    # -- accessors (kept evidence, in admission order) --------------------------
    def evidence_texts(self) -> List[str]:
        return [e.text for e in self.evidence]

    def evidence_ids(self) -> List[int]:
        return [e.chunk_id for e in self.evidence]

    def evidence_vecs(self) -> List[List[float]]:
        return [e.v_text for e in self.evidence]

    # -- admission --------------------------------------------------------------
    def admit(
        self,
        new_texts: List[str],
        new_ids: List[int],
        new_vecs: List[List[float]],
        *,
        method: str,
        query: str,
        round: int,
        dedup_threshold: float,
        params: Optional[dict] = None,
    ) -> int:
        """Dedup (always) + optional relevance filter, then accumulate. Returns n_kept.

        With ``REVEAL_ONLINE_FILTER`` unset this is exactly the legacy dedup+accumulate, so the
        answer is unchanged. Records one ``RetrievalAttempt`` with a diagnosed outcome.
        """
        n_retrieved = len(new_ids)

        # 1. redundancy filter vs the current evidence cache
        d_texts, d_ids, d_vecs = _dedup_by_similarity(
            new_texts, new_ids, new_vecs, self.evidence_vecs(), dedup_threshold)
        n_after_dedup = len(d_ids)
        n_dropped_dup = n_retrieved - n_after_dedup

        # 2. relevance / misleading filter (opt-in)
        n_dropped_irrelevant = 0
        if os.environ.get("REVEAL_ONLINE_FILTER") == "1" and n_after_dedup > 0:
            keep = _filter_supporting(self.question, self.candidates, d_texts, self.llm)
            if len(keep) < n_after_dedup:
                keep_set = set(keep)
                d_texts = [d_texts[i] for i in range(n_after_dedup) if i in keep_set]
                d_ids   = [d_ids[i]   for i in range(n_after_dedup) if i in keep_set]
                d_vecs  = [d_vecs[i]  for i in range(n_after_dedup) if i in keep_set]
                n_dropped_irrelevant = n_after_dedup - len(d_ids)

        # 3. accumulate
        for t, cid, v in zip(d_texts, d_ids, d_vecs):
            self.evidence.append(EvidenceItem(
                chunk_id=cid, text=t, v_text=v, source_query=query, method=method, round=round))
        self.seen_ids.update(d_ids)
        n_kept = len(d_ids)

        # 4. diagnose outcome
        if n_retrieved == 0:
            outcome = "empty"
        elif n_kept == 0 and n_dropped_dup == n_retrieved:
            outcome = "redundant"
        elif n_kept == 0 and n_dropped_irrelevant > 0:
            outcome = "off_target"
        elif n_kept == 0:
            outcome = "redundant"
        else:
            outcome = "productive"
        self.trajectory.append(RetrievalAttempt(
            round=round, method=method, query=query, params=params or {},
            n_retrieved=n_retrieved, n_kept=n_kept, n_dropped_dup=n_dropped_dup,
            n_dropped_irrelevant=n_dropped_irrelevant, outcome=outcome,
            failure_reason=_OUTCOME_REASON.get(outcome)))
        return n_kept

    # -- reasoning state --------------------------------------------------------
    def record_reasoning(self, verdict: dict, round: int) -> None:
        self.reasoning.append(ReasoningState(
            round=round,
            label=verdict.get("label", ""),
            rubric_score=verdict.get("rubric_score"),
            per_option_scores=verdict.get("rubric_criteria_scores") or {},
            unknown_options=list(verdict.get("unknown_options") or []),
            missing_evidence_analysis=verdict.get("missing_evidence_analysis"),
        ))

    # -- planner feedback -------------------------------------------------------
    def trajectory_notes(self) -> str:
        """Compact per-attempt log for the planner: what ran and why it didn't help.

        Only unproductive attempts carry a reason; productive ones are listed so the planner
        still sees what has been covered.
        """
        lines = []
        for a in self.trajectory:
            q = a.query if a.query else "[none]"
            if a.failure_reason:
                lines.append(f'  {a.method} | "{q}" -> {a.outcome}: {a.failure_reason}')
            else:
                lines.append(f'  {a.method} | "{q}" -> productive (+{a.n_kept} evidence)')
        return "\n".join(lines)

    # -- serialization ----------------------------------------------------------
    def summary(self) -> dict:
        """Additive block for the result dict (never read by legacy consumers)."""
        return {
            "n_evidence": len(self.evidence),
            "n_attempts": len(self.trajectory),
            "attempts": [
                {"round": a.round, "method": a.method, "query": a.query,
                 "n_retrieved": a.n_retrieved, "n_kept": a.n_kept,
                 "n_dropped_dup": a.n_dropped_dup, "n_dropped_irrelevant": a.n_dropped_irrelevant,
                 "outcome": a.outcome, "failure_reason": a.failure_reason}
                for a in self.trajectory
            ],
            "evidence": [
                {"chunk_id": e.chunk_id, "method": e.method, "round": e.round,
                 "source_query": e.source_query}
                for e in self.evidence
            ],
        }
