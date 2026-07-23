"""Online memory — REVEAL's single, question-conditioned inference-time state.

The online memory unifies the three things the answer loop maintains while it reasons over the
persistent *offline* memory (multi-scale narrative + visual index):

  1. **evidence cache**   — the compact set of retrieved fragments admitted so far;
  2. **retrieval trajectory** — every retrieval attempt: the query tried, and
                            (if it produced nothing useful) *why it failed*;
  3. **reasoning state**  — the per-round verifier verdict (sufficiency + missing-evidence gap).

These are not independent buffers: the reasoning state drives the next retrieval, the trajectory
suppresses redundant/failed retrievals, and a fragment enters the evidence cache only after
redundancy (and, optionally, relevance) filtering. This module owns that state; the main loop in
``experiments/core/reveal.py`` drives it.

What the online memory feeds back is mainline: the failure-diagnosed retrieval trajectory
(``trajectory_notes``) is passed to the planner every iter — it lets the planner see what was
tried and *why* it was unproductive, so the next plan changes strategy instead of repeating a
failed retrieval (proven +10q on egolife). Only ONE behaviour is still gated, pending A/B — the
per-retrieval relevance filter, because recall is the proven lever here and blanket filtering has
historically netted ≤0:

  - ``REVEAL_ONLINE_FILTER=1``   → per-retrieval keep-supporting / drop-misleading fragment filter
                                    (conservative; keep-if-uncertain). With it unset, ``admit`` is
                                    exactly the legacy ``_dedup_by_similarity`` + accumulate.
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
You curate retrieved evidence fragments for a multiple-choice question about a long video. Your job
is to REMOVE fragments that are irrelevant or misleading, so the answerer sees only trustworthy,
on-target evidence. Read the QUESTION, every OPTION, and every FRAGMENT before deciding.

STEP 1 — Extract the question's CONSTRAINTS: the specific subject/person, action, object, place, and
time (day/part-of-day) it asks about. Fragments carry their time span as a leading [<start>s-<end>s].

STEP 2 — DROP a fragment when it is:
  - IRRELEVANT: about a different scene/subject with no bearing on the question or any option; OR
  - MISLEADING: it looks on-topic but VIOLATES a constraint — a different person, a different
    time/day/part-of-day, a different place, or a different event than the question asks about — so
    using it as evidence would push toward a WRONG option. This is the case you must actively catch:
    superficially-relevant fragments about the wrong instance are exactly what mislead the answer.

KEEP a fragment when it:
  - matches the question's constraints and could confirm the right option; OR
  - directly helps RULE OUT an option; OR
  - you genuinely cannot tell whether it is on- or off-target. (A clear constraint violation is NOT
    "cannot tell" — drop it. Only true ambiguity is kept.)

Guard against over-deletion: do NOT drop a fragment merely because it is inconvenient for some
option or because you are guessing the answer — drop only for a concrete off-topic or
constraint-violation reason you can name.

Output ONLY this JSON — list the fragments to DROP, each with the concrete reason. Return
{"drop": []} if none should be dropped. No prose, no fences:
  {"drop": [{"i": <fragment index>, "why": "<off-topic, or the exact constraint it violates>"}]}"""


def _filter_supporting(
    question: str,
    candidates: List[str],
    texts: List[str],
    llm,
) -> List[int]:
    """Return the sorted indices of ``texts`` to KEEP.

    The model is asked only to name the fragments to DROP (each with a reason), so the default is
    to keep — a structural bias against over-pruning. Never filters when ≤1 fragment, never drops
    every fragment, and on any parse failure keeps all (conservative, protects recall).
    """
    n = len(texts)
    if n <= 1:
        return list(range(n))
    opts  = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(candidates))
    frags = "\n".join(f"[{i}] {t[:500]}" for i, t in enumerate(texts))
    user  = (f"Question: {question}\n\nOptions:\n{opts}\n\n"
             f"Fragments:\n{frags}\n\nReturn the JSON now.")
    try:
        raw    = llm.chat(
            [{"role": "system", "content": _FILTER_SYS},
             {"role": "user", "content": user}],
            max_new_tokens=500, enable_thinking=False)
        parsed = extract_json(raw) or {}
        drop   = parsed.get("drop")
        if isinstance(drop, list):
            drop_idx = set()
            for d in drop:
                v = d.get("i") if isinstance(d, dict) else d
                if isinstance(v, (int, float)) and 0 <= int(v) < n:
                    drop_idx.add(int(v))
            keep = [i for i in range(n) if i not in drop_idx]
            # Never let the filter wipe out every fragment for this retrieval.
            if keep:
                return keep
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
    round: int
    status: str = "kept"   # "kept" | "dropped_dup" | "dropped_irrelevant"


@dataclass
class RetrievalAttempt:
    round: int
    query: str
    params: dict
    n_retrieved: int
    n_kept: int
    n_dropped_dup: int
    n_dropped_irrelevant: int
    failure_reason: Optional[str]      # None ⇒ productive; else the sentence for the <reason> tag


@dataclass
class ReasoningState:
    round: int
    label: str
    rubric_score: Optional[float]
    per_option_scores: dict
    missing_evidence_analysis: Optional[str]


# ── Online memory ───────────────────────────────────────────────────────────

# Strategy diagnosis for the <reason> tag — a binary the planner can act on in one read. A
# retrieval is either productive (kept ≥1 new fragment) or brought back NOTHING NEW. The latter
# merges the two mechanically-identical dead ends: the query re-hit already-seen chunks (exact ids
# are excluded upstream in retrieval, so a reworded repeat surfaces here as zero new) OR the few
# that slipped through were near-duplicates of accumulated evidence (cosine dedup). Both mean the
# direction is exhausted — rewording won't help. (Off-target / misleading fragments are the
# relevance filter's job, not a reason category.)
_NO_NEW_EVIDENCE = (
    "NO NEW EVIDENCE — this retrieval returned only already-seen or near-duplicate chunks (or "
    "nothing at all); this query direction is exhausted. Do NOT reissue or reword it — target a "
    "DIFFERENT fact, or switch strategy (image if it must be seen, time for a specific span).")


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
        # REVEAL_ABLATE_ONLINE=1 (Online-Memory ablation): degrade to a BARE EVIDENCE STORE —
        # admit() still dedups + accumulates + tracks seen_ids (the evidence cache the answerer
        # needs), but the failure-diagnosed trajectory, the reasoning state, and the relevance
        # filter are all switched off. Nothing here changes the answer; it removes the state the
        # planner would otherwise be fed (see reveal.py) and keeps the ablation rung clean.
        # Match reveal.py's guard exactly: additive field-study mode wins over ablate
        # so the two never disagree about whether the store is bare.
        self.ablate = (os.environ.get("REVEAL_ABLATE_ONLINE") == "1"
                       and os.environ.get("REVEAL_ONLINE_ADDITIVE") != "1")

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
        query: str,
        round: int,
        dedup_threshold: float,
        params: Optional[dict] = None,
    ) -> int:
        """Dedup (always) + optional relevance filter, then accumulate. Returns n_kept.

        With ``REVEAL_ONLINE_FILTER`` unset this is exactly the legacy dedup+accumulate, so the
        answer is unchanged. Records one ``RetrievalAttempt`` with a diagnosed failure reason.
        """
        n_retrieved = len(new_ids)

        # 1. redundancy filter vs the current evidence cache
        d_texts, d_ids, d_vecs = _dedup_by_similarity(
            new_texts, new_ids, new_vecs, self.evidence_vecs(), dedup_threshold)
        n_after_dedup = len(d_ids)
        n_dropped_dup = n_retrieved - n_after_dedup

        # 2. relevance / misleading filter (opt-in; never runs under the online ablation)
        n_dropped_irrelevant = 0
        if (not self.ablate and os.environ.get("REVEAL_ONLINE_FILTER") == "1"
                and n_after_dedup > 0):
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
                chunk_id=cid, text=t, v_text=v, source_query=query, round=round))
        self.seen_ids.update(d_ids)
        n_kept = len(d_ids)

        # Online ablation: pure evidence store — skip trajectory recording.
        if self.ablate:
            return n_kept

        # 4. diagnose for the <reason> tag: binary — productive (kept ≥1 new fragment) vs
        #    no-new-evidence (kept nothing: exact-id repeats already excluded upstream surface as
        #    zero retrieved, near-dups get deduped here — both are the same exhausted direction).
        failure_reason = None if n_kept > 0 else _NO_NEW_EVIDENCE
        self.trajectory.append(RetrievalAttempt(
            round=round, query=query, params=params or {},
            n_retrieved=n_retrieved, n_kept=n_kept, n_dropped_dup=n_dropped_dup,
            n_dropped_irrelevant=n_dropped_irrelevant, failure_reason=failure_reason))
        return n_kept

    # -- reasoning state --------------------------------------------------------
    def record_reasoning(self, verdict: dict, round: int) -> None:
        if self.ablate:            # bare evidence store: no reasoning-state trace
            return
        self.reasoning.append(ReasoningState(
            round=round,
            label=verdict.get("label", ""),
            rubric_score=verdict.get("rubric_score"),
            per_option_scores=verdict.get("rubric_criteria_scores") or {},
            missing_evidence_analysis=verdict.get("missing_evidence_analysis"),
        ))

    # -- planner feedback -------------------------------------------------------
    def trajectory_notes(self) -> str:
        """Per-attempt OUTCOME for the planner: one ``<reason>`` per retrieval, in order.

        The queries themselves are shown by the planner's history_query block (the ``Already tried``
        list — the proven repeat-preventer), so the trajectory carries only the OUTCOME, not a
        duplicate ``<query>``: productive (added new evidence) vs an exhausted direction. The
        REVEAL_OL_REASON gate lives at the single call site (reveal.py's ``_online_on``), which
        passes ``None`` instead of calling this when reason is off — so no gating here.
        ``method``/``query``/``outcome``/``counts``/``unknown``/``scores`` were dropped from the
        planner feed as noise or dilution; the useful gap lives in the verifier's missing_evidence,
        and the query history in history_query.
        """
        lines = []
        for a in self.trajectory:
            reason = a.failure_reason or f"WORKED — added {a.n_kept} new evidence."
            lines.append(f"  <reason>{reason}</reason>")
        return "\n".join(lines)

    # -- serialization ----------------------------------------------------------
    def summary(self) -> dict:
        """Additive block for the result dict (never read by legacy consumers)."""
        if self.ablate:
            return {"n_evidence": len(self.evidence), "ablated_online": True}
        return {
            "n_evidence": len(self.evidence),
            "n_attempts": len(self.trajectory),
            "attempts": [
                {"round": a.round, "query": a.query,
                 "n_retrieved": a.n_retrieved, "n_kept": a.n_kept,
                 "n_dropped_dup": a.n_dropped_dup, "n_dropped_irrelevant": a.n_dropped_irrelevant,
                 "failure_reason": a.failure_reason}
                for a in self.trajectory
            ],
            "evidence": [
                {"chunk_id": e.chunk_id, "round": e.round,
                 "source_query": e.source_query}
                for e in self.evidence
            ],
        }
