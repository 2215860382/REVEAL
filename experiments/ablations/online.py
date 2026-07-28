"""Online ablation: mainline retrieval and answer model, two focused prompts.

One VLM controller chooses either a mainline retrieval strategy or the final
answer. A second prompt only consolidates retrieved evidence into Online Memory.
Both prompts use the same mainline answer VLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ablations._entrypoint import run_arm


_ONLINE_CONTROLLER_PROMPT = """\
You control retrieval and answering in a long-video multiple-choice QA system. Use the question,
options, current Online Memory, and accumulated evidence to choose exactly one action:

SEARCH when more evidence is needed. Choose one of the strategies:
- Text Retrieval: provide focused semantic queries for specific facts or events.
- Uniform Sampling: sample unseen timeline regions for broad coverage; queries must be [].

ANSWER when the current information supports a choice. Return the correct option letter.

Return only one JSON object:
{"decision":"search","strategy":"Text Retrieval|Uniform Sampling","queries":["..."]}
or
{"decision":"answer","answer":"A|B|C|D"}
"""

_ONLINE_MEMORY_PROMPT = """\
You maintain Online Memory for one long-video multiple-choice question. Update it from the question,
options, previous Online Memory, accumulated retrieved evidence, and supplied retrieval metadata.

The updated memory must contain:
1. missing_evidence_analysis: Compare the question and every option against all accumulated
   evidence. State the single most important unresolved fact needed to distinguish the remaining
   plausible options. Make it concrete and directly retrievable. If the evidence already resolves
   the question, output exactly "NONE". Do not treat absence of evidence as contradiction and do
   not invent facts.
2. history_queries: Copy every supplied attempted query exactly, in the original order. Do not
   add, rewrite, merge, remove, or reorder queries.
3. retrieval_quality: Copy every supplied retrieval-quality record exactly, in the original order,
   so each record remains aligned with its retrieval attempt. Do not reinterpret or summarize it.

Use the previous Online Memory only as continuity context. The accumulated evidence and supplied
retrieval metadata are authoritative. Return the complete replacement memory, not a delta. Do not
propose a new query, choose a retrieval strategy, or answer with an option letter.

Return only this JSON object:
{
 "missing_evidence_analysis":"one concrete unresolved fact or NONE",
 "history_queries":["first attempted query","second attempted query"],
 "retrieval_quality":["first retrieval-quality record","second retrieval-quality record"]
}
"""


from dataclasses import dataclass
from typing import Iterable, Set


PLANNER_ONLINE_FIELDS = (
    "history",
    "retrieval_quality",
    "missing_evidence_analysis",
)
ONLINE_HINT_FIELDS = PLANNER_ONLINE_FIELDS + ("coverage_gap",)


@dataclass(frozen=True)
class OnlineContext:
    """One immutable snapshot of the fields fed back after a retrieval round."""

    history: tuple[tuple[str, ...], ...]
    retrieval_quality: str
    missing_evidence_analysis: str | None
    coverage_gap: str | None


def build_online_context(
    *,
    plan_history,
    verdict: dict,
    memory,
    bank,
    max_time: float | None,
    include_coverage: bool,
) -> OnlineContext:
    """Build Online fields from one source of truth for every pipeline."""
    return OnlineContext(
        history=tuple(tuple(str(query) for query in queries) for queries in plan_history),
        retrieval_quality=memory.retrieval_quality_notes(),
        missing_evidence_analysis=(
            str(verdict.get("missing_evidence_analysis")).strip()
            if verdict.get("missing_evidence_analysis") else None
        ),
        coverage_gap=(
            coverage_gap_summary(bank, memory.seen_ids, max_time)
            if include_coverage else None),
    )


def render_online_hint(
    context: OnlineContext,
    fields: Iterable[str],
    *,
    missing_heading: str = "Previously diagnosed missing evidence",
) -> str:
    """Render Online state for reasoner-style prompts in one stable order.

    The planner has its own structured prompt builder, but it receives values
    from :func:`build_online_context` too. Reasoner-style ablations use this
    renderer so they cannot silently rename or reimplement Online fields.
    """
    enabled = frozenset(fields)
    # Evidence state is supplied as the reasoner's compact evidence payload,
    # not repeated inside the textual Online hint.
    unknown = enabled.difference(ONLINE_HINT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown Online fields: {sorted(unknown)}")

    blocks: list[str] = []
    if "history" in enabled and context.history:
        tried = [query for queries in context.history for query in queries]
        if tried:
            blocks.append(
                "Already tried (do NOT repeat):\n"
                + "\n".join(f"- {query}" for query in tried))
    if "retrieval_quality" in enabled and context.retrieval_quality:
        blocks.append(
            "Retrieval quality so far:\n" + context.retrieval_quality)
    if "coverage_gap" in enabled and context.coverage_gap:
        blocks.append("Timeline coverage gap:\n" + context.coverage_gap)
    if (
        "missing_evidence_analysis" in enabled
        and context.missing_evidence_analysis
    ):
        blocks.append(
            f"{missing_heading}:\n" + context.missing_evidence_analysis
        )
    return "\n\n".join(blocks) + ("\n\n" if blocks else "")


def coverage_gap_summary(
    bank,
    retrieved_ids: Set[int],
    max_time: float | None = None,
    max_gaps: int = 3,
) -> str:
    """Return compact timeline coverage and the largest uncovered intervals."""
    chunks = [
        c for c in bank.chunks
        if c.end_time > c.start_time
        and (max_time is None or c.start_time < max_time)
    ]
    if not chunks:
        return "coverage unavailable"

    end = min(max(c.end_time for c in chunks), max_time) if max_time is not None else max(
        c.end_time for c in chunks)
    if end <= 0:
        return "coverage unavailable"

    intervals = sorted(
        (max(0.0, c.start_time), min(end, c.end_time))
        for c in chunks if c.chunk_id in retrieved_ids and c.start_time < end
    )
    merged: list[list[float]] = []
    for start, stop in intervals:
        if stop <= start:
            continue
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, stop in merged:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < end:
        gaps.append((cursor, end))

    covered_sec = sum(stop - start for start, stop in merged)
    coverage_pct = 100.0 * covered_sec / end
    largest = sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True)[:max_gaps]
    gap_text = (
        ", ".join(f"{start:.0f}-{stop:.0f}s" for start, stop in largest)
        if largest else "none")
    return (
        f"{coverage_pct:.1f}% of the available timeline has retrieved L1 evidence; "
        f"largest uncovered intervals: {gap_text}.")


_ENV = {
    "REVEAL_UNIFIED_ONLINE": "1",
    "REVEAL_MP_L1_CAP": "8",
    "REVEAL_MP_L2_CAP": "4",
    "REVEAL_MP_L3_CAP": "2",
}


if __name__ == "__main__":
    run_arm(_ENV, default_max_iter=3)
