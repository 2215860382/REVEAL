"""Internal pipeline used by the named ablation entrypoints."""
from __future__ import annotations

import json
import os
from typing import List, Optional

from experiments.core.reveal import (
    _multi_pool_retrieve as _core_multi_pool_retrieve,
    _timeline_scan_retrieve,
)
from src.build_memory.core.schema import MemoryBank
from experiments.reveal_27b import OnlineMemory
from experiments.ablations.online import (
    _ONLINE_CONTROLLER_PROMPT,
    _ONLINE_MEMORY_PROMPT,
    build_online_context,
    render_online_hint,
)
from experiments.ablations.offline import _OFFLINE_CONTROLLER_PROMPT
from src.eval.parse_answer import parse_letter
from src.utils.extract_json import extract_json


_active_offline_controller_prompt = _OFFLINE_CONTROLLER_PROMPT


def set_offline_controller_prompt(prompt: str) -> None:
    global _active_offline_controller_prompt
    _active_offline_controller_prompt = prompt


# ── Ablation loops ────────────────────────────────────────────────────────────

def _multi_pool_retrieve(*args, **kwargs):
    """Apply the memory-layer capacities selected by the ablation entrypoint."""
    kwargs.update(
        l1_pool_cap=int(os.environ.get("REVEAL_MP_L1_CAP", "8")),
        l2_cap=int(os.environ.get("REVEAL_MP_L2_CAP", "4")),
        l3_cap=int(os.environ.get("REVEAL_MP_L3_CAP", "2")),
    )
    return _core_multi_pool_retrieve(*args, **kwargs)

def _controller_reason(model, question, candidates, evidence, *, prompt_template,
                           hint="",
                           keyframe_images=(), force_answer=False,
                           required_action=None):
    """Choose SEARCH or ANSWER with the Online controller prompt."""
    options = "\n".join(
        f"({chr(65 + i)}) {candidate}" for i, candidate in enumerate(candidates))
    fragments = (
        "\n".join(f"[E{i + 1}] {text}" for i, text in enumerate(evidence))
        or "(no evidence retrieved yet)"
    )
    if force_answer or required_action == "answer":
        constraint = "\nThe retrieval budget is exhausted. You must return the ANSWER object now."
    elif required_action == "search":
        constraint = (
            "\nNo evidence has been retrieved yet. Return SEARCH with a strategy and queries."
        )
    elif required_action == "decide":
        constraint = (
            "\nThe current Online Memory is supplied below. Return ANSWER if it supports a "
            "confident choice; otherwise return SEARCH targeting the exact missing evidence."
        )
    else:
        constraint = ""
    prompt = (
        prompt_template
        + constraint
        + f"\n\nQuestion: {question}\n\nOptions:\n{options}\n\n"
        + hint
        + f"Evidence:\n{fragments}\n\nReturn the JSON object now."
    )
    raw = model.chat_with_frames(
        keyframe_images, prompt, max_new_tokens=8192, enable_thinking=False)
    parsed = extract_json(raw) or {}
    decision = str(parsed.get("decision") or "").strip().lower()
    answer = parse_letter(str(parsed.get("answer") or ""), len(candidates)) or ""
    strategy = str(parsed.get("strategy") or "Text Retrieval").strip().lower()
    strategy = {
        "text retrieval": "query", "text": "query", "targeted": "query",
        "semantic": "query", "query": "query",
        "uniform sampling": "scan", "uniform": "scan", "scan": "scan",
        "sweep": "scan", "coverage": "scan",
    }.get(strategy, strategy)
    if strategy not in {"query", "scan"}:
        strategy = "query"
    raw_queries = parsed.get("queries")
    if isinstance(raw_queries, list):
        queries = [str(q).strip() for q in raw_queries if str(q).strip()]
    else:
        query = str(parsed.get("query") or "").strip()
        queries = [query] if query else []
    if answer:
        decision = "answer"
    elif decision != "search":
        decision = "search"
    return {
        "decision": decision,
        "answer": answer,
        "strategy": strategy,
        "queries": queries,
        "raw": raw,
    }


def _update_online_memory(model, question, candidates, evidence, previous_state, *,
                          hint="", history_queries=(), retrieval_quality=()):
    """Run the Online arm's memory-only prompt on the same mainline answer VLM."""
    options = "\n".join(
        f"({chr(65 + i)}) {candidate}" for i, candidate in enumerate(candidates))
    fragments = (
        "\n".join(f"[E{i + 1}] {text}" for i, text in enumerate(evidence))
        or "(no evidence retrieved yet)"
    )
    prompt = (
        _ONLINE_MEMORY_PROMPT
        + f"\n\nQuestion: {question}\n\nOptions:\n{options}\n\n"
        + hint
        + "Previous Online Memory:\n"
        + (json.dumps(previous_state, ensure_ascii=False) if previous_state else "(empty)")
        + "\n\nAttempted query history (copy exactly):\n"
        + json.dumps(list(history_queries), ensure_ascii=False)
        + "\n\nRetrieval-quality records (copy exactly):\n"
        + json.dumps(list(retrieval_quality), ensure_ascii=False)
        + "\n\nAccumulated evidence:\n"
        + fragments
        + "\n\nReturn the Online Memory JSON now."
    )
    raw = model.chat_with_frames([], prompt, max_new_tokens=700, enable_thinking=False)
    parsed = extract_json(raw) or {}
    missing = str(parsed.get("missing_evidence_analysis") or "").strip()
    if not missing:
        raise RuntimeError("Online Memory prompt returned an incomplete memory update")
    state = {
        "missing_evidence_analysis": missing,
        # Use the code-owned values so malformed model copies cannot corrupt
        # query suppression or retrieval diagnostics.
        "history_queries": list(history_queries),
        "retrieval_quality": list(retrieval_quality),
    }
    return state, raw


def _run_controller_loop(
    question, candidates, bank, embedder, answerer, llm,
    coarse_top_k, final_top_k, max_iter, max_time,
    *, siglip_embedder=None, video_path=None, keyframe_dir=None,
    evidence_dedup_threshold=0.9,
    controller_prompt=_ONLINE_CONTROLLER_PROMPT,
    enable_online_memory=True,
):
    """Shared retrieval/answer loop; Online optionally adds memory consolidation."""
    om = OnlineMemory(llm, question, candidates)
    model = answerer.model
    plan_history = []
    iterations = []
    q_rounds = 0
    online_state = {}

    action = _controller_reason(
        model, question, candidates, [], prompt_template=controller_prompt,
        required_action="search")
    while action["decision"] == "search" and q_rounds < max_iter:
        retrieval_target_gap = online_state.get("missing_evidence_analysis")
        requested_strategy = action["strategy"]
        strategy = requested_strategy
        queries = action["queries"]
        fallback_reason = None
        do_scan = strategy == "scan"
        q_rounds += 1
        if strategy == "query" and not queries:
            queries = [question]
        if do_scan:
            queries = []
        plan_history.append(list(queries))

        new_texts, new_ids, new_vecs = [], [], []
        if do_scan:
            new_texts, new_ids, new_vecs = _timeline_scan_retrieve(
                bank, om.seen_ids, n=final_top_k, max_time=max_time)
            iter_query = "[scan]"
        else:
            for query in queries:
                texts, ids, vecs = _multi_pool_retrieve(
                    query, bank, embedder, coarse_top_k, final_top_k,
                    om.seen_ids, max_time=max_time)
                new_texts.extend(texts)
                new_ids.extend(ids)
                new_vecs.extend(vecs)
                om.seen_ids.update(ids)
            iter_query = " | ".join(queries)

        n_before = len(om.evidence)
        om.admit(
            new_texts, new_ids, new_vecs, query=iter_query,
            round=len(iterations), dedup_threshold=evidence_dedup_threshold)
        kept_ids = om.evidence_ids()[n_before:]

        if enable_online_memory:
            context = build_online_context(
                plan_history=plan_history,
                verdict={},
                memory=om,
                bank=bank,
                max_time=max_time,
                include_coverage=True,
            )
            hint = render_online_hint(
                context, ("history", "retrieval_quality", "coverage_gap",
                          "missing_evidence_analysis"))
            history_queries = [query for queries in plan_history for query in queries]
            retrieval_quality = [
                attempt.retrieval_quality for attempt in om.trajectory]
            online_state, memory_raw = _update_online_memory(
                model, question, candidates, om.evidence_texts(), online_state,
                hint=hint,
                history_queries=history_queries,
                retrieval_quality=retrieval_quality)
            controller_hint = (
                hint
                + "Current model-derived Online Memory:\n"
                + json.dumps(online_state, ensure_ascii=False, indent=2)
                + "\n\n"
            )
            coverage_gap = context.coverage_gap
        else:
            controller_hint = ""
            memory_raw = None
            coverage_gap = None
        force_answer = q_rounds >= max_iter
        next_action = _controller_reason(
            model, question, candidates, om.evidence_texts(),
            prompt_template=controller_prompt, hint=controller_hint,
            force_answer=force_answer,
            required_action="decide")
        iterations.append({
            "verdict": "ANSWER" if next_action["decision"] == "answer" else "SEARCH",
            "iter": len(iterations),
            "evidence": om.evidence_texts(),
            "plan": {
                "think": (
                    "unified_online_reasoner" if enable_online_memory
                    else "unified_offline_reasoner"),
                "strategy": requested_strategy,
                "queries": queries,
            },
            "retrieval": {
                "executed_strategy": strategy,
                "fallback_reason": fallback_reason,
                "query": iter_query,
                "new_ids": kept_ids,
                "n_keyframes_accumulated": 0,
                "n_evidence_to_reasoner": len(om.evidence),
            },
            "online_memory_update": (
                dict(online_state) if enable_online_memory else None),
            "missing_evidence_analysis": (
                online_state["missing_evidence_analysis"]
                if enable_online_memory else None),
            "retrieval_target_gap": retrieval_target_gap,
            "coverage_gap": coverage_gap,
            "online_memory_raw": memory_raw,
            "reasoner_raw": next_action["raw"],
        })
        action = next_action

    if not action["answer"]:
        action = _controller_reason(
            model, question, candidates, om.evidence_texts(),
            prompt_template=controller_prompt, force_answer=True)
    if not action["answer"]:
        raise RuntimeError(
            "Unified Online reasoner did not return a valid answer letter; "
            f"raw={action['raw']!r}"
        )

    online_memory = None
    if enable_online_memory:
        online_memory = om.summary()
        online_memory["analyses"] = [
            {
                "round": item["iter"],
                "online_memory_update": item["online_memory_update"],
                "missing_evidence_analysis": item["missing_evidence_analysis"],
            }
            for item in iterations
        ]
    return {
        "answer": action["answer"],
        "raw": action["raw"],
        "evidence_chunk_ids": om.evidence_ids(),
        "evidence_texts": om.evidence_texts(),
        "n_keyframes_to_answer": 0,
        "n_evidence_to_answer": len(om.evidence),
        "trace_iters": iterations,
        "online_memory": online_memory,
        "evidence_state": None,
        "evidence_state_style": None,
    }


def run_reveal(
    question:      str,
    candidates:    List[str],
    bank:          MemoryBank,
    embedder,
    answerer,
    llm,
    siglip_embedder              = None,
    task_type:     Optional[str] = None,
    coarse_top_k:  int           = 8,
    final_top_k:   int           = 8,
    max_iter:      int           = 3,
    query_history_dedup_threshold:  float = 0.9,
    keyframe_dir                 = None,
    video_path: Optional[str]    = None,
    evidence_dedup_threshold:    float = 0.9,
    query_evidence_dedup_threshold: float | None = 0.9,
    rubric_judgment:       bool         = True,
    rubric_rerank:         bool         = False,
    max_time:              float | None = None,
    use_evidence_state:    bool         = False,
) -> dict:
    """Run one of the unified controller ablation arms."""
    # Every named ablation is verifier-free and uses the same single-model
    # loop.  Planner → query/image/time → rubric verifier → answerer
    # exists only in experiments/core/reveal.py (the paper mainline).
    if os.environ.get("REVEAL_UNIFIED_ONLINE") == "1":
        return _run_controller_loop(
            question, candidates, bank, embedder, answerer, llm,
            coarse_top_k, final_top_k, max_iter, max_time,
            siglip_embedder=siglip_embedder, video_path=video_path,
            keyframe_dir=keyframe_dir,
            evidence_dedup_threshold=evidence_dedup_threshold,
        )
    if os.environ.get("REVEAL_UNIFIED_OFFLINE") == "1":
        return _run_controller_loop(
            question, candidates, bank, embedder, answerer, llm,
            coarse_top_k, final_top_k, max_iter, max_time,
            siglip_embedder=siglip_embedder, video_path=video_path,
            keyframe_dir=keyframe_dir,
            evidence_dedup_threshold=evidence_dedup_threshold,
            controller_prompt=_active_offline_controller_prompt,
            enable_online_memory=False,
        )
    raise RuntimeError(
        "No ablation controller selected; use online.py, offline.py, iter.py, or base.py"
    )
