"""VEIL — iterative evidence retrieval main loop.

Planner (iter-0 decomposition + iter≥1 unified planner + query safety nets) lives in
``agents/planner.py``. Verifier lives in ``agents/verifier.py``. This file owns
only the main loop, retrieval primitives, evidence dedup, and the rubric-based
final evidence rerank.

Per-iteration retrieval: each sub-question independently fetches from the
multi-layer pool (L1 detail + L2/L3 summaries) by text-only BGE cosine → de-dup
against accumulated evidence (BGE cosine ≥ ``evidence_dedup_threshold``).
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Set

import numpy as np

from ._keyframes import keyframe_path, keyframe_paths, load_keyframe_pil
from src.build_memory.core.schema import MemoryBank
from src.agents.planner import Planner
from src.agents.verifier import Verifier, get_rubric_dict


# ── Broadcast (uniform timeline sampling) ────────────────────────────────────

def _broadcast_retrieve(
    bank: MemoryBank,
    seen_ids: Set[int],
    n: int = 8,
) -> tuple[list, list, list]:
    """Uniformly sample n unseen chunks across the video timeline."""
    texts_ev  = bank.memory_texts(with_time=True, with_asr=True)
    idx_map   = {c.chunk_id: i for i, c in enumerate(bank.chunks)}
    unseen    = [c for c in bank.chunks if c.chunk_id not in seen_ids]
    if not unseen:
        return [], [], []
    unseen.sort(key=lambda c: c.chunk_id)
    step   = max(1, len(unseen) // n)
    chosen = unseen[::step][:n]
    texts, ids, vecs = [], [], []
    for c in chosen:
        i = idx_map.get(c.chunk_id)
        if i is not None:
            texts.append(texts_ev[i])
            ids.append(c.chunk_id)
            vecs.append(c.v_text or [])
    return texts, ids, vecs


# ── Rubric-based final reranking ─────────────────────────────────────────────

def _rerank_by_rubric(
    question: str,
    candidates: List[str],
    evidence_texts: List[str],
    rubric: dict,
    llm,
) -> List[int]:
    """Ask LLM to rank evidence chunks by rubric relevance. Returns sorted indices (best first)."""
    if len(evidence_texts) <= 1:
        return list(range(len(evidence_texts)))

    from src.agents.verifier import _format_rubric_as_text
    opts    = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(candidates))
    ev_fmt  = "\n".join(f"[E{i+1}] {t[:300]}" for i, t in enumerate(evidence_texts))
    rubric_text = _format_rubric_as_text(rubric)

    user = (
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
        f"Rubric:\n{rubric_text}\n\n"
        f"Evidence chunks:\n{ev_fmt}\n\n"
        "Rank the evidence chunks from MOST to LEAST useful for satisfying the rubric and answering the question.\n"
        "Output ONLY a JSON array of 0-based indices, e.g.: [2, 0, 3, 1]\n"
        "Every index must appear exactly once."
    )
    messages = [{"role": "user", "content": user}]
    raw = llm.chat(messages, max_new_tokens=120, enable_thinking=False)

    import json as _json
    m = re.search(r'\[[\d,\s]+\]', raw)
    if m:
        try:
            order = _json.loads(m.group())
            n = len(evidence_texts)
            seen: set = set()
            valid: list = []
            for i in order:
                if isinstance(i, int) and 0 <= i < n and i not in seen:
                    seen.add(i)
                    valid.append(i)
            missing = [i for i in range(n) if i not in seen]
            return valid + missing
        except Exception:
            pass
    return list(range(len(evidence_texts)))


# ── Chunk-level evidence dedup ───────────────────────────────────────────────

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


# ── Query-driven retrieval (per sub-question) ────────────────────────────────

def _query_retrieve(
    query: str,
    bank: MemoryBank,
    embedder,
    coarse_top_k: int,
    final_top_k: int,
    exclude_ids: Set[int],
) -> tuple[List[str], List[int], List[List[float]]]:
    """Return (evidence_texts, chunk_ids, v_text_vecs), skipping already-seen chunks.
    Text-only BGE retrieval."""
    texts    = bank.memory_texts()
    texts_ev = bank.memory_texts(with_time=True, with_asr=True)
    q_vec  = embedder.encode([query])[0]
    if bank.chunks[0].v_text:
        doc_vecs = np.array([c.v_text for c in bank.chunks], dtype=np.float32)
    else:
        doc_vecs = embedder.encode(texts)
    scores = doc_vecs @ q_vec

    k      = min(coarse_top_k, len(scores))
    coarse = np.argpartition(-scores, k - 1)[:k]
    coarse = coarse[np.argsort(-scores[coarse])]

    final_idx = list(coarse[:final_top_k])

    new_idx = [i for i in final_idx if bank.chunks[i].chunk_id not in exclude_ids]
    return (
        [texts_ev[i] for i in new_idx],
        [bank.chunks[i].chunk_id for i in new_idx],
        [bank.chunks[i].v_text for i in new_idx],
    )


# ── Multi-layer pyramid retrieval ────────────────────────────────────────────

def _multi_pool_retrieve(
    query: str,
    bank: "MemoryBank",
    embedder,
    coarse_top_k: int,
    final_top_k: int,
    exclude_ids: Set[int],
    l2_cap: int = 4,
) -> tuple[List[str], List[int], List[List[float]]]:
    """Retrieve from the L1 detail pool + L2/L3 summary pools, merged by score.

    Pool sizes are env-overridable so summary depth can be swept without touching
    the CLI (VEIL's env-override pattern):

      VEIL_MP_L1_CAP   L1 detail chunks entering the pool (default = coarse_top_k)
      VEIL_MP_L2_CAP   L2 summaries entering the pool (default 4)
      VEIL_MP_L3_CAP   L3 summaries entering the pool (default 0 = off)

    Definitive config: L1=8 / L2=4 / L3=2.
    """
    if not bank.l2_chunks:
        return _query_retrieve(query, bank, embedder, coarse_top_k, final_top_k,
                               exclude_ids)

    def _envi(name, default):
        try:
            return int(os.environ.get(name, "").strip() or default)
        except ValueError:
            return default

    l1_pool_cap = _envi("VEIL_MP_L1_CAP", coarse_top_k)
    l2_cap = _envi("VEIL_MP_L2_CAP", l2_cap)
    l3_cap = _envi("VEIL_MP_L3_CAP", 0)
    final_keep = final_top_k

    q_vec = embedder.encode([query])[0]

    # L1 pool
    l1_vecs = np.array([c.v_text for c in bank.chunks], dtype=np.float32)
    l1_scores = l1_vecs @ q_vec
    l1_order = np.argsort(-l1_scores)[:l1_pool_cap]

    # Merge by score
    def _l1_ev(c):
        parts = [f"[{c.start_time:.0f}s-{c.end_time:.0f}s]", c.memory_text]
        if getattr(c, "asr", "").strip():
            parts.append(f"Speech: {c.asr}")
        return "\n".join(parts)

    candidates = []
    for i in l1_order:
        c = bank.chunks[i]
        if c.chunk_id not in exclude_ids:
            candidates.append((float(l1_scores[i]), c, _l1_ev(c)))

    # L2/L3 summary pools
    for layer, cap in ((bank.l2_chunks, l2_cap),
                       (bank.l3_chunks, l3_cap)):
        if not layer or cap <= 0:
            continue
        vecs = np.array([c.v_text for c in layer], dtype=np.float32)
        scores = vecs @ q_vec
        for i in np.argsort(-scores)[:cap]:
            c = layer[int(i)]
            if c.chunk_id not in exclude_ids:
                candidates.append((float(scores[int(i)]), c, c.memory_text))

    candidates.sort(key=lambda x: -x[0])
    selected = candidates[:final_keep]
    return (
        [ev for _, _, ev in selected],
        [c.chunk_id for _, c, _ in selected],
        [c.v_text for _, c, _ in selected],
    )


# ── Main loop ────────────────────────────────────────────────────────────────

def run_veil(
    question:      str,
    candidates:    List[str],
    bank:          MemoryBank,
    embedder,
    answerer,
    llm,
    task_type:     Optional[str] = None,
    coarse_top_k:  int           = 8,
    final_top_k:   int           = 8,
    max_iter:      int           = 3,
    query_history_dedup_threshold:  float = 0.9,
    keyframe_dir                 = None,
    kf_dedup_threshold:    float = 0.9,
    evidence_dedup_threshold:    float = 0.9,
    query_evidence_dedup_threshold: float | None = 0.9,
    answer_evidence_cap:   int | None   = None,
    answer_keyframe_cap:   int | None   = 16,
    verifier_evidence_cap: int | None   = None,
    rubric_judgment:       bool         = True,
    rubric_rerank:         bool         = False,
    verifier_implicit:     bool         = False,
) -> dict:
    """Iter-0 sub-question decomposition + iter ≥1 unified-planner repair.

    Each sub-question independently retrieves ``final_top_k`` chunks (no budget splitting);
    iter ≥1 query generation goes through Jaccard (``query_history_dedup_threshold``) + BGE drift
    (``query_evidence_dedup_threshold``) filters; if all targeted queries are filtered, the iter
    falls back to broadcast.
    """
    verifier = Verifier(llm)
    planner  = Planner(llm)
    rubric   = get_rubric_dict(question, task_type)

    all_evidence_texts: List[str]         = []
    all_chunk_ids:      List[int]         = []
    all_ev_vecs:        List[List[float]] = []
    all_keyframes:      List              = []
    all_kf_vecs:        List[List[float]] = []
    all_kf_chunk_ids:   List[int]         = []
    all_kf_ts:          List[float]       = []
    seen_ids:           Set[int]          = set()
    plan_history:       List[List[str]]   = [[question]]  # seed with original question for dedup
    iterations:         List[dict]        = []
    chunk_by_id = {c.chunk_id: c for c in bank.chunks}
    # Include L2 chunks so keyframe lookup doesn't skip them (L2 chunks have no
    # frames, so they naturally produce no keyframes).
    chunk_by_id.update({c.chunk_id: c for c in bank.l2_chunks})

    # ── Iter-0 plan: sub-question decomposition ──────────────────────────────
    current_plan: dict = planner.decompose_iter0(question, candidates)
    plan_history.append(current_plan["queries"])
    last_verdict: dict = {}

    for it in range(max_iter):

        # ── Plan (iter ≥ 1): generate new queries from last verdict + history ─
        if it > 0:
            verdict_for_planning = last_verdict
            next_plan, m_desc = planner.plan_next(
                question, candidates, all_evidence_texts, verdict_for_planning, plan_history,
                rubric_judgment=rubric_judgment,
            )
            next_plan = planner.filter_targeted_queries(
                next_plan, plan_history, embedder, all_ev_vecs,
                dedup_thresh=query_history_dedup_threshold,
                drift_threshold=query_evidence_dedup_threshold,
            )
            plan_history.append(next_plan["queries"])
            current_plan = next_plan

        do_broadcast      = (current_plan["strategy"] == "broadcast")
        queries_this_iter = current_plan["queries"] if not do_broadcast else []

        # ── Retrieve ──────────────────────────────────────────────────────────
        new_texts, new_ids, new_vecs = [], [], []

        if do_broadcast:
            new_texts, new_ids, new_vecs = _broadcast_retrieve(
                bank, seen_ids, n=final_top_k,
            )
            iter_query = "[broadcast]"
        else:
            for q in queries_this_iter:
                t, i, v = _multi_pool_retrieve(
                    q, bank, embedder,
                    coarse_top_k, final_top_k, seen_ids,
                )
                new_texts.extend(t); new_ids.extend(i); new_vecs.extend(v)
                seen_ids.update(i)
            iter_query = (" | ".join(queries_this_iter) if len(queries_this_iter) > 1
                          else (queries_this_iter[0] if queries_this_iter else ""))

        new_texts, new_ids, new_vecs = _dedup_by_similarity(
            new_texts, new_ids, new_vecs, all_ev_vecs, evidence_dedup_threshold)
        seen_ids.update(new_ids)
        all_evidence_texts.extend(new_texts)
        all_chunk_ids.extend(new_ids)
        all_ev_vecs.extend(new_vecs)

        # ── Keyframes (visual de-dup against all accumulated) ────────────────
        # Cross-chunk dedup decision based on chunk-level v_visual (one vector
        # per chunk). If chunk is accepted, append up to one frame from its
        # already-deduped within-chunk frame set.
        if keyframe_dir is not None:
            for cid in new_ids:
                c = chunk_by_id.get(cid)
                if c is None:
                    continue
                if c.v_visual and all_kf_vecs:
                    v = np.array(c.v_visual, dtype=np.float32)
                    if float(np.max(np.array(all_kf_vecs, dtype=np.float32) @ v)) >= kf_dedup_threshold:
                        continue
                paths = keyframe_paths(keyframe_dir, bank.video_id, c.chunk_id,
                                       cap=1, chunk=c)
                added = False
                for p in paths:
                    img = load_keyframe_pil(p)
                    if img is None:
                        continue
                    all_keyframes.append(img)
                    all_kf_chunk_ids.append(cid)
                    all_kf_ts.append(float(c.keyframe_ts or c.start_time or 0.0))
                    added = True
                if added:
                    all_kf_vecs.append(c.v_visual if c.v_visual else [])

        # ── Verifier ──────────────────────────────────────────────────────────
        if verifier_evidence_cap is not None and len(all_evidence_texts) > verifier_evidence_cap:
            order = _rerank_by_rubric(question, candidates, all_evidence_texts, rubric, llm)
            top_k = min(verifier_evidence_cap, len(all_evidence_texts))
            ev_for_verifier = [all_evidence_texts[i] for i in order[:top_k]]
        else:
            ev_for_verifier = all_evidence_texts

        kf_for_verifier = all_keyframes
        if answer_keyframe_cap is not None and len(kf_for_verifier) > answer_keyframe_cap:
            kf_for_verifier = kf_for_verifier[:answer_keyframe_cap]

        last_verdict = verifier.verify(question, candidates, ev_for_verifier, rubric,
                                  keyframe_images=kf_for_verifier,
                                  implicit=verifier_implicit)

        # ── Trace ────────────────────────────────────────────────────────────
        iterations.append({
            "iter":                      it,
            "query":                     iter_query,
            "new_ids":                   new_ids,
            "verdict":                   last_verdict["label"],
            "option_judgment":           last_verdict.get("option_judgment"),
            "unknown_options":           last_verdict.get("unknown_options"),
            "option_rubric_scores":      last_verdict.get("option_rubric_scores"),
            "missing_evidence_analysis": last_verdict.get("missing_evidence_analysis") or None,
            "plan_strategy":             current_plan.get("strategy"),
            "plan_reasoning":            current_plan.get("reasoning"),
            "n_keyframes_accumulated":   len(all_keyframes),
            "n_keyframes_to_verifier":   len(kf_for_verifier),
            "n_evidence_to_verifier":    len(ev_for_verifier),
        })

        if last_verdict["label"] in ("sufficient", "SUFFICIENT", "FULLY_SUFFICIENT", "ANSWER_SUFFICIENT"):
            break

    # ── Answer ───────────────────────────────────────────────────────────────
    ev_texts = all_evidence_texts
    ev_ids   = all_chunk_ids
    if answer_evidence_cap is not None:
        cap      = min(answer_evidence_cap, len(ev_texts))
        ev_texts = ev_texts[:cap]
        ev_ids   = ev_ids[:cap]

    if rubric_rerank and len(ev_texts) > 1:
        order    = _rerank_by_rubric(question, candidates, ev_texts, rubric, llm)
        ev_texts = [ev_texts[i] for i in order]
        ev_ids   = [ev_ids[i]   for i in order]

    kf_for_answer        = all_keyframes
    kf_chunk_ids_for_ans = all_kf_chunk_ids
    kf_ts_for_ans        = all_kf_ts

    if answer_keyframe_cap is not None and len(kf_for_answer) > answer_keyframe_cap:
        kf_for_answer        = kf_for_answer[:answer_keyframe_cap]
        kf_chunk_ids_for_ans = kf_chunk_ids_for_ans[:answer_keyframe_cap]
        kf_ts_for_ans        = kf_ts_for_ans[:answer_keyframe_cap]

    result = answerer.answer(question, candidates, ev_texts,
                             keyframe_images=kf_for_answer,
                             keyframe_chunk_ids=kf_chunk_ids_for_ans,
                             keyframe_ts=kf_ts_for_ans,
                             evidence_chunk_ids=ev_ids,
                             question_first=True)
    result["n_keyframes_to_answer"] = len(kf_for_answer)
    result["n_evidence_to_answer"]  = len(ev_texts)
    result["evidence_chunk_ids"] = ev_ids
    result["evidence_texts"]     = ev_texts
    result["trace_iters"]        = iterations
    return result
