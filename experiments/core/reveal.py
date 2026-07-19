"""REVEAL — iterative evidence retrieval main loop.

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

from ._keyframes import keyframe_path, keyframe_paths, load_keyframe_pil, dense_frames_from_original
from src.build_memory.core.schema import MemoryBank
from src.agents.planner import Planner
from src.agents.verifier import Verifier, get_rubric_dict
from src.agents.online_memory import OnlineMemory
from src.utils.extract_json import extract_json


# ── Time-targeted retrieval (verifier-driven temporal repair) ────────────────

def _time_range_retrieve(
    bank: "MemoryBank",
    seen_ids: Set[int],
    ranges: list[dict],
    n: int = 8,
    max_time: float | None = None,
) -> tuple[list, list, list]:
    """Fetch unseen chunks overlapping planner-provided time ranges.

    Empty ranges mean uniform uncovered-time sampling, which is the explicit
    replacement for the old broadcast strategy.
    """
    if not ranges:
        return _broadcast_retrieve(bank, seen_ids, n=n, max_time=max_time)

    texts_ev = bank.memory_texts(with_time=True, with_asr=True)
    idx_map = {c.chunk_id: i for i, c in enumerate(bank.chunks)}
    cand = []
    for c in bank.chunks:
        if c.chunk_id in seen_ids:
            continue
        if max_time is not None and c.end_time > max_time:
            continue
        for r in ranges:
            try:
                a, b = float(r["start_sec"]), float(r["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if c.end_time >= a and c.start_time <= b:
                mid = (c.start_time + c.end_time) / 2.0
                center = (a + b) / 2.0
                cand.append((abs(mid - center), c))
                break
    cand.sort(key=lambda x: x[0])
    texts, ids, vecs = [], [], []
    for _, c in cand[:n]:
        i = idx_map.get(c.chunk_id)
        if i is None:
            continue
        texts.append(texts_ev[i])
        ids.append(c.chunk_id)
        vecs.append(c.v_text or [])
    return texts, ids, vecs


# ── Broadcast (uniform timeline sampling) ────────────────────────────────────

def _broadcast_retrieve(
    bank: MemoryBank,
    seen_ids: Set[int],
    n: int = 8,
    max_time: float | None = None,
) -> tuple[list, list, list]:
    """Uniformly sample n unseen chunks across the video timeline.

    ``max_time`` (EgoLife): drop any chunk whose ``end_time`` is after the
    question's ``query_time`` — future memory is off-limits. None ⇒ no filter.
    """
    texts_ev  = bank.memory_texts(with_time=True, with_asr=True)
    idx_map   = {c.chunk_id: i for i, c in enumerate(bank.chunks)}
    unseen    = [c for c in bank.chunks if c.chunk_id not in seen_ids
                 and (max_time is None or c.end_time <= max_time)]
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
    verifier=None,
) -> List[int]:
    """Ask LLM to rank evidence chunks best-first by rubric-satisfaction +
    question relevance. Returns sorted indices."""
    if len(evidence_texts) <= 1:
        return list(range(len(evidence_texts)))

    opts    = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(candidates))
    ev_fmt  = "\n".join(f"[E{i+1}] {t[:300]}" for i, t in enumerate(evidence_texts))

    # A/B on whether the reranker sees the rubric:
    #  - default: rank by question+options relevance alone (model self-ranks).
    #  - REVEAL_RERANK_RUBRIC=1: inject ONLY the relevance-selected criteria for THIS
    #    question (~3-8, compact). We NEVER inject the full pool (~300 criteria /
    #    ~67K tokens overflows the 48K context).
    rubric_block = ""
    if (os.environ.get("REVEAL_RERANK_RUBRIC") == "1" and verifier is not None
            and isinstance(rubric, dict) and "_legacy_text" not in rubric):
        opts_for_sel = "\n".join(f"  ({chr(65+i)}) {c}" for i, c in enumerate(candidates))
        try:
            sel = verifier._relevant_criteria(question, opts_for_sel, rubric)  # cached per-question
        except Exception:
            sel = set()
        sel_crits = [c for c in (rubric.get("rubric_criteria") or [])
                     if str(c.get("name") or "").strip() in sel]
        if sel_crits:
            from src.agents.verifier import _format_rubric_for_user
            rtext = _format_rubric_for_user({
                **{k: v for k, v in rubric.items() if k != "rubric_criteria"},
                "rubric_criteria": sel_crits})
            rubric_block = f"Rubric (criteria this question hinges on):\n{rtext}\n\n"
    rank_hint = ("Rank the evidence chunks from MOST to LEAST useful for answering the "
                 "question and deciding between the options")
    if rubric_block:
        rank_hint += ", prioritising evidence that satisfies the criteria above"
    user = (
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
        f"{rubric_block}"
        f"Evidence chunks:\n{ev_fmt}\n\n"
        f"{rank_hint}.\n"
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
    max_time: float | None = None,
) -> tuple[List[str], List[int], List[List[float]]]:
    """Return (evidence_texts, chunk_ids, v_text_vecs), skipping already-seen chunks.
    Text-only BGE retrieval. ``max_time`` (EgoLife): exclude chunks whose
    ``end_time`` is after the query time (future memory off-limits)."""
    texts    = bank.memory_texts()
    texts_ev = bank.memory_texts(with_time=True, with_asr=True)
    q_vec  = embedder.encode([query])[0]
    if bank.chunks[0].v_text:
        doc_vecs = np.array([c.v_text for c in bank.chunks], dtype=np.float32)
    else:
        doc_vecs = embedder.encode(texts)
    scores = doc_vecs @ q_vec
    if max_time is not None:  # mask out future chunks before top-k selection
        mask = np.array([c.end_time <= max_time for c in bank.chunks])
        scores = np.where(mask, scores, -np.inf)

    k      = min(coarse_top_k, len(scores))
    coarse = np.argpartition(-scores, k - 1)[:k]
    coarse = coarse[np.argsort(-scores[coarse])]

    final_idx = list(coarse[:final_top_k])

    new_idx = [i for i in final_idx if bank.chunks[i].chunk_id not in exclude_ids
               and (max_time is None or bank.chunks[i].end_time <= max_time)]
    return (
        [texts_ev[i] for i in new_idx],
        [bank.chunks[i].chunk_id for i in new_idx],
        [bank.chunks[i].v_text for i in new_idx],
    )


def _visual_retrieve(
    query: str,
    bank: MemoryBank,
    siglip_embedder,
    final_top_k: int,
    exclude_ids: Set[int],
    max_time: float | None = None,
) -> tuple[List[str], List[int], List[List[float]]]:
    """SigLIP text→frame retrieval (planner `image` strategy): rank L1 chunks by the cosine
    between the query text (SigLIP text encoder) and each chunk's stored keyframe SigLIP vector
    (``v_visual``), then return the top ``final_top_k`` unseen chunks in the SAME shape as
    ``_query_retrieve``. The chunk's caption text still enters evidence; its keyframe is pulled
    downstream. ``max_time`` (EgoLife) masks future chunks. Falls back to empty if no visual
    vectors exist (e.g. text-only bank without SigLIP)."""
    texts_ev = bank.memory_texts(with_time=True, with_asr=True)
    idxs = [i for i, c in enumerate(bank.chunks) if c.v_visual]
    if not idxs:
        return [], [], []
    mat = np.asarray([bank.chunks[i].v_visual for i in idxs], dtype=np.float32)
    q = np.asarray(siglip_embedder.encode_text([query])[0], dtype=np.float32)
    scores = mat @ q                       # both L2-normed → cosine
    order = np.argsort(-scores)
    out_t, out_i, out_v = [], [], []
    for j in order:
        i = int(idxs[j])
        c = bank.chunks[i]
        if c.chunk_id in exclude_ids:
            continue
        if max_time is not None and c.end_time > max_time:
            continue
        out_t.append(texts_ev[i]); out_i.append(c.chunk_id); out_v.append(c.v_text)
        if len(out_t) >= final_top_k:
            break
    return out_t, out_i, out_v


# ── Relative-time window (egocentric) ────────────────────────────────────────

_REL_DAY_WORDS = {"yesterday": 1, "the day before yesterday": 2}


def _multi_pool_retrieve(
    query: str,
    bank: "MemoryBank",
    embedder,
    coarse_top_k: int,
    final_top_k: int,
    exclude_ids: Set[int],
    max_time: float | None = None,
) -> tuple[List[str], List[int], List[List[float]]]:
    """Retrieve from the L1 detail pool + L2/L3 summary pools, merged by score.

    Definitive config: L1=8 / L2=4 / L3=2 candidates enter the merge pool, then the
    top ``final_top_k`` by cosine are returned. The caps default to that config; the
    ``REVEAL_MP_L{1,2,3}_CAP`` env vars exist ONLY to sweep layer depth in ablations
    (e.g. L2=0/L3=0 for the L1-only ablation). The mainline eval sets none of them.

    ``max_time`` (EgoLife): every layer masks out chunks whose ``end_time`` is after
    the query time before its top-k, so future memory never enters the pool. None ⇒
    no filter (lvbench/videomme path unchanged).
    """
    if not bank.l2_chunks:
        return _query_retrieve(query, bank, embedder, coarse_top_k, final_top_k,
                               exclude_ids, max_time=max_time)

    def _envi(name, default):
        try:
            return int(os.environ.get(name, "").strip() or default)
        except ValueError:
            return default

    l1_pool_cap = _envi("REVEAL_MP_L1_CAP", 8)
    l2_cap      = _envi("REVEAL_MP_L2_CAP", 4)
    l3_cap      = _envi("REVEAL_MP_L3_CAP", 2)

    q_vec = embedder.encode([query])[0]

    # L1 pool
    l1_vecs = np.array([c.v_text for c in bank.chunks], dtype=np.float32)
    l1_scores = l1_vecs @ q_vec
    l1_mask = None
    if max_time is not None:  # future chunks can't enter the L1 pool
        l1_mask = np.array([c.end_time <= max_time for c in bank.chunks])
        l1_scores = np.where(l1_mask, l1_scores, -np.inf)

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
        if c.chunk_id not in exclude_ids and (max_time is None or c.end_time <= max_time):
            _b = 0.0
            candidates.append((float(l1_scores[i]), _b, c, _l1_ev(c)))

    # L2/L3 summary pools
    for layer, cap in ((bank.l2_chunks, l2_cap),
                       (bank.l3_chunks, l3_cap)):
        if not layer or cap <= 0:
            continue
        vecs = np.array([c.v_text for c in layer], dtype=np.float32)
        scores = vecs @ q_vec
        if max_time is not None:  # a summary covering future content is off-limits
            l_mask = np.array([c.end_time <= max_time for c in layer])
            scores = np.where(l_mask, scores, -np.inf)
        for i in np.argsort(-scores)[:cap]:
            c = layer[int(i)]
            if c.chunk_id not in exclude_ids and (max_time is None or c.end_time <= max_time):
                candidates.append((float(scores[int(i)]), 0.0, c, c.memory_text))

    candidates.sort(key=lambda x: -x[0])
    selected = candidates[:final_top_k]
    return (
        [ev for _, _, _, ev in selected],
        [c.chunk_id for _, _, c, _ in selected],
        [c.v_text for _, _, c, _ in selected],
    )


# ── Main loop ────────────────────────────────────────────────────────────────

# ── WorldMM-style implicit ablation (REVEAL_WORLDMM_IMPLICIT=1) ────────────────────
#    Removes the verifier module entirely: instead of planner → rubric-verifier →
#    answerer, a SINGLE reasoner call per round looks at the evidence gathered so far
#    and decides, holistically, whether to keep searching (emitting the next query) or
#    to answer (emitting the letter). This is the WorldMM baseline — the loop's stop
#    condition IS the model choosing "answer", with no explicit sufficiency signal.
#    Retrieval reuses ``_multi_pool_retrieve`` so the layer caps (REVEAL_MP_L*_CAP) are
#    the only difference between the M_iter (L1-only) and M_iter+hybrid (8/4/2) rungs.
_WORLDMM_SYS = """\
You are answering a multiple-choice question about a video. You have a searchable memory of the \
video. Each round you are shown the evidence gathered so far and must decide ONE action:
  - "search": the evidence is NOT yet enough to answer confidently — give a short keyword search \
query naming the specific missing fact, event, or time.
  - "answer": the evidence IS enough — give the letter of the correct option.
Return ONLY this JSON, no prose, no markdown fences:
  {"decision": "search", "query": "<keywords>"}
  or
  {"decision": "answer", "answer": "<A/B/C/D>"}"""

_WORLDMM_FORCE = """\
You must ANSWER NOW using only the evidence you already have — no more searching. If the evidence \
is unclear, make the most reasonable inference. Return ONLY this JSON:
  {"decision": "answer", "answer": "<A/B/C/D>"}"""


def _worldmm_reason(llm, question, opts_str, ev_texts, force_answer):
    ev = "\n".join(f"[E{i+1}] {t}" for i, t in enumerate(ev_texts)) or "(no evidence retrieved yet)"
    sys = _WORLDMM_FORCE if force_answer else _WORLDMM_SYS
    user = f"Question: {question}\n\nOptions:\n{opts_str}\n\nEvidence:\n{ev}\n\nReturn the JSON now."
    raw = llm.chat([{"role": "system", "content": sys},
                    {"role": "user", "content": user}],
                   max_new_tokens=300, enable_thinking=False)
    parsed = extract_json(raw) or {}
    dec = str(parsed.get("decision") or parsed.get("action") or "").lower()
    ans = str(parsed.get("answer") or "").strip()
    q   = str(parsed.get("query") or parsed.get("search_query") or "").strip()
    return dec, q, ans, raw


def _run_worldmm(question, candidates, bank, embedder, llm,
                 coarse_top_k, final_top_k, max_iter, max_time):
    """WorldMM-style verifier-free loop; returns the same dict contract as run_reveal."""
    opts_str = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(candidates))
    seen: Set[int]      = set()
    ev_texts: List[str] = []
    ev_ids:   List[int] = []
    iterations: List[dict] = []
    next_query  = question
    pred_letter = ""
    for it in range(max_iter):
        t, i, _v = _multi_pool_retrieve(next_query, bank, embedder, coarse_top_k,
                                        final_top_k, seen, max_time=max_time)
        for tt, ii in zip(t, i):
            if ii not in seen:
                seen.add(ii); ev_texts.append(tt); ev_ids.append(ii)
        force = (it == max_iter - 1)
        dec, q, ans, _raw = _worldmm_reason(llm, question, opts_str, ev_texts, force_answer=force)
        iterations.append({
            "verdict":   "ANSWER" if (dec == "answer" or force) else "SEARCH",
            "evidence":  list(ev_texts),
            "iter":      it,
            "plan":      {"think": "worldmm_implicit", "strategy": "worldmm", "queries": [next_query]},
            "retrieval": {"query": next_query, "new_ids": i,
                          "n_evidence_to_verifier": len(ev_texts)},
            "worldmm_decision": dec, "worldmm_query": q, "worldmm_answer": ans,
        })
        if dec == "answer" or force:
            pred_letter = ans[:1].upper()
            break
        next_query = q or question
    if not pred_letter:  # parse failure on the forced round → one more forced answer
        _, _, ans, _ = _worldmm_reason(llm, question, opts_str, ev_texts, force_answer=True)
        pred_letter = ans[:1].upper()
    return {
        "answer":                pred_letter,
        "raw":                   pred_letter,
        "evidence_chunk_ids":    ev_ids,
        "evidence_texts":        ev_texts,
        "n_keyframes_to_answer": 0,
        "n_evidence_to_answer":  len(ev_texts),
        "trace_iters":           iterations,
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
    kf_dedup_threshold:    float = 0.9,
    evidence_dedup_threshold:    float = 0.9,
    query_evidence_dedup_threshold: float | None = 0.9,
    answer_keyframe_cap:   int | None   = 16,
    rubric_judgment:       bool         = True,
    rubric_rerank:         bool         = False,
    max_time:              float | None = None,
) -> dict:
    """Iter-0 sub-question decomposition + iter ≥1 unified-planner repair.

    Each sub-question independently retrieves ``final_top_k`` chunks (no budget splitting);
    iter ≥1 query generation goes through Jaccard (``query_history_dedup_threshold``) + BGE drift
    (``query_evidence_dedup_threshold``) filters; if all targeted queries are filtered, the iter
    falls back to broadcast.
    """
    # WorldMM-style implicit ablation: verifier module removed, single reasoner decides
    # search-vs-answer each round. Returns early with the same dict contract.
    if os.environ.get("REVEAL_WORLDMM_IMPLICIT") == "1":
        return _run_worldmm(question, candidates, bank, embedder, llm,
                            coarse_top_k, final_top_k, max_iter, max_time)

    verifier = Verifier(llm)
    planner  = Planner(llm)
    rubric   = get_rubric_dict(question, task_type)

    # Unified online memory: evidence cache + retrieval trajectory + reasoning state.
    # ``om.seen_ids`` replaces the old local set; ``om.evidence_*()`` the old all_* lists.
    om = OnlineMemory(llm, question, candidates)
    all_keyframes:      List              = []
    all_kf_vecs:        List[List[float]] = []
    all_kf_chunk_ids:   List[int]         = []
    all_kf_ts:          List[float]       = []
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

    # Round budget: ``max_iter`` counts TEXT (query/time) rounds ONLY. An `image` round is
    # image-only (frames, no text) and does NOT consume the text budget — it's a bonus visual
    # look, capped separately by REVEAL_IMAGE_MAX_ROUNDS. A hard total-round cap prevents an
    # image-only loop from running forever. With no image strategy this is identical to the old
    # ``for it in range(max_iter)`` (text_rounds increments every round).
    text_rounds = 0
    img_rounds  = 0
    _img_max    = int(os.environ.get("REVEAL_IMAGE_MAX_ROUNDS", "2") or 2)
    it          = -1
    _max_total  = max_iter + _img_max + 1
    while text_rounds < max_iter and it + 1 < _max_total:
        it += 1

        # ── Plan (iter ≥ 1): generate new queries from last verdict + history ─
        if it > 0:
            verdict_for_planning = last_verdict
            # Failure-diagnosed trajectory feeds the planner only when opted in; default None
            # keeps the planner prompt (and thus behaviour) byte-identical to the legacy loop.
            traj_notes = (om.trajectory_notes()
                          if os.environ.get("REVEAL_ONLINE_TRAJECTORY_FEEDBACK") == "1" else None)
            next_plan, m_desc = planner.plan_next(
                question, candidates, om.evidence_texts(), verdict_for_planning, plan_history,
                rubric_judgment=rubric_judgment, trajectory_notes=traj_notes,
            )
            next_plan = planner.filter_targeted_queries(
                next_plan, plan_history, embedder, om.evidence_vecs(),
                dedup_thresh=query_history_dedup_threshold,
                drift_threshold=query_evidence_dedup_threshold,
            )
            plan_history.append(next_plan["queries"])
            current_plan = next_plan

        strategy = current_plan.get("strategy", "query")
        if strategy == "targeted":
            strategy = "query"
        if strategy == "broadcast":
            strategy = "time"
        # `image` = visual strategy (REVEAL_VISUAL_STRATEGY): SigLIP text→frame cross-modal
        # retrieval (_visual_retrieve), then contribute captions (REVEAL_VISUAL_RECALL) and/or
        # keyframes (REVEAL_VISUAL_FRAMES) — see the do_image block. Defensive: if the gate is off,
        # the planner shouldn't emit `image`, but treat any stray one as plain text query.
        if strategy == "image" and os.environ.get("REVEAL_VISUAL_STRATEGY") != "1":
            strategy = "query"
        do_time = (strategy == "time")
        # `image` retrieves visually (SigLIP text→frame) when a SigLIP encoder is available and
        # the per-question image-round budget isn't spent; otherwise it degrades to a plain text
        # query so behaviour stays defined.
        do_image = (strategy == "image" and siglip_embedder is not None
                    and img_rounds < _img_max)
        # Round accounting: an image round is a bonus visual look and does NOT consume the text
        # budget; everything else (query/time, or an image that fell back to text) counts.
        if do_image:
            img_rounds += 1
        else:
            text_rounds += 1
        queries_this_iter = current_plan.get("queries", []) if not do_time else []

        # ── Retrieve ──────────────────────────────────────────────────────────
        new_texts, new_ids, new_vecs = [], [], []
        # SigLIP-retrieved chunk ids this round (frames pulled from these, even if additive
        # text retrieval also ran); empty on non-image rounds.
        img_ids_this_round: List[int] = []

        if do_image:
            # `image` round = a bonus visual look (does NOT consume the text-round budget; see the
            # text_rounds counter). SigLIP text→frame picks the visually most-relevant moments; what
            # that contributes is set by two ORTHOGONAL switches (try each alone to ablate):
            #   REVEAL_VISUAL_RECALL=1  → the found chunks' CAPTION TEXT enters evidence (recall
            #       channel: surface gold that text retrieval missed, fed to the strong text
            #       answerer). Downstream _dedup_by_similarity drops captions already present, so
            #       re-hits are safe.
            #   REVEAL_VISUAL_FRAMES    → their keyframes are pulled and fed to the VLM (pixels
            #       channel; see the keyframe block below). Default on = legacy pixels-only.
            # Frames de-dup against everything already fed via ``all_kf_chunk_ids``.
            _vis_recall = os.environ.get("REVEAL_VISUAL_RECALL") == "1"
            for q in (queries_this_iter or [question]):
                _vt, vi, _vv = _visual_retrieve(
                    q, bank, siglip_embedder, final_top_k, set(all_kf_chunk_ids),
                    max_time=max_time)
                img_ids_this_round.extend(vi)
                if _vis_recall:
                    new_texts.extend(_vt); new_ids.extend(vi); new_vecs.extend(_vv)
            iter_query = "[visual] " + (
                " | ".join(queries_this_iter) if queries_this_iter else question)
        elif do_time:
            time_ranges = current_plan.get("time_ranges") or []
            new_texts, new_ids, new_vecs = _time_range_retrieve(
                bank, om.seen_ids, time_ranges, n=final_top_k, max_time=max_time,
            )
            if time_ranges:
                iter_query = " | ".join(
                    f"[time {float(r['start_sec']):.0f}s-{float(r['end_sec']):.0f}s]"
                    for r in time_ranges
                    if "start_sec" in r and "end_sec" in r
                ) or "[time]"
            else:
                iter_query = "[time uniform]"
        else:
            for q in queries_this_iter:
                t, i, v = _multi_pool_retrieve(
                    q, bank, embedder,
                    coarse_top_k, final_top_k, om.seen_ids, max_time=max_time,
                )
                new_texts.extend(t); new_ids.extend(i); new_vecs.extend(v)
                om.seen_ids.update(i)
            iter_query = (" | ".join(queries_this_iter) if len(queries_this_iter) > 1
                          else (queries_this_iter[0] if queries_this_iter else ""))


        # Admit into online memory: redundancy dedup (always) + optional relevance filter
        # (REVEAL_ONLINE_FILTER). With the filter off this is exactly the legacy dedup+accumulate.
        # Records one RetrievalAttempt with a diagnosed outcome (productive/redundant/off_target/empty).
        n_before = len(om.evidence)
        om.admit(new_texts, new_ids, new_vecs,
                 method=strategy, query=iter_query, round=it,
                 dedup_threshold=evidence_dedup_threshold)
        new_ids = om.evidence_ids()[n_before:]   # kept ids this round (keyframes + trace)

        # ── Keyframes (visual de-dup against all accumulated) ────────────────
        # Cross-chunk dedup decision based on chunk-level v_visual (one vector
        # per chunk). If chunk is accepted, append up to one frame from its
        # already-deduped within-chunk frame set.
        # REVEAL_KF_ONLY_ON_IMAGE: frames enter ONLY through the planner's `image` strategy.
        # On a text (query/time) round no frames are attached, so a purely-textual question
        # never gets visual distractors fed to the answerer — the model must first *decide*
        # it needs to look (pick `image`), and only then are the corresponding frames pulled.
        # Default off preserves the legacy one-frame-per-chunk behaviour (helps VideoMME).
        _kf_only_on_image = os.environ.get("REVEAL_KF_ONLY_ON_IMAGE") == "1"
        if keyframe_dir is not None and not (_kf_only_on_image and strategy != "image"):
            # Visual strategy (strategy == "image", REVEAL_VISUAL_STRATEGY): `new_ids` is already
            # ranked by SigLIP text→frame similarity (_visual_retrieve), so pull one frame from
            # each of the top ``REVEAL_IMAGE_KF_CAP`` visually-relevant chunks — i.e. the frames
            # that actually depict what the question asked about — WITHOUT the accumulated
            # visual-dedup (which would drop near-duplicate but on-topic moments). Otherwise
            # (text query round with legacy keyframes on): one deduped frame per new chunk.
            img_focus = (strategy == "image")
            _img_cap = int(os.environ.get("REVEAL_IMAGE_KF_CAP", "6") or 6)
            # image round: frames come from the SigLIP-retrieved chunks only (img_ids_this_round),
            # NOT the text-query chunks in new_ids. The pixels channel is gated by
            # REVEAL_VISUAL_FRAMES (default on = legacy pixels-only); Arm A (recall-only) sets it 0
            # to feed captions without frames. Only governs image-round frames; legacy text-round
            # keyframes (else branch) stay controlled by REVEAL_KF_ONLY_ON_IMAGE / --no-keyframes.
            _vis_frames = os.environ.get("REVEAL_VISUAL_FRAMES", "1") == "1"
            if img_focus and not _vis_frames:
                kf_targets = []
            else:
                kf_targets = img_ids_this_round[:_img_cap] if img_focus else new_ids
            kf_cap = 1
            # REVEAL_DENSE_FRAMES=N: WorldMM-style — pull N frames from the chunk's
            # ORIGINAL 30 s mp4 instead of its single stored keyframe (see
            # dense_frames_from_original). Falls back to the stored keyframe when the
            # source video can't be located (e.g. non-egolife banks, unmatched chunks).
            _dense_n = int(os.environ.get("REVEAL_DENSE_FRAMES", "0") or 0)
            for cid in kf_targets:
                c = chunk_by_id.get(cid)
                if c is None:
                    continue
                if not img_focus and c.v_visual and all_kf_vecs:
                    v = np.array(c.v_visual, dtype=np.float32)
                    if float(np.max(np.array(all_kf_vecs, dtype=np.float32) @ v)) >= kf_dedup_threshold:
                        continue
                imgs = []
                if _dense_n > 0:
                    imgs = dense_frames_from_original(c, _dense_n)
                if not imgs:
                    paths = keyframe_paths(keyframe_dir, bank.video_id, c.chunk_id,
                                           cap=kf_cap, chunk=c)
                    imgs = [im for im in (load_keyframe_pil(p) for p in paths) if im is not None]
                added = False
                for img in imgs:
                    all_keyframes.append(img)
                    all_kf_chunk_ids.append(cid)
                    all_kf_ts.append(float(c.keyframe_ts or c.start_time or 0.0))
                    added = True
                if added:
                    all_kf_vecs.append(c.v_visual if c.v_visual else [])

        # ── Verifier ──────────────────────────────────────────────────────────
        ev_for_verifier = om.evidence_texts()

        kf_for_verifier = all_keyframes
        if answer_keyframe_cap is not None and len(kf_for_verifier) > answer_keyframe_cap:
            kf_for_verifier = kf_for_verifier[:answer_keyframe_cap]

        last_verdict = verifier.verify(question, candidates, ev_for_verifier, rubric,
                                  keyframe_images=kf_for_verifier)
        om.record_reasoning(last_verdict, it)

        # ── Trace ────────────────────────────────────────────────────────────
        _attempt = om.trajectory[-1] if om.trajectory else None
        iterations.append({
            "verdict":                   last_verdict["label"],
            "evidence":                  list(ev_for_verifier),
            "unknown_options":           last_verdict.get("unknown_options") or [],
            "missing_evidence_analysis": last_verdict.get("missing_evidence_analysis") or None,
            "iter":                      it,
            "plan": {
                "think":                 current_plan.get("think") or current_plan.get("reasoning"),
                "strategy":              strategy,
                "queries":               queries_this_iter,
                "time_ranges":           current_plan.get("time_ranges") or [],
            },
            "retrieval": {
                "query":                 iter_query,
                "new_ids":               new_ids,
                "n_keyframes_accumulated": len(all_keyframes),
                "n_keyframes_to_verifier": len(kf_for_verifier),
                "n_evidence_to_verifier":  len(ev_for_verifier),
            },
            "rubric_criteria_scores":    last_verdict.get("rubric_criteria_scores"),
            "rubric_score":              last_verdict.get("rubric_score"),
            # Online-memory retrieval diagnosis for THIS round (additive; ignored by legacy
            # consumers). ``failure_reason`` is None on a productive round.
            "method":                    _attempt.method if _attempt else strategy,
            "outcome":                   _attempt.outcome if _attempt else None,
            "failure_reason":            _attempt.failure_reason if _attempt else None,
        })

        if last_verdict["label"] in ("sufficient", "SUFFICIENT", "FULLY_SUFFICIENT", "ANSWER_SUFFICIENT"):
            break

    # ── Answer ───────────────────────────────────────────────────────────────
    # 按 rubric(题目+选项)让 LLM 重排全部累积证据, 有用的排前噪声沉底, 全喂 answerer
    # (不截断)。
    ev_texts = om.evidence_texts()
    ev_ids   = om.evidence_ids()

    if rubric_rerank and len(ev_texts) > 1:
        order    = _rerank_by_rubric(question, candidates, ev_texts, rubric, llm, verifier=verifier)
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
    result["online_memory"]      = om.summary()   # additive; retrieval trajectory + evidence provenance
    return result
