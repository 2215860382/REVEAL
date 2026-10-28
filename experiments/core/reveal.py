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

from src.build_memory.core.schema import MemoryBank
from src.agents.planner import Planner
from src.agents.verifier import Verifier, get_rubric_dict
from experiments.reveal_27b import OnlineMemory
from experiments.ablations.online import (
    PLANNER_ONLINE_FIELDS,
    build_online_context,
    coverage_gap_summary,
)

ONLINE_FIELD_NAMES = PLANNER_ONLINE_FIELDS


# ── Deterministic visual routing (REVEAL_VISUAL_ROUTE) ───────────────────────
# ON by default (set REVEAL_VISUAL_ROUTE=0 to disable). A question whose text carries
# an explicit time anchor (mm:ss / h:mm:ss), OR whose video is short enough to cover
# whole (≤ REVEAL_VR_SHORT_SEC, default 10min), is handed REAL frames sampled from the
# ORIGINAL mp4 — the pixels a text caption cannot preserve — APPENDED to the accumulated
# text evidence (never replacing it). Routing is pure regex + duration IN CODE, not a
# planner/LLM choice, so the model cannot "mis-pick" a strategy. Anything not matched
# (no timestamp AND video >10min — i.e. the long-video benches) falls straight through
# to the unchanged text pipeline.
_VR_TS    = re.compile(r'(?<!\d)(\d{1,2}):([0-5]\d)(?::([0-5]\d))?(?!\d)')


def _vr_parse_seconds(question: str) -> List[float]:
    out: List[float] = []
    for m in _VR_TS.finditer(question):
        a, b, c = m.groups()
        sec = (int(a) * 3600 + int(b) * 60 + int(c)) if c is not None else (int(a) * 60 + int(b))
        out.append(float(sec))
    return out[:2]


def _ffmpeg_window_frames(video_path, start: float, end: float, n: int) -> List:
    """Extract ≤n frames in [start,end] via the ffmpeg CLI (software AV1 via libdav1d).

    decord/cv2 CANNOT decode LVBench's AV1 videos (no HW AV1 decode; 75/86 videos);
    ffmpeg's bundled libdav1d can. ``-ss`` before ``-i`` = fast keyframe seek so only
    the short window is decoded. Frames are downscaled (long edge ≤
    REVEAL_DENSE_FRAME_MAXSIDE, default 448 ≈ 256 vision tokens/img) to bound KV.
    Returns [] on any failure — caller then simply feeds no frames (text unchanged).
    """
    import subprocess
    import tempfile
    from PIL import Image
    start = max(0.0, float(start))
    dur = max(0.2, float(end) - start)
    fps = max(1, n) / dur
    mx = int(os.environ.get("REVEAL_DENSE_FRAME_MAXSIDE", "448") or 448)
    with tempfile.TemporaryDirectory() as td:
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
               "-i", str(video_path), "-vf", f"fps={fps:.4f}", "-frames:v", str(n),
               f"{td}/f_%03d.jpg"]
        try:
            subprocess.run(cmd, check=True, timeout=40,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return []
        imgs = []
        for fn in sorted(os.listdir(td)):
            try:
                im = Image.open(os.path.join(td, fn)).convert("RGB")
            except Exception:
                continue
            if mx > 0 and max(im.size) > mx:
                im.thumbnail((mx, mx), Image.BILINEAR)
            imgs.append(im)
        return imgs


def _probe_duration(video_path) -> "float | None":
    """Video duration in seconds via ffprobe (~50ms). None on any failure.

    Only called when REVEAL_VR_SHORT_SEC > 0, so zero cost on the default path.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            check=True, timeout=15, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return float(out.stdout.decode().strip())
    except Exception:
        return None


_CLIP_INDEX_CACHE: dict = {}


def _egolife_clip_index(dir_path):
    """[(abs_start, abs_end, path), …] for an egolife person footage dir, time-sorted.

    The bank's chunk start/end times live on a DAY1-anchored global timeline
    (``(day-1)*86400 + hh:mm:ss.ff``); each ``DAY{n}_{PERSON}_{ts}.mp4`` clip's
    absolute start is recoverable from its name and its length from ffprobe, so a
    global time maps to (clip, local offset).  Cached per dir. [] if not egolife-like.
    """
    key = str(dir_path)
    if key in _CLIP_INDEX_CACHE:
        return _CLIP_INDEX_CACHE[key]
    import re as _re
    from pathlib import Path as _P
    pat = _re.compile(r"DAY(\d+)_(A\d+_[A-Z]+)_(\d{8})\.mp4$")
    starts = []
    try:
        for mp4 in _P(dir_path).rglob("*.mp4"):
            m = pat.search(mp4.name)
            if not m:
                continue
            day, _, ts = m.groups()
            st = (int(day) - 1) * 86400.0 + (
                int(ts[0:2]) * 3600 + int(ts[2:4]) * 60 + int(ts[4:6]) + int(ts[6:8]) / 100.0)
            starts.append((st, str(mp4)))
    except Exception:
        starts = []
    starts.sort()
    # Clip length = next clip's start − this start (the true duration while recording
    # is continuous).  No ffprobe — probing thousands of clips is far too slow.  A
    # recording gap makes a clip "over-claim" the gap, but ffmpeg simply returns
    # fewer frames when a seek lands past real EOF, so it is harmless.
    idx = [(st, (starts[i + 1][0] if i + 1 < len(starts) else st + 60.0), path)
           for i, (st, path) in enumerate(starts)]
    _CLIP_INDEX_CACHE[key] = idx
    return idx


def _sample_window(video_path, start: float, end: float, n: int) -> List:
    """Sample ≤n frames across GLOBAL [start,end].

    Dispatches on the target: a single-file ``video_path`` (VideoMME/LVBench/
    LongVideoBench) goes straight to :func:`_ffmpeg_window_frames` — byte-identical
    to the old behaviour.  A *directory* (egolife/egor1 per-person footage, many
    day-clips) is resolved via the clip index: every clip overlapping [start,end]
    contributes frames proportional to its overlap, each sampled at its own local
    offset, then concatenated in time order.
    """
    if video_path and os.path.isdir(str(video_path)):
        idx = _egolife_clip_index(video_path)
        if not idx:
            return []
        s, e = max(0.0, float(start)), float(end)
        total = max(0.2, e - s)
        imgs: List = []
        for cst, cen, path in idx:
            ov0, ov1 = max(s, cst), min(e, cen)
            if ov1 - ov0 <= 0.05:
                continue
            n_c = max(1, int(round(n * (ov1 - ov0) / total)))
            imgs += _ffmpeg_window_frames(path, ov0 - cst, ov1 - cst, n_c)
            if len(imgs) >= n:
                break
        return imgs[:n]
    return _ffmpeg_window_frames(video_path, start, end, n)


def route_visual_frames(question: str, om, chunk_by_id: dict, video_path):
    """Real frames for this question → [(PIL, chunk_id, ts), ...]; [] otherwise.

    Deterministic priority (answer-independent — never looks at the gold):
      1. TIME  — an explicit mm:ss in the question → sample at that exact time
                 (single ±REVEAL_VR_WINDOW s window, or uniformly across a range).
      2. SHORT — video duration ≤ REVEAL_VR_SHORT_SEC (default 600s = 10min) → the
                 whole clip is coverable, so sample the ENTIRE video at
                 REVEAL_VR_SHORT_FPS (dense whole-video coverage, exactly where naive
                 frame sampling already wins).
    Anything else (no timestamp AND longer than the short cap) → no frames; the text
    pipeline is left unchanged. Decode is via ffmpeg (handles the AV1 videos
    decord/cv2 choke on). ``om``/``chunk_by_id`` are unused now (kept for a stable
    call signature).
    """
    if not video_path:
        return []

    win_pm = float(os.environ.get("REVEAL_VR_WINDOW", "4") or 4)
    n_each = int(os.environ.get("REVEAL_VR_FRAMES", "8") or 8)
    cap = int(os.environ.get("REVEAL_VR_MAXFRAMES", "12") or 12)

    # ── Priority 1: TIME — explicit mm:ss in the question → sample at that time ──
    secs = _vr_parse_seconds(question)
    if secs:
        out = []
        if len(secs) >= 2:                               # RANGE → sample across span
            t0, t1 = min(secs), max(secs)
            st, en = max(0.0, t0 - win_pm), t1 + win_pm
            step = float(os.environ.get("REVEAL_VR_RANGE_STEP", "8") or 8)  # ~1 frame / step s
            rcap = int(os.environ.get("REVEAL_VR_RANGE_FRAMES", "20") or 20)
            n = max(n_each, min(rcap, int((en - st) / max(step, 0.1)) + 1))
            for img in _sample_window(video_path, st, en, n):
                out.append((img, -1, float(t0)))
            return out
        s = secs[0]                                      # SINGLE → ±win window
        for img in _sample_window(video_path, max(0.0, s - win_pm), s + win_pm, n_each):
            out.append((img, -1, float(s)))
            if len(out) >= cap:
                break
        return out

    # ── Priority 2: SHORT — whole video ≤10min is fully coverable → sample it all ──
    # Duration-only gate (answer-independent). Samples the ENTIRE clip at a fixed fps
    # so short videos get dense whole-video coverage (not a narrow retrieval window),
    # which is exactly where naive frame sampling already scores high.
    short_sec = float(os.environ.get("REVEAL_VR_SHORT_SEC", "600") or 600)
    if short_sec > 0:
        dur = _probe_duration(video_path)
        if dur is not None and 0 < dur <= short_sec:
            sfps = float(os.environ.get("REVEAL_VR_SHORT_FPS", "1.0") or 1.0)
            n_short = max(1, int(round(dur * sfps)))
            return [(img, -1, 0.0)
                    for img in _sample_window(video_path, 0.0, dur, n_short)]

    return []


# ── Timeline scan (whole-video coverage / query-exhausted fallback) ──────────

def _timeline_scan_retrieve(
    bank: "MemoryBank",
    seen_ids: Set[int],
    n: int = 8,
    max_time: float | None = None,
) -> tuple[list, list, list]:
    """Sample ``n`` still-unretrieved chunks evenly across the whole timeline.

    This is the ``scan`` strategy: it sweeps the parts of the video that targeted
    queries have NOT reached yet, giving broad coverage. Two situations use it:
    (1) the answer is spread across the entire video and no single query or time
    window captures it (which-statement-is-correct, total counts, synopsis); and
    (2) targeted queries are exhausted — every new query the planner produced was a
    near-duplicate of one already tried (``filter_targeted_queries``), so the only
    productive move left is to cover timeline regions not yet seen. Chunks already in
    ``seen_ids`` are skipped, so each scan reaches fresh segments. ``max_time``
    (EgoLife) drops any chunk ending after the question time — future memory is
    off-limits.
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

def _rerank_by_relevance(
    question: str,
    candidates: List[str],
    evidence_texts: List[str],
    llm,
) -> List[int]:
    """Ask the LLM to rank evidence chunks by question/option relevance."""
    if len(evidence_texts) <= 1:
        return list(range(len(evidence_texts)))

    opts    = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(candidates))
    ev_fmt  = "\n".join(f"[E{i+1}] {t[:300]}" for i, t in enumerate(evidence_texts))

    rank_hint = ("Rank the evidence chunks from MOST to LEAST useful for answering the "
                 "question and deciding between the options")
    user = (
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
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
    l1_pool_cap: int = 8,
    l2_cap: int = 4,
    l3_cap: int = 2,
) -> tuple[List[str], List[int], List[List[float]]]:
    """Retrieve from the L1 detail pool + L2/L3 summary pools, merged by score.

    Definitive paper-mainline config: L1=8 / L2=4 / L3=2 candidates enter the
    merge pool, then the top ``final_top_k`` by cosine are returned.

    ``max_time`` (EgoLife): every layer masks out chunks whose ``end_time`` is after
    the query time before its top-k, so future memory never enters the pool. None ⇒
    no filter (lvbench/videomme path unchanged).
    """
    if not bank.l2_chunks:
        return _query_retrieve(query, bank, embedder, coarse_top_k, final_top_k,
                               exclude_ids, max_time=max_time)

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

def run_reveal(
    question:      str,
    candidates:    List[str],
    bank:          MemoryBank,
    embedder,
    answerer,
    llm,
    verifier_llm                  = None,
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
) -> dict:
    """Iter-0 sub-question decomposition + iter ≥1 unified-planner repair.

    Each sub-question independently retrieves ``final_top_k`` chunks (no budget splitting);
    iter ≥1 query generation goes through Jaccard (``query_history_dedup_threshold``) + BGE drift
    (``query_evidence_dedup_threshold``) filters; if all targeted queries are filtered, the planner
    falls back to the ``scan`` strategy (sweep still-unretrieved chunks across the timeline for
    broad coverage — see ``_timeline_scan_retrieve``).
    """
    verifier = Verifier(verifier_llm or llm)
    planner  = Planner(llm)
    rubric   = get_rubric_dict(question, task_type)

    # Unified online memory: evidence cache + retrieval trajectory.
    # ``om.seen_ids`` replaces the old local set; ``om.evidence_*()`` the old all_* lists.
    om = OnlineMemory(llm, question, candidates)
    all_keyframes:      List              = []
    all_kf_chunk_ids:   List[int]         = []
    all_kf_ts:          List[float]       = []
    plan_history:       List[List[str]]   = [[question]]  # seed with original question for dedup
    iterations:         List[dict]        = []
    # Experimental parity with ``ablations/online.py``.  These stay opt-in so
    # the paper mainline remains fixed, while a controlled mainline-vs-Online
    # comparison can enable the same two additions.
    use_online_coverage = os.environ.get("REVEAL_OL_COVERAGE_GAP") == "1"
    use_evidence_state = os.environ.get("REVEAL_OL_EVIDENCE_STATE") == "1"
    evidence_state_style = os.environ.get(
        "REVEAL_OL_EVIDENCE_STATE_STYLE", "legacy"
    ).strip()
    if evidence_state_style not in {"legacy", "option_strings"}:
        raise ValueError(
            "REVEAL_OL_EVIDENCE_STATE_STYLE must be legacy or option_strings"
        )
    chunk_by_id = {c.chunk_id: c for c in bank.chunks}
    # Include L2 chunks so keyframe lookup doesn't skip them (L2 chunks have no
    # frames, so they naturally produce no keyframes).
    chunk_by_id.update({c.chunk_id: c for c in bank.l2_chunks})

    # ── Iter-0 plan: sub-question decomposition ──────────────────────────────
    current_plan: dict = planner.decompose_iter0(question, candidates)
    plan_history.append(current_plan["queries"])
    last_verdict: dict = {}

    # Round budget: query (incl. its ``scan`` fallback) ≤ _q_max = max_iter
    # (CLI --max-iter, default 3). ``scan`` is NOT a separate budget — it is query's
    # remedial fallback (targeted queries exhausted), so it draws from query's budget.
    # There is NO total-round gate: the loop ends only when the verifier is SUFFICIENT,
    # or when the planner still wants a query/scan round but query's budget is spent.
    _q_max = max_iter
    q_rounds = 0
    it       = -1
    while True:
        it += 1
        n_frames_before_round = len(all_keyframes)

        # ── Plan (iter ≥ 1): generate new queries from last verdict + history ─
        if it > 0:
            # Fixed paper-mainline Online Memory: history, verifier missing
            # evidence, and retrieval quality always feed the next plan.
            online_context = build_online_context(
                plan_history=plan_history,
                verdict=last_verdict,
                memory=om,
                bank=bank,
                max_time=max_time,
                include_coverage=use_online_coverage,
            )
            _hist_for_planner = [list(queries) for queries in online_context.history]
            _verdict_for_planner = {
                "missing_evidence_analysis":
                    online_context.missing_evidence_analysis}
            planner_evidence = om.evidence_texts()
            next_plan = planner.plan_next(
                question, candidates, planner_evidence, _verdict_for_planner, _hist_for_planner,
                rubric_judgment=rubric_judgment,
                retrieval_quality=online_context.retrieval_quality,
                coverage_gap=online_context.coverage_gap,
            )
            # Dedup is a code safety net, DECOUPLED from the history_query DISPLAY switch: the
            # Jaccard/drift filter always sees the real plan_history so a repeated query is dropped
            # even in arms that hide the "Already tried" block from the planner. (Previously dedup
            # rode on _hist_for_planner, so hiding history also silently disabled dedup.)
            next_plan = planner.filter_targeted_queries(
                next_plan, plan_history, embedder, om.evidence_vecs(),
                dedup_thresh=query_history_dedup_threshold,
                drift_threshold=query_evidence_dedup_threshold,
            )
            plan_history.append(next_plan["queries"])   # always recorded (dedup + history_query)
            current_plan = next_plan

        requested_strategy = current_plan.get("strategy", "query")
        if requested_strategy == "targeted":
            requested_strategy = "query"
        strategy = requested_strategy
        fallback_reason = current_plan.get("fallback_reason")
        # ``scan`` sweeps the whole timeline on query's budget. It runs either as
        # query's remedial fallback (targeted queries exhausted, set by
        # ``filter_targeted_queries`` or below when a query round admits no new
        # evidence) OR when the planner deliberately selected it.
        scan_fallback = current_plan.get("fallback") == "scan"
        do_scan  = (strategy == "scan") or (strategy == "query" and scan_fallback)
        # query (incl. its scan fallback) ≤ _q_max. The loop ends only on
        # verifier-SUFFICIENT (below) or when query's budget is spent.
        if q_rounds >= _q_max:
            break
        q_rounds += 1
        queries_this_iter = current_plan.get("queries", []) if not do_scan else []
        if strategy == "query" and not do_scan and not queries_this_iter:
            queries_this_iter = [question]

        # ── Retrieve ──────────────────────────────────────────────────────────
        new_texts, new_ids, new_vecs = [], [], []

        if do_scan:
            # `scan` = sweep still-unretrieved chunks evenly across the whole timeline
            # (broad coverage for whole-video answers / exhausted targeted queries).
            new_texts, new_ids, new_vecs = _timeline_scan_retrieve(
                bank, om.seen_ids, n=final_top_k, max_time=max_time,
            )
            iter_query = "[scan]"
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


        # Admit into online memory with redundancy dedup and record one
        # RetrievalAttempt carrying compact retrieval quality.
        n_before = len(om.evidence)
        om.admit(new_texts, new_ids, new_vecs,
                 query=iter_query, round=it,
                 dedup_threshold=evidence_dedup_threshold)
        new_ids = om.evidence_ids()[n_before:]   # kept ids this round (keyframes + trace)

        # Text (query/scan) rounds never feed frames to the answerer — the accumulated
        # visual evidence comes solely from the deterministic TIME/SHORT route after the
        # loop (see route_visual_frames), never from retrieval-round keyframes.

        # Same update contract as the final Online ablation: only the
        # preceding compact state and this round's newly admitted evidence may
        # determine the replacement state.  Retrieval and verification still
        # retain the raw evidence; only the final answer context is replaced.
        evidence_state_changes = None
        evidence_state_raw = None
        if use_evidence_state:
            round_evidence = [
                item.text for item in om.evidence if item.round == it
            ]
            evidence_state_changes, evidence_state_raw = (
                om.update_evidence_state(
                    round_evidence,
                    round_index=it,
                    new_keyframe_images=all_keyframes[n_frames_before_round:],
                    style=evidence_state_style,
                )
            )

        # ── Verifier ──────────────────────────────────────────────────────────
        ev_for_verifier = om.evidence_texts()
        kf_for_verifier = all_keyframes

        last_verdict = verifier.verify(question, candidates, ev_for_verifier, rubric,
                                  keyframe_images=kf_for_verifier)

        # ── Trace ────────────────────────────────────────────────────────────
        _attempt = om.trajectory[-1] if om.trajectory else None
        iterations.append({
            "verdict":                   last_verdict["label"],
            "evidence":                  list(ev_for_verifier),
            "missing_evidence_analysis": last_verdict.get("missing_evidence_analysis") or None,
            "iter":                      it,
            "plan": {
                "think":                 current_plan.get("think") or current_plan.get("reasoning"),
                "strategy":              requested_strategy,
                "queries":               queries_this_iter,
            },
            "retrieval": {
                "executed_strategy":      "scan" if do_scan else strategy,
                "fallback_reason":        fallback_reason,
                "query":                 iter_query,
                "new_ids":               new_ids,
                "n_keyframes_accumulated": len(all_keyframes),
                "n_keyframes_to_verifier": len(kf_for_verifier),
                "n_evidence_to_verifier":  len(ev_for_verifier),
            },
            "rubric_criteria_scores":    last_verdict.get("rubric_criteria_scores"),
            "rubric_score":              last_verdict.get("rubric_score"),
            # Online-memory retrieval diagnosis for THIS round (additive; ignored by legacy
            # consumers). Strategy is already recorded under "plan"; this is the
            # compact quality diagnosis for the executed retrieval.
            "retrieval_quality":         _attempt.retrieval_quality if _attempt else None,
            "coverage_gap":              (
                coverage_gap_summary(bank, om.seen_ids, max_time)
                if use_online_coverage else None
            ),
            "evidence_state":            om.evidence_state if use_evidence_state else None,
            "evidence_state_changes":    evidence_state_changes,
            "evidence_state_raw":        evidence_state_raw,
        })

        if last_verdict["label"] in ("sufficient", "SUFFICIENT", "FULLY_SUFFICIENT", "ANSWER_SUFFICIENT"):
            break

    # ── Visual route (REVEAL_VISUAL_ROUTE): append REAL frames (TIME / SHORT) ──
    # ON by default (set REVEAL_VISUAL_ROUTE=0 to disable). route_visual_frames only
    # fires for a timestamped question (TIME) or a whole video ≤10min (SHORT); the
    # long-video benches (>10min, no timestamp) fall straight through to pure text,
    # so this stays a no-op there. Never runs under evidence-state mode (that path
    # feeds a consolidated string, no pixels).
    vr_used = 0
    if (os.environ.get("REVEAL_VISUAL_ROUTE", "1") != "0"
            and not use_evidence_state and keyframe_dir is not None):
        for img, cid, ts in route_visual_frames(question, om, chunk_by_id, video_path):
            all_keyframes.append(img)
            all_kf_chunk_ids.append(cid)
            all_kf_ts.append(ts)
            vr_used += 1

    # ── Answer ───────────────────────────────────────────────────────────────
    # 按 rubric(题目+选项)让 LLM 重排全部累积证据, 有用的排前噪声沉底, 全喂 answerer
    # (不截断)。
    ev_texts = om.evidence_texts()
    ev_ids   = om.evidence_ids()

    if rubric_rerank and not use_evidence_state and len(ev_texts) > 1:
        order    = _rerank_by_relevance(question, candidates, ev_texts, llm)
        ev_texts = [ev_texts[i] for i in order]
        ev_ids   = [ev_ids[i]   for i in order]

    # In the state experiment this exactly matches ``ablations/online.py``:
    # the answerer receives the compact state rather than raw fragments or
    # pixels.  Keep raw ids in the result for retrieval provenance.
    if use_evidence_state:
        answer_evidence = [
            "Consolidated question-conditioned evidence state:\n"
            + om.evidence_state
        ]
        kf_for_answer = []
    else:
        answer_evidence = ev_texts
        kf_for_answer = all_keyframes
    kf_chunk_ids_for_ans = all_kf_chunk_ids
    kf_ts_for_ans        = all_kf_ts

    result = answerer.answer(question, candidates, answer_evidence,
                             keyframe_images=kf_for_answer,
                             keyframe_chunk_ids=kf_chunk_ids_for_ans,
                             keyframe_ts=kf_ts_for_ans,
                             evidence_chunk_ids=ev_ids,
                             question_first=True)
    result["n_keyframes_to_answer"] = len(kf_for_answer)
    result["n_evidence_to_answer"]  = len(answer_evidence)
    result["evidence_chunk_ids"] = ev_ids
    result["evidence_texts"]     = answer_evidence
    result["trace_iters"]        = iterations
    result["online_memory"]      = om.summary()   # additive; retrieval trajectory + evidence provenance
    result["evidence_state"]     = om.evidence_state if use_evidence_state else None
    result["evidence_state_style"] = evidence_state_style if use_evidence_state else None
    result["visual_route_frames"] = vr_used
    return result
