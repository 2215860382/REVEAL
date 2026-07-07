"""Multi-layer summary bank builder — ADDITIVE over existing single-similarity banks.

Reads existing L1 banks (narrative.json + vectors.npz, built by
``build_single_similarity.py``) and adds coarse summary layers L2/L3 by
bottom-up grouping + LLM summarization. It does NOT re-extract frames or
re-caption — it reuses the L1 captions and BGE/SigLIP vectors.

Keyframes: by default (``--frame-mode hardlink``) the per-chunk representative
frame (``keyframes_resized``) and its full-res ``keyframes_origin`` twin are
hard-linked INTO the output bank and ``L1.jsonl`` records them as RELATIVE paths,
so the multi-layer bank is SELF-CONTAINED — layers and their photos live
together and the bank keeps working even if the source single-layer bank is later
deleted (hard links share inodes, so this costs no extra disk on the same
filesystem). Pass ``--frame-mode reference`` for the legacy behaviour (absolute
paths back into the source bank, no images copied), or ``--frame-mode copy`` for
a true portable duplicate. By default only keyframes (``keyframes_resized`` +
``keyframes_origin``) are brought — that is all the answerer/hires path reads.
``--include-raw`` additionally copies each chunk's ``frames_raw/`` subdir (large;
only needed for on-demand re-captioning, not answering).

The source single-layer bank is left untouched. New banks are written to a
SEPARATE output dir in pyramid format, loadable by
``bank_loader._load_pyramid_dir`` (L1.jsonl + L{2,3}.jsonl + *.npz + meta.json),
so ``veil.py`` ``multi_layer_mode`` can consume the layers directly.

Grouping (solves "L1 is similarity-grouped so its duration is not known in
advance"): build each upper layer BOTTOM-UP, accumulating consecutive children
until cumulative span ≥ T seconds OR child count ≥ N (whichever first). This gives
roughly-uniform time coverage while never splitting an L1 chunk.

Defaults reproduce the definitive **ml3_n12** granularity (the 79% mainline bank):
time budgets are disabled (T=100000s) so grouping is purely by count, mid fan-out
n=12 for both L2 and L3.

  L1  = existing similarity groups (variable length)
  L2  = group L1 by count n≥12,  summarize
  L3  = group L2 by count n≥12,  summarize

Usage (one bank from an existing single-layer bank):
    PYTHONPATH=. python -m src.build_memory.build_multilayer \\
        --src-memory-dir outputs/memory/videomme_multiframe \\
        --out-memory-dir outputs/memory/videomme_ml3_n12 \\
        --llm-api-url http://127.0.0.1:8003,http://127.0.0.1:8004 \\
        --embed-api-url http://127.0.0.1:9000 \\
        --workers 16 [--limit 2]

For the full one-command pipeline (videos → single-layer → self-contained ml3_n12
bank in one run), see ``build_single_similarity.py --emit-multilayer``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np

from src.utils.logging import get_logger
from src.build_memory.core.bank_loader import load_bank

log = get_logger("build_multilayer")

# ── Summary prompt (plain text, same captioning philosophy as L1) ─────────────
SUMMARY_SYSTEM = (
    "You are summarizing consecutive sub-segments of a single video into one "
    "coherent, chronological description.\n\n"
    "You will receive ordered sub-segment descriptions, each prefixed with its "
    "time range. Consolidate them into the major events in time order.\n\n"
    "Guidelines:\n"
    "- Merge related content into major events with a clear progression; keep "
    "the subjects' decisions and actions, including the purpose and method of "
    "key actions.\n"
    "- Preserve named subjects, on-screen numbers, and verbatim OCR text "
    "(quote OCR inside double quotes).\n"
    "- Keep discussions concise: core topics and conclusions only, not trivial "
    "back-and-forth.\n"
    "- Be factual and neutral; do not invent anything not in the sub-segments.\n"
    "- Do not mention sub-segments, summaries, timestamps, or how the input was "
    "produced.\n"
    "- Write ONE plain-text passage: no markdown, no headings, no bullet lists.\n"
    "- Stay strictly under {budget} words; do not report the word count.\n\n"
    "Output only the summary text."
)
SUMMARY_USER = (
    "Sub-segment descriptions (in time order):\n{children}\n\n"
    "Write the consolidated event summary for this window."
)


def _place_frame(src: Path, dst: Path, mode: str) -> bool:
    """Materialise ``src`` at ``dst`` via hard link / copy / symlink.

    Returns True if ``dst`` exists afterwards. Hard link falls back to copy when
    the two paths straddle filesystems (EXDEV). No-op if ``dst`` already exists.
    """
    if not src.is_file():
        return False
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return True
        except OSError:
            shutil.copy2(src, dst)
            return True
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return True
    shutil.copy2(src, dst)  # mode == "copy"
    return True


def _place_chunk_frames(src_vd: Path, out_vd: Path, kf_name: str,
                        mode: str, include_raw: bool) -> None:
    """Co-locate one L1 chunk's images into the output bank.

    Brings the resized representative frame (what the answerer sees) and its
    full-res ``keyframes_origin`` twin (used by ``--hires-keyframes``); with
    ``include_raw`` also the chunk's whole ``frames_raw/<cid>/`` subdir.
    """
    _place_frame(src_vd / "keyframes_resized" / kf_name,
                 out_vd / "keyframes_resized" / kf_name, mode)
    _place_frame(src_vd / "keyframes_origin" / kf_name,
                 out_vd / "keyframes_origin" / kf_name, mode)
    if include_raw:
        cid = kf_name.split("_")[0]  # "0007_0.jpg" -> "0007"
        src_raw = src_vd / "frames_raw" / cid
        if src_raw.is_dir():
            for f in src_raw.iterdir():
                if f.is_file():
                    _place_frame(f, out_vd / "frames_raw" / cid / f.name, mode)


def _group_bottom_up(nodes: List[dict], t_budget: float, n_max: int) -> List[List[dict]]:
    """Accumulate consecutive nodes until span≥t_budget OR count≥n_max."""
    groups: List[List[dict]] = []
    cur: List[dict] = []
    for nd in nodes:
        cur.append(nd)
        span = nd["t_end"] - cur[0]["t_start"]
        if span >= t_budget or len(cur) >= n_max:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _summarize(llm, group: List[dict], budget: int) -> str:
    """LLM-summarize a group's child texts into one passage."""
    children = "\n".join(
        f"[{nd['t_start']:.0f}s-{nd['t_end']:.0f}s] {nd['text']}" for nd in group
    )
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM.format(budget=budget)},
        {"role": "user", "content": SUMMARY_USER.format(children=children)},
    ]
    try:
        out = llm.chat(messages, max_new_tokens=int(budget * 2.2), enable_thinking=False)
        out = (out or "").strip()
        if out:
            return out
    except Exception as e:  # noqa: BLE001
        log.warning("summary LLM failed (%s); falling back to concat", e)
    # Fallback: concatenate child texts (truncated) so a node is never empty.
    return " ".join(nd["text"] for nd in group)[: budget * 8]


def _build_upper(llm, child_nodes: List[dict], t_budget: float, n_max: int,
                 words: int, pool) -> List[dict]:
    """Group children bottom-up and summarize each group into a parent node.

    Groups within a layer are independent → summarize them concurrently via the
    shared ``pool``. (Layers stay sequential: L3 needs L2's text.)
    """
    groups = _group_bottom_up(child_nodes, t_budget, n_max)
    futs = [pool.submit(_summarize, llm, g, words) for g in groups]
    return [{
        "idx": idx,
        "t_start": float(group[0]["t_start"]),
        "t_end": float(group[-1]["t_end"]),
        "text": fu.result(),
        "n_children": len(group),
    } for idx, (group, fu) in enumerate(zip(groups, futs))]


def build_multilayer_for_video(src_vd: Path, out_vd: Path, llm, embedder, cfg: dict, pool) -> str:
    """Build one multi-layer bank from an existing single-layer bank dir.

    ``cfg`` carries the grouping budgets (t2/n2/w2, t3/n3/w3) plus two keys that
    control keyframe co-location: ``frame_mode`` ("reference" | "hardlink" |
    "copy" | "symlink") and ``include_raw`` (bool). In "reference" mode the L1
    rows keep absolute paths into ``src_vd`` (legacy); otherwise each chunk's
    frames are placed into ``out_vd`` and recorded as relative paths so the bank
    is self-contained.
    """
    frame_mode = cfg.get("frame_mode", "hardlink")
    include_raw = cfg.get("include_raw", False)

    bank = load_bank(src_vd)
    l1 = sorted(bank.chunks, key=lambda c: c.start_time)
    if not l1:
        return "empty"

    # ── L1 nodes (reuse existing captions; resolve/co-locate keyframes) ──
    vdim = next((len(c.v_visual) for c in l1 if c.v_visual), 0)
    l1_text_vecs = np.zeros((len(l1), 1024), dtype=np.float32)
    l1_vis_vecs = np.zeros((len(l1), vdim), dtype=np.float32) if vdim else None
    l1_rows, l1_nodes = [], []
    for i, c in enumerate(l1):
        if c.v_text:
            l1_text_vecs[i] = np.asarray(c.v_text, dtype=np.float32)
        if l1_vis_vecs is not None and c.v_visual:
            l1_vis_vecs[i] = np.asarray(c.v_visual, dtype=np.float32)
        # Record the chunk's inference keyframe = keyframes_resized (what the
        # answerer sees). "reference" mode keeps an absolute path into the source
        # bank; otherwise co-locate the frame into out_vd and store it relative,
        # so the multi-layer bank ships its own photos.
        kf_ref = ""
        if c.keyframe_path:
            kf_name = Path(c.keyframe_path).name
            if frame_mode == "reference":
                kf_ref = str((src_vd / "keyframes_resized" / kf_name).resolve())
            else:
                _place_chunk_frames(src_vd, out_vd, kf_name, frame_mode, include_raw)
                kf_ref = f"keyframes_resized/{kf_name}"
        # Preserve the raw speech transcript line so L1 evidence matches the
        # single-layer bank's `<narrative>\nSpeech: <asr>` format (the narrative
        # already weaves speech in, but keep the verbatim line for fidelity).
        l1_text = c.memory_text or ""
        if (getattr(c, "asr", "") or "").strip():
            l1_text = f"{l1_text}\nSpeech: {c.asr.strip()}"
        l1_rows.append({
            "idx": i,
            "t_start": float(c.start_time),
            "t_end": float(c.end_time),
            "text": l1_text,
            "frame_paths": [kf_ref] if kf_ref else [],
            "frame_ts": [float(c.keyframe_ts or c.start_time or 0.0)],
            "visual_offsets": [i] if l1_vis_vecs is not None else [],
        })
        l1_nodes.append({"t_start": float(c.start_time), "t_end": float(c.end_time),
                         "text": l1_text})

    # ── Build upper layers bottom-up ──────────────────────────────────────────
    l2 = _build_upper(llm, l1_nodes, cfg["t2"], cfg["n2"], cfg["w2"], pool)
    l3 = _build_upper(llm, l2,       cfg["t3"], cfg["n3"], cfg["w3"], pool)

    # ── Embed upper-layer summaries (BGE-M3) ──────────────────────────────────
    def _embed(nodes):
        if not nodes:
            return np.zeros((0, 1024), dtype=np.float32)
        return embedder.encode([n["text"] for n in nodes]).astype(np.float32)
    l2_vecs, l3_vecs = _embed(l2), _embed(l3)

    # ── Write pyramid-format bank ─────────────────────────────────────────────
    out_vd.mkdir(parents=True, exist_ok=True)
    with (out_vd / "L1.jsonl").open("w") as f:
        for r in l1_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.savez_compressed(out_vd / "L1_text.npz", vectors=l1_text_vecs)
    if l1_vis_vecs is not None:
        np.savez_compressed(out_vd / "L1_visual.npz", vectors=l1_vis_vecs)
    for layer, rows, vecs in [(2, l2, l2_vecs), (3, l3, l3_vecs)]:
        with (out_vd / f"L{layer}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        np.savez_compressed(out_vd / f"L{layer}.npz", vectors=vecs)
    duration = max((c.end_time for c in l1), default=0.0)
    (out_vd / "meta.json").write_text(json.dumps({
        "video_id": src_vd.name, "duration": float(duration), "status": "ok",
        "built_by": "build_multilayer", "source_dir": str(src_vd.resolve()),
        "n_l1": len(l1), "n_l2": len(l2), "n_l3": len(l3),
        "grouping": {k: cfg[k] for k in ("t2", "n2", "t3", "n3")},
        "frame_mode": frame_mode, "include_raw": include_raw,
    }, ensure_ascii=False, indent=2))
    return f"ok L1={len(l1)} L2={len(l2)} L3={len(l3)}"


def add_multilayer_build_args(ap: argparse.ArgumentParser) -> None:
    """Register the grouping + frame-colocation flags shared by this CLI and the
    one-command pipeline in ``build_single_similarity.py --emit-multilayer``.

    Defaults reproduce the definitive **ml3_n12** bank: time budgets disabled
    (T=100000s → pure count grouping), mid fan-out n=12. Keyframes are copied
    into the bank (true self-contained duplicate);
    ``frames_raw`` is left out by default (answerer never reads it) — add it with
    ``--include-raw``.
    """
    g = ap.add_argument_group("multilayer grouping (defaults = ml3_n12)")
    g.add_argument("--l2-seconds", type=float, default=100000); g.add_argument("--l2-max", type=int, default=12)
    g.add_argument("--l3-seconds", type=float, default=100000); g.add_argument("--l3-max", type=int, default=12)
    g.add_argument("--l2-words", type=int, default=120)
    g.add_argument("--l3-words", type=int, default=300)
    g.add_argument("--frame-mode", choices=["reference", "hardlink", "copy", "symlink"],
                   default="copy",
                   help="How to bring keyframes into the bank. 'copy' (default) = "
                        "true self-contained duplicate so the source single-layer "
                        "bank can be deleted afterwards; 'hardlink' = same-fs zero-"
                        "cost link; 'reference' = legacy absolute paths into source.")
    g.add_argument("--include-raw", dest="include_raw", action="store_true", default=False,
                   help="Also bring each chunk's frames_raw/ subdir (~183MB/video; the "
                        "answerer never reads it — only needed for on-demand re-captioning).")
    g.add_argument("--no-include-raw", dest="include_raw", action="store_false",
                   help="Default: keyframes only (keyframes_resized + origin), disk-friendly.")


def cfg_from_args(args) -> dict:
    """Build the grouping+frame cfg dict consumed by ``build_multilayer_for_video``."""
    return {"t2": args.l2_seconds, "n2": args.l2_max, "w2": args.l2_words,
            "t3": args.l3_seconds, "n3": args.l3_max, "w3": args.l3_words,
            "frame_mode": args.frame_mode, "include_raw": args.include_raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-memory-dir", required=True, help="Existing single-layer banks")
    ap.add_argument("--out-memory-dir", required=True, help="New multi-layer banks (separate dir)")
    ap.add_argument("--llm-api-url", required=True, help="comma-separated = round-robin")
    ap.add_argument("--llm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--embed-api-url", default=None, help="BGE embed service; omit for local")
    ap.add_argument("--bge-model", default="/home2/ycj/Models/BAAI/bge-m3")
    ap.add_argument("--bge-gpu", default="cuda:0")
    ap.add_argument("--workers", type=int, default=8,
                    help="Videos processed concurrently")
    ap.add_argument("--summary-concurrency", type=int, default=32,
                    help="Shared cap on concurrent summary LLM calls across all videos "
                         "(same-layer groups are independent and run in parallel)")
    ap.add_argument("--limit", type=int, default=None, help="Only build first N videos (smoke test)")
    add_multilayer_build_args(ap)
    args = ap.parse_args()

    cfg = cfg_from_args(args)

    src_root = Path(args.src_memory_dir)
    out_root = Path(args.out_memory_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    from src.clients.llm_client import LLMClient
    from src.clients.embedder import BGEM3Embedder
    llm = LLMClient(model_path=args.llm_api_model, api_url=args.llm_api_url,
                    api_model=args.llm_api_model)
    if args.embed_api_url:
        embedder = BGEM3Embedder(api_url=args.embed_api_url)
    else:
        embedder = BGEM3Embedder(model_path=args.bge_model, device=args.bge_gpu)
    _embed_lock = threading.Lock()
    _safe_embedder = type("E", (), {"encode": lambda _self, t: _locked_encode(embedder, _embed_lock, t)})()

    vids = sorted(p for p in src_root.iterdir()
                  if p.is_dir() and ((p / "narrative.json").exists() or (p / "L1.jsonl").exists()))
    if args.limit:
        vids = vids[: args.limit]
    todo = [p for p in vids if not (out_root / p.name / "meta.json").exists()]
    log.info("videos: %d total, %d to build (workers=%d)", len(vids), len(todo), args.workers)

    done = [0]
    lock = threading.Lock()
    # Shared pool for leaf summary calls — caps total LLM concurrency regardless
    # of how many videos are in flight. Video threads only submit + await here.
    summ_pool = ThreadPoolExecutor(max_workers=args.summary_concurrency)

    def work(src_vd):
        t0 = time.time()
        try:
            msg = build_multilayer_for_video(src_vd, out_root / src_vd.name, llm, _safe_embedder, cfg, summ_pool)
        except Exception as e:  # noqa: BLE001
            log.error("[%s] FAILED: %s", src_vd.name, e, exc_info=True)
            msg = f"FAIL {e}"
        with lock:
            done[0] += 1
            n = done[0]
        log.info("[%d/%d] %-16s %s (%.1fs)", n, len(todo), src_vd.name, msg, time.time() - t0)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, p) for p in todo]
        for fu in as_completed(futs):
            if fu.exception():
                log.error("worker exc: %s", fu.exception())
    summ_pool.shutdown(wait=True)
    log.info("done -> %s", out_root)


def _locked_encode(embedder, lock, texts):
    with lock:
        return embedder.encode(texts)


if __name__ == "__main__":
    main()
