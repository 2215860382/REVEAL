"""Build single-granularity similarity-grouped memory banks.

Narration method: one multi-image VLM call per chunk — frames are interleaved
with timestamp labels, the overlapping subtitle lines are injected with their
time ranges, and the model writes a single chunk-level caption. This replaces
the earlier per-frame caption + LLM summary baseline (which lost frame-to-frame
continuity and laundered facts through a second LLM hop).

For the alternative structured "Memory: <paragraph>\\nKey details: <comma list>"
two-line variant, see ``build_single_similarity_newprompt.py``.

Pipeline per video:
  1. ffmpeg 1fps → JPEG frames
  2. Blank + blur filter (with failsafe)
  3. Pre-dedup (dHash, window=10) — removes near-identical consecutive frames
  4. SigLIP encode survivors → cosine-similarity grouping (θ≥0.8)
  5. Per-chunk dHash+SigLIP dedup → kept_frame_paths (visual representative subset)
     + top-2 sharpest cap on kept set → keyframe / v_visual
     all_frame_paths (post pre-dedup, no chunk-cap) is preserved for narrative VLM.
  6. SRT → speech_text + speech_lines per chunk (if subtitle_dir supplied)
  7. Narrative — single multi-image VLM call per chunk:
     - Down-sample g.all_frame_paths evenly to --narrative-max-frames (default 30)
     - One call: system prompt + segment window + transcript lines + interleaved
       (frame timestamp, image) pairs → single paragraph caption
  8. BGE-M3 encode (narrative + speech), max_length=4096
  9. Write legacy directory bank:
        {out_dir}/{video_id}/narrative.json       chunks[].{narrative, caption,
                                                  speech_text, sampled_frames,
                                                  keyframe_ts, keyframe_path, v_visual}
        {out_dir}/{video_id}/vectors.npz          narrative_vecs + chunk_ids
        {out_dir}/{video_id}/keyframes_origin/    dedup kept frames at original res
        {out_dir}/{video_id}/keyframes_resized/   same files resized to ≤ frame_max_dim
        {out_dir}/{video_id}/frames_raw/          all post-cleanup frames per chunk (original res)

Usage:
    cd /home2/ycj/Project/VEIL
    PYTHONPATH=. python -m src.build_memory.build_single_similarity \\
        --benchmark videomme \\
        --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \\
        --siglip-gpu cuda:1 --bge-gpu cuda:1 \\
        --api-url http://10.82.1.145:8001,http://10.82.1.145:8002 \\
        --api-model Qwen3.5-27B
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile

# A truncated/half-written cached JPEG (e.g. a frame written just before an
# earlier build was killed) must never take down the whole 6k-clip run: let PIL
# load whatever bytes are decodable, and _img_block below skips the rest.
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.logging import get_logger

log = get_logger("build_single_similarity")


# ── Prompts (Binlu-style segment captioning, one multi-image call per chunk) ──

SYSTEM_PROMPT = (
    "You are an expert video captioner.\n"
    "\n"
    "You will receive a short video segment represented by ordered frames, "
    "each labeled with its timestamp in the video, and optionally the "
    "transcript lines of the speech with their time ranges on the same "
    "timeline.\n"
    "Write a single paragraph caption describing the visual content and, when "
    "transcript lines are provided, the spoken content.\n"
    "\n"
    "Guidelines:\n"
    "- Describe visible people, actions, objects, and the setting.\n"
    "- Present actions and events in their temporal order; use the timestamps "
    "to align speech with what is shown.\n"
    "- When transcript lines are provided, weave what is said into the "
    "caption; do not describe sounds you cannot know from the frames or "
    "transcript.\n"
    "- Preserve names, numbers, and on-screen text (OCR) verbatim. Do not "
    "invent details.\n"
    "- Keep the caption factual and neutral.\n"
    "- Do not mention frames, transcripts, timestamps, or how the input was "
    "produced.\n"
    "- Avoid speculation about emotions or intentions unless clearly visible "
    "or stated in the transcript.\n"
    "\n"
    "Output only the final caption text."
)

USER_HEADER_TEMPLATE = (
    "Segment window: {t0} to {t1}\n"
    "{transcript_block}"
    "The following frames are ordered chronologically within the segment."
)
TRANSCRIPT_BLOCK_TEMPLATE = "Transcript lines:\n{lines}\n"
NO_TRANSCRIPT_LINE = "- No transcript lines overlap this segment."
FRAME_TS_TEMPLATE = "Frame timestamp: {ts}"
USER_TRAILER = "Write the caption for this segment."


def _hms(seconds: float) -> str:
    # Round (not truncate) — matches Binlu multiscale_segment.hms() so the
    # timestamps in the VLM prompt line up with where speech/actions actually
    # happen (truncation drifts by up to a second per chunk boundary).
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}"


# ── SRT parsing & subtitle alignment ──────────────────────────────────────────

_SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _srt_to_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(srt_path: str | Path) -> List[Tuple[float, float, str]]:
    """Parse SRT file → list of (start_sec, end_sec, text)."""
    entries: List[Tuple[float, float, str]] = []
    text_lines: List[str] = []
    t_start = t_end = 0.0
    in_block = False

    for raw in Path(srt_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        m = _SRT_TIME_RE.match(line)
        if m:
            if in_block and text_lines:
                txt = _HTML_TAG_RE.sub("", " ".join(text_lines)).strip()
                if txt:
                    entries.append((t_start, t_end, txt))
            t_start = _srt_to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
            t_end   = _srt_to_sec(m.group(5), m.group(6), m.group(7), m.group(8))
            text_lines = []
            in_block = True
        elif in_block:
            if line and not line.isdigit():
                text_lines.append(line)
            elif not line and text_lines:
                txt = _HTML_TAG_RE.sub("", " ".join(text_lines)).strip()
                if txt:
                    entries.append((t_start, t_end, txt))
                text_lines = []
                in_block = False

    if in_block and text_lines:
        txt = _HTML_TAG_RE.sub("", " ".join(text_lines)).strip()
        if txt:
            entries.append((t_start, t_end, txt))
    return entries


def align_subtitles(
    entries: List[Tuple[float, float, str]],
    t_start: float,
    t_end: float,
    max_chars: int = 500,
) -> str:
    """Return subtitle text overlapping with [t_start, t_end]."""
    parts = [txt for es, ee, txt in entries if ee >= t_start and es <= t_end]
    speech = " ".join(parts)
    if len(speech) > max_chars:
        speech = speech[:max_chars].rsplit(" ", 1)[0] + "…"
    return speech


def collect_speech_lines(
    entries: List[Tuple[float, float, str]],
    t_start: float,
    t_end: float,
) -> List[Tuple[float, float, str]]:
    """Return SRT entries overlapping [t_start, t_end], preserving timestamps."""
    return [(es, ee, txt) for es, ee, txt in entries if ee >= t_start and es <= t_end]


# ── Narrative-input frame down-sampler ────────────────────────────────────────

def _select_narrative_frames(
    paths: List[Path], timestamps: List[float], n_max: int,
) -> Tuple[List[Path], List[float]]:
    """Evenly down-sample (paths, timestamps) to at most ``n_max`` items.

    Bounds VLM input cost regardless of how many post-cleanup frames the
    chunk has. Preserves temporal order.
    """
    n = len(paths)
    if n <= n_max:
        return list(paths), list(timestamps)
    step = n / n_max
    idxs = [int(i * step) for i in range(n_max)]
    return [paths[i] for i in idxs], [timestamps[i] for i in idxs]


# ── API endpoint round-robin ──────────────────────────────────────────────────

_endpoint_state: dict = {"endpoints": [], "rr": 0, "lock": None}


def _init_endpoints(api_url: str) -> None:
    _endpoint_state["endpoints"] = [
        u.strip().rstrip("/") for u in api_url.split(",") if u.strip()
    ]
    _endpoint_state["rr"] = 0
    _endpoint_state["lock"] = asyncio.Lock()


async def _pick_endpoint() -> str:
    async with _endpoint_state["lock"]:
        eps = _endpoint_state["endpoints"]
        ep = eps[_endpoint_state["rr"] % len(eps)]
        _endpoint_state["rr"] += 1
    return ep


async def _call_api(
    session, messages, api_model: str, max_tokens: int = 150,
    max_retries: int = 5, retry_base_delay: float = 2.0,
) -> str:
    """Exponential-backoff retry on connection drops / 5xx (matches Binlu
    api_backend.py:79-101). Only retries network-level failures; 4xx and JSON
    parse errors still propagate."""
    import aiohttp
    payload_base = {
        "model": api_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        ep = await _pick_endpoint()
        try:
            async with session.post(
                f"{ep}/v1/chat/completions", json=payload_base,
                timeout=aiohttp.ClientTimeout(total=900),
            ) as r:
                data = await r.json()
            text = data["choices"][0]["message"]["content"]
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            return text.strip()
        except (aiohttp.ClientConnectionError,
                aiohttp.ServerDisconnectedError,
                asyncio.TimeoutError) as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            delay = min(retry_base_delay * (2 ** attempt), 60.0)
            await asyncio.sleep(delay)
    raise last_err  # unreachable


def _img_block(path: Path, max_side: int | None = None) -> dict | None:
    """Long-edge resize → JPEG q=85 → base64 (mirrors Binlu api_backend.py:104-119).
    Matters most for the multi-image narrative call: raw 1fps frames are 1280+
    on the long edge; sending 30 of them un-scaled triples the prompt-token
    budget and bloats attention noise, dragging caption quality down.

    Long-edge cap is env-tunable via VEIL_CAPTION_MAX_SIDE (default 448); set it
    huge (e.g. 10000) to send frames at original resolution for A/B tests."""
    if max_side is None:
        max_side = int(os.environ.get("VEIL_CAPTION_MAX_SIDE", "448"))
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        # Corrupt/unreadable cached frame: drop just this image (caller filters
        # None out) rather than crashing the whole build's asyncio.gather.
        log.warning("skip unreadable frame %s: %s", path, e)
        return None
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


# ── Narrative (single multi-image VLM call per chunk) ─────────────────────────

def _format_transcript_block(
    speech_lines: List[Tuple[float, float, str]],
    video_has_speech: bool,
) -> str:
    """Three-way subtitle handling matching Binlu segment_caption.py:155-165:
    - video has no SRT at all → block omitted entirely
    - video has SRT but this chunk has no overlap → explicit NO_TRANSCRIPT marker
    - chunk has overlapping lines → list with time ranges
    """
    if not video_has_speech:
        return ""
    if speech_lines:
        lines = "\n".join(
            f"- [{_hms(es)} --> {_hms(ee)}] {txt}"
            for es, ee, txt in speech_lines
        )
    else:
        lines = NO_TRANSCRIPT_LINE
    return TRANSCRIPT_BLOCK_TEMPLATE.format(lines=lines)


async def _narrate_chunk(
    session, sem, info: dict, api_model: str, video_has_speech: bool,
) -> str:
    fps = info["narrative_frame_paths"]
    ts = info["narrative_timestamps"]
    speech_lines = info.get("speech_lines", [])
    t_start = info["t_start"]
    t_end = info["t_end"]

    if not fps:
        return ""

    transcript_block = _format_transcript_block(speech_lines, video_has_speech)
    header = USER_HEADER_TEMPLATE.format(
        t0=_hms(t_start), t1=_hms(t_end),
        transcript_block=transcript_block,
    )
    content: list = [{"type": "text", "text": header}]
    for tt, fp in zip(ts, fps):
        block = _img_block(fp)
        if block is None:
            continue  # unreadable frame — drop it, keep the rest of the chunk
        content.append({"type": "text", "text": FRAME_TS_TEMPLATE.format(ts=_hms(tt))})
        content.append(block)
    content.append({"type": "text", "text": USER_TRAILER})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
    async with sem:
        try:
            return await _call_api(
                session, messages, api_model=api_model, max_tokens=400,
            )
        except Exception as e:
            return f"[narrative failed: {e}]"


async def _narrate_all(chunks_info: List[dict], api_model: str, concurrency: int) -> List[str]:
    import aiohttp
    # Per-video flag: if any chunk has SRT overlap, we treat the video as having
    # speech; chunks without overlap then get the NO_TRANSCRIPT marker rather
    # than dropping the transcript block silently.
    video_has_speech = any(
        c.get("speech_lines") or c.get("speech", "").strip()
        for c in chunks_info
    )
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 8)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _narrate_chunk(session, sem, c, api_model, video_has_speech)
            for c in chunks_info
        ]
        return list(await asyncio.gather(*tasks))


def narrate_chunks(
    chunks_info: List[dict], api_url: str, api_model: str, concurrency: int = 16,
) -> List[str]:
    """One multi-image VLM call per chunk → single paragraph caption."""
    _init_endpoints(api_url)
    return asyncio.run(_narrate_all(chunks_info, api_model, concurrency))


# ── Video paths ───────────────────────────────────────────────────────────────

def _get_video_paths(cfg: dict, video_dir: str | None = None) -> dict[str, str]:
    # Generic mode: point at a folder of videos, no benchmark/annotation needed.
    # video_id = filename stem. Builds EVERY video found (recursively).
    if video_dir:
        vdir = Path(video_dir)
        exts = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}
        paths: dict[str, str] = {}
        for p in sorted(vdir.rglob("*")):
            if p.suffix.lower() in exts and p.stem not in paths:
                paths[p.stem] = str(p)
        return paths
    bench = cfg["benchmark"]["name"]
    if bench == "videomme":
        from src.dataloader.videomme import load_videomme
        b = cfg["benchmark"]
        samples = load_videomme(
            parquet_path=b["parquet_path"],
            video_dir=b["video_dir"],
            duration_groups=b.get("duration_groups"),
        )
        return {s.video_id: s.video_path for s in samples}
    elif bench == "lvbench":
        from src.dataloader.lvbench import load_lvbench
        b = cfg["benchmark"]
        samples = load_lvbench(
            meta_path=b["meta_path"],
            video_dir=b["video_dir"],
        )
        return {s.video_id: s.video_path for s in samples}
    elif bench == "longvideobench":
        from src.dataloader.longvideobench import load_longvideobench
        b = cfg["benchmark"]
        samples = load_longvideobench(
            bench_root=b["bench_root"],
            video_dir=b["video_dir"],
        )
        return {s.video_id: s.video_path for s in samples}
    elif bench == "egolife":
        # Each "video" is a person; the value is a FOOTAGE DIRECTORY (thousands of
        # 30-s clips), not a single file. _prepare_person() fans it out.
        from src.dataloader.egolife import egolife_person_dirs
        return egolife_person_dirs(cfg["benchmark"]["egolife_root"])
    else:
        raise ValueError(f"Unsupported benchmark: {bench}")


# ── EgoLife: per-clip incremental builder ──────────────────────────────────────

def build_egolife_person(person, person_dir, out_dir, siglip, embedder, args, cfg, ml):
    """Build ONE person's bank from their ~30-s clips, CLIP = CHUNK (no grouping).

    EgoLife clips are already atomic 30-s units; SigLIP grouping always collapses a
    clip's frames into a single group anyway, so we skip it: each clip → one chunk
    (its ≤30 frames + aligned speech → one caption). The build is clip-incremental
    and RESUMABLE — each clip's result is cached under _clips/<stem>/ (chunk.json +
    chunk_vec.npz) and its extracted frames under frames_raw/0000/. A crash loses at
    most the in-flight batch; a restart reuses every cached clip and every extracted
    frame (no re-ffmpeg). The person bank (narrative.json + vectors.npz) is
    check-pointed after every batch, then the ml3_n12 bank is emitted once at the end.
    """
    import tempfile
    from src.dataloader.egolife import egolife_clip_plan, _speech_for
    from src.build_memory.core.frame_pipeline import (
        extract_frames, is_blank, is_blank_pixel, is_blurry, select_sharpest,
        _persist_frame, BLUR_FAILSAFE_RATIO,
    )

    egolife_root = cfg["benchmark"]["egolife_root"]
    person_out = out_dir / person
    person_out.mkdir(parents=True, exist_ok=True)
    narr_path, vec_path = person_out / "narrative.json", person_out / "vectors.npz"

    clips, transcript = egolife_clip_plan(egolife_root, person)
    log.info("[%s] %d clips, %d transcript lines", person, len(clips), len(transcript))
    if not clips:
        log.warning("[%s] no clips at %s", person, person_dir)
        return

    def _cdir(clip):
        return person_out / "_clips" / clip.stem

    def _extract(clip, tmp):
        """NVDEC decode on --extract-gpu, per-clip CPU fallback on any failure."""
        if args.extract_gpu is not None:
            try:
                raw = extract_frames(str(clip), tmp, fps=args.fps,
                                     hwaccel_device=str(args.extract_gpu))
                if raw:
                    return raw
            except Exception as e:  # noqa: BLE001 — NVDEC hiccup → fall back to CPU
                log.warning("[%s] %s NVDEC failed (%s), CPU fallback", person, clip.name, e)
            for f in tmp.glob("*.jpg"):
                f.unlink(missing_ok=True)
        return extract_frames(str(clip), tmp, fps=args.fps, hwaccel_device="")

    import shutil

    def _frames_for(clip):
        """(survivor_frames, tmp_dir_to_clean_or_None).

        Reuse persisted frames_raw/0000 if present (tmp=None, keep them). Else extract
        to a /tmp SSD dir + blank/blur filter. With --no-persist-frames we do NOT copy
        the ~30 frames to the (saturated HDD) frames_raw: caption reads them straight
        from /tmp and the consumer deletes the dir after the batch (only the keyframe
        is persisted). Resume then re-extracts, which is cheap via NVDEC."""
        raw0 = _cdir(clip) / "frames_raw" / "0000"
        have = sorted(raw0.glob("*.jpg")) if raw0.is_dir() else []
        if have:
            return have, None
        tmp = Path(tempfile.mkdtemp(prefix="veil_ego_"))
        raw = _extract(clip, tmp)
        surv: List = []
        if raw:
            # is_blank = 损坏/空帧(文件过小); is_blank_pixel = 全黑/纯色均匀帧(无内容).
            # 暗但有细节的帧不再被 is_blank_pixel 误删(核心滤镜已改为只按 std), 会保留
            # 进 caption. 只有整段全是损坏/纯黑均匀帧时 nb 才为空 → 跳过该 clip.
            nb = [p for p in raw if not is_blank(p) and not is_blank_pixel(p)]
            if nb:
                flags = [is_blurry(p) for p in nb]
                surv = nb if sum(flags) / len(flags) > BLUR_FAILSAFE_RATIO \
                    else [p for p, bl in zip(nb, flags) if not bl]
        if not surv:
            shutil.rmtree(tmp, ignore_errors=True)
            return [], None
        if args.no_persist_frames:
            return surv, tmp                       # keep on /tmp SSD; consumer cleans
        raw0.mkdir(parents=True, exist_ok=True)     # legacy: persist to HDD for reuse
        out = []
        for i, src in enumerate(surv):
            dst = raw0 / f"{i:04d}.jpg"
            _persist_frame(src, dst, None)
            out.append(dst)
        shutil.rmtree(tmp, ignore_errors=True)
        return out, None

    def _keyframe(clip, frames):
        kfo = _cdir(clip) / "keyframes_origin" / "0000_0.jpg"
        if not kfo.exists():
            kfo.parent.mkdir(parents=True, exist_ok=True)
            idx = select_sharpest([str(p) for p in frames], list(range(len(frames)))) \
                if len(frames) > 1 else 0
            _persist_frame(frames[idx], kfo, None)
            kfr = _cdir(clip) / "keyframes_resized" / "0000_0.jpg"
            kfr.parent.mkdir(parents=True, exist_ok=True)
            _persist_frame(frames[idx], kfr, args.frame_max_dim)
        return kfo

    def _checkpoint(acc):
        acc2 = sorted(acc, key=lambda x: x[0])
        chunks, nv, vv, ids = [], [], [], []
        for cid, (_abs, d, tv, vvec) in enumerate(acc2):
            dd = dict(d); dd["chunk_id"] = cid
            chunks.append(dd); nv.append(tv); vv.append(vvec); ids.append(cid)
        dur = acc2[-1][1]["end_time"] if acc2 else 0.0
        narr_path.write_text(json.dumps(
            {"video_id": person, "duration": dur, "num_chunks": len(chunks),
             "chunks": chunks}, ensure_ascii=False))
        np.savez(vec_path,
                 narrative_vecs=np.array(nv, dtype=np.float32),
                 visual_vecs=np.array(vv, dtype=np.float32),
                 chunk_ids=np.array(ids, dtype=np.int64))

    # ── Resume: load cached clips; queue the rest ───────────────────────────────
    acc, todo = [], []          # acc: (abs_start, chunk_dict, text_vec, visual_vec)
    for abs_start, clip in clips:
        cj, cv = _cdir(clip) / "chunk.json", _cdir(clip) / "chunk_vec.npz"
        if cj.exists() and cv.exists():
            try:
                d = json.loads(cj.read_text()); z = np.load(cv)
                acc.append((abs_start, d, z["text_vec"], z["visual_vec"]))
                continue
            except Exception:
                pass
        todo.append((abs_start, clip))
    log.info("[%s] resume: %d cached, %d to build", person, len(acc), len(todo))

    # ── Per-clip prep (ffmpeg extract + blank/blur filter + keyframe pick). This
    #    is CPU/IO-bound and independent per clip, so it runs on a thread pool
    #    while the previous batch's captions are still hitting the VLM. ─────────
    from concurrent.futures import ThreadPoolExecutor
    def _prep_one(abs_start, clip):
        frames, tmp = _frames_for(clip)
        if not frames:
            log.warning("[%s] %s no frames — skip", person, clip.name)
            return None
        N = len(frames)
        ts = [abs_start + i for i in range(N)]               # fps=1 ⇒ 1 s/frame
        t0, t1 = abs_start, abs_start + N
        fsel, tsel = _select_narrative_frames(frames, ts, args.narrative_max_frames)
        sl, sp = _speech_for(transcript, t0, t1)
        kf = _keyframe(clip, frames)                          # keyframe → HDD (eval)
        return {"abs": abs_start, "clip": clip, "kf": kf, "sp": sp, "tmp": tmp,
                "ts": ts, "t0": t0, "t1": t1,
                "ci": {"narrative_frame_paths": fsel,
                       "narrative_timestamps": tsel, "speech": sp,
                       "speech_lines": sl, "t_start": t0, "t_end": t1}}

    t_all = time.time()
    B = max(args.api_concurrency * 3, 48)
    prep_pool = ThreadPoolExecutor(max_workers=max(1, args.prep_workers))

    # ── Producer/consumer pipeline ──────────────────────────────────────────────
    # A background thread preps the NEXT batch (ffmpeg extract + keyframe pick,
    # CPU/GPU0-bound) while the main thread captions the CURRENT batch on the vLLM
    # endpoints. Without this, prep and caption ping-pong: during a batch's long
    # prep all 5 endpoints sit idle, and during its caption GPU0 sits idle. With a
    # 2-deep queue the endpoints stay continuously fed → ~1.5-2x throughput.
    from queue import Queue
    from threading import Thread
    SENTINEL: object = object()
    batch_q: Queue = Queue(maxsize=2)

    def _producer():
        try:
            for bi in range(0, len(todo), B):
                batch = todo[bi:bi + B]
                preps = [p for p in prep_pool.map(lambda ac: _prep_one(*ac), batch) if p]
                batch_q.put(preps)
        finally:
            batch_q.put(SENTINEL)

    Thread(target=_producer, name="ego-prep", daemon=True).start()

    # The durable resume state is the per-clip chunk.json/chunk_vec.npz (written
    # every clip below). The full-bank narrative.json (grows to 8 MB+) + vectors.npz
    # (30 MB+) are only the eval artifact, so rewriting them EVERY batch just floods
    # the shared disk (balance_dirty_pages throttling). Rewrite them every N batches
    # instead; a crash loses nothing (resume rebuilds from the per-clip caches).
    CKPT_EVERY = 10
    batch_i = 0
    while True:
        preps = batch_q.get()
        if preps is SENTINEL:
            break
        if not preps:
            continue
        # One SigLIP call for the batch's keyframes → per-chunk visual vecs.
        vvs = siglip.encode_images([str(p["kf"]) for p in preps])
        narrs = narrate_chunks([p["ci"] for p in preps], args.api_url,
                               args.api_model, concurrency=args.api_concurrency)
        etexts = [f"{n}\n\n{p['sp']}" if p["sp"].strip() else n
                  for n, p in zip(narrs, preps)]
        tvs = embedder.encode(etexts)
        for i, (p, narr) in enumerate(zip(preps, narrs)):
            cd = _cdir(p["clip"])
            kf_rel = str(p["kf"].relative_to(person_out))
            d = {"start_time": p["t0"], "end_time": p["t1"], "narrative": narr,
                 "caption": [], "speech_text": p["sp"], "sampled_frames": p["ts"],
                 "keyframe_ts": (p["t0"] + p["t1"]) / 2.0, "keyframe_path": kf_rel}
            tv = np.asarray(tvs[i], dtype=np.float32)
            vv = np.asarray(vvs[i], dtype=np.float32)
            (cd / "chunk.json").write_text(json.dumps(d, ensure_ascii=False))
            np.savez(cd / "chunk_vec.npz", text_vec=tv, visual_vec=vv)
            acc.append((p["abs"], d, tv, vv))
        # Frames lived on /tmp only for captioning (--no-persist-frames) — drop now.
        for p in preps:
            if p.get("tmp"):
                shutil.rmtree(p["tmp"], ignore_errors=True)
        batch_i += 1
        if batch_i % CKPT_EVERY == 0:
            _checkpoint(acc)          # periodic full-bank flush (eval artifact)
        log.info("[%s] built %d/%d clips (%.0fs elapsed)",
                 person, len(acc), len(clips), time.time() - t_all)

    prep_pool.shutdown(wait=True)
    _checkpoint(acc)
    log.info("[%s] saved %d chunks (%.0fs)", person, len(acc), time.time() - t_all)

    if ml is not None:
        from src.build_memory.build_multilayer import build_multilayer_for_video
        ml_out, ml_llm, ml_cfg, ml_pool = ml
        try:
            msg = build_multilayer_for_video(
                person_out, ml_out / person, ml_llm, embedder, ml_cfg, ml_pool)
            log.info("[%s] multilayer %s", person, msg)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] multilayer failed: %s", person, e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--benchmark", choices=["videomme", "lvbench", "egolife", "longvideobench"])
    src.add_argument("--config", help="Legacy YAML path")
    ap.add_argument("--video-dir", default=None,
                    help="Generic mode: build every video in this folder (recursive, "
                         "video_id = filename stem); overrides benchmark video enumeration. "
                         "Still needs --config or --benchmark for the BGE model path / out dir.")
    ap.add_argument("--siglip-model",     required=True)
    ap.add_argument("--siglip-gpu",       default="cuda:0")
    ap.add_argument("--bge-gpu",          default="cuda:0")
    ap.add_argument("--subtitle-dir",     default=None)
    ap.add_argument("--filter-from",      default=None)
    # ── EgoLife only: a "video" is a person (dir of ~30-s clips on a global
    #    timeline). --people gates which persons to build. ──
    ap.add_argument("--people", nargs="*", default=None,
                    help="EgoLife: subset of person keys to build (e.g. A1_JAKE).")
    ap.add_argument("--prep-workers", type=int, default=8,
                    help="EgoLife: threads for per-clip frame extract+keyframe "
                         "(ffmpeg/cv2, CPU-bound). Runs while captions hit the VLM.")
    ap.add_argument("--extract-gpu", default=None,
                    help="EgoLife: GPU index for ffmpeg NVDEC frame decode (e.g. 4). "
                         "Offloads decode from the contended CPU (~4x faster on "
                         "1408p clips). Falls back to CPU per-clip if NVDEC fails.")
    ap.add_argument("--no-persist-frames", action="store_true",
                    help="EgoLife: do NOT copy the ~30 per-clip frames to frames_raw "
                         "on disk. They stay on the /tmp SSD only for captioning, then "
                         "are deleted; only the keyframe is persisted. Slashes writes "
                         "to a saturated HDD; resume re-extracts (cheap via NVDEC).")
    ap.add_argument("--fps",              type=float, default=1.0)
    ap.add_argument("--theta",            type=float, default=0.80)
    ap.add_argument("--n-max",            type=int,   default=30)
    ap.add_argument("--api-url",          required=True,
                    help="vLLM OpenAI-compatible base URL(s); comma-separated"
                         " = round-robin")
    ap.add_argument("--api-model",        required=True)
    ap.add_argument("--api-concurrency",  type=int, default=16)
    ap.add_argument("--out-dir",          default=None)
    ap.add_argument("--narrative-max-frames", type=int, default=30,
                    help="Cap frames per chunk fed to the single multi-image VLM"
                         " call; evenly down-sampled from g.all_frame_paths when"
                         " exceeded")
    ap.add_argument("--frame-max-dim", type=int, default=448,
                    help="Longer-side target for keyframes_resized/. Origin"
                         " keyframes and frames_raw are always full res."
                         " Default 448 matches Binlu save_keyframe (build_index.py).")
    # ── One-command pipeline: also emit the self-contained multi-layer (ml3_n12)
    #    bank per video right after its single-layer bank is written. ──
    ap.add_argument("--emit-multilayer", action="store_true",
                    help="After each video's single-layer bank, build its "
                         "self-contained ml3_n12 multi-layer bank (layers + photos "
                         "together) so one command goes video → final bank.")
    ap.add_argument("--multilayer-out-dir", default=None,
                    help="Output dir for the multi-layer banks. Default: sibling of "
                         "--out-dir named '<out>_ml3_n12'.")
    ap.add_argument("--summary-api-url", default=None,
                    help="LLM endpoint(s) for layer summaries (default: --api-url).")
    ap.add_argument("--summary-api-model", default=None,
                    help="Model name for layer summaries (default: --api-model).")
    ap.add_argument("--summary-concurrency", type=int, default=32,
                    help="Cap on concurrent summary LLM calls (--emit-multilayer).")
    from src.build_memory.build_multilayer import add_multilayer_build_args
    add_multilayer_build_args(ap)
    args = ap.parse_args()

    if args.benchmark is not None:
        from src.build_memory.core import specs as build_specs
        cfg = build_specs.cfg_for_similarity_build(args.benchmark)
    else:
        cfg = load_config(args.config)
        bn = cfg.get("benchmark", {}).get("name")
        if bn in ("videomme", "lvbench") and not (cfg.get("memory") or {}).get("cache_dir"):
            from src.build_memory.core import specs as build_specs
            cfg.setdefault("memory", {})["cache_dir"] = str(
                build_specs.similarity_memory_cache_dir(bn)
            )

    mem_cfg = cfg.get("memory") or {}
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif mem_cfg.get("cache_dir"):
        out_dir = Path(mem_cfg["cache_dir"])
    else:
        ap.error("Pass --out-dir, or use --benchmark / YAML with memory.cache_dir set.")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir.resolve())

    if args.subtitle_dir:
        subtitle_dir = Path(args.subtitle_dir)
    else:
        sub_yaml = cfg.get("benchmark", {}).get("subtitle_dir")
        subtitle_dir = Path(sub_yaml) if sub_yaml else None
    if subtitle_dir:
        log.info("Subtitle dir: %s", subtitle_dir)

    # ── Models ────────────────────────────────────────────────────────────────
    log.info("Loading SigLIP on %s …", args.siglip_gpu)
    t0 = time.time()
    from src.clients.siglip_embedder import SigLIPEmbedder
    siglip = SigLIPEmbedder(args.siglip_model, device=args.siglip_gpu)
    log.info("  SigLIP ready (%.1fs)", time.time() - t0)

    log.info("API mode — VLM/LLM at %s (%s)", args.api_url, args.api_model)

    log.info("Loading BGE-M3 on %s …", args.bge_gpu)
    t0 = time.time()
    from src.clients.embedder import BGEM3Embedder
    embedder = BGEM3Embedder(
        model_path=cfg["models"]["embedder"]["model_path"],
        use_fp16=cfg["models"]["embedder"].get("use_fp16", True),
        device=args.bge_gpu,
    )
    log.info("  BGE-M3 ready (%.1fs)", time.time() - t0)

    # ── Multi-layer (ml3_n12) one-command pipeline setup ───────────────────────
    ml_out_dir = ml_llm = ml_pool = ml_cfg = None
    if args.emit_multilayer:
        from concurrent.futures import ThreadPoolExecutor
        from src.build_memory.build_multilayer import cfg_from_args
        from src.clients.llm_client import LLMClient
        ml_out_dir = (Path(args.multilayer_out_dir) if args.multilayer_out_dir
                      else out_dir.with_name(out_dir.name + "_ml3_n12"))
        ml_out_dir.mkdir(parents=True, exist_ok=True)
        s_url   = args.summary_api_url or args.api_url
        s_model = args.summary_api_model or args.api_model
        ml_llm = LLMClient(model_path=s_model, api_url=s_url, api_model=s_model)
        ml_pool = ThreadPoolExecutor(max_workers=args.summary_concurrency)
        ml_cfg = cfg_from_args(args)
        log.info("Emit-multilayer ON → %s (frame_mode=%s include_raw=%s)",
                 ml_out_dir, ml_cfg["frame_mode"], ml_cfg["include_raw"])

    # ── Video list ─────────────────────────────────────────────────────────────
    video_paths = _get_video_paths(cfg, args.video_dir)
    log.info("Found %d videos", len(video_paths))

    if args.filter_from:
        keep = set()
        for line in Path(args.filter_from).open():
            try:
                keep.add(json.loads(line)["video_id"])
            except Exception:
                pass
        video_paths = {k: v for k, v in video_paths.items() if k in keep}
        log.info("Filtered to %d videos", len(video_paths))

    if args.people:  # EgoLife: gate which persons to build
        want = set(args.people)
        video_paths = {k: v for k, v in video_paths.items() if k in want}
        log.info("Filtered to %d persons: %s", len(video_paths), list(video_paths))

    # ── EgoLife: dedicated clip-incremental builder (clip=chunk, resumable). It
    #    has its own loop instead of the per-video producer/consumer below, since
    #    a person is thousands of clips that must be built + check-pointed one at a
    #    time. lvbench/videomme keep the shared path untouched. ─────────────────
    if (cfg.get("benchmark", {}) or {}).get("name") == "egolife":
        ml = (ml_out_dir, ml_llm, ml_cfg, ml_pool) if args.emit_multilayer else None
        for person, person_dir in video_paths.items():
            build_egolife_person(person, person_dir, out_dir, siglip, embedder, args, cfg, ml)
        if ml_pool is not None:
            ml_pool.shutdown(wait=True)
            log.info("Multilayer banks in %s", ml_out_dir)
        log.info("Done — output in %s", out_dir)
        return

    from src.build_memory.core.frame_pipeline import clean_and_group_frames
    from queue import Queue
    from threading import Thread

    # ── Pipeline: prepare next video on a worker thread while main runs VLM ──
    # The prepare stage is ffmpeg-extract + SigLIP-encode/group + SRT parse,
    # which runs on CPU + GPU 1 (SigLIP). The build stage is the VLM caption
    # (vLLM 8000+8001 on GPU 1/3) + BGE encode + disk write. By the time the
    # main thread finishes captioning video N (~150 s), the worker has already
    # finished preparing video N+1 (~40-70 s), so the next caption can start
    # immediately. Queue size 1 caps the worker at one video ahead of main.

    def _prepare_video(video_id: str, video_path: str):
        """Pre-caption stage. Returns dict for build, or None to skip."""
        video_dir = out_dir / video_id
        narr_path = video_dir / "narrative.json"
        vec_path  = video_dir / "vectors.npz"
        if narr_path.exists() and vec_path.exists():
            log.info("[%s] already built, skipping", video_id)
            return None
        if not Path(video_path).exists():
            log.warning("[%s] not found: %s", video_id, video_path)
            return None

        srt_entries: List[Tuple[float, float, str]] = []
        if subtitle_dir is not None:
            srt_path = subtitle_dir / f"{video_id}.srt"
            if srt_path.exists():
                try:
                    srt_entries = parse_srt(srt_path)
                    log.info("[%s] %d subtitle entries", video_id, len(srt_entries))
                except Exception as e:
                    log.warning("[%s] SRT parse failed: %s", video_id, e)
            else:
                log.warning("[%s] no SRT at %s", video_id, srt_path)

        log.info("[%s] starting …", video_id)
        t_video = time.time()

        cleaned = clean_and_group_frames(
            video_path=video_path,
            out_dir=video_dir,
            siglip=siglip,
            fps=args.fps,
            theta=args.theta,
            n_max=args.n_max,
            frame_max_dim=args.frame_max_dim,
        )
        duration = max((g.t_end for g in cleaned), default=0.0)
        log.info("  [%s] %d chunks, %d kept frames total",
                 video_id, len(cleaned),
                 sum(len(g.kept_frame_paths) for g in cleaned))

        speech_texts = [
            align_subtitles(srt_entries, g.t_start, g.t_end) if srt_entries else ""
            for g in cleaned
        ]
        speech_lines_per_chunk = [
            collect_speech_lines(srt_entries, g.t_start, g.t_end) if srt_entries else []
            for g in cleaned
        ]
        picked = [
            _select_narrative_frames(g.all_frame_paths, g.all_timestamps,
                                     args.narrative_max_frames)
            for g in cleaned
        ]
        chunks_info = [
            {
                "narrative_frame_paths": fps,
                "narrative_timestamps":  ts,
                "speech":        sp,
                "speech_lines":  sl,
                "t_start": g.t_start,
                "t_end":   g.t_end,
            }
            for g, sp, sl, (fps, ts) in zip(
                cleaned, speech_texts, speech_lines_per_chunk, picked,
            )
        ]
        return {
            "video_id":     video_id,
            "video_dir":    video_dir,
            "narr_path":    narr_path,
            "vec_path":     vec_path,
            "cleaned":      cleaned,
            "duration":     duration,
            "speech_texts": speech_texts,
            "chunks_info":  chunks_info,
            "t_video":      t_video,
        }

    # Iterate videos in the order they came in (videomme loader returns samples
    # by sample_idx, and dict preserves first-seen insertion order). For
    # --filter-from, the filtered dict also preserves the same order.
    SENTINEL: object = object()
    prep_queue: Queue = Queue(maxsize=1)

    def _producer():
        try:
            for video_id, video_path in video_paths.items():
                try:
                    item = _prepare_video(video_id, video_path)
                except Exception as e:
                    log.exception("[%s] prepare failed: %s", video_id, e)
                    item = None
                prep_queue.put(item)
        finally:
            prep_queue.put(SENTINEL)

    prod_thread = Thread(target=_producer, name="prepare", daemon=True)
    prod_thread.start()

    while True:
        item = prep_queue.get()
        if item is SENTINEL:
            break
        if item is None:
            continue
        video_id     = item["video_id"]
        video_dir    = item["video_dir"]
        narr_path    = item["narr_path"]
        vec_path     = item["vec_path"]
        cleaned      = item["cleaned"]
        duration     = item["duration"]
        speech_texts = item["speech_texts"]
        chunks_info  = item["chunks_info"]
        t_video      = item["t_video"]

        t0 = time.time()
        narratives = narrate_chunks(
            chunks_info, args.api_url, args.api_model,
            concurrency=args.api_concurrency,
        )
        log.info("  [%s] narratives done (%.1fs)", video_id, time.time() - t0)

        embed_texts = [
            f"{narr}\n\n{sp}" if sp.strip() else narr
            for narr, sp in zip(narratives, speech_texts)
        ]
        sem_vecs = embedder.encode(embed_texts)

        # v_visual lives in vectors.npz["visual_vecs"] (one row per chunk in the
        # same order as narrative_vecs / chunk_ids), NOT inside narrative.json.
        narr_chunks = []
        n_vecs = []
        v_vecs = []
        chunk_ids = []
        for g, narr, speech in zip(cleaned, narratives, speech_texts):
            center = g.center_idx
            kf_path = g.kept_frame_paths[center]
            try:
                kf_rel = str(kf_path.relative_to(video_dir))
            except ValueError:
                kf_rel = str(kf_path)
            narr_chunks.append({
                "chunk_id":       g.chunk_id,
                "start_time":     g.t_start,
                "end_time":       g.t_end,
                "narrative":      narr,
                "caption":        [],
                "speech_text":    speech,
                "sampled_frames": g.all_timestamps,
                "keyframe_ts":    g.kept_timestamps[center],
                "keyframe_path":  kf_rel,
            })
            n_vecs.append(sem_vecs[g.chunk_id])
            v_vecs.append(g.kept_v_visual[center])
            chunk_ids.append(g.chunk_id)

        narr_json = {
            "video_id":   video_id,
            "duration":   duration,
            "num_chunks": len(narr_chunks),
            "chunks":     narr_chunks,
        }
        narr_path.write_text(json.dumps(narr_json, ensure_ascii=False))
        np.savez(
            vec_path,
            narrative_vecs=np.array(n_vecs, dtype=np.float32),
            visual_vecs=np.array(v_vecs, dtype=np.float32),
            chunk_ids=np.array(chunk_ids, dtype=np.int64),
        )
        log.info("[%s] saved %d chunks (%.1fs total)",
                 video_id, len(narr_chunks), time.time() - t_video)

        # ── One-command pipeline: build this video's self-contained ml3_n12
        #    bank from the single-layer bank we just wrote (layers + photos). ──
        if args.emit_multilayer:
            from src.build_memory.build_multilayer import build_multilayer_for_video
            t_ml = time.time()
            try:
                msg = build_multilayer_for_video(
                    video_dir, ml_out_dir / video_id, ml_llm, embedder, ml_cfg, ml_pool,
                )
                log.info("  [%s] multilayer %s (%.1fs)", video_id, msg, time.time() - t_ml)
            except Exception as e:  # noqa: BLE001
                log.exception("[%s] multilayer build failed: %s", video_id, e)

    prod_thread.join()
    if ml_pool is not None:
        ml_pool.shutdown(wait=True)
        log.info("Multilayer banks in %s", ml_out_dir)
    log.info("Done — output in %s", out_dir)


if __name__ == "__main__":
    main()
