"""Shared keyframe loading and visual deduplication helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_keyframe_pil(path: str):
    try:
        from PIL import Image
        p = Path(path)
        if p.is_file():
            return Image.open(p).convert("RGB")
    except Exception:
        pass
    return None


# WorldMM-style dense frames: a chunk stores only ONE keyframe, so feeding pixels
# gives the VLM 1 frame per retrieved moment (sparse). REVEAL_DENSE_FRAMES=N instead
# goes back to the ORIGINAL EgoLife 30 s mp4 for the chunk and samples N frames evenly
# across it — dense temporal coverage of the relevant segment, like WorldMM's
# retrieve-segment-then-extract. Mapping is exact: a chunk's keyframe filename
# (DAY1_A2_ALICE_11093000_0000_0.jpg) is the stem of its source clip mp4
# ({root}/A2_ALICE/DAY1/DAY1_A2_ALICE_11093000.mp4).
_EGOLIFE_ROOT_DEFAULT = "/home2/ycj/Project/WorldMM/data/EgoLife"


def _chunk_source_mp4(chunk, root: str) -> Optional[Path]:
    kp = getattr(chunk, "keyframe_path", "") or ""
    if not kp:
        return None
    parts = Path(kp).name.split("_")          # DAY1 A2 ALICE 11093000 0000 0.jpg
    if len(parts) < 4:
        return None
    day, pa, pb, wall = parts[0], parts[1], parts[2], parts[3]
    person = f"{pa}_{pb}"
    mp4 = Path(root) / person / day / f"{day}_{person}_{wall}.mp4"
    return mp4 if mp4.is_file() else None


def dense_frames_from_original(chunk, n: int, root: Optional[str] = None) -> List:
    """Sample ``n`` PIL frames evenly from the chunk's original EgoLife clip mp4.

    Returns [] if the source mp4 can't be located or decoded — callers fall back
    to the single stored keyframe. Uses decord for in-memory decode (no disk I/O).
    """
    root = root or os.environ.get("REVEAL_EGOLIFE_ROOT", _EGOLIFE_ROOT_DEFAULT)
    mp4 = _chunk_source_mp4(chunk, root)
    if mp4 is None:
        return []
    try:
        from decord import VideoReader, cpu
        from PIL import Image
        vr = VideoReader(str(mp4), ctx=cpu(0))
        tot = len(vr)
        if tot <= 0:
            return []
        if n >= tot:
            idx = list(range(tot))
        else:
            idx = sorted(set(int(i * (tot - 1) / (n - 1)) for i in range(n)))
        batch = vr.get_batch(idx).asnumpy()   # (k, H, W, 3) RGB
        return [Image.fromarray(f) for f in batch]
    except Exception:
        return []


DEFAULT_KEYFRAME_SUBDIR = "keyframes_resized"
LEGACY_KEYFRAME_SUBDIR  = "frames"


def _resolve_subdir(video_dir: Path, subdir: str) -> str:
    """Pick the actual keyframe subdir for this bank.

    New layout is ``keyframes_origin/`` (+ ``keyframes_resized/``); legacy banks
    only have ``frames/``. If the requested subdir is missing, fall back to
    legacy ``frames/`` so old banks still work.
    """
    if (video_dir / subdir).is_dir():
        return subdir
    if (video_dir / LEGACY_KEYFRAME_SUBDIR).is_dir():
        return LEGACY_KEYFRAME_SUBDIR
    return subdir


def keyframe_path(keyframe_dir, video_id: str, chunk_id: int, chunk=None,
                  subdir: str = DEFAULT_KEYFRAME_SUBDIR) -> str:
    """Return the sharpest representative frame for a chunk.

    Looks under ``{keyframe_dir}/{video_id}/{subdir}/`` (default
    ``keyframes_origin``). Falls back to ``frames/`` for legacy banks that
    predate the origin/resized split. ``chunk.keyframe_path`` is consulted
    only when its on-disk file exists; otherwise the chunk_id glob is used so
    the subdir switch (origin ↔ resized) is honoured.
    """
    video_dir = Path(keyframe_dir) / video_id
    actual = _resolve_subdir(video_dir, subdir)

    if chunk is not None:
        kp = getattr(chunk, "keyframe_path", "") or ""
        if kp:
            kp_path = Path(kp)
            if kp_path.is_absolute() and kp_path.is_file():
                return str(kp_path)
            # If chunk.keyframe_path points to a different subdir than the
            # caller wants, swap to the requested actual subdir and try glob.
            name = kp_path.name
            alt = video_dir / actual / name
            if alt.is_file():
                return str(alt)

    matches = sorted((video_dir / actual).glob(f"{chunk_id:04d}_*.jpg"))
    return str(matches[0]) if matches else ""


def _sharpest(paths: List[str]) -> str:
    """Return the path with highest Laplacian variance (sharpest). Falls back to paths[0]."""
    import cv2
    best_path, best_score = paths[0], -1.0
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        score = float(cv2.Laplacian(img, cv2.CV_64F).var())
        if score > best_score:
            best_score, best_path = score, p
    return best_path


def keyframe_paths(keyframe_dir, video_id: str, chunk_id: int, cap: int = 3,
                   chunk=None, subdir: str = DEFAULT_KEYFRAME_SUBDIR) -> List[str]:
    """All surviving frames for a chunk, capped to ``cap``.

    The sharpest frame (by Laplacian variance over existing disk files) is placed
    first so that cap=1 always returns the best frame. ``subdir`` selects
    ``keyframes_origin`` vs ``keyframes_resized``; falls back to legacy
    ``frames/`` if the requested subdir is missing.

    Cross-bank: a text bank (e.g. ml3_n12) stores the absolute keyframe path of
    its source multiframe dir on each chunk; prefer that over globbing
    ``keyframe_dir`` (which holds no images for such banks).
    """
    if chunk is not None:
        kp = getattr(chunk, "keyframe_path", "") or ""
        if kp:
            # --hires-keyframes: swap the resized frame for its full-res origin twin.
            if os.environ.get("REVEAL_HIRES_KEYFRAMES") == "1":
                hp = kp.replace("keyframes_resized", "keyframes_origin")
                if os.path.isfile(hp):
                    kp = hp
            kp_path = Path(kp)
            if kp_path.is_absolute() and kp_path.is_file():
                return [str(kp_path)]

    video_dir = Path(keyframe_dir) / video_id
    actual = _resolve_subdir(video_dir, subdir)
    frames_dir = video_dir / actual
    all_matches = [str(p) for p in sorted(frames_dir.glob(f"{chunk_id:04d}_*.jpg"))]
    # pyramid naming: c{chunk_id:05d}_f*.jpg
    if not all_matches:
        all_matches = [str(p) for p in sorted(frames_dir.glob(f"c{chunk_id:05d}_f*.jpg"))]
    if not all_matches:
        # Last resort: a relative-but-existing keyframe_path whose filename does
        # not match the chunk_id glob (EgoLife per-clip frames, e.g.
        # DAY1_..._0000_0.jpg). Glob-named banks (videomme/lvbench) never reach
        # here, so their behaviour is unchanged.
        if chunk is not None:
            kp = getattr(chunk, "keyframe_path", "") or ""
            if kp and Path(kp).is_file():
                return [str(kp)]
        return []

    best = _sharpest(all_matches) if len(all_matches) > 1 else all_matches[0]
    rest = [p for p in all_matches if p != best]
    return ([best] + rest)[:cap]


def visual_dedup(chunks_with_imgs: List[Tuple], threshold: float = 0.92) -> List:
    """Return visually-unique PIL images (cosine sim of v_visual < threshold)."""
    kept = []
    kept_vecs = []
    for c, img in chunks_with_imgs:
        if img is None:
            continue
        if c.v_visual and kept_vecs:
            v = np.array(c.v_visual, dtype=np.float32)
            mat = np.array(kept_vecs, dtype=np.float32)
            if float(np.max(mat @ v)) >= threshold:
                continue
        kept.append(img)
        kept_vecs.append(c.v_visual if c.v_visual else [])
    return kept


def load_keyframes(
    chunks,
    keyframe_dir,
    video_id: str,
    dedup_threshold: float = 0.92,
    cap: int = 8,
    subdir: str = DEFAULT_KEYFRAME_SUBDIR,
) -> List:
    """Load, dedup, and cap keyframe images for a list of MemoryChunks."""
    pairs = []
    for c in chunks:
        kp = keyframe_path(keyframe_dir, video_id, c.chunk_id, chunk=c, subdir=subdir)
        pairs.append((c, load_keyframe_pil(kp)))
    return visual_dedup(pairs, dedup_threshold)[:cap]
