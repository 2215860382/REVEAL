"""Cold backfill: give an existing multi-layer bank its own copy of the photos.

Older ``build_multilayer`` runs recorded each L1 chunk's keyframe as an ABSOLUTE
path into the source single-layer bank (``videomme_multiframe``), so the pyramid
bank held no images of its own. This script materialises those photos INTO the
bank (default ``copy``) and rewrites ``L1.jsonl`` ``frame_paths`` to RELATIVE
paths, making the bank self-contained — WITHOUT touching captions or vectors, so
a definitive eval bank stays byte-identical in content.

It is CPU-only (no models) and idempotent: frames already present are not
re-copied, and rows already relative are left as-is (their files are still
verified/backfilled when a ``--src-memory-dir`` is given).

Usage:
    PYTHONPATH=. python -m src.build_memory.backfill_frames \\
        --bank-dir outputs/memory/videomme_ml3_n12 \\
        [--src-memory-dir outputs/memory/videomme_multiframe] \\
        [--no-include-raw] [--frame-mode copy] [--limit N] [--dry-run]

``--src-memory-dir`` is only needed when a row's ``frame_paths`` are already
relative (no absolute source to resolve); for legacy absolute-path banks it is
inferred from the path itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.logging import get_logger
from src.build_memory.build_multilayer import _place_chunk_frames

log = get_logger("backfill_frames")


def _resolve_src_vd(frame_path: str, video_id: str, src_root: Path | None) -> tuple[Path, str] | None:
    """Return (src_video_dir, kf_name) for one L1 frame_path, or None if unresolvable.

    Absolute legacy path ``.../<vid>/keyframes_resized/<name>`` → parents[1] is the
    source video dir. Relative path ``keyframes_resized/<name>`` → resolve against
    ``src_root/<video_id>`` when provided.
    """
    p = Path(frame_path)
    name = p.name
    if p.is_absolute():
        # .../<vid>/keyframes_resized/<name>  → parents[1] == <vid> dir
        return p.parents[1], name
    if src_root is not None:
        return src_root / video_id, name
    return None


def _backfill_video(bank_vd: Path, src_root: Path | None, mode: str, include_raw: bool,
                    dry_run: bool) -> str:
    l1_path = bank_vd / "L1.jsonl"
    if not l1_path.exists():
        return "skip (no L1.jsonl)"
    rows = [json.loads(l) for l in l1_path.open()]
    changed = 0
    copied = 0
    for row in rows:
        fps = row.get("frame_paths") or []
        if not fps:
            continue
        new_fps = []
        for fp in fps:
            res = _resolve_src_vd(fp, bank_vd.name, src_root)
            if res is None:
                new_fps.append(fp)  # already relative & no src to backfill from
                continue
            src_vd, name = res
            if not dry_run:
                _place_chunk_frames(src_vd, bank_vd, name, mode, include_raw)
            copied += 1
            new_fps.append(f"keyframes_resized/{name}")
        if new_fps != fps:
            row["frame_paths"] = new_fps
            changed += 1
    if not dry_run and changed:
        with l1_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta_path = bank_vd / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["frame_mode"] = mode
            meta["include_raw"] = include_raw
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return f"{'DRY ' if dry_run else ''}rows={len(rows)} rewritten={changed} frames={copied}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True, help="Multi-layer bank root ({root}/{vid}/L1.jsonl)")
    ap.add_argument("--src-memory-dir", default=None,
                    help="Single-layer bank root; only needed if frame_paths are already relative")
    ap.add_argument("--frame-mode", choices=["hardlink", "copy", "symlink"], default="copy",
                    help="How to bring frames in (default copy = true self-contained duplicate)")
    ap.add_argument("--include-raw", dest="include_raw", action="store_true", default=True)
    ap.add_argument("--no-include-raw", dest="include_raw", action="store_false")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bank_root = Path(args.bank_dir)
    src_root = Path(args.src_memory_dir) if args.src_memory_dir else None
    vids = sorted(p for p in bank_root.iterdir() if p.is_dir() and (p / "L1.jsonl").exists())
    if args.limit:
        vids = vids[: args.limit]
    log.info("backfill %d videos (mode=%s include_raw=%s dry_run=%s)",
             len(vids), args.frame_mode, args.include_raw, args.dry_run)
    for i, vd in enumerate(vids, 1):
        msg = _backfill_video(vd, src_root, args.frame_mode, args.include_raw, args.dry_run)
        log.info("[%d/%d] %-16s %s", i, len(vids), vd.name, msg)
    log.info("done -> %s", bank_root)


if __name__ == "__main__":
    main()
