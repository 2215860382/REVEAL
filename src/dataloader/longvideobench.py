"""Loader for LongVideoBench (NeurIPS'24 D&B) — long-context interleaved
video+subtitle MCQ benchmark. Validation split: 1337 Qs / 753 videos.

Source: ``lvb_val.json`` (one row per question). Each row already carries the
options as a list, so — unlike LVBench — no option-splitting is needed:
  video_id, video_path ("{id}.mp4"), question, candidates (list[str]),
  correct_choice (0-based index into candidates), duration (seconds),
  question_category, level, subtitle_path ("{id}_en.json"),
  starting_timestamp_for_subtitles.

Subtitles themselves are NOT read here: the build reads per-video ``.srt`` from
``subtitle_dir`` (converted from the JSON by ``scripts/lvb_json2srt.py``), reusing
the existing SRT alignment path. This loader serves both the build (video paths)
and the eval (samples), mirroring ``src/dataloader/lvbench.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class LongVideoBenchSample:
    video_id: str
    video_path: str
    duration: float          # seconds
    question: str
    candidates: List[str]
    answer: str              # gold option text (one of candidates)
    question_type: str       # question_category (+ level), for reporting
    sample_idx: int


def load_longvideobench(
    bench_root: str | Path,
    video_dir: str | Path,
    subtitle_dir: Optional[str | Path] = None,   # unused here; build reads SRTs itself
    duration_groups: Optional[Iterable] = None,
    max_samples: Optional[int] = None,
) -> List[LongVideoBenchSample]:
    bench_root = Path(bench_root)
    video_dir = Path(video_dir)
    rows = json.loads((bench_root / "lvb_val.json").read_text(encoding="utf-8"))

    allowed_duration_groups = (
        {int(group) for group in duration_groups}
        if duration_groups is not None else None
    )

    samples: List[LongVideoBenchSample] = []
    idx = 0
    for row in rows:
        if (allowed_duration_groups is not None
                and int(row.get("duration_group", -1)) not in allowed_duration_groups):
            continue
        if max_samples is not None and len(samples) >= max_samples:
            break
        vid = str(row["video_id"])
        vpath = str(video_dir / str(row["video_path"]))
        if not Path(vpath).exists():
            continue  # skip videos whose mp4 isn't present; keep sample_idx contiguous
        cands = list(row.get("candidates") or [])
        ci = row.get("correct_choice")
        if not isinstance(ci, int) or not (0 <= ci < len(cands)):
            continue  # skip malformed
        qt = str(row.get("question_category", "") or "")
        lvl = str(row.get("level", "") or "")
        samples.append(LongVideoBenchSample(
            video_id=vid,
            video_path=vpath,
            duration=float(row.get("duration", 0.0) or 0.0),
            question=str(row.get("question", "")),
            candidates=cands,
            answer=cands[ci],
            question_type=(f"{qt}/{lvl}" if lvl else qt),
            sample_idx=idx,
        ))
        idx += 1
    return samples
