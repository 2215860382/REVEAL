"""Loader for LVBench (THUDM) — extreme-long-video MCQ benchmark.

Source: video_info.meta.jsonl (103 rows, one per video). Each row:
  key         YouTube video id  (video file = {video_dir}/{key}.mp4)
  type        video category (cartoon / sport / documentary / ...)
  video_info  {duration_minutes, fps, resolution}
  qa          list of questions, each:
      uid, question (stem + "\n(A) ..\n(B) ..\n(C) ..\n(D) .."),
      answer (letter A-D), question_type (list of str), time_reference

Options are embedded in the question text; we split them into `candidates`
and map the answer letter to the gold option text (REVEAL wants candidates +
answer-as-text, question stem separate).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

_OPT_RE = re.compile(r"\n\(([A-E])\)\s*")


@dataclass
class LVBenchSample:
    video_id: str
    video_path: str
    duration: float          # seconds
    question: str            # stem only (options moved to candidates)
    candidates: List[str]
    answer: str              # gold option text (one of candidates)
    question_type: str       # LVBench question_type list, joined by ","
    sample_idx: int


def _split_question(raw: str) -> tuple[str, List[str]]:
    """'stem\\n(A) x\\n(B) y ...' → (stem, [x, y, ...]) in A,B,C,.. order."""
    parts = _OPT_RE.split("\n" + raw.strip())  # leading \n so first marker splits
    stem = parts[0].strip()
    opts = [p.strip() for p in parts[2::2]]     # parts[1::2] = letters, parts[2::2] = texts
    return stem, opts


def load_lvbench(
    meta_path: str | Path,
    video_dir: str | Path,
    duration_groups: Optional[Iterable] = None,   # unused; kept for interface parity
    max_samples: Optional[int] = None,
) -> List[LVBenchSample]:
    video_dir = Path(video_dir)
    rows = [json.loads(l) for l in Path(meta_path).open(encoding="utf-8") if l.strip()]

    samples: List[LVBenchSample] = []
    idx = 0
    for row in rows:
        key = str(row["key"])
        dur = float(row.get("video_info", {}).get("duration_minutes", 0.0)) * 60.0
        vpath = str(video_dir / f"{key}.mp4")
        if not Path(vpath).exists():
            continue  # skip videos whose mp4 isn't downloaded (e.g. LVBench YouTube removals);
                      # keeps build + eval scoped to present videos, sample_idx stays contiguous
        for qa in row.get("qa", []):
            if max_samples is not None and len(samples) >= max_samples:
                return samples
            stem, cands = _split_question(qa["question"])
            letter = str(qa.get("answer", "")).strip().upper()
            ai = ord(letter) - ord("A") if letter[:1].isalpha() else -1
            if not (0 <= ai < len(cands)):
                continue  # skip malformed
            qt = qa.get("question_type")
            qt = ",".join(qt) if isinstance(qt, list) else str(qt or "")
            samples.append(LVBenchSample(
                video_id=key,
                video_path=vpath,
                duration=dur,
                question=stem,
                candidates=cands,
                answer=cands[ai],
                question_type=qt,
                sample_idx=idx,
            ))
            idx += 1
    return samples
