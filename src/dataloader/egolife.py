"""Loader for EgoLife-family egocentric MCQ benchmarks (Ego-R1-Bench + EgoLifeQA).

Both benchmarks share one schema (per-question dict):
    ID, query_time {date:"DAY1", time:"HHMMSSFF"}, target_time,
    type, need_audio, need_name, trigger (scene context at query_time),
    question / choice_a..choice_d / answer (letter A-D)

The "video" for a question is the questioner's *entire* egocentric life-stream,
so we key each sample by the person (e.g. ``A1_JAKE``); one memory bank per person
covers all their DAY1-DAY7 footage. ``query_time`` marks when the question is asked
(everything after it is future); ``target_time`` marks where the evidence lives.

Benchmark file layout:
    Ego-R1-Bench:  {root}/manual-benchmark/{PERSON}.json   (150 across 6 people)
                   {root}/gemini-benchmark/{PERSON}.json   (150 across 6 people)
    EgoLifeQA:     {root}/EgoLifeQA/EgoLifeQA_{PERSON}.json (500, A1_JAKE only public)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PEOPLE = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]


@dataclass
class EgoLifeSample:
    video_id: str            # person key = memory-bank name (e.g. "A1_JAKE")
    video_path: str          # person footage dir (multi-clip; not a single file)
    duration: float          # unknown at load time → 0.0
    question: str            # question stem (English)
    candidates: List[str]    # [choice_a, choice_b, choice_c, choice_d] text
    answer: str              # gold candidate text (mapped from answer letter)
    question_type: str
    sample_idx: int
    query_time: Dict = field(default_factory=dict)   # {date, time} question asked
    target_time: Dict = field(default_factory=dict)  # {date, time} evidence
    need_audio: bool = False
    trigger: str = ""


def _person_from_path(p: Path) -> str:
    """'.../EgoLifeQA_A1_JAKE.json' or '.../A1_JAKE.json' → 'A1_JAKE'."""
    stem = p.stem
    if stem.startswith("EgoLifeQA_"):
        stem = stem[len("EgoLifeQA_"):]
    return stem


def _files_for(bench: str, root: Path) -> List[Path]:
    if bench in ("egor1_manual", "egor1", "ego-r1"):
        subs = ["manual-benchmark"]
    elif bench == "egor1_gemini":
        subs = ["gemini-benchmark"]
    elif bench == "egor1_both":
        subs = ["manual-benchmark", "gemini-benchmark"]
    elif bench in ("egolifeqa", "egolife"):
        subs = ["EgoLifeQA"]
    else:
        raise ValueError(f"Unknown egolife bench: {bench}")
    files: List[Path] = []
    for sub in subs:
        files += sorted((root / sub).glob("*.json"))
    return files


# EgoLife-family question types are inconsistently spelled across the two benchmarks
# (egolifeqa is clean; Ego-R1 splits the SAME semantic type across English/中文,
# singular/plural, spaced/camel variants). Normalise to one canonical taxonomy so the
# per-type rubric consolidates instead of fragmenting into ~15 thin blocks.
_QTYPE_CANON = {
    "entity log": "EntityLog", "entitylog": "EntityLog",
    "event memory": "EventRecall", "event recollection": "EventRecall",
    "event recall": "EventRecall", "eventrecall": "EventRecall", "事件回忆": "EventRecall",
    "relationmap": "RelationMap", "relation map": "RelationMap",
    "interpersonal relationships": "RelationMap", "interpersonal relationship": "RelationMap",
    "habitinsight": "HabitInsight", "habit insight": "HabitInsight",
    "behavior habit": "HabitInsight", "behavior habits": "HabitInsight",
    "taskmaster": "TaskMaster", "task master": "TaskMaster",
    "future plan": "FuturePlan", "future plans": "FuturePlan",
}


def canon_qtype(t: str) -> str:
    """Map a raw EgoLife-family question type to its canonical name (identity if unknown)."""
    return _QTYPE_CANON.get(str(t or "").strip().lower(), str(t or "").strip())


def load_egolife(
    bench: str,
    bench_root: str | Path,
    video_dir: str | Path | None = None,
    max_samples: Optional[int] = None,
    filter_people: Optional[List[str]] = None,
) -> List[EgoLifeSample]:
    """Load an EgoLife-family MCQ benchmark into REVEAL samples.

    ``bench``: egor1_manual | egor1_gemini | egor1_both | egolifeqa
    ``bench_root``: dir containing the benchmark subfolders / json files.
    ``video_dir``: EgoLife root (…/EgoLife) holding the per-person footage dirs.
    """
    root = Path(bench_root)
    vroot = Path(video_dir) if video_dir else None
    files = _files_for(bench, root)

    samples: List[EgoLifeSample] = []
    idx = 0
    for f in files:
        person = _person_from_path(f)
        if filter_people and person not in filter_people:
            continue
        try:
            rows = json.load(f.open(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(rows, dict):
            rows = list(rows.values())
        for q in rows:
            if max_samples is not None and len(samples) >= max_samples:
                return samples
            cands = [q.get(f"choice_{c}", "") for c in ("a", "b", "c", "d")]
            cands = [c for c in cands if c != ""]
            letter = str(q.get("answer", "")).strip().upper()[:1]
            ai = ord(letter) - ord("A") if letter.isalpha() else -1
            if not (0 <= ai < len(cands)):
                continue  # malformed / missing answer
            samples.append(EgoLifeSample(
                video_id=person,
                video_path=str(vroot / person) if vroot else "",
                duration=0.0,
                question=str(q.get("question", "")).strip(),
                candidates=cands,
                answer=cands[ai],
                question_type=canon_qtype(q.get("type", "")),
                sample_idx=idx,
                query_time=q.get("query_time", {}) or {},
                target_time=q.get("target_time", {}) or {},
                need_audio=bool(q.get("need_audio", False)),
                trigger=str(q.get("trigger", "")).strip(),
            ))
            idx += 1
    return samples


# ── Build-side clip/transcript interface ─────────────────────────────────────
# EgoLife footage is delivered as ~30-s egocentric clips (1 fps ⇒ ≤30 frames each,
# matching the pipeline's n_max=30). A person's memory bank spans ALL their clips on
# a DAY1-anchored global timeline. These helpers turn a person into (ordered clips,
# transcript) so ``build_single_similarity`` can build one bank per person via the
# same engine LVBench/VideoMME use — no separate build script. Speech comes from
# EgoLifeCap/Transcript SRTs (bilingual, speaker-tagged), aligned per chunk.

_CLIP_RE = re.compile(r"DAY(\d+)_(A\d+_[A-Z]+)_(\d{8})\.mp4$")
_SRT_RE = re.compile(r"(A\d+_[A-Z]+)_DAY(\d+)_(\d{8})\.srt$")


def _hhmmssff_to_sec(s: str) -> float:
    """'11094208' → 11*3600 + 9*60 + 42 + 08/100 seconds (last 2 digits = cs)."""
    hh, mm, ss, ff = int(s[0:2]), int(s[2:4]), int(s[4:6]), int(s[6:8])
    return hh * 3600 + mm * 60 + ss + ff / 100.0


def _clip_abs_start(day: int, hhmmssff: str) -> float:
    """Absolute seconds on the person's DAY1-anchored global timeline."""
    return (day - 1) * 86400.0 + _hhmmssff_to_sec(hhmmssff)


def egolife_person_dirs(egolife_root: str | Path) -> Dict[str, str]:
    """{person: footage_dir} for every person with clips under ``egolife_root``."""
    root = Path(egolife_root)
    out: Dict[str, str] = {}
    for person in _PEOPLE:
        pdir = root / person
        if pdir.is_dir():
            out[person] = str(pdir)
    return out


def list_clips(egolife_root: str | Path, person: str) -> List[Tuple[float, Path]]:
    """All of a person's clips as (abs_start_sec, path), time-ordered."""
    out: List[Tuple[float, Path]] = []
    pdir = Path(egolife_root) / person
    for mp4 in pdir.rglob("*.mp4"):
        m = _CLIP_RE.search(mp4.name)
        if not m:
            continue
        day, _, ts = m.groups()
        out.append((_clip_abs_start(int(day), ts), mp4))
    out.sort(key=lambda x: x[0])
    return out


def load_transcript(egolife_root: str | Path, person: str) -> List[Tuple[float, float, str]]:
    """Person's Transcript SRTs → [(abs_start, abs_end, 'Speaker: text'), …]."""
    from src.build_memory.build_single_similarity import parse_srt  # lazy: avoid cycle

    tdir = Path(egolife_root) / "EgoLifeCap" / "Transcript" / person
    entries: List[Tuple[float, float, str]] = []
    if not tdir.is_dir():
        return entries
    for srt in tdir.rglob("*.srt"):
        m = _SRT_RE.search(srt.name)
        if not m:
            continue
        _, day, ts = m.groups()
        base = _clip_abs_start(int(day), ts)
        for es, ee, txt in parse_srt(srt):
            txt = txt.strip()
            if txt:
                entries.append((base + es, base + ee, txt))
    entries.sort(key=lambda x: x[0])
    return entries


def _speech_for(entries: List[Tuple[float, float, str]], t0: float, t1: float):
    """Transcript lines overlapping [t0, t1] → (speech_lines, speech_text)."""
    lines = [(s, e, txt) for s, e, txt in entries if e >= t0 and s <= t1]
    return lines, "\n".join(txt for _, _, txt in lines)


def egolife_clip_plan(egolife_root: str | Path, person: str):
    """(clips, transcript) for a person: clips=[(abs_start, Path)], transcript entries."""
    return list_clips(egolife_root, person), load_transcript(egolife_root, person)
