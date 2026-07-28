"""Canonical benchmark paths, output dirs, and default hyper-parameters for memory builders.

Not used by ``experiments/run_experiments.py`` (that reads ``configs/*.yaml``); it exists so
``src.build_memory.build_single_similarity`` does not duplicate path logic, and the legacy
fixedframe builder can reuse the same path defaults.

Eval / baselines use ``configs/*.yaml``. Builders load this module unless paths are
overridden via CLI flags or ``--config`` YAML.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

BenchName = Literal["videomme", "lvbench", "egolife", "longvideobench"]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # …/REVEAL (core → build_memory → src → REVEAL)
_OUTPUTS_ROOT = _PROJECT_ROOT / "outputs"


def outputs_root() -> Path:
    return _OUTPUTS_ROOT


def benchmark_videomme() -> dict[str, Any]:
    return {
        "name": "videomme",
        "parquet_path": "../../Datas/VideoMME/videomme/test-00000-of-00001.parquet",
        "video_dir": "../../Datas/VideoMME/videos",
        "subtitle_dir": "../../Datas/VideoMME/subtitle",
        "duration_groups": ["long"],
    }


def benchmark_lvbench() -> dict[str, Any]:
    return {
        "name": "lvbench",
        "meta_path": "../../Datas/LVBench/video_info.meta.jsonl",
        "video_dir": "../../Datas/LVBench/videos",   # {key}.mp4 (download separately)
        "subtitle_dir": None,
        "duration_groups": None,
    }


def benchmark_egolife() -> dict[str, Any]:
    # EgoLife has no single video file per sample: each "video" is a person whose
    # footage is thousands of 30-s clips under {egolife_root}/{PERSON}/. The build
    # engine enumerates clips + transcript via src.dataloader.egolife. Speech comes
    # from EgoLifeCap/Transcript SRTs (resolved per chunk on the global timeline),
    # so subtitle_dir stays None (that knob is the single-SRT-per-video path).
    return {
        "name": "egolife",
        "egolife_root": "../../Datas/EgoLife",
        "subtitle_dir": None,
        "duration_groups": None,
    }


def benchmark_longvideobench() -> dict[str, Any]:
    # LongVideoBench: one mp4 per video (val = 753 videos / 1337 Qs). Subtitles are
    # Whisper-V3 JSON (``{id}_en.json``) converted to ``{id}.srt`` by
    # scripts/lvb_json2srt.py into ``subtitle_dir``, so the existing SRT injection
    # path (parse_srt → align_subtitles) attaches them per chunk unchanged.
    return {
        "name": "longvideobench",
        "bench_root": "../../Datas/LongVideoBench",
        "video_dir": "../../Datas/LongVideoBench/videos",   # {video_id}.mp4
        "subtitle_dir": "../../Datas/LongVideoBench/subtitles_srt",  # {video_id}.srt
        "duration_groups": None,
    }


# benchmark name → its config builder
_BENCH_BUILDERS = {
    "videomme": benchmark_videomme,
    "lvbench": benchmark_lvbench,
    "egolife": benchmark_egolife,
    "longvideobench": benchmark_longvideobench,
}


def _bench(name: BenchName) -> dict[str, Any]:
    return _BENCH_BUILDERS[name]()


def embedder_bge_m3(device: str = "cuda:0") -> dict[str, Any]:
    return {
        "model_path": "../../Models/BAAI/bge-m3",
        "use_fp16": True,
        "device": device,
        "batch_size": 16,
        "max_length": 4096,
    }


def similarity_memory_cache_dir(name: BenchName) -> Path:
    return _OUTPUTS_ROOT / "memory" / {
        "videomme": "videomme_L_27b_27b",
        "lvbench": "lvbench_single",
        "egolife": "egolife_single",
        "longvideobench": "longvideobench_single",
    }[name]


def cfg_for_similarity_build(name: BenchName, bge_device: str = "cuda:0") -> dict[str, Any]:
    """Minimal config dict for ``src.build_memory.build_single_similarity`` (benchmark + embedder + out dir)."""
    bench = _bench(name)
    out = str(similarity_memory_cache_dir(name))
    return {
        "paths": {"outputs_root": str(_OUTPUTS_ROOT)},
        "benchmark": bench,
        "models": {"embedder": embedder_bge_m3(device=bge_device)},
        "memory": {"cache_dir": out},
    }


# ── Fixed wall-clock (event / dense) build defaults ─────────────────────────

FIXEDFRAME_DEFAULT_VLM = "../../Models/Qwen/Qwen3-VL-32B-Instruct"
FIXEDFRAME_DEFAULT_MAX_CHUNKS = 900
FIXEDFRAME_DEFAULT_RESOLUTION = 448

# Keys match CLI mode names "event" | "dense"
FIXEDFRAME_MODE_PARAMS: dict[str, dict[str, float | int]] = {
    "event": {"chunk_size_sec": 4.0, "frames_per_chunk": 3},
    "dense": {"chunk_size_sec": 1.0, "frames_per_chunk": 1},
}


def cfg_for_fixedframe_build(name: BenchName) -> dict[str, Any]:
    """Benchmark + paths for the legacy fixedframe CLI."""
    bench = _bench(name)
    return {
        "paths": {"outputs_root": str(_OUTPUTS_ROOT)},
        "benchmark": bench,
    }
