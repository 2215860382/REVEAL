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

BenchName = Literal["mlvu", "videomme", "longvideobench"]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # …/VEIL (core → build_memory → src → VEIL)
_OUTPUTS_ROOT = _PROJECT_ROOT / "outputs"


def outputs_root() -> Path:
    return _OUTPUTS_ROOT


def benchmark_mlvu() -> dict[str, Any]:
    return {
        "name": "mlvu",
        "split": "dev",
        "json_dir": "/home/Dataset/Dataset/MLVU/MLVU/json",
        "video_dir": "/home/Dataset/Dataset/MLVU/MLVU/video",
        "task_types": [
            "plotQA",
            "needle",
            "ego",
            "count",
            "order",
            "anomaly_reco",
            "topic_reasoning",
        ],
        "json_files": {
            "plotQA": "1_plotQA.json",
            "needle": "2_needle.json",
            "ego": "3_ego.json",
            "count": "4_count.json",
            "order": "5_order.json",
            "anomaly_reco": "6_anomaly_reco.json",
            "topic_reasoning": "7_topic_reasoning.json",
            "sub_scene": "8_sub_scene.json",
            "summary": "9_summary.json",
        },
    }


def benchmark_videomme() -> dict[str, Any]:
    return {
        "name": "videomme",
        "parquet_path": "/home2/ycj/Datas/VideoMME/videomme/test-00000-of-00001.parquet",
        "video_dir": "/home2/ycj/Datas/VideoMME/videos",
        "subtitle_dir": "/home2/ycj/Datas/VideoMME/subtitle",
        "duration_groups": ["long"],
    }


def benchmark_longvideobench() -> dict[str, Any]:
    return {
        "name": "longvideobench",
        "json_path": "/home2/ycj/Datas/LongVideoBench/lvb_val.json",
        "video_dir": "/home2/ycj/Datas/LongVideoBench/videos/videos",
        "subtitle_dir": None,
        "duration_groups": None,   # None = all 1337 val Qs; e.g. [3600] = hour-long only
    }


# benchmark name → its config builder
_BENCH_BUILDERS = {
    "mlvu": benchmark_mlvu,
    "videomme": benchmark_videomme,
    "longvideobench": benchmark_longvideobench,
}


def _bench(name: BenchName) -> dict[str, Any]:
    return _BENCH_BUILDERS[name]()


def embedder_bge_m3(device: str = "cuda:0") -> dict[str, Any]:
    return {
        "model_path": "/home2/ycj/Models/BAAI/bge-m3",
        "use_fp16": True,
        "device": device,
        "batch_size": 16,
        "max_length": 4096,
    }


def similarity_memory_cache_dir(name: BenchName) -> Path:
    return _OUTPUTS_ROOT / "memory" / {
        "mlvu": "mlvu_similarity",
        "videomme": "videomme_L_27b_27b",
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

FIXEDFRAME_DEFAULT_VLM = "/home/Dataset/Models/Qwen/Qwen3-VL-32B-Instruct"
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
