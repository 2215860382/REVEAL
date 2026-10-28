"""Measure the REAL single-stream L2/L3 summarization time for Video-MME-Long.

Rebuttal Q2 reports every benchmark's offline L2/L3 summarization time as a
MEASURED number except Video-MME-L, which was left as an estimate ("~12 h,
estimated at the measured per-summary rate"). This harness replaces that estimate
with a direct measurement.

How it works — it re-runs the exact production L2/L3 build
(``build_multilayer.build_multilayer_for_video``) over the Video-MME-L
single-layer source banks (``outputs/memory/videommeL_single``), reusing the
existing L1 captions (NO re-captioning, NO frame extraction). Every leaf summary
call ``_summarize`` is wrapped to record its wall latency, so:

    single-stream L2/L3 time  =  Σ (latency of every _summarize call)

which is independent of how many summary workers run concurrently — running with
many workers only shortens the wall clock, not this sum. That sum is exactly the
"single-stream build time" the rebuttal quotes for the other four banks.

Output goes to a throwaway temp dir with ``frame_mode=reference`` (no images
copied) and a dummy zero-embedder (BGE not needed — we discard the bank), so the
ONLY GPU work is the summary LLM calls we are timing.

Run (when a Qwen3.5-27B endpoint is free):

    PYTHONPATH=. python scripts/measure_vmme_l2l3_time.py \
        --llm-api-url http://127.0.0.1:8003 \
        --summary-concurrency 32 \
        --workers 8

Add ``--limit N`` for a smoke test over the first N videos (the reported
single-stream total is the Σ-latency, so a partial run still gives a valid
per-summary rate you can sanity-check against the full run).
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.build_memory import build_multilayer as bm
from src.build_memory.build_multilayer import build_multilayer_for_video, cfg_from_args, add_multilayer_build_args


# ── Thread-safe accumulators for the timing wrap ──────────────────────────────
class Meter:
    def __init__(self):
        self.lock = threading.Lock()
        self.n = 0                # number of _summarize calls (= L2 + L3 groups)
        self.sum_latency = 0.0    # Σ wall seconds  → single-stream time
        self.out_words = 0        # Σ output words (token proxy)

    def add(self, latency: float, out_text: str):
        with self.lock:
            self.n += 1
            self.sum_latency += latency
            self.out_words += len((out_text or "").split())


METER = Meter()
_orig_summarize = bm._summarize


def _timed_summarize(llm, group, budget):
    t0 = time.time()
    out = _orig_summarize(llm, group, budget)
    METER.add(time.time() - t0, out)
    return out


class _ZeroEmbedder:
    """Stand-in for BGE — we discard the bank, so vectors don't matter and we
    avoid occupying a GPU for embedding. Returns zeros of the right shape."""

    def encode(self, texts):
        return np.zeros((len(texts), 1024), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-memory-dir", default="outputs/memory/videommeL_single",
                    help="Video-MME-L single-layer source banks (L1 captions reused)")
    ap.add_argument("--out-memory-dir", default="/tmp/vmme_l2l3_timing",
                    help="Throwaway output dir (bank discarded; only timing matters)")
    ap.add_argument("--llm-api-url", required=True, help="Qwen3.5-27B endpoint(s), comma-separated")
    ap.add_argument("--llm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--workers", type=int, default=8, help="Videos in flight")
    ap.add_argument("--summary-concurrency", type=int, default=32,
                    help="Shared cap on concurrent summary LLM calls (wall-clock only)")
    ap.add_argument("--limit", type=int, default=None, help="First N videos (smoke)")
    add_multilayer_build_args(ap)
    args = ap.parse_args()

    # Force reference frame mode: never copy images, we only time summaries.
    args.frame_mode = "reference"
    cfg = cfg_from_args(args)

    # Install the timing wrap.
    bm._summarize = _timed_summarize

    from src.clients.llm_client import LLMClient
    llm = LLMClient(model_path=args.llm_api_model, api_url=args.llm_api_url,
                    api_model=args.llm_api_model)
    embedder = _ZeroEmbedder()

    src_root = Path(args.src_memory_dir)
    out_root = Path(args.out_memory_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    vids = sorted(p for p in src_root.iterdir()
                  if p.is_dir() and ((p / "narrative.json").exists() or (p / "L1.jsonl").exists()))
    if args.limit:
        vids = vids[: args.limit]
    print(f"[measure] {len(vids)} Video-MME-L videos; workers={args.workers} "
          f"summary_concurrency={args.summary_concurrency}", flush=True)

    summ_pool = ThreadPoolExecutor(max_workers=args.summary_concurrency)
    done = [0]
    lock = threading.Lock()
    wall_t0 = time.time()

    def work(src_vd):
        try:
            build_multilayer_for_video(src_vd, out_root / src_vd.name, llm, embedder, cfg, summ_pool)
        except Exception as e:  # noqa: BLE001
            print(f"[measure] {src_vd.name} FAILED: {e}", flush=True)
        with lock:
            done[0] += 1
            n = done[0]
        if n % 20 == 0 or n == len(vids):
            with METER.lock:
                ss = METER.sum_latency
                nn = METER.n
            print(f"[measure] {n}/{len(vids)} videos | summaries={nn} "
                  f"| single-stream so far={ss/3600:.2f} h | wall={ (time.time()-wall_t0)/60:.1f} min",
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, p) for p in vids]
        for fu in as_completed(futs):
            if fu.exception():
                print(f"[measure] worker exc: {fu.exception()}", flush=True)
    summ_pool.shutdown(wait=True)

    wall = time.time() - wall_t0
    n, ss, words = METER.n, METER.sum_latency, METER.out_words
    print("\n==================== VMME L2/L3 MEASURED ====================", flush=True)
    print(f" videos                 : {len(vids)}", flush=True)
    print(f" summaries (L2+L3)      : {n}", flush=True)
    print(f" single-stream time     : {ss/3600:.2f} h  ({ss:.0f} s)", flush=True)
    print(f" mean per summary       : {ss/max(n,1):.2f} s", flush=True)
    print(f" output words (L2+L3)   : {words:,}  (~{words/0.75/1e6:.1f}M tokens @0.75 w/tok)", flush=True)
    print(f" wall clock (this run)  : {wall/60:.1f} min", flush=True)
    print("=============================================================", flush=True)


if __name__ == "__main__":
    main()
