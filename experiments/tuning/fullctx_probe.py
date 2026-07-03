"""Full-context recall probe — answer each question from ALL chunk texts of its
video, with NO retrieval. Decides whether the bottleneck is retrieval recall or
the evidence representation / answerer.

Reuses the exact evidence-text format the live pipeline builds
(`[Ns-Ms]\\n<memory_text>` + optional `Speech: <asr>`), the same VideoMME loader,
the same Answerer, and writes a jsonl compatible with the tuning comparison.

Text-only on purpose: a long video has too many keyframes to feed, so this is the
recall upper bound for the TEXT channel. Reading:
  acc >> live (74%)  -> retrieval is dropping the answer-bearing chunk (planner lever).
  acc ~= live        -> the right text is already reaching the answerer; the lever is
                        the memory representation (lossy narrative) or the answerer.

Usage:
    PYTHONPATH=. python experiments/tuning/fullctx_probe.py \\
        --config configs/veil.yaml \\
        --memory-dir outputs/memory/videomme_multiframe \\
        --vlm-api-url http://127.0.0.1:8003,http://127.0.0.1:8004 \\
        --sample-indices-file <splits>/val_indices.txt \\
        --out outputs/results/videomme/tuning/split_runs/val_fullctx.jsonl \\
        --pipeline-name val_fullctx --workers 24
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config
from src.utils.logging import get_logger
from experiments.tuning.veil_27b import load_samples

log = get_logger("fullctx_probe")


def _build_full_evidence(bank, max_chars: int) -> list[str]:
    """All chunks of the video, in temporal order, in the live evidence format."""
    texts: list[str] = []
    for c in bank.chunks:
        parts = [f"[{c.start_time:.0f}s-{c.end_time:.0f}s]", c.memory_text]
        asr = getattr(c, "asr", "") or ""
        if asr.strip():
            parts.append(f"Speech: {asr}")
        texts.append("\n".join(parts))
    # Defensive global cap (answerer also truncates, but keep the budget explicit).
    if max_chars and sum(len(t) for t in texts) > max_chars and texts:
        per = max(1, max_chars // len(texts))
        texts = [t[:per] for t in texts]
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--memory-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm-api-url", required=True)
    ap.add_argument("--vlm-api-model", default=None)
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--sample-indices-file", default=None)
    ap.add_argument("--pipeline-name", default="fullctx")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-evidence-chars", type=int, default=120000,
                    help="Global char budget across all chunk texts (~30k tokens).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench = cfg["benchmark"]["name"]
    memory_dir = Path(args.memory_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from src.clients.vlm_client import VLMClient
    from src.agents.answerer import Answerer
    from src.build_memory.core.bank_loader import load_bank
    from src.eval.parse_answer import parse_letter

    vlm_model = args.vlm_model or cfg["models"]["vlm"]["model_path"]
    vlm = VLMClient(model_path=vlm_model, api_url=args.vlm_api_url, api_model=args.vlm_api_model)
    answerer = Answerer(vlm)
    log.info("VLM ready via %s", args.vlm_api_url)

    samples = load_samples(cfg, None)
    if args.sample_indices_file:
        sel: set[int] = set()
        for line in Path(args.sample_indices_file).open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            sel.add(int(v["sample_idx"] if isinstance(v, dict) else v))
        samples = [s for s in samples if int(s.sample_idx) in sel]
    log.info("loaded %d samples (%d videos)", len(samples), len({s.video_id for s in samples}))

    # Resume
    done_keys: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                rec = json.loads(line)
                if rec.get("record_type"):
                    continue
                done_keys.add(f"{rec['video_id']}|{rec['sample_idx']}")
            except Exception:
                pass

    _bank_cache: dict = {}
    _cache_lock = threading.Lock()

    def get_bank(vid: str):
        with _cache_lock:
            if vid in _bank_cache:
                return _bank_cache[vid]
        vd = memory_dir / vid
        if not (vd / "narrative.json").exists() and not (vd / "L1.jsonl").exists():
            return None
        bank = load_bank(vd)
        with _cache_lock:
            _bank_cache[vid] = bank
        return bank

    out_fh = out_path.open("a")
    _write_lock = threading.Lock()
    _stats_lock = threading.Lock()
    if (not out_path.exists()) or out_path.stat().st_size == 0:
        out_fh.write(json.dumps({"record_type": "header", "benchmark": bench,
                                 "pipeline": args.pipeline_name}, ensure_ascii=False) + "\n")
        out_fh.flush()

    correct_by: dict = {}
    total = len(samples)
    done_count = [0]

    def process(s):
        if f"{s.video_id}|{s.sample_idx}" in done_keys:
            return
        t0 = time.time()
        bank = get_bank(s.video_id)
        if bank is None:
            pred_letter, correct, n_ev = "", False, 0
        else:
            ev_texts = _build_full_evidence(bank, args.max_evidence_chars)
            n_ev = len(ev_texts)
            res = answerer.answer(s.question, s.candidates, ev_texts,
                                  max_evidence_chars=args.max_evidence_chars)
            raw = res.get("answer", "")
            pred_letter = raw[:1].upper() if raw else (parse_letter(raw, len(s.candidates)) or "")
            idx = ord(pred_letter) - ord("A") if pred_letter else -1
            pred_text = s.candidates[idx] if 0 <= idx < len(s.candidates) else ""
            correct = (pred_text == s.answer)
        elapsed = time.time() - t0
        rec = {"video_id": s.video_id, "sample_idx": s.sample_idx, "correct": correct,
               "question_type": s.question_type, "gold_answer": s.answer,
               "pred_letter": pred_letter, "n_chunks": n_ev, "elapsed": round(elapsed, 2)}
        with _write_lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
        with _stats_lock:
            done_count[0] += 1
            correct_by.setdefault(s.question_type, []).append(int(correct))
            n = done_count[0]
        log.info("[%d/%d] %-18s | %-12s | %s %s | %d chunks %.1fs",
                 n, total, s.question_type, s.video_id, pred_letter,
                 "✓" if correct else "✗", n_ev, elapsed)

    tasks = [s for s in samples if f"{s.video_id}|{s.sample_idx}" not in done_keys]
    log.info("tasks: %d (workers=%d)", len(tasks), args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, s) for s in tasks]
        for f in as_completed(futs):
            if f.exception():
                log.error("worker exc: %s", f.exception())
    out_fh.close()

    print("\n=== full-context recall probe ===")
    qts = sorted(correct_by)
    print(f"{args.pipeline_name:<18}" + "".join(f"{q[:8]:>10}" for q in qts) + f"{'Overall':>10}")
    row = f"{args.pipeline_name:<18}"
    allc = []
    for q in qts:
        c = correct_by[q]
        row += f"{sum(c)/len(c)*100:>9.1f}%"
        allc += c
    row += f"{(sum(allc)/len(allc)*100 if allc else 0):>9.1f}%"
    print(row)
    log.info("done — %s", out_path)


if __name__ == "__main__":
    main()
