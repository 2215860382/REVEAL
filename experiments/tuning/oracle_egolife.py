"""Oracle upper-bound for EgoLifeQA (A1_JAKE): bypass retrieval entirely and feed
the answerer the GOLD chunk window located by each question's ``target_time``.

Purpose: separate "retrieval caps accuracy" from "caption content caps accuracy".
  oracle >> baseline  ⇒ the bank CONTENT is fine, retrieval is the bottleneck
                        (better recall recovers points → EgoLifeQA could be higher)
  oracle ≈ baseline   ⇒ even the correct segment's caption lacks the answer
                        (bank/caption is the cap → needs a rebuild)

Text-only, single answerer call per question (no iteration / verifier / rerank),
matching the 59% baseline answerer. Window = covering chunk ± W (env W, default 2).
"""
from __future__ import annotations
import os, sys, json, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataloader.egolife import load_egolife, _clip_abs_start
from src.build_memory.core.bank_loader import load_bank
from src.agents.answerer import Answerer
from src.clients.vlm_client import VLMClient

ROOT       = Path("/home2/ycj/Project/REVEAL")
MEMDIR     = ROOT / "outputs/memory/egolife"
BENCH_ROOT = "/home2/ycj/Datas/EgoLife_bench"
VIDEO_DIR  = "/home2/ycj/Datas/EgoLife"
PERSON     = os.environ.get("ORACLE_PERSON", "A1_JAKE")
W          = int(os.environ.get("W", "2"))            # window half-width in chunks
WORKERS    = int(os.environ.get("WORKERS", "6"))
API        = os.environ["API"].split(",")
OUT        = ROOT / f"outputs/results/egolife/oracle_{PERSON.split('_')[0].lower()}_w{W}.jsonl"


def _abs(tt: dict):
    if not tt or not tt.get("time") or not tt.get("date"):
        return None
    try:
        day = int(str(tt["date"]).upper().replace("DAY", ""))
        return _clip_abs_start(day, str(tt["time"]))
    except Exception:
        return None


def _l1_ev(c):
    parts = [f"[{c.start_time:.0f}s-{c.end_time:.0f}s]", c.memory_text]
    if getattr(c, "asr", "").strip():
        parts.append(f"Speech: {c.asr}")
    return "\n".join(parts)


def gold_window(bank, t_abs, cutoff):
    """Chunks covering t_abs, expanded ±W, restricted to end<=cutoff (past-only)."""
    chunks = sorted(bank.chunks, key=lambda c: c.start_time)
    if cutoff is not None:
        chunks = [c for c in chunks if c.end_time <= cutoff + 1]
    if not chunks:
        return []
    # index of covering (or nearest) chunk
    cover = None
    for i, c in enumerate(chunks):
        if c.start_time <= t_abs <= c.end_time:
            cover = i; break
    if cover is None:
        cover = min(range(len(chunks)),
                    key=lambda i: abs((chunks[i].start_time + chunks[i].end_time) / 2 - t_abs))
    lo = max(0, cover - W); hi = min(len(chunks), cover + W + 1)
    return chunks[lo:hi]


def main():
    print(f"[oracle] person={PERSON} W={W} api={API}", flush=True)
    samples = load_egolife(bench="egolifeqa", bench_root=BENCH_ROOT,
                           video_dir=VIDEO_DIR, filter_people=[PERSON])
    bank = load_bank(MEMDIR / PERSON)
    vlm  = VLMClient(model_path="/home2/ycj/Models/Qwen/Qwen3.5-27B",
                     api_url=",".join(API), api_model="Qwen3.5-27B")
    answerer = Answerer(vlm)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT.open("w")
    lock = threading.Lock()
    stat = {"n": 0, "c": 0, "nogold": 0}

    def run(s):
        t_abs = _abs(s.target_time)
        cutoff = _abs(s.query_time)
        if t_abs is None:
            with lock: stat["nogold"] += 1
            return
        gw = gold_window(bank, t_abs, cutoff)
        ev = [_l1_ev(c) for c in gw]
        res = answerer.answer(s.question, s.candidates, ev, question_first=True)
        letter = (res.get("answer", "") or "")[:1].upper()
        idx = ord(letter) - ord("A") if letter else -1
        pred = s.candidates[idx] if 0 <= idx < len(s.candidates) else ""
        ok = (pred == s.answer)
        with lock:
            stat["n"] += 1; stat["c"] += int(ok)
            fh.write(json.dumps({
                "video_id": s.video_id, "sample_idx": s.sample_idx, "correct": ok,
                "question_type": s.question_type, "pred": pred, "gold": s.answer,
                "n_gold_chunks": len(gw), "t_abs": t_abs,
            }, ensure_ascii=False) + "\n"); fh.flush()
            if stat["n"] % 25 == 0:
                print(f"  {stat['n']} done  acc={100*stat['c']/stat['n']:.1f}%", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(run, samples))
    fh.close()
    n, c = stat["n"], stat["c"]
    print(f"[oracle] DONE  {c}/{n} = {100*c/max(1,n):.2f}%  (no-gold skipped={stat['nogold']})", flush=True)


if __name__ == "__main__":
    main()
