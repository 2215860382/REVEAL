"""Recall@K curve for EgoLifeQA (A1): where does the GOLD chunk (covering each
question's target_time) actually rank in retrieval? This designs the fix for the
+10pt oracle headroom:
  - recall climbs steeply with K  ⇒ just feed more evidence (raise final_top_k)
  - recall stays low even at big K ⇒ dense embedding can't find it → need temporal
    grounding / better representation, not more volume.

Compares DENSE (BGE cosine) vs HYBRID (dense+BM25 RRF). No LLM — pure ranking over
the bank's cached v_text vectors. Respects query_time cutoff (future off-limits).
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataloader.egolife import load_egolife, _clip_abs_start
from src.build_memory.core.bank_loader import load_bank
from src.clients.embedder import BGEM3Embedder
from experiments.core import _hybrid

MEMDIR = Path("/home2/ycj/Project/VEIL/outputs/memory/egolife")
PERSON = "A1_JAKE"
KS = [8, 20, 50, 100, 200, 500, 1000]


def _abs(tt):
    if not tt or not tt.get("time") or not tt.get("date"): return None
    try: return _clip_abs_start(int(str(tt["date"]).upper().replace("DAY","")), str(tt["time"]))
    except Exception: return None


def main():
    samples = load_egolife(bench="egolifeqa", bench_root="/home2/ycj/Datas/EgoLife_bench",
                           video_dir="/home2/ycj/Datas/EgoLife", filter_people=[PERSON])
    bank = load_bank(MEMDIR / PERSON)
    emb  = BGEM3Embedder(model_path="/home2/ycj/Models/BAAI/bge-m3", use_fp16=True,
                         device=os.environ.get("BGE_GPU", "cuda:1"))
    chunks = bank.chunks
    starts = np.array([c.start_time for c in chunks]); ends = np.array([c.end_time for c in chunks])
    docmat = np.array([c.v_text for c in chunks], dtype=np.float32)
    bm25 = _hybrid._BM25([ (c.memory_text + " " + getattr(c,"asr","")) for c in chunks ])

    ranks_d, ranks_h = [], []
    for s in samples:
        t = _abs(s.target_time)
        if t is None: continue
        cutoff = _abs(s.query_time)
        # gold chunk = covering target_time (nearest midpoint if none covers)
        cover = np.where((starts <= t) & (t <= ends))[0]
        gi = int(cover[0]) if len(cover) else int(np.argmin(np.abs((starts+ends)/2 - t)))
        mask = np.ones(len(chunks), bool)
        if cutoff is not None: mask = ends <= cutoff + 1
        if not mask[gi]:
            continue  # gold is after query_time (shouldn't happen) — skip
        qv = emb.encode([s.question])[0]
        dense = docmat @ qv
        dense = np.where(mask, dense, -np.inf)
        # rank of gold in dense (1-based)
        rd = int((dense > dense[gi]).sum()) + 1
        ranks_d.append(rd)
        # hybrid RRF rank
        bm = np.array(bm25.get_scores(s.question), dtype=np.float32)
        order = _hybrid.rrf_order(dense, bm, valid_mask=mask)
        rh = int(np.where(order == gi)[0][0]) + 1
        ranks_h.append(rh)

    rd = np.array(ranks_d); rh = np.array(ranks_h)
    n = len(rd)
    print(f"[recall] A1 可评估题={n}")
    print(f"  gold 排名中位: dense={int(np.median(rd))}  hybrid={int(np.median(rh))}  (总chunk≈{len(chunks)})")
    print(f"  {'K':>6} {'dense R@K':>12} {'hybrid R@K':>12}")
    for k in KS:
        print(f"  {k:>6} {100*(rd<=k).mean():>11.1f}% {100*(rh<=k).mean():>11.1f}%")


if __name__ == "__main__":
    main()
