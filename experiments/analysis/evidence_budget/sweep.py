#!/usr/bin/env python
"""Evidence-budget sweep (single-variable volume control).

Reuses mq_k3's ALREADY-RETRIEVED, rubric-reranked evidence (stored as
evidence_chunk_ids in rerank order) and only varies how many top-N segments are
fed to the answerer. Truncating texts[:N] here is byte-identical to running the
pipeline with REVEAL_EVIDENCE_CAP=N, but retrieval is run ONCE so N is the only
variable (no inter-run retrieval noise).

Self-check: N=0 (all) must reproduce mq_k3's headline accuracy (~64.7). If it
does, the reconstruction is faithful and the swept points are trustworthy.

Temp artifact — delete after ROADMAP is updated (per project workflow rule).
"""
import json, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

from src.build_memory.core.bank_loader import load_bank
from src.clients.vlm_client import VLMClient
from src.agents.answerer import Answerer

MEMORY_DIR = ROOT / "outputs" / "memory" / "egolife"
MQ = ROOT / "outputs" / "results" / "egor1" / "analysis" / "mq_k3.jsonl"
OUT = Path(__file__).resolve().parent / "egor1_budget.jsonl"
CAPS = [8, 16, 24, 32, 48, 0]          # 0 = no truncation (all evidence)
API = "http://127.0.0.1:8003"

answerer = Answerer(VLMClient(model_path="Qwen3.5-27B", api_url=API))

_banks: dict = {}
def get_bank(vid):
    if vid not in _banks:
        _banks[vid] = load_bank(MEMORY_DIR / vid)
    return _banks[vid]

def l1_ev(c):
    parts = [f"[{c.start_time:.0f}s-{c.end_time:.0f}s]", c.memory_text]
    if getattr(c, "asr", "").strip():
        parts.append(f"Speech: {c.asr}")
    return "\n".join(parts)

def id_map(bank):
    m = {}
    for c in bank.chunks:              m[c.chunk_id] = ("L1", c)
    for c in bank.l2_chunks:           m[c.chunk_id] = ("L2", c)
    for c in getattr(bank, "l3_chunks", []): m[c.chunk_id] = ("L3", c)
    return m

rows = [json.loads(l) for l in open(MQ) if l.strip()]
data = [r for r in rows if r.get("record_type") != "meta" and "evidence_chunk_ids" in r]
print(f"[budget] {len(data)} questions × {len(CAPS)} caps", flush=True)

def reconstruct(r):
    bank = get_bank(r["video_id"]); mp = id_map(bank)
    texts = []
    for i in r["evidence_chunk_ids"]:
        hit = mp.get(i)
        if hit:
            layer, c = hit
            texts.append(l1_ev(c) if layer == "L1" else c.memory_text)
    cands = r["candidates"]
    gold = r["gold_answer"]
    gold_letter = chr(ord('A') + cands.index(gold)) if gold in cands else None
    return texts, cands, gold_letter

def work(r):
    texts, cands, gold_letter = reconstruct(r)
    out = {"sample_idx": r.get("sample_idx"), "video_id": r.get("video_id"),
           "question_type": r.get("question_type"), "gold_letter": gold_letter,
           "n_available": len(texts), "by_cap": {}}
    for N in CAPS:
        ev = texts if N == 0 else texts[:N]
        letter = answerer.answer(r["question"], cands, ev).get("answer", "")
        out["by_cap"][str(N)] = {"pred": letter, "correct": bool(letter == gold_letter)}
    return out

results = []
with ThreadPoolExecutor(max_workers=3) as ex:
    for i, out in enumerate(ex.map(work, data)):
        results.append(out)
        if (i + 1) % 25 == 0:
            print(f"[budget] {i+1}/{len(data)}", flush=True)

with open(OUT, "w") as f:
    for o in results:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

print("\n==== EVIDENCE-BUDGET SWEEP (egor1) ====", flush=True)
print(f"{'cap N':>8} {'acc%':>7} {'n':>5}")
for N in CAPS:
    ok = sum(1 for o in results if o["by_cap"][str(N)]["correct"])
    n = len(results)
    tag = " (all)" if N == 0 else ""
    print(f"{N:>8}{tag} {100*ok/n:>6.1f} {n:>5}")
print(f"\nwrote {OUT}", flush=True)
