#!/usr/bin/env python
"""VideoMME evidence-VOLUME sweep (single-variable, no-method baseline).

Question: on Video-MME-L, where each L1 caption is short and a video holds a few
HUNDRED L1 segments (median 311), how far does *raw retrieval volume* alone get
you?  We strip the whole method down to: single query (the raw question) -> plain
dense BGE top-N over L1 captions -> same answerer.  No multi-query decomposition,
no L2/L3 summaries, no rerank, no rubric verify.  We sweep N and watch accuracy.

Contrast (added at plot time): REVEAL's full pipeline feeds ~39 targeted+verified
segments and reaches 78.9.  If the volume curve only matches that with hundreds
of chunks -- or plateaus below it -- then quality/verification, not quantity, is
the lever.

Alignment: questions/candidates/gold/video_id are taken verbatim from
abl_5_full.jsonl so the 900-question set is identical to the mainline.

Temp artifact -- integrate result into ROADMAP/figure then delete per workflow rule.
"""
import json, os, sys, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("NO_PROXY", "127.0.0.1,10.82.1.145,localhost")
os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import numpy as np
from src.build_memory.core.bank_loader import load_bank
from src.clients.embedder import BGEM3Embedder
from src.clients.vlm_client import VLMClient
from src.agents.answerer import Answerer

MEM   = ROOT / "outputs" / "memory" / "videommeL_1344"   # SAME bank as new pool mainline (vmme_pool_text)
SRC   = ROOT / "outputs" / "results" / "videomme" / "ablation" / "vmme_pool_text.jsonl"
OUT   = Path(__file__).resolve().parent / "videomme_volume.jsonl"
EMB   = "http://127.0.0.1:9000"
APIS  = ("http://127.0.0.1:8003,http://127.0.0.1:8006,"
         "http://10.82.1.145:8002,http://10.82.1.145:8004")
CAPS  = [8, 16, 32, 64, 128, 256, 384, 512]   # top-N fed to answerer (extend to 100s)
WORKERS = 40

emb = BGEM3Embedder(api_url=EMB)
answerer = Answerer(VLMClient(model_path="Qwen3.5-27B", api_url=APIS))

# ---- load the 900 questions (identical set to mainline) -------------------
rows = []
for l in open(SRC):
    d = json.loads(l)
    if "correct" in d and d.get("question"):
        rows.append(d)
print(f"[vol] {len(rows)} questions", flush=True)

# ---- resume: skip questions already written -------------------------------
done = set()
if OUT.exists():
    for l in open(OUT):
        try:
            d = json.loads(l)
            done.add((d["video_id"], d["sample_idx"]))
        except Exception:
            pass
print(f"[vol] resume: {len(done)} already done", flush=True)
_wlock = threading.Lock()
_wf = open(OUT, "a")

# ---- per-video L1 vectors + chunks (cache) --------------------------------
_bank, _vec = {}, {}
def get_video(vid):
    if vid not in _bank:
        b = load_bank(MEM / vid)
        v = np.load(MEM / vid / "L1_text.npz")["vectors"].astype(np.float32)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        _bank[vid], _vec[vid] = b, v
    return _bank[vid], _vec[vid]

def l1_ev(c):
    parts = [f"[{c.start_time:.0f}s-{c.end_time:.0f}s]", c.memory_text]
    if getattr(c, "asr", "").strip():
        parts.append(f"Speech: {c.asr}")
    return "\n".join(parts)

# ---- embed all queries in batches -----------------------------------------
qtexts = [r["question"] for r in rows]
qvecs = []
B = 64
for i in range(0, len(qtexts), B):
    qvecs.append(emb.encode(qtexts[i:i + B]))
    print(f"[vol] embedded {min(i+B,len(qtexts))}/{len(qtexts)}", flush=True)
qvecs = np.concatenate(qvecs, 0)
qvecs = qvecs / (np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-12)

def work(args):
    idx, r, qv = args
    try:
        return _work(args)
    except Exception as e:                 # one bad video must not kill the pool
        print(f"[vol] SKIP {r['video_id']}#{r.get('sample_idx')}: {e}", flush=True)
        return None

def _work(args):
    idx, r, qv = args
    b, v = get_video(r["video_id"])
    sim = v @ qv
    order = np.argsort(-sim)[:max(CAPS)]
    texts = [l1_ev(b.chunks[int(j)]) for j in order]
    cands = r["candidates"]; gold = r["gold_answer"]
    gl = chr(ord('A') + cands.index(gold)) if gold in cands else None
    out = {"sample_idx": r.get("sample_idx"), "video_id": r["video_id"],
           "question_type": r.get("question_type"), "gold_letter": gl,
           "n_available": len(texts), "by_cap": {}}
    for N in CAPS:
        ev = texts[:N]
        letter = answerer.answer(r["question"], cands, ev).get("answer", "")
        out["by_cap"][str(N)] = {"pred": letter, "correct": bool(letter == gl)}
    with _wlock:                       # incremental append -> crash/resume safe
        _wf.write(json.dumps(out, ensure_ascii=False) + "\n"); _wf.flush()
    return out

tasks = [(i, r, qvecs[i]) for i, r in enumerate(rows)
         if (r["video_id"], r.get("sample_idx")) not in done]
print(f"[vol] {len(tasks)} to run this session", flush=True)
results = []
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for i, o in enumerate(ex.map(work, tasks)):
        results.append(o)
        if (i + 1) % 50 == 0:
            print(f"[vol] answered {i+1}/{len(tasks)}", flush=True)
_wf.close()

# reload full file for final tally (includes prior-session rows)
results = [json.loads(l) for l in open(OUT)]
print("\n==== VIDEOMME EVIDENCE-VOLUME SWEEP (single-query dense top-N) ====", flush=True)
print(f"{'cap N':>7} {'acc%':>7} {'n':>5}")
for N in CAPS:
    ok = sum(1 for o in results if o["by_cap"][str(N)]["correct"])
    n = len(results)
    print(f"{N:>7} {100*ok/n:>6.1f} {n:>5}")
print(f"\nwrote {OUT}", flush=True)
