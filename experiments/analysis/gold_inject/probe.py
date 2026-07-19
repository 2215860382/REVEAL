#!/usr/bin/env python
"""Gold-chunk injection probe (EgoLife family) — is the caption lossy, and does
going back to the raw frames / subtitles at target_time recover the evidence?

For each MCQ we FORCE-INJECT the chunk that temporally covers ``target_time`` and
answer under four modality arms, so retrieval is taken out of the loop entirely:

  A  gold chunk CAPTION text only            -> is the evidence in the caption?
  B  A + the 1 stored keyframe of that chunk  -> does the sparse stored frame help?
  C  A + frames re-decoded from the RAW video -> user's hypothesis: caption dropped
        densely around target_time                the visual detail; re-extract recovers it
  D  A + transcript/subtitle lines at target  -> caption dropped the subtitle detail

Accuracy is then decomposed by need_audio / need_name / question type, so we can
see WHICH residual is visual (C helps), which is audio/subtitle (D helps), and
which is neither (answerer/reasoning ceiling).

Runs retrieval-free, so it only needs the VLM endpoint (no embed server).

  cd /home2/ycj/Project/REVEAL
  setsid nohup /home2/ycj/miniconda3/envs/videorag/bin/python \
      experiments/analysis/gold_inject/probe.py \
      --bench egor1 --workers 6 \
      > experiments/analysis/gold_inject/egor1.log 2>&1 &

All artifacts of this probe live in experiments/analysis/gold_inject/ :
  probe.py                 this script
  {bench}_goldinject.jsonl per-question 4-arm predictions (resumable)
  {bench}.log              run log + final summary
  figs/                    figures (made from the jsonl)
"""
from __future__ import annotations
import argparse, json, os, re, sys, glob, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative --out / memory paths resolve from repo root
os.environ.setdefault("NO_PROXY", "127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1")
os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from PIL import Image
import numpy as np
import decord  # noqa
from src.clients.vlm_client import VLMClient
from src.agents.answerer import Answerer
from src.dataloader.egolife import list_clips, load_transcript, _speech_for

EGOLIFE_ROOT = "/home2/ycj/Datas/EgoLife"
BENCHES = {
    "egor1":     ("/home2/ycj/Datas/Ego-R1-Bench/manual-benchmark/*.json", "outputs/memory/egolife"),
    "egolifeqa": ("/home2/ycj/Datas/EgoLife_bench/EgoLifeQA/*.json",       "outputs/memory/egolife"),
}
VLM_MODEL = "/home2/ycj/Models/Qwen/Qwen3.5-27B"


# ── time / bank helpers ──────────────────────────────────────────────────────
def _hhmmssff_to_sec(s: str) -> float:
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) + int(s[6:8]) / 100.0

def abs_range(tt: dict):
    """target_time {date:'DAY3', time:'11390000' | '11390000-11393000'} -> (a,b) abs sec."""
    if not tt or "date" not in tt or "time" not in tt:
        return None
    m = re.match(r"DAY(\d+)", str(tt["date"]))
    if not m:
        return None
    base = (int(m.group(1)) - 1) * 86400.0
    t = str(tt["time"])
    if "-" in t:
        a, b = t.split("-")[:2]
        return base + _hhmmssff_to_sec(a), base + _hhmmssff_to_sec(b)
    s = base + _hhmmssff_to_sec(t)
    return s, s + 1.0


_bank_cache: dict = {}
_bank_lock = threading.Lock()
def load_bank(mem_dir: str, person: str):
    with _bank_lock:
        if person in _bank_cache:
            return _bank_cache[person]
    p = os.path.join(mem_dir, person, "L1.jsonl")
    chunks = []
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            chunks.append(r)
    with _bank_lock:
        _bank_cache[person] = (os.path.join(mem_dir, person), chunks)
    return _bank_cache[person]


_clip_cache: dict = {}
def person_clips(person: str):
    if person not in _clip_cache:
        _clip_cache[person] = list_clips(EGOLIFE_ROOT, person)  # [(abs_start, Path)]
    return _clip_cache[person]

_tr_cache: dict = {}
def person_transcript(person: str):
    if person not in _tr_cache:
        _tr_cache[person] = load_transcript(EGOLIFE_ROOT, person)
    return _tr_cache[person]


def _resize(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        s = max_side / m
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return img


def decode_raw_frames(person: str, t_start: float, t_end: float, k: int = 6, margin: float = 1.5):
    """Re-decode up to k frames from the raw clip covering [t_start,t_end]±margin."""
    clips = person_clips(person)
    if not clips:
        return []
    # last clip starting at or before the target start
    cand = [(a, p) for a, p in clips if a <= t_start + 0.5]
    if not cand:
        return []
    clip_abs, path = cand[-1]
    lo, hi = t_start - margin, t_end + margin
    times = [lo + i * (hi - lo) / max(1, k - 1) for i in range(k)]
    offs = [t - clip_abs for t in times]  # seconds into the clip
    try:
        vr = decord.VideoReader(str(path))
        fps = float(vr.get_avg_fps()) or 30.0
        n = len(vr)
        idxs = sorted({min(n - 1, max(0, int(round(o * fps)))) for o in offs if o >= -0.5})
        idxs = [i for i in idxs if 0 <= i < n]
        if not idxs:
            return []
        arr = vr.get_batch(idxs).asnumpy()  # (m,H,W,3) uint8
        return [_resize(Image.fromarray(a)) for a in arr]
    except Exception as e:
        print(f"  [decode-fail] {person} {path.name}: {e}", flush=True)
        return []


def load_stored_keyframes(bank_dir: str, gold_chunks: list, cap: int = 6):
    imgs = []
    for c in gold_chunks:
        for fp in c.get("frame_paths", []):
            p = os.path.join(bank_dir, fp)
            if os.path.exists(p):
                try:
                    imgs.append(_resize(Image.open(p).convert("RGB")))
                except Exception:
                    pass
            if len(imgs) >= cap:
                return imgs
    return imgs


# ── data loading ─────────────────────────────────────────────────────────────
def load_questions(bench: str):
    pat, mem = BENCHES[bench]
    out = []
    for f in sorted(glob.glob(pat)):
        person = Path(f).stem.replace("EgoLifeQA_", "")
        rows = json.load(open(f, encoding="utf-8"))
        if isinstance(rows, dict):
            rows = list(rows.values())
        for i, q in enumerate(rows):
            cands = [q.get(f"choice_{c}", "") for c in ("a", "b", "c", "d")]
            cands = [c for c in cands if c != ""]
            letter = str(q.get("answer", "")).strip().upper()[:1]
            if letter not in "ABCD" or ord(letter) - 65 >= len(cands):
                continue
            rng = abs_range(q.get("target_time") or {})
            if rng is None:
                continue
            out.append(dict(
                key=f"{person}#{q.get('ID', i)}", person=person, mem=mem,
                question=str(q.get("question", "")).strip(), candidates=cands,
                gold_letter=letter, rng=rng,
                qtype=str(q.get("type", "")),
                need_audio=bool(q.get("need_audio", False)),
                need_name=bool(q.get("need_name", False)),
            ))
    return out


# ── one question, all arms ───────────────────────────────────────────────────
_ARMS = ["A_caption", "B_storedkf", "C_rawframes", "D_subtitle",
         "E_densevideo", "F_denseNoCap"]

def run_one(ans: Answerer, s: dict) -> dict:
    bank_dir, chunks = load_bank(s["mem"], s["person"])
    a, b = s["rng"]
    gold = [c for c in chunks if c["t_end"] >= a and c["t_start"] <= b]
    rec = dict(key=s["key"], person=s["person"], qtype=s["qtype"],
               need_audio=s["need_audio"], need_name=s["need_name"],
               gold_letter=s["gold_letter"], n_gold=len(gold), preds={})
    if not gold:
        rec["preds"] = {arm: "" for arm in _ARMS}
        return rec
    cap_texts = [c.get("text", "") for c in gold]
    q, cands = s["question"], s["candidates"]

    def _ask(imgs=(), extra_texts=()):
        texts = list(cap_texts) + list(extra_texts)
        r = ans.answer(q, cands, texts, keyframe_images=list(imgs),
                       image_timestamps=False)
        return r.get("answer", "")

    # A: caption text only
    rec["preds"]["A_caption"] = _ask()
    # B: + stored keyframe(s)
    rec["preds"]["B_storedkf"] = _ask(imgs=load_stored_keyframes(bank_dir, gold))
    # C: + re-decoded raw frames at target_time
    raw = decode_raw_frames(s["person"], a, b)
    rec["n_raw"] = len(raw)
    rec["preds"]["C_rawframes"] = _ask(imgs=raw) if raw else ""
    # D: + subtitle/transcript at target_time
    _, speech = _speech_for(person_transcript(s["person"]), a - 5, b + 5)
    rec["has_sub"] = bool(speech.strip())
    rec["preds"]["D_subtitle"] = _ask(extra_texts=[f"Transcript near this moment:\n{speech}"]) if speech.strip() else ""
    # E: + DENSE re-decoded frames (32) — is sparse 6-frame (arm C) the ceiling, or the model?
    dense = decode_raw_frames(s["person"], a, b, k=32)
    rec["n_dense"] = len(dense)
    rec["preds"]["E_densevideo"] = _ask(imgs=dense) if dense else ""
    # F: DENSE frames ONLY, no caption — does the (lossy) caption drag the answer down?
    if dense:
        rf = ans.answer(q, cands, [], keyframe_images=list(dense), image_timestamps=False)
        rec["preds"]["F_denseNoCap"] = rf.get("answer", "")
    else:
        rec["preds"]["F_denseNoCap"] = ""
    return rec


def summarize(records: list):
    def acc(rs, arm):
        ok = [1 for r in rs if r["preds"].get(arm) == r["gold_letter"]]
        n = [1 for r in rs if r["preds"].get(arm)]
        return (sum(ok), len(n), 100.0 * sum(ok) / max(1, len(n)))
    print("\n==== GOLD-INJECT PROBE SUMMARY ====")
    print(f"questions with a gold chunk: {sum(1 for r in records if r['n_gold'])}/{len(records)}")
    print(f"{'arm':<14} {'correct/answered':>18} {'acc%':>7}")
    for arm in _ARMS:
        c, n, a = acc(records, arm)
        print(f"{arm:<14} {f'{c}/{n}':>18} {a:>7.1f}")
    # decompositions
    for field in ("need_audio", "need_name", "qtype"):
        print(f"\n-- by {field} --")
        groups = defaultdict(list)
        for r in records:
            groups[r[field]].append(r)
        for g, rs in sorted(groups.items(), key=lambda x: str(x[0])):
            row = f"  {str(g):<16} n={len(rs):<4}"
            for arm in _ARMS:
                _, _, a = acc(rs, arm)
                row += f" {arm.split('_')[0]}={a:5.1f}"
            print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="egor1", choices=list(BENCHES))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--vlm-api-url", default="http://127.0.0.1:8003")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"experiments/analysis/gold_inject/{args.bench}_goldinject.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass

    qs = load_questions(args.bench)
    if args.limit:
        qs = qs[:args.limit]
    todo = [s for s in qs if s["key"] not in done]
    print(f"[probe] bench={args.bench} total={len(qs)} done={len(done)} todo={len(todo)}", flush=True)

    vlm = VLMClient(model_path=VLM_MODEL, api_url=args.vlm_api_url, api_model="Qwen3.5-27B")
    ans = Answerer(vlm)

    lock = threading.Lock()
    fout = open(out, "a")
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, ans, s): s for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  [fail] {s['key']}: {e}", flush=True)
                continue
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  ...{n_done}/{len(todo)}", flush=True)
    fout.close()

    records = [json.loads(l) for l in open(out)]
    summarize(records)


if __name__ == "__main__":
    main()
