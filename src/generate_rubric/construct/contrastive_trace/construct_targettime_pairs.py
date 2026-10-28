"""Target-time structural rubric pairs (EgoLife family).

Replaces the answerer-vote sufficiency label (select_gold_mislead's GOLD, which
depends on the deployed answerer answering correctly — the gold-anchored label
noise ceiling) with a STRUCTURAL truth derived from each question's target_time:

  POSITIVE (sufficient) = the target_time moment itself: re-decode raw frames at
      target_time and re-caption them fresh with the VLM (option-aware but
      answer-blind). Guaranteed sufficient — it literally IS the evidence moment,
      no answerer, no retrieval-recall limit.
  NEGATIVE (insufficient) = 2 "mislead" chains (dense topic-similar chunks that do
      NOT cover target_time — right topic, wrong moment) + 1 "irrelevant" chain
      (chunks far in time from target). Both past-only (query_time cutoff).

Emits <out>_chains.jsonl / <out>_pairs.jsonl in the EXACT schema select_gold_mislead
produces, so refine_rubric.py consumes them unchanged (it reads only
chain["evidence"][*].text — verified). 3 pairs/question.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# bypass any local HTTP proxy for the internal vLLM/embed endpoints (LLMClient/VLMClient
# respect *_PROXY and would 502 otherwise).
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
for _k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_k, None)

import numpy as np
from PIL import Image
import cv2  # raw-frame decode (base env has cv2, not decord)

from src.build_memory.core.bank_loader import load_bank
from src.dataloader.egolife import list_clips
from .io import write_jsonl, dump_json, load_json
from .schema import EvidenceChain, EvidenceItem, ChainPair
from .sample_chains import _evidence_items, _format_options
from .retrieval_probe import _q_vecs, _rank_chunks


# ── time / frame helpers (copied from experiments/analysis/gold_inject/probe.py to
#    avoid that script's module-level os.chdir + heavy imports) ──────────────────
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


def _resize(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        s = max_side / m
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return img


_clip_cache: dict = {}
_clip_lock = threading.Lock()


def _person_clips(video_dir: str, person: str):
    key = (video_dir, person)
    with _clip_lock:
        if key not in _clip_cache:
            _clip_cache[key] = list_clips(video_dir, person)  # [(abs_start, Path)]
    return _clip_cache[key]


def decode_raw_frames(video_dir, person, t_start, t_end, k=8, margin=1.5):
    """Re-decode up to k frames (via cv2) from the raw clip covering [t_start,t_end]±margin."""
    clips = _person_clips(video_dir, person)
    if not clips:
        return []
    cand = [(a, p) for a, p in clips if a <= t_start + 0.5]
    if not cand:
        return []
    clip_abs, path = cand[-1]
    lo, hi = t_start - margin, t_end + margin
    times = [lo + i * (hi - lo) / max(1, k - 1) for i in range(k)]
    offs = [t - clip_abs for t in times]  # seconds into the clip
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n <= 0:
            return []
        idxs = sorted({min(n - 1, max(0, int(round(o * fps)))) for o in offs if o >= -0.5})
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                frames.append(_resize(Image.fromarray(fr)))
        return frames
    except Exception as e:
        print(f"  [decode-fail] {person} {getattr(path, 'name', path)}: {e}", flush=True)
        return []
    finally:
        if cap is not None:
            cap.release()


# ── gold_window (reimplemented from oracle_egolife.py with an explicit w; the
#    original couples to a module-level env global and crashes on import) ─────────
def gold_window(bank, t_abs, cutoff, w):
    """Chunks covering t_abs, expanded ±w, restricted to end<=cutoff (past-only)."""
    chunks = sorted(bank.chunks, key=lambda c: c.start_time)
    if cutoff is not None:
        chunks = [c for c in chunks if c.end_time <= cutoff + 1]
    if not chunks:
        return []
    cover = None
    for i, c in enumerate(chunks):
        if c.start_time <= t_abs <= c.end_time:
            cover = i
            break
    if cover is None:
        cover = min(range(len(chunks)),
                    key=lambda i: abs((chunks[i].start_time + chunks[i].end_time) / 2 - t_abs))
    lo = max(0, cover - w)
    hi = min(len(chunks), cover + w + 1)
    return chunks[lo:hi]


# ── caption prompt: QUESTION+OPTION-GUIDED, but ANSWER-BLIND ────────────────────
# The question and options are provided so the caption REPORTS the exact facts needed
# to adjudicate them (who/what/color/count/order/spatial). It stays answer-blind: it
# never names which option is correct — it only states the observed facts.
_CAPTION_PROMPT = (
    "These frames are ONE specific moment from a long first-person (egocentric) video — "
    "the exact moment that the question below is about.\n\n"
    "Question: {question}\n"
    "Options:\n{options}\n\n"
    "Write a factual description of what happens in these frames. You MUST report the concrete "
    "visual facts that are RELEVANT to this question and that let someone decide among the options "
    "— e.g. who performs the action, the objects involved and their color/count/on-screen text, the "
    "order of events, and spatial relations. Ground it in what is actually visible.\n"
    "Stay answer-blind: do NOT state or hint which option is correct and do NOT declare any option "
    "as the answer — just report the observed facts (even if they happen to match one option)."
)


def _positive(cfg, s, bank, cmap, args):
    """Return (list[EvidenceItem] | None, gw_ids: list[int]).

    Positive (sufficient) = the ACTUAL bank chunk(s) covering target_time, taken by
    timestamp via the same _evidence_items path the negatives use — NOT a fresh narrow
    recaption. A recaption of the ~1s target_time window dropped the surrounding context
    (e.g. 'walks up the staircase to the upper floor to get it') that IS the answer for
    before/after/origin questions, so those recaptioned positives did not contain the
    answer yet were labelled sufficient=True. The bank chunk is a full ~30s window and
    already carries that context, so it is a genuinely-sufficient positive."""
    rng = abs_range(s.target_time)
    if rng is None or bank is None or cmap is None:
        return None, []
    a, b = rng
    t_abs = (a + b) / 2.0
    cutoff = _cutoff(s)
    gw = gold_window(bank, t_abs, cutoff, args.window_w)
    if not gw:
        return None, []
    gw_ids = [int(c.chunk_id) for c in gw]
    items = _evidence_items(cmap, gw_ids)
    if not items:
        return None, gw_ids
    return items, gw_ids


def _cutoff(s):
    from experiments.reveal_27b import _egolife_query_cutoff
    return _egolife_query_cutoff(s)


def _negatives(s, bank, cmap, gw_ids, t_abs, embedder, siglip, args, rng):
    """Return (mislead_chains, irrelevant_items).
    mislead_chains = up to 2 disjoint dense-topk-minus-gw subsets (topic-similar, wrong moment).
    irrelevant_items = random past-only chunks far in time from target."""
    cutoff = _cutoff(s)
    gw = set(int(x) for x in gw_ids)
    past = [c for c in bank.chunks if (cutoff is None or c.end_time <= cutoff + 1)]

    # dense topic ranking over (question + options)
    q = f"{s.question}\nChoices:\n" + _format_options(s.candidates)
    q_vec, vq = _q_vecs(q, embedder, siglip, s.question, s.candidates)
    ranked = _rank_chunks(past, q_vec, vq, args.alpha)
    ranked_ng = [int(c.chunk_id) for c in ranked if int(c.chunk_id) not in gw]

    mislead_chains = []
    mislead_ids = set()
    for j in range(2):  # 2 mislead chains: top mislead_k, then next mislead_k
        seg = ranked_ng[j * args.mislead_k:(j + 1) * args.mislead_k]
        items = _evidence_items(cmap, seg)
        if items:
            mislead_chains.append(items)
            mislead_ids.update(seg)

    # irrelevant: past-only, far in time, not in gw, not already used as mislead
    far = [c for c in past
           if int(c.chunk_id) not in gw
           and int(c.chunk_id) not in mislead_ids
           and abs((c.start_time + c.end_time) / 2.0 - t_abs) > args.far_sec]
    rng.shuffle(far)
    irr = _evidence_items(cmap, [int(c.chunk_id) for c in far[:args.irrelevant_k]])
    return mislead_chains, irr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--memory-dir", required=True,
                    help="offline memory bank dir for the target dataset")
    ap.add_argument("--embed-api-url", required=True, help="BGE + SigLIP")
    ap.add_argument("--vlm-api-url", default=None, help="27B (caption only)")
    ap.add_argument("--vlm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--frames-k", type=int, default=8)
    ap.add_argument("--frame-margin", type=float, default=1.5)
    ap.add_argument("--caption-max-tokens", type=int, default=256)
    ap.add_argument("--caption-max-pixels", type=int, default=None)
    ap.add_argument("--window-w", type=int, default=2)
    ap.add_argument("--mislead-k", type=int, default=4)
    ap.add_argument("--irrelevant-k", type=int, default=4)
    ap.add_argument("--far-sec", type=float, default=600.0)
    ap.add_argument("--tag", default=None, help="id prefix (default: benchmark name)")
    ap.add_argument("--idxs", default=None, help="json list of sample_idx (smoke)")
    ap.add_argument("--sample-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from src.config import load_config
    from experiments.reveal_27b import load_samples
    cfg = load_config(args.config)
    tag = args.tag or cfg["benchmark"]["name"]
    samples = {int(s.sample_idx): s for s in load_samples(cfg)}
    if args.idxs and Path(args.idxs).is_file():
        idxs = [int(x) for x in json.load(open(args.idxs)) if int(x) in samples]
    else:
        idxs = sorted(samples)
    if args.sample_limit:
        idxs = idxs[:args.sample_limit]

    from src.clients.embedder import BGEM3Embedder
    from src.clients.siglip_embedder import SigLIPEmbedder
    embedder = BGEM3Embedder(api_url=args.embed_api_url)
    siglip = SigLIPEmbedder(api_url=args.embed_api_url)
    # Positives now come straight from the timestamped bank chunks (no recaption), so no
    # VLM is needed. --vlm-api-url is kept for backward-compat but ignored.

    memory_dir = Path(args.memory_dir)
    bank_cache, cmap_cache, clock = {}, {}, threading.Lock()

    def bank_getter(vid):
        with clock:
            if vid in bank_cache:
                return bank_cache[vid], cmap_cache.get(vid)
        try:
            b = load_bank(memory_dir / vid)
            cm = {int(c.chunk_id): c for c in b.chunks}
            for nm in ("l2_chunks", "l3_chunks", "l4_chunks"):
                for c in (getattr(b, nm, None) or []):
                    cm[int(c.chunk_id)] = c
        except Exception as exc:
            print(f"[tt] bank load failed {vid}: {exc}", flush=True)
            b, cm = None, None
        with clock:
            bank_cache[vid] = b
            if cm is not None:
                cmap_cache[vid] = cm
        return b, cm

    def _work(q):
        s = samples[q]
        bank, cmap = bank_getter(s.video_id)
        if bank is None:
            return {"sample_idx": q, "video_id": s.video_id, "skip": "no-bank",
                    "pos_items": None, "mislead": [], "irrelevant": []}
        pos_items, gw_ids = _positive(cfg, s, bank, cmap, args)
        if not pos_items:
            return {"sample_idx": q, "video_id": s.video_id, "skip": "no-positive",
                    "pos_items": None, "mislead": [], "irrelevant": []}
        rng_a = abs_range(s.target_time)
        t_abs = (rng_a[0] + rng_a[1]) / 2.0
        prng = __import__("random").Random(args.seed + q)
        mislead_chains, irr = _negatives(s, bank, cmap, gw_ids, t_abs,
                                         embedder, siglip, args, prng)
        return {"sample_idx": q, "video_id": s.video_id,
                "question_type": getattr(s, "question_type", "") or "",
                "pos_items": [it.to_dict() for it in pos_items],
                "cover_id": pos_items[len(pos_items) // 2].chunk_id,
                "gw_ids": gw_ids,
                "mislead": [[it.to_dict() for it in ch] for ch in mislead_chains],
                "irrelevant": [it.to_dict() for it in irr],
                "n_mislead": len(mislead_chains), "n_irrelevant": len(irr)}

    def work(q):
        try:
            return _work(q)
        except Exception as exc:
            print(f"[tt] q={q} failed: {exc}", flush=True)
            s = samples.get(q)
            return {"sample_idx": q, "video_id": getattr(s, "video_id", ""),
                    "skip": f"error: {str(exc)[:160]}", "pos_items": None,
                    "mislead": [], "irrelevant": []}

    state_path = Path(str(args.out) + ".state.json")
    done: dict[str, dict] = {}
    if args.resume and state_path.is_file():
        done = load_json(state_path).get("done", {})
    todo = [q for q in idxs if str(q) not in done]
    print(f"[tt] {len(todo)} questions ({len(done)} cached) | tag={tag} | {args.workers}-way", flush=True)

    lock, ct = threading.Lock(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, q): q for q in todo}
        for f in as_completed(futs):
            row = f.result()
            with lock:
                done[str(row["sample_idx"])] = row
                ct += 1
                sk = row.get("skip")
                print(f"[tt {ct}/{len(todo)}] q={row['sample_idx']} "
                      + (f"SKIP({sk})" if sk else
                         f"pos=1 misl={row.get('n_mislead',0)} irr={row.get('n_irrelevant',0)}"),
                      flush=True)
                if ct % 20 == 0:
                    dump_json(state_path, {"done": done})

    rows = list(done.values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(Path(args.out), rows)
    dump_json(state_path, {"done": done})

    # ── build chains + pairs ──
    chains, pairs = [], []
    for r in rows:
        if r.get("skip") or not r.get("pos_items"):
            continue
        q = int(r["sample_idx"])
        s = samples.get(q)
        if s is None:
            continue
        qt = r.get("question_type", "") or getattr(s, "question_type", "") or ""
        vid = r["video_id"]
        pos_id = f"{tag}#{q}:pos"
        pos_ev = [EvidenceItem(**d) for d in r["pos_items"]]
        chains.append(EvidenceChain(
            sample_idx=q, video_id=vid, question_type=qt, chain_id=pos_id,
            quality="sufficient", source="targettime_bankchunk",
            evidence=pos_ev,
            notes="bank chunk(s) covering target_time", sufficient=True,
            sufficient_votes="", topk=None, config="targettime",
            n_chunks=len(pos_ev)).to_dict())

        negs = []
        for j, ch in enumerate(r.get("mislead", [])):
            if ch:
                negs.append((f"mislead{j+1}", ch))
        if r.get("irrelevant"):
            negs.append(("irrelevant", r["irrelevant"]))

        for kind, items in negs:
            neg_id = f"{tag}#{q}:{kind}"
            chains.append(EvidenceChain(
                sample_idx=q, video_id=vid, question_type=qt, chain_id=neg_id,
                quality="insufficient", source=f"neg_{kind}",
                evidence=[EvidenceItem(**d) for d in items],
                notes=kind, sufficient=False, sufficient_votes="",
                topk=None, config=kind, n_chunks=len(items)).to_dict())
            pair = ChainPair(
                sample_idx=q, video_id=vid, question_type=qt,
                pair_id=f"{tag}#{q}:{kind}->pos",
                weaker_chain_id=neg_id, stronger_chain_id=pos_id,
                pair_type=f"{kind}_vs_targettime").to_dict()
            pair["question"] = s.question
            pair["options"] = list(s.candidates)
            pair["answer"] = s.answer
            pairs.append(pair)

    cpath = str(args.out).rsplit(".jsonl", 1)[0] + "_chains.jsonl"
    ppath = str(args.out).rsplit(".jsonl", 1)[0] + "_pairs.jsonl"
    write_jsonl(cpath, chains)
    write_jsonl(ppath, pairs)

    N = len(rows) or 1
    kept = sum(1 for r in rows if r.get("pos_items") and not r.get("skip"))
    print(f"\n[tt] N={len(rows)}  positives_built={kept}/{N} ({kept/N*100:.1f}%)")
    print(f"  -> {len(chains)} chains, {len(pairs)} pairs (target: ~3/question)")
    print(f"  -> {cpath}\n  -> {ppath}")


if __name__ == "__main__":
    main()
