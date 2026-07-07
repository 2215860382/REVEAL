"""Score every pair's weak & strong chain with the holistic verifier, ONCE per unique
chain, then sweep the sufficient-threshold OFFLINE to find the value that best separates
weak from strong across all pairs.

A pair is "separated" at threshold t iff  strong_score >= t > weak_score.
Emitting scores (not just booleans) lets the threshold float during iteration, so
"rubric judges pairs correctly" is measured at the rubric's BEST threshold, not a fixed one.

Requires VEIL_VERIFIER_HOLISTIC=1 (holistic path returns 'holistic_score').

Outputs:
  --out         jsonl, one row per pair: {pair_id, sample_idx, question_type,
                weak_chain_id, strong_chain_id, weak_score, strong_score}
  stdout        offline threshold sweep + best threshold + max pair-accuracy
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.verifier import Verifier
from src.build_memory.core.bank_loader import load_bank
from experiments.core._keyframes import load_keyframes
from .io import read_jsonl, write_jsonl, dump_json, load_json
from .rubric_io import load_rubric, verify_rubric_dict


def _chain_texts(chain: dict) -> list[str]:
    return [e.get("text", "") for e in (chain.get("evidence") or [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--config", default="configs/veil.yaml")
    ap.add_argument("--llm-api-url", required=True)
    ap.add_argument("--llm-api-model", default=None)
    ap.add_argument("--workers", type=int, default=16)
    # ── multimodal alignment: feed verifier the SAME keyframes the sampler used to
    #    label chains (sample_chains._label_chain -> load_keyframes + chat_with_frames).
    #    A text-only verifier can't predict a multimodal answerer, so pair separation
    #    stalls near chance; aligning the modality is the fix. ──
    ap.add_argument("--memory-dir", default="outputs/memory/videomme_ml3_n12")
    ap.add_argument("--keyframe-dir", default=None, help="defaults to --memory-dir")
    ap.add_argument("--answer-keyframe-k", type=int, default=32)
    ap.add_argument("--no-images", action="store_true",
                    help="text-only verifier (disable multimodal alignment)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="resume from <out>.state.json")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--binary", action="store_true",
                    help="score each criterion 0/1 (no 0.5); hard yes/no per criterion")
    args = ap.parse_args()

    os.environ["VEIL_VERIFIER_HOLISTIC"] = "1"  # force holistic path
    if args.binary:
        os.environ["VEIL_VERIFIER_BINARY"] = "1"

    from src.config import load_config
    cfg = load_config(args.config)
    from experiments.tuning.veil_27b import load_samples
    samples = {s.sample_idx: s for s in load_samples(cfg)}

    chains = {c["chain_id"]: c for c in read_jsonl(args.chains)}
    pairs = list(read_jsonl(args.pairs))
    rubric = load_rubric(args.rubric)

    from src.clients.llm_client import LLMClient
    llm = LLMClient(model_path=args.llm_api_model or "Qwen3.5-27B",
                    api_url=args.llm_api_url, api_model=args.llm_api_model)
    verifier = Verifier(llm)

    # ── keyframe alignment: mirror sample_chains._label_chain exactly ──
    memory_dir = Path(args.memory_dir)
    keyframe_dir = Path(args.keyframe_dir or args.memory_dir)
    kf_cap = args.answer_keyframe_k
    use_images = not args.no_images
    _bank_cmap: dict[str, dict] = {}
    _bank_lock = threading.Lock()

    def _cmap(vid: str):
        with _bank_lock:
            if vid in _bank_cmap:
                return _bank_cmap[vid]
        try:
            b = load_bank(memory_dir / vid)
            cm = {int(c.chunk_id): c for c in b.chunks}
        except Exception as exc:
            print(f"[score] bank load failed {vid}: {exc}")
            cm = None
        with _bank_lock:
            _bank_cmap[vid] = cm
        return cm

    def _frames_for(c: dict, vid: str):
        if not use_images or not vid:
            return ()
        cm = _cmap(vid)
        if not cm:
            return ()
        ids = [int(e["chunk_id"]) for e in (c.get("evidence") or [])
               if e.get("chunk_id") is not None]
        chunks = [cm[i] for i in ids if i in cm]
        if not chunks:
            return ()
        try:
            return load_keyframes(chunks, keyframe_dir, vid, cap=kf_cap)
        except Exception as exc:
            print(f"[score] keyframe load failed {vid}: {exc}")
            return ()

    # ── unique (chain_id) work-list; a chain always belongs to one sample ──
    chain_to_sample: dict[str, int] = {}
    for p in pairs:
        si = int(p["sample_idx"])
        chain_to_sample.setdefault(p["weaker_chain_id"], si)
        chain_to_sample.setdefault(p["stronger_chain_id"], si)

    def score_chain(cid: str):
        si = chain_to_sample[cid]
        s = samples.get(si)
        c = chains.get(cid)
        if s is None or c is None:
            return cid, None, None
        rd = verify_rubric_dict(rubric, getattr(s, "question_type", ""))
        frames = _frames_for(c, getattr(s, "video_id", "") or c.get("video_id", ""))
        try:
            v = verifier.verify(s.question, s.candidates, _chain_texts(c), rd,
                                keyframe_images=frames)
            return cid, float(v.get("holistic_score", 0.0)), (v.get("criteria_scores") or {})
        except Exception as exc:
            print(f"[score] chain {cid} failed: {exc}")
            return cid, None, None

    cid_score: dict[str, float] = {}
    cid_crit: dict[str, dict] = {}          # per-chain {criterion: score} for disc measurement
    state_path = Path(str(args.out) + ".state.json")
    if args.resume and state_path.is_file():
        st = load_json(state_path)
        cid_score = {k: float(v) for k, v in (st.get("cid_score") or {}).items()}
        cid_crit = {k: (v or {}) for k, v in (st.get("cid_crit") or {}).items()}
        print(f"[score] resume: {len(cid_score)} chains cached")
    all_chains = list(chain_to_sample.keys())
    todo = [c for c in all_chains if c not in cid_score]
    print(f"[score] {len(todo)} to score ({len(cid_score)} cached) of {len(all_chains)} unique chains "
          f"over {len(pairs)} pairs | mode={'MULTIMODAL(keyframes)' if use_images else 'text-only'} "
          f"kf_cap={kf_cap} | {args.workers}-way", flush=True)
    import time as _time
    lock = threading.Lock()
    t0, ct = _time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(score_chain, c) for c in todo]):
            cid, sc, crit = f.result()
            with lock:
                if sc is not None:
                    cid_score[cid] = sc
                    cid_crit[cid] = crit or {}
                ct += 1
                if ct % args.progress_every == 0 or ct == len(todo):
                    el = _time.time() - t0
                    rate = ct / el if el else 0
                    eta = (len(todo) - ct) / rate if rate else 0
                    print(f"[score {ct}/{len(todo)}] {rate:.2f} chain/s  "
                          f"elapsed {el/60:.1f}m  ETA {eta/60:.1f}m", flush=True)
                    dump_json(state_path, {"cid_score": cid_score, "cid_crit": cid_crit})
    dump_json(state_path, {"cid_score": cid_score, "cid_crit": cid_crit})

    rows = []
    crit_rows = []                          # sidecar: per-pair per-criterion weak/strong scores
    for p in pairs:
        ws = cid_score.get(p["weaker_chain_id"])
        ss = cid_score.get(p["stronger_chain_id"])
        if ws is None or ss is None:
            continue
        rows.append({
            "pair_id": p["pair_id"],
            "sample_idx": int(p["sample_idx"]),
            "question_type": p.get("question_type", ""),
            "weaker_chain_id": p["weaker_chain_id"],
            "stronger_chain_id": p["stronger_chain_id"],
            "weak_score": round(ws, 4),
            "strong_score": round(ss, 4),
        })
        crit_rows.append({
            "pair_id": p["pair_id"],
            "question_type": p.get("question_type", ""),
            "weak_crit": cid_crit.get(p["weaker_chain_id"], {}),
            "strong_crit": cid_crit.get(p["stronger_chain_id"], {}),
        })
    write_jsonl(args.out, rows)
    # sidecar for measured discrimination (disc = mean_strong - mean_weak per criterion)
    crit_out = args.out.rsplit(".jsonl", 1)[0] + ".crit.jsonl"
    write_jsonl(crit_out, crit_rows)
    print(f"[score] per-criterion sidecar -> {crit_out}")

    # ── offline threshold sweep ──
    n = len(rows)
    best_t, best_acc = 0.75, -1.0
    print(f"\n[sweep] {n} scored pairs")
    grid = [round(0.15 + 0.05 * i, 2) for i in range(17)]  # 0.15 .. 0.95
    for t in grid:
        sep = sum(1 for r in rows if r["strong_score"] >= t > r["weak_score"])
        acc = sep / n if n else 0.0
        mark = ""
        if acc > best_acc:
            best_acc, best_t = acc, t
        print(f"  thr={t:.2f}  pair_acc={acc*100:5.1f}%  ({sep}/{n})")
    # margin diagnostics at best_t
    inv = sum(1 for r in rows if r["strong_score"] < r["weak_score"])
    print(f"\n[best] threshold={best_t:.2f}  pair_acc={best_acc*100:.1f}%")
    print(f"[diag] pairs where strong<weak (unfixable by threshold): {inv}/{n} = {inv/n*100:.1f}%")
    print(f"BEST_THR={best_t}")
    print(f"BEST_PAIR_ACC={best_acc}")


if __name__ == "__main__":
    main()
