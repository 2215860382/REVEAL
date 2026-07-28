"""Sample outcome-anchored evidence chains for rubric construction.

For every question we build a diverse pool of candidate evidence chains from three
sources, then LABEL each chain by whether the deployed answerer, given ONLY that
chain, answers correctly (majority over N samples) — the gold answer is the anchor.

Sources
  1. query retrieval : rank chunks for (question, options) under several configs
                       (query variant × modality weight) and cut nested top-k prefixes.
  2. oracle          : chains supplied via --oracle-chains (chunk_ids picked by a
                       strong reasoner / Claude offline), a clean positive source. (kept, inert
                       unless --oracle-chains is passed)

Output: one EvidenceChain row per candidate, with `sufficient` / `sufficient_votes`.
Replaces the multi-round-trace `chains_from_result.py`.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.build_memory.core.bank_loader import load_bank
from src.config import load_config
from .io import read_jsonl, write_jsonl
from .schema import EvidenceChain, EvidenceItem
from experiments.core._keyframes import load_keyframes
from experiments.reveal_27b import load_samples


def _multilayer_chain(bank, question, candidates, embedder, siglip, alpha):
    """The full deployable text config: L1@16 + L2@16 + L3all (superset of any tested bestcfg).
    This is the true SUFFICIENT positive for the 726 base-solved questions."""
    from .retrieval_probe import _q_vecs, _layer_ids  # lazy: retrieval_probe imports from us
    q = f"{question}\nChoices:\n" + _format_options(candidates)
    q_vec, vq = _q_vecs(q, embedder, siglip, question, candidates)
    l1 = _layer_ids(bank, "L1", q_vec, vq, alpha)[:16]
    l2 = _layer_ids(bank, "L2", q_vec, vq, alpha)[:16]     # L2@16 (L2 has ~11, so usually all)
    l3 = _layer_ids(bank, "L3", q_vec, vq, alpha)          # L3all (1-4 chunks)
    return [int(x) for x in (l1 + l2 + l3)]


# ── evidence formatting (mirrors answerer._PROMPT) ──────────────────────────────

_PROMPT = (
    "You are answering a multiple-choice question about a long video.\n"
    "Below are {n} relevant video segments retrieved from the video.\n\n"
    "{evidence}\n\n"
    "Question: {question}\n"
    "Choices: {choices}\n"
    "Answer with a single letter (A/B/C/D) only, no explanation:"
)


def _chunk_text(c) -> str:
    text = c.memory_text or c.visual_caption or c.asr or ""
    if c.asr:
        text = f"{text}\nASR: {c.asr}"
    return text


def _evidence_items(cmap: dict, ids: list[int]) -> list[EvidenceItem]:
    out, seen = [], set()
    for cid in ids:
        cid = int(cid)
        if cid in seen:
            continue
        seen.add(cid)
        c = cmap.get(cid)
        if c is None:
            continue
        out.append(EvidenceItem(
            chunk_id=cid,
            start_time=getattr(c, "start_time", None),
            end_time=getattr(c, "end_time", None),
            text=_chunk_text(c),
        ))
    return out


def _format_options(candidates: list[str]) -> str:
    return " ".join(f"({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))


def _format_evidence(texts: list[str]) -> str:
    if not texts:
        return "(no evidence)"
    return "\n".join(f"--- Segment {i+1} ---\n{t}" for i, t in enumerate(texts))


# ── retrieval ranking (mirrors coarse_rag.run_coarse_rag scoring) ───────────────

def _rank(bank, query: str, embedder, siglip, text_alpha: float,
          candidates: list[str], question: str) -> list[int]:
    """Return chunk-array indices ranked by descending similarity."""
    if not bank.chunks:
        return []
    q_vec = embedder.encode([query])[0]
    if bank.chunks[0].v_text:
        doc_vecs = np.array([c.v_text for c in bank.chunks], dtype=np.float32)
    else:  # fall back to on-the-fly text embedding
        doc_vecs = embedder.encode([_chunk_text(c) for c in bank.chunks])
    scores = doc_vecs @ q_vec

    if siglip is not None and bank.chunks[0].v_visual:
        vis_query = f"A video frame showing {question} Possible answers: {' or '.join(candidates)}"
        vq = siglip.encode_text([vis_query])[0]
        vis = np.array([c.v_visual for c in bank.chunks], dtype=np.float32)
        scores = text_alpha * scores + (1 - text_alpha) * (vis @ vq)

    return list(np.argsort(-scores))


def _queries(question: str, candidates: list[str], variants: list[str]) -> list[tuple[str, str]]:
    """(tag, query_text) for each requested query variant."""
    opts = _format_options(candidates)
    out = []
    for v in variants:
        if v == "q+opts":
            out.append((v, f"{question}\nChoices:\n{opts}"))
        elif v == "q":
            out.append((v, question))
        elif v == "per-opt":  # one blended query naming every option
            out.append((v, f"{question} " + " ".join(candidates)))
    return out


# ── answering / labelling ───────────────────────────────────────────────────────

def _answer_once(vlm, question, candidates, ev_texts, frames, temperature) -> str:
    prompt = _PROMPT.format(
        n=len(ev_texts), evidence=_format_evidence(ev_texts),
        question=question, choices=_format_options(candidates),
    )
    raw = vlm.chat_with_frames(
        frames, prompt, max_new_tokens=16, temperature=temperature, enable_thinking=False,
    )
    m = re.search(r"\b([A-D])\b", raw or "")
    return m.group(1) if m else ""


def _label_chain(vlm, question, candidates, gold_text, cmap, ids,
                 keyframe_dir, video_id, kf_cap, n_samples, temperature) -> tuple[bool, str]:
    """Answer with ONLY this chain n_samples times; majority-correct → sufficient."""
    items = _evidence_items(cmap, ids)
    ev_texts = [it.text for it in items]
    chunks = [cmap[it.chunk_id] for it in items if it.chunk_id in cmap]
    frames = load_keyframes(chunks, keyframe_dir, video_id, cap=kf_cap) if (chunks and keyframe_dir) else []
    votes = Counter()
    for _ in range(max(1, n_samples)):
        letter = _answer_once(vlm, question, candidates, ev_texts, frames, temperature)
        if letter:
            idx = ord(letter) - ord("A")
            pred = candidates[idx] if 0 <= idx < len(candidates) else ""
            votes["correct" if pred == gold_text else "wrong"] += 1
    correct = votes["correct"]
    total = sum(votes.values()) or 1
    return (correct * 2 > total), f"{correct}/{total}"


# ── candidate-pool construction ─────────────────────────────────────────────────

def _query_chains(bank, cmap, question, candidates, embedder, siglip,
                  variants, alphas, ladder) -> list[tuple[str, int, list[int]]]:
    """Yield (config_tag, topk, chunk_ids) for every query config × top-k prefix."""
    out = []
    max_k = max(ladder)
    for qtag, qtext in _queries(question, candidates, variants):
        for alpha in alphas:
            order = _rank(bank, qtext, embedder, siglip, alpha, candidates, question)
            if not order:
                continue
            ranked_ids = [bank.chunks[i].chunk_id for i in order[:max_k]]
            for k in ladder:
                if k <= 0 or k > len(ranked_ids):
                    k_eff = len(ranked_ids)
                else:
                    k_eff = k
                out.append((f"query:{qtag}@a={alpha}", k_eff, ranked_ids[:k_eff]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/videomme.yaml")
    ap.add_argument("--memory-dir", default="outputs/memory/videomme_multiframe")
    ap.add_argument("--keyframe-dir", default=None, help="defaults to --memory-dir")
    ap.add_argument("--embed-api-url", required=True, help="BGE + SigLIP embedding service")
    ap.add_argument("--vlm-api-url", required=True, help="deployed multimodal answerer (27B)")
    ap.add_argument("--vlm-api-model", default=None)
    ap.add_argument("--query-variants", default="q+opts,q", help="comma of: q+opts,q,per-opt")
    ap.add_argument("--alphas", default="0.6", help="comma of text_alpha values, e.g. 0.8,0.6,0.3")
    ap.add_argument("--budget-ladder", default="2,4,8,16")
    ap.add_argument("--oracle-chains", default=None, help="jsonl: sample_idx + chunk_ids (+reason); "
                    "kept for optional positive injection, inert unless passed")
    ap.add_argument("--no-siglip", action="store_true")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--answer-keyframe-k", type=int, default=32)
    ap.add_argument("--sample-limit", type=int, default=None, help="only first N questions (debug)")
    ap.add_argument("--only-idxs", default=None, help="json list of sample_idx to restrict to")
    ap.add_argument("--resume", action="store_true", help="keep rows already in --out, skip those idxs")
    ap.add_argument("--no-keyframes", action="store_true",
                    help="TEXT-ONLY labelling — no keyframes to the answerer (matches the text-anchored "
                         "79%% mainline + text-only oracle/judge sufficiency). Recommended.")
    ap.add_argument("--workers", type=int, default=8, help="parallel questions (answers round-robin across --vlm-api-url)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="outputs/rubric/chains.jsonl")
    args = ap.parse_args()

    variants = [v.strip() for v in args.query_variants.split(",") if v.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    ladder = [int(x) for x in args.budget_ladder.split(",") if x.strip()]
    keyframe_dir = None if args.no_keyframes else Path(args.keyframe_dir or args.memory_dir)

    cfg = load_config(args.config)
    samples = load_samples(cfg)
    if args.only_idxs and Path(args.only_idxs).is_file():
        keep = set(int(x) for x in __import__("json").load(open(args.only_idxs)))
        samples = [s for s in samples if int(s.sample_idx) in keep]
        print(f"[filter] --only-idxs → {len(samples)} questions")
    if args.sample_limit:
        samples = samples[: args.sample_limit]

    oracle_by_idx: dict[int, list[int]] = {}
    if args.oracle_chains and Path(args.oracle_chains).is_file():
        for row in read_jsonl(args.oracle_chains):
            oracle_by_idx[int(row["sample_idx"])] = [int(x) for x in (row.get("chunk_ids") or [])]

    # Models (all via API — no local GPU load here).
    from src.clients.embedder import BGEM3Embedder
    embedder = BGEM3Embedder(api_url=args.embed_api_url)
    siglip = None
    if not args.no_siglip:
        from src.clients.siglip_embedder import SigLIPEmbedder
        siglip = SigLIPEmbedder(api_url=args.embed_api_url)
    from src.clients.vlm_client import VLMClient
    vlm = VLMClient(model_path=args.vlm_api_model or "Qwen3.5-27B",
                    api_url=args.vlm_api_url, api_model=args.vlm_api_model)

    memory_dir = Path(args.memory_dir)
    bank_cache: dict[str, object] = {}
    cmap_cache: dict[str, dict] = {}
    cache_lock = __import__("threading").Lock()

    def _get_bank(vid):
        with cache_lock:
            if vid in bank_cache:
                return bank_cache[vid], cmap_cache.get(vid)
        try:
            b = load_bank(memory_dir / vid)
            cm = {int(c.chunk_id): c for c in b.chunks}
        except Exception as exc:
            print(f"[skip] bank load failed {vid}: {exc}")
            b, cm = None, None
        with cache_lock:
            bank_cache[vid] = b
            if cm is not None:
                cmap_cache[vid] = cm
        return b, cm

    def process(s) -> list[dict]:
        bank, cmap = _get_bank(s.video_id)
        if bank is None:
            return []
        vid = s.video_id
        pool: list[tuple[str, str, int | None, list[int]]] = []
        for cfg_tag, k, ids in _query_chains(
            bank, cmap, s.question, s.candidates, embedder, siglip, variants, alphas, ladder):
            pool.append(("query", cfg_tag, k, ids))
        # full deployable text config L1@16+L2@16+L3all — the SUFFICIENT positive for base-solved qs
        try:
            ml_ids = _multilayer_chain(bank, s.question, s.candidates, embedder, siglip, alphas[0])
            if ml_ids:
                pool.append(("multilayer", "ml:L1@16+L2@16+L3all", None, ml_ids))
        except Exception as exc:
            print(f"[warn] multilayer chain failed {s.sample_idx}: {exc}")
        if s.sample_idx in oracle_by_idx:
            pool.append(("oracle", "oracle", None, oracle_by_idx[s.sample_idx]))

        out, seen_sets = [], set()
        for src, cfg_tag, k, ids in pool:
            key = (src, tuple(sorted(set(int(i) for i in ids))))
            if not ids or key in seen_sets:
                continue
            seen_sets.add(key)
            suff, votes = _label_chain(
                vlm, s.question, s.candidates, s.answer, cmap, ids,
                keyframe_dir, vid, args.answer_keyframe_k, args.n_samples, args.temperature)
            n = len(set(int(i) for i in ids))
            out.append(EvidenceChain(
                sample_idx=s.sample_idx, video_id=vid,
                question_type=getattr(s, "question_type", ""),
                chain_id=f"{s.sample_idx}:{cfg_tag}:k{k if k is not None else n}",
                quality="sufficient" if suff else "insufficient",
                source=cfg_tag,
                evidence=_evidence_items(cmap, ids),
                notes=f"source={src}",
                sufficient=suff, sufficient_votes=votes,
                topk=k, config=cfg_tag, n_chunks=n,
            ).to_dict())
        print(f"[{s.sample_idx}] {vid} chains={len(out)} "
              f"suff={sum(1 for r in out if r['sufficient'])}", flush=True)
        return out

    # resume: per-question rows already computed are kept; done idxs are skipped.
    import threading
    out_path = Path(args.out)
    done_idxs: set[int] = set()
    rows: list[dict] = []
    if args.resume and out_path.is_file():
        for r in read_jsonl(out_path):
            rows.append(r)
            done_idxs.add(int(r["sample_idx"]))
        samples = [s for s in samples if int(s.sample_idx) not in done_idxs]
        print(f"[resume] {len(done_idxs)} questions done, {len(samples)} remaining")

    wlock = threading.Lock()
    def _flush():
        write_jsonl(out_path, rows)

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            n = 0
            for f in as_completed([ex.submit(process, s) for s in samples]):
                res = f.result()
                with wlock:
                    rows.extend(res)
                    n += 1
                    if n % 10 == 0:
                        _flush()
    else:
        for i, s in enumerate(samples):
            rows.extend(process(s))
            if i % 10 == 0:
                _flush()

    _flush()
    print(f"wrote {len(rows)} chains to {args.out}")


if __name__ == "__main__":
    main()
