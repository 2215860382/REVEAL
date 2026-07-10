"""Retrieval-method probe on RECALL-GAP questions.

Answers empirically: for the questions where the sampled retrieval (q+opts / q, top-k ladder 2/4/8/16)
never found sufficient evidence, how many become SUFFICIENT under retrieval methods we never tried —
  * top-32 / top-64      : just widen k on the same q+opts query
  * decomposed sub-queries: the mainline planner's LLM atomic decomposition, retrieve each, union
Sufficiency is judged by the SAME gold-aware text judge (judge_sufficiency) so numbers are comparable
to everything else. Doc vectors are the bank's precomputed c.v_text/c.v_visual; only the query is
embedded (BGE + SigLIP via the embed server).

Out: per-question jsonl {sample_idx, methods:{name:{n_chunks, sufficient, reason}}} + a summary of how
many recall-gap questions each method rescues (and the union).
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.utils.extract_json import extract_json
from .io import read_jsonl, write_jsonl, dump_json, load_json
from .llm import LLMRouter
from .sample_chains import _evidence_items, _chunk_text
from .judge_sufficiency import JUDGE_SYSTEM, JUDGE_USER, _fmt_options, _fmt_evidence, _gold_text


class _MsgLLM:
    """Adapt LLMRouter.chat(system,user) to the planner's llm.chat(messages,...) interface."""
    def __init__(self, router):
        self.router = router

    def chat(self, messages, max_new_tokens=256, enable_thinking=False, **kw):
        sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
        usr = "\n".join(m["content"] for m in messages if m["role"] == "user")
        return self.router.chat(sys, usr, max_tokens=max_new_tokens)


def _upper_chunks(bank):
    """L2/L3/L4 summary chunks (coarser, text-only) — loaded but never used in L1-only retrieval."""
    out = []
    for name in ("l2_chunks", "l3_chunks", "l4_chunks"):
        out.extend(getattr(bank, name, None) or [])
    return out


def _rank_chunks(chunks, q_vec, vq, alpha):
    """Rank a chunk list by descending text(+visual) similarity to the query vectors."""
    if not chunks:
        return []
    doc = np.array([c.v_text for c in chunks], dtype=np.float32)
    scores = doc @ q_vec
    if vq is not None and getattr(chunks[0], "v_visual", None):
        vis = np.array([c.v_visual for c in chunks], dtype=np.float32)
        scores = alpha * scores + (1 - alpha) * (vis @ vq)
    return [chunks[i] for i in np.argsort(-scores)]


def _q_vecs(query, embedder, siglip, question, cands):
    q_vec = embedder.encode([query])[0]
    vq = None
    if siglip is not None:
        vis_query = f"A video frame showing {question} Possible answers: {' or '.join(cands)}"
        vq = siglip.encode_text([vis_query])[0]
    return q_vec, vq


def _judge(llm, sample, cmap, ids, max_tokens):
    ev_texts = [it.text for it in _evidence_items(cmap, ids)]
    if not ev_texts:
        return False, "empty"
    user = JUDGE_USER.format(
        question=sample.question, options=_fmt_options(sample.candidates),
        gold=_gold_text(sample), n=len(ev_texts), evidence=_fmt_evidence(ev_texts))
    try:
        obj = extract_json(llm.chat(JUDGE_SYSTEM, user, max_tokens=max_tokens)) or {}
    except Exception as exc:
        return False, f"judge error: {exc}"
    return bool(obj.get("sufficient")), str(obj.get("reason") or "")[:160]


def _layer_ids(bank, name, q_vec, vq, alpha):
    """Ranked chunk_ids for one layer. L1 uses text+visual; summary layers are text-only."""
    chunks = bank.chunks if name == "L1" else (getattr(bank, f"{name.lower()}_chunks", None) or [])
    ranked = _rank_chunks(chunks, q_vec, vq if name == "L1" else None, alpha)
    return [int(c.chunk_id) for c in ranked]


# staged greedy ladder: sweep L1 k1; then at the L1 anchor sweep +L2 k2; then +L3.
L1_KS = [2, 4, 8, 16, 32, 64]
L1_ANCHOR = 16            # k1 to hold fixed while sweeping upper layers
L2_KS = [1, 2, 4, 8]      # L2 has ~11 chunks
L3_KS = [1, 2]            # L3 has 1-4 chunks


def _probe_one(llm, mllm, sample, bank_getter, embedder, siglip, alpha, subq_topk, union_cap, max_tokens):
    bank, cmap = bank_getter(sample.video_id)
    if bank is None:
        return {"sample_idx": int(sample.sample_idx), "methods": {}, "error": "no bank"}
    q_opts = f"{sample.question}\nChoices:\n" + _fmt_options(sample.candidates)
    q_vec, vq = _q_vecs(q_opts, embedder, siglip, sample.question, sample.candidates)
    l1 = _layer_ids(bank, "L1", q_vec, vq, alpha)
    l2 = _layer_ids(bank, "L2", q_vec, vq, alpha)
    l3 = _layer_ids(bank, "L3", q_vec, vq, alpha)
    methods = {}

    def judge(name, ids):
        suff, reason = _judge(llm, sample, cmap, ids, max_tokens)
        methods[name] = {"n_chunks": len(ids), "sufficient": suff, "reason": reason}

    # stage 1: L1 only, sweep k1
    for k in L1_KS:
        judge(f"L1@{k}", l1[:k])
    # stage 2: hold L1@anchor, add L2 at k2
    for k2 in L2_KS:
        judge(f"L1@{L1_ANCHOR}+L2@{k2}", l1[:L1_ANCHOR] + l2[:k2])
    # stage 3: hold L1@anchor + L2@(max), add L3
    for k3 in L3_KS:
        judge(f"L1@{L1_ANCHOR}+L2@{L2_KS[-1]}+L3@{k3}", l1[:L1_ANCHOR] + l2[:L2_KS[-1]] + l3[:k3])
    return {"sample_idx": int(sample.sample_idx), "video_id": sample.video_id, "methods": methods}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-idxs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/videomme.yaml")
    ap.add_argument("--memory-dir", default="outputs/memory/videommeL")
    ap.add_argument("--embed-api-url", required=True)
    ap.add_argument("--llm-api-url", required=True)
    ap.add_argument("--llm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--subq-topk", type=int, default=8, help="top-k retrieved per sub-question")
    ap.add_argument("--union-cap", type=int, default=24, help="max chunks in the decomp union")
    ap.add_argument("--no-siglip", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)
    from experiments.tuning.veil_27b import load_samples
    samples = {s.sample_idx: s for s in load_samples(cfg)}
    gap = [int(x) for x in json.load(open(args.gap_idxs)) if int(x) in samples]
    if args.limit:
        gap = gap[:args.limit]

    from src.clients.embedder import BGEM3Embedder
    embedder = BGEM3Embedder(api_url=args.embed_api_url)
    siglip = None
    if not args.no_siglip:
        from src.clients.siglip_embedder import SigLIPEmbedder
        siglip = SigLIPEmbedder(api_url=args.embed_api_url)

    memory_dir = Path(args.memory_dir)
    bank_cache, cmap_cache, clock = {}, {}, threading.Lock()

    def bank_getter(vid):
        with clock:
            if vid in bank_cache:
                return bank_cache[vid], cmap_cache.get(vid)
        try:
            from src.build_memory.core.bank_loader import load_bank
            b = load_bank(memory_dir / vid)
            cm = {int(c.chunk_id): c for c in b.chunks}          # L1
            for name in ("l2_chunks", "l3_chunks", "l4_chunks"):  # + summary layers
                for c in (getattr(b, name, None) or []):
                    cm[int(c.chunk_id)] = c
        except Exception as exc:
            print(f"[probe] bank load failed {vid}: {exc}")
            b, cm = None, None
        with clock:
            bank_cache[vid] = b
            if cm is not None:
                cmap_cache[vid] = cm
        return b, cm

    router = LLMRouter.from_urls(args.llm_api_url, model=args.llm_api_model)
    mllm = _MsgLLM(router)

    state_path = Path(str(args.out) + ".state.json")
    done = {}
    if args.resume and state_path.is_file():
        done = load_json(state_path).get("done", {})
        print(f"[probe] resume: {len(done)} cached")
    todo = [q for q in gap if str(q) not in done]
    print(f"[probe] {len(todo)} recall-gap questions ({len(done)} cached) | {args.workers}-way")

    lock, ct = threading.Lock(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_probe_one, router, mllm, samples[q], bank_getter, embedder, siglip,
                          args.alpha, args.subq_topk, args.union_cap, args.max_tokens): q for q in todo}
        for f in as_completed(futs):
            row = f.result()
            with lock:
                done[str(row["sample_idx"])] = row
                ct += 1
                m = row.get("methods", {})
                flags = " ".join(f"{k}={'Y' if v['sufficient'] else '.'}" for k, v in m.items())
                print(f"[probe {ct}/{len(todo)}] q={row['sample_idx']} {flags}", flush=True)
                if ct % 20 == 0:
                    dump_json(state_path, {"done": done})

    rows = list(done.values())
    write_jsonl(args.out, rows)
    dump_json(state_path, {"done": done})

    # summary — staged ladder
    N = len([r for r in rows if r.get("methods")]) or 1
    order = ([f"L1@{k}" for k in L1_KS]
             + [f"L1@{L1_ANCHOR}+L2@{k}" for k in L2_KS]
             + [f"L1@{L1_ANCHOR}+L2@{L2_KS[-1]}+L3@{k}" for k in L3_KS])
    tally = {n: 0 for n in order}
    for r in rows:
        m = r.get("methods", {})
        for n in order:
            if m.get(n, {}).get("sufficient"):
                tally[n] += 1
    print(f"\n[probe] questions probed = {N}")
    print("  -- stage 1: L1 only, sweep k1 --")
    for k in L1_KS:
        n = f"L1@{k}"; print(f"    {n:28s} {tally[n]:3d} ({tally[n]/N*100:.1f}%)")
    print(f"  -- stage 2: L1@{L1_ANCHOR} + L2 sweep --")
    for k in L2_KS:
        n = f"L1@{L1_ANCHOR}+L2@{k}"; print(f"    {n:28s} {tally[n]:3d} ({tally[n]/N*100:.1f}%)")
    print(f"  -- stage 3: L1@{L1_ANCHOR}+L2@{L2_KS[-1]} + L3 --")
    for k in L3_KS:
        n = f"L1@{L1_ANCHOR}+L2@{L2_KS[-1]}+L3@{k}"; print(f"    {n:28s} {tally[n]:3d} ({tally[n]/N*100:.1f}%)")


if __name__ == "__main__":
    main()
