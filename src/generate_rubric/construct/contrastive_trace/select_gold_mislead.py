"""Gold + misleading contrastive pairs (v6 construction).

Per question (construction-time; the model IS shown the gold answer):
  pool     = _multilayer_chain  (query retrieval: L1@16 + L2@16 + L3all, ~34 chunks)
  GOLD     = the minimal decisive subset that FORCES the correct answer
             (REDUCE -> CONFIRM -> answerer-driven growth; guarantees the closed-book answerer is
             correct on it = condition A).
  MISLEAD  = the model picks the most MISLEADING subset from pool-minus-gold: segments that look
             relevant but lack the decisive fact / point a careful reader toward a WRONG option.
             Then closed-book answer-tested: a GOOD distractor makes the answerer WRONG (condition C).

Three conditions, temp=0 closed-book (mirrors control_filter):
  A  gold evidence   -> should be CORRECT   (guaranteed by the gold extraction)
  B  no evidence     -> should be WRONG      (else the question is answerable from priors)
  C  mislead evidence-> should be WRONG      (else the distractor isn't actually misleading)
GENUINE = A and (not B) and (not C).  Only genuine questions become pairs.

Pair: positive = gold chain (sufficient) x negative = mislead chain (insufficient).

Out:
  <out>.jsonl          per-question diagnostics (gold/mislead ids, A/B/C, genuine, votes, stage)
  <out>_chains.jsonl   EvidenceChain rows (1 gold sufficient + 1 mislead insufficient per genuine q)
  <out>_pairs.jsonl    ChainPair rows (mislead weaker -> gold stronger), enriched w/ question/options/answer
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.utils.extract_json import extract_json
from .io import read_jsonl, write_jsonl, dump_json, load_json
from .llm import LLMRouter
from .schema import EvidenceChain, EvidenceItem, ChainPair
from .sample_chains import (
    _multilayer_chain, _evidence_items, _format_evidence, _format_options, _PROMPT, _chunk_text)
from .oracle_rescue import _reduce_minimal, _confirm, REDUCE_FEEDBACK, _fmt_segments
from .judge_sufficiency import _gold_text, _fmt_options


# ── misleading-evidence selection prompt (the new capability) ──────────────────
SELECT_MISLEAD_SYSTEM = """\
You build the HARDEST distractor evidence for a multiple-choice question about a long video. You are
given the question, its options, the CORRECT answer, and candidate video segments (segment_id: text)
that were retrieved for this question but do NOT contain the decisive fact.

Pick the FEWEST segment_ids (usually 1-4, at most {cap}) that together form the most MISLEADING
evidence: text that looks topically relevant and plausible, but that would lead a careful reader who
sees ONLY these segments toward a WRONG option — because it emphasises a competing detail, states a
near-miss fact, or lacks the discriminating fact so the correct option cannot be verified. Do NOT
include any segment that actually verifies or is needed to reach the correct answer. If several
segments point at the same wrong option, prefer the smallest convincing set.

Return ONE strict JSON object, no prose, no fences:
{{"chunk_ids": [<segment_id>, ...], "target_wrong_option": "<letter of the option they push toward, or ''>",
  "reason": "<=25 words: why these mislead"}}
"""

SELECT_MISLEAD_USER = """\
QUESTION:
{question}

OPTIONS:
{options}

CORRECT ANSWER: {gold}

CANDIDATE SEGMENTS (retrieved but NOT the decisive evidence; segment_id: text):
{segments}

Pick the smallest (<= {cap}) most misleading {{"chunk_ids": [...], "target_wrong_option": "...", "reason": "..."}}.
"""


def _extract_gold(llm, sample, cmap, cand, size_cap, max_tokens):
    """REDUCE->CONFIRM with feedback + size escalation (mirrors gold_evidence._extract_gold).
    Returns (picked, reason, suff, jr, stage)."""
    if not cand:
        return [], "no candidates", False, "empty", "no-cand"
    prompt_cap = len(cand)
    picked, reason = _reduce_minimal(llm, sample, cmap, cand, prompt_cap, max_tokens, size_cap)
    suff, jr = _confirm(llm, sample, cmap, picked, max_tokens)
    stage = "minimal"
    if not suff:
        fb = REDUCE_FEEDBACK.format(prev=sorted(picked) or "[]", cap=size_cap)
        grown, r2 = _reduce_minimal(llm, sample, cmap, cand, prompt_cap, max_tokens, size_cap, fb)
        if grown and set(grown) != set(picked):
            s2, j2 = _confirm(llm, sample, cmap, grown, max_tokens)
            if s2:
                picked, reason, suff, jr, stage = grown, r2, True, j2, "grow"
    if not suff:
        fb = ("\nNOTE: the discriminating fact IS present in these candidates but a tiny selection "
              "missed the context needed to make it decisive. Include enough context.\n")
        for bigger in (size_cap * 2, size_cap * 3, len(cand)):
            cap_i = min(bigger, len(cand))
            sub, ri = _reduce_minimal(llm, sample, cmap, cand, prompt_cap, max_tokens, cap_i, fb)
            si, ji = _confirm(llm, sample, cmap, sub, max_tokens)
            if si:
                picked, reason, suff, jr, stage = sub, ri, True, ji, f"esc<={cap_i}"
                break
    return picked, reason, suff, jr, stage


def _answer(vlm, sample, cmap, ids, n_samples, temperature, no_evidence=False):
    """Closed-book answer over the given chunk texts (or no evidence). Majority vote.
    Returns (correct: bool, votes: str)."""
    ev_texts = [it.text for it in _evidence_items(cmap, ids)] if ids else []
    votes = Counter()
    for _ in range(max(1, n_samples)):
        if no_evidence or not ev_texts:
            prompt = ("You are answering a multiple-choice question about a long video.\n"
                      "(No video evidence is provided.)\n\n"
                      f"Question: {sample.question}\nChoices: {_format_options(sample.candidates)}\n"
                      "Answer with a single letter (A/B/C/D) only:")
        else:
            prompt = _PROMPT.format(n=len(ev_texts), evidence=_format_evidence(ev_texts),
                                    question=sample.question, choices=_format_options(sample.candidates))
        raw = vlm.chat_with_frames([], prompt, max_new_tokens=16,
                                   temperature=temperature, enable_thinking=False)
        m = re.search(r"\b([A-D])\b", raw or "")
        if not m:
            continue
        i = ord(m.group(1)) - ord("A")
        pred = sample.candidates[i] if 0 <= i < len(sample.candidates) else ""
        votes["correct" if pred == sample.answer else "wrong"] += 1
    c, t = votes["correct"], sum(votes.values())
    return (c * 2 > t), f"{c}/{t}"


def _grow_gold(vlm, llm, sample, cmap, cand, size_cap, hard_cap, grow_step, max_tokens, n_samples, temp):
    """Gold = minimal decisive set; if the closed-book answerer still misses, grow by most-relevant
    candidates until correct (cap hard_cap). Guarantees A when it returns acorr=True.
    Returns (gold_ids, reason, judge_suff, judge_reason, stage, A_correct, votes)."""
    picked, reason, suff, jr, stage = _extract_gold(llm, sample, cmap, cand, size_cap, max_tokens)
    acorr, avotes = _answer(vlm, sample, cmap, picked, n_samples, temp) if picked else (False, "0/0")
    if acorr or not cand:
        return picked, reason, suff, jr, stage, acorr, avotes
    cur = list(dict.fromkeys(int(x) for x in picked))
    have = set(cur)
    remaining = [c for c in cand if c not in have]
    i = 0
    while i < len(remaining) and len(cur) < hard_cap:
        cur += remaining[i:i + grow_step]
        i += grow_step
        acorr, avotes = _answer(vlm, sample, cmap, cur, n_samples, temp)
        if acorr:
            suff, jr = _confirm(llm, sample, cmap, cur, max_tokens)
            return cur, reason, suff, jr, "grown", True, avotes
    return cur, reason, suff, jr, "grow-fail", False, avotes


def _select_mislead(llm, sample, cmap, pool_ids, gold_ids, size_cap, max_tokens):
    """Model picks the most misleading subset from pool-minus-gold. Returns (ids, target, reason)."""
    gold_set = set(int(x) for x in gold_ids)
    rest = [c for c in pool_ids if int(c) not in gold_set and int(c) in cmap]
    if not rest:
        return [], "", "no non-gold candidates"
    chunks = [cmap[int(i)] for i in rest]
    user = SELECT_MISLEAD_USER.format(
        question=sample.question, options=_fmt_options(sample.candidates),
        gold=_gold_text(sample), segments=_fmt_segments(chunks), cap=size_cap)
    try:
        obj = extract_json(llm.chat(SELECT_MISLEAD_SYSTEM.format(cap=size_cap), user,
                                    max_tokens=max_tokens)) or {}
    except Exception as exc:
        return [], "", f"mislead error: {exc}"
    valid = set(int(c) for c in rest)
    picked, seen = [], set()
    for x in (obj.get("chunk_ids") or []):
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi in valid and xi not in seen:
            seen.add(xi); picked.append(xi)
    return picked, str(obj.get("target_wrong_option") or ""), str(obj.get("reason") or "")[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", default=None, help="json list of sample_idx (default: all in config)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/veil.yaml")
    ap.add_argument("--memory-dir", default="outputs/memory/videomme_ml3_n12")
    ap.add_argument("--embed-api-url", required=True)
    ap.add_argument("--vlm-api-url", required=True, help="deployed 27B (answerer AND llm judge/reduce)")
    ap.add_argument("--llm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--size-cap", type=int, default=6, help="gold REDUCE target")
    ap.add_argument("--hard-cap", type=int, default=8, help="max gold chunks via answerer-growth")
    ap.add_argument("--grow-step", type=int, default=2)
    ap.add_argument("--mislead-cap", type=int, default=4, help="max misleading chunks the model may pick")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--n-samples", type=int, default=1, help="answer votes (temp=0 -> 1 is deterministic)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample-limit", type=int, default=None, help="only first N idxs (debug)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)
    from experiments.tuning.veil_27b import load_samples
    samples = {int(s.sample_idx): s for s in load_samples(cfg)}
    if args.idxs and Path(args.idxs).is_file():
        idxs = [int(x) for x in json.load(open(args.idxs)) if int(x) in samples]
    else:
        idxs = sorted(samples)
    if args.sample_limit:
        idxs = idxs[:args.sample_limit]

    from src.clients.embedder import BGEM3Embedder
    from src.clients.siglip_embedder import SigLIPEmbedder
    from src.clients.vlm_client import VLMClient
    embedder = BGEM3Embedder(api_url=args.embed_api_url)
    siglip = SigLIPEmbedder(api_url=args.embed_api_url)
    vlm = VLMClient(model_path=args.llm_api_model, api_url=args.vlm_api_url, api_model=None)
    llm = LLMRouter.from_urls(args.vlm_api_url, model=args.llm_api_model)

    memory_dir = Path(args.memory_dir)
    bank_cache, cmap_cache, clock = {}, {}, threading.Lock()

    def bank_getter(vid):
        with clock:
            if vid in bank_cache:
                return bank_cache[vid], cmap_cache.get(vid)
        try:
            from src.build_memory.core.bank_loader import load_bank
            b = load_bank(memory_dir / vid)
            cm = {int(c.chunk_id): c for c in b.chunks}
            for nm in ("l2_chunks", "l3_chunks", "l4_chunks"):
                for c in (getattr(b, nm, None) or []):
                    cm[int(c.chunk_id)] = c
        except Exception as exc:
            print(f"[v6] bank load failed {vid}: {exc}")
            b, cm = None, None
        with clock:
            bank_cache[vid] = b
            if cm is not None:
                cmap_cache[vid] = cm
        return b, cm

    def work(q):
        try:
            return _work(q)
        except Exception as exc:
            print(f"[v6] q={q} failed: {exc}", flush=True)
            s = samples.get(q)
            return {"sample_idx": q, "video_id": getattr(s, "video_id", ""), "genuine": False,
                    "gold_stage": "error", "error": str(exc)[:200],
                    "gold_chunk_ids": [], "mislead_chunk_ids": [], "n_gold": 0, "n_mislead": 0,
                    "A": False, "B": None, "C": None, "votes": {}}

    def _work(q):
        s = samples[q]
        bank, cmap = bank_getter(s.video_id)
        if bank is None:
            return {"sample_idx": q, "video_id": s.video_id, "genuine": False, "gold_stage": "no-bank",
                    "gold_chunk_ids": [], "mislead_chunk_ids": [], "n_gold": 0, "n_mislead": 0,
                    "A": False, "B": None, "C": None, "votes": {}}
        cand = [c for c in _multilayer_chain(bank, s.question, s.candidates, embedder, siglip, args.alpha)
                if c in cmap]
        gold_ids, greason, gsuff, gjr, gstage, A, avotes = _grow_gold(
            vlm, llm, s, cmap, cand, args.size_cap, args.hard_cap, args.grow_step,
            args.max_tokens, args.n_samples, args.temperature)
        mislead_ids, target, mreason = _select_mislead(
            llm, s, cmap, cand, gold_ids, args.mislead_cap, args.max_tokens)
        B, bvotes = _answer(vlm, s, cmap, [], args.n_samples, args.temperature, no_evidence=True)
        C, cvotes = _answer(vlm, s, cmap, mislead_ids, args.n_samples, args.temperature) if mislead_ids \
            else (False, "0/0")
        genuine = bool(A and (B is False) and (C is False) and gold_ids and mislead_ids)
        return {"sample_idx": q, "video_id": s.video_id, "question_type": getattr(s, "question_type", ""),
                "gold_chunk_ids": [int(x) for x in gold_ids],
                "mislead_chunk_ids": [int(x) for x in mislead_ids],
                "n_gold": len(gold_ids), "n_mislead": len(mislead_ids), "ml_n": len(cand),
                "A": bool(A), "B": bool(B), "C": bool(C), "genuine": genuine,
                "target_wrong": target, "gold_stage": gstage, "judge_sufficient": bool(gsuff),
                "gold_reason": greason, "mislead_reason": mreason,
                "votes": {"A": avotes, "B": bvotes, "C": cvotes}}

    state_path = Path(str(args.out) + ".state.json")
    done: dict[str, dict] = {}
    if args.resume and state_path.is_file():
        done = load_json(state_path).get("done", {})
    todo = [q for q in idxs if str(q) not in done]
    print(f"[v6] {len(todo)} questions ({len(done)} cached) | {args.workers}-way | temp={args.temperature}")

    lock, ct = threading.Lock(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, q): q for q in todo}
        for f in as_completed(futs):
            row = f.result()
            with lock:
                done[str(row["sample_idx"])] = row
                ct += 1
                v = row.get("votes", {})
                print(f"[v6 {ct}/{len(todo)}] q={row['sample_idx']} "
                      f"gold={row['n_gold']} misl={row['n_mislead']} "
                      f"A={'Y' if row['A'] else '.'} B={'Y' if row.get('B') else '.'} "
                      f"C={'Y' if row.get('C') else '.'} {'GENUINE' if row['genuine'] else ''} "
                      f"(A{v.get('A','')} C{v.get('C','')})", flush=True)
                if ct % 20 == 0:
                    dump_json(state_path, {"done": done})

    rows = list(done.values())
    write_jsonl(Path(args.out), rows)
    dump_json(state_path, {"done": done})

    # ── build chains + pairs from genuine questions ──
    chains, pairs = [], []
    for r in rows:
        if not r.get("genuine"):
            continue
        q = int(r["sample_idx"]); s = samples[q]
        _, cmap = bank_getter(s.video_id)
        if cmap is None:
            continue
        qtype = getattr(s, "question_type", "") or r.get("question_type", "")
        gold_items = _evidence_items(cmap, r["gold_chunk_ids"])
        misl_items = _evidence_items(cmap, r["mislead_chunk_ids"])
        if not gold_items or not misl_items:
            continue
        pos_id, neg_id = f"{q}:gold", f"{q}:mislead"
        chains.append(EvidenceChain(
            sample_idx=q, video_id=s.video_id, question_type=qtype, chain_id=pos_id,
            quality="sufficient", source="gold", evidence=gold_items,
            notes=f"stage={r.get('gold_stage')} votesA={r['votes'].get('A')}",
            sufficient=True, sufficient_votes=r["votes"].get("A", ""),
            topk=None, config="gold", n_chunks=len(gold_items)).to_dict())
        chains.append(EvidenceChain(
            sample_idx=q, video_id=s.video_id, question_type=qtype, chain_id=neg_id,
            quality="insufficient", source="model_mislead", evidence=misl_items,
            notes=f"misleading; target={r.get('target_wrong')} votesC={r['votes'].get('C')}",
            sufficient=False, sufficient_votes=r["votes"].get("C", ""),
            topk=None, config="mislead", n_chunks=len(misl_items)).to_dict())
        pair = ChainPair(
            sample_idx=q, video_id=s.video_id, question_type=qtype,
            pair_id=f"{q}:{neg_id}->{pos_id}",
            weaker_chain_id=neg_id, stronger_chain_id=pos_id,
            pair_type="mislead_vs_gold").to_dict()
        pair["question"] = s.question
        pair["options"] = list(s.candidates)
        pair["answer"] = s.answer
        pairs.append(pair)

    cpath = str(args.out).rsplit(".jsonl", 1)[0] + "_chains.jsonl"
    ppath = str(args.out).rsplit(".jsonl", 1)[0] + "_pairs.jsonl"
    write_jsonl(cpath, chains)
    write_jsonl(ppath, pairs)

    # ── stats ──
    N = len(rows) or 1
    gen = sum(1 for r in rows if r.get("genuine"))
    Aok = sum(1 for r in rows if r.get("A"))
    Bprior = sum(1 for r in rows if r.get("B"))
    Cluck = sum(1 for r in rows if r.get("C"))
    stages = Counter(r.get("gold_stage") for r in rows)
    print(f"\n[v6] N={len(rows)}  (temp={args.temperature})")
    print(f"  A gold correct       : {Aok}/{N} ({Aok/N*100:.1f}%)")
    print(f"  B prior (no-ev ✓)    : {Bprior}  (excluded)")
    print(f"  C mislead worked (✓) : {N-Cluck-sum(1 for r in rows if r.get('C') is None)} wrong / {Cluck} still-correct")
    print(f"  GENUINE (A✓ B✗ C✗)   : {gen}/{N} ({gen/N*100:.1f}%)")
    print(f"  gold stages          : {dict(stages)}")
    print(f"  -> {gen} pairs (1 gold x 1 mislead each)")
    print(f"  -> {cpath}\n  -> {ppath}")


if __name__ == "__main__":
    main()
