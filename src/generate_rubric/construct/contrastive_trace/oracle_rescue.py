"""Oracle rescue for RECALL-GAP questions — build a sufficient positive chain for every
question where query retrieval never surfaced enough evidence.

Motivation: after gold-aware `judge_sufficiency`, ~293/900 VideoMME questions have NO sufficient
query chain (retrieval missed the discriminating fact). make_pairs drops those questions — they
lose their positive anchor. Per the method, we DON'T drop them: a strong model reads the
question + options + GOLD answer and searches the WHOLE memory bank for the minimal chunk subset
that verifies / excludes, producing an oracle positive. Only construction-time (sees gold).

Because a video has up to ~440 chunks (~80k tokens of caption/asr), we can't dump everything into
one prompt reliably. Map-reduce:
  MAP     split the video's chunks into windows; per window the model lists chunk_ids whose TEXT
          holds any fact relevant to verifying/excluding an option (generous recall).
  REDUCE  from the union of candidates, the model picks the MINIMAL subset that FORCES the correct
          answer.
  CONFIRM re-run the gold-aware judge (same sufficiency bar as everything else). Kept only if it
          passes. Failures = caption-coverage gaps (the fact isn't in the bank at all) and are
          logged, not forced.

Out: <out> jsonl = {sample_idx, video_id, chunk_ids, reason, judge_sufficient, judge_reason}.
Feed the sufficient rows back either via sample_chains --oracle-chains, or directly merged into
chains_judged as source="oracle" chains (see --emit-chains).
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.utils.extract_json import extract_json
from .io import read_jsonl, write_jsonl, dump_json, load_json
from .llm import LLMRouter
from .schema import EvidenceChain
from .sample_chains import _chunk_text, _evidence_items
from .judge_sufficiency import JUDGE_SYSTEM, JUDGE_USER, _fmt_options, _fmt_evidence, _gold_text

# ── prompts ──
MAP_SYSTEM = """\
You collect CANDIDATE evidence segments. You are given a multiple-choice question, its options,
the CORRECT answer, and a batch of numbered video segments (segment_id: caption text).

List every segment_id whose TEXT contains a concrete fact that could help either VERIFY the correct
option or EXCLUDE a competing option. Be generous — include a segment if it plausibly bears on the
answer — but skip segments with no relevant fact at all. Judge only the text shown; the gold is
given so you know what to look for, not to be restated.

Return ONE strict JSON object, no prose, no fences: {"ids": [<segment_id>, ...]}  (may be empty).
"""

MAP_USER = """\
QUESTION:
{question}

OPTIONS:
{options}

CORRECT ANSWER: {gold}

SEGMENTS (segment_id: text):
{segments}

Which segment_ids hold a fact relevant to verifying the correct option or excluding another?
Return {{"ids": [...]}}.
"""

REDUCE_SYSTEM = """\
You assemble the SMALLEST sufficient evidence set for a multiple-choice question. You are given the
question, its options, the CORRECT answer, and candidate video segments (segment_id: text).

Pick the FEWEST segment_ids (usually 1-5, at most {cap}) whose COMBINED text would FORCE a careful
reader to the correct answer — it either states the fact the correct option asserts (VERIFY), or
contains the facts that rule out every competing option (EXCLUDE). Include a segment only if it
carries a fact needed for that; exclude everything else. Do not pad the set. Only if NO candidate
contains the discriminating fact at all, return an empty list.

Return ONE strict JSON object, no prose, no fences:
{{"chunk_ids": [<segment_id>, ...], "reason": "<=25 words: which facts make it sufficient, or why none do"}}
"""

REDUCE_USER = """\
QUESTION:
{question}

OPTIONS:
{options}

CORRECT ANSWER: {gold}

CANDIDATE SEGMENTS (segment_id: text):
{segments}
{feedback}
Return the smallest (<= {cap}) {{"chunk_ids": [...], "reason": "..."}} that forces the correct answer,
or an empty list only if none of these segments contain the discriminating fact.
"""

REDUCE_FEEDBACK = """
NOTE: a previous selection {prev} was judged INSUFFICIENT — it was missing the discriminating fact.
The fact IS present among these candidates. Select a DIFFERENT small subset (still <= {cap}) that
actually contains it — add the missing segment, don't just enlarge blindly.
"""


def _fmt_segments(chunks) -> str:
    return "\n".join(f"[{int(c.chunk_id)}] {_chunk_text(c)}" for c in chunks if _chunk_text(c))


def _windows(chunks, win: int):
    for i in range(0, len(chunks), win):
        yield chunks[i:i + win]


def _map_candidates(llm, sample, chunks, win, max_tokens) -> list[int]:
    """Union of relevant chunk_ids across windows (generous recall)."""
    cand: list[int] = []
    seen = set()
    valid = {int(c.chunk_id) for c in chunks}
    for w in _windows(chunks, win):
        user = MAP_USER.format(
            question=sample.question, options=_fmt_options(sample.candidates),
            gold=_gold_text(sample), segments=_fmt_segments(w))
        try:
            raw = llm.chat(MAP_SYSTEM, user, max_tokens=max_tokens)
            ids = (extract_json(raw) or {}).get("ids") or []
        except Exception as exc:
            print(f"[map] {sample.sample_idx} window failed: {exc}")
            ids = []
        for x in ids:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if xi in valid and xi not in seen:
                seen.add(xi); cand.append(xi)
    return cand


def _reduce_minimal(llm, sample, cmap, cand_ids, prompt_cap, max_tokens, size_cap, feedback=""):
    """Pick the smallest verifying subset (<= size_cap) from candidates. Returns (chunk_ids, reason).
    prompt_cap bounds how many candidates go INTO the prompt; size_cap bounds the OUTPUT set."""
    if not cand_ids:
        return [], "no relevant candidates found in bank"
    # chronological order; cap to keep the reduce prompt bounded
    cand_ids = sorted(cand_ids)[:prompt_cap]
    chunks = [cmap[i] for i in cand_ids if i in cmap]
    user = REDUCE_USER.format(
        question=sample.question, options=_fmt_options(sample.candidates),
        gold=_gold_text(sample), segments=_fmt_segments(chunks), cap=size_cap, feedback=feedback)
    try:
        raw = llm.chat(REDUCE_SYSTEM.format(cap=size_cap), user, max_tokens=max_tokens)
        obj = extract_json(raw) or {}
    except Exception as exc:
        print(f"[reduce] {sample.sample_idx} failed: {exc}")
        return [], f"reduce error: {exc}"
    valid = set(cand_ids)
    picked, seen = [], set()
    for x in (obj.get("chunk_ids") or []):
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi in valid and xi not in seen:
            seen.add(xi); picked.append(xi)
    return picked, str(obj.get("reason") or "")[:200]


def _confirm(llm, sample, cmap, chunk_ids, max_tokens):
    """Run the gold-aware judge on the assembled oracle chain. Returns (sufficient, reason)."""
    if not chunk_ids:
        return False, "empty"
    ev_texts = [it.text for it in _evidence_items(cmap, chunk_ids)]
    user = JUDGE_USER.format(
        question=sample.question, options=_fmt_options(sample.candidates),
        gold=_gold_text(sample), n=len(ev_texts), evidence=_fmt_evidence(ev_texts))
    try:
        obj = extract_json(llm.chat(JUDGE_SYSTEM, user, max_tokens=max_tokens)) or {}
    except Exception as exc:
        return False, f"judge error: {exc}"
    return bool(obj.get("sufficient")), str(obj.get("reason") or "")[:200]


def _rescue_one(llm, sample, bank_getter, win, prompt_cap, size_cap, max_tokens):
    """MAP → REDUCE → CONFIRM, confirm-gated so reduce over-pruning never loses a rescuable question,
    while keeping the oracle POSITIVE small (<= size_cap) so it doesn't dwarf the insufficient
    negatives (a huge positive would let the rubric cheat on evidence volume — the very inversion we
    fixed, in reverse):
      1. minimal reduce (<=size_cap) → confirm; if sufficient, done.
      2. else feedback-reduce (different small subset that contains the fact) → confirm; if ok, done.
      3. else confirm the FULL candidate set purely to CLASSIFY: if that's insufficient it's a true
         caption gap; if it's sufficient the fact is present but reduce couldn't isolate it, so do
         ONE final shrink-reduce constrained to <=size_cap and keep that (never the whole set).
    """
    bank, cmap = bank_getter(sample.video_id)
    if bank is None:
        return {"sample_idx": int(sample.sample_idx), "video_id": sample.video_id,
                "chunk_ids": [], "reason": "bank load failed", "stage": "no-bank",
                "n_candidates": 0, "judge_sufficient": False, "judge_reason": "no bank"}
    cand = _map_candidates(llm, sample, bank.chunks, win, max_tokens)

    # 1) minimal reduce at the tight size_cap
    picked, reason = _reduce_minimal(llm, sample, cmap, cand, prompt_cap, max_tokens, size_cap)
    suff, jr = _confirm(llm, sample, cmap, picked, max_tokens)
    stage = "minimal"

    # 2) feedback reduce (same tight cap, different subset that contains the fact)
    if not suff and cand:
        fb = REDUCE_FEEDBACK.format(prev=sorted(picked) or "[]", cap=size_cap)
        grown, reason2 = _reduce_minimal(llm, sample, cmap, cand, prompt_cap, max_tokens, size_cap, fb)
        if grown and set(grown) != set(picked):
            suff2, jr2 = _confirm(llm, sample, cmap, grown, max_tokens)
            if suff2:
                picked, reason, suff, jr, stage = grown, reason2, True, jr2, "grow"

    # 3) escalate size: the judge is context-sensitive — a fact can need 3-5 chunks of surrounding
    #    context to be decisive. Try larger caps and keep the FIRST that confirms (smallest working).
    if not suff and cand:
        suff_all, _ = _confirm(llm, sample, cmap, sorted(cand)[:prompt_cap], max_tokens)
        if suff_all:                       # fact IS in the pool; find the smallest sufficient subset
            fb = ("\nNOTE: the discriminating fact IS present in these candidates but small selections "
                  "missed the surrounding context needed to make it decisive. Include enough context.\n")
            for bigger in (size_cap * 2, size_cap * 3):
                cap_i = min(bigger, len(cand))
                sub, reason_i = _reduce_minimal(llm, sample, cmap, cand, prompt_cap, max_tokens, cap_i, fb)
                suff_i, jr_i = _confirm(llm, sample, cmap, sub, max_tokens)
                if suff_i:
                    picked, reason, suff, jr, stage = sub, reason_i, True, jr_i, f"escalate<={cap_i}"
                    break
            if not suff:                   # rescuable but reduce can't isolate: keep capped pool
                capped = sorted(cand)[:min(size_cap * 3, len(cand))]
                s_cap, j_cap = _confirm(llm, sample, cmap, capped, max_tokens)
                picked, reason, suff, jr, stage = capped, "kept capped candidate pool", s_cap, j_cap, "pool-fallback"

    return {"sample_idx": int(sample.sample_idx), "video_id": sample.video_id,
            "chunk_ids": picked, "reason": reason, "stage": stage,
            "n_candidates": len(cand), "n_picked": len(picked),
            "judge_sufficient": suff, "judge_reason": jr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-idxs", required=True, help="json list of recall-gap sample_idx")
    ap.add_argument("--out", required=True, help="oracle_chains.jsonl (sample_idx+chunk_ids+judge)")
    ap.add_argument("--config", default="configs/videomme.yaml")
    ap.add_argument("--memory-dir", default="outputs/memory/videommeL")
    ap.add_argument("--llm-api-url", required=True)
    ap.add_argument("--llm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--window", type=int, default=40, help="chunks per MAP window")
    ap.add_argument("--reduce-cap", type=int, default=100, help="max candidates INTO the reduce prompt")
    ap.add_argument("--size-cap", type=int, default=12,
                    help="max chunks in the OUTPUT oracle positive (keep it near negative sizes)")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--workers", type=int, default=32, help="parallel questions")
    ap.add_argument("--limit", type=int, default=None, help="only first N gap questions (debug)")
    ap.add_argument("--emit-chains", default=None,
                    help="also write sufficient oracle rows as source=oracle EvidenceChains here")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)
    from experiments.tuning.reveal_27b import load_samples
    samples = {s.sample_idx: s for s in load_samples(cfg)}

    gap = json.load(open(args.gap_idxs))
    gap = [int(x) for x in gap if int(x) in samples]
    if args.limit:
        gap = gap[:args.limit]

    state_path = Path(str(args.out) + ".state.json")
    done: dict[str, dict] = {}
    if args.resume and state_path.is_file():
        done = load_json(state_path).get("done", {})
        print(f"[oracle] resume: {len(done)} already rescued")
    todo = [q for q in gap if str(q) not in done]
    print(f"[oracle] {len(todo)} recall-gap questions to rescue ({len(done)} cached) | {args.workers}-way")

    memory_dir = Path(args.memory_dir)
    bank_cache: dict[str, object] = {}
    cmap_cache: dict[str, dict] = {}
    cache_lock = threading.Lock()

    def bank_getter(vid):
        with cache_lock:
            if vid in bank_cache:
                return bank_cache[vid], cmap_cache.get(vid)
        try:
            from src.build_memory.core.bank_loader import load_bank
            b = load_bank(memory_dir / vid)
            cm = {int(c.chunk_id): c for c in b.chunks}
        except Exception as exc:
            print(f"[oracle] bank load failed {vid}: {exc}")
            b, cm = None, None
        with cache_lock:
            bank_cache[vid] = b
            if cm is not None:
                cmap_cache[vid] = cm
        return b, cm

    llm = LLMRouter.from_urls(args.llm_api_url, model=args.llm_api_model)
    lock = threading.Lock()
    ct = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_rescue_one, llm, samples[q], bank_getter,
                          args.window, args.reduce_cap, args.size_cap, args.max_tokens): q for q in todo}
        for f in as_completed(futs):
            row = f.result()
            with lock:
                done[str(row["sample_idx"])] = row
                ct += 1
                mark = "OK " if row["judge_sufficient"] else "gap"
                print(f"[oracle {ct}/{len(todo)}] q={row['sample_idx']} {mark} "
                      f"picked={len(row['chunk_ids'])} cand={row.get('n_candidates')} "
                      f"stage={row.get('stage')} :: {row['judge_reason'][:55]}", flush=True)
                if ct % 20 == 0:
                    dump_json(state_path, {"done": done})

    rows = list(done.values())
    write_jsonl(args.out, rows)
    dump_json(state_path, {"done": done})
    nok = sum(1 for r in rows if r["judge_sufficient"])
    from collections import Counter
    stages = Counter(r.get("stage") for r in rows)
    sizes = [r.get("n_picked", len(r["chunk_ids"])) for r in rows if r["judge_sufficient"]]
    import statistics
    print(f"[oracle] wrote {len(rows)} rows -> {args.out}")
    print(f"[oracle] rescued(sufficient)={nok} ({nok/max(1,len(rows))*100:.1f}%)  "
          f"caption-gap(unrescuable)={len(rows)-nok}")
    print(f"[oracle] stages={dict(stages)}")
    if sizes:
        print(f"[oracle] positive size: min={min(sizes)} med={int(statistics.median(sizes))} max={max(sizes)}")

    if args.emit_chains:
        chains = []
        for r in rows:
            if not (r["judge_sufficient"] and r["chunk_ids"]):
                continue
            s = samples[int(r["sample_idx"])]
            _, cmap = bank_getter(r["video_id"])
            if cmap is None:
                continue
            ids = r["chunk_ids"]
            chains.append(EvidenceChain(
                sample_idx=int(r["sample_idx"]), video_id=r["video_id"],
                question_type=getattr(s, "question_type", ""),
                chain_id=f"{r['sample_idx']}:oracle:k{len(ids)}",
                quality="sufficient", source="oracle",
                evidence=_evidence_items(cmap, ids),
                notes=f"oracle_rescue; {r['reason']}",
                sufficient=True, sufficient_votes="oracle",
                topk=None, config="oracle", n_chunks=len(set(ids)),
            ).to_dict())
        # carry judge metadata so downstream (make_pairs) sees the same schema as chains_judged
        for c in chains:
            c["judge_sufficient"] = True
            c["judge_reason"] = "oracle_rescue"
        write_jsonl(args.emit_chains, chains)
        print(f"[oracle] emitted {len(chains)} source=oracle chains -> {args.emit_chains}")


if __name__ == "__main__":
    main()
