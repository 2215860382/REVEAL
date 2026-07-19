"""RTD refinement engine — each pair triggers one in-place edit of the type rubric.

Replaces extract_pair_criteria + aggregate_type_rubric + weight_rubric. Given the current
per-type rubric and one (wrong-chain, right-chain) pair, the LLM emits edit operations
(add / rewrite / decompose / delete / adjust_weight) that we apply to that type's criteria.

Concurrency: pairs of the SAME question type are processed sequentially (each sees the
previous edit); different types run in parallel (independent template blocks).
Resume: processed pair_ids are journalled in <out>.state.json.
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Serialises checkpoint writes AND the in-place rubric mutations they snapshot, so a
# periodic save from one type-thread never serialises another type's list mid-append.
_SAVE_LOCK = threading.Lock()
_CKPT_EVERY = 20  # pairs

from src.utils.extract_json import extract_json
from .io import read_jsonl, load_json, dump_json
from .llm import LLMRouter
from .prompts import (
    RTD_REFINE_SYSTEM, RTD_REFINE_USER_TEMPLATE, format_options, format_evidence,
)
from .rubric_io import (
    load_rubric, save_rubric, ensure_template, REPAIR_ACTIONS, as_whether,
)

_CRIT_FIELDS = ("name", "description", "score_1", "score_half", "score_0")


def _clean_criterion(c: dict) -> dict | None:
    if not isinstance(c, dict):
        return None
    name = str(c.get("name") or "").strip()
    if not name:
        return None
    out = {
        "name": name,
        "description": as_whether(str(c.get("description") or "").strip()),
        "score_1": str(c.get("score_1") or "").strip(),
        "score_half": str(c.get("score_half") or "").strip(),
        "score_0": str(c.get("score_0") or "").strip(),
    }
    try:
        out["weight"] = float(c.get("weight", 1.0))
    except (TypeError, ValueError):
        out["weight"] = 1.0
    ra = str(c.get("failure_repair_action") or c.get("repair_action") or "refine_query").strip()
    out["failure_repair_action"] = ra if ra in REPAIR_ACTIONS else "refine_query"
    return out


def _find(crits: list[dict], name: str) -> int:
    for i, c in enumerate(crits):
        if c.get("name") == name:
            return i
    return -1


def _apply_ops(crits: list[dict], ops: list[dict]) -> int:
    """Mutate `crits` in place. Returns number of applied edits."""
    applied = 0
    for op in ops or []:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op") or "").strip()
        if kind == "add":
            c = _clean_criterion(op.get("criterion") or {})
            if c and _find(crits, c["name"]) == -1:
                crits.append(c); applied += 1
        elif kind == "rewrite":
            c = _clean_criterion(op.get("criterion") or {})
            if not c:
                continue
            i = _find(crits, str(op.get("target") or c["name"]))
            if i >= 0:
                crits[i] = c
            else:
                crits.append(c)
            applied += 1
        elif kind == "decompose":
            i = _find(crits, str(op.get("target") or ""))
            parts = [x for x in (_clean_criterion(p) for p in (op.get("into") or [])) if x]
            if i >= 0 and parts:
                crits[i:i + 1] = parts
                applied += 1
            elif parts:
                crits.extend(parts); applied += 1
        elif kind == "delete":
            i = _find(crits, str(op.get("target") or ""))
            if i >= 0:
                crits.pop(i); applied += 1
        elif kind == "adjust_weight":
            i = _find(crits, str(op.get("target") or ""))
            if i >= 0:
                try:
                    crits[i]["weight"] = float(op.get("weight", crits[i].get("weight", 1.0)))
                    applied += 1
                except (TypeError, ValueError):
                    pass
    return applied


def _gold_block(s, show_gold: bool) -> str:
    """Optional analysis-only gold answer, with hard guardrails against answer-correctness
    leakage. Empty string when show_gold is False (renders to nothing in the template)."""
    if not show_gold:
        return ""
    ans = getattr(s, "answer", "") or ""
    try:
        letter = chr(65 + list(s.candidates).index(ans))
    except (ValueError, AttributeError):
        letter = "?"
    return (
        f"\nCORRECT ANSWER (analysis aid ONLY — the verifier NEVER sees this at test time):\n"
        f"({letter}) {ans}\n"
        f"Use it only to pinpoint WHICH option the evidence must verify/exclude and what the "
        f"DECISIVE fact is. FORBIDDEN: criteria like 'evidence points to/supports the correct "
        f"option' — that is answer-correctness, not sufficiency. Every criterion must stay "
        f"TYPE-GENERAL (no this-question entities, numbers, or the answer text).\n"
    )


def _propose_pair(llm, qtype, base_type_json, general_json, pair, chains, samples, max_tokens,
                  show_gold=False):
    """Propose ops for ONE pair against the READ-ONLY base rubric snapshot (no serial
    dependency between pairs → fully parallel). Direction-locked prompt. Returns (qtype, pid, ops).
    Deduplication of paraphrase proposals is deferred to consolidate (step 4a)."""
    pid = pair["pair_id"]
    s = samples.get(int(pair["sample_idx"]))
    w = chains.get(pair["weaker_chain_id"])
    st = chains.get(pair["stronger_chain_id"])
    if s is None or w is None or st is None:
        return qtype, pid, []
    user = RTD_REFINE_USER_TEMPLATE.format(
        question_type=qtype, question=s.question, options=format_options(s.candidates),
        gold_block=_gold_block(s, show_gold),
        general_json=general_json, type_json=base_type_json,
        weaker_evidence=format_evidence(w.get("evidence") or []),
        stronger_evidence=format_evidence(st.get("evidence") or []),
    )
    try:
        raw = llm.chat(RTD_REFINE_SYSTEM, user, max_tokens=max_tokens)
        ops = extract_json(raw).get("operations") or []
    except Exception as exc:
        print(f"[refine {qtype}] pair {pid} failed: {exc}")
        ops = []
    return qtype, pid, ops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs (round1) or frontier_pairs (round≥2)")
    ap.add_argument("--chains", required=True)
    ap.add_argument("--rubric-in", default=None, help="current rubric yaml (omit → coarse seed)")
    ap.add_argument("--rubric-out", required=True)
    ap.add_argument("--config", default="configs/videomme.yaml")
    ap.add_argument("--llm-api-url", required=True)
    ap.add_argument("--llm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--type-workers", type=int, default=6, help="(legacy, ignored)")
    ap.add_argument("--workers", type=int, default=48,
                    help="max TYPES refined in parallel (each type is serial internally)")
    ap.add_argument("--max-per-type", type=int, default=30,
                    help="hard cap on criteria per type; at cap the pair may only "
                         "rewrite/decompose/delete/merge, never add")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--show-gold", action="store_true",
                    help="show the gold answer to the model during refine (analysis aid, "
                         "guardrailed against answer-correctness leakage)")
    args = ap.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)
    from experiments.tuning.reveal_27b import load_samples
    samples = {s.sample_idx: s for s in load_samples(cfg)}

    chains = {c["chain_id"]: c for c in read_jsonl(args.chains)}
    pairs = list(read_jsonl(args.pairs))

    state_path = Path(str(args.rubric_out) + ".state.json")
    state = {"done": []}
    resuming = args.resume and state_path.is_file()
    if resuming:
        state = load_json(state_path)
        state.setdefault("done", [])
    state["done"] = list(dict.fromkeys(state["done"]))

    # Resume from the partial checkpointed rubric if one exists; else from --rubric-in; else seed.
    if resuming and Path(args.rubric_out).is_file():
        rubric = load_rubric(args.rubric_out)
        print(f"[resume] {len(state['done'])} pairs done; continuing from {args.rubric_out}")
    else:
        rubric = load_rubric(args.rubric_in) if args.rubric_in else load_rubric("/nonexistent")

    llm = LLMRouter.from_urls(args.llm_api_url, model=args.llm_api_model)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_type[p.get("question_type", "")].append(p)
    for qtype in by_type:
        ensure_template(rubric, qtype)

    general_json = json.dumps(rubric["general"]["rubric_criteria"], ensure_ascii=False, indent=2)
    done_set = set(state["done"])
    cap = args.max_per_type
    ntodo = sum(1 for pl in by_type.values() for p in pl if p["pair_id"] not in done_set)
    ntype = min(len(by_type), args.workers)
    print(f"[refine] {ntodo} pairs across {len(by_type)} types | serial-in-type, "
          f"{ntype}-type parallel | cap={cap}/type")

    prog = {"ct": 0}

    def _run_type(qtype, pl):
        # SERIAL within a type: each pair sees the LIVE (already-mutated) criteria + count, so a
        # pair whose mechanism was just added by an earlier pair returns empty instead of cloning
        # it → no paraphrase bloat, no post-hoc truncation. Types run in parallel (own lists).
        crits = ensure_template(rubric, qtype)["rubric_criteria"]
        edits = 0
        for p in pl:
            if p["pair_id"] in done_set:
                continue
            n = len(crits)
            atcap = n >= cap
            notice = (f"# THIS TYPE CURRENTLY HAS {n}/{cap} CRITERIA — "
                      + ("AT CAP: you may ONLY rewrite/decompose/delete/merge existing criteria; "
                         "do NOT `add`. If nothing needs fixing, return empty operations.\n"
                         if atcap else
                         "add ONLY for a genuinely NEW mechanism no existing criterion covers.\n"))
            type_block = notice + json.dumps(crits, ensure_ascii=False, indent=2)
            _, pid, ops = _propose_pair(llm, qtype, type_block, general_json,
                                        p, chains, samples, args.max_tokens,
                                        show_gold=args.show_gold)
            if atcap:
                ops = [o for o in ops if str(o.get("op")) != "add"]
            with _SAVE_LOCK:  # guard mutation + checkpoint so a save never snapshots a list mid-append
                if ops:
                    edits += _apply_ops(crits, ops)
                    if len(crits) > cap:      # backstop: drop overflow adds beyond the cap
                        del crits[cap:]
                state["done"].append(pid)
                prog["ct"] += 1
                if prog["ct"] % 50 == 0:
                    print(f"[refine] {prog['ct']}/{ntodo}  ({qtype} @ {len(crits)}/{cap})", flush=True)
                    dump_json(state_path, state)
                    save_rubric(args.rubric_out, rubric)
        return qtype, edits

    with ThreadPoolExecutor(max_workers=ntype) as ex:
        futs = [ex.submit(_run_type, qt, pl) for qt, pl in by_type.items()]
        for f in as_completed(futs):
            qt, edits = f.result()
            print(f"[refine] type={qt}: {edits} edits → "
                  f"{len(ensure_template(rubric, qt)['rubric_criteria'])} criteria")

    save_rubric(args.rubric_out, rubric)
    dump_json(state_path, state)
    tot = sum(len(ensure_template(rubric, qt)["rubric_criteria"]) for qt in by_type)
    print(f"[refine] done; total criteria={tot} across {len(by_type)} types → {args.rubric_out}")


if __name__ == "__main__":
    main()
