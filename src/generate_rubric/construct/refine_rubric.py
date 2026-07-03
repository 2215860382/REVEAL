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
from src.generate_rubric.construct.io import read_jsonl, load_json, dump_json
from src.generate_rubric.construct.llm import LLMRouter
from src.generate_rubric.construct.prompts import (
    RTD_REFINE_SYSTEM, RTD_REFINE_USER_TEMPLATE, format_options, format_evidence,
)
from src.generate_rubric.construct.rubric_io import (
    load_rubric, save_rubric, ensure_template, REPAIR_ACTIONS,
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
        "description": str(c.get("description") or "").strip(),
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


def _refine_one_type(llm, rubric, qtype, pairs, chains, samples, state, out_state_path,
                     rubric_out, max_tokens):
    """Sequentially refine one question type's template over its pairs."""
    tpl = ensure_template(rubric, qtype)
    crits = tpl["rubric_criteria"]
    general_json = json.dumps(rubric["general"]["rubric_criteria"], ensure_ascii=False, indent=2)
    n_edits = 0
    processed = 0
    for pair in pairs:
        pid = pair["pair_id"]
        if pid in state["done"]:
            continue
        s = samples.get(int(pair["sample_idx"]))
        w = chains.get(pair["weaker_chain_id"])
        st = chains.get(pair["stronger_chain_id"])
        if s is None or w is None or st is None:
            with _SAVE_LOCK:
                state["done"].append(pid)
            continue
        user = RTD_REFINE_USER_TEMPLATE.format(
            question_type=qtype, question=s.question, options=format_options(s.candidates),
            general_json=general_json,
            type_json=json.dumps(crits, ensure_ascii=False, indent=2),
            weaker_evidence=format_evidence(w.get("evidence") or []),
            stronger_evidence=format_evidence(st.get("evidence") or []),
        )
        ops = []
        try:
            raw = llm.chat(RTD_REFINE_SYSTEM, user, max_tokens=max_tokens)
            ops = extract_json(raw).get("operations") or []
        except Exception as exc:
            print(f"[refine {qtype}] pair {pid} failed: {exc}")
        # Mutate + journal under the lock so concurrent-type checkpoints see consistent state.
        with _SAVE_LOCK:
            n_edits += _apply_ops(crits, ops)
            state["done"].append(pid)
            processed += 1
            if processed % _CKPT_EVERY == 0:
                dump_json(out_state_path, state)
                save_rubric(rubric_out, rubric)
    return qtype, n_edits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs (round1) or frontier_pairs (round≥2)")
    ap.add_argument("--chains", required=True)
    ap.add_argument("--rubric-in", default=None, help="current rubric yaml (omit → coarse seed)")
    ap.add_argument("--rubric-out", required=True)
    ap.add_argument("--config", default="configs/veil.yaml")
    ap.add_argument("--llm-api-url", required=True)
    ap.add_argument("--llm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--type-workers", type=int, default=6, help="question types in parallel")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)
    from experiments.tuning.veil_27b import load_samples
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

    # Ensure templates exist before threads mutate distinct keys.
    for qtype in by_type:
        ensure_template(rubric, qtype)

    with ThreadPoolExecutor(max_workers=args.type_workers) as ex:
        futs = [ex.submit(_refine_one_type, llm, rubric, qt, pl, chains, samples,
                          state, state_path, args.rubric_out, args.max_tokens)
                for qt, pl in by_type.items()]
        for f in as_completed(futs):
            qt, n = f.result()
            print(f"[refine] type={qt} edits={n}")

    save_rubric(args.rubric_out, rubric)
    dump_json(state_path, state)
    print(f"wrote refined rubric → {args.rubric_out}")


if __name__ == "__main__":
    main()
