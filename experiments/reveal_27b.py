"""REVEAL-27B — iterative evidence retrieval with rubric judgment + rubric rerank.

Answerer / Planner / Verifier all use Qwen3.5-27B via --vlm-api-url / --llm-api-url.

Usage:
    cd /path/to/REVEAL
    PYTHONPATH=. python experiments/reveal_27b.py \\
        --config configs/videomme.yaml \\
        --vlm-api-url http://localhost:8000 \\
        --llm-api-url http://localhost:8001 \\
        --bge-gpu cuda:3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.logging import get_logger
PIPELINE_NAME = "reveal_27b"


from dataclasses import dataclass
from typing import List, Literal, Optional, Set

import numpy as np

from src.agents.verifier import _inject_images
from src.utils.extract_json import extract_json


# ── Legacy chunk-level dedup (moved verbatim from reveal.py) ──────────────────

def _dedup_by_similarity(
    new_texts: List[str],
    new_ids:   List[int],
    new_vecs:  List[List[float]],
    existing_vecs: List[List[float]],
    threshold: float = 0.85,
) -> tuple[List[str], List[int], List[List[float]]]:
    """Drop new chunks whose v_text cosine sim ≥ threshold against any already-accumulated chunk."""
    if not new_vecs or all(not v for v in new_vecs):
        return new_texts, new_ids, new_vecs
    kept_t, kept_i, kept_v = [], [], []
    running = list(existing_vecs)
    for text, cid, vec in zip(new_texts, new_ids, new_vecs):
        if not vec:
            kept_t.append(text); kept_i.append(cid); kept_v.append(vec)
            continue
        v = np.array(vec, dtype=np.float32)
        if running:
            mat = np.array(running, dtype=np.float32)
            if float(np.max(mat @ v)) >= threshold:
                continue
        kept_t.append(text); kept_i.append(cid); kept_v.append(vec)
        running.append(vec)
    return kept_t, kept_i, kept_v


# ── Records ───────────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    chunk_id: int
    text: str
    v_text: List[float]
    source_query: str
    round: int
    status: str = "kept"   # "kept" | "dropped_dup" | "dropped_irrelevant"


@dataclass
class RetrievalAttempt:
    round: int
    query: str
    params: dict
    n_retrieved: int
    n_kept: int
    n_dropped_dup: int
    n_dropped_irrelevant: int
    retrieval_quality: str            # compact WORKED / NO NEW EVIDENCE diagnosis


# ── Editable evidence state ──────────────────────────────────────────────────

_LEGACY_EVIDENCE_STATE_SYS = """\
You maintain a compact, editable evidence state for one multiple-choice video question.
The current state contains evidence-grounded information retained from earlier rounds. The new
evidence is the ONLY source from which you may introduce a new claim or change an existing claim.

The state must contain answer-relevant information extracted from evidence, not retrieval logs,
search plans, generic summaries, an answer letter, or a guessed final answer. On every update:
1. ADD newly supported facts that help distinguish the options.
2. REVISE existing claims when new evidence makes them more precise or changes their certainty.
3. REMOVE claims contradicted by stronger new evidence or no longer useful for this question.
4. Preserve useful prior facts not affected by the new evidence.
5. Keep uncertainty and contradictions explicit; never turn an inference into an observed fact.
6. If no new text or image evidence is supplied, copy the current state exactly.
7. Output the COMPLETE replacement state, not a delta or a description of your edits.
8. Keep the complete state under 1000 words and self-contained because the final answerer will see
   this state instead of the raw retrieved fragments.

Return ONLY the JSON object required by the response schema.
"""

_OPTION_STRINGS_EVIDENCE_STATE_SYS = """\
You maintain a compact, editable evidence state for one multiple-choice video question.
Maintain exactly FOUR short strings, one for each answer option in the original order. Each string
states only what the currently retrieved evidence says about that option's exact proposition.
The current strings retain evidence-grounded information from earlier rounds. New evidence is the
only source from which you may introduce or revise a claim; preserve still-valid prior information.

Rules:
1. If evidence explicitly supports the option, state the shortest decisive observed fact.
2. If evidence explicitly contradicts the option, state the shortest decisive contradiction.
3. If evidence neither supports nor contradicts it, say that no retrieved evidence addresses that
   exact proposition and that it is not directly ruled out.
4. If support comes only from an interested person/institution's explanation, say explicitly that
   it is only their stated claim and is not independently established.
5. Do not broaden or substitute concepts. For example, "facilitating processes" is not "helping
   with licensing"; obstructing a reporter is not evidence about providing physical directions.
6. Absence of evidence is not contradiction. Never convert inference into an observed fact.
7. Do not output status labels, evidence lists, missing-evidence/search recommendations, rubric
   judgments, option rankings, an answer letter, or a global conclusion.
8. Each option string must be self-contained, normally one sentence and at most 45 words. Preserve
   one useful timestamp/source tag when present, but do not repeat citations.

The four strings are the COMPLETE replacement state. Update them from the prior state and new
evidence. Return ONLY the JSON object required by the response schema.
"""


def _option_strings_evidence_state_prompt(n_options: int) -> str:
    """Keep the established four-option prompt unchanged; support other option counts."""
    if n_options == 4:
        return _OPTION_STRINGS_EVIDENCE_STATE_SYS
    number_words = {
        1: ("ONE", "one"),
        2: ("TWO", "two"),
        3: ("THREE", "three"),
        5: ("FIVE", "five"),
    }
    upper, lower = number_words.get(n_options, (str(n_options), str(n_options)))
    return _OPTION_STRINGS_EVIDENCE_STATE_SYS.replace(
        "exactly FOUR short strings", f"exactly {upper} short strings"
    ).replace("The four strings", f"The {lower} strings")


_TEMPORAL_EVIDENCE_STATE_GUARD = """

Temporal/ordering guard: when the question or an option asks which event/concept appeared first,
last, before, after, or in what sequence, record only dated atomic facts in each option string.
For every named event/concept, use its EARLIEST explicit occurrence among the retrieved evidence;
do not substitute a later repeat. Do not state a sequence, "first", "before", or "after" unless
the relevant earliest timestamps have all been compared. If an earliest occurrence is not verified,
say that the retrieved evidence has not established it.
"""


def _legacy_evidence_state_response_format() -> dict:
    """Schema used by the Online ablation that improved over Offline."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "online_evidence_state_update",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "evidence_state": {"type": "string", "minLength": 1},
                    "added": {"type": "array", "items": {"type": "string"}},
                    "revised": {"type": "array", "items": {"type": "string"}},
                    "removed": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["evidence_state", "added", "revised", "removed"],
                "additionalProperties": False,
            },
        },
    }


def _option_strings_evidence_state_response_format(
    candidates: List[str],
) -> dict:
    """Strict per-option schema; no model-generated judgments or audit fields."""
    option_properties = {
        chr(65 + i): {
            "type": "string",
            "minLength": 1,
            "maxLength": 320,
        }
        for i in range(len(candidates))
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "online_evidence_state_update",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": option_properties,
                "required": list(option_properties),
                "additionalProperties": False,
            },
        },
    }


# ── Online memory ───────────────────────────────────────────────────────────

# Strategy diagnosis for the <retrieval_quality> tag — a binary the planner can act on in one read. A
# retrieval is either productive (kept ≥1 new fragment) or brought back NOTHING NEW. The latter
# merges the two mechanically-identical dead ends: the query re-hit already-seen chunks (exact ids
# are excluded upstream in retrieval, so a reworded repeat surfaces here as zero new) OR the few
# that slipped through were near-duplicates of accumulated evidence (cosine dedup). Both mean the
# direction is exhausted — rewording won't help. (Off-target / misleading fragments are the
# relevance filter's job, not a reason category.)
_NO_NEW_EVIDENCE = (
    "NO NEW EVIDENCE — this retrieval returned only already-seen or near-duplicate chunks (or "
    "nothing at all); this query direction is exhausted. Do NOT reissue or reword it — target a "
    "DIFFERENT fact, or switch strategy (image if it must be seen, time for a specific span).")


class OnlineMemory:
    """Single question-conditioned inference-time state (see module docstring)."""

    def __init__(self, llm, question: str, candidates: List[str]):
        self.llm = llm
        self.question = question
        self.candidates = candidates
        self.evidence: List[EvidenceItem] = []
        self.trajectory: List[RetrievalAttempt] = []
        self.seen_ids: Set[int] = set()
        # Updated by the final Online ablation and, when explicitly enabled, the
        # paper-mainline Evidence-State experiment.
        self.evidence_state: str = ""
        self.evidence_state_options: dict = {}

    # -- accessors (kept evidence, in admission order) --------------------------
    def evidence_texts(self) -> List[str]:
        return [e.text for e in self.evidence]

    def evidence_ids(self) -> List[int]:
        return [e.chunk_id for e in self.evidence]

    def evidence_vecs(self) -> List[List[float]]:
        return [e.v_text for e in self.evidence]

    def update_evidence_state(
        self,
        new_evidence: List[str],
        *,
        round_index: int,
        new_keyframe_images=(),
        style: Literal["legacy", "option_strings"] = "option_strings",
        temporal_guardrails: bool = False,
    ) -> tuple[dict, str]:
        """Update and store the complete question-conditioned evidence state."""
        if style not in {"legacy", "option_strings"}:
            raise ValueError(
                "evidence-state style must be 'legacy' or 'option_strings', "
                f"got {style!r}"
            )
        opts = "\n".join(
            f"({chr(65+i)}) {candidate}"
            for i, candidate in enumerate(self.candidates)
        )
        evidence = (
            "\n".join(
                f"[R{round_index}E{i+1}] {text}"
                for i, text in enumerate(new_evidence)
            )
            or "(no new text fragment; inspect newly retrieved images if present)"
        )
        prior = (
            self.evidence_state
            or "(empty — no answer-relevant information recorded yet)"
        )
        user = (
            f"Question: {self.question}\n\nOptions:\n{opts}\n\n"
            f"Current evidence state:\n{prior}\n\n"
            f"New evidence from round {round_index}:\n{evidence}\n\n"
            "Update the evidence state now. The state itself must not mention these instructions."
        )
        umsg = {"role": "user", "content": user}
        if new_keyframe_images and getattr(self.llm, "_api_endpoints", None):
            umsg = _inject_images(umsg, new_keyframe_images)
        system_prompt = (
            _LEGACY_EVIDENCE_STATE_SYS
            if style == "legacy"
            else _option_strings_evidence_state_prompt(len(self.candidates))
        )
        if style == "option_strings" and temporal_guardrails:
            system_prompt += _TEMPORAL_EVIDENCE_STATE_GUARD
        response_format = (
            _legacy_evidence_state_response_format()
            if style == "legacy"
            else _option_strings_evidence_state_response_format(self.candidates)
        )
        raw = self.llm.chat(
            [{"role": "system", "content": system_prompt}, umsg],
            max_new_tokens=700,
            enable_thinking=False,
            response_format=response_format,
        )
        parsed = extract_json(raw) or {}
        if style == "legacy":
            state = str(parsed.get("evidence_state") or "").strip()
            if not state:
                raise RuntimeError(
                    "legacy evidence-state update returned an empty evidence_state; "
                    f"raw={str(raw)[:500]!r}"
                )
            changes = {
                key: [
                    str(item).strip()
                    for item in (parsed.get(key) or [])
                    if str(item).strip()
                ]
                for key in ("added", "revised", "removed")
            }
            self.evidence_state = state
            self.evidence_state_options = {}
            return changes, raw

        rendered = []
        state_by_option = {}
        for i, candidate in enumerate(self.candidates):
            label = chr(65 + i)
            option_state = str(parsed.get(label) or "").strip()
            if not option_state:
                raise RuntimeError(
                    f"evidence-state update missing state string for option {label}; "
                    f"raw={str(raw)[:500]!r}"
                )
            rendered.append(f"({label}) {candidate}\n{option_state}")
            state_by_option[label] = option_state
        state = "\n\n".join(rendered)
        self.evidence_state_options = state_by_option
        changes = {"added": [], "revised": [], "removed": []}
        if not state:
            raise RuntimeError(
                "evidence-state update returned an empty evidence_state; "
                f"raw={str(raw)[:500]!r}"
            )
        self.evidence_state = state
        return changes, raw

    # -- admission --------------------------------------------------------------
    def admit(
        self,
        new_texts: List[str],
        new_ids: List[int],
        new_vecs: List[List[float]],
        *,
        query: str,
        round: int,
        dedup_threshold: float,
        params: Optional[dict] = None,
    ) -> int:
        """Deduplicate, accumulate, and record retrieval quality. Returns ``n_kept``."""
        n_retrieved = len(new_ids)

        # 1. redundancy filter vs the current evidence cache
        d_texts, d_ids, d_vecs = _dedup_by_similarity(
            new_texts, new_ids, new_vecs, self.evidence_vecs(), dedup_threshold)
        n_after_dedup = len(d_ids)
        n_dropped_dup = n_retrieved - n_after_dedup

        n_dropped_irrelevant = 0

        # 2. accumulate
        for t, cid, v in zip(d_texts, d_ids, d_vecs):
            self.evidence.append(EvidenceItem(
                chunk_id=cid, text=t, v_text=v, source_query=query, round=round))
        self.seen_ids.update(d_ids)
        n_kept = len(d_ids)

        # 3. diagnose retrieval quality: productive (kept ≥1 new fragment) vs
        #    no-new-evidence (kept nothing: exact-id repeats already excluded upstream surface as
        #    zero retrieved, near-dups get deduped here — both are the same exhausted direction).
        retrieval_quality = (
            f"WORKED — added {n_kept} new evidence."
            if n_kept > 0 else _NO_NEW_EVIDENCE)
        self.trajectory.append(RetrievalAttempt(
            round=round, query=query, params=params or {},
            n_retrieved=n_retrieved, n_kept=n_kept, n_dropped_dup=n_dropped_dup,
            n_dropped_irrelevant=n_dropped_irrelevant,
            retrieval_quality=retrieval_quality))
        return n_kept

    # -- planner feedback -------------------------------------------------------
    def retrieval_quality_notes(self) -> str:
        """Per-attempt quality for the planner: one ``<retrieval_quality>`` per retrieval.

        The queries themselves are shown by the planner's history_query block (the ``Already tried``
        list — the proven repeat-preventer), so the trajectory carries only retrieval quality, not a
        duplicate ``<query>``: productive (added new evidence) vs an exhausted direction.
        ``method``/``query``/``counts``/``unknown``/``scores`` were dropped from the
        planner feed as noise or dilution; the useful gap lives in the verifier's missing_evidence,
        and the query history in history_query.
        """
        return "\n".join(
            f"  <retrieval_quality>{a.retrieval_quality}</retrieval_quality>"
            for a in self.trajectory)

    # -- serialization ----------------------------------------------------------
    def summary(self) -> dict:
        """Additive block for the result dict (never read by legacy consumers)."""
        return {
            "n_evidence": len(self.evidence),
            "n_attempts": len(self.trajectory),
            "evidence_state": self.evidence_state or None,
            "evidence_state_options": self.evidence_state_options or None,
            "attempts": [
                {"round": a.round, "query": a.query,
                 "n_retrieved": a.n_retrieved, "n_kept": a.n_kept,
                 "n_dropped_dup": a.n_dropped_dup, "n_dropped_irrelevant": a.n_dropped_irrelevant,
                 "retrieval_quality": a.retrieval_quality}
                for a in self.trajectory
            ],
            "evidence": [
                {"chunk_id": e.chunk_id, "round": e.round,
                 "source_query": e.source_query}
                for e in self.evidence
            ],
        }


log = get_logger(PIPELINE_NAME)


def _file_sha1(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_text(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _count_memory_videos(memory_dir: Path) -> int | None:
    if not memory_dir.exists():
        return None
    try:
        return sum(
            1
            for p in memory_dir.iterdir()
            if p.is_dir() and ((p / "L1.jsonl").exists() or (p / "narrative.json").exists())
        )
    except Exception:
        return None


def _egolife_query_cutoff(s) -> float | None:
    """EgoLife: absolute global-timeline second of a sample's query_time, so
    retrieval can drop memory AFTER it (future is off-limits). None for other
    benchmarks (no query_time) ⇒ no time filtering."""
    qt = getattr(s, "query_time", None)
    if not qt:
        return None
    date, tstr = qt.get("date"), qt.get("time")
    if not date or not tstr:
        return None
    try:
        from src.dataloader.egolife import _clip_abs_start
        day = int(str(date).upper().replace("DAY", ""))
        return _clip_abs_start(day, str(tstr))
    except Exception:
        return None


def load_samples(cfg: dict, filter_video_ids: set | None = None):
    bench = cfg["benchmark"]["name"]
    if bench == "videomme":
        from src.dataloader.videomme import load_videomme
        b = cfg["benchmark"]
        samples = load_videomme(
            parquet_path=b["parquet_path"],
            video_dir=b["video_dir"],
            duration_groups=b.get("duration_groups"),
        )
    elif bench == "lvbench":
        from src.dataloader.lvbench import load_lvbench
        b = cfg["benchmark"]
        samples = load_lvbench(
            meta_path=b["meta_path"],
            video_dir=b["video_dir"],
        )
    elif bench == "longvideobench":
        from src.dataloader.longvideobench import load_longvideobench
        b = cfg["benchmark"]
        samples = load_longvideobench(
            bench_root=b["bench_root"],
            video_dir=b["video_dir"],
            duration_groups=b.get("duration_groups"),
        )
    elif bench in ("egolife", "egolifeqa", "egor1_manual", "egor1_gemini", "egor1_both"):
        # EgoLife family: video_id = person; bank = that person's life-stream.
        # filter_people (optional) limits eval to persons whose bank is built.
        from src.dataloader.egolife import load_egolife
        b = cfg["benchmark"]
        samples = load_egolife(
            bench=bench,
            bench_root=b["bench_root"],
            video_dir=b.get("video_dir"),
            filter_people=b.get("filter_people"),
        )
    else:
        raise ValueError(f"Unknown benchmark: {bench}")
    if filter_video_ids is not None:
        samples = [s for s in samples if s.video_id in filter_video_ids]
    return samples


def main(run_pipeline=None):
    """Run a benchmark with an injected pipeline implementation.

    ``None`` selects the paper mainline. Ablation entrypoints pass their own
    implementation explicitly, keeping experiment routing out of this harness.
    """
    if run_pipeline is None:
        from experiments.core.reveal import run_reveal as run_pipeline
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True)
    ap.add_argument("--memory-dir",  default=None)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--filter-from", default=None,
                    help="JSONL with video_id keys; only run those videos")
    ap.add_argument("--sample-start", type=int, default=None,
                    help="Inclusive lower bound on sample_idx")
    ap.add_argument("--sample-end",   type=int, default=None,
                    help="Exclusive upper bound on sample_idx")
    ap.add_argument("--sample-indices-file", default=None,
                    help="JSONL containing sample_idx fields, or a text file with one sample_idx per line")
    ap.add_argument("--vlm-gpu",        default="cuda:0")
    ap.add_argument("--bge-gpu",        default="cuda:3")
    ap.add_argument("--llm-gpu",        default="cuda:1")
    ap.add_argument("--embed-api-url",  default=None,
                    help="Embedding service URL (BGE); skips loading locally")
    ap.add_argument("--no-keyframes",   action="store_true",
                    help="Do not load keyframe images for the answerer")
    ap.add_argument("--vlm-model",      default=None)
    ap.add_argument("--vlm-api-url",    default=None,
                    help="vLLM API base URL(s); comma-separated = round-robin")
    ap.add_argument("--vlm-api-model",  default=None)
    ap.add_argument("--llm-model",      default=None)
    ap.add_argument("--use-vllm",       action="store_true",
                    help="Use vLLM backend for local LLM")
    ap.add_argument("--llm-api-url",    default=None,
                    help="OpenAI-compatible API base URL(s); comma-separated = round-robin")
    ap.add_argument("--llm-api-model",  default=None,
                    help="Served model id for --llm-api-url (default: Qwen3.5-27B)")
    ap.add_argument("--verifier-api-url", default=None,
                    help="Optional OpenAI-compatible API used only by the verifier")
    ap.add_argument("--verifier-api-model", default=None,
                    help="Served model id for --verifier-api-url")
    ap.add_argument("--workers",             type=int, default=1,
                    help="Parallel workers (safe with API-mode LLM)")
    ap.add_argument("--pipeline-name",       default=None,
                    help="Override PIPELINE_NAME (= output jsonl filename stem)")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="Override reveal_loop.max_iter (e.g. 1 = single-round retrieval, no plan-repair loop)")
    args = ap.parse_args()
    global PIPELINE_NAME
    if args.pipeline_name:
        PIPELINE_NAME = args.pipeline_name

    cfg      = load_config(args.config)
    bench    = cfg["benchmark"]["name"]
    out_root = Path(cfg.get("paths", {}).get("outputs_root", "outputs"))

    _vl = cfg.get("reveal_loop", {})
    reveal_max_iter   = int(args.max_iter) if args.max_iter is not None else int(_vl.get("max_iter", 3))
    q_hist_dedup    = float(_vl.get("query_history_dedup_threshold", 0.9))
    ev_dedup        = float(_vl.get("evidence_dedup_threshold", 0.9))
    q_ev_dedup      = float(_vl.get("query_evidence_dedup_threshold", 0.9))

    memory_dir = Path(args.memory_dir) if args.memory_dir else \
                 out_root / "memory" / f"{bench}_L_27B"
    default_out_dir = Path(cfg.get("eval", {}).get("output_dir") or (out_root / "results" / bench))
    out_path = Path(args.out) if args.out else \
               default_out_dir / "reveal_27b.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep run metadata in a meta/ subfolder next to the results (less clutter).
    meta_path = out_path.parent / "meta" / (out_path.stem + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    rubric_override = os.environ.get("REVEAL_RUBRIC_PATH")
    rubric_path = Path(rubric_override).expanduser() if rubric_override else \
        repo_root / "outputs" / "rubric" / "rubric_contrastive.yaml"
    run_metadata = {
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "cwd": str(Path.cwd().resolve()),
        "command": [sys.executable, *sys.argv],
        "git_commit": _git_text(["rev-parse", "HEAD"]),
        "git_branch": _git_text(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git_text(["status", "--porcelain"])),
        "config_path": str(config_path.resolve()),
        "config_sha1": _file_sha1(config_path),
        "memory_dir": str(memory_dir.resolve()),
        "memory_dir_exists": memory_dir.exists(),
        "memory_video_dirs": _count_memory_videos(memory_dir),
        "rubric_path": str(rubric_path.resolve()),
        "rubric_sha1": _file_sha1(rubric_path),
        "rubric_path_env": rubric_override,
        "out_path": str(out_path.resolve()),
        "metadata_path": str(meta_path.resolve()),
        "pipeline": PIPELINE_NAME,
        "sample_start": args.sample_start,
        "sample_end": args.sample_end,
        "sample_indices_file": args.sample_indices_file,
        "vlm_api_url": args.vlm_api_url,
        "vlm_api_model": args.vlm_api_model,
        "llm_api_url": args.llm_api_url,
        "llm_api_model": args.llm_api_model,
        "verifier_api_url": args.verifier_api_url,
        "verifier_api_model": args.verifier_api_model,
        "embed_api_url": args.embed_api_url,
    }
    log.info("run_metadata: %s", json.dumps(run_metadata, ensure_ascii=False, sort_keys=True))
    sidecar_metadata = dict(run_metadata)
    sidecar_metadata.pop("command", None)
    meta_path.write_text(
        json.dumps(sidecar_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Filter video ids
    filter_vids: set | None = None
    if args.filter_from:
        filter_vids = set()
        for line in Path(args.filter_from).open():
            try: filter_vids.add(json.loads(line)["video_id"])
            except: pass
        log.info("filter: %d video_ids from %s", len(filter_vids), args.filter_from)

    samples = load_samples(cfg, filter_vids)
    if args.sample_start is not None:
        samples = [s for s in samples if int(s.sample_idx) >= args.sample_start]
    if args.sample_end is not None:
        samples = [s for s in samples if int(s.sample_idx) < args.sample_end]
    if args.sample_indices_file:
        selected_indices: set[int] = set()
        with Path(args.sample_indices_file).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                    selected_indices.add(int(value["sample_idx"] if isinstance(value, dict) else value))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid sample index line in {args.sample_indices_file}: {line!r}"
                    ) from exc
        samples = [s for s in samples if int(s.sample_idx) in selected_indices]
    log.info("loaded %d samples (%d videos)", len(samples),
             len({s.video_id for s in samples}))

    # Resume
    done_keys: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                rec = json.loads(line)
                if rec.get("record_type"):
                    continue
                key = rec.get("key")
                if not key:
                    key = f"{bench}|{rec['video_id']}|{rec['sample_idx']}|{PIPELINE_NAME}"
                done_keys.add(key)
            except Exception:
                pass
    log.info("already done: %d records", len(done_keys))

    # ── Model loading ──────────────────────────────────────────────────────────
    vlm = embedder = llm = verifier_llm = answerer = None

    from src.clients.vlm_client import VLMClient
    from src.agents.answerer import Answerer

    vlm_model = args.vlm_model or cfg["models"]["vlm"]["model_path"]
    if args.vlm_api_url:
        log.info("loading VLM via API %s ...", args.vlm_api_url)
        vlm = VLMClient(model_path=vlm_model, api_url=args.vlm_api_url,
                        api_model=args.vlm_api_model)
    else:
        log.info("loading VLM %s on %s ...", vlm_model, args.vlm_gpu)
        vlm = VLMClient(model_path=vlm_model, device=args.vlm_gpu)
    answerer = Answerer(vlm)
    log.info("  VLM ready")

    from src.clients.embedder import BGEM3Embedder
    if args.embed_api_url:
        log.info("loading BGE-M3 via API %s ...", args.embed_api_url)
        embedder = BGEM3Embedder(api_url=args.embed_api_url)
    else:
        log.info("loading BGE-M3 on %s ...", args.bge_gpu)
        t0 = time.time()
        embedder = BGEM3Embedder(
            model_path=cfg["models"]["embedder"]["model_path"],
            use_fp16=cfg["models"]["embedder"].get("use_fp16", True),
            device=args.bge_gpu,
        )
        log.info("  BGE-M3 ready (%.1fs)", time.time() - t0)

    from src.clients.llm_client import LLMClient
    if args.llm_api_url:
        api_model = args.llm_api_model or "Qwen3.5-27B"
        log.info("loading LLM via API %s ...", args.llm_api_url)
        t0 = time.time()
        llm = LLMClient(model_path=api_model, api_url=args.llm_api_url, api_model=api_model)
        log.info("  API LLM ready (%.1fs)", time.time() - t0)
    else:
        llm_model = args.llm_model or cfg["models"]["llm"]["model_path"]
        log.info("loading LLM %s on %s ...", llm_model, args.llm_gpu)
        t0 = time.time()
        llm = LLMClient(
            model_path=llm_model,
            device=args.llm_gpu,
            dtype=cfg["models"]["llm"].get("dtype", "bfloat16"),
            attn_impl=cfg["models"]["llm"].get("attn_impl", "sdpa"),
            use_vllm=args.use_vllm,
        )
        log.info("  LLM ready (%.1fs)", time.time() - t0)

    if args.verifier_api_url:
        verifier_api_model = args.verifier_api_model or "Qwen3.5-27B"
        log.info("loading verifier LLM via API %s ...", args.verifier_api_url)
        verifier_llm = LLMClient(
            model_path=verifier_api_model,
            api_url=args.verifier_api_url,
            api_model=verifier_api_model,
        )
        log.info("  verifier API LLM ready")

    # ── Thread safety (GPU models are not thread-safe) ────────────────────────
    _gpu_lock = threading.Lock()

    class _LockedModel:
        def __init__(self, model):
            self._m = model
        def __getattr__(self, name):
            attr = getattr(self._m, name)
            if callable(attr):
                def _locked(*a, **kw):
                    with _gpu_lock:
                        return attr(*a, **kw)
                return _locked
            return attr

    if embedder: embedder = _LockedModel(embedder)

    # ── Bank cache ────────────────────────────────────────────────────────────
    _bank_cache:  dict = {}
    _cache_lock = threading.Lock()

    def get_bank(video_id: str):
        with _cache_lock:
            if video_id in _bank_cache:
                return _bank_cache[video_id]
        vd = memory_dir / video_id
        if not (vd / "narrative.json").exists() and not (vd / "L1.jsonl").exists():
            return None
        from src.build_memory.core.bank_loader import load_bank
        bank = load_bank(vd)
        with _cache_lock:
            _bank_cache[video_id] = bank
        return bank

    from src.eval.parse_answer import parse_letter

    kf_dir = None if args.no_keyframes else memory_dir

    def run_sample(s):
        bank = get_bank(s.video_id)
        if bank is None:
            return None, "bank_missing"
        _ftopk = 8
        kw = dict(
            coarse_top_k=max(8, _ftopk),
            final_top_k=_ftopk,
            max_iter=reveal_max_iter,
            query_history_dedup_threshold=q_hist_dedup,
            evidence_dedup_threshold=ev_dedup,
            query_evidence_dedup_threshold=q_ev_dedup,
            keyframe_dir=kf_dir,
            video_path=getattr(s, "video_path", None),   # single mp4 per bank (VideoMME/LVBench/LongVB); egolife=dir→ignored
            rubric_rerank=True,
            max_time=_egolife_query_cutoff(s),
        )
        if verifier_llm is not None:
            kw["verifier_llm"] = verifier_llm
        return run_pipeline(s.question, s.candidates, bank, embedder, answerer, llm,
                            task_type=s.question_type, **kw), None

    # ── Run loop ──────────────────────────────────────────────────────────────
    write_header = (not out_path.exists()) or out_path.stat().st_size == 0
    out_fh      = out_path.open("a")
    _write_lock = threading.Lock()
    _stats_lock = threading.Lock()

    if write_header:
        out_fh.write(json.dumps({
            "record_type": "header",
            "benchmark": bench,
            "pipeline": PIPELINE_NAME,
            "command": run_metadata["command"],
            "metadata_path": str(meta_path.resolve()),
        }, ensure_ascii=False) + "\n")
        out_fh.flush()

    def append_record(rec: dict):
        with _write_lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()

    total      = len(samples)
    done_count = [len(done_keys)]
    correct_by: dict = {}

    def process_item(s) -> None:
        key = f"{bench}|{s.video_id}|{s.sample_idx}|{PIPELINE_NAME}"

        t0 = time.time()
        try:
            result, err = run_sample(s)
        except Exception as e:
            log.error("[%s/%s] ERROR: %s", s.video_id, s.question_type, e, exc_info=True)
            result, err = None, str(e)
        elapsed = time.time() - t0

        if result is None:
            pred_letter, pred_text, correct = "", "", False
        else:
            raw_ans = result.get("answer", "")
            pred_letter = raw_ans[:1].upper() if raw_ans else \
                          (parse_letter(result.get("raw", ""), len(s.candidates)) or "")
            idx = ord(pred_letter) - ord("A") if pred_letter else -1
            pred_text = s.candidates[idx] if 0 <= idx < len(s.candidates) else ""
            correct = (pred_text == s.answer)

        trace = result.get("trace_iters") if result else None
        rec = {
            "video_id":         s.video_id,
            "sample_idx":       s.sample_idx,
            "correct":          correct,
            "question_type":    s.question_type,
            "question":         s.question,
            "candidates":       s.candidates,
            "gold_answer":      s.answer,
            "pred_letter":      pred_letter,
            "pred_text":        pred_text,
            "evidence_chunk_ids": (result or {}).get("evidence_chunk_ids", []),
            "n_keyframes_to_answer": (result or {}).get("n_keyframes_to_answer"),
            "n_evidence_to_answer":  (result or {}).get("n_evidence_to_answer"),
            "online_memory":    (result or {}).get("online_memory"),
            "trace_iters":      trace,
            "elapsed":          round(elapsed, 2),
            "error":            err,
        }
        append_record(rec)

        with _stats_lock:
            done_count[0] += 1
            correct_by.setdefault(s.question_type, []).append(int(correct))
            n_done = done_count[0]

        n_iter = len(trace) if trace else 0
        log.info("[%d/%d] %-12s | %-16s | %s %s (iter=%d) %.1fs",
                 n_done, total, s.question_type, s.video_id,
                 pred_letter, "✓" if correct else "✗", n_iter, elapsed)

    tasks = [s for s in samples
             if f"{bench}|{s.video_id}|{s.sample_idx}|{PIPELINE_NAME}" not in done_keys]
    log.info("tasks to run: %d  (workers=%d)", len(tasks), args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_item, s) for s in tasks]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("worker exception: %s", exc)

    out_fh.close()

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n=== Accuracy ===")
    qtypes = sorted(correct_by)
    print(f"{'Pipeline':<20}" + "".join(f"{qt[:8]:>10}" for qt in qtypes) + f"{'Overall':>10}")
    row = f"{PIPELINE_NAME:<20}"
    all_c = []
    for qt in qtypes:
        c = correct_by.get(qt, [])
        acc = sum(c) / len(c) if c else float("nan")
        row += f"{acc*100:>9.1f}%"
        all_c.extend(c)
    ov = sum(all_c) / len(all_c) if all_c else float("nan")
    row += f"{ov*100:>9.1f}%"
    print(row)
    log.info("done — results in %s", out_path)


if __name__ == "__main__":
    main()
