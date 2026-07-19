#!/usr/bin/env python3
"""Summarize EgoLife rubric-eval jsonl with sufficiency score separation."""
from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path


def pct(n: int, d: int) -> str:
    return "nan" if d == 0 else f"{100.0 * n / d:.1f}%"


def quantiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    ys = sorted(xs)

    def q(frac: float) -> float:
        if len(ys) == 1:
            return ys[0]
        pos = frac * (len(ys) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ys) - 1)
        w = pos - lo
        return ys[lo] * (1 - w) + ys[hi] * w

    return {
        "n": len(xs),
        "mean": round(stats.fmean(xs), 4),
        "p10": round(q(0.10), 4),
        "p50": round(q(0.50), 4),
        "p90": round(q(0.90), 4),
    }


def iter_verdicts(rec: dict):
    for it in rec.get("trace_iters") or []:
        verdict = str(it.get("verdict") or "").upper()
        score = it.get("rubric_score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
        yield {
            "iter": it.get("iter"),
            "verdict": verdict,
            "score": score,
            "option_scores": {
                k: v for k, v in (it.get("rubric_criteria_scores") or {}).items()
                if isinstance(v, (int, float))
            },
            "n_evidence": it.get("n_evidence_to_verifier"),
            "unknown_options": it.get("unknown_options") or [],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = []
    for line in args.jsonl.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("record_type") == "header":
            continue
        rows.append(obj)

    by_type: dict[str, list[int]] = defaultdict(list)
    final_verdicts = Counter()
    final_scores_by_verdict: dict[str, list[float]] = defaultdict(list)
    all_scores_by_verdict: dict[str, list[float]] = defaultdict(list)
    final_evidence_by_verdict: dict[str, list[float]] = defaultdict(list)
    iter_counts = Counter()
    threshold_grid = [round(0.15 + 0.05 * i, 2) for i in range(17)]
    holistic_sweep = {t: Counter() for t in threshold_grid}
    option_sweep = {t: Counter() for t in threshold_grid}

    for rec in rows:
        by_type[str(rec.get("question_type") or "unknown")].append(int(bool(rec.get("correct"))))
        verdicts = list(iter_verdicts(rec))
        iter_counts[len(verdicts)] += 1
        for v in verdicts:
            if v["score"] is not None:
                all_scores_by_verdict[v["verdict"]].append(v["score"])
        if verdicts:
            f = verdicts[-1]
            final_verdicts[f["verdict"]] += 1
            if f["score"] is not None:
                final_scores_by_verdict[f["verdict"]].append(f["score"])
                for t in threshold_grid:
                    holistic_sweep[t]["sufficient" if f["score"] >= t else "insufficient"] += 1
            opt_scores = {
                str(k).upper()[:1]: float(v)
                for k, v in (f.get("option_scores") or {}).items()
                if str(k).upper()[:1]
            }
            if opt_scores:
                for t in threshold_grid:
                    # Proxy for evidence sufficiency in the per-option verifier:
                    # every answer option has enough evidence to be judged.
                    ok = len(opt_scores) >= len(rec.get("candidates") or []) and all(v >= t for v in opt_scores.values())
                    option_sweep[t]["sufficient" if ok else "insufficient"] += 1
            try:
                final_evidence_by_verdict[f["verdict"]].append(float(f["n_evidence"]))
            except (TypeError, ValueError):
                pass

    total = len(rows)
    correct = sum(int(bool(r.get("correct"))) for r in rows)
    summary = {
        "path": str(args.jsonl),
        "n_records": total,
        "overall_accuracy": None if total == 0 else round(correct / total, 4),
        "accuracy_by_type": {
            qt: {"n": len(vals), "accuracy": round(sum(vals) / len(vals), 4)}
            for qt, vals in sorted(by_type.items())
            if vals
        },
        "final_verdict_counts": dict(final_verdicts),
        "final_rubric_score_by_verdict": {
            k: quantiles(v) for k, v in sorted(final_scores_by_verdict.items())
        },
        "all_iter_rubric_score_by_verdict": {
            k: quantiles(v) for k, v in sorted(all_scores_by_verdict.items())
        },
        "final_evidence_count_by_verdict": {
            k: quantiles(v) for k, v in sorted(final_evidence_by_verdict.items())
        },
        "trace_iter_count_distribution": dict(iter_counts),
        "final_holistic_threshold_sweep": {
            str(t): dict(c) for t, c in holistic_sweep.items() if sum(c.values())
        },
        "final_option_all_options_threshold_sweep": {
            str(t): dict(c) for t, c in option_sweep.items() if sum(c.values())
        },
    }

    lines = [
        f"file: {args.jsonl}",
        f"records: {total}",
        f"overall accuracy: {correct}/{total} ({pct(correct, total)})",
        "",
        "accuracy by type:",
    ]
    for qt, vals in sorted(by_type.items()):
        lines.append(f"  {qt}: {sum(vals)}/{len(vals)} ({pct(sum(vals), len(vals))})")
    lines += [
        "",
        f"final verdict counts: {dict(final_verdicts)}",
        f"trace iteration counts: {dict(iter_counts)}",
        "",
        "final rubric_score by verdict:",
    ]
    for k, v in sorted(summary["final_rubric_score_by_verdict"].items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("all-iteration rubric_score by verdict:")
    for k, v in sorted(summary["all_iter_rubric_score_by_verdict"].items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("final evidence count by verdict:")
    for k, v in sorted(summary["final_evidence_count_by_verdict"].items()):
        lines.append(f"  {k}: {v}")
    if summary["final_holistic_threshold_sweep"]:
        lines.append("")
        lines.append("final holistic threshold sweep:")
        for t, c in summary["final_holistic_threshold_sweep"].items():
            lines.append(f"  thr={float(t):.2f}: {c}")
    if summary["final_option_all_options_threshold_sweep"]:
        lines.append("")
        lines.append("final option-score all-options threshold sweep:")
        for t, c in summary["final_option_all_options_threshold_sweep"].items():
            lines.append(f"  thr={float(t):.2f}: {c}")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        args.out.with_suffix(args.out.suffix + ".json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
