#!/usr/bin/env python3
"""
从真实eval轨迹（如generic_lvbench.jsonl）提取高质量pairs。

筛选条件：
1. 有多轮迭代（len(trace_iters) > 1）
2. iter0 verdict=INSUFFICIENT，后续某轮=SUFFICIENT
3. 最终答对了（correct=True）

输出pairs.jsonl和chains.jsonl，格式兼容refine_rubric.py。
"""
import argparse
import json
import sys
from pathlib import Path


def extract_pairs(jsonl_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    chains = []

    with open(jsonl_path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            rec = json.loads(line)

            # Skip header
            if rec.get("record_type") == "header":
                continue

            # 必须有trace_iters且答对
            if "trace_iters" not in rec or not rec.get("correct", False):
                continue

            iters = rec["trace_iters"]
            if len(iters) < 2:
                continue

            # 找iter0=INSUFFICIENT，后续某轮=SUFFICIENT
            iter0 = iters[0]
            if iter0.get("verdict") != "INSUFFICIENT":
                continue

            # 找第一个SUFFICIENT的迭代
            suff_iter = None
            for it in iters[1:]:
                if it.get("verdict") == "SUFFICIENT":
                    suff_iter = it
                    break

            if not suff_iter:
                continue

            # 构造pair
            video_id = rec.get("video_id", "")
            sample_idx = rec.get("sample_idx", 0)
            question_type = rec.get("question_type", "unknown")

            weaker_id = f"{sample_idx}:iter0"
            stronger_id = f"{sample_idx}:iter{iters.index(suff_iter)}"
            pair_id = f"{sample_idx}:{weaker_id}->{stronger_id}"

            # weaker chain
            weaker_chain = {
                "sample_idx": sample_idx,
                "video_id": video_id,
                "question_type": question_type,
                "chain_id": weaker_id,
                "quality": "insufficient",
                "evidence": iter0.get("evidence", []),
                "verdict": "INSUFFICIENT",
                "iter": 0,
            }
            chains.append(weaker_chain)

            # stronger chain
            stronger_chain = {
                "sample_idx": sample_idx,
                "video_id": video_id,
                "question_type": question_type,
                "chain_id": stronger_id,
                "quality": "sufficient",
                "evidence": suff_iter.get("evidence", []),
                "verdict": "SUFFICIENT",
                "iter": iters.index(suff_iter),
            }
            chains.append(stronger_chain)

            # pair
            pair = {
                "sample_idx": sample_idx,
                "video_id": video_id,
                "question_type": question_type,
                "pair_id": pair_id,
                "weaker_chain_id": weaker_id,
                "stronger_chain_id": stronger_id,
                "pair_type": "insufficient_vs_sufficient",
                "question": rec.get("question", ""),
                "options": rec.get("options", []),
                "correct_answer": rec.get("correct_answer", ""),
            }
            pairs.append(pair)

    # 保存
    pairs_path = out_dir / "pairs.jsonl"
    chains_path = out_dir / "chains.jsonl"

    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(chains_path, "w") as f:
        for c in chains:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Extracted {len(pairs)} pairs from {jsonl_path.name}")
    print(f"  → {pairs_path}")
    print(f"  → {chains_path}")

    # 统计question_type分布
    type_counts = {}
    for p in pairs:
        qt = p["question_type"]
        type_counts[qt] = type_counts.get(qt, 0) + 1

    print("\nQuestion type distribution:")
    for qt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {qt}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pairs from eval traces")
    parser.add_argument("jsonl", help="Input eval jsonl (e.g. generic_lvbench.jsonl)")
    parser.add_argument("--out-dir", default="outputs/rubric/pairs_from_traces",
                        help="Output directory")

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found")
        sys.exit(1)

    extract_pairs(jsonl_path, Path(args.out_dir))
