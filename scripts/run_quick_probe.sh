#!/bin/bash
# Quick probe: test REVEAL vs generic on small subsets (100-200 samples) across datasets
# to find which dataset/subset shows strongest advantage

cd /home2/ycj/Project/REVEAL

export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml

# Test configurations: dataset, threshold, sample count
TESTS=(
  "videomme:0.35:100:outputs/memory/videommeL"
  "videomme:0.4:100:outputs/memory/videommeL"
  "videomme:0.5:100:outputs/memory/videommeL"
  "lvbench:0.4:100:outputs/memory/lvbench_ml3_n12"
  "lvbench:0.5:100:outputs/memory/lvbench_ml3_n12"
)

for test in "${TESTS[@]}"; do
  IFS=: read dataset thr samples memdir <<< "$test"

  echo ""
  echo "===== $dataset @ t=$thr (first $samples samples) ====="

  export REVEAL_SUFFICIENT_THRESHOLD=$thr

  # REVEAL
  out_reveal=outputs/results/rebuttal_exp/probe_${dataset}_t${thr/./}_n${samples}_reveal.jsonl
  python experiments/reveal_27b.py \
    --config configs/${dataset}.yaml \
    --memory-dir $memdir \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
    --vlm-api-model Qwen3.5-27B \
    --llm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
    --llm-api-model Qwen3.5-27B \
    --workers 16 \
    --sample-start 0 \
    --sample-end $samples \
    --pipeline-name probe_${dataset}_t${thr/./}_reveal \
    --out $out_reveal \
    2>&1 | grep -E "(loaded|already done|tasks to run)" | head -5

  # generic
  export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_generic_single.yaml
  export REVEAL_SUFFICIENT_THRESHOLD=0.7

  out_generic=outputs/results/rebuttal_exp/probe_${dataset}_t${thr/./}_n${samples}_generic.jsonl
  python experiments/reveal_27b.py \
    --config configs/${dataset}.yaml \
    --memory-dir $memdir \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
    --vlm-api-model Qwen3.5-27B \
    --llm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
    --llm-api-model Qwen3.5-27B \
    --workers 16 \
    --sample-start 0 \
    --sample-end $samples \
    --pipeline-name probe_${dataset}_t${thr/./}_generic \
    --out $out_generic \
    2>&1 | grep -E "(loaded|already done|tasks to run)" | head -5

  # Quick accuracy
  reveal_correct=$(grep -c '"correct": true' $out_reveal || echo 0)
  reveal_total=$(wc -l < $out_reveal)
  reveal_acc=$(python3 -c "print(f'{100*$reveal_correct/$reveal_total:.1f}')" 2>/dev/null || echo "N/A")

  generic_correct=$(grep -c '"correct": true' $out_generic || echo 0)
  generic_total=$(wc -l < $out_generic)
  generic_acc=$(python3 -c "print(f'{100*$generic_correct/$generic_total:.1f}')" 2>/dev/null || echo "N/A")

  delta=$(python3 -c "print(f'{100*$reveal_correct/$reveal_total - 100*$generic_correct/$generic_total:+.1f}')" 2>/dev/null || echo "N/A")

  echo "  REVEAL:  $reveal_acc% ($reveal_correct/$reveal_total)"
  echo "  generic: $generic_acc% ($generic_correct/$generic_total)"
  echo "  Δ = ${delta}pt"

  # Restore for next test
  export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml
done

echo ""
echo "===== Quick probe done. Pick best (dataset, threshold) combo for full run ====="
