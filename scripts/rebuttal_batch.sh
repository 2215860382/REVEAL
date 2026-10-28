#!/usr/bin/env bash
# Rebuttal experiments: R3 tau-sweep + R4 variance, on the paper-reproducing pipeline
# (main HEAD d629bf2 + rubric_library.yaml + videommeL bank + 145:8001/8004 + local embed 9000).
# gate tau=0.7 already = 79.0 (==paper 79.1); this fills the remaining points.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
SUM=outputs/logs/rebuttal_batch_summary.log
mkdir -p outputs/results/videomme/rebuttal/tausweep outputs/results/videomme/rebuttal/variance
echo "=== batch start $(date '+%F %T') ===" >> "$SUM"

run() { # name thr out log
  local name="$1" thr="$2" out="$3" log="$4" attempt
  for attempt in 1 2; do
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY \
      NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 \
      REVEAL_RUBRIC_PATH=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml \
      REVEAL_SUFFICIENT_THRESHOLD="$thr" \
      "$PY" experiments/reveal_27b.py \
        --config configs/videomme.yaml --memory-dir outputs/memory/videommeL \
        --embed-api-url http://127.0.0.1:9000 \
        --vlm-api-url http://10.82.1.145:8001,http://10.82.1.145:8004 --vlm-api-model Qwen3.5-27B \
        --llm-api-url http://10.82.1.145:8001,http://10.82.1.145:8004 --llm-api-model Qwen3.5-27B \
        --workers 32 --pipeline-name "$name" --out "$out" > "$log" 2>&1
    grep -q '=== Accuracy ===' "$log" && break
  done
  local ov
  ov=$(grep -A2 '=== Accuracy ===' "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]%' | tail -1)
  echo "$(date '+%T')  $name  thr=$thr  ->  Overall=${ov:-FAIL}" >> "$SUM"
}

# R3: tau sweep (0.7 already have from gate=79.0)
run sweep_thr060 0.6  outputs/results/videomme/rebuttal/tausweep/thr060.jsonl outputs/logs/rb_thr060.log
run sweep_thr080 0.8  outputs/results/videomme/rebuttal/tausweep/thr080.jsonl outputs/logs/rb_thr080.log
run sweep_thr085 0.85 outputs/results/videomme/rebuttal/tausweep/thr085.jsonl outputs/logs/rb_thr085.log
run sweep_thr090 0.9  outputs/results/videomme/rebuttal/tausweep/thr090.jsonl outputs/logs/rb_thr090.log
# R4: variance reruns at tau=0.7 (rep1 = gate = 79.0)
run var_rep2 0.7 outputs/results/videomme/rebuttal/variance/rep2.jsonl outputs/logs/rb_var2.log
run var_rep3 0.7 outputs/results/videomme/rebuttal/variance/rep3.jsonl outputs/logs/rb_var3.log

echo "=== batch done $(date '+%F %T') ===" >> "$SUM"
