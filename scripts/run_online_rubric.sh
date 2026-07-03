#!/usr/bin/env bash
# Online (RTD) rubric construction:
#   ① sample_chains  → ② make_pairs  (once, rubric-independent)
#   round1: refine over ALL pairs           → rubric_1
#   round r≥2: frontier_select → refine     → rubric_r   (until dry / max rounds)
#   inject groundedness guard → backtest
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home2/ycj/miniconda3/envs/veil/bin/python}"
CONFIG="${CONFIG:-configs/veil.yaml}"
MEMORY_DIR="${MEMORY_DIR:-outputs/memory/videomme_multiframe}"
EMBED_API_URL="${EMBED_API_URL:-http://127.0.0.1:9000}"          # BGE + SigLIP service
VLM_API_URL="${VLM_API_URL:-http://10.82.1.145:8002,http://10.82.1.145:8004}"  # multimodal answerer
LLM_API_URL="${LLM_API_URL:-http://10.82.1.145:8002,http://10.82.1.145:8004}"  # verifier + RTD LLM
OUT_ROOT="${OUT_ROOT:-outputs/rubric/online_$(date +%Y%m%d_%H%M%S)}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
WORKERS="${WORKERS:-12}"                                          # parallelism for answer/verify calls
ORACLE_CHAINS="${ORACLE_CHAINS:-}"                                # optional jsonl (Claude/GPT)
EXTRA_SAMPLE_ARGS="${EXTRA_SAMPLE_ARGS:-}"                        # e.g. --sample-limit 20
mkdir -p "$OUT_ROOT"

# Internal vLLM/embedding endpoints must bypass any local HTTP proxy, else the
# verifier's LLMClient (which honours *_PROXY) gets 502s. Add every endpoint host
# to no_proxy. (LLMRouter already sets trust_env=False; LLMClient does not.)
_hosts=$(printf '%s\n%s\n%s\n' "$EMBED_API_URL" "$VLM_API_URL" "$LLM_API_URL" \
  | tr ',' '\n' | sed -E 's#https?://##; s#[:/].*##' | sort -u | paste -sd, -)
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$_hosts"
export no_proxy="$NO_PROXY"

CHAINS="$OUT_ROOT/chains.jsonl"
PAIRS="$OUT_ROOT/pairs.jsonl"

echo "[sample] building candidate chains + gold labels → $CHAINS"
oracle_flag=""
[ -n "$ORACLE_CHAINS" ] && oracle_flag="--oracle-chains $ORACLE_CHAINS"
"$PYTHON" -m src.generate_rubric.construct.sample_chains \
  --config "$CONFIG" --memory-dir "$MEMORY_DIR" \
  --embed-api-url "$EMBED_API_URL" --vlm-api-url "$VLM_API_URL" \
  --workers "$WORKERS" $oracle_flag $EXTRA_SAMPLE_ARGS --out "$CHAINS"

echo "[pairs] insufficient × sufficient → $PAIRS"
"$PYTHON" -m src.generate_rubric.construct.make_pairs --chains "$CHAINS" --out "$PAIRS"

echo "[round1] refine over ALL pairs"
RUBRIC="$OUT_ROOT/rubric_r1.yaml"
"$PYTHON" -m src.generate_rubric.construct.refine_rubric \
  --pairs "$PAIRS" --chains "$CHAINS" --rubric-out "$RUBRIC" \
  --config "$CONFIG" --llm-api-url "$LLM_API_URL"

r=2
while [ "$r" -le "$MAX_ROUNDS" ]; do
  FRONTIER="$OUT_ROOT/frontier_r${r}.jsonl"
  echo "[round$r] frontier_select with $RUBRIC"
  "$PYTHON" -m src.generate_rubric.construct.frontier_select \
    --pairs "$PAIRS" --chains "$CHAINS" --rubric "$RUBRIC" \
    --config "$CONFIG" --llm-api-url "$LLM_API_URL" --workers "$WORKERS" --out "$FRONTIER"

  n=$(wc -l < "$FRONTIER" | tr -d ' ')
  echo "[round$r] frontier size = $n"
  if [ "$n" -eq 0 ]; then
    echo "[round$r] frontier dry → stop"
    break
  fi

  NEXT="$OUT_ROOT/rubric_r${r}.yaml"
  "$PYTHON" -m src.generate_rubric.construct.refine_rubric \
    --pairs "$FRONTIER" --chains "$CHAINS" --rubric-in "$RUBRIC" --rubric-out "$NEXT" \
    --config "$CONFIG" --llm-api-url "$LLM_API_URL"
  RUBRIC="$NEXT"
  r=$((r + 1))
done

echo "[finalize] inject groundedness guard + backtest"
FINAL="$OUT_ROOT/rubric_final.yaml"
"$PYTHON" - "$RUBRIC" "$FINAL" <<'PY'
import sys
from src.generate_rubric.construct.rubric_io import load_rubric, save_rubric, inject_groundedness
r = load_rubric(sys.argv[1]); inject_groundedness(r); save_rubric(sys.argv[2], r)
print("final rubric →", sys.argv[2])
PY

"$PYTHON" -m src.generate_rubric.construct.backtest_report \
  --chains "$CHAINS" --pairs "$PAIRS" --rubric "$FINAL" \
  --config "$CONFIG" --llm-api-url "$LLM_API_URL" --workers "$WORKERS" \
  --out "$OUT_ROOT/backtest_report.json"

echo "done. artifacts in $OUT_ROOT ; final rubric = $FINAL"
echo "to deploy:  export VEIL_RUBRIC_PATH=$FINAL"
