#!/usr/bin/env bash
# target_time structural rubric for the EgoLife family (egolifeqa + egor1_manual).
#   ① construct pairs per config  (positive = target_time re-caption; negatives = mislead/irrelevant)
#   ② refine ONE rubric.yaml, per config in sequence (egolifeqa seeds, egor1 resumes the same file)
#   ③ (optional) score_pairs offline separability
# Runs in the `reveal` conda env (has decord+cv2+src stack). Endpoints probed live.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

PY=${PY:-/home2/ycj/miniconda3/envs/reveal/bin/python}
MEM=${MEM:-outputs/memory/egolife}
EMBED=${EMBED:-http://127.0.0.1:9000}
RB=${RB:-outputs/rubric/egolife_tt}
WORKERS=${WORKERS:-8}
CT=src.generate_rubric.construct.contrastive_trace
mkdir -p "$RB"
LOG="$RB/run.log"

# live VLM/LLM endpoints (27B): captioner + refine judge
_live(){ local a="" u; for u in http://127.0.0.1:8006 http://127.0.0.1:8003 \
  http://10.82.1.145:8001 http://10.82.1.145:8004; do
  curl -s -m4 "$u/v1/models" >/dev/null 2>&1 && a="${a:+$a,}$u"; done; echo "$a"; }
SRV=${SRV:-$(_live)}
[ -n "$SRV" ] || { echo "no live 27B endpoints"; exit 1; }
echo "[$(date +%F_%T)] SRV=$SRV EMBED=$EMBED MEM=$MEM RB=$RB WORKERS=$WORKERS" | tee -a "$LOG"

CONFIGS=("egolifeqa:configs/egolife.yaml" "egor1:configs/egor1.yaml")

# ── ① construct per config ────────────────────────────────────────────────────
for pc in "${CONFIGS[@]}"; do
  TAG=${pc%%:*}; CFG=${pc##*:}
  echo "[$(date +%F_%T)] construct $TAG ($CFG)" | tee -a "$LOG"
  "$PY" -m $CT.construct_targettime_pairs \
    --out "$RB/$TAG.jsonl" --config "$CFG" --memory-dir "$MEM" \
    --embed-api-url "$EMBED" --vlm-api-url "$SRV" --vlm-api-model Qwen3.5-27B \
    --tag "$TAG" --workers "$WORKERS" ${EXTRA_CONSTRUCT:-} --resume 2>&1 | tee -a "$LOG"
done

# ── ② refine ONE rubric.yaml, per config in sequence (2nd resumes the 1st) ─────
RUBRIC="$RB/rubric.yaml"
for pc in "${CONFIGS[@]}"; do
  TAG=${pc%%:*}; CFG=${pc##*:}
  echo "[$(date +%F_%T)] refine $TAG -> $RUBRIC" | tee -a "$LOG"
  "$PY" -m $CT.refine_rubric \
    --config "$CFG" \
    --pairs "$RB/${TAG}_pairs.jsonl" --chains "$RB/${TAG}_chains.jsonl" \
    --rubric-out "$RUBRIC" --llm-api-url "$SRV" --llm-api-model Qwen3.5-27B \
    --max-per-type 30 --show-gold --resume 2>&1 | tee -a "$LOG"
done

# ── ③ optional offline pair separability ──────────────────────────────────────
if [ "${SCORE:-0}" = "1" ]; then
  for pc in "${CONFIGS[@]}"; do
    TAG=${pc%%:*}; CFG=${pc##*:}
    "$PY" -m $CT.score_pairs \
      --config "$CFG" --pairs "$RB/${TAG}_pairs.jsonl" --chains "$RB/${TAG}_chains.jsonl" \
      --rubric "$RUBRIC" --llm-api-url "$SRV" --memory-dir "$MEM" \
      --no-images --workers "$WORKERS" --out "$RB/${TAG}_score.jsonl" --resume 2>&1 | tee -a "$LOG"
  done
fi

# rubric.yaml is now written natively in the rubric_contrastive layout (rubric_io.save_rubric).
echo "[$(date +%F_%T)] DONE -> $RUBRIC   (export REVEAL_RUBRIC_PATH=$RUBRIC)" | tee -a "$LOG"
