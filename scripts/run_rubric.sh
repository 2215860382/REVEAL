#!/usr/bin/env bash
# ── Stage 2 / 3 · Construct the evidence-sufficiency rubric (gold × mislead) ───
# ① select_gold_mislead : per question, pick minimal decisive GOLD evidence and
#    the strongest MISLEADING subset; closed-book temp-0 filter keeps only
#    genuine pairs (gold→correct, none→wrong, mislead→wrong). Emits <out>.jsonl
#    + <out>_chains.jsonl + <out>_pairs.jsonl.
# ② refine_rubric       : feed each (mislead insufficient × gold sufficient) pair
#    to the judge, which edits its own per-type rubric in place → rubric.yaml.
# ③ score_pairs         : offline pair-separation sanity check (weak vs strong).
#
# Prereqs: memory bank ($MEM) + BGE embed service ($EMBED) + 27B LLM ($SRV).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

MEM=${MEM:-outputs/memory/videommeL}
EMBED=${EMBED:-http://127.0.0.1:9000}
SRV=${SRV:-http://127.0.0.1:8003}                       # 27B: answerer + judge/reduce
RB=${RB:-outputs/rubric/pairs_v6}
WORKERS=${WORKERS:-12}
mkdir -p "$RB"
CT=src.generate_rubric.construct.contrastive_trace

# ① select gold + mislead, build clean pairs -----------------------------------
"$PY" -m $CT.select_gold_mislead \
  --out "$RB/sel.jsonl" --memory-dir "$MEM" \
  --embed-api-url "$EMBED" --vlm-api-url "$SRV" \
  --workers "$WORKERS" --temperature 0 --resume

# ② refine the rubric in place (--show-gold lets the judge see the gold answer) --
"$PY" -m $CT.refine_rubric \
  --pairs "$RB/sel_pairs.jsonl" --chains "$RB/sel_chains.jsonl" \
  --rubric-out "$RB/rubric.yaml" \
  --llm-api-url "$SRV" --max-per-type 30 --show-gold --resume

# ③ offline pair-separation score ----------------------------------------------
"$PY" -m $CT.score_pairs \
  --pairs "$RB/sel_pairs.jsonl" --chains "$RB/sel_chains.jsonl" \
  --rubric "$RB/rubric.yaml" --llm-api-url "$SRV" \
  --memory-dir "$MEM" --no-images --workers "$WORKERS" \
  --out "$RB/score.jsonl" --resume

echo "[rubric] done → $RB/rubric.yaml  (offline score → $RB/score.jsonl)"
echo "[rubric] to use it at eval time:  export VEIL_RUBRIC_PATH=$RB/rubric.yaml"
