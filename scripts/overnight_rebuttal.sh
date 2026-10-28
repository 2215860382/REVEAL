#!/usr/bin/env bash
# Unattended rebuttal experiments (EXP-1/2/3) on the paper-reproducing pipeline
# (commit d629bf2 + rubric_library.yaml + lvbench_ml3_n12 / videommeL banks).
# Endpoints: local 8001 (GPU4) + 8004 (GPU0); embedder 9000. Resumable via reveal_27b --out.
# Priority order ensures the make-or-break runs finish first.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
EMBED="http://127.0.0.1:9000"
RES=outputs/results/rebuttal_exp; mkdir -p "$RES" outputs/logs
SUM=outputs/logs/overnight_rebuttal_summary.log
PARSE=scripts/parse_exp_stats.py
RB_LIB=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
RB_GEN1=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_generic_single.yaml
RB_GENS=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_generic_structured.yaml
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 "$@"; }
log(){ echo "$(date '+%F %T')  $*" | tee -a "$SUM"; }

log "=== overnight rebuttal start ==="
# wait for both vLLM endpoints + embedder
for i in $(seq 1 240); do
  v=0; for p in 8001 8004; do np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null|grep -q Qwen && v=$((v+1)); done
  e=$(np curl -s -m4 $EMBED/health 2>/dev/null|grep -c '"bge":true')
  [ "$v" = 2 ] && [ "$e" = 1 ] && { log "endpoints ready (vLLM=2, embed=1)"; break; }
  sleep 10
done

# run <name> <config> <memdir> <rubric_path> <extra_env>
run(){
  local name="$1" cfg="$2" mem="$3" rubric="$4" extra="$5"
  local out="$RES/$name.jsonl" lg="outputs/logs/exp_$name.log"
  log ">>> RUN $name (rubric=$(basename "$rubric") env=[$extra])"
  np REVEAL_RUBRIC_PATH="$rubric" $extra \
    "$PY" experiments/reveal_27b.py \
      --config "$cfg" --memory-dir "$mem" --embed-api-url "$EMBED" \
      --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "$name" --out "$out" \
      >> "$lg" 2>&1 < /dev/null
  log "<<< DONE $name"
  $PY "$PARSE" "$out" "$name" 2>&1 | tee -a "$SUM"
}

# ---- Priority chain (critical first) ----
# 1) REVEAL positive arm on LVBench  (EXP-2 cost/freq + EXP-1 positive)
run reveal_lvbench      configs/lvbench.yaml  outputs/memory/lvbench_ml3_n12 "$RB_LIB"  ""
# 2) EXP-1 pilot: no-rubric generic single, LVBench
run generic_lvbench     configs/lvbench.yaml  outputs/memory/lvbench_ml3_n12 "$RB_GEN1" ""
# 3) EXP-3: stricter stop rule (all options decidable), LVBench
run aggmin_lvbench      configs/lvbench.yaml  outputs/memory/lvbench_ml3_n12 "$RB_LIB"  "REVEAL_SUFFICIENCY_AGG=min"
# 4) REVEAL positive arm on Video-MME-L  (EXP-2 second benchmark)
run reveal_vmme         configs/videomme.yaml outputs/memory/videommeL       "$RB_LIB"  ""
# 5) EXP-1 generic single on Video-MME-L
run generic_vmme        configs/videomme.yaml outputs/memory/videommeL       "$RB_GEN1" ""
# 6) EXP-1 variant C-ii: generic structured, LVBench
run genstruct_lvbench   configs/lvbench.yaml  outputs/memory/lvbench_ml3_n12 "$RB_GENS" ""

log "=== overnight rebuttal ALL DONE ==="
