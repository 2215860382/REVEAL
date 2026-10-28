#!/usr/bin/env bash
# Self-contained sequential driver for the remaining rubric experiments.
# Survives session teardown (setsid). Tears down vLLM + keepalive at the end.
cd /home2/ycj/Project/REVEAL

HIST=outputs/rubric/_recovered_historical
MV=outputs/rubric/_clean_multivote
mkdir -p "$MV" outputs/logs/rubric
RUBRIC=$HIST/rubric_v2_consolidated.yaml
EMBED=http://127.0.0.1:9000
VLLM=http://127.0.0.1:8001
MODEL=Qwen3.5-27B
CFG=configs/videomme.yaml
BANK=outputs/memory/videommeL

PX="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    -u AI_PROXY_HTTP -u AI_PROXY_SOCKS -u NODE_USE_ENV_PROXY \
    NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0 no_proxy=localhost,127.0.0.1,::1,0.0.0.0"

echo "##### STEP A: write calibrated threshold 0.35 into rubric YAML #####"
python - <<'PY'
import yaml
p="outputs/rubric/_recovered_historical/rubric_v2_consolidated.yaml"
d=yaml.safe_load(open(p))
n=0
def setthr(o):
    global n
    if isinstance(o,dict):
        if "sufficient_threshold" in o:
            o["sufficient_threshold"]=0.35; n+=1
        for v in o.values(): setthr(v)
    elif isinstance(o,list):
        for v in o: setthr(v)
setthr(d)
out="outputs/rubric/_recovered_historical/rubric_v2_consolidated_t035.yaml"
yaml.safe_dump(d,open(out,"w"),allow_unicode=True,sort_keys=False)
print(f"set {n} thresholds -> {out}")
PY

echo
echo "##### STEP B: ablation 2' -- relax Pass-1 (8-15) + re-sweep #####"
REVEAL_VERIFIER_PICK_RANGE=8-15 $PX \
  python -m src.generate_rubric.construct.contrastive_trace.score_pairs \
    --chains $HIST/chains.jsonl --pairs $HIST/pairs.jsonl --rubric $RUBRIC \
    --config $CFG --llm-api-url $VLLM --llm-api-model $MODEL \
    --no-images --workers 32 --progress-every 200 \
    --out $MV/sweep_pick815.jsonl --resume \
  && echo "[B] pick8-15 sweep done" || echo "[B] FAILED"
# offline fit for B
$PX python -m src.generate_rubric.construct.contrastive_trace.fit_weights_threshold \
    --agg $MV/sweep_pick815.jsonl --crit $MV/sweep_pick815.crit.jsonl \
    --out $MV/calib_pick815.json --folds 5 --fixed 0.8 2>&1 | tail -8 \
  || echo "[B] fit FAILED"

echo
echo "##### STEP C: ablation 4 -- multi-vote clean pairs (n=5, temp=0.7) #####"
$PX python -m src.generate_rubric.construct.contrastive_trace.select_gold_mislead \
    --out $MV/mv.jsonl --config $CFG --memory-dir $BANK \
    --embed-api-url $EMBED --vlm-api-url $VLLM --llm-api-model $MODEL \
    --n-samples 5 --temperature 0.7 --workers 32 --resume \
  && echo "[C] multi-vote rebuild done" || echo "[C] rebuild FAILED"

echo "----- STEP C.2: score clean pairs with same rubric + sweep -----"
if [ -f $MV/mv_pairs.jsonl ]; then
  $PX python -m src.generate_rubric.construct.contrastive_trace.score_pairs \
      --chains $MV/mv_chains.jsonl --pairs $MV/mv_pairs.jsonl --rubric $RUBRIC \
      --config $CFG --llm-api-url $VLLM --llm-api-model $MODEL \
      --no-images --workers 32 --progress-every 200 \
      --out $MV/sweep_clean.jsonl --resume \
    && echo "[C.2] clean-pair sweep done" || echo "[C.2] FAILED"
  $PX python -m src.generate_rubric.construct.contrastive_trace.fit_weights_threshold \
      --agg $MV/sweep_clean.jsonl --crit $MV/sweep_clean.crit.jsonl \
      --out $MV/calib_clean.json --folds 5 --fixed 0.8 2>&1 | tail -22 \
    || echo "[C.2] fit FAILED"
else
  echo "[C.2] no mv_pairs.jsonl -> skip"
fi

echo
echo "##### TEARDOWN: stop vLLM(8001) + keepalive #####"
KP=$(pgrep -f "while true.*8001"); [ -n "$KP" ] && kill $KP 2>/dev/null && echo "keepalive killed $KP"
PID=$(ss -ltnp 2>/dev/null | grep ':8001' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
if [ -n "$PID" ]; then
  PGID=$(ps -o pgid= -p $PID | tr -d ' ')
  kill -TERM -$PGID 2>/dev/null; sleep 8; kill -KILL -$PGID 2>/dev/null
  echo "vLLM 8001 (pid=$PID pgid=$PGID) torn down"
else
  echo "no vLLM on 8001"
fi
echo "##### ALL REMAINING EXPERIMENTS DONE #####"
