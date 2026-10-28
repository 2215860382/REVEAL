#!/usr/bin/env bash
# egor1 再连跑 2 次(run2/run3)测方差, vLLM ctx 131072。run1=repro/main.jsonl(64.67%)。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SUM=outputs/logs/egor1_variance_x2.log
: > "$SUM"; echo "=== egor1 variance x2 start $(date '+%F %T') ===" >> "$SUM"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
       -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

ensure_vllm(){
  np curl -s -m5 http://127.0.0.1:8001/v1/models 2>/dev/null|grep -q Qwen || GPU=5 PORT=8001 MAXLEN=131072 bash scripts/start_vllm_27b.sh &
  np curl -s -m5 http://127.0.0.1:8004/v1/models 2>/dev/null|grep -q Qwen || GPU=6 PORT=8004 MAXLEN=131072 bash scripts/start_vllm_27b.sh &
  while true; do
    v=0; for p in 8001 8004; do np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null|grep -q Qwen && v=$((v+1)); done
    echo "$(date '+%T') vLLM up $v/2" >> "$SUM"; [ "$v" = 2 ] && break; sleep 20
  done
  for p in 8001 8004; do
    ml=$(np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["data"][0].get("max_model_len"))' 2>/dev/null)
    echo "$(date '+%T') port $p max_model_len=$ml" >> "$SUM"
    { [ -z "$ml" ] || [ "$ml" -lt 130000 ] 2>/dev/null; } && { echo "FATAL ctx<130k on $p" >> "$SUM"; exit 2; }
  done
}
ensure_embed(){
  np curl -s -m5 http://127.0.0.1:9000/health 2>/dev/null|grep -q '"bge":true' && return 0
  setsid env -u http_proxy -u https_proxy NO_PROXY='*' HF_HUB_OFFLINE=1 \
    "$PY" scripts/start_embed_server_live.py --port 9000 --device cuda:1 --idle-timeout 86400 \
    > outputs/logs/embed_9000.log 2>&1 &
  for i in $(seq 1 30); do np curl -s -m4 http://127.0.0.1:9000/health 2>/dev/null|grep -q '"bge":true' && return 0; sleep 5; done
}
run_one(){
  local tag="$1" out="outputs/results/egor1/variance/$1.jsonl"
  mkdir -p outputs/results/egor1/variance; rm -f "$out"
  echo "$(date '+%T') START $tag" >> "$SUM"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
    "$PY" experiments/reveal_27b.py \
      --config configs/egor1.yaml --memory-dir outputs/memory/egolife \
      --embed-api-url http://127.0.0.1:9000 \
      --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "egor1_$tag" --out "$out" \
      > outputs/logs/egor1_$tag.log 2>&1 < /dev/null
  local ov=$("$PY" - "$out" <<'PY'
import json,sys
t=c=0
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("record_type"): continue
    t+=1; c+=bool(d.get("correct"))
print(f"{100*c/t:.2f}% ({c}/{t})" if t else "NO DATA")
PY
)
  echo "$(date '+%T') RESULT[$tag] = $ov" >> "$SUM"
}

# 15min 空转看门狗
( idle=0
  while true; do sleep 60
    if pgrep -f 'reveal_27b.py' >/dev/null; then idle=0; else idle=$((idle+1)); fi
    if [ "$idle" -ge 15 ]; then
      for pid in $(for d in /proc/[0-9]*; do [ "$(cat $d/comm 2>/dev/null)" = python ] || continue; tr '\0' ' ' <$d/cmdline 2>/dev/null|grep -q 'vllm.entrypoints' && echo ${d#/proc/}; done); do kill "$pid" 2>/dev/null; done
      echo "$(date '+%T') idle 15min -> vLLM killed" >> "$SUM"; exit 0
    fi
  done ) & echo "$(date '+%T') reaper=$!" >> "$SUM"

ensure_vllm; ensure_embed
run_one run2
run_one run3

echo "" >> "$SUM"
echo "=== egor1 方差(全 ctx131k, 库=egolife第一人称) ===" >> "$SUM"
echo "run1(main) = 64.67% (97/150)" >> "$SUM"
grep 'RESULT\[run2\]' "$SUM" | tail -1 | sed 's/.*RESULT/run2      =/;s/ =\]/ /' >> "$SUM"
grep 'RESULT\[run3\]' "$SUM" | tail -1 | sed 's/.*RESULT/run3      =/;s/ =\]/ /' >> "$SUM"
echo "paper     = 66.67% (100/150)" >> "$SUM"
echo "=== done $(date '+%F %T') ===" >> "$SUM"
tail -20 "$SUM"
