#!/usr/bin/env bash
# 重跑 egor1(Ego-R1-Bench 150题, 第一人称 egolife 库), vLLM 上下文 131072(≥130k)。
#   * GPU5->8001, GPU6->8004, 都 max-model-len 131072; 起后实测 /v1/models 确认 ≥130000。
#   * embed 9000 (GPU1)。15min 空转自杀看门狗守 vLLM。
#   * memory-dir=outputs/memory/egolife(第一人称), rubric=rubric_library(=rubric_pool同sha)。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SUM=outputs/logs/rerun_egor1_130k.log
: > "$SUM"; echo "=== egor1 rerun (ctx 131072) start $(date '+%F %T') ===" >> "$SUM"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
       -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

# ── 1. 起 vLLM(缺谁起谁), MAXLEN=131072 ──────────────────────────────
np curl -s -m5 http://127.0.0.1:8001/v1/models 2>/dev/null|grep -q Qwen || GPU=5 PORT=8001 MAXLEN=131072 bash scripts/start_vllm_27b.sh &
np curl -s -m5 http://127.0.0.1:8004/v1/models 2>/dev/null|grep -q Qwen || GPU=6 PORT=8004 MAXLEN=131072 bash scripts/start_vllm_27b.sh &
echo "$(date '+%T') launched vLLM 8001@GPU5 8004@GPU6 (MAXLEN=131072)" >> "$SUM"
# 等两端点就绪
while true; do
  v=0; for p in 8001 8004; do np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null|grep -q Qwen && v=$((v+1)); done
  echo "$(date '+%T') vLLM up $v/2" >> "$SUM"; [ "$v" = 2 ] && break; sleep 20
done
# 实测 max_model_len 是否 ≥130000
for p in 8001 8004; do
  ml=$(np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null | "$PY" -c 'import sys,json;d=json.load(sys.stdin);print(d["data"][0].get("max_model_len"))' 2>/dev/null)
  echo "$(date '+%T') port $p max_model_len=$ml" >> "$SUM"
  if [ -z "$ml" ] || [ "$ml" -lt 130000 ] 2>/dev/null; then
    echo "$(date '+%T') FATAL port $p ctx<130k ($ml), abort" >> "$SUM"; exit 2
  fi
done
echo "$(date '+%T') ctx>=130k OK on both endpoints" >> "$SUM"

# ── 2. 起 embed 9000 ────────────────────────────────────────────────
np curl -s -m5 http://127.0.0.1:9000/health 2>/dev/null|grep -q '"bge":true' || \
  setsid env -u http_proxy -u https_proxy NO_PROXY='*' HF_HUB_OFFLINE=1 \
    "$PY" scripts/start_embed_server_live.py --port 9000 --device cuda:1 --idle-timeout 86400 \
    > outputs/logs/embed_9000.log 2>&1 &
for i in $(seq 1 30); do np curl -s -m4 http://127.0.0.1:9000/health 2>/dev/null|grep -q '"bge":true' && break; sleep 5; done
echo "$(date '+%T') embed 9000 ready" >> "$SUM"

# ── 3. 15min 空转自杀看门狗(无 eval 进程即 kill vLLM 退出) ──────────
( idle=0
  while true; do
    sleep 60
    if pgrep -f 'reveal_27b.py' >/dev/null; then idle=0; else idle=$((idle+1)); fi
    if [ "$idle" -ge 15 ]; then
      for pid in $(for d in /proc/[0-9]*; do [ "$(cat $d/comm 2>/dev/null)" = python ] || continue; tr '\0' ' ' <$d/cmdline 2>/dev/null|grep -q 'vllm.entrypoints' && echo ${d#/proc/}; done); do kill "$pid" 2>/dev/null; done
      echo "$(date '+%T') idle 15min -> vLLM killed, reaper exit" >> "$SUM"; exit 0
    fi
  done ) &
REAPER=$!
echo "$(date '+%T') idle-reaper pid=$REAPER" >> "$SUM"

# ── 4. 跑 egor1 eval ────────────────────────────────────────────────
out=outputs/results/egor1/repro/main.jsonl; rm -f "$out"
echo "$(date '+%T') START egor1 eval" >> "$SUM"
np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
  "$PY" experiments/reveal_27b.py \
    --config configs/egor1.yaml --memory-dir outputs/memory/egolife \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 32 --pipeline-name repro_egor1_130k --out "$out" \
    > outputs/logs/repro_egor1_130k.log 2>&1 < /dev/null
ov=$("$PY" - "$out" <<'PY'
import json,sys
t=c=z=0
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("record_type"): continue
    t+=1; c+=bool(d.get("correct"))
    k=d.get("n_keyframes_to_answer");  z+= (k==0) if k is not None else 0
print(f"{100*c/t:.2f}% ({c}/{t})  0frame={z}" if t else "NO DATA")
PY
)
echo "$(date '+%T') DONE egor1 (ctx131k) -> $ov   (paper 66.7, 旧64.0)" >> "$SUM"
echo "=== egor1 rerun done $(date '+%F %T') ===" >> "$SUM"
tail -20 "$SUM"
