#!/usr/bin/env bash
# LVBench 2×2 消融:分辨率{448,1344} × θ{0.6,0.8},分组模式固定 anchor。
# 已有 448+0.6 = 60.54。本脚本按 3 阶段跑另外三格,复用帧省时间:
#   Phase1  1344 + θ0.6 : 复用现有 0.6 anchor 的 frames_raw,只重打 1344 caption
#   Phase2  448  + θ0.8 : 删帧,重抽+按 θ0.8 anchor 重分组,caption 448
#   Phase3  1344 + θ0.8 : 复用 Phase2 的 0.8 帧,只重打 1344 caption
# 每阶段:重置库 → 6 分片建库(全 GPU0)→ 等 86 好库 → 重评 → 存分。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY_BUILD=/home2/ycj/miniconda3/envs/reveal/bin/python
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SHARD_DIR=outputs/tmp/lvb_shards1344      # 已含全 86 视频的 6 分片
SUM=outputs/logs/lvbench_2x2.log
: > "$SUM"; echo "=== 2x2 ablation start $(date '+%F %T') ===" >> "$SUM"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
       -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

ensure_vllm(){
  local v p
  while true; do
    v=0; for p in 8001 8004; do np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null|grep -q Qwen && v=$((v+1)); done
    [ "$v" = 2 ] && return 0
    # 起缺的端点(GPU5->8001, GPU6->8004)
    np curl -s -m5 http://127.0.0.1:8001/v1/models 2>/dev/null|grep -q Qwen || GPU=5 PORT=8001 MAXLEN=131072 bash scripts/start_vllm_27b.sh &
    np curl -s -m5 http://127.0.0.1:8004/v1/models 2>/dev/null|grep -q Qwen || GPU=6 PORT=8004 MAXLEN=131072 bash scripts/start_vllm_27b.sh &
    echo "$(date '+%T') waiting vLLM ($v/2 up)" >> "$SUM"; sleep 20
  done
}
ensure_embed(){
  np curl -s -m5 http://127.0.0.1:9000/health 2>/dev/null | grep -q '"bge":true' && return 0
  setsid env -u http_proxy -u https_proxy NO_PROXY='*' HF_HUB_OFFLINE=1 \
    "$PY" scripts/start_embed_server_live.py --port 9000 --device cuda:1 --idle-timeout 86400 \
    > outputs/logs/embed_9000.log 2>&1 &
  for i in $(seq 1 30); do np curl -s -m4 http://127.0.0.1:9000/health 2>/dev/null|grep -q '"bge":true' && return 0; sleep 10; done
}
count_good(){
  "$PY" - <<'PY'
import json,glob,os
ok=0
for f in glob.glob("outputs/memory/lvbench_ml3_n12/*/L1.jsonl"):
    d=os.path.dirname(f)
    if not all(os.path.exists(os.path.join(d,x)) for x in ("L2.jsonl","L3.jsonl","meta.json")): continue
    bad=nc=0
    for l in open(f):
        t=(json.loads(l).get("text") or "").strip(); nc+=1
        if ("narrative failed" in t) or ("Cannot connect" in t) or ("Connect call failed" in t): bad+=1
    if nc>0 and bad/nc<=0.05: ok+=1
print(ok)
PY
}

# build_phase THETA CAPTION_SIDE
build_phase(){
  local theta="$1" side="$2"
  ensure_vllm; ensure_embed
  rm -rf outputs/memory/lvbench_ml3_n12/*
  for k in 0 1 2 3 4 5; do
    setsid env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
      NO_PROXY='*' CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 REVEAL_CAPTION_MAX_SIDE="$side" \
      PYTHONPATH=/home2/ycj/Project/REVEAL \
      "$PY_BUILD" -u -m src.build_memory.build_single_similarity \
        --config configs/lvbench.yaml \
        --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
        --siglip-gpu cuda:0 --bge-gpu cuda:0 \
        --api-url "$EP" --api-model Qwen3.5-27B \
        --theta "$theta" --n-max 30 --emit-multilayer \
        --out-dir outputs/memory/lvbench \
        --filter-from "$SHARD_DIR/shard$k.jsonl" \
        > /home2/ycj/reveal_lvb2x2_shard${k}.log 2>&1 < /dev/null &
    sleep 3
  done
  echo "$(date '+%T')   6 shards launched (theta=$theta side=$side)" >> "$SUM"
  while true; do
    g=$(count_good 2>/dev/null || echo 0)
    alive=$(pgrep -f 'src.build_memory.build_single_similarity' | wc -l)
    echo "$(date '+%T')   good=$g/86 alive=$alive" >> "$SUM"
    [ "$g" -ge 86 ] 2>/dev/null && { echo "$(date '+%T')   BANK COMPLETE" >> "$SUM"; break; }
    if [ "$alive" -eq 0 ] 2>/dev/null; then
      sleep 20; g=$(count_good 2>/dev/null||echo 0)
      [ "$g" -ge 86 ] 2>/dev/null && break
      echo "$(date '+%T')   ABORT build died at $g/86" >> "$SUM"; return 1
    fi
    sleep 120
  done
}

# eval_phase TAG
eval_phase(){
  local tag="$1"
  ensure_vllm; ensure_embed
  local out=outputs/results/lvbench/repro/main_${tag}.jsonl
  rm -f "$out"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
    "$PY" experiments/reveal_27b.py \
      --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
      --embed-api-url http://127.0.0.1:9000 \
      --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "repro_lvb_${tag}" --out "$out" \
      > outputs/logs/repro_lvb_${tag}.log 2>&1 < /dev/null
  local ov=$("$PY" - "$out" <<'PY'
import json,sys
t=c=0
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("record_type"): continue
    t+=1; c+= 1 if d.get("correct") else 0
print(f"{100*c/t:.2f}% ({c}/{t})" if t else "NO DATA")
PY
)
  echo "$(date '+%T') RESULT[$tag] = $ov" >> "$SUM"
}

# ---- Phase 1: 1344 + θ0.6 (复用现有 0.6 frames_raw,删 caption/多层) ----
echo "$(date '+%T') PHASE1 1344+0.6 (reuse 0.6 frames)" >> "$SUM"
for d in outputs/memory/lvbench/*/; do rm -f "$d/narrative.json" "$d/vectors.npz"; done
build_phase 0.60 1344 && eval_phase 1344_06

# ---- Phase 2: 448 + θ0.8 (全删重抽+按0.8重分组) ----
echo "$(date '+%T') PHASE2 448+0.8 (fresh extract+group@0.8)" >> "$SUM"
for d in outputs/memory/lvbench/*/; do rm -rf "$d/frames_raw" "$d/keyframes_origin" "$d/keyframes_resized" "$d/narrative.json" "$d/vectors.npz"; done
build_phase 0.80 448 && eval_phase 448_08

# ---- Phase 3: 1344 + θ0.8 (复用Phase2的0.8帧,只重打caption) ----
echo "$(date '+%T') PHASE3 1344+0.8 (reuse 0.8 frames)" >> "$SUM"
for d in outputs/memory/lvbench/*/; do rm -f "$d/narrative.json" "$d/vectors.npz"; done
build_phase 0.80 1344 && eval_phase 1344_08

echo "" >> "$SUM"
echo "=== 2×2 结果 (mode=anchor 固定) ===" >> "$SUM"
echo "         θ=0.6            θ=0.8" >> "$SUM"
echo "448      60.54%(已有)     $(grep 'RESULT\[448_08\]' $SUM|tail -1|sed 's/.*= //')" >> "$SUM"
echo "1344     $(grep 'RESULT\[1344_06\]' $SUM|tail -1|sed 's/.*= //')   $(grep 'RESULT\[1344_08\]' $SUM|tail -1|sed 's/.*= //')" >> "$SUM"
echo "=== 2x2 ablation done $(date '+%F %T') ===" >> "$SUM"
tail -25 "$SUM"
