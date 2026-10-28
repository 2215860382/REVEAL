#!/usr/bin/env bash
# 用 caption 分辨率 1344 重建全 86 个 LVBench 库,再重评。
#   * 6 个 build_single_similarity 分片,全在 GPU0(唯一不被别的租户/集群回收的卡),
#     REVEAL_CAPTION_MAX_SIDE=1344(默认 448)。
#   * 79 个视频复用 frames_raw(原生全分辨率)→ 只重打 caption(快,θ=0.60 分块不变,
#     纯分辨率变量);7 个缺帧的完整重抽。
#   * 等 86 个好 caption 库(badfrac<=0.05)→ 重评 LVBench(目标 65.9)。
# vLLM 存活由 scripts/vllm_idle_reaper.sh 监督。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY_BUILD=/home2/ycj/miniconda3/envs/reveal/bin/python
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SHARD_DIR=outputs/tmp/lvb_shards1344
SUM=outputs/logs/rebuild_lvbench_1344.log
: > "$SUM"; echo "=== 1344 rebuild start $(date '+%F %T') ===" >> "$SUM"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
       -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

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

# 等 vLLM 就绪(避免 captioning 打向死端点又中毒)
echo "waiting vLLM..." >> "$SUM"
while true; do
  v=0; for p in 8001 8004; do np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null|grep -q Qwen && v=$((v+1)); done
  [ "$v" = 2 ] && { echo "$(date '+%T') vLLM UP" >> "$SUM"; break; }
  sleep 15
done

# 启动 6 分片(全 GPU0, 1344)
for k in 0 1 2 3 4 5; do
  echo "$(date '+%T') launch shard$k @GPU0 CAPTION=1344" >> "$SUM"
  setsid env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    NO_PROXY='*' CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 REVEAL_CAPTION_MAX_SIDE=1344 \
    PYTHONPATH=/home2/ycj/Project/REVEAL \
    "$PY_BUILD" -u -m src.build_memory.build_single_similarity \
      --config configs/lvbench.yaml \
      --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
      --siglip-gpu cuda:0 --bge-gpu cuda:0 \
      --api-url "$EP" --api-model Qwen3.5-27B \
      --theta 0.60 --n-max 30 --emit-multilayer \
      --out-dir outputs/memory/lvbench \
      --filter-from "$SHARD_DIR/shard$k.jsonl" \
      > /home2/ycj/reveal_lvb1344_shard${k}.log 2>&1 < /dev/null &
  sleep 3
done
echo "$(date '+%T') 6 shards launched" >> "$SUM"

# 等 86 好库
while true; do
  g=$(count_good 2>/dev/null || echo 0)
  alive=$(pgrep -f 'src.build_memory.build_single_similarity' | wc -l)
  echo "$(date '+%T')  good=$g/86  shards_alive=$alive" >> "$SUM"
  [ "$g" -ge 86 ] 2>/dev/null && { echo "BANK COMPLETE ($g/86)" >> "$SUM"; break; }
  if [ "$alive" -eq 0 ] 2>/dev/null; then
    sleep 20; g=$(count_good 2>/dev/null || echo 0)
    [ "$g" -ge 86 ] 2>/dev/null && { echo "BANK COMPLETE ($g/86)" >> "$SUM"; break; }
    echo "ABORT: all shards exited at $g/86" >> "$SUM"; exit 1
  fi
  sleep 120
done

# embed 确保在
np curl -s -m5 http://127.0.0.1:9000/health 2>/dev/null | grep -q '"bge":true' || echo "WARN embed down" >> "$SUM"

# 重评
out=outputs/results/lvbench/repro/main.jsonl; rm -f "$out"
echo "$(date '+%T') START eval" >> "$SUM"
np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
  "$PY" experiments/reveal_27b.py \
    --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 32 --pipeline-name repro_lvbench --out "$out" \
    > outputs/logs/repro_lvbench.log 2>&1 < /dev/null
ov=$("$PY" - <<PY
import json
t=c=0
for l in open("$out"):
    d=json.loads(l)
    if d.get("record_type"): continue
    t+=1; c+= 1 if d.get("correct") else 0
print(f"{100*c/t:.2f}% ({c}/{t})" if t else "NO DATA")
PY
)
echo "$(date '+%T') DONE 1344 eval -> $ov  (paper 65.9, 448版=60.54)" >> "$SUM"
echo "=== 1344 rebuild done $(date '+%F %T') ===" >> "$SUM"
tail -15 "$SUM"
