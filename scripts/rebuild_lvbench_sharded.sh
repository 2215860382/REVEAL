#!/usr/bin/env bash
# 分片并行重建 LVBench 的 58 个坏视频,再重评。
#   * 5 个 build_single_similarity 进程,各 --filter-from shardK.jsonl(视频不重叠),
#     SigLIP/BGE 分散到空闲 GPU 0/2/3/4(GPU1=embed, GPU5/6=vLLM),都打 8001/8004。
#   * skip 逻辑:narrative.json+vectors.npz 双在才跳过 → 已建的 28 个好库不重复。
#   * 等到 86 个"好 caption"库(badfrac<=0.05)→ 重评 LVBench。
# vLLM 存活由 scripts/vllm_idle_reaper.sh 监督(挂了自动拉回,防再次中毒)。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY_BUILD=/home2/ycj/miniconda3/envs/reveal/bin/python
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SHARD_DIR=outputs/tmp/lvb_shards
SUM=outputs/logs/rebuild_lvbench_sharded.log
GPUS=(0 2 3 4 0)                       # shard k 的 SigLIP/BGE 物理 GPU
: > "$SUM"; echo "=== sharded rebuild start $(date '+%F %T') ===" >> "$SUM"
noproxy(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
           -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

# 好库计数:86 个多层库都存在且坏 caption 比例<=5%
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

# ── 1. 启动 5 个分片建库 ───────────────────────────────────────────────
for k in 0 1 2 3 4; do
  gpu=${GPUS[$k]}
  log=/home2/ycj/reveal_lvb_shard${k}.log
  echo "$(date '+%T') launch shard$k on GPU$gpu (filter=$SHARD_DIR/shard$k.jsonl)" >> "$SUM"
  setsid env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    NO_PROXY='*' CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    PYTHONPATH=/home2/ycj/Project/REVEAL \
    "$PY_BUILD" -u -m src.build_memory.build_single_similarity \
      --config configs/lvbench.yaml \
      --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
      --siglip-gpu cuda:0 --bge-gpu cuda:0 \
      --api-url "$EP" --api-model Qwen3.5-27B \
      --theta 0.60 --n-max 30 --emit-multilayer \
      --out-dir outputs/memory/lvbench \
      --filter-from "$SHARD_DIR/shard$k.jsonl" \
      > "$log" 2>&1 < /dev/null &
  sleep 3
done
echo "$(date '+%T') 5 shards launched" >> "$SUM"

# ── 2. 等 86 个好库(或所有分片进程结束) ──────────────────────────────
while true; do
  g=$(count_good 2>/dev/null || echo 0)
  alive=$(pgrep -f 'src.build_memory.build_single_similarity' | wc -l)
  echo "$(date '+%T')  good_banks=$g/86  shards_alive=$alive" >> "$SUM"
  [ "$g" -ge 86 ] 2>/dev/null && { echo "BANK COMPLETE ($g/86)" >> "$SUM"; break; }
  if [ "$alive" -eq 0 ] 2>/dev/null; then
    sleep 20; g=$(count_good 2>/dev/null || echo 0)
    [ "$g" -ge 86 ] 2>/dev/null && { echo "BANK COMPLETE ($g/86)" >> "$SUM"; break; }
    echo "ABORT: all shards exited at $g/86 good" >> "$SUM"; exit 1
  fi
  sleep 120
done

# ── 3. 确保 embed 9000 在(重评需要) ─────────────────────────────────
noproxy curl -s -m5 http://127.0.0.1:9000/health 2>/dev/null | grep -q '"bge":true' \
  || echo "WARN embed 9000 not ready before eval" >> "$SUM"

# ── 4. 重评 LVBench ─────────────────────────────────────────────────
out=outputs/results/lvbench/repro/main.jsonl
rm -f "$out"
echo "$(date '+%T') START lvbench re-eval" >> "$SUM"
noproxy HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
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
echo "$(date '+%T') DONE lvbench re-eval -> $ov  (paper 65.9)" >> "$SUM"
echo "=== sharded rebuild done $(date '+%F %T') ===" >> "$SUM"
tail -15 "$SUM"
