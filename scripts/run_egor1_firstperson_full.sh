#!/usr/bin/env bash
# REVEAL egor1(Ego-R1-Bench 150q=A1-A6各25) 用第一人称L1重测, 对比REVEAL原66.0%(egor1_pool_text)。
# A1的第一人称库已有(egolife_fp), 本脚本建A2-A6 → Stage2 → eval egor1(6人库,纯文本kf0) → 停vllm。
# 先等EgoLifeQA的img eval完(共用GPU0+8003)。含8003健康检查(第一人称建库不调VLM,防长空闲崩)。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
PY=/home2/ycj/miniconda3/envs/veil/bin/python
API=http://127.0.0.1:8003
SIGLIP=/home2/ycj/Models/google/siglip-large-patch16-384
BGE=/home2/ycj/Models/BAAI/bge-m3
LOG=outputs/logs/egor1_fp; mkdir -p "$LOG"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG/orch.log"; }

ensure_8003(){
  curl -s -m5 http://127.0.0.1:8003/v1/models 2>/dev/null | grep -q Qwen3.5-27B && return 0
  say "8003不可用, 重起(GPU1, noauth)"
  nohup setsid env -u VLLM_API_KEY CUDA_VISIBLE_DEVICES=1 $PY -m vllm.entrypoints.openai.api_server \
    --model /home2/ycj/Models/Qwen/Qwen3.5-27B --served-model-name Qwen3.5-27B \
    --host 127.0.0.1 --port 8003 --max-model-len 131072 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":256}' --dtype bfloat16 --max-num-seqs 32 --trust-remote-code \
    > "$LOG/vllm_8003.log" 2>&1 &
  disown
  for i in $(seq 1 40); do curl -s -m3 http://127.0.0.1:8003/v1/models 2>/dev/null | grep -q Qwen3.5-27B && break; sleep 10; done
}

# 0) 等EgoLifeQA的img eval完(共用GPU0)
say "等EgoLifeQA img eval完成(共用GPU0)..."
until [ "$(wc -l < outputs/results/egolife/egolife_fp_img.jsonl 2>/dev/null || echo 0)" -ge 500 ]; do sleep 120; done
say "EgoLifeQA完成, 开始egor1"

# 1) 建A2-A6第一人称库(不调VLM, A1已有)
if ls outputs/memory/egolife_fp_single/A6_SHURE/narrative.json >/dev/null 2>&1; then
  say "A2-A6库已建, skip"
else
  say "===== 建A2-A6第一人称库(SigLIP+BGE cuda:0, 不调VLM) ====="
  $PY -m src.build_memory.build_single_similarity \
    --benchmark egolife --people A2_ALICE A3_TASHA A4_LUCIA A5_KATRINA A6_SHURE \
    --narrative-source firstperson \
    --firstperson-cap-dir /home2/ycj/Project/WorldMM/data/EgoLife/EgoLifeCap \
    --siglip-model "$SIGLIP" --siglip-gpu cuda:0 --bge-gpu cuda:0 \
    --api-url "$API" --api-model Qwen3.5-27B \
    --fps 1.0 --extract-gpu 0 --no-persist-frames --prep-workers 8 \
    --out-dir outputs/memory/egolife_fp_single > "$LOG/build_a2a6.log" 2>&1
  say "建A2-A6 rc=$?"
fi

# 2) Stage2 multilayer(补A2-A6, A1已有)
ensure_8003
say "===== Stage2 multilayer ====="
$PY -m src.build_memory.build_multilayer \
  --src-memory-dir outputs/memory/egolife_fp_single \
  --out-memory-dir outputs/memory/egolife \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --bge-model "$BGE" --bge-gpu cuda:0 \
  --frame-mode hardlink --workers 1 --summary-concurrency 24 > "$LOG/multilayer.log" 2>&1
say "Stage2 rc=$?"

# 3) eval egor1(6人第一人称库, 纯文本kf0, 对比66.0%)
ensure_8003
say "===== eval egor1(6人第一人称, 纯文本kf0) ====="
$PY experiments/tuning/reveal_27b.py --config configs/egor1.yaml \
  --memory-dir outputs/memory/egolife --bge-gpu cuda:0 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers 8 \
  --pipeline-name egor1_fp_text \
  --out outputs/results/egor1/egor1_fp_text.jsonl > "$LOG/eval_egor1.log" 2>&1
say "eval egor1 rc=$?"

# 4) 算分对比66.0%
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG/orch.log"
import json
from collections import defaultdict
res=[r for r in (json.loads(l) for l in open('outputs/results/egor1/egor1_fp_text.jsonl') if l.strip()) if 'correct' in r]
c=sum(bool(r['correct']) for r in res)
print(f"[egor1 第一人称L1+纯文本] {c}/{len(res)}={c/len(res):.4f}  (REVEAL原egor1_pool_text=99/150=66.0%)")
bt=defaultdict(lambda:[0,0])
for r in res: t=r.get('question_type','?'); bt[t][1]+=1; bt[t][0]+=bool(r['correct'])
for t,(cc,nn) in sorted(bt.items()): print(f"    {t:16s} {cc}/{nn}={cc/nn*100:.0f}%")
PYEOF

# 5) 停vllm
say "===== 全部完成, 停vllm ====="
for pid in $(pgrep -f "vllm.entrypoints.openai.api_server"); do kill $pid 2>/dev/null && say "killed vllm $pid"; done
say "===== EGOR1 FP ALL DONE ====="
