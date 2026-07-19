#!/usr/bin/env bash
# REVEAL egolife A1_JAKE 用【官方第一人称30sec对话当L1】重建 + eval, 对比REVEAL当前:
#   纯文本主线 egolife_pool_t07_text = 60.0%   带SigLIP视觉 = 59.2%
# Stage1第一人称模式不调VLM(narrative=官方对话1:1按clip文件名join), 只SigLIP+BGE+抽帧。
# 复用WorldMM的8003(Qwen3.5-27B, limit-mm256, max-model-len131072 已设对)。GPU: cuda:0。
# 注意: 原eval脚本bge-gpu cuda:6(GPU6故障)已改cuda:0。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=.
export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
PY=/home2/ycj/miniconda3/envs/veil/bin/python
API=http://127.0.0.1:8003
SIGLIP=/home2/ycj/Models/google/siglip-large-patch16-384
BGE=/home2/ycj/Models/BAAI/bge-m3
LOG=outputs/logs/egolife_fp; mkdir -p "$LOG"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG/orch.log"; }

# 8003 ready检查(curl带auth, 避urllib缺header/代理坑)
curl -s -m5 -H "Authorization: Bearer EMPTY" "$API/v1/models" 2>/dev/null | grep -q Qwen3.5-27B \
  || { say "8003不可用, 退出"; exit 1; }

# ---------- Stage1: 第一人称L1建库(不调VLM) ----------
if [ -f outputs/memory/egolife_fp_single/A1_JAKE/narrative.json ]; then
  say "Stage1 skip(已建)"
else
  say "===== Stage1: 第一人称L1建库 A1_JAKE (narrative=官方对话, 不调VLM) ====="
  $PY -m src.build_memory.build_single_similarity \
    --benchmark egolife --people A1_JAKE \
    --narrative-source firstperson \
    --firstperson-cap-dir /home2/ycj/Project/WorldMM/data/EgoLife/EgoLifeCap \
    --siglip-model "$SIGLIP" --siglip-gpu cuda:0 --bge-gpu cuda:0 \
    --api-url "$API" --api-model Qwen3.5-27B \
    --fps 1.0 --extract-gpu 0 --no-persist-frames --prep-workers 8 \
    --out-dir outputs/memory/egolife_fp_single > "$LOG/stage1.log" 2>&1
  say "Stage1 rc=$? ($(grep -c narrative outputs/memory/egolife_fp_single/A1_JAKE/narrative.json 2>/dev/null && echo built))"
fi

# ---------- Stage2: L2/L3 multilayer ----------
if [ -d outputs/memory/egolife/A1_JAKE ]; then
  say "Stage2 skip(已建)"
else
  say "===== Stage2: multilayer 摘要 (调8003 LLM) ====="
  $PY -m src.build_memory.build_multilayer \
    --src-memory-dir outputs/memory/egolife_fp_single \
    --out-memory-dir outputs/memory/egolife \
    --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
    --bge-model "$BGE" --bge-gpu cuda:0 \
    --frame-mode hardlink --workers 1 --summary-concurrency 24 > "$LOG/stage2.log" 2>&1
  say "Stage2 rc=$?"
fi

# ---------- eval 纯文本(keyframe-k 0) vs REVEAL文本60.0% ----------
run_eval(){ # name kf outfile
  local name=$1 kf=$2 out=$3
  [ -f "$out" ] && { say "eval $name skip"; return; }
  say "===== eval $name (keyframe-k $kf) ====="
  $PY experiments/tuning/reveal_27b.py --config configs/egolife.yaml \
    --memory-dir outputs/memory/egolife --bge-gpu cuda:0 --siglip-gpu cuda:0 \
    --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
    --max-iter 3 --answer-keyframe-k "$kf" --workers 8 \
    --pipeline-name "$name" --out "$out" > "$LOG/eval_${name}.log" 2>&1
  say "eval $name rc=$?"
}
run_eval egolife_fp_text 0 outputs/results/egolife/egolife_fp_text.jsonl
run_eval egolife_fp_img  8 outputs/results/egolife/egolife_fp_img.jsonl

# ---------- 对比 ----------
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG/orch.log"
import json
def acc(f):
    res=[json.loads(l) for l in open(f) if l.strip()]
    res=[r for r in res if 'correct' in r]
    c=sum(bool(r['correct']) for r in res); return c,len(res)
print("\n===== REVEAL 第一人称L1 结果 =====")
for tag,f,base in [("第一人称L1+纯文本","outputs/results/egolife/egolife_fp_text.jsonl","REVEAL原文本 300/500=60.0%"),
                   ("第一人称L1+带图 kf8","outputs/results/egolife/egolife_fp_img.jsonl","REVEAL原SigLIP 296/500=59.2%")]:
    try:
        c,n=acc(f); print(f"[{tag}] {c}/{n} = {c/n:.4f}   (基准 {base})")
    except Exception as e: print(f"[{tag}] 读取失败: {e}")
PYEOF
say "===== REVEAL第一人称 ALL DONE ====="
