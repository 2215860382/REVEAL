#!/usr/bin/env bash
# egolife第一人称库 用rubric_pool.yaml(对齐基准60%/59.2%)重跑eval, 看第一人称L1真实增益。
# 库已建好(egolife_fp), 只eval。REVEAL_RUBRIC_PATH=pool是关键对齐变量。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml   # 对齐基准的pool方法
PY=/home2/ycj/miniconda3/envs/veil/bin/python
API=http://127.0.0.1:8003
LOG=outputs/logs/egolife_fp; mkdir -p "$LOG"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG/pool_orch.log"; }
run_eval(){ local name=$1 kf=$2 out=$3
  [ -f "$out" ] && { say "$name skip"; return; }
  say "eval $name (kf$kf, rubric_pool)"
  $PY experiments/tuning/reveal_27b.py --config configs/egolife.yaml \
    --memory-dir outputs/memory/egolife --bge-gpu cuda:0 --siglip-gpu cuda:0 \
    --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
    --max-iter 3 --answer-keyframe-k "$kf" --workers 8 \
    --pipeline-name "$name" --out "$out" > "$LOG/${name}.log" 2>&1
  say "$name rc=$?"
}
say "===== egolife第一人称 pool-rubric eval(对齐基准) ====="
run_eval egolife_fp_pool_text 0 outputs/results/egolife/egolife_fp_pool_text.jsonl
run_eval egolife_fp_pool_img  8 outputs/results/egolife/egolife_fp_pool_img.jsonl
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG/pool_orch.log"
import json
def acc(f):
    res=[r for r in (json.loads(l) for l in open(f) if l.strip()) if 'correct' in r]
    c=sum(r['correct'] is True for r in res); return c,len(res)
print("\n===== egolife第一人称L1 (pool-rubric对齐) =====")
for tag,f,base in [("第一人称L1+文本","outputs/results/egolife/egolife_fp_pool_text.jsonl","原pool文本300/500=60.0%"),
                   ("第一人称L1+图kf8","outputs/results/egolife/egolife_fp_pool_img.jsonl","原pool SigLIP296/500=59.2%")]:
    try: c,n=acc(f); print(f"[{tag}] {c}/{n}={c/n:.4f}  (基准{base}, 同rubric_pool)")
    except Exception as e: print(f"[{tag}] {e}")
PYEOF
say "===== POOL EVAL DONE ====="
