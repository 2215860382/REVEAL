#!/usr/bin/env bash
# egor1(Ego-R1-Bench 150q = A1-A6各25) 用第一人称L1新库(egolife_fp)跑, 对比旧库pool_text 66.0%。
# 现状: A1已完整; A2-A4 single已建(缺multilayer); A5-A6全空。
# 流程: (1)建A5/A6 single[GPU0,不调VLM,与egolife eval并行]
#       (2)等egolife pool_img eval完腾8003
#       (3)Stage2 multilayer 把A2-A6并入egolife_fp
#       (4)eval egor1: pool_text(kf0=默认) + pool_img(kf8带图), 6人库150q
#       (5)算分对比66% → 停vllm
# rubric_pool = 现默认; 带图用8003现配置(max-model-len 131072 + image:256)。不用Monitor, nohup自跑。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml
PY=/home2/ycj/miniconda3/envs/veil/bin/python
API=http://127.0.0.1:8003,http://127.0.0.1:8006   # 双端点(8003 GPU1 + 8006 GPU0), LLMClient逗号分流
SIGLIP=/home2/ycj/Models/google/siglip-large-patch16-384
BGE=/home2/ycj/Models/BAAI/bge-m3
LOG=outputs/logs/egor1_fp2; mkdir -p "$LOG"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG/orch.log"; }

# 用户要求: 挂了就停等你, 不自动重起vllm(配额被踢=优雅shutdown, 不是技术崩)。
ensure_8003(){
  curl -s -m5 http://127.0.0.1:8003/v1/models 2>/dev/null | grep -q Qwen3.5-27B && return 0
  say "❌ 8003不可用。已设为'挂了就停等你'——不自动重起vllm。请解禁配额后起8003(GPU1建议: env -u VLLM_API_KEY OTEL_SDK_DISABLED=true CUDA_VISIBLE_DEVICES=1 ... --enforce-eager), 再重跑本脚本(A5/A6建库+egolife已完成会自动skip, 直接进Stage2→egor1)。"
  return 1
}

# ---- (1) 建A5/A6 single: GPU0, 第一人称不调VLM, 与egolife eval并行 ----
if [ -f outputs/memory/egolife_fp_single/A6_SHURE/narrative.json ]; then
  say "A5/A6 single已建, skip"
else
  say "===== 建A5/A6第一人称single (SigLIP+BGE cuda:0, 不调VLM) ====="
  $PY -m src.build_memory.build_single_similarity \
    --benchmark egolife --people A5_KATRINA A6_SHURE \
    --narrative-source firstperson \
    --firstperson-cap-dir /home2/ycj/Project/WorldMM/data/EgoLife/EgoLifeCap \
    --siglip-model "$SIGLIP" --siglip-gpu cuda:0 --bge-gpu cuda:0 \
    --api-url "$API" --api-model Qwen3.5-27B \
    --fps 1.0 --extract-gpu 0 --no-persist-frames --prep-workers 8 \
    --out-dir outputs/memory/egolife_fp_single > "$LOG/build_a5a6.log" 2>&1
  say "建A5/A6 rc=$?"
fi

# ---- (2) 等egolife pool_img eval完成, 腾8003 (gate用真在跑的pool_img, 非死掉的fp_img) ----
say "等egolife pool_img eval完成(腾8003)..."
until [ "$(grep -c '\"correct\"' outputs/results/egolife/egolife_fp_pool_img.jsonl 2>/dev/null || echo 0)" -ge 500 ]; do sleep 120; done
say "egolife pool_img完成, 继续egor1链"

# ---- (3) Stage2 multilayer: A2-A6并入egolife_fp (A1已有会skip) ----
ensure_8003 || exit 1
say "===== Stage2 multilayer (A2-A6 → egolife_fp) ====="
$PY -m src.build_memory.build_multilayer \
  --src-memory-dir outputs/memory/egolife_fp_single \
  --out-memory-dir outputs/memory/egolife \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --bge-model "$BGE" --bge-gpu cuda:0 \
  --frame-mode hardlink --workers 1 --summary-concurrency 48 > "$LOG/multilayer.log" 2>&1
say "Stage2 rc=$?"
say "egolife_fp覆盖人数: $(ls outputs/memory/egolife | grep -cE '^A[0-9]')/6"

# ---- (4) eval egor1: pool_text(kf0) + pool_img(kf8), 6人库150q ----
run_egor1(){ local name=$1 kf=$2 out=$3
  [ -f "$out" ] && [ "$(grep -c '\"correct\"' "$out" 2>/dev/null)" -ge 150 ] && { say "$name 已完成 skip"; return; }
  ensure_8003 || return 1
  say "eval $name (kf$kf, rubric_pool)"
  $PY experiments/tuning/reveal_27b.py --config configs/egor1.yaml \
    --memory-dir outputs/memory/egolife --bge-gpu cuda:0 --siglip-gpu cuda:0 \
    --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
    --max-iter 3 --answer-keyframe-k "$kf" --workers 16 \
    --pipeline-name "$name" --out "$out" > "$LOG/${name}.log" 2>&1
  say "$name rc=$?"
}
run_egor1 egor1_fp_pool_text 0 outputs/results/egor1/egor1_fp_pool_text.jsonl
run_egor1 egor1_fp_pool_img  8 outputs/results/egor1/egor1_fp_pool_img.jsonl

# ---- (5) 算分对比66% ----
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG/orch.log"
import json
from collections import defaultdict
def show(tag,f,base):
    try:
        res=[r for r in (json.loads(l) for l in open(f) if l.strip()) if 'correct' in r]
        c=sum(r['correct'] is True for r in res)
        print(f"[{tag}] {c}/{len(res)}={c/len(res)*100:.1f}%  (基准 {base})")
        bt=defaultdict(lambda:[0,0])
        for r in res: t=r.get('question_type','?'); bt[t][1]+=1; bt[t][0]+=(r['correct'] is True)
        for t,(cc,nn) in sorted(bt.items()): print(f"    {t:16s} {cc}/{nn}={cc/nn*100:.0f}%")
    except Exception as e: print(f"[{tag}] {e}")
print("\n===== egor1 第一人称L1新库 (rubric_pool) =====")
show("egor1 第一人称+文本kf0","outputs/results/egor1/egor1_fp_pool_text.jsonl","旧库pool_text 99/150=66.0%")
show("egor1 第一人称+带图kf8","outputs/results/egor1/egor1_fp_pool_img.jsonl","旧库pool_text 99/150=66.0%")
PYEOF

# ---- (6) 停vllm ----
say "===== 全部完成, 停8003 vllm ====="
for pid in $(pgrep -f "vllm.entrypoints.openai.api_server"); do kill $pid 2>/dev/null && say "killed vllm $pid"; done
say "===== EGOR1 FP2 ALL DONE ====="
