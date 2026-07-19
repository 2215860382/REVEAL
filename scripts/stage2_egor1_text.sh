#!/usr/bin/env bash
# 文本主线: Stage2概括(text-only单库→egolife_fp) + egor1 pool_text(150q) + 对比66%。
# BGE放cuda:1(与vLLM同卡GPU1的余量), GPU0留给并行的视觉建库。8003=Qwen(GPU1)。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml
PY=/home2/ycj/miniconda3/envs/veil/bin/python
API=http://127.0.0.1:8003
BGE=/home2/ycj/Models/BAAI/bge-m3
LOG=outputs/logs/egor1_fp2; mkdir -p "$LOG"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG/text_orch.log"; }

require_8003(){ curl -s -m5 http://127.0.0.1:8003/v1/models 2>/dev/null | grep -q Qwen3.5-27B; }

require_8003 || { say "❌ 8003不可用, 停下等你"; exit 1; }

# 1) Stage2 multilayer (text-only单库→egolife_fp; A1/A5有meta会skip, 建A2/A3/A4/A6)
say "===== Stage2 multilayer (text-only → egolife_fp) ====="
$PY -m src.build_memory.build_multilayer \
  --src-memory-dir outputs/memory/egolife_fp_single \
  --out-memory-dir outputs/memory/egolife \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --bge-model "$BGE" --bge-gpu cuda:1 \
  --frame-mode hardlink --workers 1 --summary-concurrency 24 > "$LOG/multilayer_text.log" 2>&1
say "Stage2 rc=$?  egolife_fp覆盖: $(ls outputs/memory/egolife|grep -cE '^A[0-9]')/6"

# 2) egor1 pool_text (150q, 纯文本kf0)
require_8003 || { say "❌ 8003挂了, 停"; exit 1; }
say "===== eval egor1 pool_text (kf0, rubric_pool) ====="
$PY experiments/tuning/reveal_27b.py --config configs/egor1.yaml \
  --memory-dir outputs/memory/egolife --bge-gpu cuda:1 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers 8 \
  --pipeline-name egor1_fp_pool_text \
  --out outputs/results/egor1/egor1_fp_pool_text.jsonl > "$LOG/eval_egor1_pool_text.log" 2>&1
say "eval rc=$?"

# 3) 算分对比旧库66.0%
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG/text_orch.log"
import json
from collections import defaultdict
res=[r for r in (json.loads(l) for l in open('outputs/results/egor1/egor1_fp_pool_text.jsonl') if l.strip()) if 'correct' in r]
c=sum(r['correct'] is True for r in res)
print(f"\n[egor1 第一人称新库 pool_text] {c}/{len(res)}={c/len(res)*100:.1f}%  (旧第三人称库 99/150=66.0%)")
bt=defaultdict(lambda:[0,0])
for r in res: t=r.get('question_type','?'); bt[t][1]+=1; bt[t][0]+=(r['correct'] is True)
for t,(cc,nn) in sorted(bt.items()): print(f"    {t:16s} {cc}/{nn}={cc/nn*100:.0f}%")
PYEOF
say "===== TEXT MAINLINE DONE ====="
