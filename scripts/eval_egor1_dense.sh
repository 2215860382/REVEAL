#!/usr/bin/env bash
# egor1 喂帧臂(Arm B, 改进版): WorldMM式密集帧。
# REVEAL_DENSE_FRAMES=N → 每个检索到的chunk回原30s mp4采N帧(非1张存好的关键帧)。
# --answer-keyframe-k = 总帧上限(24=top-3段各8帧密集)。对比文本基线64.7%。
# 单库egolife(第一人称+视觉), rubric_pool主线。8003=Qwen(GPU5)。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml
export REVEAL_DENSE_FRAMES="${DENSE_N:-8}"          # 每段采帧数
export REVEAL_EGOLIFE_ROOT=/home2/ycj/Project/WorldMM/data/EgoLife
PY=/home2/ycj/miniconda3/envs/veil/bin/python
API=http://127.0.0.1:8003
KF="${KFCAP:-24}"                                   # 总帧上限
NAME="${NAME:-egor1_fp_dense${REVEAL_DENSE_FRAMES}_kf${KF}}"
OUT="outputs/results/egor1/${NAME}.jsonl"
LOG=outputs/logs/egor1_fp2; mkdir -p "$LOG"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG/dense_orch.log"; }

curl -s -m5 $API/v1/models 2>/dev/null | grep -q Qwen3.5-27B || { say "❌ 8003不可用, 停"; exit 1; }
say "===== egor1 喂帧臂(密集帧 N=$REVEAL_DENSE_FRAMES, 总帧cap=$KF) ====="
$PY experiments/tuning/reveal_27b.py --config configs/egor1.yaml \
  --memory-dir outputs/memory/egolife --bge-gpu cuda:1 --siglip-gpu cuda:1 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --answer-keyframe-k "$KF" --workers "${WORKERS:-8}" \
  --pipeline-name "$NAME" --out "$OUT" > "$LOG/${NAME}.log" 2>&1
say "eval rc=$?"

$PY - "$OUT" <<'PYEOF' 2>&1 | tee -a "$LOG/dense_orch.log"
import json,sys
from collections import defaultdict
f=sys.argv[1]
res=[r for r in (json.loads(l) for l in open(f) if l.strip()) if 'correct' in r]
c=sum(r['correct'] is True for r in res)
print(f"\n[egor1 喂帧臂-密集帧] {c}/{len(res)}={c/len(res)*100:.1f}%  (文本基线 97/150=64.7%, 旧散点kf 减分)")
bt=defaultdict(lambda:[0,0])
for r in res: t=r.get('question_type','?'); bt[t][1]+=1; bt[t][0]+=(r['correct'] is True)
for t,(cc,nn) in sorted(bt.items()): print(f"    {t:16s} {cc}/{nn}={cc/nn*100:.0f}%")
PYEOF
say "===== DENSE FRAMES ARM DONE ====="
