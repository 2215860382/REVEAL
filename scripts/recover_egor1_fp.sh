#!/usr/bin/env bash
# 恢复: A2/A3/A4 single是半成品(向量<clip, 24整批倍数=编码中断)。A5/A6完整。
# A2 multilayer基于坏single建的也要重建。步骤:
#  1) resume补全A2/A3/A4 single(只编缺的~761 clip, GPU0, 不调vLLM)
#  2) 校验single向量==clip, 不一致则停
#  3) 删egolife_fp的A2/A3/A4(强制Stage2重建)
#  4) exec主链run_egor1_fp2.sh → Stage2(A2-A6双端点) + egor1 pool_text+pool_img
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
PY=/home2/ycj/miniconda3/envs/veil/bin/python
LOG=outputs/logs/egor1_fp2; mkdir -p "$LOG"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG/recover.log"; }

# 1) 补全A2/A3/A4 single (resume: 复用已编向量, 只补缺的clip)
say "===== 补全A2/A3/A4 single (resume缺失clip, SigLIP+BGE cuda:0, 不调vLLM) ====="
$PY -m src.build_memory.build_single_similarity \
  --benchmark egolife --people A2_ALICE A3_TASHA A4_LUCIA \
  --narrative-source firstperson \
  --firstperson-cap-dir /home2/ycj/Project/WorldMM/data/EgoLife/EgoLifeCap \
  --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 --siglip-gpu cuda:0 --bge-gpu cuda:0 \
  --api-url http://127.0.0.1:8003 --api-model Qwen3.5-27B \
  --fps 1.0 --extract-gpu 0 --no-persist-frames --prep-workers 8 \
  --out-dir outputs/memory/egolife_fp_single > "$LOG/single_complete_a2a4.log" 2>&1
say "single补全 rc=$?"

# 2) 校验single完整 (向量==clip)
$PY - <<'PYEOF'
import os,sys,numpy as np
base='outputs/memory/egolife_fp_single'; ok=True
for p in ['A2_ALICE','A3_TASHA','A4_LUCIA','A5_KATRINA','A6_SHURE']:
    z=np.load(os.path.join(base,p,'vectors.npz')); n=int(z['chunk_ids'].shape[0])
    c=len(os.listdir(os.path.join(base,p,'_clips')))
    print(f'  {p}: vec={n} clips={c} {"OK" if n==c else "MISMATCH"}')
    ok = ok and (n==c)
sys.exit(0 if ok else 1)
PYEOF
if [ $? -ne 0 ]; then say "❌ single仍不一致, 停下等你"; exit 1; fi
say "✅ A2-A6 single全部向量==clip"

# 3) 删不完整的multilayer, 强制Stage2重建
say "删egolife_fp/{A2,A3,A4}(基于坏single或中断, 重建)"
rm -rf outputs/memory/egolife/A2_ALICE outputs/memory/egolife/A3_TASHA outputs/memory/egolife/A4_LUCIA
say "egolife_fp现存: $(ls outputs/memory/egolife|grep -E '^A[0-9]'|tr '\n' ' ')"

# 4) 交给主链: Stage2(A2-A6双端点) + egor1 pool_text+pool_img
say "===== 转入 run_egor1_fp2.sh (Stage2 + egor1) ====="
exec bash scripts/run_egor1_fp2.sh
