#!/usr/bin/env bash
# Fix the 2 videos whose multilayer bank never got an L1.jsonl (single bank is
# complete — narrative.json + vectors.npz — but the --emit-multilayer step timed
# out, so only keyframes got copied). Their questions therefore came out BLANK:
#   videomme       qVZOKel-gpE   → sample_idx 843,844,845   (3 Qs)
#   longvideobench wxWo44MoCTI   → sample_idx 39,40         (2 Qs)
# lvbench is complete (86/86), 0 affected.
#
# Steps, serialized AFTER the overnight main-table+ablation finishes so the
# rebuild's caption load never collides with the paper eval on the shared vLLM
# endpoints (that collision is exactly what broke these banks originally):
#   1. wait for overnight completion marker
#   2. delete the half-built single+multilayer dirs for both videos (clean re-extract)
#   3. rebuild both (videomme via 1-video filter list; longvideobench via resume)
#   4. verify L1.jsonl now exists
#   5. drop the blank rows from the two main-table jsonls, rerun only those idxs
#      (same --out + --pipeline-name so the numbers stay canonical), report acc.
set -uo pipefail
cd /home2/ycj/Project/VEIL
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'
export TMPDIR=/dev/shm
export VEIL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml   # threshold 0.7 baked
export VEIL_VERIFIER_AGG=max

PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP=http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004
EMB=http://127.0.0.1:9000
OVLOG=outputs/logs/after_sweep_maintable.log
LOG=outputs/logs/fix_broken_banks.log
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[$(date +%F_%T)] fix_broken_banks START"

# ── 1. Wait for overnight to finish (max ~8h) ──
echo "[$(date +%F_%T)] waiting for overnight completion marker in $OVLOG ..."
for _ in $(seq 1 480); do
  grep -q 'OVERNIGHT MAIN TABLE + RUBRIC ABLATION COMPLETE' "$OVLOG" 2>/dev/null && break
  sleep 60
done
grep -q 'OVERNIGHT MAIN TABLE + RUBRIC ABLATION COMPLETE' "$OVLOG" 2>/dev/null \
  && echo "[$(date +%F_%T)] overnight done — proceeding" \
  || { echo "[$(date +%F_%T)] WARN overnight marker not seen after 8h; proceeding anyway"; }

# ── 2. Delete half-built dirs (both single and multilayer) ──
echo "[$(date +%F_%T)] removing half-built dirs"
rm -rf outputs/memory/videommeL_single_1344/qVZOKel-gpE \
       outputs/memory/videommeL_1344/qVZOKel-gpE \
       outputs/memory/longvideobench_single/wxWo44MoCTI \
       outputs/memory/longvideobench/wxWo44MoCTI

# assemble live endpoints (each once) for caption
API=""
_add(){ curl -s -m 4 "$1/v1/models" >/dev/null 2>&1 && API="${API:+$API,}$1"; }
for u in http://127.0.0.1:8006 http://127.0.0.1:8003 http://10.82.1.145:8002 http://10.82.1.145:8004; do _add "$u"; done
echo "[$(date +%F_%T)] build endpoints=$API"
[ -n "$API" ] || { echo "no live endpoints — abort"; exit 1; }

# ── 3a. Rebuild videomme qVZOKel-gpE (1-video filter list) ──
mkdir -p outputs/lists
echo '{"video_id": "qVZOKel-gpE"}' > outputs/lists/vmme_fix_one.jsonl
echo "[$(date +%F_%T)] === REBUILD videomme qVZOKel-gpE ==="
VEIL_CAPTION_MAX_SIDE=1344 VEIL_EXTRACT_MAX_SIDE=1344 \
$PY -m src.build_memory.build_single_similarity \
  --benchmark videomme --filter-from outputs/lists/vmme_fix_one.jsonl \
  --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
  --siglip-gpu cuda:1 --bge-gpu cuda:1 \
  --api-url "$API" --api-model Qwen3.5-27B \
  --api-concurrency 24 --fps 1.0 --theta 0.80 --n-max 30 \
  --prep-workers 16 --extract-gpu 1 --no-persist-frames \
  --out-dir outputs/memory/videommeL_single_1344 \
  --emit-multilayer --multilayer-out-dir outputs/memory/videommeL_1344 \
  --summary-api-url "$API" --summary-concurrency 24
echo "[$(date +%F_%T)] === videomme rebuild rc=$? ==="

# ── 3b. Rebuild longvideobench wxWo44MoCTI (resume skips the 752 built) ──
echo "[$(date +%F_%T)] === REBUILD longvideobench wxWo44MoCTI ==="
VEIL_CAPTION_MAX_SIDE=1344 VEIL_EXTRACT_MAX_SIDE=1344 \
$PY -m src.build_memory.build_single_similarity \
  --benchmark longvideobench \
  --subtitle-dir /home2/ycj/Datas/LongVideoBench/subtitles_srt \
  --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
  --siglip-gpu cuda:1 --bge-gpu cuda:1 \
  --api-url "$API" --api-model Qwen3.5-27B \
  --api-concurrency 24 --fps 1.0 --theta 0.80 --n-max 30 \
  --prep-workers 16 --extract-gpu 1 --no-persist-frames \
  --out-dir outputs/memory/longvideobench_single \
  --emit-multilayer --multilayer-out-dir outputs/memory/longvideobench \
  --summary-api-url "$API" --summary-concurrency 24
echo "[$(date +%F_%T)] === longvideobench rebuild rc=$? ==="

# ── 4. Verify L1.jsonl now present ──
ok=1
for p in outputs/memory/videommeL_1344/qVZOKel-gpE outputs/memory/longvideobench/wxWo44MoCTI; do
  if [ -f "$p/L1.jsonl" ]; then echo "[OK] $p/L1.jsonl ($(wc -l <"$p/L1.jsonl") chunks)";
  else echo "[FAIL] $p/L1.jsonl MISSING"; ok=0; fi
done
[ "$ok" = 1 ] || { echo "[$(date +%F_%T)] rebuild incomplete — NOT patching results, abort"; exit 2; }

# ── 5. Ensure embed server up (may have idle-killed during the long rebuild) ──
if ! curl -s --max-time 3 "$EMB/health" 2>/dev/null | grep -q '"bge": *true'; then
  echo "[$(date +%F_%T)] restarting embed server on cuda:4"
  nohup $PY scripts/start_embed_server.py --port 9000 --device cuda:4 --idle-timeout 3600 \
    > outputs/logs/embed_server_9000.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -s --max-time 3 "$EMB/health" 2>/dev/null | grep -q '"bge": *true' && break; sleep 5
  done
fi
echo "[$(date +%F_%T)] embed: $(curl -s --max-time 3 $EMB/health)"

# ── 6. Drop blank rows, rerun only the affected idxs (canonical out+pipeline) ──
VMME=outputs/results/videomme/mainline_pool/vmme_pool_text.jsonl
LVBH=outputs/results/longvideobench/mainline_pool/lvbh_pool_text.jsonl
$PY - "$VMME" 843 844 845 <<'PYEOF'
import json,sys
f=sys.argv[1]; drop=set(int(x) for x in sys.argv[2:])
rows=[l for l in open(f) if l.strip()]
kept=[l for l in rows if json.loads(l).get("sample_idx") not in drop]
open(f,"w").writelines(kept)
print(f"[patch] {f}: {len(rows)}->{len(kept)} (dropped {len(rows)-len(kept)} blank)")
PYEOF
$PY - "$LVBH" 39 40 <<'PYEOF'
import json,sys
f=sys.argv[1]; drop=set(int(x) for x in sys.argv[2:])
rows=[l for l in open(f) if l.strip()]
kept=[l for l in rows if json.loads(l).get("sample_idx") not in drop]
open(f,"w").writelines(kept)
print(f"[patch] {f}: {len(rows)}->{len(kept)} (dropped {len(rows)-len(kept)} blank)")
PYEOF

COMMON=(--embed-api-url "$EMB" --vlm-api-url "$EP" --llm-api-url "$EP"
        --vlm-api-model Qwen3.5-27B --llm-api-model Qwen3.5-27B
        --no-keyframes --answer-keyframe-k 0 --max-iter 3 --workers 6)

echo "[$(date +%F_%T)] === RERUN videomme 843-845 ==="
$PY experiments/tuning/veil_27b.py --config configs/videomme.yaml \
    --memory-dir outputs/memory/videommeL_1344 "${COMMON[@]}" \
    --sample-start 843 --sample-end 846 \
    --pipeline-name vmme_pool_text --out "$VMME"
echo "[$(date +%F_%T)] === RERUN longvideobench 39-40 ==="
$PY experiments/tuning/veil_27b.py --config configs/longvideobench.yaml \
    --memory-dir outputs/memory/longvideobench "${COMMON[@]}" \
    --sample-start 39 --sample-end 41 \
    --pipeline-name lvbh_pool_text --out "$LVBH"

# ── 7. Report final accuracy on both files ──
echo "[$(date +%F_%T)] === FINAL ==="
$PY - "$VMME" "$LVBH" <<'PYEOF'
import json,sys
for f in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f) if l.strip()]
    rows=[r for r in rows if "correct" in r and r.get("sample_idx") is not None]
    blank=[r for r in rows if not (r.get("pred_letter") or "").strip()]
    n=len(rows); c=sum(r["correct"] for r in rows)
    print(f"  {f}\n    题数={n} 正确={c} acc={100*c/n:.2f}%  空预测残留={len(blank)}")
PYEOF
echo "[$(date +%F_%T)] fix_broken_banks DONE"
