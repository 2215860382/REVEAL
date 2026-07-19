#!/usr/bin/env bash
# LVBench ablation ladder — the 3 IMPLICIT-verifier rungs, same bank (outputs/memory/lvbench)
# and same pool-era pipeline as Ours, so the whole ladder is bank-consistent.
# Mirror of _vmme_ablation_pool.sh but for lvbench (full 1267, no sample cap).
#   caption+隐式 单轮 = L1(8/0/0) + implicit + 1 round
#   caption+隐式 迭代 = L1(8/0/0) + implicit + 3 rounds
#   记忆+隐式 迭代    = 3-level(8/4/2) + implicit + 3 rounds
# (caption+rubric = lvb_pool_l1rubric already done; Ours = lvb_pool_text already done.)
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml
export REVEAL_VERIFIER_AGG=max

PY=/home2/ycj/miniconda3/envs/reveal/bin/python
CFG=configs/lvbench.yaml
MEM=outputs/memory/lvbench
EMB=http://127.0.0.1:9000
EP=http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004
OUTDIR=outputs/results/lvbench/ablation
LOG=outputs/logs/lvb_ablation_pool.log
mkdir -p "$OUTDIR"
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[$(date +%F_%T)] lvb_ablation_pool START"

# ── Ensure embed server up ──
if ! curl -s --max-time 3 "$EMB/health" 2>/dev/null | grep -q '"bge": *true'; then
  echo "[$(date +%F_%T)] starting embed server on cuda:4"
  nohup $PY scripts/start_embed_server.py --port 9000 --device cuda:4 --idle-timeout 3600 \
    > outputs/logs/embed_server_9000.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -s --max-time 3 "$EMB/health" 2>/dev/null | grep -q '"bge": *true' && break; sleep 5
  done
fi
echo "[$(date +%F_%T)] embed: $(curl -s --max-time 3 $EMB/health)"

COMMON=(--config "$CFG" --memory-dir "$MEM" --embed-api-url "$EMB"
        --vlm-api-url "$EP" --llm-api-url "$EP"
        --vlm-api-model Qwen3.5-27B --llm-api-model Qwen3.5-27B
        --no-keyframes --answer-keyframe-k 0 --workers 32)

run () {   # $1=pipeline  $2=max_iter ; env pre-exported by caller
  local name="$1" mi="$2"
  echo "[$(date +%F_%T)] === $name (max-iter=$mi) START ==="
  $PY experiments/tuning/reveal_27b.py "${COMMON[@]}" \
      --max-iter "$mi" --pipeline-name "$name" --out "$OUTDIR/$name.jsonl"
  echo "[$(date +%F_%T)] === $name DONE (rc=$?) ==="
}

# 1) caption + implicit, single round
REVEAL_MP_L1_CAP=8 REVEAL_MP_L2_CAP=0 REVEAL_MP_L3_CAP=0 REVEAL_WORLDMM_IMPLICIT=1 \
  run lvb_caption_implicit_r1 1
# 2) caption + implicit, iterate
REVEAL_MP_L1_CAP=8 REVEAL_MP_L2_CAP=0 REVEAL_MP_L3_CAP=0 REVEAL_WORLDMM_IMPLICIT=1 \
  run lvb_caption_implicit 3
# 3) our memory + implicit, iterate
REVEAL_WORLDMM_IMPLICIT=1 \
  run lvb_memory_implicit 3

# ── Report the 2×2(+base) ablation grid ──
echo "[$(date +%F_%T)] === LVBENCH ABLATION GRID (all on outputs/memory/lvbench) ==="
$PY - <<'PYEOF'
import json,os
def acc(f):
    if not os.path.exists(f): return None
    rows=[json.loads(l) for l in open(f) if l.strip()]
    rows=[r for r in rows if "correct" in r and r.get("sample_idx") is not None]
    if not rows: return None
    seen={}
    for r in rows: seen[(r.get("video_id"),r.get("sample_idx"))]=r
    rr=list(seen.values())
    return 100*sum(bool(r["correct"]) for r in rr)/len(rr), len(rr)
F={
 "cap_impl_r1":"outputs/results/lvbench/ablation/lvb_caption_implicit_r1.jsonl",
 "cap_impl":   "outputs/results/lvbench/ablation/lvb_caption_implicit.jsonl",
 "cap_rubric": "outputs/results/lvbench/ablation/lvb_pool_l1rubric.jsonl",
 "mem_impl":   "outputs/results/lvbench/ablation/lvb_memory_implicit.jsonl",
 "mem_rubric": "outputs/results/lvbench/mainline_pool/lvb_pool_text.jsonl",
}
def show(k,label):
    a=acc(F[k])
    print(f"  {label:26s}: " + (f"{a[0]:.2f}%  (n={a[1]})" if a else f"(未完成 {F[k]})"))
    return a[0] if a else None
print(" 单轮基线:")
show("cap_impl_r1","caption + 隐式 (单轮)")
print(" 2×2 网格:")
ci=show("cap_impl",  "caption + 隐式")
cr=show("cap_rubric","caption + rubric")
mi=show("mem_impl",  "我们记忆 + 隐式")
mr=show("mem_rubric","我们记忆 + rubric (Ours)")
print(" 增益分解:")
if ci is not None and mi is not None: print(f"   记忆库价值(隐式行): {mi-ci:+.2f}")
if cr is not None and mr is not None: print(f"   记忆库价值(rubric行): {mr-cr:+.2f}")
if ci is not None and cr is not None: print(f"   rubric价值(caption列): {cr-ci:+.2f}")
if mi is not None and mr is not None: print(f"   rubric价值(记忆列): {mr-mi:+.2f}")
PYEOF
echo "[$(date +%F_%T)] lvb_ablation_pool DONE"
