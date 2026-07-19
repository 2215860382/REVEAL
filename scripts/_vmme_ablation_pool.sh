#!/usr/bin/env bash
# VideoMME ablation ladder — the 3 IMPLICIT-verifier rungs, rerun on the SAME bank
# (videommeL_1344) and the SAME pool-era pipeline as Ours, so the whole ladder is
# finally bank-consistent (the old abl_* were on the deleted videomme_ml3_n12 bank →
# not comparable to the 1344 Ours). Only the ablated component differs per rung.
#
#   M_base        = single-L1 (8/0/0) + implicit verifier + 1 round
#   M_iter        = single-L1 (8/0/0) + implicit verifier + 3 rounds
#   M_iter+hybrid = 3-level (8/4/2)    + implicit verifier + 3 rounds
# (M_iter+rubric = single-L1 + pool rubric  → overnight vmme_pool_l1rubric;
#  Ours          = 3-level  + pool rubric  → main-table vmme_pool_text, bank-fixed.)
#
# Chained AFTER _fix_broken_banks.sh finishes (not just overnight): these runs also hit
# qVZOKel-gpE idx 843-845, so they must wait for that bank to be rebuilt or they'd blank
# too. Serialized to keep the shared endpoints clean.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"

PY=/home2/ycj/miniconda3/envs/reveal/bin/python
CFG=configs/videomme.yaml
MEM=outputs/memory/videommeL_1344
EMB=http://127.0.0.1:9000
EP=http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004
OUTDIR=outputs/results/videomme/ablation
FIXLOG=outputs/logs/fix_broken_banks.log
LOG=outputs/logs/vmme_ablation_pool.log
mkdir -p "$OUTDIR"
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[$(date +%F_%T)] vmme_ablation_pool START"

# ── Wait for the bank-fix chain to finish (which itself waits for overnight) ──
echo "[$(date +%F_%T)] waiting for '_fix_broken_banks DONE' in $FIXLOG ..."
for _ in $(seq 1 600); do
  grep -q 'fix_broken_banks DONE' "$FIXLOG" 2>/dev/null && break
  sleep 60
done
grep -q 'fix_broken_banks DONE' "$FIXLOG" 2>/dev/null \
  && echo "[$(date +%F_%T)] bank-fix done — proceeding" \
  || echo "[$(date +%F_%T)] WARN fix marker not seen after 10h; proceeding anyway"

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
        --no-keyframes --answer-keyframe-k 0 --workers 32
        --sample-start 0 --sample-end 900)

run () {   # $1=pipeline  $2=max_iter ; env pre-exported by caller
  local name="$1" mi="$2"
  echo "[$(date +%F_%T)] === $name (max-iter=$mi) START ==="
  $PY experiments/tuning/reveal_27b.py "${COMMON[@]}" \
      --max-iter "$mi" --pipeline-name "$name" --out "$OUTDIR/$name.jsonl"
  echo "[$(date +%F_%T)] === $name DONE (rc=$?) ==="
}

# Implicit (verifier-free) rungs: one reasoner per round decides search-vs-answer and
# emits its own query — no verifier/planner/answerer (REVEAL_WORLDMM_IMPLICIT=1).
# "caption" = the raw caption bank (L1 only, before our layered-memory processing);
# "memory"  = our constructed memory bank (3-level 8/4/2). The caption↔memory gap is
# the value of building the memory bank; the implicit↔rubric gap is the value of the rubric.
# 1) caption + implicit, single round (retrieve once → forced answer)
REVEAL_MP_L1_CAP=8 REVEAL_MP_L2_CAP=0 REVEAL_MP_L3_CAP=0 REVEAL_WORLDMM_IMPLICIT=1 \
  run vmme_caption_implicit_r1 1
# 2) caption + implicit, iterate
REVEAL_MP_L1_CAP=8 REVEAL_MP_L2_CAP=0 REVEAL_MP_L3_CAP=0 REVEAL_WORLDMM_IMPLICIT=1 \
  run vmme_caption_implicit 3
# 3) our memory + implicit, iterate
REVEAL_WORLDMM_IMPLICIT=1 \
  run vmme_memory_implicit 3

# ── Report the 2×2(+base) ablation grid — all on videommeL_1344 ──
# Rows = caption bank vs our memory bank (value of building the memory bank).
# Cols = implicit verifier-free vs rubric verifier (value of the rubric).
echo "[$(date +%F_%T)] === VIDEOMME ABLATION GRID (all on videommeL_1344) ==="
$PY - <<'PYEOF'
import json,os
def acc(f):
    if not os.path.exists(f): return None
    rows=[json.loads(l) for l in open(f) if l.strip()]
    rows=[r for r in rows if "correct" in r and r.get("sample_idx") is not None]
    if not rows: return None
    return 100*sum(r["correct"] for r in rows)/len(rows), len(rows)
F={
 "cap_impl_r1":"outputs/results/videomme/ablation/vmme_caption_implicit_r1.jsonl",
 "cap_impl":   "outputs/results/videomme/ablation/vmme_caption_implicit.jsonl",
 "cap_rubric": "outputs/results/videomme/ablation/vmme_pool_l1rubric.jsonl",   # overnight
 "mem_impl":   "outputs/results/videomme/ablation/vmme_memory_implicit.jsonl",
 "mem_rubric": "outputs/results/videomme/mainline_pool/vmme_pool_text.jsonl",  # Ours
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
echo "[$(date +%F_%T)] vmme_ablation_pool DONE"
