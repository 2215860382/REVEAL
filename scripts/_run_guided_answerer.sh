#!/usr/bin/env bash
# Test the question-type-aware answerer guidance. Holds rubric = r2_merged fixed
# (only the answerer prompt changed), runs val-90 and test-90, then compares
# against the pre-guidance runs (val_round2_merged / test_r2_merged) on both
# splits so we can check the effect is consistent, not a single-split fluke.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
SPL=outputs/results/intermediate_goods/overnight_20260627_full720/splits
OUTDIR=outputs/results/videomme/tuning/split_runs
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004
EMBED=http://127.0.0.1:9000
export PYTHONPATH=.

for split in val test; do
  echo "==== guided answerer: $split ===="
  VEIL_RUBRIC_PATH=$RUBRIC "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir outputs/memory/videomme_multiframe \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --pipeline-name ${split}_r2_merged_guided \
    --sample-indices-file $SPL/${split}_indices.txt --workers 24 \
    --out $OUTDIR/${split}_r2_merged_guided.jsonl
done

echo "==================== GUIDED vs BASELINE (same rubric r2_merged) ===================="
"$PY" - <<'PYEOF'
import json
def load(f):
    o={}
    for l in open(f):
        r=json.loads(l)
        if r.get("record_type"): continue
        o[int(r["sample_idx"])]=r
    return o
B="outputs/results/videomme/tuning/split_runs"
pairs={
 "val":  ("outputs/results/videomme/tuning/split_runs/val_round2_merged.jsonl", f"{B}/val_r2_merged_guided.jsonl"),
 "test": ("outputs/results/videomme/tuning/split_runs/test_r2_merged.jsonl",   f"{B}/test_r2_merged_guided.jsonl"),
}
for split,(base_f,new_f) in pairs.items():
    try:
        base=load(base_f); new=load(new_f)
    except FileNotFoundError as e:
        print(f"{split}: missing {e}"); continue
    ids=[i for i in base if i in new]
    cb=sum(1 for i in ids if base[i].get("correct"))
    cn=sum(1 for i in ids if new[i].get("correct"))
    flips=[i for i in ids if bool(base[i].get("correct"))!=bool(new[i].get("correct"))]
    gained=[i for i in flips if new[i].get("correct")]
    lost=[i for i in flips if not new[i].get("correct")]
    print(f"{split:<5}: baseline {cb}/{len(ids)}={cb/len(ids)*100:.1f}%  ->  guided {cn}/{len(ids)}={cn/len(ids)*100:.1f}%  "
          f"({cn-cb:+d})  | 救回 {len(gained)} 救丢 {len(lost)}")
    if gained: print(f"        救回: {sorted(gained)}")
    if lost:   print(f"        救丢: {sorted(lost)}")
PYEOF
echo DONE-GUIDED
