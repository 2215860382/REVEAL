#!/usr/bin/env bash
# Recompute the evidence-volume curve on the NEW bank (videommeL_1344) so the curve
# and the Ours star point are same-bank, then redraw evidence_volume.pdf.
# Waits for the lvbench ablation to finish first (shared endpoints).
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
PYV=/home2/ycj/miniconda3/envs/reveal/bin/python       # has src deps (embedder/answerer)
PYF=/home2/ycj/miniconda3/envs/videorag/bin/python   # has matplotlib
LOG=outputs/logs/rerun_volume_fig.log
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[$(date +%F_%T)] rerun_volume START — waiting for lvb_ablation_pool DONE"
for _ in $(seq 1 240); do
  grep -q 'lvb_ablation_pool DONE' outputs/logs/lvb_ablation_pool.log 2>/dev/null && break
  sleep 60
done
grep -q 'lvb_ablation_pool DONE' outputs/logs/lvb_ablation_pool.log 2>/dev/null \
  && echo "[$(date +%F_%T)] lvb done — proceeding" \
  || echo "[$(date +%F_%T)] WARN lvb marker not seen after 4h; proceeding anyway"

# old curve was on the deleted videommeL bank — archive and recompute fresh on 1344
VJ=experiments/analysis/evidence_budget/videomme_volume.jsonl
[ -f "$VJ" ] && mv "$VJ" "${VJ%.jsonl}.oldbank.bak"
echo "[$(date +%F_%T)] recomputing volume curve on videommeL_1344 ..."
$PYV experiments/analysis/evidence_budget/videomme_volume.py
echo "[$(date +%F_%T)] redrawing evidence_volume.pdf ..."
$PYF experiments/analysis/make_volume_fig.py
echo "[$(date +%F_%T)] rerun_volume DONE"
