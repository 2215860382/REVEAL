#!/usr/bin/env bash
# Chains after _resample_pairs_v3.sh: once clean pairs exist, build TWO rubrics on the
# SAME new pairs — c1_new (no gold) vs c1_gold (--show-gold) — then score both offline.
# Answers: does letting the model see the gold answer during refine sharpen the rubric?
set -u
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
O=outputs/rubric/pairs_v3
DRV=outputs/rubric/pairs_v3_driver.log
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY AI_PROXY_HTTP AI_PROXY_SOCKS 2>/dev/null || true
URL=http://127.0.0.1:8003,http://10.82.1.145:8004,http://10.82.1.145:8005

echo "[wait] for RESAMPLE_V3_DONE (clean pairs)... $(date +%T)"
while ! grep -q "RESAMPLE_V3_DONE" $DRV 2>/dev/null; do sleep 30; done
CP=$O/pairs.clean.jsonl
N=$(wc -l <$CP 2>/dev/null || echo 0)
echo "[gate] clean pairs = $N  $(date +%T)"
if [ "$N" -lt 100 ]; then
  echo "[ABORT] only $N clean pairs (<100) — recipe likely off, not wasting refine compute."
  echo "GOLD_COMPARE_ABORTED"; exit 1
fi

refine(){ # $1 tag  $2 extra-flags
  echo "[refine $1] start $(date +%T)  flags='$2'"
  $PY -m src.generate_rubric.construct.contrastive_trace.refine_rubric \
    --pairs $CP --chains $O/chains.jsonl --rubric-out $O/rubric_$1.yaml \
    --config configs/veil.yaml --llm-api-url $URL \
    --max-per-type 30 $2 >> $O/refine_$1.log 2>&1
  echo "[refine $1] done $(date +%T)"
}
# both on the SAME pairs, only difference = --show-gold
refine c1new ""            &
P1=$!
refine c1gold "--show-gold" &
P2=$!
wait $P1 $P2

score(){ # $1 tag
  echo "[score $1] start $(date +%T)"
  $PY -m src.generate_rubric.construct.contrastive_trace.score_pairs \
    --pairs $CP --chains $O/chains.jsonl --rubric $O/rubric_$1.yaml \
    --config configs/veil.yaml --llm-api-url $URL \
    --workers 24 --no-images --resume --progress-every 50 \
    --out $O/score_$1.jsonl >> $O/score_$1.log 2>&1
  echo "[score $1] done $(date +%T)"
}
score c1new
score c1gold

echo "===== GOLD-COMPARE RESULT ====="
for t in c1new c1gold; do
  echo "--- $t ---"
  grep -E "BEST_PAIR_ACC|BEST_THR|\[best\]" $O/score_$t.log 2>/dev/null | tail -3
done
echo "GOLD_COMPARE_DONE $(date +%T)"
