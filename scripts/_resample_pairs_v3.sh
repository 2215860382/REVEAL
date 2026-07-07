#!/usr/bin/env bash
# Re-sample clean pairs from scratch (old gold_v2 subdir was manually deleted).
# temp=0 deterministic labeling, text-only (matches 79% text-only mainline).
# Stages: wait for 900-runs → sample_chains → make_pairs → validate_negatives(drop false negs).
# Stops at clean pairs for human review before the refine comparison.
set -u
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
O=outputs/rubric/pairs_v3
ND=outputs/results/videomme/nightly
mkdir -p $O
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY AI_PROXY_HTTP AI_PROXY_SOCKS 2>/dev/null || true
URL=http://127.0.0.1:8003,http://10.82.1.145:8004,http://10.82.1.145:8005
EMBED=http://127.0.0.1:9000

echo "[wait] for c1t60/c1t75 900-runs to free endpoints... $(date +%T)"
while :; do
  a=$(wc -l <$ND/def_c1t60.jsonl 2>/dev/null || echo 0)
  b=$(wc -l <$ND/def_c1t75.jsonl 2>/dev/null || echo 0)
  [ "$a" -ge 900 ] && [ "$b" -ge 900 ] && break
  sleep 30
done
echo "[wait] 900-runs done (c1t60=$a c1t75=$b). starting resample $(date +%T)"

echo "[1/3] sample_chains (temp=0, n=1, text-only) -> chains.jsonl  $(date +%T)"
$PY -m src.generate_rubric.construct.contrastive_trace.sample_chains \
  --config configs/veil.yaml --memory-dir outputs/memory/videomme_ml3_n12 \
  --embed-api-url $EMBED --vlm-api-url $URL \
  --query-variants q+opts,q --alphas 0.6 --budget-ladder 2,4,8,16 \
  --random-negatives 2 --n-samples 1 --temperature 0 --no-keyframes \
  --workers 24 --seed 17 --resume --out $O/chains.jsonl >> $O/sample.log 2>&1
echo "[1/3] chains: $(wc -l <$O/chains.jsonl) rows"

echo "[2/3] make_pairs (insufficient x sufficient) -> pairs.jsonl  $(date +%T)"
$PY -m src.generate_rubric.construct.contrastive_trace.make_pairs \
  --chains $O/chains.jsonl --max-pairs-per-question 4 --out $O/pairs.jsonl >> $O/pairs.log 2>&1
echo "[2/3] pairs: $(wc -l <$O/pairs.jsonl) rows"

echo "[3/3] validate_negatives (drop false negatives, temp0 closed-book) -> pairs.clean.jsonl  $(date +%T)"
$PY -m src.generate_rubric.construct.contrastive_trace.validate_negatives \
  --pairs $O/pairs.jsonl --chains $O/chains.jsonl \
  --config configs/veil.yaml --vlm-api-url $URL \
  --workers 24 --resume --out $O/pairs.clean.jsonl >> $O/clean.log 2>&1
echo "[3/3] clean pairs: $(wc -l <$O/pairs.clean.jsonl) rows"

# quick stats: questions covered
$PY -c "
import json
p=[json.loads(l) for l in open('$O/pairs.clean.jsonl')]
q=len({r['sample_idx'] for r in p})
print(f'[stat] clean pairs={len(p)}  questions={q}')
" 2>/dev/null
echo "RESAMPLE_V3_DONE $(date +%T)"
