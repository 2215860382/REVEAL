#!/usr/bin/env bash
# v6 (show-gold rubric) 900 阈值扫描, 复刻 def_c1t60 定版配置 (8/4/2, 纯文本, multi_pool).
# 三阈值 0.45/0.60/0.75 并发, 三端点各 ~12 worker. 完了同题号对比 def_c1t60=78.80%.
set -u
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY AI_PROXY_HTTP AI_PROXY_SOCKS 2>/dev/null || true

MEM=outputs/memory/videomme_ml3_n12
SRV=http://127.0.0.1:8003,http://10.82.1.145:8004,http://10.82.1.145:8005
EMBED=http://127.0.0.1:9000
O=outputs/results/videomme/nightly
RB=outputs/rubric/pairs_v6
SRC=$RB/rubric_v6.yaml

# --- 生成 3 个阈值副本 (general + 所有 template 的 sufficient_threshold 全改) ---
mkthr(){ # $1 = 0.45 等
  $PY - "$SRC" "$RB/rubric_v6_t${1/./}.yaml" "$1" <<'PYEOF'
import sys, yaml
src, dst, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
r = yaml.safe_load(open(src))
def setthr(d):
    if isinstance(d, dict):
        if 'sufficient_threshold' in d: d['sufficient_threshold'] = thr
        for v in d.values(): setthr(v)
    elif isinstance(d, list):
        for v in d: setthr(v)
setthr(r)
yaml.safe_dump(r, open(dst,'w'), allow_unicode=True, sort_keys=False)
print(f'wrote {dst} thr={thr}')
PYEOF
}
mkthr 0.45; mkthr 0.60; mkthr 0.75

run(){ # $1=tag(t45) $2=rubric-copy
  echo "[v6 $1] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$2 timeout 20000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool \
    --sample-start 0 --sample-end 900 --workers 12 \
    --pipeline-name v6_$1 --out $O/v6_$1.jsonl
  echo "[v6 $1] exit=$? $(date '+%T')"
}

run t45 $RB/rubric_v6_t045.yaml &
run t60 $RB/rubric_v6_t060.yaml &
run t75 $RB/rubric_v6_t075.yaml &
wait
echo "===== V6 SWEEP DONE $(date '+%T') ====="
for t in t45 t60 t75; do
  $PY -c "
import json
rows=[json.loads(l) for l in open('$O/v6_$t.jsonl') if json.loads(l).get('record_type')!='header']
n=len(rows); c=sum(1 for r in rows if r.get('correct'))
print(f'v6_$t: {c}/{n} = {c/n*100:.2f}%')
" 2>/dev/null
done
echo "V6_SWEEP_REPORT_DONE"
