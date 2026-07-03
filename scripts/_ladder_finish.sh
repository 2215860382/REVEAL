#!/usr/bin/env bash
# 2026-07-03: 纯文本消融收尾。只补真正缺的两点:
#   n2 = 行2 多轮·L1-only·无分解        (L2=0 L3=0, iter3, single-query, no-rubric, no-signal) @8004
#   d3 = 行3 多层·带分解·无rubric·无signal (L2=4 L3=2, iter3, decomp,      no-rubric, no-signal) @8005
# 复用磁盘已有: 裸RAG=naive_l1_top8; 无分解链=sqx_*/sq_*; d4/d5(在跑)=带分解·all +rubric/+signal(=79同配)。
# 冗余的 lad_n1/n3/n4/n5 一律不跑(与 sq_*/naive 重复)。d5 是孤儿进程会自然跑完。
# 逻辑: 等 d5 满900→8005跑d3; 等 d4 满900→杀ladder3的8004冗余尾(n4)→8004跑n2; 两者完→打分出表。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
export OTEL_SDK_DISABLED=true
exec >>outputs/logs/ladder_finish.log 2>&1
echo "===== ladder_finish start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
EMBED=http://127.0.0.1:9000; SA=http://127.0.0.1:8004; SB=http://127.0.0.1:8005
cnt(){ [ -f "$OUT/$1.jsonl" ] && echo $(( $(wc -l < "$OUT/$1.jsonl") - 1 )) || echo 0; }

run(){ # $1 name $2 srv $3 L2 $4 L3 $5 extra_flags
  local nm=$1 srv=$2 l2=$3 l3=$4 ef=$5
  curl -s -m5 --noproxy 127.0.0.1 "$srv/v1/models" 2>/dev/null | grep -q Qwen3.5-27B \
    || { echo "[exp] $nm SKIP: $srv 不通 $(date '+%T')"; return 1; }
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm srv=$srv L2=$l2 L3=$l3 flags=[$ef] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=$l2 VEIL_MP_L3_CAP=$l3 VEIL_MP_L4_CAP=0 \
    timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $srv --llm-api-url $srv \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 $ef \
    --pipeline-name $nm --workers 30 --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

# ── 等 d5 满900腾出 8005 → 跑 d3 ──
( while :; do d=$(cnt lad_d5_sig); echo "[gate-d5] $d/900 $(date '+%T')"; [ "$d" -ge 900 ] && break; sleep 60; done
  run lad_d3_mem "$SB" 4 2 "--no-rubric --ignore-verifier-signal" ) &

# ── 等 d4 满900 → 杀 ladder3 8004 冗余尾 → 跑 n2 ──
( while :; do d=$(cnt lad_d4_rub); echo "[gate-d4] $d/900 $(date '+%T')"; [ "$d" -ge 900 ] && break; sleep 60; done
  echo "[kill] d4完, 杀 ladder3 8004子壳(2172971)+主(2088997)+任何冗余n python $(date '+%T')"
  kill 2172971 2088997 2>/dev/null
  pkill -u ycj -f 'pipeline-name lad_n[1345]_' 2>/dev/null
  for r in lad_n1_raw lad_n3_mem lad_n4_rub lad_n5_sig; do rm -f "$OUT/$r.jsonl"; done
  sleep 5
  run lad_n2_mr "$SA" 0 0 "--single-query-iter0 --no-rubric --ignore-verifier-signal" ) &
wait
echo "[all done] 打分 $(date '+%T')"
"$PY" - <<'PY'
import json,glob,os
OUT="outputs/results/videomme/ablation"
def sc(nm):
    f=f"{OUT}/{nm}.jsonl"
    if not os.path.exists(f): return None
    n=c=0
    for line in open(f):
        try: d=json.loads(line)
        except: continue
        if d.get("record_type") or "correct" not in d: continue
        n+=1; c+=1 if d.get("correct") else 0
    return (n,c,100*c/n if n else 0)
print("== 纯文本消融四链 ==")
grid={
 "裸RAG":       ["naive_l1_top8"],
 "无分解·独占": ["sqx_s1_multiround","sqx_s2_rubric","sqx_s3_signal"],
 "无分解·all":  ["sq_s1_multiround","sq_s2_rubric","sq_s3_signal"],
 "带分解·all":  ["lad_d3_mem","lad_d4_rub","lad_d5_sig"],
 "行2多轮L1":   ["lad_n2_mr"],
}
for k,fs in grid.items():
    xs=[sc(x) for x in fs]
    print(f"{k:12s} "+" | ".join(f"{x[2]:.2f}({x[0]})" if x else "----" for x in xs))
PY
echo "===== ladder_finish done $(date '+%F %T') ====="
