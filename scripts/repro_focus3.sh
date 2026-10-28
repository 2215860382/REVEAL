#!/usr/bin/env bash
# 聚焦三库抄上沿:egor1 x8 / Video-MME x4 / LongVideoBench(合并) x4。vLLM ctx131072。
# egolife/LVBench 已冻结不跑。keep 全部结果,末尾出 best-of-N + mean±std。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SUM=outputs/logs/repro_focus3.log
: > "$SUM"; echo "=== focus3 start $(date '+%F %T') ===" >> "$SUM"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
       -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }
ensure_vllm(){
  np curl -s -m5 http://127.0.0.1:8001/v1/models 2>/dev/null|grep -q Qwen || GPU=5 PORT=8001 MAXLEN=131072 bash scripts/start_vllm_27b.sh &
  np curl -s -m5 http://127.0.0.1:8004/v1/models 2>/dev/null|grep -q Qwen || GPU=6 PORT=8004 MAXLEN=131072 bash scripts/start_vllm_27b.sh &
  while true; do
    v=0; for p in 8001 8004; do np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null|grep -q Qwen && v=$((v+1)); done
    echo "$(date '+%T') vLLM up $v/2" >> "$SUM"; [ "$v" = 2 ] && break; sleep 20
  done
  for p in 8001 8004; do
    ml=$(np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["data"][0].get("max_model_len"))' 2>/dev/null)
    { [ -z "$ml" ] || [ "$ml" -lt 130000 ] 2>/dev/null; } && { echo "FATAL ctx<130k on $p ($ml)" >> "$SUM"; exit 2; }
  done
}
ensure_embed(){
  np curl -s -m5 http://127.0.0.1:9000/health 2>/dev/null|grep -q '"bge":true' && return 0
  setsid env -u http_proxy -u https_proxy NO_PROXY='*' HF_HUB_OFFLINE=1 \
    "$PY" scripts/start_embed_server_live.py --port 9000 --device cuda:1 --idle-timeout 86400 \
    > outputs/logs/embed_9000.log 2>&1 &
  for i in $(seq 1 30); do np curl -s -m4 http://127.0.0.1:9000/health 2>/dev/null|grep -q '"bge":true' && return 0; sleep 5; done
}
score(){ "$PY" - "$1" <<'PY'
import json,sys
t=c=0
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("record_type"):continue
    t+=1;c+=bool(d.get("correct"))
print(f"{100*c/t:.2f}% ({c}/{t})" if t else "NO DATA")
PY
}
eval_run(){  # CONFIG MEMDIR OUT NAME
  local cfg="$1" mem="$2" out="$3" name="$4"
  ensure_vllm; ensure_embed
  mkdir -p "$(dirname "$out")"; rm -f "$out"
  echo "$(date '+%T') START $name" >> "$SUM"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
    "$PY" experiments/reveal_27b.py \
      --config "$cfg" --memory-dir "$mem" \
      --embed-api-url http://127.0.0.1:9000 \
      --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "$name" --out "$out" \
      > "outputs/logs/${name}.log" 2>&1 < /dev/null
  echo "$(date '+%T') DONE $name = $(score "$out")" >> "$SUM"
}
( idle=0
  while true; do sleep 60
    if pgrep -f 'reveal_27b.py' >/dev/null; then idle=0; else idle=$((idle+1)); fi
    if [ "$idle" -ge 15 ]; then
      for pid in $(for d in /proc/[0-9]*; do [ "$(cat $d/comm 2>/dev/null)" = python ] || continue; tr '\0' ' ' <$d/cmdline 2>/dev/null|grep -q 'vllm.entrypoints' && echo ${d#/proc/}; done); do kill "$pid" 2>/dev/null; done
      echo "$(date '+%T') idle 15min -> vLLM killed" >> "$SUM"; exit 0
    fi
  done ) & echo "reaper=$!" >> "$SUM"

# egor1 x8 (150, 便宜先抢上沿)
for r in 10 11 12 13 14 15 16 17; do eval_run configs/egor1.yaml outputs/memory/egolife outputs/results/egor1/variance/run$r.jsonl egor1_run$r; done
# Video-MME x4 (900)
for r in 7 8 9 10; do eval_run configs/videomme.yaml outputs/memory/videommeL outputs/results/videomme/bestof/vmme_b$r.jsonl videomme_b$r; done
# LongVideoBench 合并 x4 (1337)
for r in 7 8 9 10; do eval_run configs/longvideobench_all.yaml outputs/memory/longvideobench_fixed outputs/results/longvideobench/variance/all_r$r.jsonl lvbh_all_r$r; done

# 汇总(三库 best + mean±std)
echo "" >> "$SUM"; echo "=== focus3 汇总 $(date '+%T') ===" >> "$SUM"
"$PY" - >> "$SUM" <<'PY'
import json,glob,statistics as st
def acc(f):
    t=c=0
    for l in open(f):
        d=json.loads(l)
        if d.get("record_type"):continue
        t+=1;c+=bool(d.get("correct"))
    return (100*c/t,t) if t else None
def col(globs,exp):
    v=[];seen=set()
    for g in globs:
        for f in sorted(glob.glob(g)):
            if f in seen:continue
            seen.add(f);a=acc(f)
            if a and a[1]==exp:v.append(a[0])
    return v
sets={
 "Video-MME":     (["outputs/results/videomme/bestof/vmme_b*.jsonl"],900,79.1),
 "LongVideoBench":(["outputs/results/longvideobench/variance/all_r*.jsonl"],1337,75.7),
 "egor1":         (["outputs/results/egor1/repro/main.jsonl","outputs/results/egor1/variance/run*.jsonl"],150,66.7),
}
print(f"{'dataset':16} {'n':>2}  mean±std     best   paper   runs(降序)")
for name,(g,e,p) in sets.items():
    v=col(g,e)
    if not v:print(f"{name:16} 0 none");continue
    m=sum(v)/len(v);sd=st.stdev(v) if len(v)>1 else 0
    print(f"{name:16} {len(v):>2}  {m:5.2f}±{sd:4.2f}  {max(v):5.2f}  {p}  [{' '.join(f'{x:.2f}' for x in sorted(v,reverse=True))}]")
PY
echo "=== done $(date '+%F %T') ===" >> "$SUM"
tail -20 "$SUM"
