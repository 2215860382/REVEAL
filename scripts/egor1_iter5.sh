#!/usr/bin/env bash
# egor1 max_iter=5 实验 x4:看多迭代能否把 INSUFFICIENT 查找题救回来。vLLM ctx131072。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SUM=outputs/logs/egor1_iter5.log
: > "$SUM"; echo "=== egor1 iter5 start $(date '+%F %T') ===" >> "$SUM"
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
    echo "$(date '+%T') port $p max_model_len=$ml" >> "$SUM"
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
from collections import Counter
t=c=0; ver=Counter(); byv={}
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("record_type"):continue
    t+=1; ok=bool(d.get("correct")); c+=ok
    ti=d.get("trace_iters")
    v=ti[-1].get("verdict") if isinstance(ti,list) and ti else "?"
    ver[v]+=1; byv.setdefault(v,[0,0]); byv[v][0]+=ok; byv[v][1]+=1
ins=ver.get("INSUFFICIENT",0)
det=" ".join(f"{k}:{byv[k][0]}/{byv[k][1]}" for k in byv)
print(f"{100*c/t:.2f}% ({c}/{t})  INSUFF={ins}/{t}  [{det}]" if t else "NO DATA")
PY
}
eval_run(){  # OUT NAME
  local out="$1" name="$2"
  ensure_vllm; ensure_embed
  mkdir -p "$(dirname "$out")"; rm -f "$out"
  echo "$(date '+%T') START $name (max_iter=5)" >> "$SUM"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
    "$PY" experiments/reveal_27b.py \
      --config configs/egor1.yaml --memory-dir outputs/memory/egolife \
      --max-iter 5 \
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

for r in 1 2 3 4; do eval_run outputs/results/egor1/iter5/run$r.jsonl egor1_iter5_run$r; done

echo "" >> "$SUM"; echo "=== iter5 vs iter3 对比 $(date '+%T') ===" >> "$SUM"
"$PY" - >> "$SUM" <<'PY'
import json,glob,statistics as st
def stats(files,exp=150):
    accs=[]; ins=[]
    for f in sorted(files):
        t=c=insc=0
        for l in open(f):
            d=json.loads(l)
            if d.get("record_type"):continue
            t+=1; c+=bool(d.get("correct"))
            ti=d.get("trace_iters")
            v=ti[-1].get("verdict") if isinstance(ti,list) and ti else "?"
            insc+=(v=="INSUFFICIENT")
        if t==exp: accs.append(100*c/t); ins.append(100*insc/t)
    return accs,ins
a3,i3=stats(["outputs/results/egor1/repro/main.jsonl"]+glob.glob("outputs/results/egor1/variance/run*.jsonl"))
a5,i5=stats(glob.glob("outputs/results/egor1/iter5/run*.jsonl"))
def line(tag,a,i):
    if not a: print(f"{tag}: none"); return
    m=sum(a)/len(a); sd=st.stdev(a) if len(a)>1 else 0
    print(f"{tag}: n={len(a)} acc mean {m:.2f}±{sd:.2f} best {max(a):.2f}  INSUFF均值 {sum(i)/len(i):.1f}%  [{' '.join(f'{x:.2f}' for x in sorted(a,reverse=True))}]")
line("iter3(基线,保留4+已删)",a3,i3)
line("iter5(新)",a5,i5)
print("paper 66.7 (100/150)")
PY
echo "=== done $(date '+%F %T') ===" >> "$SUM"
tail -20 "$SUM"
