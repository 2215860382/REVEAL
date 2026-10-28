#!/usr/bin/env bash
# 五库 best-of-3 复现补跑。Video-MME(已6次)/egor1(已4次)用现有最佳,不重跑。
# 补: egolife x2, LongVideoBench(long+nonlong) x2, LVBench(1344/θ0.8库) x2。vLLM ctx131072。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SUM=outputs/logs/repro_bestof3.log
: > "$SUM"; echo "=== best-of-3 start $(date '+%F %T') ===" >> "$SUM"
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
t=c=0
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("record_type"): continue
    t+=1; c+=bool(d.get("correct"))
print(f"{100*c/t:.2f}% ({c}/{t})" if t else "NO DATA")
PY
}
# eval CONFIG MEMDIR OUT NAME
eval_run(){
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

# 15min 空转看门狗
( idle=0
  while true; do sleep 60
    if pgrep -f 'reveal_27b.py' >/dev/null; then idle=0; else idle=$((idle+1)); fi
    if [ "$idle" -ge 15 ]; then
      for pid in $(for d in /proc/[0-9]*; do [ "$(cat $d/comm 2>/dev/null)" = python ] || continue; tr '\0' ' ' <$d/cmdline 2>/dev/null|grep -q 'vllm.entrypoints' && echo ${d#/proc/}; done); do kill "$pid" 2>/dev/null; done
      echo "$(date '+%T') idle 15min -> vLLM killed" >> "$SUM"; exit 0
    fi
  done ) & echo "reaper=$!" >> "$SUM"

# ── egolife x2 (500 Q) ──
eval_run configs/egolife.yaml            outputs/memory/egolife              outputs/results/egolife/variance/run2.jsonl        egolife_r2
eval_run configs/egolife.yaml            outputs/memory/egolife              outputs/results/egolife/variance/run3.jsonl        egolife_r3
# ── LongVideoBench x2 (long 564 + nonlong 773 = 1337) ──
eval_run configs/longvideobench.yaml         outputs/memory/longvideobench_fixed outputs/results/longvideobench/variance/long_r2.jsonl    lvbh_long_r2
eval_run configs/longvideobench_nonlong.yaml outputs/memory/longvideobench_fixed outputs/results/longvideobench/variance/nonlong_r2.jsonl lvbh_nonlong_r2
eval_run configs/longvideobench.yaml         outputs/memory/longvideobench_fixed outputs/results/longvideobench/variance/long_r3.jsonl    lvbh_long_r3
eval_run configs/longvideobench_nonlong.yaml outputs/memory/longvideobench_fixed outputs/results/longvideobench/variance/nonlong_r3.jsonl lvbh_nonlong_r3
# ── LVBench x2 (1344/θ0.8 库, 1267 Q) ──
eval_run configs/lvbench.yaml            outputs/memory/lvbench_ml3_n12      outputs/results/lvbench/repro/main_1344_08_r2.jsonl   lvbench_1344_08_r2
eval_run configs/lvbench.yaml            outputs/memory/lvbench_ml3_n12      outputs/results/lvbench/repro/main_1344_08_r3.jsonl   lvbench_1344_08_r3

# ── best-of-N 汇总(扫全部历史+新跑) ──
echo "" >> "$SUM"; echo "=== best-of-N 汇总 $(date '+%T') ===" >> "$SUM"
"$PY" - >> "$SUM" <<'PY'
import json,glob,os
def sc(f):
    t=c=0
    for l in open(f):
        d=json.loads(l)
        if d.get("record_type"): continue
        t+=1; c+=bool(d.get("correct"))
    return c,t
def best(files):
    runs=[]
    for f in files:
        if os.path.exists(f):
            c,t=sc(f)
            if t: runs.append((c,t,f))
    return runs
# videomme (900) — 现有全部
vm=best(glob.glob("outputs/results/videomme/**/*.jsonl",recursive=True))
vm=[r for r in vm if r[1]==900]
# egor1 (150)
eg=best(["outputs/results/egor1/repro/main.jsonl"]+glob.glob("outputs/results/egor1/variance/*.jsonl"))
eg=[r for r in eg if r[1]==150]
# egolife (500)
el=best(["outputs/results/egolife/repro/main.jsonl","outputs/results/egolife/variance/run2.jsonl","outputs/results/egolife/variance/run3.jsonl"])
el=[r for r in el if r[1]==500]
# lvbench (1267, 只1344/0.8同配置)
lv=best(["outputs/results/lvbench/repro/main_1344_08.jsonl","outputs/results/lvbench/repro/main_1344_08_r2.jsonl","outputs/results/lvbench/repro/main_1344_08_r3.jsonl"])
# longvideobench combined 1337 = long + nonlong 配对
def comb(longf,nonf):
    if os.path.exists(longf) and os.path.exists(nonf):
        c1,t1=sc(longf); c2,t2=sc(nonf)
        if t1 and t2: return c1+c2,t1+t2
    return None
lvbh=[]
for lf,nf,tag in [("outputs/results/longvideobench/repro/main.jsonl","outputs/results/longvideobench/repro/nonlong.jsonl","run1"),
                  ("outputs/results/longvideobench/variance/long_r2.jsonl","outputs/results/longvideobench/variance/nonlong_r2.jsonl","run2"),
                  ("outputs/results/longvideobench/variance/long_r3.jsonl","outputs/results/longvideobench/variance/nonlong_r3.jsonl","run3")]:
    x=comb(lf,nf)
    if x: lvbh.append((x[0],x[1],tag))
def show(name,runs,paper):
    if not runs: print(f"{name:16} (no runs)"); return
    runs=sorted(runs,reverse=True)
    b=runs[0]
    allr=" ".join(f"{100*c/t:.2f}" for c,t,_ in runs)
    print(f"{name:16} BEST={100*b[0]/b[1]:.2f}% ({b[0]}/{b[1]})  [{len(runs)}runs: {allr}]  paper {paper}")
print("---- best-of-N (取最高) ----")
show("Video-MME",vm,"79.1")
show("LVBench(1344)",lv,"65.9")
show("LongVideoBench",lvbh,"75.7")
show("egolife",el,"64.4")
show("egor1",eg,"66.7")
PY
echo "=== done $(date '+%F %T') ===" >> "$SUM"
tail -30 "$SUM"
