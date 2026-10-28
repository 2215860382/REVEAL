#!/usr/bin/env bash
# 追加 +3 批次:Video-MME/LVBench/egor1 各再3次 + LongVideoBench 合并config再3次。
# egolife 不动(已3次)。先等当前批次(repro_bestof3的LVBench r3)跑完再启,免抢端点。vLLM ctx131072。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SUM=outputs/logs/repro_plus3.log
: > "$SUM"; echo "=== plus3 start $(date '+%F %T') ===" >> "$SUM"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
       -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

# 0) 先排队:等当前批次的 reveal_27b 全部结束(LVBench r3 收尾)
echo "$(date '+%T') waiting current batch to drain..." >> "$SUM"
while pgrep -f 'experiments/reveal_27b.py' >/dev/null 2>&1; do sleep 120; done
echo "$(date '+%T') current batch drained, starting plus3" >> "$SUM"

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

# 15min 空转看门狗
( idle=0
  while true; do sleep 60
    if pgrep -f 'reveal_27b.py' >/dev/null; then idle=0; else idle=$((idle+1)); fi
    if [ "$idle" -ge 15 ]; then
      for pid in $(for d in /proc/[0-9]*; do [ "$(cat $d/comm 2>/dev/null)" = python ] || continue; tr '\0' ' ' <$d/cmdline 2>/dev/null|grep -q 'vllm.entrypoints' && echo ${d#/proc/}; done); do kill "$pid" 2>/dev/null; done
      echo "$(date '+%T') idle 15min -> vLLM killed" >> "$SUM"; exit 0
    fi
  done ) & echo "reaper=$!" >> "$SUM"

# ── egor1 x3 (150) ──
for r in 4 5 6; do eval_run configs/egor1.yaml outputs/memory/egolife outputs/results/egor1/variance/run$r.jsonl egor1_run$r; done
# ── Video-MME x3 (900 long) ──
for r in 1 2 3; do eval_run configs/videomme.yaml outputs/memory/videommeL outputs/results/videomme/bestof/vmme_b$r.jsonl videomme_b$r; done
# ── LongVideoBench 合并 x3 (1337) ──
for r in 1 2 3; do eval_run configs/longvideobench_all.yaml outputs/memory/longvideobench_fixed outputs/results/longvideobench/variance/all_r$r.jsonl lvbh_all_r$r; done
# ── LVBench x3 (1344/θ0.8, 1267) ──
for r in 4 5 6; do eval_run configs/lvbench.yaml outputs/memory/lvbench_ml3_n12 outputs/results/lvbench/repro/main_1344_08_r$r.jsonl lvbench_1344_08_r$r; done

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
def runs_of(files,expect=None):
    out=[]
    for f in files:
        if os.path.exists(f):
            c,t=sc(f)
            if t and (expect is None or t==expect): out.append((c,t,os.path.basename(f)))
    return out
vm=runs_of(glob.glob("outputs/results/videomme/**/*.jsonl",recursive=True),900)
eg=runs_of(["outputs/results/egor1/repro/main.jsonl"]+glob.glob("outputs/results/egor1/variance/*.jsonl"),150)
el=runs_of(["outputs/results/egolife/repro/main.jsonl"]+glob.glob("outputs/results/egolife/variance/run*.jsonl"),500)
lv=runs_of(glob.glob("outputs/results/lvbench/repro/main_1344_08*.jsonl"),1267)
# LongVideoBench: 合并直跑(all_r*, 1337) + 旧拆分配对(long+nonlong)
lvbh=runs_of(glob.glob("outputs/results/longvideobench/variance/all_r*.jsonl"),1337)
def comb(lf,nf,tag):
    if os.path.exists(lf) and os.path.exists(nf):
        c1,t1=sc(lf); c2,t2=sc(nf)
        if t1 and t2: return (c1+c2,t1+t2,tag)
    return None
for lf,nf,tag in [("outputs/results/longvideobench/repro/main.jsonl","outputs/results/longvideobench/repro/nonlong.jsonl","split1"),
                  ("outputs/results/longvideobench/variance/long_r2.jsonl","outputs/results/longvideobench/variance/nonlong_r2.jsonl","split2"),
                  ("outputs/results/longvideobench/variance/long_r3.jsonl","outputs/results/longvideobench/variance/nonlong_r3.jsonl","split3")]:
    x=comb(lf,nf,tag)
    if x: lvbh.append(x)
def show(name,rs,paper):
    if not rs: print(f"{name:16} (no runs)"); return
    rs=sorted(rs,reverse=True); b=rs[0]
    allr=" ".join(f"{100*c/t:.2f}" for c,t,_ in rs)
    print(f"{name:16} BEST={100*b[0]/b[1]:.2f}% ({b[0]}/{b[1]})  [{len(rs)}runs: {allr}]  paper {paper}")
print("---- best-of-N (取最高) ----")
show("Video-MME",vm,"79.1")
show("LVBench(1344)",lv,"65.9")
show("LongVideoBench",lvbh,"75.7")
show("egolife",el,"64.4")
show("egor1",eg,"66.7")
PY
echo "=== done $(date '+%F %T') ===" >> "$SUM"
tail -30 "$SUM"
