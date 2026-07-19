#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

PY=${PY:-/home2/ycj/miniconda3/envs/reveal/bin/python}
FILTER=${FILTER:-outputs/lists/vmme_448_100.jsonl}
BUILD_LOG=${BUILD_LOG:-outputs/logs/build_videomme_448.log}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-outputs/results/videomme/compare_mem_${RUN_TS}}
LOG=${LOG:-outputs/logs/compare_videomme_mem_${RUN_TS}.log}

EMBED=${EMBED:-http://127.0.0.1:9000}
EP=${EP:-http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8001,http://10.82.1.145:8004}
WORKERS=${WORKERS:-8}
RUBRIC=${REVEAL_RUBRIC_PATH:-outputs/rubric/rubric_contrastive.yaml}

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

log() {
  echo "[compare $(date +%F_%H:%M:%S)] $*" | tee -a "$LOG"
}

count_filter() {
  "$PY" - "$FILTER" <<'PY'
import json, sys
ids = []
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        ids.append(json.loads(line)["video_id"])
print(len(set(ids)))
PY
}

missing_448() {
  "$PY" - "$FILTER" outputs/memory/videommeL_448 <<'PY'
import json, sys
from pathlib import Path
filter_path = Path(sys.argv[1])
bank = Path(sys.argv[2])
ids = []
for line in filter_path.open(encoding="utf-8"):
    if line.strip():
        ids.append(json.loads(line)["video_id"])
missing = []
for vid in sorted(set(ids)):
    vd = bank / vid
    if not ((vd / "L1.jsonl").exists() and (vd / "L2.jsonl").exists() and (vd / "L3.jsonl").exists()):
        missing.append(vid)
print(len(missing))
if missing:
    print("\n".join(missing[:20]))
PY
}

summarize_one() {
  local name="$1"
  local path="$2"
  "$PY" - "$name" "$path" <<'PY'
import collections, json, sys
name, path = sys.argv[1], sys.argv[2]
rows = []
for line in open(path, encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("record_type") == "header":
        continue
    rows.append(r)
total = len(rows)
correct = sum(1 for r in rows if r.get("correct") is True)
errors = sum(1 for r in rows if r.get("error"))
elapsed = [float(r.get("elapsed") or 0) for r in rows]
by_type = collections.defaultdict(lambda: [0, 0])
for r in rows:
    qt = r.get("question_type") or "UNKNOWN"
    by_type[qt][1] += 1
    by_type[qt][0] += int(r.get("correct") is True)
print(json.dumps({
    "name": name,
    "path": path,
    "total": total,
    "correct": correct,
    "accuracy": round(correct / total, 6) if total else 0,
    "errors": errors,
    "avg_elapsed_sec": round(sum(elapsed) / total, 2) if total else 0,
    "by_type": {
        k: {
            "correct": v[0],
            "total": v[1],
            "accuracy": round(v[0] / v[1], 6) if v[1] else 0,
        }
        for k, v in sorted(by_type.items())
    },
}, ensure_ascii=False, sort_keys=True))
PY
}

run_eval() {
  local name="$1"
  local mem="$2"
  local out="$OUT_DIR/${name}.jsonl"
  local pipeline="vmme_cmp_${name}_${RUN_TS}"
  log "eval start name=$name mem=$mem out=$out"
  env REVEAL_RUBRIC_PATH="$RUBRIC" "$PY" experiments/tuning/reveal_27b.py \
    --config configs/videomme.yaml \
    --memory-dir "$mem" \
    --filter-from "$FILTER" \
    --embed-api-url "$EMBED" \
    --vlm-api-url "$EP" --llm-api-url "$EP" \
    --vlm-api-model Qwen3.5-27B --llm-api-model Qwen3.5-27B \
    --no-keyframes --answer-keyframe-k 0 \
    --max-iter 3 \
    --workers "$WORKERS" \
    --pipeline-name "$pipeline" \
    --out "$out" >>"$LOG" 2>&1
  summarize_one "$name" "$out" | tee -a "$OUT_DIR/summary.jsonl" | tee -a "$LOG"
  log "eval done name=$name"
}

need=$(count_filter)
log "waiting for 448 bank: filter=$FILTER videos=$need"

while true; do
  if pgrep -f 'build_single_similarity.*videomme.*videommeL_448' >/dev/null; then
    current=$(find outputs/memory/videommeL_448 -name L1.jsonl 2>/dev/null | wc -l)
    log "448 build still running; L1 complete dirs=$current/$need"
    sleep 180
    continue
  fi

  miss_report=$(missing_448)
  miss_count=$(echo "$miss_report" | head -n 1)
  if [ "$miss_count" = "0" ]; then
    log "448 bank complete"
    break
  fi
  log "448 build process gone but bank incomplete; missing=$miss_count"
  echo "$miss_report" | sed '1d' | tee -a "$LOG"
  sleep 180
done

: >"$OUT_DIR/summary.jsonl"
run_eval "orig_group_448base" "outputs/memory/videommeL"
run_eval "fullres_1344" "outputs/memory/videommeL_1344"
run_eval "group_improved_448" "outputs/memory/videommeL_448"

"$PY" - "$OUT_DIR/summary.jsonl" "$OUT_DIR/summary.md" <<'PY'
import json, sys
summary_path, md_path = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(summary_path, encoding="utf-8") if line.strip()]
rows.sort(key=lambda r: r["accuracy"], reverse=True)
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# VideoMME Memory Bank Comparison\n\n")
    f.write("| rank | memory | acc | correct/total | errors | avg sec |\n")
    f.write("|---:|---|---:|---:|---:|---:|\n")
    for i, r in enumerate(rows, 1):
        f.write(f"| {i} | {r['name']} | {r['accuracy']*100:.2f}% | {r['correct']}/{r['total']} | {r['errors']} | {r['avg_elapsed_sec']:.2f} |\n")
    f.write("\n")
    for r in rows:
        f.write(f"## {r['name']}\n\n")
        f.write("| question type | acc | correct/total |\n")
        f.write("|---|---:|---:|\n")
        for qt, v in r["by_type"].items():
            f.write(f"| {qt} | {v['accuracy']*100:.2f}% | {v['correct']}/{v['total']} |\n")
        f.write("\n")
print(md_path)
PY

log "ALL DONE summary=$OUT_DIR/summary.md"
