#!/usr/bin/env bash
# Download all 103 LVBench videos from YouTube at native quality → {key}.mp4.
# Resumable (--download-archive skips completed), logs failures. LVBench has no
# val/test split, so this is the full evaluation set.
set -u
META=${META:-/home2/ycj/Datas/LVBench/video_info.meta.jsonl}
VDIR=${VDIR:-/home2/ycj/Datas/LVBench/videos}
PROXY=${PROXY:-http://127.0.0.1:10086}
PARALLEL=${PARALLEL:-3}
YTDLP=/home2/ycj/miniconda3/envs/veil/bin/yt-dlp
PY=/home2/ycj/miniconda3/envs/veil/bin/python
mkdir -p "$VDIR"
ARCHIVE="$VDIR/.downloaded.txt"
FAILLOG="$VDIR/.failed.txt"; : > "$FAILLOG"

dl(){
  local k="$1"
  "$YTDLP" --proxy "$PROXY" --no-warnings --ignore-errors \
    --download-archive "$ARCHIVE" --no-overwrites \
    --retries 5 --fragment-retries 10 \
    -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
    --merge-output-format mp4 \
    -o "$VDIR/%(id)s.%(ext)s" \
    "https://www.youtube.com/watch?v=$k" \
    && echo "[ok] $k" \
    || { echo "[FAIL] $k"; echo "$k" >> "$FAILLOG"; }
}
export -f dl; export YTDLP VDIR PROXY ARCHIVE FAILLOG

"$PY" -c "import json;[print(json.loads(l)['key']) for l in open('$META') if l.strip()]" \
  | xargs -P "$PARALLEL" -I{} bash -c 'dl "$@"' _ {}

N=$(ls "$VDIR"/*.mp4 2>/dev/null | wc -l)
F=$(sort -u "$FAILLOG" 2>/dev/null | wc -l)
echo "===== LVBENCH DOWNLOAD DONE: ${N}/103 present, ${F} failed ====="
[ "$F" -gt 0 ] && { echo "failed keys:"; sort -u "$FAILLOG"; }
