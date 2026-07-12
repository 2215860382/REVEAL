"""Convert LongVideoBench per-video JSON subtitles to SRT so the existing build
subtitle path (parse_srt → align_subtitles, expecting {subtitle_dir}/{video_id}.srt)
attaches them per chunk with ZERO build code change.

Two JSON formats occur in the dataset (handled both):
  A: {"timestamp": [start_sec, end_sec], "text": "..."}      (float seconds)
  B: {"start": "HH:MM:SS.mmm", "end": "HH:MM:SS.mmm", "line": "..."}
Only the 753 validation video_ids (from lvb_val.json) are converted.
"""
import json
import re
from pathlib import Path

_HMS = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d+)")


def _to_sec(v):
    """Accept float seconds or an 'HH:MM:SS.mmm' string → float seconds."""
    if isinstance(v, (int, float)):
        return float(v)
    m = _HMS.match(str(v).strip())
    if not m:
        try:
            return float(v)
        except Exception:
            return None
    h, mm, s, frac = m.groups()
    return int(h) * 3600 + int(mm) * 60 + int(s) + float("0." + frac)


def _entry(e):
    """Return (start_sec, end_sec, text) from either format, or None."""
    if "timestamp" in e:
        tsp = e.get("timestamp") or [None, None]
        st, en = _to_sec(tsp[0]), (_to_sec(tsp[1]) if len(tsp) > 1 else None)
        txt = (e.get("text") or "").strip()
    else:
        st, en = _to_sec(e.get("start")), _to_sec(e.get("end"))
        txt = (e.get("line") or e.get("text") or "").strip()
    if st is None or not txt:
        return None
    if en is None:
        en = st + 2.0
    return st, en, txt

ROOT = Path("/home2/ycj/Datas/LongVideoBench")
RAW  = ROOT / "subtitles_raw" / "subtitles"        # {id}_en.json
OUT  = ROOT / "subtitles_srt"                       # {video_id}.srt
OUT.mkdir(parents=True, exist_ok=True)


def _ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    rows = json.loads((ROOT / "lvb_val.json").read_text(encoding="utf-8"))
    # video_id -> (subtitle_path, offset); one row per question, dedup by video_id
    vids = {}
    for r in rows:
        vids[str(r["video_id"])] = (
            str(r.get("subtitle_path", f"{r['video_id']}_en.json")),
            float(r.get("starting_timestamp_for_subtitles", 0.0) or 0.0),
        )

    ok = miss = empty = 0
    for vid, (sub_path, offset) in vids.items():
        jf = RAW / sub_path
        if not jf.exists():
            miss += 1
            continue
        try:
            entries = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            miss += 1
            continue
        lines = []
        n = 0
        for e in entries:
            got = _entry(e)
            if got is None:
                continue
            st, en, txt = got
            n += 1
            lines.append(f"{n}\n{_ts(st + offset)} --> {_ts(en + offset)}\n{txt}\n")
        if n == 0:
            empty += 1
            continue
        (OUT / f"{vid}.srt").write_text("\n".join(lines), encoding="utf-8")
        ok += 1

    print(f"[json2srt] videos={len(vids)}  wrote={ok}  missing_json={miss}  empty={empty}  -> {OUT}")


if __name__ == "__main__":
    main()
