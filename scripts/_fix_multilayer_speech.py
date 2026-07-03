"""Post-pass: fold the raw speech transcript into multilayer L1.jsonl `text`.

The multilayer banks built before the speech fix carry only the narrative in L1
`text` (the separate speech_text was dropped). This rewrites each L1.jsonl row's
text to append `\\nSpeech: <asr>` from the source single-layer narrative.json so
L1 evidence matches the single-layer baseline. Idempotent; touches L1 only (no
LLM, no summary rebuild).

    python scripts/_fix_multilayer_speech.py <src_single_dir> <multilayer_dir>
"""
import json
import sys
from pathlib import Path


def fix_one(src_vd: Path, ml_vd: Path) -> str:
    nj = src_vd / "narrative.json"
    l1 = ml_vd / "L1.jsonl"
    if not nj.exists() or not l1.exists():
        return "skip(missing)"
    chunks = json.loads(nj.read_text()).get("chunks", [])
    # multilayer L1 idx = source chunks sorted by start_time, reindexed 0..n-1
    chunks = sorted(chunks, key=lambda c: c.get("start_time", 0.0))
    speech_by_idx = {i: (c.get("speech_text", "") or "").strip()
                     for i, c in enumerate(chunks)}
    rows = [json.loads(x) for x in l1.open() if x.strip()]
    changed = 0
    for r in rows:
        # frame_paths: answerer uses keyframes_resized, not keyframes_origin.
        fps = r.get("frame_paths", [])
        new_fps = [p.replace("/keyframes_origin/", "/keyframes_resized/") for p in fps]
        if new_fps != fps:
            r["frame_paths"] = new_fps
            changed += 1
        # speech: append the raw transcript line if not already present.
        if "\nSpeech: " not in r.get("text", ""):
            sp = speech_by_idx.get(r["idx"], "")
            if sp:
                r["text"] = f"{r['text']}\nSpeech: {sp}"
                changed += 1
    if changed:
        with l1.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return f"ok(+{changed}/{len(rows)})"


def main():
    src_root, ml_root = Path(sys.argv[1]), Path(sys.argv[2])
    dirs = sorted(p for p in ml_root.iterdir() if p.is_dir())
    n_fixed = 0
    for ml_vd in dirs:
        res = fix_one(src_root / ml_vd.name, ml_vd)
        if res.startswith("ok(+") and not res.startswith("ok(+0"):
            n_fixed += 1
    print(f"[speech-fix] processed {len(dirs)} banks, modified {n_fixed}")


if __name__ == "__main__":
    main()
