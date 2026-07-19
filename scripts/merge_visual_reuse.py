#!/usr/bin/env python3
"""复用旧egolife(第三人称)的SigLIP视觉层, 并入egolife_fp的L1(仅A2/A3/A4缺)。
视觉层=SigLIP编码物理帧, 与caption人称无关 → 直接搬, 不重建。
- 按t_start/t_end逐clip对齐(新clip中点落入旧chunk区间)
- 关键帧用硬链接(同一文件系统, 零额外磁盘)
- L1_visual.npz键='vectors'(展平帧), 每chunk用visual_offsets索引 — 与bank_loader一致
- 只写 frame_paths/frame_ts/visual_offsets + L1_visual.npz; 不碰 L1_text/L2/L3
"""
import json, os, sys, numpy as np
from pathlib import Path

ROOT = Path("/home2/ycj/Project/REVEAL/outputs/memory")
OLD  = ROOT / "egolife"      # 视觉源(第三人称)
NEW  = ROOT / "egolife_fp"   # 主线库(第一人称), 目标
PEOPLE = ["A2_ALICE", "A3_TASHA", "A4_LUCIA"]

def load_old(p):
    d = OLD / p
    rows = [json.loads(l) for l in (d / "L1.jsonl").open()]
    vecs = np.load(d / "L1_visual.npz")["vectors"]  # (sum_k, D)
    chunks = []  # (t0, t1, frame_vecs(np), frame_paths(list), frame_ts(list))
    for r in rows:
        offs = r.get("visual_offsets", []) or []
        fps  = r.get("frame_paths", []) or []
        fts  = r.get("frame_ts", []) or []
        fv   = vecs[offs] if offs else np.zeros((0, vecs.shape[1]), np.float32)
        chunks.append((r["t_start"], r["t_end"], fv, fps, fts))
    return chunks, vecs.shape[1]

def find_old(mid, old_chunks):
    for i, (a, b, *_ ) in enumerate(old_chunks):
        if a - 1 <= mid <= b + 1:
            return i
    return -1

def merge_person(p):
    old_chunks, D = load_old(p)
    nd = NEW / p
    new_rows = [json.loads(l) for l in (nd / "L1.jsonl").open()]
    (nd / "keyframes_resized").mkdir(exist_ok=True)

    flat_vecs = []      # 展平帧向量, 累积
    matched = linked = 0
    cursor = 0
    for r in new_rows:
        mid = (r["t_start"] + r["t_end"]) / 2
        oi = find_old(mid, old_chunks)
        if oi < 0:
            r["frame_paths"] = []; r["visual_offsets"] = []
            continue
        _, _, fv, fps, fts = old_chunks[oi]
        if fv.shape[0] == 0:
            r["frame_paths"] = []; r["visual_offsets"] = []
            continue
        matched += 1
        offs = []
        new_fps = []
        for j, rel in enumerate(fps):
            src = OLD / p / rel
            dst = nd / rel
            if src.is_file() and not dst.exists():
                try:
                    os.link(src, dst); linked += 1
                except FileExistsError:
                    pass
                except OSError:
                    # 跨设备等异常: 退回复制
                    import shutil; shutil.copy2(src, dst); linked += 1
            new_fps.append(rel)
            offs.append(cursor + j)
        flat_vecs.append(fv)
        cursor += fv.shape[0]
        r["frame_paths"] = new_fps
        r["visual_offsets"] = offs
        if fts: r["frame_ts"] = fts

    vectors = np.concatenate(flat_vecs, axis=0) if flat_vecs else np.zeros((0, D), np.float32)
    np.savez(nd / "L1_visual.npz", vectors=vectors.astype(np.float32))
    with (nd / "L1.jsonl").open("w") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{p}] 新chunk={len(new_rows)} 匹配={matched} 帧硬链接={linked} "
          f"L1_visual={vectors.shape}", flush=True)
    return matched, len(new_rows)

if __name__ == "__main__":
    print("===== 合并旧egolife视觉层 → egolife_fp (A2/A3/A4) =====", flush=True)
    for p in PEOPLE:
        merge_person(p)
    print("===== 合并完成; A1/A5/A6 egolife_fp已自带visual =====", flush=True)
