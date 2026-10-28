"""Compare adjacent-frame grouping vs anchor-based grouping for scene detection.

anchor mode: start a new group when a frame's cosine to the group's ANCHOR
(first frame) drops below theta -> resists gradual drift across scene cuts.
"""
import sys, shutil
from pathlib import Path
import numpy as np

STEM  = sys.argv[1]
THETA = float(sys.argv[2])
NMAX  = int(sys.argv[3]) if len(sys.argv) > 3 else 300
MODE  = sys.argv[4] if len(sys.argv) > 4 else "anchor"   # anchor | adjacent
EXPORT = "--export" in sys.argv

OUT = Path(f"/home2/ycj/Project/REVEAL/outputs/group_probe/{STEM}")
c = np.load(OUT / "cache.npz", allow_pickle=True)
v, ts, paths = c["vecs"], list(c["ts"]), list(c["paths"])
T = len(v)

groups, current, anchor = [], [0], v[0]
for t in range(1, T):
    ref = anchor if MODE == "anchor" else v[t-1]
    if float(v[t] @ ref) >= THETA and len(current) < NMAX:
        current.append(t)
    else:
        groups.append(current); current = [t]; anchor = v[t]
groups.append(current)

sizes = [len(g) for g in groups]
def hms(s): s=int(s); return f"{s//3600:02d}{(s%3600)//60:02d}{s%60:02d}"
print(f"[{MODE}] theta={THETA} n_max={NMAX} -> {len(groups)} groups | "
      f"size min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f} "
      f"median={int(np.median(sizes))} single={sum(1 for s in sizes if s==1)}")

if EXPORT:
    d = OUT / f"grp_{MODE}_th{THETA}_nmax{NMAX}"
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    for gi, g in enumerate(groups):
        center = g[len(g)//2]
        t0, t1 = ts[g[0]], ts[g[-1]] + 1
        shutil.copyfile(paths[center], d / f"g{gi:04d}_{hms(t0)}-{hms(t1)}_n{len(g)}.jpg")
    print(f"  exported -> {d}")
