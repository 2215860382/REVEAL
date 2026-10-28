"""Stage 2: load cached SigLIP vecs and re-group at a given theta/n_max, then
export one representative (center) frame per group into a browsable dir named by
group index + time window, so a human can verify "one group == one scene".

usage: group_regroup.py <stem> <theta> [n_max] [--export]
"""
import sys, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home2/ycj/Project/REVEAL")
from src.build_memory.core.frame_pipeline import group_frames

STEM  = sys.argv[1]
THETA = float(sys.argv[2])
NMAX  = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 30
EXPORT = "--export" in sys.argv

OUT = Path(f"/home2/ycj/Project/REVEAL/outputs/group_probe/{STEM}")
c = np.load(OUT / "cache.npz", allow_pickle=True)
v, ts, paths = c["vecs"], list(c["ts"]), list(c["paths"])

groups = group_frames(v, ts, theta=THETA, n_max=NMAX)
sizes = [g.size for g in groups]

def hms(s):
    s = int(s); return f"{s//3600:02d}{(s%3600)//60:02d}{s%60:02d}"

print(f"theta={THETA} n_max={NMAX}  ->  {len(groups)} groups")
print(f"  size: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.2f} "
      f"median={int(np.median(sizes))}  single={sum(1 for s in sizes if s==1)}")

if EXPORT:
    d = OUT / f"groups_theta{THETA}_nmax{NMAX}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for gi, g in enumerate(groups):
        c_path = paths[g.center_idx]
        dst = d / f"g{gi:04d}_{hms(g.t_start)}-{hms(g.t_end)}_n{g.size}.jpg"
        shutil.copyfile(c_path, dst)
    print(f"  exported {len(groups)} keyframes -> {d}")
