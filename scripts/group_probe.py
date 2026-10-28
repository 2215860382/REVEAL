"""Ad-hoc probe: run the real build pipeline's grouping on ONE video and report
how many SigLIP shot-groups (chunks) it produces. No bank persistence.

Mirrors clean_and_group_frames steps 1-5 exactly, but extracts to /dev/shm and
skips the frames_raw/keyframe disk writes — we only want the group count.
"""
import sys, time, tempfile
from pathlib import Path

sys.path.insert(0, "/home2/ycj/Project/REVEAL")

from src.build_memory.core.frame_pipeline import (
    extract_frames, is_blank, is_blank_pixel, is_blurry, group_frames,
    BLUR_FAILSAFE_RATIO, DEFAULT_GROUP_THETA, DEFAULT_GROUP_N_MAX,
)
from src.clients.siglip_embedder import SigLIPEmbedder

VIDEO = sys.argv[1]
FPS   = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
THETA = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_GROUP_THETA
NMAX  = int(sys.argv[4])   if len(sys.argv) > 4 else DEFAULT_GROUP_N_MAX
GPU   = sys.argv[5]        if len(sys.argv) > 5 else "cuda:1"
MODEL = "/home2/ycj/Models/google/siglip-large-patch16-384"

print(f"video={VIDEO}\nfps={FPS} theta={THETA} n_max={NMAX} gpu={GPU}\nmodel={MODEL}", flush=True)

t0 = time.time()
siglip = SigLIPEmbedder(MODEL, device=GPU)
print(f"[+] SigLIP loaded ({time.time()-t0:.1f}s)", flush=True)

with tempfile.TemporaryDirectory(prefix="grpprobe_", dir="/dev/shm") as tmp:
    tmp = Path(tmp)
    # 1. extract
    t = time.time()
    raw = extract_frames(VIDEO, tmp, fps=FPS)
    print(f"[1] extracted {len(raw)} frames ({time.time()-t:.1f}s)", flush=True)

    # 2. drop blank (bytes then pixel-std)
    after_size = [(i, p) for i, p in enumerate(raw) if not is_blank(p)]
    non_blank  = [(i, p) for i, p in after_size if not is_blank_pixel(p)]
    print(f"[2] after blank filter: {len(non_blank)} "
          f"(dropped {len(raw)-len(non_blank)})", flush=True)

    # 3. drop blurry with failsafe
    flags = [is_blurry(p) for _, p in non_blank]
    drop_ratio = sum(flags) / max(len(flags), 1)
    if drop_ratio > BLUR_FAILSAFE_RATIO:
        survived = non_blank
        print(f"[3] blur failsafe TRIGGERED (drop_ratio={drop_ratio:.2f}>0.5) "
              f"→ blur filter skipped", flush=True)
    else:
        survived = [pr for pr, bl in zip(non_blank, flags) if not bl]
        print(f"[3] after blur filter: {len(survived)} "
              f"(dropped {len(non_blank)-len(survived)}, ratio={drop_ratio:.2f})",
              flush=True)

    paths = [str(p) for _, p in survived]
    ts    = [i / FPS for i, _ in survived]

    # 4. SigLIP encode
    t = time.time()
    v = siglip.encode_images(paths)
    print(f"[4] SigLIP encoded {v.shape} ({time.time()-t:.1f}s)", flush=True)

    # 5. group
    groups = group_frames(v, ts, theta=THETA, n_max=NMAX)
    sizes = [g.size for g in groups]
    import numpy as np
    print("=" * 60, flush=True)
    print(f"RESULT: {len(groups)} groups (chunks)", flush=True)
    print(f"  group size: min={min(sizes)} max={max(sizes)} "
          f"mean={np.mean(sizes):.2f} median={int(np.median(sizes))}", flush=True)
    n_hitmax = sum(1 for s in sizes if s == NMAX)
    n_single = sum(1 for s in sizes if s == 1)
    print(f"  groups at n_max({NMAX}): {n_hitmax} | single-frame groups: {n_single}",
          flush=True)
    print("=" * 60, flush=True)
