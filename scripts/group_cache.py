"""Stage 1: extract frames @fps=1, resize to 448 long-edge (canonical,
resolution-invariant), SigLIP-encode once, cache vectors + timestamps + the
448 frames to a persistent dir so we can re-group at any theta instantly.
"""
import sys, subprocess, tempfile, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, "/home2/ycj/Project/REVEAL")
from src.build_memory.core.frame_pipeline import is_blank, is_blank_pixel
from src.clients.siglip_embedder import SigLIPEmbedder

VID   = sys.argv[1]
STEM  = Path(VID).stem
FPS   = 1.0
CANON = 448
GPU   = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
MODEL = "/home2/ycj/Models/google/siglip-large-patch16-384"
OUT   = Path(f"/home2/ycj/Project/REVEAL/outputs/group_probe/{STEM}")
FR448 = OUT / "frames448"
OUT.mkdir(parents=True, exist_ok=True); FR448.mkdir(exist_ok=True)

print(f"video={VID} canon={CANON} gpu={GPU}\nout={OUT}", flush=True)
t0 = time.time()
siglip = SigLIPEmbedder(MODEL, device=GPU)
print(f"[+] SigLIP loaded ({time.time()-t0:.1f}s)", flush=True)

with tempfile.TemporaryDirectory(prefix="gc_", dir="/dev/shm") as tmp:
    tmp = Path(tmp)
    t = time.time()
    subprocess.run(["ffmpeg", "-i", VID, "-vf", f"fps={FPS}", "-q:v", "2",
                    str(tmp/"f_%06d.jpg"), "-hide_banner", "-loglevel", "error", "-y"],
                   check=True)
    raw = sorted(tmp.glob("f_*.jpg"))
    print(f"[1] extracted {len(raw)} frames ({time.time()-t:.1f}s)", flush=True)

    # blank filter only (NO blur filter here — keep every frame so grouping/theta
    # tuning is not confounded by resolution-dependent blur drops).
    keep = [(i, p) for i, p in enumerate(raw)
            if not is_blank(p) and not is_blank_pixel(p)]
    print(f"[2] after blank filter: {len(keep)} (dropped {len(raw)-len(keep)})", flush=True)

    # resize to 448 long-edge, persist for viewing + encoding
    t = time.time()
    paths448, ts = [], []
    for i, p in keep:
        img = Image.open(p).convert("RGB")
        if max(img.size) > CANON:
            img.thumbnail((CANON, CANON), Image.LANCZOS)
        dst = FR448 / f"{i:06d}.jpg"       # name = original frame index = second
        img.save(dst, "JPEG", quality=90)
        paths448.append(str(dst)); ts.append(i / FPS)
    print(f"[3] resized+saved {len(paths448)} @{CANON} ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    v = siglip.encode_images(paths448)
    print(f"[4] SigLIP encoded {v.shape} ({time.time()-t:.1f}s)", flush=True)

np.savez(OUT / "cache.npz", vecs=v.astype(np.float32),
         ts=np.array(ts, dtype=np.float32),
         paths=np.array(paths448))
print(f"[+] cached → {OUT/'cache.npz'}  ({len(paths448)} frames)", flush=True)
print("DONE", flush=True)
