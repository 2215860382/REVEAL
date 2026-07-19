"""HTTP server hosting BGE-M3 + SigLIP — load once, share across experiments.

Usage:
    PYTHONPATH=. python scripts/start_embed_server.py \\
        --port 9000 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clients.embedder import BGEM3Embedder
from src.clients.siglip_embedder import SigLIPEmbedder

app = FastAPI(title="REVEAL Embedding Server")

BGE: BGEM3Embedder | None = None
SIGLIP: SigLIPEmbedder | None = None

# Idle self-termination: track the last real (non-health) request; a watchdog
# thread kills the process after --idle-timeout seconds of silence so the GPU
# isn't held once an experiment is done. /health pings do NOT count as activity.
LAST_REQ: float = time.monotonic()


@app.middleware("http")
async def _track_activity(request: Request, call_next):
    global LAST_REQ
    if request.url.path != "/health":
        LAST_REQ = time.monotonic()
    return await call_next(request)


def _idle_watchdog(timeout: float) -> None:
    while True:
        time.sleep(min(15.0, max(1.0, timeout / 4)))
        idle = time.monotonic() - LAST_REQ
        if idle >= timeout:
            print(f"[embed-server] idle {idle:.0f}s ≥ {timeout:.0f}s — self-terminating to free GPU",
                  flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
            return


class TextRequest(BaseModel):
    texts: List[str]


class PathRequest(BaseModel):
    paths: List[str]


@app.get("/health")
def health():
    return {"status": "ok", "bge": BGE is not None, "siglip": SIGLIP is not None}


@app.post("/bge/encode")
def bge_encode(req: TextRequest):
    assert BGE is not None
    return {"vectors": BGE.encode(req.texts).tolist()}


@app.post("/siglip/encode_images")
def siglip_encode_images(req: PathRequest):
    assert SIGLIP is not None
    return {"vectors": SIGLIP.encode_images(req.paths).tolist()}


@app.post("/siglip/encode_text")
def siglip_encode_text(req: TextRequest):
    assert SIGLIP is not None
    return {"vectors": SIGLIP.encode_text(req.texts).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bge-model", default="/home2/ycj/Models/BAAI/bge-m3")
    ap.add_argument("--siglip-model", default="/home2/ycj/Models/google/siglip-large-patch16-384")
    ap.add_argument("--no-fp16", action="store_true", help="load BGE in fp32 (avoids NaN/Inf on some GPUs)")
    ap.add_argument("--idle-timeout", type=float, default=300.0,
                    help="self-kill after N seconds with no (non-health) request; 0 disables. Default 300 (5 min).")
    args = ap.parse_args()

    global BGE, SIGLIP, LAST_REQ
    print(f"[embed-server] loading BGE-M3 on {args.device} (fp16={not args.no_fp16})...", flush=True)
    BGE = BGEM3Embedder(model_path=args.bge_model, use_fp16=not args.no_fp16, device=args.device)
    print(f"[embed-server] loading SigLIP on {args.device}...", flush=True)
    SIGLIP = SigLIPEmbedder(model_path=args.siglip_model, device=args.device)
    LAST_REQ = time.monotonic()  # reset so load time doesn't count toward idle
    if args.idle_timeout > 0:
        threading.Thread(target=_idle_watchdog, args=(args.idle_timeout,), daemon=True).start()
        print(f"[embed-server] idle-watchdog armed: self-kill after {args.idle_timeout:.0f}s idle", flush=True)
    print(f"[embed-server] ready, listening on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
