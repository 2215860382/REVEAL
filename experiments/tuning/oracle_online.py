"""Online-layer probe: re-extract DENSE high-res frames from the raw mp4s of the
gold segment (NOT the 1 stored keyframe/chunk) and feed them to the answerer.

This is the true "online memory" test — on the gold window we go back to the raw
video and look densely, recovering micro-facts the 30s scene-caption dropped
(who wrote the whiteboard, the menu on screen, who acted first).

Runs on the VISUAL subset (need_audio=False) where text-oracle tops out at 60.3%;
audio questions already sit at 80% from the fused ASR so vision can't add there.
Compare the printed number to that 60.3% text-oracle visual baseline.
"""
from __future__ import annotations
import os, sys, json, threading, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataloader.egolife import load_egolife, _clip_abs_start
from src.build_memory.core.bank_loader import load_bank
from src.agents.answerer import Answerer
from src.clients.vlm_client import VLMClient
import cv2
from PIL import Image

ROOT   = Path("/home2/ycj/Project/VEIL")
MEMDIR = ROOT / "outputs/memory/egolife"
VIDDIR = Path("/home2/ycj/Datas/EgoLife")
PERSON = "A1_JAKE"
W      = int(os.environ.get("W", "1"))          # window half-width (chunks)
NPC    = int(os.environ.get("NPC", "4"))         # dense frames per chunk
MAXSIDE= int(os.environ.get("MAXSIDE", "896"))
WORKERS= int(os.environ.get("WORKERS", "6"))
SUBSET = os.environ.get("SUBSET", "visual")      # visual | all
API    = os.environ["API"]
OUT    = ROOT / f"outputs/results/egolife/oracle_online_a1_{SUBSET}.jsonl"

_clip_re = re.compile(r"(DAY\d+_[A-Z0-9_]+?_\d{8})_\d+_\d+\.jpg$")


def _abs(tt):
    if not tt or not tt.get("time") or not tt.get("date"): return None
    try: return _clip_abs_start(int(str(tt["date"]).upper().replace("DAY","")), str(tt["time"]))
    except Exception: return None

def _l1_ev(c):
    parts=[f"[{c.start_time:.0f}s-{c.end_time:.0f}s]", c.memory_text]
    if getattr(c,"asr","").strip(): parts.append(f"Speech: {c.asr}")
    return "\n".join(parts)

def gold_window(bank, t, cutoff):
    ch=sorted(bank.chunks,key=lambda c:c.start_time)
    if cutoff is not None: ch=[c for c in ch if c.end_time<=cutoff+1]
    if not ch: return []
    cov=next((i for i,c in enumerate(ch) if c.start_time<=t<=c.end_time), None)
    if cov is None: cov=min(range(len(ch)),key=lambda i:abs((ch[i].start_time+ch[i].end_time)/2-t))
    return ch[max(0,cov-W):min(len(ch),cov+W+1)]

def clip_mp4(chunk):
    kp = getattr(chunk,"keyframe_path","") or ""
    m=_clip_re.search(Path(kp).name)
    if not m: return None
    stem=m.group(1)                      # DAY1_A1_JAKE_11094208
    day=stem.split("_")[0]               # DAY1
    return VIDDIR / PERSON / day / f"{stem}.mp4"

def dense_frames(mp4, k):
    if not mp4 or not mp4.exists(): return []
    cap=cv2.VideoCapture(str(mp4))
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n<=0: cap.release(); return []
    out=[]
    for i in [int(x) for x in np.linspace(0,n-1,k)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read()
        if not ok: continue
        im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
        im.thumbnail((MAXSIDE,MAXSIDE))
        out.append(im)
    cap.release(); return out

def main():
    samples=load_egolife(bench="egolifeqa",bench_root="/home2/ycj/Datas/EgoLife_bench",
                         video_dir=str(VIDDIR),filter_people=[PERSON])
    if SUBSET=="visual":
        samples=[s for s in samples if not bool(getattr(s,"need_audio",False))]
    bank=load_bank(MEMDIR/PERSON)
    vlm=VLMClient(model_path="/home2/ycj/Models/Qwen/Qwen3.5-27B",api_url=API,api_model="Qwen3.5-27B")
    ans=Answerer(vlm)
    print(f"[online] subset={SUBSET} n={len(samples)} W={W} frames/chunk={NPC} maxside={MAXSIDE}",flush=True)
    fh=OUT.open("w"); lock=threading.Lock(); st={"n":0,"c":0,"nf":0}

    def run(s):
        t=_abs(s.target_time)
        if t is None: return
        gw=gold_window(bank,t,_abs(s.query_time))
        ev=[_l1_ev(c) for c in gw]
        frames=[]
        for c in gw: frames+=dense_frames(clip_mp4(c),NPC)
        res=ans.answer(s.question,s.candidates,ev,keyframe_images=frames,question_first=True)
        letter=(res.get("answer","") or "")[:1].upper()
        idx=ord(letter)-ord("A") if letter else -1
        pred=s.candidates[idx] if 0<=idx<len(s.candidates) else ""
        ok=(pred==s.answer)
        with lock:
            st["n"]+=1; st["c"]+=int(ok); st["nf"]+=len(frames)
            fh.write(json.dumps({"sample_idx":s.sample_idx,"correct":ok,"question_type":s.question_type,
                "n_frames":len(frames),"pred":pred,"gold":s.answer},ensure_ascii=False)+"\n"); fh.flush()
            if st["n"]%20==0: print(f"  {st['n']} acc={100*st['c']/st['n']:.1f}% avgframes={st['nf']/st['n']:.1f}",flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(run,samples))
    fh.close()
    print(f"[online] DONE {st['c']}/{st['n']} = {100*st['c']/max(1,st['n']):.2f}%  (视觉题文本oracle基线=60.3%)",flush=True)

if __name__=="__main__": main()
