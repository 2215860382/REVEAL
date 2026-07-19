"""Oracle + VISION probe: does adding the gold segment's frames break past the
text-only oracle caption ceiling (68.5%)?

Same gold-window localization as oracle_egolife.py, but also loads the ORIGIN
(1408px) keyframe of each gold chunk and feeds them to the answerer alongside the
caption text. Split by need_audio (vision cannot fix audio-only questions).

If accuracy > text-oracle on the VISUAL (need_audio=False) subset ⇒ the visual /
online layer, aimed at the GOLD segment, recovers caption-omitted micro-facts.
"""
from __future__ import annotations
import os, sys, json, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataloader.egolife import load_egolife, _clip_abs_start
from src.build_memory.core.bank_loader import load_bank
from src.agents.answerer import Answerer
from src.clients.vlm_client import VLMClient
from experiments.core._keyframes import keyframe_paths, load_keyframe_pil

ROOT   = Path("/home2/ycj/Project/REVEAL")
MEMDIR = ROOT / "outputs/memory/egolife"
PERSON = "A1_JAKE"
W      = int(os.environ.get("W", "2"))
WORKERS= int(os.environ.get("WORKERS", "6"))
API    = os.environ["API"]
ORIGIN = os.environ.get("ORIGIN", "1") == "1"   # use 1408 origin frames vs 448 resized
OUT    = ROOT / f"outputs/results/egolife/oracle_vision_a1_w{W}.jsonl"


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

def load_frames(bank, chunks):
    imgs=[]
    for c in chunks:
        for p in keyframe_paths(MEMDIR, bank.video_id, c.chunk_id, cap=1, chunk=c):
            if ORIGIN:
                p = p.replace("keyframes_resized", "keyframes_origin")
            im=load_keyframe_pil(p)
            if im is not None: imgs.append(im); break
    return imgs

def main():
    samples=load_egolife(bench="egolifeqa",bench_root="/home2/ycj/Datas/EgoLife_bench",
                         video_dir="/home2/ycj/Datas/EgoLife",filter_people=[PERSON])
    bank=load_bank(MEMDIR/PERSON)
    vlm=VLMClient(model_path="/home2/ycj/Models/Qwen/Qwen3.5-27B",api_url=API,api_model="Qwen3.5-27B")
    ans=Answerer(vlm)
    print(f"[oracle-vision] W={W} origin={ORIGIN} n={len(samples)}",flush=True)
    fh=OUT.open("w"); lock=threading.Lock(); st={"n":0,"c":0}

    def run(s):
        t=_abs(s.target_time)
        if t is None: return
        gw=gold_window(bank,t,_abs(s.query_time))
        ev=[_l1_ev(c) for c in gw]
        frames=load_frames(bank,gw)
        res=ans.answer(s.question,s.candidates,ev,keyframe_images=frames,question_first=True)
        letter=(res.get("answer","") or "")[:1].upper()
        idx=ord(letter)-ord("A") if letter else -1
        pred=s.candidates[idx] if 0<=idx<len(s.candidates) else ""
        ok=(pred==s.answer)
        with lock:
            st["n"]+=1; st["c"]+=int(ok)
            fh.write(json.dumps({"sample_idx":s.sample_idx,"correct":ok,"question_type":s.question_type,
                "need_audio":bool(getattr(s,"need_audio",False)),"n_frames":len(frames),
                "pred":pred,"gold":s.answer},ensure_ascii=False)+"\n"); fh.flush()
            if st["n"]%25==0: print(f"  {st['n']} acc={100*st['c']/st['n']:.1f}%",flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(run,samples))
    fh.close()
    print(f"[oracle-vision] DONE {st['c']}/{st['n']} = {100*st['c']/max(1,st['n']):.2f}%",flush=True)

if __name__=="__main__": main()
