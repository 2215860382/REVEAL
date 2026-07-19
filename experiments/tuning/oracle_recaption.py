"""Online RE-CAPTION probe (2-stage) — the user's actual "online layer fills
caption" proposal: on the gold segment, extract dense frames, have the VLM WRITE
a QUESTION-FOCUSED description of them (attention on the asked micro-fact), then
answer text-only from (original captions + the new focused description).

Distinct from oracle_online.py which dumped frames on the MCQ answerer. Here a
dedicated perception step may surface a fact the answerer wouldn't while juggling
many images. Visual subset (need_audio=False); compare to text-oracle 60.3%.
"""
from __future__ import annotations
import os, sys, json, threading, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataloader.egolife import load_egolife, _clip_abs_start
from src.build_memory.core.bank_loader import load_bank
from src.agents.answerer import Answerer
from src.clients.vlm_client import VLMClient

ROOT=Path("/home2/ycj/Project/REVEAL"); MEMDIR=ROOT/"outputs/memory/egolife"; VIDDIR=Path("/home2/ycj/Datas/EgoLife")
PERSON="A1_JAKE"; W=int(os.environ.get("W","1")); NPC=int(os.environ.get("NPC","4"))
MAXSIDE=int(os.environ.get("MAXSIDE","896")); WORKERS=int(os.environ.get("WORKERS","6")); API=os.environ["API"]
OUT=ROOT/"outputs/results/egolife/oracle_recaption_a1_visual.jsonl"
_clip_re=re.compile(r"(DAY\d+_[A-Z0-9_]+?_\d{8})_\d+_\d+\.jpg$")

def _abs(tt):
    if not tt or not tt.get("time") or not tt.get("date"): return None
    try: return _clip_abs_start(int(str(tt["date"]).upper().replace("DAY","")),str(tt["time"]))
    except Exception: return None
def _l1_ev(c):
    p=[f"[{c.start_time:.0f}s-{c.end_time:.0f}s]",c.memory_text]
    if getattr(c,"asr","").strip(): p.append(f"Speech: {c.asr}")
    return "\n".join(p)
def gold_window(bank,t,cut):
    ch=sorted(bank.chunks,key=lambda c:c.start_time)
    if cut is not None: ch=[c for c in ch if c.end_time<=cut+1]
    if not ch: return []
    cov=next((i for i,c in enumerate(ch) if c.start_time<=t<=c.end_time),None)
    if cov is None: cov=min(range(len(ch)),key=lambda i:abs((ch[i].start_time+ch[i].end_time)/2-t))
    return ch[max(0,cov-W):min(len(ch),cov+W+1)]
def clip_mp4(c):
    m=_clip_re.search(Path(getattr(c,"keyframe_path","") or "").name)
    if not m: return None
    s=m.group(1); return VIDDIR/PERSON/s.split("_")[0]/f"{s}.mp4"
def dense(mp4,k):
    if not mp4 or not mp4.exists(): return []
    cap=cv2.VideoCapture(str(mp4)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n<=0: cap.release(); return []
    out=[]
    for i in [int(x) for x in np.linspace(0,n-1,k)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read()
        if ok:
            im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)); im.thumbnail((MAXSIDE,MAXSIDE)); out.append(im)
    cap.release(); return out

RECAP=("These frames are from the exact moment relevant to a question about an egocentric "
       "life-log. Question: {q}\nDescribe in detail EVERYTHING visible that could help answer it: "
       "who is doing what, hand-object interactions, order of actions, and any TEXT on whiteboards, "
       "phone/computer screens, or paper. Be specific with names if identifiable. 3-6 sentences.")

def main():
    samples=[s for s in load_egolife(bench="egolifeqa",bench_root="/home2/ycj/Datas/EgoLife_bench",
             video_dir=str(VIDDIR),filter_people=[PERSON]) if not bool(getattr(s,"need_audio",False))]
    bank=load_bank(MEMDIR/PERSON)
    vlm=VLMClient(model_path="/home2/ycj/Models/Qwen/Qwen3.5-27B",api_url=API,api_model="Qwen3.5-27B")
    ans=Answerer(vlm)
    print(f"[recap] visual n={len(samples)} W={W} NPC={NPC}",flush=True)
    fh=OUT.open("w"); lock=threading.Lock(); st={"n":0,"c":0}
    def run(s):
        t=_abs(s.target_time)
        if t is None: return
        gw=gold_window(bank,t,_abs(s.query_time)); ev=[_l1_ev(c) for c in gw]
        frames=[]
        for c in gw: frames+=dense(clip_mp4(c),NPC)
        focused=""
        if frames:
            try: focused=vlm.chat_with_frames(frames, RECAP.format(q=s.question), max_new_tokens=256)
            except Exception as e: focused=f"(recap failed: {e})"
        ev2=ev+([f"[FOCUSED VISUAL DESCRIPTION]\n{focused}"] if focused else [])
        res=ans.answer(s.question,s.candidates,ev2,question_first=True)
        letter=(res.get("answer","") or "")[:1].upper(); idx=ord(letter)-ord("A") if letter else -1
        pred=s.candidates[idx] if 0<=idx<len(s.candidates) else ""; ok=(pred==s.answer)
        with lock:
            st["n"]+=1; st["c"]+=int(ok)
            fh.write(json.dumps({"sample_idx":s.sample_idx,"correct":ok,"question_type":s.question_type,
                "n_frames":len(frames),"focused":focused[:400],"pred":pred,"gold":s.answer},ensure_ascii=False)+"\n"); fh.flush()
            if st["n"]%20==0: print(f"  {st['n']} acc={100*st['c']/st['n']:.1f}%",flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(run,samples))
    fh.close()
    print(f"[recap] DONE {st['c']}/{st['n']} = {100*st['c']/max(1,st['n']):.2f}%  (视觉题文本oracle=60.3%)",flush=True)
if __name__=="__main__": main()
