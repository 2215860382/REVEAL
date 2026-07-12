"""Why is the gold chunk buried? For questions where the gold (target_time chunk)
ranks DEEP under dense retrieval, print question+options+gold caption+asr, plus
where the answer text and option tokens actually sit, to characterize the
question<->gold relationship and design a retrieval that surfaces it.
"""
from __future__ import annotations
import os, sys, re
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataloader.egolife import load_egolife, _clip_abs_start
from src.build_memory.core.bank_loader import load_bank
from src.clients.embedder import BGEM3Embedder

MEMDIR=Path("/home2/ycj/Project/VEIL/outputs/memory/egolife"); PERSON="A1_JAKE"

def _abs(tt):
    if not tt or not tt.get("time") or not tt.get("date"): return None
    try: return _clip_abs_start(int(str(tt["date"]).upper().replace("DAY","")),str(tt["time"]))
    except Exception: return None

def main():
    samples=load_egolife(bench="egolifeqa",bench_root="/home2/ycj/Datas/EgoLife_bench",
                         video_dir="/home2/ycj/Datas/EgoLife",filter_people=[PERSON])
    bank=load_bank(MEMDIR/PERSON); ch=bank.chunks
    starts=np.array([c.start_time for c in ch]); ends=np.array([c.end_time for c in ch])
    docmat=np.array([c.v_text for c in ch],dtype=np.float32)
    emb=BGEM3Embedder(model_path="/home2/ycj/Models/BAAI/bge-m3",use_fp16=True,device=os.environ.get("BGE_GPU","cuda:1"))
    shown=0
    for s in samples:
        t=_abs(s.target_time)
        if t is None: continue
        cutoff=_abs(s.query_time); mask=ends<=cutoff+1 if cutoff is not None else np.ones(len(ch),bool)
        cover=np.where((starts<=t)&(t<=ends))[0]
        gi=int(cover[0]) if len(cover) else int(np.argmin(np.abs((starts+ends)/2-t)))
        if not mask[gi]: continue
        qv=emb.encode([s.question])[0]; dense=np.where(mask,docmat@qv,-np.inf)
        rank=int((dense>dense[gi]).sum())+1
        if rank<40: continue           # only DEEP-buried gold
        g=ch[gi]; cap=g.memory_text; asr=getattr(g,"asr","") or ""
        ans=s.answer
        # does the answer / options text appear literally in gold caption+asr?
        blob=(cap+" "+asr).lower()
        ans_in="✓" if ans.lower() in blob else "✗"
        opt_hits=[o for o in s.candidates if o.lower() in blob]
        print("="*90)
        print(f"[{s.question_type}] rank={rank}  Q: {s.question}")
        print(f"  选项: {s.candidates}   gold答案: {ans}  (答案文本在gold里? {ans_in};  命中选项: {opt_hits})")
        print(f"  gold caption: {cap[:600]}")
        if asr.strip(): print(f"  gold speech: {asr[:300]}")
        shown+=1
        if shown>=12: break
    print(f"\n[共展示 {shown} 道深埋题]")

if __name__=="__main__": main()
