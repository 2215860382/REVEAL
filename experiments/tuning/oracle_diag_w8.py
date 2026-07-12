"""Diagnose the oracle ceiling: for EgoLifeQA questions the oracle got WRONG,
split the cause into
  (1) EVIDENCE-INSUFFICIENT — the gold-time caption genuinely lacks the answer
      (bank/caption cap → needs a richer rebuild), vs
  (2) PROMPT/MODEL — the answer IS derivable from the fed evidence, the answerer
      still picked wrong (answerer prompt/reasoning cap → cheap to fix).

For each wrong question we rebuild the SAME W-window gold evidence, then ask a
judge LLM (same 27B) to decide, given ONLY that evidence, whether the gold answer
is supported and which option the evidence points to.
"""
from __future__ import annotations
import os, sys, json, re, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataloader.egolife import load_egolife, _clip_abs_start
from src.build_memory.core.bank_loader import load_bank
from src.clients.llm_client import LLMClient

ROOT       = Path("/home2/ycj/Project/VEIL")
MEMDIR     = ROOT / "outputs/memory/egolife"
PERSON     = "A1_JAKE"
W          = int(os.environ.get("W", "2"))
WORKERS    = int(os.environ.get("WORKERS", "6"))
API        = os.environ["API"]
ORACLE     = ROOT / "outputs/results/egolife/oracle_a1_w2.jsonl"
OUT        = ROOT / "outputs/results/egolife/oracle_diag_a1_w8.jsonl"


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

JUDGE=(
 "You are auditing whether a piece of evidence is SUFFICIENT to answer a multiple-choice "
 "question. You are told the CORRECT answer. Decide, using ONLY the evidence below, whether "
 "the correct answer can be derived.\n\n"
 "Question: {q}\nOptions:\n{opts}\nCORRECT answer: {gold}\n\nEvidence:\n{ev}\n\n"
 "Reply with a JSON object: {{\"verdict\": \"SUPPORTED\"|\"NOT_SUPPORTED\", \"points_to\": "
 "\"<the option letter the evidence best supports, or NONE>\", \"why\": \"<=20 words}}\n"
 "SUPPORTED = the evidence contains enough to conclude the correct answer. "
 "NOT_SUPPORTED = the evidence does not contain the needed fact (would require info not present)."
)

def main():
    orc={r["sample_idx"]:r for r in (json.loads(l) for l in open(ORACLE) if l.strip())}
    wrong={k for k,r in orc.items() if not r["correct"]}
    samples=[s for s in load_egolife(bench="egolifeqa",bench_root="/home2/ycj/Datas/EgoLife_bench",
             video_dir="/home2/ycj/Datas/EgoLife",filter_people=[PERSON]) if s.sample_idx in wrong]
    bank=load_bank(MEMDIR/PERSON)
    llm=LLMClient(model_path="Qwen3.5-27B",api_url=API,api_model="Qwen3.5-27B")
    print(f"[diag] oracle-wrong={len(wrong)} 可诊断={len(samples)}",flush=True)
    fh=OUT.open("w"); lock=threading.Lock(); cnt={"SUP":0,"NOT":0,"err":0}

    def run(s):
        t=_abs(s.target_time); cutoff=_abs(s.query_time)
        gw=gold_window(bank,t,cutoff) if t is not None else []
        ev="\n".join(_l1_ev(c) for c in gw) or "(none)"
        opts="\n".join(f"({chr(65+i)}) {c}" for i,c in enumerate(s.candidates))
        gl=chr(65+s.candidates.index(s.answer)) if s.answer in s.candidates else "?"
        raw=llm.chat([{"role":"user","content":JUDGE.format(q=s.question,opts=opts,gold=f"({gl}) {s.answer}",ev=ev[:8000])}],
                     max_new_tokens=120, enable_thinking=False)
        m=re.search(r'\{.*\}', raw, re.S); v="err"; pt=""
        if m:
            try: j=json.loads(m.group()); v=j.get("verdict","err"); pt=j.get("points_to","")
            except Exception: pass
        with lock:
            if v=="SUPPORTED": cnt["SUP"]+=1
            elif v=="NOT_SUPPORTED": cnt["NOT"]+=1
            else: cnt["err"]+=1
            fh.write(json.dumps({"sample_idx":s.sample_idx,"qtype":s.question_type,"verdict":v,
                "points_to":pt,"gold_letter":gl,"question":s.question,"gold":s.answer,
                "n_gold":len(gw),"evidence":ev[:4000]},ensure_ascii=False)+"\n"); fh.flush()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(run,samples))
    fh.close()
    n=cnt["SUP"]+cnt["NOT"]
    print(f"[diag] SUPPORTED(证据够,模型/prompt锅)={cnt['SUP']}  NOT_SUPPORTED(证据不足,库锅)={cnt['NOT']}  err={cnt['err']}",flush=True)
    if n: print(f"   → oracle错题里 {100*cnt['SUP']/n:.0f}% 是prompt锅 / {100*cnt['NOT']/n:.0f}% 是库锅",flush=True)

if __name__=="__main__": main()
