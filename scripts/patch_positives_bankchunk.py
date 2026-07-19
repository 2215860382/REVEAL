#!/usr/bin/env python
"""In-place patch: replace each target_time POSITIVE chain's evidence (a narrow VLM
recaption that dropped the answer context) with the ACTUAL bank chunk(s) covering
target_time — the timestamped gold evidence. Negatives and pairs are left untouched, so
this isolates the positive-label fix and needs no embed/VLM service (bank is local).

Usage: patch_positives_bankchunk.py <config.yaml> <bench> <chains.jsonl> [window_w]
"""
import sys, json, yaml
from pathlib import Path
from src.dataloader.egolife import load_egolife
from src.build_memory.core.bank_loader import load_bank
from src.generate_rubric.construct.contrastive_trace.construct_targettime_pairs import (
    abs_range, gold_window, _cutoff)
from src.generate_rubric.construct.contrastive_trace.sample_chains import _evidence_items

CFG_PATH, BENCH, CHAINS = sys.argv[1], sys.argv[2], sys.argv[3]
W = int(sys.argv[4]) if len(sys.argv) > 4 else 1

cfg = yaml.safe_load(open(CFG_PATH)); b = cfg["benchmark"]
samples = load_egolife(bench=BENCH, bench_root=b["bench_root"],
                       video_dir=b.get("video_dir"), filter_people=b.get("filter_people"))
by_idx = {s.sample_idx: s for s in samples}

_bank, _cmap = {}, {}
def bank_cmap(vid):
    if vid not in _bank:
        try:
            bk = load_bank(f"outputs/memory/egolife/{vid}")
            _bank[vid] = bk
            _cmap[vid] = {int(c.chunk_id): c for c in bk.chunks}
        except Exception as e:
            print(f"  bank load failed {vid}: {e}")
            _bank[vid] = None; _cmap[vid] = None
    return _bank[vid], _cmap[vid]

rows = [json.loads(l) for l in open(CHAINS)]
patched = missing = 0
out = []
for ch in rows:
    is_pos = str(ch.get("chain_id", "")).endswith(":pos") or ch.get("sufficient") is True
    if not is_pos:
        out.append(ch); continue
    s = by_idx.get(int(ch["sample_idx"]))
    bank, cmap = bank_cmap(ch["video_id"])
    if s is None or bank is None:
        out.append(ch); missing += 1; continue
    rng = abs_range(s.target_time)
    if rng is None:
        out.append(ch); missing += 1; continue
    t_abs = (rng[0] + rng[1]) / 2.0
    gw = gold_window(bank, t_abs, _cutoff(s), W)
    items = _evidence_items(cmap, [int(c.chunk_id) for c in gw])
    if not items:
        out.append(ch); missing += 1; continue
    ch["evidence"] = [it.to_dict() for it in items]
    ch["source"] = "targettime_bankchunk"
    ch["notes"] = "bank chunk(s) covering target_time"
    ch["n_chunks"] = len(items)
    out.append(ch); patched += 1

bak = CHAINS + ".bak_recaption"
if not Path(bak).exists():
    Path(bak).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"backed up original -> {bak}")
with open(CHAINS, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"patched positives: {patched} | skipped(missing bank/target): {missing} | total chains: {len(out)}")
