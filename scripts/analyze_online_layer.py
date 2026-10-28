#!/usr/bin/env python3
"""
EXP-5: Analyze online layer contribution on triggered questions subset.
Expected: online layer provides larger gain on triggered questions than overall.
"""
import json
import sys
from pathlib import Path

def analyze_online_layer(reveal_jsonl, ablation_jsonl):
    """Compare REVEAL (with online) vs ablation (without online) on triggered subset."""
    
    # Load REVEAL results
    reveal_qs = {}
    with open(reveal_jsonl) as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                if 'correct' not in q:  # skip metadata records
                    continue
                qid = (q.get('video_id', ''), q.get('sample_idx', 0))
                reveal_qs[qid] = q

    # Load ablation results (without online layer)
    abl_qs = {}
    if ablation_jsonl and Path(ablation_jsonl).exists():
        with open(ablation_jsonl) as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    if 'correct' not in q:
                        continue
                    qid = (q.get('video_id', ''), q.get('sample_idx', 0))
                    abl_qs[qid] = q
    
    # Identify triggered questions
    triggered = []
    not_triggered = []
    
    for qid, rq in reveal_qs.items():
        # Check if online layer was activated (admitted new evidence)
        online_used = False
        if 'online_memory' in rq and isinstance(rq['online_memory'], dict):
            admitted = rq['online_memory'].get('admit', [])
            if admitted and len(admitted) > 0:
                online_used = True
        
        if online_used:
            triggered.append(qid)
        else:
            not_triggered.append(qid)
    
    print(f"===== EXP-5: Online Layer Triggered Analysis =====")
    print(f"Total questions: {len(reveal_qs)}")
    print(f"Online layer triggered: {len(triggered)} ({100*len(triggered)/len(reveal_qs):.1f}%)")
    print(f"Not triggered: {len(not_triggered)} ({100*len(not_triggered)/len(reveal_qs):.1f}%)")
    print()
    
    # Overall accuracy
    reveal_correct = sum(1 for q in reveal_qs.values() if q.get('correct', False))
    reveal_acc = 100 * reveal_correct / len(reveal_qs)
    print(f"REVEAL overall accuracy: {reveal_acc:.2f}% ({reveal_correct}/{len(reveal_qs)})")
    
    if abl_qs:
        abl_correct = sum(1 for q in abl_qs.values() if q.get('correct', False))
        abl_acc = 100 * abl_correct / len(abl_qs)
        print(f"Ablation (no online) overall accuracy: {abl_acc:.2f}% ({abl_correct}/{len(abl_qs)})")
        print(f"Overall gain from online layer: {reveal_acc - abl_acc:+.2f}pt")
        print()
    
    # Triggered subset accuracy
    if triggered:
        rev_trig_correct = sum(1 for qid in triggered if reveal_qs[qid].get('correct', False))
        rev_trig_acc = 100 * rev_trig_correct / len(triggered)
        print(f"REVEAL on triggered subset: {rev_trig_acc:.2f}% ({rev_trig_correct}/{len(triggered)})")
        
        if abl_qs:
            abl_trig_correct = sum(1 for qid in triggered if qid in abl_qs and abl_qs[qid].get('correct', False))
            abl_trig_acc = 100 * abl_trig_correct / len(triggered)
            print(f"Ablation on triggered subset: {abl_trig_acc:.2f}% ({abl_trig_correct}/{len(triggered)})")
            print(f"★ Gain on triggered subset: {rev_trig_acc - abl_trig_acc:+.2f}pt")
            print()
    
    # Not triggered subset
    if not_triggered:
        rev_not_correct = sum(1 for qid in not_triggered if reveal_qs[qid].get('correct', False))
        rev_not_acc = 100 * rev_not_correct / len(not_triggered)
        print(f"REVEAL on not-triggered subset: {rev_not_acc:.2f}% ({rev_not_correct}/{len(not_triggered)})")
        
        if abl_qs:
            abl_not_correct = sum(1 for qid in not_triggered if qid in abl_qs and abl_qs[qid].get('correct', False))
            abl_not_acc = 100 * abl_not_correct / len(not_triggered)
            print(f"Ablation on not-triggered subset: {abl_not_acc:.2f}% ({abl_not_correct}/{len(not_triggered)})")
            print(f"Gain on not-triggered subset: {rev_not_acc - abl_not_acc:+.2f}pt (should be ~0)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_online_layer.py <reveal_jsonl> [ablation_jsonl]")
        sys.exit(1)
    
    reveal_path = sys.argv[1]
    abl_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(reveal_path).exists():
        print(f"Error: {reveal_path} not found")
        sys.exit(1)
    
    if abl_path and not Path(abl_path).exists():
        print(f"Warning: {abl_path} not found, will only analyze REVEAL triggered rate")
        abl_path = None
    
    analyze_online_layer(reveal_path, abl_path)
