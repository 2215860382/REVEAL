#!/usr/bin/env python
"""De-visualize a target_time rubric: strip visual-modality demands from criterion
names + descriptions so a TEXT-only run can satisfy them when the evidence states the
fact. The target_time rubric was refined from VLM *visual* captions, so 43% of criteria
say "visually confirms X rather than merely mentioning" — structurally unattainable and
weighted 3-4x, which dragged the holistic score down and inverted it vs correctness.

We remove the words "visually"/"visual"/"visibly"/"visible" (NOT "show/showing", which is
modality-neutral). The specificity clauses ("rather than merely mentioning") are KEPT —
once "visually" is gone they read as "a concrete description confirms, a bare mention does
not", which is a good text-valid specificity check. Weights are left untouched (isolate
the modality change). Writes a new file; the original is not modified.
"""
import re, sys, yaml

SRC = sys.argv[1] if len(sys.argv) > 1 else "outputs/rubric/rubric.yaml"
DST = sys.argv[2] if len(sys.argv) > 2 else "outputs/rubric/egolife_tt/rubric_devis.yaml"


def devis_text(s: str) -> str:
    s = re.sub(r"\bvisually\s+", "", s, flags=re.I)   # "visually confirms" -> "confirms"
    s = re.sub(r"\bvisual\s+", "", s, flags=re.I)     # "visual grounding"  -> "grounding"
    s = re.sub(r"\bvisibly\s+", "", s, flags=re.I)
    s = re.sub(r"\bvisible\b", "present", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s+([,.])", r"\1", s)
    return s.strip()


def devis_name(name: str) -> str:
    n = re.sub(r"_visual(?=_|$)", "", name, flags=re.I)   # possession_visual_grounding -> possession_grounding
    n = re.sub(r"visual_", "", n, flags=re.I)             # visual_action_attribution  -> action_attribution
    n = re.sub(r"__+", "_", n).strip("_")
    return n or name


d = yaml.safe_load(open(SRC))
n_desc = n_name = 0
still_visual = 0
examples = []
for block_key in ["general"] + list(d.get("types", [])):
    block = d.get(block_key)
    if not isinstance(block, dict):
        continue
    seen = set()
    for crit in block.get("rubric_criteria") or []:
        old_name = str(crit.get("name", ""))
        old_desc = str(crit.get("description", ""))
        new_desc = devis_text(old_desc)
        new_name = devis_name(old_name)
        # collision guard within a block
        base = new_name; i = 2
        while new_name in seen and new_name != old_name:
            new_name = f"{base}_{i}"; i += 1
        seen.add(new_name)
        if new_desc != old_desc:
            n_desc += 1
            if len(examples) < 4:
                examples.append((old_name, old_desc[:120], new_name, new_desc[:120]))
        if new_name != old_name:
            n_name += 1
        crit["name"] = new_name
        crit["description"] = new_desc
        if re.search(r"\bvisual", new_name + " " + new_desc, flags=re.I):
            still_visual += 1

yaml.safe_dump(d, open(DST, "w"), allow_unicode=True, sort_keys=False)
print(f"wrote {DST}")
print(f"descriptions changed: {n_desc} | names changed: {n_name} | criteria still containing 'visual': {still_visual}")
for on, od, nn, nd in examples:
    print(f"\n  NAME {on} -> {nn}")
    print(f"  OLD: {od}")
    print(f"  NEW: {nd}")
