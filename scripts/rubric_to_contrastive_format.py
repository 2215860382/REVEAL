#!/usr/bin/env python
"""Convert a refine_rubric (templates/type_aliases) rubric into the latest
`outputs/rubric/rubric_contrastive.yaml` layout: a top-level ``types`` list, a
``general`` block, then each question type as its OWN top-level key. The verifier
reads both layouts, but the contrastive layout is the canonical one for experiments.

  python scripts/rubric_to_contrastive_format.py IN.yaml OUT.yaml
"""
import sys
import yaml


def main():
    src, dst = sys.argv[1], sys.argv[2]
    d = yaml.safe_load(open(src)) or {}
    general = d.get("general") or {}

    # gather type blocks from either layout
    RESERVED = {"general", "types", "type_aliases", "templates", "keyword_rules"}
    blocks = {}
    for k, v in (d.get("templates") or {}).items():
        if isinstance(v, dict) and v.get("rubric_criteria"):
            blocks[k] = v
    for k, v in d.items():                       # already-flat top-level type blocks
        if k not in RESERVED and isinstance(v, dict) and v.get("rubric_criteria"):
            blocks[k] = v

    out = {"types": list(blocks.keys()), "general": general}
    for k, v in blocks.items():
        out[k] = v

    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
    print(f"[fmt] {src} → {dst}  ({len(blocks)} types: {list(blocks.keys())})")


if __name__ == "__main__":
    main()
