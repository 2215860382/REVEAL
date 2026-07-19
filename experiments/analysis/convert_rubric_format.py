#!/usr/bin/env python
"""Convert legacy-format rubric YAMLs (general + type_aliases + templates) to the
rubric_contrastive.yaml format (each question type is a TOP-LEVEL key, plus a
`types` list). The verifier parses both formats identically, so this is a purely
cosmetic normalization — verified per file by re-parsing through the code's own
_rubric_config and asserting the parsed (general, templates, thresholds) match.

Atomic writes (tmp + os.replace) so a rubric being read by a running experiment
never sees a partial file.

  /home2/ycj/miniconda3/envs/reveal/bin/python experiments/analysis/convert_rubric_format.py
"""
from __future__ import annotations
import os, sys, glob
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUBRIC_DIR = ROOT / "outputs/rubric"
sys.path.insert(0, str(ROOT))
from src.agents.verifier import _rubric_config  # noqa


def parsed_signature(path: Path):
    """(general_criteria+thr, {type: (criteria_names_weights, thr)}) as parsed by the code."""
    import src.agents.verifier as V
    # _rubric_config reads a module-level path env; call the file-loading internals.
    with open(path) as f:
        data = yaml.safe_load(f)
    general = data.get("general") or {}
    RESERVED = {"general", "types", "type_aliases", "templates", "keyword_rules"}
    templates = {}
    if data.get("templates"):
        templates = dict(data["templates"])
    else:
        for k, v in data.items():
            if k not in RESERVED and isinstance(v, dict) and v.get("rubric_criteria"):
                templates[k] = v
    def sig(sec):
        return (tuple((c.get("name"), c.get("weight")) for c in sec.get("rubric_criteria", [])),
                sec.get("sufficient_threshold"))
    gsig = sig(general)
    tsig = {t: sig(v) for t, v in templates.items()}
    return gsig, tsig


def to_format_a(data: dict) -> dict:
    """Rewrite a legacy dict into the top-level-key format."""
    if not data.get("templates"):
        return data  # already new format
    general = data.get("general") or {}
    templates = data["templates"]
    out = {"types": list(templates.keys()), "general": general}
    for t, v in templates.items():
        out[t] = v
    return out


def atomic_write_yaml(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True, width=100)
    os.replace(tmp, path)


def main():
    files = sorted(glob.glob(str(RUBRIC_DIR / "*.yaml")))
    for fp in files:
        path = Path(fp)
        data = yaml.safe_load(open(path))
        if not isinstance(data, dict):
            print(f"SKIP {path.name}: not a dict"); continue
        if not data.get("templates"):
            print(f"OK   {path.name}: already new format"); continue
        before = parsed_signature(path)
        new = to_format_a(data)
        atomic_write_yaml(path, new)
        after = parsed_signature(path)
        status = "MATCH" if before == after else "*** MISMATCH ***"
        print(f"CONV {path.name}: {len(new.get('types', []))} types -> top-level | parse {status}")
        if before != after:
            print("   !! parsed content changed — investigate before trusting this file")


if __name__ == "__main__":
    main()
