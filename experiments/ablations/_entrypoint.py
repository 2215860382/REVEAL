"""Shared CLI plumbing for named ablation arms."""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


_RESET_ENV = {
    "REVEAL_UNIFIED_ONLINE": "0",
    "REVEAL_UNIFIED_OFFLINE": "0",
}


def run_arm(
    env: Mapping[str, str],
    *,
    default_max_iter: int,
    controller_prompt: str | None = None,
    fixed_max_iter: int | None = None,
) -> None:
    """Run one reproducible arm while leaving dataset arguments on the CLI."""
    os.environ.update(_RESET_ENV)
    os.environ.update(env)
    if fixed_max_iter is not None:
        if "--max-iter" in sys.argv:
            index = sys.argv.index("--max-iter")
            sys.argv[index + 1] = str(fixed_max_iter)
        else:
            sys.argv.extend(["--max-iter", str(fixed_max_iter)])
    elif "--max-iter" not in sys.argv:
        sys.argv.extend(["--max-iter", str(default_max_iter)])

    from experiments.ablations import _pipeline
    from experiments.reveal_27b import main

    if controller_prompt is not None:
        _pipeline.set_offline_controller_prompt(controller_prompt)
    main(run_pipeline=_pipeline.run_reveal)
