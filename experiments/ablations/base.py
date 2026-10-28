"""Base: one L1 retrieval followed immediately by answering."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ablations._entrypoint import run_arm


_BASE_CONTROLLER_PROMPT = """\
You perform long-video multiple-choice QA in exactly two steps.

On the first call, retrieve evidence. Choose one strategy:
- Text Retrieval: return focused semantic queries for specific facts or events.
- Uniform Sampling: sample the timeline broadly; queries must be [].

Return only:
{"decision":"search","strategy":"Text Retrieval|Uniform Sampling","queries":["..."]}

On the second call, use the retrieved evidence to answer immediately. Do not request more
evidence. Return only:
{"decision":"answer","answer":"A|B|C|D"}
"""

_ENV = {
    "REVEAL_UNIFIED_OFFLINE": "1",
    "REVEAL_MP_L1_CAP": "8",
    "REVEAL_MP_L2_CAP": "0",
    "REVEAL_MP_L3_CAP": "0",
}


if __name__ == "__main__":
    run_arm(
        _ENV,
        default_max_iter=1,
        fixed_max_iter=1,
        controller_prompt=_BASE_CONTROLLER_PROMPT,
    )
