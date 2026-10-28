"""Offline: unified retrieval/answer controller over constructed offline memory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ablations._entrypoint import run_arm


_OFFLINE_CONTROLLER_PROMPT = """\
You control retrieval and answering in a long-video multiple-choice QA system. Use the question,
options, and accumulated evidence to choose exactly one action:

SEARCH when more evidence is needed. Choose one of the strategies:
- Text Retrieval: provide focused semantic queries for specific facts or events.
- Uniform Sampling: sample unseen timeline regions for broad coverage; queries must be [].

ANSWER when the current evidence supports a choice. Return the correct option letter.

Return only one JSON object:
{"decision":"search","strategy":"Text Retrieval|Uniform Sampling","queries":["..."]}
or
{"decision":"answer","answer":"A|B|C|D"}
"""

_ENV = {
    "REVEAL_UNIFIED_OFFLINE": "1",
    "REVEAL_MP_L1_CAP": "8",
    "REVEAL_MP_L2_CAP": "4",
    "REVEAL_MP_L3_CAP": "2",
}


if __name__ == "__main__":
    run_arm(_ENV, default_max_iter=3)
