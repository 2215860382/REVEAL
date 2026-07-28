"""Iter: offline controller with retrieval restricted to L1 memory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ablations._entrypoint import run_arm

_ENV = {
    "REVEAL_UNIFIED_OFFLINE": "1",
    "REVEAL_MP_L1_CAP": "8",
    "REVEAL_MP_L2_CAP": "0",
    "REVEAL_MP_L3_CAP": "0",
}


if __name__ == "__main__":
    run_arm(_ENV, default_max_iter=3)
