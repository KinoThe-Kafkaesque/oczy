"""Print the frozen organ identity under a bound local model directory.

Used by the G1 freeze step to bind the *verifiable* local organ identity of a
DEV instrument.  Prints one JSON object and exits nonzero on any failure.  It
never trains, never scores, and never opens a sealed or calibration file.
"""

from __future__ import annotations

import json
import sys

import torch

from .organ import QwenFrozenOrgan


def main() -> int:
    torch.set_num_threads(4)
    organ = QwenFrozenOrgan.load()
    before = organ.parameter_hash()
    print(json.dumps({"organ_hash": before}, sort_keys=True))
    after = organ.parameter_hash()
    if after != before:
        print("organ identity changed during load", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
