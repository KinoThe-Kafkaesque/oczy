"""Validate or score recorded outputs for the Pi tool-use curriculum.

Live model execution remains in the Pi benchmark/proxy surfaces. This runner
keeps the measuring instrument independent: it consumes a JSON mapping from
episode id to ordered raw model outputs and emits descriptive stage metrics.
No acceptance threshold is applied before a real-data distribution audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_tool_curriculum
from .scoring import score_episode
from .validation import validate_tool_curriculum


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, help="JSON episode-id -> output list")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stages = build_tool_curriculum()
    errors = validate_tool_curriculum(stages)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Validated tool curriculum: 6 stages, 45 episodes, thresholds inactive")
    if args.outputs is None:
        return 0

    payload = json.loads(args.outputs.read_text(encoding="utf-8"))
    for stage in stages:
        scores = [
            score_episode(ep, list(payload.get(ep.id, []))) for ep in stage.episodes
        ]
        passed = sum(score.passed for score in scores)
        print(f"{stage.id}: {passed}/{len(scores)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
