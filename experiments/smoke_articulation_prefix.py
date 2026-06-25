"""Manual smoke check: does a fixed KV prefix steer the LM answer?

The prefix is a proof-of-concept soft-prompt analog: by locking a short
piece of concept-defining text into every prompt's KV positions, we test
whether the LM can be biased toward a target word (here, "vertical").

Run from the repo root:

    .venv/bin/python experiments/smoke_articulation_prefix.py

This intentionally does NOT touch benchmark.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oczy_lm import CVecDriverConfig, LlamaCVecDriver
from experiments.cortex_agent import CortexAgent, CortexAgentConfig


def main() -> int:
    prompt = "'Profile' here means business _______."
    # Concept-defining soft-prompt prefix.  We keep it short but pack the
    # target word and nearby context so fixed KV positions can bias sampling.
    prefix = "In this codebase, profile means business vertical. "

    driver_cfg = CVecDriverConfig(n_ctx=128, n_threads=12, verbose=False)
    driver = LlamaCVecDriver.load(driver_cfg)

    agent_cfg = CortexAgentConfig(
        driver=driver_cfg,
        articulate_scale=0.0,  # not used when apply_steering=False, but explicit
    )
    agent = CortexAgent(config=agent_cfg, driver=driver)

    print("Prompt:", repr(prompt))
    print()

    # Test a couple of stop strategies; newline is too eager on this fill-in
    # task, so we illustrate both the raw continuation and a period-trimmed one.
    any_different = False
    for label, stops in [("raw", None), ("period-stopped", ["."])]:
        print(f"--- stop={label} ---")
        agent.clear_articulation_prefix()
        baseline = agent.articulate(
            prompt,
            max_tokens=16,
            temperature=0.0,
            apply_steering=False,
            stop=stops,
        )
        print("  Baseline (no prefix):", repr(baseline.strip()))

        agent.set_articulation_prefix(prefix)
        prefixed = agent.articulate(
            prompt,
            max_tokens=16,
            temperature=0.0,
            apply_steering=False,
            stop=stops,
        )
        print("  With prefix:        ", repr(prefixed.strip()))
        differs = baseline.strip().lower() != prefixed.strip().lower()
        any_different = any_different or differs
        print("  Differ:", differs)
        print()

    return 0 if any_different else 1


if __name__ == "__main__":
    sys.exit(main())
