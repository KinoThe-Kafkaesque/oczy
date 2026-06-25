"""Compare cvec vs prefix steering on a semantic consolidation probe.

Measures exact-token uptake on the probe "'Profile' here means business
_______." with target word "vertical". Tests four conditions:

1. baseline_no_steering
2. cvec_only        (proj_random + SVD, current CortexAgent path)
3. prefix_only      (soft-prompt literal-prefix injection)
4. cvec_plus_prefix (both)

Run from repo root:

    .venv/bin/python experiments/smoke_consolidation_uptake_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from oczy.lm import CVecDriverConfig, LlamaCVecDriver
from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from plastic_cortex.kv_cortex import KVCortexConfig


def _contains(answer: str, token: str) -> bool:
    # Word-level containment to avoid matching substrings inside other tokens.
    tokens = answer.lower().split()
    stripped = [t.strip(".,!?;:'\"()*[]") for t in tokens]
    return token.lower() in stripped


def evaluate_condition(
    agent_ctor,
    condition_label: str,
    probe: str,
    target: str,
    expected_substrings: list[str] | None = None,
) -> dict[str, object]:
    """Build a fresh agent from `agent_ctor`, run probe, return metrics."""
    agent = agent_ctor()
    answer = agent.articulate(
        probe,
        max_tokens=16,
        temperature=0.0,
        apply_steering=True,
        stop=["."],
    ).strip()
    exact = _contains(answer, target)
    domain = any(_contains(answer, w) for w in (expected_substrings or []))
    print(f"{condition_label:20s} | exact={int(exact)} domain={int(domain)} | {answer!r}")
    return {
        "condition": condition_label,
        "answer": answer,
        "exact_uptake": float(exact),
        "domain_uptake": float(domain),
    }


def main() -> int:
    probe = "'Profile' here means business _______."
    target = "vertical"
    domain_words = ["commercial", "economic", "business", "strategy", "market", "vertical"]

    driver_cfg = CVecDriverConfig(n_ctx=128, n_threads=12, verbose=False)
    driver = LlamaCVecDriver.load(driver_cfg)
    prefix = "In this codebase, profile means business vertical. "

    def make_baseline():
        cfg = CortexAgentConfig(driver=driver_cfg)
        return CortexAgent(config=cfg, driver=driver)

    def make_cvec_only():
        cfg = CortexAgentConfig(
            driver=driver_cfg,
            cortex=KVCortexConfig(d_cortex=8, steering_mode="proj_random"),
            articulate_scale=0.03,
            auto_consolidate=True,
        )
        agent = CortexAgent(config=cfg, driver=driver)
        # Run explicit corrections and SVD-init proj_c using the same probe.
        hiddens = []
        correction = "No, 'profile' here means business vertical, not user profile."
        for _ in range(8):
            agent.turn(correction, correction_signal=1.0, max_tokens=4, temperature=0.0)
            hiddens.append(agent._last_hidden.copy())
        agent.cortex.init_proj_c_from_svd(np.vstack(hiddens))
        # Consolidate hard to make cold state reflect the warm update.
        agent.consolidate(strength=agent.config.cortex.max_consolidation_strength)
        return agent

    def make_prefix_only():
        cfg = CortexAgentConfig(driver=driver_cfg)
        agent = CortexAgent(config=cfg, driver=driver)
        agent.set_articulation_prefix(prefix)
        return agent

    def make_cvec_plus_prefix():
        agent = make_cvec_only()
        agent.set_articulation_prefix(prefix)
        return agent

    print(f"Probe: {probe!r}")
    print(f"Target token: {target!r}")
    print()

    results = [
        evaluate_condition(make_baseline, "baseline_no_steering", probe, target, domain_words),
        evaluate_condition(make_cvec_only, "cvec_only", probe, target, domain_words),
        evaluate_condition(make_prefix_only, "prefix_only", probe, target, domain_words),
        evaluate_condition(make_cvec_plus_prefix, "cvec_plus_prefix", probe, target, domain_words),
    ]

    print()
    print("Summary:")
    for r in results:
        print(
            f"  {r['condition']:20s} exact={r['exact_uptake']:.1f} domain={r['domain_uptake']:.1f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
