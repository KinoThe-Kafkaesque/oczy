"""Metabolism loop closure probe (Experiment 05).

Tests whether repeated corrections compounded through explicit
``CortexAgent.consolidate()`` produce cold-state drift that drives a
continuous domain shift in the LM's answer, even without labels,
prefixes, or logit bias.

Mode:
  - ``--driver mock``: fast semantic-null control; expects drift_delta 0.
  - ``--driver real``: loads LFM2.5-1.2B-Instruct GGUF, runs K corrections,
    and measures domain shift.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

_PROBE = "'Profile' here means business _______."
_CORRECTION = "No, 'profile' here means business vertical, not user profile."
_DOMAIN_WORDS = ["commercial", "economic", "business", "strategy", "market", "vertical"]


def _domain_uptake(answer: str) -> float:
    """Fraction of probe domain words present in the answer."""
    tokens = {t.strip(".,!?;:'\"()*[]").lower() for t in answer.split()}
    hits = sum(1 for w in _DOMAIN_WORDS if w.lower() in tokens)
    return hits / len(_DOMAIN_WORDS)


def _compounding_index(cold_states: list[np.ndarray]) -> float:
    """‖Σ Δcold‖ / Σ ‖Δcold‖; 1.0 = perfectly additive."""
    if len(cold_states) < 2:
        return 0.0
    deltas = [cold_states[i + 1] - cold_states[i] for i in range(len(cold_states) - 1)]
    norm_sum = sum(float(np.linalg.norm(d)) for d in deltas)
    if norm_sum <= 0:
        return 0.0
    vector_sum = np.sum(deltas, axis=0)
    return float(np.linalg.norm(vector_sum) / norm_sum)


def _cold_norms(cold_states: list[np.ndarray]) -> list[float]:
    return [float(np.linalg.norm(s)) for s in cold_states]


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def _build_real_agent() -> Any:
    from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
    from oczy.lm import CVecDriverConfig, LlamaCVecDriver
    from plastic_cortex.kv_cortex import KVCortexConfig

    driver_cfg = CVecDriverConfig(n_ctx=128, n_threads=12, verbose=False)
    driver = LlamaCVecDriver.load(driver_cfg)
    cfg = CortexAgentConfig(
        driver=driver_cfg,
        cortex=KVCortexConfig(
            d_cortex=8,
            d_embd=driver.n_embd,
            n_layers=16,
            steering_mode="proj_random",
        ),
        articulate_scale=0.03,
        auto_consolidate=False,
    )
    agent = CortexAgent(config=cfg, driver=driver)
    agent.boot()
    return agent


def _build_mock_agent() -> Any:
    from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
    from oczy.experiments.multi_fact_stressor import _MockDriver
    from plastic_cortex.kv_cortex import KVCortexConfig

    class _MockCVecDriver(_MockDriver):
        """Extends the stressor mock with no-op cvec methods."""

        def set_cvec_uniform(self, *args, **kwargs) -> None:  # noqa: ARG002
            return None

        def set_cvecs_per_layer(self, *args, **kwargs) -> None:  # noqa: ARG002
            return None

        def clear_cvec(self) -> None:
            return None

    driver = _MockCVecDriver(n_embd=16)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(
            d_cortex=4,
            d_embd=driver.n_embd,
            n_layers=2,
            steering_mode="proj_random",
        ),
        articulate_scale=0.03,
        auto_consolidate=False,
    )
    agent = CortexAgent(config=cfg, driver=driver)
    agent.boot()
    return agent


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def _compounding_loop(agent: Any, correction: str, k: int) -> dict[str, Any]:
    """Run K corrections, consolidating after each, and record trajectories."""
    cold_states: list[np.ndarray] = [agent.cortex.cold_state.copy()]
    cold_drifts: list[float] = []
    hiddens: list[np.ndarray] = []

    d_cortex = agent.cortex.config.d_cortex
    did_svd = False
    for _i in range(k):
        agent.turn(correction, correction_signal=1.0, max_tokens=4, temperature=0.0)
        hidden = agent._last_hidden
        if hidden is not None:
            hiddens.append(hidden.copy())
        # SVD-init proj_c once we have enough diverse correction hiddens.
        if not did_svd and len(hiddens) >= d_cortex:
            try:
                agent.cortex.init_proj_c_from_svd(np.vstack(hiddens[:d_cortex]))
            except Exception:
                pass
            did_svd = True
        summary = agent.consolidate(strength=agent.config.cortex.max_consolidation_strength)
        cold_drifts.append(summary.get("cold_drift", 0.0))
        cold_states.append(agent.cortex.cold_state.copy())

    return {
        "cold_states": cold_states,
        "cold_drifts": cold_drifts,
        "compounding_index": _compounding_index(cold_states),
        "cold_norms": _cold_norms(cold_states),
    }


def _domain_probe(agent: Any) -> float:
    """Domain uptake on the probe with steering but no prefix/logit-bias."""
    answer = agent.articulate(
        _PROBE,
        max_tokens=16,
        temperature=0.0,
        apply_steering=True,
        use_reserved_position=False,
    )
    return _domain_uptake(answer)


# ---------------------------------------------------------------------------
# Main entrypoints
# ---------------------------------------------------------------------------


def _run_real_driver(k: int = 20) -> dict[str, float] | None:
    agent = _build_real_agent()
    comp = _compounding_loop(agent, _CORRECTION, k)
    drift_uptake = _domain_probe(agent)

    zero_agent = _build_real_agent()
    zero_uptake = _domain_probe(zero_agent)

    return {
        "metabolism_drift_delta": drift_uptake - zero_uptake,
        "compounding_index": comp["compounding_index"],
        "final_cold_norm": comp["cold_norms"][-1],
        "mean_cold_drift": float(np.mean(comp["cold_drifts"])) if comp["cold_drifts"] else 0.0,
        "zero_baseline_uptake": zero_uptake,
        "drift_uptake": drift_uptake,
    }


def _run_mock_driver(k: int = 4) -> dict[str, float]:
    agent = _build_mock_agent()
    comp = _compounding_loop(agent, _CORRECTION, k)
    drift_uptake = _domain_probe(agent)

    zero_agent = _build_mock_agent()
    zero_uptake = _domain_probe(zero_agent)

    return {
        "metabolism_drift_delta": drift_uptake - zero_uptake,
        "compounding_index": comp["compounding_index"],
        "final_cold_norm": comp["cold_norms"][-1],
        "mean_cold_drift": float(np.mean(comp["cold_drifts"])) if comp["cold_drifts"] else 0.0,
        "zero_baseline_uptake": zero_uptake,
        "drift_uptake": drift_uptake,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Metabolism loop closure probe"
    )
    parser.add_argument(
        "--driver",
        choices=["mock", "real"],
        default="mock",
    )
    parser.add_argument(
        "--corrections",
        type=int,
        default=None,
        help="Number of correction cycles (default: 4 mock, 20 real).",
    )
    args = parser.parse_args(argv)

    k = args.corrections
    if args.driver == "real":
        try:
            results = _run_real_driver(k=k if k is not None else 20)
        except Exception:
            results = None
        if results is None:
            print("ASI real_driver=failed")
            print("METRIC metabolism_drift_delta=nan")
            return 0
    else:
        results = _run_mock_driver(k=k if k is not None else 4)

    print(f"METRIC metabolism_drift_delta={results['metabolism_drift_delta']}")
    for key, value in results.items():
        if key == "metabolism_drift_delta":
            continue
        print(f"ASI {key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
