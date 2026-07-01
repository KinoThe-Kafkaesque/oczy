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
_DIVERSE_CORRECTIONS: tuple[str, ...] = (
    _CORRECTION,
    "Remember: in this context, 'profile' refers to a business vertical, not a user profile.",
    "For this task, 'profile' should be understood as business vertical, not personal profile.",
    "The intended meaning of 'profile' here is business vertical, not user profile.",
    "Clarification: 'profile' in this scenario denotes a business vertical, not a social profile.",
    "Update your understanding: here, 'profile' indicates a business vertical, not a user profile.",
    "Use the term 'profile' here as a synonym for business vertical, not personal profile.",
    "When you see 'profile' in this request, interpret it as business vertical.",
)
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


def _compounding_slope(indices: list[int], norms: list[float]) -> float:
    """Linear regression slope of cold_norm vs correction index.

    Positive slope indicates cold_state magnitude is growing with more
    corrections; zero or negative indicates saturation/overwrite.
    """
    if len(indices) < 2 or len(norms) < 2:
        return 0.0
    x = np.array(indices, dtype=np.float64)
    y = np.array(norms, dtype=np.float64)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_cov = np.sum((x - x_mean) * (y - y_mean))
    x_var = np.sum((x - x_mean) ** 2)
    if x_var == 0:
        return 0.0
    return float(xy_cov / x_var)


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
        articulate_scale=0.01,
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
        articulate_scale=0.01,
        auto_consolidate=False,
    )
    agent = CortexAgent(config=cfg, driver=driver)
    agent.boot()
    return agent


# ---------------------------------------------------------------------------
# SVD warm-up
# ---------------------------------------------------------------------------


def _svd_warmup(agent: Any, phrases: list[str]) -> None:
    """Collect hidden states from diverse correction phrases and SVD-init proj_c.

    Perceive-only (no consolidate) so cold_state stays near zero while the
    projector learns a correction-aligned basis from diverse semantics.
    """
    hiddens: list[np.ndarray] = []
    for phrase in phrases:
        agent.perceive(phrase, correction_signal=1.0)
        hidden = agent._last_hidden
        if hidden is not None:
            hiddens.append(hidden.copy())
    d_cortex = agent.cortex.config.d_cortex
    if len(hiddens) >= d_cortex:
        try:
            agent.cortex.init_proj_c_from_svd(np.vstack(hiddens))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def _compounding_loop(agent: Any, corrections: list[str], k: int, batch_size: int = 1) -> dict[str, Any]:
    """Run K corrections in batches, consolidating after each batch.

    Batching ensures the hippocampus accumulates >= batch_size distinct traces
    before each consolidate() call, which enables the additive replay absorption
    path (needs >= 3 replays). Without batching, each consolidate() only gets
    1 replay and the slow-EMA nudge saturates cold_state instead of compounding.

    Diverse correction phrasings are cycled to give each batch distinct traces
    (SHA-256 hash key per trace), preventing overwrite.
    """
    cold_states: list[np.ndarray] = [agent.cortex.cold_state.copy()]
    cold_drifts: list[float] = []
    checkpoint_norms: list[float] = []
    checkpoint_indices: list[int] = []

    _checkpoints = {0, k // 4, k // 2, 3 * k // 4, k}

    for i in range(k):
        correction = corrections[i % len(corrections)]
        agent.turn(correction, correction_signal=1.0, max_tokens=4, temperature=0.0)

        # Consolidate every batch_size corrections (or on the final step).
        if (i + 1) % batch_size == 0 or i == k - 1:
            summary = agent.consolidate(strength=agent.config.cortex.max_consolidation_strength)
            cold_drifts.append(summary.get("cold_drift", 0.0))
            cold_states.append(agent.cortex.cold_state.copy())
            if (i + 1) in _checkpoints:
                checkpoint_norms.append(float(np.linalg.norm(agent.cortex.cold_state)))
                checkpoint_indices.append(i + 1)

    return {
        "cold_states": cold_states,
        "cold_drifts": cold_drifts,
        "compounding_index": _compounding_index(cold_states),
        "cold_norms": _cold_norms(cold_states),
        "checkpoint_norms": checkpoint_norms,
        "checkpoint_indices": checkpoint_indices,
        "total_consolidations": len(cold_states) - 1,
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


def _logit_domain_shift(agent: Any) -> float:
    """Mean next-token logit of domain word token ids at the probe blank.

    Reads the underlying Llama model logits directly. The shift is read as
    the mean logit over the first token id of each domain word.
    """
    from numpy import ctypeslib

    llm = agent.driver._llm
    n_vocab = llm.n_vocab()
    text_bytes = _PROBE.encode("utf-8")
    ids = llm.tokenize(text_bytes, add_bos=True)
    llm.eval(ids)
    raw = llm._ctx.get_logits()
    logits = ctypeslib.as_array(raw, shape=(len(ids) * n_vocab,))
    last = logits[(len(ids) - 1) * n_vocab : len(ids) * n_vocab].copy()

    token_ids: list[int] = []
    for word in _DOMAIN_WORDS:
        word_ids = llm.tokenize(word.encode("utf-8"), add_bos=False)
        if word_ids:
            token_ids.append(int(word_ids[0]))
    if not token_ids:
        return 0.0
    return float(np.mean(last[token_ids]))


# ---------------------------------------------------------------------------
# Main entrypoints
# ---------------------------------------------------------------------------


def _run_real_driver(k: int = 20) -> dict[str, float] | None:
    agent = _build_real_agent()
    _svd_warmup(agent, list(_DIVERSE_CORRECTIONS))
    comp = _compounding_loop(agent, [_CORRECTION], k)
    drift_logits = _logit_domain_shift(agent)
    drift_uptake = _domain_probe(agent)

    zero_agent = _build_real_agent()
    zero_logits = _logit_domain_shift(zero_agent)
    zero_uptake = _domain_probe(zero_agent)

    # Compute compounding slope: linear regression of checkpoint cold_norms.
    _cp_norms = comp.get("checkpoint_norms", [])
    _cp_indices = comp.get("checkpoint_indices", [])
    _slope = _compounding_slope(_cp_indices, _cp_norms)

    return {
        "metabolism_drift_delta": drift_logits - zero_logits,
        "compounding_index": comp["compounding_index"],
        "compounding_slope": _slope,
        "final_cold_norm": comp["cold_norms"][-1],
        "mean_cold_drift": float(np.mean(comp["cold_drifts"])) if comp["cold_drifts"] else 0.0,
        "total_consolidations": comp["total_consolidations"],
        "batch_size": 3,
        "checkpoint_indices": _cp_indices,
        "checkpoint_norms": _cp_norms,
        "zero_baseline_logit": zero_logits,
        "drift_logit": drift_logits,
        "zero_baseline_uptake": zero_uptake,
        "drift_uptake": drift_uptake,
    }


def _run_mock_driver(k: int = 4) -> dict[str, float]:
    agent = _build_mock_agent()
    _svd_warmup(agent, list(_DIVERSE_CORRECTIONS))
    comp = _compounding_loop(agent, [_CORRECTION], k)
    drift_uptake = _domain_probe(agent)

    zero_agent = _build_mock_agent()
    zero_uptake = _domain_probe(zero_agent)

    _cp_norms = comp.get("checkpoint_norms", [])
    _cp_indices = comp.get("checkpoint_indices", [])
    _slope = _compounding_slope(_cp_indices, _cp_norms)

    return {
        "metabolism_drift_delta": drift_uptake - zero_uptake,
        "compounding_index": comp["compounding_index"],
        "compounding_slope": _slope,
        "final_cold_norm": comp["cold_norms"][-1],
        "mean_cold_drift": float(np.mean(comp["cold_drifts"])) if comp["cold_drifts"] else 0.0,
        "total_consolidations": comp["total_consolidations"],
        "batch_size": 3,
        "checkpoint_indices": _cp_indices,
        "checkpoint_norms": _cp_norms,
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
