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
import time
from pathlib import Path
from typing import Any, cast

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
_CONTROL_WORDS = [
    "document",
    "information",
    "report",
    "number",
    "system",
    "process",
]

def _domain_uptake(answer: str) -> float:
    """Fraction of probe domain words present in the answer."""
    tokens = {t.strip(".,!?;:'\"()*[]").lower() for t in answer.split()}
    hits = sum(1 for w in _DOMAIN_WORDS if w.lower() in tokens)
    return hits / len(_DOMAIN_WORDS)


def _control_uptake(answer: str) -> float:
    """Fraction of probe control words present in the answer."""
    hits = sum(1 for w in _CONTROL_WORDS if w in answer.lower())
    return hits / len(_CONTROL_WORDS)


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
    """L2 norms of each cold state snapshot."""
    return [float(np.linalg.norm(s)) for s in cold_states]

# ---------------------------------------------------------------------------
# S2.3 Magnitude-controlled drift metric
# ---------------------------------------------------------------------------


def _cvec_combined_norm(agent: Any) -> float:
    """Combined L2 norm across all per-layer cvecs.

    sqrt(sum(||v||^2 for v in cvecs)).  This is the scalar magnitude
    of the steering vector that the LM sees.
    """
    cvecs = agent.cortex.emit_all_cvecs()
    merged_sq = sum(float(np.sum(v * v)) for v in cvecs)
    return float(np.sqrt(merged_sq))


def _logit_shift_with_cvec(
    agent: Any, word_list: list[str], cvec_norm_clamp: float | None = None
) -> float:
    """Mean next-token logit of words at the probe blank, with cvec applied.

    When ``cvec_norm_clamp`` is provided, the per-layer cvecs are uniformly
    scaled down so their combined L2 norm does not exceed that budget.
    This isolates steering DIRECTION from steering LOUDNESS.

    Requires a real driver (llama-cpp-python backed).  Mock drivers should
    use the uptake-based path.
    """
    from numpy import ctypeslib

    cvecs = agent.cortex.emit_all_cvecs()
    if cvec_norm_clamp is not None and cvec_norm_clamp > 0:
        current_norm = np.sqrt(sum(float(np.sum(v * v)) for v in cvecs))
        if current_norm > cvec_norm_clamp:
            scale = cvec_norm_clamp / current_norm
            cvecs = [v * scale for v in cvecs]

    agent.driver.set_cvecs_per_layer(cvecs, scale=agent.config.articulate_scale)
    try:
        llm = agent.driver._llm
        n_vocab = llm.n_vocab()
        text_bytes = _PROBE.encode("utf-8")
        ids = llm.tokenize(text_bytes, add_bos=True)
        llm.eval(ids)
        raw = llm._ctx.get_logits()
        logits = ctypeslib.as_array(raw, shape=(len(ids) * n_vocab,))
        last = logits[(len(ids) - 1) * n_vocab : len(ids) * n_vocab].copy()

        token_ids: list[int] = []
        for word in word_list:
            word_ids = llm.tokenize(word.encode("utf-8"), add_bos=False)
            if word_ids:
                token_ids.append(word_ids[0])
        if not token_ids:
            return 0.0
        return float(np.mean(last[token_ids]))
    finally:
        agent.driver.clear_cvec()


def _domain_probe_with_cvec(
    agent: Any, cvec_norm_clamp: float | None = None
) -> float:
    """Domain uptake with cvec applied, optionally clamped.

    Mock-driver-safe path: computes clamped cvecs, pushes them to the driver,
    then calls articulate() with apply_steering=False (cvec already applied).
    """
    if cvec_norm_clamp is not None and cvec_norm_clamp > 0:
        cvecs = agent.cortex.emit_all_cvecs()
        current_norm = np.sqrt(sum(float(np.sum(v * v)) for v in cvecs))
        if current_norm > cvec_norm_clamp:
            scale = cvec_norm_clamp / current_norm
            cvecs = [v * scale for v in cvecs]
        agent.driver.set_cvecs_per_layer(cvecs, scale=agent.config.articulate_scale)
        try:
            answer = agent.articulate(
                _PROBE,
                max_tokens=8,
                temperature=0.0,
                apply_steering=False,
                use_reserved_position=False,
            )
            return _domain_uptake(answer)
        finally:
            agent.driver.clear_cvec()
    else:
        return _domain_probe(agent)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def _resolve_gguf_model_path() -> str:
    """Resolve the LFM2.5 GGUF path from env or local cache.

    Honors ``OCZY_MODEL_PATH`` (set by the Kaggle bootstrap or a local
    user) first, then falls back to the conventional HF cache layout.
    Raises ``FileNotFoundError`` when no local file can be found — never
    downloads.
    """
    import os
    from pathlib import Path

    env_path = os.environ.get("OCZY_MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return env_path
    cache_parent = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--LiquidAI--LFM2.5-1.2B-Instruct-GGUF"
    )
    target = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    if cache_parent.exists():
        for path in sorted(cache_parent.rglob(target)):
            if path.is_file():
                return str(path)
    raise FileNotFoundError(
        f"{target} not found. Set OCZY_MODEL_PATH or cache the file "
        "under ~/.cache/huggingface/hub/models--LiquidAI--"
        "LFM2.5-1.2B-Instruct-GGUF."
    )


def _build_parametrized_agent(
    alpha: float,
    threshold: int,
    corrections: tuple[str, ...],
    seed: int = 0,
) -> tuple[Any, bool]:
    """Build a real-driver agent with configurable correction parameters.

    Returns ``(agent, is_diverse)`` where ``is_diverse`` is True when more than
    one correction phrasing is supplied.
    """
    from llama_cpp import Llama

    from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
    from oczy.lm.cvec_driver import CVecDriverConfig, LlamaCVecDriver
    from plastic_cortex.kv_cortex import KVCortexConfig

    _MODEL_PATH = _resolve_gguf_model_path()

    driver_cfg = CVecDriverConfig(n_ctx=128, n_threads=12, verbose=False)
    llm = Llama(model_path=_MODEL_PATH, n_ctx=driver_cfg.n_ctx,
                n_threads=driver_cfg.n_threads, verbose=driver_cfg.verbose,
                embedding=driver_cfg.embedding)
    driver = LlamaCVecDriver(llm, driver_cfg)
    cfg = CortexAgentConfig(
        driver=driver_cfg,
        cortex=KVCortexConfig(
            d_cortex=8,
            d_embd=driver.n_embd,
            n_layers=16,
            steering_mode="proj_random",
            alpha_correction=alpha,
            consolidate_replay_threshold=threshold,
            seed=seed,
        ),
        articulate_scale=0.01,
        auto_consolidate=False,
    )
    agent = CortexAgent(config=cfg, driver=driver)
    agent.boot()
    return agent, len(corrections) > 1


def _build_real_agent() -> Any:
    agent, _ = _build_parametrized_agent(
        alpha=0.3, threshold=2, corrections=_DIVERSE_CORRECTIONS
    )
    return agent

def _build_mock_agent(seed: int = 0) -> Any:
    from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
    from oczy.experiments.multi_fact_stressor import _MockDriver
    from oczy.lm.cvec_driver import LlamaCVecDriver
    from plastic_cortex.kv_cortex import KVCortexConfig

    class _MockCVecDriver(_MockDriver):
        """Extends the stressor mock with no-op cvec methods."""

        def set_cvec_uniform(self, *args, **kwargs) -> None:  # noqa: ARG002
            return None

        def set_cvecs_per_layer(self, *args, **kwargs) -> None:  # noqa: ARG002
            return None

        def clear_cvec(self) -> None:
            return None

    driver = cast("LlamaCVecDriver", _MockCVecDriver(n_embd=16))
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(
            d_cortex=4,
            d_embd=driver.n_embd,
            n_layers=2,
            steering_mode="proj_random",
            seed=seed,
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


def _compounding_loop(
    agent: Any,
    corrections: list[str],
    k: int,
    batch_size: int = 1,
    checkpoints: set[int] | None = None,
    return_agent: bool = False,
) -> dict[str, Any]:
    """Run K corrections in batches, consolidating after each batch.

    Batching ensures the hippocampus accumulates >= batch_size distinct traces
    before each consolidate() call, which enables the additive replay absorption
    path (needs >= 3 replays). Without batching, each consolidate() only gets
    1 replay and the slow-EMA nudge saturates cold_state instead of compounding.

    Diverse correction phrasings are cycled to give each batch distinct traces
    (SHA-256 hash key per trace), preventing overwrite.

    When ``checkpoints`` is None the default set ``{0, k//4, k//2, 3*k//4, k}``
    is used; otherwise the supplied set of ints is used. When ``return_agent``
    is True the (possibly drifted) agent is included in the result dict under
    ``"agent"``.
    """
    cold_states: list[np.ndarray] = [agent.cortex.cold_state.copy()]
    cold_drifts: list[float] = []
    checkpoint_norms: list[float] = []
    checkpoint_indices: list[int] = []

    _checkpoints = checkpoints if checkpoints is not None else {0, k // 4, k // 2, 3 * k // 4, k}
    if 0 in _checkpoints:
        checkpoint_norms.append(float(np.linalg.norm(agent.cortex.cold_state)))
        checkpoint_indices.append(0)

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

    result = {
        "cold_states": cold_states,
        "cold_drifts": cold_drifts,
        "compounding_index": _compounding_index(cold_states),
        "cold_norms": _cold_norms(cold_states),
        "checkpoint_norms": checkpoint_norms,
        "checkpoint_indices": checkpoint_indices,
        "total_consolidations": len(cold_states) - 1,
    }
    if return_agent:
        result["agent"] = agent
    return result


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


def _control_probe(agent: Any) -> float:
    """Control uptake on the probe with steering but no prefix/logit-bias."""
    answer = agent.articulate(
        _PROBE,
        max_tokens=16,
        temperature=0.0,
        apply_steering=True,
        use_reserved_position=False,
    )
    return _control_uptake(answer)


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
            token_ids.append(word_ids[0])
    if not token_ids:
        return 0.0
    return float(np.mean(last[token_ids]))


def _logit_control_shift(agent: Any) -> float:
    """Mean next-token logit of control word token ids at the probe blank.

    Mirrors ``_logit_domain_shift`` but iterates over ``_CONTROL_WORDS``.
    Reads the underlying Llama model logits directly.
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
    for word in _CONTROL_WORDS:
        word_ids = llm.tokenize(word.encode("utf-8"), add_bos=False)
        if word_ids:
            token_ids.append(word_ids[0])
    if not token_ids:
        return 0.0
    return float(np.mean(last[token_ids]))


def _drift_metric_triple(
    agent: Any,
    zero_agent: Any,
    clamp_norm: float | None = None,
    use_logits: bool = True,
) -> dict[str, float]:
    """Magnitude-controlled drift metric triple.

    Returns ``delta_target``, ``delta_control`` and ``delta_target_clamped``
    relative to ``zero_agent`` (an unsteered baseline). When ``use_logits`` is
    True the real-driver logit path is used; otherwise the mock-driver
    uptake-based path is used (cvec is a no-op on the mock driver).

    All three legs apply the steering vector (cvec) before measurement.
    ``delta_target`` and ``delta_control`` use the full, unclamped steering
    vector; ``delta_target_clamped`` uses the same steering vector but
    uniformly scaled so its combined L2 norm does not exceed ``clamp_norm``.
    """
    if use_logits:
        delta_target = (
            _logit_shift_with_cvec(agent, _DOMAIN_WORDS, cvec_norm_clamp=None)
            - _logit_shift_with_cvec(zero_agent, _DOMAIN_WORDS, cvec_norm_clamp=None)
        )
        delta_control = (
            _logit_shift_with_cvec(agent, _CONTROL_WORDS, cvec_norm_clamp=None)
            - _logit_shift_with_cvec(zero_agent, _CONTROL_WORDS, cvec_norm_clamp=None)
        )
        delta_target_clamped = (
            _logit_shift_with_cvec(agent, _DOMAIN_WORDS, clamp_norm)
            - _logit_shift_with_cvec(zero_agent, _DOMAIN_WORDS, cvec_norm_clamp=None)
        )
    else:
        delta_target = _domain_probe(agent) - _domain_probe(zero_agent)
        delta_control = _control_probe(agent) - _control_probe(zero_agent)
        delta_target_clamped = (
            _domain_probe_with_cvec(agent, clamp_norm) - _domain_probe(zero_agent)
        )
    return {
        "delta_target": delta_target,
        "delta_control": delta_control,
        "delta_target_clamped": delta_target_clamped,
    }


# ---------------------------------------------------------------------------
# Main entrypoints
# ---------------------------------------------------------------------------


def _run_real_driver(k: int = 20) -> dict[str, float] | None:
    agent = _build_real_agent()
    _svd_warmup(agent, list(_DIVERSE_CORRECTIONS))
    comp = _compounding_loop(agent, list(_DIVERSE_CORRECTIONS), k, batch_size=2)

    zero_agent = _build_real_agent()

    # Magnitude-controlled drift triple (S2.3).
    clamp_norm = _cvec_combined_norm(agent)
    triple = _drift_metric_triple(agent, zero_agent, clamp_norm=clamp_norm, use_logits=True)

    drift_uptake = _domain_probe(agent)
    zero_uptake = _domain_probe(zero_agent)

    # Compute compounding slope: linear regression of checkpoint cold_norms.
    _cp_norms = comp.get("checkpoint_norms", [])
    _cp_indices = comp.get("checkpoint_indices", [])
    _slope = _compounding_slope(_cp_indices, _cp_norms)

    return {
        "metabolism_drift_delta": triple["delta_target"],
        "compounding_index": comp["compounding_index"],
        "compounding_slope": _slope,
        "final_cold_norm": comp["cold_norms"][-1],
        "mean_cold_drift": float(np.mean(comp["cold_drifts"])) if comp["cold_drifts"] else 0.0,
        "total_consolidations": comp["total_consolidations"],
        "batch_size": 2,
        "checkpoint_indices": _cp_indices,
        "checkpoint_norms": _cp_norms,
        "zero_baseline_uptake": zero_uptake,
        "drift_uptake": drift_uptake,
        "delta_target": triple["delta_target"],
        "delta_control": triple["delta_control"],
        "delta_target_clamped": triple["delta_target_clamped"],
        "cvec_combined_norm": clamp_norm,
    }


def _run_mock_driver(k: int = 4) -> dict[str, float]:
    agent = _build_mock_agent()
    _svd_warmup(agent, list(_DIVERSE_CORRECTIONS))
    comp = _compounding_loop(agent, [_CORRECTION], k)

    zero_agent = _build_mock_agent()

    triple = _drift_metric_triple(agent, zero_agent, clamp_norm=None, use_logits=False)

    drift_uptake = _domain_probe(agent)
    zero_uptake = _domain_probe(zero_agent)

    _cp_norms = comp.get("checkpoint_norms", [])
    _cp_indices = comp.get("checkpoint_indices", [])
    _slope = _compounding_slope(_cp_indices, _cp_norms)

    return {
        "metabolism_drift_delta": triple["delta_target"],
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
        "delta_target": triple["delta_target"],
        "delta_control": triple["delta_control"],
        "delta_target_clamped": triple["delta_target_clamped"],
    }


def _run_ablation_real(
    checkpoints: list[int] | None = None,
    seeds: int = 1,
    output_path: str | None = None,
) -> None:
    """S2.4 single-variable ablation on the real GGUF driver.

    Five conditions isolate the effect of alpha, threshold, and diversity:
      1. OLD          : alpha=1.0, threshold=3, bs=3, single correction
      2. NEW          : alpha=0.3, threshold=2, bs=2, diverse corrections
      3. OLD+alpha    : alpha=0.3, threshold=3, bs=3, single correction
      4. OLD+threshold: alpha=1.0, threshold=2, bs=3, single correction
      5. OLD+diverse  : alpha=1.0, threshold=3, bs=3, diverse corrections

    ``clamp_norm`` for conditions 2-5 is condition 1's final cvec norm, so the
    clamped metric isolates steering DIRECTION from LOUDNESS across conditions.
    """
    if checkpoints is None:
        checkpoints = [0, 5, 10, 15, 20]
    _run_ablation(checkpoints, seeds, use_logits=True, output_path=output_path)


def _run_ablation_mock(
    checkpoints: list[int] | None = None,
    seeds: int = 1,
    output_path: str | None = None,
) -> None:
    """S2.4 ablation on the mock driver (uptake-based, no GGUF needed)."""
    if checkpoints is None:
        checkpoints = [0, 5, 10, 15, 20]
    _run_ablation(checkpoints, seeds, use_logits=False, output_path=output_path)


def _run_ablation(
    checkpoints: list[int],
    seeds: int,
    use_logits: bool,
    output_path: str | None = None,
) -> None:
    """Shared ablation runner for both real and mock drivers.

    For each condition, iterates through every K in *checkpoints*, creating
    fresh agents each time and running the full compounding loop from scratch
    with exactly K corrections.  This produces a K-trajectory (not just an
    endpoint measurement).

    The clamp budget is captured from condition 1 (OLD) at max K using its
    OWN agent instance — not a separately-built copy — then applied to every
    other (condition, K) combination.  Agent construction is deterministically
    seeded per (condition, seed, K) so cross-instance cvec-norm drift is
    impossible.
    """
    cp_sorted = sorted(checkpoints)
    max_k = cp_sorted[-1]

    conditions: list[tuple[int, float, int, int, tuple[str, ...]]] = [
        (1, 1.0, 3, 3, (_CORRECTION,)),          # OLD
        (2, 0.3, 2, 2, _DIVERSE_CORRECTIONS),     # NEW (breakthrough)
        (3, 0.3, 3, 3, (_CORRECTION,)),           # OLD + alpha only
        (4, 1.0, 2, 3, (_CORRECTION,)),           # OLD + threshold only
        (5, 1.0, 3, 3, _DIVERSE_CORRECTIONS),     # OLD + diverse only
    ]

    def _agent_seed(cond: int, seed: int, ki: int) -> int:
        return hash((cond, seed, ki)) & 0x7FFFFFFF

    results: list[dict[str, Any]] = []

    for seed in range(seeds):
        # ── Phase 1: capture clamp budget from condition 1 at max K ──
        # Build cond-1's max-K agent first, run the full compounding loop,
        # capture clamp_norm from *this* instance, and measure the triple
        # immediately so clamped == unclamped for the budget-defining
        # checkpoint.
        bt0 = time.time()
        s_budget = _agent_seed(1, seed, max_k)
        if use_logits:
            budget_agent, _ = _build_parametrized_agent(
                1.0, 3, (_CORRECTION,), seed=s_budget,
            )
        else:
            budget_agent = _build_mock_agent(seed=s_budget)

        _svd_warmup(budget_agent, list(_DIVERSE_CORRECTIONS))
        _compounding_loop(
            budget_agent, [_CORRECTION], max_k,
            batch_size=3, checkpoints={max_k},
        )

        if use_logits:
            budget_zero, _ = _build_parametrized_agent(
                1.0, 3, (_CORRECTION,), seed=s_budget,
            )
        else:
            budget_zero = _build_mock_agent(seed=s_budget)

        clamp_norm: float | None = _cvec_combined_norm(budget_agent)
        triple = _drift_metric_triple(
            budget_agent, budget_zero, clamp_norm=clamp_norm, use_logits=use_logits,
        )
        print(
            f"ABLATION cond=1 seed={seed} K={max_k} "
            f"delta_target={triple['delta_target']} "
            f"delta_control={triple['delta_control']} "
            f"delta_target_clamped={triple['delta_target_clamped']}"
        )
        elapsed_budget = time.time() - bt0
        results.append({
            "cond": 1, "seed": seed, "K": max_k,
            "delta_target": triple["delta_target"],
            "delta_control": triple["delta_control"],
            "delta_target_clamped": triple["delta_target_clamped"],
            "wall_time_s": elapsed_budget,
        })
        print(f"ABLATION cond=1 seed={seed} wall_time_s={elapsed_budget}")

        # ── Phase 2: all remaining (condition, K) pairs ──
        for cond, alpha, threshold, batch_size, corrections in conditions:
            t0 = time.time()
            cond_rows: list[dict[str, Any]] = []
            for ki in cp_sorted:
                if cond == 1 and ki == max_k:
                    continue  # already handled in Phase 1

                s = _agent_seed(cond, seed, ki)
                if use_logits:
                    agent, _ = _build_parametrized_agent(
                        alpha, threshold, corrections, seed=s,
                    )
                else:
                    agent = _build_mock_agent(seed=s)

                _svd_warmup(agent, list(_DIVERSE_CORRECTIONS))

                if ki > 0:
                    _compounding_loop(
                        agent, list(corrections), ki,
                        batch_size=batch_size, checkpoints={ki},
                    )

                if use_logits:
                    zero_agent, _ = _build_parametrized_agent(
                        alpha, threshold, corrections, seed=s,
                    )
                else:
                    zero_agent = _build_mock_agent(seed=s)

                triple = _drift_metric_triple(
                    agent, zero_agent, clamp_norm=clamp_norm, use_logits=use_logits,
                )
                print(
                    f"ABLATION cond={cond} seed={seed} K={ki} "
                    f"delta_target={triple['delta_target']} "
                    f"delta_control={triple['delta_control']} "
                    f"delta_target_clamped={triple['delta_target_clamped']}"
                )
                cond_rows.append({
                    "cond": cond, "seed": seed, "K": ki,
                    "delta_target": triple["delta_target"],
                    "delta_control": triple["delta_control"],
                    "delta_target_clamped": triple["delta_target_clamped"],
                })
            if cond_rows:
                elapsed = time.time() - t0
                print(f"ABLATION cond={cond} seed={seed} wall_time_s={elapsed}")
                for row in cond_rows:
                    row["wall_time_s"] = elapsed
                    results.append(row)

    # Historical note: this used to hardcode the 2026-07-01 S2.4 log path,
    # which silently rewrote a frozen experiment record on every mock/test
    # run. Reports are now written only to an explicitly requested path;
    # otherwise the tables go to stdout alone.
    if output_path is not None:
        _write_ablation_report(results, cp_sorted, seeds, use_logits, output_path)


def _write_ablation_report(
    results: list[dict[str, Any]],
    checkpoints: list[int],
    seeds: int,
    use_logits: bool,
    output_path: str,
) -> None:
    """Write the S2.4 ablation markdown report to *output_path*.

    Aggregates per-(cond, K) measurements across seeds by mean, then emits
    a condition × K trajectory table, a per-condition max-K summary, a norm
    control survival table, a VERDICT, and the raw CSV.
    """
    cond_labels = {
        1: "OLD",
        2: "NEW (breakthrough)",
        3: "OLD + alpha only",
        4: "OLD + threshold only",
        5: "OLD + diverse only",
    }
    driver = "real" if use_logits else "mock"
    cp_sorted = sorted(checkpoints)

    # Aggregate across seeds: mean per (cond, K).
    agg: dict[tuple[int, int], dict[str, float]] = {}
    for row in results:
        key = (row["cond"], row["K"])
        bucket = agg.setdefault(
            key,
            {
                "delta_target": 0.0,
                "delta_control": 0.0,
                "delta_target_clamped": 0.0,
                "wall_time_s": 0.0,
                "n": 0.0,
            },
        )
        bucket["delta_target"] += row["delta_target"]
        bucket["delta_control"] += row["delta_control"]
        bucket["delta_target_clamped"] += row["delta_target_clamped"]
        bucket["wall_time_s"] += row["wall_time_s"]
        bucket["n"] += 1

    def mean(key: tuple[int, int], field: str) -> float:
        b = agg[key]
        return b[field] / b["n"]

    conds = sorted({c for c, _ in agg})
    max_k = cp_sorted[-1]

    lines: list[str] = []
    lines.append("# S2.4 Breakthrough Ablation — Magnitude-Controlled Drift")
    lines.append("")
    lines.append("**Date:** 2026-07-01")
    lines.append(f"**Driver:** {driver}")
    lines.append(f"**Seeds:** {seeds}")
    lines.append("")

    # Condition × K trajectory.
    lines.append("## Condition × K Trajectory")
    lines.append("")
    lines.append("| Cond | K | Δ Target | Δ Control | Δ Target (clamped) |")
    lines.append("|------|---|------------|------------|----------------------|")
    for ci, cond in enumerate(conds):
        for ki in cp_sorted:
            key = (cond, ki)
            if key not in agg:
                continue
            lines.append(
                f"| {cond_labels[cond]} | {ki} | "
                f"{mean(key, 'delta_target'):.6f} | "
                f"{mean(key, 'delta_control'):.6f} | "
                f"{mean(key, 'delta_target_clamped'):.6f} |"
            )
        if ci < len(conds) - 1:
            lines.append("| | | | | |")
    lines.append("")

    # Per-condition summary (max-K row).
    lines.append("## Per-Condition Summary")
    lines.append("")
    lines.append("| Cond | Max K Δ Target | Max K Δ Control | Max K Δ Clamped | Wall Time (s) |")
    lines.append("|------|------------------|-------------------|-------------------|---------------|")
    maxk_rows: dict[int, dict[str, float]] = {}
    for cond in conds:
        key = (cond, max_k)
        if key not in agg:
            continue
        dt = mean(key, "delta_target")
        dc = mean(key, "delta_control")
        dcl = mean(key, "delta_target_clamped")
        wt = mean(key, "wall_time_s")
        maxk_rows[cond] = {
            "delta_target": dt,
            "delta_control": dc,
            "delta_target_clamped": dcl,
            "wall_time_s": wt,
        }
        lines.append(
            f"| {cond_labels[cond]} | {dt:.6f} | {dc:.6f} | {dcl:.6f} | {wt:.2f} |"
        )
    lines.append("")

    # Norm control survival.
    lines.append("## Norm Control Survival")
    lines.append("")
    lines.append("| Cond | Δ Target | Δ Clamped | Survival Ratio |")
    lines.append("|------|-----------|------------|---------------|")
    survival: dict[int, float] = {}
    for cond in conds:
        if cond not in maxk_rows:
            continue
        dt = maxk_rows[cond]["delta_target"]
        dcl = maxk_rows[cond]["delta_target_clamped"]
        ratio = dcl / dt if dt != 0 else float("nan")
        survival[cond] = ratio
        ratio_str = f"{ratio:.6f}" if dt != 0 else "NaN"
        lines.append(
            f"| {cond_labels[cond]} | {dt:.6f} | {dcl:.6f} | {ratio_str} |"
        )
    lines.append("")

    # VERDICT.
    lines.append("## VERDICT")
    lines.append("")
    old_dt = maxk_rows.get(1, {}).get("delta_target", 0.0)
    single_var_deltas: dict[int, float] = {
        c: maxk_rows[c]["delta_target"] - old_dt
        for c in (3, 4, 5)
        if c in maxk_rows
    }
    if single_var_deltas:
        any_positive = any(v > 0 for v in single_var_deltas.values())
        if any_positive:
            best_cond = max(single_var_deltas, key=single_var_deltas.__getitem__)
            best_delta = single_var_deltas[best_cond]
            var_name = {3: "alpha", 4: "threshold", 5: "diversity"}[best_cond]
            lines.append(
                f"1. Largest single-variable improvement vs OLD: "
                f"**{var_name}** (cond {best_cond}), Δ_target = {best_delta:.6f}."
            )
        else:
            lines.append("1. **No single variable improves upon OLD.**")
            for c in (3, 4, 5):
                if c in single_var_deltas:
                    var_name = {3: "alpha", 4: "threshold", 5: "diversity"}[c]
                    d = single_var_deltas[c]
                    lines.append(
                        f"   - {var_name}: Δ_target = {d:+.6f} vs OLD"
                    )
    else:
        lines.append("1. Insufficient data to rank single-variable improvements.")
    new_ratio = survival.get(2, float("nan"))
    if new_ratio == new_ratio:  # not NaN
        survives = new_ratio > 0.5
        lines.append(
            f"2. NEW survival ratio = {new_ratio:.6f}; "
            f"{'survives' if survives else 'does NOT survive'} norm control (> 0.5)."
        )
        # Cross-check: does NEW clamped gain exceed OLD unclamped?
        new_dcl = maxk_rows.get(2, {}).get("delta_target_clamped", 0.0)
        if old_dt > 0 and new_dcl < old_dt:
            lines.append(
                f"3. Under norm control, NEW clamped Δ = {new_dcl:.6f} falls "
                f"BELOW OLD unclamped Δ = {old_dt:.6f} — "
                f"the raw gain is magnitude inflation."
            )
        if not survives:
            lines.append(
                "4. The 13.5x claim is retracted as magnitude inflation — "
                "the gain does not survive norm control."
            )
        else:
            lines.append("3. NEW gain survives norm control (ratio > 0.5).")
    else:
        lines.append("2. NEW survival ratio is NaN (delta_target = 0).")
        lines.append("3. Cannot evaluate norm-control retraction (NaN).")
    lines.append("")

    # Raw CSV.
    lines.append("## Raw Data (CSV for reproducibility)")
    lines.append("```csv")
    lines.append("cond,seed,K,delta_target,delta_control,delta_target_clamped")
    for row in results:
        lines.append(
            f"{row['cond']},{row['seed']},{row['K']},"
            f"{row['delta_target']},{row['delta_control']},"
            f"{row['delta_target_clamped']}"
        )
    lines.append("```")
    lines.append("")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"ABLATION report written to {output_path}")


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
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the S2.4 single-variable ablation suite.",
    )
    parser.add_argument(
        "--ablation-seeds",
        type=int,
        default=1,
        help="Number of seeds for the ablation suite (default 1).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the ablation markdown report. Omitted = stdout only "
            "(reports never overwrite historical logs implicitly)."
        ),
    )
    args = parser.parse_args(argv)

    if args.ablation:
        if args.driver == "real":
            try:
                _run_ablation_real(
                    checkpoints=[0, 5, 10, 15, 20],
                    seeds=args.ablation_seeds,
                    output_path=args.output,
                )
            except Exception:
                print("ASI ablation_real=failed")
        else:
            _run_ablation_mock(
                checkpoints=[0, 5, 10, 15, 20],
                seeds=args.ablation_seeds,
                output_path=args.output,
            )
        return 0

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
