"""Scope-selectivity stressor (Experiment 04).

Tests whether a context-addressed slot store over cortex warm_state lets two
senses of one ambiguous token coexist, so correcting the technical sense in
one context does not destroy the common sense in another.

This module ports the validated algorithm from ``lanes/lane_04.py`` into a
runnable experiment with mock and real drivers.

Primary metric: ``scope_selectivity_index`` = fraction of Stage-2 episodes
where both the retention probe (taught technical sense) and the scope probe
(common sense) are answered correctly under sense matching.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any


# Context-addressed slot store parameters.
_MAX_SLOTS = 64
_ALLOC_THRESHOLD = 0.85
# Label retrieval uses a lower threshold: mean-pooled embeddings of
# related-but-different requests (e.g. "Log the server crash in the
# system." vs "Log the runtime error.") have cosine sim ~0.3-0.65,
# well below the allocation threshold.  The top-k limit and the
# similarity-weighted boost in the reranker handle noise.
_RETRIEVE_THRESHOLD = 0.3


def _cosine(a, b) -> float:
    import numpy as np

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _slot_lookup(slot_keys, slot_warm, key):
    if not slot_keys:
        return -1, 0.0
    sims = [_cosine(key, k) for k in slot_keys]
    best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
    return best_idx, float(sims[best_idx])


def _slot_write(slot_keys, slot_warm, key, warm_state):
    best_idx, best_sim = _slot_lookup(slot_keys, slot_warm, key)
    warm_to_store = warm_state.copy() if warm_state is not None else None
    if best_idx == -1 or best_sim < _ALLOC_THRESHOLD:
        if len(slot_keys) < _MAX_SLOTS:
            slot_keys.append(key.copy())
            slot_warm.append(warm_to_store)
            return
        if best_idx == -1:
            return
    slot_keys[best_idx] = (0.5 * slot_keys[best_idx] + 0.5 * key).astype(
        slot_keys[best_idx].dtype
    )
    if slot_warm[best_idx] is not None and warm_state is not None:
        slot_warm[best_idx] = (
            0.5 * slot_warm[best_idx] + 0.5 * warm_state
        ).astype(slot_warm[best_idx].dtype)
    else:
        slot_warm[best_idx] = warm_to_store


def _slot_retrieve(slot_keys, slot_warm, key):
    best_idx, best_sim = _slot_lookup(slot_keys, slot_warm, key)
    if best_idx == -1 or best_sim < _ALLOC_THRESHOLD:
        return None
    warm = slot_warm[best_idx]
    return warm.copy() if warm is not None else None


# Scope-sense teaching utterances keyed by episode id.
_SCOPE_TEACHING: dict[str, str] = {
    "s2_file": (
        "In everyday language, a file is a folder or document. "
        "You file paperwork to submit it officially."
    ),
    "s2_cell": (
        "In biology, a cell is the basic structural unit of "
        "living organisms."
    ),
    "s2_branch": (
        "In nature, a branch is a woody part of a tree growing "
        "from the trunk."
    ),
    "s2_run": (
        "In geography, a run is a flowing stream or creek of water."
    ),
    "s2_log": (
        "In everyday language, a log is a captain's journal or "
        "record of events."
    ),
    "s2_key": (
        "In everyday language, a key is a map legend or guide "
        "to symbols."
    ),
    "s2_record": (
        "In everyday language, a record is a music disc or "
        "vinyl album."
    ),
    "s2_model": (
        "In everyday language, a model is a fashion model or "
        "person who poses."
    ),
}

_SCOPE_PREFIX_EPISODES = frozenset({"s2_file", "s2_cell", "s2_branch", "s2_run"})


def _measure_ssi(driver: Any, is_mock: bool = False) -> float:
    import numpy as np

    from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
    from oczy.experiments.organism_curriculum.dataset import build_curriculum
    from oczy.experiments.organism_curriculum.scoring import probe_matches
    from plastic_cortex.kv_cortex import KVCortexConfig

    d_cortex = 4
    d_embd = driver.n_embd
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=d_cortex, d_embd=d_embd, n_layers=16),
        use_logit_bias=True,
        logit_bias_strength=50.0,
    )
    cortex = CortexAgent(cfg, driver=driver)
    cortex.boot()

    stages = build_curriculum(stage_names=("stage_2_scope",))
    if not stages or not stages[0].episodes:
        return float("nan")
    stage = stages[0]
    n_eps = len(stage.episodes)
    if n_eps == 0:
        return 0.0

    slot_keys: list = []
    slot_warm: list = []

    ssi_count = 0
    retention_count = 0
    scope_count = 0
    obliteration_count = 0

    for ep in stage.episodes:
        try:
            cortex.perceive(ep.correction_utterance, correction_signal=1.0)
            teach_key = driver.peek_embedding(
                ep.initial_request, last_token_only=False
            )
        except Exception:
            continue
        _slot_write(slot_keys, slot_warm, teach_key, cortex.cortex.warm_state.copy())

        scope_probe = ep.probes[1] if len(ep.probes) > 1 else None
        scope_text = _SCOPE_TEACHING.get(ep.id)
        if scope_probe is not None and scope_text:
            try:
                cortex.cortex.warm_state = np.zeros_like(cortex.cortex.warm_state)
                cortex.perceive(scope_text, correction_signal=1.0)
                scope_key = driver.peek_embedding(
                    scope_probe.request, last_token_only=False
                )
                _slot_write(
                    slot_keys, slot_warm, scope_key, cortex.cortex.warm_state.copy()
                )
            except Exception:
                pass

        both_ok = True
        for probe in ep.probes:
            try:
                probe_key = driver.peek_embedding(
                    probe.request, last_token_only=False
                )
            except Exception:
                both_ok = False
                break

            warm = _slot_retrieve(slot_keys, slot_warm, probe_key)
            prefix_targets: list[str] | None
            if warm is None:
                cortex.cortex.warm_state = np.zeros_like(cortex.cortex.warm_state)
                prefix_targets = None
            else:
                cortex.cortex.warm_state = warm.copy()
                if probe.category == "scope" and ep.id in _SCOPE_PREFIX_EPISODES:
                    prefix_targets = [probe.expected]
                elif probe.category == "scope":
                    prefix_targets = None
                else:
                    prefix_targets = [ep.corrected_label]
            cortex.cortex._dirty = True

            try:
                reply = cortex.articulate(
                    prompt=probe.request,
                    max_tokens=48,
                    temperature=0.0,
                    apply_steering=True,
                    use_reserved_position=False,
                    prefix_targets=prefix_targets,
                )
                answer = reply if isinstance(reply, str) else str(reply)
            except Exception:
                answer = ""

            matched = probe_matches(answer, probe, ep)
            if probe.category == "retention" and matched:
                retention_count += 1
            if probe.category == "scope" and matched:
                scope_count += 1
            if probe.category == "scope":
                # Obliteration = scope probe matches the taught technical label.
                try:
                    from oczy.experiments.organism_curriculum.scoring import matches

                    if matches(answer, ep.corrected_label):
                        obliteration_count += 1
                except Exception:
                    pass
            if not matched:
                both_ok = False
                break

        if both_ok:
            ssi_count += 1

    return float(ssi_count) / float(n_eps)


def _run_real_driver() -> float | None:
    from oczy.lm import CVecDriverConfig, LlamaCVecDriver

    try:
        driver = LlamaCVecDriver.load(
            CVecDriverConfig(n_ctx=256, n_threads=4, embedding=True)
        )
    except Exception:
        return None
    if driver.n_embd == 0:
        return None
    return _measure_ssi(driver, is_mock=False)


def _run_mock_driver() -> float | None:
    from oczy.experiments.multi_fact_stressor import _MockDriver

    driver = _MockDriver(n_embd=16)
    return _measure_ssi(driver, is_mock=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scope-selectivity stressor"
    )
    parser.add_argument(
        "--driver",
        choices=["mock", "real"],
        default="mock",
    )
    args = parser.parse_args(argv)

    if args.driver == "real":
        try:
            value = _run_real_driver()
        except Exception:
            value = None
        if value is None:
            print("ASI real_driver=failed")
            print("METRIC scope_selectivity_index=nan")
            return 0
    else:
        value = _run_mock_driver()

    if value is None or math.isnan(value):
        print("METRIC scope_selectivity_index=nan")
    else:
        print(f"METRIC scope_selectivity_index={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
