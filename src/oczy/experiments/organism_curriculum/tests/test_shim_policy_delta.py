"""Regression test for the curriculum shim policy margin delta."""

from __future__ import annotations

import re
import warnings

from oczy.experiments.organism import OrganismAgent
from oczy.experiments.organism_curriculum.dataset import build_curriculum
from oczy.experiments.organism_curriculum.run_curriculum import (
    _DeterministicCortexShim,
    run_stage,
)


def _match_key(text: str, scores: dict[str, float]) -> str | None:
    """Find the candidate key that corresponds to ``text``.

    First tries an exact lookup, then a case-insensitive containment match,
    then falls back to the key with the greatest token overlap.
    """
    if text in scores:
        return text
    lowered = text.lower()
    for key in scores:
        if key.lower() == lowered:
            return key
        if key.lower() in lowered or lowered in key.lower():
            return key
    text_tokens = set(re.findall(r"[a-z0-9']+", lowered))
    best_key = None
    best_overlap = 0.0
    for key in scores:
        key_tokens = set(re.findall(r"[a-z0-9']+", key.lower()))
        if not key_tokens:
            continue
        overlap = len(text_tokens & key_tokens) / max(len(text_tokens), len(key_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key
    return best_key


def _corrected_key(
    corrected: str,
    wrong: str,
    before: dict[str, float],
    after: dict[str, float],
) -> str | None:
    """Map the full corrected response onto a candidate label key."""
    key = _match_key(corrected, before) or _match_key(corrected, after)
    if key is None and wrong in before and wrong in after:
        other_keys = [k for k in set(before) | set(after) if k != wrong]
        if len(other_keys) == 1:
            return other_keys[0]
    return key


def test_shim_policy_margin_delta_positive() -> None:
    """The shim probe should improve corrected-vs-wrong policy margin."""
    stages = build_curriculum(stage_names=("stage_0_grounding",))
    assert stages, "stage 0 not found"
    stage = stages[0]

    config = {
        "use_cortex_policy": True,
        "use_value_baseline": True,
        "use_acceptance_policy_reward": True,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        agent = OrganismAgent(config)

    agent.cortex_agent = _DeterministicCortexShim()
    result = run_stage(agent, stage, adapter=None, instrument_policy=True)

    margin_deltas: list[float] = []
    scored_episodes = 0
    for er in result.episode_results:
        if not (er.policy_score_before and er.policy_score_after):
            continue
        scored_episodes += 1
        before = er.policy_score_before
        after = er.policy_score_after
        wrong = er.first_answer
        corrected = er.corrected_response

        corrected_key = _corrected_key(corrected, wrong, before, after)
        before_wrong = before.get(wrong, 0.0)
        after_wrong = after.get(wrong, 0.0)
        before_corrected = (
            before.get(corrected_key, before_wrong) if corrected_key else before_wrong
        )
        after_corrected = (
            after.get(corrected_key, after_wrong) if corrected_key else after_wrong
        )

        margin_before = before_corrected - before_wrong
        margin_after = after_corrected - after_wrong
        margin_deltas.append(margin_after - margin_before)

    assert scored_episodes > 0, "no episodes had policy scores recorded"
    assert margin_deltas, "margin deltas could not be computed"
    average_margin_delta = sum(margin_deltas) / len(margin_deltas)
    assert average_margin_delta > 0.0, (
        f"expected positive margin delta, got {average_margin_delta}"
    )
