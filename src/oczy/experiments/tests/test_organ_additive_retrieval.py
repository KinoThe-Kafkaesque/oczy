"""Tests for the S3.M2a additive retrieval ablation harness.

Exercises :class:`RetrievalOrganism` (the flags-based additive subclass of
:class:`MinimalOrganism`) and the ``run_additive_one_seed`` /
``run_additive_ablation`` runners against the real
``hf-internal-testing/tiny-random-LlamaForCausalLM`` driver — fast, cached,
no network beyond the initial download.

Test style mirrors ``test_organ_ablation_smoke.py``: plain functions (no
pytest fixtures), a module-level lazy cache for the shared driver and
curriculum stage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# ``minimal_loop`` imports ``eval.v2`` from the repo root, which is not on
# sys.path under pytest (only the package ``src`` trees are).  Add it once.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from oczy.experiments.minimal_loop import MinimalOrganism, _HippocampusGuard
from oczy.experiments.organ_additive_retrieval import (
    RetrievalOrganism,
    run_additive_ablation,
    run_additive_one_seed,
)
from oczy.experiments.organism_curriculum.dataset import (
    Episode,
    Probe,
    Stage,
    build_curriculum,
    split_probes,
)
from oczy.experiments.scope_selectivity_stressor import _cosine
from oczy.lm.hf_driver import HFDriver

TEST_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# Shared fixtures (module-level lazy cache — plain functions, no pytest
# fixtures, matching the existing test style in this directory).
# ---------------------------------------------------------------------------

_driver_cache: HFDriver | None = None
_stage_cache: Stage | None = None


def _driver() -> HFDriver:
    """Load the tiny-random model once and cache it."""
    global _driver_cache
    if _driver_cache is None:
        _driver_cache = HFDriver.load(TEST_MODEL_ID)
    return _driver_cache


def _grounding_stage() -> Stage:
    """Load stage_0_grounding once and cache it."""
    global _stage_cache
    if _stage_cache is None:
        stages = build_curriculum(stage_names=["stage_0_grounding"])
        _stage_cache = stages[0]
    return _stage_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    ep_id: str = "test_ep",
    request: str = "What is X?",
    correction: str = "X means Y.",
    label: str = "Y",
    probes: tuple[Probe, ...] | None = None,
) -> Episode:
    """Build a minimal curriculum episode for unit tests."""
    if probes is None:
        probes = (
            Probe(
                request=request,
                expected=label,
                category="retention",
                match_mode="sense",
            ),
        )
    return Episode(
        id=ep_id,
        initial_request=request,
        default_response="I don't know.",
        correction_utterance=correction,
        corrected_label=label,
        corrected_response=f"{request} {correction}",
        domain="test",
        probes=probes,
    )


def _booted(**flags) -> RetrievalOrganism:
    """Fresh, booted RetrievalOrganism with the given constructor flags."""
    org = RetrievalOrganism(_driver(), **flags)
    org.boot()
    return org


def _taught_and_consolidated(org: RetrievalOrganism, episode: Episode) -> None:
    """Teach one episode then consolidate — the common pre-answer setup."""
    org.teach(episode)
    org.consolidate()


# ---------------------------------------------------------------------------
# Test 1: BASE is bit-identical to plain MinimalOrganism
# ---------------------------------------------------------------------------


def test_base_identical_to_minimal() -> None:
    """BASE condition (no flags) produces character-identical answers."""
    ep = _make_episode()
    d = _driver()

    plain = MinimalOrganism(d)
    plain.boot()
    plain.teach(ep)
    plain.consolidate()
    a_plain = plain.answer(ep.initial_request, max_tokens=16)

    ret = RetrievalOrganism(d)  # BASE: no flags
    ret.boot()
    ret.teach(ep)
    ret.consolidate()
    a_ret = ret.answer(ep.initial_request, max_tokens=16)

    assert a_plain == a_ret, f"BASE diverged: {a_plain!r} != {a_ret!r}"


# ---------------------------------------------------------------------------
# Test 2: hippocampus-at-answer engages
# ---------------------------------------------------------------------------


def test_hippocampus_at_answer_engaged() -> None:
    """With use_hippocampus_at_answer=True, answer() calls _hippo_raw.reinforce
    directly (bypassing the guard) and returns without RuntimeError."""
    ep = _make_episode()
    org = _booted(use_hippocampus_at_answer=True)
    org.teach(ep)

    # Hippocampus has stored the episode.
    status = org._hippo_raw.status()
    assert status["episode_count"] >= 1

    # answer() must not raise RuntimeError (the guard would fire if the
    # hippocampus were touched through the guarded proxy).
    result = org.answer(ep.initial_request, max_tokens=16)
    assert isinstance(result, str) and result


# ---------------------------------------------------------------------------
# Test 3: DSI populated and queried
# ---------------------------------------------------------------------------


def test_dsi_populated_and_queried() -> None:
    """With use_dsi_fact_index=True, teach() populates the DSI and retrieve()
    returns results for the taught query."""
    ep = _make_episode()
    org = _booted(use_dsi_fact_index=True)
    org.teach(ep)

    # DSI status shows occupied rows.
    status = org._dsi.status()
    assert status["n_occupied"] >= 1
    assert ep.corrected_label in status["labels"]

    # retrieve() returns results for the taught query embedding.
    hidden = org.driver.peek_embedding(ep.correction_utterance)
    hits = org._dsi.retrieve(hidden, k=3, use_lora=True)
    assert len(hits) >= 1
    assert hits[0][0] == ep.corrected_label


# ---------------------------------------------------------------------------
# Test 4: scope-slot stored and retrieved
# ---------------------------------------------------------------------------


def test_scope_slot_stored_and_retrieved() -> None:
    """With use_scope_slot_reranker=True, teach() stores a slot key+label and
    cosine similarity lookup returns the stored slot for a similar request."""
    ep = _make_episode()
    org = _booted(use_scope_slot_reranker=True)
    org.teach(ep)

    # Slot store has an entry and a label.
    assert len(org._scope_slot_keys) >= 1
    assert len(org._scope_slot_labels) >= 1
    assert org._scope_slot_labels[0] == ep.corrected_label

    # Cosine similarity lookup: query with the same embedding mode as the
    # stored key (last_token_only=False, mean-pooled).  The module stores
    # slot keys with last_token_only=False, so the query must match.
    key = org.driver.peek_embedding(ep.initial_request, last_token_only=False)
    stored_key = org._scope_slot_keys[0]
    sim = _cosine(key, stored_key)
    # Self-similarity for the same text with the same embedding mode is ~1.0.
    assert sim >= 0.3, f"cosine sim {sim} below retrieve threshold"


# ---------------------------------------------------------------------------
# Test 5: BASE condition — no retrieval injection, request passed through
# ---------------------------------------------------------------------------


def test_base_condition_no_retrieval_injection() -> None:
    """In BASE (no flags), answer() delegates to super() — the request text
    is passed through unmodified (no [Recall:]/[Fact:]/[Scope:] prefix)."""
    ep = _make_episode()
    org = _booted()  # BASE
    _taught_and_consolidated(org, ep)

    # The BASE answer path delegates to MinimalOrganism.answer, which uses
    # the guarded hippocampus proxy.  _hippo_raw is the raw object; the guard
    # wraps it as org.hippocampus.  In BASE, RetrievalOrganism.answer must
    # not touch _hippo_raw during answer — verify by confirming the guard is
    # still in place and answer succeeds.
    assert isinstance(org.hippocampus, _HippocampusGuard)

    result = org.answer(ep.initial_request, max_tokens=16)
    assert isinstance(result, str) and result

    # No retrieval prefix should appear in the *request* path.  We verify
    # indirectly: BASE answer equals a plain MinimalOrganism answer (already
    # covered by test_base_identical_to_minimal).  Here we additionally check
    # that the organism has no active component state that would inject.
    assert getattr(org, "_dsi", None) is None or org._dsi is None
    assert org._scope_slot_keys == []


# ---------------------------------------------------------------------------
# Test 6: teach populates only active components
# ---------------------------------------------------------------------------


def test_teach_populates_only_active_components() -> None:
    """With use_dsi_fact_index=True only, DSI has data but scope slots do NOT
    (no false positives)."""
    ep = _make_episode()
    org = _booted(use_dsi_fact_index=True)
    org.teach(ep)

    # DSI is populated.
    assert org._dsi is not None
    assert org._dsi.status()["n_occupied"] >= 1

    # Scope slots are NOT populated (flag was off).
    assert org._scope_slot_keys == []
    assert org._scope_slot_labels == []


# ---------------------------------------------------------------------------
# Test 7: multiple components independent (smoke)
# ---------------------------------------------------------------------------


def test_multiple_components_independent() -> None:
    """With all three flags on, teach + answer completes without crash and
    returns a string."""
    ep = _make_episode()
    org = _booted(
        use_hippocampus_at_answer=True,
        use_dsi_fact_index=True,
        use_scope_slot_reranker=True,
    )
    org.teach(ep)
    org.consolidate()

    result = org.answer(ep.initial_request, max_tokens=16)
    assert isinstance(result, str) and result


# ---------------------------------------------------------------------------
# Test 8: runner structure — run_additive_one_seed
# ---------------------------------------------------------------------------


def test_runner_structure() -> None:
    """run_additive_one_seed returns a dict with the expected keys and sane
    values for the hippocampus-at-answer condition."""
    stage = _grounding_stage()
    holdout = split_probes(stage)[1]

    # The runner calls HFDriver.load() with no model_id, which defaults to
    # the production Qwen model.  Patch the default to the tiny-random model
    # so the test stays fast and offline.
    with patch("oczy.lm.hf_driver.DEFAULT_MODEL_ID", TEST_MODEL_ID):
        result = run_additive_one_seed(
            seed=0,
            stage=stage,
            holdout_ids=holdout,
            condition_kwargs={"use_hippocampus_at_answer": True},
        )

    assert "seed" in result
    assert "holdout_accuracy" in result
    assert "vanilla_holdout_accuracy" in result
    assert "n_probes" in result

    assert result["seed"] == 0
    assert result["n_probes"] > 0
    acc = result["holdout_accuracy"]
    assert 0.0 <= acc <= 1.0, f"holdout_accuracy {acc} out of [0, 1]"
    vann = result["vanilla_holdout_accuracy"]
    assert 0.0 <= vann <= 1.0, f"vanilla_holdout_accuracy {vann} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 9: runner BASE condition
# ---------------------------------------------------------------------------


def test_runner_base_condition() -> None:
    """run_additive_one_seed with empty condition_kwargs (BASE) returns the
    expected result structure."""
    stage = _grounding_stage()
    holdout = split_probes(stage)[1]

    with patch("oczy.lm.hf_driver.DEFAULT_MODEL_ID", TEST_MODEL_ID):
        result = run_additive_one_seed(
            seed=0,
            stage=stage,
            holdout_ids=holdout,
            condition_kwargs={},
        )

    assert "seed" in result
    assert "holdout_accuracy" in result
    assert "vanilla_holdout_accuracy" in result
    assert "n_probes" in result
    assert result["n_probes"] > 0
    assert 0.0 <= result["holdout_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Test 10: all three conditions + BASE through run_additive_ablation
# ---------------------------------------------------------------------------


def test_all_three_conditions() -> None:
    """run_additive_ablation with 1 seed, 1 stage returns 4 condition keys,
    each containing the stage name."""
    stage = _grounding_stage()
    holdout_split = {stage.name: split_probes(stage)[1]}

    with patch("oczy.lm.hf_driver.DEFAULT_MODEL_ID", TEST_MODEL_ID):
        results = run_additive_ablation(
            stages=[stage],
            holdout_splits=holdout_split,
            seeds=[0],
        )

    # Four condition keys: BASE + 3 component conditions.
    assert len(results) == 4, f"expected 4 conditions, got {set(results)}"
    expected = {
        "BASE",
        "HIPPOCAMPUS_AT_ANSWER",
        "DSI_FACT_INDEX",
        "SCOPE_SLOT_RERANKER",
    }
    assert set(results.keys()) == expected, (
        f"condition keys {set(results)} != {expected}"
    )

    # Each condition has the stage name as a key.
    for cond_name, stage_results in results.items():
        assert stage.name in stage_results, (
            f"condition {cond_name} missing stage {stage.name!r}; "
            f"has {set(stage_results)}"
        )
        seed_results = stage_results[stage.name]
        assert len(seed_results) == 1, (
            f"condition {cond_name} expected 1 seed result, "
            f"got {len(seed_results)}"
        )
        sr = seed_results[0]
        assert "seed" in sr
        assert "holdout_accuracy" in sr
        assert 0.0 <= sr["holdout_accuracy"] <= 1.0
