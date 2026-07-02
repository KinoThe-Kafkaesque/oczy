"""Unit tests for the S2.5 forgetting organism (MinimalForgettingOrganism).

Covers the deletion APIs, snapshot/restore round-trips, arm order-independence
and ratio math, the no-answer-time-hippocampus-access invariant, and basic
smoke behaviour. Real-driver tests use the tiny-random Llama so they run fast
on CPU; the module-scoped driver fixture loads it once for the whole module.
"""

from __future__ import annotations

import numpy as np
import pytest

import oczy.experiments.minimal_loop_forgetting as mlf
from oczy.experiments.minimal_loop_forgetting import (
    MinimalForgettingOrganism,
    OrganismSnapshot,
    TEST_MODEL_ID,
    _compute_metrics,
)
from oczy.experiments.organism_curriculum.dataset import Episode, Probe
from oczy.lm.hf_driver import HFDriver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _patch_observe_layer() -> None:
    """The tiny-random Llama only has 2 layers, but the module default
    ``CONSOLIDATION_OBSERVE_LAYER`` is 5 (tuned for the real Qwen model).
    Patch it to a valid mid-layer for the duration of this module so
    ``teach()`` can capture a hidden state without raising.
    """
    original = mlf.CONSOLIDATION_OBSERVE_LAYER
    mlf.CONSOLIDATION_OBSERVE_LAYER = 1
    try:
        yield
    finally:
        mlf.CONSOLIDATION_OBSERVE_LAYER = original


@pytest.fixture(scope="module")
def driver() -> HFDriver:
    """Module-scoped: load the tiny model once for all tests."""
    return HFDriver.load(TEST_MODEL_ID)


@pytest.fixture
def organism(driver: HFDriver) -> MinimalForgettingOrganism:
    """Fresh organism per test using the shared driver."""
    org = MinimalForgettingOrganism(driver=driver, seed=42)
    org.boot()
    yield org
    # Clean shared driver state so one test's steering/prefix cannot leak
    # into the next test's fresh organism.
    org.driver.clear_cvec()
    org.driver.clear_articulation_prefix()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(i: int = 0, domain: str = "nautical") -> Episode:
    """Build a deterministic correction episode with a unique id."""
    return Episode(
        id=f"test_{i}",
        initial_request=f"Show the log {i}.",
        default_response=f"I'll show the system error log {i}.",
        correction_utterance=f"No, 'log' means the captain's journal {i}.",
        corrected_label=f"captain's journal {i}",
        corrected_response=f"I'll show the captain's journal {i}.",
        domain=domain,
        probes=(
            Probe(
                request=f"Show the log {i}.",
                expected=f"captain's journal {i}",
                category="retention",
                match_mode="contains",
            ),
        ),
    )


def _train_to_consolidated(org: MinimalForgettingOrganism, n: int = 3) -> dict:
    """Teach ``n`` episodes, replay them so they clear the consolidation
    gate, then consolidate. Returns the consolidation result dict.

    Replaying is required because the hippocampus only emits summaries for
    traces whose ``replay_count`` meets ``replay_threshold``; without it no
    prefix is compiled and ``delete_consolidated_artifact`` has nothing to
    clear.
    """
    episodes = [_make_episode(i) for i in range(n)]
    for ep in episodes:
        org.teach(ep)
    # Bump replay counts so consolidation produces summaries → a real prefix.
    for ep in episodes:
        org.hippocampus.reinforce(ep.initial_request, k=3)
    return org.consolidate()


# ---------------------------------------------------------------------------
# 1. Deletion APIs
# ---------------------------------------------------------------------------


def test_delete_raw_traces(organism: MinimalForgettingOrganism) -> None:
    """delete_raw_traces clears the hippocampus + replay bank and shrinks
    the pickle memory footprint."""
    for i in range(3):
        organism.teach(_make_episode(i))
    assert organism.hippocampus.status()["episode_count"] == 3

    before, after = organism.delete_raw_traces()

    assert before > after, f"memory did not shrink: {before} -> {after}"
    status = organism.hippocampus.status()
    assert status["episode_count"] == 0
    assert status["record_count"] == 0
    assert organism._episode_count == 0
    assert list(organism._replay_bank) == []


def test_delete_consolidated_artifact(organism: MinimalForgettingOrganism) -> None:
    """delete_consolidated_artifact clears the prefix + cold state and
    shrinks the pickle memory footprint."""
    result = _train_to_consolidated(organism, n=3)
    assert result["prefix_tokens"] > 0, "precondition: a prefix was compiled"
    assert organism._prefix_text is not None

    before, after = organism.delete_consolidated_artifact()

    assert before > after, f"memory did not shrink: {before} -> {after}"
    assert organism._prefix_text is None
    assert organism.prefix_token_count == 0
    # Cold/warm state reset to the boot (zero) baseline.
    assert np.all(organism.cortex.cold_state == 0)
    assert np.all(organism.cortex.warm_state == 0)


def test_delete_raw_traces_asserts_zero_count(organism: MinimalForgettingOrganism) -> None:
    """The internal assertion inside delete_raw_traces guards the post-
    condition that the hippocampus episode count is 0; verify it externally
    too."""
    for i in range(3):
        organism.teach(_make_episode(i))
    assert organism.hippocampus.status()["episode_count"] == 3

    # If the internal assert were violated this call would raise AssertionError.
    before, after = organism.delete_raw_traces()

    assert before > after
    # External re-check of the invariant the internal assert protects.
    status = organism.hippocampus.status()
    assert status["episode_count"] == 0
    assert status["record_count"] == 0
    # A second delete on an already-empty organism is a no-op that still
    # satisfies the assertion.
    before2, after2 = organism.delete_raw_traces()
    assert organism.hippocampus.status()["episode_count"] == 0


# ---------------------------------------------------------------------------
# 2. Snapshot / restore round-trips
# ---------------------------------------------------------------------------


def test_snapshot_restore_roundtrip(organism: MinimalForgettingOrganism) -> None:
    """A snapshot taken after training, restored after deletions, reproduces
    the original answer output."""
    _train_to_consolidated(organism, n=3)
    snap = organism.snapshot()
    assert isinstance(snap, OrganismSnapshot)

    baseline = organism.answer("Show the log 0.", max_tokens=24)

    # Mutate state via both deletions.
    organism.delete_raw_traces()
    organism.delete_consolidated_artifact()

    organism.restore(snap)
    after = organism.answer("Show the log 0.", max_tokens=24)

    assert after == baseline
    # Cortex state is restored bit-for-bit.
    assert np.array_equal(organism.cortex.warm_state, snap.warm_state)
    assert np.array_equal(organism.cortex.cold_state, snap.cold_state)
    assert organism._prefix_text == snap.prefix_text


def test_restore_fully_undoes_deletions(organism: MinimalForgettingOrganism) -> None:
    """After both deletions the cortex is zeroed and the prefix is gone;
    restore brings the full trained state (and answer) back."""
    _train_to_consolidated(organism, n=3)
    snap = organism.snapshot()
    baseline = organism.answer("Show the log 1.", max_tokens=24)

    organism.delete_raw_traces()
    organism.delete_consolidated_artifact()

    # Post-deletion state is the zeroed baseline.
    assert np.all(organism.cortex.warm_state == 0)
    assert organism._prefix_text is None

    organism.restore(snap)
    restored = organism.answer("Show the log 1.", max_tokens=24)

    assert restored == baseline
    assert np.array_equal(organism.cortex.warm_state, snap.warm_state)
    assert organism._prefix_text == snap.prefix_text


# ---------------------------------------------------------------------------
# 3. Arm order-independence and ratio math
# ---------------------------------------------------------------------------


def test_arm_order_independence(organism: MinimalForgettingOrganism) -> None:
    """Scoring A_full then A_none must give the same per-arm answers as
    scoring A_none then A_full, because ``restore(snap)`` resets all mutable
    state before each arm."""
    _train_to_consolidated(organism, n=3)
    snap = organism.snapshot()
    requests = [f"Show the log {i}." for i in range(3)]

    def arm_answers(del_traces: bool, del_artifact: bool) -> list[str]:
        organism.restore(snap)
        if del_traces:
            organism.delete_raw_traces()
        if del_artifact:
            organism.delete_consolidated_artifact()
        return [organism.answer(r, max_tokens=24) for r in requests]

    # Forward order: A_full first, then A_none.
    full_forward = arm_answers(False, False)
    none_forward = arm_answers(True, True)
    # Reverse order: A_none first, then A_full.
    none_reverse = arm_answers(True, True)
    full_reverse = arm_answers(False, False)

    assert full_forward == full_reverse, "A_full answers depend on arm order"
    assert none_forward == none_reverse, "A_none answers depend on arm order"


def test_ratio_math_synthetic() -> None:
    """With synthetic accuracies [0.8, 0.75, 0.3, 0.2] the ratio math is
    forgetting_survival_ratio = (0.75-0.2)/(0.8-0.2) ≈ 0.917 and
    retrieval_dependence        = (0.3-0.2)/(0.8-0.2)  ≈ 0.167.
    """
    accs = {"A_full": 0.8, "A_forget": 0.75, "A_retrieval": 0.3, "A_none": 0.2}
    results = [
        {
            "seed": 0,
            "train_time_s": 0.0,
            "memory_bytes_pre": 1000,
            "memory_bytes_post": 500,
            "arm_accuracies": accs,
            "arm_memory": {},
        }
    ]
    metrics = _compute_metrics(results)

    delta = 0.8 - 0.2
    assert metrics["raw_survival_ratios"][0] == pytest.approx((0.75 - 0.2) / delta)
    assert metrics["raw_retrieval_ratios"][0] == pytest.approx((0.3 - 0.2) / delta)
    assert metrics["n_blocked"] == 0
    assert metrics["validity_gate_passes"] == [True]
    # Summarized means match the single-seed ratios.
    assert metrics["forgetting_survival_ratio"]["mean"] == pytest.approx(0.916666, abs=1e-3)
    assert metrics["retrieval_dependence"]["mean"] == pytest.approx(0.166666, abs=1e-3)


def test_validity_gate_blocks_small_delta() -> None:
    """When A_full - A_none = 0.05 < 0.10 the validity gate blocks the seed,
    ratios are NaN, and the verdict is BLOCKED."""
    accs = {"A_full": 0.5, "A_forget": 0.48, "A_retrieval": 0.47, "A_none": 0.45}
    results = [
        {
            "seed": 0,
            "train_time_s": 0.0,
            "memory_bytes_pre": 1000,
            "memory_bytes_post": 500,
            "arm_accuracies": accs,
            "arm_memory": {},
        }
    ]
    metrics = _compute_metrics(results)

    assert metrics["n_blocked"] == 1
    assert metrics["validity_gate_passes"] == [False]
    assert metrics["verdict"] == "BLOCKED"
    assert np.isnan(metrics["raw_survival_ratios"][0])
    assert np.isnan(metrics["raw_retrieval_ratios"][0])
    # No valid ratios → no summarized ratio stats.
    assert metrics["forgetting_survival_ratio"] is None
    assert metrics["retrieval_dependence"] is None


# ---------------------------------------------------------------------------
# 4. No answer-time hippocampus access
# ---------------------------------------------------------------------------


def test_no_answer_time_hippocampus_access(organism: MinimalForgettingOrganism) -> None:
    """answer() must never touch the hippocampus. Replace ``reinforce`` with
    a tripwire; if answer() called it, the test would raise AssertionError."""
    for i in range(3):
        organism.teach(_make_episode(i))
    assert organism.hippocampus.status()["episode_count"] == 3

    def _tripwire(*args: object, **kwargs: object) -> None:
        raise AssertionError("answer() must not call hippocampus.reinforce()")

    organism.hippocampus.reinforce = _tripwire  # type: ignore[method-assign]

    # No exception → reinforce was never called at answer time.
    ans = organism.answer("Show the log 0.", max_tokens=16)
    assert isinstance(ans, str)
    assert len(ans) > 0


# ---------------------------------------------------------------------------
# 5. Smoke tests
# ---------------------------------------------------------------------------


def test_organism_builds(organism: MinimalForgettingOrganism) -> None:
    """Basic construction and property defaults after boot."""
    assert organism.driver is not None
    assert organism.seed == 42
    assert organism.memory_bytes >= 0
    assert organism.prefix_token_count == 0
    assert organism._episode_count == 0
    assert organism._consolidation_count == 0
    assert organism._prefix_text is None
    # Boot zeroes the warm state from the (fresh) cold state.
    assert np.all(organism.cortex.warm_state == 0)


def test_teach_and_answer(organism: MinimalForgettingOrganism) -> None:
    """Teaching increments the episode counter and answer() returns a string."""
    organism.teach(_make_episode(0))
    assert organism._episode_count == 1
    assert organism.hippocampus.status()["episode_count"] == 1
    assert len(organism._replay_bank) == 1

    ans = organism.answer("Show the log 0.", max_tokens=16)
    assert isinstance(ans, str)
    assert len(ans) > 0


def test_consolidate_returns_dict(organism: MinimalForgettingOrganism) -> None:
    """consolidate() returns a dict with the expected stat keys."""
    organism.teach(_make_episode(0))
    result = organism.consolidate()

    assert isinstance(result, dict)
    for key in (
        "consolidation_num",
        "hippo_summaries",
        "prefix_tokens",
        "replay_count",
        "memory_bytes",
    ):
        assert key in result, f"missing key: {key}"
    assert result["consolidation_num"] == 1
    assert result["replay_count"] == 1
    assert result["memory_bytes"] == organism.memory_bytes


def test_memory_bytes_decreases_after_deletion(organism: MinimalForgettingOrganism) -> None:
    """memory_bytes drops after raw traces are deleted."""
    for i in range(3):
        organism.teach(_make_episode(i))
    before = organism.memory_bytes

    del_before, del_after = organism.delete_raw_traces()
    after = organism.memory_bytes

    assert before > after
    assert del_before > del_after
