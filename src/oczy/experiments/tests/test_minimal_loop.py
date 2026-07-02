"""Unit tests for the MinimalOrganism metabolism loop (Sprint 2 / S2.1).

Exercises the public contract of ``MinimalOrganism`` — boot, teach, answer,
consolidate, prefix budget, memory bookkeeping — against the tiny random HF
model (``hf-internal-testing/tiny-random-LlamaForCausalLM``) so the tests run
fast and offline.  Behaviour is asserted, not plumbing: every test names the
externally observable contract it defends.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``minimal_loop`` imports ``eval.v2`` from the repo root, which is not on
# sys.path under pytest (only the package ``src`` trees are).  Add it once.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pytest
from neural_hippocampus import NeuralHippocampus

from oczy.experiments.minimal_loop import (
    MAX_PREFIX_TOKENS,
    MinimalOrganism,
    _HippocampusGuard,
    _spearman_rho,
)
from oczy.experiments.organism_curriculum.dataset import Episode, Probe
from oczy.lm.hf_driver import HFDriver

TEST_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def driver():
    with HFDriver.load(TEST_MODEL_ID) as d:
        yield d


@pytest.fixture
def org(driver):
    """Fresh, booted organism per test — hermetic, no cross-test state leak."""
    o = MinimalOrganism(driver)
    o.boot()
    return o


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    ep_id: str = "test_ep",
    request: str = "What is X?",
    correction: str = "X means Y.",
    probes: tuple[Probe, ...] | None = None,
) -> Episode:
    if probes is None:
        probes = (Probe(request=request, expected="Y", category="retention", match_mode="sense"),)
    return Episode(
        id=ep_id,
        initial_request=request,
        default_response="I don't know.",
        correction_utterance=correction,
        corrected_label="Y",
        corrected_response="X means Y.",
        domain="test",
        probes=probes,
    )


class _FakeHippocampus:
    """Hand-written fake recording every method call for guard assertions."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def store(self, **kwargs) -> str:
        self.calls.append("store")
        return "fake_id"

    def reinforce(self, query: str, k: int = 3) -> list[dict]:
        self.calls.append("reinforce")
        return []

    def consolidate(self) -> list[dict]:
        self.calls.append("consolidate")
        return []

    def status(self, include_size: bool = False) -> dict:
        self.calls.append("status")
        return {}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_minimal_organism_construction(driver) -> None:
    """Constructor wires cortex, hippocampus, and records d_cortex."""
    org = MinimalOrganism(driver, d_cortex=64, cortex_seed=7)
    assert org.d_cortex == 64
    assert org.cortex is not None
    assert org.cortex.config.d_cortex == 64
    # Hippocampus is wrapped in the guard but the raw organ is a real
    # NeuralHippocampus configured with the cortex dimension.
    assert isinstance(org._hippo_raw, NeuralHippocampus)
    assert org.hippocampus is not None


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def test_boot_resets_state(org) -> None:
    """boot() re-syncs warm_state to cold_state after warm has drifted.

    Defends the reset invariant: whatever the warm state absorbed during a
    session, a cold boot must discard it and re-load from cold.
    """
    # After fixture boot, warm == cold (both zero-initialised).
    assert np.allclose(org.cortex.warm_state, org.cortex.cold_state)

    # Teach mutates warm_state only; cold_state must be untouched.
    org.teach(_make_episode())
    assert not np.allclose(org.cortex.warm_state, org.cortex.cold_state)

    # boot() must re-sync warm to cold, discarding the warm drift.
    org.boot()
    assert np.allclose(org.cortex.warm_state, org.cortex.cold_state)


# ---------------------------------------------------------------------------
# Teaching
# ---------------------------------------------------------------------------


def test_teach_updates_cortex(org) -> None:
    """teach() perceives the correction and bumps both cortex counters."""
    assert org.cortex.update_count == 0
    assert org.cortex.correction_count == 0

    org.teach(_make_episode(correction="No, 'X' means Y."))

    assert org.cortex.update_count > 0
    assert org.cortex.correction_count > 0


# ---------------------------------------------------------------------------
# Answer / hippocampus guard
# ---------------------------------------------------------------------------


def test_answer_no_hippocampus_guard(org) -> None:
    """answer() never touches the hippocampus and clears the in-answer guard.

    Defends the spec ban on answer-time retrieval: a tracking fake replaces
    the hippocampus, answer() runs, and no non-status method may have been
    called.  The guard flag must be restored to False on exit so a later
    consolidation is not falsely blocked.
    """
    fake = _FakeHippocampus()
    org.hippocampus = _HippocampusGuard(fake)

    result = org.answer("What is X?", max_tokens=4)

    assert isinstance(result, str)
    # Guard flag restored.
    assert org.hippocampus._in_answer is False
    # No hippocampus method (status included) was consulted during answer.
    assert fake.calls == []


def test_hippocampus_guard_blocks_during_answer() -> None:
    """The guard raises RuntimeError for any non-status call while in-answer.

    Defends the ban mechanically: store/reinforce/consolidate must raise when
    the guard is armed, but status() stays callable for diagnostics.
    """
    guard = _HippocampusGuard(_FakeHippocampus())
    guard._in_answer = True

    with pytest.raises(RuntimeError, match="called during answer"):
        guard.store(query="q", answer="a", correction="c", prediction_error=1.0)
    with pytest.raises(RuntimeError, match="called during answer"):
        guard.reinforce("q", k=3)
    with pytest.raises(RuntimeError, match="called during answer"):
        guard.consolidate()

    # status() is explicitly whitelisted and must not raise.
    guard.status()


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def test_consolidate_builds_prefix(org, driver) -> None:
    """After teach + consolidate, the driver carries a non-empty prefix."""
    org.teach(_make_episode(correction="No, 'X' means Y."))
    org.consolidate()

    prefix = driver.articulation_prefix
    assert prefix is not None
    assert prefix.strip() != ""


def test_consolidate_returns_metadata(org) -> None:
    """consolidate() returns a metadata dict with the documented drift/prefix keys."""
    org.teach(_make_episode())
    meta = org.consolidate()

    assert isinstance(meta, dict)
    for key in ("cold_drift", "cvec_norm", "prefix_tokens", "prefix_overflow", "n_summaries"):
        assert key in meta, f"missing required metadata key: {key}"
    assert meta["n_summaries"] >= 1
    assert isinstance(meta["prefix_tokens"], int)
    assert isinstance(meta["prefix_overflow"], int)


def test_prefix_budget_max_48_tokens(org) -> None:
    """Corrections exceeding the 48-token budget are truncated with overflow reported.

    Defends the content-channel budget invariant: prefix_tokens never exceeds
    MAX_PREFIX_TOKENS, and any dropped tokens are counted in prefix_overflow.
    """
    # A single correction with far more than 48 distinct tokens.
    long_correction = " ".join(f"tokenword{i}" for i in range(80))
    org.teach(_make_episode(correction=long_correction, request="What is Z?"))
    meta = org.consolidate()

    assert meta["prefix_tokens"] <= MAX_PREFIX_TOKENS
    assert meta["prefix_overflow"] > 0


# ---------------------------------------------------------------------------
# Spearman rho helper
# ---------------------------------------------------------------------------


def test_spearman_rho() -> None:
    """_spearman_rho: +1 for monotonic-increasing, -1 for reversed, NaN for <3."""
    # Perfect positive monotonic relationship.
    rho_pos = _spearman_rho([1, 2, 3, 4, 5], [10.0, 20.0, 30.0, 40.0, 50.0])
    assert rho_pos == pytest.approx(1.0, abs=1e-6)

    # Perfect negative (reversed) monotonic relationship.
    rho_neg = _spearman_rho([1, 2, 3, 4, 5], [50.0, 40.0, 30.0, 20.0, 10.0])
    assert rho_neg == pytest.approx(-1.0, abs=1e-6)

    # Too few points -> NaN (guard against scipy-style crashes on tiny input).
    assert np.isnan(_spearman_rho([1, 2], [1.0, 2.0]))


# ---------------------------------------------------------------------------
# Memory bookkeeping
# ---------------------------------------------------------------------------


def test_memory_bytes_positive(org) -> None:
    """memory_bytes() reports a positive integer footprint."""
    n = org.memory_bytes()
    assert isinstance(n, int)
    assert n > 0


# ---------------------------------------------------------------------------
# Answer determinism
# ---------------------------------------------------------------------------


def test_answer_determinism_with_prefix(org) -> None:
    """Greedy decode with a consolidated prefix is deterministic across calls.

    Defends the reproducibility contract: same prefix + same request must
    yield identical output, since generate() is temperature=0 (greedy).
    """
    org.teach(_make_episode(correction="No, 'X' means Y."))
    org.consolidate()

    a = org.answer("What is X?", max_tokens=8)
    b = org.answer("What is X?", max_tokens=8)
    assert a == b
