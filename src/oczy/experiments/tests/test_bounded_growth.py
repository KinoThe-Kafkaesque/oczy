"""Tests for the Bounded-Growth Consolidation experiment (Experiment 06)."""

from __future__ import annotations

import math

import pytest

from oczy.experiments.bounded_growth.bounded_growth_eval import (
    A0bAutoencoder,
    A1Autoencoder,
    ConceptEmbeddingHypernetwork,
    _build_agent,
    _combined_footprint,
    _module_bytes,
    run,
)


# ---------------------------------------------------------------------------
# Module import / construction
# ---------------------------------------------------------------------------

def test_module_imports():
    """The experiment module imports without error."""
    from oczy.experiments.bounded_growth import bounded_growth_eval
    assert hasattr(bounded_growth_eval, "run")
    assert hasattr(bounded_growth_eval, "main")


def test_a0b_autoencoder_construction():
    ae = A0bAutoencoder(seed=42)
    assert ae.latent_dim == 32
    assert ae.residual_dim == 28
    assert ae._A.shape == (28, 1024)


def test_a1_autoencoder_construction():
    ae = A1Autoencoder(seed=42, rank=3)
    assert ae.latent_dim == 32
    assert ae.rank == 3
    assert ae._U.shape == (28, 3)
    assert ae._V.shape == (3, 1024)
    assert ae._D.shape == (16, 32)


def test_concept_embedding_hypernetwork_construction():
    hn = ConceptEmbeddingHypernetwork(latent_dim=4, seed=0)
    assert hn.latent_dim == 4
    assert hn.input_dim == 16
    assert hn.output_dim == len(hn.concepts)
    assert hn.W.shape == (hn.output_dim, hn.input_dim)


# ---------------------------------------------------------------------------
# Shape / encode checks
# ---------------------------------------------------------------------------

def test_a0b_encode_shape():
    ae = A0bAutoencoder(seed=42)
    episode = {
        "situation": "what color is the sky",
        "model_answer": "green",
        "correction": "no the sky is blue",
        "revised_answer": "blue",
        "outcome": "corrected",
    }
    dz = ae.encode(episode)
    assert dz.shape == (32,)


def test_a1_encode_shape():
    ae = A1Autoencoder(seed=42, rank=3)
    episode = {
        "situation": "capital of france",
        "model_answer": "lyon",
        "correction": "the capital is paris",
        "revised_answer": "paris",
        "outcome": "corrected",
    }
    dz = ae.encode(episode)
    assert dz.shape == (32,)


def test_a1_train_step_returns_float():
    ae = A1Autoencoder(seed=42, rank=3)
    episode = {
        "situation": "two plus two",
        "model_answer": "five",
        "correction": "two plus two is four",
        "revised_answer": "four",
        "outcome": "corrected",
    }
    loss = ae.train_step(episode, lr=0.05)
    assert isinstance(loss, float)
    assert loss >= 0.0


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def test_build_agent_a0():
    agent = _build_agent("A0")
    assert agent is not None
    assert hasattr(agent, "experience_autoencoder")
    assert hasattr(agent, "identity_hypernetwork")


def test_build_agent_a0b():
    agent = _build_agent("A0b")
    assert isinstance(agent.experience_autoencoder, A0bAutoencoder)


def test_build_agent_a1():
    agent = _build_agent("A1")
    assert isinstance(agent.experience_autoencoder, A1Autoencoder)


def test_build_agent_a2():
    agent = _build_agent("A2")
    assert isinstance(agent.experience_autoencoder, A1Autoencoder)
    assert isinstance(agent.identity_hypernetwork, ConceptEmbeddingHypernetwork)


def test_build_agent_a3():
    agent = _build_agent("A3")
    assert isinstance(agent.experience_autoencoder, A1Autoencoder)
    assert isinstance(agent.identity_hypernetwork, ConceptEmbeddingHypernetwork)
    assert agent.identity_hypernetwork.latent_dim == 2


def test_build_agent_unknown_raises():
    with pytest.raises(ValueError, match="Unknown condition"):
        _build_agent("nonsense")


# ---------------------------------------------------------------------------
# Footprint monotonicity
# ---------------------------------------------------------------------------

def test_a0b_footprint_le_a0_footprint():
    """A0b (seed-regenerable, excludes dense _A) must be <= A0 (dense _A)."""
    a0 = _build_agent("A0")
    a0b = _build_agent("A0b")
    a0_combined = _combined_footprint(a0)
    a0b_combined = _combined_footprint(a0b)
    assert a0b_combined <= a0_combined, (
        f"A0b combined ({a0b_combined}) should be <= A0 combined ({a0_combined})"
    )


def test_a1_footprint_le_a0_footprint():
    """A1 (low-rank float32) must be <= A0 (dense float64)."""
    a0 = _build_agent("A0")
    a1 = _build_agent("A1")
    a0_combined = _combined_footprint(a0)
    a1_combined = _combined_footprint(a1)
    assert a1_combined <= a0_combined, (
        f"A1 combined ({a1_combined}) should be <= A0 combined ({a0_combined})"
    )


def test_a3_footprint_le_a2_footprint():
    """A3 (ultra-compact HN) must be <= A2 (compact HN)."""
    a2 = _build_agent("A2")
    a3 = _build_agent("A3")
    a2_combined = _combined_footprint(a2)
    a3_combined = _combined_footprint(a3)
    assert a3_combined <= a2_combined, (
        f"A3 combined ({a3_combined}) should be <= A2 combined ({a2_combined})"
    )


# ---------------------------------------------------------------------------
# Full harness smoke test
# ---------------------------------------------------------------------------

def test_run_does_not_crash():
    """The full experiment harness runs without raising."""
    report = run(seed=0, n_levels=1)
    assert "m1_ratio" in report
    assert "per_condition" in report
    for cond in ("A0", "A0b", "A1", "A2", "A3", "REF-lo"):
        assert cond in report["per_condition"]
        assert "combined_footprint" in report["per_condition"][cond]


def test_run_m1_ratio_is_numeric():
    """m1_ratio should be a finite float (not NaN) when A0 has nonzero footprint."""
    report = run(seed=0, n_levels=1)
    m1 = report["m1_ratio"]
    assert not math.isnan(m1), "m1_ratio should not be NaN"
    assert m1 >= 0.0


def test_run_m1_ratio_acceptance():
    """m1_ratio should meet the <= 0.10 acceptance threshold."""
    report = run(seed=0, n_levels=1)
    m1 = report["m1_ratio"]
    assert m1 <= 0.10, f"m1_ratio {m1} exceeds acceptance threshold 0.10"
