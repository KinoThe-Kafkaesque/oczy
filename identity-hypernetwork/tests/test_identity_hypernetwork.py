"""Smoke tests for Identity Hypernetwork."""

import numpy as np
import pytest

from identity_hypernetwork import IdentityHypernetwork, IdentityLatents


def test_instantiation_and_status():
    agent = IdentityHypernetwork()
    status = agent.status()
    assert status["project"] == "identity_hypernetwork"
    assert status["ready"] is True
    assert status["latent_dim"] == 8
    assert status["num_concepts"] == 14


def test_latents_zero_initialised():
    latents = IdentityLatents(dim=8)
    assert latents.z_user.shape == (8,)
    assert np.allclose(latents.z_user, 0)
    assert np.allclose(latents.z_style, 0)
    assert np.allclose(latents.z_mistakes, 0)
    assert latents.to_array().shape == (32,)


def test_identity_vector_changes_after_lesson():
    agent = IdentityHypernetwork(seed=7)
    before = agent.latents.z_user.copy()
    agent.update_identity(
        {"token": "profile", "correct_label": "business vertical", "source": "user_correction"}
    )
    after = agent.latents.z_user.copy()
    assert not np.allclose(before, after)


def test_generated_adapter_influences_target_token_score():
    agent = IdentityHypernetwork(seed=11)
    before = agent.generate_adapters()["concept_scores"]["business"]
    agent.update_identity(
        {"token": "profile", "correct_label": "business vertical", "source": "user_correction"}
    )
    after = agent.generate_adapters()["concept_scores"]["business"]
    assert after > before + 1e-6


def test_different_z_components_affect_different_aspects():
    agent = IdentityHypernetwork(seed=3)

    z_user_before = agent.latents.z_user.copy()
    z_style_before = agent.latents.z_style.copy()

    # A user correction should only move z_user.
    agent.update_identity(
        {"token": "profile", "correct_label": "business", "source": "user_correction"}
    )
    assert not np.allclose(agent.latents.z_user, z_user_before)
    assert np.allclose(agent.latents.z_style, z_style_before)

    # A style lesson should move z_style and raise the "formal" score.
    formal_before = agent.generate_adapters()["concept_scores"]["formal"]
    z_style_before2 = agent.latents.z_style.copy()
    agent.update_identity(
        {"token": "tone", "correct_label": "formal", "source": "style"}
    )
    formal_after = agent.generate_adapters()["concept_scores"]["formal"]
    assert not np.allclose(agent.latents.z_style, z_style_before2)
    assert formal_after > formal_before + 1e-6

    # A mistake lesson should move z_mistakes and raise the "error" score.
    error_before = agent.generate_adapters()["concept_scores"]["error"]
    z_mistakes_before = agent.latents.z_mistakes.copy()
    agent.update_identity(
        {"token": "bug", "correct_label": "error patterns", "source": "mistake"}
    )
    error_after = agent.generate_adapters()["concept_scores"]["error"]
    assert not np.allclose(agent.latents.z_mistakes, z_mistakes_before)
    assert error_after > error_before + 1e-6


def test_latents_roundtrip():
    latents = IdentityLatents(dim=4)
    latents.z_user += [0.1, 0.2, 0.3, 0.4]
    latents.z_domain += [0.5, 0.6, 0.7, 0.8]
    latents.z_style += [-0.1, -0.2, -0.3, -0.4]
    latents.z_mistakes += [1.0, 2.0, 3.0, 4.0]
    data = latents.to_dict()
    restored = IdentityLatents.from_dict(data)
    assert np.allclose(latents.to_array(), restored.to_array())


def test_latents_grow_increases_dim():
    latents = IdentityLatents(dim=4)
    latents.z_user += [0.1, 0.2, 0.3, 0.4]
    grown = latents.grow(8)
    assert grown.dim == 8
    assert grown.z_user.shape == (8,)
    assert np.allclose(grown.z_user[:4], [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(grown.z_user[4:], 0)


def test_latents_grow_rejects_invalid_dim():
    latents = IdentityLatents(dim=4)
    with pytest.raises(ValueError):
        latents.grow(4)
    with pytest.raises(ValueError):
        latents.grow(2)


def test_hypernetwork_grow_increases_latent_dim():
    agent = IdentityHypernetwork(latent_dim=4, seed=1)
    agent.update_identity(
        {"token": "profile", "correct_label": "business", "source": "user_correction"}
    )
    before_latent = agent.latents.to_array().copy()

    grown = agent.grow(8)
    assert grown.latent_dim == 8
    assert grown.latents.dim == 8
    assert grown.W.shape == (14, 32)
    assert grown.status()["latent_dim"] == 8
    assert np.allclose(grown.latents.to_array()[:16], before_latent)
    assert np.allclose(grown.latents.to_array()[16:], 0)

    # Old concept ranking should be preserved because the learned latents are.
    after_scores = grown.generate_adapters()["concept_scores"]
    assert after_scores["business"] > after_scores["account"]


def test_hypernetwork_grow_rejects_invalid_dim():
    agent = IdentityHypernetwork(latent_dim=8)
    with pytest.raises(ValueError):
        agent.grow(8)
    with pytest.raises(ValueError):
        agent.grow(4)


def test_hypernetwork_grow_preserves_learning_rate_and_concepts():
    agent = IdentityHypernetwork(latent_dim=4, seed=2, learning_rate=0.42)
    grown = agent.grow(8)
    assert grown.lr == 0.42
    assert grown.concepts == agent.concepts
    assert grown.concept_index == agent.concept_index


def test_hypernetwork_grow_can_continue_learning():
    agent = IdentityHypernetwork(latent_dim=4, seed=3)
    agent.update_identity(
        {"token": "profile", "correct_label": "business", "source": "user_correction"}
    )
    grown = agent.grow(8)

    formal_before = grown.generate_adapters()["concept_scores"]["formal"]
    grown.update_identity(
        {"token": "tone", "correct_label": "formal", "source": "style"}
    )
    formal_after = grown.generate_adapters()["concept_scores"]["formal"]
    assert formal_after > formal_before + 1e-6


def test_grow_vocab_extends_W_consistently():
    agent = IdentityHypernetwork(seed=5)
    initial_rows = agent.W.shape[0]
    initial_concepts = list(agent.concepts)

    agent.grow_vocab(["git", "branch"])

    assert agent.W.shape[0] == initial_rows + 2
    assert agent.W.shape[1] == agent.input_dim
    assert agent.concepts[: len(initial_concepts)] == initial_concepts
    assert agent.concepts[-2:] == ["git", "branch"]
    assert agent.concept_index["git"] == initial_rows
    assert agent.concept_index["branch"] == initial_rows + 1
    # Row for "git" is finite, right shape, and points into the appended block.
    git_row = agent.W[agent.concept_index["git"]]
    assert git_row.shape == (agent.input_dim,)
    assert np.all(np.isfinite(git_row))
    # Idempotent: re-adding one of the new concepts must not expand W.
    agent.grow_vocab(["git", "docker"])
    assert agent.W.shape[0] == initial_rows + 3
    assert agent.concept_index["docker"] == initial_rows + 2
    assert agent.concept_index["git"] == initial_rows  # unchanged


def test_update_identity_learns_unknown_label():
    agent = IdentityHypernetwork(seed=9)
    assert "model" not in agent.concepts  # not in the initial 14-token vocab
    initial_concept_count = len(agent.concepts)
    initial_W_shape = agent.W.shape

    agent.update_identity(
        {"source": "user_correction", "correct_label": "ML model"}
    )

    # "ML model" tokenises to ["ml", "model"]; "ml" (len 2) is rejected by the
    # auto-grow filter, so "model" is registered and learned.
    assert "model" in agent.concepts
    assert agent.concept_index["model"] == initial_concept_count
    assert len(agent.concepts) == initial_concept_count + 1
    assert agent.W.shape == (initial_W_shape[0] + 1, initial_W_shape[1])
    scores = agent.generate_adapters()["concept_scores"]
    assert "model" in scores
    assert isinstance(scores["model"], float)


def test_status_reports_serialized_bytes_and_record_count():
    agent = IdentityHypernetwork()
    status = agent.status()
    assert status["record_count"] == len(agent.concepts)
    assert status["record_count"] == status["num_concepts"]
    assert isinstance(status["serialized_bytes"], int)
    assert status["serialized_bytes"] > 0
    # Growing vocab should increase both record_count and serialized_bytes.
    before_bytes = status["serialized_bytes"]
    before_count = status["record_count"]
    agent.grow_vocab(["docker", "container"])
    after = agent.status()
    assert after["record_count"] == before_count + 2
    assert after["serialized_bytes"] > before_bytes
