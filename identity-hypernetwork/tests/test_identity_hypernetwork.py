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
