"""Smoke tests for Identity Hypernetwork."""

import pickle

import numpy as np
import pytest

from identity_hypernetwork import IdentityHypernetwork, IdentityLatents


def test_instantiation_and_status():
    agent = IdentityHypernetwork()
    status = agent.status(include_size=True)
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
    assert grown.status(include_size=True)["latent_dim"] == 8
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
    status = agent.status(include_size=True)
    assert status["record_count"] == len(agent.concepts)
    assert status["record_count"] == status["num_concepts"]
    assert isinstance(status["serialized_bytes"], int)
    assert status["serialized_bytes"] > 0
    # Growing vocab should increase both record_count and serialized_bytes.
    before_bytes = status["serialized_bytes"]
    before_count = status["record_count"]
    agent.grow_vocab(["docker", "container"])
    after = agent.status(include_size=True)
    assert after["record_count"] == before_count + 2
    assert after["serialized_bytes"] > before_bytes

def test_concept_capacity_prunes_oldest_concepts_and_keeps_W_aligned():
    """Once concepts exceed max_concepts, oldest by insertion age are pruned."""
    agent = IdentityHypernetwork(
        seed=1,
        config={"max_concepts": 18, "concept_decay_fraction": 0.5},
    )

    # Add five brand-new concepts; total becomes 19 > 18.
    agent.grow_vocab(["alpha", "bravo", "charlie", "delta", "echo"])

    assert len(agent.concepts) <= agent.max_concepts
    status = agent.status()
    assert status["max_concepts"] == 18
    assert status["pruned_concepts"] > 0
    assert status["num_concepts"] == len(agent.concepts)
    assert agent.W.shape[0] == len(agent.concepts)

    # The newest concepts should survive; at least one of the oldest originals
    # must have been dropped because the cap is below the post-growth count.
    for c in ["alpha", "bravo", "charlie", "delta", "echo"]:
        assert c in agent.concepts

    # Adapter scores remain aligned with the trimmed vocabulary.
    scores = agent.generate_adapters()["concept_scores"]
    assert set(scores.keys()) == set(agent.concepts)
    assert len(scores) == agent.W.shape[0]


def test_update_identity_auto_growth_honours_concept_capacity():
    """Auto-grown concepts from unknown labels are also subject to pruning."""
    agent = IdentityHypernetwork(
        seed=2,
        config={"max_concepts": 16, "concept_decay_fraction": 0.25},
    )
    # Each lesson auto-registers a new concept, potentially triggering pruning.
    for label in ["kubernetes", "terraform", "prometheus", "grafana"]:
        agent.update_identity(
            {"source": "user_correction", "correct_label": label}
        )

    assert len(agent.concepts) <= agent.max_concepts
    assert agent.status()["pruned_concepts"] >= 0
    assert agent.W.shape[0] == len(agent.concepts)


def test_state_adapter_disabled_by_default():
    agent = IdentityHypernetwork(seed=1)
    assert agent.state_dim is None
    assert agent.W_state is None
    assert agent.state_adapters == {}
    status = agent.status()
    assert status["state_dim"] is None
    assert status["state_adapters_initialised"] is False
    with pytest.raises(TypeError):
        agent.generate_state_adapter()


def test_generate_state_adapter_lazy_initializes():
    agent = IdentityHypernetwork(seed=2)
    adapter = agent.generate_state_adapter(8)
    assert adapter.shape == (8,)
    assert agent.W_state is not None
    assert agent.W_state.shape == (8, agent.input_dim)
    assert agent.state_dim == 8
    for concept in agent.concepts:
        arr = agent.state_adapters[concept]
        assert arr.shape == (8,)
        assert np.allclose(arr, 0)


def test_update_identity_moves_state_adapter():
    agent = IdentityHypernetwork(seed=3, state_dim=8)
    target = "business"
    agent.update_identity({"source": "user_correction", "correct_label": target})
    assert agent.W_state is not None
    assert target in agent.state_adapters
    assert np.linalg.norm(agent.state_adapters[target]) > 1e-9


def test_state_adapter_affects_output():
    agent = IdentityHypernetwork(seed=4, state_dim=8)
    before = agent.generate_state_adapter(8).copy()
    agent.update_identity({"source": "user_correction", "correct_label": "business"})
    after = agent.generate_state_adapter(8)
    assert not np.allclose(before, after)


def test_pickle_roundtrip_state_adapters():
    agent = IdentityHypernetwork(seed=5, state_dim=8)
    agent.update_identity({"source": "user_correction", "correct_label": "business"})
    before = agent.generate_state_adapter(8).copy()
    data = pickle.dumps(agent, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(data)
    assert restored.state_dim == agent.state_dim
    assert np.allclose(restored.W_state, agent.W_state)
    for concept in agent.concepts:
        assert np.allclose(restored.state_adapters[concept], agent.state_adapters[concept])
    after = restored.generate_state_adapter(8)
    assert np.allclose(after, before)
