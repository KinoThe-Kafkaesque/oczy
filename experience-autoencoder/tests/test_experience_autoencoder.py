"""Tests for the Experience Autoencoder prototype."""

import json

import numpy as np
import pytest

from experience_autoencoder import (
    ExperienceAutoencoder,
    ExperienceDecoder,
    ExperienceEncoder,
)


def _profile_episode(outcome: str = "corrected") -> dict:
    return {
        "situation": "What does the profile field represent in a multi-tenant app?",
        "model_answer": "The profile field stores the user's public biography.",
        "correction": "No, profile means the business vertical the tenant belongs to.",
        "revised_answer": "The profile field identifies the business vertical for the tenant.",
        "outcome": outcome,
    }


def test_public_api_includes_expected_classes():
    assert ExperienceEncoder is not None
    assert ExperienceDecoder is not None
    assert ExperienceAutoencoder is not None


def test_encode_yields_bounded_length_vector():
    ae = ExperienceAutoencoder()
    episode = _profile_episode()
    delta_z = ae.encode(episode)

    assert delta_z.shape == (32,)
    assert np.isfinite(delta_z).all()
    assert (delta_z >= -1.0).all() and (delta_z <= 1.0).all()


def test_decode_reconstructs_key_fields_with_reasonable_accuracy():
    ae = ExperienceAutoencoder()
    episode = _profile_episode("corrected")
    decoded = ae.decode(ae.encode(episode))

    assert decoded["failure_class"] in {
        "semantic_misgrounding",
        "fact_correction",
    }

    hint = decoded["corrected_behavior_hint"]
    assert isinstance(hint, dict)
    assert len(hint) > 0
    for key, value in hint.items():
        assert isinstance(key, str)
        assert isinstance(value, str)

    triggers = decoded["trigger_conditions"]
    assert isinstance(triggers, list)
    assert len(triggers) > 0
    assert all(isinstance(t, str) for t in triggers)
    # Trigger tokens should be from the episode vocabulary, not random noise.
    episode_tokens = _episode_tokens(episode)
    overlap = len(set(triggers) & episode_tokens) / max(len(triggers), 1)
    assert overlap >= 0.25

    counters = decoded["counterexamples"]
    assert isinstance(counters, list)
    assert len(counters) > 0
    assert all(isinstance(c, str) for c in counters)

    error = ae.reconstruction_error(episode, decoded)
    assert 0.0 <= error <= 1.0
    assert error < 0.8


def test_update_identity_changes_z():
    ae = ExperienceAutoencoder()
    z0 = np.zeros(32, dtype=float)
    episode = _profile_episode("corrected")
    z1 = ae.update_identity(z0, episode)

    assert z1.shape == (32,)
    assert not np.allclose(z0, z1)


def test_compress_reduces_bytes():
    ae = ExperienceAutoencoder()
    episodes = [
        _profile_episode("corrected"),
        {
            "situation": "How do I safely parametrize a sqlx query?",
            "model_answer": "Concatenate user input into the query string.",
            "correction": "That risks SQL injection. Use parameterized queries.",
            "revised_answer": "Use the sqlx::query function with placeholders.",
            "outcome": "failed",
        },
    ]

    deltas = ae.compress(episodes)
    assert len(deltas) == len(episodes)
    assert all(d.shape == (32,) for d in deltas)

    raw_bytes = sum(len(json.dumps(ep).encode("utf-8")) for ep in episodes)
    delta_bytes = sum(d.nbytes for d in deltas)
    assert delta_bytes < raw_bytes


def test_outcome_mapping():
    ae = ExperienceAutoencoder()
    accepted = _profile_episode("accepted")
    failed = {
        "situation": "Run the deploy script.",
        "model_answer": "cd prod && rm -rf /",
        "correction": "Never run destructive commands in production.",
        "revised_answer": "Run the read-only validation script first.",
        "outcome": "failed",
    }

    assert ae.decode(ae.encode(accepted))["failure_class"] == "none"
    assert ae.decode(ae.encode(failed))["failure_class"] == "execution_error"


def _episode_tokens(episode: dict) -> set[str]:
    import re

    tokens = set()
    for key in ("situation", "model_answer", "correction", "revised_answer"):
        tokens.update(re.findall(r"[a-z0-9]+", str(episode.get(key, "")).lower()))
    return tokens
