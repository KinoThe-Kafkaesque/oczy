"""Basic tests for the Plastic Cortex toy prototype.

These tests focus on the core guarantee: explicit corrections open a stronger
write gate than ordinary text.
"""

import pytest

from plastic_cortex import PlasticCortex


def test_initial_answer_defaults_to_prior():
    """Before any correction the cortex uses its slow prior."""
    agent = PlasticCortex()

    assert agent.answer("What is a profile?") == "user profile"
    assert agent.answer("Show me my profile page.") == "user profile"
    assert agent.status()["ready"] is True


def test_correction_shifts_future_answer():
    """A single correction retargets the token toward the corrected sense."""
    agent = PlasticCortex()

    assert agent.answer("What is a profile?") == "user profile"

    agent.correct("profile means business vertical", "business vertical")

    # Same keyword, same prior question, but the corrected sense now wins.
    assert agent.answer("What is a profile?") == "business vertical"
    assert agent.answer("Open a profile view.") == "business vertical"
    assert agent.status()["corrections"] == 1


def test_correction_stronger_than_normal_text():
    """Normal conversation updates are weak; explicit corrections are strong.

    Repeatedly answering queries that contain competing cues does not flip the
    prior.  One explicit correction flips it immediately.
    """
    agent = PlasticCortex()

    # Normal plasticity only: these answers reinforce whatever was chosen.
    for _ in range(25):
        agent.answer("profile means business vertical")

    # The slow prior for "profile" is still dominant after normal exposure.
    assert agent.answer("What is a profile?") == "user profile"

    # A single correction with high plasticity overrides the prior.
    agent.correct("profile", "business vertical")
    assert agent.answer("What is a profile?") == "business vertical"

    # Sanity check that the fast-weight state recorded a correction write.
    assert agent.status()["correction_writes"] >= 1


def test_reset_state_clears_adaptation():
    """Resetting state should return the cortex to its priors."""
    agent = PlasticCortex()

    agent.correct("profile", "business vertical")
    assert agent.answer("What is a profile?") == "business vertical"

    agent.reset_state()

    assert agent.status()["fast_weights"] == {}
    assert agent.answer("What is a profile?") == "user profile"


def test_fast_weight_state_snapshot_is_serializable():
    """The fast-weight layer exposes a serializable compressed state."""
    agent = PlasticCortex()
    agent.correct("profile", "business vertical")

    snapshot = agent.fast.state_snapshot()
    assert "profile" in snapshot
    assert snapshot["profile"]["business vertical"] > snapshot["profile"]["user profile"]


def test_status_reports_serialized_bytes_and_record_count():
    """status() must expose the cross-organ serialized_bytes / record_count fields."""
    agent = PlasticCortex()

    # Drive a little learning so record_count is non-zero.
    agent.answer("What is a profile?")
    agent.correct("profile means business vertical", "business vertical")
    agent.answer("What is a profile?")

    status = agent.status()

    assert status["project"] == "plastic_cortex"
    assert status["serialized_bytes"] > 0
    assert status["record_count"] == agent.correction_count
    assert status["record_count"] >= 1
