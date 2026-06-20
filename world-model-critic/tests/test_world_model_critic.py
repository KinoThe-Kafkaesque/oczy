"""Tests for the World-Model Critic v1 prototype."""

import pytest

from world_model_critic import WorldModelCritic


AMBIGUOUS_QUERY = (
    "What are some possible reasons the result is maybe unclear or ambiguous?"
)
AMBIGUOUS_ANSWER = "It could be one of several things, perhaps likely unclear."


def test_instantiation_is_lightweight():
    critic = WorldModelCritic()
    # No heavy dependencies are initialized at construction time.
    assert critic.records == []
    assert len(critic.weights) == 4


def test_uncertainty_high_for_ambiguous_queries():
    """Before any feedback, ambiguous phrasing should look uncertain."""
    critic = WorldModelCritic()
    pred = critic.predict_acceptance(AMBIGUOUS_QUERY, AMBIGUOUS_ANSWER)

    assert 0.0 <= pred["accepted_prob"] <= 1.0
    assert 0.0 <= pred["correction_likelihood"] <= 1.0
    assert pred["accepted_prob"] == pytest.approx(1 - pred["correction_likelihood"])

    # The critic should ring uncertain (probability near 0.5 => variance near max).
    assert 0.3 < pred["correction_likelihood"] < 0.7
    assert pred["key_uncertainty"] > 0.4


def test_correction_likelihood_rises_for_similar_queries():
    """A recorded correction should make similar future queries look riskier."""
    critic = WorldModelCritic()

    original_query = "What is the profile view?"
    original_answer = "It shows the user account profile."
    similar_query = "What does the profile page display?"
    similar_answer = "The profile page displays the user account."

    baseline = critic.predict_acceptance(similar_query, similar_answer)

    critic.record_outcome(
        original_query,
        original_answer,
        "In this product, 'profile' means business vertical, not user profile.",
    )

    updated = critic.predict_acceptance(similar_query, similar_answer)

    assert updated["correction_likelihood"] > baseline["correction_likelihood"]
    # A correction on a similar query also lowers the predicted acceptance.
    assert updated["accepted_prob"] < baseline["accepted_prob"]


def test_prediction_error_decreases_with_feedback():
    """Repeated matching feedback should make the critic's predictions more accurate."""
    critic = WorldModelCritic()

    query = "Explain what a bank is."
    answer = "A bank is the side of a river."
    correction = "I meant a financial institution, not the side of a river."

    errors = []
    for _ in range(12):
        critic.predict_acceptance(query, answer)
        errors.append(critic.prediction_error(actual_was_correction=True))
        critic.record_outcome(query, answer, correction)

    # The critic should converge toward the observed correction probability.
    assert errors[-1] <= errors[0]
    assert errors[-1] < 0.2


def test_accepted_outcome_lowers_correction_likelihood():
    """Recording acceptance on a query should make similar queries look safer."""
    critic = WorldModelCritic()

    query = "What is two plus two?"
    answer = "Two plus two is four."
    similar_query = "What is 2 + 2?"

    pre = critic.predict_acceptance(similar_query, answer)
    critic.record_outcome(query, answer, None)
    post = critic.predict_acceptance(similar_query, answer)

    assert post["correction_likelihood"] <= pre["correction_likelihood"]


def test_prediction_error_with_no_prediction_is_maximal():
    critic = WorldModelCritic()
    assert critic.prediction_error(False) == 1.0
    assert critic.prediction_error(True) == 1.0


def test_status_reported_fields():
    """status() must expose the 6 cross-organ fields with correct types."""
    critic = WorldModelCritic()
    critic.predict_acceptance(AMBIGUOUS_QUERY, AMBIGUOUS_ANSWER)
    critic.record_outcome(AMBIGUOUS_QUERY, AMBIGUOUS_ANSWER, None)

    status = critic.status()

    assert set(status.keys()) == {
        "project",
        "ready",
        "record_count",
        "serialized_bytes",
        "weights",
        "ambiguous_word_count",
    }
    assert status["project"] == "world_model_critic"
    assert status["ready"] is True
    assert status["record_count"] == 1
    assert isinstance(status["serialized_bytes"], int)
    assert status["serialized_bytes"] > 0
    assert isinstance(status["weights"], list)
    assert all(isinstance(w, float) for w in status["weights"])
    assert status["ambiguous_word_count"] == len(critic.ambiguous_words)
    # status() must be a snapshot, not a live reference into the critic.
    assert status["weights"] is not critic.weights
