"""Core implementation of the World-Model Critic.

A lightweight, NumPy-free critic that predicts whether an agent answer will be
accepted or corrected.  It learns online from each outcome using hand-built
semantic features and simple logistic-regression updates.

This is intentionally minimal (v1).  It exists to make the world-model idea
concrete and testable before swapping in richer learned components.

Episode contract (cross-organ schema lives in ``oczy_common.episode``; this
organ does not import that module, but reads and writes dict keys that match
its ``Episode`` TypedDict):

- ``WorldModelCritic.predict_acceptance(query, proposed_answer)`` reads the
  ``query`` and ``proposed_answer`` (= ``answer`` field) of an in-flight
  episode and returns a correction likelihood.
- ``WorldModelCritic.record_outcome(query, proposed_answer, correction)``
  appends the episode to ``self.records`` and performs an online weight
  update.  After prediction + outcome, ``WorldModelCritic.prediction_error()``
  yields the scalar ``prediction_error`` field (in ``[0, 1]``) that the
  hippocampus consumes to decide whether to gate the write into long-term
  storage.
"""

from __future__ import annotations

import math
import pickle
import re
from typing import Optional

# Hard-coded lexical markers of uncertainty.  These are the "fast priors" the
# critic starts with; the rest of the learning is data-driven.
AMBIGUOUS_WORDS = frozenset(
    {
        "some",
        "maybe",
        "probably",
        "possibly",
        "perhaps",
        "likely",
        "often",
        "sometimes",
        "unclear",
        "ambiguous",
        "vague",
        "several",
        "few",
        "many",
        "most",
        "all",
        "any",
        "someone",
        "something",
        "anything",
        "everything",
        "everyone",
        "could",
        "might",
        "may",
        "seems",
        "appears",
    }
)

_ALPHA_NUM = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _ALPHA_NUM.findall(text.lower())


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _sigmoid(z: float) -> float:
    # Numerically stable sigmoid.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


class WorldModelCritic:
    """Predict answer acceptance/correction and learn from outcomes online."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.config = cfg

        # Learning hyperparameters.
        self.learning_rate = float(cfg.get("learning_rate", 1.0))
        self.similarity_threshold = float(cfg.get("similarity_threshold", 0.25))
        self.ambiguous_words = frozenset(cfg.get("ambiguous_words", AMBIGUOUS_WORDS))

        # Model weights for [bias, ambiguity, length_ratio, prior_correction_rate].
        # In v1 only the prior-correction weight is updated online; the fixed
        # weights act as cheap priors that keep novel queries near a conservative
        # baseline while the model learns which *similar* queries get corrected.
        self.weights = [
            float(cfg.get("bias", -0.2)),
            float(cfg.get("ambiguous_weight", 2.0)),
            float(cfg.get("length_ratio_weight", -0.1)),
            float(cfg.get("prior_correction_weight", 0.0)),
        ]
        self._learnable = tuple(cfg.get("learnable_weights", [3]))

        # Online memory: every outcome is stored as a small bag-of-words record.
        # Growth is linear; v1 does not consolidate or decay old records.
        self.records: list[dict] = []

        # Last probability of correction emitted by the model; used by
        # `prediction_error`.
        self._last_correction_prob: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_acceptance(self, query: str, proposed_answer: str) -> dict:
        """Predict the outcome of proposing `proposed_answer` for `query`.

        Returns a dict with:
          - accepted_prob: probability the answer is accepted as-is.
          - correction_likelihood: probability the user supplies a correction.
          - key_uncertainty: aggregate uncertainty (high when near 0.5).
        """
        x = self._features(query, proposed_answer)
        correction_prob = self._predict_correction_prob(x)
        accepted_prob = 1.0 - correction_prob
        # Variance of a Bernoulli with p = correction_prob; peaks at 0.5.
        uncertainty = math.sqrt(correction_prob * (1.0 - correction_prob))

        self._last_correction_prob = correction_prob
        return {
            "accepted_prob": accepted_prob,
            "correction_likelihood": correction_prob,
            "key_uncertainty": uncertainty,
        }

    def record_outcome(
        self,
        query: str,
        proposed_answer: str,
        correction: str | None,
    ) -> None:
        """Record what actually happened and perform one online update.

        `correction` is a non-empty string if the user corrected the answer,
        otherwise None/empty to indicate acceptance.
        """
        actual_correction = correction is not None and bool(str(correction).strip())
        target = 1.0 if actual_correction else 0.0

        # Prediction error is measured against the model *before* it saw this
        # outcome.
        x_prior = self._features(query, proposed_answer)
        prob_prior = self._predict_correction_prob(x_prior)
        self._last_correction_prob = prob_prior

        # Update memory first so the similarity feature for this record can be
        # learned immediately.
        self.records.append(
            {
                "query": query,
                "answer": proposed_answer,
                "tokens": _token_set(query),
                "corrected": actual_correction,
            }
        )

        # Recompute features with the new record visible (important for the
        # prior-correction-rate term) and take one gradient step.
        x_post = self._features(query, proposed_answer)
        prob_post = self._predict_correction_prob(x_post)
        error = target - prob_post
        for i in self._learnable:
            self.weights[i] += self.learning_rate * error * x_post[i]

        # Soft clamp: keep the learnable correction-history weight within a
        # modest range so a run of corrections does not drown out the fixed
        # priors for entirely novel queries.
        self.weights[3] = max(-4.0, min(4.0, self.weights[3]))

    def prediction_error(self, actual_was_correction: bool) -> float:
        """Return |predicted_correction_probability - actual_outcome|.

        If no prediction has been made yet, the error is maximal (1.0).
        """
        if self._last_correction_prob is None:
            return 1.0
        actual = 1.0 if actual_was_correction else 0.0
        return abs(self._last_correction_prob - actual)

    def status(self, include_size: bool = False) -> dict:
        """Return a standardized status snapshot for cross-organ metrics.

        The shape is shared across organs so the agent glue layer's memory
        metrics can compare them apples-to-apples.  All values are plain
        Python types so the result is JSON-serializable.
        """
        result = {
            "project": "world_model_critic",
            "ready": True,
            "record_count": len(self.records),
            "weights": list(self.weights),
            "ambiguous_word_count": len(self.ambiguous_words),
        }
        if include_size:
            result["serialized_bytes"] = len(
                pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _features(self, query: str, proposed_answer: str) -> list[float]:
        """Build the feature vector used by the logistic model."""
        q_tokens = _tokens(query)
        a_tokens = _tokens(proposed_answer)
        all_tokens = q_tokens + a_tokens

        # x0: bias.
        x0 = 1.0

        # x1: normalized ambiguity count.
        ambiguous_hits = sum(1 for t in all_tokens if t in self.ambiguous_words)
        x1 = ambiguous_hits / max(len(all_tokens), 1)

        # x2: length ratio, capped to limit numeric range.
        # The 3.0 cap prevents very long answers from saturating the
        # logistic input; beyond ~3x the query length the additional text
        # rarely changes the correction signal meaningfully.
        x2 = min(
            len(proposed_answer) / max(len(query), 1),
            3.0,
        )

        # x3: correction rate among previous similar queries (0..1).
        x3 = self._similar_correction_rate(query)

        return [x0, x1, x2, x3]

    def _similar_correction_rate(self, query: str) -> float:
        """Return the empirical correction rate for queries similar to `query`."""
        if not self.records:
            return 0.0

        query_tokens = _token_set(query)
        similar_total = 0
        similar_corrected = 0

        for rec in self.records:
            sim = _jaccard(query_tokens, rec["tokens"])
            if sim >= self.similarity_threshold:
                similar_total += 1
                if rec["corrected"]:
                    similar_corrected += 1

        if similar_total == 0:
            return 0.0
        return similar_corrected / similar_total

    def _predict_correction_prob(self, x: list[float]) -> float:
        z = sum(w * xi for w, xi in zip(self.weights, x))
        return _sigmoid(z)
