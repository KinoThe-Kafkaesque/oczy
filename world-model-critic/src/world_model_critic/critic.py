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

import numpy as np

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
        # Hidden-vector MLP configuration.
        self.d_hidden = int(cfg.get("d_hidden", 0))
        self.mlp_hidden_units = int(cfg.get("mlp_hidden_units", 16))
        self.use_hidden = bool(cfg.get("use_hidden", False))

        # MLP weights for tensor-input mode (lazy-initialized).
        self.W1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.W2: np.ndarray | None = None
        self.b2: float = 0.0
        # Optional learned value head on the MLP hidden representation.
        self.use_value_head = bool(cfg.get("use_value_head", False))
        self.value_learning_rate = float(cfg.get("value_learning_rate", 0.1))
        self.gamma = float(cfg.get("gamma", 0.95))
        self.Wv: np.ndarray | None = None
        self.bv: float = 0.0
        self._last_value: float | None = None
        self._last_td_error: float | None = None

        self.ambiguous_words = frozenset(cfg.get("ambiguous_words", AMBIGUOUS_WORDS))

        # Cap linear record growth.
        self.max_records = int(cfg.get("max_records", 10_000))
        self.record_decay_fraction = float(cfg.get("record_decay_fraction", 0.25))
        self.records_pruned = 0

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
        self.records: list[dict] = []

        # Last probability of correction emitted by the model; used by
        # `prediction_error`.
        self._last_correction_prob: float | None = None

    def __setstate__(self, state: dict) -> None:
        """Restore a pickled critic; inject defaults added after v1."""
        self.__dict__.update(state)
        self.d_hidden = int(getattr(self, "d_hidden", 0))
        self.mlp_hidden_units = int(getattr(self, "mlp_hidden_units", 16))
        self.use_hidden = bool(getattr(self, "use_hidden", False))
        self.W1 = getattr(self, "W1", None)
        self.b1 = getattr(self, "b1", None)
        self.W2 = getattr(self, "W2", None)
        self.b2 = float(getattr(self, "b2", 0.0))
        self.use_value_head = bool(getattr(self, "use_value_head", False))
        self.value_learning_rate = float(getattr(self, "value_learning_rate", 0.1))
        self.gamma = float(getattr(self, "gamma", 0.95))
        self.Wv = getattr(self, "Wv", None)
        self.bv = float(getattr(self, "bv", 0.0))
        self._last_value = getattr(self, "_last_value", None)
        self._last_td_error = getattr(self, "_last_td_error", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_acceptance(
        self,
        query: str,
        proposed_answer: str,
        lm_hidden: np.ndarray | None = None,
    ) -> dict:
        """Predict the outcome of proposing `proposed_answer` for `query`.

        Returns a dict with:
          - accepted_prob: probability the answer is accepted as-is.
          - correction_likelihood: probability the user supplies a correction.
          - key_uncertainty: aggregate uncertainty (high when near 0.5).
        """
        x = self._features(query, proposed_answer)
        if self.use_hidden and lm_hidden is not None:
            self._ensure_mlp(lm_hidden.shape[0])
            correction_prob, _ = self._mlp_forward(x, lm_hidden)
        else:
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

    def predict_value(
        self,
        query: str,
        proposed_answer: str,
        lm_hidden: np.ndarray | None = None,
    ) -> float:
        """Return the estimated future return (value) for this state.

        If the value head is disabled or no hidden vector is supplied, the
        estimate is 0.0 to keep the default API unchanged.
        """
        if not self.use_value_head or lm_hidden is None:
            return 0.0

        self._ensure_mlp(lm_hidden.shape[0])
        self._ensure_value_head()

        x = self._features(query, proposed_answer)
        # Hidden activation shared with the correction-prediction MLP.
        _, cache = self._mlp_forward(x, lm_hidden)
        h = cache[2]
        return float(self.Wv @ h + self.bv)

    def record_outcome(
        self,
        query: str,
        proposed_answer: str,
        correction: str | None,
        lm_hidden: np.ndarray | None = None,
        next_lm_hidden: np.ndarray | None = None,
        value_lm_hidden: np.ndarray | None = None,
        next_value_lm_hidden: np.ndarray | None = None,
    ) -> None:
        """Record what actually happened and perform one online update.

        `correction` is a non-empty string if the user corrected the answer,
        otherwise None/empty to indicate acceptance.

        When `use_value_head` is enabled the hidden vector is treated as a
        state representation and, optionally, `next_lm_hidden` as the next
        state, for a one-step TD update of the value estimate.
        """
        actual_correction = correction is not None and bool(str(correction).strip())
        target = 1.0 if actual_correction else 0.0

        # Prediction error is measured against the model *before* it saw this
        # outcome.
        use_mlp = self.use_hidden and lm_hidden is not None
        x_prior = self._features(query, proposed_answer)
        if use_mlp:
            self._ensure_mlp(lm_hidden.shape[0])
            prob_prior, _ = self._mlp_forward(x_prior, lm_hidden)
        else:
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

        # Bound linear organ growth: if the record list exceeds its configured
        # capacity, drop the oldest fraction of records (or however many are
        # needed to get back under the cap).
        if len(self.records) > self.max_records:
            n = len(self.records)
            remove = max(int(n * self.record_decay_fraction), n - self.max_records)
            self.records = self.records[remove:]
            self.records_pruned += remove

        # Recompute features with the new record visible (important for the
        # prior-correction-rate term) and take one gradient step.
        x_post = self._features(query, proposed_answer)

        # Hidden-vector MLP update.
        if use_mlp:
            prob_post, cache = self._mlp_forward(x_post, lm_hidden)
            error = target - prob_post
            x_input, z1, h, _ = cache
            # Gradient of MSE/cross-entropy error w.r.t. weights.
            self.W2 = self.W2 + self.learning_rate * error * h
            self.b2 += self.learning_rate * error
            grad_h = error * self.W2
            grad_z1 = grad_h * (1.0 - h * h)
            self.W1 = self.W1 + self.learning_rate * np.outer(grad_z1, x_input)
            self.b1 = self.b1 + self.learning_rate * grad_z1

        # String-logistic model update (always kept live as a fallback).
        prob_post = self._predict_correction_prob(x_post)
        error = target - prob_post
        for i in self._learnable:
            self.weights[i] += self.learning_rate * error * x_post[i]

        # Soft clamp: keep the learnable correction-history weight within a
        # modest range so a run of corrections does not drown out the fixed
        # priors for entirely novel queries.
        self.weights[3] = max(-4.0, min(4.0, self.weights[3]))

        # Optional value-head TD update.
        v_hidden = value_lm_hidden if value_lm_hidden is not None else lm_hidden
        v_next_hidden = (
            next_value_lm_hidden
            if next_value_lm_hidden is not None
            else next_lm_hidden
        )
        if self.use_value_head and v_hidden is not None:
            self._ensure_mlp(v_hidden.shape[0])
            self._ensure_value_head()
            x_value = self._features(query, proposed_answer)
            _, cache = self._mlp_forward(x_value, v_hidden)
            h = cache[2]
            v_s = float(self.Wv @ h + self.bv)
            reward = -1.0 if actual_correction else 1.0
            v_next = (
                self.predict_value(query, proposed_answer, v_next_hidden)
                if v_next_hidden is not None
                else 0.0
            )
            td_error = reward + self.gamma * v_next - v_s
            self.Wv += self.value_learning_rate * td_error * h
            self.bv += self.value_learning_rate * td_error
            self._last_value = v_s
            self._last_td_error = td_error

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
            "record_capacity": self.max_records,
            "records_pruned": self.records_pruned,
            "weights": list(self.weights),
            "ambiguous_word_count": len(self.ambiguous_words),
            "use_value_head": self.use_value_head,
            "last_value": self._last_value,
            "last_td_error": self._last_td_error,
        }
        if include_size:
            result["serialized_bytes"] = len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
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

    def _ensure_mlp(self, d_hidden: int) -> None:
        """Lazy-initialize the small hidden-vector MLP if not present."""
        n_string_features = 4
        if self.d_hidden > 0 and d_hidden != self.d_hidden:
            # Observed hidden dimension has changed; force re-initialization.
            self.W1 = None
        self.d_hidden = d_hidden
        input_dim = n_string_features + d_hidden
        if (
            self.W1 is not None
            and self.b1 is not None
            and self.W2 is not None
            and self.W1.shape == (self.mlp_hidden_units, input_dim)
            and self.b1.shape == (self.mlp_hidden_units,)
            and self.W2.shape == (self.mlp_hidden_units,)
        ):
            return
        rng = np.random.RandomState((id(self) & 0xFFFFFFFF) ^ d_hidden)
        self.W1 = rng.randn(self.mlp_hidden_units, input_dim) * 0.01
        self.b1 = rng.randn(self.mlp_hidden_units) * 0.01
        self.W2 = rng.randn(self.mlp_hidden_units) * 0.01
        self.b2 = 0.0

    def _ensure_value_head(self) -> None:
        """Lazy-initialize the linear value head on the MLP hidden state."""
        if self.Wv is not None and self.Wv.shape == (self.mlp_hidden_units,):
            return
        rng = np.random.RandomState((id(self) & 0xFFFFFFFF) ^ self.mlp_hidden_units)
        self.Wv = rng.randn(self.mlp_hidden_units) * 0.01
        self.bv = 0.0

    def _mlp_forward(
        self,
        x_str: list[float],
        lm_hidden: np.ndarray,
    ) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """Forward pass of the hidden-vector MLP.

        Returns the scalar correction probability and a cache of intermediate
        activations needed for the online gradient update.
        """
        x_input = np.concatenate([np.asarray(x_str, dtype=float), lm_hidden])
        z1 = self.W1 @ x_input + self.b1
        h = np.tanh(z1)
        z2 = float(self.W2 @ h + self.b2)
        p = _sigmoid(z2)
        return p, (x_input, z1, h, p)

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
        z = sum(w * xi for w, xi in zip(self.weights, x, strict=False))
        return _sigmoid(z)
