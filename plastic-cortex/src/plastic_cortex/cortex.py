"""Public Plastic Cortex: a correction-gated recurrent toy.

This is not an LLM.  It is a deliberately tiny word-association toy model
that proves the architecture mechanism described in ``experiments.txt``:

    recurrent state  +  fast-weight scratchpad  +  correction gating

The cortex adapts in real time: after a single correction it will lean toward
the corrected sense on future queries.
"""

from __future__ import annotations

import math
import random
import re

from .fast_weight import FastWeightLayer
from .state import TokenRNN


class PlasticCortex:
    """Correction-gated recurrent / state-space organ.

    The toy domain is the word "profile".  Before correction the cortex
    answers "user profile" when asked about a profile.  A single
    ``correct("profile", "business vertical")`` flips the bias so that later
    requests containing "profile" lean toward "business vertical".
    """

    LABELS: list[str] = ["user profile", "business vertical"]

    # Slow priors: weak default associations between tokens and labels.
    # We intentionally make "profile" strongly default to "user profile" so the
    # correction test is non-trivial.
    BASELINE: dict[str, dict[str, float]] = {
        "profile": {"user profile": 1.5},
        "user": {"user profile": 0.6},
        "account": {"user profile": 0.6},
        "login": {"user profile": 0.4},
        "business": {"business vertical": 0.6},
        "vertical": {"business vertical": 0.6},
        "industry": {"business vertical": 0.4},
        "market": {"business vertical": 0.3},
    }

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})

        # Instance copies so mutations do not leak across cortex instances.
        self.labels = list(self.LABELS)
        self.baseline = {token: dict(scores) for token, scores in self.BASELINE.items()}

        self.hidden_dim = self.config.get("hidden_dim", 8)
        self.alpha_normal = self.config.get("alpha_normal", 0.02)
        self.alpha_correction = self.config.get("alpha_correction", 5.0)
        self.recurrent_gain = self.config.get("recurrent_gain", 0.05)

        reset_rng = random.Random(42)

        self.rnn = TokenRNN(input_dim=8, hidden_dim=self.hidden_dim, seed=0)
        self.fast = FastWeightLayer(
            labels=list(self.labels),
            alpha_normal=self.alpha_normal,
            alpha_correction=self.alpha_correction,
        )

        # Fixed random projection from recurrent state to label scores.
        # It is kept small so it cannot override baseline + fast weights.
        self._recurrent_gate: dict[str, list[float]] = {
            label: [(reset_rng.random() * 2.0 - 1.0) * math.sqrt(1.0 / self.hidden_dim) for _ in range(self.hidden_dim)]
            for label in self.labels
        }

        self.answer_count = 0
        self.correction_count = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Very simple tokenizer: lower-case words."""
        return [t for t in re.findall(r"\b\w+\b", text.lower()) if t]

    def _ensure_label(self, label: str) -> str:
        """Make sure a label exists in the scoring set."""
        if label not in self.labels:
            self.labels.append(label)
            self.fast.labels.append(label)
            for row in self.fast.weights.values():
                row[label] = 0.0
            for row in self.baseline.values():
                row.setdefault(label, 0.0)
            self._recurrent_gate[label] = [
                (random.Random(hash(label) + i).random() * 2.0 - 1.0) * math.sqrt(1.0 / self.hidden_dim)
                for i in range(self.hidden_dim)
            ]
        return label

    def _recurrent_bias(self) -> dict[str, float]:
        """Small label bias derived from the recurrent state."""
        h = self.rnn.state_snapshot()
        return {
            label: sum(g * h_i for g, h_i in zip(self._recurrent_gate[label], h))
            for label in self.labels
        }

    def _score(self, tokens: list[str]) -> dict[str, float]:
        """Score each label for a token sequence."""
        scores: dict[str, float] = {label: 0.0 for label in self.labels}

        # Slow priors + fast weights from tokens.
        for token in tokens:
            baseline = self.baseline.get(token, {})
            fast = self.fast.scores(token)
            for label in self.labels:
                scores[label] += baseline.get(label, 0.0)
                scores[label] += fast.get(label, 0.0)

        # Session-level recurrent nudge.
        recurrent_bias = self._recurrent_bias()
        for label in self.labels:
            scores[label] += self.recurrent_gain * recurrent_bias[label]

        return scores

    def _best(self, scores: dict[str, float]) -> str:
        """Pick the highest-scoring label; tie-break by slow-prior label order."""
        best_label = self.labels[0]
        best_score = scores[best_label]
        for label in self.labels[1:]:
            if scores[label] > best_score:
                best_label = label
                best_score = scores[label]
        return best_label

    def answer(self, query: str) -> str:
        """Answer a query and perform a low-plasticity state update."""
        tokens = self._tokenize(query)

        # Update recurrent context only with the query tokens.
        for token in tokens:
            self.rnn.update(token)

        scores = self._score(tokens)
        chosen = self._best(scores)

        # Normal tokens write weakly toward the answer that was just produced.
        for token in tokens:
            self.fast.update(token, correction=False, target=chosen)

        self.answer_count += 1
        return chosen

    def correct(self, correction_text: str, expected_answer: str) -> None:
        """Apply an explicit correction with high plasticity.

        Args:
            correction_text: Text containing the token(s) to re-ground.
            expected_answer: The label the cortex should have produced.
        """
        expected_answer = self._ensure_label(expected_answer)
        tokens = self._tokenize(correction_text)

        # Corrections update both the recurrent state and the fast weights.
        for token in tokens:
            self.rnn.update(token)
            self.fast.update(token, correction=True, target=expected_answer)

        self.correction_count += 1

    def status(self) -> dict:
        """Return a serializable status snapshot."""
        return {
            "ready": True,
            "labels": list(self.labels),
            "hidden_state": self.rnn.state_snapshot(),
            "fast_weights": self.fast.state_snapshot(),
            "writes": self.fast.writes,
            "correction_writes": self.fast.correction_writes,
            "answers": self.answer_count,
            "corrections": self.correction_count,
        }

    def reset_state(self) -> None:
        """Reset all mutable session state."""
        self.rnn.reset_state()
        self.fast.reset_state()
        self.answer_count = 0
        self.correction_count = 0
