"""Fast-weight / state-update layer for the Plastic Cortex.

The core idea is a bounded associative scratchpad that assigns credit to
labels.  Normal tokens write weakly; explicit corrections write strongly.
"""

from __future__ import annotations


class FastWeightLayer:
    """Associative fast-weights over a fixed label set.

    For every token we keep a small vector of per-label scores.  Calling
    ``update(token, target=label, correction=True)`` opens a large write gate
    and pushes that token toward the corrected label.  Ordinary updates use a
    much smaller plasticity value.

    This is a toy implementation of the "Synaptic Scratchpad" described in
    ``experiments.txt``: a bounded matrix of fast weights where corrections
    have higher write priority than normal text.
    """

    def __init__(
        self,
        labels: list[str],
        alpha_normal: float = 0.02,
        alpha_correction: float = 2.0,
        lateral_inhibition: float = 0.1,
    ) -> None:
        self.labels = list(labels)
        self.alpha_normal = alpha_normal
        self.alpha_correction = alpha_correction
        self.lateral_inhibition = lateral_inhibition
        self.weights: dict[str, dict[str, float]] = {}
        self.writes = 0
        self.correction_writes = 0

    def update(self, token: str, correction: bool = False, target: str | None = None) -> None:
        """Write a token into the fast-weight matrix.

        Args:
            token: The surface form to associate with a label.
            correction: If True, use a strong plasticity value.
            target: The label to strengthen.  If None, this is a no-op.
        """
        if target is None or target not in self.labels:
            return

        alpha = self.alpha_correction if correction else self.alpha_normal
        token_row = self.weights.setdefault(token, {label: 0.0 for label in self.labels})

        for label in self.labels:
            if label == target:
                token_row[label] += alpha
            else:
                token_row[label] -= alpha * self.lateral_inhibition

        self.writes += 1
        if correction:
            self.correction_writes += 1

    def scores(self, token: str) -> dict[str, float]:
        """Return the per-label fast-weight scores for a token."""
        return dict(self.weights.get(token, {label: 0.0 for label in self.labels}))

    def reset_state(self) -> None:
        """Clear the fast-weight matrix."""
        self.weights.clear()
        self.writes = 0
        self.correction_writes = 0

    def state_snapshot(self) -> dict[str, dict[str, float]]:
        """Return a serializable copy of the entire fast-weight state."""
        return {token: dict(scores) for token, scores in self.weights.items()}
