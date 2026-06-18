"""Tiny recurrent-state module for the Plastic Cortex.

This is intentionally not a trained neural net.  It is a deterministic,
fixed-projection recurrent cell whose role is:

1. Turn each incoming token into a compressed hidden-state update.
2. Keep a persistent state trajectory across a conversation.
3. Give the cortex a "session context" that changes because of what it has seen.

It proves the *shape* of state-space tracking without needing torch.
"""

from __future__ import annotations

import math
import random


class TokenRNN:
    """Minimal recurrent cell driven by hashed token embeddings.

    Each token is projected to an input vector via a deterministic hash, then
    the hidden state is updated with a small Elman-style step:

        h_{t+1} = tanh(W_x @ x_t + W_h @ h_t + b)

    The weight matrices are randomly initialized from a fixed seed so the
    module is deterministic across restarts.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 8, seed: int = 0) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        rng_state = random.Random(seed)

        def rand_matrix(rows: int, cols: int) -> list[list[float]]:
            scale = math.sqrt(cols)
            return [[(rng_state.random() * 2.0 - 1.0) / scale for _ in range(cols)] for _ in range(rows)]

        self.W_x = rand_matrix(input_dim, hidden_dim)
        self.W_h = rand_matrix(hidden_dim, hidden_dim)
        self.b = [0.0] * hidden_dim
        self.reset_state()

    def _token_embedding(self, token: str) -> list[float]:
        """Deterministic pseudo embedding for a token.

        This avoids building a vocabulary table.  Any token is reduced to a
        fixed-size real vector derived from its hash.
        """
        rng = random.Random((hash(token) + 0x7F) & 0xFFFFFFFF)
        return [(rng.random() * 2.0 - 1.0) for _ in range(self.input_dim)]

    def update(self, token: str) -> None:
        """Advance the recurrent state by one token."""
        x = self._token_embedding(token)

        # Compute pre-activation: b + x @ W_x + h @ W_h
        pre = list(self.b)
        for j, x_val in enumerate(x):
            row = self.W_x[j]
            for i in range(self.hidden_dim):
                pre[i] += x_val * row[i]
        for j, h_val in enumerate(self._h):
            row = self.W_h[j]
            for i in range(self.hidden_dim):
                pre[i] += h_val * row[i]

        self._h = [math.tanh(v) for v in pre]

    def state_snapshot(self) -> list[float]:
        """Return the current compressed recurrent state."""
        return list(self._h)

    def reset_state(self) -> None:
        """Zero the recurrent state."""
        self._h = [0.0] * self.hidden_dim
