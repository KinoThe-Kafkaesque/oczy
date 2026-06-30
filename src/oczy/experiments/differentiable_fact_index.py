"""Differentiable Fact Index — DSI-style learned embedding matrix for Oczy.

Replaces the cosine-similarity scope-slot lookup with a learned fact embedding
matrix F where each row is a trainable representation of one fact/experience.
Retrieval is a single inner product: ``scores = query_hidden @ F.T``.

A LoRA-style adapter ΔW = A @ B accumulates experiential modulation so that
recent corrections bias retrieval toward the corrected sense without
overwriting the baseline fact embeddings.

Architecture
------------
    F matrix (n_facts × d_model)   — static baseline (what is true)
    ΔW = A·B adapter               — experience modulation (what happened)
    retrieval = query @ (F + ΔW).T — combined disambiguation

The F matrix maps to the DSI "model IS the index" paradigm. The LoRA adapter
maps to IncDSI's incremental constrained optimization — new experiences
modify retrieval without touching the baseline.

Use
---
    idx = DifferentiableFactIndex(n_facts=64, d_model=2048, lora_rank=8)
    idx.store(query_embedding, fact_label, is_correction=True)
    labels = idx.retrieve(query_embedding, k=3)
"""

from __future__ import annotations

import numpy as np


def _hebbian_update(
    weight: np.ndarray, query: np.ndarray, target_idx: int, lr: float = 0.01
) -> None:
    """Hebbian-style update: move target row toward query, push others away.

    weight[target_idx] += lr * query
    weight[others]      -= lr * query / (n - 1)
    """
    weight[target_idx] += lr * query
    n = weight.shape[0]
    if n > 1:
        decay = lr / (n - 1)
        for i in range(n):
            if i != target_idx:
                weight[i] -= decay * query


class DifferentiableFactIndex:
    """Learned fact embedding matrix with LoRA experience modulation.

    Parameters
    ----------
    n_facts : int
        Maximum number of facts to store (rows in F).
    d_model : int
        Dimensionality of query embeddings.
    lora_rank : int
        Rank of the LoRA adapter ΔW = A @ B.
    lr_fact : float
        Learning rate for Hebbian fact embedding updates.
    lr_lora : float
        Learning rate for LoRA adapter updates.
    """

    def __init__(
        self,
        n_facts: int = 64,
        d_model: int = 2048,
        lora_rank: int = 8,
        lr_fact: float = 0.01,
        lr_lora: float = 0.05,
    ) -> None:
        self.n_facts = n_facts
        self.d_model = d_model
        self.lora_rank = min(lora_rank, n_facts, d_model)
        self.lr_fact = lr_fact
        self.lr_lora = lr_lora

        # Static fact embedding matrix F: each row is one fact.
        scale = 1.0 / np.sqrt(d_model)
        self.F = np.random.randn(n_facts, d_model).astype(np.float32) * scale
        # Normalise so initial inner products are well-behaved.
        norms = np.linalg.norm(self.F, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.F = (self.F / norms * scale * np.sqrt(d_model)).astype(np.float32)

        # LoRA adapter: ΔW = A @ B
        self.A = np.random.randn(d_model, self.lora_rank).astype(np.float32) * scale
        self.B = np.random.randn(self.lora_rank, n_facts).astype(np.float32) * scale

        # Fact labels (one string per row).
        self._labels: list[str] = [""] * n_facts

        # Next available row index.
        self._next_idx: int = 0

        # Whether each row is occupied.
        self._occupied: list[bool] = [False] * n_facts

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        query_embedding: np.ndarray,
        label: str,
        *,
        is_correction: bool = False,
    ) -> int:
        """Store a fact keyed by *query_embedding*.

        Always allocates a new row in F (like IncDSI's frozen document
        vectors).  Corrections modulate the LoRA adapter rather than
        overwriting the baseline embedding.
        """
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        if q.shape[0] != self.d_model:
            raise ValueError(
                f"Expected query embedding of shape ({self.d_model},), "
                f"got {q.shape}"
            )

        idx = self._next_idx % self.n_facts
        # Unit-normalise so inner products are well-behaved.
        self.F[idx] = q / (np.linalg.norm(q) + 1e-8)
        self._labels[idx] = label
        self._occupied[idx] = True
        self._next_idx = (idx + 1) % self.n_facts

        if is_correction:
            self._update_lora(q, idx)

        return idx

    def _find_closest(self, q: np.ndarray) -> int | None:
        """Return the index of the row closest to *q*, or None if none."""
        if not any(self._occupied):
            return None
        scores = self.F @ q  # (n_facts,)
        best = int(np.argmax(scores))
        if not self._occupied[best]:
            return None
        # Only return if similarity is above threshold.
        sim = float(scores[best]) / (np.linalg.norm(self.F[best]) * np.linalg.norm(q) + 1e-8)
        if sim < 0.5:
            return None
        return best

    def _update_lora(self, q: np.ndarray, target_idx: int) -> None:
        """Update LoRA adapter toward the target fact.

        Moves B[:, target_idx] toward A.T @ q so that ΔW strengthens
        the target fact's retrieval score for similar future queries.
        """
        target = self.A.T @ q  # (rank,)
        delta = target - self.B[:, target_idx]
        self.B[:, target_idx] += self.lr_lora * delta
        # Lightly decay other columns.
        self.B *= (1.0 - self.lr_lora * 0.1)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        *,
        use_lora: bool = False,
    ) -> list[tuple[str, float]]:
        """Return top-k (label, score) pairs for *query_embedding*.

        Scores are inner products between the query and the combined
        (F + ΔW) matrix.  The LoRA adapter ΔW is only applied when
        *use_lora* is True.
        """
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        if q.shape[0] != self.d_model:
            raise ValueError(
                f"Expected query embedding of shape ({self.d_model},), "
                f"got {q.shape}"
            )

        effective = self.F
        if use_lora:
            effective = self.F + (self.A @ self.B).T  # (n_facts, d_model)

        scores = effective @ q  # (n_facts,)
        # Only consider occupied rows.
        mask = np.array(self._occupied, dtype=bool)
        if not mask.any():
            return []

        # Mask unoccupied rows to -inf.
        masked = np.where(mask, scores, -1e10)
        top_indices = np.argsort(masked)[::-1][:k]

        result: list[tuple[str, float]] = []
        for i in top_indices:
            if not self._occupied[i]:
                continue
            result.append((self._labels[i], float(scores[i])))
        return result

    def retrieve_baseline(
        self, query_embedding: np.ndarray, k: int = 3
    ) -> list[tuple[str, float]]:
        """Retrieve using only the baseline F matrix (no LoRA modulation)."""
        return self.retrieve(query_embedding, k, use_lora=False)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a serialisable state snapshot."""
        return {
            "F": self.F.copy(),
            "A": self.A.copy(),
            "B": self.B.copy(),
            "_labels": np.array(self._labels, dtype=object),
            "_occupied": np.array(self._occupied, dtype=bool),
            "_next_idx": np.array(self._next_idx),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Restore from a state snapshot."""
        self.F = state["F"].copy()
        self.A = state["A"].copy()
        self.B = state["B"].copy()
        self._labels = list(state["_labels"])
        self._occupied = list(state["_occupied"])
        self._next_idx = int(state["_next_idx"])
        self.n_facts = self.F.shape[0]
        self.d_model = self.F.shape[1]
        self.lora_rank = self.B.shape[0]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return a human-readable status snapshot."""
        return {
            "n_facts": self.n_facts,
            "n_occupied": sum(self._occupied),
            "d_model": self.d_model,
            "lora_rank": self.lora_rank,
            "labels": [l for l, o in zip(self._labels, self._occupied) if o],
        }
