"""Core memory primitives for the Neural Hippocampus.

This module implements a minimal surprise-gated episodic memory.  It is
intentionally small: synthetic embeddings, cosine similarity, and a simple
replay/clustering consolidation loop.  Nothing here depends on a vector DB
or an external embedding model.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
import uuid
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np


class SurpriseGatedMemory:
    """Fast hippocampal buffer that stores high-surprise episodes.

    An episode is retained only when its *surprise* (a scalar computed from a
    prediction error and a simple novelty estimate) exceeds a threshold. Once
    retained, episodes can be retrieved by cosine similarity to a query embedding,
    clustered into compressed slow-update summaries, and pruned after their
    structure has been consolidated.
    """

    def __init__(
        self,
        dim: int = 64,
        surprise_threshold: float = 0.5,
        novelty_weight: float = 0.5,
        replay_threshold: int = 1,
        cluster_similarity: float = 0.65,
        seed: int | None = None,
    ) -> None:
        """Create a surprise-gated memory.

        Args:
            dim: Dimensionality of synthetic episode embeddings.
            surprise_threshold: Minimum surprise required to write a trace.
            novelty_weight: Weight of the novelty term when blending novelty
                with prediction error.
            replay_threshold: Minimum number of replay events for an episode to
                be considered for consolidation.
            cluster_similarity: Minimum cosine similarity for an episode to be
                merged into an existing consolidation cluster.
            seed: Optional seed used only for tie-breaking, not for embedding
                content.
        """
        if dim < 2:
            raise ValueError("dim must be at least 2")
        self.dim = dim
        self.surprise_threshold = surprise_threshold
        self.novelty_weight = novelty_weight
        self.replay_threshold = replay_threshold
        self.cluster_similarity = cluster_similarity
        self._rng = np.random.default_rng(seed)

        # Fast episodic store: uuid -> episode
        self.traces: dict[str, dict[str, Any]] = {}

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def write(self, episode: dict[str, Any]) -> str | None:
        """Attempt to store an episode.  Returns its id or ``None``.

        The supplied dict should contain at least ``query`` and
        ``prediction_error``.  The following fields are populated/written:

        - ``id``: a v4-style UUID string.
        - ``embedding``: a small synthetic unit vector derived from ``query``.
        - ``novelty``: average cosine distance to existing traces.
        - ``surprise``: blended function of ``prediction_error`` and novelty.

        Other fields (``answer``, ``correction``, etc.) are stored verbatim.
        """
        query = episode.get("query", "")
        prediction_error = float(episode.get("prediction_error", 0.0))

        embedding = self._embed(str(query))
        novelty = self._novelty(embedding)
        surprise = min(
            1.0,
            prediction_error * (1.0 + self.novelty_weight * novelty),
        )

        if surprise < self.surprise_threshold:
            return None

        episode_id = str(uuid.uuid4())
        stored = {
            **episode,
            "id": episode_id,
            "embedding": embedding,
            "novelty": float(novelty),
            "surprise": float(surprise),
            "replay_count": 0,
        }
        self.traces[episode_id] = stored
        return episode_id

    def read_relevant(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Return the top-``k`` episodes most similar to ``query``.

        Returned episodes are shallow copies with ``embedding`` converted to a
        Python list.  This call increments ``replay_count`` for each returned
        episode.
        """
        if not self.traces:
            return []

        k = max(0, min(int(k), len(self.traces)))
        query_vec = self._embed(str(query))

        scored: list[tuple[float, str]] = []
        for episode_id, trace in self.traces.items():
            sim = self._cosine(query_vec, trace["embedding"])
            scored.append((sim, episode_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:k]

        results = []
        for _sim, episode_id in selected:
            self.traces[episode_id]["replay_count"] += 1
            results.append(self._export_trace(episode_id))
        return results

    def consolidate(self) -> list[dict[str, Any]]:
        """Compress frequently replayed episodes into slow-update summaries.

        The algorithm is intentionally simple and fully decoupled from any
        downstream adapter model:

        1. Select traces whose ``replay_count`` meets ``replay_threshold``.
        2. Greedy cluster by cosine similarity using ``cluster_similarity``.
        3. Emit one summary per cluster with an averaged representation.

        The returned summaries include the original ``trace_ids`` so callers can
        selectively prune raw traces afterwards.
        """
        eligible = sorted(
            (
                trace
                for trace in self.traces.values()
                if trace["replay_count"] >= self.replay_threshold
            ),
            key=lambda t: t["replay_count"],
            reverse=True,
        )

        if not eligible:
            return []

        clusters: list[dict[str, Any]] = []
        for trace in eligible:
            emb = trace["embedding"]
            best = None
            best_sim = -np.inf
            for cluster in clusters:
                sim = self._cosine(emb, cluster["center"])
                if sim > best_sim:
                    best_sim = sim
                    best = cluster

            if best is not None and best_sim >= self.cluster_similarity:
                best["traces"].append(trace)
                best["center"] = self._center([c["center"] for c in best["traces"]])
            else:
                clusters.append({
                    "traces": [trace],
                    "center": emb.copy(),
                })

        summaries = []
        for idx, cluster in enumerate(clusters):
            traces = cluster["traces"]
            count = len(traces)
            total_replay = sum(t["replay_count"] for t in traces)
            avg_surprise = float(np.mean([t["surprise"] for t in traces]))
            corrections = [t.get("correction", "") for t in traces if t.get("correction")]
            queries = [t.get("query", "") for t in traces]
            representative = queries[0]

            summaries.append({
                "id": f"slow_update_{idx}_{uuid.uuid4().hex[:8]}",
                "n_episodes": count,
                "trace_ids": [t["id"] for t in traces],
                "representative_query": representative,
                "summary_corrections": corrections,
                "avg_surprise": round(avg_surprise, 4),
                "total_replay": total_replay,
                "embedding": cluster["center"].astype(float).tolist(),
            })

        return summaries

    def decay_raw_traces(self, consolidated_ids: Sequence[str]) -> None:
        """Remove raw episodes whose structure has already been consolidated."""
        for episode_id in consolidated_ids:
            self.traces.pop(episode_id, None)

    # --------------------------------------------------------------------- #
    # Inspection helpers
    # --------------------------------------------------------------------- #

    def episode_count(self) -> int:
        return len(self.traces)

    def byte_count(self) -> int:
        """Approximate serialized size of the raw trace buffer in bytes."""
        try:
            return len(pickle.dumps(self.traces))
        except Exception as exc:  # pragma: no cover - defensive fallback
            warnings.warn(f"Could not pickle traces for sizing: {exc}")
            return sum(sys.getsizeof(t) for t in self.traces.values())

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _embed(self, text: str) -> np.ndarray:
        """Deterministic unit-vector embedding from a string.

        This is *not* semantic; it gives a stable, compact representation that
        maps the same query to the same vector and different queries to nearly
        orthogonal vectors, which is sufficient for a prototype replay test.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return vec
        return vec / norm

    def _novelty(self, embedding: np.ndarray) -> float:
        """Average cosine distance from ``embedding`` to existing traces."""
        if not self.traces:
            return 1.0
        sims = [
            self._cosine(embedding, trace["embedding"])
            for trace in self.traces.values()
        ]
        return float(1.0 - np.mean(sims))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity in [-1, 1]."""
        dot = float(np.dot(a, b))
        # Both inputs are unit vectors after embedding, but guard against drift.
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _center(embeddings: list[np.ndarray]) -> np.ndarray:
        """Mean of a list of unit vectors, re-normalized."""
        mean = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-12:
            return mean
        return mean / norm

    def _export_trace(self, episode_id: str) -> dict[str, Any]:
        """Return a serializable copy of a stored trace."""
        trace = self.traces[episode_id].copy()
        trace["embedding"] = trace["embedding"].astype(float).tolist()
        return trace
