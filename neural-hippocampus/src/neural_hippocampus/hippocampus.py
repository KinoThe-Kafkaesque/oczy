"""High-level Neural Hippocampus wrapper.

Exposes the public surface described in the project thesis:
- store/write high-surprise experience traces,
- reinforce/query replays relevant traces,
- consolidate replayed traces into slow-update summaries,
- inspect current memory state.
"""

from __future__ import annotations

from typing import Any

from .core import SurpriseGatedMemory


class NeuralHippocampus:
    """Surprise-gated neural memory, replay loop, and slow consolidation.

    This is a minimal functional prototype: the memory is synthetic, the
    consolidation is a simple clustering step, and no external vector database
    is required.  It is intended to demonstrate the fast-write / replay /
    slow-consolidate / decay loop without committing to a full differentiable
    implementation yet.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.memory = SurpriseGatedMemory(
            dim=self.config.get("dim", 64),
            surprise_threshold=self.config.get("surprise_threshold", 0.5),
            novelty_weight=self.config.get("novelty_weight", 0.5),
            replay_threshold=self.config.get("replay_threshold", 1),
            cluster_similarity=self.config.get("cluster_similarity", 0.65),
            seed=self.config.get("seed"),
        )
        self.slow_updates: list[dict[str, Any]] = []
        self.decay_after_consolidation = self.config.get(
            "decay_after_consolidation", True
        )

    def store(
        self,
        query: str,
        answer: str,
        correction: str,
        prediction_error: float,
    ) -> str | None:
        """Store a high-surprise experience episode.

        Returns the episode id if the trace was written, or ``None`` if its
        surprise fell below the configured gate.
        """
        episode = {
            "query": query,
            "answer": answer,
            "correction": correction,
            "prediction_error": float(prediction_error),
        }
        return self.memory.write(episode)

    def reinforce(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Replay the ``k`` stored episodes most relevant to ``query``.

        Each returned episode has its ``replay_count`` incremented, making it
        eligible for consolidation.
        """
        return self.memory.read_relevant(query, k=k)

    def consolidate(self) -> list[dict[str, Any]]:
        """Cluster frequently replayed traces and produce slow updates.

        If ``decay_after_consolidation`` is enabled (the default), raw traces
        that contributed to a slow update are removed from fast memory.
        """
        summaries = self.memory.consolidate()
        self.slow_updates.extend(summaries)

        if self.decay_after_consolidation:
            consolidated_ids = [
                episode_id
                for summary in summaries
                for episode_id in summary.get("trace_ids", [])
            ]
            self.memory.decay_raw_traces(consolidated_ids)

        return summaries

    def status(self) -> dict[str, Any]:
        """Return a serializable status snapshot with byte/episode counts."""
        return {
            "project": "neural_hippocampus",
            "ready": True,
            "episode_count": self.memory.episode_count(),
            "slow_update_count": len(self.slow_updates),
            "trace_bytes": self.memory.byte_count(),
        }

    def forward(self, x: Any) -> Any:
        """Placeholder one-step forward call.

        Not implemented in this minimal prototype; it exists to preserve the
        original scaffold method until a differentiable core is wired in.
        """
        raise NotImplementedError(
            "NeuralHippocampus.forward() is not implemented in the prototype."
        )
