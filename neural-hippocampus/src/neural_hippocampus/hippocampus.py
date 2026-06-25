"""High-level Neural Hippocampus wrapper.

Exposes the public surface described in the project thesis:
- store/write high-surprise experience traces,
- reinforce/query replays relevant traces,
- consolidate replayed traces into slow-update summaries,
- inspect current memory state.

Episode contract
----------------
``store()`` writes an Episode dict to the underlying SurpriseGatedMemory.
The keys it emits are constrained to the canonical Episode fields listed in
``oczy_common/episode.py`` (the cross-organ schema source of truth):

    query, answer, correction, corrected_answer, outcome,
    prediction_error, source, id, replay_count

The organ does NOT import ``oczy_common`` (it stays self-contained), but it
keeps its dict keys aligned with that contract so the glue layer can round-trip
episodes without field-name drift. ``corrected_answer`` is the recovered label
and is optional here because not every correction carries one.
"""

from __future__ import annotations

import pickle
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
        corrected_answer: str | None = None,
        hidden: np.ndarray | None = None,
    ) -> str | None:
        """Store a high-surprise experience episode.

        Returns the episode id if the trace was written, or ``None`` if its
        surprise fell below the configured gate.

        ``hidden`` is an optional LM hidden vector for the episode.  When
        provided, it is passed through to ``SurpriseGatedMemory`` and later
        surfaced as ``representative_hidden`` in consolidation summaries.
        """
        episode = {
            "query": query,
            "answer": answer,
            "correction": correction,
            "prediction_error": float(prediction_error),
        }
        if corrected_answer is not None:
            episode["corrected_answer"] = corrected_answer
        if hidden is not None:
            episode["hidden"] = hidden
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

    def status(self, include_size: bool = False) -> dict[str, Any]:
        """Return a serializable status snapshot with byte/episode counts.

        Fields:
        - ``project``: organ name tag.
        - ``ready``: always True for this prototype.
        - ``episode_count``: number of raw traces currently in fast memory.
        - ``record_count``: total episodes written (same as ``episode_count``
          for this organ; standardized key for cross-organ status consumers).
        - ``slow_update_count``: number of consolidated slow-update summaries.
        - ``trace_bytes``: approximate size of the raw trace buffer (pickle).
        - ``serialized_bytes``: only present when ``include_size=True``;
          avoids expensive pickle calls in hot loops.
        """
        result = {
            "project": "neural_hippocampus",
            "ready": True,
            "episode_count": self.memory.episode_count(),
            "record_count": self.memory.episode_count(),
            "slow_update_count": len(self.slow_updates),
            "trace_bytes": self.memory.byte_count(),
        }
        if include_size:
            result["serialized_bytes"] = len(
                pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
            )
        return result

    def forward(self, x: Any) -> Any:
        """Placeholder one-step forward call.

        Not implemented in this minimal prototype; it exists to preserve the
        original scaffold method until a differentiable core is wired in.
        """
        raise NotImplementedError(
            "NeuralHippocampus.forward() is not implemented in the prototype."
        )
