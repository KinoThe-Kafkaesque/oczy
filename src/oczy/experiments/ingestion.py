"""Ingestion pipeline contracts and shared types.

This module defines the data structures and base classes for the new
chunker → salience filter → embedder → aggregator pipeline that sits
upstream of ``CortexAgent.metabolize()``.  The pipeline turns a single
utterance into many ``ChunkSignal`` traces (stored directly in the
hippocampus, bypassing the digestive gate) and one ``TurnDigest``
(statistics consumed by the digestive gate).

The contracts are intentionally small so that concrete stage
implementations can evolve without changing call sites.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Chunk:
    """A raw segment of an utterance produced by a chunker."""

    text: str
    span: tuple[int, int]
    position: int


@dataclass(frozen=True)
class ChunkSignal:
    """An annotated chunk ready for hippocampal storage.

    The ``hidden`` field is populated only when the embedder stage has
    produced a dense representation for the chunk (same-LM or foreign).
    The ``drift`` field is populated only when the chunk has been observed
    by the cortex (sequential/parallel observation mode).
    """

    text: str
    span: tuple[int, int]
    salience: float
    embedded: bool
    hidden: np.ndarray | None
    drift: float | None
    is_correction: bool


@dataclass(frozen=True)
class TurnDigest:
    """Aggregated per-turn statistics consumed by the digestive gate."""

    n_chunks: int
    n_survived: int
    drift_max: float
    drift_mean: float
    drift_p90: float
    correction_fraction: float
    novelty_spread: float
    critic_prob_max: float


class Chunker(ABC):
    """Split an utterance into ``Chunk`` objects."""

    @abstractmethod
    def chunk(self, utterance: str) -> list[Chunk]:
        """Return ordered chunks covering ``utterance``."""
        ...


class SalienceFilter(ABC):
    """Score chunks before embedding.  Must be cheap (no LM call)."""

    @abstractmethod
    def score(
        self,
        chunks: list[Chunk],
        ctx_state: dict[str, Any] | None,
    ) -> list[float]:
        """Return a salience value in [0, 1] for each chunk."""
        ...


class Embedder(ABC):
    """Embed selected chunks."""

    embedded: bool = False

    @abstractmethod
    def embed(
        self,
        chunks: list[Chunk],
        ctx_state: dict[str, Any] | None = None,
    ) -> list[np.ndarray | None]:
        """Return one hidden vector per chunk, or None for skipped chunks."""
        ...


class Aggregator(ABC):
    """Fold chunk signals into a single ``TurnDigest``."""

    @abstractmethod
    def aggregate(
        self,
        signals: list[ChunkSignal],
        ctx_state: dict[str, Any] | None,
    ) -> TurnDigest:
        """Return a digest from the surviving chunk signals."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Best-effort token count for span bookkeeping.

    Uses a conservative whitespace + punctuation split.  The span is only
    used for relative ordering and throughput accounting, not for exact
    tokenizer alignment.
    """
    return len(text.split())


def _tokenize(text: str) -> list[str]:
    """Cheap word-level tokenization used by the lexical novelty filter."""
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", text)]


def _fixed_window_tokens(
    utterance: str,
    window_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Build fixed-window chunks over word tokens.

    ``overlap_tokens`` tokens are shared between consecutive chunks so that
    boundaries are not information cliffs.
    """
    tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", utterance)]
    if not tokens:
        return []

    stride = max(1, window_tokens - overlap_tokens)
    chunks: list[Chunk] = []
    n = len(tokens)
    for start_idx in range(0, n, stride):
        end_idx = min(start_idx + window_tokens, n)
        span_start = tokens[start_idx][0]
        span_end = tokens[end_idx - 1][1]
        text = utterance[span_start:span_end]
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    span=(span_start, span_end),
                    position=len(chunks),
                )
            )
        if end_idx == n:
            break
    return chunks


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------


class FixedWindowChunker(Chunker):
    """Split text into equal-width token windows with optional overlap."""

    def __init__(
        self,
        window_tokens: int = 256,
        overlap_tokens: int = 32,
    ) -> None:
        self.window_tokens = max(1, window_tokens)
        self.overlap_tokens = max(0, overlap_tokens)

    def chunk(self, utterance: str) -> list[Chunk]:
        return _fixed_window_tokens(
            utterance,
            self.window_tokens,
            self.overlap_tokens,
        )


class SentenceChunker(Chunker):
    """Split text on sentence terminal punctuation."""

    def __init__(self, pattern: str | None = None) -> None:
        self.pattern = pattern or r"(?<=[.!?])\s+"

    def chunk(self, utterance: str) -> list[Chunk]:
        parts = [s for s in re.split(self.pattern, utterance) if s]
        chunks: list[Chunk] = []
        cursor = 0
        for part in parts:
            start = utterance.find(part, cursor)
            if start == -1:
                start = cursor
            end = start + len(part)
            cursor = end
            text = utterance[start:end]
            chunks.append(Chunk(text=text, span=(start, end), position=len(chunks)))
        return chunks


class ParagraphChunker(Chunker):
    """Split text on blank-line boundaries."""

    def chunk(self, utterance: str) -> list[Chunk]:
        parts = [p for p in re.split(r"\n\n+", utterance) if p.strip()]
        chunks: list[Chunk] = []
        cursor = 0
        for part in parts:
            stripped = part.strip("\n")
            start = utterance.find(stripped, cursor)
            if start == -1:
                start = cursor
            end = start + len(stripped)
            cursor = end + 1
            text = utterance[start:end]
            chunks.append(Chunk(text=text, span=(start, end), position=len(chunks)))
        return chunks


class RecursiveChunker(Chunker):
    """Prefer paragraphs, then sentences, then fixed windows."""

    def __init__(
        self,
        window_tokens: int = 256,
        overlap_tokens: int = 32,
    ) -> None:
        self.window_tokens = max(1, window_tokens)
        self.overlap_tokens = max(0, overlap_tokens)

    def chunk(self, utterance: str) -> list[Chunk]:
        paragraphs = ParagraphChunker().chunk(utterance)
        chunks: list[Chunk] = []
        for para in paragraphs:
            if _estimate_tokens(para.text) <= self.window_tokens:
                chunks.append(Chunk(text=para.text, span=para.span, position=len(chunks)))
                continue

            sentences = SentenceChunker().chunk(para.text)
            para_offset = para.span[0]
            for sent in sentences:
                if _estimate_tokens(sent.text) <= self.window_tokens:
                    global_span = (para_offset + sent.span[0], para_offset + sent.span[1])
                    chunks.append(
                        Chunk(text=sent.text, span=global_span, position=len(chunks))
                    )
                    continue

                windows = _fixed_window_tokens(
                    sent.text,
                    self.window_tokens,
                    self.overlap_tokens,
                )
                for win in windows:
                    global_span = (para_offset + win.span[0], para_offset + win.span[1])
                    chunks.append(
                        Chunk(text=win.text, span=global_span, position=len(chunks))
                    )
        return chunks


# ---------------------------------------------------------------------------
# Salience filters
# ---------------------------------------------------------------------------


class PassThroughFilter(SalienceFilter):
    """Keep every chunk with maximum salience."""

    def score(
        self,
        chunks: list[Chunk],
        ctx_state: dict[str, Any] | None,
    ) -> list[float]:
        return [1.0] * len(chunks)


class CorrectionMarkerFilter(SalienceFilter):
    """Boost chunks that look like explicit corrections."""

    def __init__(self, markers: list[str] | None = None) -> None:
        self.markers = [m.lower() for m in (markers or [])]

    def score(
        self,
        chunks: list[Chunk],
        ctx_state: dict[str, Any] | None,
    ) -> list[float]:
        scores: list[float] = []
        for chunk in chunks:
            lowered = chunk.text.lower()
            score = 1.0 if any(m in lowered for m in self.markers) else 0.0
            scores.append(score)
        return scores


class LexicalNoveltyFilter(SalienceFilter):
    """Reward chunks that introduce unseen tokens.

    Maintains a running set of observed tokens and scores each chunk as
    ``1 - overlap_ratio`` with that set.  Cheap and stateful.
    """

    def __init__(self) -> None:
        self._centroid: set[str] = set()

    def score(
        self,
        chunks: list[Chunk],
        ctx_state: dict[str, Any] | None,
    ) -> list[float]:
        scores: list[float] = []
        for chunk in chunks:
            tokens = set(_tokenize(chunk.text))
            denom = max(len(tokens), 1)
            overlap = len(tokens & self._centroid)
            score = 1.0 - (overlap / denom)
            scores.append(max(0.0, min(1.0, score)))
            self._centroid |= tokens
        return scores

    def reset(self) -> None:
        """Clear the running centroid (mostly useful in tests)."""
        self._centroid.clear()


class CentroidCosineFilter(SalienceFilter):
    """Placeholder for foreign-embedder cosine novelty scoring."""

    def score(
        self,
        chunks: list[Chunk],
        ctx_state: dict[str, Any] | None,
    ) -> list[float]:
        return [1.0] * len(chunks)


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------


class NoneEmbedder(Embedder):
    """Skip embedding; signals are stored as raw text only."""

    embedded = False

    def embed(
        self,
        chunks: list[Chunk],
        ctx_state: dict[str, Any] | None = None,
    ) -> list[np.ndarray | None]:
        return [None] * len(chunks)


class SameLmEmbedder(Embedder):
    """Embed chunks using the same language model that drives the agent."""

    embedded = True

    def embed(
        self,
        chunks: list[Chunk],
        ctx_state: dict[str, Any] | None = None,
    ) -> list[np.ndarray | None]:
        if ctx_state is None:
            raise ValueError("SameLmEmbedder requires ctx_state with a 'driver'")
        driver = ctx_state.get("driver")
        if driver is None:
            raise ValueError("SameLmEmbedder requires ctx_state['driver']")

        embeddings: list[np.ndarray | None] = []
        for chunk in chunks:
            hidden = driver.peek_embedding(chunk.text, last_token_only=False)
            hidden = np.asarray(hidden, dtype=np.float32)
            if hidden.ndim > 1:
                hidden = hidden.mean(axis=0)
            hidden = hidden.reshape(-1)
            embeddings.append(hidden)
        return embeddings


class IdentityEmbedder(Embedder):
    """Flag chunks as embedded without materialising a foreign vector yet."""

    embedded = True

    def embed(
        self,
        chunks: list[Chunk],
        ctx_state: dict[str, Any] | None = None,
    ) -> list[np.ndarray | None]:
        return [None] * len(chunks)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class StatisticsAggregator(Aggregator):
    """Fold surviving chunk signals into scalar turn statistics."""

    def aggregate(
        self,
        signals: list[ChunkSignal],
        ctx_state: dict[str, Any] | None,
    ) -> TurnDigest:
        n_survived = len(signals)
        drifts = [s.drift for s in signals if s.drift is not None]
        drift_max = float(np.max(drifts)) if drifts else 0.0
        drift_mean = float(np.mean(drifts)) if drifts else 0.0
        drift_p90 = float(np.percentile(drifts, 90)) if drifts else 0.0

        if n_survived:
            correction_fraction = sum(1 for s in signals if s.is_correction) / n_survived
        else:
            correction_fraction = 0.0

        saliences = [s.salience for s in signals]
        if len(saliences) >= 2:
            novelty_spread = float(np.std(saliences, ddof=0))
        else:
            novelty_spread = 0.0

        return TurnDigest(
            n_chunks=n_survived,  # overwritten by pipeline with total chunks
            n_survived=n_survived,
            drift_max=drift_max,
            drift_mean=drift_mean,
            drift_p90=drift_p90,
            correction_fraction=correction_fraction,
            novelty_spread=novelty_spread,
            critic_prob_max=0.0,
        )


class IngestionPipeline:
    """Configurable ingestion scaffold.

    ``config`` is a plain dict with stage selectors and sub-options, e.g.:

        {
            "chunker": "fixed-window",
            "chunker_window_tokens": 256,
            "chunker_overlap_tokens": 32,
            "salience": "correction-marker",
            "salience_top_k": 4,
            "embedder": "same-lm",
            "aggregator": "stats",
            "observation_mode": "parallel",
        }

    All stages default to cheap, safe implementations so the pipeline is a
    drop-in no-op when not configured.
    """

    DEFAULT_MARKERS = ["no,", "wrong", "incorrectly", "actually"]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        driver: Any | None = None,
        cortex: Any | None = None,
    ) -> None:
        self.config = config or {}
        self._driver = driver
        self._cortex = cortex
        self._chunker = self._build_chunker()
        self._salience = self._build_salience_filter()
        self._embedder = self._build_embedder()
        self._aggregator = self._build_aggregator()

    def _build_chunker(self) -> Chunker:
        name = self.config.get("chunker", "fixed-window")
        window = int(self.config.get("chunker_window_tokens", 256))
        overlap = int(self.config.get("chunker_overlap_tokens", 32))
        if name == "sentence":
            return SentenceChunker()
        if name == "paragraph":
            return ParagraphChunker()
        if name == "recursive":
            return RecursiveChunker(window_tokens=window, overlap_tokens=overlap)
        return FixedWindowChunker(window_tokens=window, overlap_tokens=overlap)

    def _build_salience_filter(self) -> SalienceFilter:
        name = self.config.get("salience", "pass-through")
        markers = self.config.get("correction_markers", self.DEFAULT_MARKERS)
        if name == "correction-marker":
            return CorrectionMarkerFilter(markers=markers)
        if name == "lexical-novelty":
            return LexicalNoveltyFilter()
        if name == "centroid-cosine":
            return CentroidCosineFilter()
        return PassThroughFilter()

    def _build_embedder(self) -> Embedder:
        name = self.config.get("embedder", "none")
        if name == "same-lm":
            return SameLmEmbedder()
        if name == "identity":
            return IdentityEmbedder()
        return NoneEmbedder()

    def _build_aggregator(self) -> Aggregator:
        name = self.config.get("aggregator", "stats")
        if name == "stats":
            return StatisticsAggregator()
        return StatisticsAggregator()

    def process(
        self,
        utterance: str,
        ctx_state: dict[str, Any] | None = None,
    ) -> tuple[list[ChunkSignal], TurnDigest]:
        """Run the ingestion pipeline over ``utterance``.

        Returns ``(chunk_signals, turn_digest)``.  The caller is responsible
        for storing chunk signals in the hippocampus and routing the digest
        to the digestive gate.
        """
        ctx_state = dict(ctx_state) if ctx_state else {}
        if self._driver is not None and "driver" not in ctx_state:
            ctx_state["driver"] = self._driver
        if self._cortex is not None and "cortex" not in ctx_state:
            ctx_state["cortex"] = self._cortex
        ctx_state.get("last_utterance")  # consumed if provided upstream
        correction_signal = float(ctx_state.get("correction_signal", 0.0))

        chunks = self._chunker.chunk(utterance)
        if not chunks:
            chunks = [Chunk(text=utterance, span=(0, len(utterance)), position=0)]

        saliences = self._salience.score(chunks, ctx_state)
        salience_kind = self.config.get("salience", "pass-through")
        default_threshold = 0.0 if salience_kind == "pass-through" else 0.5
        threshold = float(self.config.get("salience_threshold", default_threshold))
        scored = [(chunk, float(sal)) for chunk, sal in zip(chunks, saliences, strict=True) if sal >= threshold]

        top_k = self.config.get("salience_top_k")
        if top_k is not None:
            top_k = int(top_k)
            scored.sort(key=lambda item: item[1], reverse=True)
            scored = scored[:top_k]

        survived = [chunk for chunk, _ in scored]
        salience_map = {id(chunk): sal for chunk, sal in scored}

        embeddings = self._embedder.embed(survived, ctx_state)

        markers = [m.lower() for m in self.config.get("correction_markers", self.DEFAULT_MARKERS)]

        signals: list[ChunkSignal] = []
        for chunk, hidden in zip(survived, embeddings, strict=True):
            salience = salience_map[id(chunk)]
            drift = self._compute_drift(hidden, ctx_state)
            is_correction = (
                correction_signal > 0.0
                or any(m in chunk.text.lower() for m in markers)
            )
            signals.append(
                ChunkSignal(
                    text=chunk.text,
                    span=chunk.span,
                    salience=salience,
                    embedded=self._embedder.embedded,
                    hidden=hidden,
                    drift=drift,
                    is_correction=is_correction,
                )
            )

        digest = self._aggregator.aggregate(signals, ctx_state)
        # Replace the aggregator's count with the true pre-filter total.
        digest = TurnDigest(
            n_chunks=len(chunks),
            n_survived=digest.n_survived,
            drift_max=digest.drift_max,
            drift_mean=digest.drift_mean,
            drift_p90=digest.drift_p90,
            correction_fraction=digest.correction_fraction,
            novelty_spread=digest.novelty_spread,
            critic_prob_max=digest.critic_prob_max,
        )

        return signals, digest

    def _compute_drift(
        self,
        hidden: np.ndarray | None,
        ctx_state: dict[str, Any] | None,
    ) -> float | None:
        if hidden is None:
            return None
        cortex = (ctx_state or {}).get("cortex")
        if cortex is None or not hasattr(cortex, "warm_state"):
            return None

        warm_state = cortex.warm_state
        if not isinstance(warm_state, np.ndarray):
            return None

        mode = self.config.get("observation_mode", "parallel")
        if mode == "sequential":
            return self._sequential_drift(hidden, cortex, warm_state)
        return self._parallel_drift(hidden, cortex, warm_state)

    def _parallel_drift(
        self,
        hidden: np.ndarray,
        cortex: Any,
        warm_state: np.ndarray,
    ) -> float | None:
        hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if warm_state.shape == hidden.shape:
            candidate = hidden
        else:
            # Try to project the LM hidden into cortex space when dimensions differ.
            proj = getattr(cortex, "proj_hidden", None)
            if proj is None or not hasattr(proj, "shape"):
                return None
            try:
                candidate = proj @ hidden
            except Exception:
                return None
            if candidate.shape != warm_state.shape:
                return None

        denom = max(float(np.linalg.norm(warm_state)), 1.0)
        return float(np.linalg.norm(candidate - warm_state)) / denom

    def _sequential_drift(
        self,
        hidden: np.ndarray,
        cortex: Any,
        warm_state: np.ndarray,
    ) -> float | None:
        try:
            new_warm = cortex.observe(hidden, correction_signal=0.0)
        except Exception:
            return None
        new_warm = np.asarray(new_warm, dtype=np.float32)
        denom = max(float(np.linalg.norm(warm_state)), 1.0)
        return float(np.linalg.norm(new_warm - warm_state)) / denom
