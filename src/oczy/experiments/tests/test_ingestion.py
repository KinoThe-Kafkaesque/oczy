"""Unit tests for the concrete ingestion pipeline stages."""

from __future__ import annotations

import numpy as np
import pytest

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.experiments.digestive_gate import DigestiveGate
from oczy.experiments.ingestion import (
    Chunk,
    ChunkSignal,
    IngestionPipeline,
    LexicalNoveltyFilter,
    ParagraphChunker,
    TurnDigest,
)
from plastic_cortex.kv_cortex import KVCortexConfig


class FakeDriver:
    def __init__(self, n_embd: int = 8) -> None:
        self.n_embd = n_embd

    def peek_embedding(self, text: str, last_token_only: bool = False) -> np.ndarray:
        # Deterministic, text-sensitive embedding for drift tests.
        base = np.arange(self.n_embd, dtype=np.float32) + float(len(text))
        if not last_token_only:
            return base
        return base[-1:]


class FakeCortex:
    def __init__(self, n_embd: int = 8) -> None:
        self.n_embd = n_embd
        self.warm_state = np.zeros(n_embd, dtype=np.float32)

    def observe(
        self,
        hidden: np.ndarray,
        correction_signal: float = 0.0,
    ) -> np.ndarray:
        self.warm_state = self.warm_state * 0.5 + np.asarray(hidden, dtype=np.float32) * 0.1
        return self.warm_state.copy()


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------


def test_fixed_window_chunker_splits_and_overlaps() -> None:
    pipeline = IngestionPipeline(
        {
            "chunker": "fixed-window",
            "chunker_window_tokens": 3,
            "chunker_overlap_tokens": 1,
            "salience": "pass-through",
            "embedder": "none",
        }
    )
    text = "one two three four five six seven"
    signals, digest = pipeline.process(text)

    chunks = [s.text for s in signals]
    assert chunks == ["one two three", "three four five", "five six seven"]
    assert digest.n_chunks == 3
    assert digest.n_survived == 3
    # Spans must reconstruct the source ordering without gaps (overlap allowed).
    spans = [s.span for s in signals]
    assert spans[0] == (0, 13)
    assert spans[-1][1] == len(text)


def test_sentence_chunker_splits_on_boundaries() -> None:
    pipeline = IngestionPipeline(
        {
            "chunker": "sentence",
            "salience": "pass-through",
            "embedder": "none",
        }
    )
    text = "Hello world. How are you? I am fine."
    signals, digest = pipeline.process(text)

    chunks = [s.text for s in signals]
    assert len(chunks) == 3
    assert chunks[0] == "Hello world."
    assert chunks[1] == "How are you?"
    assert chunks[2] == "I am fine."
    assert digest.n_chunks == 3


def test_paragraph_chunker_splits_on_blank_lines() -> None:
    text = "First para.\n\nSecond para\nhas two lines.\n\nThird."
    chunks = ParagraphChunker().chunk(text)
    assert len(chunks) == 3
    assert "First para." in chunks[0].text
    assert "Second para" in chunks[1].text

def test_recursive_chunker_fallbacks() -> None:
    pipeline = IngestionPipeline(
        {
            "chunker": "recursive",
            "chunker_window_tokens": 4,
            "salience": "pass-through",
            "embedder": "none",
        }
    )
    # One long paragraph that exceeds the window; should fall back to sentences/windows.
    text = "First sentence here. Second sentence follows. " * 10
    signals, digest = pipeline.process(text)
    assert digest.n_chunks > 0
    # All character spans must be in ascending order.
    assert [s.span[0] for s in signals] == sorted(s.span[0] for s in signals)


# ---------------------------------------------------------------------------
# Salience filters
# ---------------------------------------------------------------------------


def test_correction_marker_filter_marks_only_correction_chunks() -> None:
    pipeline = IngestionPipeline(
        {
            "chunker": "sentence",
            "salience": "correction-marker",
            "embedder": "none",
            "correction_markers": ["no,", "wrong", "incorrectly", "actually"],
        }
    )
    text = "No, this is wrong. It is fine."
    signals, digest = pipeline.process(text)

    assert digest.n_survived == 1  # default threshold drops non-correction chunk
    assert signals[0].salience == pytest.approx(1.0)
    assert signals[0].is_correction is True
    assert digest.correction_fraction == pytest.approx(1.0)




def test_lexical_novelty_filter_higher_score_for_novel_tokens() -> None:
    filt = LexicalNoveltyFilter()
    chunks = [
        Chunk("alpha beta gamma", (0, 16), 0),
        Chunk("alpha beta delta", (17, 33), 1),
    ]
    scores = filt.score(chunks, None)

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < scores[0]
    assert scores[1] == pytest.approx(1.0 - 2 / 3)


def test_lexical_novelty_persists_across_calls() -> None:
    filt = LexicalNoveltyFilter()
    first = filt.score([Chunk("alpha beta", (0, 10), 0)], None)
    second = filt.score([Chunk("alpha gamma", (0, 11), 0)], None)
    assert first[0] == pytest.approx(1.0)
    assert second[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Pipeline shape and embedders
# ---------------------------------------------------------------------------


def test_pipeline_returns_signals_and_digest_shape() -> None:
    pipeline = IngestionPipeline({"chunker": "sentence"})
    text = "Hello world. Goodbye world."
    signals, digest = pipeline.process(text)

    assert isinstance(signals, list)
    assert all(isinstance(s, ChunkSignal) for s in signals)
    assert isinstance(digest, TurnDigest)
    assert digest.n_chunks == 2


def test_same_lm_embedder_returns_1d_hidden() -> None:
    driver = FakeDriver(n_embd=8)
    pipeline = IngestionPipeline(
        {
            "chunker": "sentence",
            "embedder": "same-lm",
            "observation_mode": "parallel",
        }
    )
    text = "Hello world. Goodbye world."
    signals, digest = pipeline.process(text, ctx_state={"driver": driver})

    assert digest.n_chunks == 2
    assert digest.n_survived == 2
    for s in signals:
        assert s.embedded is True
        assert isinstance(s.hidden, np.ndarray)
        assert s.hidden.ndim == 1
        assert s.hidden.shape == (driver.n_embd,)


def test_same_lm_embedder_mean_pools_2d_output() -> None:
    class TwoDimDriver:
        def peek_embedding(self, text: str, last_token_only: bool = False) -> np.ndarray:
            return np.ones((3, 4), dtype=np.float32) * 2.0

    pipeline = IngestionPipeline(
        {"chunker": "fixed-window", "chunker_window_tokens": 10, "embedder": "same-lm"}
    )
    signals, _ = pipeline.process("hello world", ctx_state={"driver": TwoDimDriver()})
    assert len(signals) == 1
    np.testing.assert_array_equal(signals[0].hidden, np.ones(4, dtype=np.float32) * 2.0)


# ---------------------------------------------------------------------------
# Observation / drift
# ---------------------------------------------------------------------------


def test_parallel_observation_computes_nonzero_drift() -> None:
    driver = FakeDriver(n_embd=8)
    cortex = FakeCortex(n_embd=8)
    pipeline = IngestionPipeline(
        {
            "chunker": "fixed-window",
            "chunker_window_tokens": 2,
            "embedder": "same-lm",
            "observation_mode": "parallel",
        }
    )
    text = "The quick brown fox"
    signals, digest = pipeline.process(
        text,
        ctx_state={"driver": driver, "cortex": cortex},
    )

    assert digest.n_survived > 0
    assert all(s.drift is not None and s.drift >= 0.0 for s in signals)
    assert any(s.drift > 0.0 for s in signals)


def test_sequential_observation_updates_warm_state() -> None:
    driver = FakeDriver(n_embd=8)
    cortex = FakeCortex(n_embd=8)
    warm_before = cortex.warm_state.copy()
    pipeline = IngestionPipeline(
        {
            "chunker": "fixed-window",
            "chunker_window_tokens": 2,
            "embedder": "same-lm",
            "observation_mode": "sequential",
        }
    )
    signals, _ = pipeline.process(
        "The quick brown fox",
        ctx_state={"driver": driver, "cortex": cortex},
    )

    assert not np.array_equal(cortex.warm_state, warm_before)
    assert all(s.drift is not None for s in signals)


# ---------------------------------------------------------------------------
# Pruning and aggregation
# ---------------------------------------------------------------------------


def test_top_k_salience_pruning_reduces_n_survived() -> None:
    pipeline = IngestionPipeline(
        {
            "chunker": "sentence",
            "salience": "lexical-novelty",
            "salience_top_k": 1,
            "embedder": "none",
        }
    )
    text = "Hello world. Hello moon. Hello stars."
    signals, digest = pipeline.process(text)

    assert digest.n_chunks == 3
    assert digest.n_survived == 1
    assert signals[0].text == "Hello world."


def test_statistics_aggregator_empty_signals() -> None:
    pipeline = IngestionPipeline(
        {
            "chunker": "sentence",
            "salience": "correction-marker",
            "salience_threshold": 1.1,
        }
    )
    _, digest = pipeline.process("No, this is wrong")

    assert digest.n_chunks == 1  # one sentence chunk produced
    assert digest.n_survived == 0
    assert digest.drift_max == 0.0
    assert digest.drift_mean == 0.0
    assert digest.drift_p90 == 0.0
    assert digest.correction_fraction == 0.0
    assert digest.novelty_spread == 0.0


# ---------------------------------------------------------------------------
# Wiring convenience
# ---------------------------------------------------------------------------


def test_driver_cortex_kwargs_promoted_to_ctx_state() -> None:
    driver = FakeDriver(n_embd=8)
    cortex = FakeCortex(n_embd=8)
    pipeline = IngestionPipeline(
        {
            "chunker": "fixed-window",
            "chunker_window_tokens": 2,
            "embedder": "same-lm",
            "observation_mode": "parallel",
        },
        driver=driver,
        cortex=cortex,
    )
    signals, digest = pipeline.process("The quick brown fox")
    assert digest.n_survived > 0
    assert all(s.embedded and s.hidden is not None for s in signals)
    assert any(s.drift is not None and s.drift > 0.0 for s in signals)


def test_correction_signal_marks_all_chunks() -> None:
    pipeline = IngestionPipeline(
        {
            "chunker": "sentence",
            "salience": "pass-through",
            "embedder": "none",
        }
    )
    signals, digest = pipeline.process(
        "It is warm today. It is sunny.",
        ctx_state={"correction_signal": 1.0},
    )
    assert digest.n_survived == 2
    assert all(s.is_correction for s in signals)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Gate / agent wiring integration
# ---------------------------------------------------------------------------




class _FakeDriver:
    """Minimal driver stand-in for CortexAgent wiring smoke tests."""

    def __init__(self, n_embd: int = 8, n_layers: int = 4) -> None:
        self.n_embd = n_embd
        self.n_layers = n_layers

    def peek_embedding(self, text: str, last_token_only: bool = False) -> np.ndarray:
        base = np.arange(self.n_embd, dtype=np.float32) + float(len(text))
        return base[-1:] if last_token_only else base


def test_ingest_digest_maps_turn_digest_to_scores() -> None:
    gate = DigestiveGate()
    digest = TurnDigest(
        n_chunks=3,
        n_survived=2,
        drift_max=0.4,
        drift_mean=0.2,
        drift_p90=0.3,
        correction_fraction=0.6,
        novelty_spread=0.1,
        critic_prob_max=0.5,
    )
    scores = gate.ingest_digest(digest)
    expected_keys = {
        "critic_weight",
        "hippocampus_weight",
        "identity_weight",
        "immune_weight",
        "autoencoder_weight",
        "consolidation_pressure",
        "critic_correction_prob",
    }
    assert set(scores.keys()) == expected_keys
    assert all(isinstance(v, float) or v is None for v in scores.values())
    assert 0.0 <= scores["autoencoder_weight"] <= 1.0


def test_cortex_agent_with_pipeline_stores_chunk_signals() -> None:
    driver = _FakeDriver(n_embd=8, n_layers=4)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=8),
        use_ingestion_pipeline=True,
        ingestion={"embedder": "same-lm"},
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()
    agent.perceive("No, 'profile' means business vertical.", correction_signal=1.0)
    before = agent.neural_hippocampus.status()["episode_count"]
    meta = agent.metabolize()
    after = agent.neural_hippocampus.status()["episode_count"]
    assert after > before, "pipeline should store at least one hippocampal chunk"
    assert meta["digestive_scores"]["autoencoder_weight"] > 0.0


def test_cortex_agent_without_pipeline_unchanged() -> None:
    driver = _FakeDriver(n_embd=8, n_layers=4)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=8),
        use_ingestion_pipeline=False,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()
    agent.perceive("No, 'profile' means business vertical.", correction_signal=1.0)
    before = agent.neural_hippocampus.status()["episode_count"]
    agent.metabolize()
    after = agent.neural_hippocampus.status()["episode_count"]
    assert after > before, "legacy path should still write one hippocampus episode on high drift"
