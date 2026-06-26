"""Needle-per-turn stressor for the ingestion pipeline.

The status-quo CortexAgent metabolism embeds the whole utterance once and,
when the gate fires, stores a single trace keyed by that full string.  On a
long turn the needle is buried somewhere inside that monolithic trace; replay
retrieves it only by accident.

The new IngestionPipeline chunks the utterance, embeds each chunk, and stores
per-chunk traces.  A chunk containing the buried fact is therefore addressable
on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.experiments.digestive_gate import DigestiveGateConfig
from oczy.experiments.ingestion import FixedWindowChunker
from plastic_cortex.kv_cortex import KVCortexConfig

NEEDLE = "The secret codeword is octarine."


def _make_long_turn(
    needle: str = NEEDLE,
    *,
    needle_position: float = 0.8,
    total_length_tokens: int = 500,
) -> str:
    """Return a long whitespace-delimited text with ``needle`` buried inside.

    ``needle_position`` is a token-fraction; 0.0 puts the needle at the start,
    1.0 at the very end.  The total token count is approximate because the
    needle itself contributes several tokens.
    """
    assert 0.0 <= needle_position <= 1.0
    needle_tokens = needle.split()
    filler_len = max(0, total_length_tokens - len(needle_tokens))
    filler = ["neutral"] * filler_len
    insert_at = int(filler_len * needle_position)
    words = filler[:insert_at] + needle_tokens + filler[insert_at:]
    return " ".join(words)


class _MockDriver:
    """Deterministic stand-in LM driver with an embedding-call counter.

    Hidden vectors are deterministic and distinct for different input texts,
    which is enough for the test harness to tell the needle-bearing chunk
    apart from filler.
    """

    def __init__(self, n_embd: int = 16) -> None:
        self.n_embd = n_embd
        self.n_layers = 2
        self.embedding_calls = 0

    def peek_embedding(
        self,
        text: str,
        last_token_only: bool = True,  # noqa: ARG002
    ) -> np.ndarray:
        self.embedding_calls += 1
        idx = sum(ord(c) for c in text) % self.n_embd
        h = np.zeros(self.n_embd, dtype=np.float32)
        h[idx] = 1.0
        h[(idx + 1) % self.n_embd] = float(len(text)) * 0.05
        return h

    def generate(
        self,
        prompt: str,  # noqa: ARG002
        max_tokens: int = 64,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        stop: list[str] | str | None = None,  # noqa: ARG002
    ) -> str:
        return "mock"


def _count_chunks(text: str, window_tokens: int = 64, overlap_tokens: int = 0) -> int:
    return len(FixedWindowChunker(window_tokens, overlap_tokens).chunk(text))


def _build_agent(use_pipeline: bool) -> CortexAgent:
    driver = _MockDriver(n_embd=16)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        use_ingestion_pipeline=use_pipeline,
        auto_consolidate=False,
        # Raise the hippocampal gate so the status-quo single-trace store
        # does not fire; the pipeline path stores signals independently.
        digestive_gate=DigestiveGateConfig(novelty_threshold=1.0),
    )

    if use_pipeline:
        cfg.ingestion = {
            "chunker": "fixed-window",
            "chunker_window_tokens": 64,
            "chunker_overlap_tokens": 0,
            "salience": "pass-through",
            "embedder": "same-lm",
            "aggregator": "stats",
        }

    agent = CortexAgent(cfg, driver=driver)
    # Accept every per-chunk write; the digestive gate is what we are gating
    # for the baseline, not the memory's own surprise write.
    agent.neural_hippocampus.memory.surprise_threshold = 0.0
    agent.boot()
    return agent


def _recall_needle(agent: CortexAgent) -> bool:
    replays = agent.neural_hippocampus.reinforce(
        query="What is the secret codeword?", k=10
    )
    for ep in replays:
        query = ep.get("query", "")
        if NEEDLE.lower() in query.lower() or "octarine" in query.lower():
            return True
    return False


@pytest.mark.slow
class TestIngestionNeedle:
    """Needle-recall stressor comparing single-embed vs. chunked ingestion."""

    def test_baseline_single_embed_misses_deep_needle(self) -> None:
        agent = _build_agent(use_pipeline=False)
        long_turn = _make_long_turn(
            needle_position=0.8, total_length_tokens=512
        )

        agent.perceive(long_turn)
        agent.metabolize()

        traces_stored = agent.neural_hippocampus.status()["episode_count"]
        recall = 1 if _recall_needle(agent) else 0

        assert traces_stored <= 1, "baseline should store at most one trace"
        assert recall == 0, "baseline must miss the buried needle"

    def test_pipeline_chunking_retrieves_deep_needle(self) -> None:
        agent = _build_agent(use_pipeline=True)
        long_turn = _make_long_turn(
            needle_position=0.8, total_length_tokens=512
        )

        agent.perceive(long_turn)
        agent.metabolize()

        traces_stored = agent.neural_hippocampus.status()["episode_count"]
        recall = 1 if _recall_needle(agent) else 0

        assert traces_stored > 1, "pipeline should store multiple chunk traces"
        assert recall == 1, "pipeline must retrieve the buried needle"

    def test_pipeline_embedding_calls_scale_with_chunks(self) -> None:
        agent = _build_agent(use_pipeline=True)
        long_turn = _make_long_turn(
            needle_position=0.8, total_length_tokens=512
        )
        expected_chunks = _count_chunks(long_turn, window_tokens=64)

        agent.perceive(long_turn)
        agent.metabolize()

        calls = agent.driver.embedding_calls
        # perceive() makes one embedding call; each surviving chunk makes one
        # additional call, so the total is expected_chunks or expected_chunks+1.
        assert expected_chunks - 1 <= calls <= expected_chunks + 1, (
            f"embedding calls ({calls}) should match fixed-window chunk count "
            f"({expected_chunks})"
        )
