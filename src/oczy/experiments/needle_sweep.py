"""Needle-recall sweep benchmark.

Sweeps needle position (as a fraction of turn length) and reports recall,
trace count, and embedding-call count for a CortexAgent using either the
status-quo metabolism or the optional chunked ingestion pipeline.

Output lines are prefixed with ``METRIC`` so the autoresearch harness can
parse them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.experiments.digestive_gate import DigestiveGateConfig
from plastic_cortex.kv_cortex import KVCortexConfig

NEEDLE = "The secret codeword is octarine."
QUERY = "What is the secret codeword?"


def _make_long_turn(
    needle: str = NEEDLE,
    *,
    needle_position: float = 0.5,
    total_length_tokens: int = 512,
) -> str:
    """Return a long whitespace-delimited text with ``needle`` buried inside."""
    assert 0.0 <= needle_position <= 1.0
    needle_tokens = needle.split()
    filler_len = max(0, total_length_tokens - len(needle_tokens))
    filler = ["neutral"] * filler_len
    insert_at = int(filler_len * needle_position)
    words = filler[:insert_at] + needle_tokens + filler[insert_at:]
    return " ".join(words)


class _MockDriver:
    """Deterministic stand-in LM driver with an embedding-call counter."""

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


@dataclass(frozen=True)
class _SweepResult:
    position: float
    recall: int
    traces_stored: int
    embedding_calls: int


def _build_agent(
    use_pipeline: bool,
    ingestion: dict[str, Any] | None,
) -> CortexAgent:
    """Create a fresh CortexAgent for one needle position."""
    driver = _MockDriver(n_embd=16)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        use_ingestion_pipeline=use_pipeline,
        auto_consolidate=False,
        digestive_gate=DigestiveGateConfig(
            novelty_threshold=1.0,
            use_ingestion_pipeline=False,
        ),
    )

    if use_pipeline and ingestion is not None:
        cfg.ingestion = dict(ingestion)

    agent = CortexAgent(cfg, driver=driver)
    agent.neural_hippocampus.memory.surprise_threshold = 0.0
    agent.boot()
    return agent


def _recall_needle(agent: CortexAgent, needle: str, query: str) -> bool:
    replays = agent.neural_hippocampus.reinforce(query=query, k=10)
    needle_lower = needle.lower()
    for ep in replays:
        query_text = ep.get("query", "")
        if needle_lower in query_text.lower():
            return True
    return False


def _run_position(
    position: float,
    length: int,
    needle: str,
    query: str,
    use_pipeline: bool,
    ingestion: dict[str, Any] | None,
) -> _SweepResult:
    long_turn = _make_long_turn(
        needle=needle,
        needle_position=position,
        total_length_tokens=length,
    )
    agent = _build_agent(use_pipeline=use_pipeline, ingestion=ingestion)

    agent.perceive(long_turn)
    agent.metabolize()

    traces_stored = agent.neural_hippocampus.status()["episode_count"]
    recall = 1 if _recall_needle(agent, needle=needle, query=query) else 0
    embedding_calls = agent.driver.embedding_calls

    return _SweepResult(
        position=position,
        recall=recall,
        traces_stored=traces_stored,
        embedding_calls=embedding_calls,
    )


def _parse_positions(raw: str) -> list[float]:
    positions = [float(p.strip()) for p in raw.split(",") if p.strip()]
    for p in positions:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"position {p} must be in [0.0, 1.0]")
    return positions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Needle-recall position sweep.")
    parser.add_argument(
        "--length",
        type=int,
        default=512,
        help="Total length in whitespace tokens (default: 512).",
    )
    parser.add_argument(
        "--positions",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated needle positions (default: 0.0,0.25,0.5,0.75,1.0).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="{}",
        help='JSON config object, e.g. {"use_ingestion_pipeline": true, "ingestion": {...}}',
    )
    parser.add_argument(
        "--needle",
        type=str,
        default=NEEDLE,
        help=f"Needle text (default: {NEEDLE!r}).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=QUERY,
        help=f"Replay query text (default: {QUERY!r}).",
    )
    args = parser.parse_args(argv)

    config = json.loads(args.config) if args.config else {}
    use_pipeline = bool(config.get("use_ingestion_pipeline", False))
    ingestion = config.get("ingestion")
    if ingestion is not None and not isinstance(ingestion, dict):
        raise ValueError("config 'ingestion' must be a JSON object")

    effective_ingestion: dict[str, Any] | None = None
    if use_pipeline:
        effective_ingestion = dict(ingestion) if ingestion is not None else {}
        # Default overlap keeps the needle from being split across chunks when
        # the caller does not specify an overlap value.
        effective_ingestion.setdefault("chunker_overlap_tokens", 8)

    positions = _parse_positions(args.positions)
    results: list[_SweepResult] = []

    for position in positions:
        result = _run_position(
            position=position,
            length=args.length,
            needle=args.needle,
            query=args.query,
            use_pipeline=use_pipeline,
            ingestion=effective_ingestion,
        )
        results.append(result)
        print(
            f"METRIC length={args.length} "
            f"position={result.position:.2f} "
            f"recall={result.recall} "
            f"traces={result.traces_stored} "
            f"embedding_calls={result.embedding_calls}"
        )

    mean_recall = sum(r.recall for r in results) / len(results) if results else 0.0
    max_recall = max((r.recall for r in results), default=0)
    total_embedding_calls = sum(r.embedding_calls for r in results)

    print(
        f"METRIC length={args.length} "
        f"mean_recall={mean_recall:.2f} "
        f"max_recall={max_recall} "
        f"embedding_calls_total={total_embedding_calls}"
    )

    asi_config = {
        "use_ingestion_pipeline": use_pipeline,
        "ingestion": effective_ingestion if use_pipeline else {},
    }
    print(f"ASI config={json.dumps(asi_config)}")


if __name__ == "__main__":
    main()
