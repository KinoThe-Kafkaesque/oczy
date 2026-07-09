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
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.experiments.digestive_gate import DigestiveGateConfig
from plastic_cortex.kv_cortex import KVCortexConfig

NEEDLE = "The secret codeword is octarine."
QUERY = "What is the secret codeword?"

_GGUF_FILE_NAME = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
_GGUF_CACHE_PARENT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--LiquidAI--LFM2.5-1.2B-Instruct-GGUF"
)

_REAL_DRIVER: Any | None = None


def _resolve_gguf_path() -> Path | None:
    """Return the local GGUF path from env or HF cache, or None if missing."""
    env_path = os.environ.get("OCZY_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path
    if _GGUF_CACHE_PARENT.exists():
        for path in sorted(_GGUF_CACHE_PARENT.rglob(_GGUF_FILE_NAME)):
            if path.is_file():
                return path
    return None


def _gguf_available() -> bool:
    """True when a local GGUF can be found without downloading."""
    return _resolve_gguf_path() is not None


def _load_real_driver(n_ctx: int = 4096) -> Any:
    """Load (or reuse) the real LlamaCVecDriver backed by LFM2.5."""
    global _REAL_DRIVER
    if _REAL_DRIVER is not None and _REAL_DRIVER.config.n_ctx == n_ctx:
        return _REAL_DRIVER

    from llama_cpp import Llama

    from oczy.lm import CVecDriverConfig, LlamaCVecDriver

    config = CVecDriverConfig(n_ctx=n_ctx, n_threads=4, embedding=True)
    resolved = _resolve_gguf_path()
    if resolved is None:
        raise FileNotFoundError(
            f"{_GGUF_FILE_NAME} not found. Set OCZY_MODEL_PATH or cache the file "
            "under ~/.cache/huggingface/hub/models--LiquidAI--"
            "LFM2.5-1.2B-Instruct-GGUF."
        )
    llm = Llama(
        model_path=str(resolved),
        n_ctx=n_ctx,
        n_threads=4,
        embedding=True,
        verbose=False,
    )
    _REAL_DRIVER = LlamaCVecDriver(llm, config)
    return _REAL_DRIVER


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
    wall_seconds: float


def _build_agent(
    use_pipeline: bool,
    ingestion: dict[str, Any] | None,
    driver: Any | None = None,
) -> CortexAgent:
    """Create a fresh CortexAgent for one needle position."""
    if driver is None:
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
    driver: Any | None = None,
) -> _SweepResult:
    long_turn = _make_long_turn(
        needle=needle,
        needle_position=position,
        total_length_tokens=length,
    )

    start = time.perf_counter()
    agent = _build_agent(
        use_pipeline=use_pipeline, ingestion=ingestion, driver=driver
    )

    agent.perceive(long_turn)
    agent.metabolize()

    traces_stored = agent.neural_hippocampus.status()["episode_count"]
    recall = 1 if _recall_needle(agent, needle=needle, query=query) else 0
    embedding_calls = getattr(agent.driver, "embedding_calls", 0)
    wall_seconds = time.perf_counter() - start

    return _SweepResult(
        position=position,
        recall=recall,
        traces_stored=traces_stored,
        embedding_calls=embedding_calls,
        wall_seconds=wall_seconds,
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
    parser.add_argument(
        "--use-real-driver",
        action="store_true",
        help="Load the real LFM2.5 GGUF driver instead of the deterministic mock.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="Context size for the real driver (default: 4096).",
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

    driver: Any | None = None
    if args.use_real_driver:
        driver = _load_real_driver(args.n_ctx)

    for position in positions:
        result = _run_position(
            position=position,
            length=args.length,
            needle=args.needle,
            query=args.query,
            use_pipeline=use_pipeline,
            ingestion=effective_ingestion,
            driver=driver,
        )
        results.append(result)
        print(
            f"METRIC length={args.length} "
            f"position={result.position:.2f} "
            f"recall={result.recall} "
            f"traces={result.traces_stored} "
            f"embedding_calls={result.embedding_calls} "
            f"wall_seconds={result.wall_seconds:.6f}"
        )

    mean_recall = sum(r.recall for r in results) / len(results) if results else 0.0
    max_recall = max((r.recall for r in results), default=0)
    total_embedding_calls = sum(r.embedding_calls for r in results)
    total_wall_seconds = sum(r.wall_seconds for r in results)

    print(
        f"METRIC length={args.length} "
        f"mean_recall={mean_recall:.2f} "
        f"max_recall={max_recall} "
        f"embedding_calls_total={total_embedding_calls} "
        f"total_wall_seconds={total_wall_seconds:.6f}"
    )

    asi_config = {
        "use_ingestion_pipeline": use_pipeline,
        "ingestion": effective_ingestion if use_pipeline else {},
    }
    print(f"ASI config={json.dumps(asi_config)}")


if __name__ == "__main__":
    main()
