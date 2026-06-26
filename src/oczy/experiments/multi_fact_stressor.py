"""Multi-fact turn stressor.

Buries a novel fact and a correction-style fact in neutral filler, processes
the long turn through a CortexAgent using the chunked ingestion pipeline,
forces consolidation, and measures independent and co-recall.

Output lines are prefixed with ``METRIC`` so the autoresearch harness can
parse them.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.experiments.digestive_gate import DigestiveGateConfig
from plastic_cortex.kv_cortex import KVCortexConfig

FACT_A = "The codeword for project alpha is skylark."
FACT_B = "Correction: the codeword for project beta is not raven, it is rook."
QUERY_A = "What is the codeword for project alpha?"
QUERY_B = "What is the codeword for project beta?"
TARGET_A = "skylark"
TARGET_B = "rook"

DEFAULT_FACT_A_POSITION = 0.25
DEFAULT_FACT_B_POSITION = 0.75


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

    def set_reserved_position(self, reserved: Any) -> Any:  # noqa: ARG002
        """No-op for mock driver; present for API parity."""
        return None

    def clear_reserved_position(self) -> None:
        """No-op for mock driver; present for API parity."""
        return None


@dataclass(frozen=True)
class _ProbeResult:
    mode: str
    length: int
    use_prefix: bool
    auto_consolidated: int
    recall_a: int
    recall_b: int
    co_recall: int
    traces_stored: int
    embedding_calls: int
    cold_drift: float
    consolidation_strength: float
    memory_bytes: int

def _make_long_turn(
    fact_a: str = FACT_A,
    fact_b: str = FACT_B,
    *,
    fact_a_position: float = DEFAULT_FACT_A_POSITION,
    fact_b_position: float = DEFAULT_FACT_B_POSITION,
    total_length_tokens: int = 512,
) -> str:
    """Return a long whitespace-delimited text with two facts buried inside."""
    assert 0.0 <= fact_a_position <= 1.0
    assert 0.0 <= fact_b_position <= 1.0
    tokens_a = fact_a.split()
    tokens_b = fact_b.split()
    assert len(tokens_a) + len(tokens_b) <= total_length_tokens

    words = ["neutral"] * total_length_tokens
    idx_a = int(total_length_tokens * fact_a_position)
    idx_b = int(total_length_tokens * fact_b_position)

    # Keep facts from overlapping; if they would, nudge the later one.
    if idx_a <= idx_b < idx_a + len(tokens_a):
        idx_b = idx_a + len(tokens_a)
    if idx_b <= idx_a < idx_b + len(tokens_b):
        idx_a = idx_b + len(tokens_b)

    assert idx_a + len(tokens_a) <= total_length_tokens
    assert idx_b + len(tokens_b) <= total_length_tokens

    for i, tok in enumerate(tokens_a):
        words[idx_a + i] = tok
    for i, tok in enumerate(tokens_b):
        words[idx_b + i] = tok
    return " ".join(words)


def _build_agent(
    mode: str,
    ingestion: dict[str, Any] | None,
    driver: Any | None = None,
    auto_consolidate: bool = False,
) -> CortexAgent:
    """Create a fresh CortexAgent for one multi-fact probe run."""
    if driver is None:
        driver = _MockDriver(n_embd=16)
    use_hybrid = mode == "hybrid"
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        use_ingestion_pipeline=True,
        auto_consolidate=auto_consolidate,
        digestive_gate=DigestiveGateConfig(
            novelty_threshold=1.0,
            use_ingestion_pipeline=False,
            use_hybrid_consolidation=use_hybrid,
            consolidation_pressure_threshold=(
                0.05 if auto_consolidate else 0.25
            ),
        ),
    )

    effective_ingestion: dict[str, Any] = {
        "chunker": "fixed-window",
        "chunker_window_tokens": 64,
        "chunker_overlap_tokens": 8,
        "salience": "lexical-novelty",
        "embedder": "same-lm",
        "aggregator": "stats",
    }
    if ingestion is not None:
        effective_ingestion.update(ingestion)
    cfg.ingestion = effective_ingestion

    agent = CortexAgent(cfg, driver=driver)
    agent.neural_hippocampus.memory.surprise_threshold = 0.0
    agent.boot()
    return agent


def _recall_fact(agent: CortexAgent, query: str, target: str) -> int:
    """Return 1 if ``target`` appears in the agent's answer to ``query``.

    The query is wrapped in a brief instruction template so the Instruct-tuned
    real driver answers rather than returning empty text.
    """
    prompt = f"Answer briefly.\nQuestion: {query}\nAnswer:"
    answer = agent.articulate(prompt=prompt, apply_steering=False).lower()
    return 1 if target.lower() in answer else 0


def _run_probe(
    mode: str,
    length: int = 512,
    ingestion: dict[str, Any] | None = None,
    use_real_driver: bool = False,
    n_ctx: int = 4096,
    use_prefix: bool = False,
    auto_consolidate: bool = False,
    hybrid_cap: float = 10.0,
    max_traces: int | None = None,
) -> _ProbeResult:
    """Run one probe: perceive, metabolize, consolidate, retrieve."""
    long_turn = _make_long_turn(total_length_tokens=length)
    if use_real_driver:
        driver = _load_real_driver(n_ctx)
    else:
        driver = _MockDriver(n_embd=16)
    agent = _build_agent(
        mode=mode,
        ingestion=ingestion,
        driver=driver,
        auto_consolidate=auto_consolidate,
    )

    agent.perceive(long_turn)
    agent.metabolize()

    auto_consolidated = 0
    summary: dict[str, Any] = {}
    strength = 1.0
    if auto_consolidate and agent.should_consolidate():
        pressure = agent.digestive_gate._pressure
        gate_cfg = agent.digestive_gate.config
        threshold = gate_cfg.consolidation_pressure_threshold
        strength = 1.0 + (pressure / threshold) * 9.0 if threshold > 0 else 1.0
        if mode == "hybrid" and agent._last_digest is not None:
            raw = strength * (1.0 + agent._last_digest.drift_max)
            strength = float(raw if hybrid_cap <= 0.0 else np.clip(raw, 1.0, hybrid_cap))
        summary = agent.consolidate(strength=strength)
        auto_consolidated = 1
        agent.digestive_gate.reset()
    else:
        # In hybrid mode, mirror the strength boost CortexAgent.turn() applies
        # when auto-consolidation fires, so the comparison is between two real
        # consolidation regimes rather than just a flag.
        digest = agent._last_digest
        if mode == "hybrid" and digest is not None:
            raw = 1.0 * (1.0 + digest.drift_max)
            strength = float(raw if hybrid_cap <= 0.0 else np.clip(raw, 1.0, hybrid_cap))
        summary = agent.consolidate(strength=strength)
    if max_traces is not None and max_traces > 0:
        memory = agent.neural_hippocampus.memory
        while len(memory.traces) > max_traces:
            memory.traces.pop(next(iter(memory.traces)), None)
    memory_bytes = len(pickle.dumps(agent.neural_hippocampus))

    if use_prefix:
        from oczy.lm.cvec_driver import ReservedPosition

        prefix_text = f"{FACT_A} {FACT_B} "
        agent.set_reserved_position(
            ReservedPosition(text=prefix_text, source="multi_fact_stressor")
        )

    recall_a = _recall_fact(agent, QUERY_A, TARGET_A)
    recall_b = _recall_fact(agent, QUERY_B, TARGET_B)
    co_recall = 1 if (recall_a and recall_b) else 0

    return _ProbeResult(
        mode=mode,
        length=length,
        use_prefix=use_prefix,
        auto_consolidated=auto_consolidated,
        recall_a=recall_a,
        recall_b=recall_b,
        co_recall=co_recall,
        traces_stored=agent.neural_hippocampus.status()["episode_count"],
        embedding_calls=getattr(agent.driver, "embedding_calls", 0),
        cold_drift=float(summary.get("cold_drift", 0.0)),
        consolidation_strength=strength,
        memory_bytes=memory_bytes,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multi-fact turn stressor for CortexAgent consolidation."
    )
    parser.add_argument(
        "--length",
        type=int,
        default=512,
        help="Total turn length in whitespace tokens (default: 512).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="scalar",
        choices=["scalar", "hybrid"],
        help="Consolidation mode: scalar (S) or hybrid (H).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="{}",
        help='JSON config object for pipeline overrides, e.g. {"ingestion":{"chunker_window_tokens":32}}.',
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
    parser.add_argument(
        "--use-prefix",
        action="store_true",
        help="Set a ReservedPosition prefix containing both facts before retrieval.",
    )
    parser.add_argument(
        "--auto-consolidate",
        action="store_true",
        help="Let the DigestiveGate decide whether consolidation fires.",
    )
    parser.add_argument(
        "--hybrid-cap",
        type=float,
        default=10.0,
        help="Cap hybrid consolidation strength (default: 10.0; 0 means uncapped).",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Prune hippocampus to N most recent traces after consolidation (optional).",
    )
    args = parser.parse_args(argv)

    config = json.loads(args.config) if args.config else {}
    ingestion = config.get("ingestion")
    if ingestion is not None and not isinstance(ingestion, dict):
        raise ValueError("config 'ingestion' must be a JSON object")
    result = _run_probe(
        mode=args.mode,
        length=args.length,
        ingestion=ingestion,
        use_real_driver=args.use_real_driver,
        n_ctx=args.n_ctx,
        use_prefix=args.use_prefix,
        auto_consolidate=args.auto_consolidate,
        hybrid_cap=args.hybrid_cap,
        max_traces=args.max_traces,
    )

    print(
        f"METRIC mode={result.mode} use_prefix={result.use_prefix} "
        f"auto_consolidated={result.auto_consolidated} "
        f"length={result.length} "
        f"recall_a={result.recall_a} "
        f"recall_b={result.recall_b} "
        f"co_recall={result.co_recall} "
        f"traces={result.traces_stored} "
        f"embedding_calls={result.embedding_calls} "
        f"memory_bytes={result.memory_bytes} "
        f"cold_drift={result.cold_drift:.6f} "
        f"consolidation_strength={result.consolidation_strength:.6f}"
    )


    asi_config = {
        "mode": result.mode,
        "length": result.length,
        "use_prefix": result.use_prefix,
        "auto_consolidated": bool(result.auto_consolidated),
        "ingestion": ingestion if ingestion is not None else {},
    }
    print(
        f"ASI mode={result.mode} "
        f"auto_consolidated={result.auto_consolidated} "
        f"co_recall={result.co_recall} "
        f"traces={result.traces_stored} "
        f"config={json.dumps(asi_config)}"
    )


if __name__ == "__main__":
    main()
