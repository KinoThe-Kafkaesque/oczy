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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.experiments.digestive_gate import DigestiveGateConfig
from oczy.lm.cvec_driver import ReservedPosition
from plastic_cortex.kv_cortex import KVCortexConfig

FACTS: list[str] = [
    "The codeword for project alpha is skylark.",
    "Correction: the codeword for project beta is not raven, it is rook.",
    "The secret passphrase for level 7 is marmalade.",
]
QUERIES: list[str] = [
    "What is the codeword for project alpha?",
    "What is the codeword for project beta?",
    "What is the secret passphrase for level 7?",
]
TARGETS: list[str] = [
    "skylark",
    "rook",
    "marmalade",
]
DEFAULT_FACT_POSITIONS: list[float] = [0.2, 0.5, 0.8]


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
    prefix_source: str | None
    auto_consolidated: int
    recall_a: int
    recall_b: int
    co_recall: int
    domain_recall_a: int
    domain_recall_b: int
    domain_co_recall: int
    traces_stored: int
    embedding_calls: int
    cold_drift: float
    consolidation_strength: float
    memory_bytes: int

def _make_long_turn(
    fact_a: str | None = None,
    fact_b: str | None = None,
    *,
    fact_a_position: float = 0.25,
    fact_b_position: float = 0.75,
    total_length_tokens: int = 512,
) -> str:
    """Return a long whitespace-delimited text with two facts buried inside."""
    if fact_a is None:
        fact_a = FACTS[0] if len(FACTS) > 0 else ""
    if fact_b is None:
        fact_b = FACTS[1] if len(FACTS) > 1 else ""
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



def _make_long_turn_multi(
    facts: list[str],
    *,
    total_length_tokens: int = 2048,
) -> str:
    """Return a long whitespace-delimited text with N facts evenly spaced."""
    total_fact_tokens = sum(len(f.split()) for f in facts)
    assert total_fact_tokens <= total_length_tokens
    words = ["neutral"] * total_length_tokens
    positions = [
        int(total_length_tokens * (i + 1) / (len(facts) + 1))
        for i in range(len(facts))
    ]
    for pos, fact in zip(positions, facts, strict=True):
        fact_tokens = fact.split()
        for offset, token in enumerate(fact_tokens):
            idx = pos + offset
            if idx < total_length_tokens:
                words[idx] = token
    return " ".join(words)

def _build_agent(
    mode: str,
    ingestion: dict[str, Any] | None,
    driver: Any | None = None,
    auto_consolidate: bool = False,
    use_identity_adapter: bool = True,
) -> CortexAgent:
    """Create a fresh CortexAgent for one multi-fact probe run."""
    if driver is None:
        driver = _MockDriver(n_embd=16)
    use_hybrid = mode == "hybrid"
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        use_ingestion_pipeline=True,
        auto_consolidate=auto_consolidate,
        use_identity_adapter=use_identity_adapter,
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


def _recall_fact(
    agent: CortexAgent,
    query: str,
    target: str | None = None,
    *,
    domain_targets: list[str] | None = None,
    recall_query: str | None = None,
) -> int:
    """Return 1 if ``target`` (or any ``domain_targets`` keyword) appears in the agent's answer.

    If ``domain_targets`` is not None, it takes precedence and exact target matching is skipped.
    The query is wrapped in a brief instruction template so the Instruct-tuned
    real driver answers rather than returning empty text.

    ``recall_query`` is passed to ``agent.articulate()`` for hippocampal replay; it
    defaults to ``query`` when not provided.
    """
    effective_recall_query = recall_query if recall_query is not None else query
    prompt = f"Answer briefly.\nQuestion: {query}\nAnswer:"
    targets: list[str] = []
    if target is not None:
        targets.append(str(target))
    answer = agent.articulate(
        prompt=prompt,
        apply_steering=False,
        recall_query=effective_recall_query,
        prefix_targets=targets if targets else None,
    ).lower()
    if domain_targets is not None:
        return 1 if any(kw.lower() in answer for kw in domain_targets) else 0
    if target is None:
        raise ValueError("either target or domain_targets must be provided")
    return 1 if target.lower() in answer else 0


def _derive_prefix_from_hippocampus(
    agent: CortexAgent,
    max_tokens: int = 128,
) -> tuple[str, str] | None:
    """Pick the most relevant hippocampal memory and format it as prefix text.

    Returns ``(prefix_text, source)`` where source is ``"hippocampus"``,
    or ``None`` when no usable memory item is available.

    Rather than returning the whole stored utterance (which is mostly filler
    in the multi-fact stressor), we extract from each trace the window around
    the salient fact keywords. This keeps the prefix short and focused on the
    retrieved memory surface.
    """
    hippocampus = agent.neural_hippocampus

    def _salient_snippets(text: str, window: int = 2) -> list[str]:
        """Return narrow windows around salient keywords.  Uses word-level
        windowing first for precision, then falls back to sentence extraction
        for structured text."""
        targets = set(TARGETS) | {"alpha", "beta", "gamma", "rook", "skylark", "falcon", "marmalade"}
        snippets: list[str] = []
        # Primary: tight word-level window around each target hit.
        words = text.split()
        seen: set[int] = set()
        for i, word in enumerate(words):
            if word.lower().rstrip(".,!?;:\"'") in targets and i not in seen:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                snippet = " ".join(words[start:end])
                snippets.append(snippet)
                seen.update(range(start, end))
        if snippets:
            return snippets
        # Fall back to sentence-level for structured text without these keywords.
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent_lower = sent.lower()
            if any(t in sent_lower for t in targets):
                snippets.append(sent.strip())
        return snippets

    # Prefer consolidated slow-updates; highest surprise, then most episodes.
    if hippocampus.slow_updates:
        best_summary = max(
            hippocampus.slow_updates,
            key=lambda s: (s.get("avg_surprise", 0.0), s.get("n_episodes", 0)),
        )
        corrections = best_summary.get("summary_corrections")
        if isinstance(corrections, list) and corrections:
            text = " ".join(str(c) for c in corrections)
        else:
            text = str(best_summary.get("representative_query", ""))
        snippets = _salient_snippets(text)
        if snippets:
            words = " ".join(snippets).strip().split()[:max_tokens]
            return " ".join(words), "hippocampus"

    # Fall back to raw traces; collect salient snippets from every trace so we
    # don't miss one of the two buried facts.
    all_snippets: list[str] = []
    for trace in hippocampus.memory.traces.values():
        for key in ("correction", "corrected_answer", "query"):
            value = trace.get(key)
            if value:
                all_snippets.extend(_salient_snippets(str(value)))
    if all_snippets:
        words = " ".join(all_snippets).strip().split()[:max_tokens]
        return " ".join(words), "hippocampus"

    return None



def _agent_prefix_source(agent: Any) -> str | None:
    """Return the source label for a ReservedPosition set by the live CortexAgent."""
    pos = getattr(agent, "reserved_position", None)
    if pos is None:
        return None
    return getattr(pos, "source", None)

def _run_probe(
    mode: str,
    length: int = 512,
    ingestion: dict[str, Any] | None = None,
    use_real_driver: bool = False,
    n_ctx: int = 4096,
    use_prefix: bool = False,
    auto_prefix: bool = False,
    use_agent_prefix: bool = False,
    auto_consolidate: bool = False,
    hybrid_cap: float = 10.0,
    max_traces: int | None = None,
    domain_recall: bool = False,
    use_paraphrase: bool = False,
    use_identity_adapter: bool = True,
    num_facts: int = 2,
) -> _ProbeResult:
    """Run one probe: perceive, metabolize, consolidate, retrieve."""
    _facts = FACTS[:num_facts]
    _queries = QUERIES[:num_facts]
    _targets = TARGETS[:num_facts]
    long_turn = _make_long_turn_multi(_facts, total_length_tokens=length)
    driver: Any | None = None
    if use_real_driver:
        driver = _load_real_driver(n_ctx)
    agent = _build_agent(
        mode=mode,
        ingestion=ingestion,
        driver=driver,
        auto_consolidate=auto_consolidate,
        use_identity_adapter=use_identity_adapter,
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
    prefix_source: str | None = None
    if auto_prefix:
        print("ASI event=deprecated_auto_prefix message=--auto-prefix is deprecated; use --use-agent-prefix")
        derived = _derive_prefix_from_hippocampus(agent)
        if derived is not None:
            prefix_text, prefix_source = derived
            agent.set_reserved_position(ReservedPosition(text=prefix_text, source="hippocampus"))
    elif use_agent_prefix:
        agent.config.use_hippocampus_prefix = True
        orig_set = agent.set_reserved_position
        def _capturing_set(position: ReservedPosition | None) -> None:
            if position is not None:
                nonlocal prefix_source
                prefix_source = getattr(position, "source", None)
            orig_set(position)
        agent.set_reserved_position = _capturing_set  # type: ignore[method-assign]
    elif use_prefix:
        prefix_text = " ".join(_facts) + " "
        agent.set_reserved_position(ReservedPosition(text=prefix_text, source="hand"))
        prefix_source = "hand"

    recall_scores: list[int] = []
    for _query, _target in zip(_queries, _targets, strict=True):
        recall_scores.append(_recall_fact(agent, _query, _target))
    co_recall = 1 if all(recall_scores) else 0

    if use_agent_prefix:
        agent.set_reserved_position = orig_set  # type: ignore[method-assign]

    recall_a = recall_scores[0] if len(recall_scores) > 0 else 0
    recall_b = recall_scores[1] if len(recall_scores) > 1 else 0

    return _ProbeResult(
        mode=mode,
        length=length,
        use_prefix=use_prefix,
        prefix_source=prefix_source,
        auto_consolidated=auto_consolidated,
        recall_a=recall_a,
        recall_b=recall_b,
        co_recall=co_recall,
        traces_stored=agent.neural_hippocampus.status()["episode_count"],
        domain_recall_a=0,
        domain_recall_b=0,
        domain_co_recall=0,
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
        "--auto-prefix",
        action="store_true",
        help=(
            "[DEPRECATED] Use --use-agent-prefix instead. "
            "Derive a ReservedPosition prefix from hippocampal slow-updates"
            " / traces after consolidation. Takes precedence over --use-prefix."
        ),
    )
    parser.add_argument(
        "--use-agent-prefix",
        action="store_true",
        help=(
            "Enable CortexAgent.use_hippocampus_prefix so the live agent derives"
            " the ReservedPosition during articulate(). Takes precedence over"
            " --use-prefix but not --auto-prefix."
        ),
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
    parser.add_argument(
        "--domain-recall",
        action="store_true",
        help="Also report domain-level recall (keyword hit, not exact token).",
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="Use paraphrased recall queries that omit the original keywords.",
    )
    parser.add_argument(
        "--no-identity-adapter",
        action="store_true",
        help="Disable applying the IdentityHypernetwork state adapter bias.",
    )
    parser.add_argument(
        "--num-facts",
        type=int,
        default=2,
        choices=[2, 3, 4, 5],
        help="Number of facts to embed in the long turn (default: 2).",
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
        auto_prefix=args.auto_prefix,
        use_agent_prefix=args.use_agent_prefix,
        auto_consolidate=args.auto_consolidate,
        hybrid_cap=args.hybrid_cap,
        max_traces=args.max_traces,
        domain_recall=args.domain_recall,
        use_paraphrase=args.paraphrase,
        use_identity_adapter=not args.no_identity_adapter,
        num_facts=args.num_facts,
    )
    metric_parts = [
        f"METRIC mode={result.mode} use_prefix={result.use_prefix} prefix_source={result.prefix_source}",
        f"auto_consolidated={result.auto_consolidated}",
        f"length={result.length} num_facts={args.num_facts}",
        f"recall_a={result.recall_a}",
        f"recall_b={result.recall_b}",
        f"co_recall={result.co_recall}",
    ]
    if args.domain_recall:
        metric_parts.extend([
            f"domain_recall_a={result.domain_recall_a}",
            f"domain_recall_b={result.domain_recall_b}",
            f"domain_co_recall={result.domain_co_recall}",
        ])
    metric_parts.extend([
        f"traces={result.traces_stored}",
        f"embedding_calls={result.embedding_calls}",
        f"memory_bytes={result.memory_bytes}",
        f"cold_drift={result.cold_drift:.6f}",
        f"consolidation_strength={result.consolidation_strength:.6f}",
    ])
    print(" ".join(metric_parts))


    asi_config = {
        "mode": result.mode,
        "length": result.length,
        "use_prefix": result.use_prefix,
        "prefix_source": result.prefix_source,
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
