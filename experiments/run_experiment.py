"""Reproducible runner for the Oczy curriculum evaluation.

Usage:
    uv run python experiments/run_experiment.py
    uv run python experiments/run_experiment.py --agent NullAgent
    uv run python experiments/run_experiment.py --agent OrganismAgent
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Make the runner importable from repo root even when executed directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.baselines import (
    ContextOnlyAgent,
    FastOnlyAgent,
    HippocampusOnlyAgent,
    IdentityOnlyAgent,
    ZeroMemoryAgent,
)
from experiments.curriculum import build_curriculum
from experiments.eval_suite import EvalSuite, EvalResult, NullAgent
from experiments.logger import ExperimentLogger
from experiments.organism import OrganismAgent


@dataclass(frozen=True)
class RunConfig:
    """Seed and protocol flags for a reproducible evaluation run."""

    seed: int = 0
    consolidate: bool = True
    sense_match: bool = True
    num_repetitions: int = 1


def _agent_bytes(agent: Any) -> int:
    """Best-effort byte count for an agent instance."""
    if hasattr(agent, "memory_bytes"):
        return int(agent.memory_bytes())
    return 0


def evaluate_agent(agent: Any, name: str, config: RunConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a single agent on the full curriculum.

    Returns ``(scorecard, artifacts)`` where ``scorecard`` is the final metrics
    card and ``artifacts`` is the complete JSON-serializable result record.
    """
    curriculum = build_curriculum(seed=config.seed)
    suite = EvalSuite(curriculum, sense_match=config.sense_match)

    pre = suite.pre_test(agent)

    level_results: list[dict[str, Any]] = []
    for level in curriculum.levels():
        level_results.append(suite.run_level(agent, level))

    raw_trace_size = _agent_bytes(agent)
    post = suite.post_test(agent)

    if config.consolidate:
        consolidation = suite.consolidation_test(agent)
        consolidated_size = _agent_bytes(agent)
    else:
        consolidation = post
        consolidated_size = raw_trace_size

    result: EvalResult = suite.score(
        pre,
        post,
        consolidation,
        level_results=level_results,
        raw_trace_size=raw_trace_size,
        consolidated_size=consolidated_size,
    )

    return result.final_card, result.scorecard_json()


def _findings(name: str, scorecard: dict[str, Any], artifacts: dict[str, Any]) -> str:
    pre = artifacts.get("pre_test_scores", {})
    post = artifacts.get("post_test_scores", {})
    return (
        f"## {name}\n\n"
        f"- Sense matching: {'enabled' if artifacts.get('sense_match') else 'disabled'}\n"
        f"- Correction uptake latency: {scorecard.get('correction_uptake_latency')}\n"
        f"- Transfer score: {scorecard.get('transfer_score')}\n"
        f"- Scope control score: {scorecard.get('scope_score')}\n"
        f"- Forgetting score: {scorecard.get('forgetting_score')}\n"
        f"- Consolidation score: {scorecard.get('consolidation_score')}\n"
        f"- Identity drift score: {scorecard.get('identity_drift_score')}\n"
        f"- Memory bytes / behavior delta: {scorecard.get('memory_bytes_per_behavior_delta')}\n"
        f"- Pre-test transfer / scope / forgetting / identity: "
        f"{pre.get('transfer')} / {pre.get('scope')} / {pre.get('forgetting')} / {pre.get('identity')}\n"
        f"- Post-test transfer / scope / forgetting / identity: "
        f"{post.get('transfer')} / {post.get('scope')} / {post.get('forgetting')} / {post.get('identity')}\n"
    )


def _print_table(rows: list[tuple[str, dict[str, Any]]]) -> None:
    """Print an ASCII comparison table to stdout."""
    header = (
        f"{'Agent':<22} "
        f"{'Uptake':>8} "
        f"{'Transfer':>9} "
        f"{'Scope':>7} "
        f"{'Forget':>7} "
        f"{'Consol':>7} "
        f"{'Identity':>9} "
        f"{'Mem/Δ':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, scorecard in rows:
        def f(key: str) -> str:
            value = scorecard.get(key)
            return f"{value:.4f}" if isinstance(value, float) else str(value)
        print(
            f"{name:<22} "
            f"{f('correction_uptake_latency'):>8} "
            f"{f('transfer_score'):>9} "
            f"{f('scope_score'):>7} "
            f"{f('forgetting_score'):>7} "
            f"{f('consolidation_score'):>7} "
            f"{f('identity_drift_score'):>9} "
            f"{scorecard.get('memory_bytes_per_behavior_delta')!s:>12}"
        )


def main() -> int:
    """Run evaluation over selected agents and log the results."""
    available_agents: dict[str, type] = {
        "ZeroMemoryAgent": ZeroMemoryAgent,
        "ContextOnlyAgent": ContextOnlyAgent,
        "FastOnlyAgent": FastOnlyAgent,
        "HippocampusOnlyAgent": HippocampusOnlyAgent,
        "IdentityOnlyAgent": IdentityOnlyAgent,
        "OrganismAgent": OrganismAgent,
        "NullAgent": NullAgent,
    }

    parser = argparse.ArgumentParser(description="Run the Oczy curriculum evaluation.")
    parser.add_argument(
        "--agent",
        choices=list(available_agents.keys()),
        help="Restrict the run to a single agent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to build the curriculum (default: 0).",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip the consolidation phase.",
    )
    parser.add_argument(
        "--exact-match",
        action="store_true",
        help="Use exact string matching instead of sense-level scoring.",
    )
    args = parser.parse_args()

    config = RunConfig(
        seed=args.seed,
        consolidate=not args.no_consolidate,
        sense_match=not args.exact_match,
        num_repetitions=1,
    )

    agent_order = [args.agent] if args.agent else [
        "ZeroMemoryAgent",
        "ContextOnlyAgent",
        "FastOnlyAgent",
        "HippocampusOnlyAgent",
        "IdentityOnlyAgent",
        "OrganismAgent",
    ]

    logger = ExperimentLogger()
    rows: list[tuple[str, dict[str, Any]]] = []

    for agent_name in agent_order:
        agent_cls = available_agents[agent_name]
        agent = agent_cls()
        scorecard, artifacts = evaluate_agent(agent, agent_name, config)
        logger.log_run(
            run_id=agent_name,
            config={"seed": config.seed, "consolidate": config.consolidate, "sense_match": config.sense_match, "num_repetitions": config.num_repetitions},
            scorecard=scorecard,
            artifacts=artifacts,
        )
        logger.append_findings(agent_name, _findings(agent_name, scorecard, artifacts))
        rows.append((agent_name, scorecard))

    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
