"""Reproducible runner for the Oczy curriculum evaluation.

Usage:
    uv run python experiments/run_experiment.py
    uv run python experiments/run_experiment.py --agent NullAgent
    uv run python experiments/run_experiment.py --agent OrganismAgent
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from oczy.experiments.baselines import (
    ContextOnlyAgent,
    FastOnlyAgent,
    HippocampusOnlyAgent,
    IdentityOnlyAgent,
    ZeroMemoryAgent,
)
from oczy.experiments.curriculum import build_curriculum
from oczy.experiments.eval_suite import EvalSuite, EvalResult, NullAgent
from oczy.experiments.logger import ExperimentLogger
from oczy.experiments.organism import OrganismAgent
from oczy.common import format_row, summarize


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


# Metric keys from ``EvalResult.final_card`` that are aggregatable floats.
_AGGREGATABLE_METRICS: tuple[str, ...] = (
    "correction_uptake_latency",
    "transfer_score",
    "scope_score",
    "forgetting_score",
    "consolidation_score",
    "identity_drift_score",
    "memory_bytes_per_behavior_delta",
)


def _build_agent(agent_name: str, agent_cls: type, seed: int) -> Any:
    """Construct a fresh agent for ``seed``, seeding organs where supported.

    Baseline agents without seed-aware organs (``ZeroMemoryAgent``,
    ``ContextOnlyAgent``, ``NullAgent``) are deterministic and take no config.
    """
    if agent_name in ("ZeroMemoryAgent", "ContextOnlyAgent", "NullAgent"):
        return agent_cls()
    if agent_name == "FastOnlyAgent":
        return agent_cls(config={"seed": seed})
    if agent_name == "HippocampusOnlyAgent":
        return agent_cls(config={"neural_hippocampus": {"seed": seed}})
    if agent_name == "IdentityOnlyAgent":
        return agent_cls(config={"identity_hypernetwork": {"seed": seed}})
    if agent_name == "OrganismAgent":
        return agent_cls(
            config={
                "plastic_cortex": {"seed": seed},
                "neural_hippocampus": {"seed": seed},
                "identity_hypernetwork": {"seed": seed},
                "world_model_critic": {"seed": seed},
            }
        )
    return agent_cls()


def _aggregate_scorecards(
    scorecards: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Summarize each aggregatable metric across per-seed scorecards.

    Returns ``{metric_key: summarize(values)}``.
    """
    per_metric: dict[str, list[float]] = {}
    for card in scorecards:
        for key in _AGGREGATABLE_METRICS:
            value = card.get(key)
            if isinstance(value, (int, float)):
                per_metric.setdefault(key, []).append(float(value))
    return {key: summarize(vals) for key, vals in per_metric.items()}


def _aggregated_card(
    per_metric: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Flatten per-metric summaries into a ``<metric>_(mean|std|n|ci95)`` card."""
    card: dict[str, Any] = {}
    for key, summary in per_metric.items():
        card[f"{key}_mean"] = summary.get("mean")
        card[f"{key}_std"] = summary.get("std")
        card[f"{key}_n"] = summary.get("n")
        card[f"{key}_ci95_half"] = summary.get("ci95_half")
    return card


def _print_multi_seed_table(
    rows: list[tuple[str, dict[str, dict[str, float]]]],
    num_seeds: int,
) -> None:
    """Print an ASCII comparison table of per-metric mean ± CI across seeds."""
    print("\n=== Multi-seed comparison (%d seeds) ===" % num_seeds)
    header = f"{'Agent':<22} {'transfer_score':>20}"
    print(header)
    print("-" * len(header))
    for agent_name, per_metric in rows:
        summary = per_metric.get("transfer_score")
        if summary and summary.get("n", 0):
            print(f"{format_row(agent_name, summary):<43}")
        else:
            print(f"{agent_name:<22} {'--':>20}")
    print()
    for agent_name, per_metric in rows:
        print(f"[{agent_name}]")
        for key in _AGGREGATABLE_METRICS:
            summary = per_metric.get(key)
            if summary and summary.get("n", 0):
                print(f"  {format_row(key, summary)}")
        print()

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
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds to evaluate (default: 1).",
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

    if args.seeds > 1:
        # Multi-seed path: build fresh agents per seed and aggregate metrics.
        print("Running %d seeds..." % args.seeds)
        seeds = list(range(args.seeds))
        per_agent: dict[str, list[dict[str, Any]]] = {}
        for agent_name in agent_order:
            agent_cls = available_agents[agent_name]
            per_agent[agent_name] = []
            for seed in seeds:
                agent = _build_agent(agent_name, agent_cls, seed)
                run_config = RunConfig(
                    seed=seed,
                    consolidate=config.consolidate,
                    sense_match=config.sense_match,
                    num_repetitions=config.num_repetitions,
                )
                scorecard, artifacts = evaluate_agent(agent, agent_name, run_config)
                per_agent[agent_name].append(scorecard)
                logger.log_run(
                    run_id=f"{agent_name}/seed{seed}",
                    config={
                        "seed": seed,
                        "seeds": args.seeds,
                        "consolidate": run_config.consolidate,
                        "sense_match": run_config.sense_match,
                        "num_repetitions": run_config.num_repetitions,
                    },
                    scorecard=scorecard,
                    artifacts=artifacts,
                )

        table_rows: list[tuple[str, dict[str, dict[str, float]]]] = []
        for agent_name in agent_order:
            per_metric = _aggregate_scorecards(per_agent[agent_name])
            agg_card = _aggregated_card(per_metric)
            logger.log_run(
                run_id=f"{agent_name}-aggregated",
                config={
                    "seeds": args.seeds,
                    "consolidate": config.consolidate,
                    "sense_match": config.sense_match,
                },
                scorecard=agg_card,
                artifacts={"per_seed": per_agent[agent_name]},
            )
            findings_lines = [f"## {agent_name} (aggregated, {args.seeds} seeds)\n"]
            for key in _AGGREGATABLE_METRICS:
                summary = per_metric.get(key)
                if summary and summary.get("n", 0):
                    findings_lines.append(f"- {format_row(key, summary)}")
            logger.append_findings(
                f"{agent_name}-aggregated", "\n".join(findings_lines)
            )
            table_rows.append((agent_name, per_metric))

        _print_multi_seed_table(table_rows, args.seeds)
        return 0

    # Single-seed path (default): point estimate, not reportable.
    print("\n\nWARNING: single-seed point estimate — not reportable")
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
