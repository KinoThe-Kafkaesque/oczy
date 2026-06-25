#!/usr/bin/env python3
"""Driver that feeds the organism curriculum to an Oczy agent.

Usage:
    python experiments/organism_curriculum/run_curriculum.py
    python experiments/organism_curriculum/run_curriculum.py --agent OrganismAgent
    python experiments/organism_curriculum/run_curriculum.py --lm
    python experiments/organism_curriculum/run_curriculum.py --stages stage_0_grounding stage_1_transfer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from oczy.experiments.organism import LMBackendAgent, OrganismAgent
from oczy.experiments.organism_curriculum.dataset import Episode, Probe, Stage, build_curriculum
from oczy.experiments.organism_curriculum.scoring import categorize_results, probe_matches
from oczy.experiments.organism_curriculum.validation import validate_curriculum


@dataclass
class EpisodeResult:
    id: str
    initial_request: str
    first_answer: str
    second_answer: str
    corrected_response: str
    fixed: bool
    lm_parse_ok: bool | None = None


@dataclass
class StageResult:
    name: str
    description: str
    episode_results: list[EpisodeResult] = field(default_factory=list)
    pre_probe_results: list[tuple[Any, str, bool]] = field(default_factory=list)
    post_probe_results: list[tuple[Any, str, bool]] = field(default_factory=list)
    memory_bytes_before: int = 0
    memory_bytes_after: int = 0

    def uptake_latency(self) -> float:
        if not self.episode_results:
            return 0.0
        not_fixed = sum(1 for r in self.episode_results if not r.fixed)
        return not_fixed / len(self.episode_results)


def load_agent(agent_name: str, config: dict[str, Any]) -> OrganismAgent | LMBackendAgent:
    """Construct the requested agent class."""
    if agent_name == "LMBackendAgent":
        return LMBackendAgent(config)
    return OrganismAgent(config)


def _agent_memory_bytes(agent: Any) -> int:
    if hasattr(agent, "memory_bytes"):
        return int(agent.memory_bytes())
    return 0


def run_battery(
    agent: Any,
    stage: Stage,
    episodes: tuple[Episode, ...] | None,
) -> list[tuple[Any, str, bool]]:
    """Run all probes from ``stage`` against ``agent``.

    If ``episodes`` is supplied, only probes belonging to those episodes are
    run (used for pre/post tests scoped to the current stage).
    """
    results: list[tuple[Any, str, bool]] = []
    episode_set = set(episodes) if episodes is not None else None
    for ep in stage.episodes:
        if episode_set is not None and ep not in episode_set:
            continue
        for probe in ep.probes:
            answer = agent.answer(probe.request)
            ok = probe_matches(answer, probe, ep)
            results.append((probe, answer, ok))
    return results


def build_nl_utterance(episode: Episode) -> str:
    """Compose a single natural-language utterance from request + correction."""
    return "%s %s" % (episode.initial_request, episode.correction_utterance)


def run_stage(
    agent: Any,
    stage: Stage,
    adapter: Any | None,
) -> StageResult:
    """Present every episode in ``stage`` to ``agent`` and return metrics."""
    result = StageResult(name=stage.name, description=stage.description)
    result.memory_bytes_before = _agent_memory_bytes(agent)

    # Pre-test probes *before* this stage's acquisition episodes.
    result.pre_probe_results = run_battery(agent, stage, stage.episodes)

    for ep in stage.episodes:
        first_answer = agent.answer(ep.initial_request)

        lm_parse_ok: bool | None = None
        if adapter is not None:
            nl = build_nl_utterance(ep)
            parsed = adapter.nl_to_episode(nl)
            parsed_corrected = parsed.get("corrected_answer", "")
            lm_parse_ok = bool(
                parsed_corrected
                and parsed_corrected.lower() in ep.corrected_response.lower()
            )
            query = parsed.get("query") or ep.initial_request
            correction = parsed.get("correction") or ep.correction_utterance
            agent.learn(query, correction)
        else:
            agent.learn(ep.initial_request, ep.correction_utterance)

        second_answer = agent.answer(ep.initial_request)
        retention_probe = Probe(ep.initial_request, ep.corrected_response, "retention", "sense")
        fixed = probe_matches(second_answer, retention_probe, ep)

        result.episode_results.append(
            EpisodeResult(
                id=ep.id,
                initial_request=ep.initial_request,
                first_answer=first_answer,
                second_answer=second_answer,
                corrected_response=ep.corrected_response,
                fixed=fixed,
                lm_parse_ok=lm_parse_ok,
            )
        )

    # Post-test probes *after* acquisition.
    result.post_probe_results = run_battery(agent, stage, stage.episodes)
    result.memory_bytes_after = _agent_memory_bytes(agent)
    return result


def _shorten(text: str, width: int = 40) -> str:
    text = text.replace("\n", " ")
    if len(text) > width:
        return text[: width - 3] + "..."
    return text


def _accuracy(items: list[bool]) -> float:
    return sum(items) / len(items) if items else 0.0


def print_summary(results: list[StageResult]) -> None:
    header = "%-28s %8s %7s %6s %6s %10s" % (
        "Stage", "Episodes", "Uptake", "Pre", "Post", "Mem d"
    )
    print(header)
    print("-" * len(header))
    for sr in results:
        total = len(sr.episode_results)
        fixed = sum(1 for r in sr.episode_results if r.fixed)
        uptake = sr.uptake_latency()
        pre_acc = categorize_results(sr.pre_probe_results)
        post_acc = categorize_results(sr.post_probe_results)
        pre_total = sum(v[1] for v in pre_acc.values())
        pre_ok = sum(v[0] for v in pre_acc.values())
        post_total = sum(v[1] for v in post_acc.values())
        post_ok = sum(v[0] for v in post_acc.values())
        pre = pre_ok / pre_total if pre_total else 0.0
        post = post_ok / post_total if post_total else 0.0
        mem_delta = sr.memory_bytes_after - sr.memory_bytes_before
        print(
            "%-28s %3d/%-4d %6.2f %5.2f %5.2f %+9dB"
            % (sr.name, fixed, total, uptake, pre, post, mem_delta)
        )


def print_per_stage(results: list[StageResult]) -> None:
    for sr in results:
        print("\n%s" % sr.name)
        if sr.description:
            print("  %s" % sr.description)
        for er in sr.episode_results:
            marker = "OK" if er.fixed else ".."
            lm_info = ""
            if er.lm_parse_ok is not None:
                lm_info = " (lm=%s)" % ("ok" if er.lm_parse_ok else "fail")
            print(
                "  [%s] %-25s 1st=%-25r 2nd=%-25r%s"
                % (
                    marker,
                    er.id,
                    _shorten(er.first_answer, 22),
                    _shorten(er.second_answer, 22),
                    lm_info,
                )
            )
        post_acc = categorize_results(sr.post_probe_results)
        if post_acc:
            parts = ", ".join(
                "%s=%.2f" % (cat, acc) for cat, (_ok, _tot, acc) in sorted(post_acc.items())
            )
            print("  post-test accuracy: %s" % parts)


def write_report(
    results: list[StageResult],
    agent_name: str,
    use_lm: bool,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable: list[dict[str, Any]] = []
    for sr in results:
        serializable.append(
            {
                "name": sr.name,
                "description": sr.description,
                "memory_bytes_before": sr.memory_bytes_before,
                "memory_bytes_after": sr.memory_bytes_after,
                "uptake_latency": sr.uptake_latency(),
                "pre_accuracy": {k: v[2] for k, v in categorize_results(sr.pre_probe_results).items()},
                "post_accuracy": {k: v[2] for k, v in categorize_results(sr.post_probe_results).items()},
                "episodes": [asdict(er) for er in sr.episode_results],
            }
        )
    payload = {
        "agent": agent_name,
        "use_lm": use_lm,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": serializable,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Oczy organism curriculum.")
    p.add_argument(
        "--agent",
        choices=["OrganismAgent", "LMBackendAgent"],
        default="OrganismAgent",
        help="Agent class to evaluate (default: OrganismAgent).",
    )
    p.add_argument(
        "--config",
        default="{}",
        help="JSON config passed to the agent constructor.",
    )
    p.add_argument(
        "--lm",
        action="store_true",
        help="Feed episodes through the LM perception layer (LanguageAdapter).",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        help="Run only these stage files by basename (without .json).",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the curriculum validation smoke test.",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory for the JSON report.",
    )
    p.add_argument(
        "--report-name",
        default="run.json",
        help="Report filename.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    agent_config: dict[str, Any] = json.loads(args.config)

    stage_names = tuple(args.stages) if args.stages else None
    stages = build_curriculum(stage_names=stage_names)

    if not args.no_validate:
        report = validate_curriculum(stages)
        if not report.ok:
            print("Curriculum validation failed:")
            for e in report.errors:
                print("  - %s" % e)
            return 1
        if report.warnings:
            print("Curriculum validation warnings:")
            for w in report.warnings:
                print("  - %s" % w)

    agent = load_agent(args.agent, agent_config)

    adapter = None
    if args.lm:
        try:
            from oczy.lm import LanguageAdapter

            adapter = LanguageAdapter()
            adapter.load()
            print("LM perception adapter loaded.")
        except Exception as exc:  # noqa: BLE001
            print("Could not load LM adapter; continuing in raw mode. (%s)" % exc)

    results: list[StageResult] = []
    for stage in stages:
        if stage.consolidate_before:
            print("Consolidating before %s..." % stage.name)
            agent.consolidate()
        print("Running %s..." % stage.name)
        results.append(run_stage(agent, stage, adapter))
        if stage.consolidate_after:
            print("Consolidating after %s..." % stage.name)
            agent.consolidate()

    print("\n=== Organism curriculum summary ===")
    print_summary(results)
    print_per_stage(results)

    report_path = args.report_dir / args.report_name
    write_report(results, args.agent, args.lm, report_path)
    print("\nReport written to: %s" % report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
