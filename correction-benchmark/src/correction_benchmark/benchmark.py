"""Correction-to-Competence Benchmark runner.

The benchmark protocol is intentionally narrow so it can evaluate any agent
that exposes:

    answer(user_request: str) -> str
    correct(correction: str, expected: str) -> None
"""

from __future__ import annotations

from typing import Any

from .dataset import Episode, build_dataset
from .scorer import EpisodeResult, ProbeResult, Scorer, matches


def _evaluate_probes(agent: Any, episode: Episode) -> tuple[ProbeResult, ...]:
    """Ask every probe in ``episode`` and score the response.

    The agent's ``answer()`` is called *once* per probe.  Calling it twice
    (once for the recorded answer, once for the scored verdict) silently
    mutated stateful agents: ``HippocampusOnlyAgent`` incremented
    ``replay_count`` on stored traces, ``SkillImmuneCortex`` incremented
    ``hit_count``, and ``WorldModelCritic`` updated ``_last_correction_prob``.
    That double counting corrupted replay consolidation thresholds, hit
    counters, and downstream metrics for every stateful agent.
    """
    results: list[ProbeResult] = []
    for probe in episode.probes:
        answer = agent.answer(probe.request)
        results.append(
            ProbeResult(probe=probe, answer=answer, correct=matches(answer, probe.expected))
        )
    return tuple(results)


def run_benchmark(agent: Any) -> dict[str, float]:
    """Run the full benchmark against ``agent`` and return a score card.

    The runner walks through every episode once:

    1. Records the agent's initial answer to the ambiguous request.
    2. Delivers the user correction.
    3. Records the post-correction answer to the same request.
    4. Asks all later probes (transfer, scope, forgetting).

    Scores are aggregate across all episodes.
    """
    dataset = build_dataset()
    results: list[EpisodeResult] = []

    for episode in dataset:
        initial_answer = agent.answer(episode.request)
        agent.correct(episode.correction, episode.corrected_answer)
        post_correction_answer = agent.answer(episode.request)

        # Reuse the initial call to avoid side effects and to match the
        # post-correction call under identical conditions.
        probe_results = _evaluate_probes(agent, episode)

        results.append(
            EpisodeResult(
                episode=episode,
                initial_answer=initial_answer,
                post_correction_answer=post_correction_answer,
                probe_results=probe_results,
            )
        )

    return Scorer.score(tuple(results), agent)
