"""Evaluation suite and agent stubs for the Oczy organism curriculum.

The suite runs a pre-test, presents each curriculum level's acquisition
episodes (answer/correct/answer), runs a post-test, then a consolidation test,
and produces a JSON-serializable scorecard.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from correction_benchmark.dataset import Episode, Probe
from oczy.experiments.curriculum import Curriculum, EpisodeGroup, build_curriculum, make_pre_post_battery


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalResult:
    """Container for every artifact produced by an evaluation run."""

    pre_test_scores: dict[str, float | None]
    post_test_scores: dict[str, float | None]
    per_level_results: list[dict[str, Any]]
    final_card: dict[str, Any]
    memory_bytes: int
    raw_trace_size: int
    consolidated_size: int
    sense_match: bool = False

    def scorecard_json(self) -> dict[str, Any]:
        """Return a JSON-serializable scorecard dictionary."""
        return {
            "metrics": self.final_card,
            "pre_test_scores": self.pre_test_scores,
            "post_test_scores": self.post_test_scores,
            "memory_bytes": self.memory_bytes,
            "raw_trace_size": self.raw_trace_size,
            "consolidated_size": self.consolidated_size,
            "per_level_results": self.per_level_results,
            "sense_match": self.sense_match,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ProbeAnswer:
    probe: Probe
    answer: str
    correct: bool


@dataclass(frozen=True)
class _TestSnapshot:
    """Results of running one probe battery."""

    transfer: list[_ProbeAnswer]
    scope: list[_ProbeAnswer]
    forgetting: list[_ProbeAnswer]
    identity: list[_ProbeAnswer]

    def accuracy(self, category: str) -> float:
        battery = getattr(self, category, [])
        if not battery:
            return 0.0
        return sum(item.correct for item in battery) / len(battery)


def _normalize(text: str) -> str:
    """Lowercase, strip, and collapse whitespace for answer matching."""
    return " ".join(str(text).strip().lower().split())

# Minimal stopword set for sense-level matching: strip grammatical words and
# the high-frequency task verbs that appear in almost every agent answer.
_STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "is",
    "i'll",
    "i",
    "ll",
    "will",
    "update",
    "edit",
    "to",
    "of",
    "and",
    "for",
    "in",
    "it",
    "this",
    "with",
    "my",
    "your",
    "on",
    "at",
    "from",
    "be",
    "no",
    "here",
    "means",
    "or",
}


def _token_set(text: str) -> set[str]:
    """Return lowercased, non-stopword tokens extracted from ``text``."""
    return {
        token
        for token in re.findall(r"\b\w+\b", str(text).lower())
        if token and token not in _STOPWORDS
    }


def _ambiguous_token(episode: Episode) -> str:
    """Extract the quoted ambiguous token from an episode's correction."""
    start = episode.correction.find("'")
    end = episode.correction.find("'", start + 1)
    if start != -1 and end != -1:
        return episode.correction[start + 1 : end].lower()
    return ""


def _matches(
    answer: str,
    expected: str,
    probe: Probe | None = None,
    probe_to_episode: dict[Probe, Episode] | None = None,
    sense_match: bool = False,
    episode: Episode | None = None,
) -> bool:
    """Return whether ``answer`` matches ``expected``.

    By default matching is exact after whitespace normalization.  When
    ``sense_match`` is enabled and the originating episode is available
    (either directly or via ``probe``/``probe_to_episode``), the match
    accepts answers that share the same disambiguated sense as the expected
    answer.
    """
    if _normalize(answer) == _normalize(expected):
        return True

    if not sense_match:
        return False

    if episode is None:
        if probe is None or probe_to_episode is None or probe not in probe_to_episode:
            return False
        episode = probe_to_episode[probe]

    token = _ambiguous_token(episode)
    wrong_sense = _token_set(episode.initial_wrong_answer)
    correct_sense = _token_set(episode.corrected_answer)
    expected_tokens = _token_set(expected)

    wrong_overlap = len((expected_tokens & wrong_sense) - {token})
    correct_overlap = len((expected_tokens & correct_sense) - {token})
    expected_sense_tokens = wrong_sense if wrong_overlap > correct_overlap else correct_sense

    agent_tokens = _token_set(answer)
    if token and token in agent_tokens:
        # The ambiguous token is present in the answer; keep it in the token
        # set as requested, but do not count it toward the overlap threshold.
        pass

    shared_tokens = (agent_tokens & expected_sense_tokens) - {token}
    return len(shared_tokens) >= 2


def _respond(agent: Any, prompt: str) -> str:
    """Dispatch to the first supported query method on the agent."""
    for name in ("respond", "answer", "query"):
        if hasattr(agent, name):
            method = getattr(agent, name)
            if callable(method):
                return str(method(prompt))
    raise TypeError(f"Agent {type(agent).__name__!r} has no respond/answer/query method")


def _learn(agent: Any, request: str, correction: str) -> None:
    """Dispatch correction learning to the first supported update method."""
    for name in ("learn", "correct"):
        if not hasattr(agent, name):
            continue
        method = getattr(agent, name)
        if not callable(method):
            continue
        try:
            method(request, correction)
            return
        except TypeError:
            pass
        try:
            method(correction)
            return
        except TypeError:
            pass
    # If the agent exposes no update hook, assume stateless correction handling.


def _consolidate(agent: Any) -> None:
    """Tell the agent to drop raw traces and keep consolidated knowledge."""
    for name in ("consolidate", "drop_raw_traces", "compress"):
        if hasattr(agent, name):
            method = getattr(agent, name)
            if callable(method):
                method()
                return


def _memory_bytes(agent: Any) -> int:
    if hasattr(agent, "memory_bytes"):
        return int(agent.memory_bytes())
    try:
        return sys.getsizeof(agent)
    except Exception:
        return 0


def _run_battery(
    agent: Any,
    probes: tuple[Probe, ...],
    probe_to_episode: dict[Probe, Episode] | None = None,
    sense_match: bool = False,
) -> list[_ProbeAnswer]:
    results: list[_ProbeAnswer] = []
    for probe in probes:
        answer = _respond(agent, probe.request)
        results.append(
            _ProbeAnswer(
                probe=probe,
                answer=answer,
                correct=_matches(
                    answer,
                    probe.expected,
                    probe=probe,
                    probe_to_episode=probe_to_episode,
                    sense_match=sense_match,
                ),
            )
        )
    return results


# ---------------------------------------------------------------------------
# EvalSuite
# ---------------------------------------------------------------------------

class EvalSuite:
    """Run the Oczy organism curriculum evaluation protocol.

    The protocol is:

        1. Pre-test all forgetting, transfer, and scope probes.
        2. Present every acquisition episode in each level: answer, correct,
           answer again.
        3. Post-test the same probes.
        4. Trigger consolidation (drop raw traces) and test once more.
        5. Emit a scorecard with uptake latency, transfer, scope, forgetting,
           consolidation, memory efficiency, and identity drift metrics.
    """

    def __init__(self, curriculum: Curriculum, sense_match: bool = False) -> None:
        self.curriculum = curriculum
        self.sense_match = sense_match
        self.identity_battery = make_pre_post_battery(curriculum)
        self._transfer_probes = tuple(
            probe for level in curriculum.levels() for probe in level.group.transfer_probes
        )
        self._scope_probes = tuple(
            probe for level in curriculum.levels() for probe in level.group.scope_probes
        )
        self._forgetting_probes = tuple(
            probe for level in curriculum.levels() for probe in level.group.forgetting_probes
        )
        # Each probe object in the curriculum belongs to exactly one episode.
        self._probe_to_episode: dict[Probe, Episode] = {
            probe: episode
            for level in curriculum.levels()
            for episode in level.group.acquisition_episodes
            for probe in episode.probes
        }
        # Caches populated by ``run`` so ``score`` can also be called with the
        # canonical three snapshots alone.
        self._last_level_results: list[dict[str, Any]] = []
        self._last_raw_trace_size: int = 0
        self._last_consolidated_size: int = 0

    def pre_test(self, agent: Any) -> _TestSnapshot:
        """Run forgetting, transfer, and scope probes once and record answers."""
        return _TestSnapshot(
            transfer=_run_battery(agent, self._transfer_probes, self._probe_to_episode, self.sense_match),
            scope=_run_battery(agent, self._scope_probes, self._probe_to_episode, self.sense_match),
            forgetting=_run_battery(agent, self._forgetting_probes, self._probe_to_episode, self.sense_match),
            identity=_run_battery(agent, self.identity_battery, self._probe_to_episode, self.sense_match),
        )

    def run_level(self, agent: Any, level: Any) -> dict[str, Any]:
        """Present each acquisition episode and record correction uptake."""
        group: EpisodeGroup = level.group
        episode_results: list[dict[str, Any]] = []
        not_fixed = 0
        for episode in group.acquisition_episodes:
            first_answer = _respond(agent, episode.request)
            _learn(agent, episode.request, episode.correction)
            second_answer = _respond(agent, episode.request)
            fixed = _matches(second_answer, episode.corrected_answer, sense_match=self.sense_match, episode=episode)
            if not fixed:
                not_fixed += 1
            episode_results.append(
                {
                    "request": episode.request,
                    "expected": episode.corrected_answer,
                    "first_answer": first_answer,
                    "second_answer": second_answer,
                    "fixed_after_correction": fixed,
                    "latency": 0 if fixed else 1,
                }
            )
        total = len(group.acquisition_episodes)
        latency = not_fixed / total if total else 0.0
        return {
            "level": level.name,
            "chapter": group.chapter,
            "uptake_latency": latency,
            "episodes_presented": total,
            "episodes_not_fixed": not_fixed,
            "episodes": episode_results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def post_test(self, agent: Any) -> _TestSnapshot:
        """Rerun all probes after acquisition."""
        return self.pre_test(agent)

    def consolidation_test(self, agent: Any) -> _TestSnapshot:
        """Drop raw traces and rerun the full probe battery."""
        _consolidate(agent)
        return self.pre_test(agent)

    def score(
        self,
        pre: _TestSnapshot,
        post: _TestSnapshot,
        consolidation: _TestSnapshot,
        level_results: list[dict[str, Any]] | None = None,
        raw_trace_size: int | None = None,
        consolidated_size: int | None = None,
    ) -> EvalResult:
        """Compute the final scorecard from test snapshots and level traces."""
        if level_results is None:
            level_results = self._last_level_results
        if raw_trace_size is None:
            raw_trace_size = self._last_raw_trace_size
        if consolidated_size is None:
            consolidated_size = self._last_consolidated_size
        pre_forgetting = pre.accuracy("forgetting") or 1e-9
        pre_identity = pre.accuracy("identity") or 1e-9

        transfer_score = post.accuracy("transfer")
        scope_score = post.accuracy("scope")
        forgetting_score = min(1.0, post.accuracy("forgetting") / pre_forgetting)
        identity_drift_score = min(1.0, post.accuracy("identity") / pre_identity)

        post_all_correct = sum(
            post.accuracy(cat) * len(getattr(post, cat)) for cat in ("transfer", "scope", "forgetting", "identity")
        )
        post_total = sum(len(getattr(post, cat)) for cat in ("transfer", "scope", "forgetting", "identity"))
        post_overall = post_all_correct / post_total if post_total else 0.0

        cons_all_correct = sum(
            consolidation.accuracy(cat) * len(getattr(consolidation, cat))
            for cat in ("transfer", "scope", "forgetting", "identity")
        )
        cons_total = sum(len(getattr(consolidation, cat)) for cat in ("transfer", "scope", "forgetting", "identity"))
        cons_overall = cons_all_correct / cons_total if cons_total else 0.0

        consolidation_score = min(1.0, cons_overall / post_overall if post_overall > 0 else 0.0)

        total_episodes = sum(level["episodes_presented"] for level in level_results)
        successful_lessons = sum(
            1 for level in level_results for ep in level["episodes"] if ep["fixed_after_correction"]
        )
        memory_bytes_per_delta = consolidated_size / max(1, successful_lessons)
        correction_uptake_latency = (
            sum(level["uptake_latency"] * level["episodes_presented"] for level in level_results) / total_episodes
            if total_episodes
            else 0.0
        )

        pre_test_scores = {
            "transfer": round(pre.accuracy("transfer"), 6),
            "scope": round(pre.accuracy("scope"), 6),
            "forgetting": round(pre.accuracy("forgetting"), 6),
            "identity": round(pre.accuracy("identity"), 6),
        }
        post_test_scores = {
            "transfer": round(post.accuracy("transfer"), 6),
            "scope": round(post.accuracy("scope"), 6),
            "forgetting": round(post.accuracy("forgetting"), 6),
            "identity": round(post.accuracy("identity"), 6),
        }

        final_card: dict[str, Any] = {
            "correction_uptake_latency": round(correction_uptake_latency, 6),
            "transfer_score": round(transfer_score, 6),
            "scope_score": round(scope_score, 6),
            "forgetting_score": round(forgetting_score, 6),
            "consolidation_score": round(consolidation_score, 6),
            "memory_bytes_per_behavior_delta": round(memory_bytes_per_delta, 6),
            "identity_drift_score": round(identity_drift_score, 6),
            "curriculum_seed": getattr(self.curriculum, "seed", None),
            "num_levels": len(self.curriculum),
        }

        return EvalResult(
            pre_test_scores=pre_test_scores,
            post_test_scores=post_test_scores,
            per_level_results=level_results,
            final_card=final_card,
            memory_bytes=consolidated_size,
            raw_trace_size=raw_trace_size,
            consolidated_size=consolidated_size,
            sense_match=self.sense_match,
        )

    def run(self, agent: Any) -> EvalResult:
        """Run the full evaluation protocol and return a scorecard."""
        pre = self.pre_test(agent)
        level_results: list[dict[str, Any]] = []
        for level in self.curriculum.levels():
            level_results.append(self.run_level(agent, level))
        raw_trace_size = _memory_bytes(agent)
        post = self.post_test(agent)
        consolidation = self.consolidation_test(agent)
        consolidated_size = _memory_bytes(agent)
        self._last_level_results = level_results
        self._last_raw_trace_size = raw_trace_size
        self._last_consolidated_size = consolidated_size
        return self.score(pre, post, consolidation, level_results, raw_trace_size, consolidated_size)

    def empty_scorecard(self) -> dict[str, Any]:
        """Return an empty JSON-serializable scorecard template."""
        template_metrics = {
            "correction_uptake_latency": None,
            "transfer_score": None,
            "scope_score": None,
            "forgetting_score": None,
            "consolidation_score": None,
            "memory_bytes_per_behavior_delta": None,
            "identity_drift_score": None,
            "curriculum_seed": getattr(self.curriculum, "seed", None),
            "num_levels": len(self.curriculum),
        }
        template_scores = {"transfer": None, "scope": None, "forgetting": None, "identity": None}
        return {
            "metrics": template_metrics,
            "pre_test_scores": template_scores,
            "post_test_scores": template_scores,
            "memory_bytes": 0,
            "raw_trace_size": 0,
            "consolidated_size": 0,
            "per_level_results": [],
            "sense_match": self.sense_match,
        }


# ---------------------------------------------------------------------------
# Agent stubs
# ---------------------------------------------------------------------------

class NullAgent:
    """Agent that never emits an answer and never learns."""

    def respond(self, request: str) -> str:
        return ""

    def learn(self, request: str, correction: str) -> None:
        pass

    def consolidate(self) -> None:
        pass

    def memory_bytes(self) -> int:
        return 0


class BaselineAgent:
    """Agent that always repeats a default answer and cannot learn."""

    def __init__(self, default_answer: str = "I don't know.") -> None:
        self.default_answer = default_answer

    def respond(self, request: str) -> str:
        return self.default_answer

    def learn(self, request: str, correction: str) -> None:
        pass

    def consolidate(self) -> None:
        pass

    def memory_bytes(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    curriculum = build_curriculum(seed=0)
    suite = EvalSuite(curriculum)
    print(json.dumps(suite.empty_scorecard(), indent=2))
