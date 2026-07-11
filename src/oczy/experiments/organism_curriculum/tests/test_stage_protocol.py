"""Regression tests for the v2.2 stage-execution protocol."""

from __future__ import annotations

from typing import Any

from oczy.experiments.organism_curriculum.dataset import Episode, Probe, Stage
from oczy.experiments.organism_curriculum.run_curriculum import run_stage


class _RecordingAgent:
    def __init__(self, answer_text: str = "expected") -> None:
        self.answer_text = answer_text
        self.events: list[tuple[str, str]] = []
        self.consolidations = 0

    def answer(self, request: str) -> str:
        self.events.append(("answer", request))
        return self.answer_text

    def learn(self, request: str, correction: str) -> None:
        self.events.append(("learn", request))

    def consolidate(self) -> None:
        self.events.append(("consolidate", ""))
        self.consolidations += 1

    def memory_bytes(self) -> int:
        return self.consolidations


def _episode(idx: int, *, expected: str = "expected") -> Episode:
    return Episode(
        id=f"e{idx}",
        initial_request=f"initial-{idx}",
        default_response="wrong",
        correction_utterance=f"No, 'token{idx}' means {expected}.",
        corrected_label=expected,
        corrected_response=expected,
        domain="test",
        probes=(
            Probe(
                request=f"probe-{idx}",
                expected=expected,
                category="transfer",
                match_mode="exact",
            ),
        ),
    )


def _stage(**overrides: Any) -> Stage:
    values: dict[str, Any] = {
        "name": "test stage",
        "description": "",
        "consolidate_before": False,
        "consolidate_after": False,
        "episodes": (_episode(1), _episode(2)),
    }
    values.update(overrides)
    return Stage(**values)


def test_probe_only_stage_never_reteaches_or_reasks() -> None:
    agent = _RecordingAgent()
    result = run_stage(agent, _stage(teach_episodes=False), adapter=None)

    assert not [event for event in agent.events if event[0] == "learn"]
    assert agent.events == [("answer", "probe-1"), ("answer", "probe-2")]
    assert result.post_probe_results == result.pre_probe_results
    assert result.episode_results == []


def test_dialog_probes_are_interleaved_with_their_episode() -> None:
    agent = _RecordingAgent()
    run_stage(
        agent,
        _stage(probe_timing="after_episode"),
        adapter=None,
    )

    events = agent.events
    first_followup = events.index(("answer", "probe-1"), 2)
    second_acquisition = events.index(("answer", "initial-2"), 2)
    assert first_followup < second_acquisition


def test_consolidation_precedes_post_test_and_memory_snapshot() -> None:
    agent = _RecordingAgent()
    result = run_stage(
        agent,
        _stage(consolidate_after=True),
        adapter=None,
    )

    consolidate_idx = agent.events.index(("consolidate", ""))
    post_probe_idx = agent.events.index(("answer", "probe-1"), consolidate_idx)
    assert consolidate_idx < post_probe_idx
    assert result.memory_bytes_before == 0
    assert result.memory_bytes_after == 1


def test_semantic_mode_is_applied_to_post_test() -> None:
    episode = Episode(
        id="semantic",
        initial_request="Show the log.",
        default_response="wrong",
        correction_utterance="No, 'log' means the captain's journal.",
        corrected_label="captain's journal",
        corrected_response="captain's journal",
        domain="nautical",
        probes=(
            Probe(
                request="Bring it up.",
                expected="captain's journal",
                category="transfer",
                match_mode="sense",
            ),
        ),
    )
    agent = _RecordingAgent(answer_text="ship record")
    result = run_stage(
        agent,
        _stage(episodes=(episode,)),
        adapter=None,
        semantic=True,
    )

    assert result.post_probe_results[0][2] is True
