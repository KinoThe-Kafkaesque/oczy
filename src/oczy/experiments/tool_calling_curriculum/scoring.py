"""Deterministic, mechanism-independent scoring for tool curriculum outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .dataset import ToolEpisode


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class EpisodeScore:
    format_correct: bool
    tool_sequence_correct: bool
    params_correct: bool
    result_integrated: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.format_correct,
                self.tool_sequence_correct,
                self.params_correct,
                self.result_integrated,
            )
        )


def parse_tool_call(text: str) -> ParsedToolCall | None:
    """Parse one JSON or bracket-format tool call without fuzzy acceptance."""
    text = text.strip()
    candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for candidate in candidates:
        try:
            obj: Any = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments") or obj.get("args") or {}
        if isinstance(obj.get("name"), str) and isinstance(args, dict):
            return ParsedToolCall(
                obj["name"], {str(key): str(value) for key, value in args.items()}
            )

    match = re.fullmatch(r"\s*\[(\w+)\s*\((.*)\)\]\s*", text, re.DOTALL)
    if not match:
        return None
    args: dict[str, str] = {}
    for item in re.finditer(r"(\w+)\s*=\s*(['\"])(.*?)\2", match.group(2)):
        args[item.group(1)] = item.group(3)
    return ParsedToolCall(match.group(1), args)


def score_episode(episode: ToolEpisode, outputs: list[str]) -> EpisodeScore:
    """Score ordered calls, required parameters, and grounded answer terms."""
    calls = [call for output in outputs if (call := parse_tool_call(output))]
    expected_tools = episode.expected_tools
    observed_tools = tuple(call.name for call in calls[: len(expected_tools)])
    format_correct = not expected_tools or bool(calls)
    sequence_correct = observed_tools == expected_tools

    required_params = dict(episode.expected_params)
    params_correct = True
    if required_params:
        params_correct = bool(calls) and all(
            expected.lower() in calls[0].arguments.get(key, "").lower()
            for key, expected in required_params.items()
        )

    answer_outputs = [output.lower() for output in outputs if parse_tool_call(output) is None]
    answer_text = "\n".join(answer_outputs)
    result_integrated = all(term.lower() in answer_text for term in episode.expected_answer_terms)
    return EpisodeScore(
        format_correct=format_correct,
        tool_sequence_correct=sequence_correct,
        params_correct=params_correct,
        result_integrated=result_integrated,
    )
