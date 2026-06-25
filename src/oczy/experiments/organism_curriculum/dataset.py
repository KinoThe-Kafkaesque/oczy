"""Data model and loader for the Oczy organism curriculum.

Stages are stored as JSON files under ``stages/``.  Each file contains a
single :class:`Stage` represented as a dict.  The loader enforces the
schema and returns an ordered ``tuple[Stage, ...]``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ProbeCategory = Literal["transfer", "scope", "forgetting", "retention"]
MatchMode = Literal["exact", "sense", "contains"]


@dataclass(frozen=True)
class Probe:
    """A later question used to test whether a correction generalised."""

    request: str
    expected: str
    category: ProbeCategory
    match_mode: MatchMode = "sense"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "expected": self.expected,
            "category": self.category,
            "match_mode": self.match_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Probe":
        return cls(
            request=str(data["request"]),
            expected=str(data["expected"]),
            category=data["category"],  # type: ignore[arg-type]
            match_mode=data.get("match_mode", "sense"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class Episode:
    """One organism learning episode.

    The ``default_response`` documents what a naive agent would answer; the
    actual agent under test generates its own prior answer via
    ``agent.answer(initial_request)``.  The correction is consumed by
    ``agent.learn(initial_request, correction_utterance)``.
    """

    id: str
    initial_request: str
    default_response: str
    correction_utterance: str
    corrected_label: str
    corrected_response: str
    domain: str
    probes: tuple[Probe, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "initial_request": self.initial_request,
            "default_response": self.default_response,
            "correction_utterance": self.correction_utterance,
            "corrected_label": self.corrected_label,
            "corrected_response": self.corrected_response,
            "domain": self.domain,
            "probes": [p.to_dict() for p in self.probes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        probes = tuple(
            Probe.from_dict(p) if isinstance(p, dict) else Probe.from_dict({"request": p[0], "expected": p[1], "category": p[2]})
            for p in data.get("probes", ())
        )
        return cls(
            id=str(data["id"]),
            initial_request=str(data["initial_request"]),
            default_response=str(data.get("default_response", "")),
            correction_utterance=str(data["correction_utterance"]),
            corrected_label=str(data["corrected_label"]),
            corrected_response=str(data["corrected_response"]),
            domain=str(data.get("domain", "general")),
            probes=probes,
        )

    def ambiguous_token(self) -> str | None:
        """Extract the quoted token from a correction like ``No, 'X' means Y``."""
        text = self.correction_utterance
        start = text.find("'")
        end = text.find("'", start + 1)
        if start != -1 and end != -1:
            return text[start + 1 : end].lower()
        # Fallback: try double quotes.
        start = text.find('"')
        end = text.find('"', start + 1)
        if start != -1 and end != -1:
            return text[start + 1 : end].lower()
        return None


@dataclass(frozen=True)
class Stage:
    """One leveled chapter of the organism curriculum."""

    name: str
    description: str
    consolidate_before: bool
    consolidate_after: bool
    episodes: tuple[Episode, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "consolidate_before": self.consolidate_before,
            "consolidate_after": self.consolidate_after,
            "episodes": [ep.to_dict() for ep in self.episodes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Stage":
        episodes = tuple(
            Episode.from_dict(e) if isinstance(e, dict) else Episode.from_dict({"id": e[0], "initial_request": e[1], "default_response": e[2], "correction_utterance": e[3], "corrected_label": e[4], "corrected_response": e[5], "domain": e[6], "probes": e[7]})
            for e in data.get("episodes", ())
        )
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            consolidate_before=bool(data.get("consolidate_before", False)),
            consolidate_after=bool(data.get("consolidate_after", False)),
            episodes=episodes,
        )


STAGE_ORDER = (
    "stage_0_grounding",
    "stage_1_transfer",
    "stage_2_scope",
    "stage_3_dialog",
    "stage_4_consolidation",
    "stage_5_cross_domain",
)


def default_stages_dir() -> Path:
    """Return the directory containing the bundled stage JSON files."""
    return Path(__file__).resolve().parent / "stages"


def load_stage(path: Path) -> Stage:
    """Load a single stage from a JSON file."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Stage.from_dict(data)


def build_curriculum(
    stages_dir: Path | None = None,
    stage_names: tuple[str, ...] | None = None,
) -> tuple[Stage, ...]:
    """Load the ordered organism curriculum.

    Args:
        stages_dir: Directory containing ``stage_*.json`` files.  Defaults to
            the bundled ``stages/`` directory.
        stage_names: Subset/order of stages to load.  Defaults to
            :data:`STAGE_ORDER`.

    Returns:
        A tuple of :class:`Stage` objects in the requested order.
    """
    stages_dir = stages_dir or default_stages_dir()
    names = stage_names or STAGE_ORDER
    stages: list[Stage] = []
    for name in names:
        path = stages_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing stage file: {path}")
        stages.append(load_stage(path))
    return tuple(stages)


def extract_tokens(text: str) -> set[str]:
    """Simple alphanumeric token extractor used by validation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))
