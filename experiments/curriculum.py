"""Curriculum and eval scaffold for the Oczy Plastic World Model Agent.

This module wraps the grouped benchmark data from
`experiments.data.curriculum_dataset` into a leveled `Curriculum` object that
exposes acquisition episodes, transfer/scope/forgetting probes, and a
pre/post stability battery.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Make the parent repo root importable when this file is executed directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.data.curriculum_dataset import (
    Chapter,
    EpisodeGroup,
    build_grouped_dataset,
    chapter_prompt_template,
    chapter_skill_summary,
    _chapter_order,
)

from correction_benchmark.dataset import Episode, Probe


@dataclass(frozen=True)
class CurriculumLevel:
    """One leveled chapter in the curriculum."""

    name: str
    prompt_template: str
    expected_skill_behavior: str
    group: EpisodeGroup


class Curriculum:
    """An ordered set of `CurriculumLevel`s derived from correction episodes."""

    def __init__(self, seed: int = 0) -> None:
        grouped = build_grouped_dataset(seed=seed)
        self._levels: tuple[CurriculumLevel, ...] = tuple(
            CurriculumLevel(
                name=_level_name(chapter, idx + 1),
                prompt_template=chapter_prompt_template(chapter),
                expected_skill_behavior=chapter_skill_summary(chapter),
                group=grouped[chapter],
            )
            for idx, chapter in enumerate(_chapter_order())
        )
        self._seed = seed

    @property
    def seed(self) -> int:
        """Random seed used to derive this curriculum."""
        return self._seed

    def levels(self) -> tuple[CurriculumLevel, ...]:
        """Return the ordered list of curriculum levels."""
        return self._levels

    def __len__(self) -> int:
        return len(self._levels)

    def __iter__(self) -> Iterable[CurriculumLevel]:
        return iter(self._levels)


def _level_name(chapter: Chapter, number: int) -> str:
    """Return a human-readable level name."""
    return f"Level {number}: {chapter.replace('_', ' ').title()}"


def build_curriculum(seed: int = 0) -> Curriculum:
    """Return the full curriculum; deterministic for a given seed."""
    return Curriculum(seed=seed)


def make_pre_post_battery(curriculum: Curriculum) -> tuple[Probe, ...]:
    """Return the union of all forgetting probes across all levels.

    These probes are used for pre-training and post-training stability scoring:
    the agent should answer them correctly both before and after it learns the
    curriculum, showing that correction learning does not destroy baseline
    factual knowledge.
    """
    probes: list[Probe] = []
    for level in curriculum.levels():
        probes.extend(level.group.forgetting_probes)
    return tuple(probes)


if __name__ == "__main__":
    curriculum = build_curriculum(seed=0)
    total_probes = 0
    for level in curriculum.levels():
        group = level.group
        n_transfer = len(group.transfer_probes)
        n_scope = len(group.scope_probes)
        n_forgetting = len(group.forgetting_probes)
        n_total = n_transfer + n_scope + n_forgetting
        total_probes += n_total
        print(
            f"{level.name}: prompts={len(group.acquisition_episodes)} "
            f"transfer={n_transfer} scope={n_scope} "
            f"forgetting={n_forgetting} total={n_total}"
        )
    print(f"Pre/post stability battery size: {len(make_pre_post_battery(curriculum))}")
    print(f"Total probes across all levels: {total_probes}")
