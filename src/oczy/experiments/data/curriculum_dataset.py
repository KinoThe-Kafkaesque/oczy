"""Curriculum dataset: grouped Correction Benchmark episodes by skill chapter.

This module layers a curriculum taxonomy over
`correction_benchmark.dataset`.  It is self-contained so that it can be
imported from `experiments.curriculum` or executed directly as a smoke test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from correction_benchmark.dataset import Episode, Probe, build_dataset as _build_dataset


Chapter = Literal[
    "vocabulary_re grounding",
    "tool_use",
    "role_alignment",
    "constraint_learning",
]


@dataclass(frozen=True)
class EpisodeGroup:
    """One curriculum unit: acquisition episodes plus their probe batteries."""

    acquisition_episodes: tuple[Episode, ...]
    transfer_probes: tuple[Probe, ...]
    scope_probes: tuple[Probe, ...]
    forgetting_probes: tuple[Probe, ...]
    chapter: Chapter


# Tokens grouped by the conceptual chapter they teach.  Each inner list is the
# canonical order of ambiguous tokens inside that chapter.
_CHAPTER_TOKENS: dict[Chapter, list[str]] = {
    "vocabulary_re grounding": ["profile", "model", "batch"],
    "tool_use": ["file", "key", "module"],
    "role_alignment": ["service", "branch", "table"],
    "constraint_learning": ["cell", "record", "run"],
}


def _token_from_episode(episode: Episode) -> str:
    """Extract the ambiguous token from a correction string like 'key'."""
    # Each correction uses single quotes around the target token, e.g.
    # "No, 'model' here means ML model."
    start = episode.correction.find("'")
    end = episode.correction.find("'", start + 1)
    if start == -1 or end == -1:
        raise ValueError(f"Could not locate quoted token in: {episode.correction!r}")
    return episode.correction[start + 1 : end]


def _sort_key_for_token(token: str, chapter: Chapter) -> int:
    return _CHAPTER_TOKENS[chapter].index(token)


def _collect_probes(
    episodes: list[Episode],
    category: Literal["transfer", "scope", "forgetting"],
) -> tuple[Probe, ...]:
    """Return all probes of a given category from a list of episodes."""
    return tuple(probe for ep in episodes for probe in ep.probes if probe.category == category)


def build_grouped_dataset(seed: int = 0) -> dict[Chapter, EpisodeGroup]:
    """Return benchmark episodes grouped into four curriculum chapters.

    The mapping is deterministic; `seed` is reserved for future variations in
    episode ordering and is stored in the output for log reproducibility.
    """
    dataset = list(_build_dataset())
    token_to_episode = {_token_from_episode(ep): ep for ep in dataset}

    if set(token_to_episode.keys()) != {
        token for tokens in _CHAPTER_TOKENS.values() for token in tokens
    }:
        raise ValueError("Dataset tokens do not match the curriculum taxonomy.")

    # Deterministic order per chapter; seed is applied to a stable shuffle so
    # that the exact curriculum can vary by seed while remaining reproducible.
    rng = random.Random(seed)

    grouped: dict[Chapter, EpisodeGroup] = {}
    for chapter, tokens in _CHAPTER_TOKENS.items():
        episodes = [token_to_episode[token] for token in tokens]
        # Reproducible, seed-aware shuffle of acquisition order.
        rng.shuffle(episodes)

        transfer = list(_collect_probes(episodes, "transfer"))
        scope = list(_collect_probes(episodes, "scope"))
        forgetting = list(_collect_probes(episodes, "forgetting"))

        # Keep probe batteries in a stable, deterministic order.
        grouped[chapter] = EpisodeGroup(
            acquisition_episodes=tuple(episodes),
            transfer_probes=tuple(transfer),
            scope_probes=tuple(scope),
            forgetting_probes=tuple(forgetting),
            chapter=chapter,
        )

    return grouped


def chapter_prompt_template(chapter: Chapter) -> str:
    """Return the canonical prompt template for a curriculum chapter."""
    templates: dict[Chapter, str] = {
        "vocabulary_re grounding": (
            "You are learning how this product uses ambiguous words. "
            "When a request contains <TOKEN>, interpret it according to the "
            "domain-specific correction you received, not the everyday meaning."
        ),
        "tool_use": (
            "You are learning how to identify which physical or software "
            "tool <TOKEN> refers to in a given context. Apply the corrected "
            "tool meaning after the user correction."
        ),
        "role_alignment": (
            "You are learning how service, branch, and table map to their "
            "corrected roles in this product. Choose the meaning that aligns "
            "with the role the user clarifies."
        ),
        "constraint_learning": (
            "You are learning execution constraints for working with cells, "
            "records, and runs. After correction, honor the restricted "
            "operational meaning of <TOKEN>."
        ),
    }
    return templates[chapter]


def chapter_skill_summary(chapter: Chapter) -> str:
    """Return the expected skill behavior for a chapter."""
    summaries: dict[Chapter, str] = {
        "vocabulary_re grounding": (
            "Map each ambiguous domain term to its product-specific meaning "
            "after one correction and maintain the common meaning when no "
            "product context applies."
        ),
        "tool_use": (
            "Disambiguate tool-like nouns by situation and apply the corrected "
            "tool meaning without overwriting unrelated senses."
        ),
        "role_alignment": (
            "Resolve role/service nouns according to the clarified context and "
            "avoid over-generalizing to other legitimate senses."
        ),
        "constraint_learning": (
            "Respect operational constraints on workspace actions and apply "
            "the corrected interpretation only within the matching domain."
        ),
    }
    return summaries[chapter]


def _chapter_order() -> tuple[Chapter, ...]:
    return (
        "vocabulary_re grounding",
        "tool_use",
        "role_alignment",
        "constraint_learning",
    )


def count_probes(group: EpisodeGroup) -> dict[str, int]:
    """Return probe counts for a group."""
    return {
        "transfer": len(group.transfer_probes),
        "scope": len(group.scope_probes),
        "forgetting": len(group.forgetting_probes),
        "total": len(group.transfer_probes)
        + len(group.scope_probes)
        + len(group.forgetting_probes),
    }


if __name__ == "__main__":
    groups = build_grouped_dataset(seed=0)
    level_num = 1
    total_probes = 0
    for chapter in _chapter_order():
        group = groups[chapter]
        counts = count_probes(group)
        tokens = [_token_from_episode(ep) for ep in group.acquisition_episodes]
        print(
            f"Level {level_num}: {chapter} "
            f"(tokens={', '.join(tokens)}) "
            f"transfer={counts['transfer']} "
            f"scope={counts['scope']} "
            f"forgetting={counts['forgetting']} "
            f"total={counts['total']}"
        )
        total_probes += counts["total"]
        level_num += 1
    print(f"Cumulative probe count across all levels: {total_probes}")
