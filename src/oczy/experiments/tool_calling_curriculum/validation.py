"""Schema validation for the code-backed Pi curriculum."""

from __future__ import annotations

from .dataset import ToolStage


def validate_tool_curriculum(stages: tuple[ToolStage, ...]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    expected_counts = (8, 12, 8, 6, 8, 3)
    if tuple(len(stage.episodes) for stage in stages) != expected_counts:
        errors.append(
            "stage episode counts must be 8/12/8/6/8/3, got "
            + "/".join(str(len(stage.episodes)) for stage in stages)
        )
    for stage in stages:
        if not stage.episodes:
            errors.append(f"{stage.id}: empty stage")
        for episode in stage.episodes:
            if episode.id in seen:
                errors.append(f"duplicate episode id: {episode.id}")
            seen.add(episode.id)
            if not episode.request.strip():
                errors.append(f"{stage.id}/{episode.id}: empty request")
            if not episode.expected_tools and not episode.expected_answer_terms:
                errors.append(f"{stage.id}/{episode.id}: no observable expectation")
    return errors
