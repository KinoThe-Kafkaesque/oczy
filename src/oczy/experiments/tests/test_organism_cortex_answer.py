"""Tests for OrganismAgent cortex-LM answer delegation flag."""

from __future__ import annotations

from typing import Any

import pytest

from oczy.experiments.organism import OrganismAgent


class _MockCortexAgent:
    """Minimal CortexAgent stand-in that only implements answer()."""

    def answer(
        self,
        request: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        metabolize: bool = False,
    ) -> dict[str, Any]:
        return {"answer": "cortex reply"}


def test_organism_delegates_to_cortex_agent() -> None:
    mock_cortex = _MockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_lm_answer": True, "cortex_agent": mock_cortex}
    )
    assert organism.answer("x") == "cortex reply"


def test_organism_legacy_answer_path_still_runs() -> None:
    organism = OrganismAgent({})
    organism.plastic_cortex.answer = lambda request: "legacy plastic reply"
    assert organism.answer("x") == "legacy plastic reply"


def test_organism_missing_cortex_agent_fallback_warns() -> None:
    with pytest.warns(UserWarning, match="cortex_agent"):
        organism = OrganismAgent({"use_cortex_lm_answer": True})
    assert organism.cortex_agent is None
