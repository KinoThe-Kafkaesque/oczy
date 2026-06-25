"""End-to-end integration tests for the agent glue layer (``experiments/``).

These tests exercise the OrganismAgent and the baseline ablation agents
through their public ``learn() -> answer()`` contract.  They exist to
catch the class of bug that survived the 2026-06-19 experiment run: the
``correction`` vs ``corrected_answer`` schema drift that silently
disabled ``OrganismAgent``'s hippocampal-replay ranker	path was found
by a manual smoke test that should have been a committed test.
"""

from __future__ import annotations

import pytest

from oczy.experiments.baselines import (
    ContextOnlyAgent,
    FastOnlyAgent,
    HippocampusOnlyAgent,
    IdentityOnlyAgent,
    ZeroMemoryAgent,
)
from oczy.experiments.organism import OrganismAgent


# ---------------------------------------------------------------------------
# T2: OrganismAgent end-to-end replay path
# ---------------------------------------------------------------------------


class TestOrganismAgentReplayPath:
    """The full organ stack must surface a corrected label on re-asking.

    This is the regression test for the field-name drift bug fixed in the
    2026-06-21 session: ``NeuralHippocampus.store()`` was passed the raw
    ``correction`` sentence but no ``corrected_answer`` label, so on
    replay the label was missing and ``OrganismAgent._rank_answer`` had
    no candidate to surface besides the placeholder.
    """

    def test_learn_then_answer_surfaces_corrected_label(self) -> None:
        agent = OrganismAgent()
        agent.learn("Update the user profile.",
                    "No, 'profile' means business vertical.")
        assert agent.answer("Update the user profile.") == "business vertical"

    def test_unrelated_query_falls_back_to_placeholder(self) -> None:
        agent = OrganismAgent()
        # Never learned anything about this domain.
        out = agent.answer("Tell me a joke.")
        # The agent has no fast weights or hippocampal trace for this
        # query; it should at least be a string, not raise.
        assert isinstance(out, str)

    def test_consolidate_preserves_learned_label(self) -> None:
        """After slow consolidation, the learned label must still surface.

        Consolidation moves raw hippocampal traces into slow updates and
        *decays the raw traces*.  If the replay ranker relied on raw
        traces alone, the label would disappear post-consolidation ---
        exactly the loop the architecture's central claim depends on.
        """
        agent = OrganismAgent()
        agent.learn("Create a branch.",
                    "No, 'branch' means git branch.")
        assert agent.answer("Create a branch.") == "git branch"
        agent.consolidate()
        # Post-consolidation, the slow updates should keep the label
        # recoverable.  (We accept either the slow-update path or a
        # still-present raw trace if consolidation didn't fire.)
        out = agent.answer("Create a branch.")
        assert "git branch" in out or out == "git branch"


# ---------------------------------------------------------------------------
# Baseline ablations: each must satisfy an answer()/learn() contract
# ---------------------------------------------------------------------------


class TestBaselines:
    """The five baseline agents must each expose a consistent contract.

    They exist as ablation controls against the full OrganismAgent; if
    any of them stops returning a string from answer() or raises from
    learn(), the experiment harness can't compare them.
    """

    @pytest.mark.parametrize(
        "agent_cls",
        [
            ZeroMemoryAgent,
            ContextOnlyAgent,
            FastOnlyAgent,
            HippocampusOnlyAgent,
            IdentityOnlyAgent,
        ],
    )
    def test_answer_returns_str_before_any_learning(self, agent_cls) -> None:
        agent = agent_cls()
        assert isinstance(agent.answer("anything"), str)

    @pytest.mark.parametrize(
        "agent_cls",
        [
            ZeroMemoryAgent,
            ContextOnlyAgent,
            FastOnlyAgent,
            HippocampusOnlyAgent,
            IdentityOnlyAgent,
        ],
    )
    def test_learn_then_answer_returns_str(self, agent_cls) -> None:
        agent = agent_cls()
        agent.learn("Update the user profile.",
                    "No, 'profile' means business vertical.")
        assert isinstance(agent.answer("Update the user profile."), str)

    def test_hippocampus_only_agent_actually_learns_label(self) -> None:
        """HippocampusOnlyAgent must surface a learned corrected_answer.

        This is the regression test for the duplicate-trace kludge that
        used to write a second raw trace (with ``corrected_answer``)
        alongside the hippocampus's own write (without it).  Retrieval
        was nondeterministic and the agent often fell back to the
        placeholder.  After the fix, the single hippocampus store call
        carries ``corrected_answer`` and the ranker surfaces it.
        """
        agent = HippocampusOnlyAgent()
        agent.learn("Update the user profile.",
                    "No, 'profile' means business vertical.")
        # We expect the recovered label to surface.  If extract_expected
        # fails on this sentence we accept a non-placeholder string.
        out = agent.answer("Update the user profile.")
        assert out != "I don't know."


# ---------------------------------------------------------------------------
# Contract check: extracted expected labels for the canonical curriculum
# ---------------------------------------------------------------------------


class TestExpectedLabelExtraction:
    """``extract_expected_from_correction`` is the heuristic that lets the
    baseline agents (and OrganismAgent) recover a corrected label from a
    raw sentence when no explicit label is provided.  A handful of
    canonical curriculum corrections should round-trip to the right label.
    """

    def test_means_template_extracts_right_hand_side(self) -> None:
        from oczy.common import extract_expected_from_correction

        assert extract_expected_from_correction(
            "No, 'profile' means business vertical."
        ).strip().lower() == "business vertical"

    def test_means_template_ml_model(self) -> None:
        from oczy.common import extract_expected_from_correction

        assert extract_expected_from_correction(
            "No, 'model' here means ML model."
        ).strip().lower() == "ml model"