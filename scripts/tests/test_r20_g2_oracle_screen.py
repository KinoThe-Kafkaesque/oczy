"""Fail-closed tests for the G2 oracle screen's identity and gate logic.

The screen is the gate that decides whether the sequence continues, so its
identity checks must reject a mismatched instrument rather than silently scoring
against the wrong tasks.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "experiments_logs/artifacts/2026-10-05_r20_g2_oracle_screen/results.json"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from probe_r20_v3_oracle import with_teaching_context  # noqa: E402

from oczy.experiments.meta_cortex.contracts import DialogueMessage  # noqa: E402


@pytest.fixture(scope="module")
def result():
    return json.loads(ARTIFACT.read_text())


def test_screen_recorded_a_gate_failure_not_a_crash(result):
    assert result["gate"] == "G2"
    assert result["gate_passed"] is False
    # A failed gate still exits 0: it is a result, not an execution error.
    assert result["optimizer_steps"] == 0
    assert result["training_run"] is False


def test_screen_never_touched_forbidden_surfaces(result):
    assert result["meta_test_accessed"] is False
    assert result["calibration_accessed"] is False
    assert result["sealed_accessed"] is False
    assert result["thresholds_selected"] is False


def test_organ_identity_matched_the_frozen_binding(result):
    assert result["organ_hash_matches_frozen_binding"] is True
    assert result["organ_hash_before"] == result["organ_hash_after"]
    assert result["organ_hash_before"] == (
        "a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea"
    )


def test_screen_ran_against_the_frozen_v3_instrument(result):
    assert result["instrument_id"] == "meta_cortex/v3"
    assert result["taskgen_schema"] == "oczy/meta-cortex/taskgen/v2-dev"
    assert result["dev_view_sha256"] == (
        "593090c405758617ae9f750cf9c1075e999c92f8c7b677affcb04c3855b3ca26"
    )
    assert result["definition_sha256"] == (
        "ab99c17358a3af3efb577563b8cd33639519084fa88648c3bc1e05dce86d4bee"
    )


def test_gate_verdict_is_the_oracle_condition_only(result):
    """no_context and teaching_context are reported but never decide the gate."""
    scores = result["scores"]
    assert scores["oracle_context"]["correct"] == 0
    assert scores["oracle_context"]["total"] == 15
    assert scores["teaching_context"]["correct"] == 4
    assert scores["no_context"]["correct"] == 0
    assert result["gate_passed"] is False


def test_every_row_is_scored_by_the_exact_scorer_not_by_substring(result):
    """A row is correct only on exact normalized equality."""
    from oczy.experiments.meta_cortex.calibration import FrozenScorer

    scorer = FrozenScorer()
    for row in result["rows"]:
        assert row["correct"] == scorer.score_response(
            row["expected"], row["generated"]
        )
    # Sanity: at least one generated string *contains* its expected token and is
    # still marked incorrect, which is exactly the prose-vs-exact distinction.
    containing = [
        r
        for r in result["rows"]
        if r["condition"] == "oracle_context" and r["expected"] in r["generated"]
    ]
    assert containing, "expected at least one prose-wrapped correct token"
    assert not any(r["correct"] for r in containing)


def test_retrieval_bar_was_reported_as_a_bar_not_a_win(result):
    """The 4 correct rows are all retrieval, so no cortex claim is implied."""
    correct = [r for r in result["rows"] if r["correct"]]
    assert correct
    assert {r["condition"] for r in correct} == {"teaching_context"}


def test_teaching_context_prepends_transcript_and_keeps_system_first():
    transcript = (DialogueMessage("user", "teaching"),)
    probe = (DialogueMessage("system", "format"), DialogueMessage("user", "probe"))
    out = with_teaching_context(probe, transcript)
    assert out[0].role == "system"
    assert out[1].content == "teaching"
    assert out[-1].content == "probe"

    no_system = (DialogueMessage("user", "probe"),)
    out2 = with_teaching_context(no_system, transcript)
    assert [m.content for m in out2] == ["teaching", "probe"]
