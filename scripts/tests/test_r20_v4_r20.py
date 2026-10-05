"""Fail-closed tests for the v4-r20 materializer and the G2 successor screen.

The materializer is the gate that decides which base the successor may be built
from, so a wrong base must be refused as loudly as the v2-lineage materializer
refused the v3 lineage.  The screen's identity checks must reject a mismatched
instrument rather than silently scoring against the wrong tasks.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from materialize_r20_v4_r20 import EXIT_WRONG_BASE, main  # noqa: E402
from probe_r20_v4_r20_oracle import looks_truncated, with_teaching_context  # noqa: E402

from oczy.experiments.meta_cortex.contracts import DialogueMessage  # noqa: E402

V3_PUBLIC_ROOT = REPO / "experiments" / "r20-taskgen-v3-dev" / "instrument" / "public"
G2_ARTIFACT = REPO / "experiments_logs/artifacts/2026-10-05_r20_g2_v4_r20_oracle_screen/results.json"

pytestmark = pytest.mark.skipif(
    not V3_PUBLIC_ROOT.is_dir(),
    reason="the pinned meta_cortex/v3 public DEV view is not present in this checkout",
)


def test_materializer_writes_a_self_hashed_manifest(tmp_path: Path) -> None:
    out = tmp_path / "materialization"
    assert main(["--public-root", str(V3_PUBLIC_ROOT), "--output", str(out)]) == 0
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["instrument_id"] == "meta_cortex/v4-r20"
    assert manifest["max_new_tokens"] == 128
    assert manifest["base_max_new_tokens"] == 32
    assert manifest["base_definition_sha256"] == (
        "ab99c17358a3af3efb577563b8cd33639519084fa88648c3bc1e05dce86d4bee"
    )
    assert manifest["base_dev_view_sha256"] == (
        "593090c405758617ae9f750cf9c1075e999c92f8c7b677affcb04c3855b3ca26"
    )
    # The amendment surface is exactly the approved one.
    diff = manifest["amendment_diff"]
    assert diff["non_probe_task_fields_changed"] == 0
    assert diff["probe_payloads_changed_outside_amendment_b"] == 0
    assert diff["system_messages_added"] == 933
    assert diff["oracle_headers_rewritten"] == 35
    # Re-verification succeeds.
    assert main(
        ["--public-root", str(V3_PUBLIC_ROOT), "--output", str(out), "--verify-only"]
    ) == 0


def test_materializer_refuses_a_wrong_base(tmp_path: Path) -> None:
    """A directory that is not the pinned v3 public view exits 3, not 1."""
    empty = tmp_path / "not-an-instrument"
    empty.mkdir()
    assert (
        main(["--public-root", str(empty), "--output", str(tmp_path / "out")])
        == EXIT_WRONG_BASE
    )


def test_materializer_refuses_a_tampered_base(tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(V3_PUBLIC_ROOT.parent, copy)
    view_path = copy / "public" / "DEV_VIEW.json"
    data = json.loads(view_path.read_text())
    data["instrument_id"] = "meta_cortex/v2"
    view_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    assert (
        main(["--public-root", str(copy / "public"), "--output", str(tmp_path / "out")])
        == EXIT_WRONG_BASE
    )


def test_materializer_refuses_to_write_inside_the_frozen_instrument(tmp_path: Path) -> None:
    """The output may not live inside the base instrument tree."""
    assert (
        main(["--public-root", str(V3_PUBLIC_ROOT), "--output", str(V3_PUBLIC_ROOT.parent / "out")])
        != 0
    )


def test_materializer_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "materialization"
    assert main(["--public-root", str(V3_PUBLIC_ROOT), "--output", str(out)]) == 0
    assert main(["--public-root", str(V3_PUBLIC_ROOT), "--output", str(out)]) != 0


def test_v2_lineage_materializer_still_refuses_the_v3_base() -> None:
    """The original materializer's fail-closed behaviour is unchanged."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "materialize_dev_prompt_repair.py"),
            "--public-root",
            str(V3_PUBLIC_ROOT),
            "--output",
            "/tmp/oczy-should-never-exist-v4r20",
            "--version",
            "v3",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Not the approved v2 public DEV instrument" in result.stderr


def test_teaching_context_prepends_transcript_and_keeps_system_first() -> None:
    transcript = (DialogueMessage("user", "teaching"),)
    probe = (DialogueMessage("system", "format"), DialogueMessage("user", "probe"))
    out = with_teaching_context(probe, transcript)
    assert out[0].role == "system"
    assert out[1].content == "teaching"
    assert out[-1].content == "probe"

    no_system = (DialogueMessage("user", "probe"),)
    out2 = with_teaching_context(no_system, transcript)
    assert [m.content for m in out2] == ["teaching", "probe"]


def test_truncation_heuristic_is_descriptive_only() -> None:
    assert looks_truncated("that \"azure")
    assert not looks_truncated("q2")
    assert not looks_truncated('"grim".')
    assert looks_truncated("")


@pytest.mark.skipif(not G2_ARTIFACT.is_file(), reason="the G2 successor artifact is absent")
class TestG2SuccessorArtifact:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        return json.loads(G2_ARTIFACT.read_text())

    def test_screen_recorded_a_gate_verdict_not_a_crash(self, result):
        assert result["gate"] == "G2"
        assert result["optimizer_steps"] == 0
        assert result["training_run"] is False

    def test_screen_never_touched_forbidden_surfaces(self, result):
        assert result["meta_test_accessed"] is False
        assert result["calibration_accessed"] is False
        assert result["sealed_accessed"] is False
        assert result["thresholds_selected"] is False

    def test_organ_identity_matched_the_frozen_binding(self, result):
        assert result["organ_hash_matches_frozen_binding"] is True
        assert result["organ_hash_before"] == result["organ_hash_after"]
        assert result["organ_hash_before"] == (
            "a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea"
        )

    def test_screen_ran_against_the_frozen_v4_r20_instrument(self, result):
        assert result["instrument_id"] == "meta_cortex/v4-r20"
        assert result["taskgen_schema"] == "oczy/meta-cortex/taskgen/v2-dev"
        assert result["base_instrument_id"] == "meta_cortex/v3"
        assert result["max_new_tokens"] == 128
        assert result["base_max_new_tokens"] == 32

    def test_gate_verdict_is_the_oracle_condition_only(self, result):
        """no_context and teaching_context are reported but never decide the gate."""
        scores = result["scores"]
        assert scores["oracle_context"]["total"] == 15
        assert scores["no_context"]["total"] == 32
        assert scores["teaching_context"]["total"] == 32
        assert result["gate_passed"] == (scores["oracle_context"]["correct"] > 0)

    def test_every_row_is_scored_by_the_exact_scorer_not_by_substring(self, result):
        from oczy.experiments.meta_cortex.calibration import FrozenScorer

        scorer = FrozenScorer()
        for row in result["rows"]:
            assert row["correct"] == scorer.score_response(row["expected"], row["generated"])
        # The prose-vs-exact distinction is exercised: at least one row contains
        # its expected token wrapped in prose and is still marked incorrect.
        # (An exact-answer row trivially contains its own token and is correct,
        # so the check is scoped to rows that are not exact answers.)
        prose_wrapped = [
            r
            for r in result["rows"]
            if r["expected"] in r["generated"] and r["generated"] != r["expected"]
        ]
        assert prose_wrapped, "expected at least one prose-wrapped correct token"
        assert not any(r["correct"] for r in prose_wrapped)
