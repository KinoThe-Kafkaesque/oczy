"""Smoke test: multi-seed run_curriculum produces mean ± std report."""

from __future__ import annotations

import io
import sys
import textwrap

from oczy.experiments.organism_curriculum.run_curriculum import main


def test_multi_seed_produces_summary() -> None:
    """run_curriculum --seeds 2 produces a multi-seed summary with mean ± CI."""
    saved_stdout = sys.stdout
    try:
        buf = io.StringIO()
        sys.stdout = buf
        rc = main(
            [
                "--use-cortex-shim",
                "--seeds",
                "2",
                "--stages",
                "stage_0_grounding",
            ]
        )
        output = buf.getvalue()
    finally:
        sys.stdout = saved_stdout

    assert rc == 0, f"expected exit 0, got {rc}"
    assert "Running 2 seeds" in output, "missing 'Running 2 seeds' banner"
    assert "Multi-seed summary" in output, "missing multi-seed summary header"
    assert "±" in output, f"missing ± (mean ± CI format) in:\n{output}"
    assert "(n=2)" in output, f"missing '(n=2)' in:\n{output}"
    assert "Report written to:" in output, "missing report path"


def test_single_seed_warns() -> None:
    """run_curriculum --seeds 1 (default) emits a single-seed point estimate warning."""
    saved_stdout = sys.stdout
    try:
        buf = io.StringIO()
        sys.stdout = buf
        rc = main(
            [
                "--use-cortex-shim",
                "--stages",
                "stage_0_grounding",
            ]
        )
        output = buf.getvalue()
    finally:
        sys.stdout = saved_stdout

    assert rc == 0, f"expected exit 0, got {rc}"
    assert (
        "WARNING: single-seed point estimate — not reportable" in output
    ), f"missing single-seed warning in:\n{textwrap.shorten(output, 400)}"
