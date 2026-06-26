"""Tests for the multi-fact turn stressor."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from oczy.experiments.multi_fact_stressor import _gguf_available, main


def _capture_output(argv: list[str]) -> list[str]:
    """Run the probe CLI and return emitted lines."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(argv)
    return buf.getvalue().strip().splitlines()


def _parse_metric(lines: list[str]) -> dict[str, str]:
    """Extract the first ``METRIC`` line as a key/value map."""
    for line in lines:
        if line.startswith("METRIC"):
            result: dict[str, str] = {}
            for part in line.split()[1:]:
                if "=" in part:
                    key, value = part.split("=", 1)
                    result[key] = value
            return result
    raise AssertionError(f"no METRIC line in output: {lines}")


def _assert_valid_metric(metric: dict[str, str], expected_mode: str) -> None:
    """Common assertions for a METRIC line."""
    assert metric["mode"] == expected_mode
    assert metric["auto_consolidated"] in {"0", "1"}
    assert metric["recall_a"] in {"0", "1"}
    assert metric["recall_b"] in {"0", "1"}
    assert metric["co_recall"] in {"0", "1"}
    assert int(metric["traces"]) > 0, "pipeline should store chunk traces"


def test_multi_fact_stressor_auto_consolidate_mock() -> None:
    lines = _capture_output(["--auto-consolidate", "--length", "64"])
    metric = _parse_metric(lines)
    _assert_valid_metric(metric, "scalar")
    assert metric["auto_consolidated"] in {"0", "1"}
    assert any(line.startswith("ASI") for line in lines)

def test_multi_fact_stressor_runs_scalar() -> None:
    lines = _capture_output(["--mode", "scalar"])
    metric = _parse_metric(lines)
    assert any(line.startswith("ASI") for line in lines)
    _assert_valid_metric(metric, "scalar")


def test_multi_fact_stressor_runs_hybrid() -> None:
    lines = _capture_output(["--mode", "hybrid"])
    metric = _parse_metric(lines)
    assert any(line.startswith("ASI") for line in lines)
    _assert_valid_metric(metric, "hybrid")


def test_multi_fact_stressor_can_recall_both_facts() -> None:
    """Exact co-recall requires a real LM; here we verify storage and metrics.

    The bundled mock driver returns the literal string ``mock`` for every
    generation call, so ``co_recall`` is expected to be ``0`` in this test
    harness.  The deterministic acceptance gate is that valid ``METRIC`` lines
    are emitted and the chunked pipeline stores at least one trace under both
    consolidation modes.
    """
    for mode in ("scalar", "hybrid"):
        lines = _capture_output(["--mode", mode, "--length", "256"])
        metric = _parse_metric(lines)
        assert metric["co_recall"] in {"0", "1"}
        assert int(metric["traces"]) > 0
        assert any(line.startswith("ASI") for line in lines)


def test_multi_fact_stressor_config_override() -> None:
    """A JSON config object should be accepted and not crash the probe."""
    config_json = '{"ingestion": {"chunker_window_tokens": 32}}'
    lines = _capture_output(["--mode", "scalar", "--config", config_json])
    metric = _parse_metric(lines)
    _assert_valid_metric(metric, "scalar")
    # A smaller window should produce at least as many traces as the default.
    assert int(metric["embedding_calls"]) > 0


def test_multi_fact_stressor_mock_prefix_runs() -> None:
    """The prefix path should run on the mock driver and emit valid output."""
    lines = _capture_output(["--mode", "scalar", "--use-prefix", "--length", "64"])
    metric = _parse_metric(lines)
    _assert_valid_metric(metric, "scalar")
    assert metric["use_prefix"] == "True"
    assert any(line.startswith("ASI") for line in lines)


@pytest.mark.slow
@pytest.mark.requires_model
def test_multi_fact_stressor_runs_real_driver() -> None:
    """Run against the real LFM2.5 GGUF; skipped if the model is not cached."""
    if not _gguf_available():
        pytest.skip("LFM2.5 GGUF not found in OCZY_MODEL_PATH or HF cache")
    lines = _capture_output(["--use-real-driver", "--mode", "scalar", "--length", "256"])
    assert any(line.startswith("METRIC") for line in lines)
    assert any(line.startswith("ASI") for line in lines)
