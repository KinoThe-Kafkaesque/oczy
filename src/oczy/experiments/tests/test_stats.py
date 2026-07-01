"""Unit tests for :mod:`oczy.common.stats`."""

from __future__ import annotations

import math

import pytest

from oczy.common import format_row, run_seeded, summarize


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_empty() -> None:
    s = summarize([])
    assert s["n"] == 0
    for key in ("mean", "std", "min", "max", "ci95_half"):
        assert math.isnan(s[key]), f"{key} should be nan for empty input"


def test_summarize_single() -> None:
    s = summarize([5.0])
    assert s["n"] == 1
    assert s["mean"] == pytest.approx(5.0)
    assert s["std"] == pytest.approx(0.0)
    assert s["min"] == pytest.approx(5.0)
    assert s["max"] == pytest.approx(5.0)
    assert math.isnan(s["ci95_half"])


def test_summarize_two() -> None:
    s = summarize([1.0, 3.0])
    assert s["n"] == 2
    assert s["mean"] == pytest.approx(2.0)
    assert s["std"] == pytest.approx(1.41421356, rel=1e-3)
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(3.0)
    # t-critical for df=1 is 12.706; sem = std / sqrt(2) = 1.0
    assert s["ci95_half"] == pytest.approx(12.706, rel=1e-3)


def test_summarize_three() -> None:
    s = summarize([1.0, 2.0, 3.0])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(2.0)
    assert s["std"] == pytest.approx(1.0)
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(3.0)
    # t-critical for df=2 is 4.303; sem = 1.0 / sqrt(3); ci = 4.303 / sqrt(3)
    assert s["ci95_half"] == pytest.approx(4.303 / math.sqrt(3.0), rel=1e-3)
    assert s["ci95_half"] == pytest.approx(2.484, rel=1e-3)


# ---------------------------------------------------------------------------
# run_seeded
# ---------------------------------------------------------------------------

def test_run_seeded_scalar() -> None:
    def fn(seed: int) -> float:
        return float(seed)  # seeds 1, 2, 3 -> values 1.0, 2.0, 3.0

    out = run_seeded(fn, [1, 2, 3])
    assert list(out.keys()) == ["_all"]
    s = out["_all"]
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(2.0)
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(3.0)
    assert s["std"] == pytest.approx(1.0)


def test_run_seeded_dict() -> None:
    def fn(seed: int) -> dict[str, float]:
        return {"acc": float(seed) / 10.0, "loss": float(seed)}

    out = run_seeded(fn, [1, 2, 3])
    assert set(out.keys()) == {"acc", "loss"}
    assert out["acc"]["n"] == 3
    assert out["acc"]["mean"] == pytest.approx(0.2)  # (0.1+0.2+0.3)/3
    assert out["loss"]["mean"] == pytest.approx(2.0)
    assert out["loss"]["min"] == pytest.approx(1.0)
    assert out["loss"]["max"] == pytest.approx(3.0)


def test_run_seeded_empty_seeds() -> None:
    def fn(seed: int) -> float:
        raise AssertionError("fn should not be called with no seeds")

    assert run_seeded(fn, []) == {}


# ---------------------------------------------------------------------------
# format_row
# ---------------------------------------------------------------------------

def test_format_row_normal() -> None:
    row = format_row("acc", {"n": 5, "mean": 0.62, "ci95_half": 0.08})
    assert row == "acc: 0.6200 ± 0.0800 (n=5)"


def test_format_row_single() -> None:
    row = format_row("acc", {"n": 1, "mean": 0.5, "ci95_half": float("nan")})
    assert row == "acc: 0.5000 (n=1)"


def test_format_row_zero() -> None:
    row = format_row("acc", {"n": 0, "mean": float("nan")})
    assert row == "acc: nan (n=0)"
