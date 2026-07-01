"""Tests for the metabolism loop closure probe (Experiment 05)."""

from __future__ import annotations

import numpy as np
import pytest

import oczy.experiments.metabolism_loop as ml


def test_module_imports_without_llama() -> None:
    """Importing the module should not pull llama_cpp."""
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "import sys; import oczy.experiments.metabolism_loop; "
            "print('llama_cpp' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "False" in result.stdout


def test_domain_uptake_range() -> None:
    assert ml._domain_uptake("commercial business") == pytest.approx(2 / 6)
    assert ml._domain_uptake("") == 0.0


def test_compounding_index_perfectly_additive() -> None:
    """Repeated identical deltas give compounding_index near 1."""
    states = [np.zeros(4)]
    for _ in range(5):
        states.append(states[-1] + np.array([0.1, 0.0, 0.0, 0.0]))

    idx = ml._compounding_index(states)
    assert idx == pytest.approx(1.0, abs=1e-6)


def test_compounding_index_cancellation() -> None:
    """Opposing deltas give a low compounding_index."""
    states = [np.zeros(4)]
    for _ in range(4):
        states.append(states[-1] + np.array([0.1, 0.0, 0.0, 0.0]))
        states.append(states[-1] + np.array([-0.1, 0.0, 0.0, 0.0]))
    idx = ml._compounding_index(states)
    assert idx < 0.2


def test_mock_driver_runs_without_crash(capsys) -> None:
    assert ml.main(["--driver", "mock", "--corrections", "2"]) == 0
    out = capsys.readouterr().out
    assert "METRIC metabolism_drift_delta=" in out
    assert "ASI compounding_index=" in out


def test_default_driver_is_mock(capsys) -> None:
    assert ml.main(["--corrections", "2"]) == 0
    out = capsys.readouterr().out
    assert "METRIC metabolism_drift_delta=" in out


def test_real_driver_graceful_failure(monkeypatch, capsys) -> None:
    def _fail():
        raise RuntimeError("no gguf")

    monkeypatch.setattr(ml, "_run_real_driver", _fail)
    assert ml.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "METRIC metabolism_drift_delta=nan" in out


def test_real_driver_graceful_none(monkeypatch, capsys) -> None:
    monkeypatch.setattr(ml, "_run_real_driver", lambda k=20: None)
    assert ml.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
