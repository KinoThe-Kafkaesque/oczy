"""Tests for the conversation world model probe (Experiment 07)."""

from __future__ import annotations

import numpy as np
import pytest

import src.oczy.experiments.conversation_world_model as cwm


def test_module_imports_without_llama() -> None:
    """Importing the module should not pull llama_cpp."""
    # Test ordering makes this fragile; official check is a subprocess import.
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "import sys; import src.oczy.experiments.conversation_world_model; "
            "print('llama_cpp' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "False" in result.stdout




def test_auc_perfect_separation() -> None:
    assert cwm._auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_auc_chance_when_no_separation() -> None:
    assert cwm._auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == pytest.approx(0.5)


def test_mock_driver_runs_and_emits_metric(capsys) -> None:
    assert cwm.main(["--driver", "mock"]) == 0
    out = capsys.readouterr().out
    assert "METRIC marker_free_uptake_gap=" in out
    assert "ASI critic_auc_delta=" in out
    assert "ASI accept_pred_auc_hidden=" in out
    assert "ASI marker_free_uptake_gap=" in out


def test_default_driver_is_mock(capsys) -> None:
    assert cwm.main([]) == 0
    out = capsys.readouterr().out
    assert "METRIC marker_free_uptake_gap=" in out


def test_marker_free_uptake_gap_non_negative() -> None:
    """Lexical gate misses marker-free corrections by construction."""
    gap = cwm._marker_free_uptake()
    assert gap >= 0.0


def test_mock_embedding_deterministic() -> None:
    a = cwm._mock_embedding("hello", n_embd=16)
    b = cwm._mock_embedding("hello", n_embd=16)
    assert np.allclose(a, b)


def test_real_driver_graceful_failure(monkeypatch, capsys) -> None:
    def _fail():
        raise RuntimeError("no gguf")

    monkeypatch.setattr(cwm, "_run_real_driver", _fail)
    assert cwm.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "METRIC marker_free_uptake_gap=nan" in out


def test_real_driver_graceful_none(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cwm, "_run_real_driver", lambda: None)
    assert cwm.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "METRIC marker_free_uptake_gap=nan" in out
