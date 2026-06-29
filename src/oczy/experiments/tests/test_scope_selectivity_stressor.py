"""Tests for the scope-selectivity stressor (Experiment 04)."""

from __future__ import annotations

import pytest

import src.oczy.experiments.scope_selectivity_stressor as sss


def test_module_imports_without_llama() -> None:
    """Importing the module should not pull llama_cpp."""
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "import sys; import src.oczy.experiments.scope_selectivity_stressor; "
            "print('llama_cpp' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "False" in result.stdout


def test_cosine_basic() -> None:
    import numpy as np

    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    assert sss._cosine(a, b) == pytest.approx(1.0)
    assert sss._cosine(a, c) == pytest.approx(0.0)


def test_slot_store_allocate_and_retrieve() -> None:
    import numpy as np

    keys = []
    warm = []
    k1 = np.array([1.0, 0.0])
    w1 = np.array([0.5, 0.5])
    sss._slot_write(keys, warm, k1, w1)
    assert len(keys) == 1
    retrieved = sss._slot_retrieve(keys, warm, k1)
    assert np.allclose(retrieved, w1)


def test_mock_driver_emits_metric(capsys) -> None:
    assert sss.main(["--driver", "mock"]) == 0
    out = capsys.readouterr().out
    assert "METRIC scope_selectivity_index=" in out


def test_default_driver_is_mock(capsys) -> None:
    assert sss.main([]) == 0
    out = capsys.readouterr().out
    assert "METRIC scope_selectivity_index=" in out


def test_real_driver_graceful_failure(monkeypatch, capsys) -> None:
    def _fail():
        raise RuntimeError("no gguf")

    monkeypatch.setattr(sss, "_run_real_driver", _fail)
    assert sss.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "METRIC scope_selectivity_index=nan" in out


def test_real_driver_graceful_none(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sss, "_run_real_driver", lambda: None)
    assert sss.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "METRIC scope_selectivity_index=nan" in out
