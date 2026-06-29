"""Tests for the Layer-L hidden extraction probe (Experiment 03)."""

from __future__ import annotations

import numpy as np
import pytest

import src.oczy.experiments.layer_l_probe as layer_l_probe


@pytest.fixture
def sample_warm() -> dict[str, list[np.ndarray]]:
    """Two concepts with three 4-D warm vectors each."""
    rng = np.random.RandomState(42)
    return {
        "concept_a": [rng.standard_normal(4), rng.standard_normal(4)],
        "concept_b": [rng.standard_normal(4), rng.standard_normal(4)],
    }


def test_module_imports_without_heavy_deps() -> None:
    """Importing the module in a fresh interpreter should not pull torch/transformers/llama_cpp."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.oczy.experiments.layer_l_probe as m; "
            "heavy = {'torch', 'transformers', 'llama_cpp'}; "
            "loaded = {name.split('.')[0] for name in sys.modules if name.split('.')[0] in heavy}; "
            "print(list(loaded))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_mock_hidden_vectors_shapes_and_norms() -> None:
    vec = layer_l_probe._mock_hidden_vectors(0)
    assert vec.shape == (layer_l_probe._D_EMBD,)
    assert vec.dtype == np.float32
    assert np.linalg.norm(vec) > 0

    v9 = layer_l_probe._mock_hidden_vectors(9)
    v13 = layer_l_probe._mock_hidden_vectors(13)
    # Different layers should receive different deterministic vectors from a
    # fresh RNG, and a fixed RNG produces the same vector each call.
    assert not np.allclose(v9, v13)


def test_silhouette_function_monotonic(sample_warm) -> None:
    """Silhouette should be larger when intra-concept vectors are more similar."""
    base = layer_l_probe._silhouette(sample_warm)
    assert base is not None

    tighter = {
        "concept_a": [sample_warm["concept_a"][0], sample_warm["concept_a"][0]],
        "concept_b": [sample_warm["concept_b"][0], sample_warm["concept_b"][0]],
    }
    tight = layer_l_probe._silhouette(tighter)
    assert tight is not None
    assert tight >= base


def test_silhouette_none_on_too_few_pairs() -> None:
    """A single concept yields no inter-concept pairs."""
    rng = np.random.RandomState(0)
    assert layer_l_probe._silhouette({"only": [rng.standard_normal(4)]}) is None


def test_compute_gap_with_final() -> None:
    s = {"L9_last": 0.5, "L13_last": 0.3, "final_meanpool": 0.4}
    r = layer_l_probe._compute_gap(s)
    assert r["gap"] == pytest.approx(0.1)
    assert r["max_mid"] == pytest.approx(0.5)


def test_compute_gap_without_final() -> None:
    s = {"L9_last": 0.5}
    r = layer_l_probe._compute_gap(s)
    assert r["gap"] == 0.0
    assert r["final"] is None


def test_compute_gap_empty() -> None:
    r = layer_l_probe._compute_gap({})
    assert r["gap"] == 0.0


def test_mock_driver_runs_and_prints_metric(capsys) -> None:
    assert layer_l_probe.main(["--driver", "mock"]) == 0
    out = capsys.readouterr().out
    assert "METRIC layer_l_silhouette_gap=" in out
    assert "ASI warm_sep_silhouette_" in out


def test_run_mock_returns_silhouettes() -> None:
    sils = layer_l_probe._mock_probe()
    assert set(sils).issuperset(
        {"R_random", "last_L9", "last_L13", "last_L15", "maxpool_L14", "mean_L14"}
    )


def test_real_driver_falls_back_gracefully(monkeypatch, capsys) -> None:
    def _raise():
        raise RuntimeError("no model")

    monkeypatch.setattr(layer_l_probe, "_hf_probe", _raise)
    assert layer_l_probe.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "METRIC layer_l_silhouette_gap=" in out


def test_real_driver_falls_back_on_none(monkeypatch, capsys) -> None:
    monkeypatch.setattr(layer_l_probe, "_hf_probe", lambda: None)
    assert layer_l_probe.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "METRIC layer_l_silhouette_gap=" in out


def test_main_default_driver_is_mock(capsys) -> None:
    assert layer_l_probe.main([]) == 0
    out = capsys.readouterr().out
    assert "METRIC layer_l_silhouette_gap=" in out
