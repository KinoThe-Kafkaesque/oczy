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



# ---------------------------------------------------------------------------
# S2.3 magnitude-controlled drift metric
# ---------------------------------------------------------------------------


def test_cold_norms_returns_list_of_floats() -> None:
    states = [np.zeros(4), np.ones(4) * 2, np.ones(4) * 3]
    norms = ml._cold_norms(states)
    assert len(norms) == 3
    assert all(isinstance(n, float) for n in norms)
    assert norms[0] == pytest.approx(0.0)
    assert norms[1] == pytest.approx(4.0)  # sqrt(4*4) = 4
    assert norms[2] == pytest.approx(6.0)  # sqrt(4*9) = 6


def test_domain_probe_with_cvec_no_clamp() -> None:
    agent = ml._build_mock_agent()
    ml._svd_warmup(agent, [ml._CORRECTION])
    upright = ml._domain_probe(agent)
    clamped = ml._domain_probe_with_cvec(agent, cvec_norm_clamp=None)
    assert upright == pytest.approx(clamped)


def test_domain_probe_with_cvec_with_clamp() -> None:
    agent = ml._build_mock_agent()
    ml._svd_warmup(agent, [ml._CORRECTION])
    result = ml._domain_probe_with_cvec(agent, cvec_norm_clamp=0.01)
    assert 0.0 <= result <= 1.0


def test_control_probe_returns_float() -> None:
    agent = ml._build_mock_agent()
    result = ml._control_probe(agent)
    assert 0.0 <= result <= 1.0


def test_control_uptake_range() -> None:
    assert ml._control_uptake("document system") == pytest.approx(2 / 6)
    assert ml._control_uptake("") == 0.0
    assert ml._control_uptake("commercial business") == 0.0  # domain words, not control


def test_drift_metric_triple_mock() -> None:
    agent = ml._build_mock_agent()
    ml._svd_warmup(agent, [ml._CORRECTION])
    ml._compounding_loop(agent, [ml._CORRECTION], k=2, batch_size=1)
    zero = ml._build_mock_agent()
    triple = ml._drift_metric_triple(agent, zero, clamp_norm=1.0, use_logits=False)
    for key in ("delta_target", "delta_control", "delta_target_clamped"):
        assert key in triple
        assert isinstance(triple[key], float)
    # Mock driver is semantically null: deltas should be near zero
    assert abs(triple["delta_target"]) <= 1.0
    assert abs(triple["delta_control"]) <= 1.0


def test_compounding_loop_explicit_checkpoints() -> None:
    agent = ml._build_mock_agent()
    ml._svd_warmup(agent, [ml._CORRECTION])
    chk = {0, 2, 5}
    comp = ml._compounding_loop(agent, [ml._CORRECTION], k=5, batch_size=1, checkpoints=chk)
    assert comp["checkpoint_indices"] == [0, 2, 5]


def test_ablation_flag_parsed(capsys) -> None:
    rc = ml.main(["--driver", "mock", "--ablation"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ABLATION" in out


def test_mock_ablation_prints_five_conditions(capsys) -> None:
    rc = ml.main(["--driver", "mock", "--ablation", "--ablation-seeds", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    # Should have lines for each condition at each checkpoint
    cond1_lines = [l for l in out.splitlines() if "cond=1" in l]
    assert len(cond1_lines) >= 1
    cond5_lines = [l for l in out.splitlines() if "cond=5" in l]
    assert len(cond5_lines) >= 1


def test_cvec_combined_norm_returns_float() -> None:
    agent = ml._build_mock_agent()
    norm = ml._cvec_combined_norm(agent)
    assert isinstance(norm, float)
    assert norm >= 0.0
    # Mock agent cvecs are [np.zeros(16), np.zeros(16)] -> combined norm 0.0
    assert norm == pytest.approx(0.0)


def test_domain_probe_returns_fraction() -> None:
    agent = ml._build_mock_agent()
    result = ml._domain_probe(agent)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_drift_metric_triple_no_clamp() -> None:
    agent = ml._build_mock_agent()
    ml._svd_warmup(agent, [ml._CORRECTION])
    ml._compounding_loop(agent, [ml._CORRECTION], k=2, batch_size=1)
    zero = ml._build_mock_agent()
    triple = ml._drift_metric_triple(agent, zero, clamp_norm=None, use_logits=False)
    for key in ("delta_target", "delta_control", "delta_target_clamped"):
        assert key in triple
        assert isinstance(triple[key], float)
    # clamp_norm=None means no clamping: clamped leg equals the unclamped leg
    assert triple["delta_target"] == pytest.approx(triple["delta_target_clamped"])


def test_drift_metric_triple_specificity_sanity() -> None:
    agent = ml._build_mock_agent()
    ml._svd_warmup(agent, [ml._CORRECTION])
    ml._compounding_loop(agent, [ml._CORRECTION], k=2, batch_size=1)
    zero = ml._build_mock_agent()
    triple = ml._drift_metric_triple(agent, zero, clamp_norm=1.0, use_logits=False)
    # Both channels must exist and stay within [-1.0, 1.0]
    assert -1.0 <= triple["delta_target"] <= 1.0
    assert -1.0 <= triple["delta_control"] <= 1.0
    # Specificity channel exists: the two are distinct keys (not asserted equal)
    # Mock driver is semantically null, so both should be near zero.
    assert abs(triple["delta_target"]) <= 1.0
    assert abs(triple["delta_control"]) <= 1.0