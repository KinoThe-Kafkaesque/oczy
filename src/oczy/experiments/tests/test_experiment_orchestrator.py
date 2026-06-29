"""Tests for the combined experiment orchestrator.

These tests monkeypatch the subprocess runner so no heavy drivers or long
experiments are launched.
"""

from __future__ import annotations

import sys

import src.oczy.experiments.experiment_orchestrator as eo


def test_module_imports_without_heavy_deps() -> None:
    assert sys.modules.get("llama_cpp") is None


def test_acceptance_predicates() -> None:
    assert eo._EXPERIMENTS[0].accepted(5.0) is True
    assert eo._EXPERIMENTS[0].accepted(2.0) is False
    assert eo._EXPERIMENTS[5].accepted(0.05) is True
    assert eo._EXPERIMENTS[5].accepted(0.20) is False


def test_metric_regex_parses_lines() -> None:
    m = eo._METRIC_RE.match("METRIC foo=1.23")
    assert m
    assert m.group(1) == "foo"
    assert m.group(2) == "1.23"

    assert eo._METRIC_RE.match("ASI foo=1.23") is None


def _fake_run(exp: eo.Experiment) -> float:
    values = {
        "v2_desaturation_count": 5.0,
        "kv_slot_rank1_count": 3.0,
        "layer_l_silhouette_gap": -0.05,
        "scope_selectivity_index": 0.625,
        "metabolism_drift_delta": 0.0,
        "bounded_growth_m1_ratio": 0.07,
        "marker_free_uptake_gap": 1.0,
    }
    return values.get(exp.metric_name, float("nan"))


def test_main_mock_emits_count(monkeypatch, capsys) -> None:
    monkeypatch.setattr(eo, "_run_experiment", _fake_run)
    assert eo.main(["--driver", "mock"]) == 0
    out = capsys.readouterr().out
    # Scope module is forced to 0 under mock mode, so 4 of 7 accepted.
    assert "METRIC experiments_accepted_count=4" in out
    assert "ASI experiments_total=7" in out
    for exp in eo._EXPERIMENTS:
        assert f"ASI {exp.metric_name}=" in out


def test_default_driver_is_mock(monkeypatch, capsys) -> None:
    monkeypatch.setattr(eo, "_run_experiment", _fake_run)
    assert eo.main([]) == 0
    out = capsys.readouterr().out
    assert "METRIC experiments_accepted_count=4" in out


def test_nan_value_treated_not_accepted(monkeypatch, capsys) -> None:
    monkeypatch.setattr(eo, "_run_experiment", lambda exp: float("nan"))
    assert eo.main(["--driver", "mock"]) == 0
    out = capsys.readouterr().out
    assert "METRIC experiments_accepted_count=0" in out
    for exp in eo._EXPERIMENTS:
        assert f"ASI {exp.name}_accepted=0" in out
