"""Tests for the KV-slot fact injection probe (Experiment 02)."""

from __future__ import annotations

import oczy.experiments.kv_slot_injection as kvi


def test_module_imports_without_llama() -> None:
    """Importing the module should not pull llama_cpp."""
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "import sys; import oczy.experiments.kv_slot_injection; "
            "print('llama_cpp' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "False" in result.stdout


def test_facts_queries_targets_aligned() -> None:
    facts, queries, targets = kvi._facts_queries_targets()
    assert len(facts) == 3
    assert len(facts) == len(queries) == len(targets)
    for fact, target in zip(facts, targets, strict=True):
        assert target in fact.lower()


def test_mock_driver_returns_results() -> None:
    results = kvi._run_mock_driver()
    assert set(results).issuperset({"baseline", "live_prefix", "kv_chunk", "logit_bias"})
    for _, arr in results.items():
        assert len(arr) == 3
        for r in arr:
            assert "rank" in r
            assert "top1" in r


def test_mock_mode_emits_metric(capsys) -> None:
    assert kvi.main(["--driver", "mock"]) == 0
    out = capsys.readouterr().out
    assert "METRIC kv_slot_rank1_count=" in out
    assert "ASI live_prefix_rank1_count=" in out


def test_default_driver_is_mock(capsys) -> None:
    assert kvi.main([]) == 0
    out = capsys.readouterr().out
    assert "METRIC kv_slot_rank1_count=" in out


def test_real_driver_graceful_failure(monkeypatch, capsys) -> None:
    def _fail():
        raise RuntimeError("no gguf")

    monkeypatch.setattr(kvi, "_run_real_driver", _fail)
    assert kvi.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "METRIC kv_slot_rank1_count=" in out


def test_real_driver_graceful_none(monkeypatch, capsys) -> None:
    monkeypatch.setattr(kvi, "_run_real_driver", lambda: None)
    assert kvi.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "METRIC kv_slot_rank1_count=" in out
