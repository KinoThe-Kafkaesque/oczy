"""Deterministic regression tests for the OMP throughput probe.

Covers CLI command construction, schema validation (acceptance and
rejection), descriptive statistics, warnings, duplicate-selector CLI
rejection, atomic output, and one end-to-end subprocess smoke against a
temporary fake OMP executable.  No real provider, network, or model is
invoked — every test is deterministic and full-suite safe.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# ─── dynamic import of the probe module ───────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "benchmarks" / "omp" / "throughput_probe.py"

_spec = importlib.util.spec_from_file_location(
    "oczy_omp_throughput_probe_tests", PROBE_PATH
)
assert _spec is not None and _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = probe
_spec.loader.exec_module(probe)


# ─── compact fixtures / helpers ───────────────────────────────────────


def _success(
    ttft: float = 100.0,
    duration: float = 1000.0,
    tokens: int = 256,
    tps: float = 50.0,
) -> dict:
    return {
        "ok": True,
        "ttftMs": ttft,
        "durationMs": duration,
        "outputTokens": tokens,
        "tokensPerSecond": tps,
    }


def _failure(error: str = "boom") -> dict:
    return {"ok": False, "error": error}


def _report(selector: str, results: list[dict], thinking: str | None = None) -> dict:
    successes = [r for r in results if r.get("ok") is True]
    avg = None
    if successes:
        avg = {
            "ttftMs": sum(s["ttftMs"] for s in successes) / len(successes),
            "durationMs": sum(s["durationMs"] for s in successes) / len(successes),
            "outputTokens": sum(s["outputTokens"] for s in successes) / len(successes),
            "tokensPerSecond": sum(s["tokensPerSecond"] for s in successes) / len(successes),
        }
    rep = {
        "selector": selector,
        "model": f"resolved-{selector}",
        "results": results,
        "average": avg,
    }
    if thinking is not None:
        rep["thinking"] = thinking
    return rep


def _summary(
    reports: list[dict],
    runs: int = 2,
    max_tokens: int = 256,
    failures: int | None = None,
) -> dict:
    if failures is None:
        failures = sum(
            1 for m in reports for r in m["results"] if r.get("ok") is False
        )
    return {
        "runs": runs,
        "maxTokens": max_tokens,
        "models": reports,
        "failures": failures,
    }


# ─── command construction ─────────────────────────────────────────────


def test_build_bench_command_exact_positional() -> None:
    """omp bench MODEL... with pinned flags and embedded prompt, flat list."""
    cmd = probe.build_bench_command(
        Path("/usr/local/bin/omp"),
        ["devin/glm-5-2", "devin/kimi-k2-7"],
        runs=3,
        max_tokens=256,
        par=1,
        service_tier="none",
        prompt=probe.BENCH_PROMPT,
    )
    assert cmd[0] == "/usr/local/bin/omp"
    assert cmd[1] == "bench"
    # Positional selectors immediately after "bench", preserving order.
    assert cmd[2:4] == ["devin/glm-5-2", "devin/kimi-k2-7"]
    # Pinned flags in exact order.
    assert cmd[4:] == [
        "--runs", "3",
        "--max-tokens", "256",
        "--par", "1",
        "--service-tier", "none",
        "--prompt", probe.BENCH_PROMPT,
        "--json",
    ]
    assert all(isinstance(c, str) for c in cmd)


# ─── successful validation ────────────────────────────────────────────


def test_validate_success_two_models_two_runs() -> None:
    """A well-formed two-model/two-run payload passes validation."""
    reports = [
        _report("alpha", [_success(100, 1000, 256, 50), _success(120, 1100, 256, 55)]),
        _report("beta", [_success(80, 900, 256, 60), _success(90, 950, 256, 58)]),
    ]
    data = _summary(reports, runs=2, max_tokens=256)
    probe.validate_bench_summary(data, ["alpha", "beta"], 2, 256)


# ─── rejection tests ──────────────────────────────────────────────────


def test_reject_selector_order_mismatch() -> None:
    reports = [
        _report("beta", [_success(), _success()]),
        _report("alpha", [_success(), _success()]),
    ]
    with pytest.raises(probe.ProbeError):
        probe.validate_bench_summary(_summary(reports), ["alpha", "beta"], 2, 256)


def test_reject_runs_mismatch() -> None:
    reports = [_report("alpha", [_success()])]  # 1 result, expected 2
    with pytest.raises(probe.ProbeError):
        probe.validate_bench_summary(_summary(reports, runs=2), ["alpha"], 2, 256)


def test_reject_max_tokens_mismatch() -> None:
    reports = [_report("alpha", [_success(), _success()])]
    data = _summary(reports, runs=2, max_tokens=256)
    with pytest.raises(probe.ProbeError):
        probe.validate_bench_summary(data, ["alpha"], 2, 512)


def test_reject_non_finite_metrics() -> None:
    reports = [_report("alpha", [_success(ttft=float("nan")), _success()])]
    with pytest.raises(probe.ProbeError):
        probe.validate_bench_summary(_summary(reports), ["alpha"], 2, 256)


def test_reject_negative_metrics() -> None:
    reports = [_report("alpha", [_success(ttft=-10), _success()])]
    with pytest.raises(probe.ProbeError):
        probe.validate_bench_summary(_summary(reports), ["alpha"], 2, 256)


def test_reject_ttft_exceeds_duration() -> None:
    reports = [_report("alpha", [_success(ttft=2000, duration=1000), _success()])]
    with pytest.raises(probe.ProbeError):
        probe.validate_bench_summary(_summary(reports), ["alpha"], 2, 256)


def test_reject_failure_count_mismatch() -> None:
    reports = [_report("alpha", [_success(), _failure()])]
    data = _summary(reports, failures=0)  # claim 0 but 1 exists
    with pytest.raises(probe.ProbeError):
        probe.validate_bench_summary(data, ["alpha"], 2, 256)


def test_reject_incorrect_average() -> None:
    rep = _report(
        "alpha",
        [_success(100, 1000, 256, 50), _success(200, 2000, 256, 60)],
    )
    rep["average"]["ttftMs"] = 999  # wrong mean (should be 150)
    with pytest.raises(probe.ProbeError):
        probe.validate_bench_summary(_summary([rep]), ["alpha"], 2, 256)


# ─── preflight failure acceptance ─────────────────────────────────────


def test_accept_single_preflight_failure() -> None:
    """A single ok:false result is accepted even when runs > 1."""
    rep = _report("alpha", [_failure("connection refused")])
    data = _summary([rep], runs=3, failures=1)
    probe.validate_bench_summary(data, ["alpha"], 3, 256)


# ─── descriptive statistics ───────────────────────────────────────────


def test_descriptive_stats_sample_stdev_and_cv() -> None:
    stats = probe._descriptive_stats([10.0, 20.0, 30.0])
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(20.0)
    assert stats["median"] == pytest.approx(20.0)
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0
    # Sample stdev uses n-1 denominator → 10.0 for [10,20,30].
    assert stats["sampleStdev"] == pytest.approx(10.0)
    # CV% = stdev / mean * 100 = 50.0.
    assert stats["coefficientOfVariationPercent"] == pytest.approx(50.0)


def test_descriptive_stats_single_sample_stdev_null() -> None:
    stats = probe._descriptive_stats([42.0])
    assert stats["count"] == 1
    assert stats["mean"] == pytest.approx(42.0)
    assert stats["sampleStdev"] is None
    assert stats["coefficientOfVariationPercent"] is None


def test_descriptive_stats_empty() -> None:
    stats = probe._descriptive_stats([])
    assert stats["count"] == 0
    assert stats["mean"] is None
    assert stats["sampleStdev"] is None


def test_model_stats_no_cross_model_mixing() -> None:
    """Per-model stats use only that model's successes."""
    successes_a = [_success(ttft=100, tps=50), _success(ttft=200, tps=60)]
    successes_b = [_success(ttft=500, tps=10), _success(ttft=600, tps=20)]

    stats_a = probe._model_stats(successes_a)
    assert stats_a["ttftMs"]["mean"] == pytest.approx(150.0)
    assert stats_a["ttftMs"]["min"] == 100.0
    assert stats_a["ttftMs"]["max"] == 200.0
    assert stats_a["tokensPerSecond"]["mean"] == pytest.approx(55.0)
    assert stats_a["tokensPerSecond"]["max"] == 60.0

    stats_b = probe._model_stats(successes_b)
    assert stats_b["ttftMs"]["mean"] == pytest.approx(550.0)
    assert stats_b["tokensPerSecond"]["mean"] == pytest.approx(15.0)
    # Model B values must not appear in model A stats.
    assert stats_a["ttftMs"]["max"] < stats_b["ttftMs"]["min"]


# ─── warnings ─────────────────────────────────────────────────────────


def test_token_cap_warning() -> None:
    """outputTokens exceeding maxTokens triggers a per-run warning."""
    raw = _summary(
        [_report("alpha", [_success(tokens=300), _success(tokens=256)])],
        runs=2,
        max_tokens=256,
    )
    warnings = probe._compute_warnings(raw, 256)
    cap_warns = [w for w in warnings if "exceeds maxTokens" in w]
    assert len(cap_warns) == 1
    assert "alpha" in cap_warns[0]
    assert "300" in cap_warns[0]
    assert "256" in cap_warns[0]


def test_first_stream_token_warning_always_present() -> None:
    raw = _summary([_report("alpha", [_success()])], runs=1)
    warnings = probe._compute_warnings(raw, 256)
    assert any("TTFT" in w and "first" in w for w in warnings)


# ─── CLI: duplicate / blank selector rejection ────────────────────────


def test_duplicate_selector_rejected() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["alpha", "alpha", "--runs", "2"])


def test_blank_selector_rejected() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["", "beta"])


# ─── atomic output ────────────────────────────────────────────────────


def test_atomic_output_writes_valid_json(tmp_path: Path) -> None:
    envelope = {"schema_version": probe.SCHEMA_VERSION, "status": "ok", "data": [1, 2, 3]}
    out = tmp_path / "result.json"
    probe.write_output(envelope, str(out))
    assert out.exists()
    assert json.loads(out.read_text()) == envelope
    # No temp residue left behind.
    assert [p for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_output_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    envelope = {"schema_version": probe.SCHEMA_VERSION, "ok": True}
    probe.write_output(envelope, None)
    assert json.loads(capsys.readouterr().out) == envelope


# ─── end-to-end subprocess smoke ──────────────────────────────────────

_FAKE_OMP = """\
#!/usr/bin/env python3
import sys, json

def main():
    argv = sys.argv[1:]
    if not argv:
        return 2
    if argv[0] == "--version":
        sys.stdout.write("fake-omp 0.0.0-test\\n")
        return 0
    if argv[0] != "bench":
        return 2
    models = []
    runs = 1
    max_tokens = 256
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--runs":
            runs = int(argv[i + 1]); i += 2
        elif a == "--max-tokens":
            max_tokens = int(argv[i + 1]); i += 2
        elif a == "--json":
            i += 1
        elif a.startswith("--"):
            i += 2
        else:
            models.append(a); i += 1
    reports = []
    for idx, sel in enumerate(models):
        results = []
        for r in range(runs):
            results.append({
                "ok": True,
                "ttftMs": 100.0 + idx * 10 + r,
                "durationMs": 1000.0 + idx * 100 + r * 10,
                "outputTokens": max_tokens,
                "tokensPerSecond": 50.0 + idx + r * 0.5,
            })
        avg = {
            "ttftMs": sum(x["ttftMs"] for x in results) / len(results),
            "durationMs": sum(x["durationMs"] for x in results) / len(results),
            "outputTokens": sum(x["outputTokens"] for x in results) / len(results),
            "tokensPerSecond": sum(x["tokensPerSecond"] for x in results) / len(results),
        }
        reports.append({"selector": sel, "model": "resolved-" + sel,
                        "results": results, "average": avg})
    sys.stdout.write(json.dumps(
        {"runs": runs, "maxTokens": max_tokens, "models": reports, "failures": 0}
    ))
    return 0

sys.exit(main())
"""


def test_end_to_end_subprocess_smoke(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full probe run against a fake OMP executable — no network, no provider."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP)
    os.chmod(fake, 0o755)

    # Point proxies at a dead address: if anything touched the network the
    # run would fail, proving the probe + fake are entirely local.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "envelope.json"
    rc = probe.main([
        "devin/glm-5-2", "devin/kimi-k2-7",
        "--runs", "2",
        "--max-tokens", "256",
        "--par", "1",
        "--service-tier", "none",
        "--omp-bin", str(fake),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_OK

    # Resolved output path printed to stderr.
    assert str(out.resolve()) in capsys.readouterr().err

    env = json.loads(out.read_text())
    assert env["schema_version"] == probe.SCHEMA_VERSION
    assert env["status"] == probe.STATUS_OK
    assert env["provenance"]["ompVersion"] == "fake-omp 0.0.0-test"
    assert env["provenance"]["ompBinary"] == str(fake.resolve())
    assert env["raw"]["runs"] == 2
    assert len(env["raw"]["models"]) == 2
    assert len(env["metrics"]) == 2
    assert env["metrics"][0]["selector"] == "devin/glm-5-2"
    assert env["metrics"][1]["selector"] == "devin/kimi-k2-7"
    # No temp residue.
    assert [p for p in out.parent.iterdir() if p.suffix == ".tmp"] == []
