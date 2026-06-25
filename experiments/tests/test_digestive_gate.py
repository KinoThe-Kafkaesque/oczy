"""Unit tests for the DigestiveGate metabolic gating organ."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.digestive_gate import DigestiveGate, DigestiveGateConfig


def test_low_drift_low_correction():
    gate = DigestiveGate()
    result = gate.ingest(drift=0.1, correction_signal=0.0, novelty=0.1)

    assert result["critic_weight"] == pytest.approx(1.0)
    assert result["hippocampus_weight"] == 0.0
    assert result["identity_weight"] == 0.0
    assert result["immune_weight"] == 0.0
    assert result["autoencoder_weight"] == pytest.approx(0.2)
    assert result["consolidation_pressure"] == pytest.approx(0.01)
    assert gate.should_consolidate() is False


def test_high_drift_and_novelty():
    gate = DigestiveGate()
    result = gate.ingest(drift=0.8, correction_signal=0.0, novelty=0.8)

    assert result["critic_weight"] == pytest.approx(1.0)
    assert result["hippocampus_weight"] == 1.0
    assert result["identity_weight"] == 0.0
    assert result["immune_weight"] == 0.0
    assert result["autoencoder_weight"] == pytest.approx(0.9)
    assert result["consolidation_pressure"] == pytest.approx(0.08)
    assert gate.should_consolidate() is False


def test_correction_boost():
    base = DigestiveGate(DigestiveGateConfig(correction_boost=1.0))
    boosted = DigestiveGate(DigestiveGateConfig(correction_boost=2.0))

    base_result = base.ingest(
        drift=0.0,
        correction_signal=0.35,
        identity_relevance=1.0,
    )
    boosted_result = boosted.ingest(
        drift=0.0,
        correction_signal=0.35,
        identity_relevance=1.0,
    )

    # Without boost, correction_signal * boost == 0.35 (below identity threshold).
    assert base_result["identity_weight"] == 0.0
    assert base_result["immune_weight"] == 0.0

    # With 2x boost, the effective correction crosses the 0.5 threshold.
    assert boosted_result["identity_weight"] == pytest.approx(1.0)
    assert boosted_result["immune_weight"] == pytest.approx(1.0)
    # Consolidation input sees the boosted correction, so pressure should be higher.
    assert boosted_result["consolidation_pressure"] > base_result["consolidation_pressure"]


def test_immune_suppression():
    gate = DigestiveGate(DigestiveGateConfig(immune_suppress_identity=True))
    result = gate.ingest(
        drift=0.0,
        correction_signal=0.9,
        immune_conflict=0.9,
        identity_relevance=1.0,
    )

    assert result["identity_weight"] == 0.0
    assert result["immune_weight"] == pytest.approx(1.0)


def test_immune_suppression_disabled():
    gate = DigestiveGate(DigestiveGateConfig(immune_suppress_identity=False))
    result = gate.ingest(
        drift=0.0,
        correction_signal=0.9,
        immune_conflict=0.9,
        identity_relevance=1.0,
    )

    assert result["identity_weight"] == pytest.approx(1.0)
    assert result["immune_weight"] == pytest.approx(1.0)


def test_consolidation_pressure_accumulates():
    gate = DigestiveGate()

    gate.ingest(drift=1.0, correction_signal=0.0)
    assert gate._pressure == pytest.approx(0.1)
    assert gate.should_consolidate() is False

    gate.ingest(drift=1.0, correction_signal=0.0)
    assert gate._pressure == pytest.approx(0.19)
    assert gate.should_consolidate() is False

    gate.ingest(drift=1.0, correction_signal=0.0)
    assert gate._pressure == pytest.approx(0.25)
    assert gate.should_consolidate() is True

    # Further high-drift steps saturate at the threshold.
    gate.ingest(drift=1.0, correction_signal=0.0)
    assert gate._pressure == pytest.approx(0.25)
    assert gate.should_consolidate() is True


def test_should_consolidate_explicit_pressure():
    gate = DigestiveGate(DigestiveGateConfig(consolidation_pressure_threshold=0.25))

    assert gate.should_consolidate(pressure=0.24) is False
    assert gate.should_consolidate(pressure=0.25) is True
    assert gate.should_consolidate(pressure=1.0) is True


def test_gate_resets_pressure():
    gate = DigestiveGate()
    gate.ingest(drift=1.0, correction_signal=0.0)
    gate.ingest(drift=1.0, correction_signal=0.0)
    gate.ingest(drift=1.0, correction_signal=0.0)
    assert gate.should_consolidate() is True

    gate.reset()
    assert gate._ema == pytest.approx(0.0)
    assert gate._pressure == pytest.approx(0.0)
    assert gate.should_consolidate() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
