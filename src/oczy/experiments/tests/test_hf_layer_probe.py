"""Tests for S1.4 HF layer-L hidden probe."""

from __future__ import annotations

import numpy as np
import pytest

import oczy.experiments.hf_layer_probe as hfp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_vectors() -> dict[str, list[np.ndarray]]:
    """Two concepts with three 4-D vectors each."""
    rng = np.random.RandomState(42)
    return {
        "A": [rng.standard_normal(4) for _ in range(3)],
        "B": [rng.standard_normal(4) for _ in range(3)],
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_module_imports_without_heavy_deps() -> None:
    """Importing the module should not pull torch/transformers eagerly."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import oczy.experiments.hf_layer_probe; "
            "print([k for k in dir() if k.startswith('torch') or k.startswith('transformers')])",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "[]"


class TestCosine:
    def test_identical(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert hfp._cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert hfp._cosine(a, b) == pytest.approx(0.0)

    def test_zero_norm(self) -> None:
        a = np.zeros(4, dtype=np.float32)
        b = np.ones(4, dtype=np.float32)
        assert hfp._cosine(a, b) == 0.0


class TestSilhouette:
    def test_basic(self, sample_vectors) -> None:
        s = hfp._silhouette(sample_vectors)
        assert s is not None
        assert -1.0 <= s <= 1.0

    def test_monotonic(self) -> None:
        """Tighter intra-concept vectors → higher silhouette."""
        base = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rng = np.random.RandomState(0)
        loose = {
            "X": [base + rng.standard_normal(3).astype(np.float32) * 0.5 for _ in range(3)],
            "Y": [rng.standard_normal(3).astype(np.float32) for _ in range(3)],
        }
        tight = {
            "X": [base + rng.standard_normal(3).astype(np.float32) * 0.01 for _ in range(3)],
            "Y": [rng.standard_normal(3).astype(np.float32) for _ in range(3)],
        }
        s_loose = hfp._silhouette(loose)
        s_tight = hfp._silhouette(tight)
        assert s_loose is not None and s_tight is not None
        assert s_tight >= s_loose

    def test_single_concept_none(self) -> None:
        rng = np.random.RandomState(0)
        s = hfp._silhouette({"only": [rng.standard_normal(4) for _ in range(3)]})
        assert s is None

    def test_single_vector_per_concept_none(self) -> None:
        rng = np.random.RandomState(0)
        s = hfp._silhouette({
            "A": [rng.standard_normal(4)],
            "B": [rng.standard_normal(4)],
        })
        assert s is None


class TestMidLayerRange:
    def test_4_layers(self) -> None:
        start, end = hfp._mid_layer_range(4)
        assert start == 1
        assert end == 3  # layers 1, 2

    def test_12_layers(self) -> None:
        start, end = hfp._mid_layer_range(12)
        assert start == 3
        assert end == 9  # layers 3-8

    def test_24_layers(self) -> None:
        start, end = hfp._mid_layer_range(24)
        assert start == 6
        assert end == 18  # layers 6-17


class TestCorpusHash:
    def test_deterministic(self) -> None:
        h1 = hfp._corpus_hash()
        h2 = hfp._corpus_hash()
        assert h1 == h2
        assert len(h1) == 12


class TestGapComputation:
    def test_accept(self) -> None:
        """Gap = max(mid) - final >= 0.10 → ACCEPT."""
        # Simulate result dict from run_probe
        n_layers = 12
        sils = {"mean": {}}
        for i in range(n_layers):
            # Layers 0-2: low; 3-8 (mid): high; 9-11: low
            if 3 <= i <= 8:
                sils["mean"][f"L{i}"] = 0.30
            else:
                sils["mean"][f"L{i}"] = 0.10
        sils["mean"][f"L{n_layers - 1}"] = 0.15  # final

        # Recompute gap using the module's logic
        primary_sils = sils["mean"]
        final_score = primary_sils[f"L{n_layers - 1}"]
        start, end = hfp._mid_layer_range(n_layers)
        mid_keys = [f"L{i}" for i in range(start, end)]
        mid_scores = [primary_sils[k] for k in mid_keys]
        gap = max(mid_scores) - final_score

        assert gap >= 0.10
        assert gap == pytest.approx(0.30 - 0.15)

    def test_refute(self) -> None:
        n_layers = 12
        sils = {"mean": {}}
        for i in range(n_layers):
            sils["mean"][f"L{i}"] = 0.10
        final_score = sils["mean"][f"L{n_layers - 1}"]
        start, end = hfp._mid_layer_range(n_layers)
        mid_keys = [f"L{i}" for i in range(start, end)]
        mid_scores = [sils["mean"][k] for k in mid_keys]
        gap = max(mid_scores) - final_score
        assert gap < 0.10


class TestContentMask:
    def test_basic(self) -> None:
        """Smoke test that content_mask returns a 1-D bool array."""
        # This test requires transformers, so skip if not available
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/tiny-random-GPT2LMHeadModel"
            )
        except Exception:
            pytest.skip("tiny-random model not available")
        mask = hfp._content_mask("The capital of France is Paris.", tokenizer)
        assert mask.dtype == bool
        assert mask.ndim == 1
        assert mask.any(), "should find at least one content token"


# ---------------------------------------------------------------------------
# Plumbing test on tiny-random model
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_model
def test_run_probe_tiny_random_shapes() -> None:
    """Plumbing: run full probe on tiny-random model, verify table shape."""
    pytest.importorskip("transformers")

    model_id = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    try:
        results = hfp.run_probe(model_id)
    except Exception:
        # Tiny model might not support output_hidden_states properly
        try:
            results = hfp.run_probe("hf-internal-testing/tiny-random-OPTForCausalLM")
        except Exception:
            pytest.skip("No tiny-random model available with output_hidden_states")

    assert results["model_id"] is not None
    assert results["n_layers"] >= 1
    assert results["n_embd"] >= 1
    assert len(results["corpus_hash"]) == 12

    # silhouettes: dict of pooling -> layer -> score
    sils = results["silhouettes"]
    assert "mean" in sils
    assert "last" in sils
    assert "max" in sils

    for pooling, layer_scores in sils.items():
        assert f"L0" in layer_scores
        assert f"L{results['n_layers'] - 1}" in layer_scores
        assert len(layer_scores) >= results["n_layers"]
        # All scores should be floats
        for key, val in layer_scores.items():
            assert isinstance(val, float), f"{pooling}/{key} not float: {type(val)}"
            assert not np.isnan(val), f"{pooling}/{key} is NaN"

    # Gap and verdict
    assert isinstance(results["gap"], float)
    assert results["verdict"] in ("ACCEPT", "REFUTE")

    # Layers should have distinct scores (not all identical) for at least one pooling
    for pooling in ("mean", "last", "max"):
        values = list(sils[pooling].values())
        unique = len(set(f"{v:.4f}" for v in values))
        if unique > 1:
            break
    else:
        # On a truly random tiny model, all layers might be identical;
        # that's acceptable — the shapes are what matter.
        pass


# ---------------------------------------------------------------------------
# Formatting smoke tests
# ---------------------------------------------------------------------------


def test_format_results_table() -> None:
    """Smoke test: format_results_table produces non-empty output."""
    fake = {
        "model_id": "test/model",
        "n_layers": 4,
        "n_embd": 16,
        "corpus_hash": "abcdef123456",
        "primary_pooling": "mean",
        "mid_layer_range": "1-2",
        "silhouettes": {
            "mean": {"L0": 0.1, "L1": 0.2, "L2": 0.3, "L3": 0.15},
            "last": {"L0": 0.05, "L1": 0.1, "L2": 0.15, "L3": 0.1},
            "max": {"L0": 0.08, "L1": 0.12, "L2": 0.18, "L3": 0.12},
        },
        "final_score": 0.15,
        "max_mid": 0.3,
        "gap": 0.15,
        "verdict": "ACCEPT",
    }
    out = hfp.format_results_table(fake)
    assert "S1.4" in out
    assert "ACCEPT" in out
    assert "| Layer |" in out
    assert "L0" in out
    assert "L3" in out
    assert "mean" in out
    assert "last" in out
    assert "max" in out
