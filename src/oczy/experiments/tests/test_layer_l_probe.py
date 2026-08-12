"""Tests for the Layer-L hidden extraction probe (Experiment 03)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import oczy.experiments.layer_l_probe as layer_l_probe


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
            "import sys; import oczy.experiments.layer_l_probe as m; "
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


def test_compute_gap_compares_mid_layers_to_hf_final_meanpool() -> None:
    """The primary gap is best mid-layer silhouette minus HF final mean-pool."""
    s = {
        "R_random": 10.0,
        "last_L9": 0.52,
        "last_L13": 0.41,
        "mean_L14": 0.47,
        "final_meanpool": 0.34,
    }
    r = layer_l_probe._compute_gap(s)
    assert r["gap"] == pytest.approx(0.18)
    assert r["max_mid"] == pytest.approx(0.52)
    assert r["final"] == pytest.approx(0.34)
    assert r["mid_labels"] == ["last_L9", "last_L13", "mean_L14"]


def test_compute_gap_without_final() -> None:
    s = {"L9_last": 0.5}
    r = layer_l_probe._compute_gap(s)
    assert r["gap"] == 0.0
    assert r["final"] is None


def test_compute_gap_empty() -> None:
    r = layer_l_probe._compute_gap({})
    assert r["gap"] == 0.0


def test_hf_probe_includes_hf_final_meanpool_without_gguf(monkeypatch) -> None:
    """HF hidden states alone provide final_meanpool when llama_cpp/GGUF is absent."""
    hidden_dim = layer_l_probe._D_EMBD
    concept_axis_by_phrase = {
        phrase: idx
        for idx, phrases in enumerate(layer_l_probe._CONCEPTS.values())
        for phrase in phrases
    }

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTensor:
        def __init__(self, array: np.ndarray) -> None:
            self._array = array

        def __getitem__(self, item):
            return _FakeTensor(self._array[item])

        def to(self, dtype):
            return self

        def numpy(self) -> np.ndarray:
            return self._array

    class _FakeTokenizer:
        def __call__(self, phrase: str, return_tensors: str):
            assert return_tensors == "pt"
            return {"phrase": phrase}

    class _FakeModel:
        def eval(self) -> None:
            return None

        def __call__(self, phrase: str):
            axis = concept_axis_by_phrase[phrase]
            hidden_states = []
            for idx in range(layer_l_probe._N_HIDDEN_STATES):
                sequence = np.zeros((3, hidden_dim), dtype=np.float32)
                sequence[:, axis] = 1.0
                if idx == layer_l_probe._N_HIDDEN_STATES - 1:
                    sequence[-1, axis] = 0.0
                hidden_states.append(_FakeTensor(sequence[np.newaxis, :, :]))
            return types.SimpleNamespace(hidden_states=hidden_states)

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            return _FakeTokenizer()

    class _FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            return _FakeModel()

    class _FakeKVCortexConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeKVCortex:
        def __init__(self, config: _FakeKVCortexConfig) -> None:
            self.config = config

        def reset_warm_to_zeros(self) -> None:
            return None

        def observe(
            self, vec: np.ndarray, correction_signal: float = 0.0
        ) -> np.ndarray:
            return vec.astype(np.float32, copy=False)

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.float32 = object()
    fake_torch.no_grad = _NoGrad

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _FakeAutoModelForCausalLM
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer

    fake_plastic = types.ModuleType("plastic_cortex")
    fake_kv = types.ModuleType("plastic_cortex.kv_cortex")
    fake_kv.KVCortex = _FakeKVCortex
    fake_kv.KVCortexConfig = _FakeKVCortexConfig

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "plastic_cortex", fake_plastic)
    monkeypatch.setitem(sys.modules, "plastic_cortex.kv_cortex", fake_kv)
    monkeypatch.setitem(sys.modules, "llama_cpp", None)

    silhouettes = layer_l_probe._hf_probe()

    assert silhouettes is not None
    assert silhouettes["final_meanpool"] == pytest.approx(1.0)
    assert layer_l_probe._compute_gap(silhouettes)["final"] == pytest.approx(1.0)


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


def test_real_driver_fail_closed_on_exception(monkeypatch, capsys) -> None:
    """When _hf_probe raises, --driver real must exit 1 and emit no METRIC."""
    def _raise():
        raise RuntimeError("no model")

    monkeypatch.setattr(layer_l_probe, "_hf_probe", _raise)
    assert layer_l_probe.main(["--driver", "real"]) == 1
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "ASI real_driver_error_type=RuntimeError" in out
    assert "METRIC layer_l_silhouette_gap=" not in out


def test_real_driver_fail_closed_on_none(monkeypatch, capsys) -> None:
    """When _hf_probe returns None, --driver real must exit 1 and emit no METRIC."""
    monkeypatch.setattr(layer_l_probe, "_hf_probe", lambda: None)
    assert layer_l_probe.main(["--driver", "real"]) == 1
    out = capsys.readouterr().out
    assert "ASI real_driver=failed" in out
    assert "returned no result" in out
    assert "METRIC layer_l_silhouette_gap=" not in out


def test_real_driver_success_prints_metrics(monkeypatch, capsys) -> None:
    """When _hf_probe returns valid silhouettes, --driver real exits 0 with metrics."""
    fake_sils = {"R_random": 0.0, "mean_L14": 0.55}
    monkeypatch.setattr(layer_l_probe, "_hf_probe", lambda: fake_sils)
    assert layer_l_probe.main(["--driver", "real"]) == 0
    out = capsys.readouterr().out
    assert "METRIC layer_l_silhouette_gap=" in out
    assert "ASI warm_sep_silhouette_mean_L14=0.55" in out


def test_real_driver_fail_closed_no_mock_fallback(monkeypatch, capsys) -> None:
    """A real-driver failure must never emit mock METRIC lines."""
    def _raise():
        raise ImportError("transformers not found")

    monkeypatch.setattr(layer_l_probe, "_hf_probe", _raise)
    rc = layer_l_probe.main(["--driver", "real"])
    out = capsys.readouterr().out
    assert rc == 1
    # No mock silhouette values should appear — only the failure diagnostic.
    assert "METRIC" not in out
    assert "ASI warm_sep_silhouette_" not in out
    assert "ASI real_driver=failed" in out


def test_main_default_driver_is_mock(capsys) -> None:
    assert layer_l_probe.main([]) == 0
    out = capsys.readouterr().out
    assert "METRIC layer_l_silhouette_gap=" in out
