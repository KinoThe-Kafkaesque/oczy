"""Scientific boundaries and persistence checks for the DEV pilot."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.prepare_serialization_pilot import build_data
from scripts.serialization_pilot_contract import read_split, recovery, sha, write_json
from scripts.serialization_pilot_worker import load_numeric, save_numeric, train


def test_all_patterns_are_identified_from_training_without_probe_overlap():
    data = build_data()
    for training, probes, audit in zip(data["training"]["patterns"], data["probes"]["patterns"], data["audit"]["patterns"], strict=True):
        assert len(training["examples"]) == 3
        assert len(probes["probes"]) == 4
        assert len(audit["compatible_candidates"]) == 1
        assert {r["input"] for r in training["examples"]}.isdisjoint(r["input"] for r in probes["probes"])
        assert set(training) == {"index", "examples"}


@pytest.mark.parametrize("phase,split", [("train", "probes"), ("train", "audit"), ("restore_soft", "training"), ("restore_text", "training")])
def test_phase_firewall_rejects_forbidden_data_before_file_access(tmp_path, phase, split):
    with pytest.raises(ValueError, match="must not read"):
        read_split(tmp_path, {}, phase, split)


def test_numeric_roundtrip_and_tampering(tmp_path):
    bank = torch.randn(1, 8, 896)
    entry = save_numeric(tmp_path / "bank.npy", bank)
    manifest = {"training": {"bank_width": 8, "feature_dim": 896}}
    loaded = load_numeric(tmp_path, entry, manifest)
    assert torch.equal(bank, loaded)
    assert not loaded.requires_grad
    assert entry["payload_bytes"] == 28672
    with (tmp_path / "bank.npy").open("ab") as file:
        file.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_numeric(tmp_path, entry, manifest)


def test_invalid_state_shape_rejected_even_with_valid_file_hash(tmp_path):
    entry = save_numeric(tmp_path / "bank.npy", torch.zeros(1, 7, 896))
    with pytest.raises(ValueError, match="Invalid numeric"):
        load_numeric(tmp_path, entry, {"training": {"bank_width": 8, "feature_dim": 896}})


def test_recovery_never_hides_zero_or_negative_denominator():
    assert recovery(0, 0, 0) is None
    assert recovery(1, 1, 0) is None
    assert recovery(2, 0, 4) == 0.5
    assert recovery(0, 1, 4) == -1 / 3


def test_training_runs_without_any_probe_file_and_saves_only_numeric_state(tmp_path):
    examples = {"patterns": [{"index": 0, "examples": [{"input": "a", "target": "az"}, {"input": "b", "target": "bz"}, {"input": "c", "target": "cz"}]}]}
    write_json(tmp_path / "training.json", examples)
    config = {"seeds": [0], "steps": 2, "bank_width": 8, "feature_dim": 4,
              "initialization_std": 0.02, "learning_rate": 0.03, "gradient_clip_norm": 1.0}
    manifest = {"manifest_sha256": "test", "training": config, "generation": {"max_new_tokens": 32},
                "files": {"training": {"path": "training.json", "sha256": sha((tmp_path / "training.json").read_bytes())}}}

    class DummyOrgan:
        feature_dim = 4
        _model = torch.nn.Linear(4, 4).requires_grad_(False)
        _tokenizer = SimpleNamespace(eos_token="<EOS>")

        def teacher_forced_loss(self, messages, target, bank):
            assert target in {"az<EOS>", "bz<EOS>", "cz<EOS>"}
            return (bank - 0.5).square().mean()

        def generate(self, messages, bank, max_new_tokens):
            return "unit-test-placeholder"

        def assert_frozen(self):
            assert all(not p.requires_grad and p.grad is None for p in self._model.parameters())

    output = tmp_path / "output"
    output.mkdir()
    report = train(DummyOrgan(), SimpleNamespace(score_response=lambda expected, actual: False),
                   SimpleNamespace(root=tmp_path, phase="train", output=output), manifest)
    assert report["heldout_examples_read"] == 0
    assert report["optimizer_steps"] == 2
    state = json.loads((output / "states/state_manifest.json").read_text())["states"][0]
    before = np.load(output / "states" / state["initial"]["path"], allow_pickle=False)
    after = np.load(output / "states" / state["trained"]["path"], allow_pickle=False)
    assert not np.array_equal(before, after)
    assert after.dtype == np.float32
