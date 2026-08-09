from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from oczy.experiments.r24_tiny_decoder.phase_a_toy_v3 import (
    BASE_CONFIG as PHASE_A_BASE,
)
from oczy.experiments.r24_tiny_decoder.phase_a_toy_v3 import (
    suite_sha256 as phase_a_suite_sha256,
)
from oczy.experiments.r24_tiny_decoder.phase_a_toy_v3 import (
    train_phase_a_toy_v3,
)
from oczy.experiments.r24_tiny_decoder.phase_c_v3 import (
    BASE_CONFIG as PHASE_C_BASE,
)
from oczy.experiments.r24_tiny_decoder.phase_c_v3 import (
    CONDITIONS,
    PHASE_C_V3_SCHEMA,
    SEED_TUPLES,
    aggregate_runs,
    evaluate_test,
    load_frozen_organ,
    suite_sha256,
    train_dev,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def organ_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Small synthetic file fixture with the exact registered metadata contract."""
    path = tmp_path_factory.mktemp("r24-v3-organ")
    train_phase_a_toy_v3(
        replace(PHASE_A_BASE, steps=1, batch_size=8, dropout=0.0),
        output_dir=path,
    )
    artifact_path = path / "artifact.json"
    decoder_path = path / "decoder.pt"
    artifact = json.loads(artifact_path.read_text())
    payload = torch.load(decoder_path, map_location="cpu", weights_only=True)
    registered_config = asdict(PHASE_A_BASE)
    registered_decoder_config = dict(artifact["decoder_config"])
    registered_decoder_config["dropout"] = PHASE_A_BASE.dropout
    phase_a_suite = phase_a_suite_sha256()
    artifact["config"] = registered_config
    artifact["decoder_config"] = registered_decoder_config
    artifact["phase_a_suite_sha256"] = phase_a_suite
    artifact["phase_c_organ_gate"] = True
    artifact["suite"] = {"case": "film", "suite_sha256": phase_a_suite}
    payload["phase_a_config"] = registered_config
    payload["decoder_config"] = registered_decoder_config
    payload["phase_a_suite_sha256"] = phase_a_suite
    payload["case"] = "film"
    torch.save(payload, decoder_path)
    artifact["files"]["decoder.pt"] = {
        "sha256": _file_sha256(decoder_path),
        "bytes": decoder_path.stat().st_size,
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    return path


@pytest.fixture(scope="module")
def short_dev_run(organ_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("r24-v3-dev")
    artifact = train_dev(
        organ_dir,
        seed_index=0,
        output_dir=output,
        config=replace(PHASE_C_BASE, steps=1, task_batch_size=4),
    )
    return output, artifact


def test_strict_organ_loader_freezes_and_rejects_file_tampering(
    organ_dir: Path, tmp_path: Path
) -> None:
    organ = load_frozen_organ(organ_dir, expected_conditioning="film")
    assert organ.conditioning == "film"
    assert organ.decoder.parameter_hash() == organ.weight_hash
    assert not any(parameter.requires_grad for parameter in organ.decoder.parameters())

    copied = tmp_path / "tampered"
    copied.mkdir()
    (copied / "artifact.json").write_bytes((organ_dir / "artifact.json").read_bytes())
    payload = bytearray((organ_dir / "decoder.pt").read_bytes())
    payload[-1] ^= 1
    (copied / "decoder.pt").write_bytes(payload)
    with pytest.raises(ValueError, match="file hash"):
        load_frozen_organ(copied)


def test_one_outer_step_uses_only_cortex_and_emits_all_dev_controls(
    short_dev_run,
) -> None:
    output, artifact = short_dev_run
    assert artifact["sealed_test_accessed"] is False
    assert artifact["theta_parameter_count"] == 6_995
    assert artifact["theta0_hash"] != artifact["theta_hash"]
    assert artifact["last_trace_audit"] == {
        "writer_calls_per_task": 3,
        "consolidation_calls_per_task": 1,
        "raw_trace_count_after": 0,
        "feature_trace_count_after": 0,
        "persistent_bytes_per_task": 1_024,
    }
    result = artifact["dev"]
    assert set(result["conditions"]) == set(CONDITIONS)
    assert result["trace_audits"]["C3"]["writer_calls_per_task"] == 3
    assert result["trace_audits"]["C4"]["consolidation_calls_per_task"] == 1
    assert result["organ_hash_before_after"][0] == result["organ_hash_before_after"][1]
    assert result["theta_hash_before_after"][0] == result["theta_hash_before_after"][1]
    assert result["C8"]["logical_payload_bytes_max"] <= 1_024
    serialized = json.dumps(artifact, sort_keys=True)
    assert "x=" not in serialized
    assert "correction" not in serialized.lower()
    assert (output / "cortex.pt").is_file()


def test_test_command_rejects_forged_authorization_without_accessing_test(
    organ_dir: Path, short_dev_run, tmp_path: Path
) -> None:
    dev_dir, _ = short_dev_run
    authorization_path = tmp_path / "forged-authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "schema_version": "oczy/r24-toy-phase-c-test-authorization/v3",
                "phase_c_suite_sha256": suite_sha256(),
                "signoff_id": "forged-unit-fixture",
            }
        )
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="five complete DEV runs"):
        evaluate_test(
            organ_dir,
            dev_dir,
            authorization_path=authorization_path,
            output_dir=output,
        )
    assert not output.exists()


def _set_exact_counts(result: dict[str, object], counts: dict[str, int]) -> None:
    total = 25 * 61
    conditions = result["conditions"]
    assert isinstance(conditions, dict)
    for name, correct in counts.items():
        conditions[name] = {
            "correct": correct,
            "total": total,
            "exact_accuracy": correct / total,
        }
    per_task = result["per_task"]
    assert isinstance(per_task, dict)
    for name, correct in counts.items():
        remaining = correct
        for record in per_task.values():
            assert isinstance(record, dict)
            task_correct = min(61, remaining)
            record[name] = {
                "correct": task_correct,
                "total": 61,
                "accuracy": task_correct / 61,
            }
            remaining -= task_correct
        assert remaining == 0
    paired = result["paired_controls"]
    assert isinstance(paired, dict)
    for control in ("C1", "C2", "C4", "C5", "C6", "C8"):
        both = min(counts["C3"], counts[control])
        paired[control] = {
            "c3_only": counts["C3"] - both,
            "control_only": counts[control] - both,
            "both": both,
            "neither": total - max(counts["C3"], counts[control]),
        }


def _synthetic_run(template: dict[str, object], seed_index: int, *, c3: int = 800):
    artifact = copy.deepcopy(template)
    artifact["schema_version"] = PHASE_C_V3_SCHEMA
    artifact["phase_c_suite_sha256"] = suite_sha256()
    artifact["stage"] = "train_dev"
    artifact["config"] = asdict(PHASE_C_BASE)
    artifact["seed_index"] = seed_index
    artifact["seed_tuple"] = SEED_TUPLES[seed_index]
    artifact["organ_phase_c_gate"] = True
    artifact["theta_hash"] = f"{seed_index + 1:064x}"
    result = artifact["dev"]
    assert isinstance(result, dict)
    counts = {
        "C1": 100,
        "C2": 100,
        "C3": c3,
        "C4": 300,
        "C5": 100,
        "C6": 300,
        "C8": 30,
        "C9": 1510,
    }
    _set_exact_counts(result, counts)
    result["C7"] = {
        "same_query_total": 25 * 61,
        "state_conditioned_flip_count": 610,
        "state_conditioned_flip_rate": 610 / (25 * 61),
        "flip_to_donor_gold_count": 300,
        "flip_to_donor_gold_rate": 300 / (25 * 61),
    }
    return artifact


def test_five_seed_dev_gate_is_fail_closed_and_never_accepts(short_dev_run) -> None:
    _, template = short_dev_run
    authorized = aggregate_runs(
        [_synthetic_run(template, index) for index in range(5)], stage="dev"
    )
    assert authorized["passed"] is True
    assert authorized["decision"] == "authorize_sealed_test"

    failed_runs = [_synthetic_run(template, index) for index in range(5)]
    failed_runs[-1] = _synthetic_run(template, 4, c3=50)
    rejected = aggregate_runs(failed_runs, stage="dev")
    assert rejected["passed"] is False
    assert rejected["seed_robustness_gate"] is False


def test_dev_aggregation_rejects_forged_c7_rate(short_dev_run) -> None:
    _, template = short_dev_run
    runs = [_synthetic_run(template, index) for index in range(5)]
    result = runs[0]["dev"]
    assert isinstance(result, dict)
    c7 = result["C7"]
    assert isinstance(c7, dict)
    c7["state_conditioned_flip_rate"] = 1.0
    with pytest.raises(ValueError, match="C7 state_conditioned_flip_count"):
        aggregate_runs(runs, stage="dev")
