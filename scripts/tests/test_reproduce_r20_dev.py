"""The local reproduction path must preserve provenance and model integrity."""

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.reproduce_r20_dev import (
    compare_cell,
    materialize_model,
    namespace_prefix,
    recorded_options,
    validate_cell,
)


def provenance(phase="development"):
    return {"exit_code": 0, "job_spec": {
        "phase": phase, "profile": "cpu", "module": "oczy.experiments.meta_cortex",
        "arguments": ["collect-calibration-shard", "--task-start", "3", "--task-end", "6",
                      "--eval-seed-indices", "0,2"],
    }}


def test_reproduction_cannot_access_unrecorded_cell():
    options = recorded_options(provenance())
    validate_cell(options, 3, 2)
    with pytest.raises(ValueError, match="task"):
        validate_cell(options, 6, 2)
    with pytest.raises(ValueError, match="seed"):
        validate_cell(options, 3, 1)


@pytest.mark.parametrize("phase", ["meta-test", "oracle", "analysis"])
def test_non_dev_provenance_rejected(phase):
    with pytest.raises(ValueError, match="CPU DEV"):
        recorded_options(provenance(phase))


def test_duplicate_recorded_options_rejected():
    data = provenance()
    data["job_spec"]["arguments"] += ["--task-start", "0"]
    with pytest.raises(ValueError, match="duplicate"):
        recorded_options(data)


def test_model_copy_dereferences_snapshot_but_cannot_mutate_cache(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"frozen weights")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights").symlink_to(blob)
    artifact = {"path": "weights", "size_bytes": len(blob.read_bytes()),
                "sha256": hashlib.sha256(blob.read_bytes()).hexdigest()}
    destination = tmp_path / "copied"
    materialize_model(snapshot, destination, [artifact])
    assert not (destination / "weights").is_symlink()
    (destination / "weights").write_bytes(b"changed")
    assert blob.read_bytes() == b"frozen weights"


def test_model_tamper_is_rejected(tmp_path):
    (tmp_path / "weights").write_bytes(b"bad")
    with pytest.raises(ValueError, match="mismatch"):
        materialize_model(tmp_path, tmp_path / "out", [{
            "path": "weights", "size_bytes": 3, "sha256": "0" * 64,
        }])


@pytest.mark.parametrize("root", ["/home/user", "../../kaggle", "/kaggle/input/models/../other"])
def test_namespace_only_maps_recorded_model_location(root):
    with pytest.raises(ValueError, match="provenance-recorded"):
        namespace_prefix(Path("/model"), root, Path("/output"))


def test_namespace_is_offline_and_readonly_outside_output():
    command = namespace_prefix(Path("/model"), "/kaggle/input/models/qwen/v1", Path("/output"))
    assert "--unshare-net" in command
    assert command.count("--bind") == 1
    writable = command.index("--bind")
    assert command[writable:writable + 3] == ["--bind", "/output", "/output"]
    assert command[-6:] == ["--ro-bind", "/model", "/model", "--ro-bind", "/model", "/kaggle/input/models/qwen/v1"]


def test_identical_scores_do_not_hide_audit_divergence_or_duplicate_records():
    local = {
        "schema": "s", "definition_sha256": "d", "calibration_view_sha256": "v",
        "scorer_sha256": "sc", "organ_hash": "o",
        "seed_cell_records": [{"developmental_seed_index": 0, "evaluation_seed_index": 0,
                               "rule_fingerprint": "r", "correct": 0, "trace_count_after": 0}],
        "no_update_repeat_records": [{"developmental_seed_index": 0, "rule_fingerprint": "r",
                                      "repeat_index": 0}],
        "theta_hashes": [{"developmental_seed_index": 0, "theta_hash": "t"}],
    }
    reference = deepcopy(local)
    assert compare_cell(local, reference)["all_fields_equal"]
    reference["seed_cell_records"][0]["trace_count_after"] = 1
    assert not compare_cell(local, reference)["all_fields_equal"]
    reference = deepcopy(local)
    reference["no_update_repeat_records"] *= 2
    assert not compare_cell(local, reference)["all_fields_equal"]
