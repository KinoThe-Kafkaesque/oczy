"""Guard tests for the frozen ``meta_cortex/v3`` DEV instrument (gate G1).

A frozen instrument is only frozen if loading rejects a changed byte.  These
tests build a real freeze once (module-scoped fixture), then assert that every
tamper class fails closed and that the untouched freeze verifies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oczy.experiments.meta_cortex.instrument_v3 import (
    V2_FROZEN_HASHES,
    V3DefinitionError,
    materialize_v3_definition,
    verify_v3_definition,
)

FROZEN_ORGAN_HASH = "a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea"
FROZEN_CHAT_TEMPLATE = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"

def _kwargs(**overrides):
    """Build materialization kwargs with explicit types (no dict-unpack union)."""
    kwargs = {
        "organ_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "organ_revision": "main",
        "organ_parameter_sha256": FROZEN_ORGAN_HASH,
        "chat_template_sha256": FROZEN_CHAT_TEMPLATE,
        "source_commit": "0" * 40,
        "source_archive_sha256": "",
        "root_seed": 20260709,
        "train_tasks_per_family": 3,
        "tuning_tasks_per_family": 2,
        "calibration_tasks_per_family": 2,
        "event_min": 2,
        "event_max": 5,
        "feature_dim": 896,
        "d_cortex": 64,
        "soft_bank_width": 3,
        "max_new_tokens": 32,
        "abstain_threshold": "0",
    }
    kwargs.update(overrides)
    return kwargs


def _materialize(out: Path, **overrides):
    return materialize_v3_definition(out=out, **_kwargs(**overrides))


@pytest.fixture(scope="module")
def freeze(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("v3") / "instrument"
    _materialize(out)
    return out


def test_untouched_freeze_verifies(freeze: Path) -> None:
    definition = verify_v3_definition(freeze)
    assert definition.task_counts == {
        "meta_train": 9,
        "meta_validation_tuning": 6,
        "meta_validation_calibration": 6,
    }


def test_freeze_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "instrument"
    _materialize(out)
    with pytest.raises(V3DefinitionError, match="already exists"):
        _materialize(out)


@pytest.mark.parametrize(
    "rel_path",
    [
        "public/tasks/meta_train.jsonl",
        "public/tasks/meta_validation_tuning.jsonl",
        "public/tasks/meta_validation_calibration.jsonl",
        "public/prompts.json",
        "public/scorers.json",
        "public/endpoints.json",
        "public/probe_counts.json",
        "public/audits/leakage_summary.json",
    ],
)
def test_task_and_registry_byte_change_is_rejected(freeze: Path, tmp_path: Path, rel_path: str) -> None:
    """A single changed byte in any listed file must fail verification."""
    import shutil

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    target = copy / rel_path
    raw = target.read_bytes()
    # Flip exactly one byte inside the payload, keeping the file size identical.
    index = len(raw) // 2
    mutated = raw[:index] + bytes([raw[index] ^ 0x01]) + raw[index + 1 :]
    target.write_bytes(mutated)
    with pytest.raises(V3DefinitionError, match="File hash mismatch"):
        verify_v3_definition(copy)


def test_definition_self_hash_change_is_rejected(freeze: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    path = copy / "DEFINITION.json"
    data = json.loads(path.read_text())
    data["task_counts"]["meta_train"] = 999
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with pytest.raises(V3DefinitionError, match="self-hash mismatch"):
        verify_v3_definition(copy)


def test_registry_hash_claim_change_is_rejected(freeze: Path, tmp_path: Path) -> None:
    """Claiming a different scorer registry must not verify, even if re-signed."""
    import shutil

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    path = copy / "DEFINITION.json"
    data = json.loads(path.read_text())
    data["scorer_registry_sha256"] = "0" * 64
    # Re-sign the definition so only the v2 registry-drift check can catch it.
    import hashlib

    from oczy.experiments.meta_cortex.instrument_contracts import strict_canonical_json

    body = {k: v for k, v in data.items() if k != "definition_sha256"}
    data["definition_sha256"] = hashlib.sha256(strict_canonical_json(body)).hexdigest()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with pytest.raises(V3DefinitionError, match="registry drift"):
        verify_v3_definition(copy)


def test_unlisted_extra_file_is_rejected(freeze: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    (copy / "public" / "stowaway.json").write_text("{}\n")
    with pytest.raises(V3DefinitionError, match="Unlisted file"):
        verify_v3_definition(copy)


def test_sealed_directory_is_rejected(freeze: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    (copy / "sealed").mkdir()
    with pytest.raises(V3DefinitionError, match="must not contain sealed"):
        verify_v3_definition(copy)


def test_v3_declares_itself_dev_only(freeze: Path) -> None:
    data = json.loads((freeze / "DEFINITION.json").read_text())
    assert data["dev_only"] is True
    assert data["sealed_payload_present"] is False
    assert data["meta_test_authorized"] is False
    assert data["instrument_id"] == "meta_cortex/v3"
    assert data["taskgen_schema"] == "oczy/meta-cortex/taskgen/v2-dev"


def test_v3_inherits_v2_registries_byte_for_byte(freeze: Path) -> None:
    data = json.loads((freeze / "DEFINITION.json").read_text())
    for key, expected in V2_FROZEN_HASHES.items():
        assert data[key] == expected, f"{key} drifted from the frozen v2 value"


def test_v3_records_its_predecessor_without_reusing_v1_tasks(freeze: Path) -> None:
    """v3 names v2 as predecessor instrument and states the comparability break."""
    dev_view = json.loads((freeze / "public" / "DEV_VIEW.json").read_text())
    assert dev_view["superceded_instrument_id"] == "meta_cortex/v2"
    generator = json.loads((freeze / "public" / "generator.json").read_text())
    assert generator["superseded_lineage"] == "oczy/meta-cortex/taskgen/v1-dev"
    assert "not comparable" in generator["comparability"]


def test_task_files_are_reproducible_from_the_lineage(freeze: Path) -> None:
    """verify_v3_definition rebuilds the catalog, so this asserts determinism."""
    first = verify_v3_definition(freeze)
    second = verify_v3_definition(freeze)
    assert first.definition_sha256 == second.definition_sha256
    assert first.catalog_sha256 == second.catalog_sha256
    assert first.support_bundle_sha256 == second.support_bundle_sha256
