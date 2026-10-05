"""Guard tests for the frozen ``meta_cortex/v4-r20`` DEV instrument.

A frozen instrument is only frozen if loading rejects a changed byte.  These
tests build a real freeze once (module-scoped fixture) from a synthetic base
that *is* the pinned v3 public view, then assert that every tamper class fails
closed, that the amendments are actually carried, and that the untouched freeze
verifies.

The fixture uses a small task count (3 train / 2 tuning / 2 calibration per
family) for speed, but materializes the real v3 public DEV view when it exists
so the amendment surface is exercised on the genuine 105 public DEV tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oczy.experiments.meta_cortex.contracts import ProbeKind, TaskFamily
from oczy.experiments.meta_cortex.instrument_v4_r20 import (
    BARE_ANSWER,
    INHERITED_FROZEN_HASHES,
    INSTRUMENT_ID,
    MAX_NEW_TOKENS,
    V4R20BaseError,
    V4R20DefinitionError,
    _amended_prompt_registry_sha256,
    amend_task,
    materialize_v4_r20_definition,
    verify_base_public_view,
    verify_v4_r20_definition,
)

REPO = Path(__file__).resolve().parents[4]
V3_PUBLIC_ROOT = REPO / "experiments" / "r20-taskgen-v3-dev" / "instrument" / "public"
FROZEN_ORGAN_HASH = "a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea"
FROZEN_CHAT_TEMPLATE = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"

pytestmark = pytest.mark.skipif(
    not V3_PUBLIC_ROOT.is_dir(),
    reason="the pinned meta_cortex/v3 public DEV view is not present in this checkout",
)


def _kwargs(base_public_root: Path, **overrides):
    kwargs = {
        "base_public_root": base_public_root,
        "organ_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "organ_revision": "main",
        "organ_parameter_sha256": FROZEN_ORGAN_HASH,
        "chat_template_sha256": FROZEN_CHAT_TEMPLATE,
        "source_commit": "0" * 40,
        "source_archive_sha256": "",
        "root_seed": 20260709,
        "train_tasks_per_family": 30,
        "tuning_tasks_per_family": 5,
        "calibration_tasks_per_family": 30,
        "event_min": 2,
        "event_max": 5,
        "feature_dim": 896,
        "d_cortex": 64,
        "soft_bank_width": 3,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(scope="module")
def freeze(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("v4r20") / "instrument"
    materialize_v4_r20_definition(out=out, **_kwargs(V3_PUBLIC_ROOT))
    return out


def test_untouched_freeze_verifies(freeze: Path) -> None:
    definition = verify_v4_r20_definition(freeze)
    assert definition.task_counts == {
        "meta_train": 90,
        "meta_validation_tuning": 15,
        "meta_validation_calibration": 90,
    }
    assert definition.max_new_tokens == MAX_NEW_TOKENS == 128


def test_freeze_refuses_to_overwrite(freeze: Path) -> None:
    with pytest.raises(V4R20DefinitionError, match="already exists"):
        materialize_v4_r20_definition(out=freeze, **_kwargs(V3_PUBLIC_ROOT))


def test_wrong_base_is_refused_loudly(tmp_path: Path) -> None:
    """A base that is not the pinned v3 public view must raise V4R20BaseError."""
    with pytest.raises(V4R20BaseError, match="DEV_VIEW.json not found"):
        verify_base_public_view(tmp_path / "not-an-instrument")

    # A tampered copy of the real base is refused on the pinned value.
    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(V3_PUBLIC_ROOT.parent, copy)
    view_path = copy / "public" / "DEV_VIEW.json"
    data = json.loads(view_path.read_text())
    data["max_new_tokens"] = 64
    view_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with pytest.raises(V4R20BaseError, match="max_new_tokens"):
        verify_base_public_view(copy / "public")


def test_wrong_base_instrument_id_is_refused(tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(V3_PUBLIC_ROOT.parent, copy)
    view_path = copy / "public" / "DEV_VIEW.json"
    data = json.loads(view_path.read_text())
    data["instrument_id"] = "meta_cortex/v2"
    view_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with pytest.raises(V4R20BaseError, match="approved"):
        verify_base_public_view(copy / "public")


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
    index = len(raw) // 2
    target.write_bytes(raw[:index] + bytes([raw[index] ^ 0x01]) + raw[index + 1 :])
    with pytest.raises(V4R20DefinitionError, match="File hash mismatch"):
        verify_v4_r20_definition(copy)


def test_definition_self_hash_change_is_rejected(freeze: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    path = copy / "DEFINITION.json"
    data = json.loads(path.read_text())
    data["task_counts"]["meta_train"] = 999
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with pytest.raises(V4R20DefinitionError, match="self-hash mismatch"):
        verify_v4_r20_definition(copy)


def test_inherited_registry_drift_is_rejected(freeze: Path, tmp_path: Path) -> None:
    """Claiming a different scorer registry must not verify, even if re-signed."""
    import hashlib
    import shutil

    from oczy.experiments.meta_cortex.instrument_contracts import strict_canonical_json

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    path = copy / "DEFINITION.json"
    data = json.loads(path.read_text())
    data["scorer_registry_sha256"] = "0" * 64
    body = {k: v for k, v in data.items() if k != "definition_sha256"}
    data["definition_sha256"] = hashlib.sha256(strict_canonical_json(body)).hexdigest()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with pytest.raises(V4R20DefinitionError, match="registry drift"):
        verify_v4_r20_definition(copy)


def test_prompt_registry_must_be_the_amended_one(freeze: Path, tmp_path: Path) -> None:
    """Re-signing the definition with the v3 prompt registry must be rejected."""
    import hashlib
    import shutil

    from oczy.experiments.meta_cortex.instrument_contracts import strict_canonical_json

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    path = copy / "DEFINITION.json"
    data = json.loads(path.read_text())
    data["prompt_registry_sha256"] = data["base_prompt_registry_sha256"]
    body = {k: v for k, v in data.items() if k != "definition_sha256"}
    data["definition_sha256"] = hashlib.sha256(strict_canonical_json(body)).hexdigest()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with pytest.raises(V4R20DefinitionError, match="prompt_registry_sha256"):
        verify_v4_r20_definition(copy)


def test_unauthorized_max_new_tokens_is_rejected(freeze: Path, tmp_path: Path) -> None:
    import hashlib
    import shutil

    from oczy.experiments.meta_cortex.instrument_contracts import strict_canonical_json

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    path = copy / "DEFINITION.json"
    data = json.loads(path.read_text())
    data["max_new_tokens"] = 64
    body = {k: v for k, v in data.items() if k != "definition_sha256"}
    data["definition_sha256"] = hashlib.sha256(strict_canonical_json(body)).hexdigest()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with pytest.raises(V4R20DefinitionError, match="max_new_tokens"):
        verify_v4_r20_definition(copy)


def test_materialization_refuses_an_unauthorized_max_new_tokens(tmp_path: Path) -> None:
    with pytest.raises(V4R20DefinitionError, match="max_new_tokens"):
        materialize_v4_r20_definition(
            out=tmp_path / "instrument", **_kwargs(V3_PUBLIC_ROOT, max_new_tokens=32)
        )


def test_unlisted_extra_file_is_rejected(freeze: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    (copy / "public" / "stowaway.json").write_text("{}\n")
    with pytest.raises(V4R20DefinitionError, match="Unlisted file"):
        verify_v4_r20_definition(copy)


def test_sealed_directory_is_rejected(freeze: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "instrument"
    shutil.copytree(freeze, copy)
    (copy / "sealed").mkdir()
    with pytest.raises(V4R20DefinitionError, match="must not contain sealed"):
        verify_v4_r20_definition(copy)


def test_instrument_declares_itself_dev_only(freeze: Path) -> None:
    data = json.loads((freeze / "DEFINITION.json").read_text())
    assert data["dev_only"] is True
    assert data["sealed_payload_present"] is False
    assert data["meta_test_authorized"] is False
    assert data["instrument_id"] == INSTRUMENT_ID == "meta_cortex/v4-r20"
    assert data["taskgen_schema"] == "oczy/meta-cortex/taskgen/v2-dev"
    assert data["base_instrument_id"] == "meta_cortex/v3"
    assert data["naming"]["reason"]
    assert "collides" in data["naming"]["reason"]


def test_max_new_tokens_change_is_recorded_as_authorized(freeze: Path) -> None:
    record = json.loads((freeze / "DEFINITION.json").read_text())["max_new_tokens_change"]
    assert record["base_instrument_value"] == 32
    assert record["successor_value"] == 128
    assert record["authorized"] is True
    assert record["authorization"]["kanban_card"] == "t_37e96ee1"
    assert "NOT comparable" in record["comparability"]


def test_amendments_are_recorded_and_applied(freeze: Path) -> None:
    data = json.loads((freeze / "DEFINITION.json").read_text())
    amendments = data["amendments"]
    assert amendments["amendment_a"]["system_message"] == BARE_ANSWER
    assert set(amendments["amendment_b"]["descriptions"]) == {
        "permutation",
        "substitution",
        "conditional",
        "composition",
    }
    assert data["prompt_registry_sha256"] != data["base_prompt_registry_sha256"]
    assert data["prompt_registry_sha256"] == _amended_prompt_registry_sha256()
    for key, expected in INHERITED_FROZEN_HASHES.items():
        assert data[key] == expected, f"{key} drifted from the frozen value"


def test_every_public_probe_carries_amendment_a(freeze: Path) -> None:
    """Amendment A is on every probe of every public DEV task, system-first."""
    from oczy.experiments.meta_cortex.instrument import load_dev_view

    catalog = load_dev_view(freeze / "public").catalog
    checked = 0
    for task in (*catalog.meta_train, *catalog.meta_validation):
        for kind in ProbeKind:
            for probe in task.probes.by_kind(kind):
                assert probe.messages[0].role == "system"
                assert probe.messages[0].content == BARE_ANSWER
                checked += 1
    assert checked == 933


def test_amendment_b_rewrites_only_transformation_oracle_headers(freeze: Path) -> None:
    """Amendment B touches rule_transformation oracle headers and nothing else."""
    from oczy.experiments.meta_cortex.instrument import load_dev_view

    base = load_dev_view(V3_PUBLIC_ROOT).catalog
    amended = load_dev_view(freeze / "public").catalog
    rewritten = 0
    for base_task, amended_task in zip(
        (*base.meta_train, *base.meta_validation),
        (*amended.meta_train, *amended.meta_validation),
        strict=True,
    ):
        for kind in ProbeKind:
            for old, new in zip(
                base_task.probes.by_kind(kind), amended_task.probes.by_kind(kind), strict=True
            ):
                assert new.expected_response == old.expected_response
                if tuple(new.messages[1:]) == tuple(old.messages):
                    continue
                assert base_task.family == TaskFamily.RULE_TRANSFORMATION
                assert kind == ProbeKind.ORACLE_CONTEXT
                # Only the header line changed; examples and query are untouched.
                assert new.messages[2:] == old.messages[1:]
                assert new.messages[1].content.partition("\n")[2] == old.messages[0].content.partition("\n")[2]
                assert "Rule: composition with parameters" not in new.messages[1].content
                rewritten += 1
    assert rewritten == 35


def test_amend_task_rejects_an_already_amended_probe() -> None:
    from oczy.experiments.meta_cortex.instrument import load_dev_view

    task = load_dev_view(V3_PUBLIC_ROOT).catalog.meta_train[0]
    with pytest.raises(V4R20DefinitionError, match="already has a system"):
        amend_task(amend_task(task))


def test_task_files_are_reproducible_from_the_lineage(freeze: Path) -> None:
    first = verify_v4_r20_definition(freeze)
    second = verify_v4_r20_definition(freeze)
    assert first.definition_sha256 == second.definition_sha256
    assert first.catalog_sha256 == second.catalog_sha256
    assert first.prompt_registry_sha256 == second.prompt_registry_sha256
