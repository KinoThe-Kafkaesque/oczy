"""Scientific boundary and curriculum checks before the DEV-v2 run."""

import json

import pytest

from scripts.capability_validation_contract import messages as old_messages
from scripts.context_preservation_contract import capacity_gate, messages, read_data
from scripts.prepare_context_preservation import build_data


def test_crossed_teaching_requires_context_and_excludes_confirmation_words():
    data = build_data()
    train = data["capacity_training"]["rows"]
    for word in {r["input"] for r in train}:
        rows = [r for r in train if r["input"] == word]
        assert len(rows) == 3 and len({r["target"] for r in rows}) == 3
    assert all(r["clients"] != ["quartz"] for r in train)
    assert not {r["input"] for r in train} & {r["input"] for s in data["probes"]["confirmation"] for r in s["rows"]}


def test_v1_prompt_is_an_exact_paired_control():
    row = build_data()["probes"]["calibration"][0]
    assert messages(row, "v1", "rules") == old_messages(row, "rules")


def test_adjacent_query_and_system_repairs_change_one_field_only():
    row = build_data()["probes"]["calibration"][0]
    a, b, c = [messages(row, version, "rules") for version in ("v1", "query_v2", "concise_v2")]
    assert a[0] == b[0] and a[1] == b[1] and a[2] != b[2]
    assert b[0] != c[0] and b[1:] == c[1:]
    assert b[-1]["content"].splitlines()[-2] == "Input: pear"


@pytest.mark.parametrize("phase,role", [("train_capacity", "probes"), ("train_capacity", "oracle"),
                                      ("train_correction", "capacity_training"), ("restore_capacity", "capacity_training")])
def test_phase_denies_inputs_before_reading_files(tmp_path, phase, role):
    with pytest.raises(ValueError, match="cannot read"):
        read_data(tmp_path, {}, phase, role)


def test_expected_answer_is_not_in_probe_prompt():
    prompt = messages({"clients": ["arbitrary-client"], "input": "word", "target": "SECRET-GOLD"})
    assert "SECRET-GOLD" not in json.dumps(prompt)
    assert "arbitrary-client" in prompt[-1]["content"]


def test_capacity_gate_rejects_a_failure_omission_duplicate_and_missing_scope():
    data = build_data()["probes"]["confirmation"][0]["rows"]
    rows = [{**r, "condition": "joint_restored", "seed": s, "correct": True} for s in (0, 1, 2) for r in data]
    assert capacity_gate(rows, [0, 1, 2])
    assert not capacity_gate([{**r, "correct": i != 0} for i, r in enumerate(rows)], [0, 1, 2])
    assert not capacity_gate(rows[:-1], [0, 1, 2])
    assert not capacity_gate([*rows[:-1], rows[0]], [0, 1, 2])
    assert not capacity_gate([r for r in rows if r["category"] != "quartz"], [0, 1, 2])


def test_correction_changes_only_amber_targets():
    a, b = [s["rows"] for s in build_data()["probes"]["confirmation"]]
    for before, after in zip(a, b, strict=True):
        assert before["input"] == after["input"] and before["clients"] == after["clients"]
        assert (before["target"] != after["target"]) == (before["category"] == "amber")
