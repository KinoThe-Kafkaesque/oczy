"""Adversarial checks for the new battery's information and action contracts."""

import json
from pathlib import Path

import pytest

from scripts.capability_validation_actions import execute, parse_action, run_case
from scripts.capability_validation_contract import messages, read_data
from scripts.prepare_capability_validation import build_data


@pytest.mark.parametrize("case", build_data()["actions"]["cases"], ids=lambda c: c["name"])
def test_real_oracle_actions_have_exact_outcomes(case):
    responses = iter([*case["expected_calls"], {"final": case["expected_final"]}])
    result = run_case(case, lambda _: json.dumps(next(responses)))
    assert result["correct"]
    assert result["actual_files"] == case["expected_files"]


def test_false_completion_fails_even_with_correct_final():
    case = build_data()["actions"]["cases"][1]
    result = run_case(case, lambda _: '{"final":"done"}')
    assert result["final_exact"]
    assert not result["filesystem_exact"]
    assert not result["tool_sequence_exact"]
    assert not result["correct"]


def test_wrong_case_path_fails_and_preserved_file_damage_is_visible():
    case = build_data()["actions"]["cases"][1]
    responses = iter([{"tool": "write_file", "arguments": {"path": "report.txt", "content": "ready"}}, {"final": "done"}])
    result = run_case(case, lambda _: json.dumps(next(responses)))
    assert not result["correct"]
    assert result["actual_files"] == {"report.txt": "ready"}


def test_extra_read_fails_despite_right_final_files():
    case = build_data()["actions"]["cases"][0]
    responses = iter([*case["expected_calls"], *case["expected_calls"], {"final": "delta\n"}])
    result = run_case(case, lambda _: json.dumps(next(responses)))
    assert result["filesystem_exact"] and result["final_exact"]
    assert not result["tool_sequence_exact"] and not result["correct"]


def test_copy_must_use_actual_read_result():
    case = build_data()["actions"]["cases"][3]
    prompts = []
    responses = iter([*case["expected_calls"], {"final": "done"}])
    def actor(prompt):
        prompts.append(json.loads(json.dumps(prompt)))
        return json.dumps(next(responses))
    result = run_case(case, actor)
    assert result["correct"]
    assert "CODE-731" not in json.dumps(prompts[0])
    assert "CODE-731" in prompts[1][-1]["content"]


@pytest.mark.parametrize("text", [
    '```json\n{"final":"done"}\n```', '{"final":"done","final":"other"}',
    '{"tool":"write_file","arguments":{"path":"a","content":"x","ignored":"x"}}',
    '{"tool":"write_file","arguments":{"path":"a","content":4}}',
    '{"tool":"shell","arguments":{"path":"a"}}',
])
def test_malformed_or_permissive_calls_rejected(text):
    with pytest.raises(ValueError):
        parse_action(text)


@pytest.mark.parametrize("path", ["../outside.txt", "/tmp/outside.txt", "."])
def test_workspace_escape_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        execute(tmp_path, {"tool": "write_file", "arguments": {"path": path, "content": "x"}})
    assert list(tmp_path.iterdir()) == []


def test_symlink_rejected(tmp_path):
    (tmp_path / "link").symlink_to(Path("/tmp"))
    with pytest.raises(ValueError):
        execute(tmp_path, {"tool": "write_file", "arguments": {"path": "link/file", "content": "x"}})


@pytest.mark.parametrize("phase,role", [("train", "probes"), ("train", "oracle"), ("restore", "training"), ("actions", "training")])
def test_phase_denies_forbidden_input_before_filesystem_access(tmp_path, phase, role):
    with pytest.raises(ValueError, match="cannot read"):
        read_data(tmp_path, {}, phase, role)


def test_composition_order_and_unrelated_scope_have_independent_targets():
    data = build_data()
    for stage in data["probes"]["stages"]:
        rows = stage["probes"]
        assert len(rows) == 16
        assert all(r["target"] == r["input"] for r in rows if r["category"] == "silver")
    stage2 = data["probes"]["stages"][1]["probes"]
    stage3 = data["probes"]["stages"][2]["probes"]
    def targets(rows, clients):
        return next(r["target"] for r in rows if r["input"] == "pear" and r["clients"] == clients)
    assert targets(stage2, ["amber", "cobalt"]) == "pearvekmip"
    assert targets(stage2, ["cobalt", "amber"]) == "pearmipvek"
    assert targets(stage3, ["amber", "cobalt"]) == "pearzulmip"
    assert targets(stage3, ["cobalt"]) == targets(stage2, ["cobalt"])
    all_training = {r["input"] for lesson in data["training"]["lessons"] for r in lesson["examples"]}
    assert not all_training & {r["input"] for r in stage2 + stage3}


def test_probe_gold_target_never_enters_model_prompt():
    prompt = messages({"clients": ["amber"], "input": "pear", "target": "HIDDEN-GOLD-TARGET"})
    assert "pear" in prompt[-1]["content"]
    assert "HIDDEN-GOLD-TARGET" not in json.dumps(prompt)
