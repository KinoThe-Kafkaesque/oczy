"""Approval boundaries, frozen targets, and actual rule-function agreement."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from oczy.experiments.meta_cortex import taskgen
from oczy.experiments.meta_cortex.contracts import ProbeKind, TaskFamily, TaskGeneratorConfig
from scripts import materialize_dev_prompt_repair as repair


def described_rule(template, p1, p2, operand):
    """Execute the operations stated in the approved English descriptions."""
    if template == "permutation":
        return "".join(reversed(operand))
    if template == "conditional":
        return operand + p1 if operand[:1] in tuple("aeiou") else p2 + operand
    characters = operand if template == "substitution" else "".join(reversed(operand))
    replacement = p1 if template == "substitution" else p2
    return "".join(replacement if char in "aeiou" else char for char in characters)


@pytest.fixture
def catalog():
    return taskgen.build_dev_catalog(TaskGeneratorConfig(20260709, 30, 5))


def test_descriptions_agree_with_actual_generator_over_dev_operands(monkeypatch):
    original = taskgen._ruletrans_oracle_probes
    templates = set()
    count = 0

    def inspect(stream, spec, operands, apply_rule):
        nonlocal count
        templates.add(spec["template"])
        # Every real DEV operand plus empty/case and both conditional branches.
        for operand in (*taskgen._OPERANDS, "", "apple", "boat", "Apple", "echo"):
            assert described_rule(spec["template"], spec["param1"], spec["param2"], operand) == apply_rule(operand)
            count += 1
        return original(stream, spec, operands, apply_rule)

    monkeypatch.setattr(taskgen, "_ruletrans_oracle_probes", inspect)
    generated = taskgen.build_dev_catalog(TaskGeneratorConfig(20260709, 30, 5))
    assert len(generated.meta_train) + len(generated.meta_validation) == 105
    assert templates == set(repair.ORACLE_DESCRIPTIONS)
    assert count >= 35 * 23


def test_all_dev_tasks_preserve_every_field_except_approved_messages(catalog):
    for task in (*catalog.meta_train, *catalog.meta_validation):
        a = repair.amend_task(task, "v3")
        b = repair.amend_task(task, "v4")
        assert replace(a, probes=task.probes) == task
        assert replace(b, probes=task.probes) == task
        for kind in ProbeKind:
            for old, new_a, new_b in zip(task.probes.by_kind(kind), a.probes.by_kind(kind), b.probes.by_kind(kind), strict=True):
                assert replace(new_a, messages=old.messages) == old
                assert replace(new_b, messages=old.messages) == old
                assert new_a.messages[0].role == "system"
                assert new_a.messages[0].content == repair.BARE_ANSWER
                assert new_a.messages[1:] == old.messages
                if task.family == TaskFamily.RULE_TRANSFORMATION and kind == ProbeKind.ORACLE_CONTEXT:
                    assert new_b.messages[0] == new_a.messages[0]
                    assert new_b.messages[2:] == new_a.messages[2:]
                    assert new_b.messages[1].content.partition("\n")[2] == old.messages[0].content.partition("\n")[2]
                    template, p1, p2, _ = repair.parse_oracle(old.messages[0].content)
                    operand = old.messages[-1].content.removeprefix("Given this rule, what is the output for: ").removesuffix("?")
                    assert described_rule(template, p1, p2, operand) == old.expected_response
                else:
                    assert new_b == new_a


def test_system_instruction_precedes_teaching_transcript():
    from oczy.experiments.meta_cortex.contracts import DialogueMessage
    from scripts.probe_r20_articulation import with_teaching_context

    system = DialogueMessage("system", repair.BARE_ANSWER)
    query = DialogueMessage("user", "query")
    transcript = (DialogueMessage("user", "teaching"),)
    assert with_teaching_context((system, query), transcript) == (system, *transcript, query)
    assert with_teaching_context((query,), transcript) == (*transcript, query)


@pytest.fixture
def fake_base(monkeypatch, catalog):
    binding = SimpleNamespace(dev_view_sha256=repair.BASE_DEV_HASH,
                              definition_sha256="a" * 64, prompt_registry_sha256="b" * 64,
                              scorer_registry_sha256="c" * 64, endpoint_registry_sha256="d" * 64,
                              organ_hash="e" * 64)
    base = SimpleNamespace(binding=binding, catalog=catalog)
    monkeypatch.setattr(repair, "verified_base", lambda _: base)
    return base


def test_amendments_require_order_and_do_not_overwrite(tmp_path, fake_base):
    public = tmp_path / "original/public"
    a, b = tmp_path / "v3", tmp_path / "v4"
    with pytest.raises(ValueError, match="requires Amendment A"):
        repair.materialize(public, b, "v4")
    with pytest.raises(ValueError, match="original frozen"):
        repair.materialize(public, public / "v3", "v3")
    ma = repair.materialize(public, a, "v3")
    mb = repair.materialize(public, b, "v4", a)
    assert mb["parent_sha256"] == ma["manifest_sha256"]
    assert mb["tasks_sha256"] != ma["tasks_sha256"]
    assert mb["scorer_sha256"] == ma["scorer_sha256"]
    with pytest.raises(FileExistsError):
        repair.materialize(public, a, "v3")


@pytest.mark.parametrize("target", ["TASKS.json", "MANIFEST.json", "PARENT_MANIFEST.json"])
def test_tampering_is_rejected(tmp_path, fake_base, target):
    public = tmp_path / "original/public"
    a, b = tmp_path / "v3", tmp_path / "v4"
    repair.materialize(public, a, "v3")
    repair.materialize(public, b, "v4", a)
    file = b / target
    data = json.loads(file.read_text())
    data["unauthorized"] = True
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        repair.load_repair(public, b)


def test_unknown_rule_or_duplicate_amendment_rejected(catalog):
    with pytest.raises(ValueError, match="Unrecognized"):
        repair.parse_oracle("Rule: arbitrary Python")
    with pytest.raises(ValueError, match="already has a system"):
        repair.amend_task(repair.amend_task(catalog.meta_train[0], "v3"), "v4")
