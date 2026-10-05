"""Keep acquisition failures distinct from actual loss of previously correct knowledge."""

import pytest

from scripts.summarize_capability_validation import retention


def rows(flags, *, target="pearvek"):
    return [{"clients": ["amber"], "input": word, "target": target, "correct": flag}
            for word, flag in zip(("pear", "fig"), flags, strict=True)]


def test_never_acquired_is_not_counted_as_forgetting():
    result = retention(rows([False, False]), rows([False, False]))
    assert result["previously_correct"] == 0
    assert result["retained"] == 0
    assert result["lost_inputs"] == []


def test_only_previously_correct_lost_answers_count_as_forgetting():
    result = retention(rows([True, False]), rows([False, True]))
    assert result["previously_correct"] == 1
    assert result["retained"] == 0
    assert result["lost_inputs"] == ["pear"]
    assert result["all_old"] == result["all_after"] == {"correct": 1, "total": 2}


def test_new_target_cannot_be_called_unrelated_rule_retention():
    with pytest.raises(ValueError, match="changed target"):
        retention(rows([True, True]), rows([True, True], target="pearzul"))
