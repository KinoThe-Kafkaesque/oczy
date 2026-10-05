"""Replay verdicts must reject missing, duplicate and altered rows."""

from scripts.regression_dev.contract import compare_rows


def test_complete_row_multiset_rejects_hidden_behavior_changes():
    rows = [{"generated": "x", "correct": False, "prompt_sha256": "a"},
            {"generated": "y", "correct": True, "prompt_sha256": "b"}]
    assert compare_rows(list(reversed(rows)), rows)
    assert not compare_rows(rows[:1], rows)
    assert not compare_rows([rows[0], rows[0]], rows)
    assert not compare_rows([{**rows[0], "generated": "z"}, rows[1]], rows)
    assert not compare_rows([{**rows[0], "prompt_sha256": "c"}, rows[1]], rows)
    assert not compare_rows([], [])


def test_action_comparison_checks_actual_outcomes():
    row = {"correct": False, "actual_files": {"a.txt": "before"}, "final": "wrong"}
    assert not compare_rows([{**row, "actual_files": {"a.txt": "after"}}], [row])
