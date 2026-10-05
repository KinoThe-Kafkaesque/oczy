"""The post-run audit must reject misleading trial records."""

import pytest

from scripts.summarize_context_preservation import validate_rows


@pytest.fixture
def row():
    return {"split": "confirmation", "stage": 2, "condition": "joint_restored", "seed": 0,
            "clients": ["amber"], "input": "lime", "target": "limevek", "category": "amber",
            "prompt_sha256": "fixed", "generated": " limevek ", "correct": True}


def test_audit_accepts_frozen_whitespace_normalization(row):
    validate_rows([row], [row])


@pytest.mark.parametrize("changes,reason", [({"correct": False}, "Score mismatch"),
                                         ({"target": "lime"}, "Target changed"),
                                         ({"prompt_sha256": "altered"}, "Prompt changed"),
                                         ({"seed": 1}, "Trial coverage mismatch")])
def test_audit_rejects_modified_trial(row, changes, reason):
    with pytest.raises(AssertionError, match=reason):
        validate_rows([{**row, **changes}], [row])


def test_audit_rejects_missing_and_duplicate_trials(row):
    for rows in ([], [row, row]):
        with pytest.raises(AssertionError, match="Trial coverage mismatch"):
            validate_rows(rows, [row])
