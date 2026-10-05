"""Mutation checks against the completed, archived run; no model execution."""

import json
import shutil
from pathlib import Path

import pytest

from scripts.diversity_dev.audit import audit

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "experiments_logs/artifacts/2026-09-13_scoped_diversity_dev_v3/run"


@pytest.fixture
def run_copy(tmp_path):
    if not ARCHIVE.exists():
        pytest.skip("Completed archive not present")
    return Path(shutil.copytree(ARCHIVE, tmp_path / "run"))


def test_archived_run_audits_without_model_calls(run_copy):
    result = audit(REPO / "experiments/scoped-diversity-dev-v3", run_copy)
    assert result["verification"] == {"passed": True, "probe_rows": 608, "teaching_fits": 81, "optimizer_updates": 144}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "score", "prompt", "target"])
def test_audit_rejects_altered_evaluation(run_copy, mutation):
    path = run_copy / "evaluate/results.json"
    result = json.loads(path.read_text())
    rows = result["rows"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0].copy()
    elif mutation == "score":
        rows[0]["correct"] = not rows[0]["correct"]
    elif mutation == "prompt":
        rows[0]["prompt_sha256"] = "changed"
    else:
        rows[0]["target"] = rows[0]["generated"]
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        audit(REPO / "experiments/scoped-diversity-dev-v3", run_copy)


def test_audit_rejects_different_teaching_schedule(run_copy):
    path = run_copy / "train/trajectory.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["group"] = 1
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(AssertionError):
        audit(REPO / "experiments/scoped-diversity-dev-v3", run_copy)
