"""Focused tests for the durable legacy calibration ledger."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_KAGGLE_DIR = str(Path(__file__).resolve().parents[2] / "infrastructure" / "kaggle")
if _KAGGLE_DIR not in sys.path:
    sys.path.insert(0, _KAGGLE_DIR)

import calibration_ledger as ledger  # type: ignore[import-not-found]  # noqa: E402


def _job(key: str) -> dict[str, Any]:
    dev = int(key[1])
    start = int(key.split("-t", 1)[1].split("-", 1)[0])
    return {
        "key": key,
        "dev_seed_index": dev,
        "task_start": start,
        "task_end": start + 5,
        "state": "pending",
        "metadata_candidate": {
            "requested_remote_id": "owner/oczy-r20-cal-s7",
            "actual_remote_id": "owner/oczy-r20-s7",
            "title": "Oczy R20 S7",
            "slug_mismatch": True,
        },
        "attempts": [],
        "result": None,
        "invalid_results": [],
    }


def test_load_job_records_title_derived_remote_slug(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    job_dir = jobs_root / "d0-t15-20"
    job_dir.mkdir(parents=True)
    spec = {
        "arguments": [
            "collect-calibration-shard",
            "--dev-seed-index",
            "0",
            "--task-start",
            "15",
            "--task-end",
            "20",
            "--organ-hash",
            "a" * 64,
        ],
        "source_commit": "b" * 40,
        "source_archive_sha256": "c" * 64,
    }
    metadata = {
        "id": "owner/oczy-r20-cal-s7",
        "title": "Oczy R20 S7",
    }
    (job_dir / "job_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (job_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (job_dir / "run.py").write_text("print('ok')\n", encoding="utf-8")

    loaded = ledger._load_job(job_dir, jobs_root)

    candidate = loaded["metadata_candidate"]
    assert candidate["requested_remote_id"] == "owner/oczy-r20-cal-s7"
    assert candidate["actual_remote_id"] == "owner/oczy-r20-s7"
    assert candidate["slug_mismatch"] is True


def test_refresh_local_keeps_conflicting_valid_shards_explicit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    public = tmp_path / "public"
    public.mkdir()
    first = results / "a" / "shard-d0-t15-20.json"
    second = results / "b" / "shard-d0-t15-20.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('{"run":1}\n', encoding="utf-8")
    second.write_text('{"run":2}\n', encoding="utf-8")
    state = {
        "public_root": str(public),
        "results_root": str(tmp_path / "durable"),
        "jobs": {"d0-t15-20": _job("d0-t15-20")},
    }
    monkeypatch.setattr(ledger, "_validate_shard", lambda path, root: None)

    ledger.refresh_local(state, results)

    job = state["jobs"]["d0-t15-20"]
    assert job["state"] == "conflict"
    assert job["result"] is None
    assert {item["sha256"] for item in job["invalid_results"]} == {
        ledger._sha256(first),
        ledger._sha256(second),
    }


def test_refresh_local_deduplicates_same_bytes_across_roots(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    source_a.mkdir()
    source_b.mkdir()
    first = source_a / "shard-d1-t05-10.json"
    second = source_b / "shard-d1-t05-10.json"
    first.write_text('{"same":true}\n', encoding="utf-8")
    second.write_bytes(first.read_bytes())
    state = {
        "public_root": str(tmp_path / "public"),
        "results_root": str(tmp_path / "durable"),
        "jobs": {"d1-t05-10": _job("d1-t05-10")},
    }
    monkeypatch.setattr(ledger, "_validate_shard", lambda path, root: None)

    ledger.refresh_local(state, source_a, source_b)

    job = state["jobs"]["d1-t05-10"]
    assert job["state"] == "succeeded"
    assert job["result"]["sha256"] == ledger._sha256(first)
    assert job["invalid_results"] == []


def test_refresh_kaggle_maps_actual_title_slug(monkeypatch: Any) -> None:
    state = {
        "results_root": "/tmp/results",
        "jobs": {"d0-t15-20": _job("d0-t15-20")},
        "unmapped_remote_attempts": [],
    }
    csv_output = (
        "ref,title,author,lastRunTime,totalVotes\n"
        "owner/oczy-r20-s7,Oczy R20 S7,Owner,2026-07-15 10:00:00,0\n"
    )

    def fake_run(args: list[str], timeout: int = 120) -> tuple[int, str]:
        _ = timeout
        if args[1] == "list":
            return 0, csv_output
        if args[1] == "status":
            return 0, 'owner/oczy-r20-s7 has status "KernelWorkerStatus.RUNNING"\n'
        raise AssertionError(args)

    monkeypatch.setattr(ledger, "_run_kaggle", fake_run)

    ledger.refresh_kaggle(state, collect=False)

    job = state["jobs"]["d0-t15-20"]
    assert job["state"] == "running"
    assert job["attempts"][0]["remote_id"] == "owner/oczy-r20-s7"
    assert job["attempts"][0]["requested_remote_id"] == "owner/oczy-r20-cal-s7"
    assert state["unmapped_remote_attempts"] == []


def test_record_attempt_rejects_title_slug_mismatch() -> None:
    state = {
        "jobs": {
            "d0-t15-20": _job("d0-t15-20"),
        }
    }

    try:
        ledger.record_attempt(
            state,
            job_key="d0-t15-20",
            remote_id="owner/oczy-r20-cal-s8",
            requested_remote_id="owner/oczy-r20-cal-s8",
            title="Oczy R20 S8",
        )
    except ledger.LedgerError as exc:
        assert "resolves to 'owner/oczy-r20-s8'" in str(exc)
    else:
        raise AssertionError("expected slug mismatch to be rejected")


def test_record_attempt_persists_exact_mapping() -> None:
    state = {
        "jobs": {
            "d0-t15-20": _job("d0-t15-20"),
        }
    }

    ledger.record_attempt(
        state,
        job_key="d0-t15-20",
        remote_id="owner/oczy-r20-cal-s8",
        requested_remote_id="owner/oczy-r20-cal-s8",
        title="Oczy R20 Cal S8",
    )

    job = state["jobs"]["d0-t15-20"]
    assert job["state"] == "submitting"
    assert job["attempts"][0]["remote_id"] == "owner/oczy-r20-cal-s8"
    assert job["attempts"][0]["source"] == "local-registration"


def test_read_only_lock_does_not_rewrite_state(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    original = {
        "schema_version": ledger.SCHEMA_VERSION,
        "updated_at": 123.0,
    }
    path.write_text(json.dumps(original) + "\n", encoding="utf-8")
    before = path.read_bytes()

    with ledger._locked_state(path, write=False) as state:
        state["updated_at"] = 999.0

    assert path.read_bytes() == before
