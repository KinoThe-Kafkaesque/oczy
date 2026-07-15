"""Durable local ledger for legacy R20 DEV calibration shards.

This is a tracking and collection bridge for the pre-scheduler R20 shard jobs.
It never submits, stops, or deletes remote work.  New campaigns should use
``parallel_scheduler.py`` directly.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "oczy/r20-calibration-ledger/v1"
DEFAULT_CAMPAIGN_DIR = (
    Path.home()
    / ".local/state/oczy/remote-queue/campaigns/r20-dev-calibration-v1"
)
DEFAULT_STATE = DEFAULT_CAMPAIGN_DIR / "ledger.json"
_JOB_DIR_RE = re.compile(r"^d(?P<dev>[0-4])-t(?P<start>\d{2})-(?P<end>\d{2})$")
_SHARD_RE = re.compile(r"^shard-d(?P<dev>[0-4])-t(?P<start>\d{2})-(?P<end>\d{2})\.json$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_REMOTE_STATES = ("RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCEL")


class LedgerError(RuntimeError):
    """Invalid ledger input or state."""


def _now() -> float:
    return time.time()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _title_slug(title: str) -> str:
    """Mirror Kaggle's clean-URL title slug derivation."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@contextmanager
def _locked_state(
    path: Path, *, write: bool = True
) -> Iterator[dict[str, Any]]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(
            lock_fh.fileno(),
            fcntl.LOCK_EX if write else fcntl.LOCK_SH,
        )
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("schema_version") != SCHEMA_VERSION:
                raise LedgerError(
                    f"unsupported ledger schema: {state.get('schema_version')!r}"
                )
        else:
            state = {}
        yield state
        if write and state:
            state["updated_at"] = _now()
            _atomic_write_json(path, state)


def _parse_job_dir(path: Path) -> tuple[str, int, int, int]:
    match = _JOB_DIR_RE.fullmatch(path.name)
    if match is None:
        raise LedgerError(f"invalid shard job directory name: {path.name!r}")
    dev = int(match.group("dev"))
    start = int(match.group("start"))
    end = int(match.group("end"))
    if end - start != 5 or start % 5 != 0 or end > 90:
        raise LedgerError(f"invalid five-task range in {path.name!r}")
    return path.name, dev, start, end


def _job_spec_arg_map(spec: dict[str, Any]) -> dict[str, str]:
    raw = spec.get("arguments")
    if not isinstance(raw, list):
        raise LedgerError("job_spec.arguments must be a list")
    args = [str(item) for item in raw]
    result: dict[str, str] = {}
    index = 1 if args and args[0] == "collect-calibration-shard" else 0
    while index < len(args):
        flag = args[index]
        if not flag.startswith("--") or index + 1 >= len(args):
            raise LedgerError(f"malformed job argument list near {flag!r}")
        result[flag] = args[index + 1]
        index += 2
    return result


def _load_job(path: Path, durable_jobs_root: Path) -> dict[str, Any]:
    key, dev, start, end = _parse_job_dir(path)
    spec_path = path / "job_spec.json"
    metadata_path = path / "kernel-metadata.json"
    run_path = path / "run.py"
    for required in (spec_path, metadata_path, run_path):
        if not required.is_file():
            raise LedgerError(f"missing job artifact: {required}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    args = _job_spec_arg_map(spec)
    expected = {
        "--dev-seed-index": str(dev),
        "--task-start": str(start),
        "--task-end": str(end),
    }
    for flag, value in expected.items():
        if args.get(flag) != value:
            raise LedgerError(
                f"{key}: {flag}={args.get(flag)!r}, expected {value!r}"
            )

    requested_id = str(metadata.get("id", ""))
    title = str(metadata.get("title", ""))
    if "/" not in requested_id or not title:
        raise LedgerError(f"{key}: invalid kernel metadata identity")
    owner = requested_id.split("/", 1)[0]
    actual_id = f"{owner}/{_title_slug(title)}"
    organ_hash = args.get("--organ-hash", "")
    if not _HEX64_RE.fullmatch(organ_hash):
        raise LedgerError(f"{key}: invalid --organ-hash")

    durable_dir = durable_jobs_root / key
    return {
        "key": key,
        "dev_seed_index": dev,
        "task_start": start,
        "task_end": end,
        "state": "pending",
        "job_dir": str(durable_dir),
        "job_spec_sha256": _sha256(spec_path),
        "run_sha256": _sha256(run_path),
        "source_commit": str(spec.get("source_commit", "")),
        "source_archive_sha256": str(spec.get("source_archive_sha256", "")),
        "calibration_view_path": args.get("--calibration-view", ""),
        "organ_hash": organ_hash,
        "metadata_candidate": {
            "requested_remote_id": requested_id,
            "actual_remote_id": actual_id,
            "title": title,
            "slug_mismatch": requested_id != actual_id,
        },
        "attempts": [],
        "result": None,
        "invalid_results": [],
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise LedgerError(f"directory not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def initialize(
    *,
    state_path: Path,
    jobs_dir: Path,
    results_dir: Path,
    public_root: Path,
    campaign_dir: Path,
) -> dict[str, Any]:
    durable_jobs = campaign_dir / "jobs"
    durable_results = campaign_dir / "results"
    durable_public = campaign_dir / "instrument/public"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    _copy_tree(jobs_dir, durable_jobs)
    _copy_tree(results_dir, durable_results)
    _copy_tree(public_root, durable_public)

    jobs: dict[str, Any] = {}
    for path in sorted(durable_jobs.iterdir()):
        if not path.is_dir():
            continue
        job = _load_job(path, durable_jobs)
        jobs[job["key"]] = job
    if len(jobs) != 90:
        raise LedgerError(f"expected 90 shard jobs, found {len(jobs)}")

    state = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "r20-dev-calibration-v1",
        "created_at": _now(),
        "updated_at": _now(),
        "campaign_dir": str(campaign_dir),
        "public_root": str(durable_public),
        "results_root": str(durable_results),
        "jobs": jobs,
        "unmapped_remote_attempts": [],
    }
    _atomic_write_json(state_path, state)
    return state


def _result_key(path: Path) -> tuple[str, int, int, int] | None:
    match = _SHARD_RE.fullmatch(path.name)
    if match is None:
        return None
    dev = int(match.group("dev"))
    start = int(match.group("start"))
    end = int(match.group("end"))
    return f"d{dev}-t{start:02d}-{end:02d}", dev, start, end


def _validate_shard(path: Path, public_root: Path) -> str | None:
    try:
        from oczy.experiments.meta_cortex.calibration import load_calibration_shard
        from oczy.experiments.meta_cortex.instrument import load_calibration_view

        view = load_calibration_view(public_root)
        load_calibration_shard(path, view)
    except Exception as exc:  # exact message is part of the local audit trail
        return f"{type(exc).__name__}: {exc}"
    return None


def refresh_local(state: dict[str, Any], *roots: Path) -> None:
    public_root = Path(state["public_root"])
    durable_results = Path(state["results_root"])
    candidates: dict[str, dict[str, list[Path]]] = {}

    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("shard-d*-t*.json")):
            parsed = _result_key(path)
            if parsed is None:
                continue
            key, _, _, _ = parsed
            if key not in state["jobs"]:
                continue
            digest = _sha256(path)
            candidates.setdefault(key, {}).setdefault(digest, []).append(path)

    observed_at = _now()
    for key, job in state["jobs"].items():
        job["invalid_results"] = []
        valid: dict[str, list[Path]] = {}
        for digest, paths in candidates.get(key, {}).items():
            representative = paths[0]
            error = _validate_shard(representative, public_root)
            if error is None:
                valid[digest] = paths
                continue
            job["invalid_results"].append(
                {
                    "paths": sorted(str(path) for path in paths),
                    "sha256": digest,
                    "error": error,
                    "observed_at": observed_at,
                }
            )

        if len(valid) > 1:
            job["result"] = None
            job["state"] = "conflict"
            for digest, paths in sorted(valid.items()):
                job["invalid_results"].append(
                    {
                        "paths": sorted(str(path) for path in paths),
                        "sha256": digest,
                        "error": "conflicting valid shard bytes for one job",
                        "observed_at": observed_at,
                    }
                )
            continue

        if len(valid) == 1:
            digest, paths = next(iter(valid.items()))
            source = min(paths, key=lambda path: str(path))
            destination = durable_results / key / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.resolve() != source.resolve():
                shutil.copy2(source, destination)
            job["result"] = {
                "path": str(destination),
                "sha256": digest,
                "byte_size": source.stat().st_size,
                "validated_at": observed_at,
                "source_paths": sorted(str(path) for path in paths),
            }
            job["state"] = "succeeded"
            continue

        job["result"] = None
        if job["invalid_results"]:
            job["state"] = "invalid_result"
        elif job["state"] in ("succeeded", "conflict", "invalid_result"):
            job["state"] = "pending"


def _run_kaggle(args: list[str], timeout: int = 120) -> tuple[int, str]:
    completed = subprocess.run(
        ["kaggle", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return completed.returncode, completed.stdout + completed.stderr


def _remote_state(ref: str) -> tuple[str, str | None]:
    code, output = _run_kaggle(["kernels", "status", ref])
    if code != 0:
        return "not_found", output.strip() or None
    for remote_state in _REMOTE_STATES:
        if remote_state in output:
            return remote_state.lower(), None
    return "unknown", output.strip() or None


def _upsert_attempt(job: dict[str, Any], remote_id: str, state: str, error: str | None) -> None:
    attempts = job["attempts"]
    attempt = next((item for item in attempts if item["remote_id"] == remote_id), None)
    if attempt is None:
        candidate = job["metadata_candidate"]
        attempt = {
            "remote_id": remote_id,
            "requested_remote_id": candidate["requested_remote_id"],
            "title": candidate["title"],
            "provider": "kaggle",
            "source": "remote-discovery",
            "first_observed_at": _now(),
            "last_observed_at": _now(),
            "state": state,
            "error": error,
            "history": [],
        }
        attempts.append(attempt)
    if attempt["state"] != state or attempt.get("error") != error:
        attempt["history"].append(
            {"at": _now(), "state": state, "error": error}
        )
    attempt["state"] = state
    attempt["error"] = error
    attempt["last_observed_at"] = _now()
    if job["state"] != "succeeded":
        job["state"] = {
            "running": "running",
            "queued": "running",
            "complete": "complete_uncollected",
            "error": "failed",
            "cancel": "failed",
        }.get(state, job["state"])


def refresh_kaggle(state: dict[str, Any], *, collect: bool) -> None:
    code, output = _run_kaggle(
        ["kernels", "list", "--mine", "--page-size", "200", "--csv"]
    )
    if code != 0:
        raise LedgerError(f"kaggle kernels list failed: {output.strip()}")

    by_remote: dict[str, dict[str, Any]] = {}
    for job in state["jobs"].values():
        candidate = job["metadata_candidate"]
        by_remote[candidate["actual_remote_id"]] = job
        by_remote[candidate["requested_remote_id"]] = job
        for attempt in job["attempts"]:
            by_remote[attempt["remote_id"]] = job

    observed_orphans: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(output)):
        ref = str(row.get("ref", "")).strip()
        if not ref or "r20" not in ref:
            continue
        state_name, error = _remote_state(ref)
        job = by_remote.get(ref)
        if job is None:
            observed_orphans.append(
                {
                    "remote_id": ref,
                    "state": state_name,
                    "error": error,
                    "last_run_time": row.get("lastRunTime"),
                    "observed_at": _now(),
                }
            )
            continue
        _upsert_attempt(job, ref, state_name, error)
        if collect and state_name == "complete" and job["state"] != "succeeded":
            attempt_dir = Path(state["results_root"]) / "attempts" / ref.split("/")[-1]
            attempt_dir.mkdir(parents=True, exist_ok=True)
            collect_code, collect_output = _run_kaggle(
                ["kernels", "output", ref, "--path", str(attempt_dir), "--force"],
                timeout=1800,
            )
            if collect_code != 0:
                _upsert_attempt(job, ref, "complete", collect_output.strip())
    state["unmapped_remote_attempts"] = observed_orphans


def record_attempt(
    state: dict[str, Any],
    *,
    job_key: str,
    remote_id: str,
    requested_remote_id: str,
    title: str,
) -> None:
    job = state["jobs"].get(job_key)
    if job is None:
        raise LedgerError(f"unknown shard job: {job_key}")
    if "/" not in remote_id or "/" not in requested_remote_id:
        raise LedgerError("remote IDs must use owner/slug form")
    owner = remote_id.split("/", 1)[0]
    derived = f"{owner}/{_title_slug(title)}"
    if derived != remote_id:
        raise LedgerError(
            f"title {title!r} resolves to {derived!r}, not {remote_id!r}"
        )
    for other in state["jobs"].values():
        for attempt in other["attempts"]:
            if attempt["remote_id"] == remote_id:
                raise LedgerError(
                    f"remote ID {remote_id!r} is already registered to {other['key']}"
                )
    observed_at = _now()
    job["attempts"].append(
        {
            "remote_id": remote_id,
            "requested_remote_id": requested_remote_id,
            "title": title,
            "provider": "kaggle",
            "source": "local-registration",
            "first_observed_at": observed_at,
            "last_observed_at": observed_at,
            "state": "submitting",
            "error": None,
            "history": [
                {
                    "at": observed_at,
                    "state": "submitting",
                    "error": None,
                }
            ],
        }
    )
    if job["state"] != "succeeded":
        job["state"] = "submitting"


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    active: list[dict[str, Any]] = []
    mismatches = 0
    invalid_results = 0
    for job in state["jobs"].values():
        counts[job["state"]] = counts.get(job["state"], 0) + 1
        if job["metadata_candidate"]["slug_mismatch"]:
            mismatches += 1
        invalid_results += len(job["invalid_results"])
        for attempt in job["attempts"]:
            if attempt["state"] in ("running", "queued"):
                active.append(
                    {
                        "job": job["key"],
                        "remote_id": attempt["remote_id"],
                        "state": attempt["state"],
                    }
                )
    return {
        "schema_version": "oczy/r20-calibration-ledger-summary/v1",
        "campaign_id": state["campaign_id"],
        "state_path": state.get("state_path", ""),
        "counts": counts,
        "active_attempts": active,
        "slug_mismatches": mismatches,
        "invalid_results": invalid_results,
        "unmapped_remote_attempts": state["unmapped_remote_attempts"],
        "updated_at": state["updated_at"],
    }


def _print_summary(summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    print(f"campaign={summary['campaign_id']}")
    print(" ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print(f"active_attempts={len(summary['active_attempts'])}")
    for item in summary["active_attempts"]:
        print(f"  {item['job']} {item['state']} {item['remote_id']}")
    print(f"slug_mismatches={summary['slug_mismatches']}")
    print(f"invalid_results={summary['invalid_results']}")
    print(f"unmapped_remote_attempts={len(summary['unmapped_remote_attempts'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Snapshot a legacy campaign into durable storage")
    init.add_argument("--jobs-dir", type=Path, required=True)
    init.add_argument("--results-dir", type=Path, required=True)
    init.add_argument("--public-root", type=Path, required=True)
    init.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)

    refresh = sub.add_parser("refresh", help="Refresh local results and optional Kaggle state")
    refresh.add_argument("--results-dir", type=Path, action="append", default=[])
    refresh.add_argument("--kaggle", action="store_true")
    refresh.add_argument("--collect", action="store_true")

    record = sub.add_parser(
        "record-attempt",
        help="Register a job-to-remote mapping before submission",
    )
    record.add_argument("--job", required=True)
    record.add_argument("--remote-id", required=True)
    record.add_argument("--requested-remote-id")
    record.add_argument("--title", required=True)

    status = sub.add_parser("status", help="Print ledger status without remote actions")
    status.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state_path = args.state.expanduser().resolve()
    if args.command == "init":
        state = initialize(
            state_path=state_path,
            jobs_dir=args.jobs_dir.resolve(),
            results_dir=args.results_dir.resolve(),
            public_root=args.public_root.resolve(),
            campaign_dir=args.campaign_dir.expanduser().resolve(),
        )
        with _locked_state(state_path) as locked:
            locked.update(state)
            locked["state_path"] = str(state_path)
            refresh_local(locked, Path(locked["results_root"]), args.results_dir.resolve())
        print(f"initialized={state_path}")
        return 0

    if not state_path.is_file():
        raise LedgerError(f"ledger not found: {state_path}")

    if args.command == "record-attempt":
        requested_id = args.requested_remote_id or args.remote_id
        with _locked_state(state_path) as state:
            record_attempt(
                state,
                job_key=args.job,
                remote_id=args.remote_id,
                requested_remote_id=requested_id,
                title=args.title,
            )
            state["state_path"] = str(state_path)
        print(f"registered={args.job} remote_id={args.remote_id}")
        return 0

    if args.command == "refresh":
        with _locked_state(state_path) as state:
            roots = [Path(state["results_root"]), *[path.resolve() for path in args.results_dir]]
            refresh_local(state, *roots)
            if args.kaggle:
                refresh_kaggle(state, collect=args.collect)
                if args.collect:
                    refresh_local(state, Path(state["results_root"]))
            state["state_path"] = str(state_path)
            summary = _summary(state)
        _print_summary(summary)
        return 0

    with _locked_state(state_path, write=False) as state:
        state["state_path"] = str(state_path)
        summary = _summary(state)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
