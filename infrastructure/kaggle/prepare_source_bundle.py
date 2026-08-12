"""Prepare an immutable, commit-addressed Oczy source dataset for Kaggle.

The generated directory contains a Git archive, a provenance manifest, and
Kaggle dataset metadata. Upload remains an explicit human action. By default
the command refuses to run from a dirty worktree, even though ``git archive``
would include only the selected commit, because ambiguity about what was
actually shipped is unacceptable for a research run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "oczy/kaggle-source-bundle/v1"


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_bundle(
    *,
    repo_root: Path,
    revision: str,
    output: Path,
    dataset_id: str | None,
    allow_dirty_worktree: bool,
    force: bool,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output = output.resolve()
    commit = _git(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    short_commit = commit[:12]
    status_lines = [
        line
        for line in _git(
            repo_root, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    ]
    if status_lines and not allow_dirty_worktree:
        raise RuntimeError(
            "worktree is dirty; commit or isolate the intended source, or use "
            "--allow-dirty-worktree for a non-scored development bundle"
        )

    generated_names = ("source.tar.gz.bin", "source_manifest.json", "dataset-metadata.json")
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in generated_names if (output / name).exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite generated files in {output}")
    for path in existing:
        path.unlink()

    archive_path = output / "source.tar.gz.bin"
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar.gz",
            "--prefix=oczy/",
            f"--output={archive_path}",
            commit,
        ],
        cwd=repo_root,
        check=True,
    )
    archive_sha256 = _sha256_file(archive_path)
    tracked_files = _git(repo_root, "ls-tree", "-r", "--name-only", commit).splitlines()
    branch = _git(repo_root, "branch", "--show-current")

    resolved_dataset_id = dataset_id or f"abdellahkadem/oczy-source-{short_commit}"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": resolved_dataset_id,
        "revision_requested": revision,
        "commit": commit,
        "branch_at_packaging": branch,
        "worktree_dirty_at_packaging": bool(status_lines),
        "dirty_entry_count": len(status_lines),
        "dirty_entries": status_lines,
        "archive": {
            "filename": archive_path.name,
            "sha256": archive_sha256,
            "size_bytes": archive_path.stat().st_size,
            "prefix": "oczy/",
            "tracked_file_count": len(tracked_files),
        },
    }
    _write_json(output / "source_manifest.json", manifest)
    _write_json(
        output / "dataset-metadata.json",
        {
            "title": f"Oczy source {short_commit}",
            "id": resolved_dataset_id,
            "licenses": [{"name": "other"}],
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_bundle(
        repo_root=args.repo_root,
        revision=args.revision,
        output=args.output,
        dataset_id=args.dataset_id,
        allow_dirty_worktree=args.allow_dirty_worktree,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
