"""Pytest tests for scripts/eval_guard.py.

Each test builds a synthetic git repo in a temporary directory, commits
non-protected and protected files, then invokes eval_guard.py as a
subprocess and asserts on its exit code / stderr.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# Repo root = scripts/tests/test_eval_guard.py -> scripts/tests -> scripts -> root
REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_SCRIPT = REPO_ROOT / "scripts" / "eval_guard.py"

PROTECTED_REL_PATHS = [
    "experiments/organism_curriculum/foo.txt",
    "research/notes.md",
    "lanes/lane_a.txt",
    "eval/run.py",
    "src/oczy/experiments/organism_curriculum/scoring.py",
    "src/oczy/experiments/organism_curriculum/validation.py",
]


def _run_git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path: Path) -> None:
    """Initialise a git repo with user config and an initial commit."""
    _run_git("init", cwd=tmp_path)
    _run_git("config", "user.email", "test@test", cwd=tmp_path)
    _run_git("config", "user.name", "test", cwd=tmp_path)
    # Initial commit: a non-protected file plus the protected scaffolding.
    (tmp_path / "README.md").write_text("hello\n")
    for rel in PROTECTED_REL_PATHS:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("initial\n")
    _run_git("add", ".", cwd=tmp_path)
    _run_git("commit", "-m", "initial", cwd=tmp_path)


def _commit_change(tmp_path: Path, rel: str, content: str, msg: str) -> None:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    _run_git("add", rel, cwd=tmp_path)
    _run_git("commit", "-m", msg, cwd=tmp_path)


def _run_guard(
    tmp_path: Path,
    *,
    args: list[str] | None = None,
    env_override: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "EVAL_CHANGE_APPROVED"}
    if env_override:
        env.update(env_override)
    cmd = ["python", str(GUARD_SCRIPT)]
    if args:
        cmd.extend(args)
    return subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )


def test_no_protected_files_changed(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit_change(tmp_path, "README.md", "changed\n", "tweak readme")
    result = _run_guard(tmp_path, args=["HEAD~1...HEAD"])
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "rel",
    [
        "experiments/organism_curriculum/foo.txt",
        "research/notes.md",
        "lanes/lane_a.txt",
        "eval/run.py",
    ],
)
def test_protected_file_changed(tmp_path: Path, rel: str) -> None:
    _init_repo(tmp_path)
    _commit_change(tmp_path, rel, "tampered\n", f"touch {rel}")
    result = _run_guard(tmp_path, args=["HEAD~1...HEAD"])
    assert result.returncode == 1, result.stdout + result.stderr
    assert rel in result.stderr


def test_allow_without_env_var(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit_change(
        tmp_path,
        "experiments/organism_curriculum/foo.txt",
        "tampered\n",
        "touch protected",
    )
    result = _run_guard(tmp_path, args=["HEAD~1...HEAD", "--allow"])
    assert result.returncode == 1, result.stdout + result.stderr


def test_allow_with_env_var(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit_change(
        tmp_path,
        "experiments/organism_curriculum/foo.txt",
        "tampered\n",
        "touch protected",
    )
    result = _run_guard(
        tmp_path,
        args=["HEAD~1...HEAD", "--allow"],
        env_override={"EVAL_CHANGE_APPROVED": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
