from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pytest

KAGGLE_DIR = Path(__file__).resolve().parents[2] / "infrastructure" / "kaggle"
prepare_kernel = runpy.run_path(KAGGLE_DIR / "prepare_research_kernel.py")["prepare_kernel"]
prepare_bundle = runpy.run_path(KAGGLE_DIR / "prepare_source_bundle.py")["prepare_bundle"]
model_probe = runpy.run_path(KAGGLE_DIR / "run_qwen_model_probe.py")
artifact_manifest = model_probe["artifact_manifest"]
locate_model = model_probe["locate_model"]


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


def test_source_bundle_is_commit_addressed_and_rejects_dirty_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-m", "fixture")

    output = tmp_path / "bundle"
    manifest = prepare_bundle(
        repo_root=repo,
        revision="HEAD",
        output=output,
        dataset_id=None,
        allow_dirty_worktree=False,
        force=False,
    )

    archive = output / "source.tar.gz"
    assert manifest["commit"]
    assert manifest["dataset_id"].endswith(manifest["commit"][:12])
    assert manifest["worktree_dirty_at_packaging"] is False
    assert manifest["archive"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (
        json.loads((output / "dataset-metadata.json").read_text())["id"] == manifest["dataset_id"]
    )

    (repo / "pyproject.toml").write_text("[project]\nname='dirty'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="worktree is dirty"):
        prepare_bundle(
            repo_root=repo,
            revision="HEAD",
            output=tmp_path / "dirty-bundle",
            dataset_id=None,
            allow_dirty_worktree=False,
            force=False,
        )


def test_research_kernel_is_private_pinned_and_compilable(tmp_path: Path) -> None:
    commit = "a" * 40
    archive_sha = "b" * 64
    output = tmp_path / "job"
    spec = prepare_kernel(
        output=output,
        kernel_id="owner/oczy-development-seed-0",
        title="Oczy Development Seed 0",
        phase="development",
        profile="t4",
        source_dataset=f"owner/oczy-source-{commit[:12]}",
        source_commit=commit,
        source_archive_sha256=archive_sha,
        module="oczy.experiments.meta_cortex.train_outer",
        arguments=["--developmental-seed", "0"],
        model_source=None,
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )

    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert spec["source_commit"] == commit
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["model_sources"] == ["qwen-lm/qwen2.5/transformers/0.5b-instruct/1"]
    compile((output / "run.py").read_text(), str(output / "run.py"), "exec")


def test_meta_test_kernel_requires_manifest_and_human_signoff(tmp_path: Path) -> None:
    commit = "a" * 40
    with pytest.raises(ValueError, match="human sign-off"):
        prepare_kernel(
            output=tmp_path / "job",
            kernel_id="owner/oczy-meta-test",
            title="Oczy Meta Test",
            phase="meta-test",
            profile="t4",
            source_dataset=f"owner/oczy-source-{commit[:12]}",
            source_commit=commit,
            source_archive_sha256="b" * 64,
            module="oczy.experiments.meta_cortex.run_meta_test",
            arguments=[],
            model_source=None,
            instrument_manifest_sha256=None,
            human_signoff_id=None,
            force=False,
        )


def test_qwen_locator_and_artifact_manifest(tmp_path: Path) -> None:
    model = tmp_path / "qwen"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "hidden_size": 896}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"weights")

    assert locate_model(model) == model.resolve()
    files = {item["path"]: item for item in artifact_manifest(model)}
    assert files["model.safetensors"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
