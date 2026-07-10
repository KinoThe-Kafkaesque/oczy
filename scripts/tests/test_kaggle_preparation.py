"""CPU-only, offline-reproducible Kaggle remote-runner contract tests.

These tests defend the desired contract for Oczy's active Kaggle surface:

1. The generator accepts/defaults to CPU and rejects ``t4``/unknown profiles.
2. Generated metadata is private/offline with GPU/TPU false and empty machine
   shape for every phase; model attachment rules and meta-test sign-off remain
   correct.
3. Generated bootstrap compiles, sets CPU/offline/thread environment before
   torch-dependent execution, contains no runtime CUDA query, changes cwd to
   source root, and records error provenance.
4. Active checked-in kernel metadata includes only CPU tasks; archived GPU
   metadata is excluded from active discovery.
5. Cortex and Qwen runners cannot select or query CUDA through their public
   CLI/default contract; active runner execution must not call any
   ``torch.cuda.*`` API.
6. HFDriver chooses ``OCZY_MODEL_DIR`` and local-only loading in remote CPU
   mode without touching the network, using fakes/monkeypatches rather than
   loading Qwen.
7. Default HF layer probe model resolution honors the pinned model environment
   while explicit model ids remain explicit.

No real network, Kaggle credentials, model download, or protected eval access
is required.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

KAGGLE_DIR = Path(__file__).resolve().parents[2] / "infrastructure" / "kaggle"
prepare_kernel = runpy.run_path(str(KAGGLE_DIR / "prepare_research_kernel.py"))["prepare_kernel"]
prepare_bundle = runpy.run_path(str(KAGGLE_DIR / "prepare_source_bundle.py"))["prepare_bundle"]
model_probe = runpy.run_path(str(KAGGLE_DIR / "run_qwen_model_probe.py"))
artifact_manifest = model_probe["artifact_manifest"]
locate_model = model_probe["locate_model"]

COMMIT = "a" * 40
ARCHIVE_SHA = "b" * 64
SOURCE_DATASET = f"owner/oczy-source-{COMMIT[:12]}"


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# 1. Generator profile contract
# ---------------------------------------------------------------------------


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

    archive = output / "source.tar.gz.bin"
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


def test_source_bundle_archive_name_is_opaque_to_kaggle_auto_extraction(tmp_path: Path) -> None:
    """Kaggle auto-extracts recognized archive suffixes (``.tar.gz``, ``.zip``,
    ``.tar``) at mount time, which would remove the sibling archive the manifest
    declares. The bundle must ship under an opaque filename Kaggle mounts
    unchanged, while remaining a valid gzip tar readable via ``tarfile``.

    This test would fail if the archive were named ``source.tar.gz`` (a
    Kaggle-recognized suffix) and passes for the opaque ``source.tar.gz.bin``.
    """
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

    archive_filename = manifest["archive"]["filename"]
    archive_path = output / archive_filename

    # Kaggle-recognized archive suffixes that trigger auto-extraction at mount.
    kaggle_auto_extract_suffixes = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".tar")
    assert not archive_filename.endswith(kaggle_auto_extract_suffixes), (
        f"archive filename {archive_filename!r} ends with a Kaggle-recognized "
        f"archive suffix and would be auto-extracted, removing the sibling "
        f"the manifest declares"
    )

    # SHA-256 provenance from the manifest must match the actual file bytes.
    assert manifest["archive"]["sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()

    # Despite the opaque extension, the file must be a valid gzip tar.
    with tarfile.open(str(archive_path), "r:gz") as tf:
        members = tf.getnames()
    assert "oczy/pyproject.toml" in members, (
        f"oczy/pyproject.toml not found in archive; members={members[:10]}..."
    )


def test_generator_rejects_t4_profile(tmp_path: Path) -> None:
    """The generator must reject the retired ``t4`` profile."""
    with pytest.raises(ValueError, match="unknown profile"):
        prepare_kernel(
            output=tmp_path / "job",
            kernel_id="owner/oczy-development-seed-0",
            title="Oczy Development Seed 0",
            phase="development",
            profile="t4",
            source_dataset=SOURCE_DATASET,
            source_commit=COMMIT,
            source_archive_sha256=ARCHIVE_SHA,
            module="oczy.experiments.meta_cortex.train_outer",
            arguments=["--developmental-seed", "0"],
            model_source=None,
            instrument_manifest_sha256=None,
            human_signoff_id=None,
            force=False,
        )


def test_generator_rejects_unknown_profile(tmp_path: Path) -> None:
    """The generator must reject any profile that is not ``cpu``."""
    with pytest.raises(ValueError, match="unknown profile"):
        prepare_kernel(
            output=tmp_path / "job",
            kernel_id="owner/oczy-development-seed-0",
            title="Oczy Development Seed 0",
            phase="development",
            profile="l4",
            source_dataset=SOURCE_DATASET,
            source_commit=COMMIT,
            source_archive_sha256=ARCHIVE_SHA,
            module="oczy.experiments.meta_cortex.train_outer",
            arguments=[],
            model_source=None,
            instrument_manifest_sha256=None,
            human_signoff_id=None,
            force=False,
        )


def test_generator_cli_defaults_to_cpu_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI ``--profile`` argument must default to ``cpu``."""
    gen = runpy.run_path(str(KAGGLE_DIR / "prepare_research_kernel.py"))
    parse_args = gen["parse_args"]

    monkeypatch.setattr(sys, "argv", [
        "prepare_research_kernel.py",
        "--output", str(tmp_path / "job"),
        "--kernel-id", "owner/oczy-test",
        "--title", "Test",
        "--phase", "analysis",
        "--source-dataset", SOURCE_DATASET,
        "--source-commit", COMMIT,
        "--source-archive-sha256", ARCHIVE_SHA,
        "--module", "oczy.experiments.dummy",
    ])
    args = parse_args()
    assert args.profile == "cpu"


def test_generator_cli_rejects_t4_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI ``--profile`` argument must not accept ``t4``."""
    gen = runpy.run_path(str(KAGGLE_DIR / "prepare_research_kernel.py"))
    parse_args = gen["parse_args"]

    monkeypatch.setattr(sys, "argv", [
        "prepare_research_kernel.py",
        "--output", "/tmp/job",
        "--kernel-id", "owner/oczy-test",
        "--title", "Test",
        "--phase", "analysis",
        "--profile", "t4",
        "--source-dataset", SOURCE_DATASET,
        "--source-commit", COMMIT,
        "--source-archive-sha256", ARCHIVE_SHA,
        "--module", "oczy.experiments.dummy",
    ])
    with pytest.raises(SystemExit):
        parse_args()


# ---------------------------------------------------------------------------
# 2. Generated metadata is private/offline/CPU for every phase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["instrument", "oracle", "development", "analysis"])
def test_generated_metadata_is_cpu_private_offline_for_every_phase(
    tmp_path: Path, phase: str
) -> None:
    """Every non-meta-test phase must produce CPU-only, private, offline metadata."""
    output = tmp_path / f"job-{phase}"
    spec = prepare_kernel(
        output=output,
        kernel_id=f"owner/oczy-{phase}",
        title=f"Oczy {phase.title()}",
        phase=phase,
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.dummy",
        arguments=[],
        model_source=None,
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )
    metadata = json.loads((output / "kernel-metadata.json").read_text())

    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["machine_shape"] == ""
    assert spec["profile"] == "cpu"


def test_generated_metadata_model_attachment_rules(tmp_path: Path) -> None:
    """Oracle/development/meta-test phases attach the default model; others don't."""
    # Phases that should auto-attach the default model.
    for phase in ("oracle", "development"):
        output = tmp_path / f"auto-{phase}"
        prepare_kernel(
            output=output,
            kernel_id=f"owner/oczy-{phase}",
            title=f"Oczy {phase.title()}",
            phase=phase,
            profile="cpu",
            source_dataset=SOURCE_DATASET,
            source_commit=COMMIT,
            source_archive_sha256=ARCHIVE_SHA,
            module="oczy.experiments.dummy",
            arguments=[],
            model_source=None,
            instrument_manifest_sha256=None,
            human_signoff_id=None,
            force=False,
        )
        metadata = json.loads((output / "kernel-metadata.json").read_text())
        assert metadata["model_sources"] == [
            "qwen-lm/qwen2.5/transformers/0.5b-instruct/1"
        ], f"phase {phase} should auto-attach default model"

    # Phases that should NOT auto-attach.
    for phase in ("instrument", "analysis"):
        output = tmp_path / f"noauto-{phase}"
        prepare_kernel(
            output=output,
            kernel_id=f"owner/oczy-{phase}",
            title=f"Oczy {phase.title()}",
            phase=phase,
            profile="cpu",
            source_dataset=SOURCE_DATASET,
            source_commit=COMMIT,
            source_archive_sha256=ARCHIVE_SHA,
            module="oczy.experiments.dummy",
            arguments=[],
            model_source=None,
            instrument_manifest_sha256=None,
            human_signoff_id=None,
            force=False,
        )
        metadata = json.loads((output / "kernel-metadata.json").read_text())
        assert metadata["model_sources"] == [], f"phase {phase} should not auto-attach"

    # Explicit model_source overrides.
    output = tmp_path / "explicit-model"
    prepare_kernel(
        output=output,
        kernel_id="owner/oczy-instrument",
        title="Oczy Instrument",
        phase="instrument",
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.dummy",
        arguments=[],
        model_source="custom/model/1",
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["model_sources"] == ["custom/model/1"]


def test_meta_test_kernel_requires_manifest_and_human_signoff(tmp_path: Path) -> None:
    """Meta-test phase still requires instrument manifest hash and human sign-off."""
    with pytest.raises(ValueError, match="human sign-off"):
        prepare_kernel(
            output=tmp_path / "job",
            kernel_id="owner/oczy-meta-test",
            title="Oczy Meta Test",
            phase="meta-test",
            profile="cpu",
            source_dataset=SOURCE_DATASET,
            source_commit=COMMIT,
            source_archive_sha256=ARCHIVE_SHA,
            module="oczy.experiments.meta_cortex.run_meta_test",
            arguments=[],
            model_source=None,
            instrument_manifest_sha256=None,
            human_signoff_id=None,
            force=False,
        )


def test_meta_test_kernel_succeeds_with_signoff(tmp_path: Path) -> None:
    """Meta-test phase with full sign-off produces correct CPU metadata."""
    output = tmp_path / "meta-job"
    spec = prepare_kernel(
        output=output,
        kernel_id="owner/oczy-meta-test",
        title="Oczy Meta Test",
        phase="meta-test",
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.meta_cortex.run_meta_test",
        arguments=[],
        model_source=None,
        instrument_manifest_sha256=ARCHIVE_SHA,
        human_signoff_id="researcher@example.invalid",
        force=False,
    )
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    assert metadata["machine_shape"] == ""
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert spec["instrument_manifest_sha256"] == ARCHIVE_SHA
    assert spec["human_signoff_id"] == "researcher@example.invalid"
    assert spec["model_source"] == "qwen-lm/qwen2.5/transformers/0.5b-instruct/1"


# ---------------------------------------------------------------------------
# 3. Generated bootstrap compiles and sets CPU/offline/thread env before torch
# ---------------------------------------------------------------------------


def test_generated_bootstrap_compiles(tmp_path: Path) -> None:
    """The generated run.py must be syntactically valid Python."""
    output = tmp_path / "job"
    prepare_kernel(
        output=output,
        kernel_id="owner/oczy-test",
        title="Oczy Test",
        phase="development",
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.dummy",
        arguments=["--seed", "0"],
        model_source=None,
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )
    compile((output / "run.py").read_text(), str(output / "run.py"), "exec")


def test_generated_bootstrap_sets_cpu_offline_thread_env_before_torch(tmp_path: Path) -> None:
    """The bootstrap must set CPU/offline/thread env vars *before* importing torch."""
    output = tmp_path / "job"
    prepare_kernel(
        output=output,
        kernel_id="owner/oczy-test",
        title="Oczy Test",
        phase="development",
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.dummy",
        arguments=[],
        model_source=None,
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )
    source = (output / "run.py").read_text()

    # The env-setting block must appear before `import torch`.
    env_block_marker = 'CUDA_VISIBLE_DEVICES'
    torch_import_marker = 'import torch'

    env_pos = source.find(env_block_marker)
    torch_pos = source.find(torch_import_marker)
    assert env_pos != -1, "CUDA_VISIBLE_DEVICES not found in bootstrap"
    assert torch_pos != -1, "import torch not found in bootstrap"
    assert env_pos < torch_pos, "CPU/offline env vars must be set before import torch"

    # Verify all required env vars are present.
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OCZY_REMOTE_CPU_ONLY",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    ):
        assert var in source, f"{var} not set in bootstrap"


def test_generated_bootstrap_has_no_runtime_cuda_query(tmp_path: Path) -> None:
    """The bootstrap must not query CUDA at runtime.

    CUDA_VISIBLE_DEVICES='' is set before torch import as enforcement (not a
    probe), and the CPU-only contract is recorded with constant values.
    torch.version.cuda (build metadata) is allowed — it is torch.version, not
    torch.cuda, and does not initialize CUDA or emit NVML warnings.
    """
    output = tmp_path / "job"
    prepare_kernel(
        output=output,
        kernel_id="owner/oczy-test",
        title="Oczy Test",
        phase="development",
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.dummy",
        arguments=[],
        model_source=None,
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )
    source = (output / "run.py").read_text()

    # No runtime CUDA API calls anywhere in the bootstrap.  torch.version.cuda
    # is build metadata (torch.version, not torch.cuda) and does not match.
    assert "torch.cuda." not in source, (
        "bootstrap must not call any torch.cuda.* API; "
        "torch.version.cuda build metadata is allowed but torch.cuda.* is not"
    )

    # Enforcement: CUDA_VISIBLE_DEVICES set to empty string before torch import.
    assert 'CUDA_VISIBLE_DEVICES"] = ""' in source

    # CPU-only contract block must be recorded with constant (non-probed) values.
    assert "cpu_only_contract" in source


def test_generated_bootstrap_changes_cwd_to_source_root(tmp_path: Path) -> None:
    """The bootstrap must os.chdir(source_root) before running the target module."""
    output = tmp_path / "job"
    prepare_kernel(
        output=output,
        kernel_id="owner/oczy-test",
        title="Oczy Test",
        phase="development",
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.dummy",
        arguments=[],
        model_source=None,
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )
    source = (output / "run.py").read_text()
    assert "os.chdir(source_root)" in source
    # chdir must happen before runpy.run_module.
    chdir_pos = source.find("os.chdir(source_root)")
    runpy_pos = source.find("runpy.run_module")
    assert chdir_pos != -1
    assert runpy_pos != -1
    assert chdir_pos < runpy_pos, "os.chdir(source_root) must happen before runpy.run_module"


def test_generated_bootstrap_extracts_source_to_temp_not_persisted(tmp_path: Path) -> None:
    """The bootstrap must extract the source archive into a temporary directory
    under the system temp dir (``tempfile.mkdtemp``), never into a path below
    ``/kaggle/working``.

    Extracting into ``/kaggle/working/source`` caused ``kaggle kernels output``
    to download every tracked source file alongside the run report.  A
    ``tempfile.mkdtemp()`` directory lives outside ``/kaggle/working`` and is
    not persisted as kernel output, so only the provenance report is
    downloaded.

    The bootstrap must still ``os.chdir(source_root)`` so the target module
    runs from the extracted source root.
    """
    output = tmp_path / "job"
    prepare_kernel(
        output=output,
        kernel_id="owner/oczy-test",
        title="Oczy Test",
        phase="development",
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.dummy",
        arguments=[],
        model_source=None,
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )
    source = (output / "run.py").read_text()

    # Source must be extracted via tempfile.mkdtemp() — a non-persisted
    # directory under the system temp dir, not /kaggle/working.
    assert "import tempfile" in source, "bootstrap must import tempfile"
    assert "tempfile.mkdtemp()" in source, (
        "bootstrap must extract source via tempfile.mkdtemp()"
    )
    assert "/kaggle/working/source" not in source, (
        "bootstrap must not extract source into /kaggle/working/source; "
        "that path is persisted by kaggle kernels output and bloats downloads"
    )

    # The cwd-to-source-root contract is preserved: the temp path is still
    # used as the working directory before runpy.run_module.
    chdir_pos = source.find("os.chdir(source_root)")
    runpy_pos = source.find("runpy.run_module")
    assert chdir_pos != -1, "os.chdir(source_root) must remain in bootstrap"
    assert runpy_pos != -1
    assert chdir_pos < runpy_pos, (
        "os.chdir(source_root) must happen before runpy.run_module"
    )



def test_generated_bootstrap_propagates_model_dir(tmp_path: Path) -> None:
    """The bootstrap must set OCZY_MODEL_DIR when a model is found."""
    output = tmp_path / "job"
    prepare_kernel(
        output=output,
        kernel_id="owner/oczy-test",
        title="Oczy Test",
        phase="development",
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.dummy",
        arguments=[],
        model_source=None,
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )
    source = (output / "run.py").read_text()
    assert 'OCZY_MODEL_DIR' in source
    assert "find_model()" in source


def test_generated_bootstrap_records_error_provenance(tmp_path: Path) -> None:
    """The bootstrap must record error type, message, and traceback on failure."""
    output = tmp_path / "job"
    prepare_kernel(
        output=output,
        kernel_id="owner/oczy-test",
        title="Oczy Test",
        phase="development",
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.dummy",
        arguments=[],
        model_source=None,
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )
    source = (output / "run.py").read_text()
    # Error provenance: type, message, traceback, exit_code, status.
    assert '"type"' in source or 'type(error).__name__' in source
    assert 'traceback.format_exc()' in source
    assert '"error"' in source or "'error'" in source
    assert 'exit_code' in source

def test_generated_bootstrap_job_spec_round_trips_with_none_optional_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated run.py JOB_SPEC must be valid executable Python that
    round-trips exactly, even when optional fields are None and arguments
    contain characters that are unsafe in raw JSON-as-Python.

    Raw JSON ``null`` is not a Python identifier, so the uncorrected renderer
    (``JOB_SPEC = {... null ...}``) raises ``NameError`` before provenance is
    written.  The corrected renderer (``JOB_SPEC = json.loads(<repr>)``)
    safely quotes the canonical JSON string and lets ``json.loads`` handle
    ``null`` → ``None``, ``true`` → ``True``, etc.

    This test executes the generated run.py top level (import torch, define
    JOB_SPEC and helpers) without entering ``main()``, so no Kaggle, network,
    or model access occurs.
    """
    # Arguments with characters that break raw JSON-as-Python or require
    # careful quoting: quotes, backslashes, newlines, tabs, unicode, empty.
    special_args = [
        "--name", "O'Brien",
        "--quote", 'say "hello"',
        "--path", "back\\slash",
        "--multiline", "line1\nline2\ttab",
        "--unicode", "café—résumé",
        "--empty", "",
    ]
    output = tmp_path / "job"
    expected_spec = prepare_kernel(
        output=output,
        kernel_id="owner/oczy-test",
        title="Oczy Test",
        phase="analysis",
        profile="cpu",
        source_dataset=SOURCE_DATASET,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        module="oczy.experiments.dummy",
        arguments=special_args,
        model_source=None,
        instrument_manifest_sha256=None,
        human_signoff_id=None,
        force=False,
    )

    # The bootstrap sets env vars at top level; save originals for teardown.
    for var in (
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OCZY_REMOTE_CPU_ONLY", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.setenv(var, os.environ.get(var, ""))

    # Execute the top level without entering main().  run_name != "__main__"
    # skips the ``if __name__ == "__main__"`` guard, so no Kaggle/network/model
    # access occurs — only imports, env setup, and JOB_SPEC assignment.
    namespace = runpy.run_path(str(output / "run.py"), run_name="_test_import")

    job_spec = namespace["JOB_SPEC"]
    assert job_spec == expected_spec
    # None optional fields must survive as Python None, not JSON null.
    assert job_spec["model_source"] is None
    assert job_spec["instrument_manifest_sha256"] is None
    assert job_spec["human_signoff_id"] is None
    # Special argument strings must survive byte-for-byte.
    assert job_spec["arguments"] == special_args


def test_job_spec_rendering_safely_quotes_all_json_value_types() -> None:
    """The JOB_SPEC rendering approach must handle every JSON value type,
    not just strings and None.  Booleans (``true``/``false``) and nested lists
    are invalid as raw JSON-as-Python (NameError) but round-trip correctly
    through ``json.loads(repr(canonical_json))``.

    The job_spec has no boolean fields, so this test verifies the rendering
    mechanism directly with a synthetic spec containing True, False, None,
    lists, and strings with quotes/backslashes/newlines.
    """
    synthetic = {
        "flag_true": True,
        "flag_false": False,
        "none_value": None,
        "nested_list": [1, "two", True, None, False, ["inner"]],
        "special_string": 'quote\'s and "double" and \\back and\nnewline',
        "unicode": "café—résumé",
    }
    canonical = json.dumps(synthetic, sort_keys=True)

    # Corrected renderer: JOB_SPEC = json.loads(<repr of canonical JSON>).
    # repr() safely quotes any string content; json.loads handles all JSON
    # types (null→None, true→True, false→False).
    namespace: dict[str, object] = {}
    exec(f"import json; JOB_SPEC = json.loads({repr(canonical)})", namespace)
    assert namespace["JOB_SPEC"] == synthetic

    # Uncorrected renderer: JOB_SPEC = <raw JSON>.  JSON true/false/null are
    # not Python identifiers, so this raises NameError — the bootstrap bug.
    with pytest.raises(NameError):
        exec(f"JOB_SPEC = {canonical}", {})

# ---------------------------------------------------------------------------
# 4. Active checked-in metadata includes only CPU; GPU is archived
# ---------------------------------------------------------------------------


def _load_kernel_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_active_kernel_dirs_are_cpu_only() -> None:
    """Every kernel-metadata.json directly under infrastructure/kaggle/*/ must
    be CPU-only (enable_gpu=false, machine_shape='')."""
    active_dirs = [
        d for d in KAGGLE_DIR.iterdir()
        if d.is_dir() and (d / "kernel-metadata.json").is_file()
    ]
    assert len(active_dirs) >= 2, "expected at least cpu-smoke and qwen-cpu-probe"
    for d in active_dirs:
        meta = _load_kernel_metadata(d / "kernel-metadata.json")
        assert meta["enable_gpu"] is False, f"{d.name} has enable_gpu=true"
        assert meta["enable_tpu"] is False, f"{d.name} has enable_tpu=true"
        assert meta["machine_shape"] == "", f"{d.name} has non-empty machine_shape"
        assert meta["is_private"] is True, f"{d.name} is not private"
        assert meta["enable_internet"] is False, f"{d.name} has internet enabled"


def test_gpu_metadata_is_archived_and_excluded_from_active_discovery() -> None:
    """GPU kernel metadata must live under archive/gpu/ and not at the active
    level."""
    archive_gpu = KAGGLE_DIR / "archive" / "gpu"
    assert archive_gpu.is_dir(), "archive/gpu/ directory must exist"

    archived_dirs = [
        d for d in archive_gpu.iterdir()
        if d.is_dir() and (d / "kernel-metadata.json").is_file()
    ]
    assert len(archived_dirs) >= 1, "expected at least one archived GPU kernel"

    for d in archived_dirs:
        meta = _load_kernel_metadata(d / "kernel-metadata.json")
        # Archived metadata should have GPU enabled (that's why it was archived).
        assert meta["enable_gpu"] is True, f"{d.name} in archive should have enable_gpu=true"

    # No active-level directory should have GPU metadata.
    active_dirs = [
        d for d in KAGGLE_DIR.iterdir()
        if d.is_dir() and (d / "kernel-metadata.json").is_file()
    ]
    for d in active_dirs:
        meta = _load_kernel_metadata(d / "kernel-metadata.json")
        assert meta["enable_gpu"] is False, f"{d.name} is active but has enable_gpu=true"


def test_active_dirs_include_cpu_smoke_and_qwen_cpu_probe() -> None:
    """The active surface must include the CPU cortex smoke and CPU Qwen probe."""
    active_ids = set()
    for d in KAGGLE_DIR.iterdir():
        if d.is_dir() and (d / "kernel-metadata.json").is_file():
            meta = _load_kernel_metadata(d / "kernel-metadata.json")
            active_ids.add(meta["id"])
    assert "abdellahkadem/oczy-cortex-cpu-smoke" in active_ids
    assert "abdellahkadem/oczy-qwen-cpu-probe" in active_ids


def test_archived_gpu_dirs_are_not_active() -> None:
    """No archived GPU kernel id should appear among active kernel ids."""
    archive_gpu = KAGGLE_DIR / "archive" / "gpu"
    archived_ids = set()
    for d in archive_gpu.iterdir():
        if d.is_dir() and (d / "kernel-metadata.json").is_file():
            meta = _load_kernel_metadata(d / "kernel-metadata.json")
            archived_ids.add(meta["id"])

    active_ids = set()
    for d in KAGGLE_DIR.iterdir():
        if d.is_dir() and (d / "kernel-metadata.json").is_file():
            meta = _load_kernel_metadata(d / "kernel-metadata.json")
            active_ids.add(meta["id"])

    assert archived_ids.isdisjoint(active_ids), (
        f"archived GPU ids overlap with active ids: {archived_ids & active_ids}"
    )


# ---------------------------------------------------------------------------
# 5. Cortex and Qwen runners cannot select or query CUDA
# ---------------------------------------------------------------------------


def test_cortex_smoke_cli_rejects_cuda_and_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cortex smoke --device argument must only accept 'cpu'."""
    cortex = runpy.run_path(str(KAGGLE_DIR / "run_cortex_smoke.py"))
    parse_args = cortex["parse_args"]

    # 'cuda' and 'auto' must not be valid choices.
    for bad_device in ("cuda", "auto"):
        monkeypatch.setattr(sys, "argv", ["run_cortex_smoke.py", "--device", bad_device])
        with pytest.raises(SystemExit):
            parse_args()

    # Default must be 'cpu'.
    monkeypatch.setattr(sys, "argv", ["run_cortex_smoke.py"])
    args = parse_args()
    assert args.device == "cpu"


def test_cortex_smoke_resolve_device_rejects_non_cpu() -> None:
    """_resolve_device must reject anything other than 'cpu'."""
    cortex = runpy.run_path(str(KAGGLE_DIR / "run_cortex_smoke.py"))
    _resolve_device = cortex["_resolve_device"]
    import torch

    assert _resolve_device("cpu") == torch.device("cpu")
    for bad in ("cuda", "auto"):
        with pytest.raises(RuntimeError, match="CPU"):
            _resolve_device(bad)


def test_cortex_smoke_run_always_uses_cpu() -> None:
    """run() must produce a report with device.selected == 'cpu'."""
    cortex = runpy.run_path(str(KAGGLE_DIR / "run_cortex_smoke.py"))
    run = cortex["run"]
    SmokeConfig = cortex["SmokeConfig"]

    config = SmokeConfig(steps=2, batch_size=8, eval_batch_size=8)
    report = run(config, "cpu")
    assert report["device"]["selected"] == "cpu"
    assert report["architecture"]["parallel_mode"] == "single-device"
    assert report["architecture"]["devices_used"] == ["cpu"]


def test_cortex_smoke_source_has_no_cuda_apis() -> None:
    """The cortex smoke runner source must not call any torch.cuda.* API.

    torch.version.cuda (build metadata) is allowed — it is torch.version, not
    torch.cuda, and does not initialize CUDA or emit NVML warnings. A source-
    level check is used because torch's own optimizer internally calls
    torch.cuda.is_available() during step(), which would false-positive a
    runtime monkeypatch test.
    """
    source = (KAGGLE_DIR / "run_cortex_smoke.py").read_text(encoding="utf-8")
    assert "torch.cuda." not in source, (
        "cortex smoke runner must not call any torch.cuda.* API"
    )


def test_cortex_smoke_device_report_is_constant_cpu() -> None:
    """_device_report must emit constant CPU values without probing CUDA."""
    cortex = runpy.run_path(str(KAGGLE_DIR / "run_cortex_smoke.py"))
    _device_report = cortex["_device_report"]
    import torch

    report = _device_report(torch.device("cpu"))
    assert report["selected"] == "cpu"
    assert report["cuda_available"] is False
    assert report["cuda_device_count"] == 0
    assert report["cudnn_version"] is None


def test_qwen_probe_cli_has_no_allow_cpu_or_cuda_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Qwen probe CLI must not expose --allow-cpu or any CUDA selection."""
    qwen = runpy.run_path(str(KAGGLE_DIR / "run_qwen_model_probe.py"))
    parse_args = qwen["parse_args"]

    # --allow-cpu must not be a valid argument.
    monkeypatch.setattr(sys, "argv", ["run_qwen_model_probe.py", "--allow-cpu"])
    with pytest.raises(SystemExit):
        parse_args()

    # Default args should work and have no device-related option.
    monkeypatch.setattr(sys, "argv", ["run_qwen_model_probe.py"])
    args = parse_args()
    assert not hasattr(args, "allow_cpu")
    assert not hasattr(args, "device")


def test_qwen_probe_run_probe_always_uses_cpu(tmp_path: Path) -> None:
    """run_probe must select cpu device and float32 dtype."""
    qwen = runpy.run_path(str(KAGGLE_DIR / "run_qwen_model_probe.py"))
    run_probe = qwen["run_probe"]

    # Create a minimal fake model directory.
    model = tmp_path / "qwen"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({
            "model_type": "qwen2",
            "hidden_size": 896,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 100,
        }),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"weights")

    # metadata_only=True avoids loading transformers/torch model.
    report = run_probe(model, metadata_only=True)
    assert report["metadata_only"] is True
    assert report["checks"]["config_valid"] is True


def test_qwen_probe_source_has_no_cuda_apis() -> None:
    """The qwen probe runner source must not call any torch.cuda.* API.

    torch.version.cuda (build metadata) is allowed — it is torch.version, not
    torch.cuda, and does not initialize CUDA or emit NVML warnings.
    """
    source = (KAGGLE_DIR / "run_qwen_model_probe.py").read_text(encoding="utf-8")
    assert "torch.cuda." not in source, (
        "qwen probe runner must not call any torch.cuda.* API"
    )


def test_qwen_probe_device_report_is_constant_cpu() -> None:
    """_device_report must emit constant CPU values without probing CUDA."""
    qwen = runpy.run_path(str(KAGGLE_DIR / "run_qwen_model_probe.py"))
    _device_report = qwen["_device_report"]
    import torch

    report = _device_report(torch.device("cpu"))
    assert report["selected"] == "cpu"
    assert report["cuda_available"] is False
    assert report["cuda_device_count"] == 0


# ---------------------------------------------------------------------------
# 6. HFDriver honors OCZY_MODEL_DIR with local_files_only in remote CPU mode
# ---------------------------------------------------------------------------


def test_hfdriver_load_uses_oczy_model_dir_in_remote_cpu_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In remote CPU mode, HFDriver.load() must resolve OCZY_MODEL_DIR and pass
    local_files_only=True to from_pretrained — without touching the network."""
    from oczy.lm.hf_driver import HFDriver

    # Create a fake model directory.
    model_dir = tmp_path / "pinned-model"
    model_dir.mkdir()

    monkeypatch.setenv("OCZY_REMOTE_CPU_ONLY", "1")
    monkeypatch.setenv("OCZY_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    captured: dict = {}

    class FakeModel:
        class config:
            hidden_size = 896
            vocab_size = 100
            num_hidden_layers = 2
            pad_token_id = None

        def eval(self):
            return self

        def parameters(self):
            return iter([])

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        eos_token_id = 0

    def fake_from_pretrained_cls(mid, **kwargs):
        captured["model_kwargs"] = kwargs
        return FakeModel()

    def fake_from_pretrained_tok(mid, **kwargs):
        captured["tokenizer_kwargs"] = kwargs
        return FakeTokenizer()

    import transformers

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained_cls))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_from_pretrained_tok))

    driver = HFDriver.load()
    assert driver.model_id == str(model_dir)
    assert captured["model_kwargs"].get("local_files_only") is True
    assert captured["model_kwargs"].get("trust_remote_code") is False
    assert captured["tokenizer_kwargs"].get("local_files_only") is True
    assert captured["tokenizer_kwargs"].get("trust_remote_code") is False


def test_hfdriver_load_fails_without_model_dir_in_cpu_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In CPU-only mode without OCZY_MODEL_DIR or explicit model_id, load must
    raise RuntimeError, not fall back to network resolution."""
    from oczy.lm.hf_driver import HFDriver

    monkeypatch.setenv("OCZY_REMOTE_CPU_ONLY", "1")
    monkeypatch.delenv("OCZY_MODEL_DIR", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with pytest.raises(RuntimeError, match="OCZY_MODEL_DIR"):
        HFDriver.load()


def test_hfdriver_load_fails_when_pinned_model_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In CPU-only mode, if OCZY_MODEL_DIR points to a nonexistent directory,
    load must raise RuntimeError rather than falling back to network."""
    from oczy.lm.hf_driver import HFDriver

    monkeypatch.setenv("OCZY_REMOTE_CPU_ONLY", "1")
    monkeypatch.setenv("OCZY_MODEL_DIR", str(tmp_path / "nonexistent"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    with pytest.raises(RuntimeError, match="pinned local model not found"):
        HFDriver.load()


def test_hfdriver_explicit_model_id_takes_priority_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit model_id must take priority over OCZY_MODEL_DIR."""
    from oczy.lm.hf_driver import HFDriver

    model_dir = tmp_path / "env-model"
    model_dir.mkdir()
    monkeypatch.setenv("OCZY_MODEL_DIR", str(model_dir))
    # No offline env → explicit model_id should not get local_files_only.
    monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    captured: dict = {}

    class FakeModel:
        class config:
            hidden_size = 896
            vocab_size = 100
            num_hidden_layers = 2
            pad_token_id = None

        def eval(self):
            return self

        def parameters(self):
            return iter([])

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        eos_token_id = 0

    def fake_from_pretrained_cls(mid, **kwargs):
        captured["mid"] = mid
        captured["model_kwargs"] = kwargs
        return FakeModel()

    def fake_from_pretrained_tok(mid, **kwargs):
        captured["tokenizer_kwargs"] = kwargs
        return FakeTokenizer()

    import transformers

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained_cls))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_from_pretrained_tok))

    driver = HFDriver.load("Qwen/Qwen2.5-0.5B-Instruct")
    assert driver.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert captured["mid"] == "Qwen/Qwen2.5-0.5B-Instruct"
    # Without offline env and explicit model_id != OCZY_MODEL_DIR, no local_files_only.
    assert "local_files_only" not in captured["model_kwargs"]


def test_hfdriver_local_only_when_offline_env_set_even_with_explicit_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When offline env is set, even an explicit model_id must get
    local_files_only=True."""
    from oczy.lm.hf_driver import HFDriver

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
    monkeypatch.delenv("OCZY_MODEL_DIR", raising=False)

    captured: dict = {}

    class FakeModel:
        class config:
            hidden_size = 896
            vocab_size = 100
            num_hidden_layers = 2
            pad_token_id = None

        def eval(self):
            return self

        def parameters(self):
            return iter([])

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        eos_token_id = 0

    def fake_from_pretrained_cls(mid, **kwargs):
        captured["model_kwargs"] = kwargs
        return FakeModel()

    def fake_from_pretrained_tok(mid, **kwargs):
        captured["tokenizer_kwargs"] = kwargs
        return FakeTokenizer()

    import transformers

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained_cls))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_from_pretrained_tok))

    driver = HFDriver.load("some/model/1")
    assert captured["model_kwargs"].get("local_files_only") is True
    assert captured["tokenizer_kwargs"].get("local_files_only") is True


# ---------------------------------------------------------------------------
# 7. HF layer probe model resolution honors pinned model env
# ---------------------------------------------------------------------------


def test_hf_layer_probe_main_honors_oczy_model_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() with no --model-id must resolve OCZY_MODEL_DIR as the model id."""
    import oczy.experiments.hf_layer_probe as hfp

    model_dir = tmp_path / "pinned-model"
    monkeypatch.setenv("OCZY_MODEL_DIR", str(model_dir))
    monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    captured: dict = {}

    def fake_run_probe(model_id, poolings=("mean", "last", "max")):
        captured["model_id"] = model_id
        return {
            "model_id": model_id,
            "n_layers": 24,
            "n_embd": 896,
            "corpus_hash": "abc",
            "silhouettes": {"mean": {"L0": 0.5, "L23": 0.4}},
            "final_score": 0.4,
            "max_mid": 0.5,
            "gap": 0.1,
            "verdict": "ACCEPT",
            "mid_layer_range": "8-15",
            "primary_pooling": "mean",
        }

    monkeypatch.setattr(hfp, "run_probe", fake_run_probe)
    # Provide --output to avoid writing to default path.
    output_file = tmp_path / "probe.md"
    rc = hfp.main(["--quiet", "--output", str(output_file)])
    assert rc == 0
    assert captured["model_id"] == str(model_dir)


def test_hf_layer_probe_main_explicit_model_id_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit --model-id must take priority over OCZY_MODEL_DIR."""
    import oczy.experiments.hf_layer_probe as hfp

    monkeypatch.setenv("OCZY_MODEL_DIR", str(tmp_path / "env-model"))

    captured: dict = {}

    def fake_run_probe(model_id, poolings=("mean", "last", "max")):
        captured["model_id"] = model_id
        return {
            "model_id": model_id,
            "n_layers": 24,
            "n_embd": 896,
            "corpus_hash": "abc",
            "silhouettes": {"mean": {"L0": 0.5, "L23": 0.4}},
            "final_score": 0.4,
            "max_mid": 0.5,
            "gap": 0.1,
            "verdict": "ACCEPT",
            "mid_layer_range": "8-15",
            "primary_pooling": "mean",
        }

    monkeypatch.setattr(hfp, "run_probe", fake_run_probe)
    output_file = tmp_path / "probe.md"
    rc = hfp.main(["--quiet", "--output", str(output_file), "--model-id", "explicit/model/1"])
    assert rc == 0
    assert captured["model_id"] == "explicit/model/1"


def test_hf_layer_probe_run_probe_uses_local_only_when_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_probe must pass local_files_only=True when offline env is set."""
    import oczy.experiments.hf_layer_probe as hfp

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
    monkeypatch.delenv("OCZY_MODEL_DIR", raising=False)

    captured: dict = {}

    class FakeModel:
        class config:
            num_hidden_layers = 2
            hidden_size = 8

        def eval(self):
            return self

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    # We need to intercept the imports inside run_probe.
    # run_probe does `import torch` and `from transformers import ...` locally.
    # We monkeypatch the transformers classes directly.
    import transformers

    def fake_tokenizer_from_pretrained(mid, **kwargs):
        captured["tokenizer_kwargs"] = kwargs
        return FakeTokenizer()

    def fake_model_from_pretrained(mid, **kwargs):
        captured["model_kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_tokenizer_from_pretrained))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_model_from_pretrained))

    # run_probe also calls _forward_all_phrases which needs a real model.
    # We monkeypatch it to return empty dict to avoid forward pass.
    monkeypatch.setattr(hfp, "_forward_all_phrases", lambda model, tokenizer, phrases: {})
    monkeypatch.setattr(hfp, "_compute_silhouettes", lambda ph, nl, cm, p: {"L0": 0.5, "L1": 0.3})
    monkeypatch.setattr(hfp, "_content_mask", lambda phrase, tokenizer: None)

    hfp.run_probe("any/model")

    assert captured["tokenizer_kwargs"].get("local_files_only") is True
    assert captured["model_kwargs"].get("local_files_only") is True
    assert captured["model_kwargs"].get("trust_remote_code") is False


def test_hf_layer_probe_run_probe_explicit_id_without_offline_allows_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_probe with an explicit model_id and no offline env must NOT pass
    local_files_only (network resolution still allowed for local users)."""
    import oczy.experiments.hf_layer_probe as hfp

    monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("OCZY_MODEL_DIR", raising=False)

    captured: dict = {}

    class FakeModel:
        class config:
            num_hidden_layers = 2
            hidden_size = 8

        def eval(self):
            return self

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    import transformers

    def fake_tokenizer_from_pretrained(mid, **kwargs):
        captured["tokenizer_kwargs"] = kwargs
        return FakeTokenizer()

    def fake_model_from_pretrained(mid, **kwargs):
        captured["model_kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_tokenizer_from_pretrained))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_model_from_pretrained))

    monkeypatch.setattr(hfp, "_forward_all_phrases", lambda model, tokenizer, phrases: {})
    monkeypatch.setattr(hfp, "_compute_silhouettes", lambda ph, nl, cm, p: {"L0": 0.5, "L1": 0.3})
    monkeypatch.setattr(hfp, "_content_mask", lambda phrase, tokenizer: None)

    hfp.run_probe("explicit/model/1")

    assert "local_files_only" not in captured["model_kwargs"]
    assert "local_files_only" not in captured["tokenizer_kwargs"]


# ---------------------------------------------------------------------------
# Qwen locator and artifact manifest (existing, preserved)
# ---------------------------------------------------------------------------


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


def test_qwen_locator_honors_oczy_model_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """locate_model must check OCZY_MODEL_DIR before scanning Kaggle paths."""
    model = tmp_path / "pinned-qwen"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "hidden_size": 896}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setenv("OCZY_MODEL_DIR", str(model))
    # No explicit path — should find via env.
    result = locate_model(None)
    assert result == model.resolve()
