"""Build and verify a portable archive using an explicit research file inventory."""

import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from workbench.catalog import DATA, ROOT, digest, read, verify_data


def verify_package(root=ROOT):
    seal = root / "PACKAGE_MANIFEST.json"
    if not seal.exists():
        return {"package_seal": "repository mode; no archive seal"}
    inventory = read(seal)["files"]
    for path, expected in inventory.items():
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or digest(root / relative) != expected:
            raise ValueError("Packaged file changed: " + path)
    return {"package_seal": "verified", "packaged_files": len(inventory)}


def build(output):
    verify_data()
    files = set()
    for directory in ("workbench", "src/oczy", "scripts/diversity_dev", "scripts/regression_dev"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".json", ".html", ".css", ".js", ".md", ".npy"):
                files.add(path.relative_to(ROOT).as_posix())
    regression_manifest = read(ROOT / "experiments/capability-regression-dev-v1/MANIFEST.json")
    files.update(regression_manifest["files"])
    files.update(["experiments/capability-regression-dev-v1/MANIFEST.json", "experiments/capability-regression-dev-v1/README.md",
                  "infrastructure/__init__.py", "infrastructure/kaggle/__init__.py"])
    evidence = read(DATA / "evidence.json")
    files.update(source["path"] for source in evidence["sources"])
    files.update(["scripts/archive_dev_workbench.py", "scripts/report_dev_workbench.py"])
    for name in ("2026-09-13_scoped_diversity_dev_v3", "2026-09-13_capability_regression_dev_v1"):
        directory = ROOT / "experiments_logs/artifacts" / name
        if directory.exists():
            files.update(p.relative_to(ROOT).as_posix() for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for name in ("2026-09-13_scoped_diversity_dev_v3", "2026-09-13_capability_regression_dev_v1", "2026-09-13_capability_validation_v1",
                 "2026-09-13_context_preservation_dev_v2", "2026-09-13_language_interface_dev_v3", "2026-09-13_decoder_parity_dev_v1"):
        path = ROOT / "experiments_logs" / (name + ".md")
        if path.exists():
            files.add(path.relative_to(ROOT).as_posix())
    output = output.resolve()
    if output.exists():
        raise ValueError("Archive already exists; choose a new output name")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="oczy-package-") as temporary:
        stage = Path(temporary) / "oczy-workbench"
        stage.mkdir()
        for relative in sorted(files):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        readme = (ROOT / "workbench/README.md").read_text()
        (stage / "README.md").write_text(readme.replace("](../experiments_logs/", "](experiments_logs/"))
        (stage / "PACKAGE_MANIFEST.json").write_text(json.dumps({"name": "oczy-workbench-dev-v1", "model_weights_included": False,
                   "runtime_included": False, "files": {p.relative_to(stage).as_posix(): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}}, indent=2) + "\n")
        verify_package(stage)
        with tarfile.open(output, "w:gz") as archive:
            archive.add(stage, arcname=stage.name)
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": digest(output), "included_files": len(files) + 2}
