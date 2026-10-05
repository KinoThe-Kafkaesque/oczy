"""Archive a completed audited DEV run with its exact source and checksums."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def archive(root, run, report, destination):
    repo = Path(__file__).resolve().parents[1]
    summary = json.loads(report.with_suffix(".json").read_text())
    if not summary["verification"]["passed"]:
        raise ValueError("Only independently audited runs may be archived")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    assert summary["manifest_sha256"] == manifest["manifest_sha256"]
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root, destination / "instrument", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(run, destination / "run", ignore=shutil.ignore_patterns("__pycache__"))
    sources = set(manifest.get("execution_sources", {}))
    if not sources:
        sources = {p for p in manifest["files"] if p.endswith(".py")}
    sources.update(p.relative_to(repo).as_posix() for directory in (repo / "scripts/diversity_dev", repo / "scripts/regression_dev") for p in directory.glob("*.py"))
    sources.add("scripts/archive_dev_workbench.py")
    for path in sorted(sources):
        target = destination / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / path, target)
    for suffix in (".md", ".json"):
        shutil.copyfile(report.with_suffix(suffix), destination / report.with_suffix(suffix).name)
    shutil.copyfile(repo / "experiments_logs/artifacts/2026-09-13_capability_validation_v1/selected_runtime_provenance.json", destination / "selected_runtime_provenance.json")
    (destination / "VERIFICATION.json").write_text(json.dumps(summary["verification"], indent=2) + "\n")
    (destination / "README.md").write_text(f"# {manifest['instrument_id']} archive\n\nManifest: `{manifest['manifest_sha256']}`.\n\nCompleted local DEV run, independently audited. Original instruments are preserved.\nThis archive includes the run, registered instrument, executed/auditing source,\nmodel/runtime provenance, report and per-file checksums. Model weights are external.\nNo meta-test or optimizer beyond the registered protocol was used.\n\nVerify with `sha256sum -c SHA256SUMS` from this directory.\n")
    files = sorted(p for p in destination.rglob("*") if p.is_file())
    lines = [hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(destination).as_posix() for p in files]
    (destination / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "report", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(archive(args.root, args.run, args.report, args.destination))
