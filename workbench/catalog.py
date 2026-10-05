"""Create a portable, source-backed view from audited research artifacts."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
STUDY = "experiments/scoped-diversity-dev-v3"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(study_run=None, regression_run=None):
    DATA.mkdir(exist_ok=True)
    old = read(ROOT / "experiments_logs/2026-09-13_capability_validation_v1.json")
    previous = read(ROOT / "experiments_logs/2026-09-13_context_preservation_dev_v2.json")
    language = read(ROOT / "experiments_logs/2026-09-13_language_interface_dev_v3.json")
    decoder = read(ROOT / "experiments_logs/2026-09-13_decoder_parity_dev_v1.json")
    new_path = ROOT / "experiments_logs/2026-09-13_scoped_diversity_dev_v3.json"
    current = read(new_path) if new_path.exists() else None
    regression_path = ROOT / "experiments_logs/2026-09-13_capability_regression_dev_v1.json"
    regression = read(regression_path) if regression_path.exists() else None
    rows = current["rows"] if current else previous["rows"]
    state = ("Completed with bookkeeping recovery" if current.get("execution_recovery") else "Completed") if current else "Study in progress; showing prior DEV-v2 evidence"
    evidence = {"title": "Oczy workbench", "as_of": datetime.now(ZoneInfo("Africa/Casablanca")).date().isoformat(), "study_state": state,
                "study": current or previous, "current_study": current is not None, "regression": regression,
                "rows": rows, "action_rows": old["action_rows"], "language": language["totals"],
                "decoder_qualified": decoder["candidate_qualified"], "accumulation": old["accumulation"],
                "correction": old["correction"], "composition": old["composition"], "compression": old["compression"],
                "sources": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in
                            [ROOT / "experiments_logs/2026-09-13_capability_validation_v1.json", ROOT / "experiments_logs/2026-09-13_context_preservation_dev_v2.json",
                             ROOT / "experiments_logs/2026-09-13_language_interface_dev_v3.json", ROOT / "experiments_logs/2026-09-13_decoder_parity_dev_v1.json"] +
                            ([new_path] if current else []) + ([regression_path] if regression else [])]}
    if regression_run:
        evidence["candidate_actions"] = read(regression_run / "replay/candidate_actions.json")["rows"]
        evidence["candidate_probes"] = read(regression_run / "replay/candidate_probes.json")["rows"]
    (DATA / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if study_run:
        import shutil
        destination = DATA / "states"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(study_run / "released_diversity", destination)
        (DATA / "state_source.json").write_text(json.dumps({"study": STUDY, "state_manifest_sha256": digest(destination / "state_manifest.json")}, indent=2) + "\n")
    return evidence


def verify_data():
    from scripts.diversity_dev.contract import manifest_at
    from scripts.regression_dev.contract import verify
    manifest_at(ROOT / STUDY)
    verify(ROOT / "experiments/capability-regression-dev-v1")
    evidence = read(DATA / "evidence.json")
    for source in evidence["sources"]:
        if digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("Evidence source changed: " + source["path"])
    source_path = DATA / "state_source.json"
    if source_path.exists():
        source = read(source_path)
        if digest(DATA / "states/state_manifest.json") != source["state_manifest_sha256"]:
            raise ValueError("State release changed")
        states = read(DATA / "states/state_manifest.json")
        if states["manifest_sha256"] != read(ROOT / STUDY / "MANIFEST.json")["manifest_sha256"]:
            raise ValueError("State instrument differs")
        for state in states["states"]:
            for kind in ("initial", "trained"):
                if digest(DATA / "states" / state[kind]["path"]) != state[kind]["sha256"]:
                    raise ValueError("Learned state changed")
    return {"verified_sources": len(evidence["sources"]), "states_present": source_path.exists()}
