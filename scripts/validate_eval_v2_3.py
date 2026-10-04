#!/usr/bin/env python3
"""Validate the applied eval-v2.3 nonempty-score repair against a reference scorer.

Checks, over every shipped eval-v2 probe:

* blank and whitespace-only answers are rejected by the applied scorer
  (they passed 120/120 under the pre-v2.3 scorer);
* every expected answer still passes;
* nonempty matching is byte-for-byte identical to the reference scorer, so
  historical scores and failures reproduce unchanged;
* the eval/v2 manifest verifies without ``EVAL_CHANGE_APPROVED`` and binds
  the runtime sources (simulated drift is rejected).

The reference scorer defaults to the pre-patch revision of
``src/oczy/eval_v2/scoring.py`` (pass ``--reference-rev`` explicitly, e.g.
``HEAD`` before the v2.3 commit or its parent after).  No model is run and no
frozen score is recomputed: only synthetic answers are matched.

Usage:
    uv run python scripts/validate_eval_v2_3.py --reference-rev HEAD \
        --output experiments_logs/2026-10-04_eval_v2_3_applied.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORING_RELPATH = "src/oczy/eval_v2/scoring.py"
sys.path.insert(0, str(REPO_ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_show(revision: str, relpath: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relpath}"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"git show {revision}:{relpath} failed:\n{result.stderr}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-rev", required=True,
                        help="git revision providing the reference scoring.py")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import eval.v2 as eval_v2
    from eval.v2 import EvalIntegrityError, verify_manifest
    from oczy.experiments.organism_curriculum.dataset import build_curriculum

    reference_src = git_show(args.reference_rev, SCORING_RELPATH)
    current_src = (REPO_ROOT / SCORING_RELPATH).read_text()
    with tempfile.TemporaryDirectory(prefix="oczy-eval-v2-3-validate-") as directory:
        reference_path = Path(directory) / "reference_scoring.py"
        reference_path.write_text(reference_src)
        reference = load_module("eval_v2_scoring_reference", reference_path)
        current = load_module("eval_v2_scoring_current", REPO_ROOT / SCORING_RELPATH)

        counts = {
            "probes": 0,
            "reference_blank_correct": 0,
            "current_blank_correct": 0,
            "current_whitespace_correct": 0,
            "current_expected_correct": 0,
            "reference_expected_correct": 0,
            "nonempty_comparisons": 0,
            "nonempty_differences": 0,
        }
        for stage in build_curriculum():
            for episode in stage.episodes:
                for probe in episode.probes:
                    counts["probes"] += 1
                    counts["reference_blank_correct"] += bool(
                        reference.probe_matches("", probe, episode))
                    counts["current_blank_correct"] += bool(
                        current.probe_matches("", probe, episode))
                    counts["current_whitespace_correct"] += bool(
                        current.probe_matches(" \t\n", probe, episode))
                    counts["current_expected_correct"] += bool(
                        current.probe_matches(probe.expected, probe, episode))
                    counts["reference_expected_correct"] += bool(
                        reference.probe_matches(probe.expected, probe, episode))
                    answers = (probe.expected, episode.corrected_response,
                               "unrelated nonsense", "not " + probe.expected)
                    for answer in answers:
                        for semantic in (False, True):
                            counts["nonempty_comparisons"] += 1
                            if bool(reference.probe_matches(answer, probe, episode, semantic)) != \
                               bool(current.probe_matches(answer, probe, episode, semantic)):
                                counts["nonempty_differences"] += 1

        verify_manifest()
        manifest = json.loads((REPO_ROOT / "eval/v2/MANIFEST.json").read_text())
        bound_sources = sorted(k for k in manifest["files"] if k.startswith("../"))
        with tempfile.TemporaryDirectory(prefix="oczy-eval-v2-3-drift-") as drift_dir:
            sandbox = Path(drift_dir) / "repo"
            shutil.copytree(REPO_ROOT / "eval/v2", sandbox / "eval/v2")
            for relpath in bound_sources:
                target = (sandbox / "eval/v2" / relpath).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile((REPO_ROOT / "eval/v2" / relpath).resolve(), target)
            drift_target = (sandbox / "eval/v2" / bound_sources[0]).resolve()
            drift_target.write_text(drift_target.read_text() + "\n# simulated source drift\n")
            saved = (eval_v2._DATA_DIR, eval_v2._MANIFEST_PATH)
            eval_v2._DATA_DIR = sandbox / "eval/v2"
            eval_v2._MANIFEST_PATH = sandbox / "eval/v2/MANIFEST.json"
            try:
                verify_manifest()
            except EvalIntegrityError:
                source_drift_rejected = True
            else:
                source_drift_rejected = False
            finally:
                eval_v2._DATA_DIR, eval_v2._MANIFEST_PATH = saved

    assert counts["current_blank_correct"] == 0, counts
    assert counts["current_whitespace_correct"] == 0, counts
    assert counts["current_expected_correct"] == counts["probes"], counts
    assert counts["reference_expected_correct"] == counts["probes"], counts
    assert counts["nonempty_differences"] == 0, counts
    assert source_drift_rejected, "manifest did not bind runtime sources"

    report = {
        "status": "EVAL_V2_3_APPLIED_VALIDATED",
        "date": "2026-10-04",
        "reference_rev": args.reference_rev,
        "reference_scoring_sha256": hashlib.sha256(reference_src.encode()).hexdigest(),
        "current_scoring_sha256": hashlib.sha256(current_src.encode()).hexdigest(),
        "counts": counts,
        "manifest_version": manifest["version"],
        "bound_runtime_sources": bound_sources,
        "manifest_verifies_without_bypass": True,
        "runtime_source_drift_rejected": source_drift_rejected,
        "historical_scores_recomputed": False,
        "model_runs": 0,
        "meta_test_accessed": False,
        "remaining_limitations": [
            "Nonempty sense/substring false positives remain for the separately frozen A2 sense contract",
            "No replacement historical or live model scores",
            "Probe-level split remains unchanged; no unseen-rule claim",
        ],
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
