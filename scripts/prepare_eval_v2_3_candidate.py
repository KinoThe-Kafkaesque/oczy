"""Build and shadow-check an unapplied eval-v2.3 repair for human review."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(repo, output):
    from oczy.eval_v2.scoring import probe_matches
    from oczy.experiments.organism_curriculum.dataset import build_curriculum

    output.mkdir(parents=True, exist_ok=False)
    score_path = "src/oczy/eval_v2/scoring.py"
    init_path = "eval/v2/__init__.py"
    manifest_path = "eval/v2/MANIFEST.json"
    before = {name: (repo / name).read_text() for name in (score_path, init_path, manifest_path)}
    needle = "    result = _base_match(answer, expected, ambiguous_token, match_mode)"
    assert before[score_path].count(needle) == 1
    score = before[score_path].replace(needle, "    if not _normalize(answer) or not _normalize(expected):\n        return False\n" + needle)
    init = before[init_path].replace('return {"version": "v2.2", "files": files}', '''for relpath in FROZEN_SOURCE_FILES:
        files[relpath] = _sha256(_DATA_DIR / relpath)
    return {"version": "v2.3", "files": files}''')
    init = init.replace("Current data version: v2.2 (2026-07-11 protocol repair", "Current data version: v2.3 (2026-09-12 nonempty-score and source-binding repair;\nthe prior v2.2 protocol remains unchanged. Historical 2026-07-11 protocol repair")
    needle = '_MANIFEST_PATH = _DATA_DIR / "MANIFEST.json"'
    source_files = ("src/oczy/eval_v2/scoring.py", "src/oczy/eval_v2/validation.py",
                    "src/oczy/experiments/organism_curriculum/dataset.py",
                    "src/oczy/experiments/organism_curriculum/run_curriculum.py")
    declaration = "\n\n# Runtime scoring, validation, splitting and protocol are instrument assets.\nFROZEN_SOURCE_FILES = (\n" + "".join(f'    "../../{name}",\n' for name in source_files) + ")"
    init = init.replace(needle, needle + declaration)
    with tempfile.TemporaryDirectory(prefix="oczy-eval-v2-3-review-") as directory:
        candidate_root = Path(directory)
        shutil.copytree(repo / "eval/v2", candidate_root / "eval/v2", ignore=shutil.ignore_patterns("__pycache__"))
        for name in source_files:
            dest = candidate_root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / name, dest)
        (candidate_root / score_path).write_text(score)
        (candidate_root / init_path).write_text(init)
        instrument = load_module("candidate_eval_integrity", candidate_root / init_path)
        instrument.write_manifest()
        instrument.verify_manifest()
        candidate = load_module("candidate_eval_scoring", candidate_root / score_path)
        counts = {"probes": 0, "old_blank_correct": 0, "candidate_blank_correct": 0,
                  "candidate_whitespace_correct": 0, "candidate_expected_correct": 0,
                  "nonempty_comparisons": 0, "nonempty_differences": 0}
        for stage in build_curriculum():
            for episode in stage.episodes:
                for probe in episode.probes:
                    counts["probes"] += 1
                    counts["old_blank_correct"] += probe_matches("", probe, episode)
                    counts["candidate_blank_correct"] += candidate.probe_matches("", probe, episode)
                    counts["candidate_whitespace_correct"] += candidate.probe_matches(" \t\n", probe, episode)
                    counts["candidate_expected_correct"] += candidate.probe_matches(probe.expected, probe, episode)
                    for answer in (probe.expected, episode.corrected_response, "unrelated nonsense", "not " + probe.expected):
                        for semantic in (False, True):
                            counts["nonempty_comparisons"] += 1
                            counts["nonempty_differences"] += probe_matches(answer, probe, episode, semantic) != candidate.probe_matches(answer, probe, episode, semantic)
        assert counts["candidate_blank_correct"] == counts["candidate_whitespace_correct"] == 0
        assert counts["candidate_expected_correct"] == counts["probes"]
        assert counts["nonempty_differences"] == 0
        (candidate_root / score_path).write_text(score + "\n# simulated source drift\n")
        try:
            instrument.verify_manifest()
        except instrument.EvalIntegrityError:
            source_tamper_rejected = True
        else:
            raise AssertionError("Candidate manifest did not bind runtime scorer")
        after = {score_path: score, init_path: init,
                 manifest_path: (candidate_root / manifest_path).read_text()}
    patch = "".join("".join(difflib.unified_diff(before[name].splitlines(True), after[name].splitlines(True),
                       fromfile="a/" + name, tofile="b/" + name)) for name in before)
    (output / "eval_v2_3.patch").write_text(patch)
    report = {"status": "CANDIDATE_NOT_APPLIED_AWAITING_HUMAN_SIGNOFF", "version": "v2.3",
              "counts": counts, "runtime_source_tamper_rejected": source_tamper_rejected,
              "historical_scores_recomputed": False, "model_runs": 0,
              "source_before_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in before.items()},
              "source_after_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in after.items()},
              "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
              "remaining_limitations": ["Nonempty sense/substring false positives remain for a separate amendment",
                                        "No replacement historical or live model scores",
                                        "Probe-level split remains unchanged; no unseen-rule claim"]}
    (output / "SHADOW_VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1], args.output)


if __name__ == "__main__":
    main()
