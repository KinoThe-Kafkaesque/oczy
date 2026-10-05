"""Register all replay inputs before executing the regression worker."""


from scripts.regression_dev.contract import (
    CAP,
    DIVERSITY,
    PILOT,
    REPLAYS,
    REPO,
    canonical,
    read,
    sha,
    write_json,
)


def main():
    root = REPO / "experiments/capability-regression-dev-v1"
    root.mkdir(exist_ok=True)
    target = root / "MANIFEST.json"
    if target.exists():
        raise ValueError("Already frozen")
    parents = ["experiments/capability-validation-v1", "experiments/r23.5-serialization-dev/pilot_v1", DIVERSITY,
               "experiments/language-interface-dev-v3", "experiments/decoder-parity-dev-v1"]
    files = set()
    for parent in parents:
        for file in (REPO / parent).rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts:
                files.add(file.relative_to(REPO).as_posix())
        files.update(read(REPO / parent / "MANIFEST.json")["execution_sources"])
    for directory in (f"{CAP}/run/released_states", f"{PILOT}/released_states", f"{PILOT}/released_text"):
        files.update(p.relative_to(REPO).as_posix() for p in (REPO / directory).rglob("*") if p.is_file())
    files.update(path for path, _ in REPLAYS.values())
    files.update([f"{CAP}/run/reference/stage3-active.txt", f"{CAP}/selected_runtime_provenance.json",
                  "experiments_logs/2026-09-13_language_interface_dev_v3.json", "experiments_logs/2026-09-13_decoder_parity_dev_v1.json"])
    files.update(f"scripts/regression_dev/{p}.py" for p in ("__init__", "contract", "worker", "run"))
    manifest = {"instrument_id": "capability-regression-dev-v1", "scope": "Local DEV, no optimization or meta-test",
                "human_authorization": "get on the next objective then check previous ones against regression then see if you can make one packaged expeience",
                "candidate_manifest_sha256": read(REPO / DIVERSITY / "MANIFEST.json")["manifest_sha256"],
                "candidate_selection": "All three final states from the registered diversity run; no seed or checkpoint selection",
                "counts": {**{k: n for k, (_, n) in REPLAYS.items()}, "direct_operation": 32, "decoder_single": 32,
                           "decoder_batch4": 32, "candidate_probes": 160, "candidate_actions": 48},
                "comparison": "Complete historical row equality; exact scalar language/decoder outputs. Candidate scores use old v1 stage2 probes/actions unchanged.",
                "limitations": "Replay is one read-only process, not original phase isolation. Old failures reproduced are integrity evidence, not capability passes. Candidate joint learning is not sequential accumulation.",
                "files": {p: sha((REPO / p).read_bytes()) for p in sorted(files)}}
    manifest["manifest_sha256"] = sha(canonical(manifest))
    write_json(target, manifest)
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    main()
