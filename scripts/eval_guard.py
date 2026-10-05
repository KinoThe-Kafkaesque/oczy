#!/usr/bin/env python3
"""Guard eval assets from unauthorized changes.

Checks the revision range and staged, unstaged and untracked workspace files.
Exits 0 if no protected files changed, and 1 with offending paths otherwise.

Set EVAL_CHANGE_APPROVED=1 in the environment together with --allow to
explicitly permit a protected change (e.g. a deliberate eval update).
"""

import argparse
import os
import subprocess
import sys

PROTECTED_PATHS = [
    "scripts/regression_dev/",
    "experiments/capability-regression-dev-v1/",
    "scripts/diversity_dev/",
    "experiments/scoped-diversity-dev-v3/",
    "experiments/organism_curriculum/",
    "src/oczy/experiments/organism_curriculum/",
    "src/oczy/eval_v2/",
    "src/oczy/experiments/tool_calling_curriculum/",
    "research/",
    "lanes/",
    "eval/",
    "experiments/capability-validation-v1/",
    "experiments/context-preservation-dev-v2/",
    "experiments/language-interface-dev-v3/",
    "experiments/decoder-parity-dev-v1/",
    # R20 meta_cortex/v3 DEV instrument (gate G1 freeze, signed S1/S2/S3
    # 2026-10-04). The frozen instrument and its task-support-repaired
    # generator lineage are the measuring instrument for every downstream
    # gate; an unauthorized byte change must be refused, not silently merged.
    "experiments/r20-taskgen-v3-dev/",
    "experiments/r20-task-support-repair-v1/",
    # R20 meta_cortex/v4-r20 DEV instrument (successor freeze, user-authorized
    # 2026-10-05 via manager, kanban t_37e96ee1). Carries the approved
    # 2026-09-11 prompt amendments and the authorized max_new_tokens=128.
    "experiments/r20-taskgen-v4-r20-dev/",
]

# Keep learner/model implementation editable while guarding the instrument.
PROTECTED_FILES = {
    "scripts/probe_language_interface.py",
    "scripts/probe_decoder_parity.py",
    "src/oczy/experiments/meta_cortex/generation_v2.py",
    "scripts/context_preservation_contract.py",
    "scripts/context_preservation_worker.py",
    "scripts/prepare_context_preservation.py",
    "scripts/run_context_preservation.py",
    "scripts/capability_validation_contract.py",
    "scripts/capability_validation_actions.py",
    "scripts/capability_validation_worker.py",
    "scripts/prepare_capability_validation.py",
    "scripts/run_capability_validation.py",
    "src/oczy/experiments/meta_cortex/taskgen.py",
    "src/oczy/experiments/meta_cortex/_sealed_taskgen.py",
    "src/oczy/experiments/meta_cortex/calibration.py",
    "src/oczy/experiments/meta_cortex/contracts.py",
    "src/oczy/experiments/meta_cortex/instrument.py",
    "src/oczy/experiments/meta_cortex/instrument_contracts.py",
    "src/oczy/experiments/meta_cortex/taskgen_v2.py",
    "src/oczy/experiments/meta_cortex/instrument_v3.py",
    "src/oczy/experiments/meta_cortex/instrument_v4_r20.py",
    "src/oczy/experiments/meta_cortex/organ.py",
    "src/oczy/experiments/meta_cortex/_organ_identity_probe.py",
    "scripts/r20_task_support_check.py",
    "scripts/freeze_r20_v3_instrument.py",
    "scripts/freeze_r20_v4_r20_instrument.py",
    "scripts/materialize_r20_v4_r20.py",
    "scripts/probe_r20_v4_r20_oracle.py",
    "src/oczy/experiments/r24_tiny_decoder/toy_catalog_v3.py",
    "src/oczy/experiments/r24_tiny_decoder/corpus_v2.py",
    "src/oczy/experiments/r24_tiny_decoder/oracle.py",
    "src/oczy/experiments/r24_tiny_decoder/oracle_v2.py",
}

DEFAULT_RANGE = "origin/main...HEAD"
FALLBACK_RANGE = "HEAD~1...HEAD"


def changed_files(revision_range, *, allow_fallback=False):
    """Return committed and workspace paths, preserving both sides of renames.

    Only an implicit default range may fall back when its upstream is absent.
    An invalid explicit range fails closed rather than auditing another range.
    """
    diff = ["git", "diff", "--name-only", "--no-renames", "-z"]
    result = subprocess.run(
        [*diff, revision_range, "--"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and allow_fallback:
        sys.stderr.write(f"eval_guard: default range {revision_range!r} unavailable; checking {FALLBACK_RANGE!r}.\n")
        result = subprocess.run(
            [*diff, FALLBACK_RANGE, "--"],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        sys.stderr.write(f"eval_guard: cannot inspect range {revision_range!r}\n{result.stderr}")
        return None
    files = set(filter(None, result.stdout.split("\0")))
    for command in ([*diff, "--cached", "--"], [*diff, "--"],
                    ["git", "ls-files", "--others", "--exclude-standard", "-z"]):
        workspace = subprocess.run(command, capture_output=True, text=True)
        if workspace.returncode != 0:
            sys.stderr.write(f"eval_guard: cannot inspect workspace\n{workspace.stderr}")
            return None
        files.update(filter(None, workspace.stdout.split("\0")))
    return sorted(files)


def is_protected(path):
    """True if `path` falls under any protected path (prefix match)."""
    return path in PROTECTED_FILES or any(path.startswith(prefix) for prefix in PROTECTED_PATHS)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Guard eval assets from unauthorized changes.",
    )
    parser.add_argument(
        "revision_range",
        nargs="?",
        default=None,
        help=f"git revision range to check (default: {DEFAULT_RANGE})",
    )
    parser.add_argument(
        "--allow",
        action="store_true",
        help="permit a protected change; requires EVAL_CHANGE_APPROVED=1",
    )
    args = parser.parse_args(argv)

    files = changed_files(args.revision_range or DEFAULT_RANGE, allow_fallback=args.revision_range is None)
    if files is None:
        return 1

    offending = [f for f in files if is_protected(f)]

    if offending:
        if args.allow:
            if os.environ.get("EVAL_CHANGE_APPROVED") == "1":
                print(
                    "eval_guard: protected eval assets changed, but "
                    "EVAL_CHANGE_APPROVED=1 is set -- proceeding."
                )
                return 0
            sys.stderr.write(
                "eval_guard: --allow given but EVAL_CHANGE_APPROVED is not "
                "set to 1. Set EVAL_CHANGE_APPROVED=1 to approve an eval "
                "change.\n"
            )
            return 1

        sys.stderr.write(
            "eval_guard: refusing to proceed; protected eval assets were "
            "changed:\n"
        )
        for f in offending:
            sys.stderr.write(f"  {f}\n")
        sys.stderr.write(
            "\nIf this change is intentional, re-run with "
            "--allow and EVAL_CHANGE_APPROVED=1.\n"
        )
        return 1

    print("eval_guard: no protected eval assets changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
