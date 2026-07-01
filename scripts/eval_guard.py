#!/usr/bin/env python3
"""Guard eval assets from unauthorized changes.

Exits 0 if no protected files were changed in the given revision range,
exits 1 (with a clear message listing offending files) otherwise.

Set EVAL_CHANGE_APPROVED=1 in the environment together with --allow to
explicitly permit a protected change (e.g. a deliberate eval update).
"""

import argparse
import os
import subprocess
import sys

PROTECTED_PATHS = [
    "experiments/organism_curriculum/",
    "research/",
    "lanes/",
    "eval/",
]

DEFAULT_RANGE = "origin/main...HEAD"
FALLBACK_RANGE = "HEAD~1...HEAD"


def changed_files(revision_range):
    """Return the list of files changed in `revision_range`.

    Tries the requested range first; if git reports a non-zero exit
    (e.g. the ref is unknown), falls back to `HEAD~1...HEAD`.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", revision_range],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["git", "diff", "--name-only", FALLBACK_RANGE],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            sys.stderr.write(
                "eval_guard: git diff failed for both "
                f"{revision_range!r} and {FALLBACK_RANGE!r}\n"
            )
            sys.stderr.write(result.stderr)
            return None
    files = [line for line in result.stdout.splitlines() if line.strip()]
    return files


def is_protected(path):
    """True if `path` falls under any protected path (prefix match)."""
    for prefix in PROTECTED_PATHS:
        if path.startswith(prefix):
            return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Guard eval assets from unauthorized changes.",
    )
    parser.add_argument(
        "revision_range",
        nargs="?",
        default=DEFAULT_RANGE,
        help=f"git revision range to check (default: {DEFAULT_RANGE})",
    )
    parser.add_argument(
        "--allow",
        action="store_true",
        help="permit a protected change; requires EVAL_CHANGE_APPROVED=1",
    )
    args = parser.parse_args(argv)

    files = changed_files(args.revision_range)
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
