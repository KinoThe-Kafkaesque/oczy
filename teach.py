"""Interactive teacher REPL for the Oczy Plastic Cortex.

Usage:
    uv run python teach.py
    uv run python teach.py --reset
    uv run python teach.py --session work --checkpoint path/to/model.pkl

The REPL loads a persisted LMPlasticCortex checkpoint, remembers the last
query/answer pair, and accepts live corrections.  State is saved after every
turn so corrections survive restarts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from plastic_cortex.lm_cortex import LMPlasticCortex


DEFAULT_CHECKPOINT = Path("plastic-cortex/checkpoints/lm/model.pkl")
_REPRIMAND_MARKERS = ("no,", "no!", "wrong,", "bad,", "bad:")


def _session_dir() -> Path:
    """Return a system-appropriate directory for Oczy session files."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "oczy" / "sessions"


def _session_path(name: str) -> Path:
    return _session_dir() / f"{name}.pkl"


def _save_model(model: LMPlasticCortex, path: Path) -> None:
    """Persist model state, swallowing errors so the REPL never hangs."""
    try:
        model.save(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[warning: failed to save session: {exc}]")


def _is_correction(text: str) -> tuple[bool, str]:
    """Detect reprimand markers and return expected continuation."""
    lowered = text.strip().lower()
    for marker in _REPRIMAND_MARKERS:
        if lowered.startswith(marker):
            expected = text.strip()[len(marker) :].strip()
            return True, expected
    return False, ""


def _print_help() -> None:
    markers = ", ".join(repr(m) for m in _REPRIMAND_MARKERS)
    print(
        "Commands:\n"
        "  /help       show this message\n"
        "  /quit       leave the REPL\n"
        "  /reset      reset model state and delete the session file\n"
        "  /status     print a JSON status snapshot\n"
        "  /forget     reset fast-weight correction state\n"
        "  /save       persist model state immediately\n"
        "\n"
        f"Corrections: start with {markers}\n"
        "  /bad <expected>   correct the last query/answer pair"
    )


def _load_model(checkpoint_path: Path, reset: bool, session_path: Path) -> LMPlasticCortex:
    """Load from session, checkpoint, or start fresh."""
    if session_path.exists() and not reset:
        try:
            return LMPlasticCortex.load(session_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[failed to load session: {exc}; starting fresh]")
            session_path.unlink(missing_ok=True)

    if checkpoint_path.exists():
        try:
            return LMPlasticCortex.load(checkpoint_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[failed to load checkpoint {checkpoint_path}: {exc}; starting fresh]")

    if not reset:
        print(
            "No trained checkpoint found.\n"
            "Run: uv run python plastic-cortex/scripts/train_lm.py\n"
            "Or start with --reset to use a blank model."
        )
        raise SystemExit(1)

    print("[starting with a blank model]")
    return LMPlasticCortex()


def _teach_loop(model: LMPlasticCortex, session_path: Path) -> None:
    last_query: str | None = None
    last_answer: str | None = None

    while True:
        try:
            user_input = input("Teach > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving session...")
            _save_model(model, session_path)
            print("Bye.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()

        if lowered in {"/quit", "quit", "exit", "/exit"}:
            _save_model(model, session_path)
            print("Bye.")
            break

        if lowered == "/help":
            _print_help()
            continue

        if lowered == "/status":
            print(json.dumps(model.status(), indent=2, default=str))
            continue

        if lowered == "/forget":
            model.reset_state()
            last_query = None
            last_answer = None
            print("[fast-weight state forgotten]")
            _save_model(model, session_path)
            continue

        if lowered == "/reset":
            model.reset_state()
            if session_path.exists():
                session_path.unlink()
            last_query = None
            last_answer = None
            print("[session state reset and persisted state deleted]")
            continue

        if lowered == "/save":
            _save_model(model, session_path)
            print("[session saved]")
            continue

        if lowered.startswith("/bad "):
            expected = user_input[len("/bad ") :].strip()
            if not expected:
                print("[expected answer missing]")
                continue
            if last_query is None:
                print("[no previous query to correct]")
                continue
            model.correct(last_query, expected)
            last_answer = expected
            print("[recorded correction]")
            _save_model(model, session_path)
            continue

        is_correction, expected = _is_correction(user_input)
        if is_correction:
            if not expected:
                print("[expected answer missing]")
                continue
            if last_query is None:
                print("[no previous query to correct]")
                continue
            model.correct(last_query, expected)
            last_answer = expected
            print("[recorded correction]")
            _save_model(model, session_path)
            continue

        # Normal query.
        answer = model.answer(user_input)
        last_query = user_input
        last_answer = answer
        print(answer)
        _save_model(model, session_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Teach the Oczy Plastic Cortex with live corrections."
    )
    parser.add_argument(
        "--session",
        default="teacher",
        help="Session name used for persisted state (default: teacher).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start fresh: delete any previously saved session state.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a trained LMPlasticCortex checkpoint.",
    )
    args = parser.parse_args(argv)

    session_path = _session_path(args.session)

    if args.reset and session_path.exists():
        session_path.unlink()
        print(f"[reset session {args.session!r}]")

    _session_dir().mkdir(parents=True, exist_ok=True)

    model = _load_model(args.checkpoint, args.reset, session_path)

    if session_path.exists() and not args.reset:
        loaded_from = "session"
    elif args.checkpoint.exists():
        loaded_from = "checkpoint"
    else:
        loaded_from = "blank model"
    print(f"LMPlasticCortex ready (loaded from {loaded_from}).")

    _teach_loop(model, session_path)
    return 0


if __name__ == "__main__":
    """Entry point for the teacher REPL."""
    raise SystemExit(main())
