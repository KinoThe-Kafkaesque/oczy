"""Interactive, persistent REPL for the Oczy Plastic World Model Agent.

Usage:
    uv run python chat.py
    uv run python chat.py --reset
    uv run python chat.py --session work -- "update the profile"

The agent maintains a single OrganismAgent session. State is persisted after
every turn to a user-level session file, so corrections survive restarts.

Corrections are detected heuristically when input starts with "no,", "wrong,",
"correct:", or "expected:". Plain text issues the next request.

Type `quit`, `exit`, or press Ctrl-D/Ctrl-C to leave.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from experiments.organism import OrganismAgent


_CORRECTION_MARKERS = ("no,", "no:", "wrong,", "wrong:", "correct:", "expected:", "correction:")
_TASK_TOKENS = (
    "profile", "model", "batch", "file", "key", "module",
    "service", "branch", "table", "cell", "record", "run",
)


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


def _has_task_token(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _TASK_TOKENS)


def _is_correction(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(marker) for marker in _CORRECTION_MARKERS)


def _print_help() -> None:
    tokens = ", ".join(_TASK_TOKENS)
    print(
        "Commands:  /help  /reset  /status  /consolidate  /quit\n"
        "Corrections: start with 'no,' / 'wrong,' / 'correct:' / 'expected:'\n"
        f"Domain: this agent only handles ambiguous software commands: {tokens}\n"
        "Backend: pass --backend lm (and --lm-checkpoint PATH) to use LMPlasticCortex."
    )


def _save_agent(agent: OrganismAgent, path: Path) -> None:
    """Persist agent state, ignoring errors so chat never hangs."""
    try:
        agent.save(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[warning: failed to save session: {exc}]")


def _chat_loop(
    agent: OrganismAgent,
    session_path: Path,
    backend: str = "default",
    initial_messages: Sequence[str] | None = None,
) -> None:
    tokens = ", ".join(_TASK_TOKENS)
    if backend == "lm":
        fallback = None
    else:
        fallback = (
            "I can help with ambiguous software commands such as "
            f"{tokens}. Try one of those, or type /help."
        )
    if initial_messages:
        for message in initial_messages:
            print(f">>> {message}")
            if fallback is None or _has_task_token(message):
                print(f"<-- {agent.answer(message)}\n")
            else:
                print(f"<-- {fallback}\n")
        _save_agent(agent, session_path)

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving session...")
            _save_agent(agent, session_path)
            print("Bye.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"/quit", "quit", "exit", "/exit"}:
            _save_agent(agent, session_path)
            print("Bye.")
            break
        if lowered == "/help":
            _print_help()
            continue
        if lowered == "/reset":
            agent.reset_state()
            if session_path.exists():
                session_path.unlink()
            print("[session state reset and persisted state deleted]")
            continue
        if lowered == "/status":
            import json

            print(json.dumps(agent.status(), indent=2, default=str))
            continue
        if lowered == "/consolidate":
            agent.consolidate()
            print("[consolidated raw traces]")
            _save_agent(agent, session_path)
            continue

        if _is_correction(user_input):
            agent.correct(user_input, "")  # expected answer is extracted heuristically
            print("[correction recorded]")
            _save_agent(agent, session_path)
        elif fallback is None or _has_task_token(user_input):
            answer = agent.answer(user_input)
            print(f"<-- {answer}")
            _save_agent(agent, session_path)
        else:
            print(f"<-- {fallback}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the Oczy OrganismAgent.")
    parser.add_argument(
        "messages",
        nargs="*",
        help="Optional initial messages to send before entering the REPL.",
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session name used for persisted state (default: default).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start fresh: delete any previously saved session state.",
    )
    parser.add_argument(
        "--config",
        default="{}",
        help="JSON config passed to OrganismAgent (default: '{}').",
    )
    parser.add_argument(
        "--backend",
        choices=["default", "lm"],
        default="default",
        help="Fast organ backend to use: default PlasticCortex or LMPlasticCortex (default: default).",
    )
    parser.add_argument(
        "--lm-checkpoint",
        default="plastic-cortex/checkpoints/lm/model.pkl",
        help="Path to a saved LMPlasticCortex checkpoint (only used with --backend lm).",
    )
    args = parser.parse_args(argv)

    config: dict = {"backend": args.backend, "lm_checkpoint": args.lm_checkpoint}
    if args.config:
        import json

        config.update(json.loads(args.config))

    session_path = _session_path(args.session)

    if args.reset and session_path.exists():
        session_path.unlink()
        print(f"[reset session {args.session!r}]")

    if session_path.exists():
        try:
            agent = OrganismAgent.load(session_path)
            print(f"Oczy OrganismAgent ready (loaded session {args.session!r}).")
        except Exception as exc:  # noqa: BLE001
            print(f"[failed to load session: {exc}; starting fresh]")
            session_path.unlink(missing_ok=True)
            agent = OrganismAgent(config)
    else:
        agent = OrganismAgent(config)
        print("Oczy OrganismAgent ready (new session).")
    _chat_loop(agent, session_path, backend=args.backend, initial_messages=args.messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
