"""Interactive REPL for the Oczy Plastic World Model Agent.

Usage:
    uv run python chat.py
    uv run python chat.py -- "hello" "update the profile"

The agent maintains a single OrganismAgent session. Corrections are detected
heuristically when the user input starts with markers like "no,", "wrong,",
"correct:", or "expected:". Plain text just issues the next request.

Type `quit`, `exit`, or press Ctrl-D/Ctrl-C to leave.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from experiments.organism import OrganismAgent


_CORRECTION_MARKERS = ("no,", "no:", "wrong,", "wrong:", "correct:", "expected:", "correction:")


def _is_correction(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(marker) for marker in _CORRECTION_MARKERS)


def _print_help() -> None:
    print(
        "Commands:  /help  /reset  /status  /consolidate  /quit\n"
        "Corrections: start with 'no,' / 'wrong,' / 'correct:' / 'expected:'"
    )


def _chat_loop(agent: OrganismAgent, initial_messages: Sequence[str] | None = None) -> None:
    if initial_messages:
        for message in initial_messages:
            print(f">>> {message}")
            answer = agent.answer(message)
            print(f"<-- {answer}\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"/quit", "quit", "exit", "/exit"}:
            print("Bye.")
            break
        if lowered == "/help":
            _print_help()
            continue
        if lowered == "/reset":
            agent.reset_state()
            print("[session state reset]")
            continue
        if lowered == "/status":
            import json

            print(json.dumps(agent.status(), indent=2, default=str))
            continue
        if lowered == "/consolidate":
            agent.consolidate()
            print("[consolidated raw traces]")
            continue

        if _is_correction(user_input):
            agent.correct(user_input, "")  # expected answer is extracted heuristically
            print("[correction recorded]")
        else:
            answer = agent.answer(user_input)
            print(f"<-- {answer}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the Oczy OrganismAgent.")
    parser.add_argument(
        "messages",
        nargs="*",
        help="Optional initial messages to send before entering the REPL.",
    )
    parser.add_argument(
        "--config",
        default="{}",
        help="JSON config passed to OrganismAgent (default: '{}').",
    )
    args = parser.parse_args(argv)

    config: dict = {}
    if args.config:
        import json

        config = json.loads(args.config)

    agent = OrganismAgent(config)
    print("Oczy OrganismAgent ready. Type /help for commands.")
    _chat_loop(agent, initial_messages=args.messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
