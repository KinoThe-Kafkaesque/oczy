"""Build a training corpus from local Codex CLI session logs.

Reads ~/.codex/sessions/**/*.jsonl, extracts user/assistant messages, filters
system noise, and writes a plain-text file suitable for LMPlasticCortex.

Usage:
    uv run python plastic-cortex/scripts/build_codex_corpus.py
    uv run python plastic-cortex/scripts/build_codex_corpus.py --role assistant --lines 2000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter


SESSIONS_DIR = Path.home() / ".codex" / "sessions"
DEFAULT_OUTPUT = Path("plastic-cortex/data/codex_corpus.txt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Codex CLI session logs into a training corpus."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output text file (default: plastic-cortex/data/codex_corpus.txt).",
    )
    parser.add_argument(
        "--role",
        choices=["all", "user", "assistant"],
        default="all",
        help="Which speaker roles to include (default: all).",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=None,
        help="Limit output to the first N messages (useful for quick experiments).",
    )
    parser.add_argument(
        "--strip-code",
        action="store_true",
        default=True,
        help="Replace code blocks / inline code with a [code] placeholder.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        default=True,
        help="Print corpus statistics after writing.",
    )
    return parser.parse_args(argv)


def clean_text(text: str, strip_code: bool) -> str:
    """Normalize whitespace and optionally strip code fences."""
    text = " ".join(text.split())
    if strip_code:
        text = re.sub(r"```[\s\S]*?```", " [code] ", text)
        text = re.sub(r"`[^`]+`", " [code] ", text)
    return " ".join(text.split()).strip()


def is_noise(text: str) -> bool:
    """Drop system/context/instruction messages injected into the log."""
    if len(text) < 30 or len(text) > 1500:
        return True
    if text.startswith("<") and text.endswith(">"):
        return True
    if any(m in text for m in ["AGENTS.md", "<INSTRUCTIONS>", "<permissions instructions>"]):
        return True
    if "Headless" in text and "Stage" in text:
        return True
    if text.startswith("{") and text.endswith("}"):
        return True
    if text.startswith("[") and text.endswith("]"):
        return True
    return False


def extract_messages(
    sessions_dir: Path,
    role_filter: str,
    strip_code: bool,
) -> list[str]:
    """Extract cleaned messages from all Codex rollout JSONL files."""
    messages: list[str] = []
    for fp in sessions_dir.rglob("*.jsonl"):
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = obj.get("payload", {})
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role_filter != "all" and role != role_filter:
                continue
            content = payload.get("content", [])
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and "text" in block
            ]
            full = clean_text("\n".join(texts), strip_code)
            if is_noise(full):
                continue
            messages.append(full)
    return messages


def print_stats(messages: list[str]) -> None:
    chars = [len(m) for m in messages]
    total = sum(chars)
    print(f"Messages: {len(messages):,}")
    print(f"Total characters: {total:,}")
    print(f"Average length: {total / max(1, len(messages)):.1f}")
    if chars:
        print(f"Median length: {sorted(chars)[len(chars) // 2]}")
    words = re.findall(r"[a-zA-Z]+", " ".join(messages))
    counts = Counter(words)
    print(f"Unique words: {len(counts):,}")
    print("Top words:", ", ".join(f"{w}:{c}" for w, c in counts.most_common(15)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    messages = extract_messages(SESSIONS_DIR, args.role, args.strip_code)
    if not messages:
        print("No messages extracted.", file=sys.stderr)
        return 1
    if args.lines is not None:
        messages = messages[: args.lines]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(messages), encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.stats:
        print_stats(messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
