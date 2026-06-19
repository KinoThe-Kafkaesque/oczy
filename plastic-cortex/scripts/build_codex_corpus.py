"""Build a training corpus from local Codex CLI session logs.

Reads ~/.codex/sessions/**/*.jsonl, extracts user/assistant messages, filters
system noise, and writes a plain-text file suitable for LMPlasticCortex.

Usage:
    uv run python plastic-cortex/scripts/build_codex_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

SESSIONS_DIR = Path.home() / ".codex" / "sessions"
OUTPUT = Path("plastic-cortex/data/codex_corpus.txt")


def is_noise(text: str) -> bool:
    """Drop system/context/instruction messages injected into the log."""
    text = text.strip()
    if len(text) < 20 or len(text) > 2000:
        return True
    if text.startswith("<") and text.endswith(">"):
        return True
    if "AGENTS.md" in text or "<INSTRUCTIONS>" in text or "<permissions instructions>" in text:
        return True
    if "Headless" in text and "Stage" in text:
        return True
    if text.startswith("{") and text.endswith("}"):
        return True
    if text.startswith("[") and text.endswith("]"):
        return True
    return False


def extract_clean_text(text: str) -> str:
    """Normalize whitespace to a single line."""
    return " ".join(text.split())


def build_corpus(output: Path | None = None) -> Path:
    out = output or OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)

    files = list(SESSIONS_DIR.rglob("*.jsonl"))
    messages: list[str] = []

    for fp in files:
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
            if role not in {"user", "assistant"}:
                continue
            content = payload.get("content", [])
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and "text" in block
            ]
            full = "\n".join(texts).strip()
            if is_noise(full):
                continue
            messages.append(extract_clean_text(full))

    with out.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(msg + "\n")

    total_chars = sum(len(m) for m in messages)
    print(f"Wrote {out} ({len(messages)} lines, {total_chars:,} chars)")
    return out


if __name__ == "__main__":
    build_corpus()
