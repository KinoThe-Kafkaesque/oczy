"""Frozen, phase-scoped contracts for the six-capability DEV battery."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.serialization_pilot_contract import canonical, sha, write_json

SYSTEM = (
    "Return only the answer string, with no explanation, label, or quotation marks. "
    "Each client has a separate string formatting preference learned from corrected examples. "
    "A client with no taught preference copies the input unchanged. "
    "A new correction replaces only that client's previous preference. "
    "For multiple clients, apply their preferences in the stated order to the whole string."
)


def messages(row, context=None):
    result = [{"role": "system", "content": SYSTEM}]
    if context is not None:
        result.append({"role": "user", "content": context})
    result.append({"role": "user", "content":
                   f"Clients in order: {' -> '.join(row['clients'])}. Input: {row['input']}. Output?"})
    return result


def example_text(lessons):
    sections = []
    for lesson in lessons:
        rows = "\n".join(f"Input {r['input']} -> correct output {r['target']}" for r in lesson["examples"])
        sections.append(f"Corrected examples for client {lesson['client']}:\n{rows}")
    return "\n\n".join(sections)


def latest_lessons(lessons):
    latest = {}
    for lesson in lessons:
        latest[lesson["client"]] = lesson
    return list(latest.values())


def manifest_at(root):
    manifest = json.loads((Path(root) / "MANIFEST.json").read_text())
    if manifest["manifest_sha256"] != sha(canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"})):
        raise ValueError("Capability manifest changed")
    repo = Path(__file__).resolve().parents[1]
    for name, digest in manifest["execution_sources"].items():
        if sha((repo / name).read_bytes()) != digest:
            raise ValueError(f"Frozen execution source changed: {name}")
    return manifest


def read_data(root, manifest, phase, role):
    allowed = {"reference": {"training", "probes", "oracle"}, "train": {"training"},
               "restore": {"probes"}, "actions": {"actions"}}
    if role not in allowed.get(phase, set()):
        raise ValueError(f"{phase} cannot read {role}")
    entry = manifest["files"][role]
    raw = (Path(root) / entry["path"]).read_bytes()
    if sha(raw) != entry["sha256"]:
        raise ValueError(f"Frozen {role} changed")
    return json.loads(raw)


__all__ = ["SYSTEM", "canonical", "sha", "write_json", "messages", "example_text", "latest_lessons", "manifest_at", "read_data"]
