"""Shared integrity and prompt contract for the frozen DEV serialization pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SYSTEM = "Return only the requested answer token or string. Do not add an explanation, label, or surrounding quotation marks."
QUERY = "Apply the demonstrated formatting rule to: {input}"
REFINER = "Summarize the formatting rule illustrated by these examples in one short sentence. Include necessary literal text and ordering."


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(canonical(value) + b"\n")


def read_manifest(root):
    data = json.loads((Path(root) / "MANIFEST.json").read_text())
    if data["manifest_sha256"] != sha(canonical({k: v for k, v in data.items() if k != "manifest_sha256"})):
        raise ValueError("Pilot manifest hash mismatch")
    repo = Path(__file__).resolve().parents[1]
    for name, expected in data["execution_sources"].items():
        if sha((repo / name).read_bytes()) != expected:
            raise ValueError(f"Pilot execution source changed: {name}")
    return data


def read_split(root, manifest, phase, split):
    allowed = {"reference": {"training", "probes"}, "train": {"training"},
               "restore_text": {"probes"}, "restore_soft": {"probes"}}
    if split not in allowed.get(phase, set()):
        raise ValueError(f"{phase} must not read {split}")
    entry = manifest["files"][split]
    data = (Path(root) / entry["path"]).read_bytes()
    if sha(data) != entry["sha256"]:
        raise ValueError("Pilot split hash mismatch")
    obj = json.loads(data)
    expected = "examples" if split == "training" else "probes"
    for pattern in obj["patterns"]:
        if set(pattern) != {"index", expected}:
            raise ValueError("Unexpected fields in phase input")
        for row in pattern[expected]:
            if set(row) != {"input", "target"}:
                raise ValueError("Unexpected example/probe fields")
    return obj["patterns"]


def query_messages(input_text, context=None):
    messages = [{"role": "system", "content": SYSTEM}]
    if context is not None:
        messages.append({"role": "user", "content": context})
    messages.append({"role": "user", "content": QUERY.format(input=input_text)})
    return messages


def example_context(examples):
    return "Corrected examples:\n" + "\n".join(
        f"- The correct result for {row['input']} is {row['target']}." for row in examples)


def recovery(correct, no_context, with_context):
    denominator = with_context - no_context
    if denominator <= 0:
        return None
    return (correct - no_context) / denominator
