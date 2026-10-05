"""Shared, frozen contracts for contextual representation and preservation DEV v2."""

import json
from pathlib import Path

from scripts.capability_validation_contract import SYSTEM as V1_SYSTEM
from scripts.capability_validation_contract import manifest_at
from scripts.serialization_pilot_contract import canonical, sha, write_json

SYSTEM = (
    "Apply the requested client's latest learned string formatting rule. "
    "If no rule is known for that client, copy the input unchanged. "
    "Reply with only the output string, without quotes or explanation."
)


def messages(row, interface="concise_v2", context=None):
    if len(row["clients"]) != 1:
        raise ValueError("DEV v2 isolates single-context behavior")
    client = row["clients"][0]
    system = V1_SYSTEM if interface in ("v1", "query_v2") else SYSTEM
    if interface == "v1":
        query = f"Clients in order: {client}. Input: {row['input']}. Output?"
    elif interface in ("query_v2", "concise_v2"):
        query = f"Client: {client}\nInput: {row['input']}\nOutput:"
    else:
        raise ValueError("Unknown interface")
    result = [{"role": "system", "content": system}]
    if context is not None:
        result.append({"role": "user", "content": context})
    result.append({"role": "user", "content": query})
    return result


def read_data(root, manifest, phase, role):
    allowed = {"reference": {"probes", "oracle", "capacity_training"},
               "train_capacity": {"capacity_training"}, "restore_capacity": {"probes"},
               "train_correction": {"correction_training"}, "restore_correction": {"probes"}}
    if role not in allowed.get(phase, set()):
        raise ValueError(f"{phase} cannot read {role}")
    entry = manifest["files"][role]
    raw = (Path(root) / entry["path"]).read_bytes()
    if sha(raw) != entry["sha256"]:
        raise ValueError("Frozen phase input changed")
    return json.loads(raw)


def capacity_gate(rows, seeds):
    """Admission requires every fresh probe, every category and every seed."""
    expected = {(s, c, w) for s in seeds for c in ("amber", "cobalt", "silver", "quartz")
                for w in ("lime", "melon", "grape", "peach")}
    learned = [r for r in rows if r["condition"] == "joint_restored"]
    keys = [(r["seed"], r["category"], r["input"]) for r in learned]
    return len(keys) == len(expected) and set(keys) == expected and all(r["correct"] for r in learned)


__all__ = ["SYSTEM", "canonical", "sha", "write_json", "manifest_at", "messages", "read_data", "capacity_gate"]
