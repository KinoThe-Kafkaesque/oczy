"""Data access and measurement contracts for scoped-diversity DEV v3."""

import json
from collections import Counter

from scripts.context_preservation_contract import (
    SYSTEM,
    canonical,
    manifest_at,
    messages,
    sha,
    write_json,
)


def data(root, manifest, phase, role):
    allowed = {"train": {"capacity_training"}, "evaluate": {"capacity_training", "probes", "oracle"}}
    if role not in allowed.get(phase, set()):
        raise ValueError(f"{phase} cannot read {role}")
    entry = manifest["files"][role]
    raw = (root / entry["path"]).read_bytes()
    if sha(raw) != entry["sha256"]:
        raise ValueError("Frozen input changed")
    return json.loads(raw)


def key(row):
    return tuple(row["clients"]), row["input"]


def admission(rows, fits, manifest, probes):
    seeds = manifest["training"]["seeds"]
    expected = Counter((seed, *key(row)) for seed in seeds for row in probes)
    selected = [r for r in rows if r["condition"] == "diversity" and r["split"] == "confirmation"]
    coverage = Counter((r["seed"], *key(r)) for r in selected)
    fit_ok = (Counter(f["seed"] for f in fits) == Counter(seeds) and
              all(len(f["training_fit"]) == 27 and all(r["correct"] for r in f["training_fit"]) for f in fits))
    return bool(fit_ok and coverage == expected and all(r["correct"] for r in selected))


def example_text(rows):
    return "Corrected examples:\n" + "\n".join(
        f"Client {r['clients'][0]}, input {r['input']} -> correct output {r['target']}" for r in rows)


def reference_prompt(row, condition, teaching, table):
    if condition == "no_context":
        return messages(row)
    if condition == "direct_oracle":
        # Rule selection is explicitly supplied by this oracle baseline.
        suffix = {"amber": "vek", "cobalt": "mip"}.get(row["clients"][0], "")
        return [{"role": "system", "content": "Return only the resulting string, with no explanation or quotation marks."},
                {"role": "user", "content": f'Append the suffix "{suffix}" to the string "{row["input"]}".' if suffix
                 else f'Copy the string "{row["input"]}" exactly.'}]
    if condition == "complete_table":
        return messages(row, context=table)
    arm = condition.rsplit("_", 1)[-1]
    examples = teaching["groups"][0] if arm == "control" else teaching["rows"]
    if condition.startswith("chat_"):
        prompt = [{"role": "system", "content": SYSTEM}]
        for example in examples:
            prompt += [messages(example)[-1], {"role": "assistant", "content": example["target"]}]
        return [*prompt, messages(row)[-1]]
    if condition.startswith(("text_", "zlib_")):
        return messages(row, context=example_text(examples))
    raise ValueError("Unknown reference")


__all__ = ["SYSTEM", "canonical", "manifest_at", "messages", "sha", "write_json", "data", "key", "admission", "reference_prompt", "example_text"]
