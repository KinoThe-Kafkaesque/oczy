"""Hash-bound, DEV-only v3/v4 prompt amendments approved on 2026-09-11.

This is a diagnostic instrument, not a candidate/calibration view. The v2
loader and frozen source artifacts remain unchanged. Only the two explicitly
listed public task files are read; no sealed or calibration payload is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oczy.experiments.meta_cortex.contracts import (  # noqa: E402
    DialogueMessage,
    ProbeKind,
    TaskFamily,
)
from oczy.experiments.meta_cortex.instrument import (  # noqa: E402
    _task_to_jsonl_record,
    load_dev_view,
)
from oczy.experiments.meta_cortex.instrument_contracts import (  # noqa: E402
    strict_canonical_json,
    strict_json_loads,
)

BASE_DEV_HASH = "bfb1a4003c421df3a14345bdcc6ddabad2d6ae252a5aadd6414d0cf0ba07fea6"
TASK_FILES = ("public/tasks/meta_train.jsonl", "public/tasks/meta_validation_tuning.jsonl")
BARE_ANSWER = (
    "Return only the requested answer token or string. Do not add an explanation, "
    "label, or surrounding quotation marks."
)
APPROVAL = {
    "date": "2026-09-11",
    "human_reply": "Approve both, in order",
    "proposal": "experiments/r23.5-serialization-dev/INSTRUMENT_REPAIR_PROPOSAL.md",
    "scope": "Amendment A v3, then Amendment B v4; DEV only; no meta-test access",
}
ORACLE_DESCRIPTIONS = {
    "permutation": "Reverse the input string character by character.",
    "substitution": "Replace every lowercase vowel (a, e, i, o, u) with {param1}. Leave all other characters unchanged.",
    "conditional": "If the input begins with a lowercase vowel (a, e, i, o, u), append {param1} to the input. Otherwise, prepend {param2} to the input.",
    "composition": "Reverse the input string character by character. Then replace every lowercase vowel (a, e, i, o, u) with {param2}. Leave all other characters unchanged.",
}
HEADER = re.compile(
    r"Rule: (permutation|substitution|conditional|composition) with parameters "
    r"'([a-z]*)' and '([a-z]*)'\.\n(Worked examples:\n.+)", re.DOTALL,
)


def digest(obj):
    return hashlib.sha256(strict_canonical_json(obj)).hexdigest()


def parse_oracle(text):
    match = HEADER.fullmatch(text)
    if match is None:
        raise ValueError("Unrecognized frozen transformation oracle header")
    return match.groups()


def amend_task(task, version):
    if version not in ("v3", "v4"):
        raise ValueError("Only approved DEV v3 and v4 are supported")
    categories = {}
    for kind in ProbeKind:
        probes = []
        for probe in task.probes.by_kind(kind):
            messages = probe.messages
            if any(m.role == "system" for m in messages):
                raise ValueError("Base probe already has a system message")
            if version == "v4" and task.family == TaskFamily.RULE_TRANSFORMATION and kind == ProbeKind.ORACLE_CONTEXT:
                template, param1, param2, examples = parse_oracle(messages[0].content)
                description = ORACLE_DESCRIPTIONS[template].format(param1=param1, param2=param2)
                messages = (replace(messages[0], content=f"Rule: {description}\n{examples}"),) + messages[1:]
            probes.append(replace(probe, messages=(DialogueMessage("system", BARE_ANSWER),) + messages))
        categories[kind.value] = tuple(probes)
    return replace(task, probes=replace(task.probes, **categories))


def task_records(catalog):
    return {
        split: [_task_to_jsonl_record(t, split, i) for i, t in enumerate(tasks)]
        for split, tasks in (("meta_train", catalog.meta_train),
                             ("meta_validation_tuning", catalog.meta_validation))
    }


def amended_catalog(base, version):
    catalog = replace(base,
                      meta_train=tuple(amend_task(t, version) for t in base.meta_train),
                      meta_validation=tuple(amend_task(t, version) for t in base.meta_validation))
    return replace(catalog, catalog_sha256=digest(task_records(catalog)))


def verified_base(public_root):
    """Verify the historical public manifest without opening its sealed entries."""
    public_root = Path(public_root)
    data = strict_json_loads((public_root / "DEV_VIEW.json").read_bytes())
    if data["dev_view_sha256"] != BASE_DEV_HASH or tuple(data["task_files"]) != TASK_FILES:
        raise ValueError("Not the approved v2 public DEV instrument")
    definition = strict_json_loads((public_root.parent / "DEFINITION.json").read_bytes())
    expected = definition.pop("definition_sha256")
    if digest(definition) != expected or expected != data["definition_sha256"]:
        raise ValueError("Base definition hash mismatch")
    entries = {e["path"]: e for e in definition["public_files"]}
    for name in TASK_FILES:
        path = public_root.parent / name
        if not path.resolve().is_relative_to(public_root.resolve()):
            raise ValueError("Public task file escapes the DEV root")
        raw = path.read_bytes()
        if entries[name]["visibility"] != "public" or hashlib.sha256(raw).hexdigest() != entries[name]["sha256"]:
            raise ValueError("Base public task hash mismatch")
    return load_dev_view(public_root)


def manifest_body(base, catalog, version, parent_hash):
    return {
        "schema": "oczy/dev-prompt-amendment/v1",
        "instrument_id": f"meta_cortex/{version}", "instrument_version": version,
        "scope": "DEV_DIAGNOSTIC_ONLY", "meta_test_authorized": False,
        "approval": APPROVAL,
        "base_dev_view_sha256": base.binding.dev_view_sha256,
        "base_definition_sha256": base.binding.definition_sha256,
        "parent_sha256": parent_hash,
        "tasks_sha256": digest(task_records(catalog)),
        "base_tasks_sha256": digest(task_records(base.catalog)),
        "base_prompt_registry_sha256": base.binding.prompt_registry_sha256,
        "scorer_sha256": base.binding.scorer_registry_sha256,
        "endpoint_registry_sha256": base.binding.endpoint_registry_sha256,
        "organ_hash": base.binding.organ_hash,
        "format_instruction": BARE_ANSWER,
        "oracle_descriptions": ORACLE_DESCRIPTIONS if version == "v4" else None,
        "transform_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_counts": {"meta_train": len(catalog.meta_train), "meta_validation": len(catalog.meta_validation)},
    }


def load_repair(public_root, root):
    """Reject tampering and any change beyond the approved prompt transformations."""
    base = verified_base(public_root)
    root = Path(root)
    manifest = strict_json_loads((root / "MANIFEST.json").read_bytes())
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest(body) != manifest["manifest_sha256"]:
        raise ValueError("Amendment manifest hash mismatch")
    version = manifest["instrument_version"]
    catalog = amended_catalog(base.catalog, version)
    parent_hash = base.binding.dev_view_sha256
    if version == "v4":
        parent = strict_json_loads((root / "PARENT_MANIFEST.json").read_bytes())
        parent_body = manifest_body(base, amended_catalog(base.catalog, "v3"), "v3", parent_hash)
        if parent != dict(parent_body, manifest_sha256=digest(parent_body)):
            raise ValueError("Amendment B requires the verified Amendment A parent")
        parent_hash = parent["manifest_sha256"]
    if body != manifest_body(base, catalog, version, parent_hash):
        raise ValueError("Amendment manifest exceeds approved scope or source changed")
    if strict_json_loads((root / "TASKS.json").read_bytes()) != task_records(catalog):
        raise ValueError("Amended tasks differ from approved transformation")
    return catalog, manifest


def materialize(public_root, output, version, parent=None):
    base = verified_base(public_root)
    parent_manifest = None
    parent_hash = base.binding.dev_view_sha256
    if version == "v4":
        if parent is None:
            raise ValueError("Amendment B requires Amendment A first")
        _, parent_manifest = load_repair(public_root, parent)
        if parent_manifest["instrument_version"] != "v3":
            raise ValueError("Amendment B requires a v3 parent")
        parent_hash = parent_manifest["manifest_sha256"]
    elif parent is not None:
        raise ValueError("Amendment A starts from v2")
    catalog = amended_catalog(base.catalog, version)
    body = manifest_body(base, catalog, version, parent_hash)
    manifest = dict(body, manifest_sha256=digest(body))
    output = Path(output)
    if output.resolve().is_relative_to(Path(public_root).parent.resolve()):
        raise ValueError("Do not write inside the original frozen instrument")
    output.mkdir(parents=True, exist_ok=False)
    for name, obj in (("MANIFEST.json", manifest), ("TASKS.json", task_records(catalog)),
                      ("PARENT_MANIFEST.json", parent_manifest)):
        if obj is not None:
            (output / name).write_bytes(strict_canonical_json(obj) + b"\n")
    load_repair(public_root, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", choices=("v3", "v4"), required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    result = materialize(args.public_root, args.output, args.version, args.parent)
    print(f"{result['instrument_id']}: {result['manifest_sha256']}")


if __name__ == "__main__":
    main()
