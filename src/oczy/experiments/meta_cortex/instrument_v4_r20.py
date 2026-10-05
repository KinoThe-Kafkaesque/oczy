"""Versioned ``meta_cortex/v4-r20`` DEV instrument: the prompt-amended successor.

This module is the **G1-equivalent freeze** of a successor to the frozen
``meta_cortex/v3`` DEV instrument.  It exists because v3's gate G2 oracle
capability screen failed **0/15** (articulation block), and the recorded
diagnosis was that v3 inherited v2's prompt registry byte-for-byte and
therefore carries *neither* of the two prompt amendments the user approved on
2026-09-11 for the v2 instrument:

- **Amendment A** — an identical bare-answer system instruction on every public
  DEV probe (``2026-09-11_campaign_r20_dev_output_path.md``).
- **Amendment B** — complete oracle rule descriptions for the transformation
  oracle header, replacing the shorthand ``Rule: <template> with parameters
  'p1' and 'p2'`` form.

``meta_cortex/v4-r20`` differs from ``meta_cortex/v3`` in exactly three ways,
each of them explicitly authorized by the user via the manager on 2026-10-05
(kanban card ``t_37e96ee1``):

1. **Prompt amendments carried in.**  Every public DEV probe receives the
   Amendment A system message; every ``rule_transformation`` ``oracle_context``
   header is rewritten to the Amendment B full description.  The task-support
   repaired *task content* is otherwise byte-identical to v3.
2. **``max_new_tokens`` 32 -> 128.**  The signed v3 field truncated 10/15
   oracle generations mid-preamble.  128 is 4x the longest observed preamble
   and remains bounded.  This **changes a signed field relative to v3**; per
   the user authorization it is recorded explicitly and **v3-vs-v4-r20 scores
   are NOT comparable as a causal improvement** (same rule as the v2->v3
   transition).
3. **Prompt registry.**  The registry is the v3 registry plus two documented
   amendment entries; it is therefore *not* byte-identical to v3's.  The
   scorer and endpoint registries remain byte-identical to the frozen v2/v3
   values, which this module asserts rather than assumes.

Naming.  The user asked for "v4".  Plain ``meta_cortex/v4`` is already taken by
``experiments/r23.5-serialization-dev/instruments/v4`` (user-approved amendment
B, manifest ``d58adb749f4112f2...``), and ``meta_cortex/v3`` is doubly claimed
(the r23.5 v3 amendment ``5c944abc...`` versus the G1-frozen R20 lineage
``ab99c173...``).  The successor id is therefore the scope-qualified
``meta_cortex/v4-r20``, recorded in ``DEFINITION.json`` under ``naming``.

What this module deliberately does NOT do:

- It does not edit, read the calibration view of, or overwrite the frozen v3
  instrument, ``taskgen.py`` or ``taskgen_v2.py``.  It *reads* the v3 public DEV
  view, verifies it fail-closed against pinned hashes, and rebuilds the same
  v2-dev lineage in memory to prove the base is the approved one.
- It does not generate or open any sealed meta-test task, seed or generator.
  There is no ``sealed/`` directory, no meta-test seed commitment and no
  meta-test entry point; the verifier rejects both a ``sealed/`` directory and
  any sealed file entry.
- It does not choose, recompute or import any threshold, margin or power
  number.  Those are gate G4 and require explicit human sign-off at G5.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .contracts import (
    DevSplit,
    DevTaskCatalog,
    DialogueMessage,
    MetaTask,
    ProbeKind,
    TaskFamily,
    TaskGeneratorConfig,
)
from .instrument import (
    _compute_probe_counts,
    _endpoint_registry_obj,
    _prompt_registry_obj,
    _run_leakage_audit,
    _scorer_registry_obj,
    _task_to_jsonl_record,
    _write_canonical_json,
    _write_canonical_jsonl,
)
from .instrument_contracts import (
    CALIBRATION_VIEW_SCHEMA,
    DEV_VIEW_SCHEMA,
    ENDPOINT_SCHEMA,
    INSTRUMENT_DEFINITION_SCHEMA,
    PROMPT_SCHEMA,
    SCORER_SCHEMA,
    strict_canonical_json,
    strict_json_loads,
)
from .taskgen import MAX_COLLISION_NONCE, assert_split_firewall
from .taskgen_v2 import TASKGEN_SCHEMA_V2, build_dev_catalog_v2

__all__ = [
    "INSTRUMENT_ID",
    "INSTRUMENT_VERSION",
    "GENERATOR_ALGORITHM",
    "MAX_NEW_TOKENS",
    "BASE_INSTRUMENT_ID",
    "BASE_DEV_VIEW_SHA256",
    "BASE_DEFINITION_SHA256",
    "V3_FROZEN_HASHES",
    "INHERITED_FROZEN_HASHES",
    "BARE_ANSWER",
    "ORACLE_DESCRIPTIONS",
    "AMENDMENT_RECORD",
    "NAMING_RECORD",
    "MAX_NEW_TOKENS_RECORD",
    "amended_prompt_registry_obj",
    "amend_task",
    "V4R20BaseError",
    "V4R20DefinitionError",
    "V4R20Definition",
    "materialize_v4_r20_definition",
    "verify_v4_r20_definition",
]


INSTRUMENT_ID = "meta_cortex/v4-r20"
INSTRUMENT_VERSION = "v4-r20"
GENERATOR_ALGORITHM = "sha256-counter-rejection/v1"

#: ``max_new_tokens`` for the successor.  Raised from the signed v3 value of 32
#: under explicit user authorization (2026-10-05).  See ``MAX_NEW_TOKENS_RECORD``.
MAX_NEW_TOKENS = 128

#: The instrument whose public DEV view this successor is built from.
BASE_INSTRUMENT_ID = "meta_cortex/v3"
BASE_DEV_VIEW_SHA256 = "593090c405758617ae9f750cf9c1075e999c92f8c7b677affcb04c3855b3ca26"
BASE_DEFINITION_SHA256 = "ab99c17358a3af3efb577563b8cd33639519084fa88648c3bc1e05dce86d4bee"
BASE_CATALOG_SHA256 = "33210eb474a25925815c1112e2d6c6ea889b349c8d137fbcbe0aacac5f035219"
BASE_TASK_FILE_SHA256 = {
    "public/tasks/meta_train.jsonl": "1f1a4118adabf58fc4c863245ed8ddcfc008d9f3e3e4ab050e07ff370b19f636",
    "public/tasks/meta_validation_tuning.jsonl": "f4037013117167f182dc6ad77e26af6f8dece862d1ee79927c2ac960f9b2bef5",
    "public/tasks/meta_validation_calibration.jsonl": "13e17d9309b9f0789416f82c1694fa17239661f33515566665998c3443ba1b08",
}

#: Registries that v4-r20 must inherit byte-for-byte from the frozen v2/v3
#: instrument.  The prompt registry is intentionally NOT in this set: carrying
#: the two approved amendments is the whole point of the successor, and it is
#: the one registry change the 2026-10-05 authorization covers.
V3_FROZEN_HASHES = {
    "prompt_registry_sha256": "db624922bf4f67e3e1011b5530ade4111b479ef05391ca0a61e375cac2339735",
    "scorer_registry_sha256": "e5d746d0477c489157d1699e2ae73dfcc8ac92998719de1a06d92fcff4b1c742",
    "endpoint_registry_sha256": "669d6130075e429264a6c9eec470bbf82663559433781adcffa6ed942c9ca796",
}
INHERITED_FROZEN_HASHES = {
    "scorer_registry_sha256": V3_FROZEN_HASHES["scorer_registry_sha256"],
    "endpoint_registry_sha256": V3_FROZEN_HASHES["endpoint_registry_sha256"],
}

_FAMILY_ORDER = (
    TaskFamily.CONTEXTUAL_REMAP,
    TaskFamily.RULE_TRANSFORMATION,
    TaskFamily.FINITE_STATE,
)

#: Amendment A — the identical bare-answer system instruction prepended to every
#: public DEV probe (verbatim from the 2026-09-11 approved amendment).
BARE_ANSWER = (
    "Return only the requested answer token or string. Do not add an explanation, "
    "label, or surrounding quotation marks."
)

#: Amendment B — the approved complete English rule descriptions for the
#: transformation oracle header.  ``{param1}``/``{param2}`` are substituted with
#: the task's own parameters; the worked examples are carried through unchanged.
ORACLE_DESCRIPTIONS = {
    "permutation": "Reverse the input string character by character.",
    "substitution": (
        "Replace every lowercase vowel (a, e, i, o, u) with {param1}. "
        "Leave all other characters unchanged."
    ),
    "conditional": (
        "If the input begins with a lowercase vowel (a, e, i, o, u), append "
        "{param1} to the input. Otherwise, prepend {param2} to the input."
    ),
    "composition": (
        "Reverse the input string character by character. Then replace every "
        "lowercase vowel (a, e, i, o, u) with {param2}. Leave all other "
        "characters unchanged."
    ),
}

_ORACLE_HEADER = re.compile(
    r"Rule: (permutation|substitution|conditional|composition) with parameters "
    r"'([a-z]*)' and '([a-z]*)'\.\n(Worked examples:\n.+)",
    re.DOTALL,
)

AMENDMENT_RECORD = {
    "authorization": {
        "date": "2026-10-05",
        "relayed_by": "manager",
        "kanban_card": "t_37e96ee1",
        "verbatim": (
            "USER DECISION relayed by the manager (2026-10-05, authoritative): "
            "carry the 2026-09-11 bare-answer and full-oracle-description "
            "amendments into the successor's prompt registry"
        ),
    },
    "amendment_a": {
        "name": "bare_answer_system_instruction",
        "system_message": BARE_ANSWER,
        "applies_to": "every public DEV probe (all six probe kinds, all families)",
        "precedence": "the system message remains first when a teaching transcript is supplied",
        "source": "experiments_logs/2026-09-11_campaign_r20_dev_output_path.md",
    },
    "amendment_b": {
        "name": "complete_oracle_rule_descriptions",
        "applies_to": "rule_transformation oracle_context headers only",
        "descriptions": ORACLE_DESCRIPTIONS,
        "unchanged": "worked examples, operands, expected answers, splits, seeds and scoring",
        "source": "experiments_logs/2026-09-11_campaign_r20_dev_output_path.md",
    },
    "task_content_unchanged": (
        "every other field of every task and probe is byte-identical to the "
        "frozen meta_cortex/v3 public DEV view"
    ),
}

NAMING_RECORD = {
    "chosen_id": INSTRUMENT_ID,
    "user_intent": "the user said 'name it v4'",
    "reason": (
        "plain 'meta_cortex/v4' collides with the already-approved "
        "r23.5-serialization-dev instruments/v4 (manifest d58adb749f4112f2...), and "
        "'meta_cortex/v3' is doubly claimed (r23.5 v3 amendment 5c944abc... versus "
        "the G1-frozen R20 lineage ab99c173...); the scope-qualified id honors the "
        "'v4' intent unambiguously"
    ),
    "collisions": [
        {
            "id": "meta_cortex/v4",
            "owner": "experiments/r23.5-serialization-dev/instruments/v4",
            "manifest_sha256": "d58adb749f4112f2920f08e0dbe0ce5c6a1ee7711f562faee4e41072eccc8764",
        },
        {
            "id": "meta_cortex/v3 (r23.5 amendment A)",
            "manifest_sha256": "5c944abc72fac57b29890da46d9a3b3807c551c5c2b0370c24d771c5616162b5",
        },
    ],
    "scope_dir": "experiments/r20-taskgen-v4-r20-dev/",
}

MAX_NEW_TOKENS_RECORD = {
    "field": "max_new_tokens",
    "base_instrument_value": 32,
    "successor_value": MAX_NEW_TOKENS,
    "authorized": True,
    "authorization": {
        "date": "2026-10-05",
        "relayed_by": "manager",
        "kanban_card": "t_37e96ee1",
    },
    "justification": (
        "the G2 v3 artifact shows 10 of 15 oracle generations truncated "
        "mid-preamble at 32 new tokens; 128 is 4x the longest observed preamble "
        "and remains bounded"
    ),
    "changes_a_signed_field_relative_to": BASE_INSTRUMENT_ID,
    "comparability": (
        "v3 and v4-r20 scores are NOT comparable as a causal improvement; this is "
        "the same rule applied to the v2->v3 transition"
    ),
}


class V4R20DefinitionError(ValueError):
    """Raised when the v4-r20 instrument cannot be frozen or verified."""


# ---------------------------------------------------------------------------
# Amendment application
# ---------------------------------------------------------------------------


class V4R20BaseError(V4R20DefinitionError):
    """Raised when the *base* instrument is not the approved, pinned v3 view.

    Distinct from other definition errors so the materializer can refuse a wrong
    base with its own exit code (3) rather than a generic failure (1).
    """


def parse_oracle_header(text: str) -> tuple[str, str, str, str]:
    """Parse the frozen shorthand transformation oracle header (fail-closed)."""
    match = _ORACLE_HEADER.fullmatch(text)
    if match is None:
        raise V4R20DefinitionError(
            "Unrecognized frozen transformation oracle header in the v3 base task"
        )
    template, param1, param2, examples = match.groups()
    return template, param1, param2, examples


def amend_task(task: MetaTask, version: str = "v4") -> MetaTask:
    """Apply the approved prompt amendments to one v3 task.

    ``version`` is accepted for parity with the 2026-09-11 amendment tooling;
    only the fully amended form (Amendment A **and** B) is materialized here,
    because that is what the successor instrument is.
    """
    if version != "v4":
        raise V4R20DefinitionError(
            "meta_cortex/v4-r20 materializes only the fully amended (A + B) form"
        )
    categories: dict[str, tuple[Any, ...]] = {}
    for kind in ProbeKind:
        probes = []
        for probe in task.probes.by_kind(kind):
            messages = probe.messages
            if any(m.role == "system" for m in messages):
                raise V4R20DefinitionError("Base v3 probe already has a system message")
            if task.family == TaskFamily.RULE_TRANSFORMATION and kind == ProbeKind.ORACLE_CONTEXT:
                template, param1, param2, examples = parse_oracle_header(messages[0].content)
                description = ORACLE_DESCRIPTIONS[template].format(param1=param1, param2=param2)
                messages = (replace(messages[0], content=f"Rule: {description}\n{examples}"),) + messages[1:]
            probes.append(
                replace(probe, messages=(DialogueMessage("system", BARE_ANSWER),) + messages)
            )
        categories[kind.value] = tuple(probes)
    return replace(task, probes=replace(task.probes, **categories))


def amended_catalog(base: DevTaskCatalog) -> DevTaskCatalog:
    """Apply the approved amendments to every task of the v3 DEV catalog."""
    return replace(
        base,
        meta_train=tuple(amend_task(t) for t in base.meta_train),
        meta_validation=tuple(amend_task(t) for t in base.meta_validation),
    )


# ---------------------------------------------------------------------------
# Amended prompt registry
# ---------------------------------------------------------------------------


def amended_prompt_registry_obj() -> dict[str, Any]:
    """Return the v4-r20 prompt registry: the v3 registry plus the amendments.

    The v3 template keys are carried through unchanged so the superseded
    shorthand form stays documented; two new keys record the approved
    amendment forms.  Nothing else about the registry changes.
    """
    base = _prompt_registry_obj()
    templates = dict(base["templates"])
    templates["context.probe.format.v1"] = BARE_ANSWER
    templates["transform.oracle.header.v2"] = "Rule: {description}\nWorked examples:"
    return {
        "schema": PROMPT_SCHEMA,
        "templates": templates,
        "amendments": {
            "context.probe.format.v1": (
                "Amendment A: identical bare-answer system instruction prepended to "
                "every public DEV probe (all six probe kinds, all three families)"
            ),
            "transform.oracle.header.v2": (
                "Amendment B: complete rule description header for rule_transformation "
                "oracle_context probes; supersedes transform.oracle.header.v1, which "
                "remains listed as the superseded shorthand form"
            ),
            "unchanged": (
                "all other template keys are byte-identical to the frozen v2/v3 "
                "prompt registry"
            ),
        },
    }


def _amended_prompt_registry_sha256() -> str:
    return hashlib.sha256(strict_canonical_json(amended_prompt_registry_obj())).hexdigest()


# ---------------------------------------------------------------------------
# Seed table
# ---------------------------------------------------------------------------


def _dev_seed_table() -> dict[str, Any]:
    """Derive the v4-r20 DEV seed table, versioned by ``meta_cortex/v4-r20``."""
    from .calibration import derive_seed_table

    return derive_seed_table(
        instrument_id=INSTRUMENT_ID,
        instrument_version=INSTRUMENT_VERSION,
    )


# ---------------------------------------------------------------------------
# Base verification (fail-closed)
# ---------------------------------------------------------------------------


def verify_base_public_view(public_root: Path) -> DevTaskCatalog:
    """Verify that *public_root* is the approved v3 public DEV view.

    Fails closed on: a different instrument id/version, a dev-view or definition
    hash other than the pinned v3 values, a mismatched public task file, or a
    definition self-hash that does not recompute.  This is what makes the
    successor's lineage unambiguous: the base is pinned by bytes, not by name.
    """
    public_root = Path(public_root)
    dev_view_path = public_root / "DEV_VIEW.json"
    if not dev_view_path.is_file():
        raise V4R20BaseError(f"DEV_VIEW.json not found in {public_root}")
    dev_view = strict_json_loads(dev_view_path.read_bytes().decode("utf-8").rstrip("\n"))

    if dev_view.get("instrument_id") != BASE_INSTRUMENT_ID:
        raise V4R20BaseError(
            f"Base must be the approved {BASE_INSTRUMENT_ID!r} public DEV instrument, "
            f"got instrument_id {dev_view.get('instrument_id')!r}"
        )
    if dev_view.get("instrument_version") != "v3":
        raise V4R20BaseError("Base instrument_version must be 'v3'")
    if dev_view.get("dev_view_sha256") != BASE_DEV_VIEW_SHA256:
        raise V4R20BaseError(
            "Base dev_view_sha256 is not the pinned v3 value "
            f"{BASE_DEV_VIEW_SHA256}; refusing to build the successor"
        )
    if dev_view.get("definition_sha256") != BASE_DEFINITION_SHA256:
        raise V4R20BaseError(
            "Base definition_sha256 is not the pinned v3 value "
            f"{BASE_DEFINITION_SHA256}; refusing to build the successor"
        )
    if dev_view.get("taskgen_schema") != TASKGEN_SCHEMA_V2:
        raise V4R20BaseError(
            f"Base taskgen_schema must be {TASKGEN_SCHEMA_V2!r}, "
            f"got {dev_view.get('taskgen_schema')!r}"
        )
    if dev_view.get("prompt_registry_sha256") != V3_FROZEN_HASHES["prompt_registry_sha256"]:
        raise V4R20BaseError("Base prompt registry is not the frozen v3 registry")
    if dev_view.get("max_new_tokens") != 32:
        raise V4R20BaseError(
            f"Base max_new_tokens must be the signed v3 value 32, got {dev_view.get('max_new_tokens')!r}"
        )
    if dev_view.get("catalog_sha256") != BASE_CATALOG_SHA256:
        raise V4R20BaseError(
            "Base DEV view catalog_sha256 is not the pinned v3 full-catalog digest"
        )

    # The base DEFINITION.json must self-verify and list the pinned task bytes.
    definition_path = public_root.parent / "DEFINITION.json"
    definition = strict_json_loads(definition_path.read_bytes().decode("utf-8").rstrip("\n"))
    stored = definition.get("definition_sha256")
    computed = hashlib.sha256(
        strict_canonical_json({k: v for k, v in definition.items() if k != "definition_sha256"})
    ).hexdigest()
    if computed != stored or stored != BASE_DEFINITION_SHA256:
        raise V4R20BaseError("Base DEFINITION.json does not self-verify to the pinned v3 hash")
    if definition.get("catalog_sha256") != BASE_CATALOG_SHA256:
        raise V4R20BaseError("Base catalog_sha256 is not the pinned v3 value")
    entries = {e["path"]: e for e in definition.get("public_files", [])}
    for rel, expected in BASE_TASK_FILE_SHA256.items():
        entry = entries.get(rel)
        if entry is None:
            raise V4R20BaseError(f"Base definition does not list {rel}")
        raw = (public_root.parent / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected or entry["sha256"] != expected:
            raise V4R20BaseError(f"Base public task file {rel} is not the pinned v3 bytes")
    for rel in ("public/prompts.json", "public/scorers.json", "public/endpoints.json"):
        entry = entries.get(rel)
        if entry is None:
            raise V4R20BaseError(f"Base definition does not list {rel}")
        raw = (public_root.parent / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise V4R20BaseError(f"Base registry file {rel} does not match its recorded hash")
    if (public_root.parent / "sealed").exists():
        raise V4R20BaseError("Base instrument must not contain a sealed/ directory")

    from .instrument import load_dev_view

    return load_dev_view(public_root).catalog


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _split_validation(
    catalog: DevTaskCatalog, *, tuning_tasks_per_family: int, calibration_tasks_per_family: int
) -> tuple[tuple[MetaTask, ...], tuple[MetaTask, ...]]:
    """Split validation tasks into tuning then calibration, by family/index order."""
    validation = list(catalog.meta_validation)
    tuning: list[MetaTask] = []
    calibration: list[MetaTask] = []
    idx = 0
    for _family in _FAMILY_ORDER:
        for _ in range(tuning_tasks_per_family):
            tuning.append(validation[idx])
            idx += 1
        for _ in range(calibration_tasks_per_family):
            calibration.append(validation[idx])
            idx += 1
    if idx != len(validation):
        raise V4R20DefinitionError(
            f"validation split sizes {tuning_tasks_per_family}+"
            f"{calibration_tasks_per_family} per family do not consume "
            f"{len(validation)} validation tasks"
        )
    return tuple(tuning), tuple(calibration)


def _support_bundle_digest(bundle: Any) -> str:
    payload = {
        "lineage": bundle.lineage,
        "catalog_sha256": bundle.catalog_sha256,
        "tasks": [entry.as_dict() for entry in bundle.tasks],
    }
    return hashlib.sha256(strict_canonical_json(payload)).hexdigest()


class V4R20Definition:
    """A frozen v4-r20 DEV instrument definition and its self-verifying hashes."""

    __slots__ = (
        "definition_sha256",
        "dev_view_sha256",
        "calibration_view_sha256",
        "dev_seed_table_sha256",
        "probe_counts_sha256",
        "catalog_sha256",
        "support_bundle_sha256",
        "prompt_registry_sha256",
        "leakage_audit",
        "file_entries",
        "task_counts",
        "probe_counts",
        "max_new_tokens",
        "root",
    )

    def __init__(
        self,
        *,
        definition_sha256: str,
        dev_view_sha256: str,
        calibration_view_sha256: str,
        dev_seed_table_sha256: str,
        probe_counts_sha256: str,
        catalog_sha256: str,
        support_bundle_sha256: str,
        prompt_registry_sha256: str,
        leakage_audit: dict[str, Any],
        file_entries: tuple[dict[str, Any], ...],
        task_counts: dict[str, Any],
        probe_counts: dict[str, Any],
        max_new_tokens: int,
        root: Path,
    ) -> None:
        self.definition_sha256 = definition_sha256
        self.dev_view_sha256 = dev_view_sha256
        self.calibration_view_sha256 = calibration_view_sha256
        self.dev_seed_table_sha256 = dev_seed_table_sha256
        self.probe_counts_sha256 = probe_counts_sha256
        self.catalog_sha256 = catalog_sha256
        self.support_bundle_sha256 = support_bundle_sha256
        self.prompt_registry_sha256 = prompt_registry_sha256
        self.leakage_audit = leakage_audit
        self.file_entries = file_entries
        self.task_counts = task_counts
        self.probe_counts = probe_counts
        self.max_new_tokens = max_new_tokens
        self.root = root

    def as_json_obj(self) -> dict[str, Any]:
        return {
            "instrument_id": INSTRUMENT_ID,
            "instrument_version": INSTRUMENT_VERSION,
            "definition_sha256": self.definition_sha256,
            "dev_view_sha256": self.dev_view_sha256,
            "calibration_view_sha256": self.calibration_view_sha256,
            "dev_seed_table_sha256": self.dev_seed_table_sha256,
            "probe_counts_sha256": self.probe_counts_sha256,
            "catalog_sha256": self.catalog_sha256,
            "support_bundle_sha256": self.support_bundle_sha256,
            "prompt_registry_sha256": self.prompt_registry_sha256,
            "max_new_tokens": self.max_new_tokens,
            "task_counts": self.task_counts,
        }


def materialize_v4_r20_definition(
    *,
    base_public_root: Path,
    out: Path,
    organ_model_id: str,
    organ_revision: str,
    organ_parameter_sha256: str,
    chat_template_sha256: str,
    source_commit: str,
    source_archive_sha256: str,
    root_seed: int,
    train_tasks_per_family: int,
    tuning_tasks_per_family: int,
    calibration_tasks_per_family: int,
    event_min: int,
    event_max: int,
    feature_dim: int,
    d_cortex: int,
    soft_bank_width: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    abstain_threshold: str = "0",
) -> V4R20Definition:
    """Freeze the ``meta_cortex/v4-r20`` DEV instrument at *out*.

    The base is the approved v3 public DEV view, verified fail-closed by
    :func:`verify_base_public_view`; the amended tasks must additionally equal a
    fresh in-memory rebuild of the same v2-dev lineage with the approved
    amendments applied.  Refuses to overwrite an existing directory.
    """
    out = Path(out)
    if out.exists():
        raise V4R20DefinitionError(f"Output directory already exists: {out}")
    if max_new_tokens != MAX_NEW_TOKENS:
        raise V4R20DefinitionError(
            f"v4-r20 carries max_new_tokens={MAX_NEW_TOKENS} (user-authorized 2026-10-05); "
            f"got {max_new_tokens}"
        )

    base_catalog = verify_base_public_view(base_public_root)

    # Rebuild the v2-dev lineage in memory at the same config and require it to
    # reproduce the pinned v3 task bytes exactly.  The base is a *record* of the
    # generator, not a snapshot that could have drifted.
    tg_config = TaskGeneratorConfig(
        root_seed=root_seed,
        train_tasks_per_family=train_tasks_per_family,
        validation_tasks_per_family=tuning_tasks_per_family + calibration_tasks_per_family,
        min_events=event_min,
        max_events=event_max,
    )
    rebuilt, bundle = build_dev_catalog_v2(tg_config)
    if rebuilt.catalog_sha256 != BASE_CATALOG_SHA256:
        raise V4R20DefinitionError(
            f"rebuilt v2-dev catalog digest {rebuilt.catalog_sha256} does not match the "
            f"pinned v3 catalog {BASE_CATALOG_SHA256}"
        )
    if bundle.lineage != TASKGEN_SCHEMA_V2:
        raise V4R20DefinitionError(f"unexpected support lineage {bundle.lineage!r}")
    if bundle.catalog_sha256 != rebuilt.catalog_sha256:
        raise V4R20DefinitionError("support bundle does not bind the rebuilt catalog")
    # ``load_dev_view`` recomputes a DEV-view-only digest over the materialized
    # train+tuning records, so it is *not* the full-catalog digest.  Prove the
    # base view is the approved instrument by comparing its records instead.
    if len(base_catalog.meta_train) != train_tasks_per_family * len(_FAMILY_ORDER):
        raise V4R20DefinitionError("base DEV view train task count is not the v3 count")
    if len(base_catalog.meta_validation) != tuning_tasks_per_family * len(_FAMILY_ORDER):
        raise V4R20DefinitionError("base DEV view tuning task count is not the v3 count")
    base_records = [
        _task_to_jsonl_record(t, "meta_train", i) for i, t in enumerate(base_catalog.meta_train)
    ] + [
        _task_to_jsonl_record(t, "meta_validation_tuning", i)
        for i, t in enumerate(base_catalog.meta_validation)
    ]
    rebuilt_records = [
        _task_to_jsonl_record(t, "meta_train", i) for i, t in enumerate(rebuilt.meta_train)
    ] + [
        _task_to_jsonl_record(t, "meta_validation_tuning", i)
        for i, t in enumerate(
            _split_validation(
                rebuilt,
                tuning_tasks_per_family=tuning_tasks_per_family,
                calibration_tasks_per_family=calibration_tasks_per_family,
            )[0]
        )
    ]
    if strict_canonical_json(base_records) != strict_canonical_json(rebuilt_records):
        raise V4R20DefinitionError(
            "base DEV view records differ from a fresh rebuild of the pinned v2-dev lineage"
        )
    assert_split_firewall(rebuilt.split_audit)

    base_train = tuple(rebuilt.meta_train)
    base_tuning, base_calibration = _split_validation(
        rebuilt,
        tuning_tasks_per_family=tuning_tasks_per_family,
        calibration_tasks_per_family=calibration_tasks_per_family,
    )
    # Prove the in-memory rebuild is byte-identical to the pinned v3 task files
    # *before* amending anything.
    for rel, (role, tasks) in {
        "public/tasks/meta_train.jsonl": ("meta_train", base_train),
        "public/tasks/meta_validation_tuning.jsonl": ("meta_validation_tuning", base_tuning),
        "public/tasks/meta_validation_calibration.jsonl": (
            "meta_validation_calibration",
            base_calibration,
        ),
    }.items():
        rebuilt_bytes = b"".join(
            strict_canonical_json(_task_to_jsonl_record(t, role, i)) + b"\n"
            for i, t in enumerate(tasks)
        )
        if hashlib.sha256(rebuilt_bytes).hexdigest() != BASE_TASK_FILE_SHA256[rel]:
            raise V4R20DefinitionError(
                f"in-memory rebuild of {rel} does not reproduce the pinned v3 bytes"
            )

    train_tasks = tuple(amend_task(t) for t in base_train)
    tuning_tasks = tuple(amend_task(t) for t in base_tuning)
    calibration_tasks = tuple(amend_task(t) for t in base_calibration)

    # -- Registries ---------------------------------------------------------
    prompt_registry = amended_prompt_registry_obj()
    scorer_registry = _scorer_registry_obj()
    endpoint_registry = _endpoint_registry_obj()
    prompt_registry_sha256 = hashlib.sha256(strict_canonical_json(prompt_registry)).hexdigest()
    scorer_registry_sha256 = hashlib.sha256(strict_canonical_json(scorer_registry)).hexdigest()
    endpoint_registry_sha256 = hashlib.sha256(strict_canonical_json(endpoint_registry)).hexdigest()
    actual = {
        "scorer_registry_sha256": scorer_registry_sha256,
        "endpoint_registry_sha256": endpoint_registry_sha256,
    }
    drifted = sorted(k for k, v in actual.items() if v != INHERITED_FROZEN_HASHES[k])
    if drifted:
        raise V4R20DefinitionError(
            f"v4-r20 must inherit the frozen scorer/endpoint registries byte-for-byte; "
            f"drifted: {drifted}. The 2026-10-05 authorization covers the prompt "
            f"registry and max_new_tokens only."
        )
    if prompt_registry_sha256 == V3_FROZEN_HASHES["prompt_registry_sha256"]:
        raise V4R20DefinitionError(
            "the amended prompt registry is byte-identical to v3's; the approved "
            "amendments are not actually carried, so the successor would be pointless"
        )

    # -- Leakage audit ------------------------------------------------------
    leakage_audit = _run_leakage_audit(
        train_tasks, tuning_tasks, calibration_tasks, None, meta_test_seed_in_public=False
    )
    if not leakage_audit["passed"]:
        raise V4R20DefinitionError(
            f"DEV leakage/support audit failed: {strict_canonical_json(leakage_audit).decode()}"
        )

    seed_table = _dev_seed_table()
    dev_seed_table_sha256 = hashlib.sha256(strict_canonical_json(seed_table)).hexdigest()

    all_tasks = {
        "meta_train": train_tasks,
        "meta_validation_tuning": tuning_tasks,
        "meta_validation_calibration": calibration_tasks,
    }
    probe_counts = _compute_probe_counts(all_tasks)
    probe_counts_sha256 = hashlib.sha256(strict_canonical_json(probe_counts)).hexdigest()

    train_records = [_task_to_jsonl_record(t, "meta_train", i) for i, t in enumerate(train_tasks)]
    tuning_records = [
        _task_to_jsonl_record(t, "meta_validation_tuning", i) for i, t in enumerate(tuning_tasks)
    ]
    calibration_records = [
        _task_to_jsonl_record(t, "meta_validation_calibration", i)
        for i, t in enumerate(calibration_tasks)
    ]

    public_dir = out / "public"
    tasks_dir = public_dir / "tasks"
    audits_dir = public_dir / "audits"
    for directory in (public_dir, tasks_dir, audits_dir):
        directory.mkdir(parents=True, exist_ok=False)

    file_entries: list[dict[str, Any]] = []

    def _record(rel_path: str, sha: str, size: int, visibility: str, role: str) -> None:
        file_entries.append(
            {
                "path": rel_path,
                "sha256": sha,
                "size_bytes": size,
                "visibility": visibility,
                "role": role,
            }
        )

    for name, records, visibility, role in (
        ("meta_train", train_records, "public", "train_tasks"),
        ("meta_validation_tuning", tuning_records, "public", "tuning_tasks"),
        ("meta_validation_calibration", calibration_records, "calibration", "calibration_tasks"),
    ):
        path = tasks_dir / f"{name}.jsonl"
        sha, size = _write_canonical_jsonl(path, records)
        _record(f"public/tasks/{name}.jsonl", sha, size, visibility, role)

    generator_source = Path(__file__).resolve().parent / "taskgen_v2.py"
    generator_source_sha256 = hashlib.sha256(generator_source.read_bytes()).hexdigest()
    generator_obj = {
        "schema": TASKGEN_SCHEMA_V2,
        "algorithm": GENERATOR_ALGORITHM,
        "root_seed": root_seed,
        "family_order": [f.value for f in _FAMILY_ORDER],
        "max_collision_nonce": MAX_COLLISION_NONCE,
        "generator_source_sha256": generator_source_sha256,
        "superseded_lineage": "oczy/meta-cortex/taskgen/v1-dev",
        "base_instrument_id": BASE_INSTRUMENT_ID,
        "base_definition_sha256": BASE_DEFINITION_SHA256,
        "amendments_applied": ["amendment_a_bare_answer", "amendment_b_full_oracle_description"],
        "comparability": (
            "v3 and v4-r20 task scores are not comparable: the prompt registry and "
            "max_new_tokens changed by design, and v1/v2/v3 scores are not comparable "
            "to v4-r20 for the same reason"
        ),
    }
    gen_path = public_dir / "generator.json"
    sha, size = _write_canonical_json(gen_path, generator_obj)
    _record("public/generator.json", sha, size, "public", "generator_config")

    seeds_path = public_dir / "seeds.json"
    sha, size = _write_canonical_json(seeds_path, seed_table)
    _record("public/seeds.json", sha, size, "public", "dev_seeds")

    prompts_path = public_dir / "prompts.json"
    sha, size = _write_canonical_json(prompts_path, prompt_registry)
    _record("public/prompts.json", sha, size, "public", "prompt_registry")

    scorers_path = public_dir / "scorers.json"
    sha, size = _write_canonical_json(scorers_path, scorer_registry)
    _record("public/scorers.json", sha, size, "public", "scorer_registry")

    endpoints_path = public_dir / "endpoints.json"
    sha, size = _write_canonical_json(endpoints_path, endpoint_registry)
    _record("public/endpoints.json", sha, size, "public", "endpoint_registry")

    chat_path = public_dir / "chat_template.txt"
    sha, size = _write_canonical_json(chat_path, {"organ_model_id": organ_model_id})
    _record("public/chat_template.txt", sha, size, "public", "chat_template")

    pc_path = public_dir / "probe_counts.json"
    sha, size = _write_canonical_json(pc_path, probe_counts)
    _record("public/probe_counts.json", sha, size, "public", "probe_counts")

    leakage_summary = {
        "per_domain_counts": leakage_audit["per_domain_counts"],
        "pairwise_overlap": leakage_audit["pairwise_overlap"],
        "within_domain_duplicates": leakage_audit["within_domain_duplicates"],
        "meta_test_seed_present_in_public_files": False,
        "meta_test_records_present_in_dev_view": False,
        "meta_test_records_present_in_calibration_view": False,
        "passed": leakage_audit["passed"],
        "audit_scope": (
            "three DEV domains only; v4-r20 carries no sealed meta-test payload. The "
            "prompt amendments change probe messages, not rule/assignment/composition/"
            "paraphrase fingerprints, so the audit is unaffected by construction."
        ),
    }
    audit_path = audits_dir / "leakage_summary.json"
    sha, size = _write_canonical_json(audit_path, leakage_summary)
    _record("public/audits/leakage_summary.json", sha, size, "public", "leakage_audit")

    dev_view_body = {
        "schema": DEV_VIEW_SCHEMA,
        "instrument_id": INSTRUMENT_ID,
        "instrument_version": INSTRUMENT_VERSION,
        "taskgen_schema": TASKGEN_SCHEMA_V2,
        "decoding_mode": "greedy_skip_special_tokens",
        "superceded_instrument_id": BASE_INSTRUMENT_ID,
        "base_definition_sha256": BASE_DEFINITION_SHA256,
        "base_dev_view_sha256": BASE_DEV_VIEW_SHA256,
        "base_catalog_sha256": BASE_CATALOG_SHA256,
        "base_prompt_registry_sha256": V3_FROZEN_HASHES["prompt_registry_sha256"],
        "prompt_registry_sha256": prompt_registry_sha256,
        "scorer_registry_sha256": scorer_registry_sha256,
        "endpoint_registry_sha256": endpoint_registry_sha256,
        "organ_model_id": organ_model_id,
        "organ_revision": organ_revision,
        "organ_parameter_sha256": organ_parameter_sha256,
        "chat_template_sha256": chat_template_sha256,
        "feature_mode": "final_layer_mean_pool",
        "max_new_tokens": max_new_tokens,
        "base_max_new_tokens": 32,
        "feature_dim": feature_dim,
        "d_cortex": d_cortex,
        "soft_bank_width": soft_bank_width,
        "abstain_threshold": abstain_threshold,
        "train_tasks_per_family": train_tasks_per_family,
        "tuning_tasks_per_family": tuning_tasks_per_family,
        "family_order": [f.value for f in _FAMILY_ORDER],
        "task_files": [
            "public/tasks/meta_train.jsonl",
            "public/tasks/meta_validation_tuning.jsonl",
        ],
        "catalog_sha256": rebuilt.catalog_sha256,
        "dev_view_sha256": "",
    }
    dev_view_sha256 = hashlib.sha256(
        strict_canonical_json(
            {
                k: v
                for k, v in dev_view_body.items()
                if k not in ("dev_view_sha256", "definition_sha256")
            }
        )
    ).hexdigest()
    dev_view_body["dev_view_sha256"] = dev_view_sha256

    cal_view_body = {
        "schema": CALIBRATION_VIEW_SCHEMA,
        "instrument_id": INSTRUMENT_ID,
        "instrument_version": INSTRUMENT_VERSION,
        "taskgen_schema": TASKGEN_SCHEMA_V2,
        "definition_sha256": "",
        "scorer_sha256": scorer_registry_sha256,
        "endpoint_schema_sha256": endpoint_registry_sha256,
        "confidence_level": 0.95,
        "target_power": 0.80,
        "minimum_tasks_per_family": calibration_tasks_per_family,
        "developmental_seeds": list(seed_table["developmental"]),
        "evaluation_seeds": list(seed_table["evaluation"]),
        "no_update_repeat_seeds": list(seed_table["no_update_repeat"]),
        "task_cluster_bootstrap_seed": seed_table["task_cluster_bootstrap"],
        "calibration_tasks_per_family": {
            f.value: calibration_tasks_per_family for f in _FAMILY_ORDER
        },
        "family_order": [f.value for f in _FAMILY_ORDER],
        "task_files": ["public/tasks/meta_validation_calibration.jsonl"],
        "calibration_tasks_per_family_note": (
            "counts only; no margin, threshold or power value is set or implied here"
        ),
        "calibration_view_sha256": "",
    }
    calibration_view_sha256 = hashlib.sha256(
        strict_canonical_json(
            {
                k: v
                for k, v in cal_view_body.items()
                if k not in ("calibration_view_sha256", "definition_sha256")
            }
        )
    ).hexdigest()
    cal_view_body["calibration_view_sha256"] = calibration_view_sha256

    support_bundle_sha256 = _support_bundle_digest(bundle)
    task_counts = {
        "meta_train": len(train_tasks),
        "meta_validation_tuning": len(tuning_tasks),
        "meta_validation_calibration": len(calibration_tasks),
    }
    file_entries.sort(key=lambda e: e["path"])

    def_body = {
        "schema": INSTRUMENT_DEFINITION_SCHEMA,
        "instrument_id": INSTRUMENT_ID,
        "instrument_version": INSTRUMENT_VERSION,
        "lifecycle_state": "definition",
        "dev_only": True,
        "sealed_payload_present": False,
        "meta_test_authorized": False,
        "naming": NAMING_RECORD,
        "base_instrument_id": BASE_INSTRUMENT_ID,
        "base_definition_sha256": BASE_DEFINITION_SHA256,
        "base_dev_view_sha256": BASE_DEV_VIEW_SHA256,
        "base_catalog_sha256": BASE_CATALOG_SHA256,
        "base_task_file_sha256": BASE_TASK_FILE_SHA256,
        "base_prompt_registry_sha256": V3_FROZEN_HASHES["prompt_registry_sha256"],
        "amendments": AMENDMENT_RECORD,
        "max_new_tokens_change": MAX_NEW_TOKENS_RECORD,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "taskgen_schema": TASKGEN_SCHEMA_V2,
        "generator_algorithm": GENERATOR_ALGORITHM,
        "generator_source_sha256": generator_source_sha256,
        "prompt_schema": PROMPT_SCHEMA,
        "prompt_registry_sha256": prompt_registry_sha256,
        "scorer_schema": SCORER_SCHEMA,
        "scorer_registry_sha256": scorer_registry_sha256,
        "endpoint_schema": ENDPOINT_SCHEMA,
        "endpoint_registry_sha256": endpoint_registry_sha256,
        "organ_model_id": organ_model_id,
        "organ_revision": organ_revision,
        "organ_parameter_sha256": organ_parameter_sha256,
        "chat_template_sha256": chat_template_sha256,
        "feature_mode": "final_layer_mean_pool",
        "decoding_mode": "greedy_skip_special_tokens",
        "max_new_tokens": max_new_tokens,
        "feature_dim": feature_dim,
        "d_cortex": d_cortex,
        "soft_bank_width": soft_bank_width,
        "event_min": event_min,
        "event_max": event_max,
        "family_order": [f.value for f in _FAMILY_ORDER],
        "train_tasks_per_family": train_tasks_per_family,
        "tuning_tasks_per_family": tuning_tasks_per_family,
        "calibration_tasks_per_family": calibration_tasks_per_family,
        "task_counts": task_counts,
        "developmental_seeds": list(seed_table["developmental"]),
        "evaluation_seeds": list(seed_table["evaluation"]),
        "dev_seed_table_sha256": dev_seed_table_sha256,
        "probe_counts_sha256": probe_counts_sha256,
        "catalog_sha256": rebuilt.catalog_sha256,
        "support_bundle_sha256": support_bundle_sha256,
        "dev_view_sha256": dev_view_sha256,
        "calibration_view_sha256": calibration_view_sha256,
        "public_files": file_entries,
        "definition_sha256": "",
    }
    definition_sha256 = hashlib.sha256(
        strict_canonical_json({k: v for k, v in def_body.items() if k != "definition_sha256"})
    ).hexdigest()
    def_body["definition_sha256"] = definition_sha256

    dev_view_body["definition_sha256"] = definition_sha256
    cal_view_body["definition_sha256"] = definition_sha256

    _write_canonical_json(public_dir / "DEV_VIEW.json", dev_view_body)
    _write_canonical_json(public_dir / "CALIBRATION_VIEW.json", cal_view_body)
    _write_canonical_json(out / "DEFINITION.json", def_body)

    return V4R20Definition(
        definition_sha256=definition_sha256,
        dev_view_sha256=dev_view_sha256,
        calibration_view_sha256=calibration_view_sha256,
        dev_seed_table_sha256=dev_seed_table_sha256,
        probe_counts_sha256=probe_counts_sha256,
        catalog_sha256=rebuilt.catalog_sha256,
        support_bundle_sha256=support_bundle_sha256,
        prompt_registry_sha256=prompt_registry_sha256,
        leakage_audit=leakage_summary,
        file_entries=tuple(file_entries),
        task_counts=task_counts,
        probe_counts=probe_counts,
        max_new_tokens=max_new_tokens,
        root=out,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_v4_r20_definition(root: Path) -> V4R20Definition:
    """Verify a frozen v4-r20 DEV instrument at *root*, hashes included.

    Fails closed on: a wrong schema or identity, a self-hash mismatch, a file
    whose bytes do not match its recorded SHA-256 or size, a scorer/endpoint
    registry that drifted from the frozen v2/v3 value, an amended prompt
    registry that is byte-identical to v3's, a ``max_new_tokens`` other than the
    authorized value, an unlisted file in the tree, any sign of a sealed
    payload, and a materialized task file that differs from a fresh rebuild of
    the v2-dev lineage with the approved amendments applied.
    """
    root = Path(root)
    if not root.is_dir():
        raise V4R20DefinitionError(f"Not a directory: {root}")

    def_path = root / "DEFINITION.json"
    if not def_path.is_file():
        raise V4R20DefinitionError(f"DEFINITION.json not found in {root}")
    data = strict_json_loads(def_path.read_bytes().decode("utf-8").rstrip("\n"))

    if data.get("schema") != INSTRUMENT_DEFINITION_SCHEMA:
        raise V4R20DefinitionError(
            f"Wrong schema: expected {INSTRUMENT_DEFINITION_SCHEMA!r}, got {data.get('schema')!r}"
        )
    if data.get("instrument_id") != INSTRUMENT_ID:
        raise V4R20DefinitionError(f"instrument_id must be {INSTRUMENT_ID!r}")
    if data.get("instrument_version") != INSTRUMENT_VERSION:
        raise V4R20DefinitionError(f"instrument_version must be {INSTRUMENT_VERSION!r}")
    if data.get("taskgen_schema") != TASKGEN_SCHEMA_V2:
        raise V4R20DefinitionError(
            f"v4-r20 must bind the v2 taskgen lineage, got {data.get('taskgen_schema')!r}"
        )
    if data.get("base_definition_sha256") != BASE_DEFINITION_SHA256:
        raise V4R20DefinitionError("base_definition_sha256 is not the pinned v3 value")
    if data.get("base_dev_view_sha256") != BASE_DEV_VIEW_SHA256:
        raise V4R20DefinitionError("base_dev_view_sha256 is not the pinned v3 value")
    if data.get("max_new_tokens") != MAX_NEW_TOKENS:
        raise V4R20DefinitionError(
            f"max_new_tokens must be the authorized {MAX_NEW_TOKENS}, "
            f"got {data.get('max_new_tokens')!r}"
        )

    stored_hash = data.get("definition_sha256")
    if not isinstance(stored_hash, str) or len(stored_hash) != 64:
        raise V4R20DefinitionError("definition_sha256 missing or malformed")
    computed = hashlib.sha256(
        strict_canonical_json({k: v for k, v in data.items() if k != "definition_sha256"})
    ).hexdigest()
    if computed != stored_hash:
        raise V4R20DefinitionError(
            f"Definition self-hash mismatch: expected {stored_hash}, computed {computed}"
        )

    if data.get("sealed_payload_present") is not False:
        raise V4R20DefinitionError("v4-r20 definition must declare sealed_payload_present=false")
    if data.get("meta_test_authorized") is not False:
        raise V4R20DefinitionError("v4-r20 definition must declare meta_test_authorized=false")
    if (root / "sealed").exists():
        raise V4R20DefinitionError("v4-r20 definition directory must not contain sealed/")

    entries = data.get("public_files")
    if not isinstance(entries, list) or not entries:
        raise V4R20DefinitionError("public_files missing from definition")
    for entry in entries:
        if not isinstance(entry, dict):
            raise V4R20DefinitionError("public_files entries must be objects")
        if entry.get("visibility") == "sealed":
            raise V4R20DefinitionError(f"v4-r20 must not list a sealed file: {entry.get('path')!r}")
        path = root / entry["path"]
        if not path.is_file():
            raise V4R20DefinitionError(f"File not found: {path}")
        if path.is_symlink():
            raise V4R20DefinitionError(f"Symlink not allowed: {path}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise V4R20DefinitionError(f"File hash mismatch for {entry['path']}")
        if len(raw) != entry["size_bytes"]:
            raise V4R20DefinitionError(f"File size mismatch for {entry['path']}")

    listed = {entry["path"] for entry in entries}
    allowed_unlisted = {
        "DEFINITION.json",
        "public/DEV_VIEW.json",
        "public/CALIBRATION_VIEW.json",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel not in listed and rel not in allowed_unlisted:
            raise V4R20DefinitionError(f"Unlisted file in frozen instrument: {rel}")

    for key, expected in INHERITED_FROZEN_HASHES.items():
        if data.get(key) != expected:
            raise V4R20DefinitionError(
                f"registry drift: {key} is {data.get(key)!r}, frozen value is {expected!r}"
            )
    expected_prompt = _amended_prompt_registry_sha256()
    if data.get("prompt_registry_sha256") != expected_prompt:
        raise V4R20DefinitionError(
            "prompt_registry_sha256 is not the amended v4-r20 registry; the approved "
            "amendments are not carried as recorded"
        )
    if data.get("base_prompt_registry_sha256") != V3_FROZEN_HASHES["prompt_registry_sha256"]:
        raise V4R20DefinitionError("base_prompt_registry_sha256 is not the frozen v3 value")
    if data.get("prompt_registry_sha256") == data.get("base_prompt_registry_sha256"):
        raise V4R20DefinitionError(
            "the amended prompt registry equals the base registry; the amendments are absent"
        )

    for rel_path, key in (
        ("public/DEV_VIEW.json", "dev_view_sha256"),
        ("public/CALIBRATION_VIEW.json", "calibration_view_sha256"),
    ):
        view = strict_json_loads((root / rel_path).read_bytes().decode("utf-8").rstrip("\n"))
        if view.get(key) != data.get(key):
            raise V4R20DefinitionError(f"{rel_path} self-hash disagrees with DEFINITION.json")
        recomputed = hashlib.sha256(
            strict_canonical_json(
                {k: v for k, v in view.items() if k not in (key, "definition_sha256")}
            )
        ).hexdigest()
        if recomputed != data.get(key):
            raise V4R20DefinitionError(f"{rel_path} self-hash mismatch")
        if view.get("definition_sha256") != stored_hash:
            raise V4R20DefinitionError(f"{rel_path} does not bind this definition")
        if view.get("instrument_id") != INSTRUMENT_ID:
            raise V4R20DefinitionError(f"{rel_path} has the wrong instrument_id")

    dev_view = strict_json_loads(
        (root / "public" / "DEV_VIEW.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    if dev_view.get("max_new_tokens") != MAX_NEW_TOKENS:
        raise V4R20DefinitionError("DEV view max_new_tokens is not the authorized value")
    for task_file in dev_view.get("task_files", []):
        if "calibration" in task_file or "sealed" in task_file:
            raise V4R20DefinitionError(f"DEV view references a held-back file: {task_file}")
    cal_view = strict_json_loads(
        (root / "public" / "CALIBRATION_VIEW.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    for task_file in cal_view.get("task_files", []):
        if "meta_train" in task_file or "tuning" in task_file or "sealed" in task_file:
            raise V4R20DefinitionError(
                f"calibration view references a non-calibration file: {task_file}"
            )

    _rebuild_and_compare(root, data)

    audit = strict_json_loads(
        (root / "public" / "audits" / "leakage_summary.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    if audit.get("passed") is not True:
        raise V4R20DefinitionError("recorded leakage audit did not pass")

    probe_counts = strict_json_loads(
        (root / "public" / "probe_counts.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    if hashlib.sha256(strict_canonical_json(probe_counts)).hexdigest() != data.get(
        "probe_counts_sha256"
    ):
        raise V4R20DefinitionError("probe_counts.json does not match probe_counts_sha256")

    return V4R20Definition(
        definition_sha256=stored_hash,
        dev_view_sha256=data["dev_view_sha256"],
        calibration_view_sha256=data["calibration_view_sha256"],
        dev_seed_table_sha256=data["dev_seed_table_sha256"],
        probe_counts_sha256=data["probe_counts_sha256"],
        catalog_sha256=data["catalog_sha256"],
        support_bundle_sha256=data["support_bundle_sha256"],
        prompt_registry_sha256=data["prompt_registry_sha256"],
        leakage_audit=audit,
        file_entries=tuple(entries),
        task_counts=data["task_counts"],
        probe_counts=probe_counts,
        max_new_tokens=data["max_new_tokens"],
        root=root,
    )


def _rebuild_and_compare(root: Path, data: dict[str, Any]) -> DevTaskCatalog:
    """Rebuild the amended lineage and require it to equal the frozen files."""
    generator = strict_json_loads(
        (root / "public" / "generator.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    seed = generator.get("root_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise V4R20DefinitionError("generator.json has no usable root_seed")

    tg_config = TaskGeneratorConfig(
        root_seed=seed,
        train_tasks_per_family=data["train_tasks_per_family"],
        validation_tasks_per_family=(
            data["tuning_tasks_per_family"] + data["calibration_tasks_per_family"]
        ),
        min_events=data["event_min"],
        max_events=data["event_max"],
    )
    catalog, _bundle = build_dev_catalog_v2(tg_config)
    if catalog.catalog_sha256 != data["catalog_sha256"]:
        raise V4R20DefinitionError(
            f"rebuilt catalog digest {catalog.catalog_sha256} does not match the frozen "
            f"{data['catalog_sha256']}"
        )
    if data["catalog_sha256"] != BASE_CATALOG_SHA256:
        raise V4R20DefinitionError("frozen catalog digest is not the pinned v3 value")

    train_tasks = tuple(amend_task(t) for t in catalog.meta_train)
    base_tuning, base_calibration = _split_validation(
        catalog,
        tuning_tasks_per_family=data["tuning_tasks_per_family"],
        calibration_tasks_per_family=data["calibration_tasks_per_family"],
    )
    tuning_tasks = tuple(amend_task(t) for t in base_tuning)
    calibration_tasks = tuple(amend_task(t) for t in base_calibration)

    expected_files = {
        "public/tasks/meta_train.jsonl": ("meta_train", train_tasks),
        "public/tasks/meta_validation_tuning.jsonl": ("meta_validation_tuning", tuning_tasks),
        "public/tasks/meta_validation_calibration.jsonl": (
            "meta_validation_calibration",
            calibration_tasks,
        ),
    }
    for rel_path, (split_role, tasks) in expected_files.items():
        path = root / rel_path
        if not path.is_file():
            raise V4R20DefinitionError(f"missing task file {rel_path}")
        rebuilt = b"".join(
            strict_canonical_json(_task_to_jsonl_record(t, split_role, i)) + b"\n"
            for i, t in enumerate(tasks)
        )
        if rebuilt != path.read_bytes():
            raise V4R20DefinitionError(
                f"{rel_path} differs from a fresh rebuild of the amended v2-dev lineage"
            )
    return catalog


def assert_catalog_split_firewall(catalog: DevTaskCatalog) -> Any:
    """Public helper: assert the v2 lineage's train/validation firewall."""
    from .taskgen import audit_split_firewall

    audit = audit_split_firewall(tuple(catalog.meta_train), tuple(catalog.meta_validation))
    assert_split_firewall(audit)
    return audit


def probe_count_by_kind(tasks: Sequence[MetaTask]) -> dict[str, int]:
    """Count probes per kind across *tasks* (used by freeze reports)."""
    counts: dict[str, int] = {}
    for task in tasks:
        for kind in ProbeKind:
            name = kind.value
            counts[name] = counts.get(name, 0) + len(task.probes.by_kind(kind))
    return counts


# Keep the unused-import linters honest about names re-exported for callers.
_ = (DevSplit,)
