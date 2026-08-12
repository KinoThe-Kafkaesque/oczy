#!/usr/bin/env python3
"""
Reproducible OMP RPC riddle-reasoning benchmark (Oczy Original Riddle Benchmark v1).

Each (model, form) session launches an isolated OMP RPC process with a single
strict host tool (``submit_riddle_answers``), a frozen 20-riddle bank rendered
into one of five deterministic answer-position forms, auto-retry/auto-compaction
disabled, and exact event capture.  The probe measures exact displayed-choice
accuracy over 100 decisions per model (20 riddles × 5 forms), per-form and
per-riddle scores, majority (≥3/5) and stable (5/5) counts, category and
answer-position breakdowns, tool-format failures/repairs, timing/usage, and
paired per-riddle win/loss/tie with deterministic cluster-bootstrap intervals.

OMP model selectors are positional::

    python benchmarks/omp/riddle_reasoning_probe.py devin/glm-5-2 devin/kimi-k2-7 \\
        --forms 5 --seed 20260712 --output /tmp/riddle-bench.json

Exit codes:
    0  — every scheduled session completed (model failures are measured results)
    2  — probe/config/infrastructure failure before schedule completion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

# ─── constants ────────────────────────────────────────────────────────

SCHEMA_VERSION = "oczy/omp-riddle-reasoning-probe/v1"
PROBE_VERSION = "1.0.2"
FORM_GENERATOR_VERSION = "1.0.0"

BANK_SCHEMA_VERSION = "1.0"
BANK_BENCHMARK_VERSION = "oczy_original_riddle_benchmark_v1"

EXIT_OK = 0
EXIT_PROBE_ERROR = 2

TOOL_NAME = "submit_riddle_answers"

RIDDLES_PER_FORM = 20
OPTIONS_PER_RIDDLE = 5
DISPLAY_LETTERS = ("A", "B", "C", "D", "E")
GUESS_RATE = 0.2  # 1 / OPTIONS_PER_RIDDLE

DEFAULT_FORMS = 5
DEFAULT_SEED = 20260712
DEFAULT_MAX_FORMAT_ATTEMPTS = 3
DEFAULT_TRIAL_TIMEOUT_SECONDS = 300
DEFAULT_OMP_BIN = "omp"

STDERR_TAIL_LINES = 50

# Format-only retry instruction (never reveals correctness).
FORMAT_RETRY_INSTRUCTION = (
    "The submission has a structural format problem: {detail}. "
    "Ensure you provide exactly 20 records with unique 'number' values 1..20, "
    "each having a 'choice' field that is one of A, B, C, D, E and an "
    "'explanation' field. Then call submit_riddle_answers again."
)
ACCEPTED_TEXT = "ACCEPTED. Reply with exactly DONE."
DUPLICATE_TEXT = (
    "Answers were already accepted. Stop calling submit_riddle_answers "
    "and reply with exactly DONE."
)

# Pinned system prompt — identical across all models/sessions.
SYSTEM_PROMPT = (
    "You are a riddle-reasoning agent. You will be presented with 20 original "
    "riddles, each with five answer options labeled A through E.\n\n"
    "Rules:\n"
    "1. For each riddle, choose exactly one option (A, B, C, D, or E).\n"
    "2. Call the submit_riddle_answers tool exactly once with:\n"
    "   - trial_id: the trial identifier provided in the prompt\n"
    "   - answers: an array of exactly 20 records, each with:\n"
    "     • number: the riddle number (1..20)\n"
    "     • choice: the letter of your chosen option (A, B, C, D, or E)\n"
    "     • explanation: a brief explanation of your reasoning\n"
    "3. Every riddle number 1..20 must appear exactly once.\n"
    "4. If you receive a tool error about formatting, fix the structure and "
    "retry.\n"
    "5. Do not call any other tool.\n"
    "6. After successful submission, reply with exactly: DONE"
)

# ─── errors ───────────────────────────────────────────────────────────


class ProbeError(Exception):
    """Raised when the probe itself fails (not individual session results)."""

    def __init__(self, message: str, exit_code: int = EXIT_PROBE_ERROR) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# ─── hashing helpers ──────────────────────────────────────────────────


def _sha256_hex(s: str) -> str:
    """SHA-256 of a UTF-8 string, returned as lowercase hex."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _public_model_metadata(model: Any) -> Any:
    """Retain reproducibility fields without persisting provider credentials."""
    if not isinstance(model, dict):
        return model
    safe_keys = (
        "id",
        "name",
        "api",
        "provider",
        "baseUrl",
        "reasoning",
        "input",
        "cost",
        "contextWindow",
        "maxTokens",
        "thinking",
    )
    return {key: model[key] for key in safe_keys if key in model}


def _probe_file_sha256() -> str:
    """SHA-256 of this probe script file."""
    return _sha256_bytes(Path(__file__).resolve().read_bytes())


def _bank_file_sha256(bank_path: Path) -> str:
    """SHA-256 of the exact bytes of the bank file."""
    return _sha256_bytes(bank_path.resolve().read_bytes())


def _seeded_int(*parts: str) -> int:
    """Derive a deterministic integer from SHA-256 of colon-joined parts."""
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ─── binary resolution ────────────────────────────────────────────────


def resolve_omp_binary(omp_bin: str) -> Path:
    """Resolve the OMP binary to an absolute, executable Path."""
    resolved = shutil.which(omp_bin)
    if resolved is None:
        raise ProbeError(
            f"OMP binary not found: {omp_bin!r}. "
            "Set --omp-bin or ensure 'omp' is on PATH."
        )
    p = Path(resolved)
    if not p.is_file():
        raise ProbeError(f"OMP binary is not a regular file: {p}")
    return p


def capture_omp_version(omp_path: Path) -> str:
    """Run ``omp --version`` and return the stripped stdout."""
    try:
        result = subprocess.run(
            [str(omp_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise ProbeError(f"Failed to run 'omp --version': {exc}") from exc
    if result.returncode != 0:
        raise ProbeError(
            f"'omp --version' exited with code {result.returncode}: "
            f"{result.stderr.strip()[:200]}"
        )
    version = result.stdout.strip()
    if not version:
        version = result.stderr.strip()
    if not version:
        raise ProbeError("'omp --version' produced no output")
    return version


# ─── bank validation ──────────────────────────────────────────────────


def validate_bank(bank: dict[str, Any]) -> None:
    """Validate the frozen riddle bank structure and coverage.

    Raises ValueError on any structural, coverage, or correctness violation.
    Checks: top-level schema, exactly 20 riddles, 10 categories × 2,
    5 options per riddle, one correct option referencing an existing option_id,
    unique riddle ids, unique option_ids within each riddle, no empty fields.
    """
    if not isinstance(bank, dict):
        raise ValueError("Bank must be a JSON object")

    if bank.get("schema_version") != BANK_SCHEMA_VERSION:
        raise ValueError(
            f"Bank schema_version must be {BANK_SCHEMA_VERSION!r}, got "
            f"{bank.get('schema_version')!r}"
        )

    if bank.get("benchmark_version") != BANK_BENCHMARK_VERSION:
        raise ValueError(
            f"Bank benchmark_version must be {BANK_BENCHMARK_VERSION!r}, got "
            f"{bank.get('benchmark_version')!r}"
        )

    riddles = bank.get("riddles")
    if not isinstance(riddles, list):
        raise ValueError("Bank 'riddles' must be an array")
    if len(riddles) != RIDDLES_PER_FORM:
        raise ValueError(
            f"Bank must contain exactly {RIDDLES_PER_FORM} riddles, got {len(riddles)}"
        )

    seen_ids: set[str] = set()
    category_counts: dict[str, int] = {}

    for i, riddle in enumerate(riddles):
        if not isinstance(riddle, dict):
            raise ValueError(f"Riddle {i} must be an object")

        rid = riddle.get("id")
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"Riddle {i} 'id' must be a non-empty string")
        if rid in seen_ids:
            raise ValueError(f"Duplicate riddle id: {rid!r}")
        seen_ids.add(rid)

        category = riddle.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"Riddle {rid} 'category' must be a non-empty string")
        category_counts[category] = category_counts.get(category, 0) + 1

        difficulty = riddle.get("difficulty")
        if difficulty not in ("medium", "hard", "very_hard"):
            raise ValueError(
                f"Riddle {rid} 'difficulty' must be medium/hard/very_hard, got {difficulty!r}"
            )

        prompt = riddle.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Riddle {rid} 'prompt' must be a non-empty string")

        options = riddle.get("options")
        if not isinstance(options, list) or len(options) != OPTIONS_PER_RIDDLE:
            raise ValueError(
                f"Riddle {rid} must have exactly {OPTIONS_PER_RIDDLE} options, "
                f"got {len(options) if isinstance(options, list) else 'non-list'}"
            )

        option_ids: set[str] = set()
        for j, opt in enumerate(options):
            if not isinstance(opt, dict):
                raise ValueError(f"Riddle {rid} option {j} must be an object")
            oid = opt.get("option_id")
            if not isinstance(oid, str) or not oid:
                raise ValueError(
                    f"Riddle {rid} option {j} 'option_id' must be a non-empty string"
                )
            if oid in option_ids:
                raise ValueError(f"Riddle {rid} has duplicate option_id: {oid!r}")
            option_ids.add(oid)
            otext = opt.get("text")
            if not isinstance(otext, str) or not otext.strip():
                raise ValueError(
                    f"Riddle {rid} option {oid} 'text' must be a non-empty string"
                )

        correct_id = riddle.get("correct_option_id")
        if not isinstance(correct_id, str) or not correct_id:
            raise ValueError(
                f"Riddle {rid} 'correct_option_id' must be a non-empty string"
            )
        if correct_id not in option_ids:
            raise ValueError(
                f"Riddle {rid} 'correct_option_id' {correct_id!r} does not match "
                f"any option_id"
            )

        for field_name in ("proof", "ambiguity_audit", "originality_note"):
            val = riddle.get(field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"Riddle {rid} '{field_name}' must be a non-empty string"
                )

        # verification may be a string or a structured object.
        ver_val = riddle.get("verification")
        if isinstance(ver_val, str):
            if not ver_val.strip():
                raise ValueError(f"Riddle {rid} 'verification' must be non-empty")
        elif isinstance(ver_val, dict):
            if not ver_val:
                raise ValueError(f"Riddle {rid} 'verification' must be non-empty")
        else:
            raise ValueError(
                f"Riddle {rid} 'verification' must be a string or object, "
                f"got {type(ver_val).__name__}"
            )

    # Validate 10 categories × 2
    if len(category_counts) != 10:
        raise ValueError(
            f"Bank must have exactly 10 categories, got {len(category_counts)}"
        )
    for cat, count in category_counts.items():
        if count != 2:
            raise ValueError(
                f"Category {cat!r} must have exactly 2 riddles, got {count}"
            )


def load_and_validate_bank(bank_path: Path) -> tuple[dict[str, Any], str]:
    """Load bank JSON from file, validate it, and return (bank, sha256_of_bytes)."""
    if not bank_path.is_file():
        raise ProbeError(f"Bank file not found: {bank_path}")
    raw_bytes = bank_path.resolve().read_bytes()
    bank_sha = _sha256_bytes(raw_bytes)
    try:
        bank = json.loads(raw_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProbeError(f"Bank file is not valid JSON: {exc}") from exc
    try:
        validate_bank(bank)
    except ValueError as exc:
        raise ProbeError(f"Bank validation failed: {exc}") from exc
    return bank, bank_sha


# ─── form generation ──────────────────────────────────────────────────


def _base_permutation(seed: int, riddle_index: int, n_options: int) -> list[int]:
    """Derive a deterministic base permutation of option indices for a riddle.

    Returns a list of n_options semantic-option indices in display order.
    """
    rng = random.Random(_seeded_int("perm", str(seed), str(riddle_index)))
    perm = list(range(n_options))
    rng.shuffle(perm)
    return perm


def _cyclic_rotate(perm: list[int], shift: int) -> list[int]:
    """Cyclically rotate a permutation by shift positions."""
    n = len(perm)
    shift = shift % n
    return perm[shift:] + perm[:shift]


def generate_forms(
    bank: dict[str, Any], n_forms: int = DEFAULT_FORMS, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Generate n_forms deterministic answer-position forms from the bank.

    For each riddle, a deterministic seed-derived base option permutation is
    computed, then cyclically rotated by form index 0..n_forms-1 so that every
    semantic option occupies every displayed A–E position exactly once across
    the forms (when n_forms == OPTIONS_PER_RIDDLE).  Riddle order is shuffled
    deterministically per form.

    Each form dict contains:
      - form_index: int
      - riddle_order: list[int] — bank indices in display order
      - option_permutations: list[list[int]] — per riddle, maps display position
        (0=A, 1=B, ...) to semantic option index in bank
      - displayed: list[dict] — rendered riddles with number, prompt, options
      - sha256: str — hash of the canonical form representation
      - correct_letters: list[str] — correct answer letter per riddle (for scoring)
    """
    if n_forms < 1 or n_forms > DEFAULT_FORMS:
        raise ValueError(f"n_forms must be 1..{DEFAULT_FORMS}, got {n_forms}")

    riddles = bank["riddles"]
    n_riddles = len(riddles)
    n_options = OPTIONS_PER_RIDDLE

    # Precompute base permutations for each riddle.
    base_perms = [_base_permutation(seed, i, n_options) for i in range(n_riddles)]

    forms: list[dict[str, Any]] = []
    for form_idx in range(n_forms):
        # Deterministic riddle order shuffle per form.
        order_rng = random.Random(
            _seeded_int("order", str(seed), str(form_idx))
        )
        riddle_order = list(range(n_riddles))
        order_rng.shuffle(riddle_order)

        option_permutations: list[list[int]] = []
        displayed: list[dict[str, Any]] = []
        correct_letters: list[str] = []

        for display_number, bank_idx in enumerate(riddle_order):
            riddle = riddles[bank_idx]
            options = riddle["options"]
            correct_option_id = riddle["correct_option_id"]

            # Find the semantic index of the correct option.
            correct_semantic_idx = None
            for oi, opt in enumerate(options):
                if opt["option_id"] == correct_option_id:
                    correct_semantic_idx = oi
                    break
            assert correct_semantic_idx is not None  # validated above

            # Cyclically rotate the base permutation for this form.
            rotated = _cyclic_rotate(base_perms[bank_idx], form_idx)
            option_permutations.append(rotated)

            # Build displayed options: display position -> semantic option.
            disp_options = []
            correct_letter = None
            for pos, sem_idx in enumerate(rotated):
                opt = options[sem_idx]
                letter = DISPLAY_LETTERS[pos]
                disp_options.append(
                    {
                        "letter": letter,
                        "option_id": opt["option_id"],
                        "text": opt["text"],
                    }
                )
                if sem_idx == correct_semantic_idx:
                    correct_letter = letter

            assert correct_letter is not None
            correct_letters.append(correct_letter)

            displayed.append(
                {
                    "number": display_number + 1,
                    "bank_index": bank_idx,
                    "id": riddle["id"],
                    "category": riddle["category"],
                    "difficulty": riddle["difficulty"],
                    "prompt": riddle["prompt"],
                    "options": disp_options,
                }
            )

        # Compute form hash from canonical representation.
        form_canonical = json.dumps(
            {
                "form_index": form_idx,
                "riddle_order": riddle_order,
                "option_permutations": option_permutations,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        form_sha = _sha256_hex(form_canonical)

        forms.append(
            {
                "form_index": form_idx,
                "riddle_order": riddle_order,
                "option_permutations": option_permutations,
                "displayed": displayed,
                "correct_letters": correct_letters,
                "sha256": form_sha,
            }
        )

    return forms


def render_prompt(form: dict[str, Any], trial_id: str) -> str:
    """Render the user prompt for a form, presenting all 20 riddles."""
    lines: list[str] = []
    lines.append(f"Trial ID: {trial_id}")
    lines.append("")
    lines.append("Solve all 20 riddles below. For each, choose one option (A-E).")
    lines.append(
        "Then call submit_riddle_answers with your 20 answers "
        "(number 1..20, choice letter, explanation)."
    )
    lines.append("")

    for riddle in form["displayed"]:
        lines.append(f"Riddle {riddle['number']}:")
        lines.append(riddle["prompt"])
        lines.append("")
        for opt in riddle["options"]:
            lines.append(f"  {opt['letter']}. {opt['text']}")
        lines.append("")

    return "\n".join(lines)


# ─── scoring ──────────────────────────────────────────────────────────


def guessing_corrected_score(accuracy: float) -> float:
    """Compute guessing-corrected score: (accuracy - 0.2) / 0.8.

    Clamped to [0.0, 1.0] range.  A score of 0.0 means pure guessing (20%
    accuracy on 5-option MCQ), 1.0 means perfect accuracy.
    """
    raw = (accuracy - GUESS_RATE) / (1.0 - GUESS_RATE)
    return round(max(0.0, min(1.0, raw)), 6)


def score_submission(submission: list[dict[str, Any]], form: dict[str, Any]) -> dict[str, Any]:
    """Pure scorer: compare submitted choices to form's correct letters.

    Takes a list of {number, choice, explanation} records and a form dict.
    Returns per-riddle 0/1, correct_count, accuracy, and guessing_corrected.
    Never raises on wrong answers — structural issues are returned as flags.
    """
    correct_letters = form["correct_letters"]
    per_riddle: list[int] = [0] * RIDDLES_PER_FORM

    # Build number -> submission record map.
    by_number: dict[int, dict[str, Any]] = {}
    structural_issues: list[str] = []
    for rec in submission:
        if not isinstance(rec, dict):
            structural_issues.append("Non-dict record in submission")
            continue
        num = rec.get("number")
        if not isinstance(num, int) or num < 1 or num > RIDDLES_PER_FORM:
            structural_issues.append(f"Invalid number: {num!r}")
            continue
        if num in by_number:
            structural_issues.append(f"Duplicate number: {num}")
            continue
        by_number[num] = rec

    # Check coverage.
    missing = [n for n in range(1, RIDDLES_PER_FORM + 1) if n not in by_number]
    if missing:
        structural_issues.append(f"Missing numbers: {missing}")

    # Score each riddle.
    correct_count = 0
    for num in range(1, RIDDLES_PER_FORM + 1):
        rec = by_number.get(num)
        if rec is None:
            per_riddle[num - 1] = 0
            continue
        choice = rec.get("choice")
        if not isinstance(choice, str):
            per_riddle[num - 1] = 0
            continue
        choice_upper = choice.strip().upper()
        if choice_upper not in DISPLAY_LETTERS:
            per_riddle[num - 1] = 0
            continue
        idx = num - 1
        if choice_upper == correct_letters[idx]:
            per_riddle[idx] = 1
            correct_count += 1

    accuracy = correct_count / RIDDLES_PER_FORM
    return {
        "per_riddle": per_riddle,
        "correct_count": correct_count,
        "total": RIDDLES_PER_FORM,
        "accuracy": round(accuracy, 6),
        "guessing_corrected": guessing_corrected_score(accuracy),
        "structural_issues": structural_issues,
    }


# ─── tool schema / definition ─────────────────────────────────────────


def build_tool_schema() -> dict[str, Any]:
    """JSON Schema for submit_riddle_answers tool parameters."""
    return {
        "type": "object",
        "properties": {
            "trial_id": {"type": "string"},
            "answers": {
                "type": "array",
                "minItems": RIDDLES_PER_FORM,
                "maxItems": RIDDLES_PER_FORM,
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": RIDDLES_PER_FORM,
                        },
                        "choice": {
                            "type": "string",
                            "enum": list(DISPLAY_LETTERS),
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": ["number", "choice", "explanation"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["trial_id", "answers"],
        "additionalProperties": False,
    }


def build_tool_definition() -> dict[str, Any]:
    """RPC host tool definition for set_host_tools."""
    return {
        "name": TOOL_NAME,
        "description": (
            "Submit your answers to all 20 riddles. Provide the trial_id and "
            "an array of exactly 20 records, each with a 'number' (1..20), "
            "a 'choice' (one of A, B, C, D, E), and an 'explanation' string. "
            "Every number 1..20 must appear exactly once."
        ),
        "parameters": build_tool_schema(),
    }


def build_system_prompt() -> str:
    """Return the constant pinned system prompt."""
    return SYSTEM_PROMPT


# ─── tool argument validation (structure/coverage only) ───────────────


def validate_tool_args(
    args: Any, expected_trial_id: str
) -> tuple[bool, str, list[dict[str, Any]] | None]:
    """Validate tool arguments for structure and coverage only.

    Never checks correctness.  Returns (valid, error_detail, answers_list).
    """
    if not isinstance(args, dict):
        return False, "arguments must be an object", None

    trial_id = args.get("trial_id")
    if trial_id != expected_trial_id:
        return False, f"trial_id must be {expected_trial_id!r}", None

    answers = args.get("answers")
    if not isinstance(answers, list):
        return False, "answers must be an array", None
    if len(answers) != RIDDLES_PER_FORM:
        return False, f"answers must have exactly {RIDDLES_PER_FORM} records", None

    seen_numbers: set[int] = set()
    for i, rec in enumerate(answers):
        if not isinstance(rec, dict):
            return False, f"record {i} must be an object", None
        num = rec.get("number")
        if not isinstance(num, int) or num < 1 or num > RIDDLES_PER_FORM:
            return False, f"record {i} has invalid number {num!r}", None
        if num in seen_numbers:
            return False, f"duplicate number {num}", None
        seen_numbers.add(num)
        choice = rec.get("choice")
        if not isinstance(choice, str) or choice.strip().upper() not in DISPLAY_LETTERS:
            return False, f"record {num} has invalid choice {choice!r}", None
        explanation = rec.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            return False, f"record {num} has missing explanation", None

    missing = [n for n in range(1, RIDDLES_PER_FORM + 1) if n not in seen_numbers]
    if missing:
        return False, f"missing numbers: {missing}", None

    return True, "", answers


# ─── OMP argument repair detection ────────────────────────────────────


def _strip_harness_i(args: Any) -> Any:
    """Strip the harness ``i`` key from args for canonical comparison."""
    if isinstance(args, dict):
        return {k: v for k, v in args.items() if k != "i"}
    return args


def _detect_repair(raw_args: Any, host_args: Any) -> bool:
    """True if OMP repaired the arguments (canonical forms differ)."""
    raw_canonical = json.dumps(
        _strip_harness_i(raw_args), sort_keys=True, ensure_ascii=False,
        default=str,
    )
    host_canonical = json.dumps(
        _strip_harness_i(host_args), sort_keys=True, ensure_ascii=False,
        default=str,
    )
    return raw_canonical != host_canonical


# ─── RPC session ──────────────────────────────────────────────────────


class RpcSession:
    """Manages an isolated OMP RPC subprocess with JSONL stdin/stdout."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self._stdout_q: Queue[str | None] = Queue()
        self._stderr_q: Queue[str | None] = Queue()
        self._stdout_thread = threading.Thread(
            target=self._read_stream,
            args=(self.proc.stdout, self._stdout_q),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stream,
            args=(self.proc.stderr, self._stderr_q),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._closed = False

    @staticmethod
    def _read_stream(stream: Any, q: Queue[str | None]) -> None:
        try:
            for line in stream:
                q.put(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            q.put(None)  # EOF sentinel

    def send(self, obj: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def recv(self, timeout: float) -> str | None:
        """Return next stdout line, or None on timeout/EOF."""
        try:
            return self._stdout_q.get(timeout=timeout)
        except Empty:
            return None

    def drain_stderr(self) -> list[str]:
        lines: list[str] = []
        while True:
            try:
                line = self._stderr_q.get_nowait()
            except Empty:
                break
            if line is None:
                break
            lines.append(line)
        return lines

    def close(self, timeout: float = 5.0) -> int | None:
        if self._closed:
            return self.proc.poll()
        self._closed = True
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        self._stdout_thread.join(timeout=2)
        self._stderr_thread.join(timeout=2)
        return self.proc.poll()


# ─── event helpers ────────────────────────────────────────────────────

_DIAGNOSTIC_TYPES = frozenset(
    {
        "auto_retry_start",
        "auto_retry_end",
        "retry_fallback_applied",
        "retry_fallback_succeeded",
        "auto_compaction_start",
        "auto_compaction_end",
        "notice",
        "extension_error",
        "ttsr_triggered",
        "thinking_level_changed",
    }
)

_IGNORE_TYPES = frozenset(
    {
        "available_commands_update",
        "extension_ui_request",
        "config_update",
        "session_info_update",
        "command_output",
        "host_tool_cancel",
    }
)


def _normalize_event(frame: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact diagnostic record from an event frame."""
    ftype = frame.get("type", "unknown")
    ev: dict[str, Any] = {"type": ftype}
    for k in (
        "attempt",
        "maxAttempts",
        "delayMs",
        "errorMessage",
        "errorId",
        "success",
        "reason",
        "action",
        "level",
        "message",
        "source",
        "from",
        "to",
        "role",
        "model",
        "skipped",
    ):
        if k in frame:
            val = frame[k]
            if isinstance(val, str) and len(val) > 300:
                val = val[:300]
            ev[k] = val
    return ev


def _extract_error_text(result: Any) -> str:
    """Best-effort extraction of error text from a tool_execution_end result."""
    if result is None:
        return ""
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return str(item.get("text", ""))
        if result.get("isError"):
            return str(result.get("text", ""))
    return str(result)[:500]


def _capture_message(msg: dict[str, Any], index: int) -> dict[str, Any]:
    """Capture a per-message summary from a message_end assistant message."""
    content = msg.get("content", [])
    if not isinstance(content, list):
        content = []
    thinking_summary: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking":
            text = str(block.get("thinking", ""))
            thinking_summary = {
                "char_count": len(text),
                "sha256": _sha256_hex(text),
            }
        elif btype == "redactedThinking":
            data = str(block.get("data", ""))
            thinking_summary = {
                "char_count": len(data),
                "sha256": _sha256_hex(data),
                "redacted": True,
            }
        elif btype == "toolCall":
            tool_calls.append(
                {
                    "tool_call_id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": block.get("arguments"),
                }
            )
        elif btype == "text":
            text_parts.append(str(block.get("text", "")))

    usage = msg.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    error = None
    if msg.get("errorMessage"):
        error = {
            "message": msg.get("errorMessage"),
            "status": msg.get("errorStatus"),
            "id": msg.get("errorId"),
        }

    retry_recovery = msg.get("retryRecovery")

    return {
        "index": index,
        "ttft": msg.get("ttft"),
        "duration": msg.get("duration"),
        "usage": {
            "input": usage.get("input"),
            "output": usage.get("output"),
            "cacheRead": usage.get("cacheRead"),
            "cacheWrite": usage.get("cacheWrite"),
            "totalTokens": usage.get("totalTokens"),
        },
        "stop_reason": msg.get("stopReason"),
        "thinking": thinking_summary,
        "tool_calls": tool_calls,
        "text": " ".join(text_parts)[:2000] if text_parts else None,
        "error": error,
        "retry_recovery": retry_recovery,
    }


# ─── dataclasses ──────────────────────────────────────────────────────


@dataclass
class TrialResult:
    """Result of a single (model, form) session."""

    trial_id: str
    model: str
    form_index: int
    accepted: bool
    submission: list[dict[str, Any]] | None
    raw_tool_args: Any
    effective_tool_args: Any
    omp_argument_repair: bool
    format_attempts: int
    format_errors: list[str]
    score: dict[str, Any] | None
    ttft: float | None
    duration: float | None
    usage: dict[str, Any]
    wall_time_seconds: float
    turn_count: int
    message_count: int
    auto_retry_events: int
    auto_compaction_events: int
    diagnostic_events: list[dict[str, Any]]
    stderr_tail: list[str]
    non_json_rpc_lines: list[str]
    final_assistant_text: str
    process_exit_code: int | None
    omp_model_resolved: str | None
    probe_error: bool
    failure_category: str | None
    baseline_context_usage: dict[str, Any] | None
    final_context_usage: dict[str, Any] | None
    messages: list[dict[str, Any]]
    attempts: list[dict[str, Any]]


@dataclass
class ModelResult:
    """Aggregated results for one model across all forms."""

    model: str
    trials: list[TrialResult]
    per_form_scores: list[dict[str, Any]]
    per_riddle_scores: list[int]  # 0..5 per bank riddle
    majority_correct: int  # count of riddles with >=3/5 correct
    stable_correct: int  # count of riddles with 5/5 correct
    category_scores: dict[str, dict[str, Any]]
    answer_position_scores: dict[str, dict[str, Any]]
    tool_format_failures: int
    tool_format_repairs: int
    primary_accuracy: float
    guessing_corrected: float
    mean_wall_time_seconds: float
    mean_ttft: float | None
    total_usage: dict[str, Any]


@dataclass
class ProbeReport:
    """Top-level probe report / artifact."""

    schema_version: str
    probe_version: str
    timestamp: str
    probe_sha256: str
    bank_sha256: str
    bank_path: str
    system_prompt_sha256: str
    tool_schema_sha256: str
    form_hashes: list[str]
    forms: list[dict[str, Any]]
    config: dict[str, Any]
    omp: dict[str, Any]
    python: dict[str, Any]
    schedule: list[dict[str, Any]]
    trials: list[dict[str, Any]]
    progress: dict[str, int]
    model_results: list[dict[str, Any]]
    paired_winloss: dict[str, Any]
    bootstrap_intervals: dict[str, Any]
    infra_probe_error: bool
    ranking: list[dict[str, Any]]
    ranking_omitted: bool
    limitations: list[str]


# ─── session execution ────────────────────────────────────────────────


def _send_host_tool_result(
    session: RpcSession, call_id: str, text: str, is_error: bool
) -> None:
    session.send(
        {
            "type": "host_tool_result",
            "id": call_id,
            "result": {"content": [{"type": "text", "text": text}]},
            "isError": is_error,
        }
    )


def _wait_for_ready(
    session: RpcSession, deadline: float, trial: dict[str, Any]
) -> bool:
    """Wait for the ``ready`` frame from the RPC process."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        line = session.recv(min(remaining, 10))
        if line is None:
            if session.proc.poll() is not None:
                return False
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            trial["non_json_rpc_lines"].append(line[:500])
            return False
        if frame.get("type") == "ready":
            return True
        if frame.get("type") not in _IGNORE_TYPES:
            trial["diagnostic_events"].append(_normalize_event(frame))


def _send_and_wait(
    session: RpcSession,
    command: dict[str, Any],
    deadline: float,
    trial: dict[str, Any],
) -> dict[str, Any] | None:
    """Send a setup command and wait for its response."""
    session.send(command)
    cmd_id = command.get("id")
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        line = session.recv(min(remaining, 10))
        if line is None:
            if session.proc.poll() is not None:
                return None
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            trial["non_json_rpc_lines"].append(line[:500])
            return {"success": False, "__protocol_error": True}
        ftype = frame.get("type")
        if ftype == "response" and frame.get("id") == cmd_id:
            return frame
        if ftype in _DIAGNOSTIC_TYPES:
            trial["diagnostic_events"].append(_normalize_event(frame))
            if ftype == "auto_retry_start":
                trial["auto_retry_events"] += 1
            elif ftype == "auto_compaction_start":
                trial["auto_compaction_events"] += 1
        elif ftype == "host_tool_call":
            _send_host_tool_result(
                session, frame.get("id", ""), "Unexpected call during setup.", True
            )


# Failure categories that signify infrastructure errors.
_PROBE_ERROR_CATEGORIES = frozenset(
    {"rpc_protocol_process_error", "setup_failed", "prompt_rejected"}
)


def run_session(
    model: str,
    form: dict[str, Any],
    form_index: int,
    seed: int,
    max_format_attempts: int,
    trial_timeout: int,
    omp_path: Path,
    system_prompt: str,
    tool_definition: dict[str, Any],
) -> TrialResult:
    """Run a single isolated (model, form) session and return the trial record."""

    trial_id = f"riddle-{seed}-{form_index}-{model.replace('/', '--')}"

    prompt_text = render_prompt(form, trial_id)

    temp_dir = tempfile.mkdtemp(prefix="omp-riddle-probe-")
    argv = [
        str(omp_path),
        "--mode", "rpc",
        "--model", model,
        "--no-session",
        "--no-title",
        "--no-tools",
        "--no-lsp",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
        "--system-prompt", system_prompt,
        "--cwd", temp_dir,
    ]
    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    # Multi-riddle reasoning legitimately reuses vocabulary across explanations.
    # Disable OMP's DeepSeek/Gemini lexical-loop heuristic to prevent false
    # transport failures; every model receives the same setting.
    env["PI_NO_THINKING_LOOP_GUARD"] = "1"

    trial: dict[str, Any] = {
        "trial_id": trial_id,
        "model": model,
        "form_index": form_index,
        "probe_error": False,
        "failure_category": None,
        "accepted": False,
        "submission": None,
        "raw_tool_args": None,
        "effective_tool_args": None,
        "omp_argument_repair": False,
        "format_attempts": 0,
        "format_errors": [],
        "score": None,
        "ttft": None,
        "duration": None,
        "usage": {},
        "wall_time_seconds": 0.0,
        "turn_count": 0,
        "message_count": 0,
        "auto_retry_events": 0,
        "auto_compaction_events": 0,
        "diagnostic_events": [],
        "stderr_tail": [],
        "non_json_rpc_lines": [],
        "final_assistant_text": "",
        "process_exit_code": None,
        "omp_model_resolved": None,
        "baseline_context_usage": None,
        "final_context_usage": None,
        "messages": [],
        "attempts": [],
    }

    # Tracking structures for the event loop.
    host_tool_calls: dict[str, dict[str, Any]] = {}
    tool_execution_ends: dict[str, dict[str, Any]] = {}
    message_tool_calls: list[tuple[int, str, str, Any]] = []
    emitted_tool_call_ids: set[str] = set()
    accepted = False
    accepted_tool_call_id: str | None = None
    accepted_submission: list[dict[str, Any]] | None = None
    accepted_raw_args: Any = None
    accepted_effective_args: Any = None
    agent_ended = False
    prompt_acked = False
    failure_reason: str | None = None
    format_attempts_count = 0

    start = time.monotonic()
    deadline = start + trial_timeout

    try:
        session = RpcSession(argv, env)
    except OSError as exc:
        trial["probe_error"] = True
        trial["failure_category"] = "rpc_protocol_process_error"
        trial["diagnostic_events"].append(
            {"type": "probe_exception", "message": str(exc)[:200]}
        )
        trial["wall_time_seconds"] = round(time.monotonic() - start, 3)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return _trial_dict_to_result(trial)

    try:
        # ── wait for ready ──
        if not _wait_for_ready(session, deadline, trial):
            trial["probe_error"] = True
            failure_reason = "rpc_protocol_process_error"
            trial["failure_category"] = failure_reason
            return _trial_dict_to_result(trial)

        # ── setup phase ──
        setup_cmds = [
            {"type": "set_host_tools", "id": "cmd-1", "tools": [tool_definition]},
            {"type": "set_auto_retry", "id": "cmd-2", "enabled": False},
            {"type": "set_auto_compaction", "id": "cmd-3", "enabled": False},
            {"type": "get_state", "id": "cmd-4"},
        ]
        for cmd in setup_cmds:
            resp = _send_and_wait(session, cmd, deadline, trial)
            if resp is not None and resp.get("__protocol_error"):
                failure_reason = "rpc_protocol_process_error"
            elif resp is None or not resp.get("success"):
                failure_reason = "setup_failed"
            else:
                if cmd["type"] == "get_state":
                    state_data = resp.get("data", {})
                    if isinstance(state_data, dict):
                        trial["omp_model_resolved"] = _public_model_metadata(
                            state_data.get("model")
                        )
                        trial["baseline_context_usage"] = state_data.get("contextUsage")
                continue
            trial["probe_error"] = True
            trial["failure_category"] = failure_reason
            return _trial_dict_to_result(trial)

        # ── send prompt ──
        session.send({"type": "prompt", "id": "cmd-5", "message": prompt_text})

        # ── main event loop ──
        while not (agent_ended and prompt_acked):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure_reason = "timeout"
                try:
                    session.send({"type": "abort", "id": "cmd-abort"})
                except Exception:
                    pass
                break

            line = session.recv(min(remaining, 10))
            if line is None:
                if session.proc.poll() is not None:
                    failure_reason = "rpc_protocol_process_error"
                    break
                continue

            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                trial["non_json_rpc_lines"].append(line[:500])
                failure_reason = "rpc_protocol_process_error"
                break

            ftype = frame.get("type")

            # ── prompt response ──
            if ftype == "response" and frame.get("id") == "cmd-5":
                prompt_acked = True
                if not frame.get("success"):
                    failure_reason = "prompt_rejected"
                    break
                continue

            # ── agent_end ──
            if ftype == "agent_end":
                agent_ended = True
                msgs = frame.get("messages", [])
                if isinstance(msgs, list):
                    trial["message_count"] = len(msgs)
                    for msg in reversed(msgs):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            for block in msg.get("content", []):
                                if isinstance(block, dict) and block.get("type") == "text":
                                    trial["final_assistant_text"] = str(block.get("text", ""))
                                    break
                            if trial["final_assistant_text"]:
                                break
                continue

            # ── turn_start ──
            if ftype == "turn_start":
                trial["turn_count"] += 1
                continue

            # ── message_end ──
            if ftype == "message_end":
                msg = frame.get("message", {})
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    msg_idx = len(trial["messages"])
                    trial["messages"].append(_capture_message(msg, msg_idx))
                    # Capture TTFT/duration from first assistant message.
                    if trial["ttft"] is None and msg.get("ttft") is not None:
                        trial["ttft"] = msg.get("ttft")
                    if trial["duration"] is None and msg.get("duration") is not None:
                        trial["duration"] = msg.get("duration")
                    # Accumulate usage.
                    mu = msg.get("usage")
                    if isinstance(mu, dict):
                        for k in ("input", "output", "cacheRead", "cacheWrite", "totalTokens"):
                            v = mu.get(k)
                            if isinstance(v, (int, float)):
                                trial["usage"][k] = trial["usage"].get(k, 0) + v
                    for block in msg.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "toolCall":
                            message_tool_calls.append(
                                (
                                    msg_idx,
                                    str(block.get("id", "")),
                                    str(block.get("name", "")),
                                    block.get("arguments"),
                                )
                            )
                continue

            # ── tool_execution_end ──
            if ftype == "tool_execution_end":
                tcid = str(frame.get("toolCallId", ""))
                tool_execution_ends[tcid] = {
                    "isError": frame.get("isError", False),
                    "toolName": frame.get("toolName"),
                    "result": frame.get("result"),
                }
                if tcid not in emitted_tool_call_ids:
                    emitted_tool_call_ids.add(tcid)
                continue

            # ── host_tool_call: validate structure only, never correctness ──
            if ftype == "host_tool_call":
                tcid = str(frame.get("toolCallId", ""))
                htc_id = str(frame.get("id", ""))
                htc_name = str(frame.get("toolName", ""))
                htc_args = frame.get("arguments", {})

                host_tool_calls[tcid] = {
                    "id": htc_id,
                    "toolName": htc_name,
                    "arguments": htc_args,
                }

                if tcid not in emitted_tool_call_ids:
                    emitted_tool_call_ids.add(tcid)

                if accepted:
                    # Duplicate after accept — flag and ignore.
                    _send_host_tool_result(session, htc_id, DUPLICATE_TEXT, True)
                elif htc_name != TOOL_NAME:
                    _send_host_tool_result(
                        session, htc_id,
                        f"Unknown tool '{htc_name}'. The only available tool is {TOOL_NAME}.",
                        True,
                    )
                    format_attempts_count += 1
                    if format_attempts_count >= max_format_attempts:
                        failure_reason = "max_format_attempts_exceeded"
                        try:
                            session.send({"type": "abort", "id": "cmd-abort"})
                        except Exception:
                            pass
                else:
                    # Validate structure/coverage only.
                    valid, detail, answers_list = validate_tool_args(
                        htc_args, trial_id
                    )
                    if valid:
                        accepted = True
                        accepted_tool_call_id = tcid
                        accepted_submission = answers_list
                        accepted_raw_args = htc_args
                        accepted_effective_args = htc_args
                        _send_host_tool_result(session, htc_id, ACCEPTED_TEXT, False)
                    else:
                        format_attempts_count += 1
                        trial["format_errors"].append(detail)
                        retry_text = FORMAT_RETRY_INSTRUCTION.format(detail=detail)
                        _send_host_tool_result(session, htc_id, retry_text, True)
                        if format_attempts_count >= max_format_attempts:
                            failure_reason = "max_format_attempts_exceeded"
                            try:
                                session.send({"type": "abort", "id": "cmd-abort"})
                            except Exception:
                                pass
                continue

            # ── diagnostic events ──
            if ftype in _DIAGNOSTIC_TYPES:
                trial["diagnostic_events"].append(_normalize_event(frame))
                if ftype == "auto_retry_start":
                    trial["auto_retry_events"] += 1
                elif ftype == "auto_compaction_start":
                    trial["auto_compaction_events"] += 1
                continue

            # Ignore _IGNORE_TYPES, prompt_result, and unknown frames

        # ── prompt was never acked ──
        if not prompt_acked and failure_reason in (None, "timeout"):
            failure_reason = "prompt_rejected"

        # ── final state ──
        if agent_ended:
            final_state = _send_and_wait(
                session, {"type": "get_state", "id": "cmd-final"},
                min(deadline, time.monotonic() + 15), trial,
            )
            if final_state and final_state.get("success"):
                fd = final_state.get("data", {})
                if isinstance(fd, dict):
                    trial["final_context_usage"] = fd.get("contextUsage")
            elif failure_reason is None:
                failure_reason = "rpc_protocol_process_error"

        # ── build attempts from message_end toolCall order ──
        trial["attempts"] = _build_attempts(
            message_tool_calls, host_tool_calls, tool_execution_ends,
            accepted_tool_call_id,
        )
        trial["format_attempts"] = format_attempts_count

        # ── record accepted submission ──
        if accepted:
            trial["accepted"] = True
            trial["submission"] = accepted_submission
            trial["raw_tool_args"] = accepted_raw_args
            trial["effective_tool_args"] = accepted_effective_args
            # Score the submission (pure, never reveals correctness to model).
            trial["score"] = score_submission(accepted_submission or [], form)

        # ── detect terminal errors ──
        if failure_reason is None and not accepted:
            terminal = _detect_terminal_error(trial)
            if terminal:
                failure_reason = terminal
            elif trial["attempts"]:
                failure_reason = "tool_format_error"
            else:
                failure_reason = "missing_tool_call"

        trial["failure_category"] = failure_reason

        # ── set probe_error for infrastructure failures ──
        if failure_reason in _PROBE_ERROR_CATEGORIES:
            trial["probe_error"] = True

    except Exception as exc:
        trial["diagnostic_events"].append(
            {"type": "probe_exception", "message": str(exc)[:300]}
        )
        if not failure_reason:
            failure_reason = "rpc_protocol_process_error"
        trial["failure_category"] = failure_reason
        trial["probe_error"] = True

    finally:
        trial["wall_time_seconds"] = round(time.monotonic() - start, 3)
        trial["process_exit_code"] = session.close()
        trial["stderr_tail"] = session.drain_stderr()[-STDERR_TAIL_LINES:]
        shutil.rmtree(temp_dir, ignore_errors=True)

    return _trial_dict_to_result(trial)


def _trial_dict_to_result(trial: dict[str, Any]) -> TrialResult:
    """Convert internal trial dict to TrialResult dataclass."""
    return TrialResult(
        trial_id=trial["trial_id"],
        model=trial["model"],
        form_index=trial["form_index"],
        accepted=trial["accepted"],
        submission=trial["submission"],
        raw_tool_args=trial["raw_tool_args"],
        effective_tool_args=trial["effective_tool_args"],
        omp_argument_repair=trial["omp_argument_repair"],
        format_attempts=trial["format_attempts"],
        format_errors=trial["format_errors"],
        score=trial["score"],
        ttft=trial["ttft"],
        duration=trial["duration"],
        usage=trial["usage"],
        wall_time_seconds=trial["wall_time_seconds"],
        turn_count=trial["turn_count"],
        message_count=trial["message_count"],
        auto_retry_events=trial["auto_retry_events"],
        auto_compaction_events=trial["auto_compaction_events"],
        diagnostic_events=trial["diagnostic_events"],
        stderr_tail=trial["stderr_tail"],
        non_json_rpc_lines=trial["non_json_rpc_lines"],
        final_assistant_text=trial["final_assistant_text"],
        process_exit_code=trial["process_exit_code"],
        omp_model_resolved=trial["omp_model_resolved"],
        probe_error=trial["probe_error"],
        failure_category=trial["failure_category"],
        baseline_context_usage=trial["baseline_context_usage"],
        final_context_usage=trial["final_context_usage"],
        messages=trial["messages"],
        attempts=trial["attempts"],
    )


# ─── attempt building and classification ───────────────────────────────


def _classify_single(
    name: str,
    raw_args: Any,
    host_emitted: bool,
    te: dict[str, Any] | None,
    is_accepted: bool,
    is_duplicate: bool,
) -> str:
    """Classify a single tool-call attempt."""
    if is_accepted:
        return "accepted"
    if is_duplicate:
        return "duplicate_after_accept"
    if name != TOOL_NAME:
        return "unknown_tool_name"
    if isinstance(raw_args, dict) and "__parseError" in raw_args:
        return "malformed_json"
    if not host_emitted and te is not None:
        err_text = _extract_error_text(te.get("result")).lower()
        if (
            "not valid json" in err_text
            or "parse error" in err_text
            or "__parseerror" in err_text
            or "failed to parse" in err_text
            or "json parse" in err_text
        ):
            return "malformed_json"
        if (
            "validation" in err_text
            or "schema" in err_text
            or "required" in err_text
            or "additional properties" in err_text
            or "minitems" in err_text
            or "maxitems" in err_text
            or "enum" in err_text
            or "invalid arguments" in err_text
            or "must have" in err_text
            or "must be" in err_text
            or "expected" in err_text
        ):
            return "argument_schema_failure"
        return "tool_execution_error"
    if host_emitted:
        return "format_error"
    return "tool_execution_error"


def _build_attempts(
    message_tool_calls: list[tuple[int, str, str, Any]],
    host_tool_calls: dict[str, dict[str, Any]],
    tool_execution_ends: dict[str, dict[str, Any]],
    accepted_tool_call_id: str | None,
) -> list[dict[str, Any]]:
    """Build attempts strictly from assistant message_end toolCall order,
    enriched by toolCallId with host/execution events."""
    attempts: list[dict[str, Any]] = []
    accepted_seen = False
    for msg_idx, tcid, name, raw_args in message_tool_calls:
        htc = host_tool_calls.get(tcid)
        te = tool_execution_ends.get(tcid)
        host_emitted = htc is not None
        effective_args = htc["arguments"] if htc else raw_args

        # Detect OMP argument repair.
        omp_repair = False
        if host_emitted:
            omp_repair = _detect_repair(raw_args, effective_args)

        is_accepted = tcid == accepted_tool_call_id
        if is_accepted:
            accepted_seen = True
        is_duplicate = accepted_seen and not is_accepted and host_emitted

        category = _classify_single(
            name, raw_args, host_emitted, te, is_accepted, is_duplicate,
        )

        attempts.append(
            {
                "index": len(attempts),
                "tool_call_id": tcid,
                "tool_name": name,
                "category": category,
                "raw_arguments": raw_args,
                "effective_arguments": effective_args,
                "omp_argument_repair": omp_repair,
                "host_tool_call_emitted": host_emitted,
                "host_tool_call_id": htc["id"] if htc else None,
                "message_index": msg_idx,
            }
        )
    return attempts


def _detect_terminal_error(trial: dict[str, Any]) -> str | None:
    """Detect transport_model_error or context_overflow from assistant messages."""
    for msg in trial.get("messages", []):
        err = msg.get("error")
        if err and err.get("message"):
            em = str(err["message"]).lower()
            if (
                "context" in em
                and ("length" in em or "exceed" in em or "too long" in em)
            ):
                return "context_overflow"
            return "transport_model_error"
        sr = msg.get("stop_reason")
        if sr == "error":
            return "transport_model_error"
        if sr == "length":
            return "context_overflow"
    return None


# ─── model result aggregation ─────────────────────────────────────────


def _compute_model_result(
    model: str,
    trials: list[TrialResult],
    forms: list[dict[str, Any]],
    bank: dict[str, Any],
) -> ModelResult:
    """Aggregate trial results into a ModelResult."""
    riddles = bank["riddles"]
    n_riddles = len(riddles)

    # Per-form scores.
    per_form_scores: list[dict[str, Any]] = []
    for trial in trials:
        if trial.score is not None:
            per_form_scores.append(
                {
                    "form_index": trial.form_index,
                    "correct_count": trial.score["correct_count"],
                    "accuracy": trial.score["accuracy"],
                    "guessing_corrected": trial.score["guessing_corrected"],
                    "structural_issues": trial.score.get("structural_issues", []),
                }
            )
        else:
            per_form_scores.append(
                {
                    "form_index": trial.form_index,
                    "correct_count": 0,
                    "accuracy": 0.0,
                    "guessing_corrected": 0.0,
                    "structural_issues": ["no_valid_submission"],
                }
            )

    # Per-riddle scores (0..5) indexed by bank riddle index.
    per_riddle_scores = [0] * n_riddles
    for trial in trials:
        if trial.score is None:
            continue
        form = forms[trial.form_index]
        riddle_order = form["riddle_order"]
        for display_idx, bank_idx in enumerate(riddle_order):
            per_riddle_scores[bank_idx] += trial.score["per_riddle"][display_idx]

    # Majority correct (>=3/5) and stable correct (5/5).
    majority_correct = sum(1 for s in per_riddle_scores if s >= 3)
    stable_correct = sum(1 for s in per_riddle_scores if s == 5)

    # Category scores.
    category_scores: dict[str, dict[str, Any]] = {}
    for i, riddle in enumerate(riddles):
        cat = riddle["category"]
        if cat not in category_scores:
            category_scores[cat] = {"correct": 0, "total": 0, "per_riddle": []}
        category_scores[cat]["correct"] += per_riddle_scores[i]
        category_scores[cat]["total"] += len(trials)
        category_scores[cat]["per_riddle"].append(per_riddle_scores[i])
    for cat in category_scores:
        total = category_scores[cat]["total"]
        category_scores[cat]["accuracy"] = (
            round(category_scores[cat]["correct"] / total, 6) if total > 0 else 0.0
        )

    # Answer-position scores (A, B, C, D, E).
    answer_position_scores: dict[str, dict[str, Any]] = {}
    for letter in DISPLAY_LETTERS:
        answer_position_scores[letter] = {"correct": 0, "total": 0}

    for trial in trials:
        if trial.score is None or trial.submission is None:
            continue
        form = forms[trial.form_index]
        for rec in trial.submission:
            num = rec.get("number")
            choice = rec.get("choice", "")
            if not isinstance(num, int) or num < 1 or num > n_riddles:
                continue
            display_idx = num - 1
            correct_letter = form["correct_letters"][display_idx]
            # Track per answer-position: how often the correct answer is at
            # this letter, and how often the model chose correctly when it was.
            answer_position_scores[correct_letter]["total"] += 1
            if choice.strip().upper() == correct_letter:
                answer_position_scores[correct_letter]["correct"] += 1

    for letter in DISPLAY_LETTERS:
        total = answer_position_scores[letter]["total"]
        answer_position_scores[letter]["accuracy"] = (
            round(answer_position_scores[letter]["correct"] / total, 6)
            if total > 0
            else 0.0
        )

    # Tool format failures and repairs.
    tool_format_failures = sum(1 for t in trials if not t.accepted)
    tool_format_repairs = sum(
        1 for t in trials
        for a in t.attempts
        if a.get("omp_argument_repair")
    )

    # Primary accuracy: exact displayed-choice accuracy over all decisions.
    total_correct = sum(
        t.score["correct_count"] if t.score else 0 for t in trials
    )
    total_decisions = len(trials) * RIDDLES_PER_FORM
    primary_accuracy = (
        round(total_correct / total_decisions, 6) if total_decisions > 0 else 0.0
    )
    gc = guessing_corrected_score(primary_accuracy)

    # Timing.
    wall_vals = [t.wall_time_seconds for t in trials]
    mean_wall = round(sum(wall_vals) / len(wall_vals), 3) if wall_vals else 0.0
    ttft_vals = [t.ttft for t in trials if t.ttft is not None]
    mean_ttft = round(sum(ttft_vals) / len(ttft_vals), 3) if ttft_vals else None

    # Total usage.
    total_usage: dict[str, Any] = {}
    for t in trials:
        for k, v in t.usage.items():
            if isinstance(v, (int, float)):
                total_usage[k] = total_usage.get(k, 0) + v

    return ModelResult(
        model=model,
        trials=trials,
        per_form_scores=per_form_scores,
        per_riddle_scores=per_riddle_scores,
        majority_correct=majority_correct,
        stable_correct=stable_correct,
        category_scores=category_scores,
        answer_position_scores=answer_position_scores,
        tool_format_failures=tool_format_failures,
        tool_format_repairs=tool_format_repairs,
        primary_accuracy=primary_accuracy,
        guessing_corrected=gc,
        mean_wall_time_seconds=mean_wall,
        mean_ttft=mean_ttft,
        total_usage=total_usage,
    )


# ─── paired win/loss/tie and bootstrap ────────────────────────────────


def _paired_winloss(
    model_a: str,
    model_b: str,
    trials_a: list[TrialResult],
    trials_b: list[TrialResult],
    forms: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute paired per-riddle win/loss/tie between two models.

    Pairs by form_index, then compares per-riddle correctness.
    """
    by_form_a = {t.form_index: t for t in trials_a}
    by_form_b = {t.form_index: t for t in trials_b}

    common_forms = sorted(set(by_form_a) & set(by_form_b))
    wins = 0
    losses = 0
    ties = 0
    per_riddle_diff: list[int] = []  # per bank riddle: a_correct - b_correct

    n_riddles = RIDDLES_PER_FORM
    a_riddle_correct = [0] * n_riddles
    b_riddle_correct = [0] * n_riddles

    for fi in common_forms:
        ta = by_form_a[fi]
        tb = by_form_b[fi]
        form = forms[fi]
        for display_idx in range(n_riddles):
            # A missing/invalid submission is wrong for every displayed
            # decision, matching the primary-score contract.
            a_correct = (
                ta.score["per_riddle"][display_idx] if ta.score is not None else 0
            )
            b_correct = (
                tb.score["per_riddle"][display_idx] if tb.score is not None else 0
            )
            bank_idx = form["riddle_order"][display_idx]
            a_riddle_correct[bank_idx] += a_correct
            b_riddle_correct[bank_idx] += b_correct

    for i in range(n_riddles):
        diff = a_riddle_correct[i] - b_riddle_correct[i]
        per_riddle_diff.append(diff)
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            ties += 1

    return {
        "model_a": model_a,
        "model_b": model_b,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "per_riddle_diff": per_riddle_diff,
        "common_forms": len(common_forms),
    }


def _cluster_bootstrap_ci(
    per_riddle_diff: list[float],
    n_bootstrap: int = 10000,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Deterministic cluster-bootstrap CI for mean accuracy difference.

    Resamples riddle-level clusters with replacement.
    """
    n = len(per_riddle_diff)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_bootstrap": 0}

    rng = random.Random(seed)
    point_estimate = sum(per_riddle_diff) / n

    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        sample_sum = 0
        for _ in range(n):
            idx = rng.randint(0, n - 1)
            sample_sum += per_riddle_diff[idx]
        boot_means.append(sample_sum / n)

    boot_means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(alpha * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1.0 - alpha) * n_bootstrap))

    return {
        "mean_diff": round(point_estimate, 6),
        "ci_lower": round(boot_means[lo_idx], 6),
        "ci_upper": round(boot_means[hi_idx], 6),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


def _compute_paired_results(
    model_results: list[ModelResult],
    forms: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute all pairwise win/loss/tie and bootstrap intervals."""
    paired: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}

    for i, ma in enumerate(model_results):
        for mb in model_results[i + 1:]:
            key = f"{ma.model}__vs__{mb.model}"
            wl = _paired_winloss(
                ma.model, mb.model, ma.trials, mb.trials, forms
            )
            paired[key] = wl
            common_forms = wl["common_forms"]
            normalized_diff = (
                [diff / common_forms for diff in wl["per_riddle_diff"]]
                if common_forms
                else []
            )
            ci = _cluster_bootstrap_ci(
                normalized_diff, seed=seed + i * 100 + len(model_results)
            )
            bootstrap[key] = ci

    return paired, bootstrap


# ─── ranking ──────────────────────────────────────────────────────────


def _compute_ranking(
    model_results: list[ModelResult],
) -> tuple[list[dict[str, Any]], bool]:
    """Compute deterministic ranking. Omit if schedules are unequal/incomplete."""
    if not model_results:
        return [], True

    # All models must have the same number of trials.
    counts = [len(mr.trials) for mr in model_results]
    if len(set(counts)) > 1 or all(c == 0 for c in counts):
        return [], True

    # No model should have probe_error trials.
    any_probe_error = any(
        t.probe_error for mr in model_results for t in mr.trials
    )
    if any_probe_error:
        return [], True

    ranking = sorted(
        model_results,
        key=lambda mr: (
            -mr.primary_accuracy,
            -mr.guessing_corrected,
            -mr.stable_correct,
            -mr.majority_correct,
            mr.tool_format_failures,
            mr.mean_wall_time_seconds,
            mr.model,
        ),
    )
    ranking_list = [
        {
            "model": mr.model,
            "primary_accuracy": mr.primary_accuracy,
            "guessing_corrected": mr.guessing_corrected,
            "stable_correct": mr.stable_correct,
            "majority_correct": mr.majority_correct,
            "tool_format_failures": mr.tool_format_failures,
            "mean_wall_time_seconds": mr.mean_wall_time_seconds,
        }
        for mr in ranking
    ]
    return ranking_list, False


# ─── artifact ─────────────────────────────────────────────────────────


def _build_schedule(
    models: list[str],
    n_forms: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build a schedule rotating model order per form."""
    schedule: list[dict[str, Any]] = []
    n = len(models)
    for form_idx in range(n_forms):
        offset = form_idx % n
        order = models[offset:] + models[:offset]
        for model in order:
            schedule.append(
                {
                    "model": model,
                    "form_index": form_idx,
                    "seed": seed,
                }
            )
    return schedule


def _model_result_to_dict(mr: ModelResult) -> dict[str, Any]:
    """Convert ModelResult to a serializable dict."""
    return {
        "model": mr.model,
        "trials": [asdict(t) for t in mr.trials],
        "per_form_scores": mr.per_form_scores,
        "per_riddle_scores": mr.per_riddle_scores,
        "majority_correct": mr.majority_correct,
        "stable_correct": mr.stable_correct,
        "category_scores": mr.category_scores,
        "answer_position_scores": mr.answer_position_scores,
        "tool_format_failures": mr.tool_format_failures,
        "tool_format_repairs": mr.tool_format_repairs,
        "primary_accuracy": mr.primary_accuracy,
        "guessing_corrected": mr.guessing_corrected,
        "mean_wall_time_seconds": mr.mean_wall_time_seconds,
        "mean_ttft": mr.mean_ttft,
        "total_usage": mr.total_usage,
    }


def build_artifact(
    config: dict[str, Any],
    omp_path: Path,
    omp_version: str,
    system_prompt: str,
    tool_definition: dict[str, Any],
    bank: dict[str, Any],
    bank_sha: str,
    bank_path: str,
    forms: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    trials: list[TrialResult],
    model_results: list[ModelResult],
    paired_winloss: dict[str, Any],
    bootstrap_intervals: dict[str, Any],
    infra_probe_error: bool,
) -> dict[str, Any]:
    """Assemble the full reproducibility artifact."""
    completed = len(trials)
    total = len(schedule)

    ranking, ranking_omitted = _compute_ranking(model_results)

    # Strip proofs from bank in artifact (proofs retained outside model prompt).
    bank_public = {
        "schema_version": bank.get("schema_version"),
        "benchmark_version": bank.get("benchmark_version"),
        "provenance": bank.get("provenance"),
        "exclusions": bank.get("exclusions"),
        "categories": bank.get("categories"),
        "riddles": [
            {
                "id": r["id"],
                "category": r["category"],
                "difficulty": r["difficulty"],
                "prompt": r["prompt"],
                "options": r["options"],
                "correct_option_id": r["correct_option_id"],
                # proof/ambiguity_audit/originality_note/verification retained
                # but not included in model-facing prompt; included here for
                # reproducibility audit.
                "proof": r["proof"],
                "ambiguity_audit": r["ambiguity_audit"],
                "originality_note": r["originality_note"],
                "verification": r["verification"],
            }
            for r in bank["riddles"]
        ],
    }

    form_hashes = [f["sha256"] for f in forms]
    # Forms for artifact: strip correct_letters (scoring secret) but keep
    # structure for reproducibility.
    forms_public = [
        {
            "form_index": f["form_index"],
            "riddle_order": f["riddle_order"],
            "option_permutations": f["option_permutations"],
            "sha256": f["sha256"],
        }
        for f in forms
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "probe_version": PROBE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "probe_sha256": _probe_file_sha256(),
        "bank": {
            "sha256": bank_sha,
            "path": bank_path,
            "schema_version": bank.get("schema_version"),
            "benchmark_version": bank.get("benchmark_version"),
            "content": bank_public,
        },
        "omp": {
            "path": str(omp_path),
            "version": omp_version,
            "rpc_mode": True,
        },
        "python": {
            "version": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "config": config,
        "system_prompt": {
            "text": system_prompt,
            "sha256": _sha256_hex(system_prompt),
        },
        "tool_schema": {
            "name": TOOL_NAME,
            "definition": tool_definition,
            "sha256": _sha256_hex(
                json.dumps(tool_definition, sort_keys=True, ensure_ascii=False)
            ),
        },
        "form_generator": {
            "version": FORM_GENERATOR_VERSION,
            "n_forms": len(forms),
            "form_hashes": form_hashes,
        },
        "forms": forms_public,
        "schedule": schedule,
        "trials": [asdict(t) for t in trials],
        "progress": {"completed": completed, "total": total},
        "model_results": [_model_result_to_dict(mr) for mr in model_results],
        "paired_winloss": paired_winloss,
        "bootstrap_intervals": bootstrap_intervals,
        "infra_probe_error": infra_probe_error,
        "ranking": ranking,
        "ranking_omitted": ranking_omitted,
        "limitations": [
            "The probe does not make live model calls during validation; "
            "results depend on model availability and API configuration at "
            "run time.",
            "Provider-side compute is not standardized; model reasoning "
            "parameters use OMP defaults.  Resolved model/config is recorded "
            "per session.",
            "Guessing-corrected score assumes uniform 5-option guessing (0.2); "
            "actual guessing behavior may differ.",
            "Cluster-bootstrap intervals resample riddle-level clusters; "
            "they capture riddle-sampling uncertainty but not model-side "
            "stochasticity across repeated runs.",
            "Ranking is omitted when models have unequal trial counts or any "
            "infrastructure probe error occurred.",
            "Tool argument schema validation is performed by OMP before host "
            "tool call emission; schema failures prevent host_tool_call frames "
            "and are classified from tool_execution_end error text.",
            "Silent provider-side repairs of malformed JSON are unobservable; "
            "only OMP's validation-layer rejection and host-argument diff are "
            "detectable.",
        ],
    }


def write_artifact(artifact: dict[str, Any], output_path: str) -> None:
    """Write the artifact as pretty JSON atomically (temp + os.replace)."""
    json_str = json.dumps(artifact, indent=2, ensure_ascii=False)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(p.parent), prefix=p.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json_str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── CLI ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible OMP RPC riddle-reasoning benchmark "
            "(Oczy Original Riddle Benchmark v1)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "models",
        nargs="+",
        metavar="MODEL",
        help="OMP model selectors (positional, e.g. devin/glm-5-2)",
    )
    parser.add_argument(
        "--forms",
        type=int,
        default=DEFAULT_FORMS,
        metavar="N",
        help=f"Number of answer-position forms (default: {DEFAULT_FORMS}, range 1..5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="N",
        help=f"Deterministic seed for form generation (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--max-format-attempts",
        type=int,
        default=DEFAULT_MAX_FORMAT_ATTEMPTS,
        metavar="N",
        help=f"Max tool format attempts before aborting (default: {DEFAULT_MAX_FORMAT_ATTEMPTS})",
    )
    parser.add_argument(
        "--trial-timeout-seconds",
        type=int,
        default=DEFAULT_TRIAL_TIMEOUT_SECONDS,
        metavar="N",
        help=f"Wall-clock timeout per session (default: {DEFAULT_TRIAL_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--omp-bin",
        default=DEFAULT_OMP_BIN,
        metavar="PATH",
        help=f"OMP binary name or path (default: {DEFAULT_OMP_BIN})",
    )
    parser.add_argument(
        "--bank",
        default=None,
        metavar="PATH",
        help=(
            "Path to frozen riddle bank JSON (default: sibling "
            "riddles_v1.json next to this script)"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output artifact JSON path (required)",
    )
    args = parser.parse_args(argv)

    # ── validation ──
    seen: set[str] = set()
    for m in args.models:
        if not m or not m.strip():
            parser.error("Model selectors must be non-empty strings")
        if m in seen:
            parser.error(f"Duplicate model selector: {m!r}")
        seen.add(m)

    if args.forms < 1 or args.forms > DEFAULT_FORMS:
        parser.error(f"--forms must be between 1 and {DEFAULT_FORMS}")

    if args.max_format_attempts <= 0:
        parser.error("--max-format-attempts must be a positive integer")

    if args.trial_timeout_seconds <= 0:
        parser.error("--trial-timeout-seconds must be a positive integer")

    return args


def _default_bank_path() -> Path:
    """Return the default bank path (sibling riddles_v1.json)."""
    return Path(__file__).resolve().parent / "riddles_v1.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    models = list(args.models)
    n_forms = args.forms
    seed = args.seed
    max_format_attempts = args.max_format_attempts
    trial_timeout = args.trial_timeout_seconds

    # ── resolve bank path ──
    bank_path = Path(args.bank) if args.bank else _default_bank_path()

    # ── load and validate bank ──
    try:
        bank, bank_sha = load_and_validate_bank(bank_path)
    except ProbeError as exc:
        print(f"probe error: {exc}", file=sys.stderr)
        return exc.exit_code

    # ── resolve OMP binary ──
    try:
        omp_path = resolve_omp_binary(args.omp_bin)
        omp_version = capture_omp_version(omp_path)
    except ProbeError as exc:
        print(f"probe error: {exc}", file=sys.stderr)
        return exc.exit_code

    # ── build constant artifacts ──
    system_prompt = build_system_prompt()
    tool_definition = build_tool_definition()
    forms = generate_forms(bank, n_forms=n_forms, seed=seed)
    schedule = _build_schedule(models, n_forms, seed)

    config = {
        "models": models,
        "forms": n_forms,
        "seed": seed,
        "max_format_attempts": max_format_attempts,
        "trial_timeout_seconds": trial_timeout,
        "omp_bin": args.omp_bin,
        "bank_path": str(bank_path),
    }

    # ── write initial artifact ──
    try:
        initial = build_artifact(
            config, omp_path, omp_version,
            system_prompt, tool_definition,
            bank, bank_sha, str(bank_path),
            forms, schedule, [], [], {}, {}, False,
        )
        write_artifact(initial, args.output)
    except Exception as exc:
        print(f"probe error: failed to write initial artifact: {exc}", file=sys.stderr)
        return EXIT_PROBE_ERROR

    # ── run sessions ──
    trials: list[TrialResult] = []
    total = len(schedule)
    probe_error_aborted = False
    infra_probe_error = False

    for i, spec in enumerate(schedule):
        model = spec["model"]
        form_idx = spec["form_index"]
        form = forms[form_idx]
        print(
            f"[{i + 1}/{total}] model={model} form={form_idx}",
            file=sys.stderr,
        )
        trial = run_session(
            model=model,
            form=form,
            form_index=form_idx,
            seed=seed,
            max_format_attempts=max_format_attempts,
            trial_timeout=trial_timeout,
            omp_path=omp_path,
            system_prompt=system_prompt,
            tool_definition=tool_definition,
        )
        trials.append(trial)

        # ── checkpoint atomically after every session ──
        try:
            # Compute partial model results for checkpoint.
            partial_model_results: list[ModelResult] = []
            for m in models:
                m_trials = [t for t in trials if t.model == m]
                partial_model_results.append(
                    _compute_model_result(m, m_trials, forms, bank)
                )
            paired, bootstrap = _compute_paired_results(
                partial_model_results, forms, seed
            )
            artifact = build_artifact(
                config, omp_path, omp_version,
                system_prompt, tool_definition,
                bank, bank_sha, str(bank_path),
                forms, schedule, trials,
                partial_model_results, paired, bootstrap,
                infra_probe_error,
            )
            write_artifact(artifact, args.output)
        except Exception as exc:
            print(
                f"probe error: failed to checkpoint after session {i + 1}: {exc}",
                file=sys.stderr,
            )
            return EXIT_PROBE_ERROR

        # ── abort on probe_error (infrastructure failure) ──
        if trial.probe_error:
            print(
                f"probe error: session {i + 1} infrastructure failure "
                f"({trial.failure_category}), aborting remaining schedule.",
                file=sys.stderr,
            )
            probe_error_aborted = True
            infra_probe_error = True
            break

    # ── final aggregation ──
    model_results: list[ModelResult] = []
    for m in models:
        m_trials = [t for t in trials if t.model == m]
        model_results.append(_compute_model_result(m, m_trials, forms, bank))

    paired, bootstrap = _compute_paired_results(model_results, forms, seed)

    final_artifact = build_artifact(
        config, omp_path, omp_version,
        system_prompt, tool_definition,
        bank, bank_sha, str(bank_path),
        forms, schedule, trials,
        model_results, paired, bootstrap,
        infra_probe_error,
    )
    try:
        write_artifact(final_artifact, args.output)
    except Exception as exc:
        print(f"probe error: failed to write final artifact: {exc}", file=sys.stderr)
        return EXIT_PROBE_ERROR

    # ── final summary to stderr ──
    ranking = final_artifact.get("ranking", [])
    if ranking:
        print(
            "\nRanking (by primary_accuracy, guessing_corrected):",
            file=sys.stderr,
        )
        for r in ranking:
            print(
                f"  {r['model']}: "
                f"acc={r['primary_accuracy']:.1%} "
                f"gc={r['guessing_corrected']:.1%} "
                f"stable={r['stable_correct']} "
                f"majority={r['majority_correct']} "
                f"failures={r['tool_format_failures']} "
                f"mean_time={r['mean_wall_time_seconds']:.1f}s",
                file=sys.stderr,
            )
    elif final_artifact.get("ranking_omitted"):
        print(
            "\nRanking omitted (unequal/incomplete schedule or infra error).",
            file=sys.stderr,
        )

    print(f"\nArtifact written to {args.output}", file=sys.stderr)

    if probe_error_aborted:
        return EXIT_PROBE_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
