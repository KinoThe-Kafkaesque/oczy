#!/usr/bin/env python3
"""
Reproducible OMP RPC tool-call reliability probe for long-context needle extraction.

Each trial launches an isolated OMP RPC process with a single strict host tool,
a deterministic seeded long synthetic context with embedded AUTHENTIC_RECORD
needles and DECOY_RECORD distractors, auto-retry/auto-compaction disabled, and
exact event capture.  The probe measures first-attempt validity, model
self-correction retries, malformed/schema/semantic failure classification, and
final success.

OMP model selectors are positional::

    python benchmarks/omp/tool_reliability_probe.py devin/swe-1-7 devin/glm-5-2 devin/kimi-k2-7 \\
        --context-words 50000 --trials-per-size 3 --output /tmp/tool-reliability.json

Exit codes:
    0  — every scheduled trial completed (model failures are measured results)
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
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

# ─── constants ────────────────────────────────────────────────────────

SCHEMA_VERSION = "oczy/omp-tool-reliability-probe/v1"
PROBE_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"

EXIT_OK = 0
EXIT_PROBE_ERROR = 2

TOOL_NAME = "submit_context_evidence"
AUTHENTIC_MARKER = "AUTHENTIC_RECORD"
GUARD_MARKER = "AUTHENTIC_GUARD"
DECOY_MARKER = "DECOY_RECORD"

DEFAULT_CONTEXT_WORDS = [50000]
DEFAULT_TRIALS_PER_SIZE = 3
DEFAULT_SEED = 20260712
DEFAULT_MAX_TOOL_ATTEMPTS = 4
DEFAULT_TRIAL_TIMEOUT_SECONDS = 300
DEFAULT_OMP_BIN = "omp"

WORDS_PER_LINE = 12
STDERR_TAIL_LINES = 50
RETRY_INSTRUCTION = (
    "The submitted evidence does not match the authentic records. "
    "Re-read the context for AUTHENTIC_RECORD entries (slots alpha, beta, gamma) "
    "and the AUTHENTIC_GUARD entry. Ensure exact case, correct order "
    "(alpha, beta, gamma), and that you are not submitting DECOY_RECORD values. "
    "Then call submit_context_evidence again."
)
ACCEPTED_TEXT = "ACCEPTED. Reply with exactly DONE."
DUPLICATE_TEXT = (
    "Evidence was already accepted. Stop calling submit_context_evidence "
    "and reply with exactly DONE."
)

# Pinned filler vocabulary (120 common English words).  The SHA-256 of this
# list (as a JSON array) is recorded in the artifact for reproducibility.
FILLER_VOCAB = (
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "very", "through", "life", "child", "world", "school", "state", "family",
    "student", "group", "country", "problem", "hand", "part", "place", "case",
    "week", "company", "system", "program",
)

# ─── errors ───────────────────────────────────────────────────────────


class ProbeError(Exception):
    """Raised when the probe itself fails (not individual trial results)."""

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


def _vocab_sha256() -> str:
    """SHA-256 of the pinned filler vocabulary (JSON array encoding)."""
    return _sha256_bytes(
        json.dumps(list(FILLER_VOCAB), ensure_ascii=False).encode("utf-8")
    )


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


# ─── context generation ───────────────────────────────────────────────


def generate_context(
    seed: int, size: int, trial: int
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Generate a deterministic long context with embedded needles.

    Returns ``(context_text, expected_arguments, context_artifact)``.

    * ``context_text`` — the full prompt message sent to the model.
    * ``expected_arguments`` — the exact arguments the tool should receive.
    * ``context_artifact`` — metadata for the output artifact.
    """
    # ── filler words ──
    rng = random.Random(_seeded_int(str(seed), str(size), str(trial), "filler"))
    filler = [rng.choice(FILLER_VOCAB) for _ in range(size)]

    # ── needle keys / values / guard ──
    trial_id = _sha256_hex(f"{seed}:{size}:{trial}:trial_id")[:12]

    slots = ("alpha", "beta", "gamma")
    records_expected = []
    needle_lines: list[tuple[int, str]] = []

    for slot in slots:
        key = _sha256_hex(f"{seed}:{size}:{trial}:{slot}:key")[:20]
        value = _sha256_hex(f"{seed}:{size}:{trial}:{slot}:value")[:32]
        records_expected.append({"slot": slot, "key": key, "value": value})

    guard_value = _sha256_hex(f"{seed}:{size}:{trial}:guard")[:28]

    # ── positions (word indices into the filler list) ──
    alpha_pos = int(size * 0.15)
    beta_pos = int(size * 0.50)
    guard_pos = int(size * 0.70)
    gamma_pos = int(size * 0.85)

    needle_lines.append(
        (alpha_pos, f"AUTHENTIC_RECORD slot=alpha key={records_expected[0]['key']} value={records_expected[0]['value']}")
    )
    needle_lines.append(
        (beta_pos, f"AUTHENTIC_RECORD slot=beta key={records_expected[1]['key']} value={records_expected[1]['value']}")
    )
    needle_lines.append(
        (guard_pos, f"AUTHENTIC_GUARD token={guard_value}")
    )
    needle_lines.append(
        (gamma_pos, f"AUTHENTIC_RECORD slot=gamma key={records_expected[2]['key']} value={records_expected[2]['value']}")
    )

    # ── decoys ──
    decoy_rng = random.Random(
        _seeded_int(str(seed), str(size), str(trial), "decoy")
    )
    decoy_count = 6
    authentic_positions = {alpha_pos, beta_pos, guard_pos, gamma_pos}
    min_gap = max(size // 20, 50)
    decoy_entries: list[tuple[int, str]] = []
    for i in range(decoy_count):
        for _attempt in range(200):
            pos = decoy_rng.randint(0, size - 1)
            if all(abs(pos - ap) > min_gap for ap in authentic_positions):
                break
        slot = decoy_rng.choice(slots)
        dk = _sha256_hex(f"{seed}:{size}:{trial}:decoy:{i}:key")[:20]
        dv = _sha256_hex(f"{seed}:{size}:{trial}:decoy:{i}:value")[:32]
        decoy_entries.append(
            (pos, f"DECOY_RECORD slot={slot} key={dk} value={dv}")
        )

    # ── assemble lines ──
    insertions = sorted(needle_lines + decoy_entries, key=lambda x: x[0])
    lines: list[str] = []
    prev = 0
    for pos, text in insertions:
        segment = filler[prev:pos]
        for i in range(0, len(segment), WORDS_PER_LINE):
            lines.append(" ".join(segment[i : i + WORDS_PER_LINE]))
        lines.append(text)
        prev = pos
    # remaining filler
    segment = filler[prev:]
    for i in range(0, len(segment), WORDS_PER_LINE):
        lines.append(" ".join(segment[i : i + WORDS_PER_LINE]))

    # ── trial-id footer ──
    lines.append("")
    lines.append(f"TRIAL_ID: {trial_id}")
    lines.append("")
    lines.append(
        f"Extract the three AUTHENTIC_RECORD entries (alpha, beta, gamma) "
        f"and the AUTHENTIC_GUARD from the context above, then call "
        f"submit_context_evidence with trial_id={trial_id}."
    )

    context_text = "\n".join(lines)

    expected = {
        "trial_id": trial_id,
        "records": records_expected,
        "guard": guard_value,
        "finalize": True,
    }

    actual_words = len(context_text.split())
    artifact = {
        "requested_word_count": size,
        "actual_word_count": actual_words,
        "actual_char_count": len(context_text),
        "line_count": len(lines),
        "needle_positions": {
            "alpha": alpha_pos,
            "beta": beta_pos,
            "guard": guard_pos,
            "gamma": gamma_pos,
        },
        "decoy_count": decoy_count,
        "context_sha256": _sha256_hex(context_text),
        "trial_id": trial_id,
        "expected_arguments": expected,
    }
    return context_text, expected, artifact


# ─── tool schema / system prompt ──────────────────────────────────────


def build_tool_schema() -> dict[str, Any]:
    """JSON Schema for the submit_context_evidence host tool parameters."""
    return {
        "type": "object",
        "properties": {
            "trial_id": {"type": "string"},
            "records": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "slot": {
                            "type": "string",
                            "enum": ["alpha", "beta", "gamma"],
                        },
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["slot", "key", "value"],
                    "additionalProperties": False,
                },
            },
            "guard": {"type": "string"},
            "finalize": {"type": "boolean"},
        },
        "required": ["trial_id", "records", "guard", "finalize"],
        "additionalProperties": False,
    }


def build_tool_definition() -> dict[str, Any]:
    """RPC host tool definition for set_host_tools."""
    return {
        "name": TOOL_NAME,
        "description": (
            "Submit the authentic context evidence extracted from the long "
            "context. Provide the trial_id, an array of exactly three records "
            "in order [alpha, beta, gamma] with their slot, key, and value "
            "from the corresponding AUTHENTIC_RECORD entries, the guard token "
            "from the AUTHENTIC_GUARD entry, and finalize=true. All keys and "
            "values are case-sensitive. Ignore DECOY_RECORD entries."
        ),
        "parameters": build_tool_schema(),
    }


def build_system_prompt() -> str:
    """Constant system prompt (no authentic values)."""
    return (
        "You are a context-evidence extraction agent. Read the provided long "
        "context and extract specific marked records.\n\n"
        "Rules:\n"
        "1. Find exactly three AUTHENTIC_RECORD entries (slots: alpha, beta, "
        "gamma) and one AUTHENTIC_GUARD entry.\n"
        "2. Ignore DECOY_RECORD entries — they are not authentic.\n"
        "3. Call the submit_context_evidence tool exactly once with:\n"
        "   - trial_id: the trial identifier provided in the context\n"
        "   - records: three objects in order [alpha, beta, gamma], each with "
        "slot, key, and value from the corresponding AUTHENTIC_RECORD\n"
        "   - guard: the token from the AUTHENTIC_GUARD entry\n"
        "   - finalize: true\n"
        "4. All keys and values are case-sensitive — preserve exact case.\n"
        "5. If you receive a tool error, re-read the context and retry with "
        "corrected arguments.\n"
        "6. Do not call any other tool.\n"
        "7. After successful submission, reply with exactly: DONE"
    )


# ─── argument matching ────────────────────────────────────────────────


def arguments_match(args: Any, expected: dict[str, Any]) -> bool:
    """Exact semantic equality check for tool arguments."""
    if not isinstance(args, dict):
        return False
    if args.get("trial_id") != expected["trial_id"]:
        return False
    records = args.get("records")
    if not isinstance(records, list) or len(records) != 3:
        return False
    for i, rec in enumerate(records):
        exp = expected["records"][i]
        if not isinstance(rec, dict):
            return False
        if rec.get("slot") != exp["slot"]:
            return False
        if rec.get("key") != exp["key"]:
            return False
        if rec.get("value") != exp["value"]:
            return False
    if args.get("guard") != expected["guard"]:
        return False
    if args.get("finalize") is not True:
        return False
    return True


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
        # Join reader threads so no lines are lost.
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


# ─── trial execution ──────────────────────────────────────────────────


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
    """Send a setup command and wait for its response.

    Records diagnostic events encountered while waiting.  Handles stray
    host_tool_call frames by returning an error result.
    """
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


def run_trial(
    model: str,
    context_words: int,
    trial_index: int,
    size_index: int,
    seed: int,
    max_tool_attempts: int,
    trial_timeout: int,
    omp_path: Path,
    system_prompt: str,
    tool_definition: dict[str, Any],
) -> dict[str, Any]:
    """Run a single isolated trial and return the trial record."""

    context_text, expected, context_artifact = generate_context(
        seed, context_words, trial_index
    )

    temp_dir = tempfile.mkdtemp(prefix="omp-tool-probe-")
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

    trial: dict[str, Any] = {
        "model": model,
        "context_words": context_words,
        "trial_index": trial_index,
        "size_index": size_index,
        "context_seed": seed,
        "context": context_artifact,
        "probe_error": False,
        "outcome": {},
        "attempts": [],
        "messages": [],
        "turn_count": 0,
        "message_count": 0,
        "auto_retry_events": 0,
        "auto_compaction_events": 0,
        "diagnostic_events": [],
        "baseline_context_usage": None,
        "final_context_usage": None,
        "wall_time_seconds": 0.0,
        "stderr_tail": [],
        "non_json_rpc_lines": [],
        "final_assistant_text": "",
        "process_exit_code": None,
        "omp_model_resolved": None,
    }

    # Tracking structures for the event loop
    host_tool_calls: dict[str, dict[str, Any]] = {}
    tool_execution_ends: dict[str, dict[str, Any]] = {}
    message_tool_calls: list[tuple[int, str, str, Any]] = []
    emitted_tool_call_ids: set[str] = set()
    accepted = False
    accepted_tool_call_id: str | None = None
    agent_ended = False
    prompt_acked = False
    failure_reason: str | None = None

    start = time.monotonic()
    deadline = start + trial_timeout

    try:
        session = RpcSession(argv, env)
    except OSError as exc:
        trial["probe_error"] = True
        trial["outcome"] = _compute_outcome(
            trial, False, None, False, "rpc_protocol_process_error",
        )
        trial["diagnostic_events"].append(
            {"type": "probe_exception", "message": str(exc)[:200]}
        )
        trial["wall_time_seconds"] = round(time.monotonic() - start, 3)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return trial

    try:
        # ── wait for ready ──
        if not _wait_for_ready(session, deadline, trial):
            trial["probe_error"] = True
            failure_reason = "rpc_protocol_process_error"
            trial["outcome"] = _compute_outcome(
                trial, False, None, False, failure_reason,
            )
            return trial

        # ── setup phase: all four commands must succeed ──
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
            trial["outcome"] = _compute_outcome(
                trial, False, None, False, failure_reason,
            )
            return trial

        # ── send prompt (no _send_and_wait — events may arrive before ack) ──
        session.send({"type": "prompt", "id": "cmd-5", "message": context_text})

        # ── main event loop: process prompt ack and all agent frames ──
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

            # ── prompt response (may arrive after events) ──
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

            # ── message_end: collect tool calls in message order ──
            if ftype == "message_end":
                msg = frame.get("message", {})
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    msg_idx = len(trial["messages"])
                    trial["messages"].append(_capture_message(msg, msg_idx))
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

            # ── tool_execution_end: record for later enrichment ──
            if ftype == "tool_execution_end":
                tcid = str(frame.get("toolCallId", ""))
                tool_execution_ends[tcid] = {
                    "isError": frame.get("isError", False),
                    "toolName": frame.get("toolName"),
                    "result": frame.get("result"),
                }
                if tcid not in emitted_tool_call_ids:
                    emitted_tool_call_ids.add(tcid)
                    if (
                        len(emitted_tool_call_ids) >= max_tool_attempts
                        and not accepted
                    ):
                        failure_reason = "max_tool_attempts_exceeded"
                        try:
                            session.send({"type": "abort", "id": "cmd-abort"})
                        except Exception:
                            pass
                continue

            # ── host_tool_call: send result, track for enrichment ──
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
                    _send_host_tool_result(session, htc_id, DUPLICATE_TEXT, True)
                elif htc_name != TOOL_NAME:
                    _send_host_tool_result(
                        session, htc_id,
                        f"Unknown tool '{htc_name}'. The only available tool is {TOOL_NAME}.",
                        True,
                    )
                elif arguments_match(htc_args, expected):
                    accepted = True
                    accepted_tool_call_id = tcid
                    _send_host_tool_result(session, htc_id, ACCEPTED_TEXT, False)
                else:
                    _send_host_tool_result(session, htc_id, RETRY_INSTRUCTION, True)

                if (
                    len(emitted_tool_call_ids) >= max_tool_attempts
                    and not accepted
                ):
                    failure_reason = "max_tool_attempts_exceeded"
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

        # ── final state (only if agent ended normally) ──
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
            expected, accepted_tool_call_id,
        )

        # ── compute outcome ──
        trial["outcome"] = _compute_outcome(
            trial, accepted, accepted_tool_call_id,
            agent_ended, failure_reason,
        )

        # ── set probe_error for infrastructure failures ──
        fc = trial["outcome"].get("failure_category")
        if fc in _PROBE_ERROR_CATEGORIES:
            trial["probe_error"] = True

    except Exception as exc:
        trial["diagnostic_events"].append(
            {"type": "probe_exception", "message": str(exc)[:300]}
        )
        if not failure_reason:
            failure_reason = "rpc_protocol_process_error"
        trial["attempts"] = _build_attempts(
            message_tool_calls, host_tool_calls, tool_execution_ends,
            expected, accepted_tool_call_id,
        )
        trial["outcome"] = _compute_outcome(
            trial, accepted, accepted_tool_call_id,
            agent_ended, failure_reason,
        )
        trial["probe_error"] = True

    finally:
        trial["wall_time_seconds"] = round(time.monotonic() - start, 3)
        trial["process_exit_code"] = session.close()
        trial["stderr_tail"] = session.drain_stderr()[-STDERR_TAIL_LINES:]
        shutil.rmtree(temp_dir, ignore_errors=True)

    return trial


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
        return "semantic_mismatch"
    return "tool_execution_error"


def _build_attempts(
    message_tool_calls: list[tuple[int, str, str, Any]],
    host_tool_calls: dict[str, dict[str, Any]],
    tool_execution_ends: dict[str, dict[str, Any]],
    _expected: dict[str, Any],
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

        # Detect OMP argument repair
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
    """Detect transport_model_error or context_overflow from assistant messages.

    Scans captured messages for terminal error signals (stopReason=error,
    errorMessage, or length-based context overflow).  Returns None if no
    terminal error is found.
    """
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

# ─── outcome computation ──────────────────────────────────────────────


def _compute_outcome(
    trial: dict[str, Any],
    accepted: bool,
    _accepted_tool_call_id: str | None,
    agent_ended: bool,
    failure_reason: str | None,
) -> dict[str, Any]:
    """Compute the final trial outcome from collected data."""
    attempts = trial["attempts"]
    tool_attempts = len(attempts)

    # first_attempt_success: raw args matched with no repair
    first_attempt_success = False
    if tool_attempts > 0:
        a0 = attempts[0]
        first_attempt_success = (
            a0["category"] == "accepted"
            and not a0.get("omp_argument_repair", False)
        )

    recovered = False
    failed_before = 0
    if accepted:
        for a in attempts:
            if a["category"] == "accepted":
                break
            failed_before += 1
        recovered = failed_before > 0

    # Determine failure category
    if accepted and agent_ended:
        failure_category: str | None = None
    elif accepted and not agent_ended:
        # Tool was accepted but agent never completed
        if failure_reason in _PROBE_ERROR_CATEGORIES:
            failure_category = failure_reason
        else:
            failure_category = "accepted_no_agent_end"
    elif failure_reason:
        failure_category = failure_reason
    elif tool_attempts > 0:
        # Terminal assistant error/overflow takes precedence over tool-attempt
        # category even when earlier tool attempts exist.
        failure_category = _detect_terminal_error(trial)
        if failure_category is None:
            last_cat = None
            for a in reversed(attempts):
                if a["category"] != "accepted":
                    last_cat = a["category"]
                    break
            failure_category = last_cat or "tool_execution_error"
    elif not agent_ended:
        failure_category = "timeout"
    else:
        # Agent completed without any tool call
        failure_category = _detect_terminal_error(trial) or "missing_tool_call"

    return {
        "first_attempt_success": first_attempt_success,
        "recovered_after_tool_retry": recovered,
        "final_accepted": accepted,
        "agent_completed": agent_ended,
        "failure_category": failure_category,
        "tool_attempts": tool_attempts,
        "failed_attempts_before_accept": failed_before,
    }


# ─── summary and ranking ──────────────────────────────────────────────


_ATTEMPT_CATEGORIES = (
    "malformed_json",
    "argument_schema_failure",
    "unknown_tool_name",
    "semantic_mismatch",
    "duplicate_after_accept",
    "tool_execution_error",
    "accepted",
)


def _stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    sv = sorted(values)
    n = len(sv)
    if n % 2 == 1:
        median = sv[n // 2]
    else:
        median = (sv[n // 2 - 1] + sv[n // 2]) / 2
    return {
        "min": sv[0],
        "median": median,
        "max": sv[-1],
        "count": n,
    }


def _compute_model_summary(
    trials: list[dict[str, Any]],
    context_words: list[int],
) -> dict[str, Any]:
    n = len(trials)
    if n == 0:
        return {"overall": {"trials": 0}, "per_context_size": {}}

    outcomes = [t.get("outcome", {}) for t in trials]

    first_success = sum(1 for o in outcomes if o.get("first_attempt_success"))
    recovered = sum(1 for o in outcomes if o.get("recovered_after_tool_retry"))
    final_success = sum(1 for o in outcomes if o.get("final_accepted"))
    agent_completed = sum(1 for o in outcomes if o.get("agent_completed"))

    ta_vals = [o.get("tool_attempts", 0) for o in outcomes]
    fb_vals = [o.get("failed_attempts_before_accept", 0) for o in outcomes]
    wt_vals = [t.get("wall_time_seconds", 0) for t in trials]

    failure_cats: dict[str, int] = {}
    for o in outcomes:
        cat = o.get("failure_category")
        if cat:
            failure_cats[cat] = failure_cats.get(cat, 0) + 1

    # Attempt category counts across all trials
    attempt_category_counts: dict[str, int] = {c: 0 for c in _ATTEMPT_CATEGORIES}
    omp_repair_count = 0
    total_attempts = 0
    total_retries = 0
    for t in trials:
        trial_attempt_count = len(t.get("attempts", []))
        for a in t.get("attempts", []):
            cat = a.get("category", "tool_execution_error")
            attempt_category_counts[cat] = attempt_category_counts.get(cat, 0) + 1
            if a.get("omp_argument_repair"):
                omp_repair_count += 1
        total_attempts += trial_attempt_count
        total_retries += max(trial_attempt_count - 1, 0)
    attempt_category_counts["omp_argument_repair"] = omp_repair_count

    # Trials with any tool failure (not final_accepted, or any non-accepted attempt)
    trials_with_tool_failure = sum(
        1 for t in trials
        if not t.get("outcome", {}).get("final_accepted", False)
        or any(
            a.get("category") != "accepted"
            for a in t.get("attempts", [])
        )
    )

    # Missing tool calls
    missing_calls = sum(
        1 for o in outcomes if o.get("failure_category") == "missing_tool_call"
    )

    # Auto retries
    auto_retries = sum(t.get("auto_retry_events", 0) for t in trials)

    # Transparent retry recoveries (retryRecovery present in any message)
    transparent_recoveries = sum(
        1 for t in trials
        if any(m.get("retry_recovery") for m in t.get("messages", []))
    )

    # Specific failure category counts
    timeout_count = failure_cats.get("timeout", 0)
    context_overflow_count = failure_cats.get("context_overflow", 0)
    transport_error_count = failure_cats.get("transport_model_error", 0)
    accepted_no_agent_end_count = failure_cats.get("accepted_no_agent_end", 0)
    rpc_protocol_error_count = failure_cats.get("rpc_protocol_process_error", 0)
    setup_failed_count = failure_cats.get("setup_failed", 0)
    prompt_rejected_count = failure_cats.get("prompt_rejected", 0)
    max_tool_attempts_exceeded_count = failure_cats.get(
        "max_tool_attempts_exceeded", 0
    )
    probe_error_count = sum(1 for t in trials if t.get("probe_error"))

    ctx_tokens = [
        t["final_context_usage"]["tokens"]
        for t in trials
        if t.get("final_context_usage")
        and isinstance(t["final_context_usage"], dict)
        and t["final_context_usage"].get("tokens") is not None
    ]

    overall = {
        "trials": n,
        "first_attempt_success_count": first_success,
        "first_attempt_success_rate": round(first_success / n, 4),
        "recovered_after_retry_count": recovered,
        "recovered_after_retry_rate": round(recovered / n, 4),
        "final_success_count": final_success,
        "final_success_rate": round(final_success / n, 4),
        "agent_completion_count": agent_completed,
        "agent_completion_rate": round(agent_completed / n, 4),
        "mean_tool_attempts": round(sum(ta_vals) / n, 2),
        "mean_failed_attempts_before_accept": round(sum(fb_vals) / n, 2),
        "mean_wall_time_seconds": round(sum(wt_vals) / n, 2),
        "failure_categories": failure_cats,
        "context_token_stats": _stats(ctx_tokens),
        # Detailed attempt breakdown
        "attempt_category_counts": attempt_category_counts,
        "trials_with_tool_failure": trials_with_tool_failure,
        "total_attempts": total_attempts,
        "total_retries": total_retries,
        "missing_tool_call_count": missing_calls,
        "auto_retry_count": auto_retries,
        "transparent_retry_recovery_count": transparent_recoveries,
        "timeout_count": timeout_count,
        "context_overflow_count": context_overflow_count,
        "transport_model_error_count": transport_error_count,
        "accepted_no_agent_end_count": accepted_no_agent_end_count,
        "rpc_protocol_error_count": rpc_protocol_error_count,
        "setup_failed_count": setup_failed_count,
        "prompt_rejected_count": prompt_rejected_count,
        "max_tool_attempts_exceeded_count": max_tool_attempts_exceeded_count,
        "probe_error_count": probe_error_count,
    }

    per_size: dict[str, dict[str, Any]] = {}
    for size in context_words:
        size_trials = [t for t in trials if t["context_words"] == size]
        if size_trials:
            size_summary = _compute_model_summary(size_trials, [])
            per_size[str(size)] = size_summary["overall"]

    return {"overall": overall, "per_context_size": per_size}


def _compute_summary(
    trials: list[dict[str, Any]],
    models: list[str],
    context_words: list[int],
) -> dict[str, Any]:
    per_model: dict[str, Any] = {}
    model_trial_counts: dict[str, int] = {}
    for model in models:
        model_trials = [t for t in trials if t["model"] == model]
        model_trial_counts[model] = len(model_trials)
        per_model[model] = _compute_model_summary(model_trials, context_words)

    # Omit ranking when model trial counts differ
    counts = list(model_trial_counts.values())
    ranking_omitted = len(set(counts)) > 1 or all(c == 0 for c in counts)

    if ranking_omitted:
        return {"per_model": per_model, "ranking": [], "ranking_omitted": True}

    # Deterministic ranking with full tie-break chain:
    # final_success_rate desc, first_attempt_success_rate desc,
    # fewer failed trials asc, fewer retries asc, wall time asc, model asc.
    ranking = sorted(
        models,
        key=lambda m: (
            -per_model[m]["overall"].get("final_success_rate", 0),
            -per_model[m]["overall"].get("first_attempt_success_rate", 0),
            per_model[m]["overall"].get("trials_with_tool_failure", 0),
            per_model[m]["overall"].get("total_retries", 0),
            per_model[m]["overall"].get("mean_wall_time_seconds", 0),
            m,
        ),
    )
    ranking_list = [
        {
            "model": m,
            "final_success_rate": per_model[m]["overall"].get("final_success_rate", 0),
            "first_attempt_success_rate": per_model[m]["overall"].get(
                "first_attempt_success_rate", 0
            ),
            "trials_with_tool_failure": per_model[m]["overall"].get(
                "trials_with_tool_failure", 0
            ),
            "total_retries": per_model[m]["overall"].get("total_retries", 0),
            "mean_wall_time_seconds": per_model[m]["overall"].get(
                "mean_wall_time_seconds", 0
            ),
            "mean_tool_attempts": per_model[m]["overall"].get("mean_tool_attempts", 0),
        }
        for m in ranking
    ]

    return {
        "per_model": per_model,
        "ranking": ranking_list,
        "ranking_omitted": False,
    }


# ─── artifact ─────────────────────────────────────────────────────────


def _build_schedule(
    models: list[str],
    context_words: list[int],
    trials_per_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build a balanced schedule rotating model order per size/trial."""
    schedule: list[dict[str, Any]] = []
    n = len(models)
    for size_idx, size in enumerate(context_words):
        for trial_idx in range(trials_per_size):
            offset = (trial_idx + size_idx) % n
            order = models[offset:] + models[:offset]
            for model in order:
                schedule.append(
                    {
                        "model": model,
                        "context_words": size,
                        "trial_index": trial_idx,
                        "size_index": size_idx,
                        "context_seed": seed,
                    }
                )
    return schedule


def build_artifact(
    config: dict[str, Any],
    omp_path: Path,
    omp_version: str,
    system_prompt: str,
    tool_definition: dict[str, Any],
    schedule: list[dict[str, Any]],
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the full reproducibility artifact."""
    completed = len(trials)
    total = len(schedule)

    summary = _compute_summary(
        trials,
        config["models"],
        config["context_words"],
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "probe_version": PROBE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - system Python 3.10
        "probe_sha256": _probe_file_sha256(),
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
        "context_generator": {
            "version": GENERATOR_VERSION,
            "vocabulary_sha256": _vocab_sha256(),
            "vocabulary_size": len(FILLER_VOCAB),
        },
        "schedule": schedule,
        "trials": trials,
        "progress": {"completed": completed, "total": total},
        "limitations": [
            "Silent provider-side repairs of malformed JSON are unobservable; "
            "only OMP's validation-layer rejection and host-argument diff are "
            "detectable.  When the raw assistant arguments contain __parseError "
            "but the host_tool_call arguments are valid, omp_argument_repair is "
            "set, but the specific repair transformation is not captured.",
            "Context token counts are only available when RPC state supplies "
            "contextUsage; models that error before reporting state have null "
            "token stats.",
            "DECOY_RECORD entries share slot names with AUTHENTIC_RECORD "
            "entries; only the marker prefix distinguishes them.",
            "The filler vocabulary is pinned to 120 common English words; "
            "real-world context distributions may produce different difficulty.",
            "Tool argument schema validation is performed by OMP before host "
            "tool call emission; schema failures prevent host_tool_call frames "
            "and are classified from tool_execution_end error text.",
            "The probe does not make live model calls during validation; "
            "results depend on model availability and API configuration at "
            "run time.",
            "Ranking is omitted when models have different trial counts "
            "(e.g. due to a probe_error aborting the schedule mid-run).",
        ],
        "summary": summary,
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
            "Reproducible OMP RPC tool-call reliability probe for "
            "long-context needle extraction."
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
        "--context-words",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_WORDS),
        metavar="N",
        help="Number of filler words per context (default: %(default)s)",
    )
    parser.add_argument(
        "--trials-per-size",
        type=int,
        default=DEFAULT_TRIALS_PER_SIZE,
        metavar="N",
        help="Trials per context size per model (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="N",
        help="Deterministic seed for context generation (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tool-attempts",
        type=int,
        default=DEFAULT_MAX_TOOL_ATTEMPTS,
        metavar="N",
        help="Max tool calls before aborting (default: %(default)s)",
    )
    parser.add_argument(
        "--trial-timeout-seconds",
        type=int,
        default=DEFAULT_TRIAL_TIMEOUT_SECONDS,
        metavar="N",
        help="Wall-clock timeout per trial (default: %(default)s)",
    )
    parser.add_argument(
        "--omp-bin",
        default=DEFAULT_OMP_BIN,
        metavar="PATH",
        help="OMP binary name or path (default: %(default)s)",
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

    if not args.context_words or any(w <= 0 for w in args.context_words):
        parser.error("--context-words must be positive integers")

    if args.trials_per_size <= 0:
        parser.error("--trials-per-size must be a positive integer")

    if args.max_tool_attempts <= 0:
        parser.error("--max-tool-attempts must be a positive integer")

    if args.trial_timeout_seconds <= 0:
        parser.error("--trial-timeout-seconds must be a positive integer")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    models = list(args.models)
    context_words = list(args.context_words)
    seed = args.seed
    trials_per_size = args.trials_per_size
    max_tool_attempts = args.max_tool_attempts
    trial_timeout = args.trial_timeout_seconds

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
    schedule = _build_schedule(models, context_words, trials_per_size, seed)

    config = {
        "models": models,
        "context_words": context_words,
        "trials_per_size": trials_per_size,
        "seed": seed,
        "max_tool_attempts": max_tool_attempts,
        "trial_timeout_seconds": trial_timeout,
        "omp_bin": args.omp_bin,
    }

    # ── write initial artifact ──
    try:
        initial = build_artifact(
            config, omp_path, omp_version,
            system_prompt, tool_definition, schedule, [],
        )
        write_artifact(initial, args.output)
    except Exception as exc:
        print(f"probe error: failed to write initial artifact: {exc}", file=sys.stderr)
        return EXIT_PROBE_ERROR

    # ── run trials ──
    trials: list[dict[str, Any]] = []
    total = len(schedule)
    probe_error_aborted = False
    for i, spec in enumerate(schedule):
        print(
            f"[{i + 1}/{total}] model={spec['model']} "
            f"size={spec['context_words']} trial={spec['trial_index']}",
            file=sys.stderr,
        )
        trial = run_trial(
            model=spec["model"],
            context_words=spec["context_words"],
            trial_index=spec["trial_index"],
            size_index=spec["size_index"],
            seed=spec["context_seed"],
            max_tool_attempts=max_tool_attempts,
            trial_timeout=trial_timeout,
            omp_path=omp_path,
            system_prompt=system_prompt,
            tool_definition=tool_definition,
        )
        trials.append(trial)

        # ── checkpoint atomically after every trial ──
        try:
            artifact = build_artifact(
                config, omp_path, omp_version,
                system_prompt, tool_definition, schedule, trials,
            )
            write_artifact(artifact, args.output)
        except Exception as exc:
            print(
                f"probe error: failed to checkpoint after trial {i + 1}: {exc}",
                file=sys.stderr,
            )
            return EXIT_PROBE_ERROR

        # ── abort on probe_error (infrastructure failure) ──
        if trial.get("probe_error"):
            print(
                f"probe error: trial {i + 1} infrastructure failure "
                f"({trial['outcome'].get('failure_category', 'unknown')}), "
                f"aborting remaining schedule.",
                file=sys.stderr,
            )
            probe_error_aborted = True
            break

    # ── final summary to stderr ──
    final_artifact = build_artifact(
        config, omp_path, omp_version,
        system_prompt, tool_definition, schedule, trials,
    )
    summary = final_artifact.get("summary", {})
    ranking = summary.get("ranking", [])
    if ranking:
        print(
            "\nRanking (by final_success_rate, first_attempt_success_rate):",
            file=sys.stderr,
        )
        for r in ranking:
            print(
                f"  {r['model']}: "
                f"final={r['final_success_rate']:.1%} "
                f"first={r['first_attempt_success_rate']:.1%} "
                f"failures={r['trials_with_tool_failure']} "
                f"retries={r['total_retries']} "
                f"mean_time={r['mean_wall_time_seconds']:.1f}s",
                file=sys.stderr,
            )
    elif summary.get("ranking_omitted"):
        print(
            "\nRanking omitted (models have different trial counts).",
            file=sys.stderr,
        )

    print(f"\nArtifact written to {args.output}", file=sys.stderr)

    if probe_error_aborted:
        return EXIT_PROBE_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
