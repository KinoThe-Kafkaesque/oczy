"""Deterministic regression tests for the OMP RPC tool-call reliability probe.

Covers deterministic context generation, strict argument matching, OMP
argument-repair detection, balanced scheduling, attempt ordering and
classification, outcome prioritisation, summary accounting and ranking,
atomic artifact output, and two end-to-end subprocess smokes against
temporary fake OMP JSONL executables — one proving ordered recovery and
final success, one proving infrastructure-abort semantics.

No real OMP binary, provider, network, or model is involved.  Every test is
deterministic and full-suite safe.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# ─── dynamic import of the probe module ───────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "benchmarks" / "omp" / "tool_reliability_probe.py"

_spec = importlib.util.spec_from_file_location(
    "oczy_omp_tool_reliability_probe_tests", PROBE_PATH
)
assert _spec is not None and _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = probe
_spec.loader.exec_module(probe)


# ─── compact helpers ──────────────────────────────────────────────────


def _attempt(category: str, omp_repair: bool = False) -> dict:
    """A minimal attempt record for summary/outcome tests."""
    return {
        "index": 0,
        "tool_call_id": "tc",
        "tool_name": probe.TOOL_NAME,
        "category": category,
        "raw_arguments": {},
        "effective_arguments": {},
        "omp_argument_repair": omp_repair,
        "host_tool_call_emitted": True,
        "host_tool_call_id": "h",
        "message_index": 0,
    }


def _outcome(
    final_accepted: bool = True,
    agent_completed: bool = True,
    failure_category: str | None = None,
    tool_attempts: int = 1,
    first_attempt_success: bool = True,
    recovered: bool = False,
    failed_before: int = 0,
) -> dict:
    return {
        "first_attempt_success": first_attempt_success,
        "recovered_after_tool_retry": recovered,
        "final_accepted": final_accepted,
        "agent_completed": agent_completed,
        "failure_category": failure_category,
        "tool_attempts": tool_attempts,
        "failed_attempts_before_accept": failed_before,
    }


def _trial(
    model: str,
    outcome: dict,
    *,
    attempts: list[dict] | None = None,
    auto_retry: int = 0,
    messages: list[dict] | None = None,
    wall_time: float = 1.0,
    probe_error: bool = False,
    final_ctx: dict | None = None,
    context_words: int = 200,
) -> dict:
    msgs = messages or []
    return {
        "model": model,
        "context_words": context_words,
        "trial_index": 0,
        "size_index": 0,
        "context_seed": probe.DEFAULT_SEED,
        "context": {},
        "probe_error": probe_error,
        "outcome": outcome,
        "attempts": attempts or [],
        "messages": msgs,
        "turn_count": 1,
        "message_count": len(msgs),
        "auto_retry_events": auto_retry,
        "auto_compaction_events": 0,
        "diagnostic_events": [],
        "baseline_context_usage": None,
        "final_context_usage": final_ctx,
        "wall_time_seconds": wall_time,
        "stderr_tail": [],
        "non_json_rpc_lines": [],
        "final_assistant_text": "",
        "process_exit_code": 0,
        "omp_model_resolved": "fake",
    }


def _expected_records(seed: int, size: int, trial: int) -> dict:
    _, expected, _ = probe.generate_context(seed, size, trial)
    return expected


def _exp() -> dict:
    """A fixed expected-arguments dict for _build_attempts tests."""
    return {
        "trial_id": "T",
        "records": [
            {"slot": "alpha", "key": "ak", "value": "av"},
            {"slot": "beta", "key": "bk", "value": "bv"},
            {"slot": "gamma", "key": "gk", "value": "gv"},
        ],
        "guard": "G",
        "finalize": True,
    }


def test_public_model_metadata_strips_credentials() -> None:
    model = {
        "id": "safe-model",
        "contextWindow": 1000,
        "headers": {"Authorization": "Bearer must-not-leak"},
        "compatConfig": {"apiKey": "must-not-leak"},
    }
    public = probe._public_model_metadata(model)
    assert public == {"id": "safe-model", "contextWindow": 1000}
    assert "must-not-leak" not in json.dumps(public)


# ─── context generation ───────────────────────────────────────────────


def test_context_deterministic_same_seed_size_trial() -> None:
    c1, exp1, art1 = probe.generate_context(20260712, 300, 0)
    c2, exp2, art2 = probe.generate_context(20260712, 300, 0)
    assert c1 == c2
    assert exp1 == exp2
    assert art1 == art2
    assert art1["context_sha256"] == probe._sha256_hex(c1)


def test_context_different_trial_changes_needles() -> None:
    _, exp0, art0 = probe.generate_context(20260712, 300, 0)
    _, exp1, art1 = probe.generate_context(20260712, 300, 1)
    assert art0["trial_id"] != art1["trial_id"]
    assert art0["context_sha256"] != art1["context_sha256"]
    assert exp0["guard"] != exp1["guard"]
    assert exp0["records"][0]["key"] != exp1["records"][0]["key"]
    assert exp0["records"][0]["value"] != exp1["records"][0]["value"]


def test_context_authentic_values_absent_from_prompt_and_schema() -> None:
    _, expected, artifact = probe.generate_context(20260712, 300, 0)
    prompt = probe.build_system_prompt()
    schema_json = json.dumps(probe.build_tool_definition(), ensure_ascii=False)
    sensitive = {artifact["trial_id"], expected["guard"]}
    for rec in expected["records"]:
        sensitive.add(rec["key"])
        sensitive.add(rec["value"])
    for val in sensitive:
        assert val not in prompt
        assert val not in schema_json


def test_context_artifact_records_count_sha_positions() -> None:
    size = 250
    _, _, art = probe.generate_context(20260712, size, 2)
    assert art["requested_word_count"] == size
    assert len(art["context_sha256"]) == 64
    assert int(art["context_sha256"], 16) >= 0
    assert art["needle_positions"] == {
        "alpha": int(size * 0.15),
        "beta": int(size * 0.50),
        "guard": int(size * 0.70),
        "gamma": int(size * 0.85),
    }
    assert art["decoy_count"] == 6
    assert art["expected_arguments"]["finalize"] is True


# ─── argument matching ────────────────────────────────────────────────


def test_arguments_match_accepts_correct() -> None:
    expected = _expected_records(20260712, 200, 0)
    assert probe.arguments_match(expected, expected) is True


def test_arguments_match_rejects_variants() -> None:
    expected = _expected_records(20260712, 200, 0)

    def _bad(**overrides) -> dict:
        d = json.loads(json.dumps(expected))
        for k, v in overrides.items():
            d[k] = v
        return d

    assert probe.arguments_match(_bad(trial_id="wrong"), expected) is False
    assert probe.arguments_match(
        _bad(records=[{"slot": "alpha", "key": "WRONG", "value": "av"}] + expected["records"][1:]),
        expected,
    ) is False
    assert probe.arguments_match("nope", expected) is False
    assert probe.arguments_match({"trial_id": expected["trial_id"]}, expected) is False
    assert probe.arguments_match(_bad(guard="wrong"), expected) is False
    assert probe.arguments_match(_bad(finalize=False), expected) is False
    # swapped slot order
    swapped = json.loads(json.dumps(expected))
    swapped["records"][0], swapped["records"][1] = swapped["records"][1], swapped["records"][0]
    assert probe.arguments_match(swapped, expected) is False
    # wrong record count
    short = json.loads(json.dumps(expected))
    short["records"] = short["records"][:2]
    assert probe.arguments_match(short, expected) is False
    # records not a list
    nonlist = json.loads(json.dumps(expected))
    nonlist["records"] = "nope"
    assert probe.arguments_match(nonlist, expected) is False


# ─── OMP argument-repair detection ────────────────────────────────────


def test_strip_harness_i_only_strips_i() -> None:
    stripped = probe._strip_harness_i({"i": "intent", "trial_id": "T", "records": []})
    assert "i" not in stripped
    assert stripped["trial_id"] == "T"
    assert probe._strip_harness_i({"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert probe._strip_harness_i("nope") == "nope"


def test_detect_repair_only_i_differs_is_not_repair() -> None:
    raw = {"trial_id": "T", "i": "intent"}
    host = {"trial_id": "T", "i": "other"}
    assert probe._detect_repair(raw, host) is False


def test_detect_repair_semantic_diff_is_repair() -> None:
    raw = {"trial_id": "T", "guard": "wrong", "i": "intent"}
    host = {"trial_id": "T", "guard": "correct", "i": "other"}
    assert probe._detect_repair(raw, host) is True


# ─── schedule ─────────────────────────────────────────────────────────


def test_schedule_balanced_rotation() -> None:
    models = ["m1", "m2", "m3"]
    schedule = probe._build_schedule(models, [100, 200], 2, seed=42)
    assert len(schedule) == 12
    for m in models:
        assert sum(1 for s in schedule if s["model"] == m) == 4
    # size_idx=0 trial_idx=0 → offset 0 → [m1, m2, m3]
    assert [s["model"] for s in schedule[:3]] == ["m1", "m2", "m3"]
    # size_idx=0 trial_idx=1 → offset 1 → [m2, m3, m1]
    assert [s["model"] for s in schedule[3:6]] == ["m2", "m3", "m1"]
    # size_idx=1 trial_idx=0 → offset 1 → [m2, m3, m1]
    assert [s["model"] for s in schedule[6:9]] == ["m2", "m3", "m1"]


def test_schedule_identical_context_seeds_across_models() -> None:
    schedule = probe._build_schedule(["m1", "m2"], [200], 2, seed=99)
    assert all(s["context_seed"] == 99 for s in schedule)
    c_a, _, _ = probe.generate_context(99, 200, 0)
    c_b, _, _ = probe.generate_context(99, 200, 0)
    assert c_a == c_b


# ─── _build_attempts: ordering and classification ─────────────────────


def test_build_attempts_order_semantic_before_accepted_duplicate_after() -> None:
    expected = _exp()
    wrong = json.loads(json.dumps(expected))
    wrong["records"][0]["key"] = "WRONG"
    mtc = [
        (0, "tc-1", probe.TOOL_NAME, wrong),
        (1, "tc-2", probe.TOOL_NAME, expected),
        (2, "tc-3", probe.TOOL_NAME, expected),
    ]
    htc = {
        "tc-1": {"id": "h1", "toolName": probe.TOOL_NAME, "arguments": wrong},
        "tc-2": {"id": "h2", "toolName": probe.TOOL_NAME, "arguments": expected},
        "tc-3": {"id": "h3", "toolName": probe.TOOL_NAME, "arguments": expected},
    }
    attempts = probe._build_attempts(mtc, htc, {}, expected, "tc-2")
    cats = [a["category"] for a in attempts]
    assert cats == ["semantic_mismatch", "accepted", "duplicate_after_accept"]
    assert all(not a["omp_argument_repair"] for a in attempts)
    assert [a["index"] for a in attempts] == [0, 1, 2]


def test_build_attempts_categories_malformed_schema_unknown() -> None:
    expected = _exp()
    mtc = [
        (0, "tc-mal", probe.TOOL_NAME, {"__parseError": "bad json"}),
        (1, "tc-sch", probe.TOOL_NAME, {"trial_id": "T"}),
        (2, "tc-unk", "other_tool", {}),
    ]
    te = {
        "tc-mal": {
            "isError": True,
            "toolName": probe.TOOL_NAME,
            "result": {"content": [{"type": "text", "text": "Arguments are not valid JSON"}]},
        },
        "tc-sch": {
            "isError": True,
            "toolName": probe.TOOL_NAME,
            "result": {"content": [{"type": "text", "text": "Schema validation failed: missing records"}]},
        },
    }
    attempts = probe._build_attempts(mtc, {}, te, expected, None)
    cats = [a["category"] for a in attempts]
    assert cats == ["malformed_json", "argument_schema_failure", "unknown_tool_name"]
    assert attempts[0]["host_tool_call_emitted"] is False
    assert attempts[1]["host_tool_call_emitted"] is False


def test_build_attempts_repaired_accepted_not_raw_first_attempt_success() -> None:
    expected = _exp()
    raw_malformed = {"__parseError": "bad json", "trial_id": "T"}
    mtc = [(0, "tc-1", probe.TOOL_NAME, raw_malformed)]
    htc = {"tc-1": {"id": "h1", "toolName": probe.TOOL_NAME, "arguments": expected}}
    attempts = probe._build_attempts(mtc, htc, {}, expected, "tc-1")
    assert len(attempts) == 1
    assert attempts[0]["category"] == "accepted"
    assert attempts[0]["omp_argument_repair"] is True
    trial = {"attempts": attempts, "messages": []}
    outcome = probe._compute_outcome(trial, True, "tc-1", True, None)
    assert outcome["first_attempt_success"] is False
    assert outcome["final_accepted"] is True


# ─── _compute_outcome: prioritisation ─────────────────────────────────


def test_compute_outcome_timeout() -> None:
    trial = {"attempts": [], "messages": []}
    o = probe._compute_outcome(trial, False, None, False, "timeout")
    assert o["failure_category"] == "timeout"
    assert o["final_accepted"] is False


def test_compute_outcome_max_tool_attempts_exceeded() -> None:
    trial = {"attempts": [], "messages": []}
    o = probe._compute_outcome(trial, False, None, False, "max_tool_attempts_exceeded")
    assert o["failure_category"] == "max_tool_attempts_exceeded"


def test_compute_outcome_accepted_no_agent_end() -> None:
    trial = {"attempts": [_attempt("accepted")], "messages": []}
    o = probe._compute_outcome(trial, True, "tc-1", False, None)
    assert o["failure_category"] == "accepted_no_agent_end"
    assert o["final_accepted"] is True
    assert o["agent_completed"] is False


def test_compute_outcome_success_no_failure_category() -> None:
    trial = {"attempts": [_attempt("accepted")], "messages": []}
    o = probe._compute_outcome(trial, True, "tc-1", True, None)
    assert o["failure_category"] is None
    assert o["first_attempt_success"] is True


def test_compute_outcome_probe_error_overrides_accepted_no_agent_end() -> None:
    """Probe-error categories take precedence over accepted_no_agent_end."""
    trial = {"attempts": [_attempt("accepted")], "messages": []}
    o = probe._compute_outcome(trial, True, "tc-1", False, "rpc_protocol_process_error")
    assert o["failure_category"] == "rpc_protocol_process_error"


def test_compute_outcome_timeout_does_not_override_accepted_no_agent_end() -> None:
    """timeout (non-probe-error) does not override accepted_no_agent_end."""
    trial = {"attempts": [_attempt("accepted")], "messages": []}
    o = probe._compute_outcome(trial, True, "tc-1", False, "timeout")
    assert o["failure_category"] == "accepted_no_agent_end"


# ─── summary: retry accounting ────────────────────────────────────────


def test_summary_total_retries_uses_max_attempts_minus_one() -> None:
    trials = [
        _trial(
            "m1",
            _outcome(tool_attempts=3, first_attempt_success=False, recovered=True, failed_before=2),
            attempts=[_attempt("semantic_mismatch"), _attempt("semantic_mismatch"), _attempt("accepted")],
        ),
        _trial("m1", _outcome(tool_attempts=1), attempts=[_attempt("accepted")]),
    ]
    o = probe._compute_model_summary(trials, [200])["overall"]
    # max(3-1,0) + max(1-1,0) = 2
    assert o["total_retries"] == 2
    assert o["total_attempts"] == 4


# ─── summary: attempt category counts and OMP repairs ─────────────────


def test_summary_attempt_category_counts_and_omp_repairs() -> None:
    trials = [
        _trial(
            "m1",
            _outcome(tool_attempts=3, first_attempt_success=False, recovered=True, failed_before=2),
            attempts=[
                _attempt("malformed_json"),
                _attempt("argument_schema_failure"),
                _attempt("accepted"),
            ],
        ),
        _trial(
            "m1",
            _outcome(tool_attempts=2, first_attempt_success=False, recovered=True, failed_before=1),
            attempts=[
                _attempt("unknown_tool_name", omp_repair=True),
                _attempt("accepted", omp_repair=True),
            ],
        ),
    ]
    o = probe._compute_model_summary(trials, [200])["overall"]
    c = o["attempt_category_counts"]
    assert c["malformed_json"] == 1
    assert c["argument_schema_failure"] == 1
    assert c["unknown_tool_name"] == 1
    assert c["accepted"] == 2
    assert c["semantic_mismatch"] == 0
    assert c["duplicate_after_accept"] == 0
    assert c["tool_execution_error"] == 0
    assert c["omp_argument_repair"] == 2


# ─── summary: auto and transparent retries ────────────────────────────


def test_summary_auto_and_transparent_retries() -> None:
    trials = [
        _trial("m1", _outcome(), auto_retry=3, messages=[{"retry_recovery": None}]),
        _trial("m1", _outcome(), auto_retry=1, messages=[{"retry_recovery": {"id": "r1"}}]),
    ]
    o = probe._compute_model_summary(trials, [200])["overall"]
    assert o["auto_retry_count"] == 4
    assert o["transparent_retry_recovery_count"] == 1


# ─── summary: failure category counts ─────────────────────────────────


def test_summary_failure_category_counts() -> None:
    cats = [
        "timeout",
        "context_overflow",
        "transport_model_error",
        "accepted_no_agent_end",
        "rpc_protocol_process_error",
        "setup_failed",
        "prompt_rejected",
        "max_tool_attempts_exceeded",
        "missing_tool_call",
    ]
    trials = [_trial("m1", _outcome(final_accepted=False, agent_completed=False, failure_category=cat)) for cat in cats]
    # One additional probe_error trial with the same RPC failure category.
    trials.append(_trial("m1", _outcome(failure_category="rpc_protocol_process_error"), probe_error=True))
    o = probe._compute_model_summary(trials, [200])["overall"]
    assert o["timeout_count"] == 1
    assert o["context_overflow_count"] == 1
    assert o["transport_model_error_count"] == 1
    assert o["accepted_no_agent_end_count"] == 1
    assert o["rpc_protocol_error_count"] == 2
    assert o["setup_failed_count"] == 1
    assert o["prompt_rejected_count"] == 1
    assert o["max_tool_attempts_exceeded_count"] == 1
    assert o["missing_tool_call_count"] == 1
    assert o["probe_error_count"] == 1


# ─── summary: ranking ─────────────────────────────────────────────────


def test_summary_ranking_omitted_for_zero_counts() -> None:
    summary = probe._compute_summary([], ["m1", "m2"], [200])
    assert summary["ranking_omitted"] is True
    assert summary["ranking"] == []


def test_summary_ranking_omitted_for_unequal_counts() -> None:
    trials = [
        _trial("m1", _outcome()),
        _trial("m1", _outcome()),
        _trial("m2", _outcome()),
    ]
    summary = probe._compute_summary(trials, ["m1", "m2"], [200])
    assert summary["ranking_omitted"] is True
    assert summary["ranking"] == []


def test_summary_ranking_deterministic_for_equal_counts() -> None:
    """Equal counts: higher final_success_rate ranks first."""
    trials = [
        _trial("m1", _outcome(), wall_time=1.0),
        _trial("m1", _outcome(), wall_time=1.0),
        _trial("m2", _outcome(final_accepted=False, failure_category="semantic_mismatch"), wall_time=1.0),
        _trial("m2", _outcome(), wall_time=1.0),
    ]
    summary = probe._compute_summary(trials, ["m1", "m2"], [200])
    assert summary["ranking_omitted"] is False
    assert [r["model"] for r in summary["ranking"]] == ["m1", "m2"]
    assert summary["ranking"][0]["final_success_rate"] == 1.0
    assert summary["ranking"][1]["final_success_rate"] == 0.5


def test_summary_ranking_tiebreak_wall_time_then_name() -> None:
    """Full tie on rates → lower wall_time wins; full tie → model asc."""
    # m_a slower than m_b → m_b first
    trials = [
        _trial("m_a", _outcome(), wall_time=2.0),
        _trial("m_a", _outcome(), wall_time=2.0),
        _trial("m_b", _outcome(), wall_time=1.0),
        _trial("m_b", _outcome(), wall_time=1.0),
    ]
    summary = probe._compute_summary(trials, ["m_a", "m_b"], [200])
    assert [r["model"] for r in summary["ranking"]] == ["m_b", "m_a"]

    # Full tie including wall_time → model asc
    trials_tie = [
        _trial("m_a", _outcome(), wall_time=1.0),
        _trial("m_a", _outcome(), wall_time=1.0),
        _trial("m_b", _outcome(), wall_time=1.0),
        _trial("m_b", _outcome(), wall_time=1.0),
    ]
    summary_tie = probe._compute_summary(trials_tie, ["m_a", "m_b"], [200])
    assert [r["model"] for r in summary_tie["ranking"]] == ["m_a", "m_b"]


# ─── artifact output ──────────────────────────────────────────────────


def test_write_artifact_atomic_no_tmp_residue(tmp_path: Path) -> None:
    artifact = {"schema_version": probe.SCHEMA_VERSION, "trials": [], "progress": {"completed": 0, "total": 3}}
    out = tmp_path / "result.json"
    probe.write_artifact(artifact, str(out))
    assert json.loads(out.read_text()) == artifact
    assert [p for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_build_artifact_partial_progress(tmp_path: Path) -> None:
    config = {
        "models": ["m1"],
        "context_words": [200],
        "trials_per_size": 3,
        "seed": 42,
        "max_tool_attempts": 4,
        "trial_timeout_seconds": 30,
        "omp_bin": "fake",
    }
    schedule = probe._build_schedule(["m1"], [200], 3, 42)
    trials = [_trial("m1", _outcome())]
    artifact = probe.build_artifact(
        config,
        tmp_path / "fake",
        "fake-1.0",
        probe.build_system_prompt(),
        probe.build_tool_definition(),
        schedule,
        trials,
    )
    assert artifact["progress"] == {"completed": 1, "total": 3}
    assert len(artifact["trials"]) == 1
    assert len(artifact["schedule"]) == 3
    assert artifact["schema_version"] == probe.SCHEMA_VERSION


# ─── end-to-end: fake RPC recovery smoke ──────────────────────────────

_FAKE_OMP_RPC = r'''#!/usr/bin/env python3
import sys, json, re

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def parse_context(text):
    records = {}
    guard = None
    trial_id = None
    for line in text.split("\n"):
        if line.startswith("AUTHENTIC_RECORD"):
            m = re.match(r"AUTHENTIC_RECORD slot=(\w+) key=(\w+) value=(\w+)", line)
            if m:
                records[m.group(1)] = {"slot": m.group(1), "key": m.group(2), "value": m.group(3)}
        elif line.startswith("AUTHENTIC_GUARD"):
            m = re.match(r"AUTHENTIC_GUARD token=(\w+)", line)
            if m:
                guard = m.group(1)
        elif line.startswith("TRIAL_ID:"):
            trial_id = line.split(":", 1)[1].strip()
    return trial_id, records, guard

def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        sys.stdout.write("fake-omp-rpc 0.0.0-test\n")
        sys.stdout.flush()
        return 0
    emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctype = cmd.get("type")
        cid = cmd.get("id")
        if ctype in ("set_host_tools", "set_auto_retry", "set_auto_compaction"):
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "get_state":
            emit({"type": "response", "id": cid, "success": True,
                  "data": {"model": "fake-model", "contextUsage": {"tokens": 1000}}})
        elif ctype == "prompt":
            msg = cmd.get("message", "")
            trial_id, records, guard = parse_context(msg)
            correct = {"trial_id": trial_id,
                       "records": [records["alpha"], records["beta"], records["gamma"]],
                       "guard": guard, "finalize": True}
            wrong = dict(correct)
            wrong["guard"] = "wrongguard"
            # 1. host_tool_call with wrong args BEFORE prompt ack
            emit({"type": "host_tool_call", "id": "htc-1", "toolCallId": "tc-1",
                  "toolName": "submit_context_evidence", "arguments": wrong})
            # 2. message_end with first toolCall
            emit({"type": "message_end", "message": {"role": "assistant",
                  "content": [{"type": "toolCall", "id": "tc-1",
                   "name": "submit_context_evidence", "arguments": wrong}]}})
            # 3. host_tool_call with correct args (recovery retry)
            emit({"type": "host_tool_call", "id": "htc-2", "toolCallId": "tc-2",
                  "toolName": "submit_context_evidence", "arguments": correct})
            # 4. message_end with second toolCall
            emit({"type": "message_end", "message": {"role": "assistant",
                  "content": [{"type": "toolCall", "id": "tc-2",
                   "name": "submit_context_evidence", "arguments": correct}]}})
            # 5. agent_end BEFORE late prompt ack
            emit({"type": "agent_end", "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "DONE"}]}]})
            # 6. late prompt ack
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "abort":
            pass
    return 0

sys.exit(main())
'''


def test_end_to_end_fake_rpc_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full probe run against a fake OMP RPC process.

    The fake emits a host_tool_call before prompt ack, returns one semantic
    error then a correct retry, emits agent_end before the late prompt ack,
    and serves the final get_state.  The probe must record ordered
    attempts/recovery/final success without deadlock.
    """
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_RPC)
    os.chmod(fake, 0o755)

    # Prove no network is touched.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    rc = probe.main([
        "fake/model",
        "--context-words", "200",
        "--trials-per-size", "1",
        "--trial-timeout-seconds", "30",
        "--omp-bin", str(fake),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_OK

    artifact = json.loads(out.read_text())
    assert artifact["schema_version"] == probe.SCHEMA_VERSION
    assert artifact["progress"] == {"completed": 1, "total": 1}
    assert len(artifact["trials"]) == 1

    trial = artifact["trials"][0]
    assert trial["probe_error"] is False

    # Ordered attempts: semantic_mismatch then accepted.
    cats = [a["category"] for a in trial["attempts"]]
    assert cats == ["semantic_mismatch", "accepted"]
    assert all(not a["omp_argument_repair"] for a in trial["attempts"])

    # Recovery, not raw first-attempt success.
    o = trial["outcome"]
    assert o["first_attempt_success"] is False
    assert o["recovered_after_tool_retry"] is True
    assert o["final_accepted"] is True
    assert o["agent_completed"] is True
    assert o["failure_category"] is None
    assert o["tool_attempts"] == 2
    assert o["failed_attempts_before_accept"] == 1

    # Final get_state was served.
    assert trial["final_context_usage"] == {"tokens": 1000}
    # Agent ended with DONE text.
    assert trial["final_assistant_text"] == "DONE"

    # No temp residue.
    assert [p for p in out.parent.iterdir() if p.suffix == ".tmp"] == []


# ─── end-to-end: fake RPC setup failure abort ─────────────────────────

_FAKE_OMP_FAIL = r'''#!/usr/bin/env python3
import sys, json

def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        sys.stdout.write("fake-omp-fail 0.0.0-test\n")
        sys.stdout.flush()
        return 0
    sys.stdout.write(json.dumps({"type": "ready"}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # Emit non-JSON garbage to trigger rpc_protocol_process_error.
        sys.stdout.write("THIS IS NOT JSON GARBAGE\n")
        sys.stdout.flush()
        return 1
    return 0

sys.exit(main())
'''


def test_end_to_end_setup_failure_aborts_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-JSON/setup failure produces a partial artifact, probe_error,
    remaining schedule aborted, and exit code 2."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_FAIL)
    os.chmod(fake, 0o755)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    rc = probe.main([
        "m1", "m2",
        "--context-words", "200",
        "--trials-per-size", "2",
        "--trial-timeout-seconds", "15",
        "--omp-bin", str(fake),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_PROBE_ERROR

    # Partial artifact written.
    assert out.exists()
    artifact = json.loads(out.read_text())
    total = len(artifact["schedule"])
    assert total == 4  # 2 models × 1 size × 2 trials
    assert artifact["progress"] == {"completed": 1, "total": total}

    # Only the first (failed) trial is present.
    assert len(artifact["trials"]) == 1
    trial = artifact["trials"][0]
    assert trial["probe_error"] is True
    assert trial["outcome"]["failure_category"] == "rpc_protocol_process_error"
    # Non-JSON line was captured.
    assert len(trial["non_json_rpc_lines"]) >= 1

    # Ranking omitted: m1 has 1 trial, m2 has 0 → unequal counts.
    assert artifact["summary"]["ranking_omitted"] is True

    # No temp residue.
    assert [p for p in out.parent.iterdir() if p.suffix == ".tmp"] == []
