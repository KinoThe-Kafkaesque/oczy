"""R18 Prompt Contract Diagnostic.

DEV-only prompt audit comparing raw and chat-template prompt modes for
Research 18's consolidation distillation.  For every stage-0 teaching/dev
pair, records rendered prompt hash/length, role boundaries, correction/
request substring presence and order, token counts, truncation flags,
answer-prefix tokens, first generated token/prediction, and correctness.

Detects malformed role/template application, missing correction, request
truncation, and answer-prefix mismatch.  Emits aggregate METRIC/ASI
diagnostics and a machine-readable JSON audit.

Does NOT modify the frozen teacher metric, does NOT access holdout data,
does NOT make any H-DISTILL verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from typing import Any, cast

from oczy.eval_v2.scoring import probe_matches
from oczy.experiments.consolidation_distillation import (
    _distillation_prompts,
    _token_count,
)
from oczy.experiments.organism_curriculum.dataset import (
    Stage,
    build_curriculum,
    split_probes,
)
from oczy.lm._types import ReservedPosition
from oczy.lm.hf_driver import HFDriver

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_RAW = "raw"
MODE_CHAT = "chat_template"

_RAW_TEMPLATE_LABELS = ("raw_bare", "raw_qa", "raw_question_answer")

_RAW_ROLE_MARKERS = ("Q:", "A:", "Question:", "Answer:")
_CHAT_ROLE_MARKERS = ("<|im_start|>", "<|im_end|>", "system", "user", "assistant")

_RAW_ANSWER_CUES = ("A:", "Answer:")
_CHAT_ANSWER_CUES = ("<|im_start|>assistant",)

# ChatML wrapper used by the mock path (Qwen2.5-Instruct uses ChatML).
_CHATML_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{request}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    """Hex digest of the SHA-256 of *text*."""
    return hashlib.sha256(text.encode()).hexdigest()


def _render_raw_prompts(request: str) -> list[tuple[str, str]]:
    """Return ``(template_label, prompt_text)`` for each raw template.

    Reuses ``_distillation_prompts`` from consolidation_distillation so the
    audited prompts are byte-identical to the ones the frozen instrument
    actually distills and scores.
    """
    prompts = _distillation_prompts(request)
    return list(zip(_RAW_TEMPLATE_LABELS, prompts, strict=True))


def _render_chat_prompt(tokenizer: Any, request: str) -> str:
    """Render the chat-template prompt for *request*.

    Uses the tokenizer's registered chat template with
    ``add_generation_prompt=True`` so the model sees the assistant turn
    opener — this is the registered chat-template mode, not a novel
    prompting variant.
    """
    messages = [{"role": "user", "content": request}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _find_role_boundaries(prompt_text: str, mode: str) -> dict[str, list[int]]:
    """Return a mapping from role marker to char positions in *prompt_text*."""
    markers = _CHAT_ROLE_MARKERS if mode == MODE_CHAT else _RAW_ROLE_MARKERS
    boundaries: dict[str, list[int]] = {}
    for marker in markers:
        positions: list[int] = []
        start = 0
        while True:
            pos = prompt_text.find(marker, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + len(marker)
        if positions:
            boundaries[marker] = positions
    return boundaries


def _detect_answer_cue(
    prompt_text: str,
    mode: str,
    template_label: str,
) -> dict[str, Any]:
    """Detect the answer-prefix cue in the rendered prompt.

    The answer cue is the text that signals the model to start generating
    the answer: ``A:`` / ``Answer:`` for raw templates, or the
    ``<|im_start|>assistant`` marker for chat-template mode.
    """
    cues = _CHAT_ANSWER_CUES if mode == MODE_CHAT else _RAW_ANSWER_CUES
    cue_pos = -1
    cue_text = ""
    for cue in cues:
        pos = prompt_text.rfind(cue)
        if pos != -1:
            cue_pos = pos
            cue_text = cue
            break

    has_cue = cue_pos != -1
    # The bare raw template (just the request) intentionally has no cue.
    bare_ok = template_label == "raw_bare" and mode == MODE_RAW
    mismatch = not has_cue and not bare_ok
    return {
        "answer_cue_text": cue_text,
        "answer_cue_position": cue_pos,
        "answer_cue_present": has_cue,
        "answer_prefix_mismatch": mismatch,
    }


def _check_truncation(
    tokenizer: Any,
    prompt_text: str,
    request: str,
) -> dict[str, Any]:
    """Check whether the request survives tokenisation round-trip."""
    ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    request_in_decoded = request in decoded
    request_in_prompt = request in prompt_text
    return {
        "request_truncated": not request_in_decoded,
        "request_in_prompt": request_in_prompt,
        "decoded_char_length": len(decoded),
        "token_count_with_special": len(ids),
    }


def _first_token_prediction(
    driver: HFDriver,
    prompt: str,
) -> dict[str, Any]:
    """Return ``(token_id, token_text)`` for the greedy first token after *prompt*.

    Honours any active reserved position (teacher path).
    """
    import torch

    effective = driver._apply_reserved_prefix(prompt)
    input_ids = driver._tokenize(effective)
    with torch.no_grad():
        out = driver._model(input_ids=input_ids, use_cache=False)
    token_id = int(out.logits[0, -1, :].argmax().item())
    token_text = driver._tokenizer.decode([token_id])
    return {"first_token_id": token_id, "first_token_text": token_text}


# ---------------------------------------------------------------------------
# Per-example audit
# ---------------------------------------------------------------------------


def _audit_one_example(
    driver: HFDriver,
    tokenizer: Any,
    episode: Any,
    probe: Any,
    mode: str,
    template_label: str,
    prompt_text: str,
) -> dict[str, Any]:
    """Audit one (episode, probe, mode, template) combination.

    Records the full prompt-contract surface: structure, role boundaries,
    correction/request visibility, token counts, truncation, answer-prefix
    cue, first-token prediction, full generation, and correctness under
    both the scoring path (no correction) and the teacher path (with
    correction as reserved position).
    """
    correction = episode.correction_utterance
    request = probe.request
    expected = probe.expected

    # --- Prompt structure (scoring path) ---
    prompt_hash = _sha256(prompt_text)
    prompt_char_length = len(prompt_text)
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    token_count = _token_count(tokenizer, prompt_text)

    # --- Role boundaries ---
    role_boundaries = _find_role_boundaries(prompt_text, mode)

    # --- Answer-prefix cue ---
    cue_info = _detect_answer_cue(prompt_text, mode, template_label)

    # --- Truncation ---
    trunc_info = _check_truncation(tokenizer, prompt_text, request)

    # --- Teacher prompt structure (correction + prompt via reserved position) ---
    # The frozen instrument prepends the correction as a literal prefix.
    teacher_prompt_text = correction + " " + prompt_text
    teacher_hash = _sha256(teacher_prompt_text)
    teacher_token_count = _token_count(tokenizer, teacher_prompt_text)
    correction_present = correction in teacher_prompt_text
    correction_position = teacher_prompt_text.find(correction)
    request_in_teacher = request in teacher_prompt_text
    request_position_in_teacher = teacher_prompt_text.find(request)
    correction_before_request = (
        correction_position != -1
        and request_position_in_teacher != -1
        and correction_position < request_position_in_teacher
    )

    # --- Answer leak: expected answer must NOT be visible in scoring prompt ---
    answer_leak = expected.lower() in prompt_text.lower()

    # --- First token prediction (scoring path, no correction) ---
    driver.clear_reserved_position()
    first_tok_scoring = _first_token_prediction(driver, prompt_text)

    # --- Full prediction (scoring path) ---
    prediction = driver.generate(prompt_text, max_tokens=32)
    scoring_correct = probe_matches(prediction, probe, episode)

    # --- Teacher prediction (with correction as reserved position) ---
    driver.set_reserved_position(cast(Any, ReservedPosition(text=correction)))
    teacher_prediction = driver.generate(request, max_tokens=32)
    driver.clear_reserved_position()
    teacher_correct = probe_matches(teacher_prediction, probe, episode)

    # --- First token correctness ---
    expected_ids = tokenizer.encode(" " + expected, add_special_tokens=False)
    first_token_correct = (
        bool(expected_ids) and first_tok_scoring["first_token_id"] == expected_ids[0]
    )

    # --- Contract issues ---
    issues: list[str] = []
    if mode == MODE_CHAT and "<|im_start|>" not in role_boundaries:
        issues.append("malformed_role_template")
    if not correction_present:
        issues.append("missing_correction")
    if trunc_info["request_truncated"]:
        issues.append("request_truncated")
    if cue_info["answer_prefix_mismatch"]:
        issues.append("answer_prefix_mismatch")
    if answer_leak:
        issues.append("answer_leak")

    return {
        "episode_id": episode.id,
        "probe_id": f"{episode.id}|{probe.request}|{probe.category}",
        "mode": mode,
        "template_label": template_label,
        "model_id": driver.model_id,
        # Prompt structure
        "prompt_text": prompt_text,
        "prompt_hash": prompt_hash,
        "prompt_char_length": prompt_char_length,
        "token_count": token_count,
        "token_ids": token_ids,
        "role_boundaries": role_boundaries,
        # Answer cue
        "answer_cue_text": cue_info["answer_cue_text"],
        "answer_cue_position": cue_info["answer_cue_position"],
        "answer_cue_present": cue_info["answer_cue_present"],
        "answer_prefix_mismatch": cue_info["answer_prefix_mismatch"],
        # Truncation
        "request_truncated": trunc_info["request_truncated"],
        "request_in_prompt": trunc_info["request_in_prompt"],
        "decoded_char_length": trunc_info["decoded_char_length"],
        "token_count_with_special": trunc_info["token_count_with_special"],
        # Teacher prompt structure
        "teacher_prompt_hash": teacher_hash,
        "teacher_token_count": teacher_token_count,
        "correction_present": correction_present,
        "correction_position": correction_position,
        "request_in_teacher": request_in_teacher,
        "correction_before_request": correction_before_request,
        # Answer leak
        "answer_leak": answer_leak,
        # Generation (scoring path)
        "first_token_id": first_tok_scoring["first_token_id"],
        "first_token_text": first_tok_scoring["first_token_text"],
        "first_token_correct": first_token_correct,
        "prediction": prediction,
        "scoring_correct": scoring_correct,
        # Generation (teacher path)
        "teacher_prediction": teacher_prediction,
        "teacher_correct": teacher_correct,
        # Contract issues
        "contract_issues": issues,
        "has_contract_issue": bool(issues),
    }


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------


def _run_audit(
    driver: HFDriver,
    tokenizer: Any,
    stage: Stage,
    dev_ids: set[str],
) -> dict[str, Any]:
    """Run the full DEV-only prompt audit across raw and chat-template modes."""
    records: list[dict[str, Any]] = []

    dev_pairs: list[tuple[Any, Any, str]] = []
    for ep in stage.episodes:
        for probe in ep.probes:
            pid = f"{ep.id}|{probe.request}|{probe.category}"
            if pid in dev_ids:
                dev_pairs.append((ep, probe, pid))

    for ep, probe, _pid in dev_pairs:
        # --- Raw mode: audit each registered template ---
        for template_label, prompt_text in _render_raw_prompts(probe.request):
            record = _audit_one_example(
                driver, tokenizer, ep, probe,
                MODE_RAW, template_label, prompt_text,
            )
            records.append(record)

        # --- Chat-template mode ---
        try:
            chat_prompt = _render_chat_prompt(tokenizer, probe.request)
            record = _audit_one_example(
                driver, tokenizer, ep, probe,
                MODE_CHAT, "chat_template", chat_prompt,
            )
        except Exception as exc:
            record = {
                "episode_id": ep.id,
                "probe_id": f"{ep.id}|{probe.request}|{probe.category}",
                "mode": MODE_CHAT,
                "template_label": "chat_template",
                "model_id": driver.model_id,
                "error": str(exc),
                "contract_issues": ["malformed_role_template"],
                "has_contract_issue": True,
                "scoring_correct": False,
                "teacher_correct": False,
                "first_token_correct": False,
                "answer_prefix_mismatch": True,
                "request_truncated": False,
                "correction_present": False,
                "answer_leak": False,
            }
        records.append(record)

    return _aggregate(records, driver.model_id, stage.name, len(dev_pairs))


def _aggregate(
    records: list[dict[str, Any]],
    model_id: str,
    stage_name: str,
    dev_probe_count: int,
) -> dict[str, Any]:
    """Compute aggregate metrics from per-example records."""
    raw_records = [r for r in records if r.get("mode") == MODE_RAW]
    chat_records = [r for r in records if r.get("mode") == MODE_CHAT]

    raw_correct = sum(1 for r in raw_records if r.get("scoring_correct"))
    raw_total = len(raw_records)
    chat_correct = sum(1 for r in chat_records if r.get("scoring_correct"))
    chat_total = len(chat_records)

    raw_first_tok = sum(1 for r in raw_records if r.get("first_token_correct"))
    chat_first_tok = sum(1 for r in chat_records if r.get("first_token_correct"))

    teacher_correct = sum(1 for r in records if r.get("teacher_correct"))

    issue_count = sum(1 for r in records if r.get("has_contract_issue"))
    malformed = sum(
        1 for r in records if "malformed_role_template" in r.get("contract_issues", [])
    )
    missing_correction = sum(
        1 for r in records if "missing_correction" in r.get("contract_issues", [])
    )
    request_truncated = sum(
        1 for r in records if "request_truncated" in r.get("contract_issues", [])
    )
    answer_prefix_mismatch = sum(
        1 for r in records if "answer_prefix_mismatch" in r.get("contract_issues", [])
    )
    answer_leak = sum(
        1 for r in records if "answer_leak" in r.get("contract_issues", [])
    )

    raw_acc = raw_correct / max(raw_total, 1)
    chat_acc = chat_correct / max(chat_total, 1)
    total = len(records)

    aggregates = {
        "raw_accuracy": raw_acc,
        "chat_template_accuracy": chat_acc,
        "raw_correct": raw_correct,
        "raw_total": raw_total,
        "chat_template_correct": chat_correct,
        "chat_template_total": chat_total,
        "raw_first_token_correct_rate": raw_first_tok / max(raw_total, 1),
        "chat_template_first_token_correct_rate": chat_first_tok / max(chat_total, 1),
        "mode_accuracy_gap": raw_acc - chat_acc,
        "teacher_correct_rate": teacher_correct / max(total, 1),
        "contract_issue_count": issue_count,
        "contract_issue_rate": issue_count / max(total, 1),
        "malformed_count": malformed,
        "missing_correction_count": missing_correction,
        "request_truncated_count": request_truncated,
        "answer_prefix_mismatch_count": answer_prefix_mismatch,
        "answer_leak_count": answer_leak,
    }

    return {
        "records": records,
        "aggregates": aggregates,
        "model_id": model_id,
        "stage": stage_name,
        "dev_probe_count": dev_probe_count,
    }


# ---------------------------------------------------------------------------
# Mock driver (for local testing without a real model)
# ---------------------------------------------------------------------------


def _run_mock_audit(stage: Stage, dev_ids: set[str]) -> dict[str, Any]:
    """Deterministic mock audit that exercises the structural surface only.

    Renders raw prompts and a ChatML wrapper, audits prompt structure, and
    sets all generation/correctness fields to False/empty.  No model is
    loaded; this is for sentinel-format and structural testing only.
    """
    records: list[dict[str, Any]] = []

    dev_pairs: list[tuple[Any, Any, str]] = []
    for ep in stage.episodes:
        for probe in ep.probes:
            pid = f"{ep.id}|{probe.request}|{probe.category}"
            if pid in dev_ids:
                dev_pairs.append((ep, probe, pid))

    for ep, probe, _pid in dev_pairs:
        correction = ep.correction_utterance
        request = probe.request
        expected = probe.expected

        # Raw mode
        for template_label, prompt_text in _render_raw_prompts(request):
            teacher_prompt = correction + " " + prompt_text
            cue_info = _detect_answer_cue(prompt_text, MODE_RAW, template_label)
            role_boundaries = _find_role_boundaries(prompt_text, MODE_RAW)
            records.append({
                "episode_id": ep.id,
                "probe_id": f"{ep.id}|{probe.request}|{probe.category}",
                "mode": MODE_RAW,
                "template_label": template_label,
                "model_id": "mock",
                "prompt_text": prompt_text,
                "prompt_hash": _sha256(prompt_text),
                "prompt_char_length": len(prompt_text),
                "token_count": len(prompt_text.split()),
                "token_ids": [],
                "role_boundaries": role_boundaries,
                "answer_cue_text": cue_info["answer_cue_text"],
                "answer_cue_position": cue_info["answer_cue_position"],
                "answer_cue_present": cue_info["answer_cue_present"],
                "answer_prefix_mismatch": cue_info["answer_prefix_mismatch"],
                "request_truncated": False,
                "request_in_prompt": request in prompt_text,
                "decoded_char_length": len(prompt_text),
                "token_count_with_special": len(prompt_text.split()),
                "teacher_prompt_hash": _sha256(teacher_prompt),
                "teacher_token_count": len(teacher_prompt.split()),
                "correction_present": correction in teacher_prompt,
                "correction_position": teacher_prompt.find(correction),
                "request_in_teacher": request in teacher_prompt,
                "correction_before_request": (
                    teacher_prompt.find(correction) < teacher_prompt.find(request)
                    if correction in teacher_prompt and request in teacher_prompt
                    else False
                ),
                "answer_leak": expected.lower() in prompt_text.lower(),
                "first_token_id": -1,
                "first_token_text": "",
                "first_token_correct": False,
                "prediction": "",
                "scoring_correct": False,
                "teacher_prediction": "",
                "teacher_correct": False,
                "contract_issues": (
                    ["answer_prefix_mismatch"]
                    if cue_info["answer_prefix_mismatch"]
                    else []
                ),
                "has_contract_issue": cue_info["answer_prefix_mismatch"],
            })

        # Chat-template mode (ChatML wrapper, no tokenizer)
        chat_prompt = _CHATML_TEMPLATE.format(request=request)
        teacher_prompt = correction + " " + chat_prompt
        cue_info = _detect_answer_cue(chat_prompt, MODE_CHAT, "chat_template")
        role_boundaries = _find_role_boundaries(chat_prompt, MODE_CHAT)
        records.append({
            "episode_id": ep.id,
            "probe_id": f"{ep.id}|{probe.request}|{probe.category}",
            "mode": MODE_CHAT,
            "template_label": "chat_template",
            "model_id": "mock",
            "prompt_text": chat_prompt,
            "prompt_hash": _sha256(chat_prompt),
            "prompt_char_length": len(chat_prompt),
            "token_count": len(chat_prompt.split()),
            "token_ids": [],
            "role_boundaries": role_boundaries,
            "answer_cue_text": cue_info["answer_cue_text"],
            "answer_cue_position": cue_info["answer_cue_position"],
            "answer_cue_present": cue_info["answer_cue_present"],
            "answer_prefix_mismatch": cue_info["answer_prefix_mismatch"],
            "request_truncated": False,
            "request_in_prompt": request in chat_prompt,
            "decoded_char_length": len(chat_prompt),
            "token_count_with_special": len(chat_prompt.split()),
            "teacher_prompt_hash": _sha256(teacher_prompt),
            "teacher_token_count": len(teacher_prompt.split()),
            "correction_present": correction in teacher_prompt,
            "correction_position": teacher_prompt.find(correction),
            "request_in_teacher": request in teacher_prompt,
            "correction_before_request": (
                teacher_prompt.find(correction) < teacher_prompt.find(request)
                if correction in teacher_prompt and request in teacher_prompt
                else False
            ),
            "answer_leak": expected.lower() in chat_prompt.lower(),
            "first_token_id": -1,
            "first_token_text": "",
            "first_token_correct": False,
            "prediction": "",
            "scoring_correct": False,
            "teacher_prediction": "",
            "teacher_correct": False,
            "contract_issues": [],
            "has_contract_issue": False,
        })

    return _aggregate(records, "mock", stage.name, len(dev_pairs))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _emit_sentinels(audit: dict[str, Any]) -> None:
    """Print METRIC and ASI sentinel lines to stdout."""
    agg = audit["aggregates"]
    print(f"METRIC prompt_contract_raw_accuracy={agg['raw_accuracy']}")
    print(f"METRIC prompt_contract_chat_template_accuracy={agg['chat_template_accuracy']}")
    print(f"ASI prompt_contract_raw_correct={agg['raw_correct']}")
    print(f"ASI prompt_contract_raw_total={agg['raw_total']}")
    print(f"ASI prompt_contract_chat_template_correct={agg['chat_template_correct']}")
    print(f"ASI prompt_contract_chat_template_total={agg['chat_template_total']}")
    print(f"ASI prompt_contract_raw_first_token_correct_rate={agg['raw_first_token_correct_rate']}")
    print(f"ASI prompt_contract_chat_template_first_token_correct_rate={agg['chat_template_first_token_correct_rate']}")
    print(f"ASI prompt_contract_mode_accuracy_gap={agg['mode_accuracy_gap']}")
    print(f"ASI prompt_contract_teacher_correct_rate={agg['teacher_correct_rate']}")
    print(f"ASI prompt_contract_contract_issue_count={agg['contract_issue_count']}")
    print(f"ASI prompt_contract_contract_issue_rate={agg['contract_issue_rate']}")
    print(f"ASI prompt_contract_malformed_count={agg['malformed_count']}")
    print(f"ASI prompt_contract_missing_correction_count={agg['missing_correction_count']}")
    print(f"ASI prompt_contract_request_truncated_count={agg['request_truncated_count']}")
    print(f"ASI prompt_contract_answer_prefix_mismatch_count={agg['answer_prefix_mismatch_count']}")
    print(f"ASI prompt_contract_answer_leak_count={agg['answer_leak_count']}")
    print(f"ASI prompt_contract_model_id={audit['model_id']}")
    print(f"ASI prompt_contract_stage={audit['stage']}")
    print(f"ASI prompt_contract_dev_probe_count={audit['dev_probe_count']}")


def _emit_audit_json(audit: dict[str, Any], output_path: str | None) -> None:
    """Emit the machine-readable JSON audit to *output_path* or stderr."""
    payload = json.dumps(audit, indent=2, default=str)
    if output_path:
        with open(output_path, "w") as f:
            f.write(payload)
    else:
        print(payload, file=sys.stderr)


def _report_real_driver_failure(exc: Exception, tb: str) -> None:
    """Print fail-closed diagnostics for a real-driver failure."""
    print("ASI real_driver=failed")
    print(f"ASI real_driver_error_type={type(exc).__name__}")
    msg = str(exc).replace("\n", " ")
    print(f"ASI real_driver_error_message={msg}")
    if tb and tb.strip() != "NoneType: None":
        print("ASI real_driver_traceback:")
        for line in tb.rstrip().splitlines():
            print(f"  {line}")
    print(
        "ASI real_driver_hint=verify Qwen2.5-0.5B-Instruct is available "
        "(OCZY_MODEL_DIR or HF cache), that torch/transformers are installed, "
        "then retry; or run with --driver mock for the structural floor."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="R18 prompt contract diagnostic (DEV-only)"
    )
    parser.add_argument(
        "--driver",
        choices=["mock", "real"],
        default="real",
        help="real = HFDriver with Qwen2.5-0.5B (default); mock = structural floor",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage_0_grounding",
        help="target stage (default: stage_0_grounding)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="optional path for JSON audit output (default: stderr)",
    )
    args = parser.parse_args(argv)

    # build_curriculum verifies eval manifest integrity internally.
    stage = build_curriculum(stage_names=(args.stage,))[0]
    dev_ids, _holdout_ids = split_probes(stage, fraction=0.3, salt="v2")

    if args.driver == "real":
        try:
            driver = HFDriver.load()
        except Exception as exc:
            _report_real_driver_failure(exc, traceback.format_exc())
            return 1
        try:
            tokenizer = driver._tokenizer
            audit = _run_audit(driver, tokenizer, stage, dev_ids)
        except Exception as exc:
            _report_real_driver_failure(exc, traceback.format_exc())
            driver.close()
            return 1
        finally:
            driver.clear_reserved_position()
            driver.close()
    else:
        audit = _run_mock_audit(stage, dev_ids)

    audit["stage"] = args.stage
    _emit_sentinels(audit)
    _emit_audit_json(audit, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
