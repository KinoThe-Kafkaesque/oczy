"""Research 18 diagnostic: teacher ceiling probe (DEV-only, mechanism-level).

Pre-registered experiment: research/18-consolidation-as-distillation.md

This diagnostic explains the failed teacher prerequisite by measuring the
exact DEV probes that limit the teacher ceiling.  It compares the only two
registered prompting modes — raw per-fact-prefix teacher and the single
registered chat-template fallback — against vanilla, on the DEV split only.

It does NOT:
  - access holdout probes (DEV-only by construction);
  - introduce new prompt variants (only the two registered modes + vanilla);
  - alter the 0.2 gate or draw any H-DISTILL verdict;
  - modify consolidation_distillation, eval/, lanes/, scoring, or curriculum.

Output is machine-readable: METRIC/ASI sentinel lines on stdout, per-example
AUDIT lines on stderr.  Exits nonzero on real-driver failure (no mock fallback).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, cast

from oczy.eval_v2.scoring import probe_matches
from oczy.experiments.organism_curriculum.dataset import (
    Stage,
    build_curriculum,
    default_stages_dir,
    split_probes,
)
from oczy.lm._types import ReservedPosition
from oczy.lm.hf_driver import HFDriver

# The unchanged pre-registered validity gate from research/18.
GATE_THRESHOLD: float = 0.2

# The only stage this diagnostic examines.
TARGET_STAGE: str = "stage_0_grounding"


# ---------------------------------------------------------------------------
# Hashing / identification
# ---------------------------------------------------------------------------


def _hash_manifest() -> str:
    """Return SHA-256 over the concatenated eval/v2 MANIFEST.json contents."""
    manifest_path = default_stages_dir() / "MANIFEST.json"
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _hash_model(driver: HFDriver) -> str:
    """Return a stable identity hash for the loaded model.

    Uses model_id plus the config JSON (architecture, vocab, layers) so that
    two different checkpoints under the same name are distinguished.
    """
    cfg = getattr(driver._model, "config", None)
    cfg_dict: dict[str, Any]
    if cfg is not None:
        cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else {"repr": repr(cfg)}
    else:
        cfg_dict = {}
    payload = json.dumps(
        {"model_id": driver.model_id, "config": cfg_dict},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Prompting modes
# ---------------------------------------------------------------------------


def _build_chat_prompt(tokenizer: Any, request: str) -> str:
    """Wrap *request* in the model's registered chat template (single fallback).

    Uses ``tokenizer.apply_chat_template`` with a single user message.  This is
    the only registered fallback prompting mode from the research/18 spec —
    no new prompt variant is introduced.
    """
    messages = [{"role": "user", "content": request}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Per-mode evaluation
# ---------------------------------------------------------------------------


def _eval_mode(
    driver: HFDriver,
    dev_episodes: list[Any],
    dev_ids: set[str],
    mode: str,
) -> list[dict[str, Any]]:
    """Evaluate one prompting mode on DEV probes.

    Returns a list of per-example dicts with keys:
        episode_id, probe_request, expected_label, prediction, correct, mode

    Modes:
      - "vanilla":      raw probe request, no prefix.
      - "raw_prefix":   ReservedPosition(correction_utterance) prefix — the
                        raw per-fact-prefix teacher from the existing runner.
      - "chat_template": tokenizer.apply_chat_template wrapping the request —
                         the single registered chat-template fallback.
    """
    tokenizer = driver._tokenizer
    results: list[dict[str, Any]] = []

    for ep in dev_episodes:
        for probe in ep.probes:
            pid = f"{ep.id}|{probe.request}|{probe.category}"
            if pid not in dev_ids:
                continue

            if mode == "vanilla":
                answer = driver.generate(probe.request, max_tokens=32)
            elif mode == "raw_prefix":
                driver.set_reserved_position(
                    cast(Any, ReservedPosition(text=ep.correction_utterance))
                )
                answer = driver.generate(probe.request, max_tokens=32)
                driver.clear_reserved_position()
            elif mode == "chat_template":
                chat_prompt = _build_chat_prompt(tokenizer, probe.request)
                answer = driver.generate(chat_prompt, max_tokens=32)
            else:
                raise ValueError(f"unknown prompting mode: {mode!r}")

            correct = probe_matches(answer, probe, ep)
            results.append(
                {
                    "episode_id": ep.id,
                    "probe_request": probe.request,
                    "expected_label": probe.expected,
                    "prediction": answer,
                    "correct": correct,
                    "mode": mode,
                }
            )

    return results


def _accuracy(records: list[dict[str, Any]]) -> float:
    """Return fraction of correct records (0.0 if empty)."""
    if not records:
        return 0.0
    return sum(1 for r in records if r["correct"]) / len(records)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_audit(records: list[dict[str, Any]]) -> None:
    """Emit per-example AUDIT lines to stderr (never training data)."""
    for r in records:
        # Truncate prediction to keep lines readable; full text is not needed
        # for mechanism diagnosis.
        pred = r["prediction"].replace("\n", "\\n")
        if len(pred) > 120:
            pred = pred[:117] + "..."
        print(
            f"AUDIT episode_id={r['episode_id']} "
            f"probe_request={r['probe_request']!r} "
            f"expected_label={r['expected_label']!r} "
            f"prediction={pred!r} "
            f"correct={r['correct']} "
            f"mode={r['mode']}",
            file=sys.stderr,
        )


def _print_results(
    vanilla_acc: float,
    raw_acc: float,
    chat_acc: float,
    raw_delta: float,
    chat_delta: float,
    n_probes: int,
    model_id: str,
    model_sha: str,
    manifest_sha: str,
    stage_name: str,
) -> None:
    """Emit machine-readable METRIC/ASI sentinel lines to stdout."""
    print(f"METRIC vanilla_dev_accuracy={vanilla_acc}")
    print(f"METRIC raw_prefix_dev_accuracy={raw_acc}")
    print(f"METRIC chat_template_dev_accuracy={chat_acc}")
    print(f"METRIC raw_prefix_delta={raw_delta}")
    print(f"METRIC chat_template_delta={chat_delta}")
    print(f"METRIC gate_threshold={GATE_THRESHOLD}")
    print(f"METRIC raw_prefix_gate_pass={raw_delta >= GATE_THRESHOLD}")
    print(f"METRIC chat_template_gate_pass={chat_delta >= GATE_THRESHOLD}")

    print(f"ASI model_id={model_id}")
    print(f"ASI model_sha256={model_sha}")
    print(f"ASI manifest_sha256={manifest_sha}")
    print(f"ASI stage={stage_name}")
    print(f"ASI n_dev_probes={n_probes}")
    print("ASI split=fraction=0.3_salt=v2_dev_only")
    print("ASI verdict=NONE")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="R18 teacher ceiling diagnostic (DEV-only, mechanism-level)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=TARGET_STAGE,
        help=f"target stage (default: {TARGET_STAGE})",
    )
    args = parser.parse_args(argv)

    # Build curriculum (verifies eval manifest integrity internally) and
    # DEV/holdout split — use DEV only.
    stage: Stage = build_curriculum(stage_names=(args.stage,))[0]
    manifest_sha = _hash_manifest()
    dev_ids, _holdout_ids = split_probes(stage, fraction=0.3, salt="v2")

    dev_episode_ids = {pid.split("|")[0] for pid in dev_ids}
    dev_episodes = [ep for ep in stage.episodes if ep.id in dev_episode_ids]

    # Fail-closed: load real HFDriver; no mock fallback for remote scientific runs.
    try:
        driver = HFDriver.load()
    except Exception as exc:
        print("METRIC driver_status=unavailable", file=sys.stderr)
        print(f"ASI error=driver_load_failed: {exc}", file=sys.stderr)
        return 1

    model_sha = _hash_model(driver)
    model_id = driver.model_id

    try:
        # Evaluate all three modes on DEV.
        vanilla_records = _eval_mode(driver, dev_episodes, dev_ids, "vanilla")
        raw_records = _eval_mode(driver, dev_episodes, dev_ids, "raw_prefix")
        chat_records = _eval_mode(driver, dev_episodes, dev_ids, "chat_template")

        # Per-example audit output (stderr — never training).
        _print_audit(vanilla_records)
        _print_audit(raw_records)
        _print_audit(chat_records)

        # Aggregate accuracies and deltas.
        vanilla_acc = _accuracy(vanilla_records)
        raw_acc = _accuracy(raw_records)
        chat_acc = _accuracy(chat_records)
        raw_delta = raw_acc - vanilla_acc
        chat_delta = chat_acc - vanilla_acc
        n_probes = len(vanilla_records)

        _print_results(
            vanilla_acc=vanilla_acc,
            raw_acc=raw_acc,
            chat_acc=chat_acc,
            raw_delta=raw_delta,
            chat_delta=chat_delta,
            n_probes=n_probes,
            model_id=model_id,
            model_sha=model_sha,
            manifest_sha=manifest_sha,
            stage_name=args.stage,
        )
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
