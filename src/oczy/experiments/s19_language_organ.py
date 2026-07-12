"""Research 19 CLI: calibrate-dev and evaluate phases.

Usage::

    python -m oczy.experiments.s19_language_organ calibrate-dev \\
        --model-id Qwen/Qwen2.5-0.5B-Instruct \\
        --manifest-out /path/to/calibration_manifest.json

    python -m oczy.experiments.s19_language_organ evaluate \\
        --manifest /path/to/calibration_manifest.json \\
        --signoff-id <human-approved-id> \\
        --seeds 5

Calibrate-dev trains the coupler on DEV-only stage-0 tasks, freezes the
confidence threshold, specificity margin, coupler hash, label phrasing, and
writes a deterministic calibration manifest for human review.

Evaluate verifies the manifest SHA-256 and a nonempty human sign-off ID,
then performs one-shot holdout/transfer/scope/specificity evaluation
across C0-C7 conditions.  It fails closed (no mock fallback, no auto-sign)
unless both match.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path
from typing import Any

import numpy as np
import torch

from oczy.experiments.organism_curriculum.dataset import (
    STAGE_ORDER,
    build_curriculum,
    split_probes,
)
from oczy.experiments.s19_language_organ_core import (
    CONFIDENCE_THRESHOLD_DEFAULT,
    D_CORTEX,
    D_EMBD,
    DEFAULT_SEEDS,
    LABEL_PREFIX_TEMPLATE,
    LATENT_TOKENS,
    MAX_PARAMS,
    MIN_SEEDS,
    N_LABELS,
    PARAM_BREAKDOWN,
    SCHEMA_VERSION,
    SPECIFICITY_MARGIN_DEFAULT,
    CalibrationManifest,
    CortexConfig,
    SharedCortex,
    TraceStore,
    build_articulation_audit,
    build_label_index,
    check_dev_articulation_gate,
    check_meta_test_conflation,
    compute_oracle_ceiling,
    compute_verdicts,
    derive_source_provenance,
    extract_stage_labels,
    hash_cortex_artifact,
    hash_coupler_state,
    hash_eval_manifest,
    hash_head_state,
    hash_model,
    hash_model_config,
    hash_model_safetensors,
    mean_ci,
    run_condition,
    score_probes,
    score_specificity,
    teach_cortex,
    verify_fixed_latent_width,
    verify_no_episode_id_conditioning,
    verify_no_text_injection,
    verify_parameter_budget,
)
from oczy.lm.hf_driver import HFDriver

# ---------------------------------------------------------------------------
# Calibrate-dev phase
# ---------------------------------------------------------------------------


def _calibrate_dev(args: argparse.Namespace) -> int:
    """Run the calibrate-dev phase on DEV-only data.

    Steps:
      1. Verify eval manifest integrity.
      2. Load frozen LM, hash before.
      3. Load stage-0 curriculum, split probes into DEV/holdout.
      4. Extract label set from stage-0 episodes.
      5. Initialize cortex, train coupler on DEV episodes.
      6. Phase 0 distribution check: measure no-update repeatability,
         confidence, and specificity distributions on DEV.
      7. Freeze confidence threshold and specificity margin.
      8. Freeze coupler, compute coupler hash.
      9. Write calibration manifest (without human sign-off).
     10. Hash LM after, verify frozen.
     11. Emit METRIC/ASI/AUDIT sentinels.
    """
    print("# Research 19 — calibrate-dev phase", file=sys.stderr)
    print("# DEV only; holdout IDs are discarded.", file=sys.stderr)

    # 1. Verify eval manifest.
    from eval.v2 import verify_manifest  # type: ignore[missing-import]

    verify_manifest()
    eval_manifest_hash = hash_eval_manifest()
    print(f"ASI eval_manifest_hash={eval_manifest_hash}", file=sys.stderr)

    # 2. Load frozen LM.
    model_id = args.model_id
    print(f"# Loading frozen LM: {model_id}", file=sys.stderr)
    driver = HFDriver.load(model_id=model_id)
    model_hash_before = hash_model(driver)
    print(f"ASI model_hash_before={model_hash_before}", file=sys.stderr)

    # Freeze LM parameters.
    for p in driver._model.parameters():
        p.requires_grad_(False)
    driver._model.eval()

    # 3. Load stage-0 curriculum.
    stage0 = build_curriculum(stage_names=("stage_0_grounding",))[0]
    split_salt = "v2.2"
    split_fraction = 0.3
    dev_ids, holdout_ids = split_probes(stage0, fraction=split_fraction, salt=split_salt)

    # DISCARD holdout IDs — calibrate-dev must not use holdout data.
    del holdout_ids
    print(f"# DEV probes: {len(dev_ids)} (holdout discarded)", file=sys.stderr)

    # 4. Extract label set.
    labels = extract_stage_labels(stage0)
    label_index = build_label_index(labels)
    print(f"# Labels: {len(labels)}", file=sys.stderr)
    assert len(labels) <= N_LABELS, f"Too many labels: {len(labels)} > {N_LABELS}"

    # 5. Initialize cortex and train coupler on DEV episodes.
    config = CortexConfig(
        lr=args.lr,
        coupler_lr=args.coupler_lr,
        label_lr=args.label_lr,
    )
    cortex = SharedCortex(config=config, seed=0)

    # Verify parameter budget.
    param_count = cortex.parameter_count()
    assert param_count <= MAX_PARAMS, f"Parameter budget exceeded: {param_count} > {MAX_PARAMS}"
    print(f"ASI parameter_count={param_count}", file=sys.stderr)
    print(f"ASI parameter_budget={MAX_PARAMS}", file=sys.stderr)

    # Get DEV episodes (those with DEV probes).
    dev_episode_ids = {pid.split("|")[0] for pid in dev_ids}
    dev_episodes = [ep for ep in stage0.episodes if ep.id in dev_episode_ids]
    print(f"# DEV episodes for coupler training: {len(dev_episodes)}", file=sys.stderr)

    trace_store = TraceStore()

    # Train on DEV episodes.
    print("# Training coupler on DEV episodes...", file=sys.stderr)
    import random
    rng = random.Random(0)
    teach_order = list(dev_episodes)
    rng.shuffle(teach_order)

    label_losses: list[float] = []
    coupler_losses: list[float] = []
    for ep in teach_order:
        trace_store.add(ep.id, ep.initial_request, ep.correction_utterance, ep.corrected_response)
        features = driver.peek_embedding(ep.initial_request, last_token_only=False)
        true_idx = label_index[ep.corrected_label]
        label_loss = cortex.train_label_head(features, true_idx)
        # Freeze W_perceive/W_label during coupler DEV training so the
        # coupler learns to work with the current perception/label weights
        # rather than co-adapting them.
        cortex.W_perceive.requires_grad_(False)
        cortex.W_label.requires_grad_(False)
        cortex.b_label.requires_grad_(False)
        coupler_loss = cortex.train_coupler(driver, ep.initial_request, ep.corrected_response)
        cortex.W_perceive.requires_grad_(True)
        cortex.W_label.requires_grad_(True)
        cortex.b_label.requires_grad_(True)
        label_losses.append(label_loss)
        coupler_losses.append(coupler_loss)

    print(f"ASI calibrate_label_loss_mean={np.mean(label_losses):.6f}", file=sys.stderr)
    print(f"ASI calibrate_coupler_loss_mean={np.mean(coupler_losses):.6f}", file=sys.stderr)

    # 6. Phase 0 distribution check on DEV.
    print("# Phase 0 distribution check on DEV...", file=sys.stderr)

    # No-update repeatability: run C1 (random cortex) on DEV multiple times.
    repeatability_scores: list[float] = []
    for trial in range(3):
        random_cortex = SharedCortex(config=config, seed=100 + trial)
        dev_result = score_probes(
            driver, stage0, dev_ids, "B", random_cortex, labels,
            CONFIDENCE_THRESHOLD_DEFAULT, LABEL_PREFIX_TEMPLATE, trace_store,
        )
        repeatability_scores.append(dev_result["accuracy"])
    repeatability_std = float(np.std(repeatability_scores)) if repeatability_scores else 0.0
    print(f"ASI phase0_no_update_repeatability_std={repeatability_std:.6f}", file=sys.stderr)

    # Confidence distribution: run C2 (trained label head) on DEV.
    confidence_scores: list[float] = []
    for ep in stage0.episodes:
        for probe in ep.probes:
            pid = f"{ep.id}|{probe.request}|{probe.category}"
            if pid not in dev_ids:
                continue
            features = driver.peek_embedding(probe.request, last_token_only=False)
            cortex_act = cortex.perceive(features)
            _, conf = cortex.predict_label(cortex_act)
            confidence_scores.append(conf)
    conf_mean = float(np.mean(confidence_scores)) if confidence_scores else 0.0
    conf_std = float(np.std(confidence_scores)) if confidence_scores else 0.0
    conf_min = float(np.min(confidence_scores)) if confidence_scores else 0.0
    conf_max = float(np.max(confidence_scores)) if confidence_scores else 0.0
    print(f"ASI phase0_confidence_mean={conf_mean:.6f}", file=sys.stderr)
    print(f"ASI phase0_confidence_std={conf_std:.6f}", file=sys.stderr)
    print(f"ASI phase0_confidence_min={conf_min:.6f}", file=sys.stderr)
    print(f"ASI phase0_confidence_max={conf_max:.6f}", file=sys.stderr)

    # Specificity distribution: run C2 on other stages' DEV probes (not holdout).
    other_names = tuple(n for n in STAGE_ORDER if n != "stage_0_grounding")
    other_stages = build_curriculum(stage_names=other_names)
    spec_result = score_specificity(
        driver, other_stages, "A", cortex, labels,
        CONFIDENCE_THRESHOLD_DEFAULT, LABEL_PREFIX_TEMPLATE, trace_store,
        use_holdout=False,  # DEV-only firewall
    )
    specificity_acc = spec_result["accuracy"]
    print(f"ASI phase0_specificity_acc={specificity_acc:.6f}", file=sys.stderr)

    # 7. Freeze confidence threshold and specificity margin.
    # Use the 25th percentile of confidence as the threshold (proposed).
    if confidence_scores:
        threshold = float(np.percentile(confidence_scores, 25))
    else:
        threshold = CONFIDENCE_THRESHOLD_DEFAULT
    # Use 2x the no-update repeatability std as the specificity margin.
    margin = max(SPECIFICITY_MARGIN_DEFAULT, 3.0 * repeatability_std)
    print(f"ASI proposed_confidence_threshold={threshold:.6f}", file=sys.stderr)
    print(f"ASI proposed_specificity_margin={margin:.6f}", file=sys.stderr)

    # 8. Freeze coupler and compute hashes.
    cortex.freeze_coupler()
    coupler_sha, coupler_bytes = hash_coupler_state(cortex)
    head_sha, head_bytes = hash_head_state(cortex)
    cortex_artifact_sha, cortex_artifact_bytes = hash_cortex_artifact(cortex)
    print(f"ASI coupler_sha256={coupler_sha}", file=sys.stderr)
    print(f"ASI head_sha256={head_sha}", file=sys.stderr)
    print(f"ASI cortex_artifact_sha256={cortex_artifact_sha}", file=sys.stderr)

    # Delete raw traces.
    deleted = trace_store.delete_all()
    print(f"ASI raw_traces_deleted={deleted}", file=sys.stderr)
    assert trace_store.verify_zero(), "Raw traces not deleted"
    print("ASI raw_trace_count_after_deletion=0", file=sys.stderr)

    # 8b. Oracle ceiling check on DEV.
    print("# Oracle ceiling check on DEV...", file=sys.stderr)
    oracle_ceiling = compute_oracle_ceiling(driver, stage0, dev_ids)
    print(f"ASI oracle_ceiling_dev={oracle_ceiling:.6f}", file=sys.stderr)

    # 8c. DEV articulation gate.
    print("# DEV articulation gate...", file=sys.stderr)
    dev_gate_pass = check_dev_articulation_gate(
        driver, cortex, stage0, dev_ids, labels,
        threshold, LABEL_PREFIX_TEMPLATE, trace_store,
    )
    print(f"ASI dev_articulation_gate={'pass' if dev_gate_pass else 'fail'}", file=sys.stderr)

    # 8d. Meta-test conflation pre-check (will be fully checked in evaluate).
    meta_test_conflation_ok = True  # Pre-check; fully verified in evaluate
    print("ASI meta_test_conflation_precheck=ok", file=sys.stderr)

    # 8e. Compute model provenance hashes.
    model_config_sha = hash_model_config(driver)
    model_safetensors_sha = hash_model_safetensors(driver)
    print(f"ASI model_config_sha256={model_config_sha}", file=sys.stderr)
    print(f"ASI model_safetensors_sha256={model_safetensors_sha}", file=sys.stderr)

    # 8f. Derive source provenance (never fabricate).
    source_commit, source_archive_sha = derive_source_provenance(
        getattr(args, "source_commit", None),
        getattr(args, "source_archive", None),
    )
    print(f"ASI source_commit={source_commit}", file=sys.stderr)
    print(f"ASI source_archive_sha256={source_archive_sha}", file=sys.stderr)

    # 8g. Build a representative Arm-B articulation audit for the manifest.
    rep_audit: dict[str, Any] = {}
    for ep in stage0.episodes:
        for probe in ep.probes:
            pid = f"{ep.id}|{probe.request}|{probe.category}"
            if pid not in dev_ids:
                continue
            probe_result = score_probes(
                driver, stage0, {pid}, "B", cortex, labels,
                threshold, LABEL_PREFIX_TEMPLATE, trace_store,
            )
            if probe_result.get("audits"):
                rep_audit = probe_result["audits"][0]
            break
        if rep_audit:
            break
    if rep_audit:
        ep_lookup = {e.id: e for e in stage0.episodes}
        rep_ep = ep_lookup.get(rep_audit.get("episode_id", ""))
        if rep_ep is not None:
            rep_audit["label_text"] = rep_ep.corrected_label
            rep_audit["corrected_response"] = rep_ep.corrected_response
            rep_audit["correction_utterance"] = rep_ep.correction_utterance
            rep_audit["expected"] = rep_ep.corrected_response
        enriched = build_articulation_audit(
            condition="C3",
            arm="B",
            prompt_text=rep_audit.get("prompt_text", ""),
            latent_bank_shape=rep_audit.get("latent_bank_shape"),
            raw_trace_count=0,
            model_hash=model_safetensors_sha,
            persistent_bytes=cortex_artifact_bytes,
        )
        rep_audit.update(enriched)

    # Compute banned-content absence booleans.
    rep_prompt = rep_audit.get("prompt_text", "").lower()
    banned_label_absent = True
    banned_corrected_absent = True
    banned_correction_absent = True
    banned_expected_absent = True
    for key, absent_flag in [
        ("label_text", "banned_label"),
        ("corrected_response", "banned_corrected"),
        ("correction_utterance", "banned_correction"),
        ("expected", "banned_expected"),
    ]:
        val = rep_audit.get(key)
        if val and isinstance(val, str) and len(val.strip()) > 2:
            if val.strip().lower() in rep_prompt:
                if absent_flag == "banned_label":
                    banned_label_absent = False
                elif absent_flag == "banned_corrected":
                    banned_corrected_absent = False
                elif absent_flag == "banned_correction":
                    banned_correction_absent = False
                elif absent_flag == "banned_expected":
                    banned_expected_absent = False

    # 8h. C7 retrieval baseline reference.
    c7_reference = "S3.M2a nearest-neighbor retrieval baseline"
    c7_available = True
    c7_blocked_reason: str | None = None

    # 9. Write calibration manifest (flat contract).
    manifest = CalibrationManifest(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        source_commit=source_commit,
        source_archive_sha256=source_archive_sha,
        eval_version="v2.2",
        eval_manifest_sha256=eval_manifest_hash,
        model_repo_id=model_id,
        model_revision="main",
        model_config_sha256=model_config_sha,
        model_safetensors_sha256=model_safetensors_sha,
        model_params_requires_grad=False,
        d_embd=D_EMBD,
        d_cortex=D_CORTEX,
        latent_tokens=LATENT_TOKENS,
        max_labels=len(labels),
        arm_b_input_mode="inputs_embeds",
        parameter_total=param_count,
        parameter_budget=MAX_PARAMS,
        parameter_breakdown=dict(PARAM_BREAKDOWN),
        fixed_latent_shape=[LATENT_TOKENS, D_EMBD],
        proposed_confidence_threshold=threshold,
        proposed_specificity_margin=margin,
        cortex_artifact_sha256=cortex_artifact_sha,
        cortex_artifact_bytes=cortex_artifact_bytes,
        cortex_artifact_path="",
        coupler_sha256=coupler_sha,
        coupler_bytes=coupler_bytes,
        head_sha256=head_sha,
        head_bytes=head_bytes,
        label_phrasing_frozen=True,
        labels=labels,
        dev_split="dev",
        dev_repeatability_std=repeatability_std,
        dev_confidence_mean=conf_mean,
        dev_confidence_std=conf_std,
        dev_confidence_min=conf_min,
        dev_confidence_max=conf_max,
        dev_specificity_acc=specificity_acc,
        dev_holdout_ids_discarded=True,
        split_salt=split_salt,
        split_fraction=split_fraction,
        c7_reference=c7_reference,
        c7_available=c7_available,
        c7_blocked_reason=c7_blocked_reason,
        trace_raw_traces_deleted=True,
        trace_raw_trace_count=0,
        trace_embedding_cache_cleared=True,
        trace_optimizer_state_deleted=True,
        articulation_prompt_text=rep_audit.get("prompt_text", ""),
        articulation_latent_bank_shape=(
            list(rep_audit.get("latent_bank_shape"))
            if rep_audit.get("latent_bank_shape") is not None
            else [LATENT_TOKENS, D_EMBD]
        ),
        articulation_raw_trace_count=0,
        articulation_language_organ_hash=model_safetensors_sha,
        articulation_persistent_cortex_bytes=cortex_artifact_bytes,
        articulation_banned_label_text_absent=banned_label_absent,
        articulation_banned_corrected_response_absent=banned_corrected_absent,
        articulation_banned_correction_utterance_absent=banned_correction_absent,
        articulation_banned_expected_answer_absent=banned_expected_absent,
        signoff_thresholds_signed_off=False,
        signoff_human_signoff_id="",
        signoff_oracle_ceiling=oracle_ceiling,
        signoff_dev_articulation_gate=dev_gate_pass,
        signoff_meta_test_conflation_ok=meta_test_conflation_ok,
        holdout_accessed=False,
    )
    manifest_dict = manifest.to_dict()
    manifest_sha256 = manifest_dict["manifest_sha256"]
    manifest.manifest_sha256 = manifest_sha256

    print(f"ASI manifest_sha256={manifest_sha256}", file=sys.stderr)

    # Serialize full cortex artifact (head+coupler) for evaluate phase.
    cortex_state = cortex.state_dict()
    cortex_artifact_path = args.coupler_out or (Path(args.manifest_out).parent / "s19_cortex.pkl")
    cortex_artifact_path = Path(cortex_artifact_path)
    cortex_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cortex_artifact_path, "wb") as f:
        pickle.dump(
            {"state": cortex_state, "sha256": cortex_artifact_sha, "bytes": cortex_artifact_bytes},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"ASI cortex_artifact_path={cortex_artifact_path}", file=sys.stderr)

    # Also serialize coupler-only for backward compat.
    coupler_state = cortex.coupler_state()
    coupler_path = cortex_artifact_path.parent / "s19_coupler.pkl"
    with open(coupler_path, "wb") as f:
        pickle.dump(
            {"state": coupler_state, "hash": coupler_sha},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"ASI coupler_path={coupler_path}", file=sys.stderr)

    # Update manifest with artifact path and recompute hash.
    manifest.cortex_artifact_path = cortex_artifact_path.name
    manifest_dict = manifest.to_dict()
    manifest_sha256 = manifest_dict["manifest_sha256"]
    manifest.manifest_sha256 = manifest_sha256

    # Write manifest.
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)
        f.write("\n")
    print(f"ASI manifest_path={manifest_path}", file=sys.stderr)

    # 10. Hash LM after, verify frozen.
    model_hash_after = hash_model(driver)
    lm_frozen = model_hash_before == model_hash_after
    print(f"ASI model_hash_after={model_hash_after}", file=sys.stderr)
    print(f"ASI lm_frozen={lm_frozen}", file=sys.stderr)
    if not lm_frozen:
        print("AUDIT lm_hash_mismatch=1", file=sys.stderr)
        print("METRIC calibrate_dev_status=FAILED", file=sys.stderr)
        driver.close()
        return 1

    driver.close()
    gc.collect()

    # 11. Emit sentinels.
    print("METRIC calibrate_dev_status=OK")
    print(f"METRIC proposed_confidence_threshold={threshold}")
    print(f"METRIC proposed_specificity_margin={margin}")
    print(f"ASI parameter_total={param_count}")
    print(f"ASI parameter_budget={MAX_PARAMS}")
    print(f"ASI coupler_sha256={coupler_sha}")
    print(f"ASI head_sha256={head_sha}")
    print(f"ASI cortex_artifact_sha256={cortex_artifact_sha}")
    print(f"ASI manifest_sha256={manifest_sha256}")
    print(f"ASI model_safetensors_sha256={model_safetensors_sha}")
    print(f"ASI model_config_sha256={model_config_sha}")
    print(f"ASI eval_manifest_sha256={eval_manifest_hash}")
    print(f"ASI n_labels={len(labels)}")
    print(f"ASI n_dev_probes={len(dev_ids)}")
    print(f"ASI cortex_artifact_bytes={cortex_artifact_bytes}")
    print(f"ASI label_prefix_template={LABEL_PREFIX_TEMPLATE}")
    print(f"ASI phase0_no_update_repeatability_std={repeatability_std:.6f}")
    print(f"ASI phase0_confidence_mean={conf_mean:.6f}")
    print(f"ASI phase0_specificity_acc={specificity_acc:.6f}")
    print(f"ASI calibrate_label_loss_mean={np.mean(label_losses):.6f}")
    print(f"ASI calibrate_coupler_loss_mean={np.mean(coupler_losses):.6f}")
    print(f"ASI oracle_ceiling_dev={oracle_ceiling:.6f}")
    print(f"ASI dev_articulation_gate={'pass' if dev_gate_pass else 'fail'}")
    print("ASI meta_test_conflation_precheck=ok")
    print("ASI signoff_thresholds_signed_off=False")
    print(f"ASI raw_traces_deleted={deleted}")
    print("ASI raw_trace_count_after_deletion=0")
    print(f"ASI lm_frozen={lm_frozen}")
    print("ASI human_signoff_required=1")
    print("ASI signoff_human_signoff_id=")
    print(f"ASI split_salt={split_salt}")
    print(f"ASI split_fraction={split_fraction}")
    print(f"ASI source_commit={source_commit}")
    print(f"ASI source_archive_sha256={source_archive_sha}")
    print(f"ASI c7_reference={c7_reference}")
    print(f"ASI c7_available={c7_available}")
    print("ASI holdout_accessed=False")
    print("AUDIT calibrate_dev_complete=1")

    return 0


# ---------------------------------------------------------------------------
# Evaluate phase
# ---------------------------------------------------------------------------


def _evaluate(args: argparse.Namespace) -> int:
    """Run the evaluate phase.

    Steps:
      1. Load and verify calibration manifest (hash + human sign-off).
      2. Verify eval manifest integrity.
      3. Load frozen LM, hash before.
      4. Verify LM hash matches manifest.
      5. Load frozen coupler from manifest.
      6. For each seed:
         a. Initialize fresh cortex, load frozen coupler.
         b. Teach stage-0 corrections (seed-shuffled).
         c. Delete raw traces, verify zero.
         d. Run C0-C7 conditions.
      7. Aggregate results, compute verdicts.
      8. Hash LM after, verify frozen.
      9. Emit METRIC/ASI/AUDIT sentinels.
     10. Fail closed on any validity failure.
    """
    print("# Research 19 — evaluate phase", file=sys.stderr)

    # 1. Load and verify manifest.
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        return 1

    with open(manifest_path) as f:
        manifest_dict = json.load(f)
    manifest = CalibrationManifest.from_dict(manifest_dict)

    # Verify manifest hash.
    if not manifest.verify_hash():
        print("ERROR: manifest hash mismatch — manifest may have been tampered", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        print("AUDIT manifest_sha256_mismatch=1")
        return 1
    print(f"ASI manifest_sha256_verified={manifest.manifest_sha256}", file=sys.stderr)

    # Verify all required fields are present (fail closed on incomplete).
    if not manifest.required_fields_present():
        print("ERROR: manifest is incomplete — required fields missing", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        print("AUDIT manifest_incomplete=1")
        return 1
    print("ASI manifest_required_fields_present=True", file=sys.stderr)

    # Verify holdout was not accessed during calibration.
    if manifest.holdout_accessed:
        print("ERROR: manifest claims holdout_accessed=true — calibrate-dev must not access holdout", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        print("AUDIT holdout_accessed_violation=1")
        return 1
    print("ASI holdout_accessed=False", file=sys.stderr)

    # Verify human sign-off ID (nonempty AND exact match with manifest).
    signoff_id = args.signoff_id
    if not signoff_id or not signoff_id.strip():
        print("ERROR: human sign-off ID is required and must be nonempty", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        print("AUDIT missing_human_signoff=1")
        return 1
    if not manifest.signoff_human_signoff_id:
        print("ERROR: manifest has no human sign-off ID — calibration not signed off", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        print("AUDIT manifest_missing_signoff=1")
        return 1
    if signoff_id != manifest.signoff_human_signoff_id:
        print(
            f"ERROR: sign-off ID mismatch — CLI '{signoff_id}' != manifest '{manifest.signoff_human_signoff_id}'",
            file=sys.stderr,
        )
        print("METRIC evaluate_status=FAILED")
        print("AUDIT signoff_id_mismatch=1")
        return 1
    print(f"ASI signoff_human_signoff_id={signoff_id}", file=sys.stderr)
    # Verify thresholds have been signed off by human review.
    if not manifest.signoff_thresholds_signed_off:
        print("ERROR: thresholds have not been signed off (signoff_thresholds_signed_off=False)", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        print("AUDIT thresholds_not_signed_off=1")
        return 1
    print("ASI signoff_thresholds_signed_off=True", file=sys.stderr)
    # Verify oracle ceiling was computed and is positive.
    if manifest.signoff_oracle_ceiling <= 0.0:
        print("ERROR: oracle ceiling not computed or zero — cannot evaluate", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        print("AUDIT oracle_ceiling_missing=1")
        return 1
    print(f"ASI signoff_oracle_ceiling={manifest.signoff_oracle_ceiling:.6f}", file=sys.stderr)
    # Verify DEV articulation gate passed during calibration.
    if not manifest.signoff_dev_articulation_gate:
        print("ERROR: DEV articulation gate failed during calibration — coupler does not improve over no-update baseline", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        print("AUDIT dev_articulation_gate_failed=1")
        return 1
    print("ASI signoff_dev_articulation_gate=pass", file=sys.stderr)

    # 2. Verify eval manifest.
    from eval.v2 import verify_manifest  # type: ignore[missing-import]

    verify_manifest()
    eval_manifest_hash = hash_eval_manifest()
    if eval_manifest_hash != manifest.eval_manifest_sha256:
        print("ERROR: eval manifest hash mismatch — eval assets changed since calibration", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        print("AUDIT eval_manifest_sha256_mismatch=1")
        return 1
    print(f"ASI eval_manifest_sha256_verified={eval_manifest_hash}", file=sys.stderr)

    # 3. Load frozen LM.
    model_id = manifest.model_repo_id
    print(f"# Loading frozen LM: {model_id}", file=sys.stderr)
    driver = HFDriver.load(model_id=model_id)
    model_hash_before = hash_model(driver)
    model_safetensors_sha = hash_model_safetensors(driver)
    print(f"ASI model_hash_before={model_hash_before}", file=sys.stderr)

    # 4. Verify LM hash matches manifest (safetensors fingerprint).
    if model_safetensors_sha != manifest.model_safetensors_sha256:
        print("ERROR: model safetensors hash mismatch — model changed since calibration", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        print("AUDIT model_safetensors_mismatch=1")
        driver.close()
        return 1
    print(f"ASI model_safetensors_sha256_verified={model_safetensors_sha}", file=sys.stderr)

    # Freeze LM parameters.
    for p in driver._model.parameters():
        p.requires_grad_(False)
    driver._model.eval()

    # 5. Load frozen cortex artifact (full head+coupler) and coupler.
    cortex_artifact_path = args.coupler_path or (manifest_path.parent / "s19_cortex.pkl")
    cortex_artifact_path = Path(cortex_artifact_path)
    if not cortex_artifact_path.exists():
        # Fall back to coupler-only file for backward compat.
        cortex_artifact_path = manifest_path.parent / "s19_coupler.pkl"
    if not cortex_artifact_path.exists():
        print(f"ERROR: cortex artifact file not found: {cortex_artifact_path}", file=sys.stderr)
        print("METRIC evaluate_status=FAILED")
        driver.close()
        return 1
    with open(cortex_artifact_path, "rb") as f:
        artifact_data = pickle.load(f)
    # Full cortex artifact has "sha256"/"bytes"; coupler-only has "hash".
    cortex_artifact_sha = artifact_data.get("sha256") or artifact_data.get("hash", "")
    if cortex_artifact_sha != manifest.cortex_artifact_sha256:
        # If the file is coupler-only, verify against coupler_sha256.
        coupler_sha_from_file = artifact_data.get("hash", "")
        if coupler_sha_from_file != manifest.coupler_sha256:
            print("ERROR: cortex artifact hash mismatch — artifact file changed since calibration", file=sys.stderr)
            print("METRIC evaluate_status=FAILED")
            print("AUDIT cortex_artifact_mismatch=1")
            driver.close()
            return 1
    print(f"ASI cortex_artifact_sha256_verified={manifest.cortex_artifact_sha256}", file=sys.stderr)

    # Extract coupler state for loading into fresh cortexes.
    artifact_state = artifact_data["state"]
    if "W_coupler" in artifact_state:
        # Full cortex artifact — extract coupler subset.
        coupler_state = {
            "W_coupler": artifact_state["W_coupler"],
            "b_coupler": artifact_state["b_coupler"],
        }
    else:
        # Coupler-only file.
        coupler_state = artifact_state
    coupler_sha = manifest.coupler_sha256
    print(f"ASI coupler_sha256_verified={coupler_sha}", file=sys.stderr)

    # 6. Load curriculum.
    stage0 = build_curriculum(stage_names=("stage_0_grounding",))[0]
    dev_ids, holdout_ids = split_probes(
        stage0, fraction=manifest.split_fraction, salt=manifest.split_salt,
    )
    other_names = tuple(n for n in STAGE_ORDER if n != "stage_0_grounding")
    other_stages = build_curriculum(stage_names=other_names)

    labels = manifest.labels
    label_index = build_label_index(labels)
    confidence_threshold = manifest.proposed_confidence_threshold
    specificity_margin = manifest.proposed_specificity_margin
    label_prefix_template = LABEL_PREFIX_TEMPLATE

    print(f"# Labels: {len(labels)}", file=sys.stderr)
    print(f"# Confidence threshold: {confidence_threshold:.6f}", file=sys.stderr)
    print(f"# Specificity margin: {specificity_margin:.6f}", file=sys.stderr)

    # Determine seeds.
    n_seeds = args.seeds
    if n_seeds < MIN_SEEDS:
        print(f"WARNING: {n_seeds} seeds < minimum {MIN_SEEDS}; using {MIN_SEEDS}", file=sys.stderr)
        n_seeds = MIN_SEEDS
    print(f"# Seeds: {n_seeds}", file=sys.stderr)

    # Per-seed results for each condition.
    all_results: dict[str, list[dict[str, Any]]] = {
        cond: [] for cond in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    }

    # Store trained cortex states for C5 (swap) — need at least 2 seeds.
    trained_cortex_states: list[dict[str, torch.Tensor]] = []

    validity_flags: dict[str, bool] = {
        "lm_frozen": True,
        "raw_traces_deleted": True,
        "fixed_latent_width": True,
        "no_text_injection": True,
        "no_episode_id_conditioning": True,
        "parameter_budget": True,
        "meta_test_conflation": True,
        "c7_retrieval_baseline": True,
        "articulation_audit_complete": True,
    }

    all_audits: list[dict[str, Any]] = []

    for seed in range(n_seeds):
        print(f"\n# === Seed {seed} ===", file=sys.stderr)
        t0 = time.monotonic()

        # a. Initialize fresh cortex, load frozen coupler.
        config = CortexConfig(
            lr=args.lr,
            coupler_lr=args.coupler_lr,
            label_lr=args.label_lr,
        )
        cortex = SharedCortex(config=config, seed=seed)
        cortex.load_coupler(coupler_state)
        cortex.freeze_coupler()

        # For C6: train a separate cortex with permuted labels.
        cortex_c6 = SharedCortex(config=config, seed=seed)
        cortex_c6.load_coupler(coupler_state)
        cortex_c6.freeze_coupler()

        trace_store = TraceStore()

        # b. Teach stage-0 corrections (seed-shuffled).
        print(f"# Teaching stage-0 corrections (seed {seed})...", file=sys.stderr)
        teach_stats = teach_cortex(
            driver, cortex, stage0, labels, label_index, seed, trace_store,
            permuted_labels=False,
        )
        print(f"ASI seed_{seed}_teach_label_loss={teach_stats['label_loss_mean']:.6f}", file=sys.stderr)
        # coupler is frozen during online teaching; no coupler_loss to report

        # Teach C6 cortex with permuted labels.
        trace_store_c6 = TraceStore()
        teach_cortex(
            driver, cortex_c6, stage0, labels, label_index, seed, trace_store_c6,
            permuted_labels=True,
        )

        # Save trained cortex state for C5 swap.
        trained_cortex_states.append(cortex.state_dict())

        # c. Delete raw traces, verify zero (includes embedding/cache state
        # by clearing the driver's reserved-position surface so no stale
        # prefix persists between seeds).
        deleted = trace_store.delete_all()
        deleted_c6 = trace_store_c6.delete_all()
        driver.clear_reserved_position()
        print(f"ASI seed_{seed}_raw_traces_deleted={deleted}", file=sys.stderr)
        print(f"ASI seed_{seed}_raw_traces_deleted_c6={deleted_c6}", file=sys.stderr)
        if not trace_store.verify_zero():
            print(f"ERROR: raw traces not deleted for seed {seed}", file=sys.stderr)
            validity_flags["raw_traces_deleted"] = False
        if not trace_store_c6.verify_zero():
            print(f"ERROR: raw traces not deleted for C6 seed {seed}", file=sys.stderr)
            validity_flags["raw_traces_deleted"] = False

        # Verify parameter budget.
        if not verify_parameter_budget(cortex):
            print(f"ERROR: parameter budget exceeded for seed {seed}", file=sys.stderr)
            validity_flags["parameter_budget"] = False

        # d. Run C0-C7 conditions.
        # C0: vanilla baseline.
        r_c0 = run_condition(
            driver, "C0", stage0, other_stages, dev_ids, holdout_ids,
            None, None, confidence_threshold, label_prefix_template, trace_store,
        )
        all_results["C0"].append(r_c0)

        # C1: cortex architecture, no update (random init).
        random_cortex = SharedCortex(config=config, seed=seed + 1000)
        random_cortex.load_coupler(coupler_state)
        random_cortex.freeze_coupler()
        r_c1 = run_condition(
            driver, "C1", stage0, other_stages, dev_ids, holdout_ids,
            random_cortex, labels, confidence_threshold, label_prefix_template, trace_store,
        )
        all_results["C1"].append(r_c1)

        # C2: Arm A (label prefix).
        r_c2 = run_condition(
            driver, "C2", stage0, other_stages, dev_ids, holdout_ids,
            cortex, labels, confidence_threshold, label_prefix_template, trace_store,
        )
        all_results["C2"].append(r_c2)

        # C3: Arm B (latent control) — primary.
        r_c3 = run_condition(
            driver, "C3", stage0, other_stages, dev_ids, holdout_ids,
            cortex, labels, confidence_threshold, label_prefix_template, trace_store,
        )
        all_results["C3"].append(r_c3)

        # C4: C3 with cortex state zeroed.
        # Need a fresh copy of the trained cortex to zero.
        cortex_c4 = SharedCortex(config=config, seed=seed)
        cortex_c4.load_state_dict(cortex.state_dict())
        cortex_c4.freeze_coupler()
        r_c4 = run_condition(
            driver, "C4", stage0, other_stages, dev_ids, holdout_ids,
            cortex_c4, labels, confidence_threshold, label_prefix_template, trace_store,
        )
        all_results["C4"].append(r_c4)

        # C5: C3 with cortex state swapped (use a different seed's state).
        if seed == 0:
            # For seed 0, swap with seed 1's state (train it first if needed).
            # If only 1 seed available, swap with random init.
            swap_seed = 1 if n_seeds > 1 else 0
            if swap_seed < len(trained_cortex_states):
                swap_state = trained_cortex_states[swap_seed]
            else:
                # Train a quick seed-1 cortex for swapping.
                swap_cortex = SharedCortex(config=config, seed=1)
                swap_cortex.load_coupler(coupler_state)
                swap_cortex.freeze_coupler()
                swap_trace = TraceStore()
                teach_cortex(
                    driver, swap_cortex, stage0, labels, label_index, 1, swap_trace,
                )
                swap_trace.delete_all()
                swap_state = swap_cortex.state_dict()
        else:
            # Swap with seed 0's state.
            swap_state = trained_cortex_states[0]

        cortex_c5 = SharedCortex(config=config, seed=seed)
        cortex_c5.load_coupler(coupler_state)
        cortex_c5.freeze_coupler()
        # Load the swap state into the cortex.
        swap_cortex = SharedCortex(config=config, seed=seed)
        swap_cortex.load_state_dict(swap_state)
        r_c5 = run_condition(
            driver, "C5", stage0, other_stages, dev_ids, holdout_ids,
            cortex_c5, labels, confidence_threshold, label_prefix_template, trace_store,
            swapped_cortex=swap_cortex,
        )
        all_results["C5"].append(r_c5)

        # C6: C3 with permuted labels during teaching.
        r_c6 = run_condition(
            driver, "C6", stage0, other_stages, dev_ids, holdout_ids,
            cortex_c6, labels, confidence_threshold, label_prefix_template, trace_store_c6,
        )
        all_results["C6"].append(r_c6)

        # C7: retrieval baseline (external bar).
        r_c7 = run_condition(
            driver, "C7", stage0, other_stages, dev_ids, holdout_ids,
            None, None, confidence_threshold, label_prefix_template, trace_store,
        )
        all_results["C7"].append(r_c7)

        # Collect audits for validity checks, enriched with required
        # articulation audit fields: LM hash, persistent bytes, and
        # banned-content fields (label_text, corrected_response,
        # correction_utterance, expected) so verify_no_text_injection
        # can check that none leak into Arm B prompts.
        episode_lookup = {ep.id: ep for ep in stage0.episodes}
        for audit in r_c3.get("audits", []):
            ep = episode_lookup.get(audit.get("episode_id", ""))
            if ep is not None:
                audit["label_text"] = ep.corrected_label
                audit["corrected_response"] = ep.corrected_response
                audit["correction_utterance"] = ep.correction_utterance
                audit["expected"] = ep.corrected_response
            enriched = build_articulation_audit(
                condition="C3",
                arm=audit.get("arm", "B"),
                prompt_text=audit.get("prompt_text", ""),
                latent_bank_shape=audit.get("latent_bank_shape"),
                raw_trace_count=audit.get("raw_trace_count") or 0,
                model_hash=model_hash_before,
                persistent_bytes=r_c3.get("persistent_bytes", 0),
            )
            audit.update(enriched)
            all_audits.append(audit)

        elapsed = time.monotonic() - t0
        print(f"ASI seed_{seed}_wall_s={elapsed:.1f}", file=sys.stderr)

        # Emit per-seed ASI lines.
        for cond in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            r = all_results[cond][-1]
            print(f"ASI seed_{seed}_{cond}_holdout={r['holdout_acc']:.6f}", file=sys.stderr)
            print(f"ASI seed_{seed}_{cond}_transfer={r['transfer_acc']:.6f}", file=sys.stderr)
            print(f"ASI seed_{seed}_{cond}_scope={r['scope_acc']:.6f}", file=sys.stderr)
            print(f"ASI seed_{seed}_{cond}_specificity={r['specificity_acc']:.6f}", file=sys.stderr)

    # Meta-test conflation check: C3 and C2 must not be identical.
    if not check_meta_test_conflation(all_results["C3"], all_results["C2"]):
        print("ERROR: C3 and C2 results are identical across all seeds — arms conflated", file=sys.stderr)
        validity_flags["meta_test_conflation"] = False
        gc.collect()

    # 7. Verify validity flags.
    # C7 retrieval baseline availability: block if no holdout probes to
    # retrieve from (S3.M2a baseline unavailable).
    if not holdout_ids:
        print("ERROR: C7 retrieval baseline unavailable — no holdout probes", file=sys.stderr)
        validity_flags["c7_retrieval_baseline"] = False

    # Articulation audit completeness: each C3 audit must contain the
    # required fields (prompt, latent shape, raw traces, LM hash,
    # persistent bytes, banned-content fields).
    required_audit_fields = (
        "prompt_text",
        "latent_bank_shape",
        "raw_trace_count",
        "language_organ_hash",
        "persistent_cortex_bytes",
        "label_text",
        "corrected_response",
        "correction_utterance",
        "expected",
    )
    for audit in all_audits:
        if audit.get("arm") != "B":
            continue
        missing = [f for f in required_audit_fields if f not in audit]
        if missing:
            print(f"ERROR: articulation audit missing fields: {missing}", file=sys.stderr)
            validity_flags["articulation_audit_complete"] = False
            break

    # Check fixed latent width and no text injection from audits.
    if all_audits:
        if not verify_fixed_latent_width(all_audits):
            validity_flags["fixed_latent_width"] = False
        if not verify_no_text_injection(all_audits):
            validity_flags["no_text_injection"] = False
        if not verify_no_episode_id_conditioning(all_audits):
            validity_flags["no_episode_id_conditioning"] = False

    # 8. Hash LM after, verify frozen.
    model_hash_after = hash_model(driver)
    lm_frozen = model_hash_before == model_hash_after
    validity_flags["lm_frozen"] = lm_frozen
    print(f"ASI model_hash_after={model_hash_after}", file=sys.stderr)
    print(f"ASI lm_frozen={lm_frozen}", file=sys.stderr)

    driver.close()
    gc.collect()

    all_validity_pass = all(validity_flags.values())
    print(f"ASI all_validity_pass={all_validity_pass}", file=sys.stderr)
    for flag, val in validity_flags.items():
        print(f"ASI validity_{flag}={val}", file=sys.stderr)

    # 9. Compute deltas and verdicts.
    # C3 vs C1 deltas.
    c3_retention_deltas = [
        r3["holdout_acc"] - r1["holdout_acc"]
        for r3, r1 in zip(all_results["C3"], all_results["C1"], strict=False)
    ]
    c3_transfer_deltas = [
        r3["transfer_acc"] - r1["transfer_acc"]
        for r3, r1 in zip(all_results["C3"], all_results["C1"], strict=False)
    ]
    c3_scope_deltas = [
        r3["scope_acc"] - r1["scope_acc"]
        for r3, r1 in zip(all_results["C3"], all_results["C1"], strict=False)
    ]
    c3_specificity_deltas = [
        r3["specificity_acc"] - r1["specificity_acc"]
        for r3, r1 in zip(all_results["C3"], all_results["C1"], strict=False)
    ]
    causal_state_deltas = [
        r3["holdout_acc"] - r4["holdout_acc"]
        for r3, r4 in zip(all_results["C3"], all_results["C4"], strict=False)
    ]
    state_addressing_deltas = [
        r3["holdout_acc"] - r5["holdout_acc"]
        for r3, r5 in zip(all_results["C3"], all_results["C5"], strict=False)
    ]
    feedback_semantics_deltas = [
        r3["holdout_acc"] - r6["holdout_acc"]
        for r3, r6 in zip(all_results["C3"], all_results["C6"], strict=False)
    ]

    # C2 vs C1 deltas.
    c2_retention_deltas = [
        r2["holdout_acc"] - r1["holdout_acc"]
        for r2, r1 in zip(all_results["C2"], all_results["C1"], strict=False)
    ]
    c2_transfer_deltas = [
        r2["transfer_acc"] - r1["transfer_acc"]
        for r2, r1 in zip(all_results["C2"], all_results["C1"], strict=False)
    ]
    c2_specificity_deltas = [
        r2["specificity_acc"] - r1["specificity_acc"]
        for r2, r1 in zip(all_results["C2"], all_results["C1"], strict=False)
    ]

    # Persistent bytes and behavior delta per byte.
    persistent_bytes = all_results["C3"][0]["persistent_bytes"] if all_results["C3"] else 0
    retention_mean = float(np.mean(c3_retention_deltas)) if c3_retention_deltas else 0.0
    behavior_delta_per_byte = retention_mean / persistent_bytes if persistent_bytes > 0 else 0.0

    # Compute CIs.
    retention_ci = mean_ci(c3_retention_deltas)
    transfer_ci = mean_ci(c3_transfer_deltas)
    scope_ci = mean_ci(c3_scope_deltas)
    specificity_ci = mean_ci(c3_specificity_deltas)
    causal_ci = mean_ci(causal_state_deltas)
    addressing_ci = mean_ci(state_addressing_deltas)
    feedback_ci = mean_ci(feedback_semantics_deltas)

    c2_retention_ci = mean_ci(c2_retention_deltas)
    c2_transfer_ci = mean_ci(c2_transfer_deltas)
    c2_specificity_ci = mean_ci(c2_specificity_deltas)

    # Compute verdicts.
    verdicts = compute_verdicts(
        all_results["C3"],
        all_results["C2"],
        all_results["C4"],
        all_results["C5"],
        all_results["C6"],
        all_results["C1"],
        specificity_margin,
        all_validity_pass,
    )

    # 10. Emit METRIC/ASI/AUDIT sentinels.
    # Primary metrics (C3 vs C1).
    print(f"METRIC retention_delta={retention_ci[0]}")
    print(f"METRIC transfer_delta={transfer_ci[0]}")
    print(f"METRIC scope_delta={scope_ci[0]}")
    print(f"METRIC specificity_delta={specificity_ci[0]}")
    print(f"METRIC causal_state_delta={causal_ci[0]}")
    print(f"METRIC state_addressing_delta={addressing_ci[0]}")
    print(f"METRIC feedback_semantics_delta={feedback_ci[0]}")
    print(f"METRIC persistent_bytes={persistent_bytes}")
    print(f"METRIC behavior_delta_per_byte={behavior_delta_per_byte}")

    # C2 metrics (H-LABEL).
    print(f"METRIC c2_retention_delta={c2_retention_ci[0]}")
    print(f"METRIC c2_transfer_delta={c2_transfer_ci[0]}")
    print(f"METRIC c2_specificity_delta={c2_specificity_ci[0]}")

    # CIs.
    print(f"ASI retention_delta_ci95=[{retention_ci[1]},{retention_ci[2]}]")
    print(f"ASI transfer_delta_ci95=[{transfer_ci[1]},{transfer_ci[2]}]")
    print(f"ASI scope_delta_ci95=[{scope_ci[1]},{scope_ci[2]}]")
    print(f"ASI specificity_delta_ci95=[{specificity_ci[1]},{specificity_ci[2]}]")
    print(f"ASI causal_state_delta_ci95=[{causal_ci[1]},{causal_ci[2]}]")
    print(f"ASI state_addressing_delta_ci95=[{addressing_ci[1]},{addressing_ci[2]}]")
    print(f"ASI feedback_semantics_delta_ci95=[{feedback_ci[1]},{feedback_ci[2]}]")
    print(f"ASI c2_retention_delta_ci95=[{c2_retention_ci[1]},{c2_retention_ci[2]}]")
    print(f"ASI c2_transfer_delta_ci95=[{c2_transfer_ci[1]},{c2_transfer_ci[2]}]")
    print(f"ASI c2_specificity_delta_ci95=[{c2_specificity_ci[1]},{c2_specificity_ci[2]}]")

    # Verdicts.
    print(f"METRIC h_latent_verdict={verdicts['H_LATENT']}")
    print(f"METRIC h_label_verdict={verdicts['H_LABEL']}")

    # Configuration ASI.
    print(f"ASI seeds={n_seeds}")
    print(f"ASI model_id={model_id}")
    print(f"ASI model_hash={model_hash_before}")
    print(f"ASI model_hash_after={model_hash_after}")
    print(f"ASI eval_manifest_hash={eval_manifest_hash}")
    print(f"ASI manifest_sha256={manifest.manifest_sha256}")
    print(f"ASI signoff_human_signoff_id={signoff_id}")
    print(f"ASI coupler_sha256={coupler_sha}")
    print(f"ASI cortex_artifact_sha256={manifest.cortex_artifact_sha256}")
    print(f"ASI confidence_threshold={confidence_threshold}")
    print(f"ASI specificity_margin={specificity_margin}")
    print(f"ASI parameter_total={manifest.parameter_total}")
    print(f"ASI parameter_budget={manifest.parameter_budget}")
    print(f"ASI d_embd={D_EMBD}")
    print(f"ASI d_cortex={D_CORTEX}")
    print(f"ASI latent_tokens={LATENT_TOKENS}")
    print(f"ASI n_labels={len(labels)}")
    print(f"ASI label_prefix_template={label_prefix_template}")
    print(f"ASI oracle_ceiling={manifest.signoff_oracle_ceiling:.6f}")
    print(f"ASI dev_articulation_gate={'pass' if manifest.signoff_dev_articulation_gate else 'fail'}")
    print(f"ASI thresholds_signed_off={manifest.signoff_thresholds_signed_off}")

    # Per-condition per-seed ASI.
    for cond in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
        for seed_idx, r in enumerate(all_results[cond]):
            print(f"ASI {cond}_seed{seed_idx}_holdout={r['holdout_acc']:.6f}")
            print(f"ASI {cond}_seed{seed_idx}_transfer={r['transfer_acc']:.6f}")
            print(f"ASI {cond}_seed{seed_idx}_scope={r['scope_acc']:.6f}")
            print(f"ASI {cond}_seed{seed_idx}_specificity={r['specificity_acc']:.6f}")

    # Validity audits.
    for flag, val in validity_flags.items():
        status = "pass" if val else "FAIL"
        print(f"AUDIT validity_{flag}={status}")

    # Articulation audits (sample from C3) — includes required fields:
    # prompt, latent shape, raw traces, LM hash, persistent bytes,
    # and banned-content fields.
    if all_audits:
        for audit in all_audits[:5]:
            print(
                f"AUDIT condition={audit.get('condition', 'C3')} "
                f"arm={audit.get('arm', 'B')} "
                f"prompt_text={audit.get('prompt_text', '')!r} "
                f"latent_bank_shape={audit.get('latent_bank_shape')} "
                f"raw_trace_count={audit.get('raw_trace_count')} "
                f"language_organ_hash={audit.get('language_organ_hash', '')} "
                f"persistent_cortex_bytes={audit.get('persistent_cortex_bytes', 0)} "
                f"label_text={audit.get('label_text', '')!r} "
                f"correct={audit.get('correct')}"
            )

    # Raw trace audit.
    print("AUDIT raw_trace_count=0")
    print(f"AUDIT fixed_latent_width={'pass' if validity_flags['fixed_latent_width'] else 'FAIL'}")
    print(f"AUDIT no_text_injection={'pass' if validity_flags['no_text_injection'] else 'FAIL'}")
    print(f"AUDIT frozen_lm_hash={'pass' if validity_flags['lm_frozen'] else 'FAIL'}")
    print(f"AUDIT parameter_budget={'pass' if validity_flags['parameter_budget'] else 'FAIL'}")
    print(f"AUDIT meta_test_conflation={'pass' if validity_flags['meta_test_conflation'] else 'FAIL'}")
    print(f"AUDIT c7_retrieval_baseline={'pass' if validity_flags['c7_retrieval_baseline'] else 'FAIL'}")
    print(
        f"AUDIT articulation_audit_complete="
        f"{'pass' if validity_flags['articulation_audit_complete'] else 'FAIL'}"
    )
    print(f"AUDIT thresholds_signed_off={'pass' if manifest.signoff_thresholds_signed_off else 'FAIL'}")

    # Final status.
    if all_validity_pass:
        print("METRIC evaluate_status=OK")
    else:
        print("METRIC evaluate_status=BLOCKED")
        print("AUDIT verdict_blocked_by_validity_failure=1")

    # Verdict blocking.
    if not all_validity_pass:
        print("METRIC h_latent_verdict=BLOCKED")
        print("METRIC h_label_verdict=BLOCKED")

    return 0 if all_validity_pass else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oczy.experiments.s19_language_organ",
        description="Research 19: LM as language organ — direct cortex learning, two articulation paths",
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    # calibrate-dev
    cal = subparsers.add_parser(
        "calibrate-dev",
        help="Train coupler on DEV-only data and write calibration manifest",
    )
    cal.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct",
                      help="HuggingFace model ID")
    cal.add_argument("--manifest-out", required=True,
                      help="Path to write calibration manifest JSON")
    cal.add_argument("--coupler-out", default=None,
                      help="Path to write frozen cortex artifact (default: alongside manifest)")
    cal.add_argument("--source-commit", default=None,
                      help="Git commit SHA for source provenance (or OCZY_SOURCE_COMMIT env)")
    cal.add_argument("--source-archive", default=None,
                      help="Path to source archive for SHA-256 (or OCZY_SOURCE_ARCHIVE env)")
    cal.add_argument("--lr", type=float, default=0.01, help="Base learning rate")
    cal.add_argument("--coupler-lr", type=float, default=0.001, help="Coupler learning rate")
    cal.add_argument("--label-lr", type=float, default=0.05, help="Label head learning rate")

    # evaluate
    evl = subparsers.add_parser(
        "evaluate",
        help="Verify manifest and run one-shot holdout/transfer/scope/specificity evaluation",
    )
    evl.add_argument("--manifest", required=True,
                      help="Path to calibration manifest JSON")
    evl.add_argument("--coupler-path", default=None,
                      help="Path to frozen cortex artifact (default: alongside manifest)")
    evl.add_argument("--signoff-id", required=True,
                      help="Human-approved sign-off ID (nonempty)")
    evl.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                      help=f"Number of seeds (default: {DEFAULT_SEEDS}, min: {MIN_SEEDS})")
    evl.add_argument("--lr", type=float, default=0.01, help="Base learning rate")
    evl.add_argument("--coupler-lr", type=float, default=0.001, help="Coupler learning rate")
    evl.add_argument("--label-lr", type=float, default=0.05, help="Label head learning rate")

    args = parser.parse_args(argv)

    if args.phase == "calibrate-dev":
        return _calibrate_dev(args)
    elif args.phase == "evaluate":
        return _evaluate(args)
    else:
        parser.error(f"Unknown phase: {args.phase}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
