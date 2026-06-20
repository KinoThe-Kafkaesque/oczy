#!/usr/bin/env python3
"""Extended-learning evaluation for the Plastic World Model Agent modules.

This script trains each module on a shared 30-item word-sense disambiguation
curriculum (trivia-style), then evaluates each module on the metrics it is
actually designed for.  Unlike the end-to-end ``correction-benchmark``, this is a
component-level evaluation: we want to know whether each organ learns the right
thing from repeated correction.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from correction_benchmark import run_benchmark, OracleAgent, AlwaysWrongAgent
from experience_autoencoder import ExperienceAutoencoder
from identity_hypernetwork import IdentityHypernetwork
from neural_hippocampus import NeuralHippocampus
from plastic_cortex import PlasticCortex
from skill_immune_cortex import SkillImmuneCortex
from world_model_critic import WorldModelCritic


# ---------------------------------------------------------------------------
# Shared curriculum: 30 word-sense ambiguity corrections.
# Each tuple is: (request, correction, correct_sense, transfer_probe, scope_probe)
# ---------------------------------------------------------------------------
CURRICULUM: list[tuple[str, str, str, str, str]] = [
    ("Update the user's profile.", "No, in this product 'profile' means business vertical.",
     "business vertical", "Switch the active profile.", "Where can I edit my user profile?"),
    ("Deploy the new model.", "No, 'model' here means machine-learning model.",
     "ML model", "Retrain the model.", "Book a fashion model for the campaign."),
    ("Schedule the batch.", "No, 'batch' here means ML training batch.",
     "ML training batch", "Run a batch evaluation.", "Mix the next batch of dough."),
    ("Create a branch.", "No, 'branch' means git branch.",
     "git branch", "Merge the branch.", "Which bank branch is nearest?"),
    ("Reserve a table.", "No, 'table' means dining table.",
     "dining table", "Book a table for two.", "Create a table for users."),
    ("Start the run.", "No, 'run' means ML experiment run.",
     "ML experiment run", "Log the latest run.", "The player scored a home run."),
    ("Edit the cell.", "No, 'cell' means spreadsheet cell.",
     "spreadsheet cell", "Format the cell.", "The biology cell divides."),
    ("Play the record.", "No, 'record' means music record.",
     "music record", "Clean the record.", "Insert a new database record."),
    ("Add a module.", "No, 'module' means software module.",
     "software module", "Import the module.", "The space station module docks."),
    ("Press the key.", "No, 'key' means keyboard key.",
     "keyboard key", "Hold the key down.", "Check the map key."),
    ("Restart the service.", "No, 'service' means microservice.",
     "microservice", "Deploy the service.", "What time is the church service?"),
    ("Sharpen the file.", "No, 'file' means the metal tool.",
     "metal file tool", "Use the file to shape wood.", "Save the file to disk."),
    ("Open the terminal.", "No, 'terminal' means command-line terminal.",
     "command-line terminal", "Type into the terminal.", "The airport terminal is crowded."),
    ("Run the shell.", "No, 'shell' means Unix shell.",
     "Unix shell", "Execute a shell command.", "The turtle has a hard shell."),
    ("Check the port.", "No, 'port' means network port.",
     "network port", "Open port 8080.", "The ship arrived at the port."),
    ("Mount the volume.", "No, 'volume' means docker volume.",
     "docker volume", "Create a volume.", "Turn up the volume."),
    ("Define the schema.", "No, 'schema' means database schema.",
     "database schema", "Update the schema.", "His personality schema is fixed."),
    ("Run the migration.", "No, 'migration' means database migration.",
     "database migration", "Apply the migration.", "Study bird migration patterns."),
    ("Fix the bug.", "No, 'bug' means software bug.",
     "software bug", "File a bug report.", "A ladybug is a beneficial bug."),
    ("Push to the repo.", "No, 'repo' means git repository.",
     "git repository", "Clone the repo.", "Repossession is not repo here."),
    ("Pull the image.", "No, 'image' means docker image.",
     "docker image", "Build the image.", "Frame the photographic image."),
    ("Deploy the stack.", "No, 'stack' means software stack.",
     "software stack", "Inspect the stack trace.", "Stack the books neatly."),
    ("Run the container.", "No, 'container' means docker container.",
     "docker container", "Stop the container.", "Store goods in a shipping container."),
    ("Check the log.", "No, 'log' means system log.",
     "system log", "Tail the log.", "A log floated down the river."),
    ("Scale the app.", "No, 'scale' means horizontal scaling.",
     "horizontal scaling", "Auto-scale the app.", "Weigh food on a kitchen scale."),
    ("Map the route.", "No, 'route' means API route.",
     "API route", "Define the route.", "Plan a scenic route."),
    ("Resolve the conflict.", "No, 'conflict' means git merge conflict.",
     "git merge conflict", "Fix the conflict files.", "The story has internal conflict."),
    ("Rebase the branch.", "No, 'rebase' means git rebase.",
     "git rebase", "Perform a rebase.", "Rebase the price estimate."),
    ("Stage the changes.", "No, 'stage' means git stage.",
     "git stage", "Stage all modified files.", "The actors take the stage."),
    ("Commit the work.", "No, 'commit' means git commit.",
     "git commit", "Write a commit message.", "Commit to a decision."),
]

TRIVIA_FACTS = [
    ("The capital of France is Paris.", "Paris"),
    ("2 + 2 equals 4.", "4"),
    ("The sky is blue.", "Blue"),
    ("H2O is water.", "Water"),
    ("There are seven days in a week.", "Seven"),
    ("The first month is January.", "January"),
    ("A primary color is Red.", "Red"),
    ("10 / 2 equals 5.", "5"),
    ("The freezing point of water is 0 degrees Celsius.", "0 degrees Celsius"),
    ("Leonardo da Vinci painted the Mona Lisa.", "Leonardo da Vinci"),
    ("There are seven continents.", "Seven"),
    ("'Hola' is Spanish.", "Spanish"),
]


def first_keyword(text: str) -> str:
    """Extract the target ambiguous word from a correction sentence."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    ignore = {"no", "here", "this", "means", "product", "the", "a", "an", "in", "is", "to", "of"}
    for w in words:
        if w in ignore or len(w) < 2:
            continue
        return w
    return words[-1] if words else ""


def mem_bytes(obj: Any) -> int:
    """Approximate persistent memory size in bytes."""
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return len(json.dumps(obj, default=str).encode("utf-8"))


# ---------------------------------------------------------------------------
# Module-specific evaluations
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    module: str
    metrics: dict[str, float]
    aggregate: float


# ---------------------------------------------------------------------------
# Aggregation policy
# ---------------------------------------------------------------------------
# Every metric that enters an EvalResult.aggregate MUST be normalised to
# "higher = better, in [0, 1]" BEFORE it is combined into the aggregate.
# Cross-module ranking on `aggregate` is only meaningful because of this
# convention: a metric that is naturally "lower = better" (e.g.
# memory_bytes_per_delta) must be inverted (1 - x) before aggregation.
#
# Per-module weights differ because each module is scored on the metrics
# it was designed for, but every weight nonnegativity and sum-to-1 holds,
# and every input is on the same direction/scale.
# ---------------------------------------------------------------------------


def eval_plastic_cortex() -> EvalResult:
    """Train Plastic Cortex and measure whether corrections switch the target word."""
    cortex_config = {"alpha_correction": 8.0, "alpha_normal": 0.01, "recurrent_gain": 0.02}
    agent = PlasticCortex(config=cortex_config)

    # Learn.
    for request, correction, sense, _, _ in CURRICULUM:
        agent.answer(request)
        agent.correct(first_keyword(correction), sense)

    # Probe: for each target word, does the corrected sense win?
    correct = 0
    total = 0
    normal_only_correct = 0
    for request, correction, sense, transfer, scope in CURRICULUM:
        keyword = first_keyword(correction)
        # Direct probe.
        if agent.answer(request) == sense:
            correct += 1
        total += 1
        # Transfer probe (exact keyword present).
        if agent.answer(transfer) == sense:
            correct += 1
        total += 1

        # Normal-only control: a fresh agent exposed to the same TEXT
        # (no correction signal), to confirm that corrections are the
        # actual learning signal and not just seeing the word five times.
        # Each control is per-item by design so there is no
        # cross-contamination between curriculum entries.
        control = PlasticCortex(config=cortex_config)
        for _ in range(5):
            control.answer(request)
        if control.answer(request) == sense:
            normal_only_correct += 1

    # Plasticity ratio: correction effect over normal-text effect.
    correction_effect = correct / max(1, total)
    normal_effect = normal_only_correct / len(CURRICULUM)
    plasticity_ratio = (correction_effect - normal_effect) / max(0.01, 1.0 - normal_effect)

    metrics = {
        "correction_accuracy": correction_effect,
        "normal_control_accuracy": normal_effect,
        "plasticity_ratio": max(0.0, plasticity_ratio),
        "memory_bytes": mem_bytes(agent.status()),
    }
    aggregate = (correction_effect * 0.5) + (max(0.0, plasticity_ratio) * 0.5)
    return EvalResult("PlasticCortex", metrics, aggregate)


def eval_neural_hippocampus() -> EvalResult:
    """Train hippocampus and measure surprise-gating, replay, compression.

    Stores each curriculum correction with its ``correction`` text (the
    raw sentence) AND its ``corrected_answer`` (the recovered label).
    Previously this stored the label under ``correction=``, which only
    passed the replay_accuracy check by coincidence --- the field was
    both written and read with the wrong name, so the test passed without
    actually exercising the organ's ability to surface a label across a
    real round-trip.  This is the C2 fix from the 2026-06-21 review.
    """
    h = NeuralHippocampus()

    # Inject a mix of high-surprise corrections and low-surprise accepted answers.
    low_surprise = 0
    for fact, expected in TRIVIA_FACTS[:6]:
        h.store(fact, expected, correction=None,
                prediction_error=0.05, corrected_answer=expected)
        low_surprise += 1

    stored_high = 0
    for request, correction, sense, _, _ in CURRICULUM:
        h.store(request, "unknown", correction=correction,
                prediction_error=0.9, corrected_answer=sense)
        # A second similar query should be recognized as memory / replay.
        replayed = h.reinforce(request)
        if replayed:
            stored_high += 1

    # Replay accuracy BEFORE consolidation (when raw traces still exist).
    replay_correct = 0
    for request, correction, sense, _, _ in CURRICULUM:
        replayed = h.reinforce(request)
        if replayed and replayed[0].get("corrected_answer") == sense:
            replay_correct += 1

    before = h.status()
    summaries = h.consolidate()
    after = h.status()

    # Compression trust: slow updates must exist and raw traces must shrink.
    if before.get("trace_bytes", 0) > 0 and after.get("trace_bytes", 0) is not None:
        raw_compression = before["trace_bytes"] / max(1, after["trace_bytes"])
    else:
        raw_compression = 1.0

    # Normalize compression to [0,1] with 10x = 1.0.
    compression_ratio = min(raw_compression, 10.0) / 10.0

    # Slow-update survival score: did consolidation produce summaries?
    slow_update_rate = len(summaries) / max(1, len(CURRICULUM))

    storage_precision = stored_high / len(CURRICULUM)
    replay_accuracy = replay_correct / len(CURRICULUM)

    metrics = {
        "storage_precision": storage_precision,
        "replay_accuracy": replay_accuracy,
        "compression_ratio": compression_ratio,
        "slow_update_rate": slow_update_rate,
        "memory_bytes": mem_bytes(h.status()),
    }
    aggregate = (storage_precision * 0.25) + (replay_accuracy * 0.35) + (compression_ratio * 0.25) + (slow_update_rate * 0.15)
    return EvalResult("NeuralHippocampus", metrics, aggregate)


def eval_world_model_critic() -> EvalResult:
    """Train critic on repeated corrections and measure calibration."""
    critic = WorldModelCritic()
    target_word = "profile"

    # Repeated corrections on profile-like queries.
    for i in range(12):
        query = f"What does {target_word} mean here?"
        pred = critic.predict_acceptance(query, "unknown")
        critic.record_outcome(query, "unknown", correction=f"No, {target_word} means business vertical.")

    # Similar but not identical queries.
    similar_queries = [
        "Tell me about the profile field.",
        "Update the profile configuration.",
        "Profile settings for the business.",
    ]
    similar_likelihoods = [
        critic.predict_acceptance(q, "unknown")["correction_likelihood"]
        for q in similar_queries
    ]

    # Unrelated queries should stay low.
    unrelated_queries = [
        "How do I reset my password?",
        "What is the weather today?",
        "Tell me a joke.",
    ]
    unrelated_likelihoods = [
        critic.predict_acceptance(q, "unknown")["correction_likelihood"]
        for q in unrelated_queries
    ]

    discrimination = sum(similar_likelihoods) / max(0.001, sum(unrelated_likelihoods))
    calibration = 1.0 - abs(0.75 - (sum(similar_likelihoods) / len(similar_likelihoods)))

    metrics = {
        "similar_correction_likelihood": sum(similar_likelihoods) / len(similar_likelihoods),
        "unrelated_correction_likelihood": sum(unrelated_likelihoods) / len(unrelated_likelihoods),
        "discrimination": min(discrimination, 5.0) / 5.0,
        "calibration": max(0.0, calibration),
        "memory_bytes": mem_bytes(critic),
    }
    aggregate = (metrics["discrimination"] * 0.6) + (metrics["calibration"] * 0.4)
    return EvalResult("WorldModelCritic", metrics, aggregate)


def eval_identity_hypernetwork() -> EvalResult:
    """Train hypernetwork and measure adapter accuracy and identity change."""
    hyper = IdentityHypernetwork()

    z_before = np.asarray([hyper.latents.z_user, hyper.latents.z_domain, hyper.latents.z_style, hyper.latents.z_mistakes])
    for request, correction, sense, _, _ in CURRICULUM:
        hyper.update_identity({
            "token": first_keyword(correction),
            "correct_label": sense,
            "source": "user_correction",
        })

    # For each learned token, does the adapter score the corrected sense highest?
    correct = 0
    total = 0
    adapters = hyper.generate_adapters()
    for request, correction, sense, _, _ in CURRICULUM:
        token = first_keyword(correction)
        row = adapters.get(token, {})
        if row:
            best = max(row, key=row.get)
            if best == sense:
                correct += 1
        total += 1

    z_after = np.asarray([hyper.latents.z_user, hyper.latents.z_domain, hyper.latents.z_style, hyper.latents.z_mistakes])
    identity_shift = float(np.mean([np.linalg.norm(a - b) for a, b in zip(z_before, z_after)]))

    metrics = {
        "adapter_retrieval_accuracy": correct / max(1, total),
        "identity_shift": min(identity_shift, 10.0) / 10.0,
        "memory_bytes": mem_bytes(hyper),
    }
    aggregate = (metrics["adapter_retrieval_accuracy"] * 0.7) + (metrics["identity_shift"] * 0.3)
    return EvalResult("IdentityHypernetwork", metrics, aggregate)


def eval_skill_immune_cortex() -> EvalResult:
    """Train immune cortex and measure detector precision/recall and merging."""
    immune = SkillImmuneCortex()

    # Add detectors from curriculum corrections.
    for request, correction, sense, _, _ in CURRICULUM:
        immune.add_detector(correction, sense, response=f"distinguish {sense}")

    # Positive probes: queries that should trigger (the original request itself).
    tp = fp = fn = tn = 0
    for request, correction, sense, _, _ in CURRICULUM:
        keyword = first_keyword(correction)
        triggered = bool(immune.check(keyword, request))
        if triggered:
            tp += 1
        else:
            fn += 1

    # Negative probes: unrelated trivia.
    for fact, expected in TRIVIA_FACTS:
        keyword = first_keyword(fact)
        triggered = bool(immune.check(keyword, fact))
        if triggered:
            fp += 1
        else:
            tn += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    before_merge = len(immune.detectors)
    immune.merge_detectors()
    after_merge = len(immune.detectors)
    merge_ratio = after_merge / max(1, before_merge)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "merge_ratio": merge_ratio,
        "memory_bytes": mem_bytes(immune),
    }
    aggregate = (f1 * 0.6) + ((1.0 - merge_ratio) * 0.4)  # reward merging (compression)
    return EvalResult("SkillImmuneCortex", metrics, aggregate)


def eval_experience_autoencoder() -> EvalResult:
    """Train autoencoder and measure compression + reconstruction accuracy."""
    ae = ExperienceAutoencoder()

    raw_bytes = 0
    latent_bytes = 0
    reconstruction_errors = []

    z = None
    for request, correction, sense, _, _ in CURRICULUM:
        episode = {
            "situation": request,
            "model_answer": "unknown",
            "correction": correction,
            "revised_answer": sense,
            "outcome": "corrected",
        }
        raw_bytes += len(json.dumps(episode).encode("utf-8"))
        dz = ae.encode(episode)
        latent_bytes += dz.nbytes
        decoded = ae.decode(dz)
        reconstruction_errors.append(ae.reconstruction_error(episode, decoded))
        z = ae.update_identity(z, episode)

    compression_ratio = raw_bytes / max(1, latent_bytes)
    mean_reconstruction_error = sum(reconstruction_errors) / max(1, len(reconstruction_errors))
    identity_size = z.nbytes if z is not None else 0

    metrics = {
        "compression_ratio": min(compression_ratio, 50.0) / 50.0,
        "reconstruction_error": 1.0 - mean_reconstruction_error,
        "identity_size_bytes": identity_size,
        "memory_bytes": mem_bytes(ae),
    }
    aggregate = (metrics["compression_ratio"] * 0.5) + (metrics["reconstruction_error"] * 0.5)
    return EvalResult("ExperienceAutoencoder", metrics, aggregate)


def eval_baseline(method: str) -> EvalResult:
    """Run the canonical benchmark on a trivial baseline for reference.

    All metrics are normalised to *higher = better, in [0, 1]* before
    aggregation.  Previously ``memory_bytes_per_delta`` was averaged
    in its raw form, where a larger value is *worse* but was treated as
    better, artificially ranking PlasticCortex (tiny fast-weights) above
    OracleAgent (correct but holds a full lookup table).
    """
    agent = OracleAgent() if method == "Oracle" else AlwaysWrongAgent()
    scores = run_benchmark(agent)
    # Invert memory_bytes_per_delta so that smaller memory / more deltas
    # (both good) yield a higher score.  Cap raw bytes at 1000 so a large
    # oracle lookup table does not drive the score to exactly 0.0.
    raw_mem = scores["memory_bytes_per_delta"]
    if raw_mem == math.inf:
        normalized_mem = 0.0
    else:
        normalized_mem = 1.0 - (min(raw_mem, 1000.0) / 1000.0)
    metrics = {
        "correction_uptake_latency": 1.0 - scores["correction_uptake_latency"],
        "transfer_score": scores["transfer_score"],
        "scope_score": scores["scope_score"],
        "forgetting_score": scores["forgetting_score"],
        "memory_bytes_per_delta": normalized_mem,
    }
    aggregate = sum(metrics.values()) / len(metrics)
    return EvalResult(method, metrics, aggregate)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Extended-learning component evaluation ===\n")
    print(f"Curriculum size: {len(CURRICULUM)} word-sense corrections")
    print(f"Trivia controls: {len(TRIVIA_FACTS)} unrelated facts\n")

    evals = [
        eval_baseline("Oracle"),
        eval_baseline("Always-Wrong"),
        eval_plastic_cortex(),
        eval_neural_hippocampus(),
        eval_world_model_critic(),
        eval_identity_hypernetwork(),
        eval_skill_immune_cortex(),
        eval_experience_autoencoder(),
    ]

    for r in evals:
        print(f"--- {r.module} ---")
        for k, v in r.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
        print(f"  aggregate_score: {r.aggregate:.3f}\n")

    print("=== Ranking ===")
    ranked = sorted(evals, key=lambda x: x.aggregate, reverse=True)
    for i, r in enumerate(ranked, 1):
        print(f"{i}. {r.module:22s} {r.aggregate:.3f}")


if __name__ == "__main__":
    random.seed(0)
    main()
