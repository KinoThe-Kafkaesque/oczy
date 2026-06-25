#!/usr/bin/env python3
"""Deterministic codebase-QA benchmark harness.

Measures code_qa_accuracy with and without retrieved repository facts injected
into the prompt. Retrieval uses the KnowledgeStore's keyword overlap scorer so
no embeddings are required and the harness remains deterministic and fast.
The LFM2.5-1.2B-Instruct Q4_K_M GGUF is still used for generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Repo root is two directories above this script (experiments/codebase_qa/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.codebase_qa.knowledge_store import KnowledgeStore
from experiments.codebase_qa.cortex_agent_recall import evaluate
from experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy_lm import CVecDriverConfig, LlamaCVecDriver
from plastic_cortex.kv_cortex import KVCortexConfig

_FACTS_PATH = Path(__file__).with_name("facts.json")
_QUESTIONS_PATH = Path(__file__).with_name("questions.json")


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_prompt(question: str, context: str = "") -> str:
    body = f"Answer briefly.\nQuestion: {question}\nAnswer:"
    if context:
        return f"{context}{body}"
    return body




def _score(expected: str | list[str], answer: str) -> int:
    answer = answer.lower()
    if isinstance(expected, str):
        expected = [expected]
    return 1 if any(exp.lower() in answer for exp in expected) else 0

def _run_consolidation_uptake(driver: LlamaCVecDriver) -> dict[str, Any]:
    """Probe boot-persistent semantic consolidation via SVD-initialised proj_c.

    A fresh CortexAgent is corrected several times toward a target answer.
    The correction hidden vectors are used to SVD-initialise ``proj_c`` so
    the steering direction is aligned to real corrections and survives
    cold boot. Consolidation then moves the warm update into cold_state.
    We record answers before, immediately after, and after reboot.
    """
    import numpy as np

    probe = "'Profile' here means business _______."
    expected = ["vertical"]
    correction = "No, 'profile' here means business vertical, not user profile."
    prompt = _build_prompt(probe)
    n_turns = 8

    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=8, steering_mode="proj_random"),
        articulate_scale=0.03,
        auto_consolidate=True,
    )
    agent = CortexAgent(config=cfg, knowledge_store=None, driver=driver)
    agent.boot()
    pre_answer = agent.articulate(
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
        apply_steering=True,
    ).strip()
    pre_score = _score(expected, pre_answer)
    pre_normalised = pre_answer.lower()

    correction_hiddens: list[np.ndarray] = []
    auto_fired = False
    for _ in range(n_turns):
        result = agent.turn(
            correction,
            correction_signal=1.0,
            max_tokens=4,
            temperature=0.0,
            metabolize=True,
        )
        if result.get("consolidated"):
            auto_fired = True
        if agent._last_hidden is not None:
            correction_hiddens.append(agent._last_hidden.copy())

    if correction_hiddens:
        try:
            agent.cortex.init_proj_c_from_svd(np.vstack(correction_hiddens))
        except Exception as exc:  # noqa: BLE001
            print(f"SVD init failed: {exc}")

    if not auto_fired and agent.should_consolidate():
        auto_fired = True

    if not auto_fired:
        agent.consolidate()

    # Immediate post-consolidation answer (tests new cold/warm state).
    post_warm_answer = agent.articulate(
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
        apply_steering=True,
    ).strip()
    post_warm_score = _score(expected, post_warm_answer)
    output_shift = 1 if post_warm_answer.lower() != pre_normalised else 0

    # Reboot from cold so the post answer comes from boot-persistent state.
    agent.boot()
    post_cold_answer = agent.articulate(
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
        apply_steering=True,
    ).strip()
    post_cold_score = _score(expected, post_cold_answer)
    cold_output_shift = 1 if post_cold_answer.lower() != pre_normalised else 0

    print(
        f"Consolidation uptake probe: {probe}\n"
        f"  pre:       {pre_answer!r} | semantic: {pre_score}\n"
        f"  post_warm: {post_warm_answer!r} | semantic: {post_warm_score} | shift: {output_shift}\n"
        f"  post_cold: {post_cold_answer!r} | semantic: {post_cold_score} | cold_shift: {cold_output_shift}\n"
        f"  auto_consolidated: {auto_fired}"
    )

    return {
        "pre_score": float(pre_score),
        "post_warm_score": float(post_warm_score),
        "post_cold_score": float(post_cold_score),
        "output_shift": float(output_shift),
        "cold_output_shift": float(cold_output_shift),
        "delta": float(post_cold_score - pre_score),
        "auto_fired": 1.0 if auto_fired else 0.0,
        "pre_answer": pre_answer,
        "post_warm_answer": post_warm_answer,
        "post_cold_answer": post_cold_answer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Oczy codebase-QA benchmark.")
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of repository facts to retrieve per question (default: 3).",
    )
    parser.add_argument(
        "--no-recall",
        action="store_true",
        help="Only run the baseline (no retrieved-context) path.",
    )
    args = parser.parse_args()

    facts = _load_json(_FACTS_PATH)
    questions = _load_json(_QUESTIONS_PATH)

    print("Loading LlamaCVecDriver...")
    cfg = CVecDriverConfig(n_ctx=512, n_threads=4, embedding=True)
    driver = LlamaCVecDriver.load(cfg)

    if hasattr(driver._llm, "set_seed"):
        driver._llm.set_seed(42)

    # Keyword-only store for a deterministic, fast retrieval path.
    store = KnowledgeStore(embed_fn=None)
    for fact in facts:
        store.add_fact(fact["key"], fact["value"], fact.get("metadata", {}))

    print(f"Knowledge store status: {store.status()}")
    print(f"Benchmarking {len(questions)} questions...")
    print(f"Retrieving up to {args.k} facts per question.")

    baseline_scores: list[int] = []
    recall_scores: list[int] = []

    for idx, item in enumerate(questions, start=1):
        question = item["question"]
        expected = item["expected"]

        baseline_prompt = _build_prompt(question)
        baseline_answer = driver.generate(
            baseline_prompt,
            max_tokens=48,
            temperature=0.0,
            stop=["\n"],
        )
        baseline_hit = _score(expected, baseline_answer)
        baseline_scores.append(baseline_hit)

        if args.no_recall:
            print(
                f"Q{idx}: {question}\n"
                f"  expected: {expected!r}\n"
                f"  baseline: {baseline_answer.strip()!r} | score: {baseline_hit}"
            )
            continue

        context = store.format_context(question, k=args.k, min_score=0.18)
        recall_prompt = _build_prompt(question, context)
        recall_answer = driver.generate(
            recall_prompt,
            max_tokens=48,
            temperature=0.0,
            stop=["\n"],
        )
        recall_hit = _score(expected, recall_answer)
        recall_scores.append(recall_hit)

        print(
            f"Q{idx}: {question}\n"
            f"  expected: {expected!r}\n"
            f"  baseline: {baseline_answer.strip()!r} | score: {baseline_hit}\n"
            f"  recall:   {recall_answer.strip()!r} | score: {recall_hit}"
        )

    baseline_acc = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    recall_acc = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    recall_lift = recall_acc - baseline_acc

    print(f"METRIC baseline_accuracy={baseline_acc:.4f}")
    if not args.no_recall:
        print(f"METRIC code_qa_accuracy={recall_acc:.4f}")
        print(f"METRIC recall_lift={recall_lift:.4f}")
    cortex_subset_size = 24
    print(f"Running CortexAgent recall evaluation on {cortex_subset_size} questions...")
    cortex_res = evaluate(driver, facts, questions, subset_size=cortex_subset_size)
    print(f"METRIC cortex_agent_baseline_accuracy={cortex_res['baseline_accuracy']:.4f}")
    print(f"METRIC cortex_agent_recall_accuracy={cortex_res['recall_accuracy']:.4f}")
    print(f"METRIC cortex_agent_recall_lift={cortex_res['recall_lift']:.4f}")

    print("Running consolidation uptake evaluation...")
    cons_res = _run_consolidation_uptake(driver)
    print(f"METRIC consolidation_uptake_pre={cons_res['pre_score']:.4f}")
    print(f"METRIC consolidation_uptake_post_warm={cons_res['post_warm_score']:.4f}")
    print(f"METRIC consolidation_uptake_output_shift={cons_res['output_shift']:.4f}")
    print(f"METRIC consolidation_uptake_post_cold={cons_res['post_cold_score']:.4f}")
    print(f"METRIC consolidation_uptake_cold_output_shift={cons_res['cold_output_shift']:.4f}")
    print(f"METRIC consolidation_uptake_delta={cons_res['delta']:.4f}")
    print(f"METRIC consolidation_uptake_auto_fired={cons_res['auto_fired']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
