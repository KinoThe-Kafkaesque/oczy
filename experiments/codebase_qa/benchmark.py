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
from oczy_lm import CVecDriverConfig, LlamaCVecDriver

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
