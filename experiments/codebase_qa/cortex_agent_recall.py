"""CortexAgent recall evaluator for the codebase-QA benchmark.

Measures whether attaching a KnowledgeStore to a CortexAgent improves
answer accuracy relative to the same agent without retrieved facts. Both
agents reuse an already-loaded driver so no extra LM weight copy is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Repo root is two directories above this script (experiments/codebase_qa/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.codebase_qa.knowledge_store import KnowledgeStore
from experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy_lm import LlamaCVecDriver


def _score(expected: str | list[str], answer: str) -> int:
    """Return 1 if any expected substring appears in answer (case-insensitive)."""
    answer = answer.lower()
    if isinstance(expected, str):
        expected = [expected]
    return 1 if any(exp.lower() in answer for exp in expected) else 0


def evaluate(
    driver: LlamaCVecDriver,
    facts: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    subset_size: int = 24,
) -> dict[str, Any]:
    """Evaluate CortexAgent recall against a deterministic question subset.

    Args:
        driver: An already-loaded LlamaCVecDriver shared by both agents.
        facts: Repository facts as loaded from facts.json.
        questions: Benchmark questions as loaded from questions.json.
        subset_size: Number of questions to evaluate (first ``subset_size``).

    Returns:
        A dict with ``baseline_accuracy``, ``recall_accuracy``,
        ``recall_lift``, and per-question ``results``.
    """
    # Deterministic keyword-only store, matching benchmark.py.
    store = KnowledgeStore(embed_fn=None)
    for fact in facts:
        store.add_fact(fact["key"], fact["value"], fact.get("metadata", {}))

    cfg = CortexAgentConfig()
    baseline_agent = CortexAgent(config=cfg, knowledge_store=None, driver=driver)
    recall_agent = CortexAgent(config=cfg, knowledge_store=store, driver=driver)

    if hasattr(driver._llm, "set_seed"):
        driver._llm.set_seed(42)

    subset = questions[:subset_size]
    baseline_scores: list[int] = []
    recall_scores: list[int] = []
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(subset, start=1):
        question = item["question"]
        expected = item["expected"]
        prompt = f"Answer briefly.\nQuestion: {question}\nAnswer:"

        baseline_agent.boot()
        baseline_answer = baseline_agent.articulate(
            prompt=prompt,
            max_tokens=48,
            temperature=0.0,
            apply_steering=False,
        ).strip()
        baseline_hit = _score(expected, baseline_answer)
        baseline_scores.append(baseline_hit)

        recall_agent.boot()
        recall_answer = recall_agent.articulate(
            prompt=prompt,
            max_tokens=48,
            temperature=0.0,
            apply_steering=False,
            recall_query=question,
        ).strip()
        recall_hit = _score(expected, recall_answer)
        recall_scores.append(recall_hit)


        results.append(
            {
                "idx": idx,
                "question": question,
                "expected": expected,
                "baseline_answer": baseline_answer,
                "baseline_score": baseline_hit,
                "recall_answer": recall_answer,
                "recall_score": recall_hit,
            }
        )

    baseline_acc = (
        sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    )
    recall_acc = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    recall_lift = recall_acc - baseline_acc

    return {
        "baseline_accuracy": baseline_acc,
        "recall_accuracy": recall_acc,
        "recall_lift": recall_lift,
        "results": results,
    }
