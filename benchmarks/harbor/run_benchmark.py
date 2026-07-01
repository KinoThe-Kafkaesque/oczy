#!/usr/bin/env python3
"""Harbor QA benchmark runner — runs both agents against a question set.

Usage:
    python run_benchmark.py [--questions questions.json] [--output results.json]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent

# Default question set: 10 questions testing knowledge retention,
# cross-domain disambiguation, and factual recall — Oczy's strengths.
DEFAULT_QUESTIONS: list[dict] = [
    {
        "id": "q1_knowledge",
        "question": "What is the north-star metric for the Oczy project?",
        "expected_tokens": ["behavior", "delta", "byte", "persistent", "memory"],
        "category": "knowledge_retention",
    },
    {
        "id": "q2_knowledge",
        "question": "Which workspace packages exist in the Oczy repository?",
        "expected_tokens": [
            "correction", "benchmark", "experience", "autoencoder",
            "identity", "hypernetwork", "neural", "hippocampus",
            "plastic", "cortex", "skill", "immune", "world", "model",
        ],
        "category": "knowledge_retention",
    },
    {
        "id": "q3_disambig",
        "question": "The ship's log is missing from the archive.",
        "expected_tokens": ["captain", "journal", "log", "nautical"],
        "category": "cross_domain",
    },
    {
        "id": "q4_disambig",
        "question": "Log the server crash in the system.",
        "expected_tokens": ["system", "error", "crash", "server", "log"],
        "category": "cross_domain",
    },
    {
        "id": "q5_disambig",
        "question": "Please file these papers with the clerk.",
        "expected_tokens": ["submit", "officially", "clerk", "papers"],
        "category": "cross_domain",
    },
    {
        "id": "q6_disambig",
        "question": "Save the source file before building.",
        "expected_tokens": ["computer", "source", "disk", "save"],
        "category": "cross_domain",
    },
    {
        "id": "q7_factual",
        "question": "How does Oczy want memory to be represented?",
        "expected_tokens": ["changed", "dynamic", "process", "weights"],
        "category": "factual_recall",
    },
    {
        "id": "q8_factual",
        "question": "What does the NeuralHippocampus do in Oczy?",
        "expected_tokens": ["store", "surprise", "experience", "replay", "consolidate"],
        "category": "factual_recall",
    },
    {
        "id": "q9_factual",
        "question": "What is the role of the WorldModelCritic?",
        "expected_tokens": ["predict", "error", "surprise", "acceptance", "gate"],
        "category": "factual_recall",
    },
    {
        "id": "q10_general",
        "question": "Explain the Oczy architecture thesis in one sentence.",
        "expected_tokens": ["memory", "dynamic", "weight", "experience", "trace"],
        "category": "general_understanding",
    },
]


def run_agent(script: Path, question: str, timeout: int = 120) -> dict:
    """Run an agent script with a question and return timing + output."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            input=question,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
        )
        elapsed = time.monotonic() - start
        return {
            "answer": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
            "elapsed_sec": round(elapsed, 3),
        }
    except subprocess.TimeoutExpired:
        return {
            "answer": "",
            "stderr": "TIMEOUT",
            "exit_code": -1,
            "elapsed_sec": timeout,
        }


def score_answer(answer: str, expected_tokens: list[str]) -> dict:
    """Score an answer against expected tokens."""
    lower = answer.lower()
    matched = [t for t in expected_tokens if t.lower() in lower]
    precision = len(matched) / len(expected_tokens) if expected_tokens else 0.0
    return {
        "matched_tokens": matched,
        "precision": round(precision, 3),
        "matched_count": len(matched),
        "expected_count": len(expected_tokens),
    }


def run_benchmark(
    questions: list[dict] | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run both agents against all questions and return results."""
    if questions is None:
        questions = DEFAULT_QUESTIONS

    oczy_script = BENCHMARK_DIR / "oczy_agent.py"
    vanilla_script = BENCHMARK_DIR / "vanilla_lfm.py"

    results: dict = {
        "questions": questions,
        "oczy": {"results": [], "total_precision": 0.0, "total_time_sec": 0.0},
        "vanilla": {"results": [], "total_precision": 0.0, "total_time_sec": 0.0},
    }

    for q in questions:
        print(f"  [{q['id']}] {q['question'][:60]}...", file=sys.stderr)

        # Run Oczy.
        oczy_out = run_agent(oczy_script, q["question"])
        oczy_score = score_answer(oczy_out["answer"], q["expected_tokens"])
        results["oczy"]["results"].append({
            "id": q["id"],
            "question": q["question"],
            "answer": oczy_out["answer"],
            "score": oczy_score,
            "elapsed_sec": oczy_out["elapsed_sec"],
        })
        results["oczy"]["total_precision"] += oczy_score["precision"]
        results["oczy"]["total_time_sec"] += oczy_out["elapsed_sec"]

        # Run vanilla LFM.
        vanilla_out = run_agent(vanilla_script, q["question"])
        vanilla_score = score_answer(vanilla_out["answer"], q["expected_tokens"])
        results["vanilla"]["results"].append({
            "id": q["id"],
            "question": q["question"],
            "answer": vanilla_out["answer"],
            "score": vanilla_score,
            "elapsed_sec": vanilla_out["elapsed_sec"],
        })
        results["vanilla"]["total_precision"] += vanilla_score["precision"]
        results["vanilla"]["total_time_sec"] += vanilla_out["elapsed_sec"]

    n = len(questions)
    results["oczy"]["avg_precision"] = round(results["oczy"]["total_precision"] / n, 3) if n else 0.0
    results["vanilla"]["avg_precision"] = round(results["vanilla"]["total_precision"] / n, 3) if n else 0.0

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)

    return results


def print_report(results: dict) -> None:
    """Print a human-readable comparison report."""
    n = len(results["questions"])
    o = results["oczy"]
    v = results["vanilla"]

    print()
    print("=" * 72)
    print("  Oczy vs Vanilla LFM — Harbor QA Benchmark")
    print("=" * 72)
    print(f"  Questions: {n}")
    print(f"  Oczy avg precision:      {o['avg_precision']:.3f}  ({o['total_time_sec']:.1f}s)")
    print(f"  Vanilla LFM avg precision: {v['avg_precision']:.3f}  ({v['total_time_sec']:.1f}s)")
    print(f"  Delta:                    {o['avg_precision'] - v['avg_precision']:+.3f}")
    print()

    by_category: dict[str, dict] = {}
    for q, o_r, v_r in zip(results["questions"], o["results"], v["results"]):
        cat = q["category"]
        if cat not in by_category:
            by_category[cat] = {"oczy": 0.0, "vanilla": 0.0, "count": 0}
        by_category[cat]["oczy"] += o_r["score"]["precision"]
        by_category[cat]["vanilla"] += v_r["score"]["precision"]
        by_category[cat]["count"] += 1

    print("  By category:")
    print(f"  {'Category':<28} {'Oczy':>8} {'Vanilla':>8} {'Delta':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}")
    for cat, stats in sorted(by_category.items()):
        c = stats["count"]
        oczy_avg = stats["oczy"] / c
        van_avg = stats["vanilla"] / c
        print(f"  {cat:<28} {oczy_avg:>8.3f} {van_avg:>8.3f} {oczy_avg - van_avg:>+8.3f}")

    print()
    print("  Per-question results:")
    print(f"  {'ID':<22} {'Oczy':>6} {'Vanilla':>6} {'Delta':>6}  Answer (Oczy)")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6}  {'-'*30}")
    for q, o_r, v_r in zip(results["questions"], o["results"], v["results"]):
        delta = o_r["score"]["precision"] - v_r["score"]["precision"]
        ans = o_r["answer"][:60].replace("\n", " ")
        print(f"  {q['id']:<22} {o_r['score']['precision']:>6.3f} {v_r['score']['precision']:>6.3f} {delta:>+6.3f}  {ans}")


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Oczy vs Vanilla LFM benchmark")
    p.add_argument("--questions", type=Path, help="Path to questions JSON")
    p.add_argument("--output", type=Path, default=BENCHMARK_DIR / "results.json",
                   help="Path to output JSON")
    args = p.parse_args()

    questions = None
    if args.questions:
        questions = json.loads(args.questions.read_text())

    results = run_benchmark(questions=questions, output_path=args.output)
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
