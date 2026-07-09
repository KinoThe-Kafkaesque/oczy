#!/usr/bin/env python3
"""Benchmark: Oczy (lfm-oczy) using Pi CLI tools to explore and edit code.

Runs a set of coding tasks through `pi --model lfm-oczy --print` and
scores whether the model successfully uses tools (read, bash, edit)
to explore the codebase and make changes.

Usage:
    uv run python benchmarks/pi/run_tool_use_benchmark.py

Requires the proxy server running on localhost:8080.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark tasks
# ---------------------------------------------------------------------------

TASKS: list[dict] = [
    {
        "id": "read-file",
        "prompt": "Read the file pyproject.toml and tell me the project name. Do not guess — use the read tool.",
        "scorer": "exact_answer",
        "expected": "oczy",
        "timeout": 600,
    },
    {
        "id": "find-file",
        "prompt": "Find all Python files under src/oczy/experiments/ that contain 'CortexAgent'. Use the bash or ffgrep tool, then list the filenames.",
        "scorer": "contains_any",
        "expected": ["cortex_agent.py"],
        "timeout": 600,
    },
    {
        "id": "edit-file",
        "prompt": (
            "Create a file at /tmp/oczy_bench_marker.py with the content: "
            'print("hello from oczy benchmark")'
        ),
        "scorer": "file_exists",
        "expected": "/tmp/oczy_bench_marker.py",
        "timeout": 600,
    },
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_exact_answer(output: str, expected: str) -> dict:
    """Check if the expected string appears in the output (case-insensitive)."""
    found = expected.lower() in output.lower()
    return {"passed": found, "detail": f"expected '{expected}' in output"}


def _score_contains_any(output: str, expected: list[str]) -> dict:
    """Check if any of the expected strings appear in the output."""
    found = [e for e in expected if e.lower() in output.lower()]
    return {"passed": len(found) > 0, "detail": f"found: {found}"}


def _score_file_exists(output: str, expected: str) -> dict:
    """Check if the expected file was created."""
    exists = Path(expected).exists()
    # Clean up after scoring.
    if exists:
        Path(expected).unlink(missing_ok=True)
    return {"passed": exists, "detail": f"file {expected} exists: {exists}"}


def _score(task: dict, output: str) -> dict:
    scorer = task["scorer"]
    expected = task["expected"]
    if scorer == "exact_answer":
        return _score_exact_answer(output, expected)
    elif scorer == "contains_any":
        return _score_contains_any(output, expected)
    elif scorer == "file_exists":
        return _score_file_exists(output, expected)
    return {"passed": False, "detail": f"unknown scorer: {scorer}"}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_task(task: dict) -> dict:
    """Run a single benchmark task through pi."""
    prompt = task["prompt"]
    timeout = task.get("timeout", 180)

    # Clean up any marker files before running.
    if task["scorer"] == "file_exists":
        Path(task["expected"]).unlink(missing_ok=True)

    start = time.monotonic()
    try:
        result = subprocess.run(
            [
                "pi",
                "--model", "lfm-oczy",
                "--print",
                "--no-session",
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        elapsed = time.monotonic() - start
        output = result.stdout + result.stderr
        score = _score(task, output)
        return {
            "id": task["id"],
            "passed": score["passed"],
            "detail": score["detail"],
            "elapsed_sec": round(elapsed, 1),
            "output_preview": output[:500],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return {
            "id": task["id"],
            "passed": False,
            "detail": f"timed out after {timeout}s",
            "elapsed_sec": round(elapsed, 1),
            "output_preview": "",
            "returncode": -1,
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "id": task["id"],
            "passed": False,
            "detail": f"error: {e}",
            "elapsed_sec": round(elapsed, 1),
            "output_preview": "",
            "returncode": -1,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Verify the proxy server is running.
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=5)
        models = json.loads(resp.read())
        model_ids = [m["id"] for m in models.get("data", [])]
        if "lfm-oczy" not in model_ids:
            print("ERROR: lfm-oczy model not found on proxy server.")
            print(f"  Available: {model_ids}")
            return 1
    except Exception as e:
        print(f"ERROR: Proxy server not reachable at http://127.0.0.1:8080: {e}")
        print("Start it with:")
        print("  uv run python benchmarks/pi/proxy_server.py --port 8080 --model-path <path>")
        return 1

    print("=" * 60)
    print("  OCZY PI TOOL-USE BENCHMARK")
    print("=" * 60)
    print(f"  Tasks: {len(TASKS)}")
    print(f"  Model: lfm-oczy (via pi --model lfm-oczy)")
    print()

    results: list[dict] = []
    for task in TASKS:
        print(f"--- Running task: {task['id']} ---")
        print(f"  Prompt: {task['prompt'][:80]}...")
        result = _run_task(task)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {result['detail']} ({result['elapsed_sec']}s)")
        if result["output_preview"]:
            print(f"  Output: {result['output_preview'][:200]}")
        print()

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print("=" * 60)
    print(f"  RESULTS: {passed}/{total} passed")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['id']:<20} {status:<6} {r['elapsed_sec']:>6.1f}s  {r['detail']}")

    # Save results to logs.
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"tool_use_benchmark_{int(time.time())}.json"
    with log_path.open("w") as f:
        json.dump({
            "timestamp": int(time.time()),
            "model": "lfm-oczy",
            "passed": passed,
            "total": total,
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to: {log_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
