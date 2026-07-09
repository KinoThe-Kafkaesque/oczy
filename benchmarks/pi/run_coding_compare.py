#!/usr/bin/env python3
"""Direct coding benchmark: Oczy vs vanilla LFM via the proxy server.

Sends the llm-coding-benchmark Phase 1 prompt to both models,
records outputs, token counts, timing, and compares.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROMPT_PATH = Path("/tmp/llm-coding-benchmark/prompts/benchmark_prompt.txt")
PROXY_URL = "http://127.0.0.1:8080/v1/chat/completions"


def _read_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text()
    return "Write a hello world Python script to /tmp/hello.py"


def _call_model(model: str, prompt: str, max_tokens: int = 2048) -> dict:
    """Call the proxy server and return timing + output."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.monotonic()
    try:
        result = subprocess.run(
            [
                "curl", "-s", PROXY_URL,
                "-H", "Content-Type: application/json",
                "-d", json.dumps(payload),
            ],
            capture_output=True, text=True, timeout=600,
        )
        elapsed = time.monotonic() - start
        data = json.loads(result.stdout) if result.stdout else {}
        content = ""
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return {
            "model": model,
            "output": content,
            "output_chars": len(content),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "elapsed_sec": round(elapsed, 1),
            "error": None,
        }
    except Exception as e:
        return {
            "model": model,
            "output": "",
            "output_chars": 0,
            "elapsed_sec": time.monotonic() - start,
            "error": str(e),
        }


def _score_output(output: str) -> dict:
    """Heuristic quality scoring for coding output."""
    indicators = {
        "has_code_block": "```" in output,
        "has_ruby_class": "class " in output.lower() or "def " in output.lower(),
        "has_file_paths": "/" in output and "." in output,
        "has_commands": "`" in output or "$ " in output,
        "has_explanation": len(output.split()) > 20,
        "mentions_rails": "rails" in output.lower(),
        "mentions_gemfile": "gemfile" in output.lower() or "gem " in output.lower(),
        "mentions_docker": "docker" in output.lower(),
        "has_structure": output.count("\n") > 5,
    }
    score = sum(1 for v in indicators.values() if v)
    return {"indicators": indicators, "score": score, "max_score": len(indicators)}


def main() -> int:
    prompt = _read_prompt()
    print(f"Prompt: {len(prompt)} chars, {len(prompt.split())} words")
    print(f"Prompt preview: {prompt[:200]}...")
    print()

    results = {}
    for model in ["lfm-vanilla", "lfm-oczy"]:
        print(f"--- Running {model} ---")
        result = _call_model(model, prompt)
        scoring = _score_output(result["output"])
        result["scoring"] = scoring
        results[model] = result
        print(f"  Output: {result['output_chars']} chars in {result['elapsed_sec']}s")
        print(f"  Quality score: {scoring['score']}/{scoring['max_score']}")
        print(f"  Indicators: {json.dumps(scoring['indicators'])}")
        print(f"  Preview: {result['output'][:300]}...")
        print()

    # Comparison
    o = results.get("lfm-oczy", {})
    v = results.get("lfm-vanilla", {})
    print("=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} {'lfm-vanilla':>15} {'lfm-oczy':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    for key in ["output_chars", "elapsed_sec"]:
        vo = v.get(key, 0)
        oo = o.get(key, 0)
        print(f"  {key:<25} {vo:>15} {oo:>15}")
    vs = v.get("scoring", {}).get("score", 0)
    os_ = o.get("scoring", {}).get("score", 0)
    print(f"  {'quality_score':<25} {vs:>15} {os_:>15}")
    delta = os_ - vs
    print(f"  {'oczy_delta':<25} {'':>15} {delta:>+15}")
    print()

    # Save results
    out_path = Path("/tmp/oczy_vs_vanilla_coding.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Full results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
