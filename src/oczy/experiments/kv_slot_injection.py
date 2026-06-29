"""KV-slot fact injection probe (Experiment 02).

Tests whether text-derived KV prefill-and-reuse can force exact-token
recall on an LFM2.5-1.2B-Instruct GGUF, where residual cvecs fail.

Heavily based on ``lanes/lane_02.py``, which proved rank-1 recall via the
llama-cpp-python 0.3.31 per-sequence state APIs
(``llama_state_seq_get_data`` / ``llama_state_seq_set_data``).

Modes:
  - ``--driver mock``: fast smoke path. No real semantics; asserts wiring and
    prints NaN for the real-driver primary metric.
  - ``--driver real``: loads the cached Q4 GGUF and runs C0/C1/C2/C6
    conditions.

Primary metric: ``kv_slot_rank1_count`` = number of facts (out of 3) whose
first target token is rank-1 in the KV-chunk condition. Expected 3/3.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from typing import Any

import numpy as np


# Probe template tuned in lanes/lane_02.py. It lowers the surface-form
# ambiguity so that the natural next-token is the lowercase target.
_PROBE_TEMPLATE = "\n\nRecall the answer in lowercase. Question: {}\nAnswer:"

_FACT_POSITIONS: dict[str, str] = {
    "skylark": "alpha",
    "rook": "project-beta",
    "marmalade": "level-7",
}


def _facts_queries_targets() -> tuple[list[str], list[str], list[str]]:
    """Return the canonical 3-fact probe set."""
    facts = [
        "The secret passphrase for level 7 is marmalade.",
        "Project-beta's standard piece is rook.",
        "In group alpha the chosen call sign is skylark.",
    ]
    queries = [
        "What is the secret passphrase for level 7?",
        "What piece does project-beta use?",
        "What is the call sign in group alpha?",
    ]
    targets = ["marmalade", "rook", "skylark"]
    return facts, queries, targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _target_id(llm: Any, target: str) -> int | None:
    """Return the token id for a space-prefixed target string."""
    ids = llm.tokenize((" " + target).encode("utf-8"), add_bos=False)
    return int(ids[0]) if ids else None


def _rank_of_target(logits: np.ndarray, target_id: int) -> int:
    """Return 1-based rank of target_id in the logits array."""
    return int(1 + np.sum(logits > logits[target_id]))


def _probe_prompt(query: str) -> str:
    return _PROBE_TEMPLATE.format(query)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def _condition_baseline(
    llm: Any, fact: str, query: str, target: str
) -> dict[str, Any]:
    """No prefix/steering: measure how often the LM emits the target naturally."""
    n_vocab = llm.n_vocab()
    prompt = _probe_prompt(query)
    ids = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    if not ids:
        return {"rank": n_vocab, "top1": ""}
    llm.eval(ids)
    raw = llm._ctx.get_logits()
    logits = np.ctypeslib.as_array(raw, shape=(len(ids) * n_vocab,))
    last = logits[(len(ids) - 1) * n_vocab : len(ids) * n_vocab]
    tid = _target_id(llm, target)
    rank = _rank_of_target(last, tid) if tid is not None else n_vocab
    top1_id = int(np.argmax(last))
    top1 = llm.detokenize([top1_id]).decode("utf-8", errors="replace")
    return {"rank": rank, "top1": top1}


def _condition_live_prefix(
    llm: Any, fact: str, query: str, target: str
) -> dict[str, Any]:
    """Text prefix prepended and re-encoded each query."""
    n_vocab = llm.n_vocab()
    prompt = fact + _probe_prompt(query)
    ids = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    if not ids:
        return {"rank": n_vocab, "top1": ""}
    llm.eval(ids)
    raw = llm._ctx.get_logits()
    logits = np.ctypeslib.as_array(raw, shape=(len(ids) * n_vocab,))
    last = logits[(len(ids) - 1) * n_vocab : len(ids) * n_vocab]
    tid = _target_id(llm, target)
    rank = _rank_of_target(last, tid) if tid is not None else n_vocab
    top1_id = int(np.argmax(last))
    top1 = llm.detokenize([top1_id]).decode("utf-8", errors="replace")
    return {"rank": rank, "top1": top1}


def _condition_kv_chunk(
    llm: Any, fact: str, query: str, target: str
) -> dict[str, Any] | None:
    """Prefill fact once, snapshot per-seq state, restore per query."""
    import llama_cpp

    n_vocab = llm.n_vocab()
    ctx_p = llm._ctx.ctx

    llm.reset()
    fact_ids = llm.tokenize(fact.encode("utf-8"), add_bos=True)
    if not fact_ids:
        return None
    llm.eval(fact_ids)

    size = llama_cpp.llama_state_seq_get_size(ctx_p, 0)
    if size <= 0:
        return None
    buf = (ctypes.c_uint8 * size)()
    got = llama_cpp.llama_state_seq_get_data(ctx_p, buf, size, 0)
    if got != size:
        return None

    start = time.perf_counter()
    llm.reset()
    ret = llama_cpp.llama_state_seq_set_data(ctx_p, buf, size, 0)
    if ret != size:
        return None
    lat_ms = (time.perf_counter() - start) * 1000.0
    llm.n_tokens = len(fact_ids)

    probe = _probe_prompt(query)
    probe_ids = llm.tokenize(probe.encode("utf-8"), add_bos=False)
    if not probe_ids:
        return None
    llm.eval(probe_ids)

    raw = llm._ctx.get_logits()
    logits = np.ctypeslib.as_array(raw, shape=(len(probe_ids) * n_vocab,))
    last = logits[(len(probe_ids) - 1) * n_vocab : len(probe_ids) * n_vocab]
    tid = _target_id(llm, target)
    rank = _rank_of_target(last, tid) if tid is not None else n_vocab
    top1_id = int(np.argmax(last))
    top1 = llm.detokenize([top1_id]).decode("utf-8", errors="replace")
    return {"rank": rank, "top1": top1, "latency_ms": lat_ms}


def _condition_logit_bias(
    llm: Any, fact: str, query: str, target: str, bias: float = 20.0
) -> dict[str, Any]:
    """Known-good logit-bias reference (post-forward bias, not KV slot)."""
    n_vocab = llm.n_vocab()
    prompt = _probe_prompt(query)
    ids = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    if not ids:
        return {"rank": n_vocab, "top1": ""}
    llm.eval(ids)
    raw = llm._ctx.get_logits()
    logits = np.ctypeslib.as_array(raw, shape=(len(ids) * n_vocab,))
    last = logits[(len(ids) - 1) * n_vocab : len(ids) * n_vocab].copy()
    tid = _target_id(llm, target)
    if tid is not None:
        last[tid] += bias
    top1_id = int(np.argmax(last))
    top1 = llm.detokenize([top1_id]).decode("utf-8", errors="replace")
    return {"rank": 1 if top1_id == tid else 0, "top1": top1}


# ---------------------------------------------------------------------------
# Real driver harness
# ---------------------------------------------------------------------------


def _run_real_driver() -> dict[str, Any] | None:
    """Run C0/C1/C2/C6 on the real GGUF driver.

    Returns a results dict or ``None`` if the GGUF is missing / load fails.
    """
    from llama_cpp import Llama

    from oczy.experiments.multi_fact_stressor import _resolve_gguf_path

    resolved = _resolve_gguf_path()
    if resolved is None:
        return None

    llm = Llama(
        model_path=str(resolved),
        n_ctx=512,
        n_threads=4,
        embedding=True,
        verbose=False,
    )

    facts, queries, targets = _facts_queries_targets()

    results = {
        "baseline": [],
        "live_prefix": [],
        "kv_chunk": [],
        "logit_bias": [],
    }
    for fact, query, target in zip(facts, queries, targets, strict=True):
        llm.reset()
        results["baseline"].append(
            _condition_baseline(llm, fact, query, target)
        )
        llm.reset()
        results["live_prefix"].append(
            _condition_live_prefix(llm, fact, query, target)
        )
        llm.reset()
        kv = _condition_kv_chunk(llm, fact, query, target)
        if kv is None:
            return None
        results["kv_chunk"].append(kv)
        llm.reset()
        results["logit_bias"].append(
            _condition_logit_bias(llm, fact, query, target)
        )

    return results


def _run_mock_driver() -> dict[str, Any]:
    """Fast smoke path: return plausible NaN-ish results.

    The wiring is validated without loading any model.
    """
    _, queries, targets = _facts_queries_targets()
    dummy_rank = len(queries)  # not rank-1
    return {
        "baseline": [{"rank": dummy_rank, "top1": ""} for _ in queries],
        "live_prefix": [{"rank": dummy_rank, "top1": ""} for _ in queries],
        "kv_chunk": [{"rank": dummy_rank, "top1": ""} for _ in queries],
        "logit_bias": [{"rank": 1, "top1": t} for t in targets],
    }


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------


def _emit_metrics(results: dict[str, Any]) -> None:
    counts = {
        cond: sum(1 for r in arr if r.get("rank", 1e9) == 1)
        for cond, arr in results.items()
    }

    kv_count = counts["kv_chunk"]
    print(f"METRIC kv_slot_rank1_count={kv_count}")
    print(f"ASI live_prefix_rank1_count={counts['live_prefix']}")
    print(f"ASI logit_bias_rank1_count={counts['logit_bias']}")
    print(f"ASI baseline_rank1_count={counts['baseline']}")

    for cond, arr in results.items():
        for i, r in enumerate(arr):
            print(f"ASI rank_{cond}_{i}={r.get('rank', float('nan'))}")

    latencies = [
        r.get("latency_ms", 0.0)
        for r in results.get("kv_chunk", [])
        if "latency_ms" in r
    ]
    if latencies:
        print(f"ASI median_injection_latency_ms={float(np.median(latencies))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="KV-slot fact injection probe"
    )
    parser.add_argument(
        "--driver",
        choices=["mock", "real"],
        default="mock",
        help="mock = fast smoke path; real = load LFM2.5 GGUF",
    )
    args = parser.parse_args(argv)

    if args.driver == "real":
        try:
            results = _run_real_driver()
        except Exception:
            results = None
        if results is None:
            print("ASI real_driver=failed")
            results = _run_mock_driver()
            # Force the primary metric to NaN on real failure.
            print("METRIC kv_slot_rank1_count=nan")
            return 0
    else:
        results = _run_mock_driver()

    _emit_metrics(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
