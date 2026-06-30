#!/usr/bin/env python
"""Benchmark low-dim vs high-dim cortex on the organism curriculum.

Loads the real LFM2.5-1.2B Q4 driver once, then runs the full 6-stage
curriculum for each ``d_cortex`` value.  Captures per-stage:

  - **Performance**: post-test accuracy (retention, scope, transfer)
  - **Improvement speed**: uptake_latency (fraction of episodes NOT
    fixed on first try — lower = faster learning) and pre→post delta

Usage::

    python -m src.oczy.experiments.organism_curriculum.bench_cortex_dim
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

from oczy.experiments.organism import OrganismAgent
from oczy.experiments.organism_curriculum.dataset import build_curriculum
from oczy.experiments.organism_curriculum.run_curriculum import (
    StageResult,
    run_stage,
)
from oczy.experiments.organism_curriculum.scoring import categorize_results

# ---------------------------------------------------------------------------
# Cortex dimension sweep
# ---------------------------------------------------------------------------
D_CORTEX_VALUES = [2, 4, 8, 16, 32, 64, 128]


def _load_driver():
    """Load the real LFM2.5 driver once."""
    from oczy.lm import CVecDriverConfig, LlamaCVecDriver

    print("Loading real LlamaCVecDriver...", flush=True)
    driver = LlamaCVecDriver.load(
        CVecDriverConfig(n_ctx=128, n_threads=4, embedding=True)
    )
    print(f"Driver loaded (n_embd={driver.n_embd}).", flush=True)
    return driver


def _make_cortex_agent(driver, d_cortex: int):
    """Create a fresh CortexAgent with the given d_cortex."""
    from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
    from plastic_cortex.kv_cortex import KVCortexConfig

    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=d_cortex),
        use_policy_head=True,
        policy_learning_rate=0.001,
    )
    cortex = CortexAgent(cfg, driver=driver)
    cortex.boot()
    return cortex


def _make_organism(d_cortex: int, driver) -> OrganismAgent:
    """Create a fresh OrganismAgent with a cortex at the given d_cortex."""
    agent_config: dict[str, Any] = {}
    agent = OrganismAgent(agent_config)
    agent.cortex_agent = _make_cortex_agent(driver, d_cortex)
    # Enable policy-loop gates (matches --use-real-driver defaults).
    agent.use_cortex_policy = True
    agent.use_value_baseline = True
    agent.use_acceptance_policy_reward = True
    agent.policy_suppresses_fast_answer = True
    return agent


def _extract_metrics(sr: StageResult) -> dict[str, Any]:
    """Extract per-stage metrics from a StageResult."""
    fixed = sum(1 for r in sr.episode_results if r.fixed)
    total = len(sr.episode_results)
    uptake = sr.uptake_latency()
    pre_acc = categorize_results(sr.pre_probe_results)
    post_acc = categorize_results(sr.post_probe_results)

    def _cat(acc, key):
        if key in acc:
            return acc[key][2]  # ratio
        return 0.0

    # Sum all categories for overall pre/post
    pre_total = sum(v[1] for v in pre_acc.values())
    post_total = sum(v[1] for v in post_acc.values())
    pre_overall = sum(v[0] for v in pre_acc.values()) / max(pre_total, 1)
    post_overall = sum(v[0] for v in post_acc.values()) / max(post_total, 1)

    return {
        "name": sr.name,
        "fixed": fixed,
        "total": total,
        "uptake": round(uptake, 4),
        "pre": round(pre_overall, 4),
        "post": round(post_overall, 4),
        "improvement": round(post_overall - pre_overall, 4),
        "mem_delta": sr.memory_bytes_after - sr.memory_bytes_before,
        "pre_retention": round(_cat(pre_acc, "retention"), 4),
        "post_retention": round(_cat(post_acc, "retention"), 4),
        "pre_scope": round(_cat(pre_acc, "scope"), 4),
        "post_scope": round(_cat(post_acc, "scope"), 4),
        "pre_transfer": round(_cat(pre_acc, "transfer"), 4),
        "post_transfer": round(_cat(post_acc, "transfer"), 4),
    }


def run_benchmark() -> int:
    driver = _load_driver()
    stages = build_curriculum()
    n_stages = len(stages)

    all_results: dict[int, list[dict[str, Any]]] = {}
    all_times: dict[int, float] = {}

    for d_cortex in D_CORTEX_VALUES:
        print(f"\n{'='*70}", flush=True)
        print(f"d_cortex = {d_cortex}", flush=True)
        print(f"{'='*70}", flush=True)

        t0 = time.time()
        agent = _make_organism(d_cortex, driver)
        stage_metrics: list[dict[str, Any]] = []

        for _si, stage in enumerate(stages):
            sr = run_stage(agent, stage, adapter=None, semantic=True)
            m = _extract_metrics(sr)
            stage_metrics.append(m)
            print(
                f"  {m['name'][:28]:28s}  {m['fixed']}/{m['total']}  "
                f"uptake={m['uptake']:.2f}  pre={m['pre']:.2f}  post={m['post']:.2f}  "
                f"Δ={m['improvement']:+.2f}  mem={m['mem_delta']:+d}B",
                flush=True,
            )

        elapsed = time.time() - t0
        all_results[d_cortex] = stage_metrics
        all_times[d_cortex] = elapsed
        print(f"  elapsed: {elapsed:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n\n{'='*90}", flush=True)
    print("SUMMARY: d_cortex sweep", flush=True)
    print(f"{'='*90}\n", flush=True)

    # Performance table (post-test accuracy per stage)
    header = f"{'d_cortex':>8s}"
    for si in range(n_stages):
        header += f" | S{si} post"
    header += f" | {'avg post':>8s} | {'elapsed':>7s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for d_cortex in D_CORTEX_VALUES:
        row = f"{d_cortex:>8d}"
        posts = []
        for m in all_results[d_cortex]:
            row += f" | {m['post']:8.2f}"
            posts.append(m["post"])
        avg_post = sum(posts) / max(len(posts), 1)
        row += f" | {avg_post:8.2f} | {all_times[d_cortex]:6.1f}s"
        print(row, flush=True)

    # Improvement speed table (uptake + pre→post delta)
    print(flush=True)
    header2 = f"{'d_cortex':>8s}"
    for si in range(n_stages):
        header2 += f" | S{si} upt"
    header2 += f" | {'avg upt':>8s} | {'avg Δ':>6s}"
    print(header2, flush=True)
    print("-" * len(header2), flush=True)

    for d_cortex in D_CORTEX_VALUES:
        row = f"{d_cortex:>8d}"
        ups = []
        deltas = []
        for m in all_results[d_cortex]:
            row += f" | {m['uptake']:8.2f}"
            ups.append(m["uptake"])
            deltas.append(m["improvement"])
        avg_upt = sum(ups) / max(len(ups), 1)
        avg_delta = sum(deltas) / max(len(deltas), 1)
        row += f" | {avg_upt:8.2f} | {avg_delta:+6.2f}"
        print(row, flush=True)

    # Scope detail
    print(flush=True)
    header3 = f"{'d_cortex':>8s}"
    for si in range(n_stages):
        header3 += f" | S{si} scope"
    header3 += f" | {'avg scope':>9s}"
    print(header3, flush=True)
    print("-" * len(header3), flush=True)

    for d_cortex in D_CORTEX_VALUES:
        row = f"{d_cortex:>8d}"
        scopes = []
        for m in all_results[d_cortex]:
            row += f" | {m['post_scope']:9.2f}"
            scopes.append(m["post_scope"])
        avg_scope = sum(scopes) / max(len(scopes), 1)
        row += f" | {avg_scope:9.2f}"
        print(row, flush=True)

    # Memory footprint
    print(flush=True)
    header4 = f"{'d_cortex':>8s}"
    for si in range(n_stages):
        header4 += f" | S{si} mem"
    header4 += f" | {'total mem':>9s}"
    print(header4, flush=True)
    print("-" * len(header4), flush=True)

    for d_cortex in D_CORTEX_VALUES:
        row = f"{d_cortex:>8d}"
        mems = []
        for m in all_results[d_cortex]:
            row += f" | {m['mem_delta']:+8d}"
            mems.append(m["mem_delta"])
        row += f" | {sum(mems):+9d}"
        print(row, flush=True)

    # JSON dump for downstream analysis
    output = {
        str(d): {
            "stages": all_results[d],
            "elapsed_seconds": all_times[d],
        }
        for d in D_CORTEX_VALUES
    }
    print(f"\nJSON:\n{json.dumps(output, indent=2)}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
